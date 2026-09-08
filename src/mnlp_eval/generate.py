"""Stage A: produce hypotheses and write a self-describing run directory.

Generation is deliberately separate from scoring. The metric libraries this
project needs cannot share a process with the generation stack, and beyond that
constraint the split means a finished run can be re-scored with a new metric
years later without spending GPU hours regenerating it.
"""

from __future__ import annotations

import dataclasses
import sys
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from mnlp_eval.artifacts import Segment, atomic_write_json, write_jsonl, write_lines
from mnlp_eval.config import DecodeSpec, RunConfig
from mnlp_eval.data import load_testset
from mnlp_eval.env_capture import capture_environment
from mnlp_eval.languages import Direction, max_source_length_for
from mnlp_eval.models import SegmentOutput, Translator, build_translator
from mnlp_eval.postprocess import FLAG_EXTRA_LINES, extract_hypothesis
from mnlp_eval.prompts import get_prompt
from mnlp_eval.runspec import RunPaths, utc_now
from mnlp_eval.seeding import seed_everything

__all__ = ["DirectionReport", "GenerateReport", "generate"]


@dataclass
class DirectionReport:
    """What happened for one translation direction."""

    direction: str
    n_segments: int
    seconds: float
    generated_tokens: int
    wasted_tokens: int
    empty: int
    truncated: int
    hit_token_budget: int
    source_truncated: int
    max_source_length: int = 0
    parse_flags: dict[str, int] = field(default_factory=dict)
    data: dict[str, Any] = field(default_factory=dict)

    @property
    def tokens_per_second(self) -> float:
        return round(self.generated_tokens / self.seconds, 2) if self.seconds else 0.0

    @property
    def wasted_token_fraction(self) -> float:
        """Share of generated tokens that were discarded as trailing noise."""
        if not self.generated_tokens:
            return 0.0
        return round(self.wasted_tokens / self.generated_tokens, 4)


@dataclass
class GenerateReport:
    """Summary returned to the caller and merged into the manifest."""

    run_id: str
    directions: list[DirectionReport]
    seconds: float
    skipped: list[str] = field(default_factory=list)
    deterministic_check: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": "completed",
            "run_id": self.run_id,
            "seconds": round(self.seconds, 2),
            "skipped_directions": self.skipped,
            "directions": {
                report.direction: {
                    **{
                        key: value
                        for key, value in dataclasses.asdict(report).items()
                        if key != "direction"
                    },
                    "tokens_per_second": report.tokens_per_second,
                    "wasted_token_fraction": report.wasted_token_fraction,
                }
                for report in self.directions
            },
            "deterministic_check": self.deterministic_check,
        }


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def generate(
    config: RunConfig,
    *,
    overwrite: bool = False,
    deterministic_check: int = 0,
    only_directions: Sequence[str] | None = None,
    translator: Translator | None = None,
) -> GenerateReport:
    """Translate every direction in the suite and write the run directory.

    Args:
        config: The model and suite to evaluate.
        overwrite: Regenerate directions whose hypothesis files already exist.
        deterministic_check: If positive, re-translate this many segments of
            the first direction at batch size 1 without length bucketing, and
            report how often the result differs. Quantifies the effect of
            padding on beam search instead of assuming it away.
        only_directions: Generate just these directions of the suite, leaving
            the rest for another process. Used to shard one run across a Slurm
            array. It deliberately does not change the data specification, and
            so does not change the run identity: every task contributes to the
            same run rather than creating a one-direction run of its own.
        translator: A prebuilt translator, used by tests to avoid loading a
            real model.
    """
    paths = RunPaths.for_config(config)
    paths.init_manifest(config)
    atomic_write_json(paths.env, capture_environment())
    seed_everything(config.suite.decode.seed)

    prompt = get_prompt(config.model.prompt)
    owns_translator = translator is None
    started = time.perf_counter()

    _log(f"run {config.run_id} [{config.model.name} | {config.suite.name}]")
    _log(f"  decode: {config.suite.decode.describe()}, batch_size={config.suite.decode.batch_size}")

    if translator is None:
        _log(f"  loading {config.model.model_name_or_path or config.model.entrypoint}")
        translator = build_translator(config.model)
    info = translator.info()
    _log(
        f"  loaded in {info.load_seconds:.1f}s: {info.total_parameters / 1e6:.1f}M params, "
        f"{info.dtype} on {info.device}"
    )

    wanted = set(only_directions) if only_directions else None
    if wanted is not None:
        unknown = sorted(wanted - set(config.suite.data.directions))
        if unknown:
            msg = (
                f"direction(s) {', '.join(unknown)} are not in suite "
                f"{config.suite.name!r}, which covers {', '.join(config.suite.data.directions)}"
            )
            raise ValueError(msg)

    reports: list[DirectionReport] = []
    skipped: list[str] = []
    try:
        for direction in config.suite.data.parsed_directions:
            if wanted is not None and str(direction) not in wanted:
                continue
            if not overwrite and paths.hyps_jsonl(direction).is_file():
                _log(f"  {direction}: already present, skipping")
                skipped.append(str(direction))
                continue
            reports.append(
                _generate_direction(
                    paths, config, translator, direction, prompt.target_marker(direction)
                )
            )

        check: dict[str, Any] | None = None
        if deterministic_check > 0 and config.suite.data.parsed_directions:
            check = _deterministic_check(
                config, translator, config.suite.data.parsed_directions[0], deterministic_check
            )
    finally:
        if owns_translator:
            translator.close()

    report = GenerateReport(
        run_id=config.run_id,
        directions=reports,
        seconds=time.perf_counter() - started,
        skipped=skipped,
        deterministic_check=check,
    )
    payload = report.to_dict()
    payload["model_info"] = dataclasses.asdict(info)
    payload["completed_at"] = utc_now()
    if wanted is None:
        paths.record_stage("generate", payload)
    else:
        # One record per direction, so concurrent array tasks never write the
        # same file and no record can be lost to a read-modify-write race.
        for direction_name in sorted(wanted):
            shard = dict(payload)
            shard["directions"] = {
                name: value
                for name, value in payload["directions"].items()
                if name == direction_name
            }
            paths.record_stage("generate", shard, key=direction_name)
    _log(f"  done in {report.seconds:.1f}s -> {paths.root}")
    return report


def resolve_decode_for_direction(decode: DecodeSpec, direction: Direction) -> DecodeSpec:
    """Return the decode spec as it actually applies to one direction.

    ALMA evaluates every direction at 256 source tokens except zh-en, which it
    reruns at 512. Resolving that here means the manifest records the cap that
    was really used, instead of leaving the exception implicit in the
    translator.
    """
    if not decode.respect_alma_source_length_override:
        return decode
    resolved = max_source_length_for(direction, decode.max_source_length)
    if resolved == decode.max_source_length:
        return decode
    return dataclasses.replace(decode, max_source_length=resolved)


def _generate_direction(
    paths: RunPaths,
    config: RunConfig,
    translator: Translator,
    direction: Direction,
    target_marker: str,
) -> DirectionReport:
    testset = load_testset(config.suite.data, direction)
    resolved_decode = resolve_decode_for_direction(config.suite.decode, direction)
    if resolved_decode.max_source_length != config.suite.decode.max_source_length:
        _log(
            f"  {direction}: source length raised to {resolved_decode.max_source_length} "
            "following ALMA"
        )
    _log(f"  {direction}: {len(testset)} segments of {testset.n_available}")

    started = time.perf_counter()
    outputs = translator.translate(direction, testset.sources, config.suite.decode)
    elapsed = time.perf_counter() - started

    segments: list[Segment] = []
    flags: Counter[str] = Counter()
    for index, (source, reference, output) in enumerate(
        zip(testset.sources, testset.references, outputs, strict=True)
    ):
        parsed = extract_hypothesis(output.raw_text, target_marker)
        flags.update(parsed.flags)
        # The hypothesis was cut only if the budget ran out while still on the
        # hypothesis line. If anything followed it, the line completed and the
        # model simply carried on past the answer.
        truncated = output.hit_token_budget and FLAG_EXTRA_LINES not in parsed.flags
        segments.append(
            Segment(
                index=index,
                source=source,
                reference=reference,
                hypothesis=parsed.hypothesis,
                raw_output=output.raw_text,
                parse_status=parsed.status,
                parse_flags=list(parsed.flags),
                n_source_tokens=output.n_source_tokens,
                n_generated_tokens=output.n_generated_tokens,
                n_wasted_tokens=_wasted_tokens(translator, output, parsed.hypothesis),
                hit_token_budget=output.hit_token_budget,
                truncated=truncated,
                source_truncated=output.source_truncated,
            )
        )

    write_jsonl(paths.hyps_jsonl(direction), segments)
    write_lines(paths.hyps_text(direction), [segment.hypothesis for segment in segments])

    report = DirectionReport(
        direction=str(direction),
        n_segments=len(segments),
        seconds=round(elapsed, 2),
        generated_tokens=sum(segment.n_generated_tokens for segment in segments),
        wasted_tokens=sum(segment.n_wasted_tokens for segment in segments),
        empty=sum(1 for segment in segments if segment.parse_status == "empty"),
        truncated=sum(1 for segment in segments if segment.truncated),
        hit_token_budget=sum(1 for segment in segments if segment.hit_token_budget),
        source_truncated=sum(1 for segment in segments if segment.source_truncated),
        max_source_length=resolved_decode.max_source_length,
        parse_flags=dict(sorted(flags.items())),
        data=testset.provenance(),
    )
    warnings = []
    if report.empty:
        warnings.append(f"{report.empty} empty")
    if report.truncated:
        warnings.append(f"{report.truncated} truncated")
    if report.source_truncated:
        warnings.append(f"{report.source_truncated} source-truncated")
    if report.wasted_token_fraction >= 0.05:
        warnings.append(f"{report.wasted_token_fraction:.0%} tokens discarded")
    suffix = f" [{', '.join(warnings)}]" if warnings else ""
    _log(f"  {direction}: {elapsed:.1f}s, {report.tokens_per_second} tok/s{suffix}")
    return report


def _wasted_tokens(translator: Translator, output: SegmentOutput, hypothesis: str) -> int:
    """Count tokens generated after the hypothesis ended.

    An instruction-tuned model that was not fine-tuned to stop after the
    translation will happily fill the whole token budget with commentary. Those
    tokens cost inference time and contribute nothing, which makes this an
    efficiency signal rather than a quality one, and it is one of the clearer
    ways a poorly distilled student differs from ALMA.
    """
    if not hypothesis or not output.n_generated_tokens:
        return 0
    position = output.raw_text.find(hypothesis)
    if position < 0:
        return 0
    kept = output.raw_text[: position + len(hypothesis)]
    try:
        consumed = len(translator.tokenizer(kept, add_special_tokens=False)["input_ids"])
    except Exception:
        return 0
    return max(0, output.n_generated_tokens - consumed)


def _deterministic_check(
    config: RunConfig,
    translator: Translator,
    direction: Direction,
    limit: int,
) -> dict[str, Any]:
    """Re-translate a slice unbatched and report how often output changes.

    Length-bucketed batching gives every sequence a different amount of
    padding, and beam search is not exactly invariant to that. Rather than
    claiming the effect is negligible, the framework measures it.
    """
    testset = load_testset(config.suite.data, direction)
    sources: Sequence[str] = testset.sources[:limit]
    marker = get_prompt(config.model.prompt).target_marker(direction)

    baseline = config.suite.decode
    reference_outputs = translator.translate(direction, sources, baseline)
    unbatched = dataclasses.replace(baseline, batch_size=1, sort_by_length=False)
    comparison_outputs = translator.translate(direction, sources, unbatched)

    disagreements = 0
    for left, right in zip(reference_outputs, comparison_outputs, strict=True):
        if (
            extract_hypothesis(left.raw_text, marker).hypothesis
            != extract_hypothesis(right.raw_text, marker).hypothesis
        ):
            disagreements += 1
    return {
        "direction": str(direction),
        "n_compared": len(sources),
        "n_different": disagreements,
        "disagreement_rate": round(disagreements / len(sources), 4) if sources else 0.0,
        "reference_setting": {
            "batch_size": baseline.batch_size,
            "sort_by_length": baseline.sort_by_length,
        },
        "comparison_setting": {"batch_size": 1, "sort_by_length": False},
    }
