"""Deterministic calibration and repair sets drawn from ALMA's training data.

Each record holds the full ALMA prompt used at evaluation time. Calibration
sets have the same number of segments per direction (Icelandic has 2,009
training pairs, Russian 15,000). Repair sets use ``segments_per_direction: null``
to take every eligible segment, as ALMA's LoRA recipe does. Each set is
written with a manifest and a fingerprint.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mnlp_eval.artifacts import atomic_write_json, write_jsonl
from mnlp_eval.config import ConfigError, _reject_unknown
from mnlp_eval.languages import Direction, pair_language, parse_directions
from mnlp_eval.prompts import get_prompt, prompt_hash

__all__ = [
    "CALIBRATION_SCHEMA_VERSION",
    "CalibrationSpec",
    "build_calibration_set",
]

CALIBRATION_SCHEMA_VERSION = 1

DEFAULT_CALIBRATION_DATASET = "haoranxu/ALMA-Human-Parallel"


@dataclass(frozen=True)
class CalibrationSpec:
    """Which segments to draw, and how many."""

    name: str
    directions: list[str] = field(default_factory=list)
    dataset: str = DEFAULT_CALIBRATION_DATASET
    split: str = "train"
    #: None takes every eligible segment, unbalanced. For repair sets only.
    segments_per_direction: int | None = 128
    seed: int = 1234
    #: Source length range in characters; other segments are skipped.
    min_source_chars: int = 20
    max_source_chars: int = 600
    #: Prompt template, the same one used at evaluation.
    prompt: str = "alma"
    #: Test set whose sources are removed before drawing (used for the repair
    #: set). Removed segments are listed in the manifest.
    exclude_test_sources: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CalibrationSpec:
        payload = dict(data)
        _reject_unknown(cls, payload, "calibration spec")
        if not payload.get("name"):
            msg = "calibration spec: 'name' is required"
            raise ConfigError(msg)
        raw = payload.get("directions")
        if not raw:
            msg = "calibration spec: 'directions' is required"
            raise ConfigError(msg)
        payload["directions"] = sorted(str(item) for item in parse_directions(raw))
        spec = cls(**payload)
        spec.validate()
        return spec

    def validate(self) -> None:
        if self.segments_per_direction is not None and self.segments_per_direction < 1:
            msg = "calibration spec: segments_per_direction must be at least 1"
            raise ConfigError(msg)
        if self.min_source_chars < 0 or self.max_source_chars <= self.min_source_chars:
            msg = (
                "calibration spec: need 0 <= min_source_chars < max_source_chars, got "
                f"{self.min_source_chars} and {self.max_source_chars}"
            )
            raise ConfigError(msg)
        try:
            get_prompt(self.prompt)
        except KeyError as exc:
            msg = f"calibration spec: {exc.args[0]}"
            raise ConfigError(msg) from None
        for direction in self.parsed_directions:
            pair_language(direction)

    @property
    def parsed_directions(self) -> list[Direction]:
        return parse_directions(self.directions)

    @property
    def total_segments(self) -> int | None:
        """Known in advance only for a balanced set."""
        if self.segments_per_direction is None:
            return None
        return self.segments_per_direction * len(self.directions)


def build_calibration_set(
    spec: CalibrationSpec,
    out_dir: Path,
    *,
    contamination_check: str | None = "haoranxu/WMT22-Test",
) -> dict[str, Any]:
    """Write the calibration set described by ``spec`` and return its manifest.

    ``contamination_check`` names a test set to check the drawn sources against;
    None skips the check.
    """
    target = out_dir / spec.name
    target.mkdir(parents=True, exist_ok=True)
    template = get_prompt(spec.prompt)

    directions: dict[str, Any] = {}
    digest = hashlib.sha256()
    digest.update(f"{spec.dataset}\x00{spec.split}\x00{spec.prompt}".encode())

    excluded: dict[str, Any] = {}
    for direction in spec.parsed_directions:
        pairs = _load_pairs(spec, direction)
        if spec.exclude_test_sources:
            test_sources = _test_sources(spec.exclude_test_sources, direction)
            dropped = [pair for pair in pairs if pair[0].strip() in test_sources]
            pairs = [pair for pair in pairs if pair[0].strip() not in test_sources]
            excluded[str(direction)] = {
                "n_removed": len(dropped),
                "removed": [{"source": source, "target": text} for source, text in dropped],
            }
        eligible = [
            (source, target_text)
            for source, target_text in pairs
            if spec.min_source_chars <= len(source) <= spec.max_source_chars
        ]
        wanted = (
            len(eligible) if spec.segments_per_direction is None else spec.segments_per_direction
        )
        if len(eligible) < wanted:
            msg = (
                f"{direction}: only {len(eligible)} of {len(pairs)} segments fall within "
                f"[{spec.min_source_chars}, {spec.max_source_chars}] source characters, "
                f"but {spec.segments_per_direction} are requested. Lower "
                "segments_per_direction, or widen the character range."
            )
            raise ValueError(msg)

        # Seeded per direction so adding a direction does not change the others.
        rng = random.Random(f"{spec.seed}:{direction}")
        chosen = sorted(rng.sample(range(len(eligible)), wanted))
        records = [
            {
                "id": position,
                "direction": str(direction),
                "source": eligible[index][0],
                "target": eligible[index][1],
                "prompt": template.render(direction, eligible[index][0]),
            }
            for position, index in enumerate(chosen)
        ]
        path = target / f"{direction}.jsonl"
        write_jsonl(path, records)

        for record in records:
            digest.update(b"\x00")
            digest.update(str(record["source"]).encode())
            digest.update(b"\x01")
            digest.update(str(record["target"]).encode())
        directions[str(direction)] = {
            "path": str(path),
            "n_segments": len(records),
            "n_eligible": len(eligible),
            "n_available": len(pairs),
        }

    manifest: dict[str, Any] = {
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "name": spec.name,
        "spec": {
            "dataset": spec.dataset,
            "split": spec.split,
            "directions": spec.directions,
            "segments_per_direction": spec.segments_per_direction,
            "seed": spec.seed,
            "min_source_chars": spec.min_source_chars,
            "max_source_chars": spec.max_source_chars,
            "prompt": spec.prompt,
            "exclude_test_sources": spec.exclude_test_sources,
        },
        "prompt_fingerprint": prompt_hash(template),
        "fingerprint": digest.hexdigest()[:16],
        "total_segments": sum(entry["n_segments"] for entry in directions.values()),
        "directions": directions,
    }
    if spec.exclude_test_sources:
        manifest["excluded_test_sources"] = {
            "dataset": spec.exclude_test_sources,
            "total_removed": sum(entry["n_removed"] for entry in excluded.values()),
            "directions": excluded,
        }
    if contamination_check:
        manifest["contamination"] = _check_contamination(spec, target, contamination_check)
    atomic_write_json(target / "calibration.json", manifest)
    return manifest


def _load_pairs(spec: CalibrationSpec, direction: Direction) -> list[tuple[str, str]]:
    """Read one direction out of an ALMA-style parallel corpus.

    Configs are keyed by English-centric pair (``en-de`` and ``de-en`` both
    under ``de-en``) and rows hold a ``translation`` struct.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover
        msg = (
            "building a calibration set needs the 'datasets' package. "
            "Install the generation extra: pip install -e '.[gen]'"
        )
        raise ImportError(msg) from exc

    config = f"{pair_language(direction)}-en"
    loaded = load_dataset(spec.dataset, config, split=spec.split)
    column = "translation" if "translation" in loaded.column_names else config
    if column not in loaded.column_names:
        msg = (
            f"{spec.dataset} config {config!r} has columns {loaded.column_names}, "
            "expected either a 'translation' struct or one named after the pair"
        )
        raise ValueError(msg)

    pairs: list[tuple[str, str]] = []
    for row in loaded[column]:
        source = row.get(direction.source)
        target = row.get(direction.target)
        if source is None or target is None:
            msg = (
                f"{spec.dataset} {config}: a row is missing the {direction.source!r} "
                f"or {direction.target!r} field"
            )
            raise ValueError(msg)
        pairs.append((str(source).strip(), str(target).strip()))
    return pairs


def _test_sources(dataset: str, direction: Direction) -> set[str]:
    """Every source sentence of one test-set direction, stripped. Raises if unreachable."""
    from mnlp_eval.config import DataSpec
    from mnlp_eval.data import load_testset

    testset = load_testset(
        DataSpec.from_dict({"dataset": dataset, "directions": [str(direction)]}), direction
    )
    return {source.strip() for source in testset.sources}


def _check_contamination(spec: CalibrationSpec, target: Path, dataset: str) -> dict[str, Any]:
    """Count calibration sources that also appear in the test set."""
    from mnlp_eval.artifacts import read_jsonl_dicts
    from mnlp_eval.config import DataSpec
    from mnlp_eval.data import load_testset

    results: dict[str, Any] = {"dataset": dataset, "directions": {}, "total_collisions": 0}
    for direction in spec.parsed_directions:
        try:
            testset = load_testset(
                DataSpec.from_dict({"dataset": dataset, "directions": [str(direction)]}),
                direction,
            )
        except (OSError, ValueError, KeyError) as exc:
            results["directions"][str(direction)] = {"checked": False, "reason": str(exc)}
            continue
        test_sources = {source.strip() for source in testset.sources}
        records = read_jsonl_dicts(target / f"{direction}.jsonl")
        collisions = sum(1 for record in records if str(record["source"]).strip() in test_sources)
        results["directions"][str(direction)] = {
            "checked": True,
            "collisions": collisions,
            "n_test_segments": len(testset),
        }
        results["total_collisions"] += collisions
    return results
