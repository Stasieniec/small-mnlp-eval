"""Stage D: aggregate runs into tables.

Runs whose data or decoding settings differ are never placed in the same
table. Grouping is by measurement conditions, not by convention.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mnlp_eval.artifacts import read_json
from mnlp_eval.config import CompressionSpec, canonical_json
from mnlp_eval.languages import ALMA_PARALLEL_TRAIN_PAIRS, pair_language, resource_tier
from mnlp_eval.runspec import RunPaths, discover_runs

__all__ = ["RunSummary", "Table", "build_tables", "collect_runs", "write_summary_csv"]


@dataclass
class RunSummary:
    """One run, flattened into the pieces a report needs."""

    paths: RunPaths
    manifest: dict[str, Any]
    scores: dict[str, dict[str, Any]] = field(default_factory=dict)
    bench: dict[str, Any] | None = None
    #: Stage records, keyed by stage then by shard.
    stage_records: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def generation_complete(self) -> bool:
        return self.paths.stage_completed("generate")

    @property
    def model(self) -> str:
        return str(self.manifest["model"]["name"])

    @property
    def suite(self) -> str:
        return str(self.manifest["suite"]["name"])

    @property
    def run_id(self) -> str:
        return str(self.manifest["run_id"])

    @property
    def baseline_name(self) -> str | None:
        value = self.manifest["model"].get("baseline")
        return str(value) if value else None

    @property
    def directions(self) -> list[str]:
        return list(self.manifest["suite"]["data"]["directions"])

    @property
    def data_fingerprints(self) -> dict[str, str]:
        """Per-direction digest of the exact segments scored.

        Recorded by the generate stage. The data specification names a dataset
        and a limit; the fingerprint is the content. Two runs can share a
        specification and have been scored on different data, for example when
        a local held-out file is edited between runs.
        """
        fingerprints: dict[str, str] = {}
        for entry in self.stage_records.get("generate", {}).values():
            for direction, payload in (entry.get("directions") or {}).items():
                digest = (payload.get("data") or {}).get("fingerprint")
                if digest:
                    fingerprints[direction] = str(digest)
        return dict(sorted(fingerprints.items()))

    @property
    def comparability_key(self) -> str:
        """Identity of the measurement conditions, ignoring the model.

        Two runs sharing this key were measured the same way and may be
        compared. Two runs that do not share it may not.

        Includes the data fingerprints, not just the data specification.
        Without them, a baseline scored on a local file and a candidate scored
        after that file was edited grouped into one table with no warning, and
        a seven-BLEU "compression regression" was entirely a data change.
        """
        identity = self.manifest["identity"]
        return canonical_json(
            {
                "data": identity["data"],
                "decode": identity["decode"],
                "data_fingerprints": self.data_fingerprints,
            }
        )

    @property
    def comparability_label(self) -> str:
        decode = self.manifest["identity"]["decode"]
        data = self.manifest["identity"]["data"]
        limit = f", limit={data['limit']}" if data.get("limit") else ""
        count = len(data["directions"])
        noun = "direction" if count == 1 else "directions"
        return (
            f"{data['dataset']} [{count} {noun}{limit}], "
            f"{self.manifest.get('decode_summary', '')}, "
            f"batch_size={decode['batch_size']}, sort_by_length={decode['sort_by_length']}"
        )

    def aggregate(self, group: str) -> dict[str, Any]:
        return dict(self.scores.get(group, {}).get("aggregate") or {})

    def metric(self, group: str, name: str) -> float | None:
        value = self.aggregate(group).get(name)
        return float(value) if isinstance(value, int | float) else None

    def behaviour(self, name: str) -> float | None:
        value = self.aggregate("surface").get("behaviour", {}).get(name)
        return float(value) if isinstance(value, int | float) else None

    def per_direction(self, group: str, direction: str, name: str) -> float | None:
        payload = self.scores.get(group, {}).get("directions", {}).get(direction, {})
        value = payload.get("metrics", {}).get(name, {}).get("score")
        return float(value) if isinstance(value, int | float) else None

    @property
    def compression(self) -> CompressionSpec:
        """The system's declared compression, defaulting to uncompressed.

        Read from the manifest rather than from the config file, so a report
        built months later describes what was actually run.
        """
        payload = self.manifest["model"].get("compression")
        if not isinstance(payload, dict):
            return CompressionSpec()
        return CompressionSpec.from_dict(payload)

    @property
    def structure(self) -> dict[str, Any] | None:
        """Measured layer structure, from the bench stage."""
        if not self.bench:
            return None
        payload = self.bench.get("structure")
        if not isinstance(payload, dict) or "unavailable" in payload:
            return None
        return payload

    @property
    def main_stack(self) -> dict[str, Any] | None:
        """The layer stack holding the most parameters.

        A decoder-only model has exactly one. An encoder-decoder model has two,
        and the decoder is the one whose width dominates generation cost.
        """
        structure = self.structure
        if not structure:
            return None
        stacks = structure.get("stacks") or {}
        if not stacks:
            return None
        largest: dict[str, Any] = max(stacks.values(), key=lambda stack: stack.get("parameters", 0))
        return largest

    def bench_static(self, name: str) -> Any:
        if not self.bench:
            return None
        return self.bench.get("static", {}).get(name)

    def bench_at(self, batch_size: int, name: str) -> Any:
        if not self.bench:
            return None
        return self.bench.get("by_batch_size", {}).get(str(batch_size), {}).get(name)

    @property
    def bench_batch_sizes(self) -> list[int]:
        if not self.bench:
            return []
        return sorted(int(key) for key in self.bench.get("by_batch_size", {}))


def collect_runs(root: Path) -> list[RunSummary]:
    """Load every run under ``root`` that has a manifest."""
    summaries: list[RunSummary] = []
    for paths in discover_runs(root):
        manifest = paths.read_manifest()
        summary = RunSummary(
            paths=paths,
            manifest=manifest,
            stage_records={
                stage: paths.stage_records(stage) for stage in ("generate", "bench", "score")
            },
        )
        for group in ("surface", "neural", "metricx"):
            path = paths.scores(group)
            if path.is_file():
                summary.scores[group] = read_json(path)
        if paths.bench.is_file():
            summary.bench = read_json(paths.bench)
        summaries.append(summary)
    return summaries


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


@dataclass
class Table:
    """A rendered table, in whichever format the caller wants."""

    title: str
    columns: list[str]
    rows: list[list[str]]
    notes: list[str] = field(default_factory=list)

    def to_markdown(self) -> str:
        lines = [f"### {self.title}", ""]
        if not self.rows:
            lines.append("No data.")
            return "\n".join(lines) + "\n"
        lines.append("| " + " | ".join(self.columns) + " |")
        lines.append("| " + " | ".join("---" for _ in self.columns) + " |")
        lines.extend("| " + " | ".join(row) + " |" for row in self.rows)
        if self.notes:
            lines.append("")
            lines.extend(f"- {note}" for note in self.notes)
        return "\n".join(lines) + "\n"

    def to_csv(self) -> str:
        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow(self.columns)
        writer.writerows(self.rows)
        return buffer.getvalue()

    def to_latex(self) -> str:
        alignment = "l" + "r" * (len(self.columns) - 1)
        lines = [
            r"\begin{table}[t]",
            r"\centering",
            rf"\begin{{tabular}}{{{alignment}}}",
            r"\toprule",
            " & ".join(_latex_escape(column) for column in self.columns) + r" \\",
            r"\midrule",
        ]
        lines.extend(" & ".join(_latex_escape(cell) for cell in row) + r" \\" for row in self.rows)
        lines.extend(
            [
                r"\bottomrule",
                r"\end{tabular}",
                rf"\caption{{{_latex_escape(self.title)}}}",
                r"\end{table}",
            ]
        )
        return "\n".join(lines) + "\n"


def _latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    return text


def _fmt(value: Any, digits: int = 2, *, percent: bool = False) -> str:
    if value is None:
        return "-"
    if percent:
        return f"{float(value) * 100:.1f}%"
    return f"{float(value):.{digits}f}"


def _human_bytes(value: Any) -> str:
    if value is None:
        return "-"
    size = float(value)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if abs(size) < 1024 or unit == "GiB":
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} GiB"


def _ratio(baseline: Any, candidate: Any) -> str:
    """Compression ratio, baseline over candidate. Higher means smaller."""
    if not baseline or not candidate:
        return "-"
    return f"{float(baseline) / float(candidate):.2f}x"


# --------------------------------------------------------------------------
# Table builders
# --------------------------------------------------------------------------


def build_tables(
    summaries: list[RunSummary],
    *,
    baseline: str | None = None,
) -> list[Table]:
    """Build every table for one comparable group of runs.

    The last three answer the project's research questions directly: what the
    pruning actually removed, whether low-resource directions suffer more, and
    whether a subnetwork selected for one language pair still works on the
    others. They are omitted when the runs present cannot support them, so a
    quantization-only comparison does not carry three empty tables.
    """
    ordered = _order(summaries, baseline)
    reference = _find_baseline(ordered, baseline)
    tables = [
        _quality_table(ordered),
        _behaviour_table(ordered),
        _efficiency_table(ordered, reference),
    ]
    for optional in (
        _structure_table(ordered, reference),
        _resource_tier_table(ordered, reference),
        _transfer_table(ordered),
    ):
        if optional is not None:
            tables.append(optional)
    return tables


def _order(summaries: list[RunSummary], baseline: str | None) -> list[RunSummary]:
    """Baseline first, then alphabetical, so tables read consistently."""
    return sorted(summaries, key=lambda item: (item.model != baseline, item.model))


def _find_baseline(summaries: list[RunSummary], baseline: str | None) -> RunSummary | None:
    """Find the reference run, by explicit name then by declaration.

    An explicit ``--baseline`` that matches nothing returns None rather than
    falling back, so a typo shows up as blank ratio columns plus a warning from
    the report, rather than as ratios against an unrequested system.
    """
    if baseline:
        return next((item for item in summaries if item.model == baseline), None)
    declared = {summary.baseline_name for summary in summaries if summary.baseline_name}
    return next((item for item in summaries if item.model in declared), None)


#: Display names for the neural metric keys, and the order columns appear in.
#: Every metric present gets its own column, because they are on different
#: scales and one heading cannot stand for two of them.
_NEURAL_COLUMNS: tuple[tuple[str, str, str], ...] = (
    ("neural", "wmt22_comet_da", "COMET-22"),
    ("neural", "wmt22_cometkiwi_da", "COMETKiwi (ref-free)"),
    ("neural", "xcomet_xl", "XCOMET-XL"),
    ("neural", "xcomet_xxl", "XCOMET-XXL"),
    ("metricx", "metricx24", "MetricX-24 (lower better)"),
)


def _present_neural_columns(
    summaries: list[RunSummary],
) -> list[tuple[str, str, str]]:
    return [
        entry
        for entry in _NEURAL_COLUMNS
        if any(summary.metric(entry[0], entry[1]) is not None for summary in summaries)
    ]


def _quality_table(summaries: list[RunSummary]) -> Table:
    directions = summaries[0].directions if summaries else []
    neural = _present_neural_columns(summaries)
    columns = [
        "System",
        *[f"BLEU {d}" for d in directions],
        "BLEU avg",
        "chrF++ avg",
        *[label for _, _, label in neural],
    ]

    rows: list[list[str]] = []
    for summary in summaries:
        row = [
            summary.model,
            *[_fmt(summary.per_direction("surface", d, "bleu")) for d in directions],
            _fmt(summary.metric("surface", "bleu")),
            _fmt(summary.metric("surface", "chrf2pp")),
        ]
        row.extend(_fmt(summary.metric(group, key), 4) for group, key, _ in neural)
        rows.append(row)

    notes = [
        "Averages are unweighted macro means over directions.",
        "sacreBLEU signatures are recorded per direction in scores.surface.json.",
    ]
    if len(neural) > 1:
        notes.append(
            "Each neural metric has its own column. They are on different scales and "
            "must not be compared with one another, only across systems within a column."
        )
    if any(_fmt(summary.metric(g, k), 4) == "-" for summary in summaries for g, k, _ in neural):
        notes.append(
            "A dash means that metric was not scored for that system, not that it "
            "scored zero. Score every system with the same metrics config before "
            "quoting a comparison."
        )
    return Table(title="Translation quality", columns=columns, rows=rows, notes=notes)


def _behaviour_table(summaries: list[RunSummary]) -> Table:
    return Table(
        title="Behavioural failure modes",
        columns=[
            "System",
            "On-target",
            "Off-target",
            "Unverifiable",
            "Source language",
            "Empty",
            "Source copy",
            "Repetition",
            "Truncated",
            "Length ratio",
            "Tokens discarded",
        ],
        rows=[
            [
                summary.model,
                _fmt(summary.behaviour("on_target_rate"), percent=True),
                _fmt(summary.behaviour("off_target_rate"), percent=True),
                _fmt(summary.behaviour("unverifiable_rate"), percent=True),
                _fmt(summary.behaviour("source_language_rate"), percent=True),
                _fmt(summary.behaviour("empty_rate"), percent=True),
                _fmt(summary.behaviour("source_copy_rate"), percent=True),
                _fmt(summary.behaviour("repetition_rate"), percent=True),
                _fmt(summary.behaviour("truncation_rate"), percent=True),
                _fmt(summary.behaviour("length_ratio"), 3),
                _fmt(summary.behaviour("wasted_token_fraction"), percent=True),
            ]
            for summary in summaries
        ],
        notes=[
            "On-target, off-target and unverifiable are shares of all segments and sum "
            "to 100%. Unverifiable means the hypothesis was empty or too short to "
            "identify a language from, so a collapsed system shows up there rather "
            "than scoring a flattering 0% off-target.",
            "Language identification uses a classifier restricted to the target, the "
            "source and English, which is far more reliable on single sentences than an "
            "open choice among every language it knows.",
            "Source language is the part of off-target identified as the source, which "
            "means the model failed to translate rather than translating badly. For an "
            "out-of-English direction it is the same event as an English fallback, so "
            "only one of the two is reported.",
            "Tokens discarded is the share of generated tokens that followed the "
            "hypothesis and were thrown away. It costs inference time without "
            "affecting quality.",
        ],
    )


def _efficiency_table(summaries: list[RunSummary], reference: RunSummary | None) -> Table:
    batch_sizes = sorted({size for summary in summaries for size in summary.bench_batch_sizes})
    columns = [
        "System",
        "Params",
        "Disk",
        "Resident",
        "Bits/param",
        "Compression (disk)",
        "Compression (params)",
        "Compression (VRAM)",
    ]
    for size in batch_sizes:
        columns.extend([f"tok/s b{size}", f"Speedup b{size}", f"Peak VRAM b{size}"])
    columns.append("Prefill ms")

    rows: list[list[str]] = []
    for summary in summaries:
        row = [
            summary.model,
            f"{float(summary.bench_static('total_parameters') or 0) / 1e9:.2f}B"
            if summary.bench_static("total_parameters")
            else "-",
            _human_bytes(summary.bench_static("checkpoint_bytes")),
            _human_bytes(summary.bench_static("resident_weight_bytes")),
            _fmt(summary.bench_static("bits_per_parameter_resident")),
            _ratio(
                reference.bench_static("checkpoint_bytes") if reference else None,
                summary.bench_static("checkpoint_bytes"),
            ),
            _ratio(
                reference.bench_static("total_parameters") if reference else None,
                summary.bench_static("total_parameters"),
            ),
            _ratio(
                reference.bench_static("resident_weight_bytes") if reference else None,
                summary.bench_static("resident_weight_bytes"),
            ),
        ]
        for size in batch_sizes:
            throughput = summary.bench_at(size, "generated_tokens_per_second")
            baseline_throughput = (
                reference.bench_at(size, "generated_tokens_per_second") if reference else None
            )
            speedup = (
                f"{float(throughput) / float(baseline_throughput):.2f}x"
                if throughput and baseline_throughput
                else "-"
            )
            row.extend(
                [
                    _fmt(throughput, 1),
                    speedup,
                    _human_bytes(summary.bench_at(size, "peak_allocated_bytes")),
                ]
            )
        prefill = (summary.bench or {}).get("time_to_first_token", {}).get("median_ms")
        row.append(_fmt(prefill, 1))
        rows.append(row)

    return Table(
        title="Efficiency",
        columns=columns,
        rows=rows,
        notes=[
            "Compression ratios are baseline over system, so higher means smaller. "
            "Disk, parameter and VRAM ratios are reported separately because they "
            "diverge: load-time quantization leaves the checkpoint untouched.",
            "Throughput is measured with a fixed token budget, so a model is not "
            "credited for stopping early. Medians over repeated timed runs.",
            "Batch size 1 is memory-bandwidth bound and larger batches are compute "
            "bound, which is why both are reported.",
        ],
    )


# --------------------------------------------------------------------------
# Tables for the pruning research questions
# --------------------------------------------------------------------------


def _quality_metric(summaries: list[RunSummary]) -> tuple[str, str, str] | None:
    """One metric for every derived table, chosen once for the whole group.

    Shared with the Pareto plots so a report never states a transfer gap in
    COMET and draws the frontier in BLEU without saying so.
    """
    from mnlp_eval.report.pareto import choose_quality_metric

    return choose_quality_metric(summaries)


def _structure_table(summaries: list[RunSummary], reference: RunSummary | None) -> Table | None:
    """What structured pruning removed, per layer rather than in total.

    Parameter count alone cannot distinguish a model that lost a third of
    every FFN from one that lost eight whole layers, and the two behave very
    differently. Omitted entirely when no run measured its structure.
    """
    measured = [summary for summary in summaries if summary.main_stack]
    if not measured:
        return None

    reference_stack = reference.main_stack if reference else None
    rows: list[list[str]] = []
    for summary in summaries:
        stack = summary.main_stack
        compression = summary.compression
        if stack is None:
            rows.append([summary.model, compression.describe(), *["-"] * 7])
            continue
        attention = stack.get("attention_inner_dim") or {}
        ffn = stack.get("ffn_intermediate") or {}
        structure = summary.structure or {}
        rows.append(
            [
                summary.model,
                compression.describe(),
                str(stack.get("n_layers", "-")),
                _range(attention),
                _range(ffn),
                "yes" if stack.get("uniform") else "no",
                _kept(reference_stack, stack, "n_layers"),
                _kept(reference_stack, stack, "ffn_intermediate", nested="total"),
                _fmt(structure.get("zero_fraction"), percent=True),
            ]
        )

    notes = [
        "Measured from the loaded weights, not from the config that produced them, "
        "so a checkpoint whose config and tensors disagree shows up here.",
        "Widths are the inner dimensions of the layer stack holding the most "
        "parameters: the input width of the attention output projection, and of the "
        "FFN down projection.",
        "Uniform means every layer kept the same width. A no is the interesting "
        "answer: it means the pruning criterion spent its budget unevenly across "
        "depth, which a single sparsity number hides.",
        "Zero weights counts floating-point parameters that are exactly zero. A "
        "structurally pruned checkpoint should read near zero here, because its "
        "removed channels are gone rather than masked. A high number means the mask "
        "was applied but the weights were never compacted, so none of the claimed "
        "memory or latency saving is real yet.",
    ]
    if reference is None:
        notes.append("No baseline run was resolved, so the kept-fraction columns are blank.")
    return Table(
        title="Structure",
        columns=[
            "System",
            "Compression",
            "Layers",
            "Attention width",
            "FFN width",
            "Uniform",
            "Layers kept",
            "FFN width kept",
            "Zero weights",
        ],
        rows=rows,
        notes=notes,
    )


def _range(spread: dict[str, Any]) -> str:
    low, high = spread.get("min"), spread.get("max")
    if low is None:
        return "-"
    return str(low) if low == high else f"{low} to {high}"


def _kept(
    reference: dict[str, Any] | None,
    stack: dict[str, Any],
    key: str,
    *,
    nested: str | None = None,
) -> str:
    """Share of a structural quantity the system kept, relative to the baseline."""
    if reference is None:
        return "-"

    def value(source: dict[str, Any]) -> float | None:
        raw = source.get(key)
        if nested and isinstance(raw, dict):
            raw = raw.get(nested)
        return float(raw) if isinstance(raw, int | float) else None

    base, candidate = value(reference), value(stack)
    if not base or candidate is None:
        return "-"
    return f"{candidate / base:.1%}"


def _resource_tier_table(summaries: list[RunSummary], reference: RunSummary | None) -> Table | None:
    """Whether compression costs more in low-resource directions.

    The macro average over ten directions is dominated by the eight
    high-resource ones, so a system that has lost Icelandic entirely can still
    look mildly degraded. Splitting the average by resource tier is the whole
    of RQ2, and it is one subtraction away from data the report already has.
    """
    chosen = _quality_metric(summaries)
    if chosen is None or reference is None or len(summaries) < 2:
        return None
    group, key, label = chosen

    tiers: dict[str, list[str]] = {}
    for direction in reference.directions:
        try:
            tier = resource_tier(pair_language(direction))
        except ValueError:
            continue
        tiers.setdefault(tier, []).append(direction)
    if len(tiers) < 2:
        return None

    ordered_tiers = [tier for tier in ("high", "low", "unknown") if tier in tiers]
    rows: list[list[str]] = []
    for summary in summaries:
        if summary.run_id == reference.run_id:
            continue
        means: dict[str, float | None] = {}
        for tier in ordered_tiers:
            deltas: list[float] = []
            for direction in tiers[tier]:
                base = reference.per_direction(group, direction, key)
                candidate = summary.per_direction(group, direction, key)
                if base is not None and candidate is not None:
                    deltas.append(candidate - base)
            means[tier] = sum(deltas) / len(deltas) if deltas else None
        low, high = means.get("low"), means.get("high")
        gap = low - high if low is not None and high is not None else None
        rows.append(
            [
                summary.model,
                summary.compression.describe(),
                *[_signed(means[tier], group) for tier in ordered_tiers],
                _signed(gap, group),
            ]
        )
    if not rows:
        return None

    tier_note = "; ".join(
        f"{tier}: {', '.join(sorted({pair_language(d) for d in directions}))}"
        for tier, directions in sorted(tiers.items())
    )
    counts = ", ".join(
        f"{language} {count:,}" for language, count in sorted(ALMA_PARALLEL_TRAIN_PAIRS.items())
    )
    return Table(
        title=f"Degradation by resource tier ({label})",
        columns=[
            "System",
            "Compression",
            *[f"{tier} resource" for tier in ordered_tiers],
            "Low minus high",
        ],
        rows=rows,
        notes=[
            f"Mean change in {label} against {reference.model}, averaged within a tier.",
            f"Tiers by language: {tier_note}.",
            "A tier is set by the parallel training pairs ALMA has for that language "
            f"in haoranxu/ALMA-Human-Parallel ({counts}), because that is the data "
            "available for calibrating a pruning criterion and for repair.",
            "A negative low-minus-high figure means compression cost the "
            "low-resource directions more than the high-resource ones.",
        ],
    )


def _transfer_table(summaries: list[RunSummary]) -> Table | None:
    """How a pair-specific subnetwork does on the pairs it was not selected for.

    Needs at least one system declaring ``compression.pruned_for``. Matched
    cells are bracketed, so a reader can see at a glance whether the diagonal
    stands out from the rest of its row.
    """
    specific = [summary for summary in summaries if summary.compression.target_directions()]
    if not specific:
        return None
    chosen = _quality_metric(summaries)
    if chosen is None:
        return None
    group, key, label = chosen

    directions = summaries[0].directions
    rows: list[list[str]] = []
    for summary in summaries:
        targets = set(summary.compression.target_directions())
        cells: list[str] = []
        matched: list[float] = []
        mismatched: list[float] = []
        for direction in directions:
            value = summary.per_direction(group, direction, key)
            if value is None:
                cells.append("-")
                continue
            cells.append(f"[{value:.4g}]" if direction in targets else f"{value:.4g}")
            (matched if direction in targets else mismatched).append(value)
        matched_mean = sum(matched) / len(matched) if matched else None
        mismatched_mean = sum(mismatched) / len(mismatched) if mismatched else None
        gap = (
            matched_mean - mismatched_mean
            if matched_mean is not None and mismatched_mean is not None
            else None
        )
        rows.append(
            [
                summary.model,
                "multi" if not targets else "+".join(sorted(targets)),
                *cells,
                _fmt(matched_mean, 4),
                _fmt(mismatched_mean, 4),
                _signed(gap, group),
            ]
        )

    return Table(
        title=f"Cross-direction transfer ({label})",
        columns=[
            "System",
            "Selected for",
            *directions,
            "Matched",
            "Mismatched",
            "Specialization",
        ],
        rows=rows,
        notes=[
            "A bracketed cell is a direction the subnetwork was selected for. "
            "Specialization is the matched mean minus the mismatched mean.",
            "A multi-directional system has no matched directions, so its "
            "specialization is blank. It belongs in the table as the row every "
            "pair-specific system should be read against, column by column.",
            "Comparing specialization across systems is only meaningful when their "
            "sparsity matches, because a deeper cut lowers every column at once.",
        ],
    )


def _signed(value: float | None, group: str) -> str:
    """Format a delta, keeping the sign and using the metric's own precision."""
    if value is None:
        return "-"
    digits = 2 if group == "surface" else 4
    return f"{value:+.{digits}f}"


def write_summary_csv(summaries: list[RunSummary], path: Path) -> int:
    """Write one flat row per run, for loading into a notebook or spreadsheet."""
    records: list[dict[str, Any]] = []
    for summary in summaries:
        compression = summary.compression
        stack = summary.main_stack or {}
        record: dict[str, Any] = {
            "model": summary.model,
            "suite": summary.suite,
            "run_id": summary.run_id,
            "baseline": summary.baseline_name or "",
            "n_directions": len(summary.directions),
            "compression_family": compression.family,
            "compression_method": compression.method,
            "nominal_sparsity": compression.nominal_sparsity,
            "pruned_for": "multi"
            if compression.is_multi_directional
            else "+".join(compression.target_directions()),
            "repair": compression.repair,
            "n_layers": stack.get("n_layers"),
            "uniform_layers": stack.get("uniform"),
            "ffn_intermediate_mean": (stack.get("ffn_intermediate") or {}).get("mean"),
            "attention_inner_dim_mean": (stack.get("attention_inner_dim") or {}).get("mean"),
            "zero_fraction": (summary.structure or {}).get("zero_fraction"),
            "bleu": summary.metric("surface", "bleu"),
            "chrf2pp": summary.metric("surface", "chrf2pp"),
            "comet22": summary.metric("neural", "wmt22_comet_da"),
            "cometkiwi": summary.metric("neural", "wmt22_cometkiwi_da"),
            "xcomet_xl": summary.metric("neural", "xcomet_xl"),
            "metricx24": summary.metric("metricx", "metricx24"),
        }
        for name in (
            "on_target_rate",
            "off_target_rate",
            "unverifiable_rate",
            "off_target_rate_among_scorable",
            "source_language_rate",
            "english_fallback_rate",
            "empty_rate",
            "source_copy_rate",
            "repetition_rate",
            "truncation_rate",
            "budget_hit_rate",
            "length_ratio",
            "wasted_token_fraction",
        ):
            record[name] = summary.behaviour(name)
        for name in (
            "total_parameters",
            "non_embedding_parameters",
            "checkpoint_bytes",
            "resident_weight_bytes",
            "bits_per_parameter_resident",
            "load_seconds",
        ):
            record[name] = summary.bench_static(name)
        for size in summary.bench_batch_sizes:
            record[f"tokens_per_second_b{size}"] = summary.bench_at(
                size, "generated_tokens_per_second"
            )
            record[f"peak_allocated_bytes_b{size}"] = summary.bench_at(size, "peak_allocated_bytes")
            record[f"latency_per_sentence_ms_b{size}"] = summary.bench_at(
                size, "latency_per_sentence_ms"
            )
        for direction in summary.directions:
            record[f"bleu_{direction}"] = summary.per_direction("surface", direction, "bleu")
            record[f"chrf2pp_{direction}"] = summary.per_direction("surface", direction, "chrf2pp")
            record[f"comet22_{direction}"] = summary.per_direction(
                "neural", direction, "wmt22_comet_da"
            )
        records.append(record)

    columns: list[str] = []
    for record in records:
        for key in record:
            if key not in columns:
                columns.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, restval="")
        writer.writeheader()
        writer.writerows(records)
    return len(records)
