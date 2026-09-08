"""Stage D: aggregate runs into tables.

The one rule enforced here rather than documented: runs whose data or decoding
settings differ are never placed in the same table. That is the single easiest
way for a compression comparison to become quietly meaningless, and by week six
of a project nobody remembers which sweep used greedy decoding.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mnlp_eval.artifacts import read_json
from mnlp_eval.config import canonical_json
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
        specification and have been scored on different data, which is exactly
        what happens when a local held-out file is edited between runs.
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
    """Build every table for one comparable group of runs."""
    ordered = _order(summaries, baseline)
    reference = _find_baseline(ordered, baseline)
    return [
        _quality_table(ordered),
        _behaviour_table(ordered),
        _efficiency_table(ordered, reference),
    ]


def _order(summaries: list[RunSummary], baseline: str | None) -> list[RunSummary]:
    """Baseline first, then alphabetical, so tables read consistently."""
    return sorted(summaries, key=lambda item: (item.model != baseline, item.model))


def _find_baseline(summaries: list[RunSummary], baseline: str | None) -> RunSummary | None:
    if baseline:
        for summary in summaries:
            if summary.model == baseline:
                return summary
        return None
    # Fall back to whatever the specs declare as their baseline.
    declared = {summary.baseline_name for summary in summaries if summary.baseline_name}
    for summary in summaries:
        if summary.model in declared:
            return summary
    return None


#: Display names for the neural metric keys, and the order columns appear in.
#: Every metric present gets its own column. An earlier version had a single
#: column headed "COMET-22" that fell back to XCOMET-XL or COMETKiwi when
#: those were the only neural metrics scored, putting different metrics on
#: different scales under one heading with no note.
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


def write_summary_csv(summaries: list[RunSummary], path: Path) -> int:
    """Write one flat row per run, for loading into a notebook or spreadsheet."""
    records: list[dict[str, Any]] = []
    for summary in summaries:
        record: dict[str, Any] = {
            "model": summary.model,
            "suite": summary.suite,
            "run_id": summary.run_id,
            "baseline": summary.baseline_name or "",
            "n_directions": len(summary.directions),
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
