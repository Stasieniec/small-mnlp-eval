"""Assemble a report from a directory of runs.

Runs are grouped by their measurement conditions, and each group becomes its
own section. A group with more than one member is a comparison; a report
containing more than one group means somebody evaluated systems under different
settings, and saying so loudly is more useful than silently averaging over the
difference.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mnlp_eval.artifacts import atomic_write_json, atomic_write_text, read_jsonl
from mnlp_eval.languages import parse_direction
from mnlp_eval.report.pareto import write_pareto_plots
from mnlp_eval.report.tables import (
    RunSummary,
    Table,
    build_tables,
    collect_runs,
    write_summary_csv,
)
from mnlp_eval.runspec import utc_now

__all__ = ["ReportResult", "build_report"]


@dataclass
class ReportResult:
    """What a report run produced."""

    files: list[Path] = field(default_factory=list)
    groups: int = 0
    runs: int = 0
    warnings: list[str] = field(default_factory=list)


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def build_report(
    runs_root: Path,
    out_dir: Path,
    *,
    baseline: str | None = None,
    formats: tuple[str, ...] = ("md", "csv"),
    bootstrap_samples: int = 1000,
    bootstrap_seed: int = 12345,
    significance: bool = True,
    plots: bool = True,
) -> ReportResult:
    """Aggregate every run under ``runs_root`` into ``out_dir``."""
    summaries = collect_runs(runs_root)
    result = ReportResult(runs=len(summaries))
    if not summaries:
        result.warnings.append(f"no runs with a manifest found under {runs_root}")
        _log(result.warnings[-1])
        return result

    groups: dict[str, list[RunSummary]] = {}
    for summary in summaries:
        groups.setdefault(summary.comparability_key, []).append(summary)
    result.groups = len(groups)
    if len(groups) > 1:
        result.warnings.append(
            f"{len(groups)} distinct measurement settings found among {len(summaries)} runs. "
            "Systems are only compared within a setting, so the report has one section per "
            "setting. Re-run the odd ones out under a single suite to get one table."
        )
        _log(result.warnings[-1])

    if baseline and not any(summary.model == baseline for summary in summaries):
        available = ", ".join(sorted({summary.model for summary in summaries}))
        result.warnings.append(
            f"no run has model name {baseline!r}. Every compression ratio, speedup and "
            f"p-value is computed against the baseline, so all of them are omitted. "
            f"Available systems: {available}"
        )
        _log(result.warnings[-1])

    incomplete = [summary.model for summary in summaries if not summary.generation_complete]
    if incomplete:
        result.warnings.append(
            f"generation is incomplete for: {', '.join(sorted(set(incomplete)))}. Their "
            "scores cover fewer directions than the suite claims."
        )
        _log(result.warnings[-1])

    out_dir.mkdir(parents=True, exist_ok=True)
    sections: list[str] = [
        "# Compression evaluation report",
        "",
        f"Generated {utc_now()} from {len(summaries)} run(s) in `{runs_root}`.",
        "",
    ]
    if result.warnings:
        sections.extend(["## Warnings", ""])
        sections.extend(f"- {warning}" for warning in result.warnings)
        sections.append("")

    all_tables: list[tuple[str, Table]] = []
    for index, (_, members) in enumerate(
        sorted(groups.items(), key=lambda item: item[1][0].comparability_label), start=1
    ):
        label = members[0].comparability_label
        heading = f"## Setting {index}: {label}" if len(groups) > 1 else f"## {label}"
        sections.extend([heading, ""])
        sections.append(f"Systems: {', '.join(sorted(summary.model for summary in members))}")
        sections.append("")

        tables = build_tables(members, baseline=baseline)
        if significance:
            table = _significance_table(
                members,
                baseline=baseline,
                n_samples=bootstrap_samples,
                seed=bootstrap_seed,
            )
            if table is not None:
                tables.append(table)
        for table in tables:
            sections.append(table.to_markdown())
            all_tables.append((f"setting{index}-{_slug(table.title)}", table))

        sections.extend(_missing_notes(members))

    report_path = out_dir / "report.md"
    atomic_write_text(report_path, "\n".join(sections))
    result.files.append(report_path)

    if "csv" in formats:
        for name, table in all_tables:
            target = out_dir / f"{name}.csv"
            atomic_write_text(target, table.to_csv())
            result.files.append(target)
        summary_csv = out_dir / "summary.csv"
        write_summary_csv(summaries, summary_csv)
        result.files.append(summary_csv)
    if "tex" in formats:
        for name, table in all_tables:
            target = out_dir / f"{name}.tex"
            atomic_write_text(target, table.to_latex())
            result.files.append(target)
    if plots:
        for members in groups.values():
            result.files.extend(write_pareto_plots(members, out_dir))

    atomic_write_json(
        out_dir / "report.json",
        {
            "generated_at": utc_now(),
            "runs_root": str(runs_root),
            "n_runs": len(summaries),
            "n_settings": len(groups),
            "baseline": baseline,
            "warnings": result.warnings,
            "runs": [
                {
                    "model": summary.model,
                    "suite": summary.suite,
                    "run_id": summary.run_id,
                    "path": str(summary.paths.root),
                    "setting": summary.comparability_label,
                    "scored_groups": sorted(summary.scores),
                    "has_bench": summary.bench is not None,
                }
                for summary in summaries
            ],
        },
    )
    result.files.append(out_dir / "report.json")
    _log(f"wrote {len(result.files)} file(s) to {out_dir}")
    return result


def _resolve_baseline(members: list[RunSummary], baseline: str | None) -> RunSummary | None:
    """Find the reference run, by explicit name then by declaration."""
    if baseline:
        return next((item for item in members if item.model == baseline), None)
    declared = {item.baseline_name for item in members if item.baseline_name}
    return next((item for item in members if item.model in declared), None)


def _slug(text: str) -> str:
    return "".join(character if character.isalnum() else "-" for character in text.lower()).strip(
        "-"
    )


def _missing_notes(members: list[RunSummary]) -> list[str]:
    lines: list[str] = []
    missing_neural = [summary.model for summary in members if "neural" not in summary.scores]
    missing_bench = [summary.model for summary in members if summary.bench is None]
    if missing_neural:
        lines.append(
            f"COMET scores are missing for: {', '.join(sorted(missing_neural))}. "
            "Run the score stage in the COMET environment, see docs/environments.md."
        )
    if missing_bench:
        lines.append(
            f"Efficiency measurements are missing for: {', '.join(sorted(missing_bench))}. "
            "Run `mnlp-eval bench`."
        )
    if lines:
        return ["", *[f"- {line}" for line in lines], ""]
    return []


def _significance_table(
    members: list[RunSummary],
    *,
    baseline: str | None,
    n_samples: int,
    seed: int,
) -> Table | None:
    """Paired bootstrap of every system against the baseline, per direction."""
    from mnlp_eval.metrics.significance import paired_bootstrap_surface

    # Resolved the same way _find_baseline resolves it in tables.py. The two
    # diverged: a typo'd --baseline made the ratio columns blank while the
    # significance table quietly tested against the spec-declared baseline
    # instead, so the reader compared ratios and p-values from different
    # reference systems.
    reference = _resolve_baseline(members, baseline)
    if reference is None:
        return None

    candidates = [item for item in members if item.run_id != reference.run_id]
    if not candidates:
        return None

    rows: list[list[str]] = []
    for candidate in candidates:
        for direction_name in candidate.directions:
            baseline_path = reference.paths.hyps_jsonl(direction_name)
            candidate_path = candidate.paths.hyps_jsonl(direction_name)
            if not (baseline_path.is_file() and candidate_path.is_file()):
                continue
            baseline_segments = list(read_jsonl(baseline_path))
            candidate_segments = list(read_jsonl(candidate_path))
            if len(baseline_segments) != len(candidate_segments):
                continue
            direction = parse_direction(direction_name)
            surface = paired_bootstrap_surface(
                [segment.hypothesis for segment in baseline_segments],
                [segment.hypothesis for segment in candidate_segments],
                [segment.reference for segment in baseline_segments],
                direction.target,
                n_samples=n_samples,
                seed=seed,
            )
            comet = _comet_delta(
                reference, candidate, direction_name, n_samples=n_samples, seed=seed
            )
            rows.append(
                [
                    candidate.model,
                    direction_name,
                    _delta_cell(surface["metrics"].get("bleu")),
                    _delta_cell(surface["metrics"].get("chrf2pp")),
                    _delta_cell(comet, digits=4),
                ]
            )

    if not rows:
        return None
    return Table(
        title=f"Significance against {reference.model}",
        columns=["System", "Direction", "BLEU delta", "chrF++ delta", "COMET-22 delta"],
        rows=rows,
        notes=[
            f"Paired bootstrap resampling, {n_samples} samples, seed {seed}.",
            "Cells show the delta with its p-value; an asterisk marks p below 0.05.",
            "Surface metrics are recomputed from sufficient statistics on each "
            "resample. COMET uses the per-segment scores stored at scoring time.",
        ],
    )


def _comet_delta(
    reference: RunSummary,
    candidate: RunSummary,
    direction: str,
    *,
    n_samples: int,
    seed: int,
) -> dict[str, Any] | None:
    key = "wmt22_comet_da"
    baseline_scores = _segment_scores(reference, direction, key)
    candidate_scores = _segment_scores(candidate, direction, key)
    if not baseline_scores or not candidate_scores:
        return None
    if len(baseline_scores) != len(candidate_scores):
        return None

    from mnlp_eval.metrics.significance import bootstrap_segment_delta

    try:
        return bootstrap_segment_delta(
            baseline_scores, candidate_scores, n_samples=n_samples, seed=seed
        )
    except (ImportError, ValueError):
        # A missing numpy or a degenerate score list leaves the COMET column
        # blank rather than failing the whole report.
        return None


def _segment_scores(summary: RunSummary, direction: str, metric: str) -> list[float]:
    payload = (
        summary.scores.get("neural", {})
        .get("directions", {})
        .get(direction, {})
        .get("metrics", {})
        .get(metric, {})
    )
    scores = payload.get("segment_scores")
    return list(scores) if scores else []


def _delta_cell(result: dict[str, Any] | None, digits: int = 2) -> str:
    if not result:
        return "-"
    delta = result.get("delta")
    p_value = result.get("p_value")
    if delta is None:
        return "-"
    marker = "*" if result.get("significant_at_0.05") else ""
    p_text = f"p={p_value:.3f}" if isinstance(p_value, int | float) else "p=-"
    return f"{delta:+.{digits}f}{marker} ({p_text})"
