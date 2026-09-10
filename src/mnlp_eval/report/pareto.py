"""Quality against cost frontiers.

Two plots: quality against resident size, and quality against throughput.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mnlp_eval.report.tables import RunSummary

__all__ = ["write_pareto_plots"]


#: Quality metrics a frontier can be drawn against, best first.
_QUALITY_PREFERENCE: tuple[tuple[str, str, str], ...] = (
    ("neural", "wmt22_comet_da", "COMET-22"),
    ("neural", "xcomet_xl", "XCOMET-XL"),
    ("neural", "wmt22_cometkiwi_da", "COMETKiwi"),
    ("surface", "bleu", "BLEU"),
)


def choose_quality_metric(
    summaries: list[RunSummary],
) -> tuple[str, str, str] | None:
    """Pick one metric for the y-axis, shared by every point on the plot.

    Chosen once for the whole plot rather than per system. Falling back per
    system put a COMET value near 0.85 and a BLEU value near 30 on the same
    axis, pinned the COMET point visually at zero, and labelled the axis after
    whichever system happened to be plotted last.

    Preference order favours the metric the most systems share, so one system
    missing COMET does not drag the whole plot down to BLEU.
    """
    best: tuple[int, tuple[str, str, str]] | None = None
    for entry in _QUALITY_PREFERENCE:
        covered = sum(1 for summary in summaries if summary.metric(entry[0], entry[1]) is not None)
        if covered == len(summaries) and summaries:
            return entry
        if covered and (best is None or covered > best[0]):
            best = (covered, entry)
    return best[1] if best else None


def write_pareto_plots(
    summaries: list[RunSummary],
    out_dir: Path,
    *,
    batch_size: int | None = None,
) -> list[Path]:
    """Write the two frontier plots. Returns the files written.

    Returns an empty list when matplotlib is unavailable, since a missing plot
    should never fail a report.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    written: list[Path] = []
    out_dir.mkdir(parents=True, exist_ok=True)

    chosen = choose_quality_metric(summaries)
    if chosen is None:
        return []
    group, key, quality_label = chosen
    plotted = [summary for summary in summaries if summary.metric(group, key) is not None]
    skipped = [summary.model for summary in summaries if summary not in plotted]

    # A batch size present in every system, so the throughput axis compares
    # like with like rather than one system at batch 8 against another at 1.
    shared_batch = _shared_batch_size(plotted, batch_size)
    panels: list[dict[str, Any]] = [
        {
            "filename": "pareto-quality-vs-size.png",
            "x_label": "Resident weight size (GiB)",
            "x_of": lambda item: _to_gib(item.bench_static("resident_weight_bytes")),
            "title": "Quality against resident model size",
        },
        {
            "filename": "pareto-quality-vs-throughput.png",
            "x_label": "Decode throughput (tokens per second)",
            "x_of": lambda item: _throughput_of(item, shared_batch),
            "title": "Quality against decode throughput",
        },
    ]

    for panel in panels:
        points = []
        for summary in plotted:
            x_value = panel["x_of"](summary)
            y_value = summary.metric(group, key)
            if x_value is None or y_value is None:
                continue
            points.append((x_value, y_value, summary.model))
        if not points:
            continue

        figure, axes = plt.subplots(figsize=(7, 5))
        for x_value, y_value, label in points:
            axes.scatter(x_value, y_value, s=70)
            axes.annotate(
                label,
                (x_value, y_value),
                textcoords="offset points",
                xytext=(6, 5),
                fontsize=8,
            )
        axes.set_xlabel(panel["x_label"])
        axes.set_ylabel(quality_label)
        title = str(panel["title"])
        if skipped:
            title += f" (omits {len(skipped)} system(s) without {quality_label})"
        axes.set_title(title)
        axes.grid(visible=True, alpha=0.3)
        figure.tight_layout()
        target = out_dir / str(panel["filename"])
        figure.savefig(target, dpi=150)
        plt.close(figure)
        written.append(target)

    return written


def _to_gib(value: Any) -> float | None:
    if not value:
        return None
    return float(value) / (1024**3)


def _shared_batch_size(summaries: list[RunSummary], requested: int | None) -> int | None:
    """Return a batch size every benched system measured, if there is one."""
    benched = [set(summary.bench_batch_sizes) for summary in summaries if summary.bench]
    if not benched:
        return requested
    common = set.intersection(*benched)
    if requested in common:
        return requested
    return max(common) if common else None


def _throughput_of(summary: RunSummary, batch_size: int | None) -> float | None:
    if batch_size is None or batch_size not in summary.bench_batch_sizes:
        return None
    value = summary.bench_at(batch_size, "generated_tokens_per_second")
    return float(value) if value else None
