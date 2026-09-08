"""Quality against cost frontiers.

A compression project's central claim is a trade-off, and a trade-off is easier
to judge as a frontier than as a column of numbers. Two plots: quality against
resident size, and quality against throughput. Points on the upper-left or
upper-right frontier are the ones worth defending in a report.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mnlp_eval.report.tables import RunSummary

__all__ = ["write_pareto_plots"]


def _quality_of(summary: RunSummary) -> tuple[float | None, str]:
    """Prefer COMET-22, fall back to BLEU, so a plot is possible either way."""
    comet = summary.metric("neural", "wmt22_comet_da")
    if comet is not None:
        return comet, "COMET-22"
    return summary.metric("surface", "bleu"), "BLEU"


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
            "x_of": lambda item: _throughput_of(item, batch_size),
            "title": "Quality against decode throughput",
        },
    ]

    for panel in panels:
        points = []
        quality_label = "quality"
        for summary in summaries:
            x_value = panel["x_of"](summary)
            y_value, quality_label = _quality_of(summary)
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
        axes.set_title(panel["title"])
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


def _throughput_of(summary: RunSummary, batch_size: int | None) -> float | None:
    sizes = summary.bench_batch_sizes
    if not sizes:
        return None
    chosen = batch_size if batch_size in sizes else max(sizes)
    value = summary.bench_at(chosen, "generated_tokens_per_second")
    return float(value) if value else None
