"""Three report figures from the grid's long.csv and subnetwork masks.

- ``repair.png``: COMET before and after LoRA repair, one row per repaired system.
- ``per_direction_40.png``: COMET minus dense per direction at 40 percent, the
  multi model against the matching direction specialist, for each method.
- ``overlap_dir_slimgpt40.png``: FFN-channel Jaccard between the ten SlimGPT
  direction subnetworks at 40 percent.
- ``repair_grid.png``: method x sparsity x scope before and after repair,
  generated calibration (needs the full set of repairs).

    ./.venv/bin/python scripts/grid_figures.py --out results/grid-2026-10-06
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DIRECTIONS = [
    "cs-en",
    "de-en",
    "is-en",
    "ru-en",
    "zh-en",
    "en-cs",
    "en-de",
    "en-is",
    "en-ru",
    "en-zh",
]
LANGUAGE = {d: d.replace("en-", "").replace("-en", "") for d in DIRECTIONS}

# Reference palette (dataviz skill): categorical slots 1-2, light surface, text tokens.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
COLOR = {"slimgpt": "#2a78d6", "flap": "#eb6834"}
BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def style(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def load(out: Path) -> dict[str, dict[str, float]]:
    scores: dict[str, dict[str, float]] = defaultdict(dict)
    with (out / "long.csv").open() as handle:
        for row in csv.DictReader(handle):
            if row["comet"]:
                scores[row["system"]][row["direction"]] = float(row["comet"])
    return scores


def macro(scores: dict, system: str, directions: list[str] = DIRECTIONS) -> float:
    return float(np.mean([scores[system][d] for d in directions]))


def repair_figure(scores: dict, path: Path) -> None:
    dense = macro(scores, "alma-7b")
    rows = [
        ("SlimGPT 20% ref, multi", "slimgpt", "alma-7b-slimgpt20-ref-multi", None),
        ("SlimGPT 30% gen, multi", "slimgpt", "alma-7b-slimgpt30-gen-multi", None),
        ("SlimGPT 40% ref, multi", "slimgpt", "alma-7b-slimgpt40-ref-multi", None),
        ("SlimGPT 40% gen, multi", "slimgpt", "alma-7b-slimgpt40-gen-multi", None),
        ("SlimGPT 40% gen, 5 pair models", "slimgpt", None, "alma-7b-slimgpt40-gen-pair-{}"),
        ("FLAP 40% ref, multi", "flap", "alma-7b-flap40-ref-multi", None),
    ]
    fig, ax = plt.subplots(figsize=(7.6, 3.6), facecolor=SURFACE)
    style(ax)
    ax.grid(axis="y", visible=False)
    for y, (_label, method, multi, pair) in enumerate(reversed(rows)):
        if multi:
            before, after = macro(scores, multi), macro(scores, multi + "-lora")
        else:
            before = float(np.mean([scores[pair.format(LANGUAGE[d])][d] for d in DIRECTIONS]))
            after = float(
                np.mean([scores[pair.format(LANGUAGE[d]) + "-lora"][d] for d in DIRECTIONS])
            )
        color = COLOR[method]
        ax.plot([before, after], [y, y], color=color, linewidth=2, solid_capstyle="round", zorder=2)
        ax.scatter([before], [y], s=46, facecolor=SURFACE, edgecolor=color, linewidth=2, zorder=3)
        ax.scatter([after], [y], s=46, color=color, edgecolor=SURFACE, linewidth=1.5, zorder=3)
        ax.text(after + 0.0025, y, f"{after:.3f}", va="center", fontsize=8.5, color=INK)
        ax.text(
            before - 0.0025, y, f"{before:.3f}", va="center", ha="right", fontsize=8.5, color=INK_2
        )
    ax.axvline(dense, color=INK_2, linewidth=1)
    ax.text(dense, len(rows) - 0.45, f"dense {dense:.4f}", ha="center", fontsize=8.5, color=INK_2)
    ax.set_yticks(range(len(rows)), [row[0] for row in reversed(rows)], color=INK)
    ax.set_xlim(0.715, 0.865)
    ax.set_ylim(-0.6, len(rows) - 0.2)
    ax.set_xlabel(
        "macro COMET-22, ten directions (hollow: pruned, filled: + LoRA)", color=INK_2, fontsize=9
    )
    ax.set_title("LoRA repair", color=INK, fontsize=11, loc="left")
    handles = [
        plt.Line2D([], [], color=COLOR["slimgpt"], marker="o", linewidth=2, label="SlimGPT"),
        plt.Line2D([], [], color=COLOR["flap"], marker="o", linewidth=2, label="FLAP"),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=8.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def repair_grid_figure(scores: dict, path: Path) -> None:
    """Method x sparsity x scope after repair, generated calibration."""
    dense = macro(scores, "alma-7b")
    rows = []
    for method, label in (("slimgpt", "SlimGPT"), ("flap", "FLAP")):
        for pct in (30, 40):
            base = f"alma-7b-{method}{pct}-gen"
            rows.append((f"{label} {pct}%, multi", method, base + "-multi", None))
            rows.append((f"{label} {pct}%, 5 pair models", method, None, base + "-pair-{}"))
    fig, ax = plt.subplots(figsize=(7.6, 4.4), facecolor=SURFACE)
    style(ax)
    ax.grid(axis="y", visible=False)
    for y, (_label, method, multi, pair) in enumerate(reversed(rows)):
        if multi:
            before, after = macro(scores, multi), macro(scores, multi + "-lora")
        else:
            before = float(np.mean([scores[pair.format(LANGUAGE[d])][d] for d in DIRECTIONS]))
            after = float(
                np.mean([scores[pair.format(LANGUAGE[d]) + "-lora"][d] for d in DIRECTIONS])
            )
        color = COLOR[method]
        ax.plot([before, after], [y, y], color=color, linewidth=2, solid_capstyle="round", zorder=2)
        ax.scatter([before], [y], s=46, facecolor=SURFACE, edgecolor=color, linewidth=2, zorder=3)
        ax.scatter([after], [y], s=46, color=color, edgecolor=SURFACE, linewidth=1.5, zorder=3)
        ax.text(after + 0.0025, y, f"{after:.3f}", va="center", fontsize=8.5, color=INK)
        ax.text(
            before - 0.0025, y, f"{before:.3f}", va="center", ha="right", fontsize=8.5, color=INK_2
        )
    ax.axvline(dense, color=INK_2, linewidth=1)
    ax.text(dense, len(rows) - 0.45, f"dense {dense:.4f}", ha="center", fontsize=8.5, color=INK_2)
    ax.set_yticks(range(len(rows)), [row[0] for row in reversed(rows)], color=INK)
    ax.set_xlim(0.715, 0.865)
    ax.set_ylim(-0.6, len(rows) - 0.2)
    ax.set_xlabel(
        "macro COMET-22, ten directions (hollow: pruned, filled: + LoRA)", color=INK_2, fontsize=9
    )
    ax.set_title("LoRA repair, generated calibration", color=INK, fontsize=11, loc="left")
    handles = [
        plt.Line2D([], [], color=COLOR["slimgpt"], marker="o", linewidth=2, label="SlimGPT"),
        plt.Line2D([], [], color=COLOR["flap"], marker="o", linewidth=2, label="FLAP"),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=8.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def per_direction_figure(scores: dict, path: Path) -> None:
    dense = scores["alma-7b"]
    configs = [
        ("SlimGPT, 40%, gen", "slimgpt", "alma-7b-slimgpt40-gen"),
        ("FLAP, 40%, ref", "flap", "alma-7b-flap40-ref"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True, facecolor=SURFACE)
    x = np.arange(len(DIRECTIONS))
    lowest = 0.0
    for ax, (title, method, base) in zip(axes, configs, strict=True):
        style(ax)
        color = COLOR[method]
        multi = [scores[f"{base}-multi"][d] - dense[d] for d in DIRECTIONS]
        own = [scores[f"{base}-dir-{d}"][d] - dense[d] for d in DIRECTIONS]
        for xi, low, high in zip(x, multi, own, strict=True):
            ax.plot([xi, xi], [low, high], color=color, linewidth=2, alpha=0.5, zorder=2)
        ax.scatter(
            x,
            multi,
            s=42,
            facecolor=SURFACE,
            edgecolor=color,
            linewidth=2,
            zorder=3,
            label="multi model",
        )
        ax.scatter(
            x,
            own,
            s=42,
            color=color,
            edgecolor=SURFACE,
            linewidth=1.5,
            zorder=3,
            label="direction specialist",
        )
        ax.axhline(0, color=INK_2, linewidth=1)
        ax.axvline(4.5, color=GRID, linewidth=1)
        ax.text(2, 0.012, "into English", ha="center", fontsize=8.5, color=INK_2)
        ax.text(7, 0.012, "out of English", ha="center", fontsize=8.5, color=INK_2)
        ax.set_xticks(x, DIRECTIONS, rotation=45, ha="right", color=INK)
        ax.set_title(title, color=INK, fontsize=10.5, loc="left")
        ax.grid(axis="x", visible=False)
        lowest = min(lowest, *multi, *own)
    # Shared y: set once both panels are drawn, so neither clips the other.
    axes[0].set_ylim(lowest - 0.015, 0.03)
    axes[0].set_ylabel("COMET-22 minus dense", color=INK_2, fontsize=9)
    axes[0].legend(frameon=False, fontsize=8.5, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def overlap_figure(masks_path: Path, path: Path) -> None:
    from matplotlib.colors import LinearSegmentedColormap

    masks = np.load(masks_path)
    names = [f"alma-7b-slimgpt40-gen-dir-{d}" for d in DIRECTIONS]
    ffn = [masks[f"{name}/ffn"] for name in names]
    matrix = np.full((len(names), len(names)), np.nan)
    for i, a in enumerate(ffn):
        for j, b in enumerate(ffn):
            if i != j:
                matrix[i, j] = np.logical_and(a, b).sum() / np.logical_or(a, b).sum()
    cmap = LinearSegmentedColormap.from_list("blues", BLUES)
    cmap.set_bad(SURFACE)
    fig, ax = plt.subplots(figsize=(6.4, 5.4), facecolor=SURFACE)
    image = ax.imshow(matrix, cmap=cmap, vmin=0.6, vmax=0.9)
    for i in range(len(names)):
        for j in range(len(names)):
            if i != j:
                value = matrix[i, j]
                ax.text(
                    j,
                    i,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    color="#ffffff" if value > 0.78 else INK,
                )
    ax.set_xticks(range(len(names)), DIRECTIONS, rotation=45, ha="right", color=INK, fontsize=9)
    ax.set_yticks(range(len(names)), DIRECTIONS, color=INK, fontsize=9)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_title(
        "FFN channels kept in common (Jaccard)\nSlimGPT direction subnetworks, 40%, gen",
        color=INK,
        fontsize=10.5,
        loc="left",
    )
    bar = fig.colorbar(image, ax=ax, fraction=0.04, pad=0.02)
    bar.ax.tick_params(colors=INK_2, labelsize=8)
    bar.outline.set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="results/grid-2026-10-06")
    parser.add_argument("--masks", default=None, help="subnetworks.npz; default <out>/")
    args = parser.parse_args()
    out = Path(args.out)
    plots = out / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    scores = load(out)
    figures = [
        ("repair.png", lambda path: repair_figure(scores, path)),
        ("repair_grid.png", lambda path: repair_grid_figure(scores, path)),
        ("per_direction_40.png", lambda path: per_direction_figure(scores, path)),
        (
            "overlap_dir_slimgpt40.png",
            lambda path: overlap_figure(Path(args.masks or out / "subnetworks.npz"), path),
        ),
    ]
    for name, draw in figures:
        # A figure whose systems are not all there yet is skipped, not fatal.
        try:
            draw(plots / name)
        except (KeyError, FileNotFoundError) as exc:
            print(f"skipped {name}: missing {exc}")
            continue
        print(f"wrote {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
