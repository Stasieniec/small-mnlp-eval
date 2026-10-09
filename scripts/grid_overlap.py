"""Pack the grid's subnetwork descriptors and measure how much they share.

Every descriptor under ``subnetworks/`` is a few megabytes of kept indices,
about 900 MB for the grid, too much for git. This packs them as boolean keep
masks into one compressed ``subnetworks.npz`` (``<name>/heads``: layers x
heads, ``<name>/ffn``: layers x channels), then reports FFN-channel and head
overlap for the specialised subnetworks of chosen configurations, and between
the two calibration draws of the same configuration.

Overlap is the pooled Jaccard index over layers, with its excess over two
random selections of the same per-layer sizes. Positive excess means the two
criteria runs agree on which units matter beyond what the layer budgets force.

    ./.venv/bin/python scripts/grid_overlap.py --out results/grid-2026-10-06

Load a mask:

    masks = numpy.load("results/grid-2026-10-06/subnetworks.npz")
    ffn = masks["alma-7b-slimgpt40-gen-pair-de/ffn"]  # bool, (32, 11008)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

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
LANGUAGES = ["cs", "de", "is", "ru", "zh"]


def load_masks(path: Path) -> tuple[np.ndarray, np.ndarray]:
    descriptor = json.loads(path.read_text())
    components = descriptor["components"]
    masks = []
    for component in ("attention_heads", "ffn_channels"):
        kept = components[component]["kept"]
        width = int(components[component]["total"])
        layers = len(kept)
        mask = np.zeros((layers, width), dtype=bool)
        for layer, indices in kept.items():
            mask[int(layer), indices] = True
        masks.append(mask)
    return masks[0], masks[1]


def jaccard(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Pooled Jaccard of two keep masks, and its excess over random selection."""
    inter = float(np.logical_and(a, b).sum())
    union = float(np.logical_or(a, b).sum())
    width = a.shape[1]
    sa, sb = a.sum(1).astype(float), b.sum(1).astype(float)
    expected = sa * sb / width
    chance = expected.sum() / (sa + sb - expected).sum()
    value = inter / union if union else 1.0
    return value, value - chance


def matrix_md(names: list[str], labels: list[str], masks: dict, part: str) -> list[str]:
    lines = ["| | " + " | ".join(labels) + " |", "| :--- |" + " ---: |" * len(labels)]
    for row_name, row_label in zip(names, labels, strict=True):
        cells = []
        for col_name in names:
            if col_name == row_name:
                cells.append("-")
                continue
            value, excess = jaccard(masks[row_name][part], masks[col_name][part])
            cells.append(f"{value:.3f} ({excess:+.3f})")
        lines.append(f"| {row_label} | " + " | ".join(cells) + " |")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--subnetworks", default="subnetworks")
    parser.add_argument("--out", default="results/grid-2026-10-06")
    args = parser.parse_args()
    out = Path(args.out)

    names = {
        entry["name"]
        for entry in json.loads(Path("configs/grid/manifest.json").read_text())["models"]
    }
    names |= {
        entry["name"]
        for entry in json.loads(Path("configs/grid/manifest-seed2.json").read_text())["models"]
    }
    masks: dict[str, dict[str, np.ndarray]] = {}
    for name in sorted(names):
        path = Path(args.subnetworks) / f"{name}.json"
        if path.is_file():
            heads, ffn = load_masks(path)
            masks[name] = {"heads": heads, "ffn": ffn}
    np.savez_compressed(
        out / "subnetworks.npz",
        **{f"{name}/{part}": mask for name, parts in masks.items() for part, mask in parts.items()},
    )
    print(f"packed {len(masks)} subnetworks into {out / 'subnetworks.npz'}")

    lines = [
        "# Subnetwork overlap",
        "",
        "Pooled Jaccard index of kept units over the 32 layers, with the excess over two",
        "random selections of the same per-layer sizes in brackets. Masks for every model",
        "are in `subnetworks.npz` (see `scripts/grid_overlap.py`).",
    ]
    for method, calib, pct in [("slimgpt", "gen", 40), ("flap", "ref", 40), ("slimgpt", "ref", 20)]:
        base = f"alma-7b-{method}{pct}-{calib}"
        for part, title in (("ffn", "FFN channels"), ("heads", "attention heads")):
            pair_names = [f"{base}-pair-{lang}" for lang in LANGUAGES]
            if all(name in masks for name in pair_names):
                lines += ["", f"## {method} {calib} {pct}%: pair models, {title}", ""]
                lines += matrix_md(
                    [*pair_names, f"{base}-multi"], [*LANGUAGES, "multi"], masks, part
                )
            dir_names = [f"{base}-dir-{d}" for d in DIRECTIONS]
            if all(name in masks for name in dir_names):
                lines += ["", f"## {method} {calib} {pct}%: direction models, {title}", ""]
                lines += matrix_md(dir_names, DIRECTIONS, masks, part)

    lines += [
        "",
        "## Same configuration, two calibration draws (SlimGPT ref, seed 1234 vs 5678)",
        "",
        "| model | FFN Jaccard (excess) | heads Jaccard (excess) |",
        "| :--- | ---: | ---: |",
    ]
    for name in sorted(n for n in masks if n.startswith("alma-7b-s2-")):
        twin = name.replace("alma-7b-s2-", "alma-7b-", 1)
        if twin not in masks:
            continue
        ffn = jaccard(masks[name]["ffn"], masks[twin]["ffn"])
        heads = jaccard(masks[name]["heads"], masks[twin]["heads"])
        lines.append(
            f"| {twin} | {ffn[0]:.3f} ({ffn[1]:+.3f}) | {heads[0]:.3f} ({heads[1]:+.3f}) |"
        )

    (out / "overlap.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {out / 'overlap.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
