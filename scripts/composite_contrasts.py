"""Contrasts between composite systems that grid_report.py does not test.

A composite translates each direction with a chosen system: the multi model,
or for every direction the pair model of that direction's language. This
compares, per configuration (generated calibration, 30 and 40 percent), the
five pair models with their own-pair LoRA against the multi model with its
LoRA, on COMET-22 and MetricX-24 per segment, with the same paired bootstrap
as the report: positions resampled within each direction, the ten direction
means averaged with equal weight. It also checks how well the pilot ranked the
headline composites, against the full suite.

    ./.venv/bin/python scripts/composite_contrasts.py --suite alma10-greedy \
        --out results/full-2026-10-08 --pilot results/grid-2026-10-06
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
LANGUAGE = {d: d.replace("en-", "").replace("-en", "") for d in DIRECTIONS}
METRICS = {
    "comet": ("scores.neural.json", "wmt22_comet_da"),
    "metricx": ("scores.metricx.json", "metricx24"),
}


def segments(runs: Path, system: str, suite: str, direction: str, metric: str) -> np.ndarray:
    matches = sorted(runs.glob(f"{system}__{suite}__*"))
    if len(matches) != 1:
        msg = f"{system}: expected one {suite} run, found {len(matches)}"
        raise FileNotFoundError(msg)
    filename, key = METRICS[metric]
    payload = json.loads((matches[0] / filename).read_text())
    return np.asarray(payload["directions"][direction]["metrics"][key]["segment_scores"], float)


def bootstrap(a: dict[str, np.ndarray], b: dict[str, np.ndarray], samples: int, seed: int) -> tuple:
    """Macro of (a - b) over directions, stratified paired bootstrap CI and p."""
    rng = np.random.default_rng(seed)
    point = float(np.mean([np.mean(a[d] - b[d]) for d in a]))
    draws = np.zeros(samples)
    for d in a:
        diff = a[d] - b[d]
        index = rng.integers(0, len(diff), size=(samples, len(diff)))
        draws += diff[index].mean(axis=1)
    draws /= len(a)
    low, high = np.percentile(draws, [2.5, 97.5])
    centred = draws - draws.mean()
    p = (np.sum(np.abs(centred) >= abs(point)) + 1) / (samples + 1)
    return point, float(low), float(high), float(p)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--suite", default="alma10-greedy")
    parser.add_argument("--out", default="results/full-2026-10-08")
    parser.add_argument("--pilot", default="results/grid-2026-10-06")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()
    runs = Path(args.runs_root)

    lines = [
        "# Composite contrasts",
        "",
        "## Specialisation after repair",
        "",
        "Five pair models, each LoRA-repaired on its own pair, against the multi model",
        "LoRA-repaired on all ten directions; generated calibration. Macro over the ten",
        "directions; paired bootstrap stratified by direction (1,000 resamples). COMET:",
        "positive favours the pair models; MetricX-24 (lower is better): negative does.",
        "",
        "| method | removed | multi + LoRA COMET | pair + LoRA COMET | difference [95% CI] | p "
        "| multi + LoRA MetricX | pair + LoRA MetricX | difference [95% CI] | p |",
        "| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method, label in (("slimgpt", "SlimGPT"), ("flap", "FLAP")):
        for pct in (30, 40):
            base = f"alma-7b-{method}{pct}-gen"
            cells = []
            for metric in ("comet", "metricx"):
                multi = {
                    d: segments(runs, f"{base}-multi-lora", args.suite, d, metric)
                    for d in DIRECTIONS
                }
                pair = {
                    d: segments(runs, f"{base}-pair-{LANGUAGE[d]}-lora", args.suite, d, metric)
                    for d in DIRECTIONS
                }
                point, low, high, p = bootstrap(pair, multi, args.bootstrap, args.seed)
                digits = 4 if metric == "comet" else 3
                mean_multi = np.mean([multi[d].mean() for d in DIRECTIONS])
                mean_pair = np.mean([pair[d].mean() for d in DIRECTIONS])
                cells += [
                    f"{mean_multi:.{digits}f}",
                    f"{mean_pair:.{digits}f}",
                    f"{point:+.{digits}f} [{low:+.{digits}f}, {high:+.{digits}f}]",
                    "<0.001" if p < 0.001 else f"{p:.3f}",
                ]
            lines.append(f"| {label} | {pct}% | " + " | ".join(cells) + " |")

    # How well the 300-segment pilot predicted the full suite.
    pilot = json.loads((Path(args.pilot) / "results.json").read_text())["headline"]
    full = json.loads((Path(args.out) / "results.json").read_text())["headline"]

    def cells(headline: list) -> dict:
        found = {}
        for row in headline:
            for scope in ("multi", "pair", "dir"):
                comet = (row.get(scope) or {}).get("comet")
                if comet is not None:
                    found[(row.get("method"), row.get("calib"), row.get("sparsity"), scope)] = comet
        return found

    before, after = cells(pilot), cells(full)
    pairs = [(before[key], after[key]) for key in sorted(after) if key in before]
    if pairs:
        x, y = np.asarray(pairs).T
        rank_x, rank_y = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
        spearman = np.corrcoef(rank_x, rank_y)[0, 1]
        lines += [
            "",
            "## Pilot against full suite",
            "",
            f"Headline composites (method x calibration x sparsity x scope, {len(pairs)} cells):",
            f"Pearson r = {np.corrcoef(x, y)[0, 1]:.4f}, Spearman rho = {spearman:.4f},",
            f"mean |full - pilot| = {np.mean(np.abs(y - x)):.4f} COMET, largest "
            f"{np.max(np.abs(y - x)):.4f}.",
        ]
    (Path(args.out) / "composite_contrasts.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
