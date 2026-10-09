"""Repair results on the directions each repaired model is for.

``grid_report.py`` compares a repaired system with its source over all ten
directions, which is right for the multi models and wrong for a pair model
repaired on its own pair. This reads its ``long.csv`` and reports, per
repaired system, the macro over the directions it was pruned for (all ten for
multi), and assembles the five repaired pair models of one configuration into
a ten-direction composite, comparable with the multi model and its repair.

    ./.venv/bin/python scripts/repair_summary.py --out results/grid-2026-10-06
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

METRICS = ("comet", "bleu", "chrf")
#: Reported where every cell has it; lower is better (MetricX-24 error score).
OPTIONAL = ("metricx",)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="results/grid-2026-10-06")
    args = parser.parse_args()
    out = Path(args.out)

    rows = list(csv.DictReader((out / "long.csv").open()))
    scores: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    meta: dict[str, dict[str, str]] = {}
    for row in rows:
        values = {
            metric: float(row[metric])
            for metric in (*METRICS, *OPTIONAL)
            if row.get(metric) not in ("", None)
        }
        scores[row["system"]][row["direction"]] = values
        meta[row["system"]] = row

    def macro(system: str, directions: list[str]) -> dict[str, float] | None:
        cells = [scores[system].get(direction) for direction in directions]
        if any(cell is None or any(m not in cell for m in METRICS) for cell in cells):
            return None
        return mean_of(cells)  # type: ignore[arg-type]

    def mean_of(cells: list[dict[str, float]]) -> dict[str, float]:
        present = [m for m in (*METRICS, *OPTIONAL) if all(m in cell for cell in cells)]
        return {m: sum(cell[m] for cell in cells) / len(cells) for m in present}

    def fmt(value: dict[str, float] | None, metric: str, digits: int) -> str:
        return f"{value[metric]:.{digits}f}" if value and metric in value else "-"

    all_directions = sorted({row["direction"] for row in rows})
    lines = [
        "# Repair on the directions each model is for",
        "",
        "Macro over the repaired model's own directions (all ten for multi; its pair for a",
        "pair model, which is repaired on its pair's data only). Recovered = the share of",
        "the gap between pruned and dense that repair closes, on COMET and on MetricX-24",
        "(lower is better, so its gap is pruned minus dense).",
        "",
        "| Repaired system | Directions | Dense COMET | Pruned COMET | Repaired COMET "
        "| Recovered | Pruned BLEU | Repaired BLEU | Dense MetricX | Pruned MetricX "
        "| Repaired MetricX | Recovered (MetricX) |",
        "| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    repaired = sorted(name for name in scores if name.endswith("-lora"))
    for name in repaired:
        source = name[: -len("-lora")]
        own = (
            meta[source]["pruned_directions"].replace(";", "+").split("+") if source in meta else []
        )
        own = (
            [d for d in own if d]
            if meta.get(source, {}).get("scope") != "multi"
            else all_directions
        )
        dense, pruned, fixed = macro("alma-7b", own), macro(source, own), macro(name, own)
        if not (dense and pruned and fixed):
            lines.append(f"| {name} | {', '.join(own)} | incomplete | | | | | | | | | |")
            continue
        gap = dense["comet"] - pruned["comet"]
        recovered = (fixed["comet"] - pruned["comet"]) / gap if gap > 0 else float("nan")
        mx = "-"
        if all("metricx" in value for value in (dense, pruned, fixed)):
            mx_gap = pruned["metricx"] - dense["metricx"]
            if mx_gap > 0:
                mx = f"{(pruned['metricx'] - fixed['metricx']) / mx_gap:.0%}"
        scope = "all ten" if len(own) == len(all_directions) else ", ".join(own)
        lines.append(
            f"| {name} | {scope} | {dense['comet']:.4f} | {pruned['comet']:.4f} | "
            f"{fixed['comet']:.4f} | {recovered:.0%} | {pruned['bleu']:.2f} | "
            f"{fixed['bleu']:.2f} | {fmt(dense, 'metricx', 3)} | {fmt(pruned, 'metricx', 3)} | "
            f"{fmt(fixed, 'metricx', 3)} | {mx} |"
        )

    # Five repaired pair models of one configuration, each on its own pair.
    composites = defaultdict(dict)
    for name in repaired:
        source = name[: -len("-lora")]
        info = meta.get(source, {})
        if info.get("scope") != "pair":
            continue
        config = (info["method"], info["calib"], info["sparsity"])
        for direction in [d for d in info["pruned_directions"].replace(";", "+").split("+") if d]:
            composites[config][direction] = (source, name)
    for config, cells in sorted(composites.items()):
        if len(cells) < len(all_directions):
            continue
        method, calib, sparsity = config
        prefix = f"alma-7b-{method}{sparsity}-{calib}"
        pair_plain = [scores[cells[d][0]][d] for d in all_directions]
        pair_fixed = [scores[cells[d][1]][d] for d in all_directions]
        lines += [
            "",
            f"## {method} {calib} {sparsity}%: ten-direction composites",
            "",
            "Each direction translated by the system named in the row; pair rows use the",
            "matching pair model for each direction.",
            "",
            "| System | COMET | BLEU | chrF++ | MetricX-24 |",
            "| :--- | ---: | ---: | ---: | ---: |",
        ]
        candidates = [
            ("dense ALMA-7B", macro("alma-7b", all_directions)),
            (f"{prefix}-multi", macro(f"{prefix}-multi", all_directions)),
            (f"{prefix}-multi-lora", macro(f"{prefix}-multi-lora", all_directions)),
            ("pair models", mean_of(pair_plain)),
            ("pair models + own-pair LoRA", mean_of(pair_fixed)),
        ]
        for label, value in candidates:
            if value:
                cells = f"{value['comet']:.4f} | {value['bleu']:.2f} | {value['chrf']:.2f}"
                lines.append(f"| {label} | {cells} | {fmt(value, 'metricx', 3)} |")

    (out / "repair.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
