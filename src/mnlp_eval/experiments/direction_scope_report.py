"""Paired target-direction comparisons, with exploratory transfer results."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

from mnlp_eval.artifacts import atomic_write_json, read_json, read_jsonl
from mnlp_eval.experiments.direction_scope import DENSE
from mnlp_eval.experiments.protection_report import _scores, holm
from mnlp_eval.metrics.significance import bootstrap_segment_delta
from mnlp_eval.report.tables import collect_runs, write_summary_csv


def report(root: Path) -> None:
    manifest = read_json(root / "experiment.json")
    runs = collect_runs(root / "runs")
    by_name = {run.model: run for run in runs}
    if len(by_name) != len(runs) or set(by_name) != set(manifest["systems"]):
        raise ValueError("missing or duplicate experiment systems")
    if len({run.comparability_key for run in runs}) != 1:
        raise ValueError("evaluation settings or fingerprints differ")
    reference = by_name[DENSE]
    expected = {d for p in manifest["pairs"] for d in (f"{p}-en", f"en-{p}")}
    for run in runs:
        if not run.generation_complete or set(run.directions) != expected:
            raise ValueError(f"incomplete generation: {run.model}")
        for group in ("surface", "neural"):
            payload = run.scores.get(group, {})
            if (
                payload.get("run_id") != run.run_id
                or set(payload.get("directions", {})) != expected
                or payload.get("settings") != reference.scores.get(group, {}).get("settings")
            ):
                raise ValueError(f"incomplete or incompatible {group} scores: {run.model}")
        for direction in expected:
            left = list(read_jsonl(reference.paths.hyps_jsonl(direction)))
            right = list(read_jsonl(run.paths.hyps_jsonl(direction)))
            if not left or [(r.index, r.source, r.reference) for r in left] != [
                (r.index, r.source, r.reference) for r in right
            ]:
                raise ValueError(f"unaligned evaluation segments: {run.model}/{direction}")
            if len(_scores(run, direction)) != len(right):
                raise ValueError("COMET score count differs from evaluation count")
            for metric in ("bleu", "chrf2pp"):
                value = (
                    run.scores["surface"]["directions"][direction]["metrics"]
                    .get(metric, {})
                    .get("score")
                )
                if not isinstance(value, int | float) or not math.isfinite(value):
                    raise ValueError(f"missing or invalid {metric}: {run.model}/{direction}")
            signature = run.scores["neural"]["directions"][direction]["metrics"][
                "wmt22_comet_da"
            ].get("signature")
            ref_signature = reference.scores["neural"]["directions"][direction]["metrics"][
                "wmt22_comet_da"
            ].get("signature")
            if signature != ref_signature:
                raise ValueError("COMET signatures differ")
    # Uniform allocation fixes every layer's widths; audit realised pruning budgets.
    budgets = []
    for condition in manifest["conditions"]:
        pruning = read_json(root / "checkpoints" / condition["name"] / "prune.json")
        if pruning["calibration_segments"] != manifest["budget"]:
            raise ValueError("realised calibration budgets differ")
        budgets.append(
            {
                "system": condition["name"],
                **{
                    key: pruning[key]
                    for key in ("removed_heads", "removed_channels", "parameters_after")
                },
            }
        )
    if (
        len(
            {
                tuple(row[k] for k in ("removed_heads", "removed_channels", "parameters_after"))
                for row in budgets
            }
        )
        != 1
    ):
        raise ValueError("realised pruning budgets differ")
    scopes = {c["scope"]: c["name"] for c in manifest["conditions"]}
    comparisons: dict[str, Any] = {}
    for pair in manifest["pairs"]:
        for direction in (f"{pair}-en", f"en-{pair}"):
            for control in ("multi", f"pair-{pair}"):
                key = f"{direction}:direction-minus-{control}"
                comparisons[key] = dict(
                    direction=direction,
                    control=scopes[control],
                    candidate=scopes[f"direction-{direction}"],
                    **bootstrap_segment_delta(
                        _scores(by_name[scopes[control]], direction),
                        _scores(by_name[scopes[f"direction-{direction}"]], direction),
                        n_samples=2000,
                        seed=12345,
                    ),
                )
    adjusted = holm({key: value["p_value"] for key, value in comparisons.items()})
    for key, value in comparisons.items():
        value["p_value_holm"] = adjusted[key]
    out = root / "reports"
    out.mkdir(exist_ok=True)
    atomic_write_json(out / "comparisons.json", comparisons)
    atomic_write_json(out / "budget-audit.json", {"matched": True, "systems": budgets})
    write_summary_csv(runs, out / "systems.csv")
    rows = []
    for run in runs:
        for direction in sorted(expected):
            surface = run.scores["surface"]["directions"][direction]["metrics"]
            values = _scores(run, direction)
            rows.append(
                {
                    "system": run.model,
                    "direction": direction,
                    "comet": sum(values) / len(values),
                    "bleu": surface["bleu"]["score"],
                    "chrf": surface["chrf2pp"]["score"],
                }
            )
    with (out / "by-direction.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Direction-specific pruning pilot",
        "",
        "Positive deltas favour direction-only calibration. COMET is the primary metric.",
        "Holm correction covers all direction-versus-pair and direction-versus-multi contrasts.",
        "",
        "| Direction | Control scope | COMET delta | 95% CI | Holm p |",
        "| --- | --- | ---: | --- | ---: |",
    ]
    for key, result in comparisons.items():
        lo, hi = result["bootstrap_ci_95"]
        lines.append(
            f"| {result['direction']} | {key.split('minus-')[1]} | "
            f"{result['delta']:+.5f} | [{lo:+.5f}, {hi:+.5f}] | "
            f"{result['p_value_holm']:.4f} |"
        )
    lines.extend(
        [
            "",
            "One calibration seed; segment bootstrap does not cover calibration variability.",
            "Budgets match segment counts, not token counts; language lengths can differ.",
            "The first evaluation segments are a pilot, not a random sample.",
            "Other directions in by-direction.csv are exploratory transfer measurements.",
        ]
    )
    (out / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
