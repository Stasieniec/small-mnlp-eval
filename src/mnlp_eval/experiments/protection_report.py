"""Strict, paired analysis of the boundary-protection experiment."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

from mnlp_eval.artifacts import atomic_write_json, read_json, read_jsonl
from mnlp_eval.config import MetricsSpec, load_yaml_config
from mnlp_eval.experiments.layer_protection import CONDITIONS, CONTROL, DENSE
from mnlp_eval.metrics.significance import bootstrap_segment_delta, paired_bootstrap_surface
from mnlp_eval.report.build import build_report
from mnlp_eval.report.tables import RunSummary, Table, collect_runs, write_summary_csv


def _scores(run: RunSummary, direction: str) -> list[float]:
    metric = run.scores["neural"]["directions"][direction]["metrics"]["wmt22_comet_da"]
    values = metric.get("segment_scores", [])
    if not values or not all(isinstance(v, int | float) and math.isfinite(v) for v in values):
        raise ValueError(f"{run.model}/{direction}: missing or invalid per-segment COMET")
    return list(values)


def validate_runs(runs: list[RunSummary], expected: list[str]) -> dict[str, RunSummary]:
    """Reject incomplete, misaligned or differently scored comparisons."""
    by_name = {run.model: run for run in runs}
    if len(by_name) != len(runs) or set(by_name) != set(expected):
        raise ValueError("run set differs from experiment manifest or contains duplicate systems")
    if len({run.comparability_key for run in runs}) != 1:
        raise ValueError("experiment runs have different data, decoding or fingerprints")
    reference = by_name[CONTROL]
    for run in runs:
        if not run.generation_complete or not run.bench:
            raise ValueError(f"{run.model}: generation or benchmark is incomplete")
        if run.bench.get("run_id") != run.run_id:
            raise ValueError(f"{run.model}: benchmark belongs to another run")
        if run.bench.get("protocol") != (reference.bench or {}).get("protocol"):
            raise ValueError(f"{run.model}: benchmark protocol differs from the control")
        for group in ("surface", "neural"):
            payload = run.scores.get(group, {})
            if payload.get("run_id") != run.run_id or set(payload.get("directions", {})) != set(
                run.directions
            ):
                raise ValueError(f"{run.model}: incomplete or stale {group} scores")
            if payload.get("settings") != reference.scores.get(group, {}).get("settings"):
                raise ValueError(f"{run.model}: different {group} metric settings")
        for direction in run.directions:
            left = list(read_jsonl(reference.paths.hyps_jsonl(direction)))
            right = list(read_jsonl(run.paths.hyps_jsonl(direction)))
            if not left or [(r.index, r.source, r.reference) for r in left] != [
                (r.index, r.source, r.reference) for r in right
            ]:
                raise ValueError(f"{run.model}/{direction}: hypotheses are not segment-aligned")
            if len(_scores(run, direction)) != len(right):
                raise ValueError(f"{run.model}/{direction}: COMET count differs from hypotheses")
            for metric in ("bleu", "chrf2pp"):
                entry = run.scores["surface"]["directions"][direction]["metrics"].get(metric, {})
                value = entry.get("score")
                if not isinstance(value, int | float) or not math.isfinite(value):
                    raise ValueError(f"{run.model}/{direction}: missing {metric}")
            signature = run.scores["neural"]["directions"][direction]["metrics"][
                "wmt22_comet_da"
            ].get("signature")
            reference_signature = reference.scores["neural"]["directions"][direction]["metrics"][
                "wmt22_comet_da"
            ].get("signature")
            if signature != reference_signature:
                raise ValueError("COMET signatures differ")
    return by_name


def macro_comet_delta(
    reference: RunSummary, candidate: RunSummary, *, samples: int, seed: int
) -> dict[str, Any]:
    """Paired, stratified bootstrap: equal weight per direction, not per segment."""
    import numpy as np

    rng = np.random.default_rng(seed)
    draws = np.zeros(samples)
    observed = 0.0
    for direction in reference.directions:
        delta = np.asarray(_scores(candidate, direction)) - np.asarray(
            _scores(reference, direction)
        )
        observed += float(delta.mean()) / len(reference.directions)
        indices = rng.integers(0, len(delta), size=(samples, len(delta)))
        draws += delta[indices].mean(axis=1) / len(reference.directions)
    return {
        "delta": observed,
        "ci95": [float(v) for v in np.percentile(draws, [2.5, 97.5])],
        "p_value": float((1 + (np.abs(draws - observed) >= abs(observed)).sum()) / (samples + 1)),
        "samples": samples,
        "seed": seed,
        "method": "paired bootstrap within each direction, unweighted direction mean",
    }


def holm(p_values: dict[str, float]) -> dict[str, float]:
    """Family-wise correction across the four predeclared primary contrasts."""
    adjusted = {}
    previous = 0.0
    for i, (name, value) in enumerate(sorted(p_values.items(), key=lambda item: item[1])):
        previous = max(previous, min(1.0, (len(p_values) - i) * value))
        adjusted[name] = previous
    return adjusted


def check_budgets(root: Path, runs: dict[str, RunSummary]) -> list[dict[str, Any]]:
    """Audit actual tensors' parameter count plus descriptor selections."""
    from mnlp_eval.analysis.subnetwork import load_subnetwork

    rows = []
    matched = []
    for tag, (first, last) in CONDITIONS.items():
        name = f"alma-7b-boundary-{tag}"
        descriptor = load_subnetwork(root / "subnetworks" / f"{name}.json")
        manifest = read_json(root / "checkpoints" / name / "prune.json")
        protected = set(range(first)) | set(range(32 - last, 32))
        if set(manifest["protected_layers"]) != protected:
            raise ValueError(f"{name}: protection metadata disagrees with experiment")
        if set(descriptor.components) != {"attention_heads", "ffn_channels"}:
            raise ValueError(f"{name}: descriptor must contain heads and FFN channels")
        counts = []
        for component in descriptor.components.values():
            if set(component.kept) != {str(i) for i in range(32)}:
                raise ValueError(f"{name}: descriptor does not cover all 32 layers")
            for i in protected:
                if component.kept[str(i)] != tuple(range(component.total)):
                    raise ValueError(f"{name}: protected layer {i} was pruned")
            counts.append(component.n_total - component.n_kept)
        for component_name, key in (
            ("attention_heads", "removed_heads"),
            ("ffn_channels", "removed_channels"),
        ):
            units = descriptor.components[component_name]
            if units.n_total - units.n_kept != manifest[key]:
                raise ValueError(f"{name}: {key} disagrees with its descriptor")
        parameters = runs[name].bench_static("total_parameters")
        if parameters != manifest["parameters_after"]:
            raise ValueError(f"{name}: loaded parameter count differs from pruning manifest")
        matched.append((*counts, parameters))
        rows.append(
            {
                "system": name,
                "protect_first_n": first,
                "protect_last_n": last,
                "removed_heads": manifest["removed_heads"],
                "removed_channels": manifest["removed_channels"],
                "parameters": parameters,
            }
        )
    if len(set(matched)) != 1:
        raise ValueError("protected models do not match the control's realised compression budget")
    return rows


def report(root: Path) -> None:
    manifest = read_json(root / "experiment.json")
    metrics = MetricsSpec.from_dict(load_yaml_config(root / "configs/metrics.yaml"))
    runs = collect_runs(root / "runs")
    by_name = validate_runs(runs, manifest["systems"])
    budgets = check_budgets(root, by_name)
    out = root / "reports"
    out.mkdir(exist_ok=True)
    atomic_write_json(out / "budget-audit.json", {"matched": True, "systems": budgets})
    write_summary_csv(runs, out / "systems.csv")
    for baseline, folder in ((DENSE, "against-dense"), (CONTROL, "against-control")):
        build_report(
            root / "runs",
            out / folder,
            baseline=baseline,
            formats=("md", "csv", "tex"),
            significance=False,
            plots=False,
        )

    reference = by_name[CONTROL]
    comparisons: dict[str, Any] = {}
    direction_rows = []
    for tag in CONDITIONS:
        name = f"alma-7b-boundary-{tag}"
        if name == CONTROL:
            continue
        candidate = by_name[name]
        directions = {}
        for direction in reference.directions:
            left = list(read_jsonl(reference.paths.hyps_jsonl(direction)))
            right = list(read_jsonl(candidate.paths.hyps_jsonl(direction)))
            surface = paired_bootstrap_surface(
                [r.hypothesis for r in left],
                [r.hypothesis for r in right],
                [r.reference for r in left],
                direction.split("-")[1],
                n_samples=metrics.bootstrap_samples,
                seed=metrics.bootstrap_seed,
            )
            comet = bootstrap_segment_delta(
                _scores(reference, direction),
                _scores(candidate, direction),
                n_samples=metrics.bootstrap_samples,
                seed=metrics.bootstrap_seed,
            )
            directions[direction] = {"surface": surface, "comet": comet}
            direction_rows.append(
                {
                    "system": name,
                    "direction": direction,
                    "bleu_delta": surface["metrics"]["bleu"]["delta"],
                    "bleu_p": surface["metrics"]["bleu"]["p_value"],
                    "chrf_delta": surface["metrics"]["chrf2pp"]["delta"],
                    "chrf_p": surface["metrics"]["chrf2pp"]["p_value"],
                    "comet_delta": comet["delta"],
                    "comet_p": comet["p_value"],
                    "comet_ci_low": comet["bootstrap_ci_95"][0],
                    "comet_ci_high": comet["bootstrap_ci_95"][1],
                }
            )
        comparisons[name] = {
            "directions": directions,
            "macro_comet": macro_comet_delta(
                reference, candidate, samples=metrics.bootstrap_samples, seed=metrics.bootstrap_seed
            ),
        }
    corrected = holm(
        {name: result["macro_comet"]["p_value"] for name, result in comparisons.items()}
    )
    rows = []
    for name, result in comparisons.items():
        macro = result["macro_comet"]
        macro["p_holm"] = corrected[name]
        low, high = macro["ci95"]
        rows.append(
            [
                name,
                f"{macro['delta']:+.5f}",
                f"[{low:+.5f}, {high:+.5f}]",
                f"{macro['p_value']:.4f}",
                f"{macro['p_holm']:.4f}",
            ]
        )
    atomic_write_json(out / "comparisons.json", {"control": CONTROL, "comparisons": comparisons})
    with (out / "per-direction-deltas.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(direction_rows[0]))
        writer.writeheader()
        writer.writerows(direction_rows)
    table = Table(
        "Protection versus unprotected uniform control",
        ["System", "Macro COMET delta", "95% CI", "p", "p (Holm)"],
        rows,
        notes=[
            "Positive deltas favour protection. Budgets and loaded parameter counts match.",
            "CI is pointwise; Holm adjusts p-values across four macro COMET comparisons.",
            "Per-direction tests are exploratory and unadjusted.",
            "One calibration seed: uncertainty covers segments, not calibration variability.",
        ],
    )
    (out / "comparison.md").write_text(table.to_markdown(), encoding="utf-8")
    print(f"Complete: {out / 'comparison.md'}")
