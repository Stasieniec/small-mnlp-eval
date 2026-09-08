"""Stage B: score an existing run directory.

Reads hypotheses from disk and writes one scores file per metric group. Groups
this interpreter cannot serve are recorded as pending rather than raising, so a
scoring job in the generation environment still produces surface metrics and
says plainly what is missing and why.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from typing import Any

from mnlp_eval.artifacts import Segment, atomic_write_json, read_jsonl
from mnlp_eval.config import MetricsSpec
from mnlp_eval.env_capture import package_versions
from mnlp_eval.languages import Direction, parse_direction
from mnlp_eval.metrics.base import METRIC_GROUPS, MetricScore, group_availability
from mnlp_eval.metrics.behaviour import score_behaviour
from mnlp_eval.metrics.surface import score_surface
from mnlp_eval.runspec import RunPaths, utc_now

__all__ = ["SCORES_SCHEMA_VERSION", "score_run"]

SCORES_SCHEMA_VERSION = 1


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def score_run(
    paths: RunPaths,
    metrics: MetricsSpec,
    *,
    groups: Sequence[str] | None = None,
    overwrite: bool = False,
    allow_partial: bool = False,
) -> dict[str, Any]:
    """Score a run and write ``scores.<group>.json`` for each served group."""
    manifest = paths.read_manifest()
    directions = [parse_direction(item) for item in manifest["suite"]["data"]["directions"]]

    # Refuse to score a run that has not finished generating. Scoring the
    # subset that happens to be on disk produced a macro average over fewer
    # directions than the table claimed, recorded it as completed, and then
    # skipped rescoring it as "already scored".
    expected = {str(direction) for direction in directions}
    present = {str(direction) for direction in directions if paths.hyps_jsonl(direction).is_file()}
    if present != expected and not allow_partial:
        missing = ", ".join(sorted(expected - present)) or "none"
        msg = (
            f"{paths.root.name}: generation is incomplete, missing {missing}. "
            "Scoring now would average over fewer directions than the suite claims. "
            "Finish generation, or pass allow_partial to score the subset deliberately."
        )
        raise RuntimeError(msg)

    requested = list(groups) if groups else list(metrics.groups)
    unknown = sorted(set(requested) - set(METRIC_GROUPS))
    if unknown:
        msg = f"unknown metric group(s) {', '.join(unknown)}; known groups are {METRIC_GROUPS}"
        raise ValueError(msg)

    availability = group_availability()
    outcome: dict[str, Any] = {
        "status": "completed",
        "partial": present != expected,
        "scored_directions": sorted(present),
        "groups": {},
    }

    for group in [name for name in METRIC_GROUPS if name in requested]:
        target = paths.scores(group)
        if target.is_file() and not overwrite:
            _log(f"  {group}: already scored, skipping")
            outcome["groups"][group] = {"status": "present", "path": str(target)}
            continue
        served, reason = availability[group]
        if not served:
            _log(f"  {group}: unavailable, {reason}")
            outcome["groups"][group] = {"status": "pending", "reason": reason}
            continue

        _log(f"  {group}: scoring {len(directions)} direction(s)")
        payload = _score_group(paths, metrics, group, directions)
        atomic_write_json(target, payload)
        outcome["groups"][group] = {
            "status": "completed",
            "path": str(target),
            "aggregate": payload["aggregate"],
        }
        _log(f"  {group}: {_format_aggregate(payload['aggregate'])}")

    paths.record_stage("score", outcome)
    return outcome


def _score_group(
    paths: RunPaths,
    metrics: MetricsSpec,
    group: str,
    directions: Sequence[Direction],
) -> dict[str, Any]:
    per_direction: dict[str, Any] = {}
    for direction in directions:
        path = paths.hyps_jsonl(direction)
        if not path.is_file():
            _log(f"    {direction}: no hypotheses, skipping")
            continue
        segments = list(read_jsonl(path))
        if group == "surface":
            per_direction[str(direction)] = _surface_for(segments, direction, metrics)
        elif group == "neural":
            per_direction[str(direction)] = _neural_for(segments, direction, metrics)
        else:
            per_direction[str(direction)] = _metricx_for(segments, direction, metrics)

    return {
        "schema_version": SCORES_SCHEMA_VERSION,
        "group": group,
        "run_id": paths.read_manifest()["run_id"],
        "computed_at": utc_now(),
        "packages": package_versions(),
        "settings": _group_settings(group, metrics),
        "directions": per_direction,
        "aggregate": _macro_average(per_direction),
    }


def _group_settings(group: str, metrics: MetricsSpec) -> dict[str, Any]:
    if group == "surface":
        return {"compute_ter": metrics.compute_ter, "lid_backend": metrics.lid_backend}
    if group == "neural":
        return {
            "comet_models": list(metrics.comet_models),
            "comet_batch_size": metrics.comet_batch_size,
            "comet_gpus": metrics.comet_gpus,
        }
    return {"metricx_model": metrics.metricx_model, "metricx_tokenizer": metrics.metricx_tokenizer}


def _surface_for(
    segments: Sequence[Segment], direction: Direction, metrics: MetricsSpec
) -> dict[str, Any]:
    scores = score_surface(
        [segment.hypothesis for segment in segments],
        [segment.reference for segment in segments],
        direction.target,
        compute_ter=metrics.compute_ter,
    )
    behaviour = score_behaviour(segments, direction, lid_backend=metrics.lid_backend)
    return {
        "n_segments": len(segments),
        "metrics": {name: score.to_dict() for name, score in scores.items()},
        "behaviour": behaviour.to_dict(),
    }


def _neural_for(
    segments: Sequence[Segment], direction: Direction, metrics: MetricsSpec
) -> dict[str, Any]:
    from mnlp_eval.metrics.comet import score_comet

    sources = [segment.source for segment in segments]
    hypotheses = [segment.hypothesis for segment in segments]
    references = [segment.reference for segment in segments]

    scores: dict[str, MetricScore] = {}
    for model_name in metrics.comet_models:
        _log(f"    {direction}: {model_name}")
        score = score_comet(
            model_name,
            sources,
            hypotheses,
            references,
            batch_size=metrics.comet_batch_size,
            gpus=metrics.comet_gpus,
        )
        scores[score.name] = score
    return {
        "n_segments": len(segments),
        "metrics": {name: score.to_dict() for name, score in scores.items()},
    }


def _metricx_for(
    segments: Sequence[Segment], direction: Direction, metrics: MetricsSpec
) -> dict[str, Any]:
    from mnlp_eval.metrics.metricx import score_metricx

    _log(f"    {direction}: {metrics.metricx_model}")
    score = score_metricx(
        metrics.metricx_model,
        metrics.metricx_tokenizer,
        [segment.source for segment in segments],
        [segment.hypothesis for segment in segments],
        [segment.reference for segment in segments],
    )
    return {
        "n_segments": len(segments),
        "metrics": {score.name: score.to_dict()},
    }


def _macro_average(per_direction: dict[str, Any]) -> dict[str, Any]:
    """Unweighted mean over directions, the convention in MT result tables.

    A metric is aggregated only if every scored direction reported it, so a
    partially failed scoring pass cannot produce an average that silently
    covers fewer directions than the table claims.
    """
    if not per_direction:
        return {}

    metric_names: set[str] | None = None
    for payload in per_direction.values():
        names = set(payload.get("metrics", {}))
        metric_names = names if metric_names is None else metric_names & names
    aggregate: dict[str, Any] = {}
    for name in sorted(metric_names or set()):
        values = [
            payload["metrics"][name]["score"]
            for payload in per_direction.values()
            if payload["metrics"][name]["score"] is not None
        ]
        if len(values) == len(per_direction):
            aggregate[name] = round(sum(values) / len(values), 4)

    behaviour_keys = [
        "on_target_rate",
        "off_target_rate",
        "unverifiable_rate",
        "source_language_rate",
        "english_fallback_rate",
        "off_target_rate_among_scorable",
        "empty_rate",
        "source_copy_rate",
        "length_ratio",
        "truncation_rate",
        "budget_hit_rate",
        "repetition_rate",
        "wasted_token_fraction",
    ]
    behaviour_aggregate: dict[str, float] = {}
    for key in behaviour_keys:
        values = [
            payload["behaviour"][key]
            for payload in per_direction.values()
            if payload.get("behaviour", {}).get(key) is not None
        ]
        if values and len(values) == len(per_direction):
            behaviour_aggregate[key] = round(sum(values) / len(values), 4)
    if behaviour_aggregate:
        aggregate["behaviour"] = behaviour_aggregate

    aggregate["n_directions"] = len(per_direction)
    return aggregate


def _format_aggregate(aggregate: dict[str, Any]) -> str:
    parts = [
        f"{name}={value}"
        for name, value in aggregate.items()
        if isinstance(value, int | float) and name != "n_directions"
    ]
    return ", ".join(parts) or "no metrics aggregated"
