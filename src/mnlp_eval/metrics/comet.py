"""COMET-family metrics, run through the Python API rather than the CLI.

ALMA scores COMET by shelling out to ``comet-score`` and reading the last line
of stdout with ``tail -n 1``. Using the API instead keeps the per-segment
scores, which is what makes bootstrap significance testing possible without
re-running a 3.5 billion parameter metric model over the whole test set.

Requires the COMET environment. See docs/environments.md.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from mnlp_eval.metrics.base import MetricScore

__all__ = ["clear_model_cache", "is_reference_free", "score_comet"]

#: Loading a COMET checkpoint costs tens of seconds and several GB, so models
#: are held across directions within one scoring process.
_MODEL_CACHE: dict[str, Any] = {}

#: Fallback for checkpoints that do not expose requires_references().
_REFERENCE_FREE_MARKERS = ("kiwi", "-qe", "_qe")


def is_reference_free(model_name: str, model: Any = None) -> bool:
    """Whether a COMET checkpoint scores without a reference translation."""
    if model is not None and hasattr(model, "requires_references"):
        try:
            return not bool(model.requires_references())
        except Exception:
            pass
    lowered = model_name.lower()
    return any(marker in lowered for marker in _REFERENCE_FREE_MARKERS)


def _load(model_name: str) -> Any:
    if model_name in _MODEL_CACHE:
        return _MODEL_CACHE[model_name]
    from comet import download_model, load_from_checkpoint

    # download_model resolves from the local cache when HF_HUB_OFFLINE is set,
    # which is how this runs on compute nodes without network access.
    checkpoint = download_model(model_name)
    model = load_from_checkpoint(checkpoint)
    model.eval()
    _MODEL_CACHE[model_name] = model
    return model


def clear_model_cache() -> None:
    """Drop cached COMET models and free their GPU memory."""
    _MODEL_CACHE.clear()
    try:
        import gc

        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:  # pragma: no cover
        pass


def score_comet(
    model_name: str,
    sources: Sequence[str],
    hypotheses: Sequence[str],
    references: Sequence[str] | None = None,
    *,
    batch_size: int = 32,
    gpus: int = 1,
) -> MetricScore:
    """Score one direction with one COMET checkpoint.

    Empty hypotheses are passed through unchanged rather than filtered. A
    collapsed model should be penalised by the metric, not excused by the
    harness, and the empty rate is reported separately so the cause stays
    visible.
    """
    if len(sources) != len(hypotheses):
        msg = f"{len(sources)} sources but {len(hypotheses)} hypotheses"
        raise ValueError(msg)

    model = _load(model_name)
    reference_free = is_reference_free(model_name, model)
    if not reference_free and references is None:
        msg = f"{model_name} requires references but none were supplied"
        raise ValueError(msg)

    data: list[dict[str, str]] = []
    for index, (source, hypothesis) in enumerate(zip(sources, hypotheses, strict=True)):
        row = {"src": source, "mt": hypothesis}
        if not reference_free and references is not None:
            row["ref"] = references[index]
        data.append(row)

    output = model.predict(data, batch_size=batch_size, gpus=gpus, progress_bar=False)
    segment_scores = [round(float(value), 6) for value in output.scores]

    metric_key = model_name.split("/")[-1].lower().replace("-", "_")
    return MetricScore(
        name=metric_key,
        score=round(float(output.system_score), 6),
        signature=f"{model_name}|nrefs:{0 if reference_free else 1}|batch:{batch_size}",
        segment_scores=segment_scores,
        extra={"model": model_name, "reference_free": reference_free},
    )
