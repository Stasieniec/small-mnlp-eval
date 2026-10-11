"""COMET-family metrics through the Python API, keeping per-segment scores.

Requires the COMET environment.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from mnlp_eval.metrics.base import MetricScore

__all__ = ["accepts_references", "clear_model_cache", "is_reference_free", "score_comet"]

#: Loaded models, reused across directions.
_MODEL_CACHE: dict[str, Any] = {}

#: Name-based fallback when a checkpoint exposes neither input segments nor
#: requires_references().
_REFERENCE_FREE_MARKERS = ("kiwi", "-qe", "_qe")


def accepts_references(model_name: str, model: Any = None) -> bool:
    """Whether a checkpoint should be given the reference translation.

    Checks ``hparams.input_segments`` first. ``requires_references()`` returns
    false for XCOMET (trained on ``["mt", "src", "ref"]``) in unbabel-comet
    2.2.7, even though it uses a reference.
    """
    segments = getattr(getattr(model, "hparams", None), "input_segments", None)
    if segments is not None:
        return "ref" in segments
    if model is not None and hasattr(model, "requires_references"):
        try:
            return bool(model.requires_references())
        except Exception:
            pass
    lowered = model_name.lower()
    return not any(marker in lowered for marker in _REFERENCE_FREE_MARKERS)


def is_reference_free(model_name: str, model: Any = None) -> bool:
    """Whether a COMET checkpoint scores without a reference translation."""
    return not accepts_references(model_name, model)


def _load(model_name: str) -> Any:
    if model_name in _MODEL_CACHE:
        return _MODEL_CACHE[model_name]
    from comet import download_model, load_from_checkpoint

    # Resolves from the local cache when HF_HUB_OFFLINE is set.
    try:
        checkpoint = download_model(model_name)
    except KeyError as exc:
        # COMET reports any fetch failure as "not supported by COMET".
        import os

        msg = (
            f"COMET could not fetch {model_name}, usually because it is not in the "
            "cache: HF_HUB_OFFLINE="
            f"{os.environ.get('HF_HUB_OFFLINE', 'unset')}, HF_HOME="
            f"{os.environ.get('HF_HOME', 'unset')}. Prefetch it on a login node with the "
            "same HF_HOME: .venv-comet/bin/mnlp-eval prefetch --metrics <metrics.yaml>"
        )
        raise RuntimeError(msg) from exc
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
    """Score one direction with one COMET checkpoint. Empty hypotheses are scored as is."""
    if len(sources) != len(hypotheses):
        msg = f"{len(sources)} sources but {len(hypotheses)} hypotheses"
        raise ValueError(msg)

    model = _load(model_name)
    use_references = accepts_references(model_name, model)
    if use_references and references is None:
        msg = f"{model_name} accepts references but none were supplied"
        raise ValueError(msg)
    if use_references and references is not None and len(references) != len(sources):
        msg = f"{len(sources)} sources but {len(references)} references"
        raise ValueError(msg)

    data: list[dict[str, str]] = []
    for index, (source, hypothesis) in enumerate(zip(sources, hypotheses, strict=True)):
        row = {"src": source, "mt": hypothesis}
        if use_references and references is not None:
            row["ref"] = references[index]
        data.append(row)

    output = model.predict(data, batch_size=batch_size, gpus=gpus, progress_bar=False)
    segment_scores = [round(float(value), 6) for value in output.scores]

    metric_key = model_name.split("/")[-1].lower().replace("-", "_")
    return MetricScore(
        name=metric_key,
        score=round(float(output.system_score), 6),
        signature=f"{model_name}|nrefs:{1 if use_references else 0}|batch:{batch_size}",
        segment_scores=segment_scores,
        extra={"model": model_name, "reference_free": not use_references},
    )
