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

__all__ = ["accepts_references", "clear_model_cache", "is_reference_free", "score_comet"]

#: Loading a COMET checkpoint costs tens of seconds and several GB, so models
#: are held across directions within one scoring process.
_MODEL_CACHE: dict[str, Any] = {}

#: Last-resort fallback for a checkpoint that exposes neither its input
#: segments nor requires_references().
_REFERENCE_FREE_MARKERS = ("kiwi", "-qe", "_qe")


def accepts_references(model_name: str, model: Any = None) -> bool:
    """Whether a checkpoint should be given the reference translation.

    ``requires_references()`` is the wrong question, and asking it silently
    downgraded XCOMET to quality estimation. In unbabel-comet 2.2.7,
    ``UnifiedMetric.requires_references()`` returns true only when the model
    was trained on ``["mt", "ref"]`` exactly, meaning it *cannot* work without
    a reference. XCOMET is trained on ``["mt", "src", "ref"]``, so it
    truthfully answers false, and dropping the reference on that basis made it
    take its documented QE fallback branch: a plausible score on the same
    scale, reported under the name of the reference-based metric.

    The right question is whether the model accepts a reference at all, which
    its input segments answer directly.
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
