"""MetricX-24, one of the two WMT24 metrics task winners.

Requires the MetricX environment, which exists because upstream pins
``transformers==4.30.2``. See docs/environments.md.

Batching is implemented here rather than reusing upstream's
``metricx24/predict.py`` pipeline, which routes through ``datasets`` at a
pinned version. Only the model class is imported from upstream, so the pinned
data stack stays out of the hot path.

MetricX-24 predicts an error score in [0, 25]: lower is better. The framework
records that direction explicitly so report tables cannot sort it the wrong
way.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from mnlp_eval.metrics.base import MetricScore

__all__ = ["MAX_INPUT_LENGTH", "clear_model_cache", "score_metricx"]

#: Upstream's recommended cap for MetricX-24.
MAX_INPUT_LENGTH = 1536

_MODEL_CACHE: dict[tuple[str, str], tuple[Any, Any]] = {}


def clear_model_cache() -> None:
    _MODEL_CACHE.clear()
    try:
        import gc

        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:  # pragma: no cover
        pass


def _load(model_name: str, tokenizer_name: str) -> tuple[Any, Any]:
    key = (model_name, tokenizer_name)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    import torch
    import transformers
    from metricx24 import models

    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name, legacy=False)
    # torch_dtype="auto" as upstream's predict.py does. Omitting it upcast the
    # bfloat16 checkpoint to fp32, which drifts scores away from the published
    # values and doubles weight memory to roughly 4.9 GB for the large variant.
    model = models.MT5ForRegression.from_pretrained(model_name, torch_dtype="auto")
    model.eval()
    if torch.cuda.is_available():
        model.to("cuda")
    _MODEL_CACHE[key] = (model, tokenizer)
    return model, tokenizer


def _build_input(source: str, hypothesis: str, reference: str | None) -> str:
    if reference is None:
        return f"source: {source} candidate: {hypothesis}"
    return f"source: {source} candidate: {hypothesis} reference: {reference}"


def score_metricx(
    model_name: str,
    tokenizer_name: str,
    sources: Sequence[str],
    hypotheses: Sequence[str],
    references: Sequence[str] | None = None,
    *,
    batch_size: int = 8,
    max_input_length: int = MAX_INPUT_LENGTH,
) -> MetricScore:
    """Score one direction with MetricX-24."""
    import torch

    if references is not None and len(references) != len(sources):
        msg = f"{len(sources)} sources but {len(references)} references"
        raise ValueError(msg)

    model, tokenizer = _load(model_name, tokenizer_name)
    device = next(model.parameters()).device

    texts = [
        _build_input(source, hypothesis, references[index] if references else None)
        for index, (source, hypothesis) in enumerate(zip(sources, hypotheses, strict=True))
    ]

    segment_scores: list[float] = []
    for start in range(0, len(texts), batch_size):
        chunk = texts[start : start + batch_size]
        encoded = [
            tokenizer(text, max_length=max_input_length, truncation=True, padding=False)
            for text in chunk
        ]
        # Upstream strips the trailing EOS token before feeding the model, and
        # the checkpoints were trained that way. Keeping it shifts scores.
        for item in encoded:
            item["input_ids"] = item["input_ids"][:-1]
            item["attention_mask"] = item["attention_mask"][:-1]
        batch = tokenizer.pad(encoded, padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        segment_scores.extend(round(float(value), 6) for value in output.predictions)

    system_score = sum(segment_scores) / len(segment_scores) if segment_scores else None
    return MetricScore(
        name="metricx24",
        score=round(system_score, 6) if system_score is not None else None,
        signature=f"{model_name}|tok:{tokenizer_name}|maxlen:{max_input_length}",
        segment_scores=segment_scores,
        higher_is_better=False,
        extra={
            "model": model_name,
            "reference_free": references is None,
            "range": "[0, 25], lower is better",
        },
    )
