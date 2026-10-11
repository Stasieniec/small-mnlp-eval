"""LLM-Pruner: first-order Taylor importance over coupled weight groups.

Ma et al., NeurIPS 2023. A group scores ``|sum_i w_i * dL/dw_i|`` over all its weights. Under
grouped-query attention, k_proj and v_proj are not pruned and are left out of
the head score. The paper's LoRA step is the separate repair stage.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from mnlp_eval.prune import PruneError
from mnlp_eval.prune.groups import LayerGroups, decoder_layers

__all__ = ["ATTENTION_PROJECTIONS", "FFN_PROJECTIONS", "score"]

#: Projections whose rows or columns a head selection removes.
ATTENTION_PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj")
#: Projections whose rows or columns an FFN channel selection removes.
FFN_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")


def score(
    model: Any,
    batches: Iterable[dict[str, Any]],
    groups: tuple[LayerGroups, ...],
) -> tuple[list[list[float]], list[list[float]]]:
    """Per-layer Taylor scores for attention heads and for FFN channels.

    ``batches`` must carry ``input_ids`` and ``attention_mask``. The loss is
    next-token prediction on the prompts, with padded positions excluded.
    """
    salience = _accumulate_salience(model, batches, groups)

    head_scores: list[list[float]] = []
    channel_scores: list[list[float]] = []
    for group in groups:
        head_scores.append(_head_scores(salience[group.index], group))
        channel_scores.append(_channel_scores(salience[group.index]))
    return head_scores, channel_scores


def _accumulate_salience(
    model: Any, batches: Iterable[dict[str, Any]], groups: tuple[LayerGroups, ...]
) -> list[dict[str, Any]]:
    import torch

    layers = decoder_layers(model)
    targets: list[dict[str, Any]] = []
    for layer in layers:
        modules = {name: getattr(layer.self_attn, name) for name in ATTENTION_PROJECTIONS}
        modules.update({name: getattr(layer.mlp, name) for name in FFN_PROJECTIONS})
        targets.append(modules)

    was_grad = {name: parameter.requires_grad for name, parameter in model.named_parameters()}
    model.requires_grad_(False)
    for modules in targets:
        for module in modules.values():
            module.weight.requires_grad_(True)

    # Accumulate in float32.
    salience: list[dict[str, Any]] = [
        {
            name: torch.zeros_like(module.weight, dtype=torch.float32)
            for name, module in modules.items()
        }
        for modules in targets
    ]

    seen = 0
    try:
        for batch in batches:
            model.zero_grad(set_to_none=True)
            labels = _masked_labels(batch)
            model(**batch, labels=labels).loss.backward()
            for modules, accumulator in zip(targets, salience, strict=True):
                for name, module in modules.items():
                    if module.weight.grad is not None:
                        accumulator[name] += (module.weight * module.weight.grad).float()
            seen += 1
    finally:
        model.zero_grad(set_to_none=True)
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(was_grad.get(name, False))

    if seen == 0:
        msg = "the calibration set produced no batches, so no gradients were accumulated"
        raise PruneError(msg)
    if len(salience) != len(groups):
        msg = f"accumulated {len(salience)} layers of salience for {len(groups)} layers"
        raise PruneError(msg)
    return salience


def _masked_labels(batch: dict[str, Any]) -> Any:
    """Next-token targets, ignoring a target if it or the position predicting it is padding."""
    labels = batch["input_ids"].clone()
    mask = batch.get("attention_mask")
    if mask is None:
        return labels
    predicted_from = mask.roll(1, dims=-1)
    predicted_from[..., 0] = 0
    return labels.masked_fill((mask == 0) | (predicted_from == 0), -100)


def _head_scores(salience: dict[str, Any], group: LayerGroups) -> list[float]:
    # Rows of q_proj and columns of o_proj, both blocked by head.
    total = salience["q_proj"].view(group.num_heads, group.head_dim, -1).sum((1, 2))
    total = total + salience["o_proj"].t().view(group.num_heads, group.head_dim, -1).sum((1, 2))
    if not group.is_grouped_query:
        # Multi-head attention removes a key/value head with its query head.
        for name in ("k_proj", "v_proj"):
            total = total + salience[name].view(group.num_heads, group.head_dim, -1).sum((1, 2))
    return [float(value) for value in total.abs().tolist()]


def _channel_scores(salience: dict[str, Any]) -> list[float]:
    total = salience["gate_proj"].sum(1) + salience["up_proj"].sum(1)
    total = total + salience["down_proj"].sum(0)
    return [float(value) for value in total.abs().tolist()]
