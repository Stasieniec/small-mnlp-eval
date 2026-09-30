"""FLAP: fluctuation-based adaptive structured pruning.

Scores a group by WIFV, ``Var(X_j) * ||W[:, j]||^2`` summed over its input
channels: a channel that barely moves carries little however large it is,
because a near-constant contribution can be absorbed into a bias.

That is also the compensation. Removing channels loses
``W[:, pruned] @ mean(X)`` from every output, which does not depend on the
input, so adding it as a bias restores the mean output exactly and the method
needs no fine-tuning. Both projections are built ``bias=False``, so one is
created.

Reference: An et al., AAAI 2024.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from mnlp_eval.prune import PruneError
from mnlp_eval.prune.collect import PRUNABLE_INPUTS, InputStats
from mnlp_eval.prune.groups import LayerGroups, decoder_layers
from mnlp_eval.prune.methods import pool_heads

if TYPE_CHECKING:
    from mnlp_eval.prune.compact import LayerPlan

__all__ = ["compensate", "score"]


def score(
    model: Any, stats: dict[Any, InputStats], groups: tuple[LayerGroups, ...]
) -> tuple[list[list[float]], list[list[float]]]:
    """Per-layer WIFV scores for attention heads and for FFN channels."""
    head_scores: list[list[float]] = []
    channel_scores: list[list[float]] = []

    for group in groups:
        head_scores.append(pool_heads(_wifv(stats, model, group.index, "o_proj"), group))
        channel_scores.append(_wifv(stats, model, group.index, "down_proj").tolist())

    return head_scores, channel_scores


def compensate(
    model: Any,
    stats: dict[Any, InputStats],
    groups: tuple[LayerGroups, ...],
    plan: tuple[LayerPlan, ...],
) -> None:
    """Install the bias that replaces what was removed.

    Call before ``compact_model``, while the dense indices still line up. The
    bias is in output space, which input-channel pruning leaves alone, so it
    survives compaction.
    """
    import torch
    from torch import nn

    for group, layer_plan in zip(groups, plan, strict=True):
        kept_rows = set(group.head_rows(layer_plan.heads))
        dropped_heads = [
            channel
            for channel in range(group.num_heads * group.head_dim)
            if channel not in kept_rows
        ]
        kept_channels = set(group.validate_channels(layer_plan.channels))
        dropped_channels = [
            channel for channel in range(group.intermediate) if channel not in kept_channels
        ]

        for suffix, dropped in (("o_proj", dropped_heads), ("down_proj", dropped_channels)):
            if not dropped:
                continue
            projection, entry = _stats_for(stats, model, group.index, suffix)
            index = torch.as_tensor(dropped, device=projection.weight.device)
            # The mean output contributed by the channels about to be removed.
            lost = projection.weight.data.index_select(1, index).float() @ entry.mean.index_select(
                0, index
            )
            existing = projection.bias
            if existing is None:
                projection.bias = nn.Parameter(lost.to(projection.weight.dtype))
            else:
                existing.data = existing.data + lost.to(existing.dtype)


def _wifv(stats: dict[Any, InputStats], model: Any, layer: int, suffix: str) -> Any:
    projection, entry = _stats_for(stats, model, layer, suffix)
    # Column norms: how much of each channel's fluctuation reaches the output.
    return entry.variance * projection.weight.data.float().pow(2).sum(0)


def _stats_for(
    stats: dict[Any, InputStats], model: Any, layer: int, suffix: str
) -> tuple[Any, InputStats]:
    """A layer's projection and the statistics collected on its input."""
    block = decoder_layers(model)[layer]
    projection = getattr(block.self_attn if suffix == "o_proj" else block.mlp, suffix)
    entry = stats.get(projection)
    if entry is None:
        msg = (
            f"no input statistics for layer {layer}'s {suffix}; the calibration pass hooks "
            f"{', '.join(PRUNABLE_INPUTS)}, so it did not run, or ran on another model."
        )
        raise PruneError(msg)
    return projection, entry
