"""FLAP: fluctuation-based adaptive structured pruning (An et al., AAAI 2024).

Scores input channels by WIFV, ``Var(X_j) * ||W[:, j]||^2``. Removed channels
are compensated with a bias ``W[:, pruned] @ mean(X)`` on ``o_proj`` and
``down_proj``, so no fine-tuning is needed.

Allocations: ``uniform``, ``global`` and ``log-increase`` (``prune/budget.py``)
budget heads and channels separately. ``al-am`` is FLAP's adaptive structure
search, following ``lib/prune.py`` of CASIA-IVA-Lab/FLAP at commit 3bb57db
(lines 336 and 368-393):

1. ``o_proj`` column WIFV is squared, ``down_proj``'s is not.
2. Columns are standardised per layer and module, ``(s - mean) / std`` with
   Bessel-corrected ``torch.std``.
3. A head scores the mean of its ``head_dim`` standardised columns.
4. All heads and channels of all layers are ranked together. A head weighs
   ``4 * head_dim / 3`` channels (parameter counts: q, k, v, o vs gate, up, down).
5. The ranking is cut where the kept weight is nearest ``1 - sparsity`` of the
   total, so sparsity is the fraction of attention and MLP weights removed.

Differences from the official code (at most a unit or two at ALMA scale): exact
head weight 512/3 instead of the integer 170; the unit at the cut is kept;
a layer is never left with no heads or no channels; ``head_dim`` is read from
the model; variance is exact over non-padding tokens. Grouped-query attention
is not supported under ``al-am``.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from typing import Any

from mnlp_eval.prune import PruneError
from mnlp_eval.prune.collect import PRUNABLE_INPUTS, InputStats
from mnlp_eval.prune.compact import LayerPlan
from mnlp_eval.prune.groups import LayerGroups, decoder_layers
from mnlp_eval.prune.methods import pool_heads

__all__ = [
    "AL_AM",
    "al_am",
    "compensate",
    "require_multi_head",
    "score",
    "search_structure",
    "unit_costs",
]

#: Allocation name of FLAP's adaptive structure search.
AL_AM = "al-am"


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


def al_am(
    model: Any, stats: dict[Any, InputStats], groups: Sequence[LayerGroups], *, sparsity: float
) -> tuple[LayerPlan, ...]:
    """FLAP's adaptive structure search. Follow with :func:`compensate`, then ``compact_model``."""
    require_multi_head(groups)
    attention = [_wifv(stats, model, group.index, "o_proj") for group in groups]
    ffn = [_wifv(stats, model, group.index, "down_proj") for group in groups]
    return search_structure(attention, ffn, groups, sparsity=sparsity)


def search_structure(
    attention_columns: Sequence[Any],
    ffn_columns: Sequence[Any],
    groups: Sequence[LayerGroups],
    *,
    sparsity: float,
) -> tuple[LayerPlan, ...]:
    """Steps 1 to 5 of the module docstring, from per-column WIFV.

    Args:
        attention_columns: Per layer, a tensor holding the WIFV of each
            ``o_proj`` input column, not yet squared.
        ffn_columns: Per layer, a tensor holding the WIFV of each ``down_proj``
            input column.
        groups: The dense model's per-layer unit counts.
        sparsity: Fraction of the attention and MLP weights to remove, in
            [0, 1).

    Returns:
        One :class:`LayerPlan` per layer, holding indices into the dense model.
    """
    import torch

    if not 0.0 <= sparsity < 1.0:
        msg = f"sparsity must be in [0, 1), got {sparsity}. It is the fraction removed."
        raise PruneError(msg)
    require_multi_head(groups)
    if not len(attention_columns) == len(ffn_columns) == len(groups):
        msg = (
            f"scores cover {len(attention_columns)} and {len(ffn_columns)} layers "
            f"but the model has {len(groups)}"
        )
        raise PruneError(msg)

    head_scores, channel_scores = [], []
    for group, attention, ffn in zip(groups, attention_columns, ffn_columns, strict=True):
        attention = _columns(attention, group.num_heads * group.head_dim, group, "o_proj")
        ffn = _columns(ffn, group.intermediate, group, "down_proj")
        standardised = _standardise(attention.pow(2))
        head_scores.append(standardised.view(group.num_heads, group.head_dim).mean(1))
        channel_scores.append(_standardise(ffn))

    # Heads, then channels, layer-major (the official order, matters for ties).
    # ``owner`` and ``local`` map a ranked position back.
    scores = torch.cat(head_scores + channel_scores)
    device = scores.device
    costs = torch.cat(
        [torch.full((group.num_heads,), unit_costs(group)[0]) for group in groups]
        + [torch.full((group.intermediate,), unit_costs(group)[1]) for group in groups]
    ).to(device=device, dtype=torch.int64)
    sizes = [group.num_heads for group in groups] + [group.intermediate for group in groups]
    owner = torch.repeat_interleave(
        torch.arange(len(sizes), device=device), torch.tensor(sizes, device=device)
    )
    starts = torch.tensor([0, *sizes], device=device).cumsum(0)[:-1]
    local = torch.arange(scores.numel(), device=device) - starts[owner]

    total = int(costs.sum())
    target = (1.0 - sparsity) * total
    order = torch.sort(scores, descending=True, stable=True).indices
    keep = torch.zeros_like(scores, dtype=torch.bool)
    keep[order[: _nearest_cut(costs[order], target, base=0)]] = True

    counts = torch.bincount(owner[keep], minlength=len(sizes))
    starved = [index for index, count in enumerate(counts.tolist()) if count == 0]
    if starved:
        # Force each layer's best head and channel, re-cut the rest.
        forced = torch.zeros_like(keep)
        forced[torch.stack([order[owner[order] == unit][0] for unit in range(len(sizes))])] = True
        rest = order[~forced[order]]
        base = int(costs[forced].sum())
        keep = forced.clone()
        keep[rest[: _nearest_cut(costs[rest], target, base=base)]] = True
        layers = len(groups)
        no_heads = [unit for unit in starved if unit < layers]
        no_channels = [unit - layers for unit in starved if unit >= layers]
        found = [f"layers {no_heads} with no heads"] if no_heads else []
        found += [f"layers {no_channels} with no FFN channels"] if no_channels else []
        _log(
            f"al-am at sparsity {sparsity}: the cut left {' and '.join(found)}; kept every "
            "layer's best head and FFN channel and re-cut the rest nearest the budget"
        )

    plan = []
    for index in range(len(groups)):
        heads = local[keep & (owner == index)]
        channels = local[keep & (owner == len(groups) + index)]
        plan.append(LayerPlan(heads=tuple(heads.tolist()), channels=tuple(channels.tolist())))

    kept = int(costs[keep].sum())
    heads_total = sum(group.num_heads for group in groups)
    channels_total = sum(group.intermediate for group in groups)
    _log(
        f"al-am at sparsity {sparsity}: removed {1 - kept / total:.4f} of attention and MLP "
        f"weights, {1 - sum(len(item.heads) for item in plan) / heads_total:.4f} of heads, "
        f"{1 - sum(len(item.channels) for item in plan) / channels_total:.4f} of FFN channels"
    )
    return tuple(plan)


def unit_costs(group: LayerGroups) -> tuple[int, int]:
    """Weight of a head and of an FFN channel in the ranking, in units of ``hidden`` parameters."""
    return 4 * group.head_dim, 3


def require_multi_head(groups: Sequence[LayerGroups]) -> None:
    """Raise if any layer uses grouped-query attention."""
    grouped = [group.index for group in groups if group.is_grouped_query]
    if grouped:
        msg = (
            f"the {AL_AM} allocation is implemented for multi-head attention, and layers "
            f"{grouped[:5]} are grouped-query. A query head there shares its key and value "
            "heads, so it does not weigh 4 * head_dim / 3 FFN channels and the parameter "
            "budget would be wrong. Use uniform, global or log-increase."
        )
        raise PruneError(msg)


def compensate(
    model: Any,
    stats: dict[Any, InputStats],
    groups: tuple[LayerGroups, ...],
    plan: tuple[LayerPlan, ...],
) -> None:
    """Add the compensation bias for the pruned channels. Call before ``compact_model``."""
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


def _nearest_cut(costs: Any, target: float, *, base: int) -> int:
    """Number of ranked units to keep so that ``base`` plus their cost is nearest ``target``.

    Ties go to the shorter prefix.
    """
    import torch

    kept = torch.cat([costs.new_zeros(1), costs.cumsum(0)]) + base
    return int(torch.argmin((kept.double() - target).abs()))


def _columns(values: Any, expected: int, group: LayerGroups, suffix: str) -> Any:
    """One layer's per-column WIFV as a flat float64 tensor."""
    import torch

    flat = torch.as_tensor(values).reshape(-1).double()
    if flat.numel() != expected:
        msg = f"layer {group.index}: {flat.numel()} {suffix} column scores, expected {expected}"
        raise PruneError(msg)
    if not bool(torch.isfinite(flat).all()):
        msg = (
            f"layer {group.index}: {suffix} WIFV is not finite, so the calibration statistics "
            "overflowed or the layer's weights are broken"
        )
        raise PruneError(msg)
    return flat


def _standardise(values: Any) -> Any:
    """``(s - mean) / std`` with Bessel-corrected std. All-equal scores give zeros."""
    spread = values.std() if values.numel() > 1 else values.new_zeros(())
    if not bool(spread > 0):
        return values.new_zeros(values.shape)
    return (values - values.mean()) / spread


def _wifv(stats: dict[Any, InputStats], model: Any, layer: int, suffix: str) -> Any:
    projection, entry = _stats_for(stats, model, layer, suffix)
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


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)
