"""FLAP: fluctuation-based adaptive structured pruning.

Scores a group by WIFV, ``Var(X_j) * ||W[:, j]||^2`` summed over its input
channels: a channel that barely moves carries little however large it is,
because a near-constant contribution can be absorbed into a bias.

That is also the compensation. Removing channels loses
``W[:, pruned] @ mean(X)`` from every output, which does not depend on the
input, so adding it as a bias restores the mean output exactly and the method
needs no fine-tuning. Both projections are built ``bias=False``, so one is
created.

How many units each layer keeps is decided one of two ways. ``uniform``,
``global`` and ``log-increase`` (``prune/budget.py``) take the per-head sums of
:func:`score` and budget heads and channels separately, each to the requested
fraction, as every criterion here is budgeted. ``al-am`` is FLAP's own adaptive
structure search ("adaptive across layers and modules", the default
``--structure`` of the official code), replicated from ``lib/prune.py`` of
CASIA-IVA-Lab/FLAP at commit 3bb57db, lines 336 and 368-393:

1. Each ``o_proj`` input column's WIFV is squared; ``down_proj``'s is not. The
   paper does not mention it; the code does it before anything else.
2. Columns are standardised within one layer and one module,
   ``(s - mean) / std`` over that layer's ``o_proj`` or ``down_proj`` columns
   (4096 and 11008 on ALMA-7B), with the Bessel-corrected ``torch.std``.
3. A head scores the *mean* of its ``head_dim`` standardised columns.
4. Every head and channel of every layer goes into one ranking. A head weighs
   ``4 * head_dim / 3`` channels (512/3 at head_dim 128), their parameter
   counts: a head holds ``hidden * head_dim`` in each of q, k, v and o, a
   channel ``hidden`` in each of gate, up and down.
5. The ranking is cut where the kept weight is nearest ``1 - sparsity`` of the
   total. ``sparsity`` therefore means the fraction of the decoder's attention
   and MLP weights removed, and heads and channels lose different fractions.

Departures, none of which moves more than a unit or two at ALMA scale. The
code's weights are an integer tensor, so a head weighs 170 rather than 512/3;
this uses the exact ratio, so the parameter fraction is the one requested. The
code keeps units scoring strictly above the unit at the nearest cut, which
drops that unit as well; this keeps it. The code would leave a layer with no
heads or no channels; this keeps each layer's best of each and re-cuts the rest
nearest the budget, and says so. The code hardcodes 128 and 512/3; this reads
``head_dim``. The statistics are ``prune/collect.py``'s: the exact variance
over non-padding tokens, where the code's running estimate drops the first
sample's spread. A constant factor in the variance cancels in step 2, squared
or not.

Grouped-query attention is refused under ``al-am``: a query head there does not
own key and value rows, so it does not weigh what step 4 says, and the official
code never handles it.

Reference: An et al., AAAI 2024.
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

#: FLAP's adaptive structure search, by the allocation name a prune config uses.
#: Registered in ``prune.budget.METHOD_ALLOCATIONS``.
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
    """FLAP's adaptive structure search: which heads and channels every layer keeps.

    Follow with :func:`compensate`, then ``compact_model``, as for any plan.
    """
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

    # Heads, then channels, layer-major: the code's order, which decides only
    # exact ties. ``owner`` and ``local`` map a ranked position back.
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
    # Stable, so equal scores rank by position and the result is reproducible.
    order = torch.sort(scores, descending=True, stable=True).indices
    keep = torch.zeros_like(scores, dtype=torch.bool)
    keep[order[: _nearest_cut(costs[order], target, base=0)]] = True

    counts = torch.bincount(owner[keep], minlength=len(sizes))
    starved = [index for index, count in enumerate(counts.tolist()) if count == 0]
    if starved:
        # Each layer's best head and best channel, the first of its units in
        # the ranking; the rest is cut again nearest the same budget.
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
    """What a head and an FFN channel weigh in the ranking, in ``hidden`` parameters.

    Every projection has the unpruned hidden size on one side, so a
    multi-head head holds ``head_dim`` rows or columns in each of q, k, v and o,
    and a channel one in each of gate, up and down. The ratio is FLAP's 512/3
    at head_dim 128; as integers, the cumulative sums are exact.
    """
    return 4 * group.head_dim, 3


def require_multi_head(groups: Sequence[LayerGroups]) -> None:
    """Refuse grouped-query attention, which the ``al-am`` weights do not describe."""
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


def _nearest_cut(costs: Any, target: float, *, base: int) -> int:
    """How many of the ranked units to keep so ``base`` plus their cost is nearest ``target``.

    Keeping none is a candidate too. Ties go to the shorter prefix, the first
    minimum, as ``torch.argmin`` resolves the code's.
    """
    import torch

    kept = torch.cat([costs.new_zeros(1), costs.cumsum(0)]) + base
    return int(torch.argmin((kept.double() - target).abs()))


def _columns(values: Any, expected: int, group: LayerGroups, suffix: str) -> Any:
    """One layer's per-column WIFV as a flat float64 tensor, checked."""
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
    """``(s - mean) / std`` over one layer's columns of one module.

    ``torch.std``, so Bessel-corrected, as the code's. All-equal scores say
    nothing about which unit to drop, and standardise to zeros where the code
    would divide by zero.
    """
    spread = values.std() if values.numel() > 1 else values.new_zeros(())
    if not bool(spread > 0):
        return values.new_zeros(values.shape)
    return (values - values.mean()) / spread


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


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)
