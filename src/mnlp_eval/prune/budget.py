"""Turning importance scores into a selection under a sparsity budget.

``uniform`` gives every layer the same fraction, and is the only allocation
that leaves every layer a width a stock ``LlamaConfig`` can describe.
``global`` standardises within each layer, pools, and takes the best overall.
``log-increase`` is SlimGPT's Incremental Pruning Ratio: the fraction removed
rises logarithmically with depth, so early layers, whose errors every later
layer inherits, lose the least.

Standardising before pooling is not optional: raw importance scales with
activation magnitude, which grows with depth. It also means the allocation
follows the *shape* of a layer's score distribution, not its level, since every
layer comes out zero-mean. A layer loses more when it holds units clearly worse
than the rest of its own, not when all of its units score badly.

Heads and channels are budgeted separately. FLAP's own search, ``al-am``, pools
the two and trades a head against its parameter count in channels; it needs
FLAP's per-column scores, so it lives in ``prune/methods/flap.py`` and is
refused here.

Pure Python over floats, so this is testable without torch.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence

from mnlp_eval.prune import PruneError
from mnlp_eval.prune.compact import LayerPlan
from mnlp_eval.prune.groups import LayerGroups

__all__ = [
    "ALLOCATIONS",
    "FIRST_LAYER_SHARE",
    "METHOD_ALLOCATIONS",
    "allocate",
    "keep_counts",
    "log_increase_ratios",
]

#: How a sparsity budget is spread over the layers.
ALLOCATIONS = ("uniform", "global", "log-increase")

#: Allocations that are one criterion's own structure search rather than a
#: budget over its scores, and the method each belongs to. A prune spec accepts
#: them only with that method; :func:`allocate` and :func:`keep_counts` never.
METHOD_ALLOCATIONS = {"al-am": "flap"}

#: ``r_0`` of the Incremental Pruning Ratio as a share of the target sparsity.
#: The paper does not state it, but the curve plotted in its Figure 4 (LLaMA-7B
#: at 50 percent) fits equation 6 exactly with r_0 = 0.125, a quarter of the
#: target.
FIRST_LAYER_SHARE = 0.25


def allocate(
    head_scores: Sequence[Sequence[float]],
    channel_scores: Sequence[Sequence[float]],
    groups: Sequence[LayerGroups],
    *,
    sparsity: float,
    allocation: str = "uniform",
) -> tuple[LayerPlan, ...]:
    """Choose which heads and channels to keep, at ``sparsity`` overall.

    Args:
        head_scores: Per layer, one importance score per attention head. Higher
            is more important.
        channel_scores: Per layer, one score per FFN intermediate channel.
        groups: The dense model's per-layer unit counts, from
            :func:`mnlp_eval.prune.groups.describe_layers`.
        sparsity: Fraction of units to remove, in [0, 1).
        allocation: ``uniform``, ``global`` or ``log-increase``.

    Returns:
        One :class:`LayerPlan` per layer, holding indices into the dense model.
    """
    _check_budget(allocation, sparsity)

    _check_shapes(head_scores, groups, "head", "num_heads")
    _check_shapes(channel_scores, groups, "channel", "intermediate")

    # Grouped-query keeps the same number per group, so counts move in steps.
    head_steps = [group.num_kv_heads if group.is_grouped_query else 1 for group in groups]
    head_counts = _counts(head_scores, sparsity, allocation, steps=head_steps)
    channel_counts = _counts(channel_scores, sparsity, allocation, steps=[1] * len(groups))

    return tuple(
        LayerPlan(
            heads=_pick_heads(scores, group, keep),
            channels=_top(channels, channel_keep),
        )
        for scores, channels, group, keep, channel_keep in zip(
            head_scores, channel_scores, groups, head_counts, channel_counts, strict=True
        )
    )


def keep_counts(
    groups: Sequence[LayerGroups],
    *,
    sparsity: float,
    allocation: str,
    head_scores: Sequence[Sequence[float]] | None = None,
    channel_scores: Sequence[Sequence[float]] | None = None,
) -> tuple[list[int], list[int]]:
    """How many heads and FFN channels each layer keeps, without choosing them.

    For a criterion that picks units itself, layer by layer, once it knows how
    many to keep. ``uniform`` and ``log-increase`` need no scores; ``global``
    compares layers and so does.
    """
    _check_budget(allocation, sparsity)
    if allocation == "global":
        if head_scores is None or channel_scores is None:
            msg = "a global budget compares layers, so it needs every layer's scores"
            raise PruneError(msg)
        heads, channels = head_scores, channel_scores
    else:
        # Placeholders: these allocations read only how many units a layer has.
        heads = [[0.0] * group.num_heads for group in groups]
        channels = [[0.0] * group.intermediate for group in groups]
    _check_shapes(heads, groups, "head", "num_heads")
    _check_shapes(channels, groups, "channel", "intermediate")
    head_steps = [group.num_kv_heads if group.is_grouped_query else 1 for group in groups]
    return (
        _counts(heads, sparsity, allocation, steps=head_steps),
        _counts(channels, sparsity, allocation, steps=[1] * len(groups)),
    )


def _check_budget(allocation: str, sparsity: float) -> None:
    owner = METHOD_ALLOCATIONS.get(allocation)
    if owner is not None:
        msg = (
            f"allocation {allocation!r} is {owner}'s own structure search over per-column "
            f"scores, chosen in prune/methods/{owner}.py rather than here"
        )
        raise PruneError(msg)
    if allocation not in ALLOCATIONS:
        msg = f"allocation {allocation!r} is not one of {', '.join(ALLOCATIONS)}"
        raise PruneError(msg)
    if not 0.0 <= sparsity < 1.0:
        msg = f"sparsity must be in [0, 1), got {sparsity}. It is the fraction removed."
        raise PruneError(msg)


def _counts(
    scores: Sequence[Sequence[float]], sparsity: float, allocation: str, *, steps: Sequence[int]
) -> list[int]:
    """How many units each layer keeps."""
    sizes = [len(layer) for layer in scores]
    if allocation == "uniform":
        raw = [size - round(size * sparsity) for size in sizes]
    elif allocation == "log-increase":
        ratios = log_increase_ratios(len(sizes), sparsity)
        raw = [size - round(size * ratio) for size, ratio in zip(sizes, ratios, strict=True)]
    else:
        raw = _global_counts(scores, sparsity)
    # A layer that keeps nothing is a removed layer, a different experiment.
    return [
        min(size, max(step, count if step == 1 else round(count / step) * step))
        for count, size, step in zip(raw, sizes, steps, strict=True)
    ]


def log_increase_ratios(layers: int, sparsity: float) -> list[float]:
    """Per-layer fractions removed under SlimGPT's Incremental Pruning Ratio.

    Equation 6 of Ling et al. (NeurIPS 2024):
    ``r_i = r_0 + (r_last - r_0) * log(i + 1) / log(n)``. ``r_0`` is
    :data:`FIRST_LAYER_SHARE` of the target, read off the paper's Figure 4, and
    ``r_last`` is solved so the mean over layers equals ``sparsity``. Every
    layer holds the same number of units in a dense Llama, so the mean ratio is
    the overall unit sparsity. On ALMA-7B's 32 layers that is 0.05 rising to
    0.254 at 20 percent, 0.075 to 0.381 at 30, and 0.10 to 0.508 at 40.
    """
    if layers < 1:
        msg = "log-increase needs at least one layer"
        raise PruneError(msg)
    if layers == 1:
        return [sparsity]
    shape = [math.log(index + 1) / math.log(layers) for index in range(layers)]
    first = FIRST_LAYER_SHARE * sparsity
    last = first + (sparsity - first) / statistics.fmean(shape)
    if last >= 1.0:
        msg = (
            f"log-increase at sparsity {sparsity} would remove {last:.2f} of the last "
            "layer, which is all of it. Use a lower sparsity or another allocation."
        )
        raise PruneError(msg)
    return [first + (last - first) * value for value in shape]


def _global_counts(scores: Sequence[Sequence[float]], sparsity: float) -> list[int]:
    """Counts implied by pooling layer-standardised scores and taking the best.

    Equivalent to a global top-k, since standardising within a layer is
    monotone and preserves its internal order.
    """
    pooled = [(value, index) for index, layer in enumerate(scores) for value in _standardise(layer)]
    total = len(pooled)
    keep = total - round(total * sparsity)
    pooled.sort(key=lambda item: item[0], reverse=True)

    counts = [0] * len(scores)
    for _, index in pooled[:keep]:
        counts[index] += 1
    return counts


def _standardise(values: Sequence[float]) -> list[float]:
    """Centre and scale one layer's scores.

    All-equal scores say nothing about which unit to drop, so they standardise
    to zeros rather than winning or losing the budget on a rounding artifact.
    """
    spread = statistics.pstdev(values)
    if spread == 0.0:
        return [0.0] * len(values)
    centre = statistics.fmean(values)
    return [(value - centre) / spread for value in values]


def _pick_heads(scores: Sequence[float], group: LayerGroups, keep: int) -> tuple[int, ...]:
    """The best ``keep`` heads, spread evenly over the key/value groups."""
    if not group.is_grouped_query:
        return _top(scores, keep)
    per_group = keep // group.num_kv_heads
    chosen: list[int] = []
    for members in group.head_groups():
        ranked = sorted(members, key=lambda head: (-scores[head], head))
        chosen.extend(ranked[:per_group])
    return tuple(sorted(chosen))


def _top(scores: Sequence[float], keep: int) -> tuple[int, ...]:
    """The indices of the ``keep`` highest scores, ties broken by index."""
    ranked = sorted(range(len(scores)), key=lambda index: (-scores[index], index))
    return tuple(sorted(ranked[:keep]))


def _check_shapes(
    scores: Sequence[Sequence[float]], groups: Sequence[LayerGroups], unit: str, attribute: str
) -> None:
    if len(scores) != len(groups):
        msg = f"{unit} scores cover {len(scores)} layers but the model has {len(groups)}"
        raise PruneError(msg)
    for layer, group in zip(scores, groups, strict=True):
        expected = getattr(group, attribute)
        if len(layer) != expected:
            msg = f"layer {group.index}: got {len(layer)} {unit} scores, expected {expected}"
            raise PruneError(msg)
