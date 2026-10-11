"""SlimGPT: layer-wise structured pruning by optimal brain surgery (Ling et al., NeurIPS 2024).

A head is the ``head_dim`` columns of ``o_proj`` it writes into, an FFN channel
one column of ``down_proj``. With damped input Hessian ``H = E[x x^T]`` and
``M = [H^-1]_SS``, removing columns ``S`` costs ``tr(W_S M^-1 W_S^T)`` and the
remaining columns are updated by ``W <- W - W_S M^-1 [H^-1]_S,:`` (eq. 5).

Units are removed greedily: after each step the survivors are re-scored on the
updated weights and the Schur complement ``H^-1 - [H^-1]_:,S M^-1 [H^-1]_S,:``.
Heads go one per step (one per key/value group under GQA), channels in batches
of :data:`CHANNEL_BATCH`. Layers are pruned in order on the outputs of the
already-pruned layers, and the FFN Hessian is taken after attention is pruned.

Differences from the paper: a fixed FFN batch of 128 (the paper shrinks it
from 1024 to 8). The paper's per-layer schedule is the ``log-increase``
allocation in ``prune/budget.py``; ``global`` is our addition and ranks layers
on single-shot costs from the dense model.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from mnlp_eval.prune import PruneError
from mnlp_eval.prune.budget import keep_counts
from mnlp_eval.prune.compact import LayerPlan, compact_layer
from mnlp_eval.prune.groups import LayerGroups, decoder_layers

__all__ = ["CHANNEL_BATCH", "DAMPING", "prune"]

#: Ridge on the Hessian diagonal, as a fraction of its mean (SparseGPT's default).
DAMPING = 0.01

#: FFN channels removed per greedy step.
CHANNEL_BATCH = 128

#: The projections whose input channels index a group, in forward order.
_PROJECTIONS = ("o_proj", "down_proj")


def prune(
    model: Any,
    batches: Sequence[dict[str, Any]],
    groups: tuple[LayerGroups, ...],
    *,
    sparsity: float,
    allocation: str = "uniform",
) -> tuple[LayerPlan, ...]:
    """Allocate, select, compensate and compact each layer; return the kept units.

    Compacts as it goes, since each layer's input depends on the pruned layers before it.
    """
    layers = decoder_layers(model)

    if allocation == "global":
        head_scores, channel_scores = _dense_scores(model, batches, groups)
        head_keep, channel_keep = keep_counts(
            groups,
            sparsity=sparsity,
            allocation=allocation,
            head_scores=head_scores,
            channel_scores=channel_scores,
        )
    else:
        head_keep, channel_keep = keep_counts(groups, sparsity=sparsity, allocation=allocation)

    inputs = _layer_inputs(model, batches)
    plan: list[LayerPlan] = []
    for position, (layer, group, heads, channels) in enumerate(
        zip(layers, groups, head_keep, channel_keep, strict=True)
    ):
        try:
            layer_plan = _prune_layer(layer, group, inputs, heads=heads, channels=channels)
        except PruneError as exc:
            msg = f"layer {group.index}: {exc}"
            raise PruneError(msg) from exc
        compact_layer(layer, group, layer_plan)
        plan.append(layer_plan)
        if position + 1 < len(layers):
            inputs = _advance(layer, inputs)

    return tuple(plan)


def _dense_scores(
    model: Any, batches: Sequence[dict[str, Any]], groups: tuple[LayerGroups, ...]
) -> tuple[list[list[float]], list[list[float]]]:
    """Single-shot removal costs on the dense model, for the global budget.

    Each layer is scored on the dense outputs of the layers before it.
    """
    import torch

    layers = decoder_layers(model)
    dense = _layer_inputs(model, batches)
    head_scores: list[list[float]] = []
    channel_scores: list[list[float]] = []
    for position, (layer, group) in enumerate(zip(layers, groups, strict=True)):
        hessians = _hessians(layer, dense)
        try:
            inverse, dead = _inverse_hessian(hessians.pop("o_proj"))
            weight = layer.self_attn.o_proj.weight.data.to(torch.float64)
            costs = _head_costs(weight, inverse, dead, range(group.num_heads), group.head_dim)
            head_scores.append(costs.tolist())
            del inverse, weight

            inverse, dead = _inverse_hessian(hessians.pop("down_proj"))
            weight = layer.mlp.down_proj.weight.data.to(torch.float64)
            costs = _channel_costs(weight, inverse, dead)
            _check_costs(costs, "FFN channel")
            channel_scores.append(costs.tolist())
            del inverse, weight
        except PruneError as exc:
            msg = f"layer {group.index}: {exc}"
            raise PruneError(msg) from exc
        if position + 1 < len(layers):
            dense = _advance(layer, dense)
    return head_scores, channel_scores


def _prune_layer(
    layer: Any,
    group: LayerGroups,
    inputs: Sequence[tuple[Any, dict[str, Any], Any]],
    *,
    heads: int,
    channels: int,
) -> LayerPlan:
    """Prune one layer's heads, then its channels, leaving cut columns zero."""
    import torch

    kept_heads = tuple(range(group.num_heads))
    if heads < group.num_heads:
        projection = layer.self_attn.o_proj
        inverse, dead = _inverse_hessian(_hessians(layer, inputs, ("o_proj",))["o_proj"])
        weight = projection.weight.data.to(torch.float64, copy=True)
        kept_heads = _prune_heads(weight, inverse, dead, group, heads)
        del inverse
        # Zeroed o_proj columns silence the head, so the FFN Hessian below
        # already sees the pruned attention block.
        projection.weight.data.copy_(weight)
        del weight

    kept_channels = tuple(range(group.intermediate))
    if channels < group.intermediate:
        projection = layer.mlp.down_proj
        inverse, dead = _inverse_hessian(_hessians(layer, inputs, ("down_proj",))["down_proj"])
        weight = projection.weight.data.to(torch.float64, copy=True)
        kept_channels = _prune_channels(weight, inverse, dead, channels)
        del inverse
        projection.weight.data.copy_(weight)
        del weight

    return LayerPlan(heads=kept_heads, channels=kept_channels)


def _layer_inputs(
    model: Any, batches: Sequence[dict[str, Any]]
) -> list[tuple[Any, dict[str, Any], Any]]:
    """Capture the first decoder layer's input and keyword arguments per batch.

    Also keeps the batch's 2D padding mask, since the mask in the kwargs is the
    expanded causal one.
    """
    import torch

    layers = decoder_layers(model)
    captured: list[tuple[Any, dict[str, Any], Any]] = []
    padding: Any = None

    class Reached(Exception):
        """Stops the forward pass at the first layer."""

    def hook(_module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        hidden = args[0] if args else kwargs["hidden_states"]
        rest = {key: value for key, value in kwargs.items() if key != "hidden_states"}
        rest.pop("past_key_value", None)
        rest.pop("past_key_values", None)
        rest["use_cache"] = False
        captured.append((hidden.detach().clone(), rest, padding))
        raise Reached

    handle = layers[0].register_forward_pre_hook(hook, with_kwargs=True)
    try:
        with torch.inference_mode():
            for batch in batches:
                padding = batch.get("attention_mask")
                try:
                    model(**batch, use_cache=False)
                except Reached:
                    continue
    finally:
        handle.remove()

    if not captured:
        msg = "the calibration set produced no batches, so no activations were captured"
        raise PruneError(msg)
    return captured


def _hessians(
    layer: Any,
    inputs: Sequence[tuple[Any, dict[str, Any], Any]],
    names: Sequence[str] = _PROJECTIONS,
) -> dict[str, Any]:
    """Accumulate ``X X^T / tokens`` in float32 for the named projections.

    Padding positions are excluded. The forward pass stops at the last
    projection requested.
    """
    import torch

    wanted = tuple(names)
    last = max(wanted, key=_PROJECTIONS.index)
    modules = {"o_proj": layer.self_attn.o_proj, "down_proj": layer.mlp.down_proj}
    accumulators: dict[str, Any] = {}
    counts: dict[str, int] = {}
    mask: Any = None

    class Collected(Exception):
        """Stops the layer's forward pass once all inputs are collected."""

    def hook(name: str) -> Any:
        def capture(_module: Any, args: tuple[Any, ...]) -> None:
            flat = args[0].reshape(-1, args[0].shape[-1]).float()
            if mask is not None:
                flat = flat[mask.reshape(-1).bool()]
            existing = accumulators.get(name)
            if existing is None:
                width = int(flat.shape[-1])
                existing = torch.zeros(width, width, dtype=torch.float32, device=flat.device)
                accumulators[name] = existing
                counts[name] = 0
            existing.addmm_(flat.t(), flat)
            counts[name] += int(flat.shape[0])
            if name == last:
                raise Collected

        return capture

    handles = [modules[name].register_forward_pre_hook(hook(name)) for name in wanted]
    try:
        with torch.inference_mode():
            for hidden, kwargs, padding in inputs:
                mask = padding
                try:
                    layer(hidden, **kwargs)
                except Collected:
                    continue
    finally:
        for handle in handles:
            handle.remove()

    return {name: accumulators[name] / max(counts[name], 1) for name in wanted}


def _advance(
    layer: Any, inputs: Sequence[tuple[Any, dict[str, Any], Any]]
) -> list[tuple[Any, dict[str, Any], Any]]:
    """Run the pruned layer to produce the next layer's input."""
    import torch

    advanced: list[tuple[Any, dict[str, Any], Any]] = []
    with torch.inference_mode():
        for hidden, kwargs, padding in inputs:
            output = layer(hidden, **kwargs)
            state = output[0] if isinstance(output, tuple) else output
            advanced.append((state.detach().clone(), kwargs, padding))
    return advanced


def _inverse_hessian(hessian: Any) -> tuple[Any, Any]:
    """``H^-1`` of the damped Hessian in float64, and a mask of dead channels.

    Dead channels (zero diagonal) are handled as in SparseGPT, and their rows
    and columns are zeroed in the inverse so they cost nothing to remove.
    """
    import torch

    matrix = hessian.to(torch.float64, copy=True)
    dead = torch.diagonal(matrix) == 0
    matrix[dead, :] = 0.0
    matrix[:, dead] = 0.0
    diagonal = torch.diagonal(matrix)
    diagonal[dead] = 1.0
    diagonal += DAMPING * torch.mean(diagonal)

    factor, info = torch.linalg.cholesky_ex(matrix)
    if int(info) != 0:
        msg = (
            "the damped Hessian is not positive definite, so the calibration "
            "activations are probably not finite"
        )
        raise PruneError(msg)
    del matrix
    inverse = torch.cholesky_inverse(factor)
    del factor

    if bool(dead.any()):
        kept = torch.diagonal(inverse)[dead]
        inverse[dead, :] = 0.0
        inverse[:, dead] = 0.0
        torch.diagonal(inverse)[dead] = kept
    return inverse, dead


def _head_costs(weight: Any, inverse: Any, dead: Any, heads: Sequence[int], head_dim: int) -> Any:
    """``tr(W_S M^-1 W_S^T)`` for each head's columns ``S``, batched over heads."""
    import torch

    device = weight.device
    index = torch.as_tensor(list(heads), dtype=torch.long, device=device)
    columns = index[:, None] * head_dim + torch.arange(head_dim, device=device)
    blocks = inverse[columns[:, :, None], columns[:, None, :]]
    factor, info = torch.linalg.cholesky_ex(blocks)
    if bool(info.any()):
        msg = "a head's block of the inverse Hessian is not positive definite"
        raise PruneError(msg)

    # (out, heads, head_dim), dead columns ignored, to (heads, head_dim, out).
    sliced = weight[:, columns].masked_fill_(dead[columns], 0.0)
    solved = torch.linalg.solve_triangular(factor, sliced.permute(1, 2, 0), upper=False)
    costs = solved.pow(2).sum(dim=(1, 2))
    _check_costs(costs, "head")
    return costs


def _channel_costs(weight: Any, inverse: Any, dead: Any) -> Any:
    """``||W[:, j]||^2 / [H^-1]_jj`` per column, zero for a dead channel.

    Undefined for removed columns; the caller masks those.
    """
    import torch

    costs = torch.linalg.vector_norm(weight, dim=0).pow(2) / torch.diagonal(inverse)
    return costs.masked_fill(dead, 0.0)


def _prune_heads(
    weight: Any, inverse: Any, dead: Any, group: LayerGroups, keep: int
) -> tuple[int, ...]:
    """Remove heads greedily down to ``keep``, compensating after each step.

    Updates ``weight`` and ``inverse`` in place and returns the kept heads.
    Under GQA, the cheapest head of every key/value group goes in each step.
    """
    import torch

    partitions = group.head_groups() if group.is_grouped_query else (tuple(range(group.num_heads)),)
    removed = group.num_heads - keep
    if keep < 1 or removed < 0 or removed % len(partitions):
        msg = (
            f"cannot keep {keep} of {group.num_heads} heads evenly over "
            f"{len(partitions)} key/value group(s)"
        )
        raise PruneError(msg)

    offsets = torch.arange(group.head_dim, device=weight.device)
    remaining = set(range(group.num_heads))
    for _ in range(removed // len(partitions)):
        candidates = sorted(remaining)
        values = _head_costs(weight, inverse, dead, candidates, group.head_dim).tolist()
        costs = dict(zip(candidates, values, strict=True))
        # Ties go to the lower index.
        chosen = [
            min((costs[head], head) for head in members if head in remaining)[1]
            for members in partitions
        ]
        index = torch.as_tensor(chosen, dtype=torch.long, device=weight.device)
        _remove_columns(weight, inverse, (index[:, None] * group.head_dim + offsets).reshape(-1))
        remaining.difference_update(chosen)
    return tuple(sorted(remaining))


def _prune_channels(
    weight: Any, inverse: Any, dead: Any, keep: int, *, batch: int = CHANNEL_BATCH
) -> tuple[int, ...]:
    """Remove columns greedily down to ``keep``, ``batch`` at a time.

    Updates ``weight`` and ``inverse`` in place and returns the kept columns.
    Ties go to the lower index.
    """
    import torch

    width = int(weight.shape[1])
    if not 1 <= keep <= width:
        msg = f"cannot keep {keep} of {width} FFN channels"
        raise PruneError(msg)

    remaining = torch.ones(width, dtype=torch.bool, device=weight.device)
    left = width - keep
    while left > 0:
        step = min(batch, left)
        costs = _channel_costs(weight, inverse, dead)
        _check_costs(costs[remaining], "FFN channel")
        costs = costs.masked_fill(~remaining, math.inf)
        chosen = torch.sort(costs, stable=True).indices[:step]
        _remove_columns(weight, inverse, chosen)
        remaining[chosen] = False
        left -= step
    return tuple(int(index) for index in torch.nonzero(remaining).flatten().tolist())


def _remove_columns(weight: Any, inverse: Any, columns: Any) -> float:
    """Remove ``columns`` jointly and update the rest in place.

    Applies ``W -= W_S M^-1 [H^-1]_S,:`` and ``H^-1 -= [H^-1]_:,S M^-1 [H^-1]_S,:``
    with ``M = [H^-1]_SS``, via the Cholesky factor of ``M``. Returns the
    increase in squared output error per token.
    """
    import torch

    rows = inverse.index_select(0, columns)
    factor, info = torch.linalg.cholesky_ex(rows.index_select(1, columns))
    if int(info) != 0:
        msg = "the inverse Hessian over the removed columns is not positive definite"
        raise PruneError(msg)
    reach = torch.linalg.solve_triangular(factor, rows, upper=False)
    error = torch.linalg.solve_triangular(factor, weight.index_select(1, columns).t(), upper=False)
    del rows, factor

    weight.addmm_(error.t(), reach, alpha=-1.0)
    inverse.addmm_(reach.t(), reach, alpha=-1.0)
    weight.index_fill_(1, columns, 0.0)
    inverse.index_fill_(0, columns, 0.0)
    inverse.index_fill_(1, columns, 0.0)
    return float(error.pow(2).sum())


def _check_costs(costs: Any, unit: str) -> None:
    """Raise if any cost is non-finite or negative."""
    import torch

    if not bool(torch.isfinite(costs).all()) or bool((costs < 0).any()):
        msg = f"non-finite or negative {unit} removal costs; the Hessian is numerically broken"
        raise PruneError(msg)
