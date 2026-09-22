"""SlimGPT: layer-wise structured pruning by optimal brain surgery.

Scores an input channel by ``sum_rows W[:, j]^2 / [H^-1]_jj`` for the Hessian
``H = X X^T`` of the projection's input, and a group by the sum over its
channels.

Pruning here is a weight update, not a mask. Removing column ``j`` changes
every survivor by ``W[:, j:] -= (W[:, j] / L_jj) outer L[j, j:]`` for ``L`` the
Cholesky factor of ``H^-1``. Without that update this is a more expensive
ranking, so this module mutates the layer and the caller then compacts it.

Layers are pruned in order, each on activations the already-pruned layers
before it produce. Compensation solves for the error a layer makes given its
actual input, which after the first layer is no longer the dense model's.

Two departures from the paper: the per-layer schedule that prunes early layers
less is not implemented, and column scores are taken once from the dense
weights rather than recomputed as the block loop advances. The compensation
itself is exact.

Reference: Ling et al., NeurIPS 2024, after Frantar and Alistarh's SparseGPT.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from mnlp_eval.prune import PruneError
from mnlp_eval.prune.budget import allocate
from mnlp_eval.prune.compact import LayerPlan, compact_layer
from mnlp_eval.prune.groups import LayerGroups, decoder_layers
from mnlp_eval.prune.methods import pool_heads

__all__ = ["prune"]

#: Ridge on the Hessian diagonal, as a fraction of its mean. A calibration set
#: smaller than the input width leaves it singular. SparseGPT's default.
DAMPING = 0.01


def prune(
    model: Any,
    batches: Sequence[dict[str, Any]],
    groups: tuple[LayerGroups, ...],
    *,
    sparsity: float,
    allocation: str = "uniform",
) -> tuple[LayerPlan, ...]:
    """Score, allocate, compensate and compact, and return what was kept.

    Does its own compaction, unlike the other two: a layer's compensation has
    to be applied before the next layer sees its input.
    """
    layers = decoder_layers(model)
    inputs = _layer_inputs(model, batches)

    # Pass one: unit worth on the dense model. A global budget compares layers
    # before any of them is changed.
    head_scores: list[list[float]] = []
    channel_scores: list[list[float]] = []
    for layer, group in zip(layers, groups, strict=True):
        hessians = _hessians(layer, inputs)
        attention = _column_costs(layer.self_attn.o_proj, hessians["o_proj"])
        head_scores.append(pool_heads(attention, group))
        channel_scores.append(
            [float(value) for value in _column_costs(layer.mlp.down_proj, hessians["down_proj"])]
        )

    plan = allocate(head_scores, channel_scores, groups, sparsity=sparsity, allocation=allocation)

    # Pass two: prune in order, each layer seeing what the pruned ones produce.
    for position, (layer, group, layer_plan) in enumerate(zip(layers, groups, plan, strict=True)):
        hessians = _hessians(layer, inputs)
        _compensate(
            layer.self_attn.o_proj, hessians["o_proj"], set(group.head_rows(layer_plan.heads))
        )
        _compensate(
            layer.mlp.down_proj,
            hessians["down_proj"],
            set(group.validate_channels(layer_plan.channels)),
        )
        compact_layer(layer, group, layer_plan)
        if position + 1 < len(layers):
            inputs = _advance(layer, inputs)

    return plan


def _layer_inputs(
    model: Any, batches: Sequence[dict[str, Any]]
) -> list[tuple[Any, dict[str, Any]]]:
    """Capture what enters the first decoder layer, with its keyword arguments.

    Running the whole model once per layer is the obvious way to get a layer's
    input and is unaffordable. The mask, rotary embeddings and position ids are
    captured once and replayed. The forward pass is abandoned at the first
    layer, so nothing below it is computed.
    """
    import torch

    layers = decoder_layers(model)
    captured: list[tuple[Any, dict[str, Any]]] = []

    class Reached(Exception):
        """Abandons the forward pass once the first layer's input is in hand."""

    def hook(_module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        hidden = args[0] if args else kwargs["hidden_states"]
        rest = {key: value for key, value in kwargs.items() if key != "hidden_states"}
        # A cache would grow across replays and change the second pass.
        rest.pop("past_key_value", None)
        rest.pop("past_key_values", None)
        rest["use_cache"] = False
        captured.append((hidden.detach().clone(), rest))
        raise Reached

    handle = layers[0].register_forward_pre_hook(hook, with_kwargs=True)
    try:
        with torch.inference_mode():
            for batch in batches:
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


def _hessians(layer: Any, inputs: Sequence[tuple[Any, dict[str, Any]]]) -> dict[str, Any]:
    """Accumulate ``X X^T`` for the two projections whose inputs index groups."""
    import torch

    accumulators: dict[str, Any] = {}
    counts: dict[str, int] = {}

    def hook(name: str) -> Any:
        def capture(_module: Any, args: tuple[Any, ...]) -> None:
            flat = args[0].reshape(-1, args[0].shape[-1]).float()
            existing = accumulators.get(name)
            if existing is None:
                width = int(flat.shape[-1])
                existing = torch.zeros(width, width, dtype=torch.float32, device=flat.device)
                accumulators[name] = existing
                counts[name] = 0
            existing += flat.t() @ flat
            counts[name] += int(flat.shape[0])

        return capture

    handles = [
        layer.self_attn.o_proj.register_forward_pre_hook(hook("o_proj")),
        layer.mlp.down_proj.register_forward_pre_hook(hook("down_proj")),
    ]
    try:
        with torch.inference_mode():
            for hidden, kwargs in inputs:
                layer(hidden, **kwargs)
    finally:
        for handle in handles:
            handle.remove()

    return {name: value / max(counts[name], 1) for name, value in accumulators.items()}


def _advance(
    layer: Any, inputs: Sequence[tuple[Any, dict[str, Any]]]
) -> list[tuple[Any, dict[str, Any]]]:
    """Run the pruned layer to produce the next layer's input."""
    import torch

    advanced: list[tuple[Any, dict[str, Any]]] = []
    with torch.inference_mode():
        for hidden, kwargs in inputs:
            output = layer(hidden, **kwargs)
            state = output[0] if isinstance(output, tuple) else output
            advanced.append((state.detach().clone(), kwargs))
    return advanced


def _factor(hessian: Any) -> tuple[Any, Any]:
    """Upper Cholesky factor of ``H^-1``, and which channels were dead.

    A channel that is zero throughout the calibration set leaves a zero on the
    diagonal, which is not invertible and not worth keeping. Neutralising its
    row and column is SparseGPT's handling, and discards it first.
    """
    import torch

    width = hessian.shape[0]
    dead = torch.diagonal(hessian) == 0
    hessian[dead, :] = 0.0
    hessian[:, dead] = 0.0
    diagonal = torch.diagonal(hessian)
    diagonal[dead] = 1.0

    ridge = DAMPING * torch.mean(diagonal)
    hessian[range(width), range(width)] += ridge

    inverse = torch.cholesky_inverse(torch.linalg.cholesky(hessian))
    return torch.linalg.cholesky(inverse, upper=True), dead


def _column_costs(projection: Any, hessian: Any) -> Any:
    """Optimal-brain-surgery cost of removing each input channel."""
    factor, dead = _factor(hessian.clone())
    weight = projection.weight.data.float()
    costs = weight.pow(2).sum(0) / factor.diagonal().pow(2)
    costs[dead] = 0.0
    return costs


def _compensate(projection: Any, hessian: Any, kept: set[int]) -> None:
    """Remove columns from ``projection``, pushing their error onto the rest.

    Index order is what lets one Cholesky factor serve the whole sweep: a
    column's update only reaches columns after it. Removed columns are zeroed
    and dropped by the compaction that follows.
    """
    import torch

    factor, _ = _factor(hessian.clone())
    weight = projection.weight.data.float()

    for column in range(weight.shape[1]):
        if column in kept:
            continue
        scale = factor[column, column]
        if scale == 0:
            weight[:, column] = 0.0
            continue
        error = weight[:, column] / scale
        weight[:, column:] -= torch.outer(error, factor[column, column:])
        weight[:, column] = 0.0

    projection.weight.data = weight.to(projection.weight.dtype)
