"""Compacting a model around a selection, and recording what was kept.

:func:`compact_model` shrinks a loaded model's projections onto the kept units.
:func:`reshape_model` gives an uninitialised model the same shapes for loading
a checkpoint back. They are one function with a flag, because they must agree.

``num_key_value_groups`` is the only module attribute pruning invalidates:
attention reshapes with ``-1``, rotary embeddings carry no head dimension and
the causal mask broadcasts over heads. True from transformers 4.48, which the
``prune`` extra pins.

Weights are never zeroed in place. ``bench.structure`` reports the fraction of
parameters that are exactly zero precisely to catch that.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mnlp_eval.prune import PruneError
from mnlp_eval.prune.groups import LayerGroups, decoder_layers, describe_layers

if TYPE_CHECKING:
    from mnlp_eval.analysis.subnetwork import Subnetwork

__all__ = [
    "LayerPlan",
    "compact_layer",
    "compact_model",
    "plan_from_subnetwork",
    "reshape_model",
    "write_descriptor",
]


@dataclass(frozen=True)
class LayerPlan:
    """The units one layer keeps, indexed into the dense model."""

    heads: tuple[int, ...]
    channels: tuple[int, ...]


def compact_model(model: Any, plan: tuple[LayerPlan, ...]) -> None:
    """Shrink ``model`` in place onto ``plan``, selecting the kept weights."""
    _apply(model, plan, select=True, output_bias=False)


def reshape_model(model: Any, plan: tuple[LayerPlan, ...], *, output_bias: bool = False) -> None:
    """Give ``model`` the shapes ``plan`` implies, without carrying weights over.

    ``output_bias`` adds a bias to ``o_proj`` and ``down_proj``, which both
    architectures build without one and FLAP's compensation needs.
    """
    _apply(model, plan, select=False, output_bias=output_bias)


def _apply(model: Any, plan: tuple[LayerPlan, ...], *, select: bool, output_bias: bool) -> None:
    layers = decoder_layers(model)
    groups = describe_layers(model)
    if len(plan) != len(layers):
        msg = f"plan covers {len(plan)} layers but the model has {len(layers)}"
        raise PruneError(msg)

    for layer, group, layer_plan in zip(layers, groups, plan, strict=True):
        compact_layer(layer, group, layer_plan, select=select, output_bias=output_bias)


def compact_layer(
    layer: Any,
    group: LayerGroups,
    plan: LayerPlan,
    *,
    select: bool = True,
    output_bias: bool = False,
) -> None:
    """Shrink one decoder layer onto ``plan``.

    Exposed because SlimGPT compacts as it goes: compensation is only correct
    if the next layer sees what the pruned one produces.
    """
    head_rows = group.head_rows(plan.heads)
    kv_rows = group.kv_rows(plan.heads)
    channels = group.validate_channels(plan.channels)

    attention, mlp = layer.self_attn, layer.mlp
    _resize(attention, "q_proj", rows=head_rows, select=select)
    _resize(attention, "k_proj", rows=kv_rows, select=select)
    _resize(attention, "v_proj", rows=kv_rows, select=select)
    _resize(attention, "o_proj", columns=head_rows, select=select, add_bias=output_bias)
    _resize(mlp, "gate_proj", rows=channels, select=select)
    _resize(mlp, "up_proj", rows=channels, select=select)
    _resize(mlp, "down_proj", columns=channels, select=select, add_bias=output_bias)

    # The one attribute the attention forward reads that pruning invalidates.
    attention.num_key_value_groups = group.num_key_value_groups_after(plan.heads)


def _resize(
    parent: Any,
    name: str,
    *,
    rows: tuple[int, ...] | None = None,
    columns: tuple[int, ...] | None = None,
    select: bool,
    add_bias: bool = False,
) -> None:
    import torch
    from torch import nn

    old = getattr(parent, name)
    out_features = len(rows) if rows is not None else old.out_features
    in_features = len(columns) if columns is not None else old.in_features
    keeps_bias = old.bias is not None or add_bias

    if not select:
        setattr(
            parent,
            name,
            nn.Linear(
                in_features, out_features, bias=keeps_bias, device="meta", dtype=old.weight.dtype
            ),
        )
        return

    weight = old.weight.data
    if rows is not None:
        weight = weight.index_select(0, torch.as_tensor(rows, device=weight.device))
    if columns is not None:
        weight = weight.index_select(1, torch.as_tensor(columns, device=weight.device))

    new = nn.Linear(
        in_features, out_features, bias=keeps_bias, device="meta", dtype=old.weight.dtype
    )
    new.weight = nn.Parameter(weight, requires_grad=old.weight.requires_grad)
    if old.bias is not None:
        bias = old.bias.data
        # Only a row selection touches a bias; input channels leave it alone.
        if rows is not None:
            bias = bias.index_select(0, torch.as_tensor(rows, device=bias.device))
        new.bias = nn.Parameter(bias, requires_grad=old.bias.requires_grad)
    elif add_bias:
        new.bias = nn.Parameter(
            torch.zeros(out_features, dtype=old.weight.dtype, device=old.weight.device)
        )
    setattr(parent, name, new)


def write_descriptor(
    path: str | Path,
    plan: tuple[LayerPlan, ...],
    groups: tuple[LayerGroups, ...],
    *,
    name: str,
    base_model: str,
    method: str,
    pruned_for: str = "multi",
    notes: str = "",
) -> Subnetwork:
    """Write the kept-unit descriptor and return it parsed back.

    Reading it back is the point: it is the only record of which positions a
    run chose, and a malformed one would otherwise surface weeks later when the
    overlap analysis runs. See docs/subnetworks.md.
    """
    from mnlp_eval.analysis.subnetwork import SUBNETWORK_SCHEMA_VERSION, load_subnetwork
    from mnlp_eval.artifacts import atomic_write_json

    heads_total = {group.num_heads for group in groups}
    channels_total = {group.intermediate for group in groups}
    if len(heads_total) != 1 or len(channels_total) != 1:
        # One 'total' per component, so only a dense model of uniform width.
        msg = (
            "the dense model's layers differ in width "
            f"(heads {sorted(heads_total)}, channels {sorted(channels_total)}), "
            "which the subnetwork descriptor cannot represent"
        )
        raise PruneError(msg)

    payload = {
        "schema_version": SUBNETWORK_SCHEMA_VERSION,
        "name": name,
        "base_model": base_model,
        "pruned_for": pruned_for,
        "method": method,
        "notes": notes,
        "components": {
            "attention_heads": {
                "total": heads_total.pop(),
                "kept": {str(index): list(item.heads) for index, item in enumerate(plan)},
            },
            "ffn_channels": {
                "total": channels_total.pop(),
                "kept": {str(index): list(item.channels) for index, item in enumerate(plan)},
            },
        },
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(target, payload)
    return load_subnetwork(target)


def plan_from_subnetwork(subnetwork: Subnetwork) -> tuple[LayerPlan, ...]:
    """Rebuild a plan from a descriptor, for loading a compacted checkpoint.

    The descriptor is the single record of the per-layer widths, so there is no
    second sidecar that can disagree with it.
    """
    try:
        heads = subnetwork.components["attention_heads"]
        channels = subnetwork.components["ffn_channels"]
    except KeyError as exc:
        msg = (
            f"{subnetwork.source}: needs both 'attention_heads' and 'ffn_channels' components "
            f"to describe a compacted checkpoint, missing {exc.args[0]!r}"
        )
        raise PruneError(msg) from None

    if set(heads.kept) != set(channels.kept):
        msg = (
            f"{subnetwork.source}: the two components cover different layers, "
            f"{sorted(heads.kept)} against {sorted(channels.kept)}"
        )
        raise PruneError(msg)

    return tuple(
        LayerPlan(heads=heads.kept[str(index)], channels=channels.kept[str(index)])
        for index in range(len(heads.kept))
    )
