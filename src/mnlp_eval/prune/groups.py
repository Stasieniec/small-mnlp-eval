"""Prunable unit groups on a Llama-family decoder.

An attention head is ``head_dim`` rows of ``q_proj``, ``k_proj`` and ``v_proj``
plus the matching columns of ``o_proj``. An FFN channel is one row of
``gate_proj`` and ``up_proj`` plus one column of ``down_proj``. Fixed for this
architecture family, so it is a table rather than a traced graph.

Measured from the loaded modules, not the config, so a checkpoint whose config
and tensors disagree is visible. No torch import, so this is testable without
it. Indices are into the dense model, as docs/subnetworks.md requires.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from mnlp_eval.prune import PruneError

__all__ = [
    "SUPPORTED_MODEL_TYPES",
    "LayerGroups",
    "decoder_layers",
    "describe_layers",
]

#: ALMA-7B is ``llama``. ``qwen2`` is the smoke-test model, and grouped-query,
#: which is the case most likely to be got wrong.
SUPPORTED_MODEL_TYPES = frozenset({"llama", "qwen2"})


@dataclass(frozen=True)
class LayerGroups:
    """The prunable units of one decoder layer, measured from its modules."""

    index: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    intermediate: int
    #: Qwen2 hardcodes q, k and v biases and ignores ``config.attention_bias``,
    #: so this is measured. They index by output row and follow the heads.
    attention_bias: bool = False

    @property
    def kv_group_size(self) -> int:
        """Query heads sharing one key/value head. 1 means multi-head attention."""
        return self.num_heads // self.num_kv_heads

    @property
    def is_grouped_query(self) -> bool:
        return self.kv_group_size > 1

    def head_groups(self) -> tuple[tuple[int, ...], ...]:
        """Query head indices, grouped by the key/value head they read."""
        size = self.kv_group_size
        return tuple(
            tuple(range(group * size, (group + 1) * size)) for group in range(self.num_kv_heads)
        )

    def validate_heads(self, kept: Sequence[int]) -> tuple[int, ...]:
        """Check a query-head selection and return it sorted and deduplicated.

        Grouped-query attention must keep the same number per key/value group.
        ``repeat_kv`` wires query head ``i`` to key/value head ``i // n_rep``
        with no remapping hook, so an uneven selection does not fail: the
        survivors attend to the wrong key/value head. Multi-head attention is
        unconstrained.
        """
        unique = _checked_indices(kept, self.num_heads, "attention head", self.index)
        if not self.is_grouped_query:
            return unique

        chosen = set(unique)
        per_group = [len(chosen.intersection(group)) for group in self.head_groups()]
        if len(set(per_group)) != 1:
            msg = (
                f"layer {self.index}: grouped-query attention needs the same number of "
                f"query heads kept in every key/value group, got {per_group}. An uneven "
                "selection misroutes the survivors rather than failing."
            )
            raise PruneError(msg)
        return unique

    def kept_kv_heads(self, kept_heads: Sequence[int]) -> tuple[int, ...]:
        """Key/value heads surviving a query-head selection.

        Grouped-query keeps them all, since every group keeps a query head.
        Multi-head has one per query head, so they follow it.
        """
        validated = self.validate_heads(kept_heads)
        if self.is_grouped_query:
            return tuple(range(self.num_kv_heads))
        return validated

    def num_key_value_groups_after(self, kept_heads: Sequence[int]) -> int:
        """The ``num_key_value_groups`` the pruned attention module must carry."""
        validated = self.validate_heads(kept_heads)
        return len(validated) // len(self.kept_kv_heads(validated))

    def head_rows(self, kept_heads: Sequence[int]) -> tuple[int, ...]:
        """Rows of ``q_proj`` a selection occupies, and the ``o_proj`` columns."""
        return _expand(self.validate_heads(kept_heads), self.head_dim)

    def kv_rows(self, kept_heads: Sequence[int]) -> tuple[int, ...]:
        """Rows of ``k_proj`` and ``v_proj`` that a query-head selection leaves."""
        return _expand(self.kept_kv_heads(kept_heads), self.head_dim)

    def validate_channels(self, kept: Sequence[int]) -> tuple[int, ...]:
        """Check an FFN channel selection and return it sorted and deduplicated."""
        return _checked_indices(kept, self.intermediate, "FFN channel", self.index)


def decoder_layers(model: Any) -> list[Any]:
    """The decoder layers of a supported model, in order.

    Raises rather than guessing: pruning the wrong tensors yields a model that
    loads and translates badly, which looks like a criterion that did not work.
    """
    model_type = getattr(getattr(model, "config", None), "model_type", None)
    if model_type not in SUPPORTED_MODEL_TYPES:
        msg = (
            f"model_type {model_type!r} is not one of "
            f"{', '.join(sorted(SUPPORTED_MODEL_TYPES))}, whose head and channel coupling "
            "is the only one this module encodes."
        )
        raise PruneError(msg)

    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        msg = f"{type(model).__name__} has no model.layers to prune"
        raise PruneError(msg)
    return list(layers)


def describe_layers(model: Any) -> tuple[LayerGroups, ...]:
    """Measure every decoder layer's prunable unit counts."""
    return tuple(_describe_layer(layer, index) for index, layer in enumerate(decoder_layers(model)))


def _describe_layer(layer: Any, index: int) -> LayerGroups:
    attention = getattr(layer, "self_attn", None)
    mlp = getattr(layer, "mlp", None)
    if attention is None or mlp is None:
        msg = f"layer {index}: expected both self_attn and mlp submodules"
        raise PruneError(msg)

    head_dim = getattr(attention, "head_dim", None)
    if not isinstance(head_dim, int) or head_dim <= 0:
        msg = f"layer {index}: self_attn.head_dim is {head_dim!r}, expected a positive integer"
        raise PruneError(msg)

    query_width = _width(attention, "q_proj", "out_features", index)
    key_width = _width(attention, "k_proj", "out_features", index)
    value_width = _width(attention, "v_proj", "out_features", index)
    output_width = _width(attention, "o_proj", "in_features", index)
    if key_width != value_width:
        msg = (
            f"layer {index}: k_proj and v_proj have widths {key_width} and {value_width}. "
            "They index the same key/value heads and must agree."
        )
        raise PruneError(msg)
    if output_width != query_width:
        msg = (
            f"layer {index}: o_proj consumes {output_width} features but q_proj produces "
            f"{query_width}. Head pruning keeps these equal, so they already disagree."
        )
        raise PruneError(msg)
    for width, name in ((query_width, "q_proj"), (key_width, "k_proj")):
        if width % head_dim:
            msg = f"layer {index}: {name} width {width} is not a multiple of head_dim {head_dim}"
            raise PruneError(msg)

    num_heads = query_width // head_dim
    num_kv_heads = key_width // head_dim
    if num_kv_heads == 0 or num_heads % num_kv_heads:
        msg = (
            f"layer {index}: {num_heads} query heads over {num_kv_heads} key/value heads is "
            "not a whole number of groups, which the attention implementation requires"
        )
        raise PruneError(msg)

    intermediate = _width(mlp, "gate_proj", "out_features", index)
    for name, attribute in (("up_proj", "out_features"), ("down_proj", "in_features")):
        width = _width(mlp, name, attribute, index)
        if width != intermediate:
            msg = (
                f"layer {index}: {name}.{attribute} is {width} but gate_proj produces "
                f"{intermediate}. The three share one set of channels."
            )
            raise PruneError(msg)

    return LayerGroups(
        index=index,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        intermediate=intermediate,
        attention_bias=getattr(attention.q_proj, "bias", None) is not None,
    )


def _width(module: Any, name: str, attribute: str, index: int) -> int:
    projection = getattr(module, name, None)
    width = getattr(projection, attribute, None)
    if not isinstance(width, int) or width <= 0:
        msg = f"layer {index}: {name}.{attribute} is {width!r}, expected a positive integer"
        raise PruneError(msg)
    return width


def _checked_indices(kept: Iterable[int], total: int, unit: str, index: int) -> tuple[int, ...]:
    values = list(kept)
    if not all(isinstance(value, int) and not isinstance(value, bool) for value in values):
        msg = f"layer {index}: {unit} indices must be integers, got {values[:5]}"
        raise PruneError(msg)
    unique = sorted(set(values))
    if len(unique) != len(values):
        msg = f"layer {index}: repeated {unit} index in a selection of {len(values)}"
        raise PruneError(msg)
    if not unique:
        # An empty layer is a removed layer, which the descriptor format
        # represents as a 'layers' component rather than as an empty entry.
        msg = f"layer {index}: kept no {unit}s, which is a removed layer rather than a selection"
        raise PruneError(msg)
    outside = [value for value in unique if not 0 <= value < total]
    if outside:
        msg = f"layer {index}: {unit} indices {outside[:5]} are outside [0, {total})"
        raise PruneError(msg)
    return tuple(unique)


def _expand(units: Sequence[int], width: int) -> tuple[int, ...]:
    return tuple(unit * width + offset for unit in units for offset in range(width))
