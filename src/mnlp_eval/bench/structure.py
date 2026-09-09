"""Structural accounting for pruned models.

Parameter count and checkpoint size answer "how much smaller", but neither
answers "what was removed", and for structured pruning that is the question.
A subnetwork that drops a third of every FFN and one that drops eight whole
layers can report the same parameter count while behaving nothing alike, and
the per-layer profile is what tells them apart.

Everything here is read from the loaded model itself rather than from the
config that produced it. A pruning script that updated its config but not its
weights, or the reverse, is exactly the kind of mistake that would otherwise
be discovered while writing the report.

No torch import. Tensors are used through ``numel``, ``shape``, ``dtype`` and
``count_nonzero``, which keeps this module importable in the metric
environments and testable without a GPU.
"""

from __future__ import annotations

import re
from typing import Any

__all__ = ["STRUCTURE_SCHEMA_VERSION", "describe_structure"]

STRUCTURE_SCHEMA_VERSION = 1

#: Matches the index of a transformer layer inside a parameter name. The text
#: before the index names the stack, so an encoder-decoder model's two stacks
#: stay separate instead of collapsing into one list of doubled layers.
_LAYER_PATTERN = re.compile(r"^(?P<stack>.*?)\.(?P<index>\d+)\.(?P<rest>.*)$")

#: Leaf module names by role. Covers Llama (which is what ALMA is), Mistral,
#: BART and Marian, and T5. A model whose projections are named differently
#: reports its parameter and zero counts as usual and leaves the structural
#: fields null, rather than guessing.
_ROLE_NAMES: dict[str, frozenset[str]] = {
    "attention_query": frozenset({"q_proj", "q", "query"}),
    "attention_output": frozenset({"o_proj", "out_proj", "o"}),
    "ffn_input": frozenset({"up_proj", "gate_proj", "fc1", "wi", "wi_0", "w1"}),
    "ffn_output": frozenset({"down_proj", "fc2", "wo", "w2"}),
}

#: Config fields worth recording next to the measured structure. A mismatch
#: between the two is a bug in whoever produced the checkpoint.
_CONFIG_FIELDS = (
    "model_type",
    "hidden_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "intermediate_size",
    "vocab_size",
    "head_dim",
)


def describe_structure(model: Any, *, count_zeros: bool = True) -> dict[str, Any]:
    """Return the measured structure of ``model``.

    ``count_zeros`` walks every floating-point parameter. It is the only way
    to see a mask that was applied but never materialised, which is what an
    unstructured or a not-yet-compacted structured pruning run produces. It
    costs one reduction per tensor and is cheap next to loading the model.
    """
    config = getattr(model, "config", None)
    config_fields = {
        name: _plain_scalar(getattr(config, name, None))
        for name in _CONFIG_FIELDS
        if getattr(config, name, None) is not None
    }

    stacks: dict[str, dict[int, dict[str, Any]]] = {}
    totals = {"parameters": 0, "zero_parameters": 0, "unmeasured_parameters": 0}
    embedding_parameters = 0

    for name, tensor in _named_parameters(model):
        numel = int(tensor.numel())
        totals["parameters"] += numel
        zeros = _count_zeros(tensor) if count_zeros else None
        if zeros is None:
            totals["unmeasured_parameters"] += numel
        else:
            totals["zero_parameters"] += zeros
        if "embed" in name or name.endswith("lm_head.weight"):
            embedding_parameters += numel

        located = _locate(name)
        if located is None:
            continue
        stack_name, index, leaf = located
        layer = stacks.setdefault(stack_name, {}).setdefault(
            index, {"index": index, "parameters": 0, "zero_parameters": 0, "shapes": {}}
        )
        layer["parameters"] += numel
        if zeros is not None:
            layer["zero_parameters"] += zeros
        role = _role_of(leaf)
        if role is not None and name.endswith(".weight"):
            layer["shapes"].setdefault(role, [int(dim) for dim in tensor.shape])

    head_dim = _head_dim(config_fields)
    described = {
        stack: _describe_stack(layers, head_dim) for stack, layers in sorted(stacks.items())
    }

    result: dict[str, Any] = {
        "schema_version": STRUCTURE_SCHEMA_VERSION,
        "architecture": type(model).__name__,
        "config": config_fields,
        "head_dim": head_dim,
        "stacks": described,
        "total_parameters": totals["parameters"],
        "embedding_parameters": embedding_parameters,
        "zero_parameters": totals["zero_parameters"] if count_zeros else None,
        "unmeasured_parameters": totals["unmeasured_parameters"],
        "zero_fraction": None,
    }
    measured = totals["parameters"] - totals["unmeasured_parameters"]
    if count_zeros and measured > 0:
        result["zero_fraction"] = round(totals["zero_parameters"] / measured, 6)
    if totals["unmeasured_parameters"]:
        result["zero_note"] = (
            "Zeros were counted over floating-point parameters only. "
            f"{totals['unmeasured_parameters']} parameters are stored in an integer "
            "dtype, which is what a quantized checkpoint looks like, and a zero byte "
            "there is a quantization level rather than a pruned weight."
        )
    if not described:
        result["structure_note"] = (
            "No transformer layer stack was recognised in the parameter names, so "
            "per-layer structure is unavailable. Parameter and zero counts are "
            "unaffected."
        )
    return result


def _named_parameters(model: Any) -> list[tuple[str, Any]]:
    getter = getattr(model, "named_parameters", None)
    if getter is None:
        msg = f"{type(model).__name__} has no named_parameters(); cannot measure structure"
        raise TypeError(msg)
    return list(getter())


def _locate(name: str) -> tuple[str, int, str] | None:
    """Split a parameter name into its layer stack, layer index and leaf module."""
    match = _LAYER_PATTERN.match(name)
    if match is None:
        return None
    rest = match.group("rest")
    parts = rest.split(".")
    if len(parts) < 2:
        return None
    return match.group("stack"), int(match.group("index")), parts[-2]


def _role_of(leaf: str) -> str | None:
    for role, names in _ROLE_NAMES.items():
        if leaf in names:
            return role
    return None


def _count_zeros(tensor: Any) -> int | None:
    """Zeros in a floating-point tensor, or None when the dtype makes it meaningless."""
    if not _is_floating(tensor):
        return None
    try:
        nonzero = int(tensor.count_nonzero())
    except (AttributeError, TypeError, RuntimeError):
        return None
    return int(tensor.numel()) - nonzero


def _is_floating(tensor: Any) -> bool:
    dtype = getattr(tensor, "dtype", None)
    flag = getattr(dtype, "is_floating_point", None)
    if isinstance(flag, bool):
        return flag
    return "float" in str(dtype).lower()


def _head_dim(config_fields: dict[str, Any]) -> int | None:
    explicit = config_fields.get("head_dim")
    if isinstance(explicit, int) and explicit > 0:
        return explicit
    hidden = config_fields.get("hidden_size")
    heads = config_fields.get("num_attention_heads")
    if isinstance(hidden, int) and isinstance(heads, int) and heads > 0 and hidden % heads == 0:
        return hidden // heads
    return None


def _describe_stack(layers: dict[int, dict[str, Any]], head_dim: int | None) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for index in sorted(layers):
        layer = layers[index]
        shapes = layer.pop("shapes")
        # A Linear weight is stored as (out_features, in_features), so the
        # inner width of a block is the input width of the module that leaves
        # it: the attention output projection, and the FFN down projection.
        attention_inner = _in_features(shapes.get("attention_output"))
        ffn_intermediate = _in_features(shapes.get("ffn_output"))
        if ffn_intermediate is None:
            ffn_intermediate = _out_features(shapes.get("ffn_input"))
        record = {
            **layer,
            "attention_inner_dim": attention_inner,
            "attention_heads": (
                attention_inner // head_dim
                if attention_inner is not None and head_dim and attention_inner % head_dim == 0
                else None
            ),
            "ffn_intermediate": ffn_intermediate,
        }
        if layer["parameters"]:
            record["zero_fraction"] = round(layer["zero_parameters"] / layer["parameters"], 6)
        records.append(record)

    return {
        "n_layers": len(records),
        "uniform": _is_uniform(records),
        "attention_inner_dim": _spread(records, "attention_inner_dim"),
        "ffn_intermediate": _spread(records, "ffn_intermediate"),
        "parameters": sum(record["parameters"] for record in records),
        "layers": records,
    }


def _in_features(shape: list[int] | None) -> int | None:
    return int(shape[1]) if shape and len(shape) == 2 else None


def _out_features(shape: list[int] | None) -> int | None:
    return int(shape[0]) if shape and len(shape) == 2 else None


def _is_uniform(records: list[dict[str, Any]]) -> bool:
    """Whether every layer has the same width.

    False is the interesting answer: it means pruning allocated different
    budgets to different layers, which is the thing a single sparsity number
    hides.
    """
    widths = {(record["attention_inner_dim"], record["ffn_intermediate"]) for record in records}
    return len(widths) <= 1


def _spread(records: list[dict[str, Any]], key: str) -> dict[str, Any] | None:
    values = [record[key] for record in records if record[key] is not None]
    if not values:
        return None
    return {
        "min": min(values),
        "max": max(values),
        "mean": round(sum(values) / len(values), 2),
        "total": sum(values),
    }


def _plain_scalar(value: Any) -> Any:
    if isinstance(value, bool | int | float | str) or value is None:
        return value
    return str(value)
