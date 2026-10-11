"""Loader for a structurally pruned checkpoint.

Layers have different widths, which ``LlamaConfig`` cannot express, so
``config.json`` stays dense and the per-layer widths come from the subnetwork
descriptor next to the weights. Used with ``loader: custom`` and
``entrypoint: recipes.pruned:load``.

Notes:

- ``config.json`` must stay dense: ``Qwen2Config`` derives ``head_dim`` from
  ``num_attention_heads``.
- ``ignore_mismatched_sizes=True`` reinitialises mismatched tensors.
- ``strict=True`` fails on tied models because ``save_pretrained`` drops
  ``lm_head.weight``; the assert below checks the rest.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

__all__ = ["load", "load_model"]

#: Written into the checkpoint directory; the copy under ``subnetworks/`` is identical.
DESCRIPTOR_NAME = "subnetwork.json"


def load_model(checkpoint: str | Path, dtype: str = "auto") -> tuple[Any, Any]:
    """Build the compacted model without a tokenizer; return it with its descriptor."""
    from accelerate import init_empty_weights
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoModelForCausalLM

    from mnlp_eval.analysis.subnetwork import load_subnetwork
    from mnlp_eval.models.loading import resolve_dtype
    from mnlp_eval.prune.compact import plan_from_subnetwork, reshape_model

    directory = Path(checkpoint).expanduser()
    subnetwork = load_subnetwork(directory / DESCRIPTOR_NAME)
    plan = plan_from_subnetwork(subnetwork)

    shards = sorted(directory.glob("*.safetensors"))
    if not shards:
        msg = (
            f"{directory} holds no .safetensors file. A compacted checkpoint is written with "
            "save_pretrained, which produces one; a directory of .bin weights is not read here."
        )
        raise FileNotFoundError(msg)
    state: dict[str, Any] = {}
    for shard in shards:
        state.update(load_file(shard))

    config = AutoConfig.from_pretrained(directory)
    # include_buffers=False: inv_freq is not in the state dict and would stay on meta.
    with init_empty_weights(include_buffers=False):
        model = AutoModelForCausalLM.from_config(config)

    # Per projection: FLAP adds a bias only where units were dropped, so a layer
    # can have an o_proj bias and no down_proj bias.
    biases = {key for key in state if key.endswith((".o_proj.bias", ".down_proj.bias"))}
    reshape_model(
        model,
        plan,
        output_bias=lambda index, suffix: (
            (f"model.layers.{index}.{'self_attn' if suffix == 'o_proj' else 'mlp'}.{suffix}.bias")
            in biases
        ),
    )

    # assign=True breaks the embedding tie, hence tie_weights below.
    disagreement = (
        f"{directory}: the subnetwork descriptor disagrees with the weights. The descriptor "
        "is the record of the per-layer widths, so either it was edited after the checkpoint "
        "was written, or the two came from different runs."
    )
    try:
        missing, unexpected = model.load_state_dict(state, strict=False, assign=True)
    except RuntimeError as exc:
        # A width disagreement lands here naming the layer; a key one below.
        raise ValueError(f"{disagreement}\n{exc}") from exc
    tied = set(getattr(type(model), "_tied_weights_keys", None) or ())
    unexplained = sorted(set(missing) - tied)
    if unexplained or unexpected:
        msg = f"{disagreement} Missing {unexplained[:5]}, unexpected {sorted(unexpected)[:5]}."
        raise ValueError(msg)
    model.tie_weights()

    if dtype != "auto":
        model = model.to(resolve_dtype(dtype))
    model.eval()
    return model, subnetwork


def load(checkpoint: str, dtype: str = "auto", **extra: Any) -> dict[str, Any]:
    """Build a compacted model and its tokenizer.

    Args:
        checkpoint: Directory with the pruned weights, the dense
            ``config.json`` and the subnetwork descriptor.
        dtype: Cast the loaded weights. ``auto`` keeps whatever was saved.
        extra: Anything else from the config's ``kwargs``, kept in the manifest.

    Returns:
        A mapping with ``model``, ``tokenizer``, ``kind`` and ``extra``.
    """
    from transformers import AutoTokenizer

    from mnlp_eval.models.loading import default_device

    model, subnetwork = load_model(checkpoint, dtype)
    model = model.to(default_device())
    tokenizer = AutoTokenizer.from_pretrained(Path(checkpoint).expanduser(), padding_side="left")
    return {
        "model": model,
        "tokenizer": tokenizer,
        "kind": "causal",
        "extra": {
            "subnetwork": subnetwork.name,
            "method": subnetwork.method,
            "unit_sparsity": round(subnetwork.overall_sparsity, 6),
            **extra,
        },
    }
