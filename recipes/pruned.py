"""Loader for a structurally pruned checkpoint.

A global budget gives each layer its own width and ``LlamaConfig`` holds one
scalar ``num_attention_heads``, so the config on disk stays dense and the
per-layer widths come from the subnetwork descriptor beside the weights. Point
a model config here with ``loader: custom`` and
``entrypoint: recipes.pruned:load``; the ``prune`` stage emits one.

Three traps, each of which yields a working model rather than an error:

- ``config.json`` must stay dense. ``Qwen2Config`` never stores ``head_dim``
  and derives it from ``num_attention_heads``, so a reduced head count there
  silently changes it for every layer.
- ``ignore_mismatched_sizes=True`` does not adapt shapes, it randomly
  reinitialises every tensor whose shape disagrees.
- ``strict=True`` cannot be used: both architectures declare
  ``_tied_weights_keys = ["lm_head.weight"]`` and ``save_pretrained`` strips
  tied keys, so a tied model always reports that one missing. The assert below
  restores the strictness that matters.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

__all__ = ["load", "load_model"]

#: Written into the checkpoint directory, so it carries its own shapes. The
#: copy under ``subnetworks/`` is the same bytes and feeds the overlap analysis.
DESCRIPTOR_NAME = "subnetwork.json"


def load_model(checkpoint: str | Path, dtype: str = "auto") -> tuple[Any, Any]:
    """Build the compacted model alone, and return it with its descriptor.

    Separate from :func:`load` so the surgery, the load check and the retie are
    testable on a random model, which has no tokenizer.
    """
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
    # include_buffers=False is load bearing: inv_freq is non-persistent and so
    # absent from the state dict, and would stay on the meta device.
    with init_empty_weights(include_buffers=False):
        model = AutoModelForCausalLM.from_config(config)

    reshape_model(
        model,
        plan,
        output_bias=any(key.endswith("o_proj.bias") for key in state),
    )

    # assign=True is the only way to populate a meta-device model, and it
    # breaks the embedding tie, hence tie_weights below.
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
        checkpoint: Directory holding the pruned weights, its dense
            ``config.json`` and its subnetwork descriptor. The name is fixed by
            the custom-loader contract, or the report's checkpoint-size and
            bits-per-parameter columns come out empty.
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
