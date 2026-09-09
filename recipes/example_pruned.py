"""Template for a model that cannot be loaded from a config alone.

Structured pruning usually changes the architecture, so the checkpoint needs
the code that produced it. Copy this file, replace the loading logic, and point
a model config at it:

    name: alma-7b-wanda-50
    loader: custom
    prompt: alma
    baseline: alma-7b
    entrypoint: recipes.example_pruned:load
    kwargs:
      checkpoint: /scratch-shared/$USER/wanda-50
      sparsity: 0.5
    compression:
      family: pruning
      nominal_sparsity: 0.5
      pruned_for: multi
      subnetwork: subnetworks/wanda-50.json

Everything downstream is unchanged: prompting, batching, hypothesis extraction,
all metrics, efficiency measurement, compression ratios, and significance
against the baseline. The compression block is what the structure, resource
tier and transfer tables are built from; see docs/plugging-in-a-model.md.

A structured pruning method should also write a subnetwork descriptor at
pruning time, listing the indices it kept. It cannot be recovered from the
saved checkpoint afterwards. See docs/subnetworks.md.
"""

from __future__ import annotations

from typing import Any

__all__ = ["load"]


def load(
    checkpoint: str,
    sparsity: float | None = None,
    dtype: str = "bfloat16",
    **extra: Any,
) -> dict[str, Any]:
    """Build a pruned model and its tokenizer.

    Args:
        checkpoint: Directory holding the pruned weights.
        sparsity: Recorded in the run manifest so a sweep is self-describing.
        dtype: Weight dtype to load in.
        extra: Anything else passed through the config's ``kwargs``.

    Returns:
        A mapping with ``model`` and ``tokenizer``. ``kind`` and ``extra`` are
        optional; ``extra`` is stored in the manifest, which is the right place
        for method metadata that a report should be able to cite.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[dtype]

    # Replace this with whatever your pruning method needs. Unstructured
    # methods that only zero weights load through the standard path; structured
    # methods that change layer shapes usually need trust_remote_code, or their
    # own model class imported here.
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    )
    model.eval()
    if torch.cuda.is_available():
        model = model.to("cuda")

    return {
        "model": model,
        "tokenizer": tokenizer,
        "kind": "causal",
        "extra": {"sparsity": sparsity, "method": "example", **extra},
    }
