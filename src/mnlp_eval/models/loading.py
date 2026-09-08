"""Shared helpers for building models from a :class:`ModelSpec`.

Kept separate from the translator classes so that a custom loader written by a
teammate can reuse the dtype, quantization, and checkpoint-resolution logic
without inheriting anything.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mnlp_eval.config import ModelSpec

__all__ = [
    "build_quantization_config",
    "checkpoint_bytes",
    "default_device",
    "resolve_checkpoint_dir",
    "resolve_dtype",
]

#: Weight file suffixes counted towards on-disk checkpoint size. Tokenizer
#: files and configs are excluded: they are identical across compression
#: variants and would dilute the compression ratio.
_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".gguf", ".ckpt", ".msgpack")

#: Quantization methods whose configuration lives inside the checkpoint. For
#: these, passing a config is optional and only needed to override defaults.
_CHECKPOINT_DEFINED = frozenset({"gptq", "awq", "compressed_tensors"})


def resolve_dtype(name: str) -> Any:
    """Map a dtype name from a config file onto a torch dtype."""
    import torch

    if name == "auto":
        return "auto"
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    try:
        return mapping[name]
    except KeyError:
        msg = f"unsupported dtype {name!r}; expected one of auto, float32, float16, bfloat16"
        raise ValueError(msg) from None


def default_device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def build_quantization_config(spec: ModelSpec) -> Any | None:
    """Translate a :class:`QuantizationSpec` into a transformers config object.

    Returns ``None`` when the checkpoint already carries its own quantization
    metadata and no overrides were requested, which is the normal case for
    GPTQ, AWQ and compressed-tensors checkpoints.
    """
    quant = spec.quantization
    if quant is None:
        return None

    options = dict(quant.options)
    if quant.method in _CHECKPOINT_DEFINED and not options:
        return None

    if quant.method == "bitsandbytes":
        from transformers import BitsAndBytesConfig

        compute_dtype = options.pop("bnb_4bit_compute_dtype", None)
        if isinstance(compute_dtype, str):
            options["bnb_4bit_compute_dtype"] = resolve_dtype(compute_dtype)
        elif compute_dtype is not None:
            options["bnb_4bit_compute_dtype"] = compute_dtype
        return BitsAndBytesConfig(**options)

    class_names = {
        "gptq": "GPTQConfig",
        "awq": "AwqConfig",
        "hqq": "HqqConfig",
        "torchao": "TorchAoConfig",
        "compressed_tensors": "CompressedTensorsConfig",
    }
    import transformers

    class_name = class_names[quant.method]
    config_class = getattr(transformers, class_name, None)
    if config_class is None:
        msg = (
            f"this transformers version does not provide {class_name}, needed for "
            f"quantization method {quant.method!r}. Upgrade transformers, or drop the "
            "inline options and let the checkpoint's own metadata be used."
        )
        raise RuntimeError(msg)
    return config_class(**options)


def resolve_checkpoint_dir(model_name_or_path: str, revision: str | None = None) -> str | None:
    """Return the local directory holding a checkpoint, if it can be found.

    Works for a local path and for a Hub repository already present in the
    cache. Returns ``None`` rather than raising, because checkpoint size is a
    reported metric and not a precondition for evaluating a model.
    """
    candidate = Path(model_name_or_path).expanduser()
    if candidate.is_dir():
        return str(candidate.resolve())
    try:
        from huggingface_hub import snapshot_download

        return snapshot_download(model_name_or_path, revision=revision, local_files_only=True)
    except Exception:
        return None


def checkpoint_bytes(directory: str | Path | None) -> int | None:
    """Sum the weight files in a checkpoint directory.

    Follows symlinks, since the Hugging Face cache stores blobs separately from
    the snapshot tree and the snapshot entries are links.
    """
    if directory is None:
        return None
    root = Path(directory)
    if not root.is_dir():
        return None
    total = 0
    found = False
    for path in root.rglob("*"):
        if path.suffix.lower() in _WEIGHT_SUFFIXES and path.is_file():
            total += path.stat().st_size
            found = True
    return total if found else None
