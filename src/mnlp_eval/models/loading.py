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

#: Weight-file formats, in the order a checkpoint's primary format is chosen.
#: Many Hub repositories ship the same weights more than once (PyTorch plus
#: safetensors plus Flax), so summing every weight-ish file inflated the disk
#: figure severalfold and corrupted the disk compression ratio. Only the first
#: format present is counted.
_FORMAT_PRIORITY = (".safetensors", ".bin", ".pt", ".pth", ".gguf", ".ckpt")

#: File stems that carry a weight suffix but are not model weights. Trainer
#: state in particular is often several times the size of the model.
_NON_WEIGHT_STEMS = frozenset(
    {
        "training_args",
        "optimizer",
        "scheduler",
        "rng_state",
        "trainer_state",
        "scaler",
    }
)

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

        # str() is not redundant: snapshot_download is typed Any when
        # huggingface-hub is absent, which is the case in the lint environment.
        return str(snapshot_download(model_name_or_path, revision=revision, local_files_only=True))
    except Exception:
        return None


def checkpoint_bytes(directory: str | Path | None) -> int | None:
    """Size of a checkpoint's weights on disk, counting one format only.

    Follows symlinks, since the Hugging Face cache stores blobs separately from
    the snapshot tree and the snapshot entries are links. Counts only the
    highest-priority format present, so a repository shipping both
    ``pytorch_model.bin`` and ``model.safetensors`` is not counted twice, and
    skips trainer state that happens to share a weight suffix.
    """
    if directory is None:
        return None
    root = Path(directory)
    if not root.is_dir():
        return None

    by_format: dict[str, int] = {}
    for path in root.rglob("*"):
        suffix = path.suffix.lower()
        if suffix not in _FORMAT_PRIORITY or not path.is_file():
            continue
        if path.stem.lower() in _NON_WEIGHT_STEMS:
            continue
        by_format[suffix] = by_format.get(suffix, 0) + path.stat().st_size

    for suffix in _FORMAT_PRIORITY:
        if by_format.get(suffix):
            return by_format[suffix]
    return None
