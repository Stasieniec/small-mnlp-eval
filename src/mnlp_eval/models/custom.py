"""The escape hatch for models that need their own loading code.

Structured pruning usually cannot be expressed as a checkpoint plus a config:
the architecture itself changed, so loading needs the code that produced it.
Rather than absorbing that code into the framework, a spec points at a function
and the framework calls it.

The contract is deliberately forgiving about the return value, because the
point is that a teammate writes a handful of lines and gets the whole
pipeline. Accepted returns:

* a :class:`~mnlp_eval.models.base.Translator`, for full control
* a ``(model, tokenizer)`` tuple, the common case
* a mapping with ``model`` and ``tokenizer`` keys, optionally ``kind`` and
  ``extra``

Whatever is returned, the tokenizer must behave like a Hugging Face one. The
generation loop reads ``pad_token_id``, ``eos_token_id``, ``eos_token`` and
``padding_side``, and calls the object itself, ``pad()`` and ``decode()``. A
bare callable is not enough, and the resulting AttributeError says nothing
about this contract, so it is stated here.

The ``checkpoint`` kwarg name is load-bearing when ``model_name_or_path`` is
unset: it is what the framework inspects to find the checkpoint on disk, and
naming it something else leaves checkpoint size, bits per parameter and the
disk compression ratio empty.
"""

from __future__ import annotations

import dataclasses
import importlib
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mnlp_eval.models.base import Translator
from mnlp_eval.models.hf_causal import CausalTranslator
from mnlp_eval.models.hf_seq2seq import Seq2SeqTranslator
from mnlp_eval.models.loading import resolve_checkpoint_dir
from mnlp_eval.prompts import PromptTemplate, get_prompt

if TYPE_CHECKING:
    from mnlp_eval.config import ModelSpec

__all__ = ["load_custom"]


def _import_entrypoint(entrypoint: str) -> Any:
    module_name, _, attribute = entrypoint.partition(":")
    # A recipe usually lives in the repository checkout rather than an
    # installed package, so the working directory has to be importable.
    cwd = str(Path.cwd())
    if cwd not in sys.path:
        sys.path.insert(0, cwd)
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        msg = (
            f"cannot import {module_name!r} from entrypoint {entrypoint!r}. "
            f"Working directory {cwd} is on sys.path; check the module path and that "
            "any package directories contain an __init__.py."
        )
        raise ImportError(msg) from exc
    try:
        return getattr(module, attribute)
    except AttributeError:
        available = ", ".join(name for name in dir(module) if not name.startswith("_"))
        msg = f"{module_name!r} has no attribute {attribute!r}. Available names: {available}"
        raise AttributeError(msg) from None


def _infer_kind(model: Any) -> str:
    config = getattr(model, "config", None)
    if getattr(config, "is_encoder_decoder", False):
        return "seq2seq"
    return "causal"


#: Fields the framework applies for the built-in loaders but cannot apply for
#: a custom one, because the entrypoint owns loading. Setting them is almost
#: always a mistake, and for a compression project a silently ignored
#: `quantization:` block means evaluating an unquantized model under a
#: quantized model's name.
_UNAPPLIED_FIELDS = (
    "dtype",
    "quantization",
    "adapter",
    "device_map",
    "attn_implementation",
    "trust_remote_code",
)


def _reject_unapplied_fields(spec: ModelSpec) -> None:
    defaults = {
        field.name: field.default
        for field in dataclasses.fields(spec)
        if field.name in _UNAPPLIED_FIELDS
    }
    offenders = [name for name, default in defaults.items() if getattr(spec, name) != default]
    if not offenders:
        return
    msg = (
        f"model {spec.name}: loader 'custom' cannot apply {', '.join(offenders)}, because "
        "the entrypoint does the loading. Pass what the loader needs through 'kwargs' and "
        "apply it there, or use a built-in loader."
    )
    raise ValueError(msg)


def load_custom(spec: ModelSpec, prompt: PromptTemplate | None = None) -> Translator:
    """Build a translator by calling a user-supplied entrypoint."""
    if not spec.entrypoint:
        msg = f"model {spec.name}: loader 'custom' requires 'entrypoint'"
        raise ValueError(msg)
    _reject_unapplied_fields(spec)

    prompt = prompt or get_prompt(spec.prompt)
    loader = _import_entrypoint(spec.entrypoint)
    started = time.perf_counter()
    result = loader(**spec.kwargs)
    load_seconds = time.perf_counter() - started

    if isinstance(result, Translator):
        return result

    kind: str | None = None
    extra: dict[str, Any] = {"entrypoint": spec.entrypoint}
    if isinstance(result, Mapping):
        try:
            model = result["model"]
            tokenizer = result["tokenizer"]
        except KeyError:
            msg = (
                f"{spec.entrypoint} returned a mapping with keys {sorted(result)}; "
                "'model' and 'tokenizer' are required"
            )
            raise TypeError(msg) from None
        kind = result.get("kind")
        extra.update(result.get("extra") or {})
    elif isinstance(result, tuple) and len(result) == 2:
        model, tokenizer = result
    else:
        msg = (
            f"{spec.entrypoint} returned {type(result).__name__}. Return a Translator, "
            "a (model, tokenizer) tuple, or a mapping with 'model' and 'tokenizer' keys."
        )
        raise TypeError(msg)

    kind = kind or _infer_kind(model)
    translator_class = Seq2SeqTranslator if kind == "seq2seq" else CausalTranslator
    checkpoint = spec.model_name_or_path or spec.kwargs.get("checkpoint")
    return translator_class(
        spec=spec,
        prompt=prompt,
        model=model,
        tokenizer=tokenizer,
        load_seconds=load_seconds,
        checkpoint_dir=resolve_checkpoint_dir(str(checkpoint), spec.revision)
        if checkpoint
        else None,
        extra=extra,
    )
