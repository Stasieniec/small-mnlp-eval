"""Loader registry. A model spec names a loader; this maps the name to code."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Final

from mnlp_eval.models.base import Translator
from mnlp_eval.models.custom import load_custom
from mnlp_eval.models.hf_causal import load_hf_causal
from mnlp_eval.models.hf_seq2seq import load_hf_seq2seq, load_nllb
from mnlp_eval.prompts import PromptTemplate, get_prompt

if TYPE_CHECKING:
    from mnlp_eval.config import ModelSpec

__all__ = ["available_loaders", "build_translator"]

Loader = Callable[["ModelSpec", PromptTemplate | None], Translator]

_LOADERS: Final[dict[str, Loader]] = {
    "hf_causal": load_hf_causal,
    "hf_seq2seq": load_hf_seq2seq,
    "nllb": load_nllb,
    "custom": load_custom,
}


def available_loaders() -> list[str]:
    return sorted(_LOADERS)


def build_translator(spec: ModelSpec) -> Translator:
    """Instantiate the translator described by ``spec``."""
    try:
        loader = _LOADERS[spec.loader]
    except KeyError:
        known = ", ".join(available_loaders())
        msg = (
            f"model {spec.name}: unknown loader {spec.loader!r}; available loaders are {known}. "
            "Use loader 'custom' with an entrypoint for anything else."
        )
        raise KeyError(msg) from None
    return loader(spec, get_prompt(spec.prompt))
