"""Prompt templates, and a hash that pins prompt behaviour into a manifest.

Comparing a compressed model against a baseline is only meaningful if both saw
byte-identical prompts. Every run records :func:`prompt_hash`, which is derived
from rendering a fixed set of probe inputs, so any edit to a template changes
the recorded hash and the two runs stop being comparable by construction.
"""

from __future__ import annotations

import abc
import hashlib
from typing import ClassVar, Final

from mnlp_eval.languages import Direction, parse_direction

__all__ = [
    "Alma2ChatPrompt",
    "AlmaPrompt",
    "Aya101Prompt",
    "PassthroughPrompt",
    "PromptTemplate",
    "available_prompts",
    "get_prompt",
    "prompt_hash",
]


class PromptTemplate(abc.ABC):
    """Turns a source sentence into the exact text handed to the model."""

    name: ClassVar[str]

    #: When true, the causal-LM loader wraps :meth:`render` output in the
    #: tokenizer's chat template before tokenizing. Used by X-ALMA.
    requires_chat_template: ClassVar[bool] = False

    @abc.abstractmethod
    def render(self, direction: Direction, source: str) -> str:
        """Return the prompt text for one source sentence."""

    def target_marker(self, direction: Direction) -> str:
        """Return the trailing cue that precedes the translation.

        Post-processing strips this marker if the model echoes it back. Return
        an empty string for templates that have no such cue.
        """
        del direction
        return ""


class AlmaPrompt(PromptTemplate):
    """The prompt used by ALMA and ALMA-R.

    Byte-identical to ``get_prompt`` in ``utils/utils.py`` of ``fe1ixxu/ALMA``.
    Note that the upstream README writes "into" while the upstream code writes
    "to"; the code is what produced the published scores, so "to" is correct.
    """

    name: ClassVar[str] = "alma"

    def render(self, direction: Direction, source: str) -> str:
        src = direction.source_name
        tgt = direction.target_name
        return f"Translate this from {src} to {tgt}:\n{src}: {source}\n{tgt}:"

    def target_marker(self, direction: Direction) -> str:
        return f"{direction.target_name}:"


class Alma2ChatPrompt(AlmaPrompt):
    """The ALMA prompt wrapped in the model's chat template, as X-ALMA needs.

    The wrapping itself is applied by the loader, which owns the tokenizer.
    """

    name: ClassVar[str] = "alma_chat"
    requires_chat_template: ClassVar[bool] = True


class PassthroughPrompt(PromptTemplate):
    """No prompt at all, for encoder-decoder models such as Marian and NLLB.

    These models select the target language through a forced BOS token or a
    dedicated checkpoint, not through the input text.
    """

    name: ClassVar[str] = "passthrough"

    def render(self, direction: Direction, source: str) -> str:
        del direction
        return source


class Aya101Prompt(PromptTemplate):
    """Instruction prefix used by ALMA when evaluating Aya-101 baselines."""

    name: ClassVar[str] = "aya101"

    def render(self, direction: Direction, source: str) -> str:
        return f"Translate to {direction.target_name}: {source}"


_REGISTRY: Final[dict[str, type[PromptTemplate]]] = {
    cls.name: cls for cls in (AlmaPrompt, Alma2ChatPrompt, PassthroughPrompt, Aya101Prompt)
}


def available_prompts() -> list[str]:
    return sorted(_REGISTRY)


def get_prompt(name: str) -> PromptTemplate:
    """Look up a prompt template by name."""
    try:
        return _REGISTRY[name]()
    except KeyError:
        known = ", ".join(available_prompts())
        msg = f"unknown prompt template {name!r}; available templates are: {known}"
        raise KeyError(msg) from None


# Fixed probes. Chosen to exercise both translation directions and a
# non-ASCII source. Changing this tuple invalidates every recorded prompt
# hash, so it must stay frozen.
_PROBES: Final[tuple[tuple[str, str], ...]] = (
    ("de-en", "Das ist ein Test."),
    ("en-de", "This is a test."),
    ("is-en", "Petta er profun."),
)


def prompt_hash(template: PromptTemplate) -> str:
    """Return a stable 16-character digest of a template's rendered output."""
    digest = hashlib.sha256()
    digest.update(template.name.encode())
    digest.update(b"\x00chat=" + str(template.requires_chat_template).encode())
    for spec, sentence in _PROBES:
        direction = parse_direction(spec)
        digest.update(b"\x00")
        digest.update(template.render(direction, sentence).encode())
        digest.update(b"\x01")
        digest.update(template.target_marker(direction).encode())
    return digest.hexdigest()[:16]
