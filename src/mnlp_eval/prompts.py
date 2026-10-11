"""Prompt templates, and a hash of their rendered output that each run records."""

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

    Same as ``get_prompt`` in ``utils/utils.py`` of ``fe1ixxu/ALMA`` (the code
    says "to", the upstream README says "into").
    """

    name: ClassVar[str] = "alma"

    def render(self, direction: Direction, source: str) -> str:
        src = direction.source_name
        tgt = direction.target_name
        return f"Translate this from {src} to {tgt}:\n{src}: {source}\n{tgt}:"

    def target_marker(self, direction: Direction) -> str:
        return f"{direction.target_name}:"


class Alma2ChatPrompt(AlmaPrompt):
    """The ALMA prompt wrapped in the model's chat template (X-ALMA). The loader applies it."""

    name: ClassVar[str] = "alma_chat"
    requires_chat_template: ClassVar[bool] = True


class PassthroughPrompt(PromptTemplate):
    """No prompt, for encoder-decoder models such as Marian and NLLB."""

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


# Fixed probe inputs for prompt_hash. Changing them changes every hash.
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
