"""Encoder-decoder translation backends.

These are not compression targets. They exist as reference systems: a real MT
model with known quality proves the metric pipeline produces sensible numbers,
which a prompted 0.5B model cannot do. A harness that has only ever seen
garbage output cannot tell "pipeline correct" from "pipeline broken".
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, ClassVar

from mnlp_eval.languages import NLLB_CODE, Direction
from mnlp_eval.models.base import Translator
from mnlp_eval.models.loading import (
    build_quantization_config,
    default_device,
    resolve_checkpoint_dir,
    resolve_dtype,
)
from mnlp_eval.prompts import PromptTemplate, get_prompt

if TYPE_CHECKING:
    from mnlp_eval.config import ModelSpec

__all__ = ["NllbTranslator", "Seq2SeqTranslator", "load_hf_seq2seq", "load_nllb"]


class Seq2SeqTranslator(Translator):
    """A generic encoder-decoder model, such as Marian or mT5."""

    kind: ClassVar[str] = "seq2seq"

    @property
    def strips_prompt_from_output(self) -> bool:
        # Encoder-decoder output contains only the decoded target sequence.
        return False

    def build_input(self, direction: Direction, source: str) -> str:
        return self.prompt.render(direction, source)


class NllbTranslator(Seq2SeqTranslator):
    """NLLB, which selects languages through tokenizer state and a forced BOS."""

    kind: ClassVar[str] = "seq2seq_nllb"

    def prepare_direction(self, direction: Direction) -> None:
        self.tokenizer.src_lang = NLLB_CODE[direction.source]

    def generate_kwargs(self, direction: Direction) -> dict[str, Any]:
        target = NLLB_CODE[direction.target]
        token_id = self.tokenizer.convert_tokens_to_ids(target)
        if token_id is None or token_id == self.tokenizer.unk_token_id:
            msg = f"tokenizer has no language token for {target!r}"
            raise RuntimeError(msg)
        return {"forced_bos_token_id": token_id}


def _load(
    spec: ModelSpec,
    prompt: PromptTemplate | None,
    translator_class: type[Seq2SeqTranslator],
) -> Seq2SeqTranslator:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    prompt = prompt or get_prompt(spec.prompt)
    # ModelSpec.validate already enforces this for every non-custom loader;
    # restating it here keeps the guarantee local and checkable.
    if not spec.model_name_or_path:
        msg = f"model {spec.name}: loader {spec.loader!r} requires 'model_name_or_path'"
        raise ValueError(msg)
    reference = spec.model_name_or_path
    started = time.perf_counter()

    tokenizer = AutoTokenizer.from_pretrained(
        spec.tokenizer_name_or_path or reference,
        revision=spec.revision,
        trust_remote_code=spec.trust_remote_code,
    )

    load_kwargs: dict[str, Any] = {
        "revision": spec.revision,
        "trust_remote_code": spec.trust_remote_code,
        **spec.kwargs,
    }
    quantization_config = build_quantization_config(spec)
    if quantization_config is not None:
        load_kwargs["quantization_config"] = quantization_config
    if spec.dtype != "auto":
        load_kwargs["torch_dtype"] = resolve_dtype(spec.dtype)
    if spec.device_map:
        load_kwargs["device_map"] = spec.device_map

    model = AutoModelForSeq2SeqLM.from_pretrained(reference, **load_kwargs)
    if not spec.device_map and quantization_config is None:
        model = model.to(default_device())
    model.eval()

    return translator_class(
        spec=spec,
        prompt=prompt,
        model=model,
        tokenizer=tokenizer,
        load_seconds=time.perf_counter() - started,
        checkpoint_dir=resolve_checkpoint_dir(reference, spec.revision),
    )


def load_hf_seq2seq(spec: ModelSpec, prompt: PromptTemplate | None = None) -> Seq2SeqTranslator:
    return _load(spec, prompt, Seq2SeqTranslator)


def load_nllb(spec: ModelSpec, prompt: PromptTemplate | None = None) -> Seq2SeqTranslator:
    return _load(spec, prompt, NllbTranslator)
