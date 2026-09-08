"""Prompted decoder-only translation, the path ALMA itself uses."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, ClassVar

from mnlp_eval.languages import Direction
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

__all__ = ["CausalTranslator", "load_hf_causal"]


class CausalTranslator(Translator):
    """A causal language model prompted to translate."""

    kind: ClassVar[str] = "causal"

    @property
    def add_special_tokens(self) -> bool:
        # A chat template already emits BOS, so letting the tokenizer add
        # another one shifts every position by a token and quietly degrades
        # output. Only relevant for X-ALMA style prompts.
        return not self.prompt.requires_chat_template

    def build_input(self, direction: Direction, source: str) -> str:
        text = self.prompt.render(direction, source)
        if not self.prompt.requires_chat_template:
            return text
        applied = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False,
            add_generation_prompt=True,
        )
        return str(applied)


def load_hf_causal(spec: ModelSpec, prompt: PromptTemplate | None = None) -> CausalTranslator:
    """Load a decoder-only model, optionally quantized, with a PEFT adapter."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

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
        padding_side="left",
        trust_remote_code=spec.trust_remote_code,
    )

    load_kwargs: dict[str, Any] = {
        "revision": spec.revision,
        "trust_remote_code": spec.trust_remote_code,
        "low_cpu_mem_usage": True,
        **spec.kwargs,
    }
    quantization_config = build_quantization_config(spec)
    if quantization_config is not None:
        load_kwargs["quantization_config"] = quantization_config
    if spec.attn_implementation:
        load_kwargs["attn_implementation"] = spec.attn_implementation
    if spec.dtype != "auto":
        load_kwargs["torch_dtype"] = resolve_dtype(spec.dtype)
    if spec.device_map:
        load_kwargs["device_map"] = spec.device_map

    model = AutoModelForCausalLM.from_pretrained(reference, **load_kwargs)

    extra: dict[str, Any] = {}
    if spec.adapter is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, spec.adapter.path, revision=spec.adapter.revision)
        extra["adapter"] = spec.adapter.path
        if spec.adapter.merge:
            # Merging removes the adapter's runtime overhead, which would
            # otherwise show up as a latency penalty unrelated to compression.
            model = model.merge_and_unload()
            extra["adapter_merged"] = True

    if not spec.device_map and quantization_config is None:
        model = model.to(default_device())
    model.eval()

    return CausalTranslator(
        spec=spec,
        prompt=prompt,
        model=model,
        tokenizer=tokenizer,
        load_seconds=time.perf_counter() - started,
        checkpoint_dir=resolve_checkpoint_dir(reference, spec.revision),
        extra=extra,
    )
