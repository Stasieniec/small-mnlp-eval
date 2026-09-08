"""The translator interface every model plugs into.

A translator turns source sentences into raw model output. It does not
post-process, score, or know anything about metrics. Keeping that boundary
sharp is what lets a pruned model with custom modelling code reuse the entire
downstream pipeline.

The batching loop lives here rather than in each loader, so a new backend only
has to say how to build its inputs and how to read its outputs.
"""

from __future__ import annotations

import abc
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

from mnlp_eval.languages import Direction, max_source_length_for
from mnlp_eval.prompts import PromptTemplate

if TYPE_CHECKING:
    from mnlp_eval.config import DecodeSpec, ModelSpec

__all__ = ["SegmentOutput", "Translator", "TranslatorInfo"]


@dataclass
class SegmentOutput:
    """Raw output for one source sentence, before post-processing."""

    raw_text: str
    n_source_tokens: int = 0
    n_generated_tokens: int = 0
    #: True when generation stopped because the token budget ran out rather
    #: than because the model emitted EOS. Whether that truncated the
    #: translation itself is decided downstream, once the hypothesis is known.
    hit_token_budget: bool = False
    source_truncated: bool = False


@dataclass
class TranslatorInfo:
    """Static facts about a loaded model, for the manifest and for bench."""

    kind: str
    load_seconds: float
    total_parameters: int
    non_embedding_parameters: int
    dtype: str
    device: str
    checkpoint_dir: str | None = None
    extra: dict[str, Any] | None = None


class Translator(abc.ABC):
    """Base class implementing the shared generation loop."""

    kind: ClassVar[str] = "abstract"

    def __init__(
        self,
        spec: ModelSpec,
        prompt: PromptTemplate,
        model: Any,
        tokenizer: Any,
        *,
        load_seconds: float = 0.0,
        checkpoint_dir: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        self.spec = spec
        self.prompt = prompt
        self.model = model
        self.tokenizer = tokenizer
        self._load_seconds = load_seconds
        self._checkpoint_dir = checkpoint_dir
        self._extra = extra or {}
        self._configure_tokenizer()

    # ------------------------------------------------------------------
    # Subclass hooks
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def build_input(self, direction: Direction, source: str) -> str:
        """Return the exact text fed to the tokenizer for one sentence."""

    def generate_kwargs(self, direction: Direction) -> dict[str, Any]:
        """Extra ``generate`` arguments for a direction, such as a forced BOS."""
        del direction
        return {}

    @property
    def strips_prompt_from_output(self) -> bool:
        """Whether ``generate`` echoes the input back in its output.

        True for decoder-only models, false for encoder-decoder models.
        """
        return True

    @property
    def add_special_tokens(self) -> bool:
        """Whether the tokenizer should add its own special tokens."""
        return True

    def prepare_direction(self, direction: Direction) -> None:
        """Hook called once before translating a direction.

        Used by models that carry language state on the tokenizer, such as
        NLLB's ``src_lang``.
        """
        del direction

    # ------------------------------------------------------------------
    # Shared machinery
    # ------------------------------------------------------------------

    def _configure_tokenizer(self) -> None:
        tokenizer = self.tokenizer
        if tokenizer.pad_token_id is None:
            # Llama-family checkpoints, ALMA included, ship without a pad
            # token. Reusing EOS is the standard choice and is safe here
            # because padding is masked out and never generated into.
            tokenizer.pad_token = tokenizer.eos_token
        if self.strips_prompt_from_output:
            # Right padding would put pad tokens between the prompt and the
            # continuation, which silently corrupts decoder-only generation.
            tokenizer.padding_side = "left"

    @property
    def device(self) -> Any:
        return getattr(self.model, "device", "cpu")

    def info(self) -> TranslatorInfo:
        total, non_embedding = self.parameter_counts()
        dtype = str(getattr(self.model, "dtype", self.spec.dtype)).replace("torch.", "")
        return TranslatorInfo(
            kind=self.kind,
            load_seconds=round(self._load_seconds, 3),
            total_parameters=total,
            non_embedding_parameters=non_embedding,
            dtype=dtype,
            device=str(self.device),
            checkpoint_dir=self._checkpoint_dir,
            extra=dict(self._extra) or None,
        )

    def parameter_counts(self) -> tuple[int, int]:
        """Return total and non-embedding parameter counts.

        Non-embedding is reported separately because embedding tables dominate
        small models and are usually left untouched by pruning, so a raw total
        understates how much of the transformer was actually removed.
        """
        if not hasattr(self.model, "parameters"):
            return 0, 0
        try:
            import torch.nn as nn
        except ImportError:  # pragma: no cover
            return 0, 0
        total = sum(parameter.numel() for parameter in self.model.parameters())
        embedding = sum(
            parameter.numel()
            for module in self.model.modules()
            if isinstance(module, nn.Embedding)
            for parameter in module.parameters(recurse=False)
        )
        return total, max(total - embedding, 0)

    def translate(
        self,
        direction: Direction,
        sources: Sequence[str],
        decode: DecodeSpec,
    ) -> list[SegmentOutput]:
        """Translate ``sources`` and return raw output in the original order."""
        import torch

        max_source_length = (
            max_source_length_for(direction, decode.max_source_length)
            if decode.respect_alma_source_length_override
            else decode.max_source_length
        )
        self.prepare_direction(direction)
        add_special = self.add_special_tokens
        texts = [self.build_input(direction, source) for source in sources]
        encoded = [
            self.tokenizer(
                text,
                add_special_tokens=add_special,
                truncation=True,
                max_length=max_source_length,
            )
            for text in texts
        ]
        # A truncated source is a silent quality ceiling, so it is counted.
        untruncated_lengths = [
            len(self.tokenizer(text, add_special_tokens=add_special)["input_ids"]) for text in texts
        ]

        order = list(range(len(texts)))
        if decode.sort_by_length:
            order.sort(key=lambda index: len(encoded[index]["input_ids"]))

        results: list[SegmentOutput | None] = [None] * len(texts)
        gen_kwargs = self._resolve_generate_kwargs(direction, decode)

        for start in range(0, len(order), decode.batch_size):
            indices = order[start : start + decode.batch_size]
            batch = self.tokenizer.pad(
                [encoded[index] for index in indices],
                padding=True,
                return_tensors="pt",
            ).to(self.device)
            with torch.inference_mode():
                sequences = self.model.generate(**batch, **gen_kwargs)
            prompt_length = batch["input_ids"].shape[1] if self.strips_prompt_from_output else 0
            for position, index in enumerate(indices):
                continuation = sequences[position, prompt_length:]
                generated, complete = self._trim_continuation(continuation)
                results[index] = SegmentOutput(
                    raw_text=self.tokenizer.decode(generated, skip_special_tokens=True),
                    n_source_tokens=len(encoded[index]["input_ids"]),
                    n_generated_tokens=len(generated),
                    hit_token_budget=not complete,
                    source_truncated=untruncated_lengths[index] > max_source_length,
                )

        missing = [index for index, value in enumerate(results) if value is None]
        if missing:  # pragma: no cover
            msg = f"generation produced no output for {len(missing)} segment(s)"
            raise RuntimeError(msg)
        return [value for value in results if value is not None]

    def _resolve_generate_kwargs(self, direction: Direction, decode: DecodeSpec) -> dict[str, Any]:
        """Build ``generate`` arguments, pinning every sampling knob explicitly.

        Several ALMA-family checkpoints inherit a ``generation_config.json``
        from Llama 2 that enables sampling. Passing these values explicitly
        stops the checkpoint from quietly deciding how it is evaluated.
        """
        kwargs: dict[str, Any] = {
            "do_sample": decode.do_sample,
            "num_beams": decode.num_beams,
            "max_new_tokens": decode.max_new_tokens,
            "num_return_sequences": 1,
            "length_penalty": decode.length_penalty,
            "early_stopping": decode.early_stopping,
            "use_cache": True,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if decode.do_sample:
            for label in ("temperature", "top_p", "top_k"):
                value = getattr(decode, label)
                if value is not None:
                    kwargs[label] = value
        else:
            # Explicit nulls keep transformers from reinstating checkpoint
            # defaults and warning about unused sampling parameters.
            kwargs.update({"temperature": None, "top_p": None, "top_k": None})
        kwargs.update(self.generate_kwargs(direction))
        return kwargs

    def _trim_continuation(self, continuation: Any) -> tuple[Any, bool]:
        """Cut a continuation at its first EOS and say whether one was found.

        A continuation with no EOS hit the token budget, which is a truncated
        translation rather than a finished one. Compressed models run into this
        far more often than baselines, so it is tracked per segment.
        """
        eos_id = self.tokenizer.eos_token_id
        if eos_id is None:
            return continuation, True
        matches = (continuation == eos_id).nonzero()
        if matches.numel() == 0:
            return continuation, False
        first = int(matches[0].item() if matches.dim() == 1 else matches[0, 0].item())
        return continuation[: first + 1], True

    def close(self) -> None:
        """Release GPU memory. Safe to call more than once."""
        self.model = None
        try:
            import gc

            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:  # pragma: no cover
            pass

    def __enter__(self) -> Translator:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def timed_load(loader: Any, *args: Any, **kwargs: Any) -> tuple[Any, float]:
    """Call ``loader`` and return its result with the elapsed seconds."""
    start = time.perf_counter()
    result = loader(*args, **kwargs)
    return result, time.perf_counter() - start
