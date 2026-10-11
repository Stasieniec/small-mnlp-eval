"""Translator interface and the shared batched generation loop.

A translator turns source sentences into raw model output; post-processing and
scoring happen elsewhere.
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
    #: Padded width of the batch this segment was in (used for the FLOPs estimate).
    n_padded_source_tokens: int = 0
    n_generated_tokens: int = 0
    #: Generation stopped at the token budget without emitting EOS.
    hit_token_budget: bool = False
    source_truncated: bool = False
    #: Only captured for calibration.
    input_token_ids: list[int] | None = None
    generated_token_ids: list[int] | None = None


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
    unpinned_generation_settings: dict[str, Any] | None = None
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
        self._terminators: frozenset[int] | None = None
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
            # Llama-family checkpoints (ALMA included) have no pad token.
            tokenizer.pad_token = tokenizer.eos_token
        if self.strips_prompt_from_output:
            # Decoder-only generation needs left padding.
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
            unpinned_generation_settings=self.unpinned_generation_settings() or None,
            extra=dict(self._extra) or None,
        )

    def parameter_counts(self) -> tuple[int, int]:
        """Return total and non-embedding parameter counts."""
        if not hasattr(self.model, "parameters"):
            return 0, 0
        try:
            import torch.nn as nn
        except ImportError:  # pragma: no cover
            return 0, 0

        # Prefer num_parameters(), which counts packed 4-bit weights correctly.
        total = 0
        counter = getattr(self.model, "num_parameters", None)
        if callable(counter):
            try:
                total = int(counter())
            except Exception:  # a failure here just falls back to the manual sum
                total = 0
        if not total:
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
        *,
        capture_token_ids: bool = False,
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
        # Untruncated lengths, to flag truncated sources.
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
            # Encoder-decoder output starts with decoder_start_token_id; skip it.
            prompt_length = batch["input_ids"].shape[1] if self.strips_prompt_from_output else 1
            for position, index in enumerate(indices):
                continuation = sequences[position, prompt_length:]
                generated, n_content, complete = self._trim_continuation(continuation)
                results[index] = SegmentOutput(
                    raw_text=self.tokenizer.decode(generated, skip_special_tokens=True),
                    n_source_tokens=len(encoded[index]["input_ids"]),
                    n_padded_source_tokens=int(batch["input_ids"].shape[1]),
                    n_generated_tokens=n_content,
                    hit_token_budget=not complete,
                    source_truncated=untruncated_lengths[index] > max_source_length,
                    input_token_ids=list(encoded[index]["input_ids"])
                    if capture_token_ids
                    else None,
                    generated_token_ids=generated.tolist() if capture_token_ids else None,
                )

        missing = [index for index, value in enumerate(results) if value is None]
        if missing:  # pragma: no cover
            msg = f"generation produced no output for {len(missing)} segment(s)"
            raise RuntimeError(msg)
        return [value for value in results if value is not None]

    #: Generation settings always set from the decode spec, overriding the
    #: checkpoint's generation_config.json. Other checkpoint settings (e.g.
    #: forced_bos_token_id for NLLB) are kept and recorded by
    #: :meth:`unpinned_generation_settings`.
    POLICY_KEYS: ClassVar[frozenset[str]] = frozenset(
        {
            "do_sample",
            "num_beams",
            "max_new_tokens",
            "num_return_sequences",
            "length_penalty",
            "early_stopping",
            "repetition_penalty",
            "no_repeat_ngram_size",
            "num_beam_groups",
            "diversity_penalty",
            "temperature",
            "top_p",
            "top_k",
            "typical_p",
            "epsilon_cutoff",
            "eta_cutoff",
            "min_new_tokens",
            "min_length",
            "penalty_alpha",
        }
    )

    def unpinned_generation_settings(self) -> dict[str, Any]:
        """Non-default checkpoint generation settings outside :attr:`POLICY_KEYS`."""
        generation_config = getattr(self.model, "generation_config", None)
        differ = getattr(generation_config, "to_diff_dict", None)
        if not callable(differ):
            return {}
        try:
            diff = differ()
        except Exception:
            return {}
        return {
            key: value
            for key, value in sorted(diff.items())
            if key not in self.POLICY_KEYS and key != "transformers_version"
        }

    def _resolve_generate_kwargs(self, direction: Direction, decode: DecodeSpec) -> dict[str, Any]:
        """Build ``generate`` arguments, setting every key in :attr:`POLICY_KEYS` explicitly.

        Some ALMA checkpoints inherit sampling from Llama 2's generation config,
        and Qwen2.5 ships a repetition penalty.
        """
        kwargs: dict[str, Any] = {
            "do_sample": decode.do_sample,
            "num_beams": decode.num_beams,
            "max_new_tokens": decode.max_new_tokens,
            "num_return_sequences": 1,
            "length_penalty": decode.length_penalty,
            "early_stopping": decode.early_stopping,
            "repetition_penalty": decode.repetition_penalty,
            "no_repeat_ngram_size": decode.no_repeat_ngram_size,
            "num_beam_groups": 1,
            "diversity_penalty": 0.0,
            "use_cache": True,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if decode.do_sample:
            for label in ("temperature", "top_p", "top_k"):
                value = getattr(decode, label)
                if value is not None:
                    kwargs[label] = value
        else:
            # Explicit None so checkpoint sampling defaults are not used.
            kwargs.update({"temperature": None, "top_p": None, "top_k": None})
        kwargs.update(self.generate_kwargs(direction))
        return kwargs

    def terminator_ids(self) -> frozenset[int]:
        """Every token id that ends generation: the tokenizer's EOS plus the generation config's.

        E.g. Qwen2.5 has ``eos_token_id = 151645`` in the tokenizer but
        ``[151645, 151643]`` in its generation config.
        """
        if self._terminators is not None:
            return self._terminators

        candidates: list[Any] = [getattr(self.tokenizer, "eos_token_id", None)]
        generation_config = getattr(self.model, "generation_config", None)
        candidates.append(getattr(generation_config, "eos_token_id", None))

        ids: set[int] = set()
        for candidate in candidates:
            if candidate is None:
                continue
            values = candidate if isinstance(candidate, list | tuple | set) else [candidate]
            for value in values:
                try:
                    ids.add(int(value))
                except (TypeError, ValueError):  # pragma: no cover - defensive
                    continue
        self._terminators = frozenset(ids)
        return self._terminators

    def _trim_continuation(self, continuation: Any) -> tuple[Any, int, bool]:
        """Cut a continuation at its first terminator.

        Returns the kept tokens, the number of content tokens (excluding the
        terminator) and whether a terminator was found.
        """
        terminators = self.terminator_ids()
        if not terminators:
            return continuation, len(continuation), True

        import torch

        mask = torch.zeros_like(continuation, dtype=torch.bool)
        for token_id in terminators:
            mask |= continuation == token_id
        matches = mask.nonzero()
        if matches.numel() == 0:
            return continuation, len(continuation), False
        first = int(matches.reshape(-1)[0].item())
        return continuation[: first + 1], first, True

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
