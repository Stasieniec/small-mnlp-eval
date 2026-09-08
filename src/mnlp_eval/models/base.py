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
    #: Padded width of the batch this segment was in. Prefill computes over the
    #: padded width, not the unpadded token count, so an analytic FLOPs
    #: estimate that uses the latter undercounts badly at batch sizes above one.
    n_padded_source_tokens: int = 0
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
            unpinned_generation_settings=self.unpinned_generation_settings() or None,
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

        # A 4-bit bitsandbytes checkpoint stores two weights per uint8 element,
        # so summing numel() reports half the parameters and would credit
        # weight-only quantization with a fake parameter-count compression
        # ratio. transformers' own num_parameters() corrects for this, so use
        # it when the model provides it.
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
            # Encoder-decoder output opens with decoder_start_token_id, which
            # the model was given rather than generated. Counting it made the
            # fixed token budget differ by one per segment between
            # encoder-decoder and decoder-only systems.
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
                )

        missing = [index for index, value in enumerate(results) if value is None]
        if missing:  # pragma: no cover
            msg = f"generation produced no output for {len(missing)} segment(s)"
            raise RuntimeError(msg)
        return [value for value in results if value is not None]

    #: Decoding knobs this framework treats as evaluation policy and always
    #: sets itself, so a checkpoint's generation_config.json cannot change how
    #: it is measured. Anything not listed here is left to the checkpoint,
    #: because it is a property of the architecture rather than of the
    #: evaluation: forced_bos_token_id (NLLB needs it to select a target
    #: language), forced_eos_token_id and bad_words_ids (Marian ships both),
    #: renormalize_logits, suppress_tokens, decoder_start_token_id. Whatever a
    #: checkpoint sets outside this list is recorded by
    #: :meth:`unpinned_generation_settings` so it is at least visible.
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
        """Non-default generation settings this framework does not override.

        Recorded in the run manifest. A reader can then see that, say, a Marian
        checkpoint forces an EOS token, without the framework having to guess
        whether overriding that would break the model.
        """
        generation_config = getattr(self.model, "generation_config", None)
        differ = getattr(generation_config, "to_diff_dict", None)
        if not callable(differ):
            return {}
        try:
            diff = differ()
        except Exception:  # recording provenance must never fail a run
            return {}
        return {
            key: value
            for key, value in sorted(diff.items())
            if key not in self.POLICY_KEYS and key != "transformers_version"
        }

    def _resolve_generate_kwargs(self, direction: Direction, decode: DecodeSpec) -> dict[str, Any]:
        """Build ``generate`` arguments, pinning every policy knob explicitly.

        Several ALMA-family checkpoints inherit a ``generation_config.json``
        from Llama 2 that enables sampling, and Qwen2.5 ships a repetition
        penalty. Passing these values explicitly stops the checkpoint from
        deciding how it is evaluated. See :attr:`POLICY_KEYS` for the boundary
        between what is pinned and what is deliberately left alone.
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
            # Explicit nulls keep transformers from reinstating checkpoint
            # defaults and warning about unused sampling parameters.
            kwargs.update({"temperature": None, "top_p": None, "top_k": None})
        kwargs.update(self.generate_kwargs(direction))
        return kwargs

    def terminator_ids(self) -> frozenset[int]:
        """Every token id that ends generation for this model.

        Reading only ``tokenizer.eos_token_id`` is wrong for a large family of
        models. Qwen2.5, for instance, has ``tokenizer.eos_token_id = 151645``
        while its generation config lists ``[151645, 151643]``, and 151643 is
        also its pad token. A model that stopped on the second terminator would
        otherwise look like it never stopped, so its padding would be counted
        as generated text: the token count, the throughput, the budget-hit rate
        and the truncation rate would all be wrong at once, and a clean
        translation would be flagged as truncated.
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

        Returns the kept tokens, the count of tokens the model actually
        produced as content, and whether a terminator was found. The
        terminator itself is excluded from the content count: it is a control
        token, and counting it made ``n_wasted_tokens`` report 1 for output as
        clean as ``"Hello."``.

        A continuation with no terminator ran out of token budget. Whether that
        truncated the translation is decided downstream, once the hypothesis is
        known.
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
