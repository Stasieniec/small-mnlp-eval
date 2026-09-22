"""Test doubles.

Kept in their own module rather than in conftest so that helper functions in a
test file can import them directly.
"""

from __future__ import annotations

from typing import Any

from mnlp_eval.config import ModelSpec
from mnlp_eval.languages import Direction
from mnlp_eval.models.base import SegmentOutput, Translator


class FakeTokenizer:
    """The smallest object satisfying what the translator touches."""

    def __init__(self) -> None:
        self.pad_token_id: int | None = 0
        self.eos_token_id: int | None = 1
        self.eos_token = "</s>"
        self.pad_token = "<pad>"
        self.padding_side = "right"

    def __call__(self, text: str, **_: Any) -> dict[str, list[int]]:
        tokens = text.split()
        return {"input_ids": list(range(len(tokens))), "attention_mask": [1] * len(tokens)}


class StubTranslator(Translator):
    """A translator that echoes canned output, bypassing the batching loop.

    Lets the generate, score and report stages be exercised end to end with no
    model.
    """

    kind = "stub"

    def __init__(
        self,
        spec: ModelSpec,
        prompt: Any,
        outputs: list[str] | None = None,
        *,
        template: str = "translation of {source}",
    ) -> None:
        super().__init__(spec=spec, prompt=prompt, model=object(), tokenizer=FakeTokenizer())
        self._outputs = outputs
        self._template = template
        self.calls: list[tuple[str, int, int]] = []

    def build_input(self, direction: Direction, source: str) -> str:
        return self.prompt.render(direction, source)

    def parameter_counts(self) -> tuple[int, int]:
        return 1_000_000, 900_000

    def translate(self, direction: Direction, sources: Any, decode: Any) -> list[SegmentOutput]:
        sources = list(sources)
        self.calls.append((str(direction), len(sources), decode.batch_size))
        if self._outputs is not None:
            texts = [self._outputs[index % len(self._outputs)] for index in range(len(sources))]
        else:
            texts = [self._template.format(source=source) for source in sources]
        return [
            SegmentOutput(
                raw_text=text,
                n_source_tokens=len(source.split()),
                n_generated_tokens=len(text.split()),
                hit_token_budget=False,
                source_truncated=False,
            )
            for source, text in zip(sources, texts, strict=True)
        ]


def tiny_llama(**overrides: Any) -> Any:
    """A two-layer multi-head Llama, the architecture family ALMA-7B is.

    Tiny and randomly initialised, so the pruning tests need no GPU, network or
    weights. Seeded, since criterion tests compare two runs of one model.
    """
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    settings: dict[str, Any] = {
        "vocab_size": 64,
        "hidden_size": 32,
        "intermediate_size": 48,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "max_position_embeddings": 64,
        "tie_word_embeddings": False,
    }
    settings.update(overrides)
    torch.manual_seed(0)
    model = LlamaForCausalLM(LlamaConfig(**settings))
    model.eval()
    return model


def tiny_qwen2(**overrides: Any) -> Any:
    """A two-layer grouped-query Qwen2 with tied embeddings, as Qwen2.5-0.5B is.

    Grouped-query and tied embeddings are the two cases that behave differently
    from ALMA under pruning.
    """
    import torch
    from transformers import Qwen2Config, Qwen2ForCausalLM

    settings: dict[str, Any] = {
        "vocab_size": 64,
        "hidden_size": 32,
        "intermediate_size": 48,
        "num_hidden_layers": 2,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "max_position_embeddings": 64,
        "tie_word_embeddings": True,
    }
    settings.update(overrides)
    torch.manual_seed(0)
    model = Qwen2ForCausalLM(Qwen2Config(**settings))
    model.eval()
    return model


def token_batches(count: int = 3, rows: int = 2, tokens: int = 6, pad: int = 0) -> list[Any]:
    """Calibration-shaped batches of random token ids with an attention mask.

    ``pad`` masks that many leading positions, as the translators pad, so a
    test can tell whether padded positions were excluded.
    """
    import torch

    torch.manual_seed(1)
    batches = []
    for _ in range(count):
        ids = torch.randint(0, 64, (rows, tokens))
        attention = torch.ones_like(ids)
        if pad:
            attention[:, :pad] = 0
        batches.append({"input_ids": ids, "attention_mask": attention})
    return batches
