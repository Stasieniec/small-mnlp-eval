"""The shared generation loop in ``models.base``.

This is the only place hypotheses are reordered, and until these tests existed
nothing exercised it: ``tests/stubs.StubTranslator`` overrides ``translate``
wholesale. A mutation swapping ``results[index]`` for ``results[position]``
pairs every hypothesis with the wrong source and reference, collapsing BLEU,
and the rest of the suite passed with that bug in place.

The fake model and tokenizer here encode each segment's own index into its
token ids, so a misalignment is detected directly rather than inferred from a
score. Requires torch, but not a GPU and not a model download.
"""

from __future__ import annotations

from typing import Any

import pytest

from mnlp_eval.config import DecodeSpec, ModelSpec
from mnlp_eval.languages import parse_direction
from mnlp_eval.models.hf_causal import CausalTranslator
from mnlp_eval.models.hf_seq2seq import Seq2SeqTranslator
from mnlp_eval.prompts import get_prompt

torch = pytest.importorskip("torch", reason="the generation loop is written against torch")

pytestmark = pytest.mark.torch

DE_EN = parse_direction("de-en")

PAD = 0
EOS = 1
ALT_EOS = 2
DECODER_START = 3
PROMPT_BASE = 100
CONTENT_BASE = 1000


class Batch(dict):
    """Stands in for a transformers BatchEncoding."""

    def to(self, device: Any) -> Batch:
        del device
        return self


class FakeTokenizer:
    """Encodes each segment's index into its ids so misalignment is visible."""

    def __init__(self, eos_token_id: Any = EOS) -> None:
        self.pad_token_id = PAD
        self.eos_token_id = eos_token_id
        self.eos_token = "</s>"
        self.pad_token = "<pad>"
        self.padding_side = "right"

    @staticmethod
    def _index_of(text: str) -> int:
        digits = "".join(character for character in text if character.isdigit())
        return int(digits) if digits else 0

    def __call__(self, text: str, **kwargs: Any) -> dict[str, list[int]]:
        index = self._index_of(text)
        # Lengths vary between 1 and 5 tokens, so length bucketing actually
        # reorders the batch and padding differs between neighbours.
        length = index % 5 + 1
        ids = [PROMPT_BASE + index] * length
        limit = kwargs.get("max_length")
        if kwargs.get("truncation") and limit is not None:
            ids = ids[:limit]
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}

    def pad(self, encodings: list[dict[str, list[int]]], **_: Any) -> Batch:
        width = max(len(item["input_ids"]) for item in encodings)
        input_ids, attention = [], []
        for item in encodings:
            padding = width - len(item["input_ids"])
            if self.padding_side == "left":
                input_ids.append([PAD] * padding + item["input_ids"])
                attention.append([0] * padding + item["attention_mask"])
            else:
                input_ids.append(item["input_ids"] + [PAD] * padding)
                attention.append(item["attention_mask"] + [0] * padding)
        return Batch(
            input_ids=torch.tensor(input_ids),
            attention_mask=torch.tensor(attention),
        )

    def decode(self, ids: Any, skip_special_tokens: bool = False) -> str:
        del skip_special_tokens
        words = []
        for value in ids.tolist():
            if value >= CONTENT_BASE:
                words.append(f"hyp{value - CONTENT_BASE}")
            elif value >= PROMPT_BASE:
                words.append(f"prompt{value - PROMPT_BASE}")
        return " ".join(words)


class FakeGenerationConfig:
    def __init__(self, eos_token_id: Any = None) -> None:
        self.eos_token_id = eos_token_id

    def to_diff_dict(self) -> dict[str, Any]:
        return {"eos_token_id": self.eos_token_id, "repetition_penalty": 1.1}


class FakeModel:
    """Emits the content token matching each row's prompt index."""

    device = "cpu"

    def __init__(
        self,
        *,
        stop_token: int = EOS,
        emit_terminator: bool = True,
        extra_tokens: int = 0,
        generation_eos: Any = None,
        is_encoder_decoder: bool = False,
    ) -> None:
        self.generation_config = FakeGenerationConfig(generation_eos)
        self.stop_token = stop_token
        self.emit_terminator = emit_terminator
        self.extra_tokens = extra_tokens
        self.calls: list[dict[str, Any]] = []
        self.is_encoder_decoder = is_encoder_decoder

    def generate(self, **kwargs: Any) -> Any:
        input_ids = kwargs["input_ids"]
        self.calls.append({key: value for key, value in kwargs.items() if key != "input_ids"})
        budget = kwargs.get("max_new_tokens", 8)
        rows = []
        for row in input_ids.tolist():
            index = max(value - PROMPT_BASE for value in row if value >= PROMPT_BASE)
            prefix = [DECODER_START] if self.is_encoder_decoder else list(row)
            tail = [CONTENT_BASE + index, *([CONTENT_BASE + index] * self.extra_tokens)]
            if self.emit_terminator:
                tail.append(self.stop_token)
            tail = tail[:budget]
            tail += [PAD] * (budget - len(tail))
            rows.append(prefix + tail)
        return torch.tensor(rows)

    def modules(self) -> list[Any]:
        return []

    def parameters(self) -> list[Any]:
        return []


def _causal(model: FakeModel, tokenizer: FakeTokenizer | None = None) -> CausalTranslator:
    spec = ModelSpec.from_dict(
        {"name": "fake", "loader": "hf_causal", "prompt": "alma", "model_name_or_path": "fake"}
    )
    return CausalTranslator(
        spec=spec,
        prompt=get_prompt("alma"),
        model=model,
        tokenizer=tokenizer or FakeTokenizer(),
    )


def _sources(count: int) -> list[str]:
    return [f"Segment {index} source text" for index in range(count)]


# --------------------------------------------------------------------------
# Ordering: the bug this file exists for
# --------------------------------------------------------------------------


@pytest.mark.parametrize("batch_size", [1, 3, 4, 16])
@pytest.mark.parametrize("sort_by_length", [True, False])
def test_output_stays_aligned_with_its_source(batch_size: int, sort_by_length: bool) -> None:
    sources = _sources(13)
    translator = _causal(FakeModel())
    outputs = translator.translate(
        DE_EN,
        sources,
        DecodeSpec(num_beams=1, batch_size=batch_size, sort_by_length=sort_by_length),
    )
    assert len(outputs) == len(sources)
    # hypN must come back at position N regardless of how the batch was formed.
    assert [output.raw_text for output in outputs] == [
        f"hyp{index}" for index in range(len(sources))
    ]


def test_length_bucketing_actually_reorders_the_work() -> None:
    # If it did not, the alignment test above would prove nothing.
    sources = _sources(10)
    translator = _causal(FakeModel())
    sorted_batches = FakeModel()
    translator.model = sorted_batches
    translator.translate(DE_EN, sources, DecodeSpec(num_beams=1, batch_size=3))
    bucketed_widths = [call["attention_mask"].shape[1] for call in sorted_batches.calls]

    unsorted_model = FakeModel()
    translator.model = unsorted_model
    translator.translate(
        DE_EN, sources, DecodeSpec(num_beams=1, batch_size=3, sort_by_length=False)
    )
    plain_widths = [call["attention_mask"].shape[1] for call in unsorted_model.calls]
    assert bucketed_widths != plain_widths


def test_empty_input_never_touches_the_model() -> None:
    model = FakeModel()
    assert _causal(model).translate(DE_EN, [], DecodeSpec()) == []
    assert model.calls == []


# --------------------------------------------------------------------------
# Terminators and token accounting
# --------------------------------------------------------------------------


def test_secondary_terminator_from_the_generation_config_is_honoured() -> None:
    # Qwen2.5 has tokenizer.eos_token_id 151645 and generation_config
    # [151645, 151643]. Reading only the tokenizer made a model that stopped on
    # the second one look like it never stopped, so its padding was counted as
    # generated text and clean output was flagged as truncated.
    model = FakeModel(stop_token=ALT_EOS, generation_eos=[EOS, ALT_EOS])
    translator = _causal(model)
    assert translator.terminator_ids() == frozenset({EOS, ALT_EOS})
    outputs = translator.translate(DE_EN, _sources(4), DecodeSpec(num_beams=1, max_new_tokens=32))
    for output in outputs:
        assert not output.hit_token_budget
        assert output.n_generated_tokens == 1


def test_unknown_terminator_is_reported_as_budget_exhaustion() -> None:
    translator = _causal(FakeModel(emit_terminator=False))
    outputs = translator.translate(DE_EN, _sources(3), DecodeSpec(num_beams=1, max_new_tokens=6))
    assert all(output.hit_token_budget for output in outputs)


def test_a_list_valued_tokenizer_eos_does_not_crash() -> None:
    # The idiom tokenizer.eos_token_id = model.generation_config.eos_token_id
    # makes this a list, which a scalar comparison could not handle.
    translator = _causal(FakeModel(stop_token=ALT_EOS), FakeTokenizer(eos_token_id=[EOS, ALT_EOS]))
    assert translator.terminator_ids() == frozenset({EOS, ALT_EOS})
    outputs = translator.translate(DE_EN, _sources(2), DecodeSpec(num_beams=1))
    assert all(not output.hit_token_budget for output in outputs)


def test_the_terminator_is_not_counted_as_content() -> None:
    # Counting it reported n_wasted_tokens of 1 for output as clean as "Hello."
    translator = _causal(FakeModel(extra_tokens=2))
    output = translator.translate(DE_EN, _sources(1), DecodeSpec(num_beams=1))[0]
    assert output.n_generated_tokens == 3


def test_decoder_start_token_is_not_counted_for_encoder_decoder() -> None:
    # Counting it made the fixed token budget differ by one per segment between
    # encoder-decoder and decoder-only systems.
    spec = ModelSpec.from_dict(
        {
            "name": "fake",
            "loader": "hf_seq2seq",
            "prompt": "passthrough",
            "model_name_or_path": "fake",
        }
    )
    translator = Seq2SeqTranslator(
        spec=spec,
        prompt=get_prompt("passthrough"),
        model=FakeModel(is_encoder_decoder=True),
        tokenizer=FakeTokenizer(),
    )
    output = translator.translate(DE_EN, _sources(1), DecodeSpec(num_beams=1))[0]
    assert output.n_generated_tokens == 1
    assert output.raw_text == "hyp0"


def test_padded_width_is_recorded_for_the_flops_estimate() -> None:
    outputs = _causal(FakeModel()).translate(
        DE_EN, _sources(6), DecodeSpec(num_beams=1, batch_size=6, sort_by_length=False)
    )
    widths = {output.n_padded_source_tokens for output in outputs}
    assert widths == {5}
    assert any(output.n_source_tokens < 5 for output in outputs)


def test_source_truncation_is_flagged_at_the_boundary() -> None:
    sources = _sources(5)
    outputs = _causal(FakeModel()).translate(
        DE_EN, sources, DecodeSpec(num_beams=1, max_source_length=3)
    )
    # Segment N encodes to N % 5 + 1 tokens, so segments 3 and 4 (four and
    # five tokens) exceed a cap of three and the rest do not.
    assert [output.source_truncated for output in outputs] == [
        False,
        False,
        False,
        True,
        True,
    ]


# --------------------------------------------------------------------------
# Generation policy
# --------------------------------------------------------------------------


def test_checkpoint_sampling_and_repetition_settings_are_overridden() -> None:
    model = FakeModel()
    _causal(model).translate(DE_EN, _sources(2), DecodeSpec(num_beams=5))
    call = model.calls[0]
    assert call["do_sample"] is False
    assert call["num_beams"] == 5
    assert call["temperature"] is None
    assert call["top_p"] is None
    # The checkpoint's generation config asks for 1.1; evaluation policy wins,
    # because a repetition penalty suppresses the collapse mode the
    # behavioural metrics measure.
    assert call["repetition_penalty"] == 1.0
    assert call["no_repeat_ngram_size"] == 0
    assert call["num_beam_groups"] == 1


def test_sampling_parameters_are_forwarded_when_sampling_is_requested() -> None:
    model = FakeModel()
    decode = DecodeSpec(do_sample=True, num_beams=1, temperature=0.6, top_p=0.9)
    _causal(model).translate(DE_EN, _sources(2), decode)
    call = model.calls[0]
    assert call["do_sample"] is True
    assert call["temperature"] == 0.6
    assert call["top_p"] == 0.9


def test_unpinned_generation_settings_are_disclosed() -> None:
    translator = _causal(FakeModel(generation_eos=[EOS, ALT_EOS]))
    disclosed = translator.unpinned_generation_settings()
    # Pinned by policy, so it must not appear as an unpinned surprise.
    assert "repetition_penalty" not in disclosed
    # Not pinned, so it must be recorded for the reader.
    assert disclosed["eos_token_id"] == [EOS, ALT_EOS]
