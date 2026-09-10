"""Structural accounting for pruned models.

Structured pruning is the whole experiment, so the module that says what was
removed needs to be right on the cases that matter: uneven per-layer budgets,
whole layers dropped, masks applied but not compacted, and quantized weights
where a zero byte is not a pruned weight.

Tensors are fakes: describe_structure never imports torch, which also keeps
it usable in the metric environments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from mnlp_eval.bench.structure import describe_structure


@dataclass
class FakeDtype:
    is_floating_point: bool


FLOAT = FakeDtype(True)
INT8 = FakeDtype(False)


class FakeTensor:
    def __init__(self, shape: tuple[int, ...], *, zeros: int = 0, dtype: FakeDtype = FLOAT) -> None:
        self.shape = shape
        self.dtype = dtype
        self._zeros = zeros

    def numel(self) -> int:
        total = 1
        for dim in self.shape:
            total *= dim
        return total

    def count_nonzero(self) -> int:
        return self.numel() - self._zeros


class FakeConfig:
    def __init__(self, **fields: Any) -> None:
        for key, value in fields.items():
            setattr(self, key, value)


class FakeModel:
    """A Llama-shaped parameter listing, which is what ALMA is."""

    def __init__(self, layers: list[tuple[int, int]], **overrides: Any) -> None:
        self.config = FakeConfig(
            model_type="llama",
            hidden_size=64,
            num_hidden_layers=len(layers),
            num_attention_heads=8,
            intermediate_size=128,
            vocab_size=100,
            **overrides,
        )
        self._parameters: list[tuple[str, FakeTensor]] = [
            ("model.embed_tokens.weight", FakeTensor((100, 64))),
        ]
        for index, (attention_width, ffn_width) in enumerate(layers):
            prefix = f"model.layers.{index}"
            self._parameters.extend(
                [
                    (f"{prefix}.self_attn.q_proj.weight", FakeTensor((attention_width, 64))),
                    (f"{prefix}.self_attn.o_proj.weight", FakeTensor((64, attention_width))),
                    (f"{prefix}.mlp.up_proj.weight", FakeTensor((ffn_width, 64))),
                    (f"{prefix}.mlp.down_proj.weight", FakeTensor((64, ffn_width))),
                ]
            )

    def named_parameters(self) -> list[tuple[str, FakeTensor]]:
        return self._parameters


def test_dense_model_is_uniform_with_the_expected_head_count() -> None:
    structure = describe_structure(FakeModel([(64, 128)] * 4))
    stack = structure["stacks"]["model.layers"]

    assert structure["head_dim"] == 8
    assert stack["n_layers"] == 4
    assert stack["uniform"] is True
    assert [layer["attention_heads"] for layer in stack["layers"]] == [8, 8, 8, 8]
    assert stack["ffn_intermediate"] == {"min": 128, "max": 128, "mean": 128.0, "total": 512}


def test_uneven_pruning_is_reported_as_non_uniform() -> None:
    # The case a single sparsity number hides: the same parameter count spent
    # very differently across depth.
    structure = describe_structure(FakeModel([(64, 128), (64, 128), (32, 64), (16, 32)]))
    stack = structure["stacks"]["model.layers"]

    assert stack["uniform"] is False
    assert stack["ffn_intermediate"] == {"min": 32, "max": 128, "mean": 88.0, "total": 352}
    assert [layer["attention_heads"] for layer in stack["layers"]] == [8, 8, 4, 2]


def test_dropped_layers_shrink_the_stack_not_the_widths() -> None:
    structure = describe_structure(FakeModel([(64, 128)] * 2))
    stack = structure["stacks"]["model.layers"]

    assert stack["n_layers"] == 2
    assert stack["uniform"] is True


def test_a_mask_that_was_never_compacted_shows_up_as_zero_weights() -> None:
    """The failure this column exists for.

    A pruning run that zeroed its removed channels but never rebuilt the
    tensors reports the full parameter count, the full checkpoint size and the
    full latency. Only the zero fraction reveals that none of the claimed
    saving is real.
    """
    model = FakeModel([(64, 128)] * 2)
    for name, tensor in model.named_parameters():
        if name.endswith("down_proj.weight"):
            tensor._zeros = tensor.numel() // 2

    structure = describe_structure(model)

    assert structure["zero_parameters"] == 2 * (64 * 128 // 2)
    # Reported to six decimals, which is the resolution the JSON artifact keeps.
    assert structure["zero_fraction"] == pytest.approx(
        8192 / structure["total_parameters"], abs=1e-6
    )
    # Half of one of the layer's four weight matrices, and down_proj is a third
    # of the layer's parameters: 4096 zeros out of 24576.
    layer = structure["stacks"]["model.layers"]["layers"][0]
    assert layer["zero_parameters"] == 4096
    assert layer["zero_fraction"] == pytest.approx(4096 / 24576, abs=1e-6)


def test_integer_weights_are_excluded_from_the_zero_count() -> None:
    """A zero byte in a 4-bit checkpoint is a quantization level, not a pruned weight."""
    model = FakeModel([(64, 128)])
    model._parameters.append(("model.quantized.weight", FakeTensor((64, 64), zeros=64, dtype=INT8)))

    structure = describe_structure(model)

    assert structure["unmeasured_parameters"] == 4096
    assert structure["zero_parameters"] == 0
    assert "zero_note" in structure
    # The fraction divides by what was actually measured, not by everything.
    assert structure["zero_fraction"] == 0.0


def test_zero_counting_can_be_switched_off() -> None:
    structure = describe_structure(FakeModel([(64, 128)]), count_zeros=False)

    assert structure["zero_fraction"] is None
    assert structure["zero_parameters"] is None


def test_encoder_and_decoder_stacks_stay_separate() -> None:
    """Pooling them would report a 6-layer model as having 12 identical layers."""

    class Seq2Seq:
        config = FakeConfig(model_type="marian", hidden_size=64, num_attention_heads=8)

        def named_parameters(self) -> list[tuple[str, FakeTensor]]:
            parameters = []
            for stack, width in (("encoder", 128), ("decoder", 64)):
                for index in range(3):
                    prefix = f"model.{stack}.layers.{index}"
                    parameters.extend(
                        [
                            (f"{prefix}.self_attn.out_proj.weight", FakeTensor((64, 64))),
                            (f"{prefix}.fc2.weight", FakeTensor((64, width))),
                        ]
                    )
            return parameters

    structure = describe_structure(Seq2Seq())

    assert set(structure["stacks"]) == {"model.encoder.layers", "model.decoder.layers"}
    assert structure["stacks"]["model.encoder.layers"]["ffn_intermediate"]["min"] == 128
    assert structure["stacks"]["model.decoder.layers"]["ffn_intermediate"]["min"] == 64


def test_unrecognised_parameter_names_still_report_counts() -> None:
    class Opaque:
        def named_parameters(self) -> list[tuple[str, FakeTensor]]:
            return [("something.odd", FakeTensor((8, 8)))]

    structure = describe_structure(Opaque())

    assert structure["stacks"] == {}
    assert structure["total_parameters"] == 64
    assert "structure_note" in structure


def test_a_model_without_parameters_is_rejected_clearly() -> None:
    with pytest.raises(TypeError, match="named_parameters"):
        describe_structure(object())
