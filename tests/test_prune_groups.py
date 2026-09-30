"""Prunable unit groups.

Every later stage indexes through this module, so an off-by-one here becomes a
quality number nobody can explain. The cases that matter are grouped-query
attention, where an uneven selection misroutes rather than fails, and the
boundary between a selection and a removed layer.

The modules are faked: this is index arithmetic and must stay testable without
torch, which the first CI job does not install.
"""

from __future__ import annotations

from typing import Any

import pytest

from mnlp_eval.prune import PruneError
from mnlp_eval.prune.groups import LayerGroups, decoder_layers, describe_layers


class FakeLinear:
    """The three attributes ``groups`` reads off a projection."""

    def __init__(self, in_features: int, out_features: int, *, bias: bool = False) -> None:
        self.in_features = in_features
        self.out_features = out_features
        self.bias = [0.0] * out_features if bias else None


def build_model(
    *,
    model_type: str = "llama",
    hidden: int = 256,
    num_heads: int = 8,
    num_kv_heads: int | None = None,
    head_dim: int = 32,
    intermediate: int = 512,
    num_layers: int = 2,
    attention_bias: bool = False,
) -> Any:
    """A duck-typed stand-in for a decoder, shaped like Llama or Qwen2."""
    kv_heads = num_heads if num_kv_heads is None else num_kv_heads

    def layer() -> Any:
        attention = type(
            "FakeAttention",
            (),
            {
                "head_dim": head_dim,
                "q_proj": FakeLinear(hidden, num_heads * head_dim, bias=attention_bias),
                "k_proj": FakeLinear(hidden, kv_heads * head_dim, bias=attention_bias),
                "v_proj": FakeLinear(hidden, kv_heads * head_dim, bias=attention_bias),
                "o_proj": FakeLinear(num_heads * head_dim, hidden),
            },
        )()
        mlp = type(
            "FakeMLP",
            (),
            {
                "gate_proj": FakeLinear(hidden, intermediate),
                "up_proj": FakeLinear(hidden, intermediate),
                "down_proj": FakeLinear(intermediate, hidden),
            },
        )()
        return type("FakeLayer", (), {"self_attn": attention, "mlp": mlp})()

    inner = type("FakeInner", (), {"layers": [layer() for _ in range(num_layers)]})()
    config = type("FakeConfig", (), {"model_type": model_type})()
    return type("FakeModel", (), {"config": config, "model": inner})()


def test_multi_head_layers_are_measured_from_the_projections() -> None:
    layers = describe_layers(build_model(num_heads=8, head_dim=32, intermediate=512))

    assert len(layers) == 2
    assert layers[0] == LayerGroups(
        index=0, num_heads=8, num_kv_heads=8, head_dim=32, intermediate=512
    )
    assert layers[0].kv_group_size == 1
    assert not layers[0].is_grouped_query


def test_grouped_query_layers_report_their_group_size_and_biases() -> None:
    # Qwen2.5-0.5B's shape, with the q, k and v biases Qwen2 hardcodes.
    layers = describe_layers(
        build_model(
            model_type="qwen2",
            hidden=896,
            num_heads=14,
            num_kv_heads=2,
            head_dim=64,
            intermediate=4864,
            attention_bias=True,
        )
    )

    assert layers[0].kv_group_size == 7
    assert layers[0].is_grouped_query
    assert layers[0].attention_bias
    assert layers[0].head_groups() == (tuple(range(7)), tuple(range(7, 14)))


def test_an_unsupported_architecture_is_refused_rather_than_guessed_at() -> None:
    with pytest.raises(PruneError, match="not one of"):
        decoder_layers(build_model(model_type="mistral"))


def test_a_disagreement_between_coupled_projections_is_refused() -> None:
    model = build_model()
    model.model.layers[1].mlp.down_proj = FakeLinear(999, 256)

    with pytest.raises(PruneError, match="layer 1: down_proj"):
        describe_layers(model)


class TestHeadSelection:
    mha = LayerGroups(index=0, num_heads=8, num_kv_heads=8, head_dim=4, intermediate=16)
    gqa = LayerGroups(index=3, num_heads=14, num_kv_heads=2, head_dim=4, intermediate=16)

    def test_multi_head_attention_takes_any_subset(self) -> None:
        assert self.mha.validate_heads([5, 0, 2]) == (0, 2, 5)

    def test_a_key_value_head_follows_its_query_head_under_multi_head_attention(self) -> None:
        assert self.mha.kept_kv_heads([0, 2, 5]) == (0, 2, 5)
        assert self.mha.num_key_value_groups_after([0, 2, 5]) == 1

    def test_grouped_query_attention_takes_an_even_selection(self) -> None:
        kept = [0, 1, 2, 7, 8, 9]

        assert self.gqa.validate_heads(kept) == (0, 1, 2, 7, 8, 9)
        # Every group keeps something, so k_proj and v_proj stay at full width.
        assert self.gqa.kept_kv_heads(kept) == (0, 1)
        assert self.gqa.num_key_value_groups_after(kept) == 3

    def test_grouped_query_attention_refuses_an_uneven_selection(self) -> None:
        # Four from the first group, two from the second. repeat_kv would
        # misroute the survivors rather than raise, so catch it here.
        with pytest.raises(PruneError, match="same number of query heads"):
            self.gqa.validate_heads([0, 1, 2, 3, 7, 8])

    def test_rows_follow_the_heads_in_order(self) -> None:
        assert self.mha.head_rows([1, 3]) == (4, 5, 6, 7, 12, 13, 14, 15)

    def test_key_value_rows_stay_full_width_under_grouped_query_attention(self) -> None:
        assert self.gqa.kv_rows([0, 7]) == tuple(range(8))

    @pytest.mark.parametrize(
        ("kept", "expected"),
        [
            ([], "removed layer"),
            ([0, 0, 1], "repeated"),
            ([0, 99], "outside"),
            ([True, 2], "must be integers"),
        ],
    )
    def test_a_malformed_selection_is_refused(self, kept: list[int], expected: str) -> None:
        with pytest.raises(PruneError, match=expected):
            self.mha.validate_heads(kept)

    def test_channels_are_checked_against_the_intermediate_width(self) -> None:
        assert self.mha.validate_channels([15, 0]) == (0, 15)
        with pytest.raises(PruneError, match="FFN channel indices"):
            self.mha.validate_channels([16])
