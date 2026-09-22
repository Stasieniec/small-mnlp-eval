"""Compaction, and the round trip through a saved checkpoint.

Pruning nothing must change nothing. An off-by-one does not crash, it yields a
model that translates slightly worse, which is indistinguishable from a
criterion that did not work.

The second case is the save and load round trip on a tied-embedding model with
layers of differing widths, which is what a global budget produces and what
the config format cannot express.

Needs torch and transformers, but no GPU, network or weights.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from mnlp_eval.prune import PruneError
from mnlp_eval.prune.compact import (
    LayerPlan,
    compact_model,
    plan_from_subnetwork,
    write_descriptor,
)
from mnlp_eval.prune.groups import describe_layers
from stubs import tiny_llama as build_llama
from stubs import tiny_qwen2 as build_qwen2

pytestmark = pytest.mark.torch


def logits(model: Any) -> Any:
    import torch

    with torch.inference_mode():
        return model(torch.arange(8).unsqueeze(0)).logits.clone()


def full_plan(model: Any) -> tuple[LayerPlan, ...]:
    return tuple(
        LayerPlan(heads=tuple(range(group.num_heads)), channels=tuple(range(group.intermediate)))
        for group in describe_layers(model)
    )


@pytest.mark.parametrize("build", [build_llama, build_qwen2], ids=["llama", "qwen2"])
def test_pruning_nothing_leaves_the_logits_bit_identical(build: Any) -> None:
    import torch

    model = build()
    before = logits(model)

    compact_model(model, full_plan(model))

    assert torch.equal(logits(model), before)


def test_compaction_shrinks_every_coupled_projection() -> None:
    model = build_llama()
    plan = (
        LayerPlan(heads=(0, 2), channels=tuple(range(12))),
        LayerPlan(heads=(1,), channels=tuple(range(24))),
    )

    compact_model(model, plan)

    first, second = model.model.layers
    head_dim = first.self_attn.head_dim
    assert first.self_attn.q_proj.out_features == 2 * head_dim
    assert first.self_attn.o_proj.in_features == 2 * head_dim
    assert first.mlp.gate_proj.out_features == 12
    assert first.mlp.down_proj.in_features == 12
    # Layers keep independent widths, which is what a global budget produces
    # and what the config format cannot express.
    assert second.self_attn.q_proj.out_features == head_dim
    assert second.mlp.up_proj.out_features == 24
    # Multi-head attention: a key/value head follows its query head.
    assert second.self_attn.k_proj.out_features == head_dim
    assert second.self_attn.num_key_value_groups == 1
    assert logits(model).shape == (1, 8, 64)


def test_compaction_keeps_grouped_query_attention_consistent() -> None:
    model = build_qwen2()
    # Two of four query heads from each of the two key/value groups.
    plan = (
        LayerPlan(heads=(0, 1, 4, 5), channels=tuple(range(24))),
        LayerPlan(heads=(2, 3, 6, 7), channels=tuple(range(24))),
    )

    compact_model(model, plan)

    attention = model.model.layers[0].self_attn
    head_dim = attention.head_dim
    assert attention.q_proj.out_features == 4 * head_dim
    # Every group kept something, so the key and value projections stay whole.
    assert attention.k_proj.out_features == 2 * head_dim
    assert attention.num_key_value_groups == 2
    # Qwen2 hardcodes q, k and v biases, and they index by output row.
    assert attention.q_proj.bias.shape == (4 * head_dim,)
    assert attention.k_proj.bias.shape == (2 * head_dim,)
    assert logits(model).shape == (1, 8, 64)


def test_an_uneven_grouped_query_selection_is_refused_before_surgery() -> None:
    model = build_qwen2()
    plan = (
        LayerPlan(heads=(0, 1, 2, 4), channels=tuple(range(24))),
        LayerPlan(heads=(0, 1, 4, 5), channels=tuple(range(24))),
    )

    with pytest.raises(PruneError, match="same number of query heads"):
        compact_model(model, plan)


def test_a_plan_that_does_not_cover_every_layer_is_refused() -> None:
    model = build_llama()

    with pytest.raises(PruneError, match="covers 1 layers but the model has 2"):
        compact_model(model, (LayerPlan(heads=(0,), channels=(0,)),))


class TestRoundTrip:
    """Save a compacted model, load it back, and expect the same function."""

    @staticmethod
    def save(model: Any, plan: tuple[LayerPlan, ...], directory: Path) -> None:
        groups = describe_layers(model)
        compact_model(model, plan)
        model.save_pretrained(directory)
        write_descriptor(
            directory / "subnetwork.json",
            plan,
            groups,
            name="test-subnetwork",
            base_model="test",
            method="test",
        )

    @pytest.mark.parametrize("build", [build_llama, build_qwen2], ids=["llama", "qwen2"])
    def test_a_compacted_checkpoint_loads_back_to_the_same_function(
        self, build: Any, tmp_path: Path
    ) -> None:
        import torch
        from recipes.pruned import load_model

        model = build()
        # Uneven on purpose: a uniform plan would pass even if the loader
        # ignored the descriptor and trusted the dense config.
        step = model.config.num_attention_heads // model.config.num_key_value_heads
        plan = (
            LayerPlan(
                heads=tuple(range(0, model.config.num_attention_heads, max(step, 2))),
                channels=tuple(range(12)),
            ),
            LayerPlan(
                heads=tuple(range(model.config.num_attention_heads)), channels=tuple(range(24))
            ),
        )
        self.save(model, plan, tmp_path)
        expected = logits(model)

        loaded, subnetwork = load_model(tmp_path)

        assert torch.equal(logits(loaded), expected)
        assert subnetwork.name == "test-subnetwork"
        # The descriptor is the only record of the widths, and the loader is
        # the only consumer that must agree with it.
        assert plan_from_subnetwork(subnetwork) == plan

    def test_a_descriptor_that_disagrees_with_the_weights_fails_at_load(
        self, tmp_path: Path
    ) -> None:
        import json

        from recipes.pruned import load_model

        model = build_llama()
        plan = (
            LayerPlan(heads=(0, 1), channels=tuple(range(12))),
            LayerPlan(heads=(0, 1), channels=tuple(range(12))),
        )
        self.save(model, plan, tmp_path)

        descriptor = tmp_path / "subnetwork.json"
        payload = json.loads(descriptor.read_text())
        payload["components"]["ffn_channels"]["kept"]["0"] = list(range(11))
        descriptor.write_text(json.dumps(payload))

        with pytest.raises(ValueError, match="disagrees with the weights"):
            load_model(tmp_path)

    def test_a_tied_checkpoint_comes_back_tied(self, tmp_path: Path) -> None:
        from recipes.pruned import load_model

        model = build_qwen2()
        self.save(model, full_plan(model), tmp_path)

        loaded, _ = load_model(tmp_path)

        # assign=True breaks the tie, so untied here means lm_head holds
        # uninitialised storage.
        assert loaded.lm_head.weight.data_ptr() == loaded.model.embed_tokens.weight.data_ptr()


def test_the_descriptor_round_trips_through_the_analysis_loader(tmp_path: Path) -> None:
    model = build_llama()
    groups = describe_layers(model)
    plan = (
        LayerPlan(heads=(0, 3), channels=(1, 2, 3)),
        LayerPlan(heads=(1,), channels=(0, 4)),
    )

    subnetwork = write_descriptor(
        tmp_path / "flap50.json",
        plan,
        groups,
        name="flap50",
        base_model="test/model",
        method="FLAP",
        pruned_for="de-en,en-de",
    )

    assert subnetwork.components["attention_heads"].total == 4
    assert subnetwork.components["attention_heads"].kept == {"0": (0, 3), "1": (1,)}
    assert subnetwork.components["ffn_channels"].total == 48
    assert plan_from_subnetwork(subnetwork) == plan
