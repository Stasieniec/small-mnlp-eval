"""LLM-Pruner's Taylor criterion.

A group that cannot move the loss must score zero, grouped-query attention must
not count its untouched key and value rows into a head's score, and the
backward pass must leave the model's gradient state as it found it. The last is
not cosmetic: a leak would silently change what the next method in a sweep
computes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from mnlp_eval.prune import PruneError
from mnlp_eval.prune.budget import allocate
from mnlp_eval.prune.compact import compact_model, write_descriptor
from mnlp_eval.prune.groups import describe_layers
from mnlp_eval.prune.methods import llm_pruner
from stubs import tiny_llama, token_batches

pytestmark = pytest.mark.torch


def test_scores_have_one_entry_per_prunable_unit() -> None:
    model = tiny_llama()
    groups = describe_layers(model)

    heads, channels = llm_pruner.score(model, token_batches(), groups)

    assert [len(layer) for layer in heads] == [4, 4]
    assert [len(layer) for layer in channels] == [48, 48]
    assert all(value >= 0.0 for layer in heads for value in layer)


def test_a_head_whose_weights_are_zero_cannot_move_the_loss() -> None:
    import torch

    model = tiny_llama()
    groups = describe_layers(model)
    head_dim = groups[0].head_dim
    with torch.no_grad():
        # Head 2 contributes nothing, so removing it costs exactly nothing.
        for name in ("q_proj", "k_proj", "v_proj"):
            getattr(model.model.layers[0].self_attn, name).weight[2 * head_dim : 3 * head_dim] = 0.0
        model.model.layers[0].self_attn.o_proj.weight[:, 2 * head_dim : 3 * head_dim] = 0.0

    heads, _ = llm_pruner.score(model, token_batches(), groups)

    assert heads[0][2] == pytest.approx(0.0, abs=1e-6)
    assert heads[0].index(min(heads[0])) == 2


def test_an_ffn_channel_whose_weights_are_zero_scores_zero() -> None:
    import torch

    model = tiny_llama()
    groups = describe_layers(model)
    with torch.no_grad():
        model.model.layers[1].mlp.gate_proj.weight[9] = 0.0
        model.model.layers[1].mlp.up_proj.weight[9] = 0.0
        model.model.layers[1].mlp.down_proj.weight[:, 9] = 0.0

    _, channels = llm_pruner.score(model, token_batches(), groups)

    assert channels[1][9] == pytest.approx(0.0, abs=1e-6)


def test_grouped_query_heads_ignore_the_shared_key_value_salience() -> None:
    import torch

    from mnlp_eval.prune.groups import LayerGroups

    grouped = LayerGroups(index=0, num_heads=8, num_kv_heads=2, head_dim=4, intermediate=8)
    quiet = {
        "q_proj": torch.ones(32, 16),
        "o_proj": torch.ones(16, 32),
        "k_proj": torch.zeros(8, 16),
        "v_proj": torch.zeros(8, 16),
    }
    loud = dict(quiet, k_proj=torch.full((8, 16), 1e6), v_proj=torch.full((8, 16), 1e6))

    # These rows are never removed under grouped-query attention, and each
    # belongs to every query head in its group, so counting one would add the
    # same quantity to four heads for a decision it plays no part in.
    assert llm_pruner._head_scores(loud, grouped) == llm_pruner._head_scores(quiet, grouped)


def test_multi_head_attention_does_count_the_key_value_salience() -> None:
    import torch

    from mnlp_eval.prune.groups import LayerGroups

    plain = LayerGroups(index=0, num_heads=4, num_kv_heads=4, head_dim=4, intermediate=8)
    quiet = {
        "q_proj": torch.ones(16, 16),
        "o_proj": torch.ones(16, 16),
        "k_proj": torch.zeros(16, 16),
        "v_proj": torch.zeros(16, 16),
    }
    loud = dict(quiet, k_proj=torch.full((16, 16), 5.0))

    # Here the key/value head goes with its one query head, so it counts.
    assert llm_pruner._head_scores(loud, plain) != llm_pruner._head_scores(quiet, plain)


def test_the_gradient_state_of_the_model_is_restored() -> None:
    model = tiny_llama()
    groups = describe_layers(model)
    before = {name: parameter.requires_grad for name, parameter in model.named_parameters()}

    llm_pruner.score(model, token_batches(), groups)

    after = {name: parameter.requires_grad for name, parameter in model.named_parameters()}
    assert after == before
    assert all(parameter.grad is None for parameter in model.parameters())


def test_an_empty_calibration_set_is_refused() -> None:
    model = tiny_llama()

    with pytest.raises(PruneError, match="no batches"):
        llm_pruner.score(model, [], describe_layers(model))


def test_the_whole_criterion_produces_a_loadable_checkpoint(tmp_path: Path) -> None:
    from recipes.pruned import load_model

    model = tiny_llama()
    groups = describe_layers(model)
    heads, channels = llm_pruner.score(model, token_batches(), groups)
    plan = allocate(heads, channels, groups, sparsity=0.5, allocation="global")

    compact_model(model, plan)
    model.save_pretrained(tmp_path)
    subnetwork = write_descriptor(
        tmp_path / "subnetwork.json",
        plan,
        groups,
        name="llm-pruner50",
        base_model="test/model",
        method="LLM-Pruner",
    )

    loaded, _ = load_model(tmp_path)

    assert abs(subnetwork.overall_sparsity - 0.5) < 0.05
    # No compensation step, so no bias is added.
    assert loaded.model.layers[0].mlp.down_proj.bias is None


def test_tokens_behind_the_padding_mask_do_not_reach_the_scores() -> None:
    model = tiny_llama()
    groups = describe_layers(model)
    batches = token_batches(pad=2)
    scrambled = [
        {"input_ids": batch["input_ids"].clone(), "attention_mask": batch["attention_mask"]}
        for batch in batches
    ]
    for batch in scrambled:
        batch["input_ids"][:, :2] = (batch["input_ids"][:, :2] + 7) % 64

    # What sits behind the mask is not data. If it moves a score, either the
    # mask is not applied or a pad position is predicting the first real token.
    assert llm_pruner.score(model, scrambled, groups) == llm_pruner.score(model, batches, groups)


def test_scores_are_reproducible_for_the_same_calibration_data() -> None:
    model = tiny_llama()
    groups = describe_layers(model)

    first = llm_pruner.score(model, token_batches(), groups)
    second = llm_pruner.score(model, token_batches(), groups)

    assert first == second


def test_every_prunable_projection_contributes(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: set[str] = set()
    model = tiny_llama()
    groups = describe_layers(model)
    original = llm_pruner._head_scores

    def spy(salience: dict[str, Any], group: Any) -> Any:
        seen.update(salience)
        return original(salience, group)

    monkeypatch.setattr(llm_pruner, "_head_scores", spy)
    llm_pruner.score(model, token_batches(), groups)

    assert seen == set(llm_pruner.ATTENTION_PROJECTIONS) | set(llm_pruner.FFN_PROJECTIONS)
