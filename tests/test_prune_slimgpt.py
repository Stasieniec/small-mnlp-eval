"""SlimGPT's criterion and its error compensation.

The compensation is the method. A ranking that picks the same columns without
the weight update is far cheaper, so the test that matters is that the update
reduces the error the layer makes on the activations its Hessian came from.
tests/test_prune_slimgpt_exact.py checks it against the closed-form optimum.

Pruning nothing must also change nothing: a projection that loses no unit is
never round-tripped through the float64 working copy, and an off-by-one in the
bookkeeping would corrupt a model asked to keep all of it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from mnlp_eval.prune import PruneError
from mnlp_eval.prune.compact import write_descriptor
from mnlp_eval.prune.groups import describe_layers
from mnlp_eval.prune.methods import pool_heads, slimgpt
from stubs import tiny_llama, tiny_qwen2, token_batches

pytestmark = pytest.mark.torch


def layer_inputs(model: Any, batches: list[dict[str, Any]], name: str) -> Any:
    """Every activation that reached ``name``, stacked as (tokens, channels)."""
    import torch

    seen: list[Any] = []

    def capture(_module: Any, args: tuple[Any, ...]) -> None:
        seen.append(args[0].reshape(-1, args[0].shape[-1]).float().clone())

    module = dict(model.named_modules())[name]
    handle = module.register_forward_pre_hook(capture)
    try:
        with torch.inference_mode():
            for batch in batches:
                model(**batch)
    finally:
        handle.remove()
    return torch.cat(seen, dim=0)


@pytest.mark.parametrize("allocation", ["uniform", "log-increase", "global"])
@pytest.mark.parametrize("build", [tiny_llama, tiny_qwen2], ids=["llama", "qwen2"])
def test_pruning_nothing_leaves_the_model_alone(build: Any, allocation: str) -> None:
    import torch

    model = build()
    groups = describe_layers(model)
    batches = token_batches()
    with torch.inference_mode():
        before = model(**batches[0]).logits.clone()

    plan = slimgpt.prune(model, batches, groups, sparsity=0.0, allocation=allocation)

    assert [len(layer.heads) for layer in plan] == [group.num_heads for group in groups]
    assert [len(layer.channels) for layer in plan] == [group.intermediate for group in groups]
    with torch.inference_mode():
        assert torch.equal(model(**batches[0]).logits, before)


def test_the_compensation_reduces_the_error_the_layer_makes() -> None:
    import torch

    model = tiny_llama()
    batches = token_batches(count=6)
    name = "model.layers.0.mlp.down_proj"
    activations = layer_inputs(model, batches, name).double()
    hessian = activations.t() @ activations / activations.shape[0]

    projection = dict(model.named_modules())[name]
    dense = projection.weight.data.double().clone()
    target = activations @ dense.t()

    kept = list(range(0, 48, 2))
    dropped = [column for column in range(48) if column not in set(kept)]
    index = torch.as_tensor(kept)
    naive = activations.index_select(1, index) @ dense.index_select(1, index).t()

    inverse, _ = slimgpt._inverse_hessian(hessian)
    updated = dense.clone()
    slimgpt._remove_columns(updated, inverse, torch.as_tensor(dropped))
    compensated = activations.index_select(1, index) @ updated.index_select(1, index).t()

    # The removed columns' contribution moves onto the survivors. Without
    # this the criterion is a ranking and the inverse Hessian is wasted.
    assert (compensated - target).pow(2).sum() < 0.5 * (naive - target).pow(2).sum()
    # Zeroed on the way out, so compaction drops nothing load bearing.
    assert torch.count_nonzero(updated.index_select(1, torch.as_tensor(dropped))) == 0


def test_a_dead_channel_is_the_first_thing_discarded() -> None:
    import torch

    model = tiny_llama()
    groups = describe_layers(model)
    batches = token_batches()
    inputs = slimgpt._layer_inputs(model, batches)
    hessian = slimgpt._hessians(model.model.layers[0], inputs)["down_proj"]
    # No variance leaves a zero on the diagonal, which is not invertible.
    hessian[5, :] = 0.0
    hessian[:, 5] = 0.0

    inverse, dead = slimgpt._inverse_hessian(hessian)
    weight = model.model.layers[0].mlp.down_proj.weight.data.double().clone()
    costs = slimgpt._channel_costs(weight, inverse, dead)

    assert dead.nonzero().flatten().tolist() == [5]
    assert costs[5] == 0.0
    assert torch.isfinite(costs).all()
    assert groups[0].intermediate == costs.numel()

    kept = slimgpt._prune_channels(weight, inverse, dead, 47, batch=1)
    assert 5 not in kept


def test_a_kept_dead_channel_keeps_its_weight() -> None:
    import torch

    model = tiny_llama()
    inputs = slimgpt._layer_inputs(model, token_batches())
    hessian = slimgpt._hessians(model.model.layers[0], inputs)["down_proj"]
    hessian[5, :] = 0.0
    hessian[:, 5] = 0.0
    inverse, _ = slimgpt._inverse_hessian(hessian)
    dense = model.model.layers[0].mlp.down_proj.weight.data.double().clone()
    weight = dense.clone()

    slimgpt._remove_columns(weight, inverse, torch.as_tensor([0, 7, 30]))

    # The calibration set says nothing about a channel it never activated, so
    # compensation must not move it: it stays decoupled in the inverse.
    assert torch.equal(weight[:, 5], dense[:, 5])
    assert not torch.allclose(weight[:, 6], dense[:, 6])


def test_tokens_behind_the_padding_mask_do_not_reach_the_hessian() -> None:
    import torch

    model = tiny_llama()
    batches = token_batches(pad=2)
    scrambled = [
        {"input_ids": batch["input_ids"].clone(), "attention_mask": batch["attention_mask"]}
        for batch in batches
    ]
    for batch in scrambled:
        batch["input_ids"][:, :2] = (batch["input_ids"][:, :2] + 7) % 64

    layer = model.model.layers[0]
    clean = slimgpt._hessians(layer, slimgpt._layer_inputs(model, batches))
    noisy = slimgpt._hessians(layer, slimgpt._layer_inputs(model, scrambled))

    # What sits behind the mask is not data, and here it reaches further than
    # a score: the Hessian is also what the compensating weight update solves
    # against, so a padded position moves the surviving weights.
    for name in ("o_proj", "down_proj"):
        assert torch.allclose(clean[name], noisy[name], atol=1e-5)


def test_pool_heads_sums_the_columns_of_one_head() -> None:
    import torch

    model = tiny_llama()
    groups = describe_layers(model)
    costs = torch.arange(32, dtype=torch.float32)

    scores = pool_heads(costs, groups[0])

    # FLAP and LLM-Pruner pool per-column scores this way. SlimGPT scores a
    # head as a block instead, because its columns compensate for each other.
    # Four heads of eight channels each, over 0..31.
    assert scores == [28.0, 92.0, 156.0, 220.0]


def test_a_head_score_that_does_not_match_the_layer_is_refused() -> None:
    import torch

    groups = describe_layers(tiny_llama())

    with pytest.raises(PruneError, match="expected 32"):
        pool_heads(torch.zeros(30), groups[0])


def test_an_empty_calibration_set_is_refused() -> None:
    model = tiny_llama()

    with pytest.raises(PruneError, match="no batches"):
        slimgpt.prune(model, [], describe_layers(model), sparsity=0.5)


@pytest.mark.parametrize("build", [tiny_llama, tiny_qwen2], ids=["llama", "qwen2"])
def test_the_whole_criterion_produces_a_loadable_checkpoint(build: Any, tmp_path: Path) -> None:
    import torch
    from recipes.pruned import load_model

    model = build()
    groups = describe_layers(model)
    batches = token_batches()

    plan = slimgpt.prune(model, batches, groups, sparsity=0.5, allocation="global")

    model.save_pretrained(tmp_path)
    subnetwork = write_descriptor(
        tmp_path / "subnetwork.json",
        plan,
        groups,
        name="slimgpt50",
        base_model="test/model",
        method="SlimGPT",
    )
    loaded, _ = load_model(tmp_path)

    assert abs(subnetwork.overall_sparsity - 0.5) < 0.05
    for layer, group in zip(plan, groups, strict=True):
        group.validate_heads(layer.heads)
    assert loaded.model.layers[0].mlp.down_proj.bias is None
    # The compensated weights, not only the shapes, survive the round trip.
    with torch.inference_mode():
        assert torch.equal(loaded(**batches[0]).logits, model(**batches[0]).logits)


def test_the_dense_scores_see_each_layer_s_own_input(monkeypatch: pytest.MonkeyPatch) -> None:
    import torch

    model = tiny_llama()
    groups = describe_layers(model)
    batches = token_batches()
    inputs = slimgpt._layer_inputs(model, batches)
    expected = slimgpt._advance(model.model.layers[0], inputs)[0][0].clone()

    seen: list[Any] = []
    hessians = slimgpt._hessians

    def spy(layer: Any, given: Any, *args: Any) -> Any:
        seen.append(given[0][0].clone())
        return hessians(layer, given, *args)

    monkeypatch.setattr(slimgpt, "_hessians", spy)
    slimgpt.prune(model, batches, groups, sparsity=0.5, allocation="global")

    # Calls one and two are the scoring pass. Scoring layer 1 on the
    # embeddings rather than on layer 0's output misranked every layer past
    # the first in the 50 percent pilot.
    assert torch.equal(seen[0], inputs[0][0])
    assert torch.allclose(seen[1], expected)
    assert not torch.allclose(seen[1], inputs[0][0])


def test_later_layers_are_pruned_on_what_earlier_ones_now_produce() -> None:
    import torch

    model = tiny_llama()
    batches = token_batches()
    inputs = slimgpt._layer_inputs(model, batches)
    first = model.model.layers[0]
    before = slimgpt._advance(first, inputs)[0][0].clone()

    # The next layer's Hessian must come from the pruned layer's actual
    # output, so the driver has to re-run it rather than cache.
    with torch.no_grad():
        first.mlp.down_proj.weight.mul_(0.5)
    after = slimgpt._advance(first, inputs)[0][0]

    assert not torch.allclose(before, after)
