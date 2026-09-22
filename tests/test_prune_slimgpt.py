"""SlimGPT's criterion and its error compensation.

The compensation is the method. A ranking that picks the same columns without
the weight update is far cheaper, so the test that matters is that the update
reduces the error the layer makes on the activations its Hessian came from.

Pruning nothing must also change nothing: compensation sweeps every column, and
an off-by-one there would corrupt a model asked to keep all of it.
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


def test_pruning_nothing_leaves_the_model_alone() -> None:
    import torch

    model = tiny_llama()
    groups = describe_layers(model)
    batches = token_batches()
    with torch.inference_mode():
        before = model(**batches[0]).logits.clone()

    plan = slimgpt.prune(model, batches, groups, sparsity=0.0, allocation="uniform")

    assert [len(layer.heads) for layer in plan] == [4, 4]
    assert [len(layer.channels) for layer in plan] == [48, 48]
    with torch.inference_mode():
        assert torch.allclose(model(**batches[0]).logits, before, atol=1e-4)


def test_the_compensation_reduces_the_error_the_layer_makes() -> None:
    import torch

    model = tiny_llama()
    batches = token_batches(count=6)
    name = "model.layers.0.mlp.down_proj"
    activations = layer_inputs(model, batches, name)
    hessian = activations.t() @ activations / activations.shape[0]

    projection = dict(model.named_modules())[name]
    dense = projection.weight.data.float().clone()
    target = activations @ dense.t()

    kept = sorted(range(0, 48, 2))
    index = torch.as_tensor(kept)
    naive = activations.index_select(1, index) @ dense.index_select(1, index).t()

    slimgpt._compensate(projection, hessian, set(kept))
    updated = projection.weight.data.float()
    compensated = activations.index_select(1, index) @ updated.index_select(1, index).t()

    # The removed columns' contribution moves onto the survivors. Without
    # this the criterion is a ranking and the Cholesky work is wasted.
    assert (compensated - target).pow(2).sum() < (naive - target).pow(2).sum()
    # Zeroed on the way out, so compaction drops nothing load bearing.
    dropped = [column for column in range(48) if column not in set(kept)]
    assert torch.count_nonzero(updated.index_select(1, torch.as_tensor(dropped))) == 0


def test_a_dead_channel_is_the_first_thing_discarded() -> None:
    import torch

    model = tiny_llama()
    groups = describe_layers(model)
    batches = token_batches()
    inputs = slimgpt._layer_inputs(model, batches)
    hessians = slimgpt._hessians(model.model.layers[0], inputs)
    # No variance leaves a zero on the diagonal, which is not invertible.
    hessians["down_proj"][5, :] = 0.0
    hessians["down_proj"][:, 5] = 0.0

    costs = slimgpt._column_costs(model.model.layers[0].mlp.down_proj, hessians["down_proj"])

    assert costs[5] == 0.0
    assert torch.isfinite(costs).all()
    assert groups[0].intermediate == costs.numel()


def test_head_costs_pool_the_columns_of_one_head() -> None:
    import torch

    model = tiny_llama()
    groups = describe_layers(model)
    costs = torch.arange(32, dtype=torch.float32)

    scores = pool_heads(costs, groups[0])

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
    from recipes.pruned import load_model

    model = build()
    groups = describe_layers(model)

    plan = slimgpt.prune(model, token_batches(), groups, sparsity=0.5, allocation="global")

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
