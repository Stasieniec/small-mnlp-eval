"""SlimGPT's update against the closed-form optimum, and its greedy selection.

Every weight update here has a closed form to compare against. Removing the
columns ``S`` of ``W`` and refitting the rest under the damped Hessian ``Hd``
minimises ``tr((W - W') Hd (W - W')^T)`` subject to ``W'[:, S] = 0``. Setting
the gradient in the kept columns ``K`` to zero gives ``W'_K Hd_KK = W Hd_:,K``.
A sweep that compensates a column only with the columns after it misses this
for every set but a trailing one, and the last heads of ``o_proj`` then get no
compensation at all, which is the bug these tests were written against.
"""

from __future__ import annotations

from typing import Any

import pytest

from mnlp_eval.prune.budget import keep_counts
from mnlp_eval.prune.groups import LayerGroups, describe_layers
from mnlp_eval.prune.methods import slimgpt
from stubs import tiny_llama, tiny_qwen2, token_batches

pytestmark = pytest.mark.torch


def correlated(tokens: int, width: int, seed: int = 0) -> Any:
    """Inputs whose channels are strongly correlated, as a layer's are."""
    import torch

    generator = torch.Generator().manual_seed(seed)
    base = torch.randn(tokens, width, dtype=torch.float64, generator=generator)
    mix = torch.randn(width, width, dtype=torch.float64, generator=generator)
    return base @ mix


def hessian(inputs: Any) -> Any:
    return inputs.t() @ inputs / inputs.shape[0]


def damped(matrix: Any) -> Any:
    """The Hessian the criterion solves against, ridge included."""
    import torch

    ridge = slimgpt.DAMPING * matrix.diagonal().mean()
    return matrix + ridge * torch.eye(matrix.shape[0], dtype=matrix.dtype)


def optimum(weight: Any, matrix: Any, kept: list[int]) -> Any:
    """The least-squares refit onto ``kept``, zero everywhere else."""
    import torch

    index = torch.as_tensor(kept)
    block = matrix.index_select(0, index).index_select(1, index)
    refit = torch.linalg.solve(block, matrix.index_select(0, index) @ weight.t()).t()
    result = torch.zeros_like(weight)
    result[:, index] = refit
    return result


def error(weight: Any, other: Any, matrix: Any) -> float:
    """``tr((W - W') H (W - W')^T)``, the squared output error per token."""
    import torch

    difference = weight - other
    return float(torch.trace(difference @ matrix @ difference.t()))


def head_columns(heads: list[int], head_dim: int) -> list[int]:
    return [head * head_dim + offset for head in heads for offset in range(head_dim)]


def multi_head(num_heads: int, head_dim: int, num_kv_heads: int | None = None) -> LayerGroups:
    return LayerGroups(
        index=0,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads or num_heads,
        head_dim=head_dim,
        intermediate=8,
    )


def close(actual: Any, expected: Any) -> bool:
    import torch

    scale = float(expected.abs().max())
    return bool(torch.allclose(actual, expected, rtol=0.0, atol=1e-10 * scale))


@pytest.mark.parametrize(
    "removed",
    [list(range(24, 32)), list(range(8)), [1, 5, 6, 13, 22, 31]],
    ids=["last-heads", "first-heads", "scattered"],
)
def test_removing_columns_reaches_the_least_squares_optimum(removed: list[int]) -> None:
    import torch

    inputs = correlated(400, 32)
    matrix = damped(hessian(inputs))
    weight = torch.randn(16, 32, dtype=torch.float64, generator=torch.Generator().manual_seed(3))
    kept = [column for column in range(32) if column not in removed]

    inverse, _ = slimgpt._inverse_hessian(hessian(inputs))
    updated = weight.clone()
    cost = slimgpt._remove_columns(updated, inverse, torch.as_tensor(removed))

    expected = optimum(weight, matrix, kept)
    assert close(updated, expected)
    assert torch.count_nonzero(updated[:, removed]) == 0
    # The reported cost is the error the removal actually adds.
    assert cost == pytest.approx(error(weight, expected, matrix), rel=1e-10)
    # The inverse is downdated to the inverse of the survivors' Hessian, which
    # is what the next greedy step scores against.
    index = torch.as_tensor(kept)
    survivors = torch.linalg.inv(matrix.index_select(0, index).index_select(1, index))
    assert close(inverse.index_select(0, index).index_select(1, index), survivors)
    assert torch.count_nonzero(inverse[removed]) == 0
    assert torch.count_nonzero(inverse[:, removed]) == 0


def test_the_last_heads_are_compensated() -> None:
    import torch

    head_dim = 4
    inputs = correlated(400, 8 * head_dim)
    matrix = damped(hessian(inputs))
    weight = torch.randn(16, 32, dtype=torch.float64, generator=torch.Generator().manual_seed(4))
    removed = head_columns([6, 7], head_dim)

    inverse, _ = slimgpt._inverse_hessian(hessian(inputs))
    updated = weight.clone()
    slimgpt._remove_columns(updated, inverse, torch.as_tensor(removed))
    naive = weight.clone()
    naive[:, removed] = 0.0

    # A sweep in index order has no later column to push the last heads'
    # error onto, so it equals naive zeroing there.
    assert error(weight, updated, matrix) < 0.5 * error(weight, naive, matrix)
    assert not torch.allclose(updated[:, :24], weight[:, :24])


def test_removing_in_steps_ends_where_one_joint_removal_does() -> None:
    import torch

    inputs = correlated(400, 32, seed=1)
    weight = torch.randn(16, 32, dtype=torch.float64, generator=torch.Generator().manual_seed(5))

    stepwise = weight.clone()
    inverse, _ = slimgpt._inverse_hessian(hessian(inputs))
    for removed in ([30, 31, 2], [7], [12, 13, 14, 15]):
        slimgpt._remove_columns(stepwise, inverse, torch.as_tensor(removed))

    joint = weight.clone()
    inverse, _ = slimgpt._inverse_hessian(hessian(inputs))
    slimgpt._remove_columns(joint, inverse, torch.as_tensor([30, 31, 2, 7, 12, 13, 14, 15]))

    # So the greedy order decides which units go, never how good the refit is.
    assert close(stepwise, joint)


def test_greedy_heads_end_at_the_optimum_for_what_they_keep() -> None:
    import torch

    group = multi_head(8, 4)
    inputs = correlated(400, 32, seed=2)
    weight = torch.randn(16, 32, dtype=torch.float64, generator=torch.Generator().manual_seed(6))

    inverse, dead = slimgpt._inverse_hessian(hessian(inputs))
    updated = weight.clone()
    kept = slimgpt._prune_heads(updated, inverse, dead, group, 3)

    assert len(kept) == 3
    assert close(updated, optimum(weight, damped(hessian(inputs)), head_columns(list(kept), 4)))


def test_greedy_channels_end_at_the_optimum_for_what_they_keep() -> None:
    import torch

    inputs = correlated(400, 40, seed=3)
    weight = torch.randn(16, 40, dtype=torch.float64, generator=torch.Generator().manual_seed(7))

    inverse, dead = slimgpt._inverse_hessian(hessian(inputs))
    updated = weight.clone()
    kept = slimgpt._prune_channels(updated, inverse, dead, 13, batch=6)

    assert len(kept) == 13
    assert close(updated, optimum(weight, damped(hessian(inputs)), list(kept)))


def test_head_selection_does_not_depend_on_position() -> None:
    import torch

    head_dim = 4
    group = multi_head(8, head_dim)
    inputs = correlated(400, 32, seed=4)
    weight = torch.randn(16, 32, dtype=torch.float64, generator=torch.Generator().manual_seed(8))
    order = [5, 2, 7, 0, 3, 6, 1, 4]
    columns = torch.as_tensor(head_columns(order, head_dim))

    def run(given_inputs: Any, given_weight: Any) -> tuple[Any, tuple[int, ...], float, Any]:
        inverse, dead = slimgpt._inverse_hessian(hessian(given_inputs))
        single = slimgpt._head_costs(given_weight, inverse, dead, range(8), head_dim)
        updated = given_weight.clone()
        kept = slimgpt._prune_heads(updated, inverse, dead, group, 4)
        return single, kept, error(given_weight, updated, damped(hessian(given_inputs))), updated

    single, kept, loss, _ = run(inputs, weight)
    moved_single, moved_kept, moved_loss, _ = run(
        inputs.index_select(1, columns), weight.index_select(1, columns)
    )

    # Head i of the permuted layer is head order[i] of the original. Scoring
    # on a trailing submatrix makes late heads look more important than
    # early ones, which this catches.
    assert torch.allclose(moved_single, single[torch.as_tensor(order)], rtol=1e-10)
    assert sorted(order[head] for head in moved_kept) == list(kept)
    assert moved_loss == pytest.approx(loss, rel=1e-10)


def test_channel_selection_does_not_depend_on_position() -> None:
    import torch

    inputs = correlated(400, 24, seed=5)
    weight = torch.randn(16, 24, dtype=torch.float64, generator=torch.Generator().manual_seed(9))
    order = torch.randperm(24, generator=torch.Generator().manual_seed(10))

    def run(given_inputs: Any, given_weight: Any) -> tuple[tuple[int, ...], float]:
        inverse, dead = slimgpt._inverse_hessian(hessian(given_inputs))
        updated = given_weight.clone()
        kept = slimgpt._prune_channels(updated, inverse, dead, 10, batch=3)
        return kept, error(given_weight, updated, damped(hessian(given_inputs)))

    kept, loss = run(inputs, weight)
    moved_kept, moved_loss = run(inputs.index_select(1, order), weight.index_select(1, order))

    assert sorted(int(order[channel]) for channel in moved_kept) == list(kept)
    assert moved_loss == pytest.approx(loss, rel=1e-10)


def twins(tokens: int, width: int, block: int) -> Any:
    """Independent inputs, except that block 1 nearly duplicates block 0.

    Either twin alone is nearly free to remove, since the other absorbs it.
    Removing both is expensive, which a single-shot ranking cannot see.
    """
    import torch

    generator = torch.Generator().manual_seed(11)
    inputs = torch.randn(tokens, width, dtype=torch.float64, generator=generator)
    noise = torch.randn(tokens, block, dtype=torch.float64, generator=generator)
    inputs[:, block : 2 * block] = inputs[:, :block] + 0.01 * noise
    return inputs


def test_heads_are_rescored_after_every_removal() -> None:
    import torch

    head_dim = 2
    group = multi_head(4, head_dim)
    inputs = twins(500, 8, head_dim)
    matrix = damped(hessian(inputs))
    scales = torch.tensor([1.0, 1.0, 0.6, 0.9], dtype=torch.float64).repeat_interleave(head_dim)
    weight = torch.ones(6, 8, dtype=torch.float64) * scales

    inverse, dead = slimgpt._inverse_hessian(hessian(inputs))
    single = slimgpt._head_costs(weight, inverse, dead, range(4), head_dim)
    updated = weight.clone()
    kept = slimgpt._prune_heads(updated, inverse, dead, group, 2)

    # Scored once, the twins are the two cheapest and both go.
    assert sorted(torch.argsort(single)[:2].tolist()) == [0, 1]
    # Rescored, the surviving twin carries both and is the dearest.
    assert len(set(kept) & {0, 1}) == 1
    assert 3 in kept
    joint = optimum(weight, matrix, head_columns([2, 3], head_dim))
    assert error(weight, updated, matrix) < 0.5 * error(weight, joint, matrix)


def test_channels_are_rescored_between_batches() -> None:
    import torch

    inputs = twins(500, 6, 1)
    matrix = damped(hessian(inputs))
    weight = torch.ones(6, 6, dtype=torch.float64) * torch.tensor(
        [1.0, 1.0, 0.5, 0.7, 0.8, 0.9], dtype=torch.float64
    )

    def run(batch: int) -> tuple[tuple[int, ...], float]:
        inverse, dead = slimgpt._inverse_hessian(hessian(inputs))
        updated = weight.clone()
        kept = slimgpt._prune_channels(updated, inverse, dead, 4, batch=batch)
        return kept, error(weight, updated, matrix)

    greedy, greedy_loss = run(1)
    batched, batched_loss = run(2)

    # Within a batch the costs are single-shot, so both twins go together.
    assert batched == (2, 3, 4, 5)
    assert len(set(greedy) & {0, 1}) == 1
    assert 2 not in greedy
    assert greedy_loss < 0.5 * batched_loss


def test_grouped_query_removes_one_head_per_group_per_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    head_dim = 2
    group = multi_head(8, head_dim, num_kv_heads=2)
    inputs = correlated(400, 16, seed=6)
    weight = torch.randn(6, 16, dtype=torch.float64, generator=torch.Generator().manual_seed(12))

    steps: list[list[int]] = []
    remove = slimgpt._remove_columns

    def spy(given: Any, inverse: Any, columns: Any) -> float:
        steps.append(sorted({int(column) // head_dim for column in columns}))
        return remove(given, inverse, columns)

    monkeypatch.setattr(slimgpt, "_remove_columns", spy)
    inverse, dead = slimgpt._inverse_hessian(hessian(inputs))
    updated = weight.clone()
    kept = slimgpt._prune_heads(updated, inverse, dead, group, 4)

    # repeat_kv wires query head i to key/value head i // 4 here, so an
    # uneven selection would misroute the survivors.
    assert group.validate_heads(kept) == kept
    assert len(steps) == 2
    for step in steps:
        assert [head // 4 for head in step] == [0, 1]
    assert close(updated, optimum(weight, damped(hessian(inputs)), head_columns(list(kept), 2)))


def test_an_early_stop_collects_the_same_attention_hessian() -> None:
    import torch

    model = tiny_llama()
    inputs = slimgpt._layer_inputs(model, token_batches(pad=2))
    layer = model.model.layers[0]

    alone = slimgpt._hessians(layer, inputs, ("o_proj",))
    both = slimgpt._hessians(layer, inputs)

    assert set(alone) == {"o_proj"}
    assert torch.equal(alone["o_proj"], both["o_proj"])


@pytest.mark.parametrize("allocation", ["uniform", "log-increase", "global"])
@pytest.mark.parametrize("build", [tiny_llama, tiny_qwen2], ids=["llama", "qwen2"])
def test_each_layer_keeps_what_the_budget_allots(
    build: Any, allocation: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    import torch

    model = build()
    groups = describe_layers(model)
    batches = token_batches()
    allotted: list[tuple[list[int], list[int]]] = []

    def spy(*args: Any, **kwargs: Any) -> tuple[list[int], list[int]]:
        result = keep_counts(*args, **kwargs)
        allotted.append(result)
        return result

    monkeypatch.setattr(slimgpt, "keep_counts", spy)
    plan = slimgpt.prune(model, batches, groups, sparsity=0.5, allocation=allocation)

    assert len(allotted) == 1
    heads, channels = allotted[0]
    assert [len(layer.heads) for layer in plan] == heads
    assert [len(layer.channels) for layer in plan] == channels
    for layer, layer_plan, group in zip(model.model.layers, plan, groups, strict=True):
        group.validate_heads(layer_plan.heads)
        group.validate_channels(layer_plan.channels)
        assert layer.self_attn.o_proj.in_features == len(layer_plan.heads) * group.head_dim
        assert layer.mlp.down_proj.in_features == len(layer_plan.channels)
    with torch.inference_mode():
        assert torch.isfinite(model(**batches[0]).logits).all()


@pytest.mark.parametrize("allocation", ["uniform", "log-increase"])
def test_a_schedule_needs_no_dense_scoring_pass(
    allocation: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = tiny_llama()

    def refuse(*_args: Any) -> Any:
        raise AssertionError("scored the dense model for a budget that reads no scores")

    monkeypatch.setattr(slimgpt, "_dense_scores", refuse)
    slimgpt.prune(
        model, token_batches(), describe_layers(model), sparsity=0.5, allocation=allocation
    )


@pytest.mark.parametrize("build", [tiny_llama, tiny_qwen2], ids=["llama", "qwen2"])
def test_the_ffn_hessian_sees_the_pruned_attention(
    build: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = build()
    groups = describe_layers(model)
    silenced: list[set[int]] = []
    hessians = slimgpt._hessians

    def spy(layer: Any, given: Any, names: Any = slimgpt._PROJECTIONS) -> Any:
        if tuple(names) == ("down_proj",):
            weight = layer.self_attn.o_proj.weight.data
            per_head = weight.view(weight.shape[0], -1, groups[0].head_dim)
            silenced.append(set(per_head.abs().sum(dim=(0, 2)).eq(0).nonzero().flatten().tolist()))
        return hessians(layer, given, names)

    monkeypatch.setattr(slimgpt, "_hessians", spy)
    plan = slimgpt.prune(model, token_batches(), groups, sparsity=0.5, allocation="uniform")

    # Collected after the heads were cut and compensated, before compaction:
    # the cut heads' columns are zero and nothing else is.
    assert silenced == [
        set(range(group.num_heads)) - set(layer.heads)
        for layer, group in zip(plan, groups, strict=True)
    ]


def test_the_first_layer_ends_at_the_least_squares_optimum() -> None:
    import torch

    model = tiny_llama()
    groups = describe_layers(model)
    batches = token_batches(count=4, pad=1)
    layer = model.model.layers[0]
    attention = slimgpt._hessians(layer, slimgpt._layer_inputs(model, batches), ("o_proj",))
    dense = layer.self_attn.o_proj.weight.data.double().clone()

    plan = slimgpt.prune(model, batches, groups, sparsity=0.5, allocation="uniform")

    kept = head_columns(list(plan[0].heads), groups[0].head_dim)
    expected = optimum(dense, damped(attention["o_proj"].double()), kept)[:, kept]
    actual = layer.self_attn.o_proj.weight.data.double()
    # float32 storage, so to float32 rounding.
    assert torch.allclose(actual, expected, rtol=1e-5, atol=1e-6 * float(expected.abs().max()))
