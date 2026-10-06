"""Spreading a sparsity budget over the layers.

An off-by-one here yields a model that is not at the sparsity it claims, which
no downstream stage can detect: the report quotes the requested figure beside a
measured one and a small gap reads as rounding.

For ``global`` the property that matters is standardising before pooling, so
its test uses layers with identical score shapes at wildly different scales.
"""

from __future__ import annotations

import pytest

from mnlp_eval.prune import PruneError
from mnlp_eval.prune.budget import FIRST_LAYER_SHARE, allocate, keep_counts, log_increase_ratios
from mnlp_eval.prune.groups import LayerGroups


def mha(index: int, num_heads: int = 4, intermediate: int = 8) -> LayerGroups:
    return LayerGroups(
        index=index,
        num_heads=num_heads,
        num_kv_heads=num_heads,
        head_dim=2,
        intermediate=intermediate,
    )


def gqa(index: int, num_heads: int = 8, num_kv_heads: int = 2) -> LayerGroups:
    return LayerGroups(
        index=index, num_heads=num_heads, num_kv_heads=num_kv_heads, head_dim=2, intermediate=8
    )


def flat(layers: int, width: int) -> list[list[float]]:
    """Scores that ascend within each layer, identically across layers."""
    return [[float(index) for index in range(width)] for _ in range(layers)]


def test_uniform_gives_every_layer_the_same_fraction() -> None:
    groups = [mha(0), mha(1)]

    plan = allocate(flat(2, 4), flat(2, 8), groups, sparsity=0.5, allocation="uniform")

    assert [len(layer.heads) for layer in plan] == [2, 2]
    assert [len(layer.channels) for layer in plan] == [4, 4]
    # Highest scores kept, and indices come back sorted and dense-indexed.
    assert plan[0].heads == (2, 3)
    assert plan[0].channels == (4, 5, 6, 7)


def test_uniform_is_unmoved_by_a_layer_scoring_far_higher_than_another() -> None:
    groups = [mha(0), mha(1)]
    heads = [[1.0, 2.0, 3.0, 4.0], [100.0, 200.0, 300.0, 400.0]]

    plan = allocate(heads, flat(2, 8), groups, sparsity=0.5, allocation="uniform")

    assert [len(layer.heads) for layer in plan] == [2, 2]


def test_global_spends_more_of_the_budget_where_the_scores_say_to() -> None:
    groups = [mha(0), mha(1)]
    # Every layer is zero-mean after standardising, so unevenness comes from
    # distribution shape: one keeper here against one reject there.
    heads = [[0.0, 0.0, 0.0, 10.0], [0.0, 10.0, 10.0, 10.0]]

    plan = allocate(heads, flat(2, 8), groups, sparsity=0.5, allocation="global")

    assert sum(len(layer.heads) for layer in plan) == 4
    assert len(plan[0].heads) == 1
    assert len(plan[1].heads) == 3
    assert plan[0].heads == (3,)


def test_global_standardises_before_pooling_so_scale_does_not_buy_budget() -> None:
    groups = [mha(0), mha(1)]
    # Identical shape, scaled a thousandfold. Raw pooling would take all four
    # of the second layer's heads and none of the first's.
    heads = [[1.0, 2.0, 3.0, 4.0], [1000.0, 2000.0, 3000.0, 4000.0]]

    plan = allocate(heads, flat(2, 8), groups, sparsity=0.5, allocation="global")

    assert [len(layer.heads) for layer in plan] == [2, 2]


def test_global_hits_the_requested_budget_overall() -> None:
    groups = [mha(index, num_heads=8, intermediate=16) for index in range(4)]
    heads = [[float((index * 7 + layer * 3) % 11) for index in range(8)] for layer in range(4)]

    plan = allocate(heads, flat(4, 16), groups, sparsity=0.25, allocation="global")

    assert sum(len(layer.heads) for layer in plan) == 24
    assert sum(len(layer.channels) for layer in plan) == 48


def test_a_layer_never_loses_everything() -> None:
    groups = [mha(0), mha(1)]
    heads = [[0.0, 0.0, 0.0, 0.0], [1.0, 2.0, 3.0, 4.0]]

    plan = allocate(heads, flat(2, 8), groups, sparsity=0.9, allocation="global")

    # An empty layer is a removed layer, which this is not.
    assert all(layer.heads for layer in plan)
    assert all(layer.channels for layer in plan)


class TestLogIncrease:
    def test_the_ratios_average_to_the_target_and_rise_with_depth(self) -> None:
        ratios = log_increase_ratios(32, 0.2)

        assert ratios[0] == pytest.approx(FIRST_LAYER_SHARE * 0.2)
        assert ratios == sorted(ratios)
        assert sum(ratios) / len(ratios) == pytest.approx(0.2)
        # The figures quoted in the docstring and in docs/pruning.md.
        assert ratios[-1] == pytest.approx(0.254, abs=1e-3)
        assert log_increase_ratios(32, 0.4)[-1] == pytest.approx(0.508, abs=1e-3)

    def test_it_reproduces_the_papers_figure_at_fifty_percent(self) -> None:
        # SlimGPT Figure 4, LLaMA-7B: 0.125 at the first layer, 0.656 at the
        # last. The plotted curve averages 0.516 rather than 0.5; held to an
        # exact mean, the last layer comes out a little lower.
        ratios = log_increase_ratios(32, 0.5)

        assert ratios[0] == pytest.approx(0.125)
        assert ratios[-1] == pytest.approx(0.635, abs=1e-3)

    def test_it_follows_the_papers_logarithmic_shape(self) -> None:
        import math

        ratios = log_increase_ratios(8, 0.3)

        first, last = ratios[0], ratios[-1]
        for index, ratio in enumerate(ratios):
            expected = first + (last - first) * math.log(index + 1) / math.log(8)
            assert ratio == pytest.approx(expected)

    def test_the_first_layer_loses_least_and_the_last_loses_most(self) -> None:
        groups = [mha(index, num_heads=8, intermediate=64) for index in range(4)]

        plan = allocate(flat(4, 8), flat(4, 64), groups, sparsity=0.25, allocation="log-increase")

        kept = [len(layer.channels) for layer in plan]
        assert kept[0] == 64 - round(64 * FIRST_LAYER_SHARE * 0.25)
        assert kept == sorted(kept, reverse=True)
        assert sum(kept) == pytest.approx(192, abs=2)

    def test_scores_decide_which_units_go_but_not_how_many(self) -> None:
        groups = [mha(0), mha(1)]
        heads = [[1.0, 2.0, 3.0, 4.0], [100.0, 200.0, 300.0, 400.0]]

        plan = allocate(heads, flat(2, 8), groups, sparsity=0.25, allocation="log-increase")

        # Layer 0 loses round(4 * 0.0625) = 0 heads, layer 1 round(4 * 0.4375) = 2.
        assert plan[0].heads == (0, 1, 2, 3)
        assert plan[1].heads == (2, 3)

    def test_a_budget_that_would_empty_the_last_layer_is_refused(self) -> None:
        with pytest.raises(PruneError, match="all of it"):
            log_increase_ratios(32, 0.8)


class TestGroupedQueryAttention:
    def test_heads_are_kept_evenly_across_the_key_value_groups(self) -> None:
        groups = [gqa(0)]
        # Group one outscores group two throughout, so an unconstrained top-k
        # would take all four from it and repeat_kv would misroute them.
        heads = [[10.0, 11.0, 12.0, 13.0, 1.0, 2.0, 3.0, 4.0]]

        plan = allocate(heads, flat(1, 8), groups, sparsity=0.5, allocation="uniform")

        assert plan[0].heads == (2, 3, 6, 7)
        groups[0].validate_heads(plan[0].heads)

    def test_the_kept_count_is_rounded_to_a_whole_number_per_group(self) -> None:
        groups = [gqa(0, num_heads=8, num_kv_heads=2)]

        plan = allocate(flat(1, 8), flat(1, 8), groups, sparsity=0.4, allocation="uniform")

        # 8 heads at 0.4 is 4.8 kept, rounded to 4 so each group keeps 2.
        assert len(plan[0].heads) % 2 == 0
        groups[0].validate_heads(plan[0].heads)

    def test_global_allocation_also_respects_the_groups(self) -> None:
        groups = [gqa(0), gqa(1)]
        heads = [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 9.0]]

        plan = allocate(heads, flat(2, 8), groups, sparsity=0.5, allocation="global")

        for layer, group in zip(plan, groups, strict=True):
            group.validate_heads(layer.heads)


def test_sparsity_zero_keeps_everything() -> None:
    groups = [mha(0), mha(1)]

    plan = allocate(flat(2, 4), flat(2, 8), groups, sparsity=0.0, allocation="global")

    assert [layer.heads for layer in plan] == [(0, 1, 2, 3)] * 2
    assert [len(layer.channels) for layer in plan] == [8, 8]


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"sparsity": 1.0}, "fraction removed"),
        ({"sparsity": -0.1}, "fraction removed"),
        ({"allocation": "magic"}, "is not one of"),
    ],
)
def test_a_bad_budget_is_refused(kwargs: dict[str, object], expected: str) -> None:
    settings: dict[str, object] = {"sparsity": 0.5, "allocation": "uniform"}
    settings.update(kwargs)

    with pytest.raises(PruneError, match=expected):
        allocate(flat(2, 4), flat(2, 8), [mha(0), mha(1)], **settings)  # type: ignore[arg-type]


def test_scores_that_do_not_match_the_model_are_refused() -> None:
    with pytest.raises(PruneError, match="expected 4"):
        allocate(flat(2, 3), flat(2, 8), [mha(0), mha(1)], sparsity=0.5)

    with pytest.raises(PruneError, match="cover 1 layers but the model has 2"):
        allocate(flat(1, 4), flat(1, 8), [mha(0), mha(1)], sparsity=0.5)


class TestKeepCounts:
    def test_it_matches_what_allocate_keeps(self) -> None:
        groups = [mha(index, num_heads=8, intermediate=64) for index in range(4)]
        for allocation in ("uniform", "log-increase"):
            plan = allocate(flat(4, 8), flat(4, 64), groups, sparsity=0.3, allocation=allocation)

            heads, channels = keep_counts(groups, sparsity=0.3, allocation=allocation)

            assert heads == [len(layer.heads) for layer in plan]
            assert channels == [len(layer.channels) for layer in plan]

    def test_global_needs_scores(self) -> None:
        groups = [mha(0), mha(1)]

        with pytest.raises(PruneError, match="scores"):
            keep_counts(groups, sparsity=0.25, allocation="global")
