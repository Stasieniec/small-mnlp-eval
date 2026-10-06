"""The calibration pass and the FLAP criterion.

Statistics must ignore padding, or a pad activation the model would never see
there shrinks the variance the criterion reads. And the bias compensation must
recover the mean output it removed, which is why the method claims to work
without fine-tuning; if it does not, the bias apparatus is dead weight.

The ``al-am`` search is checked against what the official code does: squared
attention scores, standardisation within each layer and module, heads weighed
by parameter count, and a cut nearest the parameter budget.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from mnlp_eval.prune import PruneError
from mnlp_eval.prune.budget import ALLOCATIONS, METHOD_ALLOCATIONS, allocate
from mnlp_eval.prune.collect import collect_input_stats, load_calibration_prompts
from mnlp_eval.prune.compact import LayerPlan, compact_model, write_descriptor
from mnlp_eval.prune.groups import LayerGroups, describe_layers
from mnlp_eval.prune.methods import flap
from stubs import tiny_llama, tiny_qwen2, token_batches


@pytest.fixture
def calibration_set(tmp_path: Path) -> Path:
    root = tmp_path / "multi-2dir"
    root.mkdir()
    for direction in ("de-en", "en-de"):
        records = [
            {
                "id": index,
                "direction": direction,
                "source": "s",
                "target": "t",
                "prompt": f"Translate this from X to Y:\nX: segment {index}\nY:",
            }
            for index in range(4)
        ]
        (root / f"{direction}.jsonl").write_text(
            "\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8"
        )
    return root


def test_calibration_prompts_are_read_in_a_stable_order(calibration_set: Path) -> None:
    prompts = load_calibration_prompts(calibration_set)

    assert len(prompts) == 8
    assert prompts == load_calibration_prompts(calibration_set)
    assert all(prompt.startswith("Translate this") for prompt in prompts)


def test_a_pair_specific_run_reads_only_its_own_directions(calibration_set: Path) -> None:
    assert len(load_calibration_prompts(calibration_set, directions=["de-en"])) == 4

    with pytest.raises(PruneError, match=r"no calibration data for \['is-en'\]"):
        load_calibration_prompts(calibration_set, directions=["is-en"])


def test_a_set_whose_manifest_records_test_collisions_is_refused(calibration_set: Path) -> None:
    manifest = calibration_set / "calibration.json"
    manifest.write_text(
        json.dumps({"contamination": {"dataset": "wmt", "total_collisions": 0}}), encoding="utf-8"
    )
    assert len(load_calibration_prompts(calibration_set)) == 8

    manifest.write_text(
        json.dumps({"contamination": {"dataset": "wmt", "total_collisions": 2}}), encoding="utf-8"
    )
    with pytest.raises(PruneError, match="2 segment"):
        load_calibration_prompts(calibration_set)


def test_an_empty_calibration_directory_says_how_to_build_one(tmp_path: Path) -> None:
    with pytest.raises(PruneError, match="mnlp-eval calibration"):
        load_calibration_prompts(tmp_path)


class TestWithTorch:
    pytestmark = pytest.mark.torch

    def test_padded_positions_are_left_out_of_the_statistics(self) -> None:
        model = tiny_llama()

        unmasked = collect_input_stats(model, iter(token_batches(pad=0)))
        masked = collect_input_stats(model, iter(token_batches(pad=2)))

        projection = model.model.layers[0].mlp.down_proj
        # Three batches of two sequences: six tokens each unmasked, four masked.
        assert unmasked[projection].count == 3 * 2 * 6
        assert masked[projection].count == 3 * 2 * 4

    def test_every_prunable_projection_is_covered(self) -> None:
        model = tiny_llama()

        stats = collect_input_stats(model, iter(token_batches()))

        assert len(stats) == 2 * 2
        assert set(stats) == {
            projection
            for layer in model.model.layers
            for projection in (layer.self_attn.o_proj, layer.mlp.down_proj)
        }

    def test_a_channel_that_never_moves_scores_zero(self) -> None:
        import torch

        model = tiny_llama()
        groups = describe_layers(model)
        stats = collect_input_stats(model, iter(token_batches()))
        # Freeze one FFN channel's fluctuation. FLAP's claim is that a channel
        # which does not vary carries nothing a bias cannot replace.
        entry = stats[model.model.layers[0].mlp.down_proj]
        entry.m2 = entry.m2.clone()
        entry.m2[7] = 0.0

        _, channels = flap.score(model, stats, groups)

        assert channels[0][7] == 0.0
        assert channels[0][7] < max(channels[0])
        assert torch.tensor(channels[0]).min() >= 0.0

    def test_scores_have_one_entry_per_prunable_unit(self) -> None:
        model = tiny_llama()
        groups = describe_layers(model)

        heads, channels = flap.score(
            model, stats := collect_input_stats(model, iter(token_batches())), groups
        )

        assert stats
        assert [len(layer) for layer in heads] == [4, 4]
        assert [len(layer) for layer in channels] == [48, 48]

    def test_the_bias_recovers_the_mean_output_that_pruning_removed(self) -> None:
        import torch

        model = tiny_llama()
        groups = describe_layers(model)
        batches = token_batches()
        stats = collect_input_stats(model, iter(batches))
        heads, channels = flap.score(model, stats, groups)
        plan = allocate(heads, channels, groups, sparsity=0.5, allocation="uniform")

        projection = model.model.layers[0].mlp.down_proj
        entry = stats[projection]
        dense_mean = projection.weight.data.float() @ entry.mean

        plain = tiny_llama()
        compact_model(plain, plan)
        bare = plain.model.layers[0].mlp.down_proj
        kept = torch.as_tensor(plan[0].channels)
        without = bare.weight.data.float() @ entry.mean.index_select(0, kept)

        flap.compensate(model, stats, groups, plan)
        compact_model(model, plan)
        compensated = model.model.layers[0].mlp.down_proj
        with_bias = (
            compensated.weight.data.float() @ entry.mean.index_select(0, kept)
            + compensated.bias.data.float()
        )

        # The bias is exactly the mean contribution of the removed channels,
        # so the compensated layer reproduces the dense mean output.
        assert torch.allclose(with_bias, dense_mean, atol=1e-4)
        assert (without - dense_mean).abs().max() > (with_bias - dense_mean).abs().max()

    def test_the_whole_criterion_produces_a_loadable_checkpoint(self, tmp_path: Path) -> None:
        from recipes.pruned import load_model

        model = tiny_llama()
        groups = describe_layers(model)
        stats = collect_input_stats(model, iter(token_batches()))
        heads, channels = flap.score(model, stats, groups)
        plan = allocate(heads, channels, groups, sparsity=0.5, allocation="global")

        flap.compensate(model, stats, groups, plan)
        compact_model(model, plan)
        model.save_pretrained(tmp_path)
        subnetwork = write_descriptor(
            tmp_path / "subnetwork.json",
            plan,
            groups,
            name="flap50",
            base_model="test/model",
            method="FLAP",
        )

        loaded, _ = load_model(tmp_path)

        assert abs(subnetwork.overall_sparsity - 0.5) < 0.05
        # The compensation bias has to survive the round trip, or the loaded
        # model is a different function from the one that was pruned.
        assert loaded.model.layers[0].mlp.down_proj.bias is not None
        assert loaded.model.layers[0].self_attn.o_proj.bias is not None


def mha(index: int, num_heads: int = 4, head_dim: int = 8, intermediate: int = 48) -> LayerGroups:
    return LayerGroups(
        index=index,
        num_heads=num_heads,
        num_kv_heads=num_heads,
        head_dim=head_dim,
        intermediate=intermediate,
    )


def kept_cost(plan: Sequence[LayerPlan], groups: Sequence[LayerGroups]) -> int:
    """What a plan keeps, in the search's own weights."""
    total = 0
    for layer, group in zip(plan, groups, strict=True):
        head, channel = flap.unit_costs(group)
        total += head * len(layer.heads) + channel * len(layer.channels)
    return total


def dense_cost(groups: Sequence[LayerGroups]) -> int:
    return kept_cost(
        [LayerPlan(tuple(range(g.num_heads)), tuple(range(g.intermediate))) for g in groups],
        groups,
    )


def random_columns(groups: Sequence[LayerGroups], seed: int = 0) -> tuple[list[Any], list[Any]]:
    """Non-negative per-column WIFV, skewed like the real thing."""
    import torch

    generator = torch.Generator().manual_seed(seed)
    attention = [
        torch.rand(g.num_heads * g.head_dim, generator=generator).pow(3) * (index + 1)
        for index, g in enumerate(groups)
    ]
    ffn = [
        torch.rand(g.intermediate, generator=generator).pow(3) * 10**index
        for index, g in enumerate(groups)
    ]
    return attention, ffn


def test_al_am_is_flaps_own_allocation_not_a_budget() -> None:
    assert METHOD_ALLOCATIONS[flap.AL_AM] == "flap"
    assert flap.AL_AM not in ALLOCATIONS


def test_a_head_weighs_its_parameter_count_in_channels() -> None:
    # q, k, v and o hold hidden * head_dim each per head; gate, up and down
    # hidden each per channel. FLAP's code hardcodes 512/3 for head_dim 128.
    head, channel = flap.unit_costs(mha(0, head_dim=128))
    assert head / channel == pytest.approx(512 / 3)
    head, channel = flap.unit_costs(mha(0, head_dim=8))
    assert head / channel == pytest.approx(4 * 8 / 3)


class TestAlAm:
    pytestmark = pytest.mark.torch

    def test_removing_one_head_spends_512_thirds_channels_of_budget(self) -> None:
        import torch

        groups = [mha(0, num_heads=4, head_dim=128, intermediate=1000)]
        # Head 0 ranks last of everything. Channel i ranks below channel i + 1,
        # and all 999 of them below heads 1 to 3 and the outlier channel.
        attention = torch.cat([torch.zeros(128), torch.ones(384)])
        ffn = torch.cat([1 + torch.arange(999.0) * 1e-6, torch.tensor([1e4])])
        total = 4 * 512 + 1000 * 3

        plan = flap.search_structure([attention], [ffn], groups, sparsity=512 / total)
        # One head's worth of parameters is exactly that head, and no channel.
        assert plan[0].heads == (1, 2, 3)
        assert len(plan[0].channels) == 1000

        plan = flap.search_structure([attention], [ffn], groups, sparsity=(512 + 6 * 3) / total)
        assert plan[0].heads == (1, 2, 3)
        assert plan[0].channels == tuple(range(6, 1000))

    @pytest.mark.parametrize("sparsity", [0.1, 0.2, 0.3, 0.5])
    def test_the_cut_is_nearest_the_parameter_budget(self, sparsity: float) -> None:
        groups = [mha(index, num_heads=8, head_dim=16, intermediate=96) for index in range(4)]
        attention, ffn = random_columns(groups)

        plan = flap.search_structure(attention, ffn, groups, sparsity=sparsity)

        target = (1 - sparsity) * dense_cost(groups)
        head, _ = flap.unit_costs(groups[0])
        # Successive cuts differ by at most a head, so the nearest is within half.
        assert abs(kept_cost(plan, groups) - target) <= head / 2
        removed_heads = 1 - sum(len(layer.heads) for layer in plan) / 32
        removed_channels = 1 - sum(len(layer.channels) for layer in plan) / 384
        # One budget over both, so the two fractions are free to differ.
        assert removed_heads != pytest.approx(removed_channels)

    def test_standardisation_is_within_each_layer_and_module(self) -> None:
        groups = [mha(index) for index in range(3)]
        attention, ffn = random_columns(groups, seed=3)
        # Powers of two, so the rescaling is exact in floating point.
        scaled_attention = [attention[0], attention[1] * 1024, attention[2] / 1024]
        scaled_ffn = [ffn[0] * 1024, ffn[1], ffn[2] * 2**-20]

        plan = flap.search_structure(attention, ffn, groups, sparsity=0.4)

        assert flap.search_structure(scaled_attention, scaled_ffn, groups, sparsity=0.4) == plan

    def test_attention_columns_are_squared_before_standardising(self) -> None:
        import torch

        groups = [mha(0, num_heads=2, head_dim=2, intermediate=64)]
        # Head 0's columns are 0 and 10, head 1's both 6: head 1 has the higher
        # mean, head 0 the higher mean square. 63 channels just below their
        # mean and one far above put both heads near the bottom of the ranking.
        attention = torch.tensor([0.0, 10.0, 6.0, 6.0])
        ffn = torch.cat([1 + torch.arange(63.0) * 1e-3, torch.tensor([1000.0])])
        one_head = 8 / (2 * 8 + 64 * 3)

        plan = flap.search_structure([attention], [ffn], groups, sparsity=one_head)
        # Feeding square roots undoes the squaring, so this is the search on the
        # unsquared scores, which keeps both heads and drops channels instead.
        unsquared = flap.search_structure([attention.sqrt()], [ffn], groups, sparsity=one_head)

        assert plan[0].heads == (0,)
        assert len(plan[0].channels) == 64
        assert unsquared[0].heads == (0, 1)
        assert len(unsquared[0].channels) < 64

    def test_every_layer_keeps_a_head_and_a_channel(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        import torch

        groups = [mha(0), mha(1)]
        # Layer 0's heads are indistinguishable, so all score 0, below most
        # channels; layer 1 has a standout head. Uncorrected, the cut at 0.5
        # leaves layer 0 no heads at all.
        attention = [torch.ones(32), torch.cat([torch.full((8,), 10.0), torch.ones(24)])]
        ffn = [torch.cat([torch.zeros(4), 1 + torch.arange(44.0) * 1e-3]) for _ in range(2)]

        plan = flap.search_structure(attention, ffn, groups, sparsity=0.5)

        assert all(layer.heads and layer.channels for layer in plan)
        assert plan[0].heads == (0,)
        assert "layers [0] with no heads" in capsys.readouterr().err
        # The rest is re-cut, so the budget still holds to within a head.
        target = 0.5 * dense_cost(groups)
        assert abs(kept_cost(plan, groups) - target) <= flap.unit_costs(groups[0])[0] / 2

        # Past what the floor allows, only the floor is left.
        plan = flap.search_structure(attention, ffn, groups, sparsity=0.99)
        assert [(len(layer.heads), len(layer.channels)) for layer in plan] == [(1, 1), (1, 1)]

    def test_sparsity_zero_keeps_everything(self) -> None:
        groups = [mha(0), mha(1)]
        attention, ffn = random_columns(groups)

        plan = flap.search_structure(attention, ffn, groups, sparsity=0.0)

        assert kept_cost(plan, groups) == dense_cost(groups)

    def test_grouped_query_attention_is_refused(self) -> None:
        model = tiny_qwen2()

        with pytest.raises(PruneError, match="multi-head"):
            flap.al_am(model, {}, describe_layers(model), sparsity=0.2)

    def test_a_bad_input_is_refused(self) -> None:
        import torch

        groups = [mha(0)]
        attention, ffn = random_columns(groups)

        with pytest.raises(PruneError, match="fraction removed"):
            flap.search_structure(attention, ffn, groups, sparsity=1.0)
        with pytest.raises(PruneError, match="expected 32"):
            flap.search_structure([attention[0][:-1]], ffn, groups, sparsity=0.2)
        broken = ffn[0].clone()
        broken[3] = torch.nan
        with pytest.raises(PruneError, match="not finite"):
            flap.search_structure(attention, [broken], groups, sparsity=0.2)

    def test_the_search_compensates_compacts_and_reloads(self, tmp_path: Path) -> None:
        import torch
        from recipes.pruned import load_model

        model = tiny_llama()
        groups = describe_layers(model)
        stats = collect_input_stats(model, iter(token_batches()))
        plan = flap.al_am(model, stats, groups, sparsity=0.3)

        def projection(index: int, suffix: str) -> Any:
            layer = model.model.layers[index]
            return layer.self_attn.o_proj if suffix == "o_proj" else layer.mlp.down_proj

        kept = {
            (group.index, suffix): list(units)
            for group in groups
            for suffix, units in (
                ("o_proj", group.head_rows(plan[group.index].heads)),
                ("down_proj", plan[group.index].channels),
            )
        }
        dense = {
            key: projection(*key).weight.data.float() @ stats[projection(*key)].mean for key in kept
        }
        means = {key: stats[projection(*key)].mean for key in kept}

        flap.compensate(model, stats, groups, plan)
        compact_model(model, plan)
        dropped = {key for key, units in kept.items() if len(units) < len(means[key])}
        # The search removed heads and channels both, so both biases are tested.
        assert {suffix for _, suffix in dropped} == {"o_proj", "down_proj"}
        for key, units in kept.items():
            compacted = projection(*key)
            # A bias only where something was removed, and it restores the mean.
            assert (compacted.bias is not None) == (key in dropped)
            if key in dropped:
                output = compacted.weight.data.float() @ means[key][units] + compacted.bias.float()
                assert torch.allclose(output, dense[key], atol=1e-4)

        model.save_pretrained(tmp_path)
        write_descriptor(
            tmp_path / "subnetwork.json",
            plan,
            groups,
            name="flap-al-am30",
            base_model="test/model",
            method="FLAP",
        )
        loaded, _ = load_model(tmp_path)

        ids = torch.randint(0, 64, (2, 7), generator=torch.Generator().manual_seed(5))
        with torch.inference_mode():
            assert torch.allclose(loaded(ids).logits, model(ids).logits, atol=1e-5)
