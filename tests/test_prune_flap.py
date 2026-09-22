"""The calibration pass and the FLAP criterion.

Statistics must ignore padding, or a pad activation the model would never see
there shrinks the variance the criterion reads. And the bias compensation must
recover the mean output it removed, which is why the method claims to work
without fine-tuning; if it does not, the bias apparatus is dead weight.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mnlp_eval.prune import PruneError
from mnlp_eval.prune.budget import allocate
from mnlp_eval.prune.collect import collect_input_stats, load_calibration_prompts
from mnlp_eval.prune.compact import compact_model, write_descriptor
from mnlp_eval.prune.groups import describe_layers
from mnlp_eval.prune.methods import flap
from stubs import tiny_llama, token_batches


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
