"""Boundary protection must preserve weights and match actual compression."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from mnlp_eval.config import ConfigError, load_yaml_config
from mnlp_eval.experiments.layer_protection import (
    CONDITIONS,
    CONTROL,
    DENSE,
    budget_preview,
    submission_plan,
)
from mnlp_eval.prune import PruneError
from mnlp_eval.prune.budget import allocate, protected_indices
from mnlp_eval.prune.groups import LayerGroups, describe_layers
from mnlp_eval.prune.spec import PruneSpec, _select
from stubs import tiny_llama, tiny_qwen2, token_batches


def test_all_five_alma_conditions_match_the_legacy_control_budget() -> None:
    rows = budget_preview()
    assert len(rows) == 5
    for row in rows:
        assert row["removed_heads"] == 192
        assert row["removed_channels"] == 70464
        eligible = range(row["first"], 32 - row["last"])
        for key, width in (("heads_kept_by_layer", 32), ("channels_kept_by_layer", 11008)):
            counts = row[key]
            assert all(counts[i] == width for i in set(range(32)) - set(eligible))
            assert max(counts[i] for i in eligible) - min(counts[i] for i in eligible) <= 1
    for tag, protection in CONDITIONS.items():
        spec = PruneSpec.from_dict(
            load_yaml_config(f"configs/experiments/layer-protection/{tag}.yaml")
        )
        assert (spec.protect_first_n, spec.protect_last_n) == protection
        assert (spec.method, spec.allocation, spec.calibration_text, spec.sparsity) == (
            "slimgpt",
            "uniform",
            "prompt+target",
            0.2,
        )


@pytest.mark.parametrize("first,last", [(-1, 0), (1.5, 0), (True, 0), (0, "2")])
def test_invalid_counts_are_refused(first: Any, last: Any) -> None:
    with pytest.raises(PruneError, match="non-negative integers"):
        protected_indices(32, first, last)
    with pytest.raises(ConfigError, match="non-negative integer"):
        PruneSpec.from_dict(
            {
                "name": "test",
                "method": "slimgpt",
                "model_name_or_path": "m",
                "calibration": "c",
                "protect_first_n": first,
                "protect_last_n": last,
            }
        )


@pytest.mark.parametrize("allocation", ["uniform", "global", "log-increase"])
def test_zero_protection_preserves_existing_allocation(allocation: str) -> None:
    groups = [LayerGroups(i, 8, 8, 4, 16) for i in range(8)]
    heads = [[float(j) for j in range(8)] for _ in groups]
    channels = [[float(j) for j in range(16)] for _ in groups]
    plain = allocate(heads, channels, groups, sparsity=0.2, allocation=allocation)
    explicit = allocate(
        heads,
        channels,
        groups,
        sparsity=0.2,
        allocation=allocation,
        protect_first_n=0,
        protect_last_n=0,
    )
    assert plain == explicit
    protected = allocate(
        heads,
        channels,
        groups,
        sparsity=0.2,
        allocation=allocation,
        protect_first_n=1,
        protect_last_n=1,
    )
    assert protected[0].heads == protected[-1].heads == tuple(range(8))
    for attr in ("heads", "channels"):
        assert sum(len(getattr(p, attr)) for p in plain) == sum(
            len(getattr(p, attr)) for p in protected
        )


def test_impossible_protection_and_budget_are_rejected() -> None:
    with pytest.raises(PruneError, match="overlap"):
        protected_indices(4, 3, 2)
    groups = [LayerGroups(i, 4, 4, 4, 8) for i in range(4)]
    with pytest.raises(PruneError, match="insufficient capacity"):
        allocate([[1.0] * 4] * 4, [[1.0] * 8] * 4, groups, sparsity=0.5, protect_first_n=3)
    with pytest.raises(PruneError, match="insufficient capacity"):
        allocate([[1.0] * 4] * 4, [[1.0] * 8] * 4, groups, sparsity=0.2, protect_first_n=4)
    all_kept = allocate([[1.0] * 4] * 4, [[1.0] * 8] * 4, groups, sparsity=0, protect_first_n=4)
    assert all(len(p.heads) == 4 and len(p.channels) == 8 for p in all_kept)


def test_grouped_query_budget_remains_representable() -> None:
    groups = [LayerGroups(i, 8, 2, 4, 16) for i in range(4)]
    plan = allocate([[1.0] * 8] * 4, [[1.0] * 16] * 4, groups, sparsity=0.25, protect_first_n=1)
    assert sum(8 - len(p.heads) for p in plan) == 8
    for group, selected in zip(groups, plan, strict=True):
        group.validate_heads(selected.heads)


@pytest.mark.torch
@pytest.mark.parametrize("method", ["slimgpt", "flap", "llm-pruner"])
@pytest.mark.parametrize("build", [tiny_llama, tiny_qwen2])
def test_protected_weights_unchanged_and_checkpoint_reloads(
    method: str, build: Any, tmp_path: Path
) -> None:
    import torch
    from recipes.pruned import load_model

    from mnlp_eval.prune.compact import write_descriptor

    model = build(num_hidden_layers=4)
    groups = describe_layers(model)
    protected = [model.model.layers[i] for i in (0, 3)]
    before = [{k: v.clone() for k, v in layer.state_dict().items()} for layer in protected]
    spec = PruneSpec(
        name="protected",
        method=method,
        model_name_or_path="m",
        calibration="c",
        sparsity=0.25,
        protect_first_n=1,
        protect_last_n=1,
        dtype="float32",
    )
    batches = token_batches(count=4)
    plan = _select(spec, model, groups, lambda: batches)
    for layer, original in zip(protected, before, strict=True):
        assert layer.state_dict().keys() == original.keys()
        assert all(torch.equal(layer.state_dict()[key], value) for key, value in original.items())
    model.eval()
    with torch.inference_mode():
        expected = model(**batches[0]).logits.clone()
    model.save_pretrained(tmp_path)
    write_descriptor(
        tmp_path / "subnetwork.json",
        plan,
        groups,
        name="protected",
        base_model="test",
        method=method,
        protected_layers=(0, 3),
    )
    loaded, _ = load_model(tmp_path)
    with torch.inference_mode():
        torch.testing.assert_close(loaded(**batches[0]).logits, expected)
    assert sum(p.numel() for p in loaded.parameters()) < sum(
        p.numel() for p in build(num_hidden_layers=4).parameters()
    )


@pytest.mark.torch
def test_slimgpt_skips_protected_hessians_but_forwards_their_outputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    from mnlp_eval.prune.methods import slimgpt

    model = tiny_llama(num_hidden_layers=4)
    batches = token_batches()
    inputs = slimgpt._layer_inputs(model, batches)
    expected = slimgpt._advance(model.model.layers[0], inputs)[0][0]
    original = slimgpt._hessians
    seen = []

    def record(layer: Any, activations: Any) -> Any:
        assert layer is not model.model.layers[0] and layer is not model.model.layers[3]
        seen.append(activations[0][0].clone())
        return original(layer, activations)

    monkeypatch.setattr(slimgpt, "_hessians", record)
    slimgpt.prune(
        model, batches, describe_layers(model), sparsity=0.25, protect_first_n=1, protect_last_n=1
    )
    assert len(seen) == 4
    torch.testing.assert_close(seen[0], expected)
    torch.testing.assert_close(seen[2], expected)


def test_job_dag_waits_for_all_scores_and_benchmarks() -> None:
    systems = [DENSE, CONTROL, "alma-7b-boundary-first4"]
    jobs = submission_plan(systems)
    seen = set()
    for stage, system, deps in jobs:
        assert set(deps) <= seen
        seen.add(f"{stage}:{system}")
    assert set(jobs[-1][2]) == {
        f"{stage}:{system}" for stage in ("score", "bench") for system in systems
    }
    assert ("prune", DENSE, []) not in jobs
