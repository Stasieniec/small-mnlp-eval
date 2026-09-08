"""Run directory layout and manifest handling."""

from __future__ import annotations

from pathlib import Path

import pytest

from mnlp_eval.config import ConfigError, ModelSpec, RunConfig
from mnlp_eval.runspec import RunPaths, discover_runs


def test_manifest_is_created_once_and_is_idempotent(run_config: RunConfig) -> None:
    paths = RunPaths.for_config(run_config)
    first = paths.init_manifest(run_config)
    second = paths.init_manifest(run_config)
    assert first == second
    assert first["run_id"] == run_config.run_id
    assert first["prompt_fingerprint"]


def test_manifest_refuses_to_adopt_a_different_run(
    run_config: RunConfig, model_spec: ModelSpec
) -> None:
    paths = RunPaths.for_config(run_config)
    paths.init_manifest(run_config)
    other = RunConfig(
        model=ModelSpec.from_dict(
            {**model_spec.__dict__, "dtype": "float16", "quantization": None, "adapter": None}
        ),
        suite=run_config.suite,
        output_root=run_config.output_root,
    )
    with pytest.raises(RuntimeError, match="Refusing to overwrite"):
        RunPaths(paths.root).init_manifest(other)


def test_manifest_round_trips_into_a_run_config(run_config: RunConfig) -> None:
    # The bench and score stages act on a run directory alone, which is all a
    # Slurm job chain has.
    paths = RunPaths.for_config(run_config)
    manifest = paths.init_manifest(run_config)
    restored = RunConfig.from_manifest(manifest, run_config.output_root)
    assert restored.run_id == run_config.run_id
    assert restored.model.name == run_config.model.name
    assert restored.suite.decode == run_config.suite.decode


def test_hand_edited_manifest_is_detected(run_config: RunConfig) -> None:
    paths = RunPaths.for_config(run_config)
    manifest = paths.init_manifest(run_config)
    manifest["model"]["dtype"] = "float16"
    with pytest.raises(ConfigError, match="does not match the identity"):
        RunConfig.from_manifest(manifest, run_config.output_root)


def test_stage_recording_accumulates(run_config: RunConfig) -> None:
    paths = RunPaths.for_config(run_config)
    paths.init_manifest(run_config)
    assert not paths.stage_completed("generate")
    paths.record_stage("generate", {"status": "completed", "n": 4})
    paths.record_stage("score", {"status": "pending"})
    paths.record_stage("generate", {"extra": True})
    manifest = paths.read_manifest()
    assert paths.stage_completed("generate")
    assert not paths.stage_completed("score")
    assert manifest["stages"]["generate"]["n"] == 4
    assert manifest["stages"]["generate"]["extra"] is True
    assert "recorded_at" in manifest["stages"]["generate"]


def test_missing_manifest_gives_an_actionable_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="mnlp-eval generate"):
        RunPaths(tmp_path).read_manifest()


def test_paths_are_derived_from_the_run_directory(run_config: RunConfig) -> None:
    paths = RunPaths.for_config(run_config)
    assert paths.manifest.name == "manifest.json"
    assert paths.hyps_jsonl("de-en").name == "de-en.jsonl"
    assert paths.hyps_text("de-en").name == "de-en.txt"
    assert paths.scores("surface").name == "scores.surface.json"
    assert paths.bench.name == "bench.json"


def test_discover_runs_skips_directories_without_a_manifest(
    run_config: RunConfig, tmp_path: Path
) -> None:
    paths = RunPaths.for_config(run_config)
    paths.init_manifest(run_config)
    (run_config.output_root / "not-a-run").mkdir(parents=True, exist_ok=True)
    found = [item.root.name for item in discover_runs(run_config.output_root)]
    assert found == [run_config.slug]
    assert list(discover_runs(tmp_path / "absent")) == []
