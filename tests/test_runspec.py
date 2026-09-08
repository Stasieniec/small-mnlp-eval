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


def test_stage_records_live_outside_the_manifest(run_config: RunConfig) -> None:
    # The manifest is the run's identity and is written once. Merging stage
    # reports into it meant an unlocked read-modify-write, and concurrent
    # Slurm array tasks lost records in every measured trial.
    paths = RunPaths.for_config(run_config)
    manifest = paths.init_manifest(run_config)
    assert "stages" not in manifest

    assert not paths.stage_completed("bench")
    paths.record_stage("bench", {"status": "completed", "n": 4})
    paths.record_stage("score", {"status": "pending"})

    assert paths.stage_completed("bench")
    assert not paths.stage_completed("score")
    assert paths.stages()["bench"]["n"] == 4
    assert "recorded_at" in paths.stages()["bench"]
    # The manifest is untouched by stage recording.
    assert paths.read_manifest() == manifest


def test_sharded_stage_records_do_not_share_a_file(run_config: RunConfig) -> None:
    paths = RunPaths.for_config(run_config)
    paths.init_manifest(run_config)
    paths.record_stage("generate", {"status": "completed"}, key="de-en")
    paths.record_stage("generate", {"status": "pending"}, key="en-de")

    assert paths.stage_path("generate", "de-en") != paths.stage_path("generate", "en-de")
    assert set(paths.stage_records("generate")) == {"de-en", "en-de"}
    assert paths.stage_completed("generate", key="de-en")
    assert not paths.stage_completed("generate", key="en-de")
    assert paths.generated_directions() == {"de-en"}


def test_generate_is_incomplete_until_every_suite_direction_reports(
    model_spec: ModelSpec, local_testset: Path, tmp_path: Path
) -> None:
    # Checking only that the records present are completed would call a
    # half-finished shard set done, which is how a partially generated run got
    # scored and reported as whole.
    from mnlp_eval.config import SuiteSpec

    suite = SuiteSpec.from_dict(
        {
            "name": "two-way",
            "data": {
                "dataset": f"local:jsonl:{local_testset}",
                "directions": ["de-en", "en-de"],
            },
        }
    )
    config = RunConfig(model_spec, suite, output_root=tmp_path / "runs")
    paths = RunPaths.for_config(config)
    paths.init_manifest(config)
    assert paths.suite_directions() == ["de-en", "en-de"]

    paths.record_stage("generate", {"status": "completed"}, key="de-en")
    assert not paths.stage_completed("generate")

    paths.record_stage("generate", {"status": "completed"}, key="en-de")
    assert paths.stage_completed("generate")


def test_concurrent_stage_writers_do_not_lose_records(run_config: RunConfig) -> None:
    # The regression this design exists to prevent. With a single mutable
    # manifest this lost at least one record in 60 of 60 trials.
    import concurrent.futures

    paths = RunPaths.for_config(run_config)
    paths.init_manifest(run_config)
    stages = [f"generate:{index}" for index in range(12)]

    def write(name: str) -> None:
        stage, _, key = name.partition(":")
        paths.record_stage(stage, {"status": "completed"}, key=key)

    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
        list(pool.map(write, stages))

    assert len(paths.stage_records("generate")) == 12


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
