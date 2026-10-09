"""The final grid: its configs, the twin lock, the status logic and the job script.

No Slurm and no GPU. The job script runs under bash with fake ``squeue``,
``scancel`` and ``nvidia-smi`` on PATH and, where stages really run, a fake
``mnlp-eval`` that writes what each stage would.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]


def _load_grid() -> Any:
    spec = importlib.util.spec_from_file_location("grid", REPO / "scripts" / "grid.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolve string annotations through sys.modules.
    sys.modules["grid"] = module
    spec.loader.exec_module(module)
    return module


grid = _load_grid()
MODELS = grid.build_grid()
BY_NAME = {model.name: model for model in MODELS}


# --------------------------------------------------------------------------
# The grid and its configs
# --------------------------------------------------------------------------


class TestGrid:
    def test_192_models_with_unique_names(self) -> None:
        assert len(MODELS) == 192
        assert len(BY_NAME) == 192

    def test_names_follow_the_scheme(self) -> None:
        pattern = re.compile(
            r"^alma-7b-(slimgpt|flap)(20|30|40)-(ref|gen)-"
            r"(multi|pair-(cs|de|is|ru|zh)|dir-(cs|de|is|ru|zh)-en|dir-en-(cs|de|is|ru|zh))$"
        )
        assert all(pattern.match(name) for name in BY_NAME)
        for example in (
            "alma-7b-slimgpt30-gen-dir-en-is",
            "alma-7b-flap20-ref-multi",
            "alma-7b-slimgpt40-ref-pair-zh",
        ):
            assert example in BY_NAME

    def test_every_axis_is_balanced(self) -> None:
        def count(attribute: str) -> dict[Any, int]:
            found: dict[Any, int] = {}
            for model in MODELS:
                value = getattr(model, attribute)
                found[value] = found.get(value, 0) + 1
            return found

        assert count("method") == {"slimgpt": 96, "flap": 96}
        assert count("tag") == {"ref": 96, "gen": 96}
        assert count("sparsity") == {0.2: 64, 0.3: 64, 0.4: 64}
        assert count("scope") == {"multi": 12, "pair": 60, "dir": 120}

    def test_method_sets_allocation_and_tag_sets_text(self) -> None:
        for model in MODELS:
            spec = model.prune_spec()
            assert spec["allocation"] == {"slimgpt": "log-increase", "flap": "al-am"}[model.method]
            assert (
                spec["calibration_text"]
                == {"ref": "prompt+target", "gen": "prompt+generated"}[model.tag]
            )

    def test_common_fields(self) -> None:
        for model in MODELS:
            spec = model.prune_spec()
            assert spec["model_name_or_path"] == "/scratch-shared/scur0560/models/alma-7b-bf16"
            assert spec["dtype"] == "bfloat16"
            assert spec["batch_size"] == 4
            assert spec["max_length"] == 768
            assert spec["seed"] == 1234
            assert spec["name"] == model.name
            assert spec["sparsity"] == model.sparsity

    def test_calibration_and_directions_per_scope(self) -> None:
        suffix = {"ref": "", "gen": "-generated"}
        for model in MODELS:
            spec = model.prune_spec()
            if model.scope == "multi":
                assert spec["calibration"] == f"data/calibration/multi-10dir{suffix[model.tag]}"
                # Unset, so the descriptor says multi rather than a list of ten.
                assert "directions" not in spec
                assert len(model.pruned_directions) == 10
                assert model.pruned_for == "multi"
            elif model.scope == "pair":
                lang = model.scope_key
                assert spec["calibration"] == f"data/calibration/pair-{lang}-640{suffix[model.tag]}"
                assert spec["directions"] == sorted([f"{lang}-en", f"en-{lang}"])
                assert model.pruned_for == ",".join(sorted([f"{lang}-en", f"en-{lang}"]))
            else:
                direction = model.scope_key
                assert spec["calibration"] == (
                    f"data/calibration/dir-{direction}-1280{suffix[model.tag]}"
                )
                assert spec["directions"] == [direction]
                assert model.pruned_for == direction

    def test_priority_puts_multi_then_pair_then_dir_and_low_sparsity_first(self) -> None:
        assert [model.priority for model in MODELS] == list(range(192))
        scopes = [model.scope for model in MODELS]
        assert scopes == ["multi"] * 12 + ["pair"] * 60 + ["dir"] * 120
        for scope in grid.SCOPES:
            sparsities = [model.sparsity for model in MODELS if model.scope == scope]
            assert sparsities == sorted(sparsities)

    def test_specs_pass_prune_spec(self) -> None:
        warnings = grid.validate_specs(model.prune_spec() for model in MODELS)
        # Only an allocation another branch has not registered may be excused.
        assert all("allocation" in warning for warning in warnings)

    def test_written_configs_and_manifest(self, tmp_path: Path) -> None:
        written = grid.write_configs(tmp_path, MODELS)
        assert len(written) == 192
        for model in MODELS:
            text = (tmp_path / model.prune_config).read_text(encoding="utf-8")
            first, *rest = text.splitlines()
            assert first.startswith("# ")
            assert not any(line.startswith("#") for line in rest)
            assert yaml.safe_load(text) == model.prune_spec()

        manifest = json.loads((tmp_path / grid.MANIFEST).read_text(encoding="utf-8"))
        assert manifest["count"] == 192
        entries = manifest["models"]
        assert [entry["priority"] for entry in entries] == list(range(192))
        required = {
            "name",
            "method",
            "allocation",
            "calibration_text",
            "tag",
            "sparsity",
            "scope",
            "scope_key",
            "pruned_directions",
            "prune_config",
            "model_config",
            "suite",
            "priority",
        }
        for entry in entries:
            assert required <= set(entry)
            assert entry["model_config"] == f"configs/models/{entry['name']}.yaml"
            assert entry["prune_config"] == f"configs/prune/grid/{entry['name']}.yaml"
            assert entry["suite"] == "configs/suites/alma10-greedy-300.yaml"
            assert entry["scope_key"] in {"multi", *grid.LANGUAGES, *grid.DIRECTIONS}

    def test_repair_spec_is_alma_recipe_defaults(self) -> None:
        from mnlp_eval.prune.repair import RepairSpec

        spec = grid.repair_spec("alma-7b-slimgpt20-ref-multi")
        assert spec == {
            "name": "alma-7b-slimgpt20-ref-multi-lora",
            "source": "configs/models/alma-7b-slimgpt20-ref-multi.yaml",
            "data": "data/calibration/repair-multi-clean",
        }
        assert RepairSpec.from_dict(spec).rank == 16


# --------------------------------------------------------------------------
# Submission
# --------------------------------------------------------------------------


def _target(name: str = "alma-7b-flap20-ref-multi", mode: str = "prune") -> Any:
    config = f"configs/prune/grid/{name}.yaml" if mode == "prune" else f"configs/repair/{name}"
    return grid.Target(name, mode, "multi", config, f"configs/models/{name}.yaml", 0)


class TestSbatchCommand:
    def test_twins_differ_only_in_partition_and_cores(self, tmp_path: Path) -> None:
        commands = {
            partition: grid.sbatch_command(
                _target(), partition, "03:00:00", state=tmp_path, prune_out="/scratch/x"
            )
            for partition in ("gpu_a100", "gpu_h100")
        }
        a100, h100 = commands["gpu_a100"], commands["gpu_h100"]
        assert "--cpus-per-task=18" in a100
        assert "--cpus-per-task=16" in h100
        for command in (a100, h100):
            assert "--job-name=g-alma-7b-flap20-ref-multi" in command
            assert "--gpus=1" in command
            assert "--nodes=1" in command
            assert command[-1] == "slurm/grid_pipeline.sbatch"
        differing = {item for item in a100 if item not in h100}
        assert differing == {"--partition=gpu_a100", "--cpus-per-task=18"}

    def test_export_carries_every_variable_without_stray_commas(self, tmp_path: Path) -> None:
        command = grid.sbatch_command(
            _target(), "gpu_a100", "03:00:00", state=tmp_path, prune_out="/scratch/x"
        )
        export = next(item for item in command if item.startswith("--export="))
        assignments = export.removeprefix("--export=").split(",")
        assert assignments[0] == "ALL"
        values = dict(item.split("=", 1) for item in assignments[1:])
        assert values["GRID_NAME"] == "alma-7b-flap20-ref-multi"
        assert values["GRID_MODE"] == "prune"
        assert values["PRUNE_CONFIG"] == "configs/prune/grid/alma-7b-flap20-ref-multi.yaml"
        assert values["MODEL_CONFIG"] == "configs/models/alma-7b-flap20-ref-multi.yaml"
        assert values["SUITE_CONFIG"] == "configs/suites/alma10-greedy-300.yaml"
        assert values["GRID_DRY_RUN"] == "0"

    def test_repair_passes_its_config_as_repair_config(self, tmp_path: Path) -> None:
        command = grid.sbatch_command(
            _target("x-lora", "repair"), "gpu_h100", "10:00:00", state=tmp_path, prune_out="/o"
        )
        export = next(item for item in command if item.startswith("--export="))
        assert "REPAIR_CONFIG=configs/repair/x-lora" in export
        assert "PRUNE_CONFIG" not in export

    def test_unknown_partition_is_refused(self) -> None:
        assert grid.parse_partitions("gpu_a100,gpu_h100") == ["gpu_a100", "gpu_h100"]
        with pytest.raises(grid.GridError):
            grid.parse_partitions("gpu_a100,gpu_mig")

    def test_select_filters_and_keeps_priority_order(self) -> None:
        targets = grid.grid_targets()
        chosen = grid.select(targets, match=r"-gen-pair-", scope="pair")
        assert len(chosen) == 30
        assert [t.priority for t in chosen] == sorted(t.priority for t in chosen)
        only = grid.select(targets, only=["alma-7b-flap40-gen-dir-zh-en"])
        assert [t.name for t in only] == ["alma-7b-flap40-gen-dir-zh-en"]
        with pytest.raises(grid.GridError):
            grid.select(targets, only=["alma-7b-nope"])


# --------------------------------------------------------------------------
# Queue parsing and status
# --------------------------------------------------------------------------


class TestQueue:
    def test_parse_squeue_with_and_without_a_node(self) -> None:
        text = (
            "101 g-alma-7b-flap20-ref-multi PENDING gpu_a100 \n"
            "102 g-alma-7b-flap20-ref-multi RUNNING gpu_h100 gcn12\n"
            "103 calgen-pair-de RUNNING gpu_a100 gcn3\n"
            "104 g-alma-7b-flap30-ref-multi COMPLETED gpu_a100 gcn4\n"
        )
        jobs = grid.parse_squeue(text)
        assert jobs[0] == grid.QueuedJob("101", "g-alma-7b-flap20-ref-multi", "PENDING", "gpu_a100")
        assert jobs[1].node == "gcn12"
        grouped = grid.queued_by_target(jobs)
        # Not a grid job, and a finished one, are both left out.
        assert set(grouped) == {"alma-7b-flap20-ref-multi"}
        assert [job.job_id for job in grouped["alma-7b-flap20-ref-multi"]] == ["101", "102"]

    @pytest.mark.parametrize(
        ("returncode", "stdout", "stderr", "expected"),
        [
            (0, "RUNNING\n", "", "alive"),
            (0, "PENDING\n", "", "alive"),
            (0, "COMPLETING\n", "", "alive"),
            (0, "COMPLETED\n", "", "dead"),
            (0, "CANCELLED\n", "", "dead"),
            (0, "", "", "dead"),
            (1, "", "slurm_load_jobs error: Invalid job id specified", "dead"),
            (1, "", "slurm_load_jobs error: Socket timed out on send/recv operation", "unknown"),
        ],
    )
    def test_squeue_liveness(
        self, returncode: int, stdout: str, stderr: str, expected: str
    ) -> None:
        assert grid.squeue_liveness(returncode, stdout, stderr) == expected

    def test_parse_sacct(self) -> None:
        text = "201|TIMEOUT|03:00:12|0:0\n202|CANCELLED by 1234|00:00:05|0:15\n"
        parsed = grid.parse_sacct(text)
        assert parsed["201"] == {"state": "TIMEOUT", "elapsed": "03:00:12", "exit_code": "0:0"}
        assert parsed["202"]["state"] == "CANCELLED"


def _job(state: str, job_id: str = "7") -> Any:
    return grid.QueuedJob(job_id, "g-m", state, "gpu_a100", "gcn1" if state == "RUNNING" else "")


class TestClassify:
    def test_done_wins(self) -> None:
        assert grid.classify({"state": "done"}, [_job("PENDING")], True)[0] == "done"

    def test_running_and_pending_come_from_the_queue(self) -> None:
        assert grid.classify(None, [_job("PENDING"), _job("RUNNING", "8")], True)[0] == "running"
        assert grid.classify({"state": "failed"}, [_job("PENDING")], True)[0] == "pending"

    def test_failed(self) -> None:
        category, detail = grid.classify({"state": "failed", "stage": "generate"}, [], True)
        assert category == "failed"
        assert "generate" in detail

    def test_running_without_a_job_is_stale(self) -> None:
        category, detail = grid.classify({"state": "running", "job_id": "9"}, [], True)
        assert category == "failed"
        assert detail.startswith("stale")

    def test_submitted_but_vanished_is_lost(self) -> None:
        category, detail = grid.classify(None, [], True)
        assert category == "failed"
        assert detail.startswith("lost")

    def test_never_submitted(self) -> None:
        assert grid.classify(None, [], False)[0] == "not-submitted"

    def test_a_dry_run_job_does_not_count_as_done(self) -> None:
        assert grid.classify({"state": "done", "dry_run": True}, [], True)[0] == "failed"
        assert grid.classify({"state": "done", "dry_run": True}, [], False)[0] == "not-submitted"


class TestMark:
    def test_a_new_attempt_keeps_stage_times_and_records_the_old_one(self, tmp_path: Path) -> None:
        env = {"SLURM_JOB_ID": "1", "SLURM_JOB_PARTITION": "gpu_a100", "GRID_GPU": "A100"}
        grid.mark(
            "m", state="running", stage="prune", seconds={"prune": 61.0}, env=env, root=tmp_path
        )
        grid.mark("m", state="failed", stage="generate", exit_code=1, env=env, root=tmp_path)
        env2 = {**env, "SLURM_JOB_ID": "2", "SLURM_JOB_PARTITION": "gpu_h100"}
        status = grid.mark("m", state="running", stage="start", env=env2, root=tmp_path)
        assert status["job_id"] == "2"
        assert status["partition"] == "gpu_h100"
        assert status["stage_seconds"] == {"prune": 61.0}
        assert status["attempts"] == [
            {"job_id": "1", "state": "failed", "stage": "generate", "exit_code": 1}
        ]
        assert status["exit_code"] is None

    def test_failure_keeps_the_end_of_the_log(self, tmp_path: Path) -> None:
        log = tmp_path / "job.out"
        log.write_text("".join(f"line {i}\n" for i in range(100)) + "Traceback: boom\n")
        env = {"SLURM_JOB_ID": "3", "GRID_LOG": str(log)}
        status = grid.mark("m", state="failed", stage="prune", env=env, root=tmp_path)
        assert status["error"][-1] == "Traceback: boom"
        assert len(status["error"]) == 25


# --------------------------------------------------------------------------
# The twin lock
# --------------------------------------------------------------------------


class TestLockDecision:
    def test_owner_alive_means_yield(self) -> None:
        assert grid.lock_decision({"job_id": "5"}, "6", "alive", 1000.0) == "yield"

    def test_owner_unknown_means_yield(self) -> None:
        assert grid.lock_decision({"job_id": "5"}, "6", "unknown", 1000.0) == "yield"

    def test_owner_dead_means_takeover(self) -> None:
        assert grid.lock_decision({"job_id": "5"}, "6", "dead", 1.0) == "takeover"

    def test_own_lock_after_requeue_means_takeover(self) -> None:
        assert grid.lock_decision({"job_id": "6"}, "6", None, 1.0) == "takeover"

    def test_young_lock_without_owner_means_wait(self) -> None:
        assert grid.lock_decision(None, "6", None, 3.0) == "wait"
        assert grid.lock_decision(None, "6", None, grid.OWNER_GRACE + 1) == "takeover"


class FakeClock:
    def __init__(self) -> None:
        self.offset = 0.0
        self.sleeps: list[float] = []

    def now(self) -> float:
        import time

        return time.time() + self.offset

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.offset += seconds


class TestAcquireLock:
    def test_first_job_acquires_and_records_itself(self, tmp_path: Path) -> None:
        outcome, _ = grid.acquire_lock("m", "11", info={"partition": "gpu_a100"}, root=tmp_path)
        assert outcome == "acquired"
        owner = json.loads((tmp_path / "locks" / "m" / "owner.json").read_text())
        assert owner["job_id"] == "11"
        assert owner["partition"] == "gpu_a100"

    def test_twin_yields_to_a_live_owner(self, tmp_path: Path) -> None:
        grid.acquire_lock("m", "11", root=tmp_path)
        outcome, message = grid.acquire_lock(
            "m", "12", root=tmp_path, liveness=lambda _job: "alive"
        )
        assert outcome == "held"
        assert "11" in message
        assert grid._owner_id(tmp_path / "locks" / "m") == "11"

    def test_stale_lock_is_taken_over(self, tmp_path: Path) -> None:
        grid.acquire_lock("m", "11", root=tmp_path)
        outcome, message = grid.acquire_lock("m", "12", root=tmp_path, liveness=lambda _job: "dead")
        assert outcome == "acquired"
        assert "stale" in message
        assert grid._owner_id(tmp_path / "locks" / "m") == "12"
        # Nothing of the old lock is left behind.
        assert sorted(path.name for path in (tmp_path / "locks").iterdir()) == ["m"]

    def test_lock_without_owner_is_waited_for_then_taken(self, tmp_path: Path) -> None:
        (tmp_path / "locks" / "m").mkdir(parents=True)
        clock = FakeClock()
        outcome, _ = grid.acquire_lock(
            "m", "12", root=tmp_path, sleep=clock.sleep, now=clock.now, liveness=lambda _job: "dead"
        )
        assert outcome == "acquired"
        assert sum(clock.sleeps) >= grid.OWNER_GRACE

    def test_takeover_refuses_a_lock_whose_owner_changed(self, tmp_path: Path) -> None:
        grid.acquire_lock("m", "13", root=tmp_path)
        lock = tmp_path / "locks" / "m"
        # Judged stale while job 11 owned it; job 13 has taken it since.
        assert not grid._take_over(lock, "11", "12", FakeClock().now)
        assert grid._owner_id(lock) == "13"

    def test_only_the_owner_releases(self, tmp_path: Path) -> None:
        grid.acquire_lock("m", "11", root=tmp_path)
        assert not grid.release_lock("m", "12", root=tmp_path)
        assert (tmp_path / "locks" / "m").is_dir()
        assert grid.release_lock("m", "11", root=tmp_path)
        assert not (tmp_path / "locks" / "m").exists()


# --------------------------------------------------------------------------
# Stage completeness
# --------------------------------------------------------------------------


def _checkpoint(path: Path, kind: str = "prune", *, shards: int = 2) -> None:
    path.mkdir(parents=True, exist_ok=True)
    own = "calibration_provenance.json" if kind == "prune" else "repair.json"
    for name in ("config.json", "subnetwork.json", own):
        (path / name).write_text("{}")
    names = [f"model-0000{i + 1}-of-0000{shards}.safetensors" for i in range(shards)]
    for name in names:
        (path / name).write_bytes(b"x")
    weight_map = {f"w{i}": name for i, name in enumerate(names)}
    (path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))


class TestCompleteness:
    def test_sharded_checkpoint_needs_every_shard(self, tmp_path: Path) -> None:
        _checkpoint(tmp_path / "c")
        assert grid.checkpoint_complete(tmp_path / "c", "prune")[0]
        (tmp_path / "c" / "model-00002-of-00002.safetensors").unlink()
        ok, reason = grid.checkpoint_complete(tmp_path / "c", "prune")
        assert not ok
        assert "model-00002" in reason

    def test_unsharded_checkpoint(self, tmp_path: Path) -> None:
        path = tmp_path / "c"
        _checkpoint(path)
        for shard in path.glob("model-*"):
            shard.unlink()
        (path / "model.safetensors.index.json").unlink()
        assert not grid.checkpoint_complete(path, "prune")[0]
        (path / "model.safetensors").write_bytes(b"x")
        assert grid.checkpoint_complete(path, "prune")[0]

    def test_repair_needs_repair_json(self, tmp_path: Path) -> None:
        _checkpoint(tmp_path / "c", "prune")
        assert not grid.checkpoint_complete(tmp_path / "c", "repair")[0]

    def test_stage_needs_the_cli_manifest(self, tmp_path: Path) -> None:
        _checkpoint(tmp_path / "c")
        config = tmp_path / "m.yaml"
        config.write_text("name: m\n")
        subnetwork = tmp_path / "s.json"
        subnetwork.write_text("{}")
        marker = tmp_path / "m.prune.json"
        arguments = {
            "checkpoint": tmp_path / "c",
            "model_config": config,
            "marker": marker,
            "subnetwork": subnetwork,
        }
        # Everything written but prune exited non-zero: not complete.
        assert not grid.stage_complete("prune", **arguments)[0]
        marker.write_text("{}")
        assert grid.stage_complete("prune", **arguments)[0]

    def test_scores_state(self, tmp_path: Path) -> None:
        (tmp_path / "hyps").mkdir()
        hyps = tmp_path / "hyps" / "de-en.jsonl"
        hyps.write_text("{}\n")
        assert grid.scores_state(tmp_path, "surface")[0] == 1
        scores = tmp_path / "scores.surface.json"
        scores.write_text("{}")
        os.utime(scores, (hyps.stat().st_mtime + 10,) * 2)
        assert grid.scores_state(tmp_path, "surface")[0] == 0
        os.utime(scores, (hyps.stat().st_mtime - 10,) * 2)
        assert grid.scores_state(tmp_path, "surface")[0] == 2

    def test_generation_complete(self, tmp_path: Path) -> None:
        directions = ["de-en", "en-de"]
        (tmp_path / "manifest.json").write_text(
            json.dumps({"suite": {"data": {"directions": directions}}})
        )
        (tmp_path / "hyps").mkdir()
        (tmp_path / "hyps" / "de-en.jsonl").write_text("{}\n")
        assert not grid.generation_complete(tmp_path)[0]
        (tmp_path / "hyps" / "en-de.jsonl").write_text("{}\n")
        (tmp_path / "stages").mkdir()
        record = {"status": "completed", "directions": {d: {} for d in directions}}
        (tmp_path / "stages" / "generate.json").write_text(json.dumps(record))
        assert grid.generation_complete(tmp_path)[0]

    def test_last_json_object_skips_progress_output(self) -> None:
        text = 'loading\n{"not": "this"}\nmore\n{\n  "name": "m",\n  "unit_sparsity": 0.3\n}\n'
        assert grid.last_json_object(text) == {"name": "m", "unit_sparsity": 0.3}
        assert grid.last_json_object("no json here\n") is None


class TestAdopt:
    def test_an_outside_checkpoint_becomes_complete_for_the_job(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GRID_STATE_DIR", str(tmp_path / "state"))
        target = next(t for t in grid.grid_targets() if t.name == NAME)
        out = tmp_path / "checkpoints"
        with pytest.raises(grid.GridError):
            grid.adopt(target, root=tmp_path, prune_out=out)
        _checkpoint(out / NAME)
        (tmp_path / "configs" / "models").mkdir(parents=True)
        (tmp_path / target.model_config).write_text("name: x\n")
        (tmp_path / "subnetworks").mkdir()
        (tmp_path / "subnetworks" / f"{NAME}.json").write_text("{}")

        notes = grid.adopt(target, root=tmp_path, prune_out=out)

        assert "not resolved" in notes[-1]
        marker = tmp_path / "state" / "status" / f"{NAME}.prune.json"
        assert json.loads(marker.read_text())["adopted"] is True
        assert grid.stage_complete(
            "prune",
            checkpoint=out / NAME,
            model_config=tmp_path / target.model_config,
            marker=marker,
            subnetwork=tmp_path / "subnetworks" / f"{NAME}.json",
        )[0]


# --------------------------------------------------------------------------
# slurm/grid_pipeline.sbatch under bash
# --------------------------------------------------------------------------

NAME = "alma-7b-slimgpt30-gen-dir-en-is"

FAKE_SQUEUE = """#!/usr/bin/env bash
# Jobs in FAKE_ALIVE are RUNNING; any other id is unknown to Slurm.
echo "squeue $*" >> "$FAKE_CALLS"
job=""
while (( $# )); do [[ $1 == -j ]] && job=$2; shift; done
for alive in ${FAKE_ALIVE:-}; do
    if [[ $job == "$alive" ]]; then echo RUNNING; exit 0; fi
done
echo "slurm_load_jobs error: Invalid job id specified" >&2
exit 1
"""

FAKE_CLI = """#!{python}
# Writes what each mnlp-eval stage would, or fails where FAKE_FAIL says.
import json, os, sys
from pathlib import Path

args = sys.argv[1:]
command, options, index = args[0], {{}}, 1
while index < len(args):
    if index + 1 < len(args) and not args[index + 1].startswith("--"):
        options[args[index][2:]] = args[index + 1]
        index += 2
    else:
        options[args[index][2:]] = True
        index += 1
with open(os.environ["FAKE_CALLS"], "a") as handle:
    handle.write(" ".join(["cli", *args]) + "\\n")
stage = command + (":" + options["groups"] if command == "score" else "")
if stage in os.environ.get("FAKE_FAIL", "").split(","):
    print("Traceback: fake failure in " + stage, file=sys.stderr)
    sys.exit(1)
if stage in os.environ.get("FAKE_SILENT", "").split(","):
    sys.exit(0)
name = os.environ["GRID_NAME"]
run = Path(os.environ["FAKE_RUN_DIR"])
if command == "prune":
    out = Path(options["out"]) / name
    out.mkdir(parents=True, exist_ok=True)
    for file in ("config.json", "subnetwork.json", "calibration_provenance.json"):
        (out / file).write_text("{{}}")
    (out / "model.safetensors").write_bytes(b"x")
    Path(options["subnetwork-dir"]).mkdir(parents=True, exist_ok=True)
    (Path(options["subnetwork-dir"]) / (name + ".json")).write_text("{{}}")
    Path(options["model-config-dir"]).mkdir(parents=True, exist_ok=True)
    (Path(options["model-config-dir"]) / (name + ".yaml")).write_text("name: " + name)
    print("selecting heads")
    print(json.dumps({{"name": name, "unit_sparsity": 0.3}}, indent=2))
elif command == "run-dir":
    print(run)
elif command == "generate":
    directions = ["de-en", "en-de"]
    for sub in ("hyps", "stages"):
        (run / sub).mkdir(parents=True, exist_ok=True)
    manifest = {{"suite": {{"data": {{"directions": directions}}}}}}
    (run / "manifest.json").write_text(json.dumps(manifest))
    for direction in directions:
        (run / "hyps" / (direction + ".jsonl")).write_text("{{}}\\n")
    record = {{"status": "completed", "directions": dict.fromkeys(directions, {{}})}}
    (run / "stages" / "generate.json").write_text(json.dumps(record))
elif command == "score":
    (Path(options["run"]) / ("scores." + options["groups"] + ".json")).write_text("{{}}")
"""


class Pipeline:
    """Runs the job script as Slurm would, against fakes in a tmp dir."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.bin = root / "bin"
        self.bin.mkdir()
        tools = {
            "squeue": FAKE_SQUEUE,
            "scancel": '#!/usr/bin/env bash\necho "scancel $*" >> "$FAKE_CALLS"\n',
            "nvidia-smi": '#!/usr/bin/env bash\n[[ $# -eq 0 ]] && echo table || echo "FAKE GPU"\n',
            "scontrol": "#!/usr/bin/env bash\nexit 1\n",
            "mnlp-eval": FAKE_CLI.format(python=sys.executable),
        }
        for name, text in tools.items():
            path = self.bin / name
            path.write_text(text)
            path.chmod(0o755)
        self.state = root / "state"
        self.calls = root / "calls"
        self.calls.touch()
        self.out = root / "checkpoints"
        self.model_config = root / "configs" / f"{NAME}.yaml"
        self.run_dir = root / "runs" / "run"

    def run(
        self, job_id: str, *, dry: bool = False, **extra: str
    ) -> subprocess.CompletedProcess[str]:
        log = self.root / f"job-{job_id}.out"
        env = {
            **os.environ,
            "PATH": f"{self.bin}:{os.environ['PATH']}",
            "SLURM_SUBMIT_DIR": str(REPO),
            "SLURM_JOB_ID": job_id,
            "SLURM_JOB_PARTITION": "gpu_a100",
            "SLURMD_NODENAME": "gcn1",
            # Slurm always sets it and the job script uses it under set -u.
            "USER": os.environ.get("USER", "scur0000"),
            "GRID_NAME": NAME,
            "GRID_MODE": "prune",
            "PRUNE_CONFIG": f"configs/prune/grid/{NAME}.yaml",
            "MODEL_CONFIG": str(self.model_config),
            "SUITE_CONFIG": "configs/suites/alma10-greedy-300.yaml",
            "GRID_DRY_RUN": "1" if dry else "0",
            "GRID_STATE_DIR": str(self.state),
            "GRID_LOG": str(log),
            "GRID_PYTHON": sys.executable,
            "GRID_CLI": str(self.bin / "mnlp-eval"),
            "GRID_COMET_CLI": str(self.bin / "mnlp-eval"),
            "GRID_SUBNETWORK_DIR": str(self.root / "subnetworks"),
            "GRID_SCRATCH_ROOT": str(self.root),
            "PRUNE_OUT": str(self.out),
            "FAKE_CALLS": str(self.calls),
            "FAKE_RUN_DIR": str(self.run_dir),
            # `module` is a shell function on Snellius; here it does nothing.
            "BASH_FUNC_module%%": "() {  return 0\n}",
            **extra,
        }
        with log.open("w") as handle:
            result = subprocess.run(
                ["bash", str(REPO / "slurm" / "grid_pipeline.sbatch")],
                cwd=REPO,
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        result.stdout = log.read_text()
        return result

    def status(self) -> dict[str, Any]:
        status: dict[str, Any] = json.loads((self.state / "status" / f"{NAME}.json").read_text())
        return status

    def lock(self, owner: str) -> None:
        lock = self.state / "locks" / NAME
        lock.mkdir(parents=True)
        (lock / "owner.json").write_text(json.dumps({"job_id": owner}))

    def cli_calls(self) -> list[str]:
        return [line for line in self.calls.read_text().splitlines() if line.startswith("cli ")]


@pytest.fixture
def pipeline(tmp_path: Path) -> Pipeline:
    if shutil.which("bash") is None:
        pytest.skip("needs bash")
    return Pipeline(tmp_path)


class TestPipelineScript:
    def test_syntax(self) -> None:
        subprocess.run(["bash", "-n", str(REPO / "slurm" / "grid_pipeline.sbatch")], check=True)

    def test_dry_run_takes_the_lock_cancels_the_twin_and_finishes(self, pipeline: Pipeline) -> None:
        result = pipeline.run("100", dry=True)
        assert result.returncode == 0, result.stdout
        assert (
            f"scancel -u {os.environ.get('USER', 'scur0000')} --name=g-{NAME} --state=PENDING"
            in (pipeline.calls.read_text())
        )
        assert "[dry-run]" in result.stdout
        assert pipeline.cli_calls() == []
        status = pipeline.status()
        assert status["state"] == "done"
        assert status["gpu"] == "FAKE GPU"
        # Released on exit, and the log is indexed.
        assert not (pipeline.state / "locks" / NAME).exists()
        assert (pipeline.state / "logs" / f"{NAME}.out").is_symlink()

    def test_a_twin_exits_quietly_while_the_owner_lives(self, pipeline: Pipeline) -> None:
        pipeline.lock("200")
        result = pipeline.run("201", dry=True, FAKE_ALIVE="200")
        assert result.returncode == 0
        assert "holds the lock" in result.stdout
        assert "scancel" not in pipeline.calls.read_text()
        assert not (pipeline.state / "status" / f"{NAME}.json").exists()
        assert grid._owner_id(pipeline.state / "locks" / NAME) == "200"

    def test_a_stale_lock_is_taken_over(self, pipeline: Pipeline) -> None:
        pipeline.lock("300")
        result = pipeline.run("301", dry=True, FAKE_ALIVE="")
        assert result.returncode == 0, result.stdout
        assert "took over the stale lock of job 300" in result.stdout
        assert pipeline.status()["job_id"] == "301"
        assert not (pipeline.state / "locks" / NAME).exists()

    def test_stages_run_and_a_failure_resumes_where_it_stopped(self, pipeline: Pipeline) -> None:
        failed = pipeline.run("400", FAKE_FAIL="score:neural")
        assert failed.returncode != 0
        status = pipeline.status()
        assert status["state"] == "failed"
        assert status["stage"] == "score_neural"
        assert any("fake failure in score:neural" in line for line in status["error"])
        assert (pipeline.state / "status" / f"{NAME}.prune.json").is_file()
        assert not (pipeline.state / "locks" / NAME).exists()

        done = pipeline.run("401")
        assert done.returncode == 0, done.stdout
        status = pipeline.status()
        assert status["state"] == "done"
        assert status["skipped"] == ["prune", "generate", "score_surface"]
        assert status["run_dir"] == str(pipeline.run_dir)
        assert status["attempts"][-1]["job_id"] == "400"
        commands = [call.split()[1] for call in pipeline.cli_calls()]
        # The second job repeated only run-dir and the neural scoring.
        assert commands == [
            "prune",
            "run-dir",
            "generate",
            "score",
            "score",
            "run-dir",
            "score",
        ]
        manifest = json.loads((pipeline.state / "status" / f"{NAME}.prune.json").read_text())
        assert manifest == {"name": NAME, "unit_sparsity": 0.3}

    def test_a_rebuilt_model_does_not_inherit_old_hypotheses(self, pipeline: Pipeline) -> None:
        assert pipeline.run("450").returncode == 0
        # The checkpoint went (scratch purges it after 14 days), so prune runs again.
        shutil.rmtree(pipeline.out / NAME)
        result = pipeline.run("451")
        assert result.returncode == 0, result.stdout
        stale = pipeline.root / "runs-stale" / "run.451"
        assert (stale / "hyps" / "de-en.jsonl").is_file()
        assert pipeline.status()["skipped"] == []

    def test_a_failed_prune_leaves_no_marker(self, pipeline: Pipeline) -> None:
        result = pipeline.run("500", FAKE_FAIL="prune")
        assert result.returncode != 0
        assert pipeline.status()["stage"] == "prune"
        assert not (pipeline.state / "status" / f"{NAME}.prune.json").exists()

    def test_a_partial_checkpoint_is_removed_before_pruning(self, pipeline: Pipeline) -> None:
        stale = pipeline.out / NAME / "model-00003-of-00003.safetensors"
        stale.parent.mkdir(parents=True)
        stale.write_bytes(b"from an earlier attempt")
        result = pipeline.run("600")
        assert result.returncode == 0, result.stdout
        assert not stale.exists()
        assert (pipeline.out / NAME / "model.safetensors").is_file()

    def test_a_score_group_that_writes_nothing_fails(self, pipeline: Pipeline) -> None:
        result = pipeline.run("700", FAKE_SILENT="score:neural")
        assert result.returncode != 0
        assert pipeline.status()["stage"] == "score_neural"
        assert "wrote no scores.neural.json" in result.stdout
