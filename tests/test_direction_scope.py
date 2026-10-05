"""Scientific design and resource scheduling invariants for the scope pilot."""

import pytest

from mnlp_eval.experiments.direction_scope import matrix, submission_plan


def test_matched_budget_and_complete_controls():
    conditions = matrix(["de", "is"], 1280, "flap")
    scopes = {c["scope"]: c for c in conditions}
    assert len(conditions) == 7
    assert len({c["name"] for c in conditions}) == 7
    for condition in conditions:
        assert condition["segments_per_direction"] * len(condition["directions"]) == 1280
    assert len(scopes["multi"]["directions"]) == 10
    for pair in ("de", "is"):
        for direction in (f"{pair}-en", f"en-{pair}"):
            assert scopes[f"direction-{direction}"]["directions"] == [direction]
            assert direction in scopes[f"pair-{pair}"]["directions"]


@pytest.mark.parametrize(
    "pairs,budget", [([], 1280), (["de", "de"], 1280), (["fr"], 1280), (["de"], 1279), (["de"], 0)]
)
def test_invalid_design_rejected(pairs, budget):
    with pytest.raises(ValueError):
        matrix(pairs, budget, "flap")


@pytest.mark.parametrize("concurrency", [1, 2])
def test_dependency_lanes_bound_gpu_concurrency(concurrency):
    systems = ["dense", *[c["name"] for c in matrix(["de", "is"], 1280, "flap")]]
    remaining = dict(submission_plan(systems, concurrency))
    completed = set()
    while remaining:
        ready = [name for name, deps in remaining.items() if set(deps) <= completed]
        assert 0 < len(ready) <= concurrency
        completed.update(ready)
        for name in ready:
            del remaining[name]


def test_dry_run_requests_shared_single_gpu_and_does_not_submit(tmp_path, monkeypatch, capsys):
    import argparse

    from mnlp_eval.artifacts import atomic_write_json
    from mnlp_eval.experiments.direction_scope import submit

    atomic_write_json(
        tmp_path / "experiment.json",
        {
            "systems": ["alma-7b", "pair", "direction"],
            "input_sha256": {},
            "python": "/python",
            "hf_home": None,
        },
    )
    monkeypatch.setattr("subprocess.run", lambda *_a, **_k: pytest.fail("dry run submitted a job"))
    submit(argparse.Namespace(out=tmp_path, concurrency=2, dry_run=True))
    commands = capsys.readouterr().out.splitlines()
    assert len(commands) == 4
    assert all("--gpus 1" in c and "--exclusive" not in c for c in commands[:3])
    assert all("--time 02:00:00" in c for c in commands[:3])
    assert "--dependency afterok:1" in commands[2]
    assert "--dependency afterok:1:2:3" in commands[3]
    assert "--gpus" not in commands[3]
    assert not (tmp_path / "jobs.json").exists()


def test_partial_submission_records_ids_and_prevents_duplicates(tmp_path, monkeypatch):
    import argparse
    import subprocess

    from mnlp_eval.artifacts import atomic_write_json, read_json
    from mnlp_eval.experiments.direction_scope import submit

    atomic_write_json(
        tmp_path / "experiment.json",
        {
            "systems": ["alma-7b", "pair"],
            "input_sha256": {},
            "python": "/python",
            "hf_home": None,
        },
    )
    calls = []

    def queue(command, **kwargs):
        calls.append(command)
        if len(calls) > 1:
            raise subprocess.CalledProcessError(1, command)
        return subprocess.CompletedProcess(command, 0, stdout="42;cluster\n")

    monkeypatch.setattr("subprocess.run", queue)
    args = argparse.Namespace(out=tmp_path, concurrency=1, dry_run=False)
    with pytest.raises(subprocess.CalledProcessError):
        submit(args)
    assert read_json(tmp_path / "jobs.json") == {"alma-7b": "42"}
    with pytest.raises(ValueError, match="duplicate"):
        submit(args)
    assert len(calls) == 2


@pytest.fixture
def scope_results(tmp_path):
    from mnlp_eval.artifacts import Segment, atomic_write_json, write_jsonl
    from mnlp_eval.config import ModelSpec, RunConfig, SuiteSpec
    from mnlp_eval.runspec import RunPaths

    conditions = matrix(["de"], 1280, "flap")
    names = ["alma-7b", *[c["name"] for c in conditions]]
    atomic_write_json(
        tmp_path / "experiment.json",
        {
            "systems": names,
            "pairs": ["de"],
            "conditions": conditions,
            "budget": 1280,
        },
    )
    paths_by_name = {}
    for name in names:
        config = RunConfig(
            model=ModelSpec(name=name, loader="hf_causal", model_name_or_path=name),
            suite=SuiteSpec.from_dict(
                {
                    "name": "scope",
                    "data": {
                        "dataset": "test",
                        "directions": ["de-en", "en-de"],
                    },
                }
            ),
            output_root=tmp_path / "runs",
        )
        paths = RunPaths.for_config(config)
        paths.init_manifest(config)
        paths_by_name[name] = paths
        directions = {}
        shift = 0.1 if "direction-" in name else 0
        for direction in ("de-en", "en-de"):
            write_jsonl(
                paths.hyps_jsonl(direction),
                [
                    Segment(
                        index=i,
                        source=f"source {i}",
                        reference=f"reference {i}",
                        hypothesis=f"translation {i}",
                    )
                    for i in range(5)
                ],
            )
            directions[direction] = {"data": {"fingerprint": direction}}
        paths.record_stage("generate", {"status": "completed", "directions": directions})
        for group in ("surface", "neural"):
            metrics = (
                {"bleu": {"score": 20.0}, "chrf2pp": {"score": 40.0}}
                if group == "surface"
                else {
                    "wmt22_comet_da": {
                        "score": 0.8 + shift,
                        "signature": "test",
                        "segment_scores": [0.8 + shift] * 5,
                    },
                }
            )
            atomic_write_json(
                paths.scores(group),
                {
                    "run_id": config.run_id,
                    "settings": {},
                    "directions": {d: {"metrics": metrics} for d in directions},
                    "aggregate": {key: value["score"] for key, value in metrics.items()},
                },
            )
        if name != "alma-7b":
            atomic_write_json(
                tmp_path / "checkpoints" / name / "prune.json",
                {
                    "removed_heads": 192,
                    "removed_channels": 70464,
                    "parameters_after": 12345,
                    "calibration_segments": 1280,
                },
            )
    return tmp_path, paths_by_name, conditions


def test_report_computes_target_contrasts_without_benchmarks(scope_results):
    from mnlp_eval.artifacts import read_json
    from mnlp_eval.experiments.direction_scope_report import report

    root, _, _ = scope_results
    report(root)
    contrasts = read_json(root / "reports/comparisons.json")
    assert len(contrasts) == 4
    for comparison in contrasts.values():
        assert comparison["delta"] == pytest.approx(0.1)
        assert comparison["bootstrap_ci_95"] == pytest.approx([0.1, 0.1])
        assert comparison["p_value_holm"] < 0.05
    assert (root / "reports/by-direction.csv").is_file()
    assert (root / "reports/comparison.md").is_file()


@pytest.mark.parametrize("damage", ["alignment", "scores", "budget", "calibration", "settings"])
def test_report_rejects_invalid_scientific_comparisons(scope_results, damage):
    from mnlp_eval.artifacts import atomic_write_json, read_json
    from mnlp_eval.experiments.direction_scope_report import report

    root, paths_by_name, conditions = scope_results
    candidate = conditions[-1]["name"]
    paths = paths_by_name[candidate]
    if damage == "alignment":
        path = paths.hyps_jsonl("de-en")
        path.write_text(path.read_text().replace("source 0", "other source"))
    elif damage in ("scores", "settings"):
        path = paths.scores("neural")
        payload = read_json(path)
        if damage == "scores":
            payload["directions"]["de-en"]["metrics"]["wmt22_comet_da"]["segment_scores"] = [0.9]
        else:
            payload["settings"] = {"different": True}
        atomic_write_json(path, payload)
    else:
        path = root / "checkpoints" / candidate / "prune.json"
        payload = read_json(path)
        payload["parameters_after" if damage == "budget" else "calibration_segments"] += 1
        atomic_write_json(path, payload)
    with pytest.raises(ValueError):
        report(root)


def test_prepare_freezes_matched_inputs_and_single_direction_specs(tmp_path, monkeypatch):
    import argparse
    import sys
    import types
    from pathlib import Path

    from mnlp_eval.artifacts import atomic_write_json, read_json
    from mnlp_eval.config import load_yaml_config
    from mnlp_eval.experiments.direction_scope import prepare, verify_inputs

    hub = types.ModuleType("huggingface_hub")
    hub.snapshot_download = lambda _model: "/cache/pinned-alma-revision"
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setattr("subprocess.run", lambda *_a, **_k: None)
    monkeypatch.setattr("mnlp_eval.experiments.direction_scope._call", lambda *_a, **_k: None)

    def calibration(spec, out):
        folder = out / spec.name
        folder.mkdir(parents=True)
        for direction in spec.directions:
            (folder / f"{direction}.jsonl").write_text('{"source": "training source"}\n')
        payload = {
            "total_segments": spec.segments_per_direction * len(spec.directions),
            "fingerprint": spec.name,
            "contamination": {
                "total_collisions": 0,
                "directions": {d: {"checked": True} for d in spec.directions},
            },
        }
        atomic_write_json(folder / "calibration.json", payload)
        return payload

    monkeypatch.setattr("mnlp_eval.data.calibration.build_calibration_set", calibration)
    monkeypatch.setattr(
        "mnlp_eval.data.loaders.load_testset",
        lambda *_a: types.SimpleNamespace(
            sources=["evaluation source"],
            references=["evaluation reference"],
            provenance=lambda: {"fingerprint": "test"},
        ),
    )
    out = tmp_path / "experiment"
    prepare(
        argparse.Namespace(
            out=out,
            pairs=["de"],
            budget=1280,
            method="flap",
            seed=1234,
            limit=200,
            comet_python=Path("/comet-python"),
        )
    )
    manifest = read_json(out / "experiment.json")
    assert len(manifest["systems"]) == 5
    assert manifest["python"] == sys.executable
    verify_inputs(out, manifest)
    for condition in manifest["conditions"]:
        spec = load_yaml_config(out / "configs" / f"prune-{condition['name']}.yaml")
        assert set(spec["directions"]) == set(condition["directions"])
        assert spec["allocation"] == "uniform"
        assert spec["calibration_text"] == "prompt+target"
        assert spec["model_name_or_path"] == "/cache/pinned-alma-revision"
    suite = load_yaml_config(out / "configs/suite.yaml")
    assert suite["data"]["directions"] == ["de-en", "en-de"]
    assert suite["data"]["dataset"].startswith("local:text:")
    assert not (out / "jobs.json").exists()
    # Detect any change to a frozen input, regardless of its file name.
    frozen = next((out / "data/evaluation").iterdir())
    frozen.write_text("changed\n")
    with pytest.raises(ValueError, match="input changed"):
        verify_inputs(out, manifest)
