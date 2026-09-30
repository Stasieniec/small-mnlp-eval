"""The experiment must fail closed and compare aligned segment scores."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from mnlp_eval.artifacts import Segment, atomic_write_json, write_jsonl
from mnlp_eval.config import ModelSpec, RunConfig, SuiteSpec
from mnlp_eval.experiments.layer_protection import CONDITIONS, CONTROL, DENSE, submit, verify_inputs
from mnlp_eval.experiments.protection_report import holm, macro_comet_delta, report, validate_runs
from mnlp_eval.report.tables import collect_runs
from mnlp_eval.runspec import RunPaths


def make_run(root: Path, name: str, *, shift: float = 0.0) -> RunPaths:
    config = RunConfig(
        model=ModelSpec(name=name, loader="hf_causal", model_name_or_path=name),
        suite=SuiteSpec.from_dict(
            {
                "name": "tiny",
                "data": {"dataset": "test", "directions": ["de-en", "en-de"]},
                "decode": {"num_beams": 1},
            }
        ),
        output_root=root / "runs",
    )
    paths = RunPaths.for_config(config)
    paths.init_manifest(config)
    directions = {}
    for direction in config.suite.data.directions:
        records = [
            Segment(
                index=i,
                source=f"source {i}",
                reference=f"a proper translation {i}",
                hypothesis=f"a proper translation {i}",
            )
            for i in range(5)
        ]
        write_jsonl(paths.hyps_jsonl(direction), records)
        directions[direction] = {"data": {"fingerprint": direction}}
    paths.record_stage("generate", {"status": "completed", "directions": directions})
    for group in ("surface", "neural"):
        per_dir = {}
        for direction in config.suite.data.directions:
            metric = (
                {"bleu": {"score": 100.0}, "chrf2pp": {"score": 100.0}}
                if group == "surface"
                else {
                    "wmt22_comet_da": {
                        "score": 0.8 + shift,
                        "signature": "test-comet",
                        "segment_scores": [
                            0.7 + shift,
                            0.8 + shift,
                            0.9 + shift,
                            0.75 + shift,
                            0.85 + shift,
                        ],
                    }
                }
            )
            per_dir[direction] = {"n_segments": 5, "metrics": metric}
        aggregate = {key: value["score"] for key, value in metric.items()}
        atomic_write_json(
            paths.scores(group),
            {
                "run_id": config.run_id,
                "settings": {},
                "directions": per_dir,
                "aggregate": aggregate,
            },
        )
    atomic_write_json(
        paths.bench,
        {
            "run_id": config.run_id,
            "protocol": {"repeats": 5},
            "static": {"total_parameters": 12345},
            "by_batch_size": {},
        },
    )
    return paths


def test_macro_bootstrap_and_holm_are_correct(tmp_path: Path) -> None:
    make_run(tmp_path, CONTROL)
    make_run(tmp_path, "candidate", shift=0.1)
    runs = validate_runs(collect_runs(tmp_path / "runs"), [CONTROL, "candidate"])
    result = macro_comet_delta(runs[CONTROL], runs["candidate"], samples=99, seed=42)
    assert result["delta"] == pytest.approx(0.1)
    assert result["ci95"] == pytest.approx([0.1, 0.1])
    assert result["p_value"] == 0.01
    assert holm({"a": 0.01, "b": 0.02, "c": 0.8, "d": 0.1}) == {
        "a": 0.04,
        "b": 0.06,
        "d": 0.2,
        "c": 0.8,
    }


@pytest.mark.parametrize("damage", ["alignment", "scores", "benchmark", "missing-system"])
def test_report_refuses_invalid_comparisons(tmp_path: Path, damage: str) -> None:
    make_run(tmp_path, CONTROL)
    candidate = make_run(tmp_path, "candidate")
    if damage == "alignment":
        records = [
            Segment(index=i, source=f"different {i}", reference="r", hypothesis="h")
            for i in range(5)
        ]
        write_jsonl(candidate.hyps_jsonl("de-en"), records)
    elif damage == "scores":
        atomic_write_json(candidate.scores("neural"), {"directions": {}})
    elif damage == "benchmark":
        candidate.bench.unlink()
    expected = (
        [CONTROL, "candidate", "absent"] if damage == "missing-system" else [CONTROL, "candidate"]
    )
    with pytest.raises(ValueError):
        validate_runs(collect_runs(tmp_path / "runs"), expected)


def test_complete_report_contains_direct_tests_and_budget_audit(tmp_path: Path) -> None:
    systems = [DENSE] + [f"alma-7b-boundary-{tag}" for tag in CONDITIONS]
    atomic_write_json(tmp_path / "experiment.json", {"systems": systems})
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/metrics.yaml").write_text("bootstrap_samples: 20\n")
    for name in systems:
        make_run(tmp_path, name, shift=0 if name == CONTROL else 0.01)
    for tag, (first, last) in CONDITIONS.items():
        name = f"alma-7b-boundary-{tag}"
        protected = list(range(first)) + list(range(32 - last, 32))
        # Tiny descriptors with an identical total removal at different depths.
        eligible = [i for i in range(32) if i not in protected]
        pruned = set(eligible[:8])
        kept = {str(i): [1] if i in pruned else [0, 1] for i in range(32)}
        atomic_write_json(
            tmp_path / "subnetworks" / f"{name}.json",
            {
                "name": name,
                "components": {
                    key: {"total": 2, "kept": kept} for key in ("attention_heads", "ffn_channels")
                },
            },
        )
        atomic_write_json(
            tmp_path / "checkpoints" / name / "prune.json",
            {
                "protected_layers": protected,
                "parameters_after": 12345,
                "removed_heads": 8,
                "removed_channels": 8,
            },
        )
    report(tmp_path)
    results = json.loads((tmp_path / "reports/comparisons.json").read_text())
    assert results["control"] == CONTROL
    assert len(results["comparisons"]) == 4
    assert all("p_holm" in row["macro_comet"] for row in results["comparisons"].values())
    assert (tmp_path / "reports/against-dense/report.md").is_file()
    assert (tmp_path / "reports/against-control/report.md").is_file()
    assert len((tmp_path / "reports/per-direction-deltas.csv").read_text().splitlines()) == 9


def test_submit_records_ids_and_requires_all_upstream_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    systems = [DENSE, CONTROL]
    atomic_write_json(
        tmp_path / "experiment.json",
        {"systems": systems, "input_sha256": {}, "python": "/venv/bin/python", "hf_home": None},
    )
    calls = []

    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout=f"{len(calls)};cluster\n")

    monkeypatch.setattr(subprocess, "run", fake_run)
    args = argparse.Namespace(
        out=tmp_path, partition="gpu_a100", cpu_partition="rome", time="08:00:00", dry_run=False
    )
    submit(args)
    assert len(calls) == 8
    assert "afterok:2:3:6:7" in calls[-1]
    assert all("--kill-on-invalid-dep=yes" in command for command in calls)
    assert "--exclusive" in calls[2]
    assert "--gpus" not in calls[-1]
    with pytest.raises(ValueError, match="duplicate"):
        submit(args)


def test_input_edits_are_detected(tmp_path: Path) -> None:
    (tmp_path / "input").write_text("changed")
    with pytest.raises(ValueError, match="input changed"):
        verify_inputs(tmp_path, {"input_sha256": {"input": "original"}})


@pytest.mark.torch
def test_prepare_freezes_inputs_and_resolves_every_condition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from types import SimpleNamespace

    import huggingface_hub
    import transformers

    from mnlp_eval.data import calibration, loaders
    from mnlp_eval.experiments import layer_protection as experiment

    root = tmp_path / "experiment"
    suite = tmp_path / "suite.yaml"
    suite.write_text("name: tiny\ndata:\n  dataset: test\n  directions: [de-en, en-de]\n")

    def build(spec: Any, out: Path) -> dict[str, Any]:
        target = out / spec.name
        target.mkdir(parents=True)
        for direction in spec.directions:
            (target / f"{direction}.jsonl").write_text(
                json.dumps({"source": "calibration", "target": "reference", "prompt": "translate"})
                + "\n"
            )
        manifest = {
            "fingerprint": "frozen",
            "contamination": {
                "total_collisions": 0,
                "directions": {d: {"checked": True} for d in spec.directions},
            },
        }
        atomic_write_json(target / "calibration.json", manifest)
        return manifest

    monkeypatch.setattr(calibration, "build_calibration_set", build)
    monkeypatch.setattr(
        loaders,
        "load_testset",
        lambda _spec, direction: SimpleNamespace(
            sources=("evaluation",),
            references=("translation",),
            provenance=lambda: {"fingerprint": str(direction)},
        ),
    )
    monkeypatch.setattr(
        huggingface_hub, "snapshot_download", lambda _repo: "/cache/pinned-snapshot"
    )
    monkeypatch.setattr(
        transformers.AutoConfig,
        "from_pretrained",
        lambda _path: SimpleNamespace(
            num_hidden_layers=32,
            num_attention_heads=32,
            num_key_value_heads=32,
            intermediate_size=11008,
        ),
    )
    monkeypatch.setattr(experiment, "_call", lambda *_args: None)
    monkeypatch.setattr(
        subprocess, "run", lambda *_args, **_kwargs: subprocess.CompletedProcess([], 0, stdout="")
    )
    import mnlp_eval.env_capture

    monkeypatch.setattr(mnlp_eval.env_capture, "capture_environment", lambda: {"test": True})
    args = argparse.Namespace(
        out=root,
        repo=Path.cwd(),
        suite=suite,
        seed=1234,
        comet_python=Path(".venv-comet/bin/python"),
        global_reference=True,
    )
    experiment.prepare(args)
    manifest = json.loads((root / "experiment.json").read_text())
    assert len(manifest["systems"]) == 7
    verify_inputs(root, manifest)
    for name in manifest["systems"][1:]:
        spec = experiment.PruneSpec.from_dict(
            experiment.load_yaml_config(root / "configs" / f"prune-{name}.yaml")
        )
        assert spec.model_name_or_path == "/cache/pinned-snapshot"
        assert Path(spec.calibration).is_dir()
    with pytest.raises(ValueError, match="already exists"):
        experiment.prepare(args)


def test_run_command_prepares_then_submits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    from mnlp_eval.experiments import layer_protection as experiment

    calls = []
    monkeypatch.setattr(experiment, "prepare", lambda args: calls.append(("prepare", args.out)))
    monkeypatch.setattr(experiment, "submit", lambda args: calls.append(("submit", args.out)))
    monkeypatch.setattr(sys, "argv", ["layer-protection", "run", str(tmp_path)])
    experiment.main()
    assert calls == [("prepare", tmp_path), ("submit", tmp_path)]
