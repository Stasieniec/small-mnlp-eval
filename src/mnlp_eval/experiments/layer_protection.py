"""Prepare and submit the fixed five-condition boundary-protection experiment.

Each experiment owns its inputs, code snapshot, checkpoints, runs and reports.
Preparation downloads on the login node; jobs run offline from the snapshot.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from mnlp_eval.artifacts import atomic_write_json, read_json
from mnlp_eval.config import ModelSpec, RunConfig, SuiteSpec, load_yaml_config
from mnlp_eval.prune.budget import allocate
from mnlp_eval.prune.groups import LayerGroups
from mnlp_eval.prune.spec import PruneSpec

CONDITIONS = {
    "none": (0, 0),
    "first4": (4, 0),
    "last2": (0, 2),
    "first4-last2": (4, 2),
    "first3-last1": (3, 1),
}
CONTROL = "alma-7b-boundary-none"
DENSE = "alma-7b"
GLOBAL = "alma-7b-boundary-global-reference"


def budget_preview() -> list[dict[str, Any]]:
    """Predict exact realised counts for ALMA, without importing torch."""
    groups = [LayerGroups(i, 32, 32, 128, 11008) for i in range(32)]
    rows = []
    for name, (first, last) in CONDITIONS.items():
        plan = allocate(
            [[0.0] * 32 for _ in groups],
            [[0.0] * 11008 for _ in groups],
            groups,
            sparsity=0.2,
            protect_first_n=first,
            protect_last_n=last,
        )
        rows.append(
            {
                "condition": name,
                "first": first,
                "last": last,
                "removed_heads": sum(32 - len(p.heads) for p in plan),
                "removed_channels": sum(11008 - len(p.channels) for p in plan),
                "heads_kept_by_layer": [len(p.heads) for p in plan],
                "channels_kept_by_layer": [len(p.channels) for p in plan],
            }
        )
    return rows


def _yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _call(python: str, root: Path, *args: str) -> None:
    subprocess.run(
        [python, "-m", "mnlp_eval.cli", "--runs-root", str(root / "runs"), *args], check=True
    )


def prepare(args: argparse.Namespace) -> None:
    """Download, snapshot, and validate before any jobs are submitted."""
    from huggingface_hub import snapshot_download
    from transformers import AutoConfig

    from mnlp_eval.data.calibration import CalibrationSpec, build_calibration_set
    from mnlp_eval.data.loaders import load_testset
    from mnlp_eval.env_capture import capture_environment

    root = args.out.resolve()
    repo = args.repo.resolve()
    comet_python = str(args.comet_python.absolute())
    if args.seed < 0:
        raise ValueError("seed must be non-negative")
    if root.exists():
        raise ValueError(
            f"{root} already exists; use submit to resume preparation's output, "
            "or choose a fresh experiment directory"
        )
    # Triton builds a Python extension on first GPU use. Check both interpreters
    # before downloading inputs or submitting a DAG that cannot run.
    header_check = (
        "import pathlib, sysconfig; "
        "p = pathlib.Path(sysconfig.get_path('include')) / 'Python.h'; "
        "assert p.is_file(), f'Missing {p}; see docs/layer-protection.md'"
    )
    for interpreter in (sys.executable, comet_python):
        subprocess.run([interpreter, "-c", header_check], check=True)
    # Missing COMET is a hard failure, never a silently omitted primary metric.
    subprocess.run([comet_python, "-c", "import comet"], check=True)
    suite_payload = load_yaml_config(args.suite)
    suite = SuiteSpec.from_dict(suite_payload)
    if "de-en" not in suite.data.directions:
        raise ValueError("the experiment's efficiency benchmark requires de-en in the suite")
    root.mkdir(parents=True)
    for folder in ("configs", "checkpoints", "subnetworks", "runs", "logs", "reports"):
        (root / folder).mkdir()
    source = root / "source"
    for folder in ("src", "recipes", "configs", "scripts", "slurm"):
        shutil.copytree(
            repo / folder, source / folder, ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
        )
    shutil.copyfile(repo / "pyproject.toml", source / "pyproject.toml")

    calibration_payload = load_yaml_config(repo / "configs/calibration/multi-10dir.yaml")
    calibration_payload["seed"] = args.seed
    calibration = build_calibration_set(
        CalibrationSpec.from_dict(calibration_payload), root / "data/calibration"
    )
    contamination = calibration.get("contamination", {})
    checked = contamination.get("directions", {})
    if (
        contamination.get("total_collisions", 0)
        or set(checked) != set(calibration_payload["directions"])
        or not all(entry.get("checked") for entry in checked.values())
    ):
        raise ValueError("calibration must pass contamination checks in every direction")

    # Freeze the exact evaluation strings, including document ordering and limit.
    data_dir = root / "data/evaluation"
    data_dir.mkdir(parents=True)
    provenance = {}
    for direction in suite.data.parsed_directions:
        data = load_testset(suite.data, direction)
        provenance[str(direction)] = data.provenance()
        for language, texts in (
            (direction.source, data.sources),
            (direction.target, data.references),
        ):
            if any("\n" in text or "\r" in text for text in texts):
                raise ValueError("plain-text evaluation snapshot cannot contain embedded newlines")
            (data_dir / f"{suite.data.split}.{direction}.{language}").write_text(
                "\n".join(texts) + "\n", encoding="utf-8"
            )
        # The fixed pilot is a predeclared comparison, not a tuning set. Also
        # refuse collisions for a user-supplied validation suite.
        records = root / "data/calibration/multi-10dir" / f"{direction}.jsonl"
        sources = {s.strip() for s in data.sources}
        if records.exists() and any(
            json.loads(line)["source"].strip() in sources
            for line in records.read_text().splitlines()
        ):
            raise ValueError(f"calibration overlaps evaluation in {direction}")
    suite_payload["data"]["dataset"] = f"local:text:{data_dir}"
    _yaml(root / "configs/suite.yaml", suite_payload)
    atomic_write_json(root / "evaluation-provenance.json", provenance)

    # A resolved Hub snapshot pins dense/pruned systems to identical weights.
    dense = load_yaml_config(repo / "configs/models/alma-7b.yaml")
    dense["model_name_or_path"] = snapshot_download(dense["model_name_or_path"])
    model_config = AutoConfig.from_pretrained(dense["model_name_or_path"])
    shape = (
        model_config.num_hidden_layers,
        model_config.num_attention_heads,
        model_config.num_key_value_heads,
        model_config.intermediate_size,
    )
    if shape != (32, 32, 32, 11008):
        raise ValueError(f"the fixed ALMA experiment expects (32, 32, 32, 11008), got {shape}")
    _yaml(root / "configs/alma-7b.yaml", dense)
    _yaml(root / "configs/metrics.yaml", load_yaml_config(repo / "configs/metrics/default.yaml"))
    _call(
        comet_python,
        root,
        "prefetch",
        "--metrics",
        str(root / "configs/metrics.yaml"),
        "--strict",
        "--json",
    )

    names = [DENSE]
    for tag, (first, last) in CONDITIONS.items():
        payload = load_yaml_config(repo / f"configs/experiments/layer-protection/{tag}.yaml")
        payload.update(
            model_name_or_path=dense["model_name_or_path"],
            calibration=str(root / "data/calibration/multi-10dir"),
            seed=args.seed,
        )
        spec = PruneSpec.from_dict(payload)
        assert (spec.protect_first_n, spec.protect_last_n) == (first, last)
        _yaml(root / "configs" / f"prune-{spec.name}.yaml", asdict(spec))
        names.append(spec.name)
    # This is a contextual reference, not part of the matched uniform ablation.
    if args.global_reference:
        payload = load_yaml_config(repo / "configs/prune/slimgpt-20-multi-target.yaml")
        payload.update(
            name=GLOBAL,
            model_name_or_path=dense["model_name_or_path"],
            calibration=str(root / "data/calibration/multi-10dir"),
            seed=args.seed,
        )
        _yaml(root / "configs" / f"prune-{GLOBAL}.yaml", asdict(PruneSpec.from_dict(payload)))
        names.append(GLOBAL)

    atomic_write_json(root / "environment.json", capture_environment())
    manifest = {
        "systems": names,
        "control": CONTROL,
        "seed": args.seed,
        "python": sys.executable,
        "comet_python": comet_python,
        "hf_home": os.environ.get("HF_HOME"),
        "suite_source": str(args.suite),
        "calibration_fingerprint": calibration["fingerprint"],
        "budget_preview": budget_preview(),
        "prepared": True,
    }
    # Hash inputs/code, not generated model configs or checkpoints. Every job
    # verifies these before running so edits cannot silently mix experiments.
    files = [*source.rglob("*"), *(root / "data").rglob("*"), *(root / "configs").glob("*")]
    manifest["input_sha256"] = {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(files)
        if path.is_file()
    }
    atomic_write_json(root / "experiment.json", manifest)
    print(f"Prepared {root}. Submit with: bash scripts/run_layer_protection.sh submit {root}")


def verify_inputs(root: Path, manifest: dict[str, Any]) -> None:
    for relative, digest in manifest["input_sha256"].items():
        if hashlib.sha256((root / relative).read_bytes()).hexdigest() != digest:
            raise ValueError(f"experiment input changed: {relative}; use a fresh experiment")


def submission_plan(systems: list[str]) -> list[tuple[str, str, list[str]]]:
    """Job DAG, with direct dependencies expressed as stable keys."""
    jobs: list[tuple[str, str, list[str]]] = []
    finished = []
    previous_bench = None
    for system in systems:
        prune = f"prune:{system}"
        if system != DENSE:
            jobs.append(("prune", system, []))
        jobs.append(("generate", system, [] if system == DENSE else [prune]))
        jobs.append(("score", system, [f"generate:{system}"]))
        dependencies = [f"generate:{system}"]
        if previous_bench:
            dependencies.append(previous_bench)
        jobs.append(("bench", system, dependencies))
        previous_bench = f"bench:{system}"
        finished.extend([f"score:{system}", previous_bench])
    jobs.append(("report", "all", finished))
    return jobs


def submit(args: argparse.Namespace) -> None:
    root = args.out.resolve()
    manifest = read_json(root / "experiment.json")
    verify_inputs(root, manifest)
    if (root / "jobs.json").exists():
        raise ValueError(
            "jobs.json already exists: refusing duplicate submissions; "
            "inspect job IDs and rerun failed stages explicitly"
        )
    ids: dict[str, str] = {}
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{root / 'source/src'}:{root / 'source'}"
    if manifest["hf_home"]:
        env["HF_HOME"] = manifest["hf_home"]
    for stage, system, dependencies in submission_plan(manifest["systems"]):
        key = f"{stage}:{system}"
        command = [
            "sbatch",
            "--parsable",
            "--kill-on-invalid-dep=yes",
            "--job-name",
            f"boundary-{stage}-{system}",
            "--output",
            str(root / "logs" / f"{stage}-{system}-%j.out"),
            "--partition",
            args.cpu_partition if stage == "report" else args.partition,
            "--cpus-per-task",
            "16" if stage == "report" else "18",
            "--time",
            "02:00:00" if stage == "report" else args.time,
        ]
        if stage != "report":
            command += ["--gpus", "1"]
        if stage == "bench":
            command += ["--exclusive"]
        if dependencies:
            command += ["--dependency", "afterok:" + ":".join(ids[d] for d in dependencies)]
        command += [
            str(root / "source/slurm/layer_protection.sbatch"),
            manifest["python"],
            str(root),
            stage,
            system,
        ]
        if args.dry_run:
            import shlex

            print(shlex.join(command))
            ids[key] = str(len(ids) + 1)
        else:
            result = subprocess.run(command, env=env, capture_output=True, text=True, check=True)
            ids[key] = result.stdout.strip().split(";")[0]
            if not ids[key].isdigit():
                raise ValueError(f"unexpected sbatch output: {result.stdout}")
            atomic_write_json(root / "jobs.json", ids)
            print(f"{key}: {ids[key]}", flush=True)
    print(f"Results will be written to {root / 'reports'}")


def run_stage(root: Path, stage: str, system: str) -> None:
    manifest = read_json(root / "experiment.json")
    verify_inputs(root, manifest)
    if stage == "report":
        from mnlp_eval.experiments.protection_report import report

        report(root)
        return
    if system not in manifest["systems"]:
        raise ValueError(f"unknown system: {system}")
    config = root / "configs" / f"{system}.yaml"
    python = manifest["python"]
    if stage == "prune":
        if (root / "checkpoints" / system).exists():
            raise ValueError(
                "checkpoint directory exists; refusing to overwrite experiment weights"
            )
        _call(
            python,
            root,
            "prune",
            "--spec",
            str(root / "configs" / f"prune-{system}.yaml"),
            "--out",
            str(root / "checkpoints"),
            "--model-config-dir",
            str(root / "configs"),
            "--subnetwork-dir",
            str(root / "subnetworks"),
        )
        return
    run = RunConfig(
        model=ModelSpec.from_dict(load_yaml_config(config)),
        suite=SuiteSpec.from_dict(load_yaml_config(root / "configs/suite.yaml")),
        output_root=root / "runs",
    )
    if stage == "generate":
        _call(
            python,
            root,
            "generate",
            "--model",
            str(config),
            "--suite",
            str(root / "configs/suite.yaml"),
        )
    elif stage == "score":
        for interpreter, group in ((python, "surface"), (manifest["comet_python"], "neural")):
            _call(
                interpreter,
                root,
                "score",
                "--run",
                str(run.directory),
                "--groups",
                group,
                "--metrics",
                str(root / "configs/metrics.yaml"),
            )
            if not (run.directory / f"scores.{group}.json").is_file():
                raise ValueError(f"{group} scoring did not produce results")
    elif stage == "bench":
        _call(python, root, "bench", "--run", str(run.directory), "--set", "bench.repeats=5")
    else:
        raise ValueError(f"unknown stage: {stage}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("plan", help="print the fixed matrix and exact unit budgets; no downloads/jobs")
    prep = sub.add_parser(
        "prepare", aliases=["run"], help="freeze inputs; 'run' also submits the complete experiment"
    )
    prep.add_argument("out", type=Path)
    prep.add_argument("--repo", type=Path, default=Path.cwd())
    prep.add_argument("--suite", type=Path, default=Path("configs/suites/alma10-greedy-300.yaml"))
    prep.add_argument("--comet-python", type=Path, default=Path(".venv-comet/bin/python"))
    prep.add_argument("--seed", type=int, default=1234)
    prep.add_argument("--global-reference", action=argparse.BooleanOptionalAction, default=True)
    prep.add_argument("--partition", default="gpu_a100")
    prep.add_argument("--cpu-partition", default="rome")
    prep.add_argument("--time", default="08:00:00")
    prep.set_defaults(dry_run=False)
    queue = sub.add_parser("submit", help="queue pruning, generation, scoring, bench and reports")
    queue.add_argument("out", type=Path)
    queue.add_argument("--partition", default="gpu_a100")
    queue.add_argument("--cpu-partition", default="rome")
    queue.add_argument("--time", default="08:00:00")
    queue.add_argument("--dry-run", action="store_true")
    stage = sub.add_parser("stage", help="run one stage, normally called by Slurm")
    stage.add_argument("out", type=Path)
    stage.add_argument("stage", choices=["prune", "generate", "score", "bench", "report"])
    stage.add_argument("system", nargs="?", default="all")
    args = parser.parse_args()
    if args.command == "plan":
        print(json.dumps(budget_preview(), indent=2))
    elif args.command in ("prepare", "run"):
        prepare(args)
        if args.command == "run":
            submit(args)
    elif args.command == "submit":
        submit(args)
    else:
        run_stage(args.out.resolve(), args.stage, args.system)


if __name__ == "__main__":
    main()
