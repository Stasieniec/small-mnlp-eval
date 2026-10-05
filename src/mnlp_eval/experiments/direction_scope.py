"""Small, matched-budget direction / pair / multilingual pruning experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from mnlp_eval.artifacts import atomic_write_json, read_json
from mnlp_eval.config import load_yaml_config
from mnlp_eval.experiments.layer_protection import _call, _yaml, run_stage, verify_inputs
from mnlp_eval.languages import ALMA_DIRECTIONS
from mnlp_eval.prune.spec import PruneSpec

DENSE = "alma-7b"


def matrix(pairs: list[str], budget: int, method: str) -> list[dict[str, Any]]:
    """One shared multilingual control, one per pair, one per direction."""
    if (
        not pairs
        or len(set(pairs)) != len(pairs)
        or any(f"{p}-en" not in ALMA_DIRECTIONS for p in pairs)
    ):
        raise ValueError("pairs must be distinct ALMA languages: cs, de, is, ru, zh")
    if budget <= 0 or budget % len(ALMA_DIRECTIONS):
        raise ValueError("calibration budget must be a positive multiple of ten")
    scopes = [("multi", list(ALMA_DIRECTIONS))]
    for pair in pairs:
        scopes.append((f"pair-{pair}", [f"{pair}-en", f"en-{pair}"]))
        scopes.extend((f"direction-{d}", [d]) for d in (f"{pair}-en", f"en-{pair}"))
    return [
        {
            "name": f"alma-7b-{method}20-scope-{tag}",
            "scope": tag,
            "directions": directions,
            "segments_per_direction": budget // len(directions),
            "total_segments": budget,
        }
        for tag, directions in scopes
    ]


def prepare(args: argparse.Namespace) -> None:
    """Freeze code, calibration, evaluation strings and model revision before submission."""
    from huggingface_hub import snapshot_download

    from mnlp_eval.config import SuiteSpec
    from mnlp_eval.data.calibration import CalibrationSpec, build_calibration_set
    from mnlp_eval.data.loaders import load_testset

    conditions = matrix(args.pairs, args.budget, args.method)
    root = args.out.resolve()
    repo = Path.cwd()
    if root.exists():
        raise ValueError("experiment directory already exists; use a fresh scratch directory")
    if args.seed < 0 or args.limit < 1:
        raise ValueError("seed must be nonnegative and evaluation limit positive")
    comet_python = str(args.comet_python.absolute())
    header_check = (
        "import pathlib, sysconfig; "
        "p = pathlib.Path(sysconfig.get_path('include')) / 'Python.h'; "
        "assert p.is_file(), f'Missing {p}; see docs/layer-protection.md'"
    )
    for interpreter in (sys.executable, comet_python):
        subprocess.run([interpreter, "-c", header_check], check=True)
    subprocess.run([comet_python, "-c", "import comet"], check=True)
    root.mkdir(parents=True)
    for folder in ("configs", "checkpoints", "subnetworks", "runs", "logs", "reports"):
        (root / folder).mkdir()
    source = root / "source"
    for folder in ("src", "recipes", "configs", "scripts", "slurm"):
        shutil.copytree(
            repo / folder, source / folder, ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
        )
    shutil.copyfile(repo / "pyproject.toml", source / "pyproject.toml")
    for condition in conditions:
        calibration = build_calibration_set(
            CalibrationSpec(
                name=condition["scope"],
                directions=condition["directions"],
                segments_per_direction=condition["segments_per_direction"],
                seed=args.seed,
            ),
            root / "data/calibration",
        )
        check = calibration.get("contamination", {})
        checks = check.get("directions", {})
        if (
            calibration["total_segments"] != args.budget
            or check.get("total_collisions", 0)
            or set(checks) != set(condition["directions"])
            or not all(item.get("checked") for item in checks.values())
        ):
            raise ValueError(f"invalid or contaminated calibration: {condition['scope']}")
        condition["calibration_fingerprint"] = calibration["fingerprint"]
    suite_payload = load_yaml_config(repo / "configs/suites/alma10-greedy-300.yaml")
    suite_payload["name"] = f"scope-{'-'.join(args.pairs)}-{args.limit}"
    suite_payload["data"].update(
        directions=[d for p in args.pairs for d in (f"{p}-en", f"en-{p}")],
        limit=args.limit,
    )
    suite = SuiteSpec.from_dict(suite_payload)
    evaluation = root / "data/evaluation"
    evaluation.mkdir(parents=True)
    provenance = {}
    for direction in suite.data.parsed_directions:
        data = load_testset(suite.data, direction)
        provenance[str(direction)] = data.provenance()
        sources = {s.strip() for s in data.sources}
        for path in (root / "data/calibration").glob(f"*/{direction}.jsonl"):
            if any(
                json.loads(line)["source"].strip() in sources
                for line in path.read_text().splitlines()
            ):
                raise ValueError(f"calibration overlaps evaluation: {path}")
        for language, texts in (
            (direction.source, data.sources),
            (direction.target, data.references),
        ):
            if any("\n" in text or "\r" in text for text in texts):
                raise ValueError("evaluation text contains embedded newlines")
            (evaluation / f"{suite.data.split}.{direction}.{language}").write_text(
                "\n".join(texts) + "\n", encoding="utf-8"
            )
    suite_payload["data"]["dataset"] = f"local:text:{evaluation}"
    _yaml(root / "configs/suite.yaml", suite_payload)
    atomic_write_json(root / "evaluation-provenance.json", provenance)
    dense = load_yaml_config(repo / "configs/models/alma-7b.yaml")
    dense["model_name_or_path"] = snapshot_download(dense["model_name_or_path"])
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
    for condition in conditions:
        spec = PruneSpec.from_dict(
            {
                "name": condition["name"],
                "method": args.method,
                "model_name_or_path": dense["model_name_or_path"],
                "calibration": str(root / "data/calibration" / condition["scope"]),
                "directions": condition["directions"],
                "calibration_text": "prompt+target",
                "sparsity": 0.2,
                "allocation": "uniform",
                "seed": args.seed,
            }
        )
        _yaml(root / "configs" / f"prune-{spec.name}.yaml", asdict(spec))
    manifest = {
        "systems": [DENSE, *[c["name"] for c in conditions]],
        "conditions": conditions,
        "pairs": args.pairs,
        "budget": args.budget,
        "seed": args.seed,
        "method": args.method,
        "python": sys.executable,
        "comet_python": comet_python,
        "hf_home": os.environ.get("HF_HOME"),
        "evaluation_limit": args.limit,
        "prepared": True,
    }
    files = [*source.rglob("*"), *(root / "data").rglob("*"), *(root / "configs").glob("*")]
    manifest["input_sha256"] = {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(files)
        if p.is_file()
    }
    atomic_write_json(root / "experiment.json", manifest)
    print(f"Prepared {root}; review with submit --dry-run before submitting.")


def submission_plan(systems: list[str], concurrency: int) -> list[tuple[str, list[str]]]:
    if concurrency not in (1, 2):
        raise ValueError("GPU concurrency must be 1 or 2")
    return [
        (name, [systems[i - concurrency]] if i >= concurrency else [])
        for i, name in enumerate(systems)
    ]


def submit(args: argparse.Namespace) -> None:
    root = args.out.resolve()
    manifest = read_json(root / "experiment.json")
    verify_inputs(root, manifest)
    if (root / "jobs.json").exists():
        raise ValueError("jobs.json exists; refusing duplicate submission")
    ids: dict[str, str] = {}
    env = dict(os.environ, PYTHONPATH=f"{root / 'source/src'}:{root / 'source'}")
    if manifest["hf_home"]:
        env["HF_HOME"] = manifest["hf_home"]
    plan = submission_plan(manifest["systems"], args.concurrency)
    for name, dependencies in [*plan, ("report", manifest["systems"])]:
        report_job = name == "report"
        command = [
            "sbatch",
            "--parsable",
            "--kill-on-invalid-dep=yes",
            "--job-name",
            f"scope-{name}",
            "--output",
            str(root / "logs" / f"{name}-%j.out"),
            "--partition",
            "rome" if report_job else "gpu_a100",
            "--cpus-per-task",
            "16" if report_job else "18",
            "--time",
            "00:30:00" if report_job else "02:00:00",
        ]
        if not report_job:
            command += ["--gpus", "1"]
        if dependencies:
            command += ["--dependency", "afterok:" + ":".join(ids[d] for d in dependencies)]
        command += [
            str(root / "source/slurm/direction_scope.sbatch"),
            manifest["python"],
            str(root),
            name,
        ]
        if args.dry_run:
            print(shlex.join(command))
            ids[name] = str(len(ids) + 1)
        else:
            result = subprocess.run(command, env=env, capture_output=True, text=True, check=True)
            job_id = result.stdout.strip().split(";")[0]
            if not job_id.isdigit():
                raise ValueError(f"unexpected sbatch output: {result.stdout}")
            ids[name] = job_id
            atomic_write_json(root / "jobs.json", ids)
            print(f"{name}: {job_id}", flush=True)


def stage(root: Path, system: str) -> None:
    manifest = read_json(root / "experiment.json")
    verify_inputs(root, manifest)
    if system == "report":
        from mnlp_eval.experiments.direction_scope_report import report

        report(root)
        return
    if system not in manifest["systems"]:
        raise ValueError(f"unknown system: {system}")
    # Each subprocess exits before the next loads weights, freeing GPU memory.
    if system != DENSE:
        run_stage(root, "prune", system)
    run_stage(root, "generate", system)
    run_stage(root, "score", system)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("plan", "prepare"):
        child = sub.add_parser(command)
        child.add_argument("--pairs", nargs="+", default=["de", "is"])
        child.add_argument("--budget", type=int, default=1280)
        child.add_argument("--method", choices=["flap", "slimgpt"], default="flap")
        if command == "prepare":
            child.add_argument("out", type=Path)
            child.add_argument("--limit", type=int, default=200)
            child.add_argument("--seed", type=int, default=1234)
            child.add_argument("--comet-python", type=Path, default=Path(".venv-comet/bin/python"))
    queue = sub.add_parser("submit")
    queue.add_argument("out", type=Path)
    queue.add_argument("--dry-run", action="store_true")
    queue.add_argument("--concurrency", type=int, choices=[1, 2], default=2)
    worker = sub.add_parser("stage")
    worker.add_argument("out", type=Path)
    worker.add_argument("system")
    args = parser.parse_args()
    if args.command == "plan":
        print(json.dumps(matrix(args.pairs, args.budget, args.method), indent=2))
    elif args.command == "prepare":
        prepare(args)
    elif args.command == "submit":
        submit(args)
    else:
        stage(args.out.resolve(), args.system)


if __name__ == "__main__":
    main()
