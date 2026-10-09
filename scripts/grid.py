#!/usr/bin/env python3
"""The final experiment grid: 192 pruned ALMA-7B models, their configs, jobs and status.

method (SlimGPT log-increase, FLAP al-am) x calibration text (prompt+target,
prompt+generated) x sparsity (20, 30, 40) x scope (all ten directions, one
pair, one direction). Every model is pruned, evaluated on all ten directions
of alma10-greedy-300 and scored, in one job: slurm/grid_pipeline.sbatch.

Snellius rejects a shared job that names two partitions, so each model is
submitted twice, to gpu_a100 and to gpu_h100, under one job name. The first
twin to start takes grid-state/locks/<name> and cancels the other; a twin
that starts anyway finds the lock held by a live job and exits.

On a login node:

    ./.venv/bin/python scripts/grid.py configs
    ./.venv/bin/python scripts/grid.py submit --scope multi [--dry-run]
    ./.venv/bin/python scripts/grid.py status [--verbose]
    ./.venv/bin/python scripts/grid.py resubmit-failed
    ./.venv/bin/python scripts/grid.py repairs --source alma-7b-slimgpt20-ref-multi

``lock``, ``unlock``, ``mark``, ``check`` and ``save-manifest`` are what the
job script calls. Standard library only, plus yaml where configs are read or
written, so it runs on a login node and inside the job before anything heavy.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import getpass
import json
import os
import re
import shlex
import shutil
import socket
import subprocess
import sys
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]

MODEL_PATH = "/scratch-shared/scur0560/models/alma-7b-bf16"
SUITE = "configs/suites/alma10-greedy-300.yaml"
METRICS = "configs/metrics/default.yaml"
REPAIR_DATA = "data/calibration/repair-multi-clean"
PRUNE_CONFIG_DIR = "configs/prune/grid"
REPAIR_CONFIG_DIR = "configs/repair/grid"
MODEL_CONFIG_DIR = "configs/models"
MANIFEST = "configs/grid/manifest.json"
SBATCH_SCRIPT = "slurm/grid_pipeline.sbatch"
LOG_DIR = "slurm-logs/grid"
JOB_PREFIX = "g-"

LANGUAGES = ("cs", "de", "is", "ru", "zh")
DIRECTIONS = tuple(f"{lang}-en" for lang in LANGUAGES) + tuple(f"en-{lang}" for lang in LANGUAGES)
#: Each method with the per-layer allocation it runs under.
METHODS = {"slimgpt": "log-increase", "flap": "al-am"}
TEXTS = {"ref": "prompt+target", "gen": "prompt+generated"}
SPARSITIES = (0.2, 0.3, 0.4)
SCOPES = ("multi", "pair", "dir")
COMMON = {
    "model_name_or_path": MODEL_PATH,
    "dtype": "bfloat16",
    "batch_size": 4,
    # 512 source tokens plus 256 generated, the cap generated calibration
    # needs. The reference arm uses it too, so the two differ in text alone.
    "max_length": 768,
    "seed": 1234,
}

#: A quarter node of each GPU partition: 72 cores over 4 A100s, 64 over 4 H100s.
PARTITION_CPUS = {"gpu_a100": 18, "gpu_h100": 16}
DEFAULT_PARTITIONS = "gpu_a100,gpu_h100"
#: The pilot took about 13 minutes to prune, 10 to generate and 3 to score on
#: an A100. Short limits backfill sooner; repair is one to two hours of LoRA.
DEFAULT_TIME = {"prune": "03:00:00", "repair": "10:00:00"}

#: Slurm states after which a job will not run again.
TERMINAL = frozenset(
    {
        "BOOT_FAIL",
        "CANCELLED",
        "COMPLETED",
        "DEADLINE",
        "FAILED",
        "NODE_FAIL",
        "OUT_OF_MEMORY",
        "PREEMPTED",
        "REVOKED",
        "SPECIAL_EXIT",
        "TIMEOUT",
    }
)
#: mkdir and the owner record are two steps. A lock younger than this with no
#: owner yet is a twin between them, not a dead job.
OWNER_GRACE = 120.0
SACCT_FIELDS = "JobID,State,Elapsed,ExitCode"
#: Exit status of ``lock`` when a live twin holds the lock.
LOCK_HELD = 10


class GridError(RuntimeError):
    """A grid command that cannot go on."""


# --------------------------------------------------------------------------
# The grid
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class GridModel:
    """One pruned model of the grid."""

    method: str
    tag: str
    sparsity: float
    scope: str
    scope_key: str
    priority: int = 0

    @property
    def pct(self) -> int:
        return round(self.sparsity * 100)

    @property
    def name(self) -> str:
        scope = "multi" if self.scope == "multi" else f"{self.scope}-{self.scope_key}"
        return f"alma-7b-{self.method}{self.pct}-{self.tag}-{scope}"

    @property
    def allocation(self) -> str:
        return METHODS[self.method]

    @property
    def calibration_text(self) -> str:
        return TEXTS[self.tag]

    @property
    def pruned_directions(self) -> list[str]:
        if self.scope == "multi":
            return list(DIRECTIONS)
        if self.scope == "pair":
            # Sorted, as PruneSpec normalises them, so pruned_for matches.
            return sorted([f"{self.scope_key}-en", f"en-{self.scope_key}"])
        return [self.scope_key]

    @property
    def pruned_for(self) -> str:
        return "multi" if self.scope == "multi" else ",".join(self.pruned_directions)

    @property
    def calibration(self) -> str:
        base = {
            "multi": "multi-10dir",
            "pair": f"pair-{self.scope_key}-640",
            "dir": f"dir-{self.scope_key}-1280",
        }[self.scope]
        return f"data/calibration/{base}{'-generated' if self.tag == 'gen' else ''}"

    @property
    def prune_config(self) -> str:
        return f"{PRUNE_CONFIG_DIR}/{self.name}.yaml"

    @property
    def model_config(self) -> str:
        return f"{MODEL_CONFIG_DIR}/{self.name}.yaml"

    def prune_spec(self) -> dict[str, Any]:
        spec: dict[str, Any] = {
            "name": self.name,
            "method": self.method,
            "allocation": self.allocation,
            "sparsity": self.sparsity,
            "calibration": self.calibration,
            "calibration_text": self.calibration_text,
        }
        # Each calibration set already holds exactly these directions, so this
        # selects nothing new; it is set because pruned_for derives from it.
        if self.scope != "multi":
            spec["directions"] = self.pruned_directions
        spec.update(COMMON)
        return spec

    def header(self) -> str:
        scope = "all ten directions" if self.scope == "multi" else self.pruned_for
        return (
            f"# Grid: {self.method} {self.allocation}, {self.pct}% removed, "
            f"{self.calibration_text} calibration, pruned for {scope}. "
            "Written by scripts/grid.py configs.\n"
        )

    def manifest_entry(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "method": self.method,
            "allocation": self.allocation,
            "calibration_text": self.calibration_text,
            "tag": self.tag,
            "sparsity": self.sparsity,
            "scope": self.scope,
            "scope_key": self.scope_key,
            "pruned_directions": self.pruned_directions,
            "pruned_for": self.pruned_for,
            "calibration": self.calibration,
            "prune_config": self.prune_config,
            "model_config": self.model_config,
            "suite": SUITE,
            "priority": self.priority,
        }


def _scope_keys() -> list[tuple[str, str]]:
    return (
        [("multi", "multi")]
        + [("pair", lang) for lang in LANGUAGES]
        + [("dir", direction) for direction in DIRECTIONS]
    )


def build_grid() -> list[GridModel]:
    """Every model, in submission order: multi, pair, dir; lower sparsity first."""
    keys = _scope_keys()
    models = [
        GridModel(method, tag, sparsity, scope, key)
        for scope, key in keys
        for sparsity in SPARSITIES
        for method in METHODS
        for tag in TEXTS
    ]
    methods, tags = list(METHODS), list(TEXTS)

    def order(model: GridModel) -> tuple[int, float, int, int, int]:
        return (
            SCOPES.index(model.scope),
            model.sparsity,
            methods.index(model.method),
            tags.index(model.tag),
            keys.index((model.scope, model.scope_key)),
        )

    models.sort(key=order)
    return [replace(model, priority=index) for index, model in enumerate(models)]


def validate_specs(specs: Iterable[Mapping[str, Any]]) -> list[str]:
    """Check each spec with PruneSpec.from_dict; return warnings, raise on errors.

    An allocation the installed package rejects (another branch may not have
    registered it yet) is retried as uniform, so the rest is still checked.
    """
    try:
        from mnlp_eval.config import ConfigError
        from mnlp_eval.prune.spec import PruneSpec
    except Exception as exc:  # a half-edited module is as unusable as a missing one
        return [f"mnlp_eval is not importable ({exc}); configs not validated"]
    rejected: dict[str, int] = {}
    for spec in specs:
        try:
            parsed = PruneSpec.from_dict(dict(spec))
        except ConfigError as exc:
            if "allocation" not in str(exc):
                raise
            rejected[spec["allocation"]] = rejected.get(spec["allocation"], 0) + 1
            parsed = PruneSpec.from_dict({**spec, "allocation": "uniform"})
        expected = sorted(spec["directions"]) if spec.get("directions") else None
        if parsed.directions != expected:
            msg = f"{spec['name']}: directions normalised to {parsed.directions}"
            raise GridError(msg)
        if parsed.pruned_for != ("multi" if expected is None else ",".join(expected)):
            raise GridError(f"{spec['name']}: pruned_for is {parsed.pruned_for}")
    return [
        f"allocation {name!r} is rejected by the installed PruneSpec; "
        f"{count} config(s) validated with it replaced by 'uniform'"
        for name, count in sorted(rejected.items())
    ]


def render_yaml(spec: Mapping[str, Any]) -> str:
    """Block style, one key per line, with a list on one line as a config is written by hand."""
    import yaml

    lines = []
    for key, value in spec.items():
        if isinstance(value, list):
            lines.append(f"{key}: [{', '.join(str(item) for item in value)}]")
        else:
            lines.append(yaml.safe_dump({key: value}, default_flow_style=False).rstrip("\n"))
    return "\n".join(lines) + "\n"


def write_configs(root: Path, models: Sequence[GridModel]) -> list[Path]:
    import yaml

    written = []
    for model in models:
        spec = model.prune_spec()
        text = model.header() + render_yaml(spec)
        if yaml.safe_load(text) != spec:
            raise GridError(f"{model.name}: the written config does not read back as written")
        path = root / model.prune_config
        _write_text(path, text)
        written.append(path)
    manifest = {
        "description": "Final grid; written by scripts/grid.py configs, do not edit.",
        "model_name_or_path": MODEL_PATH,
        "suite": SUITE,
        "count": len(models),
        "models": [model.manifest_entry() for model in models],
    }
    _write_text(root / MANIFEST, json.dumps(manifest, indent=2) + "\n")
    return written


# --------------------------------------------------------------------------
# Jobs: grid models and repairs
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Target:
    """One pipeline job's worth of work, as the job script receives it."""

    name: str
    mode: str  # prune | repair
    scope: str  # multi | pair | dir | repair
    config: str  # the prune or repair config, relative to the repository
    model_config: str
    priority: int


def grid_targets() -> list[Target]:
    return [
        Target(m.name, "prune", m.scope, m.prune_config, m.model_config, m.priority)
        for m in build_grid()
    ]


def repair_targets(root: Path = REPO) -> list[Target]:
    paths = sorted((root / REPAIR_CONFIG_DIR).glob("*.yaml"))
    return [
        Target(
            path.stem,
            "repair",
            "repair",
            f"{REPAIR_CONFIG_DIR}/{path.name}",
            f"{MODEL_CONFIG_DIR}/{path.stem}.yaml",
            10_000 + index,
        )
        for index, path in enumerate(paths)
    ]


def repair_spec(source: str, data: str = REPAIR_DATA) -> dict[str, Any]:
    # Everything else is RepairSpec's defaults, which are ALMA's LoRA recipe.
    return {
        "name": f"{source}-lora",
        "source": f"{MODEL_CONFIG_DIR}/{source}.yaml",
        "data": data,
    }


def select(
    targets: Sequence[Target],
    *,
    only: Sequence[str] | None = None,
    match: str | None = None,
    scope: str | None = None,
) -> list[Target]:
    chosen = list(targets)
    if only:
        known = {target.name for target in targets}
        unknown = sorted(set(only) - known)
        if unknown:
            raise GridError(f"not in the grid: {', '.join(unknown)}")
        chosen = [target for target in chosen if target.name in set(only)]
    if match:
        pattern = re.compile(match)
        chosen = [target for target in chosen if pattern.search(target.name)]
    if scope:
        chosen = [target for target in chosen if target.scope == scope]
    return sorted(chosen, key=lambda target: target.priority)


def preflight(target: Target, root: Path = REPO) -> str | None:
    """Why the job would fail at once, or None if its inputs are in place."""
    import yaml

    config = root / target.config
    if not config.is_file():
        return f"{target.config} missing; run 'grid.py configs'"
    spec = yaml.safe_load(config.read_text(encoding="utf-8")) or {}
    if target.mode == "repair":
        source = root / str(spec.get("source", ""))
        if not source.is_file():
            return f"source {spec.get('source')} not written yet; prune it first"
        data = root / str(spec.get("data", ""))
        if not (data / "calibration.json").is_file():
            return f"repair data {spec.get('data')} not built"
        return None
    calibration = root / str(spec.get("calibration", ""))
    manifest = _read_json(calibration / "calibration.json")
    if manifest is None:
        return f"calibration {spec.get('calibration')} not built"
    if spec.get("calibration_text") == "prompt+generated":
        if not manifest.get("complete"):
            return f"generated calibration {spec.get('calibration')} incomplete"
        producer = manifest.get("identity", {}).get("model", {}).get("model_name_or_path")
        if producer != spec.get("model_name_or_path"):
            return f"{spec.get('calibration')} was generated by {producer}, not the pruned model"
    return None


# --------------------------------------------------------------------------
# State: grid-state/{locks,status,logs}/ and submissions.jsonl
# --------------------------------------------------------------------------


def state_dir() -> Path:
    configured = os.environ.get("GRID_STATE_DIR")
    return Path(configured) if configured else REPO / "grid-state"


def _now() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


def _write_text(path: Path, text: str) -> None:
    """Write through a rename, so a reader never sees half a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def status_path(name: str, root: Path | None = None) -> Path:
    return (root or state_dir()) / "status" / f"{name}.json"


def read_status(name: str, root: Path | None = None) -> dict[str, Any] | None:
    return _read_json(status_path(name, root))


def _tail(path: Path, lines: int) -> list[str]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    # tqdm redraws with carriage returns; keep only the last frame of a line.
    return [line.rsplit("\r", 1)[-1] for line in text.splitlines()[-lines:]]


def mark(
    name: str,
    *,
    state: str | None = None,
    stage: str | None = None,
    seconds: Mapping[str, float] | None = None,
    skipped: str | None = None,
    run_dir: str | None = None,
    exit_code: int | None = None,
    env: Mapping[str, str] = os.environ,
    root: Path | None = None,
) -> dict[str, Any]:
    """Record a job's progress in status/<name>.json."""
    root = root or state_dir()
    path = status_path(name, root)
    status = _read_json(path) or {"name": name}
    job_id = env.get("SLURM_JOB_ID", "")
    if status.get("job_id") != job_id:
        # A new attempt. Stage times carry over, since a skipped stage's output
        # is the earlier attempt's work.
        if status.get("job_id"):
            previous = {key: status.get(key) for key in ("job_id", "state", "stage", "exit_code")}
            status.setdefault("attempts", []).append(previous)
        status.update(started=_now(), skipped=[], exit_code=None, error=None, finished=None)
    status.update(
        name=name,
        mode=env.get("GRID_MODE", status.get("mode")),
        job_id=job_id,
        partition=env.get("SLURM_JOB_PARTITION", ""),
        node=env.get("SLURMD_NODENAME", ""),
        gpu=env.get("GRID_GPU", status.get("gpu", "")),
        log=env.get("GRID_LOG", status.get("log", "")),
        dry_run=env.get("GRID_DRY_RUN", "0") == "1",
    )
    if state:
        status["state"] = state
    if stage:
        status["stage"] = stage
    if seconds:
        status.setdefault("stage_seconds", {}).update({k: round(v, 1) for k, v in seconds.items()})
    if skipped and skipped not in status.setdefault("skipped", []):
        status["skipped"].append(skipped)
    if run_dir:
        status["run_dir"] = run_dir
    if exit_code is not None:
        status["exit_code"] = exit_code
    if state == "failed" and status.get("log"):
        status["error"] = _tail(Path(status["log"]), 25)
    if state == "done":
        status["finished"] = _now()
        status["error"] = None
    status["updated"] = _now()
    _write_text(path, json.dumps(status, indent=2, sort_keys=True) + "\n")
    if state == "running" and status.get("log"):
        _index_log(name, Path(status["log"]), root)
    return status


def _index_log(name: str, log: Path, root: Path) -> None:
    """logs/<name>.out points at the log of the job that last worked on it."""
    link = root / "logs" / f"{name}.out"
    link.parent.mkdir(parents=True, exist_ok=True)
    target = log if log.is_absolute() else Path.cwd() / log
    temporary = link.with_name(f".{link.name}.{os.getpid()}")
    with contextlib.suppress(FileNotFoundError):
        temporary.unlink()
    temporary.symlink_to(target)
    temporary.replace(link)


def record_submission(entry: Mapping[str, Any], root: Path | None = None) -> None:
    path = (root or state_dir()) / "submissions.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(entry), sort_keys=True) + "\n")


def read_submissions(root: Path | None = None) -> list[dict[str, Any]]:
    path = (root or state_dir()) / "submissions.jsonl"
    if not path.is_file():
        return []
    entries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        with contextlib.suppress(ValueError):
            entries.append(json.loads(line))
    return entries


# --------------------------------------------------------------------------
# Slurm queries
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class QueuedJob:
    job_id: str
    name: str
    state: str
    partition: str = ""
    node: str = ""

    @property
    def active(self) -> bool:
        return self.state not in TERMINAL

    @property
    def target(self) -> str | None:
        return self.name[len(JOB_PREFIX) :] if self.name.startswith(JOB_PREFIX) else None


def parse_squeue(text: str) -> list[QueuedJob]:
    """Parse ``squeue -h -o '%i %j %T %P %N'``; a pending job has no node."""
    jobs = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        parts += [""] * (5 - len(parts))
        jobs.append(QueuedJob(*parts[:5]))
    return jobs


def queued_by_target(jobs: Iterable[QueuedJob]) -> dict[str, list[QueuedJob]]:
    grouped: dict[str, list[QueuedJob]] = {}
    for job in jobs:
        if job.target and job.active:
            grouped.setdefault(job.target, []).append(job)
    return grouped


def _user() -> str:
    return os.environ.get("USER") or getpass.getuser()


def squeue_jobs() -> list[QueuedJob]:
    result = subprocess.run(
        ["squeue", "-u", _user(), "-h", "-o", "%i %j %T %P %N"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        # Without the queue, "not queued" cannot be told from "squeue failed",
        # and submitting on a guess would double up jobs.
        raise GridError(f"squeue failed: {result.stderr.strip()}")
    return parse_squeue(result.stdout)


def squeue_liveness(returncode: int, stdout: str, stderr: str) -> str:
    """'alive', 'dead' or 'unknown' from ``squeue -h -j ID -o %T``."""
    if returncode == 0:
        states = [line.split()[0] for line in stdout.splitlines() if line.strip()]
        if not states:
            return "dead"
        # A finished job stays visible for MinJobAge, in its final state.
        return "dead" if all(state in TERMINAL for state in states) else "alive"
    if "invalid job id" in stderr.lower():
        return "dead"
    return "unknown"


def job_liveness(job_id: str, *, tries: int = 3) -> str:
    verdict = "unknown"
    for attempt in range(tries):
        try:
            result = subprocess.run(
                ["squeue", "-h", "-j", job_id, "-o", "%T"],
                capture_output=True,
                text=True,
                check=False,
                timeout=60,
            )
        except (OSError, subprocess.TimeoutExpired):
            verdict = "unknown"
        else:
            verdict = squeue_liveness(result.returncode, result.stdout, result.stderr)
        if verdict != "unknown":
            return verdict
        if attempt + 1 < tries:
            time.sleep(5)
    return verdict


def parse_sacct(text: str) -> dict[str, dict[str, str]]:
    """Parse ``sacct -X -n -P -o JobID,State,Elapsed,ExitCode``."""
    found = {}
    for line in text.splitlines():
        parts = line.split("|")
        if len(parts) < 4:
            continue
        found[parts[0]] = {
            "state": parts[1].split()[0] if parts[1] else "",
            "elapsed": parts[2],
            "exit_code": parts[3],
        }
    return found


def sacct_jobs(job_ids: Sequence[str]) -> dict[str, dict[str, str]]:
    if not job_ids:
        return {}
    # sacct refuses wide date ranges here, but -j needs none.
    try:
        result = subprocess.run(
            ["sacct", "-X", "-n", "-P", "-j", ",".join(job_ids), "-o", SACCT_FIELDS],
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
        )
    except (OSError, subprocess.TimeoutExpired):
        return {}
    return parse_sacct(result.stdout) if result.returncode == 0 else {}


# --------------------------------------------------------------------------
# The twin lock
# --------------------------------------------------------------------------


def lock_decision(
    owner: Mapping[str, Any] | None, job_id: str, liveness: str | None, age: float
) -> str:
    """What a job that found the lock already there does: yield, wait or takeover."""
    owner_id = str(owner.get("job_id", "")) if owner else ""
    if not owner_id:
        return "wait" if age < OWNER_GRACE else "takeover"
    if owner_id == str(job_id):
        # A requeued job keeps its id, and the lock is its own from before.
        return "takeover"
    if liveness == "dead":
        return "takeover"
    # Alive, or squeue could not say: never risk two jobs writing one model.
    return "yield"


def _owner_id(lock: Path) -> str:
    owner = _read_json(lock / "owner.json")
    return str(owner.get("job_id", "")) if owner else ""


def _remove_lock(lock: Path, tag: str) -> bool:
    """Rename then delete, so the name is free at once and atomically."""
    graveyard = lock.with_name(f".stale-{lock.name}-{tag}-{time.time_ns()}")
    try:
        lock.rename(graveyard)
    except FileNotFoundError:
        return False
    shutil.rmtree(graveyard, ignore_errors=True)
    return True


def _take_over(lock: Path, stale_owner: str, tag: str, now: Callable[[], float]) -> bool:
    """Remove a stale lock, unless someone else got there first.

    Two jobs can both judge one lock stale. Only the holder of the takeover
    guard may remove it, and only if the owner is still the one it judged,
    so the second never removes a lock the first has just taken.
    """
    guard = lock.with_name(f".takeover-{lock.name}")
    try:
        guard.mkdir()
    except FileExistsError:
        with contextlib.suppress(FileNotFoundError, OSError):
            if now() - guard.stat().st_mtime > OWNER_GRACE:
                guard.rmdir()
        return False
    try:
        if not lock.exists() or _owner_id(lock) != stale_owner:
            return False
        return _remove_lock(lock, tag)
    finally:
        with contextlib.suppress(OSError):
            guard.rmdir()


def acquire_lock(
    name: str,
    job_id: str,
    *,
    info: Mapping[str, str] | None = None,
    root: Path | None = None,
    liveness: Callable[[str], str] = job_liveness,
    sleep: Callable[[float], None] = time.sleep,
    now: Callable[[], float] = time.time,
    attempts: int = 60,
) -> tuple[str, str]:
    """Take locks/<name> for this job: ('acquired' | 'held' | 'error', message)."""
    locks = (root or state_dir()) / "locks"
    locks.mkdir(parents=True, exist_ok=True)
    lock = locks / name
    note = "acquired"
    for _ in range(attempts):
        try:
            lock.mkdir()
        except FileExistsError:
            pass
        else:
            record = {"job_id": str(job_id), "acquired": _now(), "host": socket.gethostname()}
            record.update(info or {})
            _write_text(lock / "owner.json", json.dumps(record, indent=2) + "\n")
            return "acquired", note
        owner = _read_json(lock / "owner.json")
        try:
            age = now() - lock.stat().st_mtime
        except FileNotFoundError:
            continue
        owner_id = str(owner.get("job_id", "")) if owner else ""
        verdict = liveness(owner_id) if owner_id and owner_id != str(job_id) else None
        decision = lock_decision(owner, job_id, verdict, age)
        if decision == "yield":
            return "held", f"lock held by job {owner_id} ({verdict})"
        if decision == "wait":
            sleep(5)
            continue
        if _take_over(lock, owner_id, str(job_id), now):
            note = f"took over the stale lock of job {owner_id or 'unknown'}"
        else:
            sleep(1)
    return "error", f"could not settle {lock} after {attempts} attempts"


def release_lock(name: str, job_id: str, root: Path | None = None) -> bool:
    lock = (root or state_dir()) / "locks" / name
    if _owner_id(lock) != str(job_id):
        return False
    return _remove_lock(lock, str(job_id))


def read_locks(root: Path | None = None) -> dict[str, dict[str, Any]]:
    locks = (root or state_dir()) / "locks"
    if not locks.is_dir():
        return {}
    return {
        path.name: _read_json(path / "owner.json") or {}
        for path in sorted(locks.iterdir())
        if path.is_dir() and not path.name.startswith(".")
    }


def clear_stale_lock(name: str, root: Path | None = None) -> str | None:
    """Remove locks/<name> if its owner is gone; return what was removed."""
    lock = (root or state_dir()) / "locks" / name
    if not lock.is_dir():
        return None
    owner_id = _owner_id(lock)
    try:
        age = time.time() - lock.stat().st_mtime
    except FileNotFoundError:
        return None
    verdict = job_liveness(owner_id) if owner_id else None
    if lock_decision({"job_id": owner_id} if owner_id else None, "", verdict, age) != "takeover":
        return None
    if _take_over(lock, owner_id, "login", time.time):
        return f"stale lock of job {owner_id or 'unknown'}"
    return None


# --------------------------------------------------------------------------
# Stage completeness, for the job script
# --------------------------------------------------------------------------


def checkpoint_complete(path: Path, kind: str) -> tuple[bool, str]:
    """Whether a pruned (or repaired) checkpoint was written to the end."""
    own = "calibration_provenance.json" if kind == "prune" else "repair.json"
    for required in ("config.json", "subnetwork.json", own):
        if not (path / required).is_file():
            return False, f"{path / required} missing"
    index = path / "model.safetensors.index.json"
    if index.is_file():
        payload = _read_json(index)
        shards = sorted(set((payload or {}).get("weight_map", {}).values()))
        if not shards:
            return False, f"{index} unreadable or empty"
        missing = [
            shard
            for shard in shards
            if not (path / shard).is_file() or (path / shard).stat().st_size == 0
        ]
        if missing:
            return False, f"{path} lacks {', '.join(missing)}"
    elif not (path / "model.safetensors").is_file():
        return False, f"{path} has no weights"
    return True, "complete"


def stage_complete(
    kind: str,
    *,
    checkpoint: Path,
    model_config: Path,
    marker: Path,
    subnetwork: Path | None = None,
) -> tuple[bool, str]:
    """Prune or repair is complete when everything it writes is there.

    The marker is the CLI's manifest, saved only when it exited 0. Both write
    every file before their last check (prune's achieved sparsity, repair's
    held-out loss), so the files alone would call a failed run done.
    """
    ok, reason = checkpoint_complete(checkpoint, kind)
    if not ok:
        return False, reason
    required = [model_config, marker] + ([subnetwork] if subnetwork else [])
    for path in required:
        if not path.is_file():
            return False, f"{path} missing"
    return True, "complete"


def generation_complete(run_dir: Path) -> tuple[bool, str]:
    manifest = _read_json(run_dir / "manifest.json")
    if manifest is None:
        return False, f"{run_dir} has no manifest"
    directions = manifest.get("suite", {}).get("data", {}).get("directions", [])
    missing = [d for d in directions if not (run_dir / "hyps" / f"{d}.jsonl").is_file()]
    if not directions or missing:
        return False, f"hypotheses missing for {', '.join(missing) or 'every direction'}"
    try:
        from mnlp_eval.runspec import RunPaths
    except Exception:
        return True, "every direction has hypotheses"
    if not RunPaths(run_dir).stage_completed("generate"):
        return False, "generation stage records incomplete"
    return True, "complete"


def scores_state(run_dir: Path, group: str) -> tuple[int, str]:
    """0 scored, 1 not scored, 2 scored before the latest hypotheses."""
    scores = run_dir / f"scores.{group}.json"
    if not scores.is_file():
        return 1, f"{scores} missing"
    hyps = list((run_dir / "hyps").glob("*.jsonl"))
    newest = max((path.stat().st_mtime for path in hyps), default=0.0)
    if scores.stat().st_mtime < newest:
        return 2, f"{scores} is older than the hypotheses"
    return 0, "complete"


def last_json_object(text: str) -> dict[str, Any] | None:
    """The last top-level JSON object printed, as the CLI prints its manifest."""
    lines = text.splitlines()
    for start in range(len(lines) - 1, -1, -1):
        if lines[start].startswith("{"):
            with contextlib.suppress(ValueError):
                payload = json.loads("\n".join(lines[start:]))
                if isinstance(payload, dict):
                    return payload
    return None


# --------------------------------------------------------------------------
# Status
# --------------------------------------------------------------------------

CATEGORIES = ("done", "running", "pending", "failed", "not-submitted")


def classify(
    status: Mapping[str, Any] | None, queued: Sequence[QueuedJob], submitted: bool
) -> tuple[str, str]:
    """(category, detail) for one target from its status file and the queue."""
    if status and status.get("dry_run"):
        # A --job-dry-run job only echoed its stages; nothing of the model exists.
        status = None
    if status and status.get("state") == "done":
        return "done", ""
    running = [job for job in queued if job.state in {"RUNNING", "COMPLETING", "CONFIGURING"}]
    if running:
        job = running[0]
        return "running", f"job {job.job_id} {job.partition} {job.node}"
    if queued:
        return "pending", " ".join(f"{job.job_id}:{job.partition}" for job in queued)
    if status:
        state = status.get("state")
        job_id = status.get("job_id", "?")
        if state == "failed":
            return "failed", f"{status.get('stage', '?')} failed, job {job_id}"
        if state == "running":
            return "failed", f"stale: job {job_id} left the queue during {status.get('stage')}"
        return "failed", f"lost: state {state!r} but nothing queued"
    if submitted:
        return "failed", "lost: submitted, nothing queued and no status written"
    return "not-submitted", ""


def gather(targets: Sequence[Target], jobs: Sequence[QueuedJob]) -> list[dict[str, Any]]:
    queued = queued_by_target(jobs)
    submitted = {entry.get("name") for entry in read_submissions()}
    rows = []
    for target in targets:
        status = read_status(target.name)
        category, detail = classify(status, queued.get(target.name, []), target.name in submitted)
        rows.append(
            {
                "target": target,
                "status": status or {},
                "category": category,
                "detail": detail,
                "queued": queued.get(target.name, []),
            }
        )
    return rows


def _elapsed(since: str | None) -> str:
    if not since:
        return "?"
    with contextlib.suppress(ValueError):
        seconds = (dt.datetime.now(dt.UTC) - dt.datetime.fromisoformat(since)).total_seconds()
        return f"{int(seconds // 3600)}h{int(seconds % 3600 // 60):02d}m"
    return "?"


def print_status(rows: Sequence[dict[str, Any]], *, verbose: bool) -> None:
    scopes = [*SCOPES, "repair"]
    counts = {scope: dict.fromkeys(CATEGORIES, 0) for scope in scopes}
    for row in rows:
        counts[row["target"].scope][row["category"]] += 1
    header = f"{'scope':<8}{'total':>7}" + "".join(f"{c:>15}" for c in CATEGORIES)
    print(header)
    totals = dict.fromkeys(CATEGORIES, 0)
    for scope in scopes:
        row_counts = counts[scope]
        if scope == "repair" and not any(row_counts.values()):
            continue
        for category in CATEGORIES:
            totals[category] += row_counts[category]
        cells = "".join(f"{row_counts[c]:>15}" for c in CATEGORIES)
        print(f"{scope:<8}{sum(row_counts.values()):>7}{cells}")
    print(f"{'all':<8}{sum(totals.values()):>7}" + "".join(f"{totals[c]:>15}" for c in CATEGORIES))

    running = [row for row in rows if row["category"] == "running"]
    if running:
        print("\nrunning:")
        for row in running:
            status = row["status"]
            stage = status.get("stage", "starting")
            gpu = status.get("gpu") or ""
            print(
                f"  {row['target'].name:<44} {row['detail']:<34} {stage:<14} "
                f"{_elapsed(status.get('started')):>6}  {gpu}"
            )
    pending = [row for row in rows if row["category"] == "pending"]
    if pending and verbose:
        print("\npending:")
        for row in pending:
            print(f"  {row['target'].name:<44} {row['detail']}")
    failed = [row for row in rows if row["category"] == "failed"]
    if failed:
        # A job that died before writing status is known only by its submissions.
        latest: dict[str, list[str]] = {}
        for submission in read_submissions():
            latest.setdefault(submission["name"], []).append(str(submission["job_id"]))
        for row in failed:
            job_id = row["status"].get("job_id")
            row["jobs"] = [str(job_id)] if job_id else latest.get(row["target"].name, [])[-2:]
        accounting = sacct_jobs([i for row in failed for i in row["jobs"] if i.isdigit()])
        print("\nfailed (resubmit with: grid.py resubmit-failed):")
        for row in failed:
            status = row["status"]
            slurm = " ".join(
                f"[{i} {accounting[i]['state']} after {accounting[i]['elapsed']}]"
                for i in row["jobs"]
                if i in accounting
            )
            print(f"  {row['target'].name:<44} {row['detail']} {slurm}".rstrip())
            lines = [line for line in status.get("error") or [] if line.strip()]
            log = status.get("log")
            if not lines and not log:
                logs = [
                    REPO / LOG_DIR / f"{JOB_PREFIX}{row['target'].name}-{i}.out"
                    for i in row["jobs"]
                ]
                log = next((str(path) for path in logs if path.is_file()), None)
                lines = [line for line in _tail(Path(log), 15) if line.strip()] if log else []
            for line in lines[-(15 if verbose else 3) :]:
                print(f"      | {line[:160]}")
            if log:
                print(f"      log: {log}")
    stale = []
    live = {row["target"].name for row in rows if row["category"] in {"running", "pending"}}
    for name, holder in read_locks().items():
        if name not in live:
            stale.append(f"{name} (job {holder.get('job_id', '?')})")
    if stale:
        print("\nlocks with no queued job (cleared by resubmit-failed):")
        for line in stale:
            print(f"  {line}")


# --------------------------------------------------------------------------
# Submission
# --------------------------------------------------------------------------

_TIME = re.compile(r"^(\d+-)?\d{1,2}:\d{2}:\d{2}$")


def parse_partitions(text: str) -> list[str]:
    partitions = [item.strip() for item in text.split(",") if item.strip()]
    unknown = sorted(set(partitions) - set(PARTITION_CPUS))
    if unknown or not partitions:
        known = ", ".join(PARTITION_CPUS)
        raise GridError(f"unknown partition(s) {', '.join(unknown) or '(none)'}; known: {known}")
    return partitions


def sbatch_command(
    target: Target,
    partition: str,
    time_limit: str,
    *,
    state: Path,
    prune_out: str,
    job_dry_run: bool = False,
) -> list[str]:
    config_var = "PRUNE_CONFIG" if target.mode == "prune" else "REPAIR_CONFIG"
    exports = [
        f"GRID_NAME={target.name}",
        f"GRID_MODE={target.mode}",
        f"{config_var}={target.config}",
        f"MODEL_CONFIG={target.model_config}",
        f"SUITE_CONFIG={SUITE}",
        f"GRID_STATE_DIR={state}",
        f"PRUNE_OUT={prune_out}",
        f"GRID_DRY_RUN={1 if job_dry_run else 0}",
    ]
    # A comma inside a value would be read as the start of the next assignment.
    bad = [item for item in exports if "," in item]
    if bad:
        raise GridError(f"cannot pass through --export: {', '.join(bad)}")
    return [
        "sbatch",
        "--parsable",
        f"--job-name={JOB_PREFIX}{target.name}",
        f"--partition={partition}",
        f"--cpus-per-task={PARTITION_CPUS[partition]}",
        "--gpus=1",
        "--nodes=1",
        f"--time={time_limit}",
        f"--output={LOG_DIR}/%x-%j.out",
        "--export=" + ",".join(["ALL", *exports]),
        SBATCH_SCRIPT,
    ]


def submit_targets(
    targets: Sequence[Target],
    *,
    partitions: Sequence[str],
    time_limit: str | None,
    dry_run: bool,
    job_dry_run: bool = False,
    reason: str = "submit",
) -> int:
    """Submit one twin per partition for each target; return how many targets got a job."""
    state = state_dir().resolve()
    prune_out = os.environ.get("PRUNE_OUT") or f"/scratch-shared/{_user()}/checkpoints"
    if not dry_run:
        (REPO / LOG_DIR).mkdir(parents=True, exist_ok=True)
        state.mkdir(parents=True, exist_ok=True)
    submitted = 0
    for target in targets:
        limit = time_limit or DEFAULT_TIME[target.mode]
        ids = []
        for partition in partitions:
            command = sbatch_command(
                target, partition, limit, state=state, prune_out=prune_out, job_dry_run=job_dry_run
            )
            if dry_run:
                print(shlex.join(command))
                continue
            result = subprocess.run(command, cwd=REPO, capture_output=True, text=True, check=False)
            if result.returncode != 0:
                print(f"  {target.name}: sbatch on {partition} failed: {result.stderr.strip()}")
                continue
            job_id = result.stdout.strip().split(";")[0]
            ids.append(f"{partition}:{job_id}")
            record_submission(
                {
                    "time": _now(),
                    "name": target.name,
                    "mode": target.mode,
                    "partition": partition,
                    "job_id": job_id,
                    "time_limit": limit,
                    "reason": reason,
                    "job_dry_run": job_dry_run,
                }
            )
        if ids:
            submitted += 1
            print(f"  {target.name:<44} {' '.join(ids)}")
    return submitted


def plan_submission(
    targets: Sequence[Target],
    jobs: Sequence[QueuedJob],
    *,
    want: str,
    check_inputs: bool,
) -> tuple[list[Target], dict[str, list[str]]]:
    """Split targets into those to submit and those skipped, by reason.

    ``want`` is 'new' for submit (anything not done, queued or failed) or
    'failed' for resubmit-failed.
    """
    queued = queued_by_target(jobs)
    submitted = {entry.get("name") for entry in read_submissions()}
    chosen: list[Target] = []
    skipped: dict[str, list[str]] = {}
    for target in targets:
        category, _ = classify(
            read_status(target.name), queued.get(target.name, []), target.name in submitted
        )
        eligible = category == "failed" if want == "failed" else category == "not-submitted"
        if not eligible:
            skipped.setdefault(category, []).append(target.name)
            continue
        if check_inputs:
            problem = preflight(target)
            if problem:
                skipped.setdefault(f"not ready: {problem}", []).append(target.name)
                continue
        chosen.append(target)
    return chosen, skipped


def print_skipped(skipped: Mapping[str, Sequence[str]], *, verbose: bool = False) -> None:
    for reason, names in skipped.items():
        shown = ", ".join(names[:4] if not verbose else names)
        more = f" and {len(names) - 4} more" if len(names) > 4 and not verbose else ""
        print(f"  skipped {len(names)} ({reason}): {shown}{more}")


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------


def command_configs(_args: argparse.Namespace) -> int:
    models = build_grid()
    names = [model.name for model in models]
    if len(set(names)) != len(names):
        raise GridError("duplicate model names in the grid")
    specs = [model.prune_spec() for model in models]
    for warning in validate_specs(specs):
        print(f"warning: {warning}", file=sys.stderr)
    written = write_configs(REPO, models)
    extra = sorted(
        path.name
        for path in (REPO / PRUNE_CONFIG_DIR).glob("*.yaml")
        if path.stem not in set(names)
    )
    if extra:
        print(f"warning: not in the grid, left alone: {', '.join(extra)}", file=sys.stderr)
    print(f"wrote {len(written)} prune configs to {PRUNE_CONFIG_DIR}/ and {MANIFEST}")
    return 0


def _check_time(value: str | None) -> str | None:
    if value is not None and not _TIME.match(value):
        raise GridError(f"--time {value!r} is not [D-]HH:MM:SS")
    return value


def command_submit(args: argparse.Namespace) -> int:
    partitions = parse_partitions(args.partitions)
    time_limit = _check_time(args.time)
    targets = select(grid_targets(), only=args.only, match=args.match, scope=args.scope)
    chosen, skipped = plan_submission(
        targets, squeue_jobs(), want="new", check_inputs=not args.no_preflight
    )
    if args.limit is not None:
        chosen = chosen[: args.limit]
    print(f"{len(targets)} selected; submitting {len(chosen)} on {', '.join(partitions)}")
    count = submit_targets(
        chosen,
        partitions=partitions,
        time_limit=time_limit,
        dry_run=args.dry_run,
        job_dry_run=args.job_dry_run,
    )
    print_skipped(skipped, verbose=args.verbose)
    if "failed" in skipped:
        print("  failed models are left for: grid.py resubmit-failed")
    verb = "would submit" if args.dry_run else "submitted"
    print(
        f"{verb} {len(chosen) if args.dry_run else count} model(s), {len(partitions)} twin(s) each"
    )
    return 0


def command_status(args: argparse.Namespace) -> int:
    try:
        jobs = squeue_jobs()
    except GridError as exc:
        print(f"warning: {exc}; queue state unknown", file=sys.stderr)
        jobs = []
    rows = gather(grid_targets() + repair_targets(), jobs)
    if args.json:
        payload = [
            {
                "name": row["target"].name,
                "mode": row["target"].mode,
                "scope": row["target"].scope,
                "category": row["category"],
                "detail": row["detail"],
                "status": row["status"],
            }
            for row in rows
        ]
        print(json.dumps(payload, indent=2))
        return 0
    print_status(rows, verbose=args.verbose)
    return 0


def command_resubmit_failed(args: argparse.Namespace) -> int:
    partitions = parse_partitions(args.partitions)
    time_limit = _check_time(args.time)
    targets = select(
        grid_targets() + repair_targets(), only=args.only, match=args.match, scope=args.scope
    )
    chosen, skipped = plan_submission(
        targets, squeue_jobs(), want="failed", check_inputs=not args.no_preflight
    )
    if args.limit is not None:
        chosen = chosen[: args.limit]
    for target in chosen:
        if args.dry_run:
            print(f"  would clear {target.name}")
            continue
        removed = clear_stale_lock(target.name)
        if removed:
            print(f"  {target.name}: removed {removed}")
        status = read_status(target.name)
        if status:
            # The failure stays readable under previous_failures.
            failure = {k: status.get(k) for k in ("job_id", "state", "stage", "exit_code", "error")}
            status.setdefault("previous_failures", []).append(failure)
            status.update(state="queued", error=None, updated=_now())
            _write_text(
                status_path(target.name), json.dumps(status, indent=2, sort_keys=True) + "\n"
            )
    print(f"resubmitting {len(chosen)} failed target(s)")
    count = submit_targets(
        chosen,
        partitions=partitions,
        time_limit=time_limit,
        dry_run=args.dry_run,
        reason="resubmit-failed",
    )
    print_skipped({k: v for k, v in skipped.items() if k != "not-submitted"}, verbose=args.verbose)
    done = len(chosen) if args.dry_run else count
    print(f"{'would resubmit' if args.dry_run else 'resubmitted'} {done}")
    return 0


def command_repairs(args: argparse.Namespace) -> int:
    import yaml

    partitions = parse_partitions(args.partitions)
    time_limit = _check_time(args.time) or DEFAULT_TIME["repair"]
    grid_names = {target.name for target in grid_targets()}
    for source in args.source:
        if source not in grid_names and not (REPO / MODEL_CONFIG_DIR / f"{source}.yaml").is_file():
            raise GridError(f"{source} is neither a grid model nor an existing model config")
    names = []
    scopes = {entry["name"]: entry for entry in _read_json(REPO / MANIFEST)["models"]}
    for source in args.source:
        spec = repair_spec(source, args.data)
        entry = scopes.get(source)
        if entry is not None and entry["scope"] != "multi":
            # A specialist is repaired on its own directions only, so it stays
            # a specialist; the multi models see every direction, as ALMA did.
            spec["directions"] = list(entry["pruned_directions"])
        try:
            from mnlp_eval.prune.repair import RepairSpec
        except Exception as exc:
            print(f"warning: RepairSpec not importable ({exc}); not validated", file=sys.stderr)
        else:
            RepairSpec.from_dict(spec)
        scope_note = (
            f" on its own directions ({', '.join(spec['directions'])})"
            if spec.get("directions")
            else ""
        )
        header = (
            f"# LoRA repair of {source}{scope_note}, ALMA's recipe (RepairSpec defaults). "
            "Written by scripts/grid.py repairs.\n"
        )
        path = REPO / REPAIR_CONFIG_DIR / f"{spec['name']}.yaml"
        _write_text(path, header + yaml.safe_dump(spec, sort_keys=False))
        print(f"wrote {path.relative_to(REPO)}")
        names.append(spec["name"])
    targets = [target for target in repair_targets() if target.name in set(names)]
    chosen, skipped = plan_submission(
        targets, squeue_jobs(), want="new", check_inputs=not args.no_preflight
    )
    count = submit_targets(
        chosen, partitions=partitions, time_limit=time_limit, dry_run=args.dry_run
    )
    print_skipped(skipped, verbose=True)
    done = len(chosen) if args.dry_run else count
    print(f"{'would submit' if args.dry_run else 'submitted'} {done} repair(s)")
    return 0


def adopt(target: Target, *, root: Path = REPO, prune_out: Path | None = None) -> list[str]:
    """Accept a checkpoint written outside the grid, such as a canary's, as this target's.

    Only by hand: a canary may predate a fix, and the job script never adopts.
    Writes the stage marker the job looks for; marks the model done as well
    if its run directory is already generated and scored. Returns what it did.
    """
    out = prune_out or Path(os.environ.get("PRUNE_OUT") or f"/scratch-shared/{_user()}/checkpoints")
    checkpoint = out / target.name
    ok, reason = checkpoint_complete(checkpoint, target.mode)
    if not ok:
        raise GridError(f"{target.name}: cannot adopt, {reason}")
    model_config = root / target.model_config
    required = [model_config]
    if target.mode == "prune":
        required.append(root / "subnetworks" / f"{target.name}.json")
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise GridError(f"{target.name}: cannot adopt, missing {', '.join(missing)}")
    record: dict[str, Any] = {"adopted": True, "adopted_at": _now(), "checkpoint": str(checkpoint)}
    with contextlib.suppress(Exception):
        from mnlp_eval.analysis.subnetwork import load_subnetwork

        record["unit_sparsity"] = round(
            load_subnetwork(checkpoint / "subnetwork.json").overall_sparsity, 6
        )
    marker = state_dir() / "status" / f"{target.name}.{target.mode}.json"
    _write_text(marker, json.dumps(record, indent=2) + "\n")
    done = [f"adopted {checkpoint} (unit sparsity {record.get('unit_sparsity', '?')})"]

    command = [str(root / ".venv/bin/mnlp-eval"), "run-dir", "--model", target.model_config]
    try:
        result = subprocess.run(
            [*command, "--suite", SUITE], cwd=root, capture_output=True, text=True, check=False
        )
    except OSError:
        run_dir = None
    else:
        run_dir = Path(result.stdout.strip()) if result.returncode == 0 else None
    if run_dir is None:
        return [*done, "run directory not resolved; submit to generate and score"]
    run_path = run_dir if run_dir.is_absolute() else root / run_dir
    scored = all(scores_state(run_path, group)[0] == 0 for group in ("surface", "neural"))
    if generation_complete(run_path)[0] and scored:
        status = read_status(target.name) or {"name": target.name}
        status.update(
            state="done", stage="done", mode=target.mode, run_dir=str(run_dir), adopted=True
        )
        status.update(dry_run=False, updated=_now(), finished=_now())
        _write_text(status_path(target.name), json.dumps(status, indent=2, sort_keys=True) + "\n")
        return [*done, f"{run_dir} is generated and scored; marked done"]
    return [*done, f"{run_dir} incomplete; submit to finish it"]


def command_adopt(args: argparse.Namespace) -> int:
    targets = select(grid_targets() + repair_targets(), only=args.name)
    for target in targets:
        for line in adopt(target):
            print(f"{target.name}: {line}")
    return 0


def command_lock(args: argparse.Namespace) -> int:
    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id:
        raise GridError("lock: SLURM_JOB_ID is not set")
    info = {
        "partition": os.environ.get("SLURM_JOB_PARTITION", ""),
        "node": os.environ.get("SLURMD_NODENAME", ""),
    }
    outcome, message = acquire_lock(args.name, job_id, info=info)
    print(f"lock {args.name}: {message}")
    return {"acquired": 0, "held": LOCK_HELD}.get(outcome, 1)


def command_unlock(args: argparse.Namespace) -> int:
    job_id = os.environ.get("SLURM_JOB_ID", "")
    released = release_lock(args.name, job_id)
    print(f"lock {args.name}: {'released' if released else 'not ours, left alone'}")
    return 0


def command_mark(args: argparse.Namespace) -> int:
    seconds = {}
    for item in args.seconds or []:
        stage, _, value = item.partition("=")
        seconds[stage] = float(value)
    mark(
        args.name,
        state=args.state,
        stage=args.stage,
        seconds=seconds,
        skipped=args.skipped,
        run_dir=args.run_dir,
        exit_code=args.exit_code,
    )
    return 0


def command_check(args: argparse.Namespace) -> int:
    if args.kind in {"prune", "repair"}:
        ok, reason = stage_complete(
            args.kind,
            checkpoint=Path(args.checkpoint),
            model_config=Path(args.model_config),
            marker=Path(args.marker),
            subnetwork=Path(args.subnetwork) if args.subnetwork else None,
        )
        code = 0 if ok else 1
    elif args.kind == "generate":
        ok, reason = generation_complete(Path(args.run_dir))
        code = 0 if ok else 1
    else:
        code, reason = scores_state(Path(args.run_dir), args.group)
    print(f"check {args.kind}: {reason}")
    return code


def command_save_manifest(args: argparse.Namespace) -> int:
    text = Path(args.stdout).read_text(encoding="utf-8", errors="replace")
    payload = last_json_object(text)
    if payload is None:
        payload = {"unparsed_stdout": text[-4000:]}
        print(f"warning: no JSON manifest in {args.stdout}", file=sys.stderr)
    _write_text(Path(args.out), json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="grid.py", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("configs", help="write configs/prune/grid/*.yaml and configs/grid/manifest.json")

    def submission_options(command: argparse.ArgumentParser, *, scopes: Sequence[str]) -> None:
        command.add_argument("--only", nargs="+", action="extend", metavar="NAME")
        command.add_argument("--match", metavar="REGEX", help="re.search on the model name")
        command.add_argument("--scope", choices=scopes)
        command.add_argument("--limit", type=int, help="at most N models, in priority order")
        command.add_argument("--partitions", default=DEFAULT_PARTITIONS)
        command.add_argument("--time", help="time limit, [D-]HH:MM:SS")
        command.add_argument("--dry-run", action="store_true", help="print sbatch commands only")
        command.add_argument(
            "--no-preflight", action="store_true", help="submit even if inputs are missing"
        )
        command.add_argument("--verbose", action="store_true", help="list every skipped name")

    submit = sub.add_parser("submit", help="submit twin jobs for models not done or queued")
    submission_options(submit, scopes=SCOPES)
    submit.add_argument(
        "--job-dry-run",
        action="store_true",
        help="the jobs take the lock and echo their stages instead of running them",
    )

    status = sub.add_parser("status", help="done, running, pending, failed, not submitted")
    status.add_argument("--verbose", action="store_true")
    status.add_argument("--json", action="store_true", help="one record per target")

    resubmit = sub.add_parser("resubmit-failed", help="clear stale locks and resubmit failures")
    submission_options(resubmit, scopes=(*SCOPES, "repair"))

    repairs = sub.add_parser("repairs", help="write LoRA repair configs and submit them")
    repairs.add_argument("--source", action="append", required=True, metavar="NAME")
    repairs.add_argument("--data", default=REPAIR_DATA)
    repairs.add_argument("--partitions", default=DEFAULT_PARTITIONS)
    repairs.add_argument("--time", help=f"default {DEFAULT_TIME['repair']}")
    repairs.add_argument("--dry-run", action="store_true")
    repairs.add_argument("--no-preflight", action="store_true")

    adopt_cmd = sub.add_parser(
        "adopt", help="accept checkpoints written outside the grid (e.g. canaries) as done"
    )
    adopt_cmd.add_argument("name", nargs="+")

    # Called by slurm/grid_pipeline.sbatch.
    lock = sub.add_parser("lock", help="(job) take the twin lock; exit 10 if a live twin has it")
    lock.add_argument("name")
    unlock = sub.add_parser("unlock", help="(job) release the twin lock if this job holds it")
    unlock.add_argument("name")
    mark_cmd = sub.add_parser("mark", help="(job) update status/<name>.json")
    mark_cmd.add_argument("name")
    mark_cmd.add_argument("--state", choices=("running", "done", "failed"))
    mark_cmd.add_argument("--stage")
    mark_cmd.add_argument("--seconds", action="append", metavar="STAGE=N")
    mark_cmd.add_argument("--skipped", metavar="STAGE")
    mark_cmd.add_argument("--run-dir")
    mark_cmd.add_argument("--exit-code", type=int)
    check = sub.add_parser("check", help="(job) exit 0 if a stage is complete")
    check.add_argument("kind", choices=("prune", "repair", "generate", "score"))
    check.add_argument("--checkpoint")
    check.add_argument("--model-config")
    check.add_argument("--marker")
    check.add_argument("--subnetwork")
    check.add_argument("--run-dir")
    check.add_argument("--group")
    save = sub.add_parser("save-manifest", help="(job) keep the JSON a CLI command printed")
    save.add_argument("--stdout", required=True)
    save.add_argument("--out", required=True)
    return parser


COMMANDS: dict[str, Callable[[argparse.Namespace], int]] = {
    "configs": command_configs,
    "submit": command_submit,
    "status": command_status,
    "resubmit-failed": command_resubmit_failed,
    "repairs": command_repairs,
    "lock": command_lock,
    "unlock": command_unlock,
    "mark": command_mark,
    "check": command_check,
    "save-manifest": command_save_manifest,
    "adopt": command_adopt,
}


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return COMMANDS[args.command](args)
    except GridError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
