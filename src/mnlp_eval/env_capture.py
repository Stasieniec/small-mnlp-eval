"""Capture the environment a run happened in.

Efficiency numbers from a shared cluster are only comparable if you know what
they were measured on. Clock throttling, a driver upgrade between job
submissions, or a different attention backend all move latency by more than
some compression methods do, so the run records enough state to notice.

Nothing here may fail. A run on a machine without CUDA, without git, or without
torch still needs to produce an env record.
"""

from __future__ import annotations

import os
import platform
import shutil
import socket
import subprocess
import sys
from importlib import metadata
from typing import Any

__all__ = ["capture_environment", "package_versions"]

#: Packages worth recording. Absent ones are reported as null rather than
#: omitted, so a diff between two env files lines up.
_TRACKED_PACKAGES = (
    "mnlp-eval",
    "torch",
    "transformers",
    "tokenizers",
    "accelerate",
    "datasets",
    "peft",
    "bitsandbytes",
    "optimum",
    "vllm",
    "numpy",
    "sacrebleu",
    "unbabel-comet",
    "pytorch-lightning",
    "py3langid",
)

_SLURM_KEYS = (
    "SLURM_JOB_ID",
    "SLURM_JOB_NAME",
    "SLURM_ARRAY_JOB_ID",
    "SLURM_ARRAY_TASK_ID",
    "SLURM_JOB_PARTITION",
    "SLURM_JOB_NODELIST",
    "SLURM_GPUS_ON_NODE",
    "SLURM_CPUS_PER_TASK",
)

_HF_KEYS = (
    "HF_HOME",
    "HF_HUB_OFFLINE",
    "TRANSFORMERS_OFFLINE",
    "HF_DATASETS_OFFLINE",
    "CUDA_VISIBLE_DEVICES",
    "TOKENIZERS_PARALLELISM",
    "PYTORCH_CUDA_ALLOC_CONF",
)


def package_versions() -> dict[str, str | None]:
    """Return installed versions of the tracked packages."""
    versions: dict[str, str | None] = {}
    for name in _TRACKED_PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _run(command: list[str]) -> str | None:
    if shutil.which(command[0]) is None:
        return None
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=15, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def _git_state() -> dict[str, Any]:
    sha = _run(["git", "rev-parse", "HEAD"])
    if sha is None:
        return {"available": False}
    status = _run(["git", "status", "--porcelain"])
    return {
        "available": True,
        "commit": sha,
        "branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "dirty": bool(status),
        "dirty_files": len(status.splitlines()) if status else 0,
    }


def _torch_state() -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {"available": False}

    state: dict[str, Any] = {
        "available": True,
        "version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version()
        if torch.backends.cudnn.is_available()
        else None,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "devices": [],
    }
    if not torch.cuda.is_available():
        return state
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        state["devices"].append(
            {
                "index": index,
                "name": props.name,
                "total_memory_bytes": props.total_memory,
                "capability": f"{props.major}.{props.minor}",
                "multi_processor_count": props.multi_processor_count,
            }
        )
    return state


def _nvidia_smi_state() -> dict[str, Any]:
    """Driver version plus the clock and thermal state at capture time."""
    query = (
        "driver_version,name,memory.total,clocks.sm,clocks.max.sm,"
        "temperature.gpu,power.draw,power.limit,clocks_throttle_reasons.active"
    )
    raw = _run(["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader"])
    if raw is None:
        return {"available": False}
    labels = [
        "driver_version",
        "name",
        "memory_total",
        "clock_sm",
        "clock_sm_max",
        "temperature_c",
        "power_draw_w",
        "power_limit_w",
        "throttle_reasons",
    ]
    devices = []
    for line in raw.splitlines():
        values = [part.strip() for part in line.split(",")]
        if len(values) != len(labels):
            continue
        devices.append(dict(zip(labels, values, strict=True)))
    return {"available": bool(devices), "devices": devices}


def capture_environment(*, include_pip_freeze: bool = True) -> dict[str, Any]:
    """Return a JSON-safe snapshot of the current execution environment."""
    record: dict[str, Any] = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "python": {
            "version": sys.version.split()[0],
            "executable": sys.executable,
            "implementation": platform.python_implementation(),
        },
        "packages": package_versions(),
        "git": _git_state(),
        "torch": _torch_state(),
        "nvidia_smi": _nvidia_smi_state(),
        "slurm": {key: os.environ[key] for key in _SLURM_KEYS if key in os.environ},
        "hf_env": {key: os.environ[key] for key in _HF_KEYS if key in os.environ},
    }
    if include_pip_freeze:
        record["installed_distributions"] = _installed_distributions()
    return record


def _installed_distributions() -> list[str]:
    """Every installed distribution and its version.

    Read from importlib.metadata rather than by shelling out to pip. All three
    environments are created with ``uv venv``, which does not install pip, so
    ``python -m pip freeze`` failed and this field was silently null in every
    env.json on the cluster. For a framework whose purpose is knowing what a
    number was measured on, that was the one field that would answer which
    torch, CUDA and bitsandbytes produced a given latency.
    """
    entries: set[str] = set()
    for distribution in metadata.distributions():
        name = distribution.metadata["Name"]
        if not name:
            continue
        entries.add(f"{name}=={distribution.version}")
    return sorted(entries, key=str.lower)
