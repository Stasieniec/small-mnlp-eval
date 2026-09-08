"""Run directory layout and the manifest that makes a run self-describing.

A run directory is the unit of exchange between stages and between machines. It
must be interpretable on its own, months later, without the config files that
produced it, so the manifest carries the fully resolved configuration rather
than a pointer to it.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mnlp_eval.artifacts import atomic_write_json, read_json
from mnlp_eval.config import RunConfig, _plain
from mnlp_eval.languages import Direction

__all__ = ["MANIFEST_NAME", "SCHEMA_VERSION", "RunPaths", "discover_runs", "utc_now"]

MANIFEST_NAME = "manifest.json"
SCHEMA_VERSION = 1


def utc_now() -> str:
    """Return the current UTC time as an ISO 8601 string with seconds."""
    return dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat()


@dataclass(frozen=True)
class RunPaths:
    """Every path inside one run directory."""

    root: Path

    @classmethod
    def for_config(cls, config: RunConfig) -> RunPaths:
        return cls(config.directory)

    @property
    def manifest(self) -> Path:
        return self.root / MANIFEST_NAME

    @property
    def env(self) -> Path:
        return self.root / "env.json"

    @property
    def bench(self) -> Path:
        return self.root / "bench.json"

    @property
    def hyps_dir(self) -> Path:
        return self.root / "hyps"

    @property
    def logs_dir(self) -> Path:
        return self.root / "logs"

    def hyps_jsonl(self, direction: Direction | str) -> Path:
        return self.hyps_dir / f"{direction}.jsonl"

    def hyps_text(self, direction: Direction | str) -> Path:
        return self.hyps_dir / f"{direction}.txt"

    def scores(self, group: str) -> Path:
        return self.root / f"scores.{group}.json"

    def ensure(self) -> None:
        self.hyps_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Manifest
    # ------------------------------------------------------------------

    def read_manifest(self) -> dict[str, Any]:
        if not self.manifest.is_file():
            msg = f"no manifest at {self.manifest}; run 'mnlp-eval generate' first"
            raise FileNotFoundError(msg)
        return read_json(self.manifest)

    def init_manifest(self, config: RunConfig) -> dict[str, Any]:
        """Create the manifest, or return the existing one unchanged.

        Refuses to reuse a directory whose manifest describes a different run,
        which would otherwise happen if someone hand-edited a run slug.
        """
        if self.manifest.is_file():
            existing = self.read_manifest()
            if existing.get("run_id") != config.run_id:
                msg = (
                    f"{self.root} already holds run {existing.get('run_id')!r}, "
                    f"which is not {config.run_id!r}. Refusing to overwrite."
                )
                raise RuntimeError(msg)
            return existing
        manifest: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "run_id": config.run_id,
            "slug": config.slug,
            "created_at": utc_now(),
            "model": _plain(config.model),
            "suite": _plain(config.suite),
            "identity": config.identity(),
            "prompt_fingerprint": config.model.prompt_fingerprint,
            "decode_summary": config.suite.decode.describe(),
            "stages": {},
        }
        self.ensure()
        atomic_write_json(self.manifest, manifest)
        return manifest

    def record_stage(self, stage: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Merge a stage report into the manifest and persist it."""
        manifest = self.read_manifest()
        stages = manifest.setdefault("stages", {})
        entry = dict(stages.get(stage) or {})
        entry.update(payload)
        entry.setdefault("recorded_at", utc_now())
        stages[stage] = entry
        atomic_write_json(self.manifest, manifest)
        return manifest

    def stage_completed(self, stage: str) -> bool:
        try:
            manifest = self.read_manifest()
        except FileNotFoundError:
            return False
        entry = (manifest.get("stages") or {}).get(stage) or {}
        return bool(entry.get("status") == "completed")


def discover_runs(root: Path) -> Iterator[RunPaths]:
    """Yield every run directory under ``root``, sorted by slug."""
    if not root.is_dir():
        return
    for candidate in sorted(root.iterdir()):
        if (candidate / MANIFEST_NAME).is_file():
            yield RunPaths(candidate)
