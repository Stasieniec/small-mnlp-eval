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

    @property
    def stages_dir(self) -> Path:
        return self.root / "stages"

    def stage_path(self, stage: str, key: str | None = None) -> Path:
        """Path of one stage record.

        Stage reports live in their own files rather than inside the manifest.
        Merging them into a single mutable document meant a read-modify-write
        with no lock, and concurrent Slurm array tasks lost records in 60 of 60
        measured trials. One file per stage, and per direction where a stage is
        sharded, removes the shared mutable state entirely.
        """
        name = f"{stage}.{key}.json" if key else f"{stage}.json"
        return self.stages_dir / name

    def hyps_jsonl(self, direction: Direction | str) -> Path:
        return self.hyps_dir / f"{direction}.jsonl"

    def hyps_text(self, direction: Direction | str) -> Path:
        return self.hyps_dir / f"{direction}.txt"

    def scores(self, group: str) -> Path:
        return self.root / f"scores.{group}.json"

    def ensure(self) -> None:
        self.hyps_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.stages_dir.mkdir(parents=True, exist_ok=True)

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
        }
        self.ensure()
        # Written once and never rewritten. The manifest is the run's identity,
        # and identity does not change; anything that accumulates during a run
        # goes to stages/ instead.
        atomic_write_json(self.manifest, manifest)
        return manifest

    def record_stage(
        self, stage: str, payload: dict[str, Any], key: str | None = None
    ) -> dict[str, Any]:
        """Write one stage record. Never touches the manifest."""
        entry = dict(payload)
        entry.setdefault("recorded_at", utc_now())
        if key:
            entry.setdefault("key", key)
        self.stages_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(self.stage_path(stage, key), entry)
        return entry

    def stage_records(self, stage: str) -> dict[str, dict[str, Any]]:
        """Every record for one stage, keyed by shard (``""`` when unsharded)."""
        if not self.stages_dir.is_dir():
            return {}
        records: dict[str, dict[str, Any]] = {}
        for path in sorted(self.stages_dir.glob(f"{stage}*.json")):
            parts = path.name[: -len(".json")].split(".", 1)
            if parts[0] != stage:
                continue
            try:
                records[parts[1] if len(parts) > 1 else ""] = read_json(path)
            except (OSError, ValueError):
                continue
        return records

    def stages(self) -> dict[str, Any]:
        """All stage records, for display and for the report."""
        collected: dict[str, Any] = {}
        for stage in ("generate", "bench", "score"):
            records = self.stage_records(stage)
            if not records:
                continue
            collected[stage] = records[""] if set(records) == {""} else records
        return collected

    def suite_directions(self) -> list[str]:
        """The directions this run's suite covers, from its manifest."""
        try:
            manifest = self.read_manifest()
        except FileNotFoundError:
            return []
        return list(manifest.get("suite", {}).get("data", {}).get("directions", []))

    def stage_completed(self, stage: str, key: str | None = None) -> bool:
        """Whether a stage finished.

        For ``generate``, complete means every direction in the suite reported
        a completed record. Checking only that the records present are
        completed would call a half-finished shard set done, which is how a
        partially generated run got scored and reported as whole.
        """
        records = self.stage_records(stage)
        if key is not None:
            return bool(records.get(key, {}).get("status") == "completed")
        if not records:
            return False
        if not all(entry.get("status") == "completed" for entry in records.values()):
            return False
        if stage == "generate":
            expected = set(self.suite_directions())
            if expected:
                return expected <= self.generated_directions()
        return True

    def generated_directions(self) -> set[str]:
        """Directions whose generation finished, from the stage records."""
        found: set[str] = set()
        for shard, entry in self.stage_records("generate").items():
            if entry.get("status") != "completed":
                continue
            if shard:
                found.add(shard)
            found.update(entry.get("directions", {}))
        return found


def discover_runs(root: Path) -> Iterator[RunPaths]:
    """Yield every run directory under ``root``, sorted by slug."""
    if not root.is_dir():
        return
    for candidate in sorted(root.iterdir()):
        if (candidate / MANIFEST_NAME).is_file():
            yield RunPaths(candidate)
