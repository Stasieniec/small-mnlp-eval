#!/usr/bin/env python3
"""Aggregate the ALMA-7B structured-pruning grid into tables, plots and JSON.

Usage::

    ./.venv/bin/python scripts/grid_report.py [--out results/grid-2026-10-06] \
        [--runs-root runs] [--manifest configs/grid/manifest.json] [--bootstrap 1000]

The grid is method {slimgpt, flap} x calibration text {ref, gen} x sparsity
{20, 30, 40} x scope {multi, pair, dir}. The grid manifest (written by
``scripts/grid.py configs``) names the systems and their attributes; run
directories under ``--runs-root`` hold the scores. The dense ``alma-7b`` run is
the baseline, and any ``<source>-lora`` run is treated as a LoRA repair of
``<source>``.

A model is complete when every direction it is meant to be evaluated on is
scored (BLEU, chrF++ and COMET): all ten for dense, multi models and their
repairs, the manifest's ``pruned_directions`` for pair and dir models (a
``-lora`` system inherits its source's). On the pilot suite every model has
all ten; on the full suite (``--suite alma10-greedy``) specialists are only
evaluated on their own directions. Specialists that do have all ten feed the
transfer section; the others are left out of it.

MetricX-24 (``scores.metricx.json``, an error score in [0, 25], lower is
better) is reported next to COMET where it exists and as ``-`` elsewhere.
COMET stays the primary metric.

Every contrast (COMET, MetricX-24, BLEU) is system minus baseline on the macro
over the directions both sides have, each direction weighted equally, so a
delta is the difference of the macro columns. Its 95% CI and two-sided p come
from a stratified paired bootstrap: each resample redraws segment positions
within each direction, the same positions for both systems.

Written into ``--out``:

* ``long.csv``: one row per (system, direction) with every metric and attribute.
* ``summary.md``: coverage, headline (COMET, then MetricX-24), ref vs gen,
  SlimGPT vs FLAP, per-direction, sparsity curve, transfer, behaviour, repair
  and structure.
* ``transfer/<config>.md`` and ``transfer/<config>.csv``: the full pair (5x10)
  and direction (10x10) transfer matrices for one method x calib x sparsity.
* ``structure.csv`` and ``structure/<name>.layers.csv``: removed parameters and
  per-layer kept units.
* ``plots/*.png`` and ``results.json``.

Built to be rerun on a partial grid: anything missing is shown as ``-``, and a
macro average over fewer than ten directions carries an ``(n/10)`` marker.
Parsed runs, BLEU sufficient statistics, subnetwork summaries and bootstrap
results are cached under ``<out>/.cache``, keyed by file size and mtime, so a
rerun only re-reads what changed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import sys
import tempfile
import time
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from mnlp_eval.metrics.significance import DEFAULT_SEED

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_OUT = "results/grid-2026-10-06"
DEFAULT_SUITE = "alma10-greedy-300"
DENSE = "alma-7b"
COMET_KEY = "wmt22_comet_da"
METRICX_KEY = "metricx24"
CACHE_VERSION = 1
#: The parsed-run cache has its own version, bumped when the parsed structure
#: changes (2: MetricX fields), so the BLEU and bootstrap caches stay valid.
RUN_CACHE_VERSION = 2
#: Signature of the stored bootstrap results. v2: COMET and MetricX contrasts
#: became a stratified bootstrap of the macro, so pooled v1 results are dropped.
BOOTSTRAP_CACHE = "v2"
#: Tag in the cache key of every per-segment contrast bootstrap.
SEGMENT_BOOTSTRAP = "stratified-macro"

PAIR_LANGS = ("cs", "de", "is", "ru", "zh")
INTO_EN = tuple(f"{lang}-en" for lang in PAIR_LANGS)
OUT_OF_EN = tuple(f"en-{lang}" for lang in PAIR_LANGS)
DIRECTIONS = INTO_EN + OUT_OF_EN
N_DIRS = len(DIRECTIONS)

METHODS = ("slimgpt", "flap")
CALIBS = ("ref", "gen")
SPARSITIES = (20, 30, 40)
SCOPES = ("multi", "pair", "dir")
METHOD_LABEL = {"slimgpt": "SlimGPT", "flap": "FLAP"}
SCOPE_TITLE = {
    "multi": "multi (one model)",
    "pair": "pair (matched pair model)",
    "dir": "dir (matched direction model)",
}

#: Behavioural rates read from scores.surface.json, in long.csv column order.
BEHAVIOUR_FIELDS = (
    "off_target_rate",
    "repetition_rate",
    "budget_hit_rate",
    "truncation_rate",
    "empty_rate",
    "length_ratio",
    "source_copy_rate",
    "unverifiable_rate",
    "off_target_rate_open",
    "source_language_rate",
    "wasted_token_fraction",
)

LONG_COLUMNS = (
    "system",
    "run",
    "method",
    "calib",
    "sparsity",
    "scope",
    "scope_key",
    "pruned_directions",
    "repaired",
    "source_system",
    "direction",
    "into_english",
    "in_scope",
    "n_segments",
    "comet",
    "bleu",
    "chrf",
    "metricx",
    "comet_minus_dense",
    "bleu_minus_dense",
    "chrf_minus_dense",
    "metricx_minus_dense",
    "comet_minus_multi",
    "metricx_minus_multi",
    *BEHAVIOUR_FIELDS,
    "bleu_token_length_ratio",
)

#: Colours: one hue per method (validated categorical slots 1 and 2), a
#: blue-gray-red diverging ramp for the transfer heatmaps.
METHOD_COLOUR = {"slimgpt": "#2a78d6", "flap": "#eb6834"}
EXTRA_COLOURS = ("#1baf7a", "#4a3aa7", "#e87ba4", "#008300")
DIVERGING = ("#e34948", "#f0efec", "#2a78d6")
INK = "#0b0b0b"
MUTED = "#52514e"
SCOPE_MARKER = {"multi": "o", "pair": "s", "dir": "^"}

#: Llama-2-7B shapes, used when a checkpoint config is not available.
LLAMA_7B_DIMS = {
    "hidden_size": 4096,
    "head_dim": 128,
    "vocab_size": 32000,
    "num_attention_heads": 32,
    "intermediate_size": 11008,
    "tie_word_embeddings": False,
}

NAME_PATTERN = re.compile(
    r"(?:^|-)(?P<method>slimgpt|flap)(?P<pct>\d+)-(?P<calib>ref|gen)-"
    r"(?:(?P<multi>multi)|pair-(?P<pair>[a-z]{2})|dir-(?P<dir>[a-z]{2}-[a-z]{2}))"
    r"(?P<lora>-lora)?$"
)


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------


def _num(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value) if math.isfinite(value) else None


def _mean(values: Iterable[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return sum(present) / len(present) if present else None


def pair_of(direction: str) -> str:
    """The non-English side of an English-centric direction."""
    source, target = direction.split("-")
    return target if source == "en" else source


def reverse(direction: str) -> str:
    source, target = direction.split("-")
    return f"{target}-{source}"


def scope_directions(scope: str, key: str) -> tuple[str, ...]:
    if scope == "multi":
        return DIRECTIONS
    if scope == "pair" and key:
        return (f"{key}-en", f"en-{key}")
    if scope == "dir" and key:
        return (key,)
    return ()


def _file_sig(path: Path) -> list[int] | None:
    try:
        stat = path.stat()
    except OSError:
        return None
    return [stat.st_mtime_ns, stat.st_size]


def _read_json(path: Path, warnings: list[str] | None = None) -> Any:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        if warnings is not None:
            warnings.append(f"could not read {path}: {exc}")
        return None


def _write_text(path: Path, text: str) -> None:
    """Write through a temporary file and a rename, without fsync."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(text)
        Path(name).replace(path)
    except BaseException:
        Path(name).unlink(missing_ok=True)
        raise


def _log(message: str) -> None:
    print(f"[grid-report] {message}", file=sys.stderr, flush=True)


# Formatting ---------------------------------------------------------------


def f4(value: float | None) -> str:
    return "-" if value is None else f"{value:.4f}"


def f2(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}"


def f3(value: float | None) -> str:
    """MetricX-24 scores and deltas (an error score in [0, 25])."""
    return "-" if value is None else f"{value:.3f}"


def signed(value: float | None, digits: int) -> str:
    return "-" if value is None else f"{value:+.{digits}f}"


def pct(value: float | None, digits: int = 1) -> str:
    return "-" if value is None else f"{100 * value:.{digits}f}%"


def fmt_p(value: float | None) -> str:
    if value is None:
        return "-"
    return "<0.001" if value < 0.001 else f"{value:.3f}"


def with_coverage(text: str, n: int, total: int = N_DIRS) -> str:
    if text == "-" or n >= total:
        return text
    return f"{text} ({n}/{total})"


def md_table(header: Sequence[str], rows: Iterable[Sequence[str]], labels: int = 1) -> list[str]:
    """A GitHub markdown table, label columns left-aligned and the rest right."""
    align = [":---" if index < labels else "---:" for index in range(len(header))]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(align) + " |"]
    lines.extend("| " + " | ".join(str(cell) for cell in row) + " |" for row in rows)
    return lines


# --------------------------------------------------------------------------
# Systems
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class System:
    """One evaluated model and the grid attributes that place it."""

    name: str
    method: str = "dense"
    calib: str = ""
    sparsity: int = 0
    scope: str = ""
    scope_key: str = ""
    pruned_directions: tuple[str, ...] = ()
    repaired: bool = False
    source: str = ""
    priority: Any = None

    @property
    def is_dense(self) -> bool:
        return self.method == "dense"

    @property
    def config(self) -> str:
        return f"{self.method}{self.sparsity}-{self.calib}"

    @property
    def cell(self) -> tuple[str, str, int, str, str]:
        return (self.method, self.calib, self.sparsity, self.scope, self.scope_key)

    def in_scope(self, direction: str) -> bool:
        if self.is_dense or self.scope == "multi":
            return True
        return direction in self.pruned_directions

    @property
    def expected_directions(self) -> tuple[str, ...]:
        """The directions this system must be scored on to count as complete.

        All ten for dense and multi models (and their repairs); the pruned
        directions for a pair or dir model, which a ``-lora`` repair inherits.
        A specialist scored on more (the pilot, the Tier-3 full-suite runs) is
        still complete, and also has transfer data.
        """
        if self.is_dense or self.scope == "multi":
            return DIRECTIONS
        own = tuple(d for d in DIRECTIONS if d in self.pruned_directions)
        return own or DIRECTIONS

    def label(self) -> str:
        if self.is_dense:
            return f"dense ({self.name})"
        text = config_label(self.method, self.calib, self.sparsity)
        if self.scope != "multi":
            text += f" {self.scope}-{self.scope_key}"
        return text + (" +LoRA" if self.repaired else "")


def config_label(method: str, calib: str, sparsity: int) -> str:
    return f"{METHOD_LABEL.get(method, method)} {calib} {sparsity}%"


def parse_name(name: str) -> System | None:
    """Infer grid attributes from a name such as ``alma-7b-slimgpt30-gen-dir-en-is``."""
    match = NAME_PATTERN.search(name)
    if match is None:
        return None
    if match["multi"]:
        scope, key = "multi", ""
    elif match["pair"]:
        scope, key = "pair", match["pair"]
    else:
        scope, key = "dir", match["dir"]
    lora = bool(match["lora"])
    return System(
        name=name,
        method=match["method"],
        calib=match["calib"],
        sparsity=int(match["pct"]),
        scope=scope,
        scope_key=key,
        pruned_directions=scope_directions(scope, key),
        repaired=lora,
        source=name[: -len("-lora")] if lora else "",
    )


def _norm_method(value: Any) -> str:
    text = str(value or "").lower()
    if "slimgpt" in text:
        return "slimgpt"
    if "flap" in text:
        return "flap"
    if "llm" in text and "pruner" in text:
        return "llm-pruner"
    return re.sub(r"[^a-z0-9]+", "", text)


def _norm_calib(entry: dict[str, Any]) -> str:
    """Calibration tag: ``ref`` (prompt+target) or ``gen`` (prompt+generated)."""
    tag = str(entry.get("tag") or "").lower()
    tokens = set(re.split(r"[^a-z]+", tag))
    if "gen" in tokens or "generated" in tokens:
        return "gen"
    if tokens & {"ref", "reference", "target"}:
        return "ref"
    text = str(entry.get("calibration_text") or "").lower()
    if "generat" in text:
        return "gen"
    if "target" in text or "ref" in text:
        return "ref"
    return text


def _norm_sparsity(value: Any) -> int:
    number = _num(value)
    if number is None and isinstance(value, str):
        try:
            number = float(value.strip().rstrip("%"))
        except ValueError:
            number = None
    if number is None:
        return 0
    return round(number * 100) if number <= 1 else round(number)


def _norm_dirs(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        items = value.split(",")
    elif isinstance(value, list | tuple):
        items = [str(item) for item in value]
    else:
        return ()
    return tuple(item.strip() for item in items if item.strip())


def _norm_scope_key(scope: str, value: Any, directions: tuple[str, ...]) -> str:
    key = str(value or "").strip()
    if scope == "multi":
        return ""
    for prefix in ("pair-", "dir-"):
        if key.startswith(prefix):
            key = key[len(prefix) :]
    if not key and directions:
        key = pair_of(directions[0]) if scope == "pair" else directions[0]
    return key


def system_from_entry(entry: dict[str, Any]) -> System | None:
    """Normalise one grid manifest entry. Missing fields fall back to the name."""
    name = str(entry.get("name") or "").strip()
    if not name:
        return None
    parsed = parse_name(name)
    method = _norm_method(entry.get("method")) or (parsed.method if parsed else "")
    calib = _norm_calib(entry) or (parsed.calib if parsed else "")
    sparsity = _norm_sparsity(entry.get("sparsity")) or (parsed.sparsity if parsed else 0)
    scope = str(entry.get("scope") or (parsed.scope if parsed else "")).lower()
    directions = _norm_dirs(entry.get("pruned_directions"))
    key = _norm_scope_key(scope, entry.get("scope_key"), directions)
    if not key and parsed is not None:
        key = parsed.scope_key
    if not directions:
        directions = scope_directions(scope, key)
    repaired = bool(entry.get("repair")) or name.endswith("-lora")
    source = str(entry.get("source") or (name[: -len("-lora")] if repaired else ""))
    if not method or scope not in SCOPES:
        return None
    return System(
        name=name,
        method=method,
        calib=calib,
        sparsity=sparsity,
        scope=scope,
        scope_key=key,
        pruned_directions=directions,
        repaired=repaired,
        source=source if repaired else "",
        priority=entry.get("priority"),
    )


def load_manifest(path: Path, warnings: list[str]) -> tuple[list[System], str | None] | None:
    """Return the grid systems and the suite they name, or None if absent."""
    payload = _read_json(path, warnings)
    if payload is None:
        return None
    entries: list[dict[str, Any]] = []
    if isinstance(payload, list):
        entries = [entry for entry in payload if isinstance(entry, dict)]
    elif isinstance(payload, dict):
        for key in ("models", "systems", "entries", "grid"):
            if isinstance(payload.get(key), list):
                entries = [entry for entry in payload[key] if isinstance(entry, dict)]
                break
        else:
            entries = [
                {"name": name, **entry}
                for name, entry in payload.items()
                if isinstance(entry, dict)
            ]
    systems: list[System] = []
    suites: set[str] = set()
    for entry in entries:
        system = system_from_entry(entry)
        if system is None:
            warnings.append(f"manifest entry not understood, skipped: {entry.get('name')!r}")
            continue
        systems.append(system)
        if entry.get("suite"):
            suites.add(Path(str(entry["suite"])).stem)
    suite = suites.pop() if len(suites) == 1 else None
    return systems, suite


# --------------------------------------------------------------------------
# Runs
# --------------------------------------------------------------------------


@dataclass
class RunInfo:
    """The parsed scores of one run directory."""

    name: str
    slug: str
    path: Path
    run_id: str
    created_at: str
    compression: dict[str, Any]
    checkpoint: str | None
    has_surface: bool
    has_neural: bool
    directions: dict[str, dict[str, Any]]
    fingerprints: dict[str, str]
    neural_sig: Any
    mtime: int
    has_metricx: bool = False
    metricx_sig: Any = None

    def value(self, direction: str, key: str) -> float | None:
        return _num(self.directions.get(direction, {}).get(key))

    def segments(self, direction: str, metric: str = "comet") -> list[float] | None:
        found = self.directions.get(direction, {}).get(f"{metric}_segments")
        return found or None

    @property
    def scored_directions(self) -> list[str]:
        keys = ("comet", "bleu", "chrf")
        return [d for d in DIRECTIONS if all(self.value(d, key) is not None for key in keys)]

    def covers(self, directions: Iterable[str]) -> bool:
        """Whether BLEU, chrF++ and COMET exist for every one of ``directions``."""
        return set(directions) <= set(self.scored_directions)

    @property
    def all_directions(self) -> bool:
        """Scored on all ten directions: the condition for entering the transfer section."""
        return len(self.scored_directions) == N_DIRS

    def metricx_directions(self) -> list[str]:
        return [d for d in DIRECTIONS if self.value(d, "metricx") is not None]


def _metric_score(metrics: dict[str, Any], name: str) -> float | None:
    payload = metrics.get(name)
    return _num(payload.get("score")) if isinstance(payload, dict) else None


def parse_run(path: Path) -> dict[str, Any]:
    """Flatten one run directory into the fields the report needs."""
    errors: list[str] = []
    manifest = _read_json(path / "manifest.json", errors) or {}
    model = manifest.get("model") or {}
    suite = manifest.get("suite") or {}
    compression = model.get("compression")
    data: dict[str, Any] = {
        "name": str(model.get("name") or path.name.split("__")[0]),
        "suite": str(suite.get("name") or ""),
        "run_id": str(manifest.get("run_id") or ""),
        "created_at": str(manifest.get("created_at") or ""),
        "compression": compression if isinstance(compression, dict) else {},
        "checkpoint": (model.get("kwargs") or {}).get("checkpoint"),
        "has_surface": False,
        "has_neural": False,
        "has_metricx": False,
        "directions": {},
        "fingerprints": {},
        "errors": errors,
    }
    surface = _read_json(path / "scores.surface.json", errors)
    if isinstance(surface, dict):
        data["has_surface"] = True
        for direction, entry in (surface.get("directions") or {}).items():
            record = data["directions"].setdefault(direction, {})
            metrics = entry.get("metrics") or {}
            record["bleu"] = _metric_score(metrics, "bleu")
            record["chrf"] = _metric_score(metrics, "chrf2pp")
            extra = (metrics.get("bleu") or {}).get("extra") or {}
            record["bleu_token_length_ratio"] = _num(extra.get("token_length_ratio"))
            behaviour = entry.get("behaviour") or {}
            for name in BEHAVIOUR_FIELDS:
                record[name] = _num(behaviour.get(name))
            record["n_segments"] = entry.get("n_segments")
    neural = _read_json(path / "scores.neural.json", errors)
    if isinstance(neural, dict):
        data["has_neural"] = True
        for direction, entry in (neural.get("directions") or {}).items():
            payload = (entry.get("metrics") or {}).get(COMET_KEY) or {}
            record = data["directions"].setdefault(direction, {})
            record["comet"] = _num(payload.get("score"))
            segments = payload.get("segment_scores")
            record["comet_segments"] = (
                [float(value) for value in segments] if isinstance(segments, list) else None
            )
            if record.get("n_segments") is None:
                record["n_segments"] = entry.get("n_segments")
    # Optional: absent for every pilot run, so a missing file is not an error.
    metricx = _read_json(path / "scores.metricx.json", errors)
    if isinstance(metricx, dict):
        data["has_metricx"] = True
        for direction, entry in (metricx.get("directions") or {}).items():
            payload = (entry.get("metrics") or {}).get(METRICX_KEY) or {}
            record = data["directions"].setdefault(direction, {})
            record["metricx"] = _num(payload.get("score"))
            segments = payload.get("segment_scores")
            record["metricx_segments"] = (
                [float(value) for value in segments] if isinstance(segments, list) else None
            )
            if record.get("n_segments") is None:
                record["n_segments"] = entry.get("n_segments")
    stages = path / "stages"
    if stages.is_dir():
        for stage in sorted(stages.glob("generate*.json")):
            record = _read_json(stage, errors) or {}
            for direction, payload in (record.get("directions") or {}).items():
                fingerprint = ((payload or {}).get("data") or {}).get("fingerprint")
                if fingerprint:
                    data["fingerprints"][direction] = str(fingerprint)
    return data


#: Files whose signatures make up the first entries of ``run_signature``.
RUN_FILES = ("manifest.json", "scores.surface.json", "scores.neural.json", "scores.metricx.json")


def run_signature(path: Path) -> list[Any]:
    parts: list[Any] = [_file_sig(path / name) for name in RUN_FILES]
    stages = path / "stages"
    if stages.is_dir():
        parts.append(
            sorted([stage.name, _file_sig(stage)] for stage in stages.glob("generate*.json"))
        )
    return parts


# --------------------------------------------------------------------------
# Cache
# --------------------------------------------------------------------------


class Cache:
    """Per-key JSON files under one directory, validated by a signature."""

    def __init__(self, root: Path | None) -> None:
        self.root = root
        self._bootstrap: dict[str, Any] | None = None
        self._bootstrap_dirty = False

    def _path(self, kind: str, key: str) -> Path:
        assert self.root is not None
        safe = re.sub(r"[^A-Za-z0-9_.+-]", "_", key)
        return self.root / kind / f"{safe}.json"

    @staticmethod
    def version(kind: str) -> int:
        return RUN_CACHE_VERSION if kind == "runs" else CACHE_VERSION

    def get(self, kind: str, key: str, signature: Any) -> Any:
        if self.root is None:
            return None
        payload = _read_json(self._path(kind, key))
        if not isinstance(payload, dict) or payload.get("version") != self.version(kind):
            return None
        if payload.get("signature") != signature:
            return None
        return payload.get("data")

    def put(self, kind: str, key: str, signature: Any, data: Any) -> None:
        if self.root is None:
            return
        payload = {"version": self.version(kind), "signature": signature, "data": data}
        try:
            _write_text(self._path(kind, key), json.dumps(payload))
        except OSError as exc:
            _log(f"cache write failed for {kind}/{key}: {exc}")

    def _bootstraps(self) -> dict[str, Any]:
        if self._bootstrap is None:
            loaded = self.get("bootstrap", "results", BOOTSTRAP_CACHE) if self.root else None
            self._bootstrap = loaded if isinstance(loaded, dict) else {}
        return self._bootstrap

    def bootstrap_get(self, key: str) -> dict[str, Any] | None:
        found = self._bootstraps().get(key)
        return found if isinstance(found, dict) else None

    def bootstrap_put(self, key: str, value: dict[str, Any]) -> None:
        self._bootstraps()[key] = value
        self._bootstrap_dirty = True

    def flush(self) -> None:
        if self._bootstrap_dirty and self._bootstrap is not None:
            self.put("bootstrap", "results", BOOTSTRAP_CACHE, self._bootstrap)
            self._bootstrap_dirty = False


def _digest(payload: Any) -> str:
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()


# --------------------------------------------------------------------------
# Context
# --------------------------------------------------------------------------

Composite = dict[str, tuple[System, RunInfo]]


@dataclass
class Context:
    args: argparse.Namespace
    out: Path
    suite: str
    cache: Cache
    systems: dict[str, System] = field(default_factory=dict)
    grid: list[System] = field(default_factory=list)
    runs: dict[str, RunInfo] = field(default_factory=dict)
    index: dict[tuple[str, str, int, str, str], System] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    manifest_found: bool = False
    methods: tuple[str, ...] = METHODS
    calibs: tuple[str, ...] = CALIBS
    sparsities: tuple[int, ...] = SPARSITIES
    composites: dict[tuple[str, str, int, str], Composite] = field(default_factory=dict)
    bleu_stats: dict[str, dict[str, Any]] = field(default_factory=dict)
    bleu_dirty: set[str] = field(default_factory=set)
    transfer: dict[str, dict[str, Any]] = field(default_factory=dict)
    structure: dict[str, dict[str, Any]] = field(default_factory=dict)
    results: dict[str, Any] = field(default_factory=dict)
    n_bootstraps: int = 0

    @property
    def dense(self) -> RunInfo | None:
        return self.runs.get(DENSE)

    def warn(self, message: str) -> None:
        if message not in self.warnings:
            self.warnings.append(message)

    def complete(self, system: System) -> bool:
        """Scored on every direction the system is meant to be evaluated on."""
        run = self.runs.get(system.name)
        return run is not None and run.covers(system.expected_directions)

    def in_scope_scored(self, system: System) -> int:
        run = self.runs.get(system.name)
        if run is None:
            return 0
        return len(set(system.expected_directions) & set(run.scored_directions))

    @property
    def has_metricx(self) -> bool:
        return any(run.metricx_directions() for run in self.runs.values())

    def configs(self) -> list[tuple[str, str, int]]:
        return [(m, c, s) for m in self.methods for c in self.calibs for s in self.sparsities]


def _ordered(found: Iterable[Any], canonical: Sequence[Any]) -> tuple[Any, ...]:
    extra = sorted(set(found) - set(canonical), key=str)
    return (*canonical, *extra)


def load_systems(ctx: Context) -> None:
    result = load_manifest(ctx.args.manifest, ctx.warnings)
    systems: list[System]
    if result is None:
        ctx.warn(
            f"grid manifest {ctx.args.manifest} not found; systems inferred from run "
            "directory names, so the expected-model count is unknown"
        )
        systems = []
        if ctx.args.runs_root.is_dir():
            for child in sorted(ctx.args.runs_root.iterdir()):
                parts = child.name.split("__")
                if len(parts) < 3 or parts[1] != ctx.suite:
                    continue
                parsed = parse_name(parts[0])
                if parsed is not None and not parsed.repaired:
                    systems.append(parsed)
    else:
        ctx.manifest_found = True
        systems, suite = result
        if suite and ctx.args.suite is None:
            ctx.suite = suite
    seen: set[str] = set()
    for system in systems:
        if system.name in seen:
            continue
        seen.add(system.name)
        ctx.systems[system.name] = system
        if system.repaired:
            continue
        ctx.grid.append(system)
        if system.cell in ctx.index:
            ctx.warn(f"{system.name} and {ctx.index[system.cell].name} occupy the same grid cell")
        ctx.index.setdefault(system.cell, system)
    ctx.systems.setdefault(DENSE, System(name=DENSE))
    ctx.methods = _ordered((s.method for s in ctx.grid), METHODS)
    ctx.calibs = _ordered((s.calib for s in ctx.grid), CALIBS)
    ctx.sparsities = _ordered((s.sparsity for s in ctx.grid), SPARSITIES)


def discover_runs(ctx: Context) -> None:
    """Pick one run directory per known system, preferring complete scores."""
    root = ctx.args.runs_root
    if not root.is_dir():
        ctx.warn(f"runs root {root} does not exist")
        return
    candidates: dict[str, list[RunInfo]] = {}
    for child in sorted(root.iterdir()):
        parts = child.name.split("__")
        if len(parts) < 3 or parts[1] != ctx.suite:
            continue
        prefix = parts[0]
        source = prefix[: -len("-lora")] if prefix.endswith("-lora") else None
        if prefix not in ctx.systems and source not in ctx.systems:
            continue
        if not (child / "manifest.json").is_file():
            continue
        signature = run_signature(child)
        data = ctx.cache.get("runs", child.name, signature)
        if data is None:
            data = parse_run(child)
            ctx.cache.put("runs", child.name, signature, data)
        for error in data.get("errors") or []:
            ctx.warn(error)
        if data["suite"] != ctx.suite:
            continue
        mtimes = [part[0] for part in signature[: len(RUN_FILES)] if part]
        run = RunInfo(
            name=data["name"],
            slug=child.name,
            path=child,
            run_id=data["run_id"],
            created_at=data["created_at"],
            compression=data["compression"],
            checkpoint=data["checkpoint"],
            has_surface=data["has_surface"],
            has_neural=data["has_neural"],
            directions=data["directions"],
            fingerprints=data["fingerprints"],
            neural_sig=signature[RUN_FILES.index("scores.neural.json")],
            mtime=max(mtimes) if mtimes else 0,
            has_metricx=bool(data.get("has_metricx")),
            metricx_sig=signature[RUN_FILES.index("scores.metricx.json")],
        )
        candidates.setdefault(run.name, []).append(run)

    for name in sorted(candidates):
        if name in ctx.systems or not name.endswith("-lora"):
            continue
        source = ctx.systems.get(name[: -len("-lora")])
        if source is None or source.is_dense:
            continue
        ctx.systems[name] = System(
            name=name,
            method=source.method,
            calib=source.calib,
            sparsity=source.sparsity,
            scope=source.scope,
            scope_key=source.scope_key,
            pruned_directions=source.pruned_directions,
            repaired=True,
            source=source.name,
        )

    for name, runs in sorted(candidates.items()):
        if name not in ctx.systems:
            continue
        ranked = sorted(
            runs,
            key=lambda r: (
                len(r.scored_directions),
                r.has_neural,
                r.has_surface,
                len(r.metricx_directions()),
                r.mtime,
            ),
            reverse=True,
        )
        chosen = ranked[0]
        ctx.runs[name] = chosen
        if len(ranked) > 1:
            others = ", ".join(
                f"{r.slug} ({len(r.scored_directions)}/{N_DIRS} scored)" for r in ranked[1:]
            )
            ctx.warn(
                f"{name}: {len(ranked)} run directories for suite {ctx.suite}; using "
                f"{chosen.slug} ({len(chosen.scored_directions)}/{N_DIRS} scored), "
                f"ignoring {others}"
            )

    dense = ctx.dense
    if dense is None:
        ctx.warn(f"dense baseline {DENSE} has no run for suite {ctx.suite}")
        return
    for run in ctx.runs.values():
        for direction, fingerprint in run.fingerprints.items():
            expected = dense.fingerprints.get(direction)
            if expected and fingerprint != expected:
                ctx.warn(
                    f"{run.name} {direction}: data fingerprint {fingerprint} differs from the "
                    f"dense run's {expected}; its segments are not comparable"
                )


def composite(ctx: Context, method: str, calib: str, sparsity: int, scope: str) -> Composite:
    """Map each direction to the model that translates it under ``scope``.

    multi: the one multi model. pair: ``pair-<l>`` for both directions of
    language l. dir: ``dir-<d>`` for direction d.
    """
    key = (method, calib, sparsity, scope)
    if key in ctx.composites:
        return ctx.composites[key]
    found: Composite = {}
    for direction in DIRECTIONS:
        if scope == "multi":
            scope_key = ""
        elif scope == "pair":
            scope_key = pair_of(direction)
        else:
            scope_key = direction
        system = ctx.index.get((method, calib, sparsity, scope, scope_key))
        run = ctx.runs.get(system.name) if system else None
        if system is not None and run is not None and direction in run.directions:
            found[direction] = (system, run)
    ctx.composites[key] = found
    return found


def single(ctx: Context, name: str) -> Composite:
    system, run = ctx.systems.get(name), ctx.runs.get(name)
    if system is None or run is None:
        return {}
    return {d: (system, run) for d in DIRECTIONS if d in run.directions}


def macro(
    comp: Composite, key: str, directions: Sequence[str] = DIRECTIONS
) -> tuple[float | None, int]:
    values = [comp[d][1].value(d, key) for d in directions if d in comp]
    present = [value for value in values if value is not None]
    return (round(sum(present) / len(present), 6) if present else None), len(present)


# --------------------------------------------------------------------------
# Paired bootstrap
# --------------------------------------------------------------------------


def bleu_from_stats(stats: Any) -> np.ndarray:
    """Corpus BLEU from summed sacreBLEU sufficient statistics, vectorised.

    ``stats[..., :]`` is ``[sys_len, ref_len, correct_1..4, total_1..4]``, as
    sacreBLEU's ``_extract_corpus_statistics`` returns per segment. Matches
    ``BLEU.compute_bleu`` with its defaults: exp smoothing, no effective order.
    """
    s = np.asarray(stats, dtype=np.float64)
    sys_len, ref_len = s[..., 0], s[..., 1]
    correct, total = s[..., 2:6], s[..., 6:10]
    zero = correct == 0
    smooth = np.cumprod(np.where(zero, 2.0, 1.0), axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        precisions = np.where(zero, 100.0 / (smooth * total), 100.0 * correct / total)
        log_mean = np.log(precisions).mean(axis=-1)
        ratio = ref_len / np.where(sys_len > 0, sys_len, 1.0)
        bp = np.where(sys_len < ref_len, np.where(sys_len > 0, np.exp(1.0 - ratio), 0.0), 1.0)
        score = bp * np.exp(log_mean)
    dead = (total <= 0).any(axis=-1) | (correct.sum(axis=-1) == 0)
    return np.where(dead, 0.0, score)


def macro_bleu_bootstrap(
    pairs: Sequence[tuple[np.ndarray, np.ndarray]], *, n_samples: int, seed: int
) -> dict[str, Any]:
    """Paired bootstrap on macro BLEU over directions, resampled within each.

    ``significance.paired_bootstrap_surface`` tests one direction with one
    tokenizer, so it cannot give an interval on a ten-direction macro that
    mixes the 13a and zh tokenizers. This resamples segments within each
    direction (the same indices for both systems), recomputes corpus BLEU per
    direction from sufficient statistics and averages. The p-value follows
    ``bootstrap_segment_delta``: two-sided, centred, add-one smoothed.
    """
    base = float(np.mean([bleu_from_stats(a.sum(axis=0)) for a, _ in pairs]))
    system = float(np.mean([bleu_from_stats(b.sum(axis=0)) for _, b in pairs]))
    observed = system - base
    rng = np.random.default_rng(seed)
    deltas = np.zeros(n_samples)
    for a, b in pairs:
        m = a.shape[0]
        indices = rng.integers(0, m, size=(n_samples, m))
        offsets = np.arange(n_samples)[:, None] * m
        weights = np.bincount((indices + offsets).ravel(), minlength=n_samples * m)
        weights = weights.reshape(n_samples, m).astype(np.float64)
        deltas += bleu_from_stats(weights @ b) - bleu_from_stats(weights @ a)
    deltas /= len(pairs)
    centred = np.abs(deltas - observed) >= abs(observed)
    p_value = float((1 + int(centred.sum())) / (n_samples + 1))
    return {
        "method": "stratified paired bootstrap on macro corpus BLEU",
        "n_samples": n_samples,
        "seed": seed,
        "n_segments": int(sum(a.shape[0] for a, _ in pairs)),
        "baseline_score": round(base, 4),
        "system_score": round(system, 4),
        "delta": round(observed, 4),
        "p_value": round(p_value, 6),
        "significant_at_0.05": p_value < 0.05,
        "bootstrap_ci_95": [
            round(float(np.percentile(deltas, 2.5)), 4),
            round(float(np.percentile(deltas, 97.5)), 4),
        ],
    }


def _macro_means(pairs: Sequence[tuple[np.ndarray, np.ndarray]]) -> tuple[float, float, float]:
    """Baseline macro, system macro and their difference: equal weight per direction."""
    base = float(np.mean([a.mean() for a, _ in pairs]))
    system = float(np.mean([b.mean() for _, b in pairs]))
    observed = float(np.mean([b.mean() - a.mean() for a, b in pairs]))
    return base, system, observed


def macro_segment_bootstrap(
    pairs: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    n_samples: int,
    seed: int,
    higher_is_better: bool = True,
) -> dict[str, Any]:
    """Stratified paired bootstrap on the macro of a per-segment metric.

    ``pairs`` holds one (baseline, system) pair of per-segment score arrays per
    direction, paired by position. Each resample redraws segment positions
    independently within each direction, the same positions for both systems,
    takes the per-direction mean difference and averages the directions with
    equal weight. The observed delta is therefore the difference of the two
    macros, whatever the directions' sizes, unlike pooling all segments, which
    weights directions by their segment counts.

    Seed, p-value and CI mirror ``significance.bootstrap_segment_delta``: one
    ``default_rng(seed)``, directions drawn in order; two-sided centred p,
    ``P(|d_b - d| >= |d|)``, add-one smoothed; percentile 95% CI on the delta.
    With a single direction it reproduces that helper exactly.
    """
    base, system, observed = _macro_means(pairs)
    rng = np.random.default_rng(seed)
    deltas = np.zeros(n_samples)
    for a, b in pairs:
        indices = rng.integers(0, a.shape[0], size=(n_samples, a.shape[0]))
        deltas += b[indices].mean(axis=1) - a[indices].mean(axis=1)
    deltas /= len(pairs)
    centred = np.abs(deltas - observed) >= abs(observed)
    p_value = float((1 + int(centred.sum())) / (n_samples + 1))
    improved = observed > 0 if higher_is_better else observed < 0
    return {
        "method": "stratified paired bootstrap on the macro of per-segment scores",
        "n_samples": n_samples,
        "seed": seed,
        "n_segments": int(sum(a.shape[0] for a, _ in pairs)),
        "n_directions": len(pairs),
        "baseline_score": round(base, 6),
        "system_score": round(system, 6),
        "delta": round(observed, 6),
        "improved": bool(improved),
        "higher_is_better": higher_is_better,
        "p_value": round(p_value, 6),
        "significant_at_0.05": p_value < 0.05,
        "bootstrap_ci_95": [
            round(float(np.percentile(deltas, 2.5)), 6),
            round(float(np.percentile(deltas, 97.5)), 6),
        ],
    }


def bleu_stats(ctx: Context, run: RunInfo, direction: str) -> tuple[np.ndarray, Any] | None:
    """Per-segment BLEU statistics for one run and direction, cached by hyps file."""
    path = run.path / "hyps" / f"{direction}.jsonl"
    signature = _file_sig(path)
    if signature is None:
        return None
    store = ctx.bleu_stats.get(run.slug)
    if store is None:
        store = ctx.cache.get("bleu", run.slug, "per-direction") or {}
        ctx.bleu_stats[run.slug] = store
    entry = store.get(direction)
    if not isinstance(entry, dict) or entry.get("signature") != signature:
        from sacrebleu.metrics import BLEU

        from mnlp_eval.languages import sacrebleu_tokenizer

        hypotheses: list[str] = []
        references: list[str] = []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    record = json.loads(line)
                    hypotheses.append(str(record.get("hypothesis") or ""))
                    references.append(str(record.get("reference") or ""))
        metric = BLEU(tokenize=sacrebleu_tokenizer(direction.split("-")[1]))
        rows = metric._extract_corpus_statistics(hypotheses, [references])
        entry = {"signature": signature, "stats": [[int(v) for v in row] for row in rows]}
        store[direction] = entry
        ctx.bleu_dirty.add(run.slug)
    stats = np.asarray(entry["stats"], dtype=np.float64)
    if stats.ndim != 2 or stats.shape[0] == 0:
        return None
    stored = run.value(direction, "bleu")
    recomputed = float(bleu_from_stats(stats.sum(axis=0)))
    if stored is not None and abs(recomputed - stored) > 0.01:
        ctx.warn(
            f"{run.name} {direction}: BLEU from hyps ({recomputed:.2f}) differs from the "
            f"scored value ({stored:.2f}); hyps may have changed after scoring, so BLEU "
            "bootstraps that need it are skipped"
        )
        return None
    return stats, signature


def flush_bleu_cache(ctx: Context) -> None:
    for slug in sorted(ctx.bleu_dirty):
        ctx.cache.put("bleu", slug, "per-direction", ctx.bleu_stats[slug])
    ctx.bleu_dirty.clear()


#: Metrics with stored per-segment scores: label, which way is better, and the
#: run attribute holding the signature of the scores file they come from.
SEGMENT_METRICS = {
    "comet": ("COMET", True, "neural_sig"),
    "metricx": ("MetricX-24", False, "metricx_sig"),
}


def segment_contrast(
    ctx: Context, base: Composite, system: Composite, metric: str = "comet"
) -> dict[str, Any] | None:
    """System minus base on the macro of a per-segment metric over shared directions.

    Equal weight per direction, so the delta is the difference of the macro
    columns; the CI and p come from ``macro_segment_bootstrap``. The delta is
    always system minus base: for MetricX-24 (lower is better) a negative delta
    means the system is better.
    """
    label, higher_is_better, sig_attr = SEGMENT_METRICS[metric]
    directions: list[str] = []
    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    key_parts: list[Any] = []
    for direction in DIRECTIONS:
        if direction not in base or direction not in system:
            continue
        run_a, run_b = base[direction][1], system[direction][1]
        seg_a, seg_b = run_a.segments(direction, metric), run_b.segments(direction, metric)
        if not seg_a or not seg_b:
            continue
        if len(seg_a) != len(seg_b):
            ctx.warn(
                f"{direction}: {run_a.name} has {len(seg_a)} {label} segments but "
                f"{run_b.name} has {len(seg_b)}; left out of the paired contrast"
            )
            continue
        directions.append(direction)
        pairs.append((np.asarray(seg_a, dtype=np.float64), np.asarray(seg_b, dtype=np.float64)))
        key_parts.append(
            [
                direction,
                run_a.slug,
                getattr(run_a, sig_attr),
                run_b.slug,
                getattr(run_b, sig_attr),
            ]
        )
    if not directions:
        return None
    base_score, system_score, delta = _macro_means(pairs)
    result: dict[str, Any] = {
        "metric": metric,
        "higher_is_better": higher_is_better,
        "directions": directions,
        "n_directions": len(directions),
        "n_segments": int(sum(a.shape[0] for a, _ in pairs)),
        "baseline_score": round(base_score, 6),
        "system_score": round(system_score, 6),
        "delta": round(delta, 6),
        "p_value": None,
        "bootstrap_ci_95": None,
        "n_samples": 0,
    }
    n_samples = ctx.args.bootstrap
    if n_samples > 0:
        key = _digest([metric, SEGMENT_BOOTSTRAP, n_samples, ctx.args.seed, key_parts])
        boot = ctx.cache.bootstrap_get(key)
        if boot is None:
            boot = macro_segment_bootstrap(
                pairs, n_samples=n_samples, seed=ctx.args.seed, higher_is_better=higher_is_better
            )
            ctx.n_bootstraps += 1
            ctx.cache.bootstrap_put(key, boot)
        result.update(
            {
                "delta": boot["delta"],
                "p_value": boot["p_value"],
                "bootstrap_ci_95": boot["bootstrap_ci_95"],
                "n_samples": n_samples,
            }
        )
    return result


def comet_contrast(ctx: Context, base: Composite, system: Composite) -> dict[str, Any] | None:
    """System minus base on macro COMET over the shared directions."""
    return segment_contrast(ctx, base, system, "comet")


def metricx_contrast(ctx: Context, base: Composite, system: Composite) -> dict[str, Any] | None:
    """System minus base on MetricX-24: negative means the system is better."""
    return segment_contrast(ctx, base, system, "metricx")


def bleu_contrast(ctx: Context, base: Composite, system: Composite) -> dict[str, Any] | None:
    """System minus base on macro BLEU over shared directions."""
    directions = [
        d
        for d in DIRECTIONS
        if d in base
        and d in system
        and base[d][1].value(d, "bleu") is not None
        and system[d][1].value(d, "bleu") is not None
    ]
    if not directions:
        return None
    base_score = _mean(base[d][1].value(d, "bleu") for d in directions) or 0.0
    system_score = _mean(system[d][1].value(d, "bleu") for d in directions) or 0.0
    result: dict[str, Any] = {
        "metric": "bleu",
        "directions": directions,
        "n_directions": len(directions),
        "baseline_score": round(base_score, 4),
        "system_score": round(system_score, 4),
        "delta": round(system_score - base_score, 4),
        "p_value": None,
        "bootstrap_ci_95": None,
        "n_samples": 0,
    }
    n_samples = ctx.args.bootstrap
    if n_samples <= 0:
        return result
    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    key_parts: list[Any] = []
    for direction in directions:
        run_a, run_b = base[direction][1], system[direction][1]
        found_a, found_b = bleu_stats(ctx, run_a, direction), bleu_stats(ctx, run_b, direction)
        if found_a is None or found_b is None or found_a[0].shape != found_b[0].shape:
            result["note"] = f"no usable hyps for {direction}; BLEU bootstrap skipped"
            return result
        pairs.append((found_a[0], found_b[0]))
        key_parts.append([direction, run_a.slug, found_a[1], run_b.slug, found_b[1]])
    key = _digest(["bleu", n_samples, ctx.args.seed, key_parts])
    boot = ctx.cache.bootstrap_get(key)
    if boot is None:
        boot = macro_bleu_bootstrap(pairs, n_samples=n_samples, seed=ctx.args.seed)
        ctx.n_bootstraps += 1
        ctx.cache.bootstrap_put(key, boot)
    result.update(
        {
            "delta": boot["delta"],
            "p_value": boot["p_value"],
            "bootstrap_ci_95": boot["bootstrap_ci_95"],
            "n_samples": n_samples,
        }
    )
    return result


def delta_text(
    result: dict[str, Any] | None, digits: int, *, with_p: bool = False, total: int = N_DIRS
) -> str:
    if not result:
        return "-"
    text = signed(result.get("delta"), digits)
    ci = result.get("bootstrap_ci_95")
    if ci:
        text += f" [{ci[0]:+.{digits}f}, {ci[1]:+.{digits}f}]"
    if with_p and result.get("p_value") is not None:
        p_value = fmt_p(result["p_value"])
        text += f" p{p_value}" if p_value.startswith("<") else f" p={p_value}"
    return with_coverage(text, int(result.get("n_directions") or 0), total)


def p_text(result: dict[str, Any] | None) -> str:
    return fmt_p(result.get("p_value")) if result else "-"


# --------------------------------------------------------------------------
# Sections
# --------------------------------------------------------------------------


def _status_counts(ctx: Context) -> Counter[str]:
    counts: Counter[str] = Counter()
    status_dir = ctx.args.status_dir
    if not status_dir.is_dir():
        return counts
    for system in ctx.grid:
        payload = _read_json(status_dir / f"{system.name}.json")
        if not isinstance(payload, dict):
            continue
        for key in ("state", "status", "stage", "phase"):
            value = payload.get(key)
            if isinstance(value, str) and value:
                counts[value] += 1
                break
    return counts


def section_coverage(ctx: Context) -> list[str]:
    grid = ctx.grid
    complete = [s for s in grid if ctx.complete(s)]
    partial = [s for s in grid if s.name in ctx.runs and not ctx.complete(s)]
    missing = [s for s in grid if s.name not in ctx.runs]
    loras = [s for s in ctx.systems.values() if s.repaired]
    lora_done = [s for s in loras if ctx.complete(s)]
    specialists = [s for s in grid if s.scope != "multi" and s.name in ctx.runs]
    all_ten = Counter(s.scope for s in specialists if ctx.runs[s.name].all_directions)
    expected = f"{len(grid)}" if ctx.manifest_found else f"{len(grid)} (inferred, no manifest)"
    lines = ["## Coverage", ""]
    lines.append(
        f"Grid models complete: **{len(complete)}/{expected}** (BLEU, chrF++ and COMET on every "
        f"direction the model is evaluated on: all {N_DIRS} for multi models, the model's own "
        "pruned directions for pair and dir models; a `-lora` system follows its source); "
        f"{len(partial)} partially scored; {len(missing)} without a run. LoRA-repaired "
        f"systems: {len(lora_done)}/{len(loras)} complete."
    )
    lines.append("")
    lines.append(
        f"Specialists scored on all {N_DIRS} directions, which feed the transfer section: "
        f"{all_ten['pair']} pair and {all_ten['dir']} dir models, of {len(specialists)} "
        "specialists with a run."
    )
    with_mx = {name: run for name, run in ctx.runs.items() if run.metricx_directions()}
    mx_grid = [s for s in grid if s.name in with_mx]
    mx_full = [
        s
        for s in mx_grid
        if set(s.expected_directions) <= set(with_mx[s.name].metricx_directions())
    ]
    mx_lora = [s for s in loras if s.name in with_mx]
    lines.append("")
    lines.append(
        f"MetricX-24 (`scores.metricx.json`): {len(mx_grid)}/{len(grid)} grid models "
        f"({len(mx_full)} on every in-scope direction), {len(mx_lora)}/{len(loras)} LoRA "
        f"systems, dense baseline {'yes' if DENSE in with_mx else 'no'}."
    )
    dense = ctx.dense
    if dense is not None:
        comet, _ = macro(single(ctx, DENSE), "comet")
        bleu, _ = macro(single(ctx, DENSE), "bleu")
        chrf, _ = macro(single(ctx, DENSE), "chrf")
        metricx, n_mx = macro(single(ctx, DENSE), "metricx")
        mx_text = f", MetricX-24 {with_coverage(f3(metricx), n_mx)}" if metricx is not None else ""
        lines.append("")
        lines.append(
            f"Dense baseline `{DENSE}` (run {dense.run_id}, {len(dense.scored_directions)}/"
            f"{N_DIRS} directions): COMET {f4(comet)}, BLEU {f2(bleu)}, chrF++ {f2(chrf)}"
            f"{mx_text}."
        )
    else:
        lines.extend(["", f"Dense baseline `{DENSE}`: no run found."])
    states = _status_counts(ctx)
    if states:
        listed = ", ".join(f"{state} {count}" for state, count in sorted(states.items()))
        lines.extend(["", f"grid-state status files: {listed}."])
    lines.append("")
    rows = []
    for method, calib, sparsity in ctx.configs():
        cells = []
        for scope in SCOPES:
            members = [
                s
                for s in grid
                if s.cell[:4] == (method, calib, sparsity, scope)
                and (scope != "multi" or s.scope_key == "")
            ]
            done = sum(1 for s in members if ctx.complete(s))
            part = sum(1 for s in members if s.name in ctx.runs and not ctx.complete(s))
            text = f"{done}/{len(members)}" if members else "-"
            if part:
                text += f" (+{part} partial)"
            cells.append(text)
        rows.append([config_label(method, calib, sparsity), *cells])
    lines.extend(md_table(["Config", "multi", "pair", "dir"], rows))
    if partial:
        lines.append("")
        listed = ", ".join(
            f"{s.name} ({ctx.in_scope_scored(s)}/{len(s.expected_directions)})" for s in partial
        )
        lines.append(f"Partially scored (in-scope directions scored): {listed}.")
    off_grid = [
        s
        for s in grid
        if s.method not in METHODS or s.calib not in CALIBS or s.sparsity not in SPARSITIES
    ]
    if off_grid:
        lines.append("")
        lines.append(
            "Systems outside the method x calib x sparsity grid (in long.csv, and in the tables "
            f"as extra rows where they fit): {', '.join(s.name for s in off_grid)}."
        )
    ctx.results["coverage"] = {
        "rule": (
            "complete = BLEU, chrF++ and COMET on every expected direction: all ten for dense "
            "and multi (and their repairs), pruned_directions for pair and dir (and theirs)"
        ),
        "expected": len(grid),
        "manifest_found": ctx.manifest_found,
        "complete": len(complete),
        "partial": len(partial),
        "missing": len(missing),
        "partial_systems": {s.name: ctx.in_scope_scored(s) for s in partial},
        "partial_expected": {s.name: len(s.expected_directions) for s in partial},
        "specialists_all_directions": {"pair": all_ten["pair"], "dir": all_ten["dir"]},
        "lora_systems": len(loras),
        "lora_complete": len(lora_done),
        "metricx": {
            "grid_models": len(mx_grid),
            "grid_models_all_in_scope": len(mx_full),
            "lora_systems": len(mx_lora),
            "dense": DENSE in with_mx,
        },
        "status_counts": dict(states),
    }
    return lines


def section_headline(ctx: Context) -> list[str]:
    lines = ["## Headline: one multi model vs matched specialists", ""]
    lines.append(
        f"Macro over the {N_DIRS} directions. For pair and dir, each direction is translated "
        "by the matching specialised model: `pair-<l>` for both directions of language l, "
        "`dir-<d>` for direction d. Bold marks the best complete scope per metric. The "
        "delta columns are specialist minus multi on the macro over the shared directions, "
        "so they equal the difference of the macro columns, with a stratified paired "
        "bootstrap 95% CI and two-sided p: each resample redraws segment positions within "
        "each direction, the same positions for both models, and averages the per-direction "
        "mean differences with equal weight. "
        f"`(n/{N_DIRS})` means only n directions exist yet."
    )
    lines.append("")
    metrics = (("comet", f4), ("bleu", f2), ("chrf", f2))
    header = ["Method", "Calib", "Sparsity"]
    for scope in SCOPES:
        header.extend([f"{scope} COMET", f"{scope} BLEU", f"{scope} chrF++"])
    header.extend(["pair - multi COMET", "dir - multi COMET"])
    rows: list[list[str]] = []
    dense = single(ctx, DENSE)
    if dense:
        cells = []
        for key, fmt in metrics:
            value, n = macro(dense, key)
            cells.append(with_coverage(fmt(value), n))
        rows.append(["dense", "-", "0%", *(cells * len(SCOPES)), "-", "-"])
    records = []
    for method, calib, sparsity in ctx.configs():
        values: dict[str, dict[str, tuple[float | None, int]]] = {}
        for scope in SCOPES:
            comp = composite(ctx, method, calib, sparsity, scope)
            values[scope] = {key: macro(comp, key) for key, _ in metrics}
        cells = []
        for key, fmt in metrics:
            complete = {
                scope: values[scope][key][0]
                for scope in SCOPES
                if values[scope][key][0] is not None and values[scope][key][1] == N_DIRS
            }
            best = max(complete.values()) if len(complete) >= 2 else None
            for scope in SCOPES:
                value, n = values[scope][key]
                text = with_coverage(fmt(value), n)
                if best is not None and scope in complete and fmt(complete[scope]) == fmt(best):
                    text = f"**{text}**"
                cells.append(text)
        # Reorder from metric-major to scope-major to match the header.
        ordered = [cells[m * len(SCOPES) + s] for s in range(len(SCOPES)) for m in range(3)]
        multi = composite(ctx, method, calib, sparsity, "multi")
        deltas = {
            scope: comet_contrast(ctx, multi, composite(ctx, method, calib, sparsity, scope))
            for scope in ("pair", "dir")
        }
        rows.append(
            [
                METHOD_LABEL.get(method, method),
                calib,
                f"{sparsity}%",
                *ordered,
                delta_text(deltas["pair"], 4, with_p=True),
                delta_text(deltas["dir"], 4, with_p=True),
            ]
        )
        for scope in SCOPES:
            values[scope]["metricx"] = macro(
                composite(ctx, method, calib, sparsity, scope), "metricx"
            )
        record_keys = (*(key for key, _ in metrics), "metricx")
        records.append(
            {
                "method": method,
                "calib": calib,
                "sparsity": sparsity,
                **{
                    scope: {key: values[scope][key][0] for key in record_keys}
                    | {"n_directions": {key: values[scope][key][1] for key in record_keys}}
                    for scope in SCOPES
                },
                "pair_minus_multi_comet": deltas["pair"],
                "dir_minus_multi_comet": deltas["dir"],
                # Filled by section_headline_metricx.
                "pair_minus_multi_metricx": None,
                "dir_minus_multi_metricx": None,
            }
        )
    lines.extend(md_table(header, rows, labels=3))
    ctx.results["headline"] = records
    return lines


def section_headline_metricx(ctx: Context) -> list[str]:
    """The COMET headline again on MetricX-24, where lower is better."""
    lines = ["## Headline, MetricX-24 (lower is better)", ""]
    records = {(r["method"], r["calib"], r["sparsity"]): r for r in ctx.results["headline"]}
    if not ctx.has_metricx:
        lines.append(f"No MetricX-24 scores (`scores.metricx.json`) for suite `{ctx.suite}` yet.")
        return lines
    lines.append(
        "MetricX-24 is an error score in [0, 25]: lower is better. Same scope composites and "
        "macro over directions as the COMET headline; bold marks the lowest (best) complete "
        "scope. The delta columns are specialist minus multi on macro MetricX, with the same "
        "stratified paired bootstrap 95% CI and two-sided p: **a negative delta means the "
        f"specialist is better**. `-` means no MetricX scores yet; `(n/{N_DIRS})` means only "
        "n directions have them."
    )
    lines.append("")
    header = [
        "Method",
        "Calib",
        "Sparsity",
        *[f"{scope} MetricX" for scope in SCOPES],
        "pair - multi MetricX",
        "dir - multi MetricX",
    ]
    rows: list[list[str]] = []
    dense = single(ctx, DENSE)
    if dense:
        value, n = macro(dense, "metricx")
        rows.append(["dense", "-", "0%", *[with_coverage(f3(value), n)] * len(SCOPES), "-", "-"])
    for method, calib, sparsity in ctx.configs():
        values = {
            scope: macro(composite(ctx, method, calib, sparsity, scope), "metricx")
            for scope in SCOPES
        }
        complete = {
            scope: value
            for scope, (value, n) in values.items()
            if value is not None and n == N_DIRS
        }
        best = min(complete.values()) if len(complete) >= 2 else None
        cells = []
        for scope in SCOPES:
            value, n = values[scope]
            text = with_coverage(f3(value), n)
            if best is not None and scope in complete and f3(complete[scope]) == f3(best):
                text = f"**{text}**"
            cells.append(text)
        multi = composite(ctx, method, calib, sparsity, "multi")
        deltas = {
            scope: metricx_contrast(ctx, multi, composite(ctx, method, calib, sparsity, scope))
            for scope in ("pair", "dir")
        }
        record = records.get((method, calib, sparsity))
        if record is not None:
            record["pair_minus_multi_metricx"] = deltas["pair"]
            record["dir_minus_multi_metricx"] = deltas["dir"]
        rows.append(
            [
                METHOD_LABEL.get(method, method),
                calib,
                f"{sparsity}%",
                *cells,
                delta_text(deltas["pair"], 3, with_p=True),
                delta_text(deltas["dir"], 3, with_p=True),
            ]
        )
    lines.extend(md_table(header, rows, labels=3))
    return lines


def _contrast_section(
    ctx: Context,
    title: str,
    intro: str,
    groups: Sequence[tuple[list[str], tuple[str, str, int, str], tuple[str, str, int, str]]],
    names: tuple[str, str],
    header_labels: list[str],
    result_key: str,
) -> list[str]:
    first, second = names
    intro += (
        f" MetricX-24 is lower-is-better; its delta is also {second} minus {first}, so a "
        f"negative MetricX delta means {second} is better."
    )
    lines = [title, "", intro, ""]
    header = [
        *header_labels,
        f"COMET {first}",
        f"COMET {second}",
        f"{second} - {first} [95% CI]",
        "p",
        f"BLEU {first}",
        f"BLEU {second}",
        f"{second} - {first} [95% CI]",
        "p",
        f"MetricX {first}",
        f"MetricX {second}",
        f"MetricX {second} - {first} [95% CI]",
        "p",
    ]
    rows = []
    records = []
    for labels, base_key, system_key in groups:
        base = composite(ctx, *base_key)
        system = composite(ctx, *system_key)
        comet = comet_contrast(ctx, base, system)
        bleu = bleu_contrast(ctx, base, system)
        metricx = metricx_contrast(ctx, base, system)
        rows.append(
            [
                *labels,
                f4(comet["baseline_score"]) if comet else "-",
                f4(comet["system_score"]) if comet else "-",
                delta_text(comet, 4),
                p_text(comet),
                f2(bleu["baseline_score"]) if bleu else "-",
                f2(bleu["system_score"]) if bleu else "-",
                delta_text(bleu, 2),
                p_text(bleu),
                f3(metricx["baseline_score"]) if metricx else "-",
                f3(metricx["system_score"]) if metricx else "-",
                delta_text(metricx, 3),
                p_text(metricx),
            ]
        )
        records.append(
            {
                "baseline": dict(
                    zip(("method", "calib", "sparsity", "scope"), base_key, strict=True)
                ),
                "system": dict(
                    zip(("method", "calib", "sparsity", "scope"), system_key, strict=True)
                ),
                "comet": comet,
                "bleu": bleu,
                "metricx": metricx,
            }
        )
    lines.extend(md_table(header, rows, labels=len(header_labels)))
    ctx.results[result_key] = records
    return lines


def section_ref_vs_gen(ctx: Context) -> list[str]:
    groups = [
        (
            [METHOD_LABEL.get(method, method), f"{sparsity}%", scope],
            (method, "ref", sparsity, scope),
            (method, "gen", sparsity, scope),
        )
        for method in ctx.methods
        for sparsity in ctx.sparsities
        for scope in SCOPES
    ]
    intro = (
        "Calibrating on prompt + dense-generated continuation (gen) instead of prompt + "
        "reference (ref). Scores are the scope composites of the headline, over the "
        "directions both sides have, each direction weighted equally. COMET and MetricX-24 use "
        "a stratified paired bootstrap of the macro (segment positions resampled within each "
        "direction, the same for both sides); BLEU does the same, recomputing corpus BLEU "
        "per direction from sufficient statistics."
    )
    return _contrast_section(
        ctx,
        "## Ref vs gen calibration text",
        intro,
        groups,
        ("ref", "gen"),
        ["Method", "Sparsity", "Scope"],
        "ref_vs_gen",
    )


def section_method(ctx: Context) -> list[str]:
    groups = [
        (
            [calib, f"{sparsity}%", scope],
            ("flap", calib, sparsity, scope),
            ("slimgpt", calib, sparsity, scope),
        )
        for calib in ctx.calibs
        for sparsity in ctx.sparsities
        for scope in SCOPES
    ]
    intro = (
        "Deltas are SlimGPT minus FLAP at the same calibration text, sparsity and scope; "
        "positive means SlimGPT is better on COMET and BLEU. Same composites and tests as above."
    )
    return _contrast_section(
        ctx,
        "## SlimGPT vs FLAP",
        intro,
        groups,
        ("FLAP", "SlimGPT"),
        ["Calib", "Sparsity", "Scope"],
        "slimgpt_vs_flap",
    )


def _multi_rows(ctx: Context) -> list[tuple[str, Composite]]:
    rows = [(f"dense ({DENSE})", single(ctx, DENSE))]
    for method, calib, sparsity in ctx.configs():
        rows.append(
            (
                config_label(method, calib, sparsity),
                composite(ctx, method, calib, sparsity, "multi"),
            )
        )
    return rows


def section_per_direction(ctx: Context) -> list[str]:
    lines = ["## Per-direction scores, multi models", ""]
    lines.append(
        "One row per multi model (plus dense). into-EN and out-of-EN are macros over the "
        "five directions on each side."
    )
    per_direction: dict[str, dict[str, dict[str, float | None]]] = {}
    for key, fmt, title in (
        ("comet", f4, "COMET"),
        ("bleu", f2, "BLEU"),
        ("metricx", f3, "MetricX-24 (lower is better)"),
    ):
        lines.extend(["", f"### {title}", ""])
        if key == "metricx" and not ctx.has_metricx:
            lines.append("No MetricX-24 scores yet.")
        header = ["System", *INTO_EN, "into-EN", *OUT_OF_EN, "out-of-EN", "all"]
        rows = []
        for label, comp in _multi_rows(ctx):
            into, n_into = macro(comp, key, INTO_EN)
            out, n_out = macro(comp, key, OUT_OF_EN)
            overall, n_all = macro(comp, key)
            cells = [fmt(comp[d][1].value(d, key)) if d in comp else "-" for d in DIRECTIONS]
            rows.append(
                [
                    label,
                    *cells[:5],
                    with_coverage(fmt(into), n_into, 5),
                    *cells[5:],
                    with_coverage(fmt(out), n_out, 5),
                    with_coverage(fmt(overall), n_all),
                ]
            )
            per_direction.setdefault(label, {})[key] = {
                d: (comp[d][1].value(d, key) if d in comp else None) for d in DIRECTIONS
            }
        if key != "metricx" or ctx.has_metricx:
            lines.extend(md_table(header, rows))
    ctx.results["multi_per_direction"] = per_direction
    return lines


def section_sparsity(ctx: Context) -> list[str]:
    header = ["Method", "Calib", "Scope", "0% (dense)", *[f"{s}%" for s in ctx.sparsities]]
    records = [
        {"method": method, "calib": calib, "scope": scope, "comet": {}, "metricx": {}}
        for method in ctx.methods
        for calib in ctx.calibs
        for scope in SCOPES
    ]
    lines: list[str] = []
    dense_values: dict[str, float | None] = {}
    for key, fmt, title in (
        ("comet", f4, "## Sparsity curve (macro COMET)"),
        ("metricx", f3, "## Sparsity curve (macro MetricX-24, lower is better)"),
    ):
        if lines:
            lines.append("")
        lines.extend([title, ""])
        dense_value, _ = macro(single(ctx, DENSE), key)
        dense_values[key] = dense_value
        rows = []
        for record in records:
            method, calib, scope = record["method"], record["calib"], record["scope"]
            cells = []
            for sparsity in ctx.sparsities:
                value, n = macro(composite(ctx, method, calib, sparsity, scope), key)
                cells.append(with_coverage(fmt(value), n))
                record[key][str(sparsity)] = value
            rows.append([METHOD_LABEL.get(method, method), calib, scope, fmt(dense_value), *cells])
        if key == "metricx" and not ctx.has_metricx:
            lines.append("No MetricX-24 scores yet.")
        else:
            lines.extend(md_table(header, rows, labels=3))
    ctx.results["sparsity_curve"] = {
        "dense": dense_values["comet"],
        "dense_metricx": dense_values["metricx"],
        "rows": records,
    }
    return lines


PAIR_CATEGORIES = ("own", "other")
DIR_CATEGORIES = ("own", "reverse", "same_target", "same_source", "other")


def transfer_category(scope: str, key: str, direction: str) -> str:
    if scope == "pair":
        return "own" if pair_of(direction) == key else "other"
    if direction == key:
        return "own"
    if direction == reverse(key):
        return "reverse"
    if direction.split("-")[1] == key.split("-")[1]:
        return "same_target"
    if direction.split("-")[0] == key.split("-")[0]:
        return "same_source"
    return "other"


def compute_transfer(ctx: Context, method: str, calib: str, sparsity: int) -> dict[str, Any]:
    multi = composite(ctx, method, calib, sparsity, "multi")
    multi_comet = {d: multi[d][1].value(d, "comet") for d in multi}
    data: dict[str, Any] = {"multi": multi_comet, "multi_system": None}
    if multi:
        data["multi_system"] = next(iter(multi.values()))[0].name
    for scope, keys, categories in (
        ("pair", PAIR_LANGS, PAIR_CATEGORIES),
        ("dir", DIRECTIONS, DIR_CATEGORIES),
    ):
        rows = []
        for key in keys:
            system = ctx.index.get((method, calib, sparsity, scope, key))
            run = ctx.runs.get(system.name) if system else None
            # Only models scored on all ten directions: one evaluated on its own
            # directions alone has no transfer cells, and a part-scored one would
            # weight the categories unevenly.
            if system is None or run is None or not run.all_directions:
                continue
            cells = {}
            for direction in DIRECTIONS:
                comet = run.value(direction, "comet")
                base = multi_comet.get(direction)
                cells[direction] = {
                    "category": transfer_category(scope, key, direction),
                    "comet": comet,
                    "bleu": run.value(direction, "bleu"),
                    "chrf": run.value(direction, "chrf"),
                    "multi_comet": base,
                    "comet_minus_multi": (
                        comet - base if comet is not None and base is not None else None
                    ),
                }
            rows.append({"key": key, "system": system.name, "cells": cells})
        summary = {}
        for category in categories:
            chosen = [c for r in rows for c in r["cells"].values() if c["category"] == category]
            values = [c["comet"] for c in chosen if c["comet"] is not None]
            deltas = [c["comet_minus_multi"] for c in chosen if c["comet_minus_multi"] is not None]
            summary[category] = {
                "comet": _mean(values),
                "n_cells": len(values),
                "minus_multi": _mean(deltas),
                "n_paired": len(deltas),
            }
        data[scope] = rows
        data[f"{scope}_summary"] = summary
    return data


def _summary_cell(entry: dict[str, Any]) -> str:
    if entry["comet"] is None:
        return "-"
    text = f4(entry["comet"])
    if entry["minus_multi"] is not None:
        text += f" ({signed(entry['minus_multi'], 4)})"
    return text


def section_transfer(ctx: Context) -> list[str]:
    lines = ["## Transfer", ""]
    pair_rows = []
    dir_rows = []
    for method, calib, sparsity in ctx.configs():
        config = f"{method}{sparsity}-{calib}"
        data = compute_transfer(ctx, method, calib, sparsity)
        ctx.transfer[config] = data
        label = config_label(method, calib, sparsity)
        pair = data["pair_summary"]
        own, other = pair["own"]["comet"], pair["other"]["comet"]
        pair_rows.append(
            [
                label,
                f"{len(data['pair'])}/5",
                _summary_cell(pair["own"]),
                _summary_cell(pair["other"]),
                signed(own - other, 4) if own is not None and other is not None else "-",
            ]
        )
        summary = data["dir_summary"]
        own, rev = summary["own"]["comet"], summary["reverse"]["comet"]
        dir_rows.append(
            [
                label,
                f"{len(data['dir'])}/10",
                *[_summary_cell(summary[c]) for c in DIR_CATEGORIES],
                signed(own - rev, 4) if own is not None and rev is not None else "-",
            ]
        )
        if data["pair"] or data["dir"]:
            write_transfer_files(ctx, config, label, data)
    n_pair = sum(len(data["pair"]) for data in ctx.transfer.values())
    n_dir = sum(len(data["dir"]) for data in ctx.transfer.values())
    n_specialists = sum(1 for s in ctx.grid if s.scope != "multi" and s.name in ctx.runs)
    lines.append(
        f"Only specialised models scored on all {N_DIRS} directions enter: {n_pair} pair and "
        f"{n_dir} dir models here, of {n_specialists} specialists with a run (the Models "
        "columns count them per config). Specialists evaluated only on their own directions, "
        "as on the full suite, are left out. Each cell is the mean "
        "COMET over the matrix cells in that category, with the mean of (specialist minus the "
        "multi model of the same config, on the same direction) in parentheses. Pair: own = "
        "the two directions of the model's language, other = the remaining eight. Dir: own = "
        "the calibration direction, reverse = its reverse, same target / same source = other "
        "directions sharing the target / source language (into-English models have same-target "
        "neighbours, out-of-English models same-source ones), other = the rest. Full matrices "
        "are in `transfer/<config>.md` and `.csv`."
    )
    lines.extend(["", "### Pair models (5 x 10)", ""])
    lines.extend(md_table(["Config", "Models", "Own pair", "Other 8", "Own - other"], pair_rows))
    lines.extend(["", "### Direction models (10 x 10)", ""])
    lines.extend(
        md_table(
            [
                "Config",
                "Models",
                "Own",
                "Reverse",
                "Same target",
                "Same source",
                "Other",
                "Own - reverse",
            ],
            dir_rows,
        )
    )
    ctx.results["transfer_models"] = {
        "rule": f"specialists scored on all {N_DIRS} directions only",
        "pair": n_pair,
        "dir": n_dir,
        "specialists_with_a_run": n_specialists,
    }
    ctx.results["transfer"] = {
        config: {
            "multi_system": data["multi_system"],
            "pair_models": len(data["pair"]),
            "dir_models": len(data["dir"]),
            "pair_summary": data["pair_summary"],
            "dir_summary": data["dir_summary"],
        }
        for config, data in ctx.transfer.items()
    }
    return lines


def write_transfer_files(ctx: Context, config: str, label: str, data: dict[str, Any]) -> None:
    folder = ctx.out / "transfer"
    folder.mkdir(parents=True, exist_ok=True)
    dense = ctx.dense
    lines = [f"# Transfer matrices: {label}", ""]
    lines.append(
        "Rows are specialised models, columns the evaluated direction. Bracketed cells are the "
        "model's own directions. The multi and dense rows are references."
    )
    for scope, title in (("pair", "Pair models"), ("dir", "Direction models")):
        rows = data[scope]
        if not rows:
            continue
        for key, fmt, heading in (
            ("comet", f4, "COMET"),
            ("comet_minus_multi", lambda v: signed(v, 4), "COMET minus multi"),
            ("bleu", f2, "BLEU"),
        ):
            lines.extend(["", f"## {title}: {heading}", ""])
            table = []
            for row in rows:
                cells = []
                for direction in DIRECTIONS:
                    cell = row["cells"][direction]
                    text = fmt(cell[key])
                    if cell["category"] == "own" and text != "-":
                        text = f"[{text}]"
                    cells.append(text)
                table.append([f"{scope}-{row['key']}", *cells])
            if key in ("comet", "bleu"):
                metric = "comet" if key == "comet" else "bleu"
                multi = composite(ctx, *config_parts(config), "multi")
                table.append(
                    [
                        "multi",
                        *[
                            fmt(multi[d][1].value(d, metric)) if d in multi else "-"
                            for d in DIRECTIONS
                        ],
                    ]
                )
                if dense is not None:
                    table.append(["dense", *[fmt(dense.value(d, metric)) for d in DIRECTIONS]])
            lines.extend(md_table(["Model", *DIRECTIONS], table))
    _write_text(folder / f"{config}.md", "\n".join(lines) + "\n")

    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(
        [
            "config",
            "scope",
            "system",
            "calibrated_on",
            "direction",
            "category",
            "comet",
            "bleu",
            "chrf",
            "multi_comet",
            "comet_minus_multi",
        ]
    )
    for scope in ("pair", "dir"):
        for row in data[scope]:
            for direction in DIRECTIONS:
                cell = row["cells"][direction]
                writer.writerow(
                    [
                        config,
                        scope,
                        row["system"],
                        row["key"],
                        direction,
                        cell["category"],
                        *[
                            _csv_value(cell[k])
                            for k in ("comet", "bleu", "chrf", "multi_comet", "comet_minus_multi")
                        ],
                    ]
                )
    _write_text(folder / f"{config}.csv", buffer.getvalue())


def config_parts(config: str) -> tuple[str, str, int]:
    match = re.fullmatch(r"(?P<method>.+?)(?P<pct>\d+)-(?P<calib>.+)", config)
    assert match is not None
    return match["method"], match["calib"], int(match["pct"])


def section_behaviour(ctx: Context) -> list[str]:
    lines = ["## Behaviour, multi models", ""]
    lines.append(
        "Macro over directions of the per-direction rates (share of all segments). Hit budget "
        "= generation ran out of tokens without EOS; truncated = the hypothesis itself was cut "
        "off; repetition = looping flagged by post-processing; length ratio = hypothesis / "
        "reference characters. All systems are in long.csv."
    )
    lines.append("")
    fields = (
        ("off_target_rate", "Off-target", pct),
        ("repetition_rate", "Repetition", pct),
        ("budget_hit_rate", "Hit budget", pct),
        ("truncation_rate", "Truncated", pct),
        ("empty_rate", "Empty", pct),
        ("source_copy_rate", "Source copy", pct),
        ("length_ratio", "Length ratio", lambda v: "-" if v is None else f"{v:.3f}"),
    )
    rows = []
    records = {}
    for label, comp in _multi_rows(ctx):
        cells = []
        record = {}
        for key, _, fmt in fields:
            value, n = macro(comp, key)
            cells.append(with_coverage(fmt(value), n))
            record[key] = value
        rows.append([label, *cells])
        records[label] = record
    lines.extend(md_table(["System", *[title for _, title, _ in fields]], rows))
    ctx.results["behaviour_multi"] = records
    return lines


def recovered_share(
    dense: float | None,
    pruned: float | None,
    repaired: float | None,
    *,
    higher_is_better: bool = True,
) -> float | None:
    """Share of the pruning loss that the repair won back.

    The loss is how far the pruned model sits from dense in the bad direction
    (below it for COMET, above it for MetricX-24), the gain how far the repair
    moved in the good direction. Both flip sign together for a lower-is-better
    metric, so the ratio equals ``(repaired - pruned) / (dense - pruned)``
    either way; it is written out so the direction is explicit. 1 = the whole
    gap closed, 0 = nothing, negative = the repair made it worse.
    """
    if dense is None or pruned is None or repaired is None:
        return None
    sign = 1.0 if higher_is_better else -1.0
    loss = sign * (dense - pruned)
    if abs(loss) <= 1e-9:
        return None
    return sign * (repaired - pruned) / loss


def section_repair(ctx: Context) -> list[str]:
    lines = ["## Repair (LoRA)", ""]
    loras = sorted((s for s in ctx.systems.values() if s.repaired), key=lambda s: s.name)
    if not loras:
        lines.append("No `-lora` systems found yet.")
        ctx.results["repair"] = []
        return lines
    lines.append(
        "Macro over the directions all three systems have (a specialist's repair is evaluated "
        "on its own directions on the full suite). Recovered = share of the pruning loss "
        "won back, (repaired - pruned) / (dense - pruned): 1 means the repair closed the whole "
        "gap to dense, 0 means nothing; the ratio is the same for a lower-is-better metric. The "
        "deltas are repaired minus pruned on the macro, stratified paired bootstrap; on "
        "MetricX-24 (lower is better) a negative delta means the repair helped."
    )
    lines.append("")
    rows = []
    records = []
    dense = single(ctx, DENSE)
    per_direction_rows = []
    metrics: list[tuple[str, Any, str, int, bool]] = [
        ("comet", f4, "COMET", 4, True),
        ("bleu", f2, "BLEU", 2, True),
        ("chrf", f2, "chrF++", 2, True),
        ("metricx", f3, "MetricX-24", 3, False),
    ]
    for lora in loras:
        source = single(ctx, lora.source)
        repaired = single(ctx, lora.name)
        expected = len(lora.expected_directions)
        shared = [d for d in DIRECTIONS if d in source and d in repaired and d in dense]
        record: dict[str, Any] = {"system": lora.name, "source": lora.source, "metrics": {}}
        for key, fmt, title, digits, higher_is_better in metrics:
            # Per metric, so a MetricX file missing for one of the three
            # shrinks the MetricX macro rather than mixing direction sets.
            have = [
                d
                for d in shared
                if all(comp[d][1].value(d, key) is not None for comp in (dense, source, repaired))
            ]
            d_value, _ = macro(dense, key, have)
            p_value, n = macro(source, key, have)
            r_value, _ = macro(repaired, key, have)
            recovered = recovered_share(
                d_value, p_value, r_value, higher_is_better=higher_is_better
            )
            gain = r_value - p_value if r_value is not None and p_value is not None else None
            if key != "metricx" or ctx.has_metricx:
                rows.append(
                    [
                        lora.name,
                        title,
                        fmt(d_value),
                        fmt(p_value),
                        fmt(r_value),
                        with_coverage(signed(gain, digits), n, expected),
                        pct(recovered),
                    ]
                )
            record["metrics"][key] = {
                "dense": d_value,
                "pruned": p_value,
                "repaired": r_value,
                "recovered": recovered,
                "higher_is_better": higher_is_better,
                "n_directions": n,
            }
        record["comet_repaired_minus_pruned"] = comet_contrast(ctx, source, repaired)
        record["metricx_repaired_minus_pruned"] = metricx_contrast(ctx, source, repaired)
        records.append(record)
        for label, comp, own in (
            (f"dense ({DENSE})", dense, N_DIRS),
            (lora.source, source, expected),
            (lora.name, repaired, expected),
        ):
            # Over the shared directions, so a pair repair on the full suite is
            # not set against a ten-direction dense macro; each system's own
            # directions while the three share none yet.
            if shared:
                overall, n = macro(comp, "comet", shared)
                total = len(shared)
            else:
                overall, n = macro(comp, "comet")
                total = own
            per_direction_rows.append(
                [
                    label,
                    *[f4(comp[d][1].value(d, "comet")) if d in comp else "-" for d in DIRECTIONS],
                    with_coverage(f4(overall), n, total),
                ]
            )
    lines.extend(
        md_table(
            ["System", "Metric", "Dense", "Pruned", "Repaired", "Repaired - pruned", "Recovered"],
            rows,
            labels=2,
        )
    )
    by_name = {lora.name: lora for lora in loras}
    lines.extend(["", "Repaired - pruned COMET with 95% CI:", ""])
    for record in records:
        contrast = record["comet_repaired_minus_pruned"]
        total = len(by_name[record["system"]].expected_directions)
        lines.append(f"- {record['system']}: {delta_text(contrast, 4, with_p=True, total=total)}")
    if ctx.has_metricx:
        lines.extend(
            ["", "Repaired - pruned MetricX-24 with 95% CI (negative = the repair helped):", ""]
        )
        for record in records:
            contrast = record["metricx_repaired_minus_pruned"]
            total = len(by_name[record["system"]].expected_directions)
            lines.append(
                f"- {record['system']}: {delta_text(contrast, 3, with_p=True, total=total)}"
            )
    lines.extend(["", "### Per-direction COMET", ""])
    lines.append(
        "The macro column averages the directions all three systems have (each system's own "
        "while they share none yet)."
    )
    lines.append("")
    lines.extend(md_table(["System", *DIRECTIONS, "macro"], per_direction_rows))
    ctx.results["repair"] = records
    return lines


# Structure ------------------------------------------------------------------


def _subnetwork_candidates(ctx: Context, system: System) -> list[Path]:
    run = ctx.runs.get(system.name)
    paths: list[Path] = []
    if run is not None and run.checkpoint:
        paths.append(Path(run.checkpoint) / "subnetwork.json")
    paths.append(ctx.args.checkpoints_root / system.name / "subnetwork.json")
    if run is not None and run.compression.get("subnetwork"):
        declared = Path(str(run.compression["subnetwork"]))
        paths.append(declared if declared.is_absolute() else REPO_ROOT / declared)
    paths.append(REPO_ROOT / "subnetworks" / f"{system.name}.json")
    return paths


def _load_subnetwork(ctx: Context, system: System) -> dict[str, Any] | None:
    """Per-layer kept counts from a subnetwork descriptor, cached by file signature."""
    for path in _subnetwork_candidates(ctx, system):
        signature = _file_sig(path)
        if signature is None:
            continue
        key = f"{system.name}"
        cached = ctx.cache.get("structure", key, [str(path), signature])
        if cached is not None:
            return cached
        payload = _read_json(path, ctx.warnings)
        if not isinstance(payload, dict):
            continue
        components = {}
        for name, component in (payload.get("components") or {}).items():
            kept = component.get("kept") or {}
            layers = sorted((k for k in kept if k.isdigit()), key=int)
            components[name] = {
                "total_per_layer": int(component.get("total") or 0),
                "kept_per_layer": [len(kept[layer]) for layer in layers],
            }
        dims = dict(LLAMA_7B_DIMS)
        config = _read_json(path.parent / "config.json")
        if isinstance(config, dict):
            for name in ("hidden_size", "vocab_size", "tie_word_embeddings"):
                if config.get(name) is not None:
                    dims[name] = config[name]
            heads = config.get("num_attention_heads")
            if config.get("head_dim"):
                dims["head_dim"] = config["head_dim"]
            elif heads and config.get("hidden_size"):
                dims["head_dim"] = config["hidden_size"] // heads
        data = {"path": str(path), "components": components, "dims": dims}
        ctx.cache.put("structure", key, [str(path), signature], data)
        return data
    return None


def parameter_fractions(
    components: dict[str, dict[str, Any]], dims: dict[str, Any]
) -> dict[str, Any]:
    """Removed parameter shares from kept head and channel counts.

    Assumes Llama layers: a head carries q, k, v and o slices (4 x hidden x
    head_dim, no grouped-query attention), an FFN channel carries gate, up and
    down slices (3 x hidden). Norms and embeddings are never pruned.
    """
    hidden = int(dims["hidden_size"])
    per_head = 4 * hidden * int(dims["head_dim"])
    per_channel = 3 * hidden
    heads = components.get("attention_heads") or {}
    ffn = components.get("ffn_channels") or {}
    n_layers = int(heads.get("layers") or ffn.get("layers") or 0)
    if not n_layers:
        return {}
    heads_total = int(heads.get("total_per_layer") or dims["num_attention_heads"])
    ffn_total = int(ffn.get("total_per_layer") or dims["intermediate_size"])
    heads_kept = int(heads.get("kept", heads_total * n_layers))
    ffn_kept = int(ffn.get("kept", ffn_total * n_layers))
    layer_params = n_layers * (heads_total * per_head + ffn_total * per_channel + 2 * hidden)
    embeddings = int(dims["vocab_size"]) * hidden * (1 if dims.get("tie_word_embeddings") else 2)
    total = layer_params + embeddings + hidden
    removed = (heads_total * n_layers - heads_kept) * per_head + (
        ffn_total * n_layers - ffn_kept
    ) * per_channel
    return {
        "heads_removed_frac": 1 - heads_kept / (heads_total * n_layers),
        "ffn_removed_frac": 1 - ffn_kept / (ffn_total * n_layers),
        "params_removed_frac_layers": removed / layer_params,
        "params_removed_frac_all": removed / total,
        "params_removed": removed,
        "params_total": total,
    }


def compute_structure(ctx: Context, system: System) -> dict[str, Any] | None:
    info: dict[str, Any] = {"sources": []}
    components: dict[str, dict[str, Any]] = {}
    prune = _read_json(ctx.args.status_dir / f"{system.name}.prune.json")
    if isinstance(prune, dict):
        info["sources"].append("prune manifest")
        for key in ("unit_sparsity", "parameter_sparsity", "calibration_segments"):
            if prune.get(key) is not None:
                info[key] = prune[key]
        for name, entry in (prune.get("components") or {}).items():
            if isinstance(entry, dict) and entry.get("layers"):
                components[name] = {
                    "layers": entry["layers"],
                    "total_per_layer": entry.get("total_per_layer"),
                    "kept": entry.get("kept"),
                }
    dims = dict(LLAMA_7B_DIMS)
    if system.scope == "multi" or not components:
        sub = _load_subnetwork(ctx, system)
        if sub is not None:
            info["sources"].append("subnetwork")
            info["subnetwork"] = sub["path"]
            dims = sub["dims"]
            info["layers"] = sub["components"]
            for name, entry in sub["components"].items():
                counts = entry["kept_per_layer"]
                components[name] = {
                    "layers": len(counts),
                    "total_per_layer": entry["total_per_layer"],
                    "kept": sum(counts),
                }
            if "unit_sparsity" not in info:
                kept = sum(c["kept"] for c in components.values())
                total = sum(c["layers"] * c["total_per_layer"] for c in components.values())
                info["unit_sparsity"] = 1 - kept / total if total else None
    if not info["sources"]:
        return None
    info.update(parameter_fractions(components, dims))
    return info


def section_structure(ctx: Context) -> list[str]:
    lines = ["## Structure", ""]
    if ctx.args.no_structure:
        lines.append("Skipped (--no-structure).")
        return lines
    for system in ctx.grid:
        found = compute_structure(ctx, system)
        if found is not None:
            ctx.structure[system.name] = found
    if not ctx.structure:
        lines.append(
            "No prune manifests (grid-state/status/<name>.prune.json) or subnetwork descriptors "
            "found yet."
        )
        return lines
    lines.append(
        f"Removed parameters as a share of the whole model (embeddings included), from kept "
        f"head and FFN channel counts. Found for {len(ctx.structure)}/{len(ctx.grid)} models; "
        "pair and dir cells show the mean and range over their models. Per-model figures are "
        "in structure.csv."
    )
    lines.append("")
    rows = []
    for method, calib, sparsity in ctx.configs():
        cells = []
        for scope in SCOPES:
            values = [
                ctx.structure[s.name].get("params_removed_frac_all")
                for s in ctx.grid
                if s.cell[:4] == (method, calib, sparsity, scope) and s.name in ctx.structure
            ]
            values = [v for v in values if v is not None]
            if not values:
                cells.append("-")
            elif scope == "multi" or len(values) == 1:
                cells.append(pct(values[0]))
            else:
                mean = pct(sum(values) / len(values))
                cells.append(f"{mean} ({pct(min(values))} to {pct(max(values))}, n={len(values)})")
        rows.append([config_label(method, calib, sparsity), *cells])
    lines.extend(md_table(["Config", "multi", "pair", "dir"], rows))

    layer_rows = []
    for method, calib, sparsity in ctx.configs():
        system = ctx.index.get((method, calib, sparsity, "multi", ""))
        info = ctx.structure.get(system.name) if system else None
        layers = (info or {}).get("layers") or {}
        if not layers:
            continue
        cells = [config_label(method, calib, sparsity), pct(info.get("params_removed_frac_all"))]
        for name in ("attention_heads", "ffn_channels"):
            entry = layers.get(name)
            if not entry or not entry["kept_per_layer"]:
                cells.extend(["-", "-"])
                continue
            counts = entry["kept_per_layer"]
            total = entry["total_per_layer"] or 1
            first, last = counts[:4], counts[-4:]
            if name == "attention_heads":
                cells.append(f"{min(counts)} / {sum(counts) / len(counts):.1f} / {max(counts)}")
                cells.append(f"{sum(first) / len(first):.1f} / {sum(last) / len(last):.1f}")
            else:
                cells.append(
                    f"{pct(min(counts) / total, 0)} / {pct(sum(counts) / len(counts) / total, 0)}"
                    f" / {pct(max(counts) / total, 0)}"
                )
                cells.append(
                    f"{pct(sum(first) / len(first) / total, 0)} / "
                    f"{pct(sum(last) / len(last) / total, 0)}"
                )
        layer_rows.append(cells)
    if layer_rows:
        lines.extend(["", "### Per-layer kept units, multi models", ""])
        lines.append(
            "Heads are counts per layer (of 32); FFN is the kept share of 11008 channels. "
            "min / mean / max over layers, then mean of the first four / last four layers."
        )
        lines.append("")
        lines.extend(
            md_table(
                [
                    "System",
                    "Params removed",
                    "Heads min/mean/max",
                    "Heads first4/last4",
                    "FFN min/mean/max",
                    "FFN first4/last4",
                ],
                layer_rows,
            )
        )
    write_structure_files(ctx)
    ctx.results["structure"] = {
        name: {k: v for k, v in info.items() if k != "layers"}
        for name, info in ctx.structure.items()
    }
    return lines


def write_structure_files(ctx: Context) -> None:
    columns = [
        "system",
        "method",
        "calib",
        "sparsity",
        "scope",
        "scope_key",
        "sources",
        "unit_sparsity",
        "parameter_sparsity",
        "heads_removed_frac",
        "ffn_removed_frac",
        "params_removed_frac_layers",
        "params_removed_frac_all",
        "calibration_segments",
    ]
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(columns)
    for system in ctx.grid:
        info = ctx.structure.get(system.name)
        if info is None:
            continue
        writer.writerow(
            [
                system.name,
                system.method,
                system.calib,
                system.sparsity,
                system.scope,
                system.scope_key,
                "+".join(info["sources"]),
                *[_csv_value(info.get(key)) for key in columns[7:]],
            ]
        )
    _write_text(ctx.out / "structure.csv", buffer.getvalue())
    for system in ctx.grid:
        layers = (ctx.structure.get(system.name) or {}).get("layers")
        if system.scope != "multi" or not layers:
            continue
        heads = (layers.get("attention_heads") or {}).get("kept_per_layer") or []
        ffn = (layers.get("ffn_channels") or {}).get("kept_per_layer") or []
        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow(["layer", "heads_kept", "heads_total", "ffn_kept", "ffn_total"])
        for layer in range(max(len(heads), len(ffn))):
            writer.writerow(
                [
                    layer,
                    heads[layer] if layer < len(heads) else "",
                    (layers.get("attention_heads") or {}).get("total_per_layer", ""),
                    ffn[layer] if layer < len(ffn) else "",
                    (layers.get("ffn_channels") or {}).get("total_per_layer", ""),
                ]
            )
        _write_text(ctx.out / "structure" / f"{system.name}.layers.csv", buffer.getvalue())


# long.csv -------------------------------------------------------------------


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return value


def write_long_csv(ctx: Context) -> int:
    dense = ctx.dense
    ordered = [ctx.systems[DENSE], *ctx.grid]
    ordered += sorted((s for s in ctx.systems.values() if s.repaired), key=lambda s: s.name)
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(LONG_COLUMNS))
    writer.writeheader()
    count = 0
    for system in ordered:
        run = ctx.runs.get(system.name)
        if run is None:
            continue
        multi_system = None
        if not system.is_dense:
            multi_system = ctx.index.get(
                (system.method, system.calib, system.sparsity, "multi", "")
            )
        multi_run = ctx.runs.get(multi_system.name) if multi_system else None
        for direction in DIRECTIONS:
            record = run.directions.get(direction, {})
            row: dict[str, Any] = {
                "system": system.name,
                "run": run.slug,
                "method": system.method,
                "calib": system.calib,
                "sparsity": system.sparsity,
                "scope": system.scope,
                "scope_key": system.scope_key,
                "pruned_directions": "+".join(system.pruned_directions),
                "repaired": system.repaired,
                "source_system": system.source,
                "direction": direction,
                "into_english": direction.endswith("-en"),
                "in_scope": system.in_scope(direction),
                "n_segments": record.get("n_segments"),
            }
            for key in ("comet", "bleu", "chrf", "metricx"):
                value = run.value(direction, key)
                row[key] = value
                base = dense.value(direction, key) if dense is not None else None
                row[f"{key}_minus_dense"] = (
                    value - base if value is not None and base is not None else None
                )
            for key in ("comet", "metricx"):
                value = run.value(direction, key)
                multi = multi_run.value(direction, key) if multi_run is not None else None
                row[f"{key}_minus_multi"] = (
                    value - multi if value is not None and multi is not None else None
                )
            for key in (*BEHAVIOUR_FIELDS, "bleu_token_length_ratio"):
                row[key] = record.get(key)
            writer.writerow({key: _csv_value(value) for key, value in row.items()})
            count += 1
    _write_text(ctx.out / "long.csv", buffer.getvalue())
    return count


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------


def _method_colour(method: str, ctx: Context) -> str:
    if method in METHOD_COLOUR:
        return METHOD_COLOUR[method]
    extras = [m for m in ctx.methods if m not in METHOD_COLOUR]
    return EXTRA_COLOURS[extras.index(method) % len(EXTRA_COLOURS)]


def make_plots(ctx: Context) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        ctx.warn("matplotlib not available; plots skipped")
        return []
    folder = ctx.out / "plots"
    folder.mkdir(parents=True, exist_ok=True)
    style = {
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#e5e4df",
        "grid.linewidth": 0.6,
        "axes.edgecolor": "#8a8984",
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "font.size": 9,
        "axes.titlesize": 10,
    }
    written: list[str] = []
    with plt.rc_context(style):
        for plot in (plot_sparsity, plot_transfer, plot_ref_vs_gen, plot_structure):
            try:
                written.extend(plot(ctx, plt, folder))
            except Exception as exc:  # a broken plot must not lose the tables
                ctx.warn(f"plot {plot.__name__} failed: {exc}")
            finally:
                plt.close("all")
    return written


def _save(fig: Any, path: Path) -> str:
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    return str(path)


def plot_sparsity(ctx: Context, plt: Any, folder: Path) -> list[str]:
    fig, axes = plt.subplots(1, len(SCOPES), figsize=(12.5, 4.2), sharey=True)
    dense_comet, _ = macro(single(ctx, DENSE), "comet")
    any_data = False
    for ax, scope in zip(axes, SCOPES, strict=True):
        for method in ctx.methods:
            for calib in ctx.calibs:
                xs, ys = [], []
                for sparsity in ctx.sparsities:
                    value, n = macro(composite(ctx, method, calib, sparsity, scope), "comet")
                    if value is not None and n == N_DIRS:
                        xs.append(sparsity)
                        ys.append(value)
                if not xs:
                    continue
                any_data = True
                ax.plot(
                    xs,
                    ys,
                    color=_method_colour(method, ctx),
                    linestyle="-" if calib == "ref" else "--",
                    marker="o",
                    markersize=5,
                    linewidth=1.6,
                    label=f"{METHOD_LABEL.get(method, method)}, {calib} calibration",
                )
        if dense_comet is not None:
            ax.axhline(
                dense_comet, color=MUTED, linewidth=1.0, linestyle=":", label="dense ALMA-7B"
            )
        ax.set_title(SCOPE_TITLE[scope])
        ax.set_xticks(list(ctx.sparsities))
        ax.set_xlim(min(ctx.sparsities) - 5, max(ctx.sparsities) + 5)
        ax.set_xlabel("sparsity (%)")
    if not any_data:
        plt.close(fig)
        return []
    axes[0].set_ylabel(f"macro COMET-22 ({N_DIRS} directions)")
    handles: dict[str, Any] = {}
    for ax in axes:
        for handle, label in zip(*ax.get_legend_handles_labels(), strict=True):
            handles.setdefault(label, handle)
    fig.legend(
        list(handles.values()),
        list(handles.keys()),
        loc="lower center",
        ncol=min(len(handles), 5),
        frameon=False,
        bbox_to_anchor=(0.5, -0.06),
    )
    fig.suptitle("COMET vs sparsity (complete composites only)", y=1.0)
    path = _save(fig, folder / "comet_vs_sparsity.png")
    plt.close(fig)
    return [path]


def plot_transfer(ctx: Context, plt: Any, folder: Path) -> list[str]:
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("diverging", list(DIVERGING)).with_extremes(
        bad="#ffffff"
    )
    written = []
    pairs = [(m, c) for m in ctx.methods for c in ctx.calibs]
    for sparsity in ctx.sparsities:
        for scope, keys in (("dir", DIRECTIONS), ("pair", PAIR_LANGS)):
            matrices = {}
            for method, calib in pairs:
                data = ctx.transfer.get(f"{method}{sparsity}-{calib}")
                if not data or not data[scope]:
                    continue
                matrix = np.full((len(keys), N_DIRS), np.nan)
                by_key = {row["key"]: row for row in data[scope]}
                for i, key in enumerate(keys):
                    row = by_key.get(key)
                    if row is None:
                        continue
                    for j, direction in enumerate(DIRECTIONS):
                        value = row["cells"][direction]["comet_minus_multi"]
                        if value is not None:
                            matrix[i, j] = 100 * value
                if np.isfinite(matrix).any():
                    matrices[(method, calib)] = matrix
            if not matrices:
                continue
            limit = max(float(np.nanmax(np.abs(m))) for m in matrices.values()) or 1.0
            # Only the methods and calibrations that have data get a panel row/column.
            methods = [m for m in ctx.methods if any((m, c) in matrices for c in ctx.calibs)]
            calibs = [c for c in ctx.calibs if any((m, c) in matrices for m in methods)]
            n_rows, n_cols = len(methods), len(calibs)
            height = 5.2 if scope == "dir" else 3.0
            fig, axes = plt.subplots(
                n_rows,
                n_cols,
                figsize=(6.2 * n_cols, height * n_rows),
                squeeze=False,
                constrained_layout=True,
            )
            image = None
            for r, method in enumerate(methods):
                for c, calib in enumerate(calibs):
                    ax = axes[r][c]
                    ax.grid(False)
                    title = f"{METHOD_LABEL.get(method, method)}, {calib} calibration"
                    matrix = matrices.get((method, calib))
                    if matrix is None:
                        ax.set_axis_off()
                        ax.set_title(f"{title}: no data", color=MUTED)
                        continue
                    image = ax.imshow(matrix, cmap=cmap, vmin=-limit, vmax=limit, aspect="auto")
                    ax.set_title(title)
                    ax.set_xticks(range(N_DIRS), DIRECTIONS, rotation=45, ha="right")
                    labels = [f"{scope}-{k}" for k in keys]
                    ax.set_yticks(range(len(keys)), labels)
                    ax.set_xlabel("evaluated direction")
                    ax.set_ylabel("calibrated on")
                    for i, key in enumerate(keys):
                        for j, direction in enumerate(DIRECTIONS):
                            value = matrix[i, j]
                            if not np.isfinite(value):
                                continue
                            strong = abs(value) > 0.6 * limit
                            ax.text(
                                j,
                                i,
                                f"{value:+.1f}",
                                ha="center",
                                va="center",
                                fontsize=6.5,
                                color="white" if strong else INK,
                            )
                            own = direction == key if scope == "dir" else pair_of(direction) == key
                            if own:
                                ax.add_patch(
                                    plt.Rectangle(
                                        (j - 0.5, i - 0.5),
                                        1,
                                        1,
                                        fill=False,
                                        edgecolor=INK,
                                        linewidth=1.2,
                                    )
                                )
            if image is not None:
                bar = fig.colorbar(image, ax=axes, shrink=0.8)
                bar.set_label("COMET x100, specialist minus multi")
            fig.suptitle(
                f"{scope} transfer at {sparsity}% sparsity: specialist minus multi on each "
                "direction (boxed = own)"
            )
            written.append(_save(fig, folder / f"transfer_{scope}_s{sparsity}.png"))
            plt.close(fig)
    return written


def plot_ref_vs_gen(ctx: Context, plt: Any, folder: Path) -> list[str]:
    from matplotlib.lines import Line2D

    macro_points = []
    cell_points = []
    for method in ctx.methods:
        for sparsity in ctx.sparsities:
            for scope in SCOPES:
                ref = composite(ctx, method, "ref", sparsity, scope)
                gen = composite(ctx, method, "gen", sparsity, scope)
                shared = [d for d in DIRECTIONS if d in ref and d in gen]
                pairs = [(ref[d][1].value(d, "comet"), gen[d][1].value(d, "comet")) for d in shared]
                pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
                for x, y in pairs:
                    cell_points.append((method, scope, x, y))
                if len(pairs) == N_DIRS:
                    x = sum(p[0] for p in pairs) / N_DIRS
                    y = sum(p[1] for p in pairs) / N_DIRS
                    macro_points.append((method, scope, sparsity, x, y))
    if not cell_points:
        return []
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2))
    panels = (
        (
            axes[0],
            [(m, s, x, y) for m, s, _, x, y in macro_points],
            "macro over 10 directions",
            1.0,
        ),
        (axes[1], cell_points, "one point per direction", 0.55),
    )
    for ax, points, title, alpha in panels:
        if not points:
            ax.set_axis_off()
            ax.set_title(f"{title}: no complete pairs yet", color=MUTED)
            continue
        for method, scope, x, y in points:
            ax.scatter(
                x,
                y,
                color=_method_colour(method, ctx),
                marker=SCOPE_MARKER[scope],
                s=46 if alpha == 1.0 else 22,
                alpha=alpha,
                edgecolors="white",
                linewidths=0.6,
            )
        values = [p[2] for p in points] + [p[3] for p in points]
        low, high = min(values), max(values)
        pad = (high - low) * 0.05 or 0.01
        ax.plot(
            [low - pad, high + pad],
            [low - pad, high + pad],
            color=MUTED,
            linewidth=0.9,
            linestyle="--",
        )
        ax.set_xlim(low - pad, high + pad)
        ax.set_ylim(low - pad, high + pad)
        ax.set_aspect("equal")
        ax.set_xlabel("COMET-22, ref calibration (prompt + reference)")
        ax.set_ylabel("COMET-22, gen calibration (prompt + dense output)")
        ax.set_title(title)
    for _, _, sparsity, x, y in macro_points:
        axes[0].annotate(
            f"{sparsity}",
            (x, y),
            textcoords="offset points",
            xytext=(5, 3),
            fontsize=7,
            color=MUTED,
        )
    handles = [
        Line2D(
            [],
            [],
            color=_method_colour(m, ctx),
            marker="o",
            linestyle="",
            label=METHOD_LABEL.get(m, m),
        )
        for m in ctx.methods
    ] + [Line2D([], [], color=MUTED, marker=SCOPE_MARKER[s], linestyle="", label=s) for s in SCOPES]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(handles),
        frameon=False,
        bbox_to_anchor=(0.5, -0.04),
    )
    fig.suptitle(
        "Ref vs gen calibration text (above the dashed line: gen is better; labels = sparsity %)"
    )
    path = _save(fig, folder / "ref_vs_gen.png")
    plt.close(fig)
    return [path]


def plot_structure(ctx: Context, plt: Any, folder: Path) -> list[str]:
    series = []
    for method, calib, sparsity in ctx.configs():
        system = ctx.index.get((method, calib, sparsity, "multi", ""))
        layers = (ctx.structure.get(system.name) or {}).get("layers") if system else None
        if layers:
            series.append((method, calib, sparsity, layers))
    if not series:
        return []
    sparsities = [s for s in ctx.sparsities if any(item[2] == s for item in series)]
    fig, axes = plt.subplots(
        2, len(sparsities), figsize=(4.4 * len(sparsities), 6.0), squeeze=False, sharey="row"
    )
    for col, sparsity in enumerate(sparsities):
        for row, (name, title) in enumerate(
            (("attention_heads", "heads kept"), ("ffn_channels", "FFN channels kept"))
        ):
            ax = axes[row][col]
            for method, calib, s, layers in series:
                entry = layers.get(name)
                if s != sparsity or not entry or not entry["kept_per_layer"]:
                    continue
                total = entry["total_per_layer"] or 1
                counts = entry["kept_per_layer"]
                ax.plot(
                    range(len(counts)),
                    [100 * c / total for c in counts],
                    color=_method_colour(method, ctx),
                    linestyle="-" if calib == "ref" else "--",
                    linewidth=1.4,
                    label=f"{METHOD_LABEL.get(method, method)}, {calib}",
                )
            ax.set_title(f"{sparsity}%: {title}")
            ax.set_xlabel("layer")
            if col == 0:
                ax.set_ylabel("kept (%)")
    handles: dict[str, Any] = {}
    for ax in axes.flat:
        for handle, label in zip(*ax.get_legend_handles_labels(), strict=True):
            handles.setdefault(label, handle)
    fig.legend(
        list(handles.values()),
        list(handles.keys()),
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, -0.04),
    )
    fig.suptitle("Per-layer kept units, multi models")
    fig.tight_layout()
    path = _save(fig, folder / "structure_multi.png")
    plt.close(fig)
    return [path]


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=Path(DEFAULT_OUT))
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--manifest", type=Path, default=Path("configs/grid/manifest.json"))
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=1000,
        help="paired bootstrap resamples per contrast; 0 skips CIs and p-values",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--suite", default=None, help=f"suite name; default from the manifest, else {DEFAULT_SUITE}"
    )
    parser.add_argument("--status-dir", type=Path, default=Path("grid-state/status"))
    parser.add_argument(
        "--checkpoints-root", type=Path, default=Path("/scratch-shared/scur0560/checkpoints")
    )
    parser.add_argument("--cache-dir", type=Path, default=None, help="default: <out>/.cache")
    parser.add_argument("--no-cache", action="store_true", help="ignore and do not write the cache")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--no-structure", action="store_true")
    return parser.parse_args(argv)


def build(args: argparse.Namespace) -> Context:
    started = time.monotonic()
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    cache_root = None if args.no_cache else (args.cache_dir or out / ".cache")
    ctx = Context(args=args, out=out, suite=args.suite or DEFAULT_SUITE, cache=Cache(cache_root))

    load_systems(ctx)
    discover_runs(ctx)
    _log(
        f"{len(ctx.grid)} grid systems, {len(ctx.runs)} runs selected "
        f"({time.monotonic() - started:.1f}s)"
    )

    sections = [
        section_coverage,
        section_headline,
        section_headline_metricx,
        section_ref_vs_gen,
        section_method,
        section_per_direction,
        section_sparsity,
        section_transfer,
        section_behaviour,
        section_repair,
        section_structure,
    ]
    body: list[str] = []
    for section in sections:
        body.extend(section(ctx))
        body.append("")
    ctx.cache.flush()
    flush_bleu_cache(ctx)
    _log(f"tables built, {ctx.n_bootstraps} new bootstraps ({time.monotonic() - started:.1f}s)")

    n_rows = write_long_csv(ctx)
    plots = [] if args.no_plots else make_plots(ctx)

    generated = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    header = [
        "# Pruning grid report",
        "",
        f"Suite `{ctx.suite}`, generated {generated} from `{args.runs_root}` and "
        f"`{args.manifest}`. Paired bootstrap: "
        + (f"{args.bootstrap} resamples, seed {args.seed}." if args.bootstrap > 0 else "off.")
        + " Missing cells are `-`.",
        "",
    ]
    footer = ["## Warnings", ""]
    footer.extend([f"- {warning}" for warning in ctx.warnings] or ["None."])
    _write_text(out / "summary.md", "\n".join(header + body + footer) + "\n")

    dense = single(ctx, DENSE)
    ctx.results.update(
        {
            "generated_at": generated,
            "suite": ctx.suite,
            "runs_root": str(args.runs_root),
            "manifest": str(args.manifest),
            "bootstrap": {"n_samples": args.bootstrap, "seed": args.seed},
            "dense": {
                "system": DENSE,
                "run": ctx.dense.slug if ctx.dense else None,
                **{key: macro(dense, key)[0] for key in ("comet", "bleu", "chrf", "metricx")},
            },
            "runs": {name: run.slug for name, run in sorted(ctx.runs.items())},
            "plots": plots,
            "warnings": ctx.warnings,
        }
    )
    _write_text(out / "results.json", json.dumps(ctx.results, indent=2, sort_keys=True) + "\n")
    _log(
        f"wrote {out}/summary.md, long.csv ({n_rows} rows), results.json, {len(plots)} plots; "
        f"{len(ctx.warnings)} warnings ({time.monotonic() - started:.1f}s)"
    )
    return ctx


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    build(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
