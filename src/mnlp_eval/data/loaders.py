"""Test set loading, with a fingerprint that proves two runs saw the same data.

The three ALMA test sets on the Hub share one schema: a single struct column
named after the direction, holding the two language fields. Verified against
the datasets API for ``haoranxu/WMT22-Test``, ``WMT23-Test`` and
``FLORES-200``, so one loader serves all three.

Local sources are supported too, because a distilled student is often evaluated
on a held-out slice of the ALMA training data that never reaches the Hub.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mnlp_eval.config import DataSpec
from mnlp_eval.languages import Direction

__all__ = ["TestSet", "available_datasets", "load_testset"]

#: Hub datasets known to use ALMA's one-struct-column layout.
ALMA_STYLE_DATASETS = (
    "haoranxu/WMT22-Test",
    "haoranxu/WMT23-Test",
    "haoranxu/FLORES-200",
)

_LOCAL_JSONL_PREFIX = "local:jsonl:"
_LOCAL_TEXT_PREFIX = "local:text:"


@dataclass(frozen=True)
class TestSet:
    """Aligned sources and references for one direction, with provenance."""

    direction: Direction
    sources: tuple[str, ...]
    references: tuple[str, ...]
    dataset: str
    split: str
    n_available: int

    def __post_init__(self) -> None:
        if len(self.sources) != len(self.references):
            msg = (
                f"{self.dataset} {self.direction}: {len(self.sources)} sources but "
                f"{len(self.references)} references"
            )
            raise ValueError(msg)
        if not self.sources:
            msg = f"{self.dataset} {self.direction}: no segments loaded"
            raise ValueError(msg)

    def __len__(self) -> int:
        return len(self.sources)

    @property
    def fingerprint(self) -> str:
        """Digest of the exact segments used.

        Two runs with the same fingerprint provably scored the same data, which
        matters once a suite carries a ``limit`` or a local file gets edited.
        """
        digest = hashlib.sha256()
        digest.update(f"{self.dataset}\x00{self.split}\x00{self.direction}".encode())
        for source, reference in zip(self.sources, self.references, strict=True):
            digest.update(b"\x00")
            digest.update(source.encode())
            digest.update(b"\x01")
            digest.update(reference.encode())
        return digest.hexdigest()[:16]

    def provenance(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "split": self.split,
            "direction": str(self.direction),
            "n_used": len(self),
            "n_available": self.n_available,
            "fingerprint": self.fingerprint,
        }


def available_datasets() -> list[str]:
    return [
        *ALMA_STYLE_DATASETS,
        f"{_LOCAL_JSONL_PREFIX}<path>",
        f"{_LOCAL_TEXT_PREFIX}<directory>",
    ]


def load_testset(spec: DataSpec, direction: Direction) -> TestSet:
    """Load one direction of a test set as described by ``spec``."""
    if spec.dataset.startswith(_LOCAL_JSONL_PREFIX):
        pairs, n_available = _load_local_jsonl(
            Path(spec.dataset[len(_LOCAL_JSONL_PREFIX) :]), direction
        )
    elif spec.dataset.startswith(_LOCAL_TEXT_PREFIX):
        pairs, n_available = _load_local_text(
            Path(spec.dataset[len(_LOCAL_TEXT_PREFIX) :]), direction, spec.split
        )
    else:
        pairs, n_available = _load_hub(spec.dataset, direction, spec.split)

    if spec.limit is not None:
        pairs = pairs[: spec.limit]

    sources = tuple(source for source, _ in pairs)
    references = tuple(reference for _, reference in pairs)
    return TestSet(
        direction=direction,
        sources=sources,
        references=references,
        dataset=spec.dataset,
        split=spec.split,
        n_available=n_available,
    )


def _load_hub(dataset: str, direction: Direction, split: str) -> tuple[list[tuple[str, str]], int]:
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover
        msg = (
            "loading a Hub dataset needs the 'datasets' package. "
            "Install the generation extra: pip install -e '.[gen]'"
        )
        raise ImportError(msg) from exc

    config = str(direction)
    loaded = load_dataset(dataset, config, split=split)
    if config not in loaded.column_names:
        msg = (
            f"{dataset} config {config!r} has columns {loaded.column_names}, "
            f"expected a single struct column named {config!r}. "
            "This loader targets ALMA's test set layout."
        )
        raise ValueError(msg)

    pairs: list[tuple[str, str]] = []
    for row in loaded[config]:
        source = row.get(direction.source)
        reference = row.get(direction.target)
        if source is None or reference is None:
            msg = (
                f"{dataset} {config}: a row is missing the {direction.source!r} or "
                f"{direction.target!r} field"
            )
            raise ValueError(msg)
        pairs.append((str(source), str(reference)))
    return pairs, len(pairs)


def _load_local_jsonl(path: Path, direction: Direction) -> tuple[list[tuple[str, str]], int]:
    """Read ALMA's local JSON Lines layout, or a flat source/target layout.

    Accepts either ``{"translation": {"de": ..., "en": ...}}``, which is what
    ALMA's ``human_written_data`` uses, or ``{"de": ..., "en": ...}``.
    """
    if not path.is_file():
        msg = f"local test file not found: {path}"
        raise FileNotFoundError(msg)
    pairs: list[tuple[str, str]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                msg = f"{path}:{line_number}: malformed JSON"
                raise ValueError(msg) from exc
            payload = row.get("translation", row)
            source = payload.get(direction.source)
            reference = payload.get(direction.target)
            if source is None or reference is None:
                msg = (
                    f"{path}:{line_number}: expected keys {direction.source!r} and "
                    f"{direction.target!r}, found {sorted(payload)}"
                )
                raise ValueError(msg)
            pairs.append((str(source), str(reference)))
    return pairs, len(pairs)


def _load_local_text(
    directory: Path, direction: Direction, split: str
) -> tuple[list[tuple[str, str]], int]:
    """Read ALMA's plain-text layout, ``<split>.<src>-<tgt>.<lang>``."""
    stem = f"{split}.{direction}"
    source_path = directory / f"{stem}.{direction.source}"
    reference_path = directory / f"{stem}.{direction.target}"
    for path in (source_path, reference_path):
        if not path.is_file():
            msg = f"local test file not found: {path}"
            raise FileNotFoundError(msg)
    sources = source_path.read_text(encoding="utf-8").splitlines()
    references = reference_path.read_text(encoding="utf-8").splitlines()
    if len(sources) != len(references):
        msg = (
            f"{directory}: {source_path.name} has {len(sources)} lines but "
            f"{reference_path.name} has {len(references)}"
        )
        raise ValueError(msg)
    return list(zip(sources, references, strict=True)), len(sources)
