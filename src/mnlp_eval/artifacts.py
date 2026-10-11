"""On-disk artifact formats and atomic writes.

Stages exchange files because the metric environments have incompatible
dependencies. Every write goes through a temporary file and a rename.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "Segment",
    "atomic_write_json",
    "atomic_write_text",
    "read_json",
    "read_jsonl",
    "read_jsonl_dicts",
    "write_jsonl",
    "write_lines",
]


@dataclass
class Segment:
    """One evaluated sentence, as stored in ``hyps/<direction>.jsonl``."""

    index: int
    source: str
    reference: str
    hypothesis: str
    raw_output: str = ""
    parse_status: str = "ok"
    parse_flags: list[str] = field(default_factory=list)
    n_source_tokens: int = 0
    n_generated_tokens: int = 0
    #: Tokens generated after the hypothesis ended, which were discarded.
    n_wasted_tokens: int = 0
    #: Generation ran out of budget without emitting EOS.
    hit_token_budget: bool = False
    #: The hypothesis itself was cut off by the token budget.
    truncated: bool = False
    source_truncated: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Segment:
        known = set(cls.__dataclass_fields__)
        return cls(**{key: value for key, value in payload.items() if key in known})


def atomic_write_text(path: Path, text: str, *, fsync: bool = True) -> None:
    """Write ``text`` to ``path`` atomically, creating parent directories.

    Uses a :func:`tempfile.mkstemp` name so concurrent jobs on different nodes
    do not share a temp file. ``fsync`` flushes before the rename.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(text)
            if fsync:
                stream.flush()
                os.fsync(stream.fileno())
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def atomic_write_json(path: Path, payload: Any) -> None:
    """Write ``payload`` as indented, key-sorted JSON, atomically."""
    text = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False)
    atomic_write_text(path, text + "\n")


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        loaded: dict[str, Any] = json.load(handle)
    return loaded


def write_jsonl(path: Path, records: Iterable[Segment | dict[str, Any]]) -> int:
    """Write segments as JSON Lines, atomically. Returns the record count."""
    lines: list[str] = []
    for record in records:
        payload = record.to_dict() if isinstance(record, Segment) else record
        lines.append(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    atomic_write_text(path, "\n".join(lines) + ("\n" if lines else ""))
    return len(lines)


def read_jsonl_dicts(path: Path) -> Iterator[dict[str, Any]]:
    """Stream raw dict records from a JSON Lines file (see :func:`read_jsonl` for segments)."""
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                msg = f"{path}:{line_number}: malformed JSON Lines record"
                raise ValueError(msg) from exc
            if not isinstance(payload, dict):
                msg = f"{path}:{line_number}: expected an object, got {type(payload).__name__}"
                raise ValueError(msg)
            yield payload


def read_jsonl(path: Path) -> Iterator[Segment]:
    """Stream segments from a JSON Lines file."""
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                msg = f"{path}:{line_number}: malformed JSON Lines record"
                raise ValueError(msg) from exc
            try:
                yield Segment.from_dict(payload)
            except TypeError as exc:
                msg = f"{path}:{line_number}: incomplete segment record ({exc})"
                raise ValueError(msg) from exc


def write_lines(path: Path, values: Iterable[str]) -> int:
    """Write one value per line, as in ALMA's plain-text output. Newlines become spaces."""
    cleaned = [value.replace("\n", " ").replace("\r", " ") for value in values]
    atomic_write_text(path, "\n".join(cleaned) + ("\n" if cleaned else ""))
    return len(cleaned)
