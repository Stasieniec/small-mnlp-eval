"""On-disk artifact formats and crash-safe writes.

Stages communicate through files rather than function calls, because the three
metric families in use have mutually incompatible dependency pins and cannot
share a Python process. See docs/environments.md.

Every write goes through a temporary file and an atomic rename. Slurm jobs get
killed at wall-clock limits, and a half-written scores file that looks complete
is worse than no file at all.
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
    "write_jsonl",
    "write_lines",
]


@dataclass
class Segment:
    """One evaluated sentence, as stored in ``hyps/<direction>.jsonl``.

    ``raw_output`` is retained deliberately. When a compressed model collapses,
    the recovered hypothesis alone does not explain what happened, and
    regenerating a 10,000 sentence run to find out is expensive.
    """

    index: int
    source: str
    reference: str
    hypothesis: str
    raw_output: str = ""
    parse_status: str = "ok"
    parse_flags: list[str] = field(default_factory=list)
    n_source_tokens: int = 0
    n_generated_tokens: int = 0
    #: Tokens generated after the hypothesis ended, which were discarded. A
    #: model that will not stop talking costs real inference time without
    #: affecting quality, so the two are reported separately.
    n_wasted_tokens: int = 0
    #: Generation ran out of budget without emitting EOS.
    hit_token_budget: bool = False
    #: The hypothesis itself was cut off, which is a quality problem. Distinct
    #: from hit_token_budget: a model that finished the translation and then
    #: kept generating hits the budget without truncating the translation.
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

    The temporary name comes from :func:`tempfile.mkstemp`, not from the
    process id. A PID-derived name is only unique within one host, and Slurm
    array tasks on different nodes collide: two writers shared a temp file, one
    truncated the other's content and the survivor renamed an interleaving of
    both over the target, while the loser died on a missing file. Measured at 1
    corrupted manifest in 8 concurrent trials.

    ``fsync`` flushes to disk before the rename so a node failure cannot leave
    a durable rename pointing at unwritten content.
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
                # A record missing required fields would otherwise surface as a
                # bare TypeError from dataclass construction, with nothing to
                # say which file or line produced it.
                msg = f"{path}:{line_number}: incomplete segment record ({exc})"
                raise ValueError(msg) from exc


def write_lines(path: Path, values: Iterable[str]) -> int:
    """Write one value per line, matching ALMA's plain-text output format.

    Newlines inside a hypothesis would corrupt the line alignment that
    sacreBLEU and ``comet-score`` rely on, so they are replaced with spaces.
    The unmodified text stays available in the JSON Lines file.
    """
    cleaned = [value.replace("\n", " ").replace("\r", " ") for value in values]
    atomic_write_text(path, "\n".join(cleaned) + ("\n" if cleaned else ""))
    return len(cleaned)
