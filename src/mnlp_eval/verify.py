"""Test-set provenance check.

The framework loads test sets from the Hub while ALMA's published numbers were
produced from JSON files committed to their repository. If the two ever differ,
every comparison with their tables is quietly wrong. This makes that a checkable
claim rather than an assumption: it compares the Hub dataset against ALMA's own
files, segment by segment, and reports the first difference it finds.

ALMA's layout: ``human_written_data/<nonEnglish><en>/test.<src>-<tgt>.json``,
a single JSON array of ``{"translation": {"<src>": ..., "<tgt>": ...}}`` objects.
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from mnlp_eval.config import DataSpec
from mnlp_eval.data import load_testset
from mnlp_eval.languages import parse_direction

__all__ = ["verify_testset"]

ALMA_RAW_BASE = "https://raw.githubusercontent.com/fe1ixxu/ALMA/master/human_written_data"


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _alma_relative_path(direction_name: str) -> str:
    direction = parse_direction(direction_name)
    non_english = direction.source if direction.source != "en" else direction.target
    return f"{non_english}en/test.{direction}.json"


def _load_reference(direction_name: str, reference_root: str | None) -> list[dict[str, str]]:
    relative = _alma_relative_path(direction_name)
    if reference_root:
        path = Path(reference_root).expanduser() / relative
        if not path.is_file():
            # Accept either a repository root or the data directory itself.
            path = Path(reference_root).expanduser() / "human_written_data" / relative
        if not path.is_file():
            msg = f"reference file not found: {path}"
            raise FileNotFoundError(msg)
        payload = json.loads(path.read_text(encoding="utf-8"))
    else:
        url = f"{ALMA_RAW_BASE}/{relative}"
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            msg = (
                f"cannot download {url}: {exc}. On a machine without network access, "
                "clone fe1ixxu/ALMA and pass --reference-root."
            )
            raise FileNotFoundError(msg) from exc
    return [row["translation"] for row in payload]


def verify_testset(
    *,
    dataset: str = "haoranxu/WMT22-Test",
    directions: list[str],
    reference_root: str | None = None,
) -> bool:
    """Compare a Hub test set against ALMA's committed files.

    Returns true only if every direction matches exactly.
    """
    _log(f"verifying {dataset} against ALMA's human_written_data")
    all_match = True

    for direction_name in directions:
        direction = parse_direction(direction_name)
        try:
            reference = _load_reference(direction_name, reference_root)
        except (FileNotFoundError, KeyError, json.JSONDecodeError) as exc:
            _log(f"  {direction_name}: cannot load reference, {exc}")
            all_match = False
            continue

        hub = load_testset(DataSpec(directions=[direction_name], dataset=dataset), direction)
        mismatch = _first_mismatch(hub, reference, direction)
        if mismatch is None:
            _log(f"  {direction_name}: match, {len(hub)} segments")
        else:
            all_match = False
            _log(f"  {direction_name}: MISMATCH, {mismatch}")

    _log("all directions match" if all_match else "at least one direction does not match")
    return all_match


def _first_mismatch(hub: Any, reference: list[dict[str, str]], direction: Any) -> str | None:
    if len(hub) != len(reference):
        return f"{len(hub)} segments on the Hub but {len(reference)} in ALMA's file"
    for index, row in enumerate(reference):
        expected_source = row.get(direction.source, "")
        expected_reference = row.get(direction.target, "")
        if hub.sources[index] != expected_source:
            return (
                f"source differs at segment {index}: "
                f"Hub {hub.sources[index]!r} against ALMA {expected_source!r}"
            )
        if hub.references[index] != expected_reference:
            return (
                f"reference differs at segment {index}: "
                f"Hub {hub.references[index]!r} against ALMA {expected_reference!r}"
            )
    return None
