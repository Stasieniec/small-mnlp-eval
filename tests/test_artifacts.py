"""Artifact formats and crash-safe writes."""

from __future__ import annotations

from pathlib import Path

import pytest

from mnlp_eval.artifacts import (
    Segment,
    atomic_write_json,
    atomic_write_text,
    read_json,
    read_jsonl,
    write_jsonl,
    write_lines,
)


def test_segment_round_trips_through_jsonl(tmp_path: Path) -> None:
    segments = [
        Segment(
            index=index,
            source=f"source {index}",
            reference=f"reference {index}",
            hypothesis=f"hypothesis {index}",
            raw_output=f" hypothesis {index}\ntrailing",
            parse_flags=["extra_lines"],
            n_generated_tokens=5,
            n_wasted_tokens=2,
            hit_token_budget=True,
        )
        for index in range(3)
    ]
    path = tmp_path / "de-en.jsonl"
    assert write_jsonl(path, segments) == 3
    restored = list(read_jsonl(path))
    assert restored == segments


def test_unknown_jsonl_fields_are_ignored(tmp_path: Path) -> None:
    # Lets an older run directory still be read after the schema grows.
    path = tmp_path / "x.jsonl"
    path.write_text(
        '{"index": 0, "source": "a", "reference": "b", "hypothesis": "c", "future": 1}\n',
        encoding="utf-8",
    )
    assert next(iter(read_jsonl(path))).hypothesis == "c"


def test_malformed_jsonl_names_the_line(tmp_path: Path) -> None:
    path = tmp_path / "x.jsonl"
    path.write_text(
        '{"index": 0, "source": "a", "reference": "b", "hypothesis": "c"}\nnot json\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"x\.jsonl:2: malformed"):
        list(read_jsonl(path))


def test_incomplete_jsonl_record_names_the_line(tmp_path: Path) -> None:
    path = tmp_path / "x.jsonl"
    path.write_text('{"index": 0}\n', encoding="utf-8")
    with pytest.raises(ValueError, match=r"x\.jsonl:1: incomplete"):
        list(read_jsonl(path))


def test_blank_jsonl_lines_are_skipped(tmp_path: Path) -> None:
    path = tmp_path / "x.jsonl"
    path.write_text(
        '{"index": 0, "source": "a", "reference": "b", "hypothesis": "c"}\n\n\n', encoding="utf-8"
    )
    assert len(list(read_jsonl(path))) == 1


def test_empty_jsonl_writes_no_trailing_newline(tmp_path: Path) -> None:
    path = tmp_path / "empty.jsonl"
    assert write_jsonl(path, []) == 0
    assert path.read_text(encoding="utf-8") == ""


def test_write_lines_flattens_newlines(tmp_path: Path) -> None:
    # A newline inside a hypothesis would break the line alignment that
    # sacreBLEU and comet-score depend on.
    path = tmp_path / "de-en.txt"
    assert write_lines(path, ["one line", "two\nlines", "three\r\nlines"]) == 3
    assert path.read_text(encoding="utf-8").splitlines() == [
        "one line",
        "two lines",
        "three  lines",
    ]


def test_atomic_writes_create_parent_directories(tmp_path: Path) -> None:
    target = tmp_path / "deep" / "nested" / "file.json"
    atomic_write_json(target, {"a": 1})
    assert read_json(target) == {"a": 1}


def test_atomic_writes_leave_no_temporary_files(tmp_path: Path) -> None:
    atomic_write_text(tmp_path / "a.txt", "hello")
    assert [path.name for path in tmp_path.iterdir()] == ["a.txt"]


def test_json_is_written_sorted_for_stable_diffs(tmp_path: Path) -> None:
    path = tmp_path / "a.json"
    atomic_write_json(path, {"b": 1, "a": 2})
    assert path.read_text(encoding="utf-8").index('"a"') < path.read_text(encoding="utf-8").index(
        '"b"'
    )


def test_non_ascii_survives_a_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "x.jsonl"
    write_jsonl(
        path,
        [
            Segment(
                index=0, source="Petta er profun", reference="Ich mag Kaffee", hypothesis="Это тест"
            )
        ],
    )
    assert next(iter(read_jsonl(path))).hypothesis == "Это тест"
