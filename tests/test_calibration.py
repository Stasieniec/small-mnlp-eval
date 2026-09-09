"""Calibration and repair data.

Every subnetwork in the sweep is selected on data this module produces, so a
silent change here changes what the pruning criterion optimises for every
system at once. The properties that matter are that the draw is balanced,
deterministic, reproducible from the config alone, and disjoint from the test
sets.

The Hub is stubbed. These tests must run in CI with no network.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import pytest

from mnlp_eval.artifacts import read_jsonl_dicts
from mnlp_eval.config import ConfigError
from mnlp_eval.data.calibration import CalibrationSpec, build_calibration_set


class FakeDataset:
    def __init__(self, rows: list[dict[str, str]], column: str = "translation") -> None:
        self.column_names = [column]
        self._rows = rows
        self._column = column

    def __getitem__(self, key: str) -> list[dict[str, str]]:
        assert key == self._column
        return self._rows


@pytest.fixture
def stub_hub(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Stand in for ``datasets.load_dataset`` with a deterministic corpus."""
    calls: dict[str, Any] = {"configs": []}

    def load_dataset(dataset: str, config: str, split: str) -> FakeDataset:
        calls["configs"].append((dataset, config, split))
        language = config.split("-")[0]
        # Lengths vary so the source-character filter is observable.
        rows = [
            {
                "en": f"English sentence {index} " + "long " * (index % 20),
                language: f"{language} sentence {index} " + "lang " * (index % 20),
            }
            for index in range(200)
        ]
        return FakeDataset(rows)

    module = types.ModuleType("datasets")
    module.load_dataset = load_dataset  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "datasets", module)
    return calls


@pytest.fixture
def no_contamination_check(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "mnlp_eval.data.load_testset",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("no network in tests")),
    )


def build(tmp_path: Path, **overrides: Any) -> dict[str, Any]:
    payload = {"name": "test-set", "directions": ["de-en", "en-de"], "segments_per_direction": 5}
    payload.update(overrides)
    return build_calibration_set(
        CalibrationSpec.from_dict(payload), tmp_path, contamination_check=None
    )


def test_every_direction_contributes_the_same_number_of_segments(
    tmp_path: Path, stub_hub: dict[str, Any]
) -> None:
    """Balanced by construction.

    Proportional sampling would leave Icelandic, which has a seventh of the
    parallel data the other languages have, nearly absent from the
    multi-directional calibration set.
    """
    manifest = build(tmp_path, directions=["de-en", "en-de", "is-en"], segments_per_direction=7)

    counts = {
        direction: len(list(read_jsonl_dicts(tmp_path / "test-set" / f"{direction}.jsonl")))
        for direction in manifest["directions"]
    }
    assert counts == {"de-en": 7, "en-de": 7, "is-en": 7}
    assert manifest["total_segments"] == 21


def test_the_draw_is_reproducible_from_the_config(tmp_path: Path, stub_hub: dict[str, Any]) -> None:
    first = build(tmp_path / "a")
    second = build(tmp_path / "b")

    assert first["fingerprint"] == second["fingerprint"]
    assert list(read_jsonl_dicts(tmp_path / "a" / "test-set" / "de-en.jsonl")) == list(
        read_jsonl_dicts(tmp_path / "b" / "test-set" / "de-en.jsonl")
    )


def test_a_different_seed_draws_different_segments(
    tmp_path: Path, stub_hub: dict[str, Any]
) -> None:
    first = build(tmp_path / "a", seed=1)
    second = build(tmp_path / "b", seed=2)

    assert first["fingerprint"] != second["fingerprint"]


def test_adding_a_direction_leaves_the_others_untouched(
    tmp_path: Path, stub_hub: dict[str, Any]
) -> None:
    """Seeded per direction on purpose.

    A single stream would mean extending the suite silently re-drew every
    other direction, and every subnetwork calibrated before the change would
    stop being comparable with the ones after it.
    """
    build(tmp_path / "a", directions=["de-en", "en-de"])
    build(tmp_path / "b", directions=["de-en", "en-de", "is-en"])

    assert list(read_jsonl_dicts(tmp_path / "a" / "test-set" / "de-en.jsonl")) == list(
        read_jsonl_dicts(tmp_path / "b" / "test-set" / "de-en.jsonl")
    )


def test_records_carry_the_prompt_the_model_is_evaluated_under(
    tmp_path: Path, stub_hub: dict[str, Any]
) -> None:
    """Calibrating on bare sentences calibrates on a distribution ALMA never sees."""
    build(tmp_path)

    record = next(iter(read_jsonl_dicts(tmp_path / "test-set" / "de-en.jsonl")))
    assert record["prompt"] == (
        f"Translate this from German to English:\nGerman: {record['source']}\nEnglish:"
    )


def test_source_length_filtering_is_applied(tmp_path: Path, stub_hub: dict[str, Any]) -> None:
    """Very short segments are titles; very long ones dominate a calibration batch."""
    manifest = build(tmp_path, min_source_chars=25, max_source_chars=45)

    assert manifest["directions"]["de-en"]["n_available"] == 200
    assert manifest["directions"]["de-en"]["n_eligible"] < 200
    assert all(
        25 <= len(record["source"]) <= 45
        for record in read_jsonl_dicts(tmp_path / "test-set" / "de-en.jsonl")
    )


def test_asking_for_more_than_the_data_holds_is_an_error(
    tmp_path: Path, stub_hub: dict[str, Any]
) -> None:
    with pytest.raises(ValueError, match="are requested"):
        build(tmp_path, segments_per_direction=5000)


def test_both_directions_of_a_pair_read_one_config(
    tmp_path: Path, stub_hub: dict[str, Any]
) -> None:
    """ALMA's parallel data is keyed by English-centric pair, unlike its test sets."""
    build(tmp_path, directions=["de-en", "en-de"])

    assert {config for _, config, _ in stub_hub["configs"]} == {"de-en"}


def test_contamination_against_the_test_set_is_counted(
    tmp_path: Path, stub_hub: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The check that makes a calibration set defensible.

    A collision means the pruning criterion saw the evaluation data, and every
    quality number downstream of the subnetwork is void.
    """
    spec = CalibrationSpec.from_dict(
        {"name": "leaky", "directions": ["de-en"], "segments_per_direction": 3}
    )
    build_calibration_set(spec, tmp_path, contamination_check=None)
    drawn = [record["source"] for record in read_jsonl_dicts(tmp_path / "leaky" / "de-en.jsonl")]

    class FakeTestSet:
        sources = tuple(drawn[:2])

        def __len__(self) -> int:
            return len(self.sources)

    monkeypatch.setattr("mnlp_eval.data.load_testset", lambda *_a, **_k: FakeTestSet())
    manifest = build_calibration_set(spec, tmp_path, contamination_check="haoranxu/WMT22-Test")

    assert manifest["contamination"]["total_collisions"] == 2


def test_an_unreachable_test_set_is_reported_not_swallowed(
    tmp_path: Path, stub_hub: dict[str, Any], no_contamination_check: None
) -> None:
    spec = CalibrationSpec.from_dict(
        {"name": "offline", "directions": ["de-en"], "segments_per_direction": 3}
    )

    manifest = build_calibration_set(spec, tmp_path, contamination_check="haoranxu/WMT22-Test")

    entry = manifest["contamination"]["directions"]["de-en"]
    assert entry["checked"] is False
    assert "no network" in entry["reason"]


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"directions": ["de-en"]}, "'name' is required"),
        ({"name": "x"}, "'directions' is required"),
        ({"name": "x", "directions": ["de-en"], "segments_per_direction": 0}, "at least 1"),
        ({"name": "x", "directions": ["de-en"], "prompt": "nope"}, "prompt"),
        ({"name": "x", "directions": ["de-fr"]}, "not English-centric"),
        (
            {"name": "x", "directions": ["de-en"], "min_source_chars": 100, "max_source_chars": 10},
            "min_source_chars",
        ),
    ],
)
def test_malformed_specs_are_rejected(payload: dict[str, Any], message: str) -> None:
    with pytest.raises((ConfigError, ValueError), match=message):
        CalibrationSpec.from_dict(payload)
