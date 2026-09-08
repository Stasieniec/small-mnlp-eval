"""Shared fixtures.

The core test suite runs without torch, without a GPU and without network
access, so it is cheap enough to run on every commit. Tests that need any of
those are marked ``gpu``, ``network`` or ``slow`` and are excluded from CI.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mnlp_eval.config import ModelSpec, RunConfig, SuiteSpec


@pytest.fixture
def local_testset(tmp_path: Path) -> Path:
    """A four-segment de-en test file in ALMA's local JSON Lines layout."""
    rows = [
        {"translation": {"de": "Das ist ein Test.", "en": "This is a test."}},
        {"translation": {"de": "Der Hund schlaeft.", "en": "The dog is sleeping."}},
        {"translation": {"de": "Ich mag Kaffee sehr.", "en": "I like coffee a lot."}},
        {"translation": {"de": "Wir fahren morgen los.", "en": "We leave tomorrow."}},
    ]
    path = tmp_path / "test.de-en.jsonl"
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8"
    )
    return path


@pytest.fixture
def model_spec() -> ModelSpec:
    return ModelSpec.from_dict(
        {
            "name": "stub-model",
            "loader": "hf_causal",
            "prompt": "alma",
            "model_name_or_path": "stub/model",
            "dtype": "float32",
        }
    )


@pytest.fixture
def suite_spec(local_testset: Path) -> SuiteSpec:
    return SuiteSpec.from_dict(
        {
            "name": "stub-suite",
            "data": {"dataset": f"local:jsonl:{local_testset}", "directions": ["de-en"]},
            "decode": {"num_beams": 1, "max_new_tokens": 16, "batch_size": 2},
        }
    )


@pytest.fixture
def run_config(model_spec: ModelSpec, suite_spec: SuiteSpec, tmp_path: Path) -> RunConfig:
    return RunConfig(model=model_spec, suite=suite_spec, output_root=tmp_path / "runs")
