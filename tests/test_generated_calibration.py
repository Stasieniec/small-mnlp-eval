"""Cache identity, interruption recovery, and exact generated-token replay."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from mnlp_eval.artifacts import read_jsonl_dicts, write_jsonl
from mnlp_eval.config import DecodeSpec, ModelSpec, SuiteSpec, load_yaml_config
from mnlp_eval.data import generated_calibration as gc
from mnlp_eval.languages import parse_direction
from mnlp_eval.models import SegmentOutput
from mnlp_eval.prompts import get_prompt
from mnlp_eval.prune.collect import load_calibration_prompts, tokenized_batches
from mnlp_eval.prune.spec import PruneSpec


@pytest.fixture
def setup_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    source, out = tmp_path / "source", tmp_path / "generated"
    source.mkdir()
    (source / "calibration.json").write_text(json.dumps({"contamination": {"total_collisions": 0}}))
    for name in ("de-en", "en-de", "cs-en"):
        direction = parse_direction(name)
        rows = [
            {
                "id": index,
                "direction": name,
                "source": text,
                "target": f"ref {index}",
                "prompt": get_prompt("alma").render(direction, text),
            }
            for index, text in enumerate(("long sentence", "short"))
        ]
        write_jsonl(source / f"{name}.jsonl", rows)
    model = ModelSpec.from_dict(load_yaml_config("configs/models/alma-7b.yaml"))
    suite = SuiteSpec.from_dict(load_yaml_config("configs/suites/alma10-greedy.yaml"))
    calls: list[str] = []
    loads: list[ModelSpec] = []
    closed: list[bool] = []
    fail: list[str] = []

    class FakeTranslator:
        def translate(
            self,
            direction: Any,
            sources: list[str],
            decode: DecodeSpec,
            *,
            capture_token_ids: bool = False,
        ) -> list[SegmentOutput]:
            assert capture_token_ids
            assert decode.num_beams == 1 and not decode.do_sample
            calls.append(str(direction))
            if str(direction) in fail:
                raise RuntimeError("interrupted")
            return [
                SegmentOutput(
                    raw_text="" if index else "translation\nextra text",
                    input_token_ids=[1, index + 10],
                    generated_token_ids=[index + 20, 2],
                    n_source_tokens=2,
                    n_generated_tokens=1,
                    source_truncated=bool(index),
                    hit_token_budget=not bool(index),
                )
                for index, _ in enumerate(sources)
            ]

        def close(self) -> None:
            closed.append(True)

    def build(spec: ModelSpec) -> FakeTranslator:
        loads.append(spec)
        return FakeTranslator()

    monkeypatch.setattr(
        gc,
        "_resolve_model",
        lambda spec: (
            replace(spec, revision="dense-sha"),
            {"repository": spec.model_name_or_path, "revision": "dense-sha"},
        ),
    )
    monkeypatch.setattr(gc, "build_translator", build)
    monkeypatch.setattr(gc, "seed_everything", lambda _: None)
    return source, out, model, suite, calls, loads, closed, fail


def test_generate_and_reuse(setup_cache: Any) -> None:
    source, out, model, suite, calls, loads, closed, _ = setup_cache
    manifest = gc.generate_calibration(source, model, suite, out)
    assert manifest["complete"] and manifest["total_segments"] == 6
    assert len(loads) == len(closed) == 1
    assert loads[0].revision == "dense-sha"
    rows = list(read_jsonl_dicts(out / "de-en.jsonl"))
    assert rows[0]["target"] == "ref 0"
    assert rows[0]["generated"] == "translation\nextra text"
    assert rows[0]["hypothesis"] == "translation"
    assert rows[1]["generated"] == ""  # Keep empty answers, and their tokens.
    assert rows[1]["source_truncated"]
    stats = manifest["directions"]["de-en"]
    assert stats["source_tokens"] == 4 and stats["generated_tokens"] == 2
    assert stats["empty_outputs"] == stats["source_truncated"] == stats["hit_token_budget"] == 1
    assert gc.generate_calibration(source, model, suite, out) == manifest
    assert len(loads) == 1 and len(calls) == 3


def test_resume_completed_directions(setup_cache: Any) -> None:
    source, out, model, suite, calls, loads, closed, fail = setup_cache
    fail.append("de-en")
    with pytest.raises(RuntimeError, match="interrupted"):
        gc.generate_calibration(source, model, suite, out)
    assert calls == ["cs-en", "de-en"]
    assert closed == [True]
    with pytest.raises(ValueError, match="incomplete"):
        load_calibration_prompts(out, with_generated=True)
    with pytest.raises(ValueError, match="incomplete"):
        load_calibration_prompts(out, with_target=True)
    before = (out / "cs-en.jsonl").read_bytes()
    fail.clear()
    gc.generate_calibration(source, model, suite, out)
    assert calls == ["cs-en", "de-en", "de-en", "en-de"]
    assert (out / "cs-en.jsonl").read_bytes() == before
    assert len(loads) == len(closed) == 2


@pytest.mark.parametrize("change", ["source", "decode", "model", "code", "revision"])
def test_refuse_stale_cache(setup_cache: Any, monkeypatch: pytest.MonkeyPatch, change: str) -> None:
    source, out, model, suite, _, loads, _, _ = setup_cache
    gc.generate_calibration(source, model, suite, out)
    if change == "source":
        rows = list(read_jsonl_dicts(source / "de-en.jsonl"))
        rows[0]["target"] = "changed reference"
        write_jsonl(source / "de-en.jsonl", rows)
    elif change == "decode":
        suite = replace(suite, decode=replace(suite.decode, batch_size=1))
    elif change == "model":
        model = replace(model, dtype="float32")
    elif change == "code":
        monkeypatch.setattr(gc, "_implementation", lambda: {"new": "code"})
    else:
        monkeypatch.setattr(
            gc,
            "_resolve_model",
            lambda spec: (replace(spec, revision="new-sha"), {"revision": "new-sha"}),
        )
    with pytest.raises(ValueError, match="settings/input changed"):
        gc.generate_calibration(source, model, suite, out)
    assert len(loads) == 1


def test_pair_selection_replays_tokens_and_keeps_references(setup_cache: Any) -> None:
    source, out, model, suite, *_ = setup_cache
    gc.generate_calibration(source, model, suite, out)
    sequences = load_calibration_prompts(out, directions=["de-en", "en-de"], with_generated=True)
    assert sequences == [[1, 10, 20, 2], [1, 11, 21, 2]] * 2
    refs = load_calibration_prompts(out, directions=["de-en"], with_target=True)
    rows = list(read_jsonl_dicts(source / "de-en.jsonl"))
    assert refs == [row["prompt"] + row["target"] for row in rows]
    spec = PruneSpec.from_dict(
        {
            "name": "pair",
            "method": "flap",
            "model_name_or_path": "dense",
            "calibration": str(out),
            "calibration_text": "prompt+generated",
            "directions": ["de-en", "en-de"],
            "max_length": 768,
        }
    )
    assert spec.calibration_text == "prompt+generated"


def test_tampering_is_rejected(setup_cache: Any) -> None:
    source, out, model, suite, *_ = setup_cache
    gc.generate_calibration(source, model, suite, out)
    path = out / "de-en.jsonl"
    path.write_text(path.read_text().replace("translation", "corruption"))
    with pytest.raises(ValueError, match="content changed"):
        load_calibration_prompts(out, with_generated=True)


def test_wrong_prompt_and_non_greedy_are_rejected(setup_cache: Any) -> None:
    source, out, model, suite, _, loads, *_ = setup_cache
    with pytest.raises(ValueError, match="greedy"):
        gc.generate_calibration(
            source, model, replace(suite, decode=replace(suite.decode, num_beams=5)), out
        )
    rows = list(read_jsonl_dicts(source / "de-en.jsonl"))
    rows[0]["prompt"] = "wrong prompt"
    write_jsonl(source / "de-en.jsonl", rows)
    with pytest.raises(ValueError, match="prompt or direction"):
        gc.generate_calibration(source, model, suite, out)
    assert not loads


def test_cached_tokens_are_padded_without_retokenization() -> None:
    class Tensor:
        def to(self, device: str) -> Any:
            return self

    class Tokenizer:
        def pad(self, rows: Any, **kwargs: Any) -> Any:
            assert rows == [
                {"input_ids": [1, 4, 2], "attention_mask": [1, 1, 1]},
                {"input_ids": [1, 2], "attention_mask": [1, 1]},
            ]
            return {"input_ids": Tensor(), "attention_mask": Tensor()}

    assert len(list(tokenized_batches([[1, 4, 2], [1, 2]], Tokenizer(), max_length=3))) == 1
    with pytest.raises(ValueError, match="exceeds max_length"):
        list(tokenized_batches([[1, 4, 2]], Tokenizer(), max_length=2))


def test_cli_generates_cache(setup_cache: Any, capsys: pytest.CaptureFixture[str]) -> None:
    from mnlp_eval.cli import main

    source, out, *_ = setup_cache
    assert (
        main(
            [
                "calibration-generate",
                "--calibration",
                str(source),
                "--model",
                "configs/models/alma-7b.yaml",
                "--decode-suite",
                "configs/suites/alma10-greedy.yaml",
                "--out",
                str(out),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["complete"]


def test_duplicate_union_prompts_generated_once(setup_cache: Any) -> None:
    source, out, model, suite, *_ = setup_cache
    path = source / "de-en.jsonl"
    rows = list(read_jsonl_dicts(path))
    write_jsonl(path, [*rows, dict(rows[0], id=999)])
    cache = gc.generate_calibration(source, model, suite, out)
    assert cache["total_segments"] == 6
    assert cache["directions"]["de-en"]["n_segments"] == 2
    assert gc.prompt_id("de-en", rows[0]["prompt"]) != gc.prompt_id("en-de", rows[0]["prompt"])


def test_hub_revision_is_pinned_before_loading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys
    import types

    calls: list[dict[str, Any]] = []

    def snapshot(repository: str, **kwargs: Any) -> str:
        assert repository == "dense/model"
        calls.append(kwargs)
        return str(tmp_path / "snapshots" / "immutable-sha")

    hub = types.ModuleType("huggingface_hub")
    hub.snapshot_download = snapshot  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    spec = ModelSpec(name="dense", loader="hf_causal", model_name_or_path="dense/model")
    pinned, resolved = gc._resolve_model(spec)
    assert pinned.revision == resolved["revision"] == "immutable-sha"
    assert calls == [
        {"revision": None, "allow_patterns": ["config.json"], "local_files_only": True}
    ]


def test_local_checkpoint_edits_change_identity(tmp_path: Path) -> None:
    checkpoint = tmp_path / "dense"
    checkpoint.mkdir()
    config = checkpoint / "config.json"
    config.write_text("{}")
    model = ModelSpec(name="dense", loader="hf_causal", model_name_or_path=str(checkpoint))
    _, before = gc._resolve_model(model)
    config.write_text('{"changed": true}')
    _, after = gc._resolve_model(model)
    assert gc.fingerprint(before) != gc.fingerprint(after)
