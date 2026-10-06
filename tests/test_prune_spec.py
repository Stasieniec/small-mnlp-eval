"""The pruning stage end to end.

What the criterion tests cannot cover: that a run writes a checkpoint, a
descriptor in both places it is needed, and a model config the rest of the
harness accepts, and that the two descriptors agree. The Hub is stubbed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from mnlp_eval.config import ConfigError, load_yaml_config
from mnlp_eval.prune.spec import PruneSpec, run_pruning
from stubs import tiny_llama

pytestmark = pytest.mark.torch


class StubTokenizer:
    """Splits on whitespace and pads on the left. Enough for a calibration pass.

    Takes one text too, returning plain lists as a real tokenizer does without
    ``return_tensors``, because reference calibration appends eos to each text
    before padding.
    """

    pad_token_id: int | None = 0
    eos_token_id: int | None = 1
    eos_token = "</s>"
    pad_token = "<pad>"

    def __call__(self, texts: str | list[str], **kwargs: Any) -> Any:
        limit = int(kwargs.get("max_length", 16))
        if isinstance(texts, str):
            ids = self._ids(texts)[:limit]
            return {"input_ids": ids, "attention_mask": [1] * len(ids)}
        return self.pad([{"input_ids": self._ids(text)[:limit]} for text in texts])

    def pad(self, rows: list[dict[str, list[int]]], **_: Any) -> Any:
        import torch

        width = max(len(row["input_ids"]) for row in rows)
        ids = torch.tensor(
            [[0] * (width - len(row["input_ids"])) + row["input_ids"] for row in rows]
        )
        mask = torch.tensor(
            [[0] * (width - len(row["input_ids"])) + [1] * len(row["input_ids"]) for row in rows]
        )
        return _Encoded({"input_ids": ids, "attention_mask": mask})

    @staticmethod
    def _ids(text: str) -> list[int]:
        return [(len(word) * 7 + index) % 64 for index, word in enumerate(text.split())]

    def save_pretrained(self, directory: str | Path) -> None:
        Path(directory, "tokenizer_config.json").write_text("{}", encoding="utf-8")


class _Encoded(dict):
    def to(self, device: str) -> Any:
        return {key: value.to(device) for key, value in self.items()}

    def items(self) -> Any:
        return super().items()


@pytest.fixture
def calibration(tmp_path: Path) -> Path:
    root = tmp_path / "calib"
    root.mkdir()
    for direction in ("de-en", "en-de"):
        records = [
            {
                "id": index,
                "direction": direction,
                "source": "s",
                "target": "t",
                "prompt": f"Translate this from German to English word {index} and more text here",
            }
            for index in range(6)
        ]
        (root / f"{direction}.jsonl").write_text(
            "\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8"
        )
    return root


@pytest.fixture
def model_config_dir(tmp_path: Path) -> Path:
    """The baseline the emitted config extends, copied so a run leaves no
    stray model config in configs/models.
    """
    import shutil

    directory = tmp_path / "models"
    directory.mkdir()
    shutil.copy("configs/models/alma-7b.yaml", directory / "alma-7b.yaml")
    return directory


@pytest.fixture
def stub_hub(monkeypatch: pytest.MonkeyPatch) -> None:
    import transformers

    def tokenizer(*_args: Any, **_kwargs: Any) -> StubTokenizer:
        return StubTokenizer()

    def model(*_args: Any, **_kwargs: Any) -> Any:
        return tiny_llama()

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", tokenizer)
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", model)


@pytest.mark.parametrize("method", ["flap", "llm-pruner", "slimgpt"])
def test_a_run_writes_a_checkpoint_a_descriptor_and_a_model_config(
    method: str, calibration: Path, tmp_path: Path, model_config_dir: Path, stub_hub: None
) -> None:
    from mnlp_eval.analysis.subnetwork import load_subnetwork

    spec = PruneSpec.from_dict(
        {
            "name": f"stub-{method}50",
            "method": method,
            "model_name_or_path": "stub/model",
            "calibration": str(calibration),
            "sparsity": 0.5,
            "allocation": "global",
            "dtype": "float32",
            "batch_size": 3,
            "max_length": 12,
        }
    )

    manifest = run_pruning(
        spec,
        tmp_path / "out",
        subnetwork_dir=tmp_path / "subnetworks",
        config_dir=model_config_dir,
    )

    checkpoint = Path(manifest["checkpoint"])
    assert (checkpoint / "model.safetensors").is_file()
    assert (checkpoint / "config.json").is_file()
    assert abs(manifest["unit_sparsity"] - 0.5) < 0.05

    # The overlap analysis reads one copy and the loader the other, so a
    # disagreement means the model loaded is not the model analysed.
    beside = load_subnetwork(checkpoint / "subnetwork.json")
    alongside = load_subnetwork(Path(manifest["subnetwork"]))
    assert beside.components["attention_heads"].kept == (
        alongside.components["attention_heads"].kept
    )
    assert beside.components["ffn_channels"].kept == alongside.components["ffn_channels"].kept


def test_flap_al_am_prunes_to_a_parameter_budget_and_reloads(
    calibration: Path, tmp_path: Path, model_config_dir: Path, stub_hub: None
) -> None:
    import yaml
    from recipes.pruned import load_model

    spec = PruneSpec.from_dict(
        {
            "name": "stub-flap-al-am30",
            "method": "flap",
            "model_name_or_path": "stub/model",
            "calibration": str(calibration),
            "sparsity": 0.3,
            "allocation": "al-am",
            "dtype": "float32",
            "batch_size": 3,
            "max_length": 12,
        }
    )

    manifest = run_pruning(
        spec,
        tmp_path / "out",
        subnetwork_dir=tmp_path / "subnetworks",
        config_dir=model_config_dir,
    )

    # The stub's 4 heads of head_dim 8 and 48 channels per layer: a head is
    # 32 / 544 of the two layers' weights, and the cut is nearest the budget.
    assert abs(manifest["parameter_sparsity"] - 0.3) <= 16 / 544
    components = manifest["components"]
    assert components["attention_heads"]["kept"] >= 2
    assert components["ffn_channels"]["kept"] >= 2
    notes = json.loads(Path(manifest["subnetwork"]).read_text())["notes"]
    assert notes.startswith("al-am budget")
    assert "of attention and MLP weights" in notes
    emitted = yaml.safe_load(Path(manifest["model_config"]).read_text())
    assert emitted["compression"]["method"] == "FLAP, al-am budget"

    loaded, _ = load_model(manifest["checkpoint"])
    assert len(loaded.model.layers) == 2


def test_target_calibration_reads_the_reference_after_the_prompt(calibration: Path) -> None:
    from mnlp_eval.prune.collect import load_calibration_prompts

    plain = load_calibration_prompts(calibration, directions=["de-en"])
    with_target = load_calibration_prompts(calibration, directions=["de-en"], with_target=True)

    # ALMA joins prompt and reference with no space; the fixture's target is "t".
    assert with_target == [prompt + "t" for prompt in plain]


@pytest.mark.parametrize("method", ["flap", "slimgpt"])
def test_target_calibration_is_recorded_where_the_report_reads_it(
    method: str,
    calibration: Path,
    tmp_path: Path,
    model_config_dir: Path,
    stub_hub: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import yaml

    from mnlp_eval.prune import collect

    batched: list[Any] = []
    tokenize = collect.tokenized_batches

    def recording(*args: Any, **kwargs: Any) -> Any:
        for batch in tokenize(*args, **kwargs):
            batched.append(batch["input_ids"])
            yield batch

    monkeypatch.setattr(collect, "tokenized_batches", recording)

    def run(calibration_text: str) -> dict[str, Any]:
        spec = PruneSpec.from_dict(
            {
                "name": f"stub-{method}-{calibration_text.replace('+', '-')}",
                "method": method,
                "model_name_or_path": "stub/model",
                "calibration": str(calibration),
                "calibration_text": calibration_text,
                "sparsity": 0.5,
                "allocation": "global",
                "dtype": "float32",
                "batch_size": 3,
                "max_length": 16,
            }
        )
        return run_pruning(
            spec,
            tmp_path / "out",
            subnetwork_dir=tmp_path / "subnetworks",
            config_dir=model_config_dir,
        )

    plain = run("prompt")
    plain_batches, batched[:] = list(batched), []
    target = run("prompt+target")

    # The reference ends with eos, as a cached generation does; the prompt alone
    # does not. Left padding puts every row's last token in the last column.
    assert batched and plain_batches
    assert all((ids[:, -1] == StubTokenizer.eos_token_id).all() for ids in batched)
    assert not any((ids[:, -1] == StubTokenizer.eos_token_id).any() for ids in plain_batches)

    # The descriptor is what the overlap analysis reads, so it must say which.
    assert "prompt+target text" in json.loads(Path(target["subnetwork"]).read_text())["notes"]
    assert "prompt text" in json.loads(Path(plain["subnetwork"]).read_text())["notes"]
    emitted = yaml.safe_load(Path(target["model_config"]).read_text())
    assert emitted["compression"]["method"].endswith("prompt+target calibration")
    assert (
        "calibration"
        not in yaml.safe_load(Path(plain["model_config"]).read_text())["compression"]["method"]
    )


def test_the_emitted_model_config_is_one_the_harness_accepts(
    calibration: Path, tmp_path: Path, model_config_dir: Path, stub_hub: None
) -> None:
    from mnlp_eval.config import ModelSpec

    spec = PruneSpec.from_dict(
        {
            "name": "stub-flap50-de",
            "method": "flap",
            "model_name_or_path": "stub/model",
            "calibration": str(calibration),
            "directions": ["de-en", "en-de"],
            "sparsity": 0.25,
            "dtype": "float32",
            "batch_size": 3,
            "max_length": 12,
        }
    )

    manifest = run_pruning(
        spec,
        tmp_path / "out",
        subnetwork_dir=tmp_path / "subnetworks",
        config_dir=model_config_dir,
    )
    # Loaded through the extends chain, so a config that cannot find its
    # baseline fails here rather than at the start of a cluster run.
    model_spec = ModelSpec.from_dict(load_yaml_config(manifest["model_config"]))

    assert model_spec.loader == "custom"
    assert model_spec.entrypoint == "recipes.pruned:load"
    # Without pruned_for the report cannot build the transfer matrix at all.
    assert model_spec.compression.pruned_for == "de-en,en-de"
    assert model_spec.compression.family == "pruning"
    assert model_spec.kwargs["checkpoint"] == manifest["checkpoint"]


def test_the_shipped_configs_all_parse() -> None:
    for path in sorted(Path("configs/prune").glob("*.yaml")):
        spec = PruneSpec.from_dict(load_yaml_config(path))
        assert spec.method in {"flap", "llm-pruner", "slimgpt"}
        assert 0.0 <= spec.sparsity < 1.0


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"method": "wanda"}, "is not one of"),
        ({"allocation": "clairvoyant"}, "is not one of"),
        ({"sparsity": 1.0}, "fraction removed"),
        ({"name": ""}, "'name' is required"),
        ({"batch_size": 0}, "at least 1"),
        ({"calibration_text": "reference"}, "is not one of"),
        ({"method": "slimgpt", "allocation": "al-am"}, "only with method 'flap'"),
        ({"method": "llm-pruner", "allocation": "al-am"}, "only with method 'flap'"),
    ],
)
def test_a_bad_spec_is_refused(payload: dict[str, Any], expected: str) -> None:
    base = {
        "name": "x",
        "method": "flap",
        "model_name_or_path": "stub/model",
        "calibration": "data/calibration/x",
    }
    base.update(payload)

    with pytest.raises(ConfigError, match=expected):
        PruneSpec.from_dict(base)


def test_an_unknown_field_is_refused_rather_than_ignored() -> None:
    with pytest.raises(ConfigError, match="prune spec"):
        PruneSpec.from_dict(
            {
                "name": "x",
                "method": "flap",
                "model_name_or_path": "stub/model",
                "calibration": "data/calibration/x",
                "sparsty": 0.5,
            }
        )


def test_the_cli_runs_the_stage(
    calibration: Path, tmp_path: Path, stub_hub: None, capsys: pytest.CaptureFixture[str]
) -> None:
    from mnlp_eval.cli import main

    config = tmp_path / "prune.yaml"
    config.write_text(
        "\n".join(
            [
                "name: cli-flap50",
                "method: flap",
                "model_name_or_path: stub/model",
                f"calibration: {calibration}",
                "sparsity: 0.5",
                "dtype: float32",
                "batch_size: 3",
                "max_length: 12",
            ]
        ),
        encoding="utf-8",
    )

    code = main(
        [
            "prune",
            "--spec",
            str(config),
            "--out",
            str(tmp_path / "out"),
            "--subnetwork-dir",
            str(tmp_path / "subnetworks"),
            "--model-config-dir",
            str(tmp_path / "models"),
            "--set",
            "sparsity=0.25",
        ]
    )

    assert code == 0
    manifest = json.loads(capsys.readouterr().out)
    assert manifest["requested_sparsity"] == 0.25
    assert abs(manifest["unit_sparsity"] - 0.25) < 0.05
    # Heads and channels each lose a quarter, so the weights do too.
    assert manifest["parameter_sparsity"] == pytest.approx(0.25)
    assert Path(manifest["subnetwork"]).is_file()


class TestPruneSpecCommand:
    """``prune-spec`` is how the Slurm scripts resolve a config.

    They must not parse YAML themselves: the ``extends`` chain and the
    ``directions`` normalisation live in the config loader, and a shell
    reimplementation of either would drift.
    """

    def test_a_field_prints_one_bare_line(self, capsys: pytest.CaptureFixture[str]) -> None:
        from mnlp_eval.cli import main

        assert (
            main(["prune-spec", "--spec", "configs/prune/slimgpt-50-multi.yaml", "--field", "name"])
            == 0
        )

        assert capsys.readouterr().out == "alma-7b-slimgpt50-multi\n"

    def test_a_list_field_is_joined_so_a_shell_can_use_it(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from mnlp_eval.cli import main

        assert (
            main(["prune-spec", "--spec", "configs/prune/flap-50-de.yaml", "--field", "pruned_for"])
            == 0
        )

        assert capsys.readouterr().out == "de-en,en-de\n"

    def test_an_extended_config_resolves_through_its_parent(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from mnlp_eval.cli import main

        # flap-50-de.yaml sets only name, calibration and directions; method
        # and sparsity come from the config it extends.
        assert (
            main(["prune-spec", "--spec", "configs/prune/flap-50-de.yaml", "--field", "method"])
            == 0
        )

        assert capsys.readouterr().out == "flap\n"

    def test_the_whole_spec_prints_as_json(self, capsys: pytest.CaptureFixture[str]) -> None:
        from mnlp_eval.cli import main

        assert main(["prune-spec", "--spec", "configs/prune/flap-50-multi.yaml"]) == 0

        payload = json.loads(capsys.readouterr().out)
        assert payload["method"] == "flap"
        assert payload["allocation"] == "global"

    def test_a_broken_config_fails_before_anything_is_queued(self, tmp_path: Path) -> None:
        from mnlp_eval.cli import main

        config = tmp_path / "bad.yaml"
        config.write_text(
            "name: x\nmethod: wanda\nmodel_name_or_path: m\ncalibration: c\n", encoding="utf-8"
        )

        assert main(["prune-spec", "--spec", str(config), "--field", "name"]) == 1


@pytest.mark.parametrize("method", ["flap", "slimgpt"])
def test_generated_cache_prunes_and_records_provenance(
    method: str,
    calibration: Path,
    tmp_path: Path,
    model_config_dir: Path,
    stub_hub: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dataclasses import replace

    import torch
    import transformers

    from mnlp_eval.artifacts import read_jsonl_dicts, write_jsonl
    from mnlp_eval.config import ModelSpec, SuiteSpec
    from mnlp_eval.data import generated_calibration as gc
    from mnlp_eval.languages import parse_direction
    from mnlp_eval.models import SegmentOutput
    from mnlp_eval.prompts import get_prompt

    (calibration / "calibration.json").write_text("{}")
    for path in calibration.glob("*.jsonl"):
        rows = list(read_jsonl_dicts(path))
        for row in rows:
            row["source"] = f"source {row['id']}"
            row["prompt"] = get_prompt("alma").render(parse_direction(path.stem), row["source"])
        write_jsonl(path, rows)

    class Generator:
        def translate(self, direction: Any, sources: Any, decode: Any, **kwargs: Any) -> Any:
            return [
                SegmentOutput("generated", input_token_ids=[1, 3, 4], generated_token_ids=[5, 6, 2])
                for _ in sources
            ]

        def close(self) -> None:
            pass

    class ReplayTokenizer(StubTokenizer):
        def pad(self, rows: Any, **kwargs: Any) -> Any:
            return {
                key: torch.tensor([row[key] for row in rows])
                for key in ("input_ids", "attention_mask")
            }

    revisions: list[str] = []

    def tokenizer(*args: Any, **kwargs: Any) -> Any:
        revisions.append(kwargs["revision"])
        return ReplayTokenizer()

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", tokenizer)
    monkeypatch.setattr(gc, "build_translator", lambda _: Generator())
    monkeypatch.setattr(
        gc,
        "_resolve_model",
        lambda spec: (
            replace(spec, revision="pinned-sha"),
            {"repository": "stub/model", "revision": "pinned-sha"},
        ),
    )
    out = tmp_path / "generated"
    cache = gc.generate_calibration(
        calibration,
        ModelSpec(name="dense", loader="hf_causal", model_name_or_path="stub/model"),
        SuiteSpec.from_dict(load_yaml_config("configs/suites/alma10-greedy.yaml")),
        out,
    )
    spec = PruneSpec.from_dict(
        {
            "name": f"generated-{method}",
            "method": method,
            "model_name_or_path": "stub/model",
            "calibration": str(out),
            "calibration_text": "prompt+generated",
            "dtype": "float32",
            "directions": ["de-en", "en-de"],
            "max_length": 12,
        }
    )
    report = run_pruning(
        spec, tmp_path / "out", subnetwork_dir=tmp_path / "subnetworks", config_dir=model_config_dir
    )
    assert revisions == ["pinned-sha"]
    assert report["calibration_segments"] == 12
    checkpoint = Path(report["checkpoint"])
    provenance = json.loads((checkpoint / "calibration_provenance.json").read_text())
    assert provenance["cache_fingerprint"] == cache["fingerprint"]
    assert provenance["text"] == "prompt+generated"
    assert provenance["directions"] == ["de-en", "en-de"]
    assert cache["fingerprint"] in (checkpoint / "subnetwork.json").read_text()

    from mnlp_eval.prune import PruneError

    for changed, message in (
        (replace(spec, model_name_or_path="wrong/model"), "different dense"),
        (replace(spec, max_length=1), "exceeds max_length"),
    ):
        with pytest.raises(PruneError, match=message):
            run_pruning(
                changed,
                tmp_path / "out",
                subnetwork_dir=tmp_path / "subnetworks",
                config_dir=model_config_dir,
            )
    assert revisions == ["pinned-sha"]  # Refuse before another model/tokenizer load.
