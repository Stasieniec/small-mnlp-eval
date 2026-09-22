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
    """Splits on whitespace and pads. Enough for a calibration pass."""

    pad_token_id: int | None = 0
    eos_token = "</s>"
    pad_token = "<pad>"

    def __call__(self, texts: list[str], **kwargs: Any) -> Any:
        import torch

        limit = int(kwargs.get("max_length", 16))
        rows = [
            [(len(word) * 7 + index) % 64 for index, word in enumerate(text.split())][:limit]
            for text in texts
        ]
        width = max(len(row) for row in rows)
        ids = torch.tensor([[0] * (width - len(row)) + row for row in rows])
        mask = torch.tensor([[0] * (width - len(row)) + [1] * len(row) for row in rows])
        return _Encoded({"input_ids": ids, "attention_mask": mask})

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
