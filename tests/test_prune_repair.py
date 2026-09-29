"""LoRA repair of a pruned checkpoint.

What matters, in order: the loss is taken on the reference and eos only, so
repair teaches translation rather than prompt echoing; the repaired checkpoint
keeps the pruned shapes and loads through the same recipe; and the emitted
model config inherits everything from the pruned one except the checkpoint and
the repair note, so the report pairs the two systems.

Uses a real fast tokenizer built in memory, so the tokenizer save and load path
is the production one. No GPU, network or weights.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from mnlp_eval.config import ConfigError, ModelSpec, load_yaml_config
from mnlp_eval.prune import PruneError
from mnlp_eval.prune.compact import LayerPlan, compact_model, write_descriptor
from mnlp_eval.prune.groups import describe_layers
from mnlp_eval.prune.repair import (
    IGNORE,
    Example,
    RepairSpec,
    encode_example,
    length_grouped_batches,
    resolve_source,
    run_repair,
    train_lora,
)
from stubs import tiny_llama

pytestmark = pytest.mark.torch

WORDS = [f"w{index}" for index in range(56)]


def word_tokenizer() -> Any:
    """Whitespace word-level tokenizer over a 64-entry vocabulary, bos 2, eos 1."""
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    from transformers import PreTrainedTokenizerFast

    vocab = {
        "<unk>": 0,
        "</s>": 1,
        "<s>": 2,
        ":": 3,
        **{word: 4 + i for i, word in enumerate(WORDS)},
    }
    backend = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Sequence(
        [pre_tokenizers.WhitespaceSplit(), pre_tokenizers.Punctuation()]
    )
    backend.post_processor = processors.TemplateProcessing(
        single="<s> $A", special_tokens=[("<s>", 2)]
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=backend, bos_token="<s>", eos_token="</s>", unk_token="<unk>"
    )


def sentence(seed: int, length: int) -> str:
    return " ".join(WORDS[(seed * 5 + index * 3) % len(WORDS)] for index in range(length))


def records(direction: str, count: int) -> list[dict[str, Any]]:
    return [
        {
            "id": index,
            "direction": direction,
            "source": sentence(index, 4),
            "target": " " + sentence(index + 1, 3),
            "prompt": f"w0 w1 {sentence(index, 4)} w2 :",
        }
        for index in range(count)
    ]


class TestEncoding:
    def test_the_loss_is_on_the_reference_and_eos_only(self) -> None:
        tokenizer = word_tokenizer()

        example = encode_example(tokenizer, "w0 w1 w5 :", " w7 w8", max_length=64)

        assert example is not None
        assert example.input_ids == (2, 4, 5, 9, 3, 11, 12, 1)
        assert example.labels == (IGNORE, IGNORE, IGNORE, IGNORE, IGNORE, 11, 12, 1)
        assert example.n_target_tokens == 3

    def test_truncation_keeps_eos_as_the_last_token(self) -> None:
        tokenizer = word_tokenizer()

        example = encode_example(tokenizer, "w0 :", " w7 w8 w9 w10", max_length=5)

        assert example is not None
        assert len(example.input_ids) == 5
        assert example.input_ids[-1] == 1
        assert example.labels[-1] == 1

    def test_a_segment_whose_reference_is_truncated_away_is_dropped(self) -> None:
        tokenizer = word_tokenizer()

        assert encode_example(tokenizer, "w0 w1 w2 w3 :", " w7", max_length=4) is None


def test_length_grouping_uses_every_segment_exactly_once() -> None:
    examples = [
        Example(direction="de-en", input_ids=tuple(range(n % 9 + 2)), labels=(IGNORE, 1))
        for n in range(37)
    ]

    batches = length_grouped_batches(examples, 4, seed=3, window=2)

    flat = sorted(index for batch in batches for index in batch)
    assert flat == list(range(37))
    assert batches == length_grouped_batches(examples, 4, seed=3, window=2)
    assert batches != length_grouped_batches(examples, 4, seed=4, window=2)


def small_spec(**overrides: Any) -> RepairSpec:
    settings: dict[str, Any] = {
        "name": "tiny-lora",
        "source": "unused.yaml",
        "data": "unused",
        "rank": 4,
        "alpha": 8,
        "dropout": 0.0,
        "learning_rate": 1e-2,
        "epochs": 6,
        "batch_size": 4,
        "gradient_accumulation": 2,
        "warmup_ratio": 0.0,
        "max_length": 64,
        "gradient_checkpointing": False,
        "log_every": 1000,
    }
    settings.update(overrides)
    return RepairSpec(**settings)


def test_training_lowers_the_loss_and_merges_back_to_the_pruned_shapes() -> None:
    tokenizer = word_tokenizer()
    model = tiny_llama()
    plan = (
        LayerPlan(heads=(0, 2), channels=tuple(range(20))),
        LayerPlan(heads=(0, 1, 2, 3), channels=tuple(range(40))),
    )
    compact_model(model, plan)
    shapes = {name: tuple(p.shape) for name, p in model.named_parameters()}
    examples = [
        example
        for record in records("de-en", 16)
        if (example := encode_example(tokenizer, record["prompt"], record["target"], max_length=64))
    ]

    merged, log = train_lora(model, examples, examples[:8], small_spec(), pad_id=1, device="cpu")

    assert log["held_out_loss_after"] < log["held_out_loss_before"]
    assert log["optimizer_steps"] == 6 * 2
    assert {name: tuple(p.shape) for name, p in merged.named_parameters()} == shapes
    assert not any("lora" in name for name, _ in merged.named_parameters())


@pytest.fixture
def pruned_system(tmp_path: Path) -> dict[str, Path]:
    """A compacted tiny Llama on disk, its descriptor, tokenizer and model config."""
    model = tiny_llama()
    groups = describe_layers(model)
    plan = (
        LayerPlan(heads=(1, 3), channels=tuple(range(0, 48, 2))),
        LayerPlan(heads=(0, 1, 2, 3), channels=tuple(range(36))),
    )
    compact_model(model, plan)
    checkpoint = tmp_path / "checkpoints" / "tiny-pruned"
    model.save_pretrained(checkpoint)
    word_tokenizer().save_pretrained(checkpoint)
    write_descriptor(
        checkpoint / "subnetwork.json",
        plan,
        groups,
        name="tiny-pruned",
        base_model="test",
        method="SlimGPT, global budget",
    )

    config_dir = tmp_path / "models"
    config_dir.mkdir()
    (config_dir / "base.yaml").write_text(
        yaml.safe_dump(
            {"name": "base", "loader": "hf_causal", "prompt": "alma", "model_name_or_path": "x"}
        ),
        encoding="utf-8",
    )
    source = config_dir / "tiny-pruned.yaml"
    source.write_text(
        yaml.safe_dump(
            {
                "extends": "base.yaml",
                "name": "tiny-pruned",
                "baseline": "base",
                "loader": "custom",
                "entrypoint": "recipes.pruned:load",
                "kwargs": {"checkpoint": str(checkpoint)},
                "compression": {
                    "family": "pruning",
                    "method": "SlimGPT, global budget",
                    "nominal_sparsity": 0.2,
                    "pruned_for": "multi",
                },
            }
        ),
        encoding="utf-8",
    )

    data = tmp_path / "repair-set"
    data.mkdir()
    for direction in ("de-en", "en-de"):
        (data / f"{direction}.jsonl").write_text(
            "\n".join(json.dumps(record) for record in records(direction, 12)) + "\n",
            encoding="utf-8",
        )
    return {"checkpoint": checkpoint, "source": source, "data": data, "config_dir": config_dir}


def test_a_repair_run_writes_a_loadable_checkpoint_and_a_paired_config(
    pruned_system: dict[str, Path], tmp_path: Path
) -> None:
    import torch
    from recipes.pruned import load_model
    from safetensors.torch import load_file

    spec = small_spec(
        name="tiny-pruned-lora",
        source=str(pruned_system["source"]),
        data=str(pruned_system["data"]),
        held_out=4,
        epochs=2,
    )

    manifest = run_repair(spec, tmp_path / "out", config_dir=pruned_system["config_dir"])

    target = Path(manifest["checkpoint"])
    assert manifest["segments"] == {"train": 20, "held_out": 4, "dropped": 0}
    assert manifest["directions"] == ["de-en", "en-de"]
    assert json.loads((target / "repair.json").read_text())["name"] == "tiny-pruned-lora"

    # Same shapes as the pruned parent, loaded through the same recipe.
    repaired, subnetwork = load_model(target)
    parent, _ = load_model(pruned_system["checkpoint"])
    assert subnetwork.name == "tiny-pruned"
    shapes = {name: tuple(p.shape) for name, p in parent.named_parameters()}
    assert {name: tuple(p.shape) for name, p in repaired.named_parameters()} == shapes
    # And different weights: the adapter was merged, not dropped.
    before = load_file(next(pruned_system["checkpoint"].glob("*.safetensors")))
    after = load_file(next(target.glob("*.safetensors")))
    assert any(not torch.equal(before[key], after[key]) for key in before)

    # The config inherits the pruned system and changes only what repair did.
    config = load_yaml_config(manifest["model_config"])
    spec_after = ModelSpec.from_dict(config)
    assert spec_after.name == "tiny-pruned-lora"
    assert spec_after.baseline == "base"
    assert spec_after.kwargs["checkpoint"] == str(target)
    assert spec_after.compression.nominal_sparsity == 0.2
    assert spec_after.compression.repair.startswith("LoRA r=4 alpha=8")
    assert "repaired" in spec_after.compression.describe()


def test_repairing_a_repaired_system_is_refused(
    pruned_system: dict[str, Path], tmp_path: Path
) -> None:
    spec = small_spec(
        name="tiny-pruned-lora",
        source=str(pruned_system["source"]),
        data=str(pruned_system["data"]),
        held_out=2,
        epochs=1,
    )
    manifest = run_repair(spec, tmp_path / "out", config_dir=pruned_system["config_dir"])

    with pytest.raises(PruneError, match="already repaired"):
        resolve_source(manifest["model_config"])


def test_a_source_that_is_not_a_pruned_checkpoint_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "dense.yaml"
    path.write_text(
        yaml.safe_dump({"name": "d", "loader": "hf_causal", "model_name_or_path": "x"}),
        encoding="utf-8",
    )

    with pytest.raises(PruneError, match=r"recipes\.pruned:load"):
        resolve_source(path)


def test_the_shipped_repair_configs_all_parse() -> None:
    paths = sorted(Path("configs/repair").glob("*.yaml"))

    assert paths
    for path in paths:
        spec = RepairSpec.from_dict(load_yaml_config(path))
        assert spec.name.endswith("-lora")
        assert Path(spec.source).parent == Path("configs/models")


@pytest.mark.parametrize(
    ("override", "expected"),
    [
        ({"rank": 0}, "rank must be at least 1"),
        ({"dropout": 1.0}, "dropout"),
        ({"learning_rate": 0.0}, "learning_rate"),
        ({"unknown_field": 1}, "unknown_field"),
    ],
)
def test_a_bad_spec_is_refused(override: dict[str, Any], expected: str) -> None:
    payload = {"name": "x", "source": "s.yaml", "data": "d", **override}

    with pytest.raises(ConfigError, match=expected):
        RepairSpec.from_dict(payload)
