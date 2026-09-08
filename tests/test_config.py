"""Configuration parsing, validation, and run identity."""

from __future__ import annotations

from pathlib import Path

import pytest

from mnlp_eval.config import (
    BenchSpec,
    ConfigError,
    DataSpec,
    DecodeSpec,
    MetricsSpec,
    ModelSpec,
    RunConfig,
    SuiteSpec,
    canonical_json,
    load_yaml_config,
)

BASE_MODEL = {
    "name": "alma-7b-r",
    "loader": "hf_causal",
    "prompt": "alma",
    "model_name_or_path": "haoranxu/ALMA-7B-R",
    "dtype": "bfloat16",
}


def _suite(**decode: object) -> SuiteSpec:
    return SuiteSpec.from_dict(
        {
            "name": "suite",
            "data": {"directions": ["de-en", "en-de"]},
            "decode": decode or {},
        }
    )


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------


def test_unknown_key_is_rejected_with_the_valid_keys_listed() -> None:
    # A typo in a teammate's YAML must fail immediately, not silently fall back
    # to a default and invalidate a comparison.
    with pytest.raises(ConfigError, match="unknown key"):
        ModelSpec.from_dict({**BASE_MODEL, "modelname": "oops"})


def test_missing_required_keys_are_rejected() -> None:
    with pytest.raises(ConfigError, match="'name' is required"):
        ModelSpec.from_dict({"loader": "hf_causal"})
    with pytest.raises(ConfigError, match="'loader' is required"):
        ModelSpec.from_dict({"name": "x"})


def test_model_path_is_required_unless_the_loader_is_custom() -> None:
    with pytest.raises(ConfigError, match="model_name_or_path"):
        ModelSpec.from_dict({"name": "x", "loader": "hf_causal"})


def test_custom_loader_requires_a_dotted_entrypoint() -> None:
    with pytest.raises(ConfigError, match="requires 'entrypoint'"):
        ModelSpec.from_dict({"name": "x", "loader": "custom"})
    with pytest.raises(ConfigError, match=r"package\.module:function"):
        ModelSpec.from_dict({"name": "x", "loader": "custom", "entrypoint": "nocolon"})
    spec = ModelSpec.from_dict(
        {"name": "x", "loader": "custom", "entrypoint": "recipes.wanda:load"}
    )
    assert spec.entrypoint == "recipes.wanda:load"


def test_invalid_dtype_and_prompt_are_rejected() -> None:
    with pytest.raises(ConfigError, match="dtype"):
        ModelSpec.from_dict({**BASE_MODEL, "dtype": "fp16"})
    with pytest.raises(ConfigError, match="unknown prompt template"):
        ModelSpec.from_dict({**BASE_MODEL, "prompt": "nope"})


def test_quantization_accepts_inline_or_nested_options() -> None:
    inline = ModelSpec.from_dict(
        {**BASE_MODEL, "quantization": {"method": "bitsandbytes", "load_in_4bit": True}}
    )
    nested = ModelSpec.from_dict(
        {
            **BASE_MODEL,
            "quantization": {"method": "bitsandbytes", "options": {"load_in_4bit": True}},
        }
    )
    assert inline.quantization is not None
    assert inline.quantization.options == nested.quantization.options  # type: ignore[union-attr]


def test_unsupported_quantization_method_is_rejected() -> None:
    with pytest.raises(ConfigError, match="unsupported method"):
        ModelSpec.from_dict({**BASE_MODEL, "quantization": {"method": "magic"}})


def test_sampling_parameters_without_sampling_are_rejected() -> None:
    # Silently ignoring temperature while running beam search is how a run ends
    # up not doing what its config says.
    with pytest.raises(ConfigError, match="do_sample is false"):
        DecodeSpec.from_dict({"temperature": 0.6})
    with pytest.raises(ConfigError, match="do_sample is false"):
        DecodeSpec.from_dict({"top_p": 0.9, "top_k": 50})


def test_sampling_parameters_are_allowed_when_sampling_is_on() -> None:
    spec = DecodeSpec.from_dict({"do_sample": True, "temperature": 0.6, "top_p": 0.9})
    assert spec.temperature == 0.6
    assert not spec.is_deterministic


def test_decode_defaults_are_deterministic() -> None:
    spec = DecodeSpec()
    assert spec.is_deterministic
    assert not spec.do_sample


def test_decode_rejects_nonsensical_values() -> None:
    for payload in ({"num_beams": 0}, {"batch_size": 0}, {"max_new_tokens": 0}):
        with pytest.raises(ConfigError):
            DecodeSpec.from_dict(payload)


def test_data_spec_rejects_a_non_positive_limit() -> None:
    with pytest.raises(ConfigError, match="limit must be positive"):
        DataSpec.from_dict({"directions": ["de-en"], "limit": 0})


def test_metrics_spec_rejects_unknown_groups() -> None:
    with pytest.raises(ConfigError, match="unknown group"):
        MetricsSpec.from_dict({"groups": ["surface", "magic"]})


def test_bench_spec_rejects_empty_batch_sizes() -> None:
    with pytest.raises(ConfigError, match="batch_sizes"):
        BenchSpec.from_dict({"batch_sizes": []})


def test_decode_describe_reads_naturally() -> None:
    assert DecodeSpec(num_beams=5).describe() == "beam5, max_new_tokens=256"
    assert DecodeSpec(num_beams=1).describe() == "greedy, max_new_tokens=256"
    assert DecodeSpec(do_sample=True).describe().startswith("sample")


# --------------------------------------------------------------------------
# Run identity
# --------------------------------------------------------------------------


def test_renaming_a_model_does_not_change_its_run_identity() -> None:
    suite = _suite()
    original = RunConfig(ModelSpec.from_dict(BASE_MODEL), suite)
    renamed = RunConfig(ModelSpec.from_dict({**BASE_MODEL, "name": "something-else"}), suite)
    assert original.run_id == renamed.run_id


def test_notes_and_baseline_do_not_change_run_identity() -> None:
    suite = _suite()
    plain = RunConfig(ModelSpec.from_dict(BASE_MODEL), suite)
    annotated = RunConfig(
        ModelSpec.from_dict({**BASE_MODEL, "notes": "hello", "baseline": "other"}), suite
    )
    assert plain.run_id == annotated.run_id


@pytest.mark.parametrize(
    "change",
    [
        {"dtype": "float16"},
        {"model_name_or_path": "haoranxu/ALMA-7B"},
        {"revision": "abc123"},
        {"prompt": "alma_chat"},
        {"trust_remote_code": True},
        {"quantization": {"method": "bitsandbytes", "load_in_4bit": True}},
        {"attn_implementation": "eager"},
    ],
)
def test_anything_that_changes_the_numbers_changes_run_identity(change: dict) -> None:
    suite = _suite()
    original = RunConfig(ModelSpec.from_dict(BASE_MODEL), suite)
    altered = RunConfig(ModelSpec.from_dict({**BASE_MODEL, **change}), suite)
    assert original.run_id != altered.run_id


@pytest.mark.parametrize(
    "decode",
    [
        {"num_beams": 1},
        {"batch_size": 8},
        {"max_new_tokens": 128},
        {"seed": 7},
        {"sort_by_length": False},
        {"length_penalty": 1.1},
    ],
)
def test_decode_changes_change_run_identity(decode: dict) -> None:
    model = ModelSpec.from_dict(BASE_MODEL)
    assert RunConfig(model, _suite()).run_id != RunConfig(model, _suite(**decode)).run_id


def test_data_changes_change_run_identity() -> None:
    model = ModelSpec.from_dict(BASE_MODEL)
    wide = SuiteSpec.from_dict({"name": "s", "data": {"directions": ["de-en", "en-de"]}})
    narrow = SuiteSpec.from_dict({"name": "s", "data": {"directions": ["de-en"]}})
    limited = SuiteSpec.from_dict({"name": "s", "data": {"directions": ["de-en"], "limit": 10}})
    ids = {RunConfig(model, s).run_id for s in (wide, narrow, limited)}
    assert len(ids) == 3


def test_output_root_does_not_change_run_identity() -> None:
    model = ModelSpec.from_dict(BASE_MODEL)
    suite = _suite()
    left = RunConfig(model, suite, output_root=Path("runs"))
    right = RunConfig(model, suite, output_root=Path("/scratch/runs"))
    assert left.run_id == right.run_id


def test_run_identity_is_stable_across_key_order() -> None:
    reordered = dict(reversed(list(BASE_MODEL.items())))
    suite = _suite()
    assert (
        RunConfig(ModelSpec.from_dict(BASE_MODEL), suite).run_id
        == RunConfig(ModelSpec.from_dict(reordered), suite).run_id
    )


def test_canonical_json_sorts_keys() -> None:
    assert canonical_json({"b": 1, "a": 2}) == '{"a":2,"b":1}'


def test_slug_carries_the_names_and_the_hash() -> None:
    config = RunConfig(ModelSpec.from_dict(BASE_MODEL), _suite())
    assert config.slug == f"alma-7b-r__suite__{config.run_id}"
    assert config.directory.name == config.slug


# --------------------------------------------------------------------------
# YAML loading
# --------------------------------------------------------------------------


def test_extends_merges_nested_mappings(tmp_path: Path) -> None:
    (tmp_path / "base.yaml").write_text(
        "name: base\nloader: hf_causal\nmodel_name_or_path: x\ndtype: bfloat16\n"
        "kwargs:\n  a: 1\n  b: 2\n",
        encoding="utf-8",
    )
    (tmp_path / "child.yaml").write_text(
        "extends: base.yaml\nname: child\ndtype: float16\nkwargs:\n  b: 3\n", encoding="utf-8"
    )
    payload = load_yaml_config(tmp_path / "child.yaml")
    assert payload["name"] == "child"
    assert payload["dtype"] == "float16"
    assert payload["model_name_or_path"] == "x"
    # Nested mappings merge rather than replace, so the parent's 'a' survives.
    assert payload["kwargs"] == {"a": 1, "b": 3}


def test_extends_detects_a_cycle(tmp_path: Path) -> None:
    (tmp_path / "a.yaml").write_text("extends: b.yaml\n", encoding="utf-8")
    (tmp_path / "b.yaml").write_text("extends: a.yaml\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="circular"):
        load_yaml_config(tmp_path / "a.yaml")


def test_missing_config_file_is_reported_clearly(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="config file not found"):
        load_yaml_config(tmp_path / "absent.yaml")


def test_non_mapping_config_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "list.yaml").write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="must be a mapping"):
        load_yaml_config(tmp_path / "list.yaml")


def test_empty_config_loads_as_an_empty_mapping(tmp_path: Path) -> None:
    (tmp_path / "empty.yaml").write_text("", encoding="utf-8")
    assert load_yaml_config(tmp_path / "empty.yaml") == {}


def test_shipped_configs_all_parse() -> None:
    root = Path(__file__).resolve().parents[1] / "configs"
    if not root.is_dir():
        pytest.skip("configs directory not present")
    for path in sorted((root / "models").glob("*.yaml")):
        ModelSpec.from_dict(load_yaml_config(path))
    for path in sorted((root / "suites").glob("*.yaml")):
        SuiteSpec.from_dict(load_yaml_config(path))
    for path in sorted((root / "metrics").glob("*.yaml")):
        MetricsSpec.from_dict(load_yaml_config(path))
