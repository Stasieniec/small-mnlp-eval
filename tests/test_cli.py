"""Argument parsing and override handling."""

from __future__ import annotations

from pathlib import Path

import pytest

from mnlp_eval.cli import (
    _apply_overrides,
    _build_bench_spec,
    _build_run_config,
    build_parser,
    main,
)
from mnlp_eval.config import ConfigError, ModelSpec


def test_parser_requires_a_subcommand(capsys: pytest.CaptureFixture) -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_every_subcommand_has_a_handler() -> None:
    parser = build_parser()
    for command, arguments in [
        ("info", []),
        ("prefetch", ["--models", "a.yaml"]),
        ("generate", ["--model", "m.yaml", "--suite", "s.yaml"]),
        ("bench", ["--run", "runs/x"]),
        ("score", []),
        ("report", []),
        ("run", ["--model", "m.yaml", "--suite", "s.yaml"]),
        ("verify-testset", []),
    ]:
        args = parser.parse_args([command, *arguments])
        assert callable(args.handler), command


def test_bench_needs_a_run_or_a_model_and_suite() -> None:
    assert main(["bench"]) == 1


# --------------------------------------------------------------------------
# Overrides
# --------------------------------------------------------------------------


def test_overrides_create_nested_paths() -> None:
    payload = _apply_overrides({}, ["suite.decode.batch_size=2"])
    assert payload == {"suite": {"decode": {"batch_size": 2}}}


def test_override_values_are_parsed_as_yaml_scalars() -> None:
    payload = _apply_overrides(
        {},
        [
            "a.int=4",
            "a.float=0.5",
            "a.bool=true",
            "a.none=null",
            "a.list=[1, 2]",
            "a.string=hello",
        ],
    )
    assert payload["a"] == {
        "int": 4,
        "float": 0.5,
        "bool": True,
        "none": None,
        "list": [1, 2],
        "string": "hello",
    }


def test_overrides_replace_a_non_mapping_on_the_path() -> None:
    payload = _apply_overrides({"a": 1}, ["a.b=2"])
    assert payload == {"a": {"b": 2}}


@pytest.mark.parametrize("override", ["novalue", "=value", ".=1"])
def test_malformed_overrides_are_rejected(override: str) -> None:
    with pytest.raises(ConfigError, match="malformed override"):
        _apply_overrides({}, [override])


def test_overrides_reach_the_bench_spec() -> None:
    args = build_parser().parse_args(
        ["bench", "--run", "x", "--set", "bench.subset_size=16", "--set", "bench.repeats=1"]
    )
    spec = _build_bench_spec(args)
    assert spec.subset_size == 16
    assert spec.repeats == 1


def test_bench_direction_flag_wins_over_the_config() -> None:
    args = build_parser().parse_args(["bench", "--run", "x", "--bench-direction", "en-is"])
    assert _build_bench_spec(args).direction == "en-is"


# --------------------------------------------------------------------------
# Run configuration assembly
# --------------------------------------------------------------------------


def _write_configs(tmp_path: Path) -> tuple[Path, Path]:
    model = tmp_path / "model.yaml"
    model.write_text(
        "name: m\nloader: hf_causal\nprompt: alma\nmodel_name_or_path: x\ndtype: bfloat16\n",
        encoding="utf-8",
    )
    suite = tmp_path / "suite.yaml"
    suite.write_text(
        "name: s\ndata:\n  directions: [de-en, en-de]\ndecode:\n  num_beams: 5\n",
        encoding="utf-8",
    )
    return model, suite


def test_run_config_is_assembled_from_two_files(tmp_path: Path) -> None:
    model, suite = _write_configs(tmp_path)
    args = build_parser().parse_args(["generate", "--model", str(model), "--suite", str(suite)])
    config = _build_run_config(args)
    assert config.model.name == "m"
    assert config.suite.data.directions == ["de-en", "en-de"]
    assert config.suite.decode.num_beams == 5


def test_shorthand_flags_and_set_both_apply(tmp_path: Path) -> None:
    model, suite = _write_configs(tmp_path)
    args = build_parser().parse_args(
        [
            "generate",
            "--model",
            str(model),
            "--suite",
            str(suite),
            "--limit",
            "16",
            "--directions",
            "is-en",
            "--set",
            "model.dtype=float16",
            "--set",
            "suite.decode.num_beams=1",
        ]
    )
    config = _build_run_config(args)
    assert config.suite.data.limit == 16
    assert config.suite.data.directions == ["is-en"]
    assert config.model.dtype == "float16"
    assert config.suite.decode.num_beams == 1


def test_an_override_changes_the_run_identity(tmp_path: Path) -> None:
    # An override is a different experiment, not a variant of the same one.
    model, suite = _write_configs(tmp_path)
    parser = build_parser()
    plain = _build_run_config(
        parser.parse_args(["generate", "--model", str(model), "--suite", str(suite)])
    )
    altered = _build_run_config(
        parser.parse_args(
            [
                "generate",
                "--model",
                str(model),
                "--suite",
                str(suite),
                "--set",
                "model.dtype=float16",
            ]
        )
    )
    assert plain.run_id != altered.run_id


def test_runs_root_is_honoured(tmp_path: Path) -> None:
    model, suite = _write_configs(tmp_path)
    args = build_parser().parse_args(
        [
            "--runs-root",
            str(tmp_path / "elsewhere"),
            "generate",
            "--model",
            str(model),
            "--suite",
            str(suite),
        ]
    )
    assert _build_run_config(args).output_root == tmp_path / "elsewhere"


# --------------------------------------------------------------------------
# Error handling
# --------------------------------------------------------------------------


def test_a_bad_config_exits_non_zero_without_a_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("name: x\nloader: hf_causal\ndtype: nonsense\n", encoding="utf-8")
    suite = tmp_path / "suite.yaml"
    suite.write_text("name: s\ndata:\n  directions: [de-en]\n", encoding="utf-8")
    assert main(["generate", "--model", str(bad), "--suite", str(suite)]) == 1
    assert "error:" in capsys.readouterr().err


def test_a_missing_config_exits_non_zero(tmp_path: Path) -> None:
    suite = tmp_path / "suite.yaml"
    suite.write_text("name: s\ndata:\n  directions: [de-en]\n", encoding="utf-8")
    assert main(["generate", "--model", str(tmp_path / "absent.yaml"), "--suite", str(suite)]) == 1


def test_score_on_an_empty_runs_root_exits_non_zero(tmp_path: Path) -> None:
    assert main(["--runs-root", str(tmp_path / "absent"), "score"]) == 1


def test_prefetch_with_nothing_to_do_exits_non_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    assert main(["prefetch", "--no-metrics"]) == 1


def test_info_runs_in_any_environment(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    assert main(["--runs-root", str(tmp_path), "info"]) == 0
    output = capsys.readouterr().out
    assert "metric groups in this environment" in output
    assert "loaders" in output


def test_metrics_overrides_apply_to_score(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from mnlp_eval.cli import _load_metrics

    monkeypatch.chdir(tmp_path)
    args = build_parser().parse_args(["score", "--groups", "neural", "--set", "comet_batch_size=8"])
    assert _load_metrics(args.metrics, args.set).comet_batch_size == 8


def test_metrics_overrides_apply_to_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from mnlp_eval.cli import _load_metrics

    monkeypatch.chdir(tmp_path)
    args = build_parser().parse_args(["report", "--set", "bootstrap_samples=200"])
    assert _load_metrics(args.metrics, args.set).bootstrap_samples == 200


def test_metrics_default_to_the_shipped_config(monkeypatch: pytest.MonkeyPatch) -> None:
    from mnlp_eval.cli import DEFAULT_METRICS_CONFIG, _load_metrics

    repository_root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(repository_root)
    if not DEFAULT_METRICS_CONFIG.is_file():
        pytest.skip("default metrics config not present")
    assert "surface" in _load_metrics(None).groups


def test_prefetch_tolerates_a_checkpoint_that_does_not_exist_yet(tmp_path: Path) -> None:
    """The shipped pruning templates carry placeholder scratch paths.

    slurm/prefetch.sh globs every config in the directory and runs with
    --strict, so treating an absent local checkpoint as a download failure
    would break the documented Snellius setup step for everyone until the last
    teammate's checkpoint landed.
    """
    from mnlp_eval.prefetch import prefetch

    spec = ModelSpec.from_dict(
        {
            "name": "not-yet",
            "loader": "hf_causal",
            "model_name_or_path": str(tmp_path / "absent"),
        }
    )

    report = prefetch([spec], [])

    entry = report["models"]["not-yet:model"]
    assert entry["ok"] is True
    assert "does not exist yet" in entry["warning"]


def test_prefetch_reports_a_present_local_checkpoint_without_a_warning(tmp_path: Path) -> None:
    from mnlp_eval.prefetch import prefetch

    checkpoint = tmp_path / "present"
    checkpoint.mkdir()
    spec = ModelSpec.from_dict(
        {"name": "here", "loader": "hf_causal", "model_name_or_path": str(checkpoint)}
    )

    report = prefetch([spec], [])

    assert report["models"]["here:model"] == {"ok": True, "note": "local directory"}
