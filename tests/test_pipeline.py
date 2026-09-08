"""The generate, score and report stages, driven by a stub model.

No network, no GPU, no model weights. This is the test that would catch a
regression in the plumbing that every teammate's model depends on.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mnlp_eval.config import MetricsSpec, ModelSpec, RunConfig, SuiteSpec
from mnlp_eval.generate import generate
from mnlp_eval.prompts import get_prompt
from mnlp_eval.report import build_report, collect_runs
from mnlp_eval.report.tables import build_tables
from mnlp_eval.runspec import RunPaths
from mnlp_eval.score import score_run
from stubs import StubTranslator

SURFACE_ONLY = MetricsSpec.from_dict({"groups": ["surface"]})


def _stub(config: RunConfig, **kwargs: object) -> StubTranslator:
    return StubTranslator(config.model, get_prompt(config.model.prompt), **kwargs)  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# Generate
# --------------------------------------------------------------------------


def test_generate_writes_a_complete_run_directory(run_config: RunConfig) -> None:
    report = generate(run_config, translator=_stub(run_config))
    paths = RunPaths.for_config(run_config)

    assert paths.manifest.is_file()
    assert paths.env.is_file()
    assert paths.hyps_jsonl("de-en").is_file()
    assert paths.hyps_text("de-en").is_file()
    assert report.directions[0].n_segments == 4
    assert paths.stage_completed("generate")

    recorded = paths.stages()["generate"]
    assert recorded["directions"]["de-en"]["n_segments"] == 4
    assert recorded["model_info"]["kind"] == "stub"
    assert recorded["directions"]["de-en"]["data"]["fingerprint"]


def test_generate_text_and_jsonl_stay_aligned(run_config: RunConfig) -> None:
    generate(run_config, translator=_stub(run_config))
    paths = RunPaths.for_config(run_config)
    lines = paths.hyps_text("de-en").read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in paths.hyps_jsonl("de-en").read_text().splitlines()]
    assert lines == [record["hypothesis"] for record in records]


def test_generate_skips_existing_directions_then_overwrites_on_request(
    run_config: RunConfig,
) -> None:
    generate(run_config, translator=_stub(run_config, template="first {source}"))
    second = generate(run_config, translator=_stub(run_config, template="second {source}"))
    assert second.skipped == ["de-en"]
    text = RunPaths.for_config(run_config).hyps_text("de-en").read_text()
    assert "first" in text

    generate(run_config, translator=_stub(run_config, template="third {source}"), overwrite=True)
    assert "third" in RunPaths.for_config(run_config).hyps_text("de-en").read_text()


def test_generate_counts_empty_output_rather_than_hiding_it(run_config: RunConfig) -> None:
    # ALMA's parser turns unrecoverable output into an empty string with no
    # trace. Here it is a counted, reported failure.
    report = generate(run_config, translator=_stub(run_config, outputs=["", "ok output"]))
    assert report.directions[0].empty == 2


def test_generate_records_trailing_commentary_as_wasted_tokens(run_config: RunConfig) -> None:
    outputs = ["The translation.\nExplanation: this German sentence means something."]
    report = generate(run_config, translator=_stub(run_config, outputs=outputs))
    direction = report.directions[0]
    assert direction.parse_flags.get("extra_lines") == 4
    assert direction.wasted_tokens > 0
    assert 0 < direction.wasted_token_fraction < 1


def test_generate_honours_the_data_limit(
    model_spec: ModelSpec, local_testset: Path, tmp_path: Path
) -> None:
    suite = SuiteSpec.from_dict(
        {
            "name": "limited",
            "data": {
                "dataset": f"local:jsonl:{local_testset}",
                "directions": ["de-en"],
                "limit": 2,
            },
            "decode": {"num_beams": 1, "batch_size": 2},
        }
    )
    config = RunConfig(model_spec, suite, output_root=tmp_path / "runs")
    report = generate(config, translator=_stub(config))
    assert report.directions[0].n_segments == 2
    assert report.directions[0].data["n_available"] == 4


def test_deterministic_check_reports_a_disagreement_rate(run_config: RunConfig) -> None:
    report = generate(run_config, translator=_stub(run_config), deterministic_check=4)
    check = report.deterministic_check
    assert check is not None
    assert check["n_compared"] == 4
    assert check["comparison_setting"] == {"batch_size": 1, "sort_by_length": False}
    # The stub is order independent, so there is nothing to disagree about.
    assert check["disagreement_rate"] == 0.0


# --------------------------------------------------------------------------
# Score
# --------------------------------------------------------------------------


def test_score_writes_surface_metrics_and_leaves_neural_pending(run_config: RunConfig) -> None:
    generate(run_config, translator=_stub(run_config))
    paths = RunPaths.for_config(run_config)
    outcome = score_run(paths, MetricsSpec.from_dict({"groups": ["surface", "neural"]}))

    assert outcome["groups"]["surface"]["status"] == "completed"
    assert paths.scores("surface").is_file()

    neural = outcome["groups"]["neural"]
    # In an environment without COMET the group is reported as pending with a
    # reason, never silently dropped and never fatal.
    if neural["status"] == "pending":
        assert "unbabel-comet" in neural["reason"]


def test_score_is_idempotent_unless_told_otherwise(run_config: RunConfig) -> None:
    generate(run_config, translator=_stub(run_config))
    paths = RunPaths.for_config(run_config)
    score_run(paths, SURFACE_ONLY)
    again = score_run(paths, SURFACE_ONLY)
    assert again["groups"]["surface"]["status"] == "present"
    forced = score_run(paths, SURFACE_ONLY, overwrite=True)
    assert forced["groups"]["surface"]["status"] == "completed"


def test_score_output_carries_provenance(run_config: RunConfig) -> None:
    generate(run_config, translator=_stub(run_config))
    paths = RunPaths.for_config(run_config)
    score_run(paths, SURFACE_ONLY)
    payload = json.loads(paths.scores("surface").read_text(encoding="utf-8"))
    assert payload["run_id"] == run_config.run_id
    assert payload["packages"]["sacrebleu"]
    assert payload["computed_at"]
    assert "tok:13a" in payload["directions"]["de-en"]["metrics"]["bleu"]["signature"]


def test_score_rejects_unknown_groups(run_config: RunConfig) -> None:
    generate(run_config, translator=_stub(run_config))
    with pytest.raises(ValueError, match="unknown metric group"):
        score_run(RunPaths.for_config(run_config), SURFACE_ONLY, groups=["magic"])


def test_macro_average_requires_every_direction(
    model_spec: ModelSpec, local_testset: Path, tmp_path: Path
) -> None:
    # A partially failed scoring pass must not produce an average that quietly
    # covers fewer directions than the table claims.
    suite = SuiteSpec.from_dict(
        {
            "name": "two-way",
            "data": {
                "dataset": f"local:jsonl:{local_testset}",
                "directions": ["de-en", "en-de"],
            },
            "decode": {"num_beams": 1, "batch_size": 2},
        }
    )
    config = RunConfig(model_spec, suite, output_root=tmp_path / "runs")
    generate(config, translator=_stub(config))
    paths = RunPaths.for_config(config)
    paths.hyps_jsonl("en-de").unlink()

    # Scoring an incomplete run is refused by default, because averaging over
    # the subset that happens to be on disk produced a macro average covering
    # fewer directions than the table claimed.
    with pytest.raises(RuntimeError, match="generation is incomplete"):
        score_run(paths, SURFACE_ONLY)

    score_run(paths, SURFACE_ONLY, allow_partial=True)
    payload = json.loads(paths.scores("surface").read_text(encoding="utf-8"))
    assert payload["aggregate"]["n_directions"] == 1
    assert set(payload["directions"]) == {"de-en"}


def test_sharded_generation_shares_one_run(
    model_spec: ModelSpec, local_testset: Path, tmp_path: Path
) -> None:
    # How a Slurm array must shard a suite. Passing --directions instead put
    # each direction in its own run id, so the six-direction macro average the
    # suite promises was unreachable through the documented submission path.
    suite = SuiteSpec.from_dict(
        {
            "name": "two-way",
            "data": {
                "dataset": f"local:jsonl:{local_testset}",
                "directions": ["de-en", "en-de"],
            },
            "decode": {"num_beams": 1, "batch_size": 2},
        }
    )
    config = RunConfig(model_spec, suite, output_root=tmp_path / "runs")

    first = generate(config, only_directions=["de-en"], translator=_stub(config))
    assert [report.direction for report in first.directions] == ["de-en"]
    paths = RunPaths.for_config(config)
    assert not paths.stage_completed("generate")

    generate(config, only_directions=["en-de"], translator=_stub(config))
    # One run directory, both directions, and the stage is complete only once
    # every shard has reported.
    assert paths.stage_completed("generate")
    assert paths.generated_directions() == {"de-en", "en-de"}
    assert len(list((tmp_path / "runs").iterdir())) == 1

    score_run(paths, SURFACE_ONLY)
    payload = json.loads(paths.scores("surface").read_text(encoding="utf-8"))
    assert payload["aggregate"]["n_directions"] == 2


def test_sharding_rejects_a_direction_outside_the_suite(run_config: RunConfig) -> None:
    with pytest.raises(ValueError, match="not in suite"):
        generate(run_config, only_directions=["ru-en"], translator=_stub(run_config))


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------


def _two_systems(local_testset: Path, tmp_path: Path, suite: SuiteSpec) -> Path:
    runs_root = tmp_path / "runs"
    for name, template in (("baseline", "good {source}"), ("compressed", "worse {source}")):
        spec = ModelSpec.from_dict(
            {
                "name": name,
                "loader": "hf_causal",
                "prompt": "alma",
                "model_name_or_path": f"stub/{name}",
                "dtype": "float32",
                "baseline": None if name == "baseline" else "baseline",
            }
        )
        config = RunConfig(spec, suite, output_root=runs_root)
        generate(config, translator=StubTranslator(spec, get_prompt("alma"), template=template))
        score_run(RunPaths.for_config(config), SURFACE_ONLY)
    return runs_root


def test_report_places_comparable_runs_in_one_table(
    local_testset: Path, tmp_path: Path, suite_spec: SuiteSpec
) -> None:
    runs_root = _two_systems(local_testset, tmp_path, suite_spec)
    result = build_report(runs_root, tmp_path / "reports", baseline="baseline", plots=False)
    assert result.groups == 1
    assert result.runs == 2
    assert not result.warnings
    text = (tmp_path / "reports" / "report.md").read_text(encoding="utf-8")
    assert "Translation quality" in text
    assert "Significance against baseline" in text


def test_report_refuses_to_mix_incomparable_runs(
    model_spec: ModelSpec, local_testset: Path, tmp_path: Path
) -> None:
    # The single easiest way for a compression comparison to become
    # meaningless, so the report says so loudly instead of averaging over it.
    runs_root = tmp_path / "runs"
    for beams in (1, 4):
        suite = SuiteSpec.from_dict(
            {
                "name": f"beam{beams}",
                "data": {"dataset": f"local:jsonl:{local_testset}", "directions": ["de-en"]},
                "decode": {"num_beams": beams, "batch_size": 2},
            }
        )
        config = RunConfig(model_spec, suite, output_root=runs_root)
        generate(config, translator=_stub(config))
        score_run(RunPaths.for_config(config), SURFACE_ONLY)

    result = build_report(runs_root, tmp_path / "reports", plots=False)
    assert result.groups == 2
    assert any("distinct measurement settings" in warning for warning in result.warnings)
    text = (tmp_path / "reports" / "report.md").read_text(encoding="utf-8")
    assert "Setting 1" in text
    assert "Setting 2" in text


def test_report_notes_what_is_missing(
    local_testset: Path, tmp_path: Path, suite_spec: SuiteSpec
) -> None:
    runs_root = _two_systems(local_testset, tmp_path, suite_spec)
    build_report(runs_root, tmp_path / "reports", baseline="baseline", plots=False)
    text = (tmp_path / "reports" / "report.md").read_text(encoding="utf-8")
    assert "Efficiency measurements are missing" in text
    assert "COMET scores are missing" in text


def test_report_handles_an_empty_runs_directory(tmp_path: Path) -> None:
    result = build_report(tmp_path / "absent", tmp_path / "reports", plots=False)
    assert result.runs == 0
    assert result.files == []
    assert any("no runs" in warning for warning in result.warnings)


def test_report_emits_every_requested_format(
    local_testset: Path, tmp_path: Path, suite_spec: SuiteSpec
) -> None:
    runs_root = _two_systems(local_testset, tmp_path, suite_spec)
    build_report(
        runs_root,
        tmp_path / "reports",
        baseline="baseline",
        formats=("md", "csv", "tex"),
        plots=False,
    )
    out = tmp_path / "reports"
    assert (out / "summary.csv").is_file()
    assert list(out.glob("*.tex"))
    assert (out / "report.json").is_file()


def test_summary_csv_has_one_row_per_run(
    local_testset: Path, tmp_path: Path, suite_spec: SuiteSpec
) -> None:
    import csv

    runs_root = _two_systems(local_testset, tmp_path, suite_spec)
    build_report(runs_root, tmp_path / "reports", baseline="baseline", plots=False)
    with (tmp_path / "reports" / "summary.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert {row["model"] for row in rows} == {"baseline", "compressed"}
    assert all(row["bleu"] for row in rows)


def test_baseline_is_ordered_first_in_tables(
    local_testset: Path, tmp_path: Path, suite_spec: SuiteSpec
) -> None:
    runs_root = _two_systems(local_testset, tmp_path, suite_spec)
    summaries = collect_runs(runs_root)
    tables = build_tables(summaries, baseline="baseline")
    assert tables[0].rows[0][0] == "baseline"


def test_latex_escapes_special_characters() -> None:
    from mnlp_eval.report.tables import Table

    table = Table(title="100% of runs", columns=["a_b"], rows=[["50%"]])
    rendered = table.to_latex()
    assert r"a\_b" in rendered
    assert r"50\%" in rendered


def test_source_length_override_is_recorded(model_spec: ModelSpec, tmp_path: Path) -> None:
    # ALMA reruns zh-en at 512 source tokens. The manifest records the cap that
    # actually applied, so the exception is visible rather than implicit.
    import json

    rows = [{"translation": {"zh": "这是一个测试。", "en": "This is a test."}}]
    path = tmp_path / "test.zh-en.jsonl"
    path.write_text(json.dumps(rows[0], ensure_ascii=False) + "\n", encoding="utf-8")

    suite = SuiteSpec.from_dict(
        {
            "name": "zh",
            "data": {"dataset": f"local:jsonl:{path}", "directions": ["zh-en"]},
            "decode": {"num_beams": 1, "max_source_length": 256, "batch_size": 1},
        }
    )
    config = RunConfig(model_spec, suite, output_root=tmp_path / "runs")
    report = generate(config, translator=_stub(config))
    assert report.directions[0].max_source_length == 512


def test_source_length_override_can_be_disabled(
    model_spec: ModelSpec, local_testset: Path, tmp_path: Path
) -> None:
    from mnlp_eval.generate import resolve_decode_for_direction
    from mnlp_eval.languages import parse_direction

    suite = SuiteSpec.from_dict(
        {
            "name": "no-override",
            "data": {"dataset": f"local:jsonl:{local_testset}", "directions": ["de-en"]},
            "decode": {"respect_alma_source_length_override": False},
        }
    )
    resolved = resolve_decode_for_direction(suite.decode, parse_direction("zh-en"))
    assert resolved.max_source_length == 256
