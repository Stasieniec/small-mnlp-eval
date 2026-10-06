"""Tests for scripts/grid_report.py, the pruning-grid aggregator.

The fixture builds a fake grid manifest and fake run directories with the same
JSON layout as real runs. COMET levels are chosen so that only correct matching
gives the expected composites: a direction model scores 0.85 on its own
direction and 0.70 elsewhere, a pair model 0.82 on its two directions and 0.71
elsewhere, every multi model 0.80, dense 0.86. BLEU is computed from the fake
hypotheses with the project's own surface scorer, so the aggregator's
hyps-versus-stored-BLEU check holds.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import shutil
import sys
import zlib
from pathlib import Path
from typing import Any

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "grid_report.py"
_spec = importlib.util.spec_from_file_location("grid_report", SCRIPT)
assert _spec is not None and _spec.loader is not None
gr = importlib.util.module_from_spec(_spec)
sys.modules["grid_report"] = gr
_spec.loader.exec_module(gr)

SUITE = "alma10-greedy-300"
N_SEG = 6
WORDS = ("alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india")


def grid_name(method: str, calib: str, sparsity: int, scope: str, key: str = "") -> str:
    suffix = "multi" if scope == "multi" else f"{scope}-{key}"
    return f"alma-7b-{method}{sparsity}-{calib}-{suffix}"


def level(name: str, direction: str) -> float:
    """The synthetic COMET of one system on one direction."""
    if name == "alma-7b":
        return 0.86
    system = gr.parse_name(name)
    assert system is not None
    if system.repaired:
        return 0.83
    shift = (0.01 if system.calib == "gen" else 0.0) - (0.02 if system.method == "flap" else 0.0)
    if system.scope == "multi":
        return 0.80 + shift
    if system.scope == "pair":
        return (0.82 if gr.pair_of(direction) == system.scope_key else 0.71) + shift
    return (0.85 if direction == system.scope_key else 0.70) + shift


def write_run(
    runs_root: Path,
    name: str,
    *,
    run_hash: str = "000000000000",
    surface: bool = True,
    neural: bool = True,
    directions: tuple[str, ...] = gr.DIRECTIONS,
) -> Path:
    from mnlp_eval.metrics.surface import score_surface

    slug = f"{name}__{SUITE}__{run_hash}"
    root = runs_root / slug
    (root / "hyps").mkdir(parents=True)
    (root / "stages").mkdir()
    lora = name.endswith("-lora")
    manifest = {
        "run_id": run_hash,
        "slug": slug,
        "created_at": "2026-10-06T00:00:00+00:00",
        "model": {
            "name": name,
            "compression": {
                "family": "pruning" if name != "alma-7b" else "none",
                "repair": "LoRA" if lora else "",
            },
            "kwargs": {},
        },
        "suite": {"name": SUITE, "data": {"directions": list(gr.DIRECTIONS), "limit": 300}},
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    rng = np.random.default_rng(zlib.crc32(f"{name}{run_hash}".encode()))
    surface_dirs: dict[str, Any] = {}
    neural_dirs: dict[str, Any] = {}
    for direction in directions:
        value = level(name, direction)
        keep = max(1, min(12, round((value - 0.6) * 40)))
        references = [f"seg{i} {direction} " + " ".join(WORDS) for i in range(N_SEG)]
        hypotheses = [" ".join(ref.split()[:keep]) for ref in references]
        records = [
            {"index": i, "source": f"src {i}", "reference": ref, "hypothesis": hyp}
            for i, (ref, hyp) in enumerate(zip(references, hypotheses, strict=True))
        ]
        (root / "hyps" / f"{direction}.jsonl").write_text(
            "\n".join(json.dumps(r) for r in records) + "\n"
        )
        scores = score_surface(hypotheses, references, direction.split("-")[1])
        surface_dirs[direction] = {
            "n_segments": N_SEG,
            "metrics": {key: score.to_dict() for key, score in scores.items()},
            "behaviour": {
                "off_target_rate": round(1 - value, 4),
                "repetition_rate": 0.0,
                "budget_hit_rate": 0.0,
                "truncation_rate": 0.0,
                "empty_rate": 0.0,
                "length_ratio": round(keep / 13, 4),
            },
        }
        noise = rng.normal(0, 0.03, N_SEG)
        segments = [float(v) for v in value + noise - noise.mean()]
        neural_dirs[direction] = {
            "n_segments": N_SEG,
            "metrics": {
                gr.COMET_KEY: {
                    "score": round(float(np.mean(segments)), 6),
                    "segment_scores": segments,
                    "higher_is_better": True,
                    "signature": "fake",
                    "extra": {},
                }
            },
        }
    if surface:
        (root / "scores.surface.json").write_text(
            json.dumps({"group": "surface", "directions": surface_dirs, "aggregate": {}})
        )
    if neural:
        (root / "scores.neural.json").write_text(
            json.dumps({"group": "neural", "directions": neural_dirs, "aggregate": {}})
        )
    stage = {
        "status": "completed",
        "directions": {d: {"data": {"fingerprint": f"fp-{d}"}} for d in directions},
    }
    (root / "stages" / "generate.json").write_text(json.dumps(stage))
    return root


def manifest_entry(method: str, calib: str, sparsity: int, scope: str, key: str = "") -> dict:
    return {
        "name": grid_name(method, calib, sparsity, scope, key),
        "method": method,
        "allocation": "global",
        "calibration_text": "prompt+target" if calib == "ref" else "prompt+generated",
        "tag": calib,
        "sparsity": sparsity / 100,
        "scope": scope,
        "scope_key": key or None,
        "pruned_directions": list(gr.scope_directions(scope, key)),
        "prune_config": "x.yaml",
        "model_config": "y.yaml",
        "suite": f"configs/suites/{SUITE}.yaml",
        "priority": 0,
    }


def full_config(method: str, calib: str, sparsity: int) -> list[dict]:
    entries = [manifest_entry(method, calib, sparsity, "multi")]
    entries += [manifest_entry(method, calib, sparsity, "pair", lang) for lang in gr.PAIR_LANGS]
    entries += [manifest_entry(method, calib, sparsity, "dir", d) for d in gr.DIRECTIONS]
    return entries


@pytest.fixture(scope="module")
def template(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A grid of 35 manifest entries, of which 33 have complete runs."""
    root = tmp_path_factory.mktemp("grid")
    entries = full_config("slimgpt", "ref", 20) + full_config("slimgpt", "gen", 20)
    entries += [
        manifest_entry("flap", "ref", 20, "multi"),
        manifest_entry("flap", "ref", 30, "multi"),
    ]
    entries += [manifest_entry("flap", "gen", 40, "dir", "en-is")]
    (root / "manifest.json").write_text(json.dumps(entries))
    runs = root / "runs"
    for entry in entries:
        if entry["name"] in (
            grid_name("flap", "ref", 30, "multi"),
            grid_name("flap", "gen", 40, "dir", "en-is"),
        ):
            continue
        write_run(runs, entry["name"])
    write_run(runs, "alma-7b")
    return root


@pytest.fixture
def grid(template: Path, tmp_path: Path) -> Path:
    target = tmp_path / "grid"
    shutil.copytree(template, target)
    return target


def run_report(grid: Path, *extra: str, bootstrap: int = 50) -> tuple[Path, dict[str, Any]]:
    out = grid / "out"
    argv = [
        "--out",
        str(out),
        "--runs-root",
        str(grid / "runs"),
        "--manifest",
        str(grid / "manifest.json"),
        "--bootstrap",
        str(bootstrap),
        "--status-dir",
        str(grid / "status"),
        "--checkpoints-root",
        str(grid / "checkpoints"),
        *extra,
    ]
    assert gr.main(argv) == 0
    return out, json.loads((out / "results.json").read_text())


def headline_row(results: dict, method: str, calib: str, sparsity: int) -> dict:
    for row in results["headline"]:
        if (row["method"], row["calib"], row["sparsity"]) == (method, calib, sparsity):
            return row
    raise AssertionError(f"no headline row for {method} {calib} {sparsity}")


# --------------------------------------------------------------------------


def test_parse_name() -> None:
    system = gr.parse_name("alma-7b-slimgpt30-gen-dir-en-is")
    assert system is not None
    assert system.cell == ("slimgpt", "gen", 30, "dir", "en-is")
    assert system.pruned_directions == ("en-is",)
    pair = gr.parse_name("alma-7b-slimgpt40-ref-pair-zh")
    assert pair is not None and pair.pruned_directions == ("zh-en", "en-zh")
    lora = gr.parse_name("alma-7b-flap20-ref-multi-lora")
    assert lora is not None and lora.repaired and lora.source == "alma-7b-flap20-ref-multi"
    assert gr.parse_name("alma-7b-slimgpt20-multi-target") is None


def test_manifest_entry_normalisation() -> None:
    entry = manifest_entry("slimgpt", "gen", 30, "pair", "zh")
    entry["scope_key"] = "pair-zh"
    entry["tag"] = None
    system = gr.system_from_entry(entry)
    assert system is not None
    assert system.cell == ("slimgpt", "gen", 30, "pair", "zh")
    assert system.in_scope("en-zh") and not system.in_scope("en-de")


@pytest.mark.parametrize("target", ["en", "zh"])
def test_bleu_from_stats_matches_sacrebleu(target: str) -> None:
    from sacrebleu.metrics import BLEU

    rng = np.random.default_rng(3)
    vocabulary = ["the", "cat", "sat", "on", "a", "mat", "dog", "ran", "home", "fast"]
    references = [" ".join(rng.choice(vocabulary, size=rng.integers(3, 12))) for _ in range(40)]
    hypotheses = [" ".join(rng.choice(vocabulary, size=rng.integers(0, 12))) for _ in range(40)]
    hypotheses[0] = ""
    metric = BLEU(tokenize="zh" if target == "zh" else "13a")
    stats = np.asarray(metric._extract_corpus_statistics(hypotheses, [references]))
    expected = metric.corpus_score(hypotheses, [references]).score
    assert float(gr.bleu_from_stats(stats.sum(axis=0))) == pytest.approx(expected, abs=1e-9)
    # Short output with no 4-gram matches still follows sacreBLEU's exp smoothing.
    short = ["cat", "the mat", "dog ran"]
    short_stats = np.asarray(metric._extract_corpus_statistics(short, [references[:3]]))
    assert float(gr.bleu_from_stats(short_stats.sum(axis=0))) == pytest.approx(
        metric.corpus_score(short, [references[:3]]).score, abs=1e-9
    )
    assert float(gr.bleu_from_stats(np.zeros(10))) == 0.0


def test_headline_matches_specialists_to_directions(grid: Path) -> None:
    out, results = run_report(grid, "--no-plots")
    row = headline_row(results, "slimgpt", "ref", 20)
    assert row["multi"]["comet"] == pytest.approx(0.80, abs=1e-6)
    # Only correct matching gives these: own-direction levels everywhere.
    assert row["pair"]["comet"] == pytest.approx(0.82, abs=1e-6)
    assert row["dir"]["comet"] == pytest.approx(0.85, abs=1e-6)
    assert row["dir"]["n_directions"]["comet"] == 10

    delta = row["dir_minus_multi_comet"]
    assert delta["delta"] == pytest.approx(0.05, abs=1e-6)
    assert delta["n_directions"] == 10
    low, high = delta["bootstrap_ci_95"]
    assert low <= 0.05 <= high
    assert delta["p_value"] < 0.05
    assert row["pair_minus_multi_comet"]["delta"] == pytest.approx(0.02, abs=1e-6)

    gen = headline_row(results, "slimgpt", "gen", 20)
    assert gen["dir"]["comet"] == pytest.approx(0.86, abs=1e-6)
    flap = headline_row(results, "flap", "ref", 20)
    assert flap["multi"]["comet"] == pytest.approx(0.78, abs=1e-6)
    assert flap["pair"]["comet"] is None and flap["dir_minus_multi_comet"] is None

    summary = (out / "summary.md").read_text()
    assert "| SlimGPT | ref | 20% | 0.8000 |" in summary
    assert "**0.8500**" in summary  # dir is the best complete scope

    transfer = results["transfer"]["slimgpt20-ref"]["dir_summary"]
    assert transfer["own"]["comet"] == pytest.approx(0.85, abs=1e-6)
    assert transfer["own"]["minus_multi"] == pytest.approx(0.05, abs=1e-6)
    assert transfer["reverse"]["comet"] == pytest.approx(0.70, abs=1e-6)
    assert transfer["same_target"]["n_cells"] == 5 * 4  # the five into-English models
    assert transfer["same_source"]["n_cells"] == 5 * 4  # the five out-of-English models
    assert transfer["other"]["n_cells"] == 10 * 4
    pair = results["transfer"]["slimgpt20-ref"]["pair_summary"]
    assert pair["own"]["comet"] == pytest.approx(0.82, abs=1e-6)
    assert pair["other"]["minus_multi"] == pytest.approx(-0.09, abs=1e-6)
    assert (out / "transfer" / "slimgpt20-ref.md").is_file()
    assert (out / "transfer" / "slimgpt20-ref.csv").is_file()

    contrast = next(
        r
        for r in results["ref_vs_gen"]
        if r["baseline"] == {"method": "slimgpt", "calib": "ref", "sparsity": 20, "scope": "dir"}
    )
    assert contrast["comet"]["delta"] == pytest.approx(0.01, abs=1e-6)
    assert contrast["bleu"]["bootstrap_ci_95"] is not None
    assert results["coverage"]["complete"] == 33
    assert results["coverage"]["expected"] == 35


def test_partial_grid(grid: Path) -> None:
    runs = grid / "runs"
    # Three dir models gone, one pair model generated but not COMET-scored,
    # one dir model scored on only part of the directions.
    for d in ("cs-en", "en-cs", "zh-en"):
        shutil.rmtree(runs / f"{grid_name('slimgpt', 'ref', 20, 'dir', d)}__{SUITE}__000000000000")
    (
        runs
        / f"{grid_name('slimgpt', 'ref', 20, 'pair', 'de')}__{SUITE}__000000000000"
        / "scores.neural.json"
    ).unlink()
    name = grid_name("slimgpt", "gen", 20, "dir", "en-de")
    shutil.rmtree(runs / f"{name}__{SUITE}__000000000000")
    write_run(runs, name, directions=("en-de", "de-en"))
    # A second, incomplete run of a model that already has a complete one.
    write_run(runs, grid_name("flap", "ref", 20, "multi"), run_hash="111111111111", neural=False)

    out, results = run_report(grid)
    coverage = results["coverage"]
    assert coverage["missing"] == 2 + 3
    assert coverage["partial"] == 2
    assert coverage["complete"] == 33 - 3 - 2

    row = headline_row(results, "slimgpt", "ref", 20)
    assert row["dir"]["n_directions"]["comet"] == 7
    assert row["dir"]["comet"] == pytest.approx(0.85, abs=1e-6)
    assert row["pair"]["n_directions"]["comet"] == 8
    assert row["pair"]["n_directions"]["bleu"] == 10
    assert row["dir_minus_multi_comet"]["n_directions"] == 7

    summary = (out / "summary.md").read_text()
    assert "(7/10)" in summary
    assert "| SlimGPT | ref | 30% | - |" in summary
    assert "0/1" in summary  # flap ref 30 multi has no run
    assert any("2 run directories" in w for w in results["warnings"])
    assert results["runs"][grid_name("flap", "ref", 20, "multi")].endswith("000000000000")

    with (out / "long.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    systems = {r["system"] for r in rows}
    assert grid_name("slimgpt", "ref", 20, "dir", "cs-en") not in systems
    partial = [r for r in rows if r["system"] == name]
    assert len(partial) == 10
    assert sum(1 for r in partial if r["comet"]) == 2


def test_bootstrap_wiring(grid: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[int, int, int]] = []
    real = gr.bootstrap_segment_delta

    def spy(baseline: Any, system: Any, **kwargs: Any) -> dict[str, Any]:
        calls.append((len(baseline), len(system), kwargs["n_samples"]))
        return real(baseline, system, **kwargs)

    monkeypatch.setattr(gr, "bootstrap_segment_delta", spy)
    _, results = run_report(grid, "--no-plots", "--no-structure", bootstrap=7)
    assert calls
    assert all(n == 7 for _, _, n in calls)
    assert all(a == b for a, b, _ in calls)
    assert (10 * N_SEG, 10 * N_SEG, 7) in calls
    row = headline_row(results, "slimgpt", "ref", 20)
    assert row["dir_minus_multi_comet"]["n_samples"] == 7

    # A rerun on unchanged inputs takes every bootstrap from the cache.
    calls.clear()
    run_report(grid, "--no-plots", "--no-structure", bootstrap=7)
    assert calls == []

    # --bootstrap 0 skips resampling but still reports the deltas.
    _, results = run_report(grid, "--no-plots", "--no-structure", "--no-cache", bootstrap=0)
    assert calls == []
    delta = headline_row(results, "slimgpt", "ref", 20)["dir_minus_multi_comet"]
    assert delta["p_value"] is None and delta["bootstrap_ci_95"] is None
    assert delta["delta"] == pytest.approx(0.05, abs=1e-6)


def test_long_csv_columns(grid: Path) -> None:
    out, _ = run_report(grid, "--no-plots", bootstrap=0)
    with (out / "long.csv").open() as handle:
        reader = csv.DictReader(handle)
        assert tuple(reader.fieldnames or ()) == gr.LONG_COLUMNS
        rows = list(reader)
    assert len(rows) == 34 * 10  # 33 grid runs plus dense
    by_key = {(r["system"], r["direction"]): r for r in rows}

    pair = grid_name("slimgpt", "ref", 20, "pair", "cs")
    flags = {d: by_key[(pair, d)]["in_scope"] for d in gr.DIRECTIONS}
    assert {d for d, flag in flags.items() if flag == "True"} == {"cs-en", "en-cs"}
    assert by_key[(pair, "cs-en")]["scope_key"] == "cs"
    assert by_key[(pair, "cs-en")]["pruned_directions"] == "cs-en+en-cs"

    multi = grid_name("slimgpt", "ref", 20, "multi")
    assert all(by_key[(multi, d)]["in_scope"] == "True" for d in gr.DIRECTIONS)

    own = by_key[(grid_name("slimgpt", "ref", 20, "dir", "en-is"), "en-is")]
    assert float(own["comet_minus_multi"]) == pytest.approx(0.05, abs=1e-6)
    assert float(own["comet_minus_dense"]) == pytest.approx(-0.01, abs=1e-6)
    assert own["sparsity"] == "20" and own["calib"] == "ref" and own["method"] == "slimgpt"
    assert own["into_english"] == "False" and own["repaired"] == "False"
    assert float(own["off_target_rate"]) == pytest.approx(0.15)
    dense = by_key[("alma-7b", "de-en")]
    assert dense["method"] == "dense" and dense["into_english"] == "True"


def test_repair_section(grid: Path) -> None:
    source = grid_name("slimgpt", "ref", 20, "multi")
    write_run(grid / "runs", f"{source}-lora")
    out, results = run_report(grid, "--no-plots", bootstrap=20)
    (record,) = results["repair"]
    assert record["source"] == source
    comet = record["metrics"]["comet"]
    assert comet["repaired"] == pytest.approx(0.83, abs=1e-6)
    assert comet["recovered"] == pytest.approx(0.5, abs=1e-6)
    assert record["comet_repaired_minus_pruned"]["delta"] == pytest.approx(0.03, abs=1e-6)
    summary = (out / "summary.md").read_text()
    assert f"{source}-lora" in summary and "50.0%" in summary
    with (out / "long.csv").open() as handle:
        lora_rows = [r for r in csv.DictReader(handle) if r["system"] == f"{source}-lora"]
    assert len(lora_rows) == 10 and lora_rows[0]["repaired"] == "True"
    assert lora_rows[0]["source_system"] == source


def test_missing_manifest_infers_from_names(grid: Path) -> None:
    (grid / "manifest.json").unlink()
    _, results = run_report(grid, "--no-plots", bootstrap=0)
    assert not results["coverage"]["manifest_found"]
    assert results["coverage"]["complete"] == 33
    assert headline_row(results, "slimgpt", "ref", 20)["dir"]["comet"] == pytest.approx(0.85)
    assert any("not found" in w for w in results["warnings"])


def test_structure_from_subnetwork_and_prune_manifest(grid: Path) -> None:
    multi = grid_name("slimgpt", "ref", 20, "multi")
    folder = grid / "checkpoints" / multi
    folder.mkdir(parents=True)
    heads = {str(layer): list(range(32 if layer < 2 else 24)) for layer in range(32)}
    ffn = {str(layer): list(range(11008 if layer < 2 else 8000)) for layer in range(32)}
    descriptor = {
        "name": multi,
        "schema_version": 1,
        "components": {
            "attention_heads": {"total": 32, "kept": heads},
            "ffn_channels": {"total": 11008, "kept": ffn},
        },
    }
    (folder / "subnetwork.json").write_text(json.dumps(descriptor))
    pair = grid_name("slimgpt", "ref", 20, "pair", "de")
    (grid / "status").mkdir()
    (grid / "status" / f"{pair}.prune.json").write_text(
        json.dumps(
            {
                "unit_sparsity": 0.2,
                "calibration_segments": 640,
                "components": {
                    "attention_heads": {"layers": 32, "total_per_layer": 32, "kept": 820},
                    "ffn_channels": {"layers": 32, "total_per_layer": 11008, "kept": 281805},
                },
            }
        )
    )
    (grid / "status" / f"{pair}.json").write_text(json.dumps({"state": "done"}))
    out, results = run_report(grid, bootstrap=0)

    info = results["structure"][multi]
    per_head, per_channel = 4 * 4096 * 128, 3 * 4096
    removed = 30 * 8 * per_head + 30 * 3008 * per_channel
    total = 32 * (32 * per_head + 11008 * per_channel + 2 * 4096) + 2 * 32000 * 4096 + 4096
    assert info["params_removed_frac_all"] == pytest.approx(removed / total)
    assert info["heads_removed_frac"] == pytest.approx(30 * 8 / (32 * 32))
    assert results["structure"][pair]["calibration_segments"] == 640
    assert results["coverage"]["status_counts"] == {"done": 1}
    layers = (out / "structure" / f"{multi}.layers.csv").read_text().splitlines()
    assert layers[1] == "0,32,32,11008,11008" and layers[3] == "2,24,32,8000,11008"
    summary = (out / "summary.md").read_text()
    assert "24 / 24.5 / 32" in summary

    plots = {Path(p).name for p in results["plots"]}
    assert {"comet_vs_sparsity.png", "ref_vs_gen.png", "structure_multi.png"} <= plots
    assert {"transfer_dir_s20.png", "transfer_pair_s20.png"} <= plots
    assert all((out / "plots" / name).stat().st_size > 0 for name in plots)
