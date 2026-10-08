"""Tests for scripts/grid_report.py, the pruning-grid aggregator.

The fixture builds a fake grid manifest and fake run directories with the same
JSON layout as real runs. COMET levels are chosen so that only correct matching
gives the expected composites: a direction model scores 0.85 on its own
direction and 0.70 elsewhere, a pair model 0.82 on its two directions and 0.71
elsewhere, every multi model 0.80, dense 0.86. BLEU is computed from the fake
hypotheses with the project's own surface scorer, so the aggregator's
hyps-versus-stored-BLEU check holds. MetricX-24, where a run has it, is
20 * (1 - COMET level): an error score, lower is better (multi 4.0, dense 2.8).

Two grids: a pilot-style one where every model has all ten directions, and a
full-suite one where specialists are evaluated only on their own directions.
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
FULL_SUITE = "alma10-greedy"
N_SEG = 6
#: Full-suite segment counts differ by direction (the real suite has 1,000 to
#: 2,037), so a mean pooled over segments is not the macro over directions.
FULL_SIZES = {d: N_SEG + 2 * i for i, d in enumerate(gr.DIRECTIONS)}
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


def metricx_level(name: str, direction: str) -> float:
    """The synthetic MetricX-24: an error score that falls as COMET rises."""
    return round(20 * (1 - level(name, direction)), 6)


def write_metricx(
    root: Path, name: str, directions: tuple[str, ...], sizes: dict[str, int] | None = None
) -> None:
    """A scores.metricx.json in the layout ``mnlp-eval score --groups metricx`` writes."""
    rng = np.random.default_rng(zlib.crc32(f"metricx {name}".encode()))
    per_direction = {}
    for direction in directions:
        n_seg = (sizes or {}).get(direction, N_SEG)
        noise = rng.normal(0, 0.5, n_seg)
        segments = [float(v) for v in metricx_level(name, direction) + noise - noise.mean()]
        per_direction[direction] = {
            "n_segments": n_seg,
            "metrics": {
                gr.METRICX_KEY: {
                    "score": round(float(np.mean(segments)), 6),
                    "higher_is_better": False,
                    "signature": "google/metricx-24-hybrid-large-v2p6|tok:google/mt5-xl",
                    "segment_scores": segments,
                    "extra": {"reference_free": False, "range": "[0, 25], lower is better"},
                }
            },
        }
    payload = {"group": "metricx", "directions": per_direction, "aggregate": {}}
    (root / "scores.metricx.json").write_text(json.dumps(payload))


def write_run(
    runs_root: Path,
    name: str,
    *,
    run_hash: str = "000000000000",
    surface: bool = True,
    neural: bool = True,
    metricx: bool = False,
    directions: tuple[str, ...] = gr.DIRECTIONS,
    suite: str = SUITE,
    sizes: dict[str, int] | None = None,
) -> Path:
    from mnlp_eval.metrics.surface import score_surface

    slug = f"{name}__{suite}__{run_hash}"
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
        "suite": {"name": suite, "data": {"directions": list(gr.DIRECTIONS), "limit": 300}},
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    rng = np.random.default_rng(zlib.crc32(f"{name}{run_hash}".encode()))
    surface_dirs: dict[str, Any] = {}
    neural_dirs: dict[str, Any] = {}
    for direction in directions:
        value = level(name, direction)
        n_seg = (sizes or {}).get(direction, N_SEG)
        keep = max(1, min(12, round((value - 0.6) * 40)))
        references = [f"seg{i} {direction} " + " ".join(WORDS) for i in range(n_seg)]
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
            "n_segments": n_seg,
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
        noise = rng.normal(0, 0.03, n_seg)
        segments = [float(v) for v in value + noise - noise.mean()]
        neural_dirs[direction] = {
            "n_segments": n_seg,
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
    if metricx:
        write_metricx(root, name, directions, sizes)
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


@pytest.fixture(scope="module")
def full_template(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A full-suite grid of 33 manifest entries: specialists on their own directions.

    SlimGPT gen 40% (MetricX on every run): multi on all ten directions, pair
    models on their own pair, dir models on their own direction, except
    dir-en-is, which (like the Tier-3 runs) has all ten. SlimGPT ref 40% multi
    on all ten, with MetricX. FLAP ref 40% (no MetricX): multi on all ten, four
    pair models on their own pair, pair-cs on cs-en only, no dir runs. Dense on
    all ten with MetricX, and two repairs with MetricX: the gen multi model's
    on all ten, pair-is's on its own pair. The manifest names the pilot suite,
    as the real one does; the report is run with ``--suite alma10-greedy``.
    Directions have FULL_SIZES segments, 6 to 24.
    """
    root = tmp_path_factory.mktemp("full")
    entries = full_config("slimgpt", "gen", 40) + full_config("flap", "ref", 40)
    entries.append(manifest_entry("slimgpt", "ref", 40, "multi"))
    (root / "manifest.json").write_text(json.dumps(entries))
    runs = root / "runs"
    tier3 = grid_name("slimgpt", "gen", 40, "dir", "en-is")
    for entry in entries:
        system = gr.system_from_entry(entry)
        assert system is not None
        if system.method == "flap" and system.scope == "dir":
            continue
        directions = system.pruned_directions
        if system.scope == "multi" or system.name == tier3:
            directions = gr.DIRECTIONS
        if system.name == grid_name("flap", "ref", 40, "pair", "cs"):
            directions = ("cs-en",)
        write_run(
            runs,
            system.name,
            suite=FULL_SUITE,
            directions=directions,
            metricx=system.method == "slimgpt",
            sizes=FULL_SIZES,
        )
    write_run(runs, "alma-7b", suite=FULL_SUITE, metricx=True, sizes=FULL_SIZES)
    write_run(
        runs,
        f"{grid_name('slimgpt', 'gen', 40, 'multi')}-lora",
        suite=FULL_SUITE,
        metricx=True,
        sizes=FULL_SIZES,
    )
    write_run(
        runs,
        f"{grid_name('slimgpt', 'gen', 40, 'pair', 'is')}-lora",
        suite=FULL_SUITE,
        directions=("is-en", "en-is"),
        metricx=True,
        sizes=FULL_SIZES,
    )
    return root


@pytest.fixture
def full_grid(full_template: Path, tmp_path: Path) -> Path:
    target = tmp_path / "full"
    shutil.copytree(full_template, target)
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
    # The pair model without COMET is partial. The dir model scored on its own
    # direction (en-de) and one other is complete under the in-scope rule, but
    # has too few directions to enter the transfer section.
    assert coverage["partial"] == 1
    assert coverage["partial_systems"] == {grid_name("slimgpt", "ref", 20, "pair", "de"): 0}
    assert coverage["partial_expected"] == {grid_name("slimgpt", "ref", 20, "pair", "de"): 2}
    assert coverage["complete"] == 33 - 3 - 1
    assert results["transfer"]["slimgpt20-gen"]["dir_models"] == 9
    assert results["transfer"]["slimgpt20-ref"]["dir_models"] == 7

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


def spy_on_bootstrap(
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[tuple[int, ...], int, bool]]:
    """Record (per-direction sizes, n_samples, higher_is_better) per segment bootstrap."""
    calls: list[tuple[tuple[int, ...], int, bool]] = []
    real = gr.macro_segment_bootstrap

    def spy(pairs: Any, **kwargs: Any) -> dict[str, Any]:
        assert all(a.shape == b.shape for a, b in pairs)  # paired within each direction
        sizes = tuple(a.shape[0] for a, _ in pairs)
        calls.append((sizes, kwargs["n_samples"], kwargs.get("higher_is_better", True)))
        return real(pairs, **kwargs)

    monkeypatch.setattr(gr, "macro_segment_bootstrap", spy)
    return calls


def test_bootstrap_wiring(grid: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = spy_on_bootstrap(monkeypatch)
    _, results = run_report(grid, "--no-plots", "--no-structure", bootstrap=7)
    assert calls
    assert all(n == 7 for _, n, _ in calls)
    assert ((N_SEG,) * 10, 7, True) in calls
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


# --------------------------------------------------------------------------
# Full suite: specialists on their own directions, MetricX-24


def summary_line(summary: str, prefix: str) -> str:
    (line,) = [line for line in summary.splitlines() if line.startswith(prefix)]
    return line


def test_pilot_without_metricx(grid: Path) -> None:
    """No run has MetricX: '-' everywhere, one coverage line, no warnings."""
    out, results = run_report(grid, "--no-plots", bootstrap=0)
    assert results["coverage"]["metricx"] == {
        "grid_models": 0,
        "grid_models_all_in_scope": 0,
        "lora_systems": 0,
        "dense": False,
    }
    assert results["coverage"]["specialists_all_directions"] == {"pair": 10, "dir": 20}
    assert results["transfer_models"]["dir"] == 20
    assert results["dense"]["metricx"] is None
    row = headline_row(results, "slimgpt", "ref", 20)
    assert row["dir"]["metricx"] is None and row["dir_minus_multi_metricx"] is None
    assert row["dir"]["comet"] == pytest.approx(0.85, abs=1e-6)
    assert not any("metricx" in w.lower() for w in results["warnings"])
    summary = (out / "summary.md").read_text()
    assert "No MetricX-24 scores (`scores.metricx.json`) for suite" in summary
    assert "MetricX-24 (`scores.metricx.json`): 0/35 grid models" in summary
    assert "| SlimGPT | 20% | dir | 0.8500 | 0.8600 |" in summary
    assert summary_line(summary, "| SlimGPT | 20% | dir |").endswith("| - | - | - | - |")
    with (out / "long.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert all(
        r["metricx"] == r["metricx_minus_dense"] == r["metricx_minus_multi"] == "" for r in rows
    )


def test_full_suite_completeness_and_coverage(full_grid: Path) -> None:
    out, results = run_report(full_grid, "--suite", FULL_SUITE, "--no-plots", bootstrap=0)
    coverage = results["coverage"]
    assert coverage["expected"] == 33
    assert coverage["complete"] == 16 + 1 + 1 + 4
    assert coverage["partial"] == 1
    assert coverage["partial_systems"] == {grid_name("flap", "ref", 40, "pair", "cs"): 1}
    assert coverage["missing"] == 10
    assert coverage["lora_systems"] == 2 and coverage["lora_complete"] == 2
    assert coverage["specialists_all_directions"] == {"pair": 0, "dir": 1}
    assert coverage["metricx"] == {
        "grid_models": 17,
        "grid_models_all_in_scope": 17,
        "lora_systems": 2,
        "dense": True,
    }
    # A missing scores.metricx.json is not worth a warning.
    assert results["warnings"] == []

    summary = (out / "summary.md").read_text()
    assert "Grid models complete: **22/33**" in summary
    assert "the model's own pruned directions for pair and dir models" in summary
    assert "| SlimGPT gen 40% | 1/1 | 5/5 | 10/10 |" in summary
    assert "| FLAP ref 40% | 1/1 | 4/5 (+1 partial) | 0/10 |" in summary
    assert f"{grid_name('flap', 'ref', 40, 'pair', 'cs')} (1/2)" in summary
    assert "0 pair and 1 dir models, of 20 specialists with a run" in summary
    assert "MetricX-24 (`scores.metricx.json`): 17/33 grid models (17 on every" in summary
    assert "MetricX-24 2.800." in summary  # dense line in the coverage section

    with (out / "long.csv").open() as handle:
        rows = {(r["system"], r["direction"]): r for r in csv.DictReader(handle)}
    pair = grid_name("slimgpt", "gen", 40, "pair", "de")
    assert rows[(pair, "en-de")]["comet"] and not rows[(pair, "en-cs")]["comet"]
    assert rows[(pair, "en-cs")]["in_scope"] == "False"
    own = rows[(grid_name("slimgpt", "gen", 40, "dir", "en-is"), "en-is")]
    assert float(own["metricx"]) == pytest.approx(2.8, abs=1e-6)
    assert float(own["metricx_minus_multi"]) == pytest.approx(-1.0, abs=1e-6)
    assert float(own["metricx_minus_dense"]) == pytest.approx(0.0, abs=1e-6)
    multi = rows[(grid_name("slimgpt", "gen", 40, "multi"), "zh-en")]
    assert float(multi["metricx_minus_dense"]) == pytest.approx(1.0, abs=1e-6)
    assert rows[(grid_name("flap", "ref", 40, "multi"), "zh-en")]["metricx"] == ""


def test_full_suite_headlines(full_grid: Path) -> None:
    out, results = run_report(full_grid, "--suite", FULL_SUITE, "--no-plots", bootstrap=50)
    row = headline_row(results, "slimgpt", "gen", 40)
    for scope, comet, metricx in (("multi", 0.81, 3.8), ("pair", 0.83, 3.4), ("dir", 0.86, 2.8)):
        assert row[scope]["comet"] == pytest.approx(comet, abs=1e-6)
        assert row[scope]["metricx"] == pytest.approx(metricx, abs=1e-6)
        assert row[scope]["n_directions"]["comet"] == 10
        assert row[scope]["n_directions"]["metricx"] == 10
    # Specialist minus multi on an error score: negative = the specialist is better.
    delta = row["dir_minus_multi_metricx"]
    assert delta["metric"] == "metricx" and delta["higher_is_better"] is False
    assert delta["delta"] == pytest.approx(-1.0, abs=1e-6)
    assert delta["n_directions"] == 10 and delta["n_samples"] == 50
    low, high = delta["bootstrap_ci_95"]
    assert low <= -1.0 <= high < 0
    assert delta["p_value"] < 0.05
    assert row["pair_minus_multi_metricx"]["delta"] == pytest.approx(-0.4, abs=1e-6)
    assert row["dir_minus_multi_comet"]["delta"] == pytest.approx(0.05, abs=1e-6)

    flap = headline_row(results, "flap", "ref", 40)
    assert flap["pair"]["n_directions"]["comet"] == 9
    assert flap["dir"]["comet"] is None and flap["multi"]["metricx"] is None
    assert flap["dir_minus_multi_metricx"] is None

    summary = (out / "summary.md").read_text()
    comet_table, metricx_table = summary.split("## Headline, MetricX-24 (lower is better)")
    metricx_table = metricx_table.split("\n## ")[0]
    # COMET: the highest complete scope is bold; MetricX: the lowest.
    assert "| SlimGPT | gen | 40% | 0.8100 |" in comet_table
    assert "**0.8600**" in comet_table
    line = summary_line(metricx_table, "| SlimGPT | gen | 40% |")
    assert line.startswith("| SlimGPT | gen | 40% | 3.800 | 3.400 | **2.800** | -0.400 [")
    assert "| -1.000 [" in line
    assert "**3.800**" not in metricx_table
    assert "negative delta means the specialist is" in metricx_table
    assert "| dense | - | 0% | 2.800 | 2.800 | 2.800 | - | - |" in metricx_table
    assert "| FLAP | ref | 40% | - | - | - | - | - |" in metricx_table

    # Contrast sections carry MetricX next to COMET; gen - ref < 0 = gen better.
    contrast = next(
        r
        for r in results["ref_vs_gen"]
        if r["baseline"] == {"method": "slimgpt", "calib": "ref", "sparsity": 40, "scope": "multi"}
    )
    assert contrast["comet"]["delta"] == pytest.approx(0.01, abs=1e-6)
    assert contrast["metricx"]["delta"] == pytest.approx(-0.2, abs=1e-6)
    assert contrast["metricx"]["bootstrap_ci_95"] is not None
    method = next(
        r
        for r in results["slimgpt_vs_flap"]
        if r["baseline"] == {"method": "flap", "calib": "ref", "sparsity": 40, "scope": "multi"}
    )
    assert method["comet"]["delta"] == pytest.approx(0.02, abs=1e-6)
    assert method["metricx"] is None  # FLAP has no MetricX yet
    assert "MetricX gen - ref [95% CI]" in summary
    assert summary_line(summary, "| SlimGPT | 40% | multi |").endswith(" |")
    assert "| 4.000 | 3.800 | -0.200 [" in summary_line(summary, "| SlimGPT | 40% | multi |")

    # Per-direction and sparsity sections have MetricX versions.
    per_direction = results["multi_per_direction"]["SlimGPT gen 40%"]["metricx"]
    assert per_direction["en-de"] == pytest.approx(3.8, abs=1e-6)
    assert "### MetricX-24 (lower is better)" in summary
    curve = results["sparsity_curve"]
    assert curve["dense_metricx"] == pytest.approx(2.8, abs=1e-6)
    dir_curve = next(
        r
        for r in curve["rows"]
        if (r["method"], r["calib"], r["scope"]) == ("slimgpt", "gen", "dir")
    )
    assert dir_curve["metricx"]["40"] == pytest.approx(2.8, abs=1e-6)
    assert dir_curve["comet"]["40"] == pytest.approx(0.86, abs=1e-6)
    assert "## Sparsity curve (macro MetricX-24, lower is better)" in summary
    assert results["dense"]["metricx"] == pytest.approx(2.8, abs=1e-6)


def test_full_suite_transfer_uses_only_all_ten_models(full_grid: Path) -> None:
    out, results = run_report(full_grid, "--suite", FULL_SUITE, bootstrap=0)
    assert results["transfer_models"]["pair"] == 0
    assert results["transfer_models"]["dir"] == 1
    data = results["transfer"]["slimgpt40-gen"]
    assert data["pair_models"] == 0 and data["dir_models"] == 1
    summary = data["dir_summary"]
    assert summary["own"]["n_cells"] == 1
    assert summary["own"]["comet"] == pytest.approx(0.86, abs=1e-6)
    assert summary["own"]["minus_multi"] == pytest.approx(0.05, abs=1e-6)
    assert summary["reverse"]["n_cells"] == 1
    assert summary["same_source"]["n_cells"] == 4
    assert summary["same_target"]["n_cells"] == 0
    assert summary["other"]["n_cells"] == 4
    assert summary["other"]["comet"] == pytest.approx(0.71, abs=1e-6)
    assert results["transfer"]["flap40-ref"]["dir_models"] == 0
    rows = (out / "transfer" / "slimgpt40-gen.csv").read_text().splitlines()
    assert len(rows) == 1 + 10  # header plus the one dir model's ten cells
    assert not (out / "transfer" / "flap40-ref.csv").exists()
    md = (out / "summary.md").read_text()
    assert "| SlimGPT gen 40% | 1/10 |" in md
    assert "| SlimGPT gen 40% | 0/5 | - | - | - |" in md
    plots = {Path(p).name for p in results["plots"]}
    assert "transfer_dir_s40.png" in plots and "transfer_pair_s40.png" not in plots
    assert "comet_vs_sparsity.png" in plots


def test_full_suite_repair(full_grid: Path) -> None:
    out, results = run_report(full_grid, "--suite", FULL_SUITE, "--no-plots", bootstrap=50)
    records = {r["system"]: r for r in results["repair"]}
    multi = records[f"{grid_name('slimgpt', 'gen', 40, 'multi')}-lora"]
    comet, metricx = multi["metrics"]["comet"], multi["metrics"]["metricx"]
    assert comet["recovered"] == pytest.approx(0.4, abs=1e-6)  # (0.83 - 0.81) / (0.86 - 0.81)
    # Lower is better: pruned 3.8, repaired 3.4, dense 2.8 closes 0.4 of the 1.0 gap.
    assert metricx["pruned"] == pytest.approx(3.8, abs=1e-6)
    assert metricx["repaired"] == pytest.approx(3.4, abs=1e-6)
    assert metricx["dense"] == pytest.approx(2.8, abs=1e-6)
    assert metricx["recovered"] == pytest.approx(0.4, abs=1e-6)
    assert metricx["higher_is_better"] is False
    contrast = multi["metricx_repaired_minus_pruned"]
    assert contrast["delta"] == pytest.approx(-0.4, abs=1e-6)
    assert contrast["bootstrap_ci_95"][1] < 0

    pair_name = f"{grid_name('slimgpt', 'gen', 40, 'pair', 'is')}-lora"
    pair = records[pair_name]
    assert pair["metrics"]["comet"]["n_directions"] == 2
    assert pair["comet_repaired_minus_pruned"]["directions"] == ["is-en", "en-is"]
    summary = (out / "summary.md").read_text()
    # Complete on its own pair, so no (2/10) marker.
    for prefix in (f"| {pair_name} | COMET |", f"- {pair_name}:"):
        for line in [line for line in summary.splitlines() if line.startswith(prefix)]:
            assert "/10)" not in line
    assert summary_line(summary, f"| {multi['system']} | MetricX-24 |").endswith(
        "| 2.800 | 3.800 | 3.400 | -0.400 | 40.0% |"
    )
    assert "Repaired - pruned MetricX-24 with 95% CI (negative = the repair helped):" in summary


def test_recovered_share_direction() -> None:
    assert gr.recovered_share(0.86, 0.81, 0.83) == pytest.approx(0.4)
    assert gr.recovered_share(2.8, 3.8, 3.4, higher_is_better=False) == pytest.approx(0.4)
    # A repair that makes MetricX worse recovers a negative share.
    assert gr.recovered_share(2.8, 3.8, 4.0, higher_is_better=False) == pytest.approx(-0.2)
    assert gr.recovered_share(2.8, 2.8, 3.0, higher_is_better=False) is None
    assert gr.recovered_share(None, 3.8, 3.4, higher_is_better=False) is None


def test_metricx_bootstrap_wiring(full_grid: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = spy_on_bootstrap(monkeypatch)
    argv = ("--suite", FULL_SUITE, "--no-plots", "--no-structure")
    _, results = run_report(full_grid, *argv, bootstrap=7)
    metricx_calls = [c for c in calls if c[2] is False]
    all_ten = tuple(FULL_SIZES[d] for d in gr.DIRECTIONS)
    assert metricx_calls and all(n == 7 for _, n, _ in calls)
    assert (all_ten, 7, False) in metricx_calls  # dir - multi on all ten directions
    own_pair = (FULL_SIZES["is-en"], FULL_SIZES["en-is"])
    assert (own_pair, 7, False) in metricx_calls  # pair-is repair on its own pair
    assert (all_ten, 7, True) in calls  # COMET: higher is better
    assert headline_row(results, "slimgpt", "gen", 40)["dir_minus_multi_metricx"]["n_samples"] == 7

    calls.clear()
    run_report(full_grid, *argv, bootstrap=7)
    assert calls == []  # every bootstrap, MetricX included, came from the cache

    _, results = run_report(full_grid, *argv, "--no-cache", bootstrap=0)
    assert calls == []
    delta = headline_row(results, "slimgpt", "gen", 40)["dir_minus_multi_metricx"]
    assert delta["p_value"] is None and delta["bootstrap_ci_95"] is None
    assert delta["delta"] == pytest.approx(-1.0, abs=1e-6)
    assert "| SlimGPT | gen | 40% | 3.800 | 3.400 | **2.800** | -0.400 | -1.000 |" in (
        (full_grid / "out" / "summary.md").read_text()
    )


def test_metricx_file_invalidates_run_cache(full_grid: Path) -> None:
    argv = ("--suite", FULL_SUITE, "--no-plots", "--no-structure")
    _, results = run_report(full_grid, *argv, bootstrap=0)
    assert headline_row(results, "flap", "ref", 40)["multi"]["metricx"] is None
    name = grid_name("flap", "ref", 40, "multi")
    root = full_grid / "runs" / f"{name}__{FULL_SUITE}__000000000000"
    write_metricx(root, name, gr.DIRECTIONS, FULL_SIZES)
    _, results = run_report(full_grid, *argv, bootstrap=0)
    assert headline_row(results, "flap", "ref", 40)["multi"]["metricx"] == pytest.approx(4.4)
    assert results["coverage"]["metricx"]["grid_models"] == 18


def test_run_cache_version(tmp_path: Path) -> None:
    cache = gr.Cache(tmp_path)
    cache.put("runs", "x", [1], {"a": 1})
    cache.put("bleu", "x", [1], {"b": 1})
    assert cache.get("runs", "x", [1]) == {"a": 1}
    path = tmp_path / "runs" / "x.json"
    assert json.loads(path.read_text())["version"] == gr.RUN_CACHE_VERSION == 2
    # A run parsed by the previous version lacks the MetricX fields: ignored.
    stale = json.loads(path.read_text()) | {"version": 1}
    path.write_text(json.dumps(stale))
    assert cache.get("runs", "x", [1]) is None
    # The BLEU and bootstrap caches keep their version and stay valid.
    assert json.loads((tmp_path / "bleu" / "x.json").read_text())["version"] == 1
    assert cache.get("bleu", "x", [1]) == {"b": 1}


# --------------------------------------------------------------------------
# Stratified bootstrap of the macro


@pytest.mark.parametrize("higher_is_better", [True, False])
def test_macro_bootstrap_single_direction_matches_segment_helper(higher_is_better: bool) -> None:
    """One direction: the same seed, CI and p convention as the project helper."""
    from mnlp_eval.metrics.significance import DEFAULT_SEED, bootstrap_segment_delta

    rng = np.random.default_rng(7)
    a, b = rng.normal(0.80, 0.1, 57), rng.normal(0.81, 0.1, 57)
    ours = gr.macro_segment_bootstrap(
        [(a, b)], n_samples=400, seed=DEFAULT_SEED, higher_is_better=higher_is_better
    )
    theirs = bootstrap_segment_delta(
        list(a), list(b), n_samples=400, seed=DEFAULT_SEED, higher_is_better=higher_is_better
    )
    for key in (
        "delta",
        "baseline_score",
        "system_score",
        "p_value",
        "bootstrap_ci_95",
        "improved",
        "n_segments",
        "significant_at_0.05",
    ):
        assert ours[key] == theirs[key], key


def test_macro_bootstrap_unequal_sizes() -> None:
    """Directions of 3, 40 and 400 segments count equally, unlike a pooled mean."""
    rng = np.random.default_rng(11)
    pairs = []
    for size, shift in ((3, 0.10), (40, 0.0), (400, -0.05)):
        a = rng.normal(0.8, 0.05, size)
        pairs.append((a, a + shift + rng.normal(0, 0.01, size)))
    macro_a = np.mean([a.mean() for a, _ in pairs])
    macro_b = np.mean([b.mean() for _, b in pairs])
    pooled = (
        np.concatenate([b for _, b in pairs]).mean() - np.concatenate([a for a, _ in pairs]).mean()
    )
    result = gr.macro_segment_bootstrap(pairs, n_samples=500, seed=3)
    assert result["delta"] == pytest.approx(macro_b - macro_a, abs=1e-6)
    assert result["baseline_score"] == pytest.approx(macro_a, abs=1e-6)
    assert result["system_score"] == pytest.approx(macro_b, abs=1e-6)
    assert abs(result["delta"] - pooled) > 0.03  # pooling would be dominated by the 400
    low, high = result["bootstrap_ci_95"]
    assert low <= result["delta"] <= high
    assert result["n_directions"] == 3 and result["n_segments"] == 443
    assert gr.macro_segment_bootstrap(pairs, n_samples=500, seed=3) == result
    assert gr.macro_segment_bootstrap(pairs, n_samples=500, seed=4)["bootstrap_ci_95"] != [
        low,
        high,
    ]
    worse = gr.macro_segment_bootstrap(pairs, n_samples=50, seed=3, higher_is_better=False)
    assert worse["improved"] is (result["delta"] < 0)


def shift_scores(run: Path, group: str, key: str, offsets: dict[str, float]) -> None:
    """Add a per-direction offset to every segment score of one metric in a run."""
    path = run / f"scores.{group}.json"
    payload = json.loads(path.read_text())
    for direction, offset in offsets.items():
        metric = payload["directions"][direction]["metrics"][key]
        metric["segment_scores"] = [v + offset for v in metric["segment_scores"]]
        metric["score"] = round(float(np.mean(metric["segment_scores"])), 6)
    path.write_text(json.dumps(payload))


def test_full_suite_contrasts_equal_macro_differences(full_grid: Path) -> None:
    """Unequal directions and a multi model whose gap varies by direction.

    The gen 40% multi model gains 0.01 x i COMET and loses 0.1 x i MetricX on
    the i-th direction, and later directions have more segments, so a pooled
    delta would differ from the macro difference (dir - multi COMET: pooled
    -0.006, macro +0.005). Every contrast must equal its macro difference.
    """
    multi = grid_name("slimgpt", "gen", 40, "multi")
    run = full_grid / "runs" / f"{multi}__{FULL_SUITE}__000000000000"
    shift_scores(run, "neural", gr.COMET_KEY, {d: 0.01 * i for i, d in enumerate(gr.DIRECTIONS)})
    shift_scores(run, "metricx", gr.METRICX_KEY, {d: -0.1 * i for i, d in enumerate(gr.DIRECTIONS)})
    out, results = run_report(full_grid, "--suite", FULL_SUITE, "--no-plots", bootstrap=200)

    row = headline_row(results, "slimgpt", "gen", 40)
    assert row["multi"]["comet"] == pytest.approx(0.855, abs=1e-6)
    assert row["multi"]["metricx"] == pytest.approx(3.35, abs=1e-6)
    for scope in ("pair", "dir"):
        for metric in ("comet", "metricx"):
            contrast = row[f"{scope}_minus_multi_{metric}"]
            macro_difference = row[scope][metric] - row["multi"][metric]
            assert contrast["delta"] == pytest.approx(macro_difference, abs=2e-6), (scope, metric)
            low, high = contrast["bootstrap_ci_95"]
            assert low <= contrast["delta"] <= high
    assert row["dir_minus_multi_comet"]["delta"] == pytest.approx(0.005, abs=1e-6)
    assert row["dir_minus_multi_metricx"]["delta"] == pytest.approx(-0.55, abs=1e-6)
    assert row["dir_minus_multi_comet"]["n_segments"] == sum(FULL_SIZES.values())

    summary = (out / "summary.md").read_text()
    line = summary_line(summary, "| SlimGPT | gen | 40% | 0.8550 |")
    assert line.split(" | ")[-1].startswith("+0.0050 [")

    # The contrast sections' score columns are the headline macros.
    contrast = next(
        r
        for r in results["ref_vs_gen"]
        if r["baseline"] == {"method": "slimgpt", "calib": "ref", "sparsity": 40, "scope": "multi"}
    )
    ref = headline_row(results, "slimgpt", "ref", 40)["multi"]
    assert contrast["comet"]["baseline_score"] == pytest.approx(ref["comet"], abs=1e-6)
    assert contrast["comet"]["system_score"] == pytest.approx(0.855, abs=1e-6)
    assert contrast["comet"]["delta"] == pytest.approx(0.055, abs=1e-6)
    assert contrast["metricx"]["delta"] == pytest.approx(3.35 - 4.0, abs=1e-6)

    (repair,) = [r for r in results["repair"] if r["system"] == f"{multi}-lora"]
    comet = repair["metrics"]["comet"]
    assert repair["comet_repaired_minus_pruned"]["delta"] == pytest.approx(
        comet["repaired"] - comet["pruned"], abs=2e-6
    )
    metricx = repair["metrics"]["metricx"]
    assert repair["metricx_repaired_minus_pruned"]["delta"] == pytest.approx(
        metricx["repaired"] - metricx["pruned"], abs=2e-6
    )


def test_bootstrap_cache_drops_pooled_results(grid: Path) -> None:
    """Results stored under the old (pooled) cache signature are not reused."""
    out = grid / "out"
    stale = {"version": gr.CACHE_VERSION, "signature": "v1", "data": {"k": {"delta": 9.0}}}
    (out / ".cache" / "bootstrap").mkdir(parents=True)
    (out / ".cache" / "bootstrap" / "results.json").write_text(json.dumps(stale))
    assert gr.Cache(out / ".cache").bootstrap_get("k") is None
    run_report(grid, "--no-plots", "--no-structure", bootstrap=5)
    stored = json.loads((out / ".cache" / "bootstrap" / "results.json").read_text())
    assert stored["signature"] == gr.BOOTSTRAP_CACHE == "v2"
    assert "k" not in stored["data"]
    methods = {entry["method"] for entry in stored["data"].values()}
    assert "stratified paired bootstrap on the macro of per-segment scores" in methods
