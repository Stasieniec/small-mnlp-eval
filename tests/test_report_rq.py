"""The report tables that answer the project's research questions.

Built against hand-assembled summaries rather than through the pipeline, so
each table is exercised on the exact shape of data that makes it interesting:
a subnetwork that lost its low-resource direction and nothing else, a
pair-specific subnetwork evaluated on pairs it was never selected for, and a
mask that was applied but never compacted.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mnlp_eval.report.tables import (
    RunSummary,
    _resource_tier_table,
    _structure_table,
    _transfer_table,
    build_tables,
)
from mnlp_eval.runspec import RunPaths

DIRECTIONS = ["de-en", "en-de", "is-en", "en-is"]


def summary(
    name: str,
    comet: dict[str, float],
    *,
    tmp_path: Path,
    baseline: str | None = None,
    compression: dict[str, Any] | None = None,
    structure: dict[str, Any] | None = None,
) -> RunSummary:
    return RunSummary(
        paths=RunPaths(tmp_path / name),
        manifest={
            "run_id": f"id-{name}",
            "model": {"name": name, "baseline": baseline, "compression": compression or {}},
            "suite": {"name": "suite", "data": {"directions": DIRECTIONS}},
            "identity": {"data": {}, "decode": {}},
        },
        scores={
            "neural": {
                "aggregate": {"wmt22_comet_da": sum(comet.values()) / len(comet)},
                "directions": {
                    direction: {"metrics": {"wmt22_comet_da": {"score": value}}}
                    for direction, value in comet.items()
                },
            }
        },
        bench={"static": {}, "structure": structure} if structure is not None else None,
    )


def uniform_structure(layers: int, ffn: int, *, zero_fraction: float = 0.0) -> dict[str, Any]:
    return {
        "zero_fraction": zero_fraction,
        "stacks": {
            "model.layers": {
                "n_layers": layers,
                "uniform": True,
                "parameters": layers * ffn,
                "attention_inner_dim": {
                    "min": 4096,
                    "max": 4096,
                    "mean": 4096.0,
                    "total": 4096 * layers,
                },
                "ffn_intermediate": {
                    "min": ffn,
                    "max": ffn,
                    "mean": float(ffn),
                    "total": ffn * layers,
                },
                "layers": [],
            }
        },
    }


# --------------------------------------------------------------------------
# Structure
# --------------------------------------------------------------------------


def test_structure_table_reports_what_was_removed(tmp_path: Path) -> None:
    even = dict.fromkeys(DIRECTIONS, 0.85)
    base = summary("alma-7b", even, tmp_path=tmp_path, structure=uniform_structure(32, 11008))
    pruned = summary(
        "pruned",
        even,
        tmp_path=tmp_path,
        baseline="alma-7b",
        compression={"family": "pruning", "nominal_sparsity": 0.5},
        structure=uniform_structure(32, 5504),
    )

    table = _structure_table([base, pruned], base)

    assert table is not None
    row = dict(zip(table.columns, table.rows[1], strict=True))
    assert row["System"] == "pruned"
    assert row["Layers"] == "32"
    assert row["FFN width"] == "5504"
    assert row["Uniform"] == "yes"
    assert row["Layers kept"] == "100.0%"
    assert row["FFN width kept"] == "50.0%"


def test_structure_table_shows_a_mask_that_was_never_compacted(tmp_path: Path) -> None:
    """Same shape, same parameter count, no saving delivered."""
    even = dict.fromkeys(DIRECTIONS, 0.85)
    base = summary("alma-7b", even, tmp_path=tmp_path, structure=uniform_structure(32, 11008))
    masked = summary(
        "masked",
        even,
        tmp_path=tmp_path,
        baseline="alma-7b",
        structure=uniform_structure(32, 11008, zero_fraction=0.5),
    )

    table = _structure_table([base, masked], base)

    assert table is not None
    row = dict(zip(table.columns, table.rows[1], strict=True))
    assert row["FFN width kept"] == "100.0%"
    assert row["Zero weights"] == "50.0%"


def test_structure_table_is_omitted_when_nothing_measured_it(tmp_path: Path) -> None:
    even = dict.fromkeys(DIRECTIONS, 0.85)
    runs = [summary("a", even, tmp_path=tmp_path), summary("b", even, tmp_path=tmp_path)]

    assert _structure_table(runs, runs[0]) is None


def test_non_uniform_pruning_is_visible_as_a_width_range(tmp_path: Path) -> None:
    even = dict.fromkeys(DIRECTIONS, 0.85)
    structure = uniform_structure(32, 11008)
    stack = structure["stacks"]["model.layers"]
    stack["uniform"] = False
    stack["ffn_intermediate"] = {"min": 2048, "max": 11008, "mean": 6000.0, "total": 192000}

    table = _structure_table(
        [summary("uneven", even, tmp_path=tmp_path, structure=structure)], None
    )

    assert table is not None
    row = dict(zip(table.columns, table.rows[0], strict=True))
    assert row["FFN width"] == "2048 to 11008"
    assert row["Uniform"] == "no"


# --------------------------------------------------------------------------
# Resource tier
# --------------------------------------------------------------------------


def test_low_resource_degradation_is_separated_from_the_macro_average(tmp_path: Path) -> None:
    """RQ2. A system that lost Icelandic alone still looks mild on the average."""
    base = summary("alma-7b", dict.fromkeys(DIRECTIONS, 0.85), tmp_path=tmp_path)
    pruned = summary(
        "pruned",
        {"de-en": 0.84, "en-de": 0.84, "is-en": 0.60, "en-is": 0.60},
        tmp_path=tmp_path,
        baseline="alma-7b",
        compression={"family": "pruning", "nominal_sparsity": 0.5},
    )

    table = _resource_tier_table([base, pruned], base)

    assert table is not None
    row = dict(zip(table.columns, table.rows[0], strict=True))
    assert row["high resource"] == "-0.0100"
    assert row["low resource"] == "-0.2500"
    assert row["Low minus high"] == "-0.2400"


def test_resource_tier_table_needs_both_tiers(tmp_path: Path) -> None:
    high_only = ["de-en", "en-de"]
    runs = []
    for name in ("alma-7b", "pruned"):
        run = summary(name, dict.fromkeys(high_only, 0.85), tmp_path=tmp_path)
        run.manifest["suite"]["data"]["directions"] = high_only
        runs.append(run)

    assert _resource_tier_table(runs, runs[0]) is None


def test_resource_tier_table_needs_a_baseline(tmp_path: Path) -> None:
    even = dict.fromkeys(DIRECTIONS, 0.85)
    runs = [summary("a", even, tmp_path=tmp_path), summary("b", even, tmp_path=tmp_path)]

    assert _resource_tier_table(runs, None) is None


# --------------------------------------------------------------------------
# Transfer
# --------------------------------------------------------------------------


def test_transfer_table_marks_matched_directions_and_measures_the_gap(tmp_path: Path) -> None:
    """RQ3. A de-calibrated subnetwork keeps German and loses the rest."""
    multi = summary(
        "prune50-multi",
        dict.fromkeys(DIRECTIONS, 0.80),
        tmp_path=tmp_path,
        compression={"family": "pruning", "nominal_sparsity": 0.5, "pruned_for": "multi"},
    )
    specific = summary(
        "prune50-de",
        {"de-en": 0.84, "en-de": 0.84, "is-en": 0.60, "en-is": 0.60},
        tmp_path=tmp_path,
        compression={
            "family": "pruning",
            "nominal_sparsity": 0.5,
            "pruned_for": "de-en,en-de",
        },
    )

    table = _transfer_table([multi, specific])

    assert table is not None
    rows = {row[0]: dict(zip(table.columns, row, strict=True)) for row in table.rows}
    assert rows["prune50-de"]["Selected for"] == "de-en+en-de"
    assert rows["prune50-de"]["de-en"] == "[0.84]"
    assert rows["prune50-de"]["is-en"] == "0.6"
    assert rows["prune50-de"]["Matched"] == "0.8400"
    assert rows["prune50-de"]["Mismatched"] == "0.6000"
    assert rows["prune50-de"]["Specialization"] == "+0.2400"
    # The multi-directional row is the comparison, and has no diagonal.
    assert rows["prune50-multi"]["Selected for"] == "multi"
    assert rows["prune50-multi"]["Specialization"] == "-"


def test_transfer_table_is_omitted_without_a_pair_specific_system(tmp_path: Path) -> None:
    even = dict.fromkeys(DIRECTIONS, 0.85)
    runs = [
        summary("alma-7b", even, tmp_path=tmp_path),
        summary("quantized", even, tmp_path=tmp_path, compression={"family": "quantization"}),
    ]

    assert _transfer_table(runs) is None


def test_build_tables_adds_the_research_question_tables_when_they_apply(
    tmp_path: Path,
) -> None:
    base = summary(
        "alma-7b",
        dict.fromkeys(DIRECTIONS, 0.85),
        tmp_path=tmp_path,
        structure=uniform_structure(32, 11008),
    )
    specific = summary(
        "prune50-de",
        {"de-en": 0.84, "en-de": 0.84, "is-en": 0.60, "en-is": 0.60},
        tmp_path=tmp_path,
        baseline="alma-7b",
        compression={"family": "pruning", "nominal_sparsity": 0.5, "pruned_for": "de-en,en-de"},
        structure=uniform_structure(32, 5504),
    )

    titles = [table.title for table in build_tables([base, specific], baseline="alma-7b")]

    assert titles[:3] == ["Translation quality", "Behavioural failure modes", "Efficiency"]
    assert "Structure" in titles
    assert any(title.startswith("Degradation by resource tier") for title in titles)
    assert any(title.startswith("Cross-direction transfer") for title in titles)
