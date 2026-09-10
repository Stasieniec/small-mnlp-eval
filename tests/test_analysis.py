"""Subnetwork descriptors and overlap analysis.

The chance baseline is the part that has to be right. Two independently chosen
50 percent subnetworks share about a third of what they keep, so a Jaccard
index of 0.33 is the null result rather than evidence of shared structure.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from mnlp_eval.analysis.overlap import compare_subnetworks, component_overlap, overlap_report
from mnlp_eval.analysis.subnetwork import (
    SubnetworkError,
    load_subnetwork,
    load_subnetworks,
)


def write_descriptor(path: Path, name: str, kept: dict[str, list[int]], total: int = 8) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "name": name,
                "base_model": "haoranxu/ALMA-7B",
                "pruned_for": name,
                "components": {"attention_heads": {"total": total, "kept": kept}},
            }
        ),
        encoding="utf-8",
    )
    return path


# --------------------------------------------------------------------------
# Descriptor validation
# --------------------------------------------------------------------------


def test_a_valid_descriptor_reports_its_sparsity(tmp_path: Path) -> None:
    path = write_descriptor(tmp_path / "a.json", "a", {"0": [0, 1, 2, 3], "1": [4, 5, 6, 7]})

    subnetwork = load_subnetwork(path)

    assert subnetwork.name == "a"
    assert subnetwork.components["attention_heads"].sparsity == pytest.approx(0.5)
    assert subnetwork.overall_sparsity == pytest.approx(0.5)


def test_a_flat_kept_list_is_a_single_unlayered_component(tmp_path: Path) -> None:
    """Whole-layer pruning is not per-layer, so it uses one bucket."""
    path = tmp_path / "layers.json"
    path.write_text(
        json.dumps(
            {
                "name": "drop8",
                "components": {"layers": {"total": 32, "kept": list(range(24))}},
            }
        ),
        encoding="utf-8",
    )

    subnetwork = load_subnetwork(path)

    assert set(subnetwork.components["layers"].kept) == {"*"}
    assert subnetwork.components["layers"].sparsity == pytest.approx(0.25)


@pytest.mark.parametrize(
    ("kept", "message"),
    [
        ({"0": [0, 1, 99]}, "outside"),
        ({"0": [0, 1, 1]}, "repeated"),
        ({"0": []}, "kept nothing"),
        ({"0": [0, "1"]}, "must be integers"),
    ],
)
def test_malformed_selections_are_rejected(tmp_path: Path, kept: dict, message: str) -> None:
    path = write_descriptor(tmp_path / "bad.json", "bad", kept)

    with pytest.raises(SubnetworkError, match=message):
        load_subnetwork(path)


def test_duplicate_names_are_rejected(tmp_path: Path) -> None:
    """Two descriptors under one name would collapse into a self-comparison."""
    first = write_descriptor(tmp_path / "one.json", "same", {"0": [0, 1]})
    second = write_descriptor(tmp_path / "two.json", "same", {"0": [2, 3]})

    with pytest.raises(SubnetworkError, match="both named"):
        load_subnetworks([first, second])


def test_a_future_schema_version_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "future.json"
    path.write_text(
        json.dumps({"schema_version": 99, "components": {"heads": {"total": 4, "kept": [0]}}}),
        encoding="utf-8",
    )

    with pytest.raises(SubnetworkError, match="schema_version"):
        load_subnetwork(path)


# --------------------------------------------------------------------------
# Overlap
# --------------------------------------------------------------------------


def test_identical_subnetworks_overlap_completely(tmp_path: Path) -> None:
    kept = {"0": [0, 1, 2, 3], "1": [0, 2, 4, 6]}
    left = load_subnetwork(write_descriptor(tmp_path / "l.json", "l", kept))
    right = load_subnetwork(write_descriptor(tmp_path / "r.json", "r", kept))

    result = component_overlap(
        left.components["attention_heads"], right.components["attention_heads"]
    )

    assert result.jaccard == 1.0
    assert result.overlap_coefficient == 1.0
    assert result.excess == pytest.approx(1.0 - result.chance_jaccard)


def test_disjoint_subnetworks_overlap_not_at_all(tmp_path: Path) -> None:
    left = load_subnetwork(write_descriptor(tmp_path / "l.json", "l", {"0": [0, 1, 2, 3]}))
    right = load_subnetwork(write_descriptor(tmp_path / "r.json", "r", {"0": [4, 5, 6, 7]}))

    result = component_overlap(
        left.components["attention_heads"], right.components["attention_heads"]
    )

    assert result.jaccard == 0.0
    # Disjoint is not the null result. Chance would have shared a third of
    # them, so being disjoint is evidence of anti-correlation.
    assert result.excess == pytest.approx(-result.chance_jaccard)
    assert result.chance_jaccard == pytest.approx(1 / 3)


def test_independent_selections_sit_at_chance(tmp_path: Path) -> None:
    """The result the analysis exists to make visible.

    Raw Jaccard reads as substantial agreement. The excess over chance, which
    is what the report quotes, reads as approximately nothing.
    """
    rng = random.Random(20260909)
    total, keep, layers = 128, 64, 32
    kept = {
        name: {str(layer): sorted(rng.sample(range(total), keep)) for layer in range(layers)}
        for name in ("a", "b")
    }
    left = load_subnetwork(write_descriptor(tmp_path / "a.json", "a", kept["a"], total=total))
    right = load_subnetwork(write_descriptor(tmp_path / "b.json", "b", kept["b"], total=total))

    result = component_overlap(
        left.components["attention_heads"], right.components["attention_heads"]
    )

    assert result.jaccard == pytest.approx(1 / 3, abs=0.02)
    assert result.chance_jaccard == pytest.approx(1 / 3, abs=1e-9)
    assert abs(result.excess) < 0.02


def test_overlap_pools_positions_across_layers(tmp_path: Path) -> None:
    """The same head index in different layers is not agreement."""
    left = load_subnetwork(write_descriptor(tmp_path / "l.json", "l", {"0": [0, 1], "1": [2, 3]}))
    right = load_subnetwork(write_descriptor(tmp_path / "r.json", "r", {"0": [2, 3], "1": [0, 1]}))

    result = component_overlap(
        left.components["attention_heads"], right.components["attention_heads"]
    )

    assert result.intersection == 0
    assert result.jaccard == 0.0


def test_different_widths_cannot_be_compared(tmp_path: Path) -> None:
    left = load_subnetwork(write_descriptor(tmp_path / "l.json", "l", {"0": [0, 1]}, total=8))
    right = load_subnetwork(write_descriptor(tmp_path / "r.json", "r", {"0": [0, 1]}, total=16))

    with pytest.raises(ValueError, match="different models"):
        component_overlap(left.components["attention_heads"], right.components["attention_heads"])


def test_the_overlap_coefficient_detects_a_nested_subnetwork(tmp_path: Path) -> None:
    """A sparser subnetwork contained in a denser one is the transfer story."""
    dense = load_subnetwork(write_descriptor(tmp_path / "d.json", "d", {"0": [0, 1, 2, 3]}))
    sparse = load_subnetwork(write_descriptor(tmp_path / "s.json", "s", {"0": [0, 1]}))

    result = component_overlap(
        dense.components["attention_heads"], sparse.components["attention_heads"]
    )

    assert result.overlap_coefficient == 1.0
    assert result.jaccard == pytest.approx(0.5)


def test_comparison_covers_every_pair_and_renders(tmp_path: Path) -> None:
    paths = [
        write_descriptor(tmp_path / f"{name}.json", name, kept)
        for name, kept in (
            ("de", {"0": [0, 1, 2, 3]}),
            ("is", {"0": [2, 3, 4, 5]}),
            ("multi", {"0": [0, 2, 4, 6]}),
        )
    ]

    analysis = compare_subnetworks(load_subnetworks(list(paths)))

    assert [(pair["left"], pair["right"]) for pair in analysis["pairs"]] == [
        ("de", "is"),
        ("de", "multi"),
        ("is", "multi"),
    ]
    rendered = overlap_report(analysis)
    assert "Subnetwork overlap" in rendered
    assert "Excess" in rendered


def test_a_component_only_one_descriptor_has_is_skipped(tmp_path: Path) -> None:
    first = write_descriptor(tmp_path / "a.json", "a", {"0": [0, 1]})
    second = tmp_path / "b.json"
    second.write_text(
        json.dumps(
            {
                "name": "b",
                "components": {
                    "attention_heads": {"total": 8, "kept": {"0": [0, 1]}},
                    "ffn_channels": {"total": 16, "kept": {"0": [0, 1, 2]}},
                },
            }
        ),
        encoding="utf-8",
    )

    analysis = compare_subnetworks(load_subnetworks([first, second]))

    assert analysis["components_compared"] == ["attention_heads"]
    assert analysis["components_skipped"] == ["ffn_channels"]


def test_one_descriptor_is_not_a_comparison(tmp_path: Path) -> None:
    only = load_subnetwork(write_descriptor(tmp_path / "a.json", "a", {"0": [0, 1]}))

    with pytest.raises(ValueError, match="at least two"):
        compare_subnetworks([only])
