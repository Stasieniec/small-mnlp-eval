"""FLOPs estimation and efficiency accounting."""

from __future__ import annotations

import pytest

from mnlp_eval.bench.efficiency import _human, _weight_footprint
from mnlp_eval.bench.flops import (
    DEVICE_PEAK_BF16_FLOPS,
    device_peak_flops,
    estimate_generation_flops,
)

PARAMETERS = 6_000_000_000


def test_greedy_flops_are_two_per_parameter_per_token() -> None:
    flops = estimate_generation_flops(
        PARAMETERS, prefill_positions=10, generated_tokens=20, num_beams=1
    )
    assert flops == 2 * PARAMETERS * 30


def test_beam_search_multiplies_both_terms() -> None:
    # transformers expands input_ids by the beam count before prefill, so
    # prefill is beam-expanded too.
    single = estimate_generation_flops(
        PARAMETERS, prefill_positions=100, generated_tokens=50, num_beams=1
    )
    beamed = estimate_generation_flops(
        PARAMETERS, prefill_positions=100, generated_tokens=50, num_beams=5
    )
    assert single is not None
    assert beamed == single * 5


def test_padded_positions_drive_the_prefill_term() -> None:
    # Prefill computes over the padding, so use the padded batch width.
    unpadded = estimate_generation_flops(
        PARAMETERS, prefill_positions=33, generated_tokens=0, num_beams=1
    )
    padded = estimate_generation_flops(
        PARAMETERS, prefill_positions=190, generated_tokens=0, num_beams=1
    )
    assert unpadded is not None
    assert padded is not None
    assert padded > unpadded * 5


def test_encoder_decoder_returns_nothing_rather_than_a_wrong_number() -> None:
    # The encoder runs once unexpanded, the decoder beam-expanded.
    assert (
        estimate_generation_flops(
            PARAMETERS,
            prefill_positions=100,
            generated_tokens=50,
            num_beams=5,
            architecture="seq2seq",
        )
        is None
    )


def test_zero_beams_is_treated_as_one() -> None:
    assert (
        estimate_generation_flops(PARAMETERS, prefill_positions=1, generated_tokens=1, num_beams=0)
        == 2 * PARAMETERS * 2
    )


@pytest.mark.parametrize(
    ("device", "expected"),
    [
        ("NVIDIA A100-SXM4-40GB", 312e12),
        ("NVIDIA A100 80GB PCIe", 312e12),
        ("NVIDIA L40S", 362e12),
        ("NVIDIA GeForce RTX 4090", 165e12),
    ],
)
def test_known_devices_resolve_to_their_vendor_peak(device: str, expected: float) -> None:
    peak, _ = device_peak_flops(device)
    assert peak == expected


def test_unknown_device_reports_no_utilisation_and_says_why() -> None:
    peak, note = device_peak_flops("NVIDIA RTX 1000 Ada Generation Laptop GPU")
    assert peak is None
    assert "DEVICE_PEAK_BF16_FLOPS" in note


def test_peak_table_holds_only_plausible_values() -> None:
    for name, value in DEVICE_PEAK_BF16_FLOPS.items():
        assert 1e13 <= value <= 2e15, f"{name} peak looks wrong: {value}"


def test_weight_footprint_is_the_allocator_delta() -> None:
    assert _weight_footprint({"allocated_bytes": 100}, {"allocated_bytes": 500}) == 400


def test_weight_footprint_is_none_when_it_cannot_be_measured() -> None:
    assert _weight_footprint({}, {}) is None
    assert _weight_footprint({"allocated_bytes": None}, {"allocated_bytes": 5}) is None


def test_weight_footprint_never_goes_negative() -> None:
    assert _weight_footprint({"allocated_bytes": 500}, {"allocated_bytes": 100}) == 0


@pytest.mark.parametrize(
    ("value", "expected"),
    [(None, "unknown"), (0, "0.00 B"), (1024, "1.00 KiB"), (1536 * 1024**2, "1.50 GiB")],
)
def test_byte_formatting(value: object, expected: str) -> None:
    assert _human(value) == expected
