"""Analytic FLOPs estimation and model FLOPs utilisation.

Reported because Zhu et al., Section 2.1, lists both. Wall-clock time is the
number a compression report turns on.

Two limits are recorded in the field names rather than in a footnote:

* The estimate is analytic, from parameter and token counts. It is not a
  traced operator count, and it ignores normalisation, activation and softmax
  cost.
* MFU is computed against the device's dense bf16 peak. A 4-bit model executes
  low-precision matmuls with a different ceiling, so its figure is not a
  utilisation number in the usual sense, and the field is named
  ``mfu_bf16_equivalent``.
"""

from __future__ import annotations

from typing import Final

__all__ = ["DEVICE_PEAK_BF16_FLOPS", "device_peak_flops", "estimate_generation_flops"]

#: Dense bf16 or fp16 peak throughput in FLOP/s, without sparsity doubling.
#: Vendor figures. Extend this table rather than guessing at runtime: a wrong
#: peak silently corrupts every MFU number in a report.
DEVICE_PEAK_BF16_FLOPS: Final[dict[str, float]] = {
    "a100": 312e12,
    "a100-sxm4-40gb": 312e12,
    "a100-sxm4-80gb": 312e12,
    "a100-pcie-40gb": 312e12,
    "h100-sxm": 989e12,
    "h100-pcie": 756e12,
    "h200": 989e12,
    "l40s": 362e12,
    "l4": 121e12,
    "a6000": 155e12,
    "rtx-6000-ada": 364e12,
    "rtx-4090": 165e12,
    "v100": 125e12,
}


def device_peak_flops(device_name: str) -> tuple[float | None, str]:
    """Look up a device's dense bf16 peak, or explain why it is unknown."""
    normalised = device_name.lower().replace("nvidia ", "").replace(" ", "-")
    for key, value in DEVICE_PEAK_BF16_FLOPS.items():
        if key in normalised:
            return value, key
    return None, (
        f"no peak FLOPS entry for {device_name!r}; add one to "
        "DEVICE_PEAK_BF16_FLOPS in mnlp_eval/bench/flops.py to enable MFU"
    )


def estimate_generation_flops(
    non_embedding_parameters: int,
    *,
    prefill_positions: int,
    generated_tokens: int,
    num_beams: int = 1,
    architecture: str = "causal",
) -> int | None:
    """Estimate forward FLOPs for one decoder-only generation workload.

    Uses the standard ``2 N`` FLOPs per parameter per token approximation.
    Both terms are multiplied by ``num_beams``, because transformers expands
    ``input_ids`` by the beam count before the prefill pass.
    ``prefill_positions`` must be the padded width the model computed over, not
    the sum of unpadded token counts.

    Returns ``None`` for encoder-decoder models. Their cost splits between an
    encoder that runs once unexpanded and a decoder that runs beam-expanded,
    and ``non_embedding_parameters`` covers both, so a single ``2 N`` figure
    would overstate the decode term.
    """
    if architecture != "causal":
        return None
    per_token = 2 * non_embedding_parameters
    beams = max(num_beams, 1)
    prefill = per_token * prefill_positions * beams
    decode = per_token * generated_tokens * beams
    return int(prefill + decode)
