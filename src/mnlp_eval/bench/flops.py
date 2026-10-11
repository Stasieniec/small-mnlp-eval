"""Analytic FLOPs estimate and model FLOPs utilisation (Zhu et al., Section 2.1).

The estimate uses parameter and token counts only and ignores normalisation,
activation and softmax cost. MFU is relative to the device's dense bf16 peak,
hence ``mfu_bf16_equivalent`` for quantized models.
"""

from __future__ import annotations

from typing import Final

__all__ = ["DEVICE_PEAK_BF16_FLOPS", "device_peak_flops", "estimate_generation_flops"]

#: Dense bf16/fp16 peak throughput in FLOP/s, without sparsity (vendor figures).
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

    Uses ``2 N`` FLOPs per parameter per token. Both terms are multiplied by
    ``num_beams``, since transformers expands ``input_ids`` before prefill.
    ``prefill_positions`` is the padded width. Returns ``None`` for
    encoder-decoder models.
    """
    if architecture != "causal":
        return None
    per_token = 2 * non_embedding_parameters
    beams = max(num_beams, 1)
    prefill = per_token * prefill_positions * beams
    decode = per_token * generated_tokens * beams
    return int(prefill + decode)
