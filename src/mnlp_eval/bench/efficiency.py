"""Bench stage: efficiency measurement.

Every model generates the same number of tokens (``min_new_tokens ==
max_new_tokens``), at each configured batch size. Warmup batches are discarded,
the median of several repeats is reported, and peak memory is taken from both
the torch allocator and NVML.
"""

from __future__ import annotations

import dataclasses
import statistics
import sys
import time
from typing import Any

from mnlp_eval.artifacts import atomic_write_json
from mnlp_eval.bench.flops import device_peak_flops, estimate_generation_flops
from mnlp_eval.bench.structure import describe_structure
from mnlp_eval.config import BenchSpec, RunConfig
from mnlp_eval.data import load_testset
from mnlp_eval.languages import parse_direction
from mnlp_eval.models import Translator, build_translator
from mnlp_eval.models.loading import checkpoint_bytes
from mnlp_eval.runspec import RunPaths, utc_now
from mnlp_eval.seeding import seed_everything

__all__ = ["BENCH_SCHEMA_VERSION", "bench_run"]

BENCH_SCHEMA_VERSION = 2


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _cuda_memory_snapshot() -> dict[str, int | None]:
    try:
        import torch

        if not torch.cuda.is_available():
            return {}
        return {
            "allocated_bytes": torch.cuda.memory_allocated(),
            "reserved_bytes": torch.cuda.memory_reserved(),
            "max_allocated_bytes": torch.cuda.max_memory_allocated(),
            "max_reserved_bytes": torch.cuda.max_memory_reserved(),
        }
    except ImportError:  # pragma: no cover
        return {}


def _reset_peak_memory() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
    except ImportError:  # pragma: no cover
        pass


def _synchronize() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:  # pragma: no cover
        pass


def _nvml_process_memory() -> dict[str, Any]:
    """Total device memory in use according to NVML (includes the CUDA context)."""
    try:
        import pynvml
    except ImportError:
        return {"available": False, "reason": "nvidia-ml-py not installed"}
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        result: dict[str, Any] = {
            "available": True,
            "used_bytes": int(info.used),
            "total_bytes": int(info.total),
        }
        try:
            result["sm_clock_mhz"] = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)
            result["temperature_c"] = pynvml.nvmlDeviceGetTemperature(
                handle, pynvml.NVML_TEMPERATURE_GPU
            )
            result["power_watts"] = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
        except pynvml.NVMLError:
            pass
        pynvml.nvmlShutdown()
    except Exception as exc:
        return {"available": False, "reason": str(exc)}
    return result


def bench_run(
    config: RunConfig,
    spec: BenchSpec,
    *,
    translator: Translator | None = None,
    paths: RunPaths | None = None,
) -> dict[str, Any]:
    """Measure efficiency for one run and write ``bench.json``.

    ``paths`` overrides where the result is written (used by ``bench --run <dir>``).
    """
    paths = paths or RunPaths.for_config(config)
    paths.init_manifest(config)
    seed_everything(config.suite.decode.seed)

    direction = parse_direction(spec.direction)
    testset = load_testset(
        dataclasses.replace(config.suite.data, limit=spec.subset_size, directions=[str(direction)]),
        direction,
    )
    sources = list(testset.sources[: spec.subset_size])
    _log(f"bench {config.model.name}: {len(sources)} segments of {direction}")

    _reset_peak_memory()
    before_load = _cuda_memory_snapshot()
    owns_translator = translator is None
    if translator is None:
        translator = build_translator(config.model)
    _synchronize()
    after_load = _cuda_memory_snapshot()

    info = translator.info()
    weights_bytes = _weight_footprint(before_load, after_load)

    disk_bytes = checkpoint_bytes(info.checkpoint_dir)
    static: dict[str, Any] = {
        "total_parameters": info.total_parameters,
        "non_embedding_parameters": info.non_embedding_parameters,
        "dtype": info.dtype,
        "device": info.device,
        "load_seconds": info.load_seconds,
        "checkpoint_dir": info.checkpoint_dir,
        "checkpoint_bytes": disk_bytes,
        "resident_weight_bytes": weights_bytes,
        "bits_per_parameter_on_disk": round(disk_bytes * 8 / info.total_parameters, 3)
        if disk_bytes and info.total_parameters
        else None,
        "bits_per_parameter_resident": round(weights_bytes * 8 / info.total_parameters, 3)
        if weights_bytes and info.total_parameters
        else None,
    }
    _log(
        f"  {info.total_parameters / 1e9:.2f}B params, "
        f"disk={_human(disk_bytes)}, resident={_human(weights_bytes)}"
    )

    structure = _measure_structure(translator, measure=spec.measure_structure)

    measurements: dict[str, Any] = {}
    try:
        for batch_size in spec.batch_sizes:
            measurements[str(batch_size)] = _measure_batch_size(
                translator, config, spec, direction, sources, batch_size, info
            )
        ttft = _measure_ttft(translator, config, spec, direction, sources)
    finally:
        if owns_translator:
            translator.close()

    payload = {
        "schema_version": BENCH_SCHEMA_VERSION,
        "run_id": config.run_id,
        "model": config.model.name,
        "measured_at": utc_now(),
        "protocol": {
            **dataclasses.asdict(spec),
            "num_beams": config.suite.decode.num_beams,
            "fixed_token_budget": spec.force_fixed_length,
            "note": (
                "min_new_tokens equals max_new_tokens, so every model generates the same "
                "number of tokens and latency does not reward early stopping"
            ),
        },
        "data": testset.provenance(),
        "static": static,
        "structure": structure,
        "time_to_first_token": ttft,
        "by_batch_size": measurements,
        "device_state_after": _nvml_process_memory(),
    }
    atomic_write_json(paths.bench, payload)
    paths.record_stage("bench", {"status": "completed", "path": str(paths.bench)})
    return payload


def _measure_structure(translator: Translator, *, measure: bool) -> dict[str, Any] | None:
    """Describe the loaded model's layer structure. Errors are recorded, not raised."""
    if not measure:
        return None
    model = getattr(translator, "model", None)
    if model is None:
        return {"unavailable": "the translator exposes no model object"}
    try:
        structure = describe_structure(model)
    except (AttributeError, TypeError, RuntimeError) as exc:
        return {"unavailable": f"{type(exc).__name__}: {exc}"}
    stacks = structure.get("stacks") or {}
    for name, stack in stacks.items():
        spread = stack.get("ffn_intermediate") or {}
        width = (
            f"{spread.get('min')}"
            if spread.get("min") == spread.get("max")
            else f"{spread.get('min')} to {spread.get('max')}"
        )
        _log(
            f"  {name}: {stack['n_layers']} layers, FFN width {width}, "
            f"{'uniform' if stack['uniform'] else 'non-uniform'}"
        )
    zero_fraction = structure.get("zero_fraction")
    if zero_fraction:
        _log(f"  {zero_fraction:.2%} of floating-point weights are exactly zero")
    return structure


def _weight_footprint(before: dict[str, int | None], after: dict[str, int | None]) -> int | None:
    """Bytes the weights occupy, as the allocator delta across model loading (None without CUDA)."""
    start = before.get("allocated_bytes")
    end = after.get("allocated_bytes")
    if start is None or end is None:
        return None
    return max(end - start, 0)


def _measure_batch_size(
    translator: Translator,
    config: RunConfig,
    spec: BenchSpec,
    direction: Any,
    sources: list[str],
    batch_size: int,
    info: Any,
) -> dict[str, Any]:
    decode = dataclasses.replace(
        config.suite.decode,
        batch_size=batch_size,
        max_new_tokens=spec.max_new_tokens,
        sort_by_length=False,
    )
    warmup_slice = sources[: max(batch_size * spec.warmup_batches, batch_size)]

    # Force a fixed number of generated tokens.
    forced = {"min_new_tokens": spec.max_new_tokens} if spec.force_fixed_length else {}
    original_generate_kwargs = translator.generate_kwargs

    def patched(direction_arg: Any) -> dict[str, Any]:
        return {**original_generate_kwargs(direction_arg), **forced}

    translator.generate_kwargs = patched  # type: ignore[assignment,method-assign]
    try:
        for _ in range(spec.warmup_batches):
            translator.translate(direction, warmup_slice, decode)

        _reset_peak_memory()
        durations: list[float] = []
        generated_tokens = 0
        source_tokens = 0
        for _ in range(spec.repeats):
            _synchronize()
            started = time.perf_counter()
            outputs = translator.translate(direction, sources, decode)
            _synchronize()
            durations.append(time.perf_counter() - started)
            generated_tokens = sum(output.n_generated_tokens for output in outputs)
            source_tokens = sum(output.n_source_tokens for output in outputs)
            # The FLOPs estimate needs the padded prefill width.
            prefill_positions = sum(
                output.n_padded_source_tokens or output.n_source_tokens for output in outputs
            )
        peak = _cuda_memory_snapshot()
    finally:
        translator.generate_kwargs = original_generate_kwargs  # type: ignore[method-assign]

    median = statistics.median(durations)
    flops = estimate_generation_flops(
        info.non_embedding_parameters,
        prefill_positions=prefill_positions,
        generated_tokens=generated_tokens,
        num_beams=config.suite.decode.num_beams,
        architecture="causal" if translator.kind.startswith("causal") else "seq2seq",
    )
    peak_device_flops, peak_note = device_peak_flops(_device_name())
    mfu = (
        round(flops / (median * peak_device_flops), 5)
        if flops and peak_device_flops and median > 0
        else None
    )

    result = {
        "batch_size": batch_size,
        "n_segments": len(sources),
        "repeats": spec.repeats,
        "seconds": [round(value, 4) for value in durations],
        "seconds_median": round(median, 4),
        "seconds_min": round(min(durations), 4),
        "seconds_max": round(max(durations), 4),
        "seconds_stdev": round(statistics.stdev(durations), 4) if len(durations) > 1 else 0.0,
        "sentences_per_second": round(len(sources) / median, 3) if median else None,
        "generated_tokens": generated_tokens,
        "generated_tokens_per_second": round(generated_tokens / median, 2) if median else None,
        "latency_per_sentence_ms": round(median * 1000 / len(sources), 3) if sources else None,
        "peak_allocated_bytes": peak.get("max_allocated_bytes"),
        "peak_reserved_bytes": peak.get("max_reserved_bytes"),
        "source_tokens": source_tokens,
        "prefill_positions": prefill_positions,
        "estimated_forward_flops": flops,
        "mfu_bf16_equivalent": mfu,
    }
    if flops is None:
        result["flops_note"] = "not estimated for encoder-decoder models; see mnlp_eval.bench.flops"
    elif mfu is None:
        result["mfu_note"] = peak_note
    _log(
        f"  batch={batch_size}: {result['seconds_median']}s median, "
        f"{result['generated_tokens_per_second']} tok/s, "
        f"peak={_human(result['peak_allocated_bytes'])}"
    )
    return result


def _measure_ttft(
    translator: Translator,
    config: RunConfig,
    spec: BenchSpec,
    direction: Any,
    sources: list[str],
) -> dict[str, Any]:
    """Prefill latency, measured as a single-token generation at batch size 1.

    Includes one decode step, so it approximates time to first token.
    """
    decode = dataclasses.replace(
        config.suite.decode,
        batch_size=1,
        max_new_tokens=1,
        num_beams=1,
        sort_by_length=False,
    )
    sample = sources[: min(len(sources), 16)]
    for _ in range(2):
        translator.translate(direction, sample[:2], decode)
    durations = []
    for _ in range(spec.repeats):
        _synchronize()
        started = time.perf_counter()
        translator.translate(direction, sample, decode)
        _synchronize()
        durations.append((time.perf_counter() - started) / len(sample))
    median = statistics.median(durations)
    return {
        "n_segments": len(sample),
        "repeats": spec.repeats,
        "median_ms": round(median * 1000, 3),
        "min_ms": round(min(durations) * 1000, 3),
        "max_ms": round(max(durations) * 1000, 3),
        "method": "single-token generation at batch size 1, greedy, prefill dominated",
    }


def _device_name() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return str(torch.cuda.get_device_name(0))
    except ImportError:  # pragma: no cover
        pass
    return "cpu"


def _human(value: Any) -> str:
    if value is None:
        return "unknown"
    size = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(size) < 1024 or unit == "TiB":
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} TiB"
