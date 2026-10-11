#!/usr/bin/env bash
# End-to-end check on a small local slice, a few minutes on a 6 GB GPU.
#
# opus-mt-de-en is a real MT system, so its scores check the metric pipeline.
# Qwen2.5-0.5B-Instruct covers the prompted decoder-only path and hypothesis
# extraction.
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON=${PYTHON:-./.venv/bin/python}
CLI=${CLI:-./.venv/bin/mnlp-eval}
LIMIT=${LIMIT:-64}
RUNS_ROOT=${RUNS_ROOT:-runs-smoke}
REPORT_DIR=${REPORT_DIR:-reports-smoke}

if [[ ! -x "${CLI}" ]]; then
    echo "mnlp-eval not found at ${CLI}. Create the environment first:" >&2
    echo "  uv venv --python 3.11 .venv" >&2
    echo "  uv pip install --python .venv/bin/python -e '.[gen,surface,dev]'" >&2
    exit 1
fi

export HF_HUB_DISABLE_PROGRESS_BARS=1
export TOKENIZERS_PARALLELISM=false

# Start clean: the assertions key runs by model name.
rm -rf "${RUNS_ROOT}" "${REPORT_DIR}"

echo "=== environment ==="
"${CLI}" --runs-root "${RUNS_ROOT}" info

echo
echo "=== test set provenance ==="
"${CLI}" verify-testset --directions de-en

echo
echo "=== 1/3 reference MT system: opus-mt-de-en ==="
"${CLI}" --runs-root "${RUNS_ROOT}" run \
    --model configs/models/opus-mt-de-en.yaml \
    --suite configs/suites/smoke-de-en.yaml \
    --limit "${LIMIT}" \
    --deterministic-check 8 \
    --set bench.subset_size=16 \
    --set bench.max_new_tokens=32 \
    --set bench.repeats=2 \
    --groups surface

echo
echo "=== 2/3 half precision, to exercise the compression path ==="
"${CLI}" --runs-root "${RUNS_ROOT}" run \
    --model configs/models/opus-mt-de-en-fp16.yaml \
    --suite configs/suites/smoke-de-en.yaml \
    --limit "${LIMIT}" \
    --set bench.subset_size=16 \
    --set bench.max_new_tokens=32 \
    --set bench.repeats=2 \
    --groups surface

echo
echo "=== 3/3 prompted decoder-only path: Qwen2.5-0.5B-Instruct ==="
"${CLI}" --runs-root "${RUNS_ROOT}" run \
    --model configs/models/qwen2.5-0.5b-instruct.yaml \
    --suite configs/suites/smoke-de-en.yaml \
    --limit "${LIMIT}" \
    --no-bench \
    --groups surface

echo
echo "=== neural metrics, if the COMET environment exists ==="
COMET_EXPECTED=0
if [[ -x ./.venv-comet/bin/mnlp-eval ]]; then
    COMET_EXPECTED=1
    # Smaller COMET batch to fit in 6 GB.
    ./.venv-comet/bin/mnlp-eval --runs-root "${RUNS_ROOT}" score \
        --groups neural --set comet_batch_size=8
else
    echo "skipped: .venv-comet not found, see README.md"
fi
export COMET_EXPECTED

echo
echo "=== report ==="
"${CLI}" --runs-root "${RUNS_ROOT}" report \
    --out "${REPORT_DIR}" --baseline opus-mt-de-en --formats md,csv

echo
echo "=== assertions ==="
"${PYTHON}" - "${RUNS_ROOT}" <<'PYEOF'
import json
import os
import sys
from pathlib import Path


runs_root = Path(sys.argv[1])
failures: list[str] = []


def check(condition: bool, message: str) -> None:
    print(f"  {'ok  ' if condition else 'FAIL'} {message}")
    if not condition:
        failures.append(message)


runs = {path.name.split("__")[0]: path for path in runs_root.iterdir() if path.is_dir()}
check(len(runs) == 3, f"three runs present (found {len(runs)}: {sorted(runs)})")

reference = runs.get("opus-mt-de-en")
if reference is None:
    failures.append("reference run missing")
else:
    scores = json.loads((reference / "scores.surface.json").read_text())
    bleu = scores["aggregate"]["bleu"]
    behaviour = scores["aggregate"]["behaviour"]
    # A real MT system on WMT22 de-en scores around 30; a broken pipeline would
    # be in single digits.
    check(bleu > 20, f"reference BLEU is plausible: {bleu}")
    check(behaviour["empty_rate"] == 0.0, "reference produced no empty output")
    check(behaviour["on_target_rate"] == 1.0, "reference stayed on target on every segment")
    check(
        abs(
            behaviour["on_target_rate"]
            + behaviour["off_target_rate"]
            + behaviour["unverifiable_rate"]
            - 1.0
        )
        < 1e-6,
        "language shares sum to 1, so a collapse cannot hide in the denominator",
    )
    check(behaviour["truncation_rate"] == 0.0, "reference truncated nothing")

    bench = json.loads((reference / "bench.json").read_text())
    bits = bench["static"]["bits_per_parameter_resident"]
    check(31.0 < bits < 33.0, f"float32 reads 32 bits per parameter: {bits}")

half = runs.get("opus-mt-de-en-fp16")
if half is not None and reference is not None:
    bench = json.loads((half / "bench.json").read_text())
    bits = bench["static"]["bits_per_parameter_resident"]
    check(15.0 < bits < 17.0, f"float16 reads 16 bits per parameter: {bits}")
    full = json.loads((reference / "bench.json").read_text())["static"]
    ratio = full["resident_weight_bytes"] / bench["static"]["resident_weight_bytes"]
    # Not exactly 2: some buffers stay fp32.
    check(1.95 < ratio < 2.05, f"fp16 halves resident size: {ratio:.4f}x")
    half_bleu = json.loads((half / "scores.surface.json").read_text())["aggregate"]["bleu"]
    reference_bleu = json.loads((reference / "scores.surface.json").read_text())[
        "aggregate"
    ]["bleu"]
    check(
        abs(half_bleu - reference_bleu) < 1.0,
        f"fp16 quality matches fp32: {half_bleu} against {reference_bleu}",
    )

# score exits 0 when a metric group is unavailable, so check the files exist.
if os.environ.get("COMET_EXPECTED") == "1":
    for name, path in sorted(runs.items()):
        scores = path / "scores.neural.json"
        if not scores.is_file():
            check(False, f"{name}: scores.neural.json is missing")
            continue
        payload = json.loads(scores.read_text())
        value = payload["aggregate"].get("wmt22_comet_da")
        check(
            isinstance(value, float) and 0.0 < value < 1.0,
            f"{name}: COMET-22 is a plausible score: {value}",
        )

prompted = runs.get("qwen2.5-0.5b-instruct")
if prompted is not None:
    scores = json.loads((prompted / "scores.surface.json").read_text())
    behaviour = scores["aggregate"]["behaviour"]
    # The prompted model keeps talking after the translation; extraction should
    # still recover a hypothesis.
    check(behaviour["empty_rate"] == 0.0, "prompted model yielded parseable output")
    check(behaviour["wasted_token_fraction"] > 0.1, "trailing commentary was detected")

if failures:
    print(f"\n{len(failures)} assertion(s) failed")
    raise SystemExit(1)
print("\nsmoke test passed")
PYEOF

echo
echo "Runs in ${RUNS_ROOT}, report in ${REPORT_DIR}/report.md"
