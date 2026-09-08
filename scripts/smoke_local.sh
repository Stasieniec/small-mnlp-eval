#!/usr/bin/env bash
# End-to-end check on a small local slice. Runs on 6 GB of VRAM in a couple of
# minutes, and on CPU given patience.
#
# Two models on purpose. opus-mt-de-en is a real MT system, so its scores prove
# the metric pipeline is wired correctly; a harness that has only ever seen a
# prompted 0.5B model cannot distinguish "pipeline correct" from "pipeline
# broken". Qwen2.5-0.5B-Instruct then exercises the prompted decoder-only path,
# including hypothesis extraction from a model that keeps talking past the
# translation.
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
if [[ -x ./.venv-comet/bin/mnlp-eval ]]; then
    # A smaller COMET batch keeps the 580M metric model inside 6 GB alongside
    # whatever else the card is holding.
    ./.venv-comet/bin/mnlp-eval --runs-root "${RUNS_ROOT}" score \
        --groups neural --set comet_batch_size=8
else
    echo "skipped: .venv-comet not found, see docs/environments.md"
fi

echo
echo "=== report ==="
"${CLI}" --runs-root "${RUNS_ROOT}" report \
    --out "${REPORT_DIR}" --baseline opus-mt-de-en --formats md,csv

echo
echo "=== assertions ==="
"${PYTHON}" - "${RUNS_ROOT}" <<'PYEOF'
import json
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
    # A real MT system on WMT22 de-en lands near the low thirties. A broken
    # pipeline, a mangled prompt or a misaligned reference file would show up
    # as single digits, which is the failure this threshold catches.
    check(bleu > 20, f"reference BLEU is plausible: {bleu}")
    check(behaviour["empty_rate"] == 0.0, "reference produced no empty output")
    check(behaviour["off_target_rate"] == 0.0, "reference stayed on target")
    check(behaviour["truncation_rate"] == 0.0, "reference truncated nothing")

    bench = json.loads((reference / "bench.json").read_text())
    bits = bench["static"]["bits_per_parameter_resident"]
    # float32 must read 32 bits per parameter. If this drifts, the size
    # accounting that every compression ratio depends on is wrong.
    check(31.0 < bits < 33.0, f"float32 reads 32 bits per parameter: {bits}")

half = runs.get("opus-mt-de-en-fp16")
if half is not None:
    bench = json.loads((half / "bench.json").read_text())
    bits = bench["static"]["bits_per_parameter_resident"]
    check(15.0 < bits < 17.0, f"float16 reads 16 bits per parameter: {bits}")

prompted = runs.get("qwen2.5-0.5b-instruct")
if prompted is not None:
    scores = json.loads((prompted / "scores.surface.json").read_text())
    behaviour = scores["aggregate"]["behaviour"]
    # The prompted model is not fine-tuned to stop after the translation, so it
    # fills the token budget with commentary. Extraction should recover a clean
    # hypothesis regardless, which is the whole point of that code path.
    check(behaviour["empty_rate"] == 0.0, "prompted model yielded parseable output")
    check(behaviour["wasted_token_fraction"] > 0.1, "trailing commentary was detected")

if failures:
    print(f"\n{len(failures)} assertion(s) failed")
    raise SystemExit(1)
print("\nsmoke test passed")
PYEOF

echo
echo "Runs in ${RUNS_ROOT}, report in ${REPORT_DIR}/report.md"
