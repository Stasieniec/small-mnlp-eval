#!/usr/bin/env bash
# Download every model, dataset and metric checkpoint into HF_HOME.
# Run this on a LOGIN NODE. Compute nodes have no internet access.
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ -z "${HF_HOME:-}" ]]; then
    echo "HF_HOME is not set. Point it at scratch, for example:" >&2
    echo "  export HF_HOME=/scratch-shared/\$USER/hf_home" >&2
    exit 1
fi
mkdir -p "$HF_HOME"
unset HF_HUB_OFFLINE || true

MODELS=(configs/models/*.yaml)
SUITES=(configs/suites/wmt22-6dir-beam5.yaml configs/suites/wmt22-6dir-greedy.yaml)

echo "Prefetching models and datasets into $HF_HOME"
./.venv/bin/mnlp-eval prefetch --models "${MODELS[@]}" --suites "${SUITES[@]}" --no-metrics

if [[ -x ./.venv-comet/bin/mnlp-eval ]]; then
    echo
    echo "Prefetching COMET checkpoints"
    if [[ -z "${HF_TOKEN:-}" ]]; then
        echo "  HF_TOKEN is not set; gated checkpoints (COMETKiwi, XCOMET) will be skipped." >&2
    fi
    ./.venv-comet/bin/mnlp-eval prefetch --metrics configs/metrics/full.yaml
else
    echo "COMET environment not found, skipping. See docs/environments.md."
fi

if [[ -x ./.venv-metricx/bin/mnlp-eval ]]; then
    echo
    echo "Prefetching MetricX checkpoints"
    ./.venv-metricx/bin/mnlp-eval prefetch --metrics configs/metrics/full.yaml
fi

echo
echo "Done. Batch jobs can now run with HF_HUB_OFFLINE=1."
