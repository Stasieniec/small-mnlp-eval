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
# Every suite, including wmt22-alma-repro. Omitting it meant the four extra
# directions it adds (cs, zh both ways) were never downloaded, so the
# framework's own correctness evidence failed offline on direction two.
SUITES=(configs/suites/*.yaml)

echo "Prefetching models and datasets into $HF_HOME"
# --strict so a failed download is an error here rather than a surprise hours
# later inside a GPU job.
./.venv/bin/mnlp-eval prefetch --models "${MODELS[@]}" --suites "${SUITES[@]}" \
    --no-metrics --strict

if [[ -x ./.venv-comet/bin/mnlp-eval ]]; then
    echo
    echo "Prefetching COMET checkpoints"
    if [[ -z "${HF_TOKEN:-}" ]]; then
        echo "  HF_TOKEN is not set; gated checkpoints (COMETKiwi, XCOMET) will be skipped." >&2
    fi
    ./.venv-comet/bin/mnlp-eval prefetch --metrics configs/metrics/full.yaml --strict
else
    echo "COMET environment not found, skipping. See docs/environments.md."
fi

if [[ -x ./.venv-metricx/bin/mnlp-eval ]]; then
    echo
    echo "Prefetching MetricX checkpoints"
    ./.venv-metricx/bin/mnlp-eval prefetch --metrics configs/metrics/full.yaml --strict
fi

echo
echo "Done. Batch jobs can now run with HF_HUB_OFFLINE=1."
