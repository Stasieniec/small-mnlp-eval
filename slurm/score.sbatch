#!/usr/bin/env bash
#SBATCH --job-name=mnlp-score
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --time=04:00:00
#SBATCH --output=slurm-logs/score-%j.out
#
# COMET and MetricX need their own Python environments (see README.md).
# Scoped to one run directory with --run.
set -euo pipefail

module load 2025
module load Python/3.13.1-GCCcore-14.2.0

cd "${SLURM_SUBMIT_DIR}"

: "${MODEL_CONFIG:?set by slurm/submit_sweep.sh}"
: "${SUITE_CONFIG:?set by slurm/submit_sweep.sh}"

export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TOKENIZERS_PARALLELISM=false

METRICS_CONFIG="${METRICS_CONFIG:-configs/metrics/default.yaml}"
RUN_DIR=$(./.venv/bin/mnlp-eval run-dir --model "${MODEL_CONFIG}" --suite "${SUITE_CONFIG}")
echo "scoring ${RUN_DIR}"

# Surface metrics only need sacreBLEU.
./.venv/bin/mnlp-eval score --run "${RUN_DIR}" --groups surface --metrics "${METRICS_CONFIG}"

if [[ -x ./.venv-comet/bin/mnlp-eval ]]; then
    ./.venv-comet/bin/mnlp-eval score --run "${RUN_DIR}" --groups neural \
        --metrics "${METRICS_CONFIG}"
fi

if [[ -x ./.venv-metricx/bin/mnlp-eval ]]; then
    PYTHONPATH="${METRICX_SRC:-$PWD/third_party/metricx}" \
        ./.venv-metricx/bin/mnlp-eval score --run "${RUN_DIR}" --groups metricx \
        --metrics "${METRICS_CONFIG}"
fi
