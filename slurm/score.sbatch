#!/usr/bin/env bash
#SBATCH --job-name=mnlp-score
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --time=04:00:00
#SBATCH --output=slurm-logs/score-%j.out
#
# Scoring runs as its own job because COMET and MetricX cannot share a Python
# environment with the generation stack. See docs/environments.md.
set -euo pipefail

module load 2023
module load Python/3.11.3-GCCcore-12.3.0

cd "${SLURM_SUBMIT_DIR}"
mkdir -p slurm-logs

export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

METRICS_CONFIG="${METRICS_CONFIG:-configs/metrics/default.yaml}"

# Surface metrics need nothing beyond sacreBLEU, so they run in whichever
# environment is present.
./.venv/bin/mnlp-eval score --groups surface --metrics "${METRICS_CONFIG}"

if [[ -x ./.venv-comet/bin/mnlp-eval ]]; then
    ./.venv-comet/bin/mnlp-eval score --groups neural --metrics "${METRICS_CONFIG}"
fi

if [[ -x ./.venv-metricx/bin/mnlp-eval ]]; then
    ./.venv-metricx/bin/mnlp-eval score --groups metricx --metrics "${METRICS_CONFIG}"
fi
