#!/usr/bin/env bash
#SBATCH --job-name=mnlp-pilot
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --time=02:00:00
#SBATCH --output=slurm-logs/pilot-%j.out
#
# Generate every direction of a reduced suite in one job. The run id is the
# same as generate.sbatch would give.
set -euo pipefail

module load 2025
module load Python/3.13.1-GCCcore-14.2.0

cd "${SLURM_SUBMIT_DIR}"

: "${MODEL_CONFIG:?set by slurm/submit_pilot.sh}"
: "${SUITE_CONFIG:?set by slurm/submit_pilot.sh}"

export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-1}
export TOKENIZERS_PARALLELISM=false

echo "model=${MODEL_CONFIG} suite=${SUITE_CONFIG}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

./.venv/bin/mnlp-eval generate \
    --model "${MODEL_CONFIG}" \
    --suite "${SUITE_CONFIG}" \
    --deterministic-check "${DETERMINISTIC_CHECK:-0}"
