#!/usr/bin/env bash
#SBATCH --job-name=mnlp-generate
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --time=08:00:00
#SBATCH --output=slurm-logs/generate-%A_%a.out
#
# One job generates all directions with one model load. As an array, each
# task generates one direction of the same run.
#
# The direction is looked up from the suite here because a comma in an
# `sbatch --export` value splits it. Sharding uses --only-direction, which
# keeps the run id (--directions would change it).
set -euo pipefail

module load 2025
module load Python/3.13.1-GCCcore-14.2.0

cd "${SLURM_SUBMIT_DIR}"

: "${MODEL_CONFIG:?set by slurm/submit_sweep.sh}"
: "${SUITE_CONFIG:?set by slurm/submit_sweep.sh}"

# Offline; slurm/prefetch.sh downloads everything from a login node.
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-1}
export TOKENIZERS_PARALLELISM=false

DIRECTION_ARGS=()
DIRECTION=all
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    mapfile -t PAIRS < <(./.venv/bin/mnlp-eval suite-directions --suite "${SUITE_CONFIG}")
    INDEX=${SLURM_ARRAY_TASK_ID}
    if (( INDEX >= ${#PAIRS[@]} )); then
        echo "array task ${INDEX} exceeds the ${#PAIRS[@]} direction(s) in ${SUITE_CONFIG}" >&2
        exit 1
    fi
    DIRECTION="${PAIRS[INDEX]}"
    DIRECTION_ARGS=(--only-direction "${DIRECTION}")
fi

echo "model=${MODEL_CONFIG} suite=${SUITE_CONFIG} direction=${DIRECTION}"

./.venv/bin/mnlp-eval generate \
    --model "${MODEL_CONFIG}" \
    --suite "${SUITE_CONFIG}" \
    "${DIRECTION_ARGS[@]}" \
    --deterministic-check "${DETERMINISTIC_CHECK:-0}"
