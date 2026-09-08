#!/usr/bin/env bash
#SBATCH --job-name=mnlp-generate
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --time=08:00:00
#SBATCH --output=slurm-logs/generate-%A_%a.out
#
# Generation and efficiency measurement for one model, one direction per array
# task. Submit through slurm/submit_sweep.sh, which sets MODEL_CONFIG,
# SUITE_CONFIG and DIRECTIONS and wires up the dependency on scoring.
set -euo pipefail

module load 2023
module load Python/3.11.3-GCCcore-12.3.0

cd "${SLURM_SUBMIT_DIR}"
mkdir -p slurm-logs

# Compute nodes have no network. Everything must already be in HF_HOME, which
# slurm/prefetch.sh populates from a login node.
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

readarray -t PAIRS <<< "${DIRECTIONS//,/$'\n'}"
DIRECTION="${PAIRS[${SLURM_ARRAY_TASK_ID:-0}]}"

echo "model=${MODEL_CONFIG} suite=${SUITE_CONFIG} direction=${DIRECTION}"

./.venv/bin/mnlp-eval generate \
    --model "${MODEL_CONFIG}" \
    --suite "${SUITE_CONFIG}" \
    --directions "${DIRECTION}" \
    --deterministic-check "${DETERMINISTIC_CHECK:-0}"

# Efficiency is measured once per model, on the first array task only, so the
# measurement is not repeated six times and does not compete with itself.
if [[ "${SLURM_ARRAY_TASK_ID:-0}" == "0" && "${SKIP_BENCH:-0}" != "1" ]]; then
    ./.venv/bin/mnlp-eval bench \
        --model "${MODEL_CONFIG}" \
        --suite "${SUITE_CONFIG}" \
        --bench-direction "${DIRECTION}"
fi
