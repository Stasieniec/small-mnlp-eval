#!/usr/bin/env bash
#SBATCH --job-name=mnlp-calibration-gen
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --time=01:00:00
#SBATCH --output=slurm-logs/calibration-gen-%j.out
# Generate the union once with the dense model; resubmit to resume directions.
set -euo pipefail

module load 2025
module load Python/3.13.1-GCCcore-14.2.0
cd "${SLURM_SUBMIT_DIR}"

CALIBRATION_INPUT=${CALIBRATION_INPUT:-data/calibration/multi-10dir}
CALIBRATION_OUT=${CALIBRATION_OUT:-data/calibration/multi-10dir-generated}
DENSE_MODEL_CONFIG=${DENSE_MODEL_CONFIG:-configs/models/alma-7b.yaml}
DECODE_SUITE=${DECODE_SUITE:-configs/suites/alma10-greedy.yaml}

# A submission shell may not export HF_HOME. Resolve the repository's standard
# scratch cache explicitly instead of silently falling back to ~/.cache.
export HF_HOME=${HF_HOME:-/scratch-shared/${USER}/hf_home}
export HF_HUB_CACHE=${HF_HOME}/hub
printf 'HF_HOME=%s HF_HUB_CACHE=%s\n' "${HF_HOME}" "${HF_HUB_CACHE}"
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-1}
export TOKENIZERS_PARALLELISM=false

./.venv/bin/mnlp-eval calibration-generate \
    --calibration "${CALIBRATION_INPUT}" \
    --model "${DENSE_MODEL_CONFIG}" \
    --decode-suite "${DECODE_SUITE}" \
    --out "${CALIBRATION_OUT}"
