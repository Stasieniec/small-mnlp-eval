#!/usr/bin/env bash
#SBATCH --job-name=mnlp-prune
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --time=04:00:00
#SBATCH --output=slurm-logs/prune-%j.out
#
# Select and compact one subnetwork, once per pruning config.
#
# Memory: the dense model is 13.5 GB in bfloat16; SlimGPT adds the cached layer
# inputs and one layer's Hessians (down_proj's is 11008 x 11008 float32). If it
# does not fit on a 40 GB A100, lower segments_per_direction, not max_length.
set -euo pipefail

module load 2025
module load Python/3.13.1-GCCcore-14.2.0

cd "${SLURM_SUBMIT_DIR}"

: "${PRUNE_CONFIG:?set by slurm/submit_prune.sh}"

# Checkpoints go to scratch; the emitted model config records this absolute path.
PRUNE_OUT=${PRUNE_OUT:-/scratch-shared/${USER}/checkpoints}
mkdir -p "${PRUNE_OUT}"

# Weights come from slurm/prefetch.sh, calibration data from `mnlp-eval calibration`.
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-1}
export TOKENIZERS_PARALLELISM=false

echo "prune=${PRUNE_CONFIG} out=${PRUNE_OUT}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# Descriptors go into the repository, where `mnlp-eval overlap` reads them.
./.venv/bin/mnlp-eval prune \
    --spec "${PRUNE_CONFIG}" \
    --out "${PRUNE_OUT}" \
    --subnetwork-dir subnetworks \
    --model-config-dir configs/models
