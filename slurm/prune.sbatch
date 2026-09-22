#!/usr/bin/env bash
#SBATCH --job-name=mnlp-prune
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --time=04:00:00
#SBATCH --output=slurm-logs/prune-%j.out
#
# Select and compact one subnetwork. Once per pruning config, before any
# generation.
#
# One GPU, not an exclusive node: nothing here is a timing measurement, so
# sharing the node costs only wall clock. SlimGPT is the expensive one, at
# roughly an hour on ALMA-7B: two passes over the calibration set, a Hessian
# and a Cholesky inverse per projection, and a per-column compensation sweep.
# FLAP finishes in minutes. The four-hour limit covers the slowest with room
# for the 13.5 GB checkpoint write.
#
# Memory on one 40 GB A100 is comfortable but not spacious: the dense model is
# 13.5 GB in bfloat16, SlimGPT adds the cached layer inputs for the whole
# calibration set plus one layer's Hessians, and down_proj's Hessian alone is
# 11008 squared in float32. Lower segments_per_direction in the calibration
# config if it does not fit; do not lower max_length, which changes what the
# Hessian measures and stops the sweep being comparable across methods.
set -euo pipefail

module load 2025
module load Python/3.12.3-GCCcore-13.3.0

cd "${SLURM_SUBMIT_DIR}"

: "${PRUNE_CONFIG:?set by slurm/submit_prune.sh}"

# Checkpoints run to tens of gigabytes and the home quota is small, so they go
# to scratch. The emitted model config records this path, which is why it is
# resolved to an absolute one rather than left relative to the submit dir.
PRUNE_OUT=${PRUNE_OUT:-/scratch-shared/${USER}/checkpoints}
mkdir -p "${PRUNE_OUT}"

# The dense checkpoint and the calibration data are both already on disk: the
# weights from slurm/prefetch.sh and the calibration set from
# `mnlp-eval calibration` on a login node. A job that downloads mid-run is a
# job whose input may differ from the last one's.
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-1}
export TOKENIZERS_PARALLELISM=false

echo "prune=${PRUNE_CONFIG} out=${PRUNE_OUT}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# Descriptors stay in the repository: they are small, they are the only record
# of which positions a run chose, and `mnlp-eval overlap` reads them from here.
./.venv/bin/mnlp-eval prune \
    --spec "${PRUNE_CONFIG}" \
    --out "${PRUNE_OUT}" \
    --subnetwork-dir subnetworks \
    --model-config-dir configs/models
