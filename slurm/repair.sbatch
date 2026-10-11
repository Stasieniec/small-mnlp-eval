#!/usr/bin/env bash
#SBATCH --job-name=mnlp-repair
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --time=08:00:00
#SBATCH --output=slurm-logs/repair-%j.out
#
# LoRA-repair one pruned checkpoint and merge the adapter into it.
#
# One epoch over ALMA-Human-Parallel (about 100k sequences), roughly one to two
# hours on an A100. Exits non-zero if held-out loss did not fall, so afterok
# jobs do not run.
set -euo pipefail

module load 2025
module load Python/3.13.1-GCCcore-14.2.0

cd "${SLURM_SUBMIT_DIR}"

: "${REPAIR_CONFIG:?set by slurm/submit_repair.sh}"

REPAIR_OUT=${REPAIR_OUT:-/scratch-shared/${USER}/checkpoints}
mkdir -p "${REPAIR_OUT}"

export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-1}
export TOKENIZERS_PARALLELISM=false

echo "repair=${REPAIR_CONFIG} out=${REPAIR_OUT}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

./.venv/bin/mnlp-eval repair \
    --spec "${REPAIR_CONFIG}" \
    --out "${REPAIR_OUT}" \
    --model-config-dir configs/models
