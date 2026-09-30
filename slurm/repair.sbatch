#!/usr/bin/env bash
#SBATCH --job-name=mnlp-repair
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --time=08:00:00
#SBATCH --output=slurm-logs/repair-%j.out
#
# LoRA-repair one pruned checkpoint and merge the adapter into it. Once per
# repair config, after the prune job that wrote its source.
#
# The recipe is one epoch over every eligible segment of ALMA-Human-Parallel,
# about a hundred thousand sequences. On one A100 that is an estimated one to
# two hours for a 20 percent model and less for 50, with gradient
# checkpointing keeping memory well under 40 GB. The eight-hour limit is
# headroom for that estimate being wrong. The job prints a progress line every
# ten optimizer steps with throughput and elapsed time, so the real figure is
# in the log after the first few minutes.
#
# The job exits non-zero if held-out loss did not fall, after writing the
# checkpoint and repair.json, so evaluation chained behind it with afterok
# does not run on a repair that did nothing.
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
