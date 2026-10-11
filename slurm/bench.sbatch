#!/usr/bin/env bash
#SBATCH --job-name=mnlp-bench
#SBATCH --partition=gpu_a100
#SBATCH --gpus=4
#SBATCH --cpus-per-task=72
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --output=slurm-logs/bench-%j.out
#
# Efficiency measurement, once per model, on an exclusive node so other jobs
# do not affect latency.
set -euo pipefail

module load 2025
module load Python/3.13.1-GCCcore-14.2.0

cd "${SLURM_SUBMIT_DIR}"

: "${MODEL_CONFIG:?set by slurm/submit_sweep.sh}"
: "${SUITE_CONFIG:?set by slurm/submit_sweep.sh}"

export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TOKENIZERS_PARALLELISM=false
# Single-device latency: keep device_map from spreading the model.
export CUDA_VISIBLE_DEVICES=0

./.venv/bin/mnlp-eval bench \
    --model "${MODEL_CONFIG}" \
    --suite "${SUITE_CONFIG}" \
    ${BENCH_CONFIG:+--bench "${BENCH_CONFIG}"}
