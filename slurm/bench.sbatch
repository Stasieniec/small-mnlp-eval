#!/usr/bin/env bash
#SBATCH --job-name=mnlp-bench
#SBATCH --partition=gpu_a100
#SBATCH --gpus=4
#SBATCH --cpus-per-task=72
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --output=slurm-logs/bench-%j.out
#
# Efficiency measurement, once per model, on an exclusive node.
#
# Exclusive because latency is the measurement. A quarter-node allocation
# shares the physical GPU node with up to three other jobs, and the resulting
# contention moves throughput by more than some compression methods do. The
# run records SM clock, temperature and power draw so a throttled measurement
# is at least visible, but the right fix is not to share.
#
# Separate from generation so it is measured once rather than once per
# direction, and so generation can keep using a quarter node.
set -euo pipefail

module load 2025
module load Python/3.12.3-GCCcore-13.3.0

cd "${SLURM_SUBMIT_DIR}"

: "${MODEL_CONFIG:?set by slurm/submit_sweep.sh}"
: "${SUITE_CONFIG:?set by slurm/submit_sweep.sh}"

export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TOKENIZERS_PARALLELISM=false
# Pin to one GPU: the framework measures single-device latency, and leaving
# four visible would let a device_map spread the model and change what is
# being measured.
export CUDA_VISIBLE_DEVICES=0

./.venv/bin/mnlp-eval bench \
    --model "${MODEL_CONFIG}" \
    --suite "${SUITE_CONFIG}" \
    ${BENCH_CONFIG:+--bench "${BENCH_CONFIG}"}
