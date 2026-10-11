#!/usr/bin/env bash
#SBATCH --job-name=mnlp-report
#SBATCH --partition=rome
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --output=slurm-logs/report-%j.out
#
# Reads JSON only, no GPU. rome bills in eighths of a node, so 16 cores.
set -euo pipefail

module load 2025
module load Python/3.13.1-GCCcore-14.2.0

cd "${SLURM_SUBMIT_DIR}"

: "${BASELINE:?name of the uncompressed system, for ratios and significance}"

./.venv/bin/mnlp-eval report \
    --baseline "${BASELINE}" \
    --formats md,csv,tex \
    --metrics "${METRICS_CONFIG:-configs/metrics/default.yaml}"
