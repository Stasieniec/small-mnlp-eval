#!/usr/bin/env bash
#SBATCH --job-name=mnlp-report
#SBATCH --partition=rome
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --output=slurm-logs/report-%j.out
#
# Reporting reads JSON only, so it needs no GPU. Sixteen cores because the
# rome partition bills in eighths of a 128-core node and four would cost the
# same as sixteen.
set -euo pipefail

module load 2025
module load Python/3.12.3-GCCcore-13.3.0

cd "${SLURM_SUBMIT_DIR}"

: "${BASELINE:?name of the uncompressed system, for ratios and significance}"

./.venv/bin/mnlp-eval report \
    --baseline "${BASELINE}" \
    --formats md,csv,tex \
    --metrics "${METRICS_CONFIG:-configs/metrics/default.yaml}"
