#!/usr/bin/env bash
#SBATCH --job-name=mnlp-report
#SBATCH --partition=rome
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=slurm-logs/report-%j.out
#
# Reporting reads JSON only, so it needs no GPU.
set -euo pipefail

module load 2023
module load Python/3.11.3-GCCcore-12.3.0

cd "${SLURM_SUBMIT_DIR}"
mkdir -p slurm-logs

./.venv/bin/mnlp-eval report \
    --baseline "${BASELINE:-alma-7b-r}" \
    --formats md,csv,tex \
    --metrics "${METRICS_CONFIG:-configs/metrics/default.yaml}"
