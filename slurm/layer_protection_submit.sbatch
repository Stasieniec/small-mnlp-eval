#!/usr/bin/env bash
#SBATCH --job-name=boundary-prepare
#SBATCH --partition=rome
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=slurm-logs/boundary-prepare-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
# Preparation and submission logs also stay in a file.
bash scripts/run_layer_protection.sh run "${1:?Provide a fresh experiment directory}"
