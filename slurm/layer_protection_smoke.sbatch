#!/usr/bin/env bash
#SBATCH --job-name=boundary-smoke
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00
#SBATCH --output=slurm-logs/boundary-smoke-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source scripts/layer_protection_env.sh
for interpreter in .venv/bin/python .venv-comet/bin/python; do
    check_python_headers "$interpreter"
    "$interpreter" scripts/layer_protection_gpu_smoke.py
done
