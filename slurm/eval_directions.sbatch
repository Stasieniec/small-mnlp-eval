#!/usr/bin/env bash
#SBATCH --job-name=mnlp-eval
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --time=04:00:00
#SBATCH --output=slurm-logs/eval/%x-%j.out
#
# Generate and score one existing model on a suite, in one allocation,
# optionally on a subset of its directions. For the full-suite evaluation of
# chosen systems: a pair or direction specialist is evaluated only where it is
# meant to be used, a multi model on everything.
#
#   sbatch --export=ALL,MODEL_CONFIG=configs/models/<name>.yaml,\
#   SUITE_CONFIG=configs/suites/alma10-greedy.yaml,DIRECTIONS=de-en+en-de \
#       slurm/eval_directions.sbatch
#
# DIRECTIONS is joined with "+", because a comma inside --export ends the
# assignment (see generate.sbatch). Empty means every direction of the suite.
# --only-direction keeps the suite's run identity, so a subset run lands in the
# same run directory a full one would, and is scored with --allow-partial.
# A100 only: H100 nodes ran this generation at about half speed (host-bound).
set -euo pipefail
module load 2025
module load Python/3.13.1-GCCcore-14.2.0
cd "${SLURM_SUBMIT_DIR}"
export HF_HOME=${HF_HOME:-/scratch-shared/${USER}/hf_home}
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
: "${MODEL_CONFIG:?path to a model config}"
: "${SUITE_CONFIG:?path to a suite config}"
METRICS_CONFIG=${METRICS_CONFIG:-configs/metrics/default.yaml}

ONLY=()
PARTIAL=()
if [[ -n "${DIRECTIONS:-}" ]]; then
    IFS=+ read -r -a WANTED <<< "${DIRECTIONS}"
    for direction in "${WANTED[@]}"; do
        ONLY+=(--only-direction "${direction}")
    done
    PARTIAL=(--allow-partial)
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "model=${MODEL_CONFIG} suite=${SUITE_CONFIG} directions=${DIRECTIONS:-all} start=$(date +%T)"

./.venv/bin/mnlp-eval generate --model "${MODEL_CONFIG}" --suite "${SUITE_CONFIG}" "${ONLY[@]}"
echo "generated $(date +%T)"
RUN_DIR=$(./.venv/bin/mnlp-eval run-dir --model "${MODEL_CONFIG}" --suite "${SUITE_CONFIG}")
./.venv/bin/mnlp-eval score --run "${RUN_DIR}" --groups surface --metrics "${METRICS_CONFIG}" \
    "${PARTIAL[@]}"
./.venv-comet/bin/mnlp-eval score --run "${RUN_DIR}" --groups neural --metrics "${METRICS_CONFIG}" \
    "${PARTIAL[@]}"
echo "scored $(date +%T) ${RUN_DIR}"
