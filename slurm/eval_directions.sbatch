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
# slurm/submit_eval.sh submits each system twice, to gpu_a100 and gpu_h100,
# because Snellius refuses a shared job naming both partitions. The first twin
# to start takes grid-state/full/locks/<name>, cancels the other while it is
# pending, and runs; a twin that starts anyway finds the lock held by a live job
# and exits 0. A lock whose owner has left the queue is stale and taken over.
# H100 nodes generate at about two thirds of A100 speed here (host-bound), which
# still beats waiting in a full A100 queue.
set -euo pipefail
module load 2025
module load Python/3.13.1-GCCcore-14.2.0
cd "${SLURM_SUBMIT_DIR}"
export HF_HOME=${HF_HOME:-/scratch-shared/${USER}/hf_home}
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
: "${MODEL_CONFIG:?path to a model config}"
: "${SUITE_CONFIG:?path to a suite config}"
METRICS_CONFIG=${METRICS_CONFIG:-configs/metrics/default.yaml}

LOCKS=grid-state/full/locks
LOCK=${LOCKS}/${SLURM_JOB_NAME}
mkdir -p "${LOCKS}"
if ! mkdir "${LOCK}" 2>/dev/null; then
    OWNER=$(cat "${LOCK}/owner" 2>/dev/null || true)
    if [[ -z "${OWNER}" ]]; then
        # The twin may have made the lock and not yet written its id.
        sleep 10
        OWNER=$(cat "${LOCK}/owner" 2>/dev/null || true)
    fi
    if [[ -n "${OWNER}" && "${OWNER}" != "${SLURM_JOB_ID}" ]] \
        && squeue -h -j "${OWNER}" 2>/dev/null | grep -q .; then
        echo "twin ${OWNER} holds ${LOCK}; nothing to do"
        exit 0
    fi
    echo "taking over stale lock ${LOCK} from ${OWNER:-unknown}"
fi
echo "${SLURM_JOB_ID}" > "${LOCK}/owner"
trap 'rm -rf "${LOCK}"' EXIT
scancel -u "${USER}" --name="${SLURM_JOB_NAME}" --state=PENDING 2>/dev/null \
    || echo "warning: could not cancel the pending twin" >&2

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
