#!/usr/bin/env bash
#SBATCH --job-name=grid
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=03:00:00
#SBATCH --output=slurm-logs/grid/%x-%j.out
#
# One model of the final grid, start to finish: prune (or LoRA repair), then
# generate all ten directions of the suite, then score with surface metrics
# and COMET. Submitted by scripts/grid.py, which sets the partition, cores and
# time on the command line and passes GRID_NAME, GRID_MODE (prune|repair),
# PRUNE_CONFIG or REPAIR_CONFIG, MODEL_CONFIG and SUITE_CONFIG.
#
# Each model is submitted twice, to gpu_a100 and gpu_h100, because Snellius
# rejects a shared job that names both partitions. The first twin to start
# takes grid-state/locks/<name> and cancels the other while it is pending; a
# twin that starts anyway sees the lock held by a live job and exits 0. A lock
# whose owner has left the queue is stale and taken over.
#
# Every stage is skipped when its output is complete, so a resubmitted model
# resumes where it stopped. Progress goes to grid-state/status/<name>.json.
#
# GRID_DRY_RUN=1 takes the lock and writes status as usual but echoes each
# stage's commands instead of running them.
set -Eeuo pipefail

cd "${SLURM_SUBMIT_DIR:?run through sbatch, or set SLURM_SUBMIT_DIR}"
DRY_RUN=${GRID_DRY_RUN:-0}

if [[ "${DRY_RUN}" != 1 ]]; then
    module load 2025
    module load Python/3.13.1-GCCcore-14.2.0
fi

: "${SLURM_JOB_ID:?}"
: "${GRID_NAME:?set by scripts/grid.py}"
: "${GRID_MODE:?set by scripts/grid.py}"
: "${MODEL_CONFIG:?set by scripts/grid.py}"
: "${SUITE_CONFIG:?set by scripts/grid.py}"
case "${GRID_MODE}" in
    prune) : "${PRUNE_CONFIG:?set by scripts/grid.py}" ;;
    repair) : "${REPAIR_CONFIG:?set by scripts/grid.py}" ;;
    *)
        echo "error: GRID_MODE must be prune or repair, got ${GRID_MODE}" >&2
        exit 1
        ;;
esac
if [[ "$(basename "${MODEL_CONFIG}" .yaml)" != "${GRID_NAME}" ]]; then
    echo "error: MODEL_CONFIG ${MODEL_CONFIG} does not belong to ${GRID_NAME}" >&2
    exit 1
fi

export PYTHONUNBUFFERED=1
export HF_HOME=${HF_HOME:-/scratch-shared/${USER}/hf_home}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-1}
export TOKENIZERS_PARALLELISM=false

export GRID_STATE_DIR=${GRID_STATE_DIR:-grid-state}
if [[ -z "${GRID_LOG:-}" ]]; then
    GRID_LOG=$(scontrol show job "${SLURM_JOB_ID}" 2>/dev/null \
        | grep -o 'StdOut=[^ ]*' | cut -d= -f2) || true
    GRID_LOG=${GRID_LOG:-${PWD}/slurm-logs/grid/g-${GRID_NAME}-${SLURM_JOB_ID}.out}
fi
export GRID_LOG
export GRID_DRY_RUN=${DRY_RUN}
PRUNE_OUT=${PRUNE_OUT:-/scratch-shared/${USER}/checkpoints}
CHECKPOINT=${PRUNE_OUT%/}/${GRID_NAME}
SUBNETWORK_DIR=${GRID_SUBNETWORK_DIR:-subnetworks}
CONFIG_DIR=$(dirname "${MODEL_CONFIG}")
METRICS_CONFIG=${METRICS_CONFIG:-configs/metrics/default.yaml}
STATUS_DIR=${GRID_STATE_DIR}/status
CLI=${GRID_CLI:-./.venv/bin/mnlp-eval}
COMET_CLI=${GRID_COMET_CLI:-./.venv-comet/bin/mnlp-eval}
GRID=("${GRID_PYTHON:-./.venv/bin/python}" scripts/grid.py)
mkdir -p "${STATUS_DIR}"

echo "grid ${GRID_NAME} mode=${GRID_MODE} job=${SLURM_JOB_ID}" \
    "partition=${SLURM_JOB_PARTITION:-?} node=${SLURMD_NODENAME:-?} dry_run=${DRY_RUN}"

# --- Twin lock ---------------------------------------------------------------

LOCK_STATUS=0
"${GRID[@]}" lock "${GRID_NAME}" || LOCK_STATUS=$?
if (( LOCK_STATUS == 10 )); then
    echo "the twin of this job holds the lock; nothing to do"
    exit 0
elif (( LOCK_STATUS != 0 )); then
    exit "${LOCK_STATUS}"
fi

STAGE=start
on_exit() {
    local code=$?
    trap - EXIT
    if (( code != 0 )); then
        echo "failed in stage ${STAGE} with exit ${code}" >&2
        "${GRID[@]}" mark "${GRID_NAME}" --state failed --stage "${STAGE}" \
            --exit-code "${code}" || true
    fi
    "${GRID[@]}" unlock "${GRID_NAME}" || true
    exit "${code}"
}
trap on_exit EXIT
trap 'echo "error: line ${LINENO}: ${BASH_COMMAND} exited $?" >&2' ERR
# Time limit or scancel. Slurm signals the whole job, so the CLI below has
# already been told to stop; this records where.
trap 'echo "received SIGTERM" >&2; exit 143' TERM

# The twin is pending on the other partition. Cancelling it is an
# optimisation: if it starts anyway it finds this job alive and exits.
scancel -u "${USER}" --name="g-${GRID_NAME}" --state=PENDING \
    || echo "warning: could not cancel the pending twin" >&2

nvidia-smi || true
GRID_GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n 1) \
    || GRID_GPU=unknown
export GRID_GPU=${GRID_GPU:-unknown}
"${GRID[@]}" mark "${GRID_NAME}" --state running --stage start

# --- Helpers -----------------------------------------------------------------

run() {
    if [[ "${DRY_RUN}" == 1 ]]; then
        echo "[dry-run] $(printf '%q ' "$@")"
    else
        "$@"
    fi
}

STAGE_START=0
begin() {
    STAGE=$1
    STAGE_START=${SECONDS}
    echo "== ${STAGE} $(date '+%F %T')"
    "${GRID[@]}" mark "${GRID_NAME}" --state running --stage "${STAGE}"
}

finish() {
    "${GRID[@]}" mark "${GRID_NAME}" --state running --stage "${STAGE}" \
        --seconds "${STAGE}=$(( SECONDS - STAGE_START ))"
}

skip() {
    echo "== $1 already complete, skipping"
    "${GRID[@]}" mark "${GRID_NAME}" --skipped "$1"
}

# Removes a partial checkpoint before pruning again, so no shard of an earlier
# attempt survives into the new one. Refuses anything but <PRUNE_OUT>/<name>
# under the user's own scratch.
remove_partial() {
    local target=$1 parent scratch
    if [[ ! "${GRID_NAME}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
        echo "error: refusing to remove ${target}: unexpected name" >&2
        return 1
    fi
    parent=$(cd "$(dirname "${target}")" && pwd -P)
    # GRID_SCRATCH_ROOT exists for tests/test_grid.py, which prunes into a tmp dir.
    scratch=${GRID_SCRATCH_ROOT:-/scratch-shared/${USER}}
    scratch=$(cd "${scratch}" 2>/dev/null && pwd -P || echo "${scratch}")
    if [[ "${parent}" != "$(cd "${PRUNE_OUT}" && pwd -P)" ]]; then
        echo "error: refusing to remove ${target}: not directly in ${PRUNE_OUT}" >&2
        return 1
    fi
    echo "removing partial checkpoint ${target}"
    if [[ "${DRY_RUN}" == 1 ]]; then
        echo "[dry-run] rm -rf -- ${parent}/${GRID_NAME}"
        return 0
    fi
    if [[ "${parent}" != "${scratch}"/* ]]; then
        echo "error: refusing to remove ${target}: not under ${scratch}" >&2
        return 1
    fi
    rm -rf -- "${parent:?}/${GRID_NAME:?}"
}

# Runs a CLI command that prints its JSON manifest last on stdout, and keeps
# that manifest only if the command exited 0. Its existence is the stage's
# completion marker.
run_with_manifest() {
    local marker=$1
    shift
    if [[ "${DRY_RUN}" == 1 ]]; then
        run "$@"
        return 0
    fi
    local captured="${marker%.json}.stdout"
    "$@" | tee "${captured}"
    "${GRID[@]}" save-manifest --stdout "${captured}" --out "${marker}"
    rm -f -- "${captured}"
}

# --- (a) prune or repair -----------------------------------------------------

REBUILT=0

if [[ "${GRID_MODE}" == prune ]]; then
    MARKER=${STATUS_DIR}/${GRID_NAME}.prune.json
    CHECK=(check prune --checkpoint "${CHECKPOINT}" --model-config "${MODEL_CONFIG}"
        --subnetwork "${SUBNETWORK_DIR}/${GRID_NAME}.json" --marker "${MARKER}")
    if "${GRID[@]}" "${CHECK[@]}"; then
        skip prune
    else
        begin prune
        if [[ -e "${CHECKPOINT}" ]]; then
            remove_partial "${CHECKPOINT}"
        fi
        run rm -f -- "${MARKER}"
        run mkdir -p "${PRUNE_OUT}"
        run_with_manifest "${MARKER}" "${CLI}" prune \
            --spec "${PRUNE_CONFIG}" \
            --out "${PRUNE_OUT}" \
            --subnetwork-dir "${SUBNETWORK_DIR}" \
            --model-config-dir "${CONFIG_DIR}"
        [[ "${DRY_RUN}" == 1 ]] || "${GRID[@]}" "${CHECK[@]}"
        REBUILT=1
        finish
    fi
else
    MARKER=${STATUS_DIR}/${GRID_NAME}.repair.json
    CHECK=(check repair --checkpoint "${CHECKPOINT}" --model-config "${MODEL_CONFIG}"
        --marker "${MARKER}")
    if "${GRID[@]}" "${CHECK[@]}"; then
        skip repair
    else
        # repair replaces its own output directory, so nothing to remove here.
        begin repair
        run rm -f -- "${MARKER}"
        run mkdir -p "${PRUNE_OUT}"
        run_with_manifest "${MARKER}" "${CLI}" repair \
            --spec "${REPAIR_CONFIG}" \
            --out "${PRUNE_OUT}" \
            --model-config-dir "${CONFIG_DIR}"
        [[ "${DRY_RUN}" == 1 ]] || "${GRID[@]}" "${CHECK[@]}"
        REBUILT=1
        finish
    fi
fi

# --- (b) generate ------------------------------------------------------------

STAGE=run-dir
if [[ "${DRY_RUN}" == 1 && ! -f "${MODEL_CONFIG}" ]]; then
    RUN_DIR=runs/DRY-RUN__${GRID_NAME}
else
    RUN_DIR=$("${CLI}" run-dir --model "${MODEL_CONFIG}" --suite "${SUITE_CONFIG}")
fi
echo "run directory ${RUN_DIR}"
# Fresh weights make any hypotheses already on disk someone else's; generate
# would otherwise keep them and fill in only the missing directions. Moved out
# of runs/ so nothing that scans it finds two runs of one model.
if (( REBUILT )) && [[ -d "${RUN_DIR}" ]]; then
    STALE_ROOT=$(dirname "$(dirname "${RUN_DIR}")")/runs-stale
    STALE=${STALE_ROOT}/$(basename "${RUN_DIR}").${SLURM_JOB_ID}
    echo "moving ${RUN_DIR}, generated by earlier weights, to ${STALE}"
    run mkdir -p "${STALE_ROOT}"
    run mv -- "${RUN_DIR}" "${STALE}"
fi
"${GRID[@]}" mark "${GRID_NAME}" --run-dir "${RUN_DIR}"

if "${GRID[@]}" check generate --run-dir "${RUN_DIR}"; then
    skip generate
else
    begin generate
    # Skips any direction already on disk, so a resumed job only does the rest.
    run "${CLI}" generate --model "${MODEL_CONFIG}" --suite "${SUITE_CONFIG}"
    [[ "${DRY_RUN}" == 1 ]] || "${GRID[@]}" check generate --run-dir "${RUN_DIR}"
    finish
fi

# --- (c) score ---------------------------------------------------------------

# `score` skips a group whose file exists, and reports a group its environment
# cannot serve as pending with exit 0, so the file is checked afterwards.
score_group() {
    local cli=$1 group=$2 state=0
    local extra=()
    "${GRID[@]}" check score --run-dir "${RUN_DIR}" --group "${group}" || state=$?
    if (( state == 0 )); then
        skip "score_${group}"
        return 0
    fi
    if (( state == 2 )); then
        extra=(--overwrite)
    fi
    begin "score_${group}"
    run "${cli}" score --run "${RUN_DIR}" --groups "${group}" \
        --metrics "${METRICS_CONFIG}" "${extra[@]}"
    if [[ "${DRY_RUN}" != 1 ]] \
        && ! "${GRID[@]}" check score --run-dir "${RUN_DIR}" --group "${group}"; then
        echo "error: ${cli} wrote no scores.${group}.json" >&2
        return 1
    fi
    finish
}

score_group "${CLI}" surface
score_group "${COMET_CLI}" neural

STAGE=done
"${GRID[@]}" mark "${GRID_NAME}" --state done --stage done --run-dir "${RUN_DIR}"
echo "== ${GRID_NAME} done"
