#!/usr/bin/env bash
# Queue one repair job per config.
#
#   bash slurm/submit_repair.sh configs/repair/slimgpt-20-multi-lora.yaml [more.yaml ...]
#
# AFTER=<job id> makes every job wait for that job to succeed, which is how a
# repair queues behind the prune job that writes its source checkpoint:
#
#   AFTER=12345 bash slurm/submit_repair.sh configs/repair/slimgpt-20-multi-lora.yaml
#
# Prints one line per job, "<name> <job id>", after the human-readable ones,
# so a caller can chain on it.
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p slurm-logs

CLI=./.venv/bin/mnlp-eval
if [[ ! -x "${CLI}" ]]; then
    echo "mnlp-eval not found at ${CLI}; see slurm/README.md" >&2
    exit 1
fi
if [[ $# -eq 0 ]]; then
    echo "usage: [AFTER=<job id>] bash $0 <repair.yaml> [more.yaml ...]" >&2
    exit 1
fi

REPAIR_OUT=${REPAIR_OUT:-/scratch-shared/${USER}/checkpoints}
DEPENDENCY=()
if [[ -n "${AFTER:-}" ]]; then
    DEPENDENCY=(--dependency "afterok:${AFTER}")
fi

for CONFIG in "$@"; do
    # Validates the spec and fails before queueing if it is broken.
    NAME=$("${CLI}" repair-spec --spec "${CONFIG}" --field name)
    DATA=$("${CLI}" repair-spec --spec "${CONFIG}" --field data)
    SOURCE=$("${CLI}" repair-spec --spec "${CONFIG}" --field source)
    if [[ ! -d "${DATA}" ]]; then
        echo "repair data not found: ${DATA}" >&2
        echo "Build it on a LOGIN NODE:" >&2
        echo "  ./.venv/bin/mnlp-eval calibration --spec configs/calibration/repair-multi.yaml" >&2
        exit 1
    fi
    if [[ ! -f "${SOURCE}" && -z "${AFTER:-}" ]]; then
        echo "${SOURCE} does not exist and no AFTER job was given to wait for." >&2
        echo "It is written by the prune job; queue this behind it with AFTER=<job id>." >&2
        exit 1
    fi
    JOB=$(
        sbatch --parsable \
            --job-name "repair-${NAME}" \
            "${DEPENDENCY[@]}" \
            --export "ALL,REPAIR_CONFIG=${CONFIG},REPAIR_OUT=${REPAIR_OUT}" \
            slurm/repair.sbatch
    )
    echo "  ${NAME} -> job ${JOB}, checkpoint ${REPAIR_OUT}/${NAME}" >&2
    echo "${NAME} ${JOB}"
done
