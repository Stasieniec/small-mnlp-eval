#!/usr/bin/env bash
# Queue one pruning job per config.
#
#   bash slurm/submit_prune.sh configs/prune/flap-50-multi.yaml [more.yaml ...]
#   bash slurm/submit_prune.sh                 # every config in configs/prune
#
# The jobs are independent: each reads the dense checkpoint and writes its own
# subnetwork, so they run in parallel rather than in a chain.
#
# This deliberately does not chain into slurm/submit_sweep.sh. The evaluation
# sweep is well over a hundred GPU-hours, and the achieved sparsity and the
# overlap between the three subnetworks are worth looking at before spending
# it. The follow-up command is printed at the end.
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p slurm-logs subnetworks

CLI=./.venv/bin/mnlp-eval
if [[ ! -x "${CLI}" ]]; then
    echo "mnlp-eval not found at ${CLI}; see slurm/README.md" >&2
    exit 1
fi

if [[ $# -eq 0 ]]; then
    CONFIGS=(configs/prune/*.yaml)
else
    CONFIGS=("$@")
fi

PRUNE_OUT=${PRUNE_OUT:-/scratch-shared/${USER}/checkpoints}

# Every config names the calibration set it needs. Checking here turns a
# missing set into a failure now rather than four hours into a GPU job, and
# `mnlp-eval calibration` needs the Hub, so it cannot be fixed from a
# compute node.
MISSING=()
for CONFIG in "${CONFIGS[@]}"; do
    CALIBRATION=$("${CLI}" prune-spec --spec "${CONFIG}" --field calibration)
    if [[ ! -d "${CALIBRATION}" ]]; then
        MISSING+=("${CALIBRATION}")
    fi
done
if (( ${#MISSING[@]} > 0 )); then
    printf 'calibration data not found: %s\n' "${MISSING[@]}" >&2
    echo >&2
    echo "Build it on a LOGIN NODE, which has the Hub access it needs:" >&2
    echo "  ./.venv/bin/mnlp-eval calibration --spec configs/calibration/multi-10dir.yaml" >&2
    exit 1
fi

MODEL_CONFIGS=()
for CONFIG in "${CONFIGS[@]}"; do
    # Validates the spec and fails before queueing if it is broken.
    NAME=$("${CLI}" prune-spec --spec "${CONFIG}" --field name)
    JOB=$(
        sbatch --parsable \
            --job-name "prune-${NAME}" \
            --export "ALL,PRUNE_CONFIG=${CONFIG},PRUNE_OUT=${PRUNE_OUT}" \
            slurm/prune.sbatch
    )
    echo "  ${NAME} -> job ${JOB}, checkpoint ${PRUNE_OUT}/${NAME}"
    MODEL_CONFIGS+=("configs/models/${NAME}.yaml")
done

SWEEP_ARGS=$(printf ' \\\n      %s' "${MODEL_CONFIGS[@]}")

cat <<NEXT

Queued ${#CONFIGS[@]} pruning job(s). Watch with: squeue -u \$USER

Each job prints a manifest with the sparsity it actually achieved. Once they
are done, check that the three subnetworks are not the same selection:

  ./.venv/bin/mnlp-eval overlap --subnetwork-dir subnetworks/

then evaluate them against the dense baseline:

  BASELINE=alma-7b bash slurm/submit_sweep.sh \\
      configs/suites/alma10-greedy.yaml \\
      configs/models/alma-7b.yaml${SWEEP_ARGS}
NEXT
