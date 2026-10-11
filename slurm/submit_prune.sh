#!/usr/bin/env bash
# Queue one pruning job per config.
#
#   bash slurm/submit_prune.sh configs/prune/flap-50-multi.yaml [more.yaml ...]
#   bash slurm/submit_prune.sh                 # every config in configs/prune but smoke-*
#
# Prints "<name> <job id>" per job on stdout and everything else on stderr, so
#   JOB=$(bash slurm/submit_prune.sh configs/prune/x.yaml | awk '{print $2}')
# gives the job id to chain repair or evaluation on with AFTER=.
#
# PARTITION=gpu_h100 overrides the partition in prune.sbatch.
# The jobs are independent. The follow-up sweep command is printed at the end.
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p slurm-logs subnetworks

CLI=./.venv/bin/mnlp-eval
if [[ ! -x "${CLI}" ]]; then
    echo "mnlp-eval not found at ${CLI}; see README.md" >&2
    exit 1
fi

if [[ $# -eq 0 ]]; then
    # Skip the smoke-* configs.
    CONFIGS=()
    for CONFIG in configs/prune/*.yaml; do
        [[ "$(basename "${CONFIG}")" == smoke-* ]] || CONFIGS+=("${CONFIG}")
    done
else
    CONFIGS=("$@")
fi

PRUNE_OUT=${PRUNE_OUT:-/scratch-shared/${USER}/checkpoints}

# Check the calibration sets exist; building one needs the Hub (login node).
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
    # Also validates the spec.
    NAME=$("${CLI}" prune-spec --spec "${CONFIG}" --field name)
    JOB=$(
        sbatch --parsable \
            --job-name "prune-${NAME}" \
            ${PARTITION:+--partition "${PARTITION}"} \
            --export "ALL,PRUNE_CONFIG=${CONFIG},PRUNE_OUT=${PRUNE_OUT}" \
            slurm/prune.sbatch
    )
    echo "  ${NAME} -> job ${JOB}, checkpoint ${PRUNE_OUT}/${NAME}" >&2
    # stdout: NAME JOB
    echo "${NAME} ${JOB}"
    MODEL_CONFIGS+=("configs/models/${NAME}.yaml")
done

SWEEP_ARGS=$(printf ' \\\n      %s' "${MODEL_CONFIGS[@]}")

cat >&2 <<NEXT

Queued ${#CONFIGS[@]} pruning job(s). Watch with: squeue -u \$USER

Each job prints a manifest with the sparsity it actually achieved. Once they
are done, check that the three subnetworks are not the same selection:

  ./.venv/bin/mnlp-eval overlap --subnetwork-dir subnetworks/

then evaluate them against the dense baseline:

  BASELINE=alma-7b bash slurm/submit_sweep.sh \\
      configs/suites/alma10-greedy.yaml \\
      configs/models/alma-7b.yaml${SWEEP_ARGS}
NEXT
