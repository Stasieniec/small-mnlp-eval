#!/usr/bin/env bash
# Queue a reduced-suite evaluation: generate, score, report. No bench.
#
#   BASELINE=alma-7b bash slurm/submit_pilot.sh \
#       configs/suites/alma10-greedy-300.yaml \
#       configs/models/alma-7b.yaml configs/models/alma-7b-slimgpt50-multi.yaml
#
# Differs from submit_sweep.sh in exactly two ways, both of which only pay off
# on a reduced suite:
#
#   - one generation job per system instead of an array of ten, because at a
#     few hundred segments per direction the model load dominates;
#   - no bench. BenchSpec is a fixed 128-segment protocol, so its cost does not
#     fall with the suite, and it asks for an exclusive four-GPU node, which
#     bills at four times the rate of the one GPU it uses. On a pilot it would
#     cost more than everything else combined. Run slurm/bench.sbatch by hand
#     when efficiency numbers are actually wanted.
#
# Scoring reuses slurm/score.sbatch unchanged, which scopes itself to one run
# with --run. That guard matters here: `score` with no --run reaches every run
# directory on disk, including ones still generating.
#
# Two optional settings for chaining behind pruning and repair:
#
#   AFTER=<job id>  every generation job waits for that job to succeed. The
#                   model config may then not exist yet: prune and repair
#                   write it, and the generation job reads it when it starts.
#   REPORT=0        queue no report job, for when several calls feed one
#                   report submitted separately at the end.
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p slurm-logs

CLI=./.venv/bin/mnlp-eval
if [[ ! -x "${CLI}" ]]; then
    echo "mnlp-eval not found at ${CLI}; see slurm/README.md" >&2
    exit 1
fi

if [[ $# -lt 2 ]]; then
    echo "usage: BASELINE=<name> bash $0 <suite.yaml> <model.yaml> [more.yaml ...]" >&2
    exit 1
fi

SUITE_CONFIG="$1"
shift

BASELINE=${BASELINE:-}
if [[ -z "${BASELINE}" ]]; then
    echo "BASELINE is not set. Every compression ratio and p-value is computed" >&2
    echo "against it, and the report drops all of them if it cannot find it." >&2
    exit 1
fi

mapfile -t PAIRS < <("${CLI}" suite-directions --suite "${SUITE_CONFIG}")
echo "suite ${SUITE_CONFIG}: ${#PAIRS[@]} direction(s) [${PAIRS[*]}]"

DEPENDENCY=()
if [[ -n "${AFTER:-}" ]]; then
    # Without the kill flag a failed upstream job leaves this one pending
    # forever with reason DependencyNeverSatisfied.
    DEPENDENCY=(--dependency "afterok:${AFTER}" --kill-on-invalid-dep=yes)
fi

SCORE_JOBS=()
for MODEL_CONFIG in "$@"; do
    if [[ -f "${MODEL_CONFIG}" ]]; then
        RUN_DIR=$("${CLI}" run-dir --model "${MODEL_CONFIG}" --suite "${SUITE_CONFIG}")
        echo "  ${MODEL_CONFIG} -> ${RUN_DIR}"
    elif [[ -n "${AFTER:-}" ]]; then
        echo "  ${MODEL_CONFIG} (written by job ${AFTER}, which this waits for)"
    else
        echo "${MODEL_CONFIG} does not exist, and no AFTER job was given to write it." >&2
        exit 1
    fi

    GENERATE_JOB=$(
        sbatch --parsable \
            --job-name "pilot-$(basename "${MODEL_CONFIG}" .yaml)" \
            "${DEPENDENCY[@]}" \
            --export "ALL,MODEL_CONFIG=${MODEL_CONFIG},SUITE_CONFIG=${SUITE_CONFIG}" \
            slurm/pilot.sbatch
    )
    echo "    generate ${GENERATE_JOB}"

    SCORE_JOB=$(
        sbatch --parsable \
            --job-name "score-$(basename "${MODEL_CONFIG}" .yaml)" \
            --dependency "afterok:${GENERATE_JOB}" --kill-on-invalid-dep=yes \
            --export "ALL,MODEL_CONFIG=${MODEL_CONFIG},SUITE_CONFIG=${SUITE_CONFIG}" \
            slurm/score.sbatch
    )
    echo "    score    ${SCORE_JOB}"
    SCORE_JOBS+=("${SCORE_JOB}")
done

if [[ "${REPORT:-1}" == "0" ]]; then
    echo
    echo "No report queued (REPORT=0). Score jobs: ${SCORE_JOBS[*]}"
    exit 0
fi

DEPENDENCY=$(IFS=:; echo "afterok:${SCORE_JOBS[*]}")
REPORT_JOB=$(
    sbatch --parsable \
        --dependency "${DEPENDENCY}" --kill-on-invalid-dep=yes \
        --export "ALL,BASELINE=${BASELINE}" \
        slurm/report.sbatch
)

cat <<NEXT

report ${REPORT_JOB} (baseline ${BASELINE}), waits on every scoring job.

Watch with: squeue -u \$USER
Tables land in reports/ once the report job finishes.
NEXT
