#!/usr/bin/env bash
# Submit a full sweep: one generation array per model sharded by direction,
# then efficiency, then scoring, then one report.
#
#   bash slurm/submit_sweep.sh <suite.yaml> <model.yaml> [model.yaml ...]
#
# Per model the chain is: generate array -> bench -> score. The report waits on
# every scoring job. Nothing is passed through --export that contains a comma,
# because commas are Slurm's delimiter between assignments inside an --export
# value and silently truncate it.
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p slurm-logs

if [[ $# -lt 2 ]]; then
    echo "usage: $0 <suite.yaml> <model.yaml> [model.yaml ...]" >&2
    exit 1
fi

CLI=./.venv/bin/mnlp-eval
if [[ ! -x "${CLI}" ]]; then
    echo "mnlp-eval not found at ${CLI}; see slurm/README.md" >&2
    exit 1
fi

SUITE_CONFIG="$1"
shift

# Resolving here validates the suite before anything is queued, and gives the
# array size. The CLI is the single source of truth for the direction list.
mapfile -t PAIRS < <("${CLI}" suite-directions --suite "${SUITE_CONFIG}")
N_DIRECTIONS=${#PAIRS[@]}
if (( N_DIRECTIONS == 0 )); then
    echo "suite ${SUITE_CONFIG} lists no directions" >&2
    exit 1
fi
echo "suite ${SUITE_CONFIG}: ${N_DIRECTIONS} direction(s) [${PAIRS[*]}]"

BASELINE=${BASELINE:-}
SCORE_JOBS=()
for MODEL_CONFIG in "$@"; do
    # Validates the model config, and fails before queueing if it is broken.
    RUN_DIR=$("${CLI}" run-dir --model "${MODEL_CONFIG}" --suite "${SUITE_CONFIG}")
    NAME=$(basename "${MODEL_CONFIG}" .yaml)
    echo "  ${NAME} -> ${RUN_DIR}"

    GENERATE_JOB=$(
        sbatch --parsable \
            --job-name "gen-${NAME}" \
            --array "0-$((N_DIRECTIONS - 1))" \
            --export "ALL,MODEL_CONFIG=${MODEL_CONFIG},SUITE_CONFIG=${SUITE_CONFIG}" \
            slurm/generate.sbatch
    )
    echo "    generate array ${GENERATE_JOB} (${N_DIRECTIONS} tasks)"

    # afterok on an array job id waits for every task in the array.
    BENCH_JOB=$(
        sbatch --parsable \
            --job-name "bench-${NAME}" \
            --dependency "afterok:${GENERATE_JOB}" \
            --export "ALL,MODEL_CONFIG=${MODEL_CONFIG},SUITE_CONFIG=${SUITE_CONFIG}" \
            slurm/bench.sbatch
    )
    echo "    bench ${BENCH_JOB}"

    SCORE_JOB=$(
        sbatch --parsable \
            --job-name "score-${NAME}" \
            --dependency "afterok:${BENCH_JOB}" \
            --export "ALL,MODEL_CONFIG=${MODEL_CONFIG},SUITE_CONFIG=${SUITE_CONFIG}" \
            slurm/score.sbatch
    )
    echo "    score ${SCORE_JOB}"
    SCORE_JOBS+=("${SCORE_JOB}")
done

if [[ -z "${BASELINE}" ]]; then
    echo
    echo "BASELINE is not set, so no report job was queued. Compression ratios," >&2
    echo "speedups and significance are all computed against it. Re-run with:" >&2
    echo "  BASELINE=<model name> bash $0 ..." >&2
    exit 1
fi

DEPENDENCY=$(IFS=:; echo "afterok:${SCORE_JOBS[*]}")
REPORT_JOB=$(
    sbatch --parsable \
        --dependency "${DEPENDENCY}" \
        --export "ALL,BASELINE=${BASELINE}" \
        slurm/report.sbatch
)
echo
echo "report ${REPORT_JOB} (baseline ${BASELINE})"
echo "watch with: squeue -u \$USER"
