#!/usr/bin/env bash
# Submit a full sweep: generation per (model, direction), then scoring per
# model, then one report.
#
#   bash slurm/submit_sweep.sh <suite.yaml> <model.yaml> [model.yaml ...]
#
# Scoring for a model depends on its generation array finishing, and the report
# depends on every scoring job, so the chain is correct without babysitting.
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p slurm-logs

if [[ $# -lt 2 ]]; then
    echo "usage: $0 <suite.yaml> <model.yaml> [model.yaml ...]" >&2
    exit 1
fi

SUITE_CONFIG="$1"
shift

DIRECTIONS=$(
    ./.venv/bin/python -c "
from mnlp_eval.config import SuiteSpec, load_yaml_config
print(','.join(SuiteSpec.from_dict(load_yaml_config('${SUITE_CONFIG}')).data.directions))
"
)
N_DIRECTIONS=$(awk -F, '{print NF}' <<< "${DIRECTIONS}")
echo "suite ${SUITE_CONFIG}: ${N_DIRECTIONS} direction(s) [${DIRECTIONS}]"

SCORE_JOBS=()
for MODEL_CONFIG in "$@"; do
    NAME=$(basename "${MODEL_CONFIG}" .yaml)
    GENERATE_JOB=$(
        sbatch --parsable \
            --job-name "gen-${NAME}" \
            --array "0-$((N_DIRECTIONS - 1))" \
            --export "ALL,MODEL_CONFIG=${MODEL_CONFIG},SUITE_CONFIG=${SUITE_CONFIG},DIRECTIONS=${DIRECTIONS}" \
            slurm/generate.sbatch
    )
    echo "  ${NAME}: generation array ${GENERATE_JOB}"

    SCORE_JOB=$(
        sbatch --parsable \
            --job-name "score-${NAME}" \
            --dependency "afterok:${GENERATE_JOB}" \
            --export "ALL,METRICS_CONFIG=${METRICS_CONFIG:-configs/metrics/default.yaml}" \
            slurm/score.sbatch
    )
    echo "  ${NAME}: scoring ${SCORE_JOB}"
    SCORE_JOBS+=("${SCORE_JOB}")
done

DEPENDENCY=$(IFS=:; echo "afterok:${SCORE_JOBS[*]}")
REPORT_JOB=$(
    sbatch --parsable \
        --dependency "${DEPENDENCY}" \
        --export "ALL,BASELINE=${BASELINE:-alma-7b-r}" \
        slurm/report.sbatch
)
echo "report ${REPORT_JOB}"
echo
echo "Watch with: squeue -u \$USER"
