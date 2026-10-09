#!/usr/bin/env bash
# Queue the full-suite evaluation of chosen systems, each on its own directions.
#
#   bash slurm/submit_eval.sh alma-7b-slimgpt20-ref-multi alma-7b-slimgpt40-gen-multi-lora \
#       alma-7b-slimgpt40-gen-pair-{cs,de,is,ru,zh}-lora
#
# A name is a model config under configs/models/. A pair or direction model of
# the grid (or its -lora repair) is evaluated on the directions it was pruned
# for, read from configs/grid/manifest.json; anything else on every direction.
# SUITE_CONFIG defaults to the full alma10-greedy suite. One job per system and
# partition (PARTITIONS, default "gpu_a100 gpu_h100"), slurm/eval_directions.sbatch;
# DRY_RUN=1 prints the commands.
# ALL_DIRECTIONS=1 evaluates every name on every direction (transfer matrices).
# Never queue a model twice with different directions: both jobs write one run
# directory, and the first to score would leave partial scores the other skips.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p slurm-logs/eval
SUITE_CONFIG=${SUITE_CONFIG:-configs/suites/alma10-greedy.yaml}

for NAME in "$@"; do
    MODEL_CONFIG=configs/models/${NAME}.yaml
    if [[ ! -f "${MODEL_CONFIG}" ]]; then
        echo "${MODEL_CONFIG} does not exist" >&2
        exit 1
    fi
    if [[ "${ALL_DIRECTIONS:-0}" == 1 ]]; then
        DIRECTIONS=""
    else
    DIRECTIONS=$(./.venv/bin/python - "${NAME}" <<'PY'
import json, sys
name = sys.argv[1].removesuffix("-lora")
models = json.load(open("configs/grid/manifest.json"))["models"]
entry = next((m for m in models if m["name"] == name), None)
print("" if entry is None or entry["scope"] == "multi" else "+".join(entry["pruned_directions"]))
PY
)
    fi
    # Realistic limits, so the jobs can backfill into gaps on a full cluster.
    # H100 nodes run this host-bound workload slower; the limits allow for that.
    if [[ -z "${DIRECTIONS}" ]]; then
        TIME=${TIME_ALL:-01:45:00}
    elif [[ "${DIRECTIONS}" == *+* ]]; then
        TIME=${TIME_PAIR:-00:50:00}
    else
        TIME=${TIME_DIR:-00:40:00}
    fi
    # One twin per partition; the job script lets the first to start run.
    JOBS=()
    for PARTITION in ${PARTITIONS:-gpu_a100 gpu_h100}; do
        CPUS=18
        [[ "${PARTITION}" == gpu_h100 ]] && CPUS=16
        COMMAND=(sbatch --parsable --job-name "eval-${NAME}" --partition "${PARTITION}"
            --cpus-per-task "${CPUS}" --time "${TIME}"
            --export "ALL,MODEL_CONFIG=${MODEL_CONFIG},SUITE_CONFIG=${SUITE_CONFIG},DIRECTIONS=${DIRECTIONS}"
            slurm/eval_directions.sbatch)
        if [[ "${DRY_RUN:-0}" == 1 ]]; then
            echo "${COMMAND[*]}"
        else
            JOBS+=("${PARTITION}:$("${COMMAND[@]}" 2>/dev/null)")
        fi
    done
    [[ "${DRY_RUN:-0}" == 1 ]] || echo "${NAME} [${DIRECTIONS:-all}] ${JOBS[*]}"
done
