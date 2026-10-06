#!/usr/bin/env bash
# Queue the full-suite evaluation of chosen systems, each on its own directions.
#
#   bash slurm/submit_eval.sh alma-7b-slimgpt20-ref-multi alma-7b-slimgpt40-gen-multi-lora \
#       alma-7b-slimgpt40-gen-pair-{cs,de,is,ru,zh}-lora
#
# A name is a model config under configs/models/. A pair or direction model of
# the grid (or its -lora repair) is evaluated on the directions it was pruned
# for, read from configs/grid/manifest.json; anything else on every direction.
# SUITE_CONFIG defaults to the full alma10-greedy suite. One A100 job per
# system, slurm/eval_directions.sbatch; DRY_RUN=1 prints the commands.
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
    DIRECTIONS=$(./.venv/bin/python - "${NAME}" <<'PY'
import json, sys
name = sys.argv[1].removesuffix("-lora")
models = json.load(open("configs/grid/manifest.json"))["models"]
entry = next((m for m in models if m["name"] == name), None)
print("" if entry is None or entry["scope"] == "multi" else "+".join(entry["pruned_directions"]))
PY
)
    COMMAND=(sbatch --parsable --job-name "eval-${NAME}"
        --export "ALL,MODEL_CONFIG=${MODEL_CONFIG},SUITE_CONFIG=${SUITE_CONFIG},DIRECTIONS=${DIRECTIONS}"
        slurm/eval_directions.sbatch)
    if [[ "${DRY_RUN:-0}" == 1 ]]; then
        echo "${COMMAND[*]}"
    else
        echo "${NAME} [${DIRECTIONS:-all}] job $("${COMMAND[@]}")"
    fi
done
