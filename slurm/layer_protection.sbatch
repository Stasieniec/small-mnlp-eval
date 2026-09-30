#!/usr/bin/env bash
# Resources and dependencies are supplied by the experiment driver.
set -euo pipefail
PYTHON="$1"
EXPERIMENT="$2"
STAGE="$3"
SYSTEM="$4"
cd "${EXPERIMENT}/source"
source scripts/layer_protection_env.sh
check_python_headers "${PYTHON}"
export PYTHONPATH="$PWD/src:$PWD"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
exec "${PYTHON}" -m mnlp_eval.experiments.layer_protection stage \
    "${EXPERIMENT}" "${STAGE}" "${SYSTEM}"
