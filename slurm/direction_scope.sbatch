#!/usr/bin/env bash
# Resources are supplied by direction_scope.submit; never request an exclusive node.
set -euo pipefail
PYTHON="$1"
EXPERIMENT="$2"
SYSTEM="$3"
cd "${EXPERIMENT}/source"
source scripts/layer_protection_env.sh
check_python_headers "${PYTHON}"
export PYTHONPATH="$PWD/src:$PWD"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
exec "${PYTHON}" -m mnlp_eval.experiments.direction_scope stage "${EXPERIMENT}" "${SYSTEM}"
