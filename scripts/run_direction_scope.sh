#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/layer_protection_env.sh
PYTHON=${PYTHON:-./.venv/bin/python}
check_python_headers "${PYTHON}"
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
exec "${PYTHON}" -m mnlp_eval.experiments.direction_scope "$@"
