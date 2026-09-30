#!/usr/bin/env bash
# Source this file before using the experiment's Python environments on Snellius.
module load 2023
module load Python/3.11.3-GCCcore-12.3.0

check_python_headers() {
    "$1" -c 'import pathlib, sysconfig
header = pathlib.Path(sysconfig.get_path("include")) / "Python.h"
if not header.is_file():
    raise SystemExit(f"Missing {header}; recreate the environment with the Snellius Python module (docs/layer-protection.md).")'
}
