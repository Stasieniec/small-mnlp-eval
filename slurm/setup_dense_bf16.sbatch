#!/usr/bin/env bash
#SBATCH --job-name=mnlp-dense-bf16
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=slurm-logs/setup/dense-bf16-%j.out
#
# One-off: write ALMA-7B as bfloat16 safetensors on scratch.
#
# The Hub checkpoint is float32 .bin shards, and loading them took eight of a
# prune job's fifteen minutes in the pilot. Every job loads the dense model
# with torch_dtype=bfloat16 anyway, so the copy holds exactly the tensors those
# jobs already used; the check below compares every one of them.
set -euo pipefail

module load 2025
module load Python/3.13.1-GCCcore-14.2.0
cd "${SLURM_SUBMIT_DIR}"

export HF_HOME=${HF_HOME:-/scratch-shared/${USER}/hf_home}
export HF_HUB_OFFLINE=1
OUT=${DENSE_BF16_OUT:-/scratch-shared/${USER}/models/alma-7b-bf16}

./.venv/bin/python - "${OUT}" <<'PY'
import sys, time
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

out = Path(sys.argv[1])
start = time.time()
model = AutoModelForCausalLM.from_pretrained(
    "haoranxu/ALMA-7B", torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
)
print(f"loaded in {time.time() - start:.0f}s", flush=True)
model.save_pretrained(out, safe_serialization=True, max_shard_size="5GB")
AutoTokenizer.from_pretrained("haoranxu/ALMA-7B").save_pretrained(out)

saved = {}
for shard in sorted(out.glob("*.safetensors")):
    saved.update(load_file(shard))
state = model.state_dict()
tied = set(getattr(type(model), "_tied_weights_keys", None) or ())
missing = sorted(set(state) - set(saved) - tied)
assert not missing, f"missing from the copy: {missing[:5]}"
for key, tensor in saved.items():
    assert torch.equal(tensor, state[key]), f"{key} differs"
print(f"verified {len(saved)} tensors bit-identical, {time.time() - start:.0f}s total")
PY
ls -la "${OUT}"
