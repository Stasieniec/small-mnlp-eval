# Environments

This project needs three Python environments. That is not a stylistic choice.

## Why three

| Package | Hard pins | Consequence |
| --- | --- | --- |
| `unbabel-comet` 2.2.7 | `numpy<2.0.0`, `torchmetrics<0.11.0`, `jsonargparse==3.13.1`, `protobuf<5.0.0` | Cannot coexist with a current generation stack |
| `metricx` (GitHub, not pip installable) | `transformers[torch]==4.30.2`, `datasets==2.13.1` | Cannot coexist with COMET or with the generation stack |
| generation stack | torch, transformers 4.x, numpy 2.x, bitsandbytes | What compression work is developed against |

`torchmetrics<0.11.0` is a 2022 release. Resolving all three sets of pins into
one environment either fails outright or silently downgrades numpy underneath
the generation code, which is worse.

The framework therefore joins its stages through files on disk and never
through function calls across a metric boundary. The core `mnlp_eval` package
depends only on `pyyaml`, precisely so it can be installed in all three.

Two consequences worth having anyway:

- A finished run can be re-scored with a new metric years later without
  spending GPU hours regenerating hypotheses.
- Scoring runs as its own Slurm job, on its own node, with its own wall-clock
  limit.

## Creating them

```bash
# 1. Generation. The default environment for everything except neural metrics.
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e ".[gen,surface,dev]"

# 2. COMET. BLEU and chrF++ also work here, so it can score both groups.
uv venv --python 3.11 .venv-comet
uv pip install --python .venv-comet/bin/python -r envs/comet-requirements.txt

# 3. MetricX. Only needed for the final report, and NOT a plain pip install:
#    upstream ships no pyproject.toml or setup.py, so it has to be cloned onto
#    PYTHONPATH. This script does the clone, the venv and a verification.
bash scripts/setup_metricx_env.sh
```

The MetricX step is the one environment a single `uv pip install -r` cannot
build. `pip install git+https://github.com/google-research/metricx.git` fails
with "does not appear to be a Python project", because the repository root
holds only a README, a licence, `requirements.txt` and the `metricx23` and
`metricx24` directories. The package is imported as `metricx24`, so the clone
root goes on `PYTHONPATH`. `slurm/score.sbatch` reads `METRICX_SRC` and
defaults to `$PWD/third_party/metricx`.

Check what any environment can do:

```bash
.venv/bin/mnlp-eval info
```

It prints each metric group as available or unavailable, and for unavailable
groups it says which environment is needed.

## Which environment runs which stage

| Stage | Environment | Reason |
| --- | --- | --- |
| `prefetch` | each of them, once | Each holds different metric checkpoints |
| `generate` | `.venv` | Needs torch and transformers |
| `bench` | `.venv` | Must measure the same stack that generated |
| `score --groups surface` | any | Pure Python plus sacreBLEU |
| `score --groups neural` | `.venv-comet` | COMET-22, COMETKiwi, XCOMET |
| `score --groups metricx` | `.venv-metricx` | MetricX-24 |
| `report` | any | Reads JSON only |

A `score` call for a group the current interpreter cannot serve records the
group as pending with the reason, and exits zero. It never fails the job and
never silently omits the group.

## Known traps

**`pkg_resources` is missing.** `torchmetrics<0.11` imports `pkg_resources` at
module load, and setuptools 81 removed it. `envs/comet-requirements.txt` pins
`setuptools<81` for this reason. Without the pin the environment installs
cleanly and then fails on `import comet`.

**transformers 5.x.** The `gen` extra caps transformers below 5.0, because
COMET requires `transformers<5.0` and keeping both environments on one major
version removes a class of tokenizer-drift questions that would otherwise have
to be ruled out by hand. ALMA's current `install_alma.sh` pins 4.51.1, though
that release postdates both ALMA papers; ALMA-7B-R's own
`generation_config.json` records `transformers_version: 4.36.2`, which is what
actually produced the checkpoint.

**Gated checkpoints.** `Unbabel/wmt22-cometkiwi-da` and `Unbabel/XCOMET-XL`
require accepting a licence on the Hub. Export `HF_TOKEN` before prefetching.
XCOMET-XL is a 13.9 GB fp32 checkpoint under CC-BY-NC-SA-4.0 and needs an
A100-class GPU.

**Download footprint.** Prefetching the full metrics config pulls roughly 16 GB
of COMET checkpoints. MetricX references `google/mt5-xl` for its sentencepiece
vocabulary only, so prefetch restricts that repository to its tokenizer files;
fetching it whole would be 45 GB of PyTorch, TensorFlow and Flax weights the
framework never loads.

**Quantization extras.** `bitsandbytes` and `peft` live in the `quant` extra,
not in `gen`. The shipped `alma-7b-r-bnb-*` configs need them. Install with
`uv pip install --python .venv/bin/python -e ".[gen,quant,surface,dev]"`, or
expect an error naming `bitsandbytes` metadata rather than the extra.

**No pip in these environments.** `uv venv` does not install pip, so
`python -m pip show` fails. Use `uv pip show --python .venv/bin/python`. The
run manifest records installed distributions through `importlib.metadata` for
the same reason.
