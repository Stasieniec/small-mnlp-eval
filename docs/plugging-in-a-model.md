# Plugging in a model

A compressed checkpoint should become a fully evaluated system without
touching the framework.

## One YAML file

Load-time quantization:

```yaml
extends: alma-7b.yaml
name: alma-7b-bnb-nf4
baseline: alma-7b
quantization:
  method: bitsandbytes
  load_in_4bit: true
  bnb_4bit_quant_type: nf4
  bnb_4bit_compute_dtype: bfloat16
```

A checkpoint quantized elsewhere and saved:

```yaml
name: alma-7b-gptq-4bit
loader: hf_causal
prompt: alma
baseline: alma-7b
model_name_or_path: /scratch-shared/$USER/alma-7b-gptq
quantization:
  method: gptq
```

For GPTQ, AWQ and compressed-tensors the metadata is already in the
checkpoint, so `method` alone is enough.

A base checkpoint with a LoRA adapter:

```yaml
name: alma-7b-prune50-multi-lora
extends: alma-7b-prune50-multi.yaml
adapter:
  path: /scratch-shared/$USER/prune50-multi-lora
  merge: true
```

`merge: true` folds the adapter into the base weights. Leave it merged unless
you want to measure adapter overhead, since an unmerged adapter shows up as a
latency penalty unrelated to compression.

Loaders: `hf_causal`, `hf_seq2seq`, `nllb`, `custom`. Quantization methods:
`bitsandbytes`, `gptq`, `awq`, `compressed_tensors`, `hqq`, `torchao`.

Always set `baseline`. It is what the report computes ratios, speedups and
p-values against.

## The compression block

```yaml
compression:
  family: pruning              # none, pruning, quantization or distillation
  method: structured, FFN intermediate channels and attention heads
  nominal_sparsity: 0.5        # fraction removed, not the fraction kept
  pruned_for: de-en,en-de      # or "multi"
  calibration: configs/calibration/pair-de.yaml
  subnetwork: subnetworks/prune50-de.json
  repair: LoRA r=16 on ALMA-Human-Parallel
```

Descriptive, so it takes no part in run identity and can be filled in later
without orphaning existing runs. `nominal_sparsity` is the sweep's x-axis;
`pruned_for` is what the transfer matrix is built from; `subnetwork` links the
config to its descriptor. Without them those tables are omitted.

`nominal_sparsity` is what the method aimed for. What it achieved is measured
independently by the bench stage from the loaded model's per-layer widths and
its share of exactly-zero weights. If the two disagree, believe the
measurement.

## Custom loading

Structured pruning usually changes the architecture, so the checkpoint needs
the code that produced it:

```yaml
name: alma-7b-wanda-50
loader: custom
prompt: alma
baseline: alma-7b
entrypoint: recipes.wanda:load
kwargs:
  checkpoint: /scratch-shared/$USER/wanda-50
  sparsity: 0.5
```

The entrypoint is `module.path:function`, and the working directory is on
`sys.path` so `recipes/` resolves without installing anything. It may return a
`(model, tokenizer)` tuple, a mapping with `model`, `tokenizer` and optional
`kind` and `extra` (which lands in the manifest), or a `Translator` subclass.
See `recipes/example_pruned.py`.

Two traps:

- **The `checkpoint` kwarg name is load-bearing.** With `model_name_or_path`
  unset, `checkpoint` is what the framework inspects to find the weights on
  disk. Name it otherwise and checkpoint size, bits per parameter and the disk
  compression ratio all read as a dash.
- **A `custom` loader cannot apply `dtype`, `quantization`, `adapter`,
  `device_map`, `attn_implementation` or `trust_remote_code`**, since your
  entrypoint does the loading. Setting them is an error rather than ignored: an
  ignored `quantization:` block would mean evaluating an unquantized model
  under a quantized model's name.

The tokenizer must behave like a Hugging Face one. The generation loop reads
`pad_token_id`, `eos_token_id`, `eos_token` and `padding_side`, and calls the
object itself, `pad()` and `decode()`.

## Checking before spending an allocation

```bash
python -c "from mnlp_eval.config import ModelSpec, load_yaml_config;
           print(ModelSpec.from_dict(load_yaml_config('configs/models/yours.yaml')))"

mnlp-eval run --model configs/models/yours.yaml \
              --suite configs/suites/smoke-de-en.yaml \
              --limit 4 --no-bench --groups surface

mnlp-eval run-dir --model configs/models/yours.yaml \
                  --suite configs/suites/smoke-de-en.yaml
```

`--no-bench` matters: `--limit` caps the quality run only, and the efficiency
stage has its own subset size.

Unknown keys are rejected rather than ignored, including in `--set` overrides,
so a typo fails immediately instead of an hour into a job.

## What lands in a run directory

```
runs/<model>__<suite>__<hash>/
  manifest.json          identity, written once
  env.json               versions, device, driver, clocks, packages
  stages/generate.json   or generate.<direction>.json when sharded
  hyps/<direction>.jsonl one record per segment
  hyps/<direction>.txt   one hypothesis per line, ALMA's output format
  bench.json
  scores.<group>.json
```

Each JSON Lines record carries `source`, `reference`, `hypothesis`,
`raw_output`, `parse_status`, `parse_flags`, token counts, `hit_token_budget`,
`truncated` and `source_truncated`. `raw_output` is kept because when a
compressed model collapses, the recovered hypothesis alone does not explain
what happened.
