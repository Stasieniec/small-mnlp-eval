# Plugging in a model

The goal: a compressed checkpoint becomes a fully evaluated system without
touching the framework.

## The common case: one YAML file

Most models need no code. Write a file in `configs/models/` and the whole
pipeline works.

Weight-only 4-bit quantization applied at load time:

```yaml
extends: alma-7b-r.yaml
name: alma-7b-r-bnb-nf4
baseline: alma-7b-r
quantization:
  method: bitsandbytes
  load_in_4bit: true
  bnb_4bit_quant_type: nf4
  bnb_4bit_use_double_quant: true
  bnb_4bit_compute_dtype: bfloat16
```

A checkpoint you quantized elsewhere and saved, for example with
`llm-compressor` or `GPTQModel`:

```yaml
name: alma-7b-r-gptq-4bit
loader: hf_causal
prompt: alma
baseline: alma-7b-r
model_name_or_path: /scratch-shared/$USER/alma-7b-r-gptq
dtype: bfloat16
quantization:
  method: gptq
```

For GPTQ, AWQ and compressed-tensors the quantization metadata already lives in
the checkpoint, so `method` alone is enough. Add options only to override a
default.

A distilled student with a LoRA adapter:

```yaml
name: alma-7b-r-distilled-lora
loader: hf_causal
prompt: alma
baseline: alma-7b-r
model_name_or_path: haoranxu/ALMA-7B-Pretrain
adapter:
  path: /scratch-shared/$USER/distilled-lora
  merge: true
```

`merge: true` folds the adapter into the base weights. Leave it merged unless
you specifically want to measure adapter overhead, since an unmerged adapter
shows up as a latency penalty that has nothing to do with compression.

Supported `loader` values: `hf_causal`, `hf_seq2seq`, `nllb`, `custom`.
Supported `quantization.method` values: `bitsandbytes`, `gptq`, `awq`,
`compressed_tensors`, `hqq`, `torchao`.

Always set `baseline` on a compression variant. It is what lets the report
compute compression ratios, speedups and significance against the right system.

## The escape hatch: one function

Structured pruning usually changes the architecture, so the checkpoint cannot
be loaded without the code that produced it. Write a loader and point at it:

```yaml
name: alma-7b-wanda-50
loader: custom
prompt: alma
baseline: alma-7b-r
entrypoint: recipes.wanda:load
kwargs:
  checkpoint: /scratch-shared/$USER/wanda-50
  sparsity: 0.5
```

The entrypoint is `module.path:function`. The working directory is added to
`sys.path`, so `recipes/` in a checkout resolves without installing anything.

Your function receives `kwargs` and may return any of three things:

```python
def load(checkpoint: str, sparsity: float):
    model = MyPrunedModel.from_pretrained(checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    # 1. A tuple. The simplest option; encoder-decoder is detected from
    #    model.config.is_encoder_decoder.
    return model, tokenizer

    # 2. A mapping, if you want to declare the kind or record metadata.
    #    Anything under "extra" lands in the run manifest.
    return {
        "model": model,
        "tokenizer": tokenizer,
        "kind": "causal",
        "extra": {"sparsity": sparsity, "method": "wanda"},
    }

    # 3. A Translator, if you need to control prompting or generation.
```

See `recipes/example_pruned.py` for a working template.

## What you get for free

Once the config exists, everything downstream works unchanged: ALMA-compatible
prompting, left-padded length-bucketed batching, hypothesis extraction with an
audit trail, BLEU and chrF++ with signatures, COMET, MetricX, behavioural
failure metrics, efficiency measurement, compression ratios, bootstrap
significance against the baseline, and a place in the report tables.

## Checking before you commit a GPU allocation

```bash
# Validates the config without loading any weights.
.venv/bin/python -c "
from mnlp_eval.config import ModelSpec, load_yaml_config
print(ModelSpec.from_dict(load_yaml_config('configs/models/yours.yaml')))
"

# Four segments, on whatever GPU is to hand.
.venv/bin/mnlp-eval run --model configs/models/yours.yaml \
                        --suite configs/suites/smoke-de-en.yaml \
                        --limit 4 --groups surface
```

Unknown keys are rejected rather than ignored, so a typo fails immediately
instead of falling back to a default an hour into a Slurm job.
