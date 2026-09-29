# Structured pruning

Three criteria over one pipeline. `mnlp-eval prune` turns a dense checkpoint
and a calibration set into a compacted checkpoint, a subnetwork descriptor and
a model config that the rest of the harness runs unchanged.

```bash
mnlp-eval calibration --spec configs/calibration/multi-10dir.yaml
mnlp-eval prune --spec configs/prune/flap-50-multi.yaml
mnlp-eval run --model configs/models/alma-7b-flap50-multi.yaml \
              --suite configs/suites/alma10-greedy.yaml
```

## What is pruned

Two unit types on a Llama-family decoder, defined in `prune/groups.py`:

| unit | removed from | removed from |
| --- | --- | --- |
| attention head | rows of `q_proj`, `k_proj`, `v_proj` | columns of `o_proj` |
| FFN channel | rows of `gate_proj`, `up_proj` | column of `down_proj` |

`o_proj` and `down_proj` are the projections whose *input* channels index the
groups, which is why every activation-based criterion here hooks exactly those
two. Hidden size is never pruned, so the residual stream keeps its width and
layers can differ from each other.

Grouped-query attention constrains the selection. `repeat_kv` expands key/value
head `j` into `n_rep` contiguous copies, wiring query head `i` to key/value head
`i // n_rep` with no remapping hook, so an uneven selection does not fail: the
surviving heads silently attend to the wrong key/value head. The same number of
query heads is therefore kept in every group, and the key and value projections
are left at full width. ALMA-7B is multi-head, so this only binds on the
Qwen2.5-0.5B smoke path.

## The three criteria

| | statistic | needs | compensation |
| --- | --- | --- | --- |
| FLAP | mean and variance of each input channel | one forward pass | bias, added to `o_proj` and `down_proj` |
| LLM-Pruner | accumulated `W * dL/dW` | forward and backward | none |
| SlimGPT | `H = X X^T` per projection | forward, Cholesky inverse | weight update on the survivors |

What each criterion measures, and why, is in its module docstring under
`prune/methods/`; what it means for a reported number is in
[protocol.md](protocol.md).

One departure from the SlimGPT paper: column scores are taken once from the
dense weights rather than recomputed as the block loop advances. The
compensation itself is exact. The paper's per-layer schedule is the
`log-increase` budget below.

## Budget

`uniform` gives every layer the same fraction. `global` standardises each
layer's scores, pools them, and takes the best overall.

Standardising before pooling is not optional: raw importance scales with
activation magnitude, which grows with depth, so pooling raw scores hands the
budget to whichever layers have the largest activations. It also means the
allocation responds to the *shape* of a layer's score distribution rather than
its level. Every layer comes out zero-mean, so a layer whose units all score
badly does not thereby lose more of them; a layer loses more when it has units
clearly worse than the rest of its own.

`log-increase` is SlimGPT's Incremental Pruning Ratio (Ling et al., NeurIPS
2024, equation 6): layer `i` of `n` loses
`r_0 + (r_last - r_0) * log(i + 1) / log(n)` of its units. The paper does not
state `r_0`; here it is 0, so layer 0 is left whole, and `r_last` is solved so
the mean over layers is the requested sparsity. On ALMA-7B that is 0 at layer
0 rising to 0.27 at layer 31 for 20 percent, and to 0.68 for 50. The reasoning
is error accumulation: every later layer inherits an early layer's error, and
the paper's ablation (its Table 6) has this beating uniform, and uniform
beating the decreasing schedules. Scores still decide which units a layer
loses; the schedule decides only how many.

Heads and channels are budgeted separately, each to the requested sparsity.
FLAP pools the two into one ranking, which trades an attention head against an
FFN channel; they are not the same size, so that trade is made in a unit that
does not mean anything.

## Output

A run writes four things:

- the compacted checkpoint, with `config.json` left **dense and unedited**.
  `Qwen2Config` never stores `head_dim` and derives it as
  `hidden_size // num_attention_heads`, so a reduced head count in the config
  would silently change `head_dim` for every layer.
- `subnetwork.json` beside the weights, so the checkpoint carries its own
  shapes, and the same bytes under `subnetworks/`, which is what
  `mnlp-eval overlap` reads. Both are written from the same plan.
- `configs/models/<name>.yaml`, extending `alma-7b.yaml`, with the
  `compression:` block filled in and `loader: custom` pointing at
  `recipes/pruned.py`. It goes beside the baseline because `extends` resolves
  relative to the declaring file.

There is no separate shapes file. Layer `i` keeps
`len(kept["attention_heads"]["kept"][str(i)])` heads, so the descriptor is the
single record of the widths and there is no second artifact to disagree with
it. See [subnetworks.md](subnetworks.md).

## Loading a pruned checkpoint

A global budget gives each layer its own width, and `LlamaConfig` holds one
scalar `num_attention_heads`. `recipes/pruned.py` builds the dense architecture
on the meta device, shrinks each layer to the descriptor, and loads.

Three traps it handles, all of which produce a working model rather than an
error if got wrong:

- `ignore_mismatched_sizes=True` does not adapt shapes. It randomly
  reinitialises every tensor whose shape disagrees. Never use it here.
- `strict=True` cannot be used directly. Both architectures declare
  `_tied_weights_keys = ["lm_head.weight"]` and `save_pretrained` strips tied
  keys, so a tied model always reports that one missing. The loader asserts
  that nothing else is, which is the strictness that matters.
- `assign=True` is required to populate a meta-device model, and it breaks the
  embedding tie, so `tie_weights()` is called afterwards.

## Repair

`mnlp-eval repair` trains a LoRA adapter on a pruned checkpoint and merges it
back in. The result has the pruned shapes, loads through `recipes/pruned.py`
and is evaluated like any other system. Its model config extends the pruned
one and sets `compression.repair`, so the report pairs the two.

```bash
mnlp-eval calibration --spec configs/calibration/repair-multi.yaml
mnlp-eval repair --spec configs/repair/slimgpt-20-multi-lora.yaml
```

The recipe is ALMA's own LoRA stage (`runs/parallel_ft_lora.sh` and
`utils/utils.py` in `fe1ixxu/ALMA`), and the defaults in `RepairSpec` are that
recipe:

| | |
| --- | --- |
| adapter | rank 16, alpha 32, dropout 0.05, every linear layer but the output head |
| optimiser | AdamW, lr 2e-3, weight decay 0.01, gradient clipping at 1.0 |
| schedule | inverse square root after 1 percent warmup, one epoch |
| batch | 128 sequences per step: 8 per micro-batch, 16 accumulation steps |
| text | prompt with the reference appended directly, then eos, at most 511 tokens |
| loss | on the reference and eos only, token-averaged over the whole step |
| data | every segment of ALMA-Human-Parallel, both directions of every pair |

ALMA reaches its batch with eight processes; one GPU reaches the same batch by
accumulating. Gradient checkpointing is on to fit one A100, which changes
memory and speed but not the result.

The repair data is unbalanced on purpose, unlike calibration. Repair tries to
restore what ALMA-7B could do, and ALMA-7B was fine-tuned on this data at these
proportions.

500 segments are held out and scored before and after training.
`repair.json` beside the checkpoint records both, with the loss curve, and the
command fails, after writing everything, if the held-out loss did not fall.

## Transformers version

The pruning extra pins `transformers>=4.48,<5`, tighter than `gen`. Since 4.48
attention forward reshapes with `(*input_shape, -1, self.head_dim)`, so head
counts come off tensor shapes and `num_key_value_groups` is the only attribute
the compaction has to patch. Before 4.48 it reshaped with an explicit
`self.num_heads`, which the pin excludes.

## Cost

For SlimGPT on ALMA-7B at 1,280 calibration segments, peak GPU memory is about
26 GB: weights 13.5 GB, hidden state buffers 10.7 GB, one layer's Hessians
0.9 GB (`down_proj` alone is 11008 squared in float32), plus Cholesky
workspace. The knob is the number of calibration segments, which the buffer is
linear in. Do not shrink `max_length` instead: it changes what the Hessian
measures and the sweep stops being comparable across methods.

LLM-Pruner backpropagates through the whole model, so its budget is backward
activations rather than a fixed statistic. Lower `batch_size` rather than
reaching for the other knobs.

## Testing

`pytest tests/ -m torch` covers all of it with randomly initialised two-layer
models, no GPU and no network. The checks worth knowing about:

- pruning at sparsity 0 leaves logits bit-identical, which catches an
  index-mapping error that a quality metric would show only as a small
  regression;
- a compacted checkpoint reloads to the same function, on a tied-embedding
  model with layers of differing widths;
- FLAP's bias reproduces the dense mean output exactly;
- SlimGPT's compensation measurably reduces the layer's reconstruction error
  against naive truncation, which is the claim that justifies its cost;
- tokens behind the padding mask do not move any score, and for SlimGPT not
  the Hessian either, which the compensation solves against as well as ranks
  by.
