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

SlimGPT prunes layer by layer, each layer on the activations the
already-pruned layers before it produce. For a projection with damped input
Hessian `H` (ridge 1 percent of the mean diagonal, padding excluded), removing
the columns `S` and refitting the rest by least squares costs
`tr(W_S M^-1 W_S^T)` with `M = [H^-1]_SS`; the refit is
`W <- W - W_S M^-1 [H^-1]_S,:`, and `H^-1` is downdated by the matching Schur
complement, all in float64. That is the exact optimum over every surviving
column, whatever its position. Units go greedily, each step re-scoring the
survivors against the current weights and inverse: heads one per step (one
per key/value group under grouped-query), FFN channels in batches of 128 (the
paper shrinks its batch from 1024 to 8; its ablation puts a fixed size within
noise). Within a layer the heads are pruned first and the FFN Hessian is taken
on the pruned attention. How many units a layer keeps comes from the budget
below; the paper's schedule is `log-increase`.

Before 6 October 2026 this followed SparseGPT's Cholesky sweep in column
order. That compensates a removed column only with the columns after it, so
the last heads of `o_proj` got no compensation, and it scored columns on the
trailing submatrix, so late units looked more important. Results from before
that date (the 29 September pilot, the layer-protection and direction-scope
experiments) used it.

## Calibration text

By default the criteria read each calibration prompt alone, which ends at the
cue for the translation. `calibration_text: prompt+target` appends the
reference translation, joined as ALMA's training joins them, so the model also
reads the target language as if it had written it. With the prompt alone, the
`en-xx` directions never put a word of the target language through the model,
and units that matter for producing Czech, Icelandic or Chinese are judged only
on how they respond to reading English.

The variant also sees about 1.7 times the calibration tokens (134k against 78k
on multi-10dir), a second change, so `slimgpt-20-multi-256` calibrates on
prompts alone at 256 segments per direction as the control (156k tokens; its
128-sample set is contained in it, though sorted row order differs). SlimGPT's
peak memory for either is about 21 GB on ALMA-7B, so both fit a 40 GB A100.

### Dense-generated translations

Generate the shared `multi-10dir` pool once (1,280 prompts at 128 per direction):

```bash
mnlp-eval calibration-generate \
  --calibration data/calibration/multi-10dir \
  --model configs/models/alma-7b.yaml \
  --decode-suite configs/suites/alma10-greedy.yaml \
  --out data/calibration/multi-10dir-generated
```

Run this in the generation environment on a GPU, after prefetching the dense
model. The matching Slurm job is `slurm/calibration_generate.sbatch`; create
`slurm-logs` before submitting it. On offline compute nodes set
`HF_HUB_OFFLINE=1`. It loads the dense
model once for all missing directions and uses the same translation loop,
length sorting, source limits (including zh-en's 512), and greedy settings as
evaluation. This command generates calibration data only; it never loads WMT
test examples. The input set's contamination check is carried into the cache.

Point a pruning config at the cache:

```yaml
calibration: data/calibration/multi-10dir-generated
calibration_text: prompt+generated
directions: [de-en, en-de]  # Both directions of one pair; omit for multi.
max_length: 768
```

Both SlimGPT and FLAP use this mode. The original `target` remains available
for `prompt+target` comparisons on exactly the same records. Generated mode
replays the actual input and continuation token IDs, including EOS, rather
than retokenizing the cleaned hypothesis. This preserves prompt truncation,
token boundaries and extra output text. Empty answers remain in the set.
`max_length: 768` covers the 512-token source cap plus the 256-token generation
budget; shorter caps are rejected when they would cut a cached sequence.
For a controlled reference/generated comparison, use the same pruning cap
and records in both arms. Token counts still differ between their responses.
Reference mode retains its existing text tokenization; inspect the cache's
`source_truncated` counts because generated mode replays evaluation's source
cap, while reference mode truncates the combined text at the pruning cap.

The cache stores raw text, extracted translations, diagnostic flags and token
counts alongside the exact tokens. Its identity includes input content,
model/tokenizer revision, prompt, decoding settings and generation code/library
versions. Directions are written atomically; rerunning the same command
validates and skips completed ones. Interrupted caches cannot be used for
pruning. Changed inputs/settings require a fresh output directory; corrupted
files are rejected. The pruning checkpoint records its calibration mode,
directions and cache fingerprint in `calibration_provenance.json`, and the
subnetwork descriptor carries the fingerprint too. Hub pruning loads the
same pinned revision that generated the cache. Local checkpoints are checked
by file sizes and modification times; keep their contents immutable.

The matched 20% pilot config is `configs/prune/slimgpt-20-multi-generated.yaml`.
It retains the global budget and 512-token pruning cap of the existing
prompt+reference pilot. The initial generated cache's longest sequence is
494 tokens, so it fits without clipping. Evaluate with
`configs/suites/alma10-greedy-300.yaml` for the same 300 examples per direction.

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
state `r_0`, but the curve in its Figure 4 (LLaMA-7B at 50 percent) fits
equation 6 exactly with `r_0` a quarter of the target, so that is used here,
and `r_last` is solved so the mean over layers is the requested sparsity. On
ALMA-7B that is 0.05 at layer 0 rising to 0.254 at layer 31 for 20 percent,
0.075 to 0.381 for 30, and 0.10 to 0.508 for 40. The reasoning
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

Other departures from ALMA: the final adapter is kept where ALMA kept the best
of evaluations every 5 percent, and the prompt is fully masked where ALMA left
its last token as a label. The module docstring lists them all.

At least 500 segments are held out and scored before and after training. They
are chosen by English sentence, with every translation of each, because ALMA's
data is partly multi-way parallel and every pair appears in both directions; a
split by record would leave most held-out sentences in training under another
direction. It is still the training distribution, so treat the pair of numbers
as a check that training worked rather than as generalisation. `repair.json`
beside the checkpoint records both, with the loss curve, and the command fails,
after writing everything, if the held-out loss did not fall or is not finite.

## Transformers version

The pruning extra pins `transformers>=4.48,<5`, tighter than `gen`. Since 4.48
attention forward reshapes with `(*input_shape, -1, self.head_dim)`, so head
counts come off tensor shapes and `num_key_value_groups` is the only attribute
the compaction has to patch. Before 4.48 it reshaped with an explicit
`self.num_heads`, which the pin excludes.

## Cost

For SlimGPT on ALMA-7B at 1,280 calibration segments, peak GPU memory is the
13.5 GB of weights, two copies of the hidden states (a layer's input and
output), and about 4 GB of float64 matrices for `down_proj` (11008 squared:
the inverse Hessian, its factor and the weights). The linear algebra takes
under a second per layer; forward passes dominate, about five minutes in all. The knob is the number of calibration segments, which the buffer is
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
- SlimGPT's compensated weights equal the closed-form least-squares optimum
  for removed columns at any position, and its selection does not change when
  columns are permuted;
- tokens behind the padding mask do not move any score, and for SlimGPT not
  the Hessian either, which the compensation solves against as well as ranks
  by.
