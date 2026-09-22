# Evaluation protocol

What is measured, how, and which parts are enforced in code rather than by
convention.

## Comparability

**Run identity.** A run is identified by a hash of the model, data and decode
specifications. Changing any of them produces a different run directory rather
than overwriting one. Labels do not participate, so renaming a model or editing
its notes leaves its identity intact.

**Prompt fingerprinting.** Every run records a digest of the prompt template
rendered against fixed probe inputs. A stray whitespace change alters the
digest, and two runs with different digests are not comparable.

**Sampling is off unless requested.** `do_sample` defaults to false, and
setting `temperature`, `top_p` or `top_k` while it is false is a configuration
error rather than a silently ignored field. Several ALMA-family checkpoints
inherit a `generation_config.json` from Llama 2 that enables sampling, so the
generator passes every sampling parameter explicitly instead of letting the
checkpoint decide how it is evaluated. ALMA's README quick-start samples while
its evaluation scripts use pure beam search; the scripts produced the published
numbers.

**The report refuses to mix settings.** Runs are grouped by data and decode
identity, and each group becomes its own section with a warning.

**Padding effects are measured, not assumed.** Length-bucketed batching gives
each sequence a different amount of padding, and beam search is not exactly
invariant to that. `--deterministic-check N` re-translates N segments at batch
size 1 with bucketing off and reports the disagreement rate.

## Matching ALMA

| Setting | Value | Source |
| --- | --- | --- |
| Prompt | `Translate this from {Src} to {Tgt}:\n{Src}: {sentence}\n{Tgt}:` | `utils/utils.py:get_prompt` |
| Beams | 5 | `evals/alma_7b.sh` |
| `max_new_tokens` | 256 | `evals/alma_7b.sh` |
| `max_source_length` | 256, raised to 512 for zh-en | `evals/alma_7b.sh` |
| Sampling | off | `evals/alma_7b.sh` |
| Seed | 42 | `evals/alma_7b.sh` |
| Padding side | left, for decoder-only | ALMA's tokenizer setup |
| BLEU tokenizer | `13a`, `zh` for Chinese, `ja-mecab` for Japanese | `evals/eval_generation.sh` |
| Test sets | `haoranxu/WMT22-Test`, `WMT23-Test`, `FLORES-200` | ALMA README |

The upstream README writes "into" where the code writes "to". The code is
authoritative.

`mnlp-eval verify-testset` confirms the Hub datasets match ALMA's committed
`human_written_data` files segment by segment.

Exact agreement with ALMA's published scores is not expected: their harness
pads through the Hugging Face `Trainer`, and padding differences perturb beam
search. `configs/suites/wmt22-alma-repro.yaml` disables length bucketing to get
as close as a single-process harness can. It does not match their batch size
and cannot: `evals/alma_7b.sh` passes `--per_device_eval_batch_size 2` under
`accelerate launch` with DeepSpeed, and per-device 2 across several devices is
a different padding pattern from a single process at any batch size. The
recorded delta is the framework's own correctness evidence.

## Hypothesis extraction

Generation decodes only the newly generated tokens, by slicing off the prompt
at its tokenized length. ALMA instead string-splits the full decoded sequence
on the target-language cue inside two sequential `try` blocks and returns an
empty string when both fail. For a project about degradation that is the worst
available behaviour, since a collapsed model then scores a legitimate-looking
zero.

Generation stops at the first of any of a model's terminator tokens, taken from
the union of the tokenizer's and the generation config's. Reading only
`tokenizer.eos_token_id` is wrong for a large family of models: Qwen2.5 lists
two terminators and its pad token is the second, so a model that stopped
normally looks like it never stopped and its padding counts as generated text.

Every hypothesis carries a status and flags recording what had to be done to
recover it: `skipped_blank`, `stripped_marker`, `extra_lines`, `repetition`.
The raw output is retained in `hyps/<direction>.jsonl` so a parsing question can
be answered without regenerating a ten-thousand segment run.

## Quality metrics

| Metric | Checkpoint | Notes |
| --- | --- | --- |
| BLEU | sacreBLEU | Signature recorded per direction |
| chrF++ | sacreBLEU | `nc:6 nw:2 space:no` |
| TER | sacreBLEU | Optional, lower is better |
| COMET-22 | `Unbabel/wmt22-comet-da` | Primary metric, ungated |
| COMETKiwi | `Unbabel/wmt22-cometkiwi-da` | Reference-free, gated |
| XCOMET-XL | `Unbabel/XCOMET-XL` | Gated |
| MetricX-24 | `google/metricx-24-hybrid-large-v2p6-bfloat16` | Error score in [0, 25], lower is better |

The WMT24 Metrics Shared Task ranked metrics by weighted average correlation
over six tasks: MetaMetrics-MT 0.725, MetricX-24-Hybrid 0.721, XCOMET 0.719,
COMET-22 0.688, BLEURT-20 0.686, BLEU 0.589 (23rd of 26). BLEU is reported but
not relied on.

Two caveats, since it is easy to overclaim. The ranked XCOMET submission was an
ensemble of XCOMET-XXL and XCOMET-XL, not XCOMET-XL alone; and
MetricX-24-Hybrid is the mT5-XXL model, where the `large` variant configured
here scores 0.705 against 0.716 by Google's own numbers. These are the
affordable members of two winning families, not the winning systems.
MetaMetrics-MT outscored both and is not implemented.

Per-segment neural scores are stored at scoring time, so bootstrap significance
does not require reloading a multi-billion-parameter metric model.

Aggregation is an unweighted macro mean over directions. A metric is aggregated
only if every scored direction reported it, so a partially failed scoring pass
cannot produce an average covering fewer directions than the table claims.

## Behavioural metrics

| Metric | Meaning |
| --- | --- |
| `on_target_rate` | Segments confirmed to be in the target language |
| `off_target_rate` | The signature failure of compressed multilingual models, invisible in every quality metric |
| `unverifiable_rate` | Too empty or too short to identify. These three sum to one |
| `source_language_rate` | Output in the source language: the model failed to translate at all |
| `english_fallback_rate` | Reported only where the source is not English, since otherwise it is the same event |
| `empty_rate` | Empty output, worth separating from genuine mistranslation |
| `source_copy_rate` | Copying the source, case-insensitive and ignoring punctuation |
| `repetition_rate` | Degenerate cycles, a classic low-bit collapse mode |
| `truncation_rate` | The hypothesis itself was cut off |
| `budget_hit_rate` | Generation ran out of budget, whether or not the hypothesis was cut |
| `wasted_token_fraction` | Tokens generated after the hypothesis and discarded |
| `length_ratio` | Hypothesis to reference characters, catching systematic under-generation |

**Truncation versus budget** are separate signals. A model that finishes the
translation and keeps generating commentary hits the budget without truncating
anything, which is an efficiency problem. A model that runs out of budget
mid-sentence has produced a damaged translation, which is a quality problem.
Conflating them makes a verbose instruction-tuned model look broken. On the
first 16 segments of WMT22 de-en at beam 4 with `max_new_tokens=128`,
`Qwen2.5-0.5B-Instruct` exhausts the budget on 11 of 16 segments, truncates
none, and discards 83 percent of its generated tokens as trailing commentary.

**Language identification** is restricted to the languages plausible for the
direction: the target, the source, and English. A restricted decision is more
reliable on a single sentence than an open choice among every language the
classifier knows. The unrestricted rate is reported as `off_target_rate_open`.
Hypotheses shorter than 12 characters are excluded, since identification on a
two-word fragment is noise.

**Every language rate divides by the total segment count**, not by the
classifiable subset. Dividing by the subset lets a model that emits nothing on
95 percent of segments report an off-target rate of zero, because the few
segments it does produce are the ones it got right. The subset figure is
retained under the explicit name `off_target_rate_among_scorable` and is never
the headline.

## Structure and sparsity

Read from the loaded model rather than from its config, so a checkpoint whose
config and tensors disagree is visible. Per layer: attention and FFN width, the
head count where the head dimension is known, and the share of parameters that
are exactly zero. Encoder and decoder stacks are kept separate.

## Pruning method

`compression.method` records which criterion selected the subnetwork and how
the budget was spread, because the three differ in what they measure and two of
the three change the surviving weights rather than only selecting among them.

- **FLAP** ranks a unit by the variance of its input feature weighted by the
  norm of the weights reading it, and adds the removed units' mean output back
  as a bias. A FLAP checkpoint therefore carries biases on `o_proj` and
  `down_proj` that the dense model does not have.
- **LLM-Pruner** ranks by a first-order Taylor expansion of the calibration
  loss and changes nothing else. Its published results assume a LoRA repair,
  which is a separate stage and appears in `compression.repair`.
- **SlimGPT** ranks by the optimal-brain-surgery cost and applies the
  compensating weight update to the surviving columns. Its numbers are not
  comparable to a run of the same criterion without that update.

`uniform` and `global` name how the sparsity budget was spread over the layers.
Only `uniform` leaves every layer the same width. See
[pruning.md](pruning.md).

A `uniform: false` result means the pruning criterion spent its budget unevenly
across depth, which a single sparsity number hides. A high zero fraction means
a mask was applied but the weights were never compacted, so the parameter
count, checkpoint size and latency are all unchanged and no claimed saving is
real. Integer-dtype parameters are excluded from the zero count, since a zero
byte in a 4-bit checkpoint is a quantization level rather than a pruned weight.

Recognised projection names cover Llama, which is what ALMA is, and the
BART-style naming Marian and NLLB share. Another architecture reports parameter
and zero counts as usual and leaves the structural fields null.

## Efficiency

Metrics follow Section 2.1 of Zhu et al., "A Survey on Model Compression for
Large Language Models": model size, FLOPs, MFU, inference time, speedup ratio,
compression ratio.

**Three compression ratios.** Checkpoint bytes, parameter count and resident
VRAM are reported separately because they diverge: load-time quantization
leaves the checkpoint untouched, so its disk ratio reads 1.00x while its VRAM
ratio is 2.00x or better. `bits_per_parameter` is reported for both disk and
resident memory and is the quickest available sanity check: an fp32 model must
read 32, an fp16 model 16, a 4-bit model near 4.

**Fixed token budget.** Every model generates exactly the same number of
tokens, with `min_new_tokens` equal to `max_new_tokens`. Otherwise a model that
stops early looks faster, and a compression method that happens to shorten
outputs would be credited with a speedup it did not deliver. Natural throughput
is reported separately by the generate stage.

**Two batch sizes.** Batch size 1 is memory-bandwidth bound, where weight-only
quantization helps most; a large batch is compute bound, where dequantization
overhead can make a quantized model slower. Reporting one number would let a
method look good or bad through that choice alone.

**Measurement procedure.** Warmup iterations are discarded, several timed
repeats are taken, and the median is reported with its spread. Peak memory
comes from both the torch allocator, which is the right figure for comparing
model footprints, and NVML, which includes the CUDA context and is the right
figure for capacity planning. Device name, driver, SM clock, temperature and
power draw are recorded, because a throttled GPU moves latency by more than
some compression methods do.

Treat a throughput difference smaller than the measured spread as no
difference. Casting a 74M-parameter Marian model to fp16 on an RTX 1000 Ada
measured 0.86x at batch 1 and 0.88x at batch 8 over nine repeats, with a
run-to-run spread of 2.3 to 10 percent of the median. Halving the weights
bought no speed, and a three-repeat measurement of the same thing had suggested
a batch-size crossover that did not reproduce.

**FLOPs and MFU are labelled honestly.** FLOPs are an analytic estimate from
parameter and token counts, not a traced operator count, and are not estimated
for encoder-decoder models. MFU is computed against the device's dense bf16
peak, so a 4-bit model's figure is not a utilisation number in the usual sense;
the field is named `mfu_bf16_equivalent`. On a device absent from the
peak-FLOPS table, MFU is null with a note naming the table to extend.

## Significance

Deltas are reported with p-values from paired bootstrap resampling. A 0.3 BLEU
drop on 1000 segments is not evidence of anything.

Surface metrics use sacreBLEU's `PairedTest`, which recomputes BLEU and chrF++
from sufficient statistics on each resample rather than averaging per-sentence
scores. Neural metrics resample the stored per-segment scores.

**The two are not the same estimator.** sacreBLEU centres the absolute score
difference on its own bootstrap mean and counts one-sided; the segment-level
path is a two-sided centred bootstrap, `P(|d_b - d| >= |d|)`. Both are paired
and add-one smoothed, so neither reports exactly zero from a finite number of
samples, but their rejection rates under the null differ and a p-value from one
is not directly comparable with a p-value from the other.

Both are genuinely paired: the same resampled indices apply to both systems,
which removes test-set difficulty as a source of variance.

The table applies one asterisk threshold per cell and no multiplicity
correction across directions, systems and metrics. With ten directions and
several systems that is many tests, so treat a lone asterisk as weak evidence.
