# Evaluation protocol

What is measured, how, and which parts are enforced in code rather than by
convention.

## The comparability contract

A compression comparison is worthless if the systems were not measured
identically. Rather than documenting that and hoping, the framework makes it
structural.

**Run identity.** A run is identified by a hash of three things: the model
specification, the data specification, and the decode specification. Changing
any of them produces a different run directory rather than overwriting an
existing one. Labels do not participate: renaming a model, editing its `notes`,
or changing `device_map` leaves its identity intact, so no existing run is
orphaned by a cosmetic edit.

**Prompt fingerprinting.** Every run records a digest derived from rendering
the prompt template against fixed probe inputs. An accidental whitespace change
to a template changes the digest, and two runs with different digests are not
comparable.

**Sampling is off unless requested.** `do_sample` defaults to false, and
setting `temperature`, `top_p` or `top_k` while it is false is a configuration
error rather than a silently ignored field. Several ALMA-family checkpoints
inherit a `generation_config.json` from Llama 2 that enables sampling, so the
generator passes every sampling parameter explicitly instead of letting the
checkpoint decide how it is evaluated. Note that ALMA's own README quick-start
samples while its evaluation scripts use pure beam search; the scripts produced
the published numbers.

**The report refuses to mix settings.** Runs are grouped by their data and
decode identity, and each group becomes its own section with a warning. A greedy
sweep and a beam-5 headline run never land in the same table.

**Padding effects are measured, not assumed.** Length-bucketed batching gives
every sequence a different amount of padding, and beam search is not exactly
invariant to that. `--deterministic-check N` re-translates N segments at batch
size 1 with bucketing disabled and reports the disagreement rate.

## Matching ALMA

The protocol reproduces ALMA's evaluation so scores stay comparable with the
published tables.

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
`human_written_data` files segment by segment, so bit-compatibility of the test
data is a checked fact rather than an assumption.

Exact agreement with ALMA's published scores is not expected: their harness
pads through the Hugging Face `Trainer`, and padding differences perturb beam
search slightly. `configs/suites/wmt22-alma-repro.yaml` disables length
bucketing and uses their batch size to get as close as possible. The recorded
delta is the framework's own correctness evidence and belongs in the report.

## Hypothesis extraction

Generation decodes only the newly generated tokens, by slicing off the prompt at
its tokenized length. ALMA instead string-splits the full decoded sequence on
the target-language cue inside three nested `try` blocks, returning an empty
string when all of them fail. For a project about degradation that is the worst
possible behaviour: a collapsed model scores a legitimate-looking zero and
nobody notices.

Every hypothesis carries a status and a set of flags recording what had to be
done to recover it:

| Flag | Meaning |
| --- | --- |
| `skipped_blank` | Blank lines preceded the hypothesis |
| `stripped_marker` | The model echoed the target-language cue back |
| `extra_lines` | Content after the hypothesis was discarded |
| `repetition` | The output ends in a repeated n-gram |

The raw output is retained in `hyps/<direction>.jsonl` so a parsing question can
be answered without regenerating a ten-thousand segment run.

`mnlp_eval.postprocess.alma_legacy_clean` reimplements upstream's parser
faithfully, so the framework can quantify how often it would have produced an
empty string on the same outputs.

## Quality metrics

| Metric | Checkpoint | Notes |
| --- | --- | --- |
| BLEU | sacreBLEU | Signature recorded per direction |
| chrF++ | sacreBLEU | `nc:6 nw:2 space:no` |
| TER | sacreBLEU | Optional, lower is better |
| COMET-22 | `Unbabel/wmt22-comet-da` | Primary metric, ungated |
| COMETKiwi | `Unbabel/wmt22-cometkiwi-da` | Reference-free, gated |
| XCOMET-XL | `Unbabel/XCOMET-XL` | WMT24 joint winner, gated |
| MetricX-24 | `google/metricx-24-hybrid-large-v2p6-bfloat16` | WMT24 joint winner, error score in [0, 25], lower is better |

For reference, the WMT24 Metrics Shared Task ranked metrics by weighted average
correlation over six tasks: MetaMetrics-MT 0.725, MetricX-24-Hybrid 0.721,
XCOMET 0.719, COMET-22 0.688, BLEURT-20 0.686. BLEU is far below all of them,
which is why it is reported but not relied upon.

Per-segment neural scores are stored at scoring time. Without them, bootstrap
significance would require reloading a multi-billion-parameter metric model and
rescoring the whole test set.

Aggregation is an unweighted macro mean over directions, the convention in MT
result tables. A metric is aggregated only if every scored direction reported
it, so a partially failed scoring pass cannot produce an average that covers
fewer directions than the table claims.

## Behavioural metrics

Quality scores answer how good the output is on average. These answer how it
fails, which is what distinguishes compression methods that degrade gracefully
from those that collapse on part of the input.

| Metric | Why it matters |
| --- | --- |
| `off_target_rate` | The signature failure of compressed multilingual models, and invisible in every quality metric |
| `source_language_rate` | Output in the source language means the model failed to translate at all, not that it translated badly |
| `english_fallback_rate` | Falling back to English when the target is not English |
| `empty_rate` | Empty output is a scoring artefact worth separating from genuine mistranslation |
| `source_copy_rate` | Copying the source, detected case-insensitively and ignoring punctuation |
| `repetition_rate` | Degenerate cycles, a classic low-bit collapse mode |
| `truncation_rate` | The hypothesis itself was cut off |
| `budget_hit_rate` | Generation ran out of budget, whether or not the hypothesis was cut |
| `wasted_token_fraction` | Tokens generated after the hypothesis and discarded |
| `length_ratio` | Hypothesis to reference characters, which catches systematic under-generation |

Two of these deserve elaboration.

**Truncation versus budget.** These are separate signals because they mean
different things. A model that finishes the translation and then keeps
generating commentary hits the token budget without truncating anything. A model
that runs out of budget mid-sentence has produced a damaged translation. The
first is an efficiency problem, the second a quality problem. Conflating them
makes an instruction-tuned model look broken when it is merely verbose. In
measured runs, `Qwen2.5-0.5B-Instruct` hits the budget on every segment while
truncating almost none of them.

**Language identification.** The detector is restricted to the languages
plausible for the direction: the target, the source, and English. A restricted
decision is far more reliable on a single sentence than an open choice among 142
languages, and it answers the actionable question directly. The unrestricted
rate is reported alongside it as `off_target_rate_open`. Hypotheses shorter than
12 characters are excluded, because language identification on a two-word
fragment is noise and counting that noise as off-target would manufacture a
finding.

## Efficiency

Metrics follow Section 2.1 of the course reference survey (Zhu et al., "A Survey
on Model Compression for Large Language Models"): model size, FLOPs, MFU,
inference time, speedup ratio, compression ratio.

**Three compression ratios, not one.** Checkpoint bytes, parameter count, and
resident VRAM are reported separately because they diverge. Load-time
quantization such as bitsandbytes leaves the checkpoint untouched, so its disk
ratio reads 1.00x while its VRAM ratio is 2.00x or better. Quoting only the
flattering one is the standard way these reports go wrong. `bits_per_parameter`
is reported for both disk and resident memory, and is the quickest sanity check
available: an fp32 model must read 32, an fp16 model 16, a 4-bit model near 4.

**Fixed token budget.** Every model is forced to generate exactly the same
number of tokens, with `min_new_tokens` equal to `max_new_tokens`. Without this,
a model that stops early looks faster, and a compression method that happens to
shorten outputs would be credited with a speedup it did not deliver. Natural
throughput is separately reported by the generate stage.

**Two batch sizes.** Batch size 1 is memory-bandwidth bound, where weight-only
quantization helps most. A large batch is compute bound, where dequantization
overhead can make a quantized model slower than the baseline. Reporting one
number would let a method look good or bad purely through that choice. This is
not hypothetical: measured on this repository, casting a Marian model to fp16
gives 0.93x throughput at batch size 1 and 1.18x at batch size 8. A single
batch size would have supported either conclusion.

**Measurement procedure.** Warmup iterations are discarded, several timed
repeats are taken, and the median is reported with its spread. Peak memory comes
from both the torch allocator, which is the right figure for comparing model
footprints, and NVML, which includes the CUDA context and is the right figure
for capacity planning. Device name, driver, SM clock, temperature and power draw
are recorded, because these runs share a cluster and a throttled GPU moves
latency by more than some compression methods do.

**FLOPs and MFU are labelled honestly.** FLOPs are an analytic estimate from
parameter and token counts, not a traced operator count. MFU is computed against
the device's dense bf16 peak, so a 4-bit model's figure is not a utilisation
number in the usual sense; the field is named `mfu_bf16_equivalent` for that
reason. On a device absent from the peak-FLOPS table, MFU is reported as null
with a note naming the table to extend, rather than being guessed.

## Significance

Deltas are reported with p-values from paired bootstrap resampling. A 0.3 BLEU
drop on 1000 segments is not evidence of anything, and saying so is more useful
than reporting it as a finding.

Surface metrics use sacreBLEU's `PairedTest`, which recomputes BLEU and chrF++
from sufficient statistics on each resample. That is correct in a way that
averaging per-sentence BLEU is not. Neural metrics resample the stored
per-segment scores. Both use the same centred p-value with add-one smoothing, so
a p-value of exactly zero is never reported from a finite number of samples, and
both are paired: the same resampled indices apply to both systems, which removes
test-set difficulty as a source of variance.
