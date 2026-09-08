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
bucketing to get as close as a single-process harness can. It does not match
their batch size, and cannot: `evals/alma_7b.sh` passes
`--per_device_eval_batch_size 2` under `accelerate launch` with a DeepSpeed
config, so per-device 2 across several devices is a different padding pattern
from a single process at any batch size. The recorded delta is the framework's
own correctness evidence and belongs in the report.

## Hypothesis extraction

Generation decodes only the newly generated tokens, by slicing off the prompt at
its tokenized length. ALMA instead string-splits the full decoded sequence on
the target-language cue inside two sequential `try` blocks, the first of which
tries three candidate lines, and returns an empty string when both fail. For a
project about degradation that is the worst possible behaviour: a collapsed
model scores a legitimate-looking zero and nobody notices.

Generation also stops at the first of *any* of a model's terminator tokens,
taken from the union of the tokenizer's and the generation config's. Reading
only `tokenizer.eos_token_id` is wrong for a large family of models: Qwen2.5
lists two terminators and its pad token is the second one, so a model that
stopped normally looked like it never stopped and its padding was counted as
generated text.

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
closely enough to demonstrate the failure, and is unit tested, but no stage
calls it: the framework does not currently quantify upstream's empty-string
rate on its own outputs. Two known divergences, both making the copy
*optimistic*: where the first three lines after the cue are all blank, upstream
returns an empty string from its first block while the copy falls through to
the second and can return something non-empty; and upstream uses a bare
`except` where the copy catches `IndexError` only.

## Quality metrics

| Metric | Checkpoint | Notes |
| --- | --- | --- |
| BLEU | sacreBLEU | Signature recorded per direction |
| chrF++ | sacreBLEU | `nc:6 nw:2 space:no` |
| TER | sacreBLEU | Optional, lower is better |
| COMET-22 | `Unbabel/wmt22-comet-da` | Primary metric, ungated |
| COMETKiwi | `Unbabel/wmt22-cometkiwi-da` | Reference-free, gated |
| XCOMET-XL | `Unbabel/XCOMET-XL` | XL member of a WMT24 winning family, gated |
| MetricX-24 | `google/metricx-24-hybrid-large-v2p6-bfloat16` | large variant of a WMT24 winning family, error score in [0, 25], lower is better |

The WMT24 Metrics Shared Task ranked metrics by weighted average correlation
over six tasks: MetaMetrics-MT 0.725, MetricX-24-Hybrid 0.721, XCOMET 0.719,
COMET-22 0.688, BLEURT-20 0.686, BLEU 0.589 (23rd of 26). BLEU is reported but
not relied upon.

Two caveats on the winners, since it is easy to overclaim here. The ranked
XCOMET submission was an ensemble of XCOMET-XXL and XCOMET-XL, not XCOMET-XL
alone; and MetricX-24-Hybrid is the mT5-XXL model, where the `large` variant
configured here scores 0.705 against 0.716 by Google's own numbers. These are
therefore the affordable members of two winning families, not the winning
systems. MetaMetrics-MT outscored both and is not implemented.

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
| `on_target_rate` | Share of all segments confirmed to be in the target language |
| `off_target_rate` | The signature failure of compressed multilingual models, and invisible in every quality metric |
| `unverifiable_rate` | Share too empty or too short to identify. These three sum to one |
| `source_language_rate` | Output in the source language means the model failed to translate at all, not that it translated badly |
| `english_fallback_rate` | Falling back to English. Reported only where the source is not English, since otherwise it is the same event as `source_language_rate` |
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
makes an instruction-tuned model look broken when it is merely verbose.
Measured on the first 16 segments of WMT22 de-en at beam 4 with
`max_new_tokens=128`, `Qwen2.5-0.5B-Instruct` exhausts the budget on 11 of 16
segments, truncates none of them, and discards 83 percent of its generated
tokens as trailing commentary.

**Language identification.** The detector is restricted to the languages
plausible for the direction: the target, the source, and English. A restricted
decision is far more reliable on a single sentence than an open choice among
every language the classifier knows, and it answers the actionable question
directly. The unrestricted rate is reported alongside it as
`off_target_rate_open`. Hypotheses shorter than 12 characters are excluded from
classification, because language identification on a two-word fragment is
noise.

**Every language rate divides by the total segment count, not by the
classifiable subset.** An earlier version divided by the subset, which made a
model that emitted nothing on 95 percent of segments report an off-target rate
of zero: the most flattering possible value for the most broken possible
system, printed next to rates that used the full denominator.
`on_target_rate`, `off_target_rate` and `unverifiable_rate` now sum to one by
construction, which is what makes that failure impossible to reintroduce
unnoticed. The subset figure is retained under the explicit name
`off_target_rate_among_scorable`, for comparison with the literature, and is
never the headline.

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
averaging per-sentence BLEU is not, and it is what the MT literature reports.
Neural metrics resample the stored per-segment scores.

**The two are not the same estimator.** sacreBLEU centres the absolute score
difference on its own bootstrap mean and counts one-sided; the segment-level
path is a two-sided centred bootstrap, `P(|d_b - d| >= |d|)`. Both are paired
and both are add-one smoothed, so neither reports a p-value of exactly zero
from a finite number of samples, but their rejection rates under the null
differ and a p-value from one is not directly comparable with a p-value from
the other. An earlier version of this document claimed they were the same
statistic. They are not.

Both are genuinely paired: the same resampled indices apply to both systems,
which removes test-set difficulty as a source of variance.

The table applies one asterisk threshold per cell and no multiplicity
correction across directions, systems and metrics. With ten directions and
several systems that is many tests, so treat a lone asterisk as weak
evidence.
