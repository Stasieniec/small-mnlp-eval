# Overnight run, 2026-09-29

Running log of the unattended GPU run on scur0560 (Snellius), branch
`claude/busy-bell-pyhq1v`. Times are CEST. The morning write-up is at the top
once it exists; the log below is chronological.

## Log

### Phase 0: setup (login node int4)

- 19:08 Checked out `claude/busy-bell-pyhq1v` at `a2e4532`. The account had
  nothing set up: no `.venv`, no `.venv-comet`, no `HF_HOME`, no
  `/scratch-shared/scur0560`.
- 19:10 `HF_HOME=/scratch-shared/$USER/hf_home` created and exported in
  `~/.bashrc`. Batch scripts do not read `~/.bashrc`; they inherit the
  submitting shell through `--export ALL`, so every submission below was made
  from a shell with `HF_HOME` exported.
- 19:12 Both venvs built with `uv venv --managed-python --python 3.11`
  (CPython 3.11.16 under `~/.local/share/uv`, `Python.h` present). `.venv`:
  torch 2.14.0+cu130, transformers 4.55.4, peft 0.21.1. `.venv-comet`:
  unbabel-comet 2.2.7, torch 2.14.0+cu130.
- 19:17 `pytest tests/ -q`: all pass.
- 19:18 Budget (`accinfo`, `budget-overview`): shared course account
  `gpuuva086`, 406,439 SBU remaining, 388,359 SBU left for dispatch after
  other users' running jobs. Billing weights from `scontrol show partition`:
  `gpu_a100` bills `gres/gpu=128, cpu=7.11` per hour, max-of-TRES (a whole
  node is 512 either way), so one A100 with 18 cores is 128 SBU/h, as the
  brief assumed. `gpu_h100` is 192 per GPU and 12 per core. The plan's
  40-45 A100-hours is about 5,800 SBU, 1.5% of what is left, so nothing is
  dropped for budget. No per-user job or QoS limits
  (`sacctmgr show assoc`, `show qos`).
- 19:19-19:21 Calibration sets built, no collisions: multi-10dir (1,280
  segments, fingerprint 44f1fd1f413fbce1), multi-10dir-256 (2,560,
  2be4440baf50dce0), pair-cs (83ae5b5b5f939a31), pair-de (ce4605a66bf86226),
  pair-is (0cac80da404e13b2), pair-ru (c24f4e603a1db741), pair-zh
  (75876fd8936b8368), 256 segments each.
- 19:21 **repair-multi FAILED the contamination check: 2 of 117,404 segments
  collide with haoranxu/WMT22-Test.** Per the brief this is a hard failure:
  no repair runs tonight on this data, and nothing works around it. The two
  collisions (exact source-string matches):
  - en-cs, training record 5463, source `Amazing.` (train target `Nádherný.`),
    equal to WMT22 en-cs test segment 1356 (reference `Úžasné.`);
  - ru-en, training record 614, source `Ну и что?` (train target `So what?`),
    equal to WMT22 ru-en test segment 311 (reference `So what?`).
  Both lie outside the first 300 segments that `alma10-greedy-300` evaluates.
  `mnlp-eval calibration` exits 1 but still writes the set to disk under its
  usable name, and `submit_repair.sh` only checks that the directory exists,
  so the directory was renamed to `data/calibration/repair-multi.CONTAMINATED`
  so that nothing can train on it by accident.
- 19:22 Prefetch done: ALMA-7B, ALMA-7B-R, NLLB-600M, opus-mt-de-en,
  Qwen2.5-0.5B-Instruct, WMT22 and FLORES-200 for all ten directions,
  `Unbabel/wmt22-comet-da` and the `xlm-roberta-large` encoder files.
  COMETKiwi and XCOMET-XL failed as expected (gated, no `HF_TOKEN`); the
  default metric set does not use them.
- 19:23 COMET offline check on the login node passes:
  `HF_HUB_OFFLINE=1 .venv-comet/bin/python -c "... load_from_checkpoint(download_model('Unbabel/wmt22-comet-da'))"`
  prints `ok`.
- 19:24 Probe job 27370124 (1 A100, 50 s, COMPLETED): driver 595.91.07 on
  A100-SXM4-40GB, so the cu130 torch wheels work; `torch.cuda.is_available()`
  true, a bf16 matmul and a `torch.compile` (triton JIT, the step that needed
  `Python.h`) succeed in both venvs; `HF_HOME` reaches the job. Its
  `AllocTRES` shows `billing=128`, confirming the rate. The probe was worth
  it because `default_device()` falls back to CPU silently, so a driver
  mismatch would have had every prune job grind on CPU for four hours.

### Queued at 19:27

The smoke chain, the dense baseline, and all fourteen prune jobs with an
evaluation (generate, then score) behind each. The three real repair jobs
and their evaluations were NOT queued (repair-multi is contaminated, above).
The report is queued once the last score job id is known.

| system | prune | repair | generate | score |
| --- | --- | --- | --- | --- |
| smoke-qwen-slimgpt20 | 27370221 | | 27370224 | 27370225 |
| smoke-qwen-slimgpt20-lora | | 27370222 | 27370226 | 27370227 |
| alma-7b (dense) | | | 27370229 | 27370230 |
| alma-7b-slimgpt20-multi | 27370233 | | 27370234 | 27370235 |
| alma-7b-slimgpt50-multi | 27370236 | | 27370238 | 27370239 |
| alma-7b-slimgpt20-cs | 27370240 | | 27370242 | 27370243 |
| alma-7b-slimgpt20-de | 27370245 | | 27370246 | 27370247 |
| alma-7b-slimgpt20-is | 27370248 | | 27370249 | 27370250 |
| alma-7b-slimgpt20-ru | 27370251 | | 27370252 | 27370253 |
| alma-7b-slimgpt20-zh | 27370254 | | 27370256 | 27370257 |
| alma-7b-slimgpt20-multi-uniform | 27370262 | | 27370263 | 27370264 |
| alma-7b-slimgpt20-multi-log | 27370267 | | 27370268 | 27370269 |
| alma-7b-slimgpt50-multi-log | 27370270 | | 27370271 | 27370273 |
| alma-7b-slimgpt20-multi-target | 27370275 | | 27370276 | 27370278 |
| alma-7b-slimgpt20-multi-256 | 27370279 | | 27370280 | 27370281 |
| alma-7b-flap20-multi | 27370282 | | 27370283 | 27370284 |
| alma-7b-flap20-de | 27370285 | | 27370286 | 27370287 |
