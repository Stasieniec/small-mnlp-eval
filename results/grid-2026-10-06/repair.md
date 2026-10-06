# Repair on the directions each model is for

Macro over the repaired model's own directions (all ten for multi; its pair for a
pair model, which is repaired on its pair's data only). Recovered = (repaired -
pruned) / (dense - pruned) on COMET.

| Repaired system | Directions | Dense COMET | Pruned COMET | Repaired COMET | Recovered | Pruned BLEU | Repaired BLEU | Pruned chrF++ | Repaired chrF++ |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| alma-7b-flap40-ref-multi-lora | all ten | 0.8474 | 0.7393 | 0.8112 | 67% | 20.50 | 25.73 | 40.94 | 46.84 |
| alma-7b-slimgpt20-ref-multi-lora | all ten | 0.8474 | 0.8374 | 0.8381 | 7% | 27.72 | 28.41 | 49.03 | 49.43 |
| alma-7b-slimgpt30-gen-multi-lora | all ten | 0.8474 | 0.8229 | 0.8324 | 39% | 25.88 | 27.59 | 47.58 | 48.65 |
| alma-7b-slimgpt40-gen-multi-lora | all ten | 0.8474 | 0.7905 | 0.8179 | 48% | 22.92 | 26.36 | 44.44 | 47.40 |
| alma-7b-slimgpt40-gen-pair-cs-lora | cs-en, en-cs | 0.8614 | 0.8275 | 0.8348 | 21% | 26.87 | 28.96 | 50.18 | 52.16 |
| alma-7b-slimgpt40-gen-pair-de-lora | de-en, en-de | 0.8461 | 0.8175 | 0.8307 | 46% | 24.20 | 26.45 | 49.76 | 51.40 |
| alma-7b-slimgpt40-gen-pair-is-lora | en-is, is-en | 0.8523 | 0.8120 | 0.8274 | 38% | 23.22 | 23.97 | 47.52 | 48.25 |
| alma-7b-slimgpt40-gen-pair-ru-lora | en-ru, ru-en | 0.8538 | 0.8151 | 0.8284 | 34% | 25.77 | 27.51 | 49.25 | 51.39 |
| alma-7b-slimgpt40-gen-pair-zh-lora | en-zh, zh-en | 0.8234 | 0.7741 | 0.7970 | 46% | 20.83 | 23.51 | 33.04 | 35.46 |
| alma-7b-slimgpt40-ref-multi-lora | all ten | 0.8474 | 0.7914 | 0.8193 | 50% | 23.13 | 26.13 | 44.56 | 47.40 |

## slimgpt gen 40%: ten-direction composites

Each direction translated by the system named in the row; pair rows use the
matching pair model for each direction.

| System | COMET | BLEU | chrF++ |
| :--- | ---: | ---: | ---: |
| dense ALMA-7B | 0.8474 | 30.35 | 51.14 |
| alma-7b-slimgpt40-gen-multi | 0.7905 | 22.92 | 44.44 |
| alma-7b-slimgpt40-gen-multi-lora | 0.8179 | 26.36 | 47.40 |
| pair models | 0.8092 | 24.18 | 45.95 |
| pair models + own-pair LoRA | 0.8236 | 26.08 | 47.73 |
