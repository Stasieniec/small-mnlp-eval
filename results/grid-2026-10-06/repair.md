# Repair on the directions each model is for

Macro over the repaired model's own directions (all ten for multi; its pair for a
pair model, which is repaired on its pair's data only). Recovered = the share of
the gap between pruned and dense that repair closes, on COMET and on MetricX-24
(lower is better, so its gap is pruned minus dense).

| Repaired system | Directions | Dense COMET | Pruned COMET | Repaired COMET | Recovered | Pruned BLEU | Repaired BLEU | Dense MetricX | Pruned MetricX | Repaired MetricX | Recovered (MetricX) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| alma-7b-flap30-gen-multi-lora | all ten | 0.8474 | 0.7830 | 0.8242 | 64% | 23.97 | 27.09 | - | - | - | - |
| alma-7b-flap30-gen-pair-cs-lora | cs-en, en-cs | 0.8614 | 0.8206 | 0.8382 | 43% | 27.46 | 30.33 | - | - | - | - |
| alma-7b-flap30-gen-pair-de-lora | de-en, en-de | 0.8461 | 0.8117 | 0.8281 | 48% | 25.11 | 27.45 | - | - | - | - |
| alma-7b-flap30-gen-pair-is-lora | en-is, is-en | 0.8523 | 0.7975 | 0.8218 | 44% | 24.16 | 25.06 | - | - | - | - |
| alma-7b-flap30-gen-pair-ru-lora | en-ru, ru-en | 0.8538 | 0.8061 | 0.8278 | 46% | 25.39 | 27.87 | - | - | - | - |
| alma-7b-flap30-gen-pair-zh-lora | en-zh, zh-en | 0.8234 | 0.7648 | 0.8009 | 62% | 19.83 | 24.75 | - | - | - | - |
| alma-7b-flap40-gen-multi-lora | all ten | 0.8474 | 0.7392 | 0.8110 | 66% | 20.12 | 25.85 | - | - | - | - |
| alma-7b-flap40-gen-pair-cs-lora | cs-en, en-cs | 0.8614 | 0.7759 | 0.8270 | 60% | 22.90 | 28.63 | - | - | - | - |
| alma-7b-flap40-gen-pair-de-lora | de-en, en-de | 0.8461 | 0.7846 | 0.8157 | 51% | 21.72 | 25.90 | - | - | - | - |
| alma-7b-flap40-gen-pair-is-lora | en-is, is-en | 0.8523 | 0.7267 | 0.8045 | 62% | 18.42 | 21.83 | - | - | - | - |
| alma-7b-flap40-gen-pair-ru-lora | en-ru, ru-en | 0.8538 | 0.7684 | 0.8174 | 57% | 20.86 | 26.71 | - | - | - | - |
| alma-7b-flap40-gen-pair-zh-lora | en-zh, zh-en | 0.8234 | 0.7158 | 0.7904 | 69% | 15.87 | 23.88 | - | - | - | - |
| alma-7b-flap40-ref-multi-lora | all ten | 0.8474 | 0.7393 | 0.8112 | 67% | 20.50 | 25.73 | - | - | - | - |
| alma-7b-slimgpt20-ref-multi-lora | all ten | 0.8474 | 0.8374 | 0.8381 | 7% | 27.72 | 28.41 | - | - | - | - |
| alma-7b-slimgpt30-gen-multi-lora | all ten | 0.8474 | 0.8229 | 0.8324 | 39% | 25.88 | 27.59 | - | - | - | - |
| alma-7b-slimgpt30-gen-pair-cs-lora | cs-en, en-cs | 0.8614 | 0.8433 | 0.8476 | 24% | 29.13 | 31.09 | - | - | - | - |
| alma-7b-slimgpt30-gen-pair-de-lora | de-en, en-de | 0.8461 | 0.8340 | 0.8368 | 23% | 25.88 | 27.90 | - | - | - | - |
| alma-7b-slimgpt30-gen-pair-is-lora | en-is, is-en | 0.8523 | 0.8316 | 0.8364 | 24% | 26.75 | 26.19 | - | - | - | - |
| alma-7b-slimgpt30-gen-pair-ru-lora | en-ru, ru-en | 0.8538 | 0.8290 | 0.8382 | 37% | 27.06 | 28.94 | - | - | - | - |
| alma-7b-slimgpt30-gen-pair-zh-lora | en-zh, zh-en | 0.8234 | 0.8004 | 0.8090 | 37% | 23.79 | 25.23 | - | - | - | - |
| alma-7b-slimgpt40-gen-multi-lora | all ten | 0.8474 | 0.7905 | 0.8179 | 48% | 22.92 | 26.36 | - | - | - | - |
| alma-7b-slimgpt40-gen-pair-cs-lora | cs-en, en-cs | 0.8614 | 0.8275 | 0.8348 | 21% | 26.87 | 28.96 | - | - | - | - |
| alma-7b-slimgpt40-gen-pair-de-lora | de-en, en-de | 0.8461 | 0.8175 | 0.8307 | 46% | 24.20 | 26.45 | - | - | - | - |
| alma-7b-slimgpt40-gen-pair-is-lora | en-is, is-en | 0.8523 | 0.8120 | 0.8274 | 38% | 23.22 | 23.97 | - | - | - | - |
| alma-7b-slimgpt40-gen-pair-ru-lora | en-ru, ru-en | 0.8538 | 0.8151 | 0.8284 | 34% | 25.77 | 27.51 | - | - | - | - |
| alma-7b-slimgpt40-gen-pair-zh-lora | en-zh, zh-en | 0.8234 | 0.7741 | 0.7970 | 46% | 20.83 | 23.51 | - | - | - | - |
| alma-7b-slimgpt40-ref-multi-lora | all ten | 0.8474 | 0.7914 | 0.8193 | 50% | 23.13 | 26.13 | - | - | - | - |

## flap gen 30%: ten-direction composites

Each direction translated by the system named in the row; pair rows use the
matching pair model for each direction.

| System | COMET | BLEU | chrF++ | MetricX-24 |
| :--- | ---: | ---: | ---: | ---: |
| dense ALMA-7B | 0.8474 | 30.35 | 51.14 | - |
| alma-7b-flap30-gen-multi | 0.7830 | 23.97 | 44.79 | - |
| alma-7b-flap30-gen-multi-lora | 0.8242 | 27.09 | 48.10 | - |
| pair models | 0.8001 | 24.39 | 45.37 | - |
| pair models + own-pair LoRA | 0.8234 | 27.09 | 48.20 | - |

## flap gen 40%: ten-direction composites

Each direction translated by the system named in the row; pair rows use the
matching pair model for each direction.

| System | COMET | BLEU | chrF++ | MetricX-24 |
| :--- | ---: | ---: | ---: | ---: |
| dense ALMA-7B | 0.8474 | 30.35 | 51.14 | - |
| alma-7b-flap40-gen-multi | 0.7392 | 20.12 | 40.79 | - |
| alma-7b-flap40-gen-multi-lora | 0.8110 | 25.85 | 47.05 | - |
| pair models | 0.7543 | 19.95 | 40.86 | - |
| pair models + own-pair LoRA | 0.8110 | 25.39 | 46.74 | - |

## slimgpt gen 30%: ten-direction composites

Each direction translated by the system named in the row; pair rows use the
matching pair model for each direction.

| System | COMET | BLEU | chrF++ | MetricX-24 |
| :--- | ---: | ---: | ---: | ---: |
| dense ALMA-7B | 0.8474 | 30.35 | 51.14 | - |
| alma-7b-slimgpt30-gen-multi | 0.8229 | 25.88 | 47.58 | - |
| alma-7b-slimgpt30-gen-multi-lora | 0.8324 | 27.59 | 48.65 | - |
| pair models | 0.8276 | 26.52 | 48.06 | - |
| pair models + own-pair LoRA | 0.8336 | 27.87 | 49.19 | - |

## slimgpt gen 40%: ten-direction composites

Each direction translated by the system named in the row; pair rows use the
matching pair model for each direction.

| System | COMET | BLEU | chrF++ | MetricX-24 |
| :--- | ---: | ---: | ---: | ---: |
| dense ALMA-7B | 0.8474 | 30.35 | 51.14 | - |
| alma-7b-slimgpt40-gen-multi | 0.7905 | 22.92 | 44.44 | - |
| alma-7b-slimgpt40-gen-multi-lora | 0.8179 | 26.36 | 47.40 | - |
| pair models | 0.8092 | 24.18 | 45.95 | - |
| pair models + own-pair LoRA | 0.8236 | 26.08 | 47.73 | - |
