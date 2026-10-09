# Repair on the directions each model is for

Macro over the repaired model's own directions (all ten for multi; its pair for a
pair model, which is repaired on its pair's data only). Recovered = the share of
the gap between pruned and dense that repair closes, on COMET and on MetricX-24
(lower is better, so its gap is pruned minus dense).

| Repaired system | Directions | Dense COMET | Pruned COMET | Repaired COMET | Recovered | Pruned BLEU | Repaired BLEU | Dense MetricX | Pruned MetricX | Repaired MetricX | Recovered (MetricX) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| alma-7b-flap30-gen-multi-lora | all ten | 0.8475 | 0.7823 | 0.8239 | 64% | 23.55 | 27.05 | 2.937 | 5.231 | 3.831 | 61% |
| alma-7b-flap30-gen-pair-cs-lora | cs-en, en-cs | 0.8675 | 0.8269 | 0.8462 | 47% | 27.81 | 31.01 | 3.698 | 5.428 | 4.543 | 51% |
| alma-7b-flap30-gen-pair-de-lora | de-en, en-de | 0.8431 | 0.8139 | 0.8293 | 53% | 25.27 | 27.27 | 2.257 | 3.046 | 2.620 | 54% |
| alma-7b-flap30-gen-pair-is-lora | en-is, is-en | 0.8508 | 0.7908 | 0.8216 | 51% | 22.75 | 23.80 | 3.757 | 6.697 | 5.364 | 45% |
| alma-7b-flap30-gen-pair-ru-lora | en-ru, ru-en | 0.8538 | 0.8079 | 0.8312 | 51% | 25.16 | 27.83 | 2.692 | 4.263 | 3.455 | 51% |
| alma-7b-flap30-gen-pair-zh-lora | en-zh, zh-en | 0.8222 | 0.7660 | 0.8021 | 64% | 21.30 | 26.20 | 2.281 | 3.771 | 2.809 | 65% |
| alma-7b-flap40-gen-multi-lora | all ten | 0.8475 | 0.7417 | 0.8113 | 66% | 20.17 | 25.91 | 2.937 | 6.665 | 4.252 | 65% |
| alma-7b-flap40-gen-pair-cs-lora | cs-en, en-cs | 0.8675 | 0.7797 | 0.8312 | 59% | 23.20 | 28.68 | 3.698 | 7.246 | 5.156 | 59% |
| alma-7b-flap40-gen-pair-de-lora | de-en, en-de | 0.8431 | 0.7825 | 0.8176 | 58% | 21.67 | 25.46 | 2.257 | 3.859 | 2.976 | 55% |
| alma-7b-flap40-gen-pair-is-lora | en-is, is-en | 0.8508 | 0.7268 | 0.8018 | 61% | 18.10 | 21.56 | 3.757 | 9.505 | 6.246 | 57% |
| alma-7b-flap40-gen-pair-ru-lora | en-ru, ru-en | 0.8538 | 0.7773 | 0.8181 | 53% | 21.60 | 26.22 | 2.692 | 5.345 | 3.963 | 52% |
| alma-7b-flap40-gen-pair-zh-lora | en-zh, zh-en | 0.8222 | 0.7239 | 0.7904 | 68% | 17.13 | 24.97 | 2.281 | 4.923 | 3.101 | 69% |
| alma-7b-flap40-ref-multi-lora | all ten | 0.8475 | 0.7414 | 0.8111 | 66% | 20.31 | 25.72 | 2.937 | 6.697 | 4.216 | 66% |
| alma-7b-slimgpt20-ref-multi-lora | all ten | 0.8475 | 0.8361 | 0.8391 | 26% | 27.76 | 28.51 | 2.937 | 3.356 | 3.234 | 29% |
| alma-7b-slimgpt30-gen-multi-lora | all ten | 0.8475 | 0.8227 | 0.8312 | 34% | 25.94 | 27.48 | 2.937 | 3.815 | 3.474 | 39% |
| alma-7b-slimgpt30-gen-pair-cs-lora | cs-en, en-cs | 0.8675 | 0.8500 | 0.8566 | 37% | 29.44 | 31.70 | 3.698 | 4.357 | 4.082 | 42% |
| alma-7b-slimgpt30-gen-pair-de-lora | de-en, en-de | 0.8431 | 0.8298 | 0.8353 | 41% | 25.75 | 27.44 | 2.257 | 2.625 | 2.470 | 42% |
| alma-7b-slimgpt30-gen-pair-is-lora | en-is, is-en | 0.8508 | 0.8322 | 0.8365 | 23% | 26.03 | 25.47 | 3.757 | 4.757 | 4.590 | 17% |
| alma-7b-slimgpt30-gen-pair-ru-lora | en-ru, ru-en | 0.8538 | 0.8336 | 0.8395 | 30% | 27.59 | 28.74 | 2.692 | 3.375 | 3.147 | 33% |
| alma-7b-slimgpt30-gen-pair-zh-lora | en-zh, zh-en | 0.8222 | 0.7980 | 0.8087 | 44% | 25.11 | 26.86 | 2.281 | 2.956 | 2.566 | 58% |
| alma-7b-slimgpt40-gen-multi-lora | all ten | 0.8475 | 0.7937 | 0.8196 | 48% | 23.27 | 26.16 | 2.937 | 4.866 | 3.915 | 49% |
| alma-7b-slimgpt40-gen-pair-cs-lora | cs-en, en-cs | 0.8675 | 0.8310 | 0.8447 | 37% | 27.45 | 29.77 | 3.698 | 5.089 | 4.519 | 41% |
| alma-7b-slimgpt40-gen-pair-de-lora | de-en, en-de | 0.8431 | 0.8153 | 0.8287 | 48% | 24.21 | 26.24 | 2.257 | 2.977 | 2.686 | 40% |
| alma-7b-slimgpt40-gen-pair-is-lora | en-is, is-en | 0.8508 | 0.8108 | 0.8252 | 36% | 22.65 | 23.53 | 3.757 | 5.863 | 5.251 | 29% |
| alma-7b-slimgpt40-gen-pair-ru-lora | en-ru, ru-en | 0.8538 | 0.8153 | 0.8306 | 40% | 25.42 | 27.13 | 2.692 | 3.964 | 3.490 | 37% |
| alma-7b-slimgpt40-gen-pair-zh-lora | en-zh, zh-en | 0.8222 | 0.7718 | 0.7948 | 46% | 21.92 | 25.01 | 2.281 | 3.621 | 2.971 | 48% |
| alma-7b-slimgpt40-ref-multi-lora | all ten | 0.8475 | 0.7931 | 0.8195 | 48% | 23.24 | 26.10 | 2.937 | 4.857 | 3.916 | 49% |

## flap gen 30%: ten-direction composites

Each direction translated by the system named in the row; pair rows use the
matching pair model for each direction.

| System | COMET | BLEU | chrF++ | MetricX-24 |
| :--- | ---: | ---: | ---: | ---: |
| dense ALMA-7B | 0.8475 | 30.32 | 51.13 | 2.937 |
| alma-7b-flap30-gen-multi | 0.7823 | 23.55 | 44.56 | 5.231 |
| alma-7b-flap30-gen-multi-lora | 0.8239 | 27.05 | 48.09 | 3.831 |
| pair models | 0.8011 | 24.46 | 45.49 | 4.641 |
| pair models + own-pair LoRA | 0.8261 | 27.22 | 48.33 | 3.758 |

## flap gen 40%: ten-direction composites

Each direction translated by the system named in the row; pair rows use the
matching pair model for each direction.

| System | COMET | BLEU | chrF++ | MetricX-24 |
| :--- | ---: | ---: | ---: | ---: |
| dense ALMA-7B | 0.8475 | 30.32 | 51.13 | 2.937 |
| alma-7b-flap40-gen-multi | 0.7417 | 20.17 | 40.85 | 6.665 |
| alma-7b-flap40-gen-multi-lora | 0.8113 | 25.91 | 47.03 | 4.252 |
| pair models | 0.7580 | 20.34 | 41.31 | 6.175 |
| pair models + own-pair LoRA | 0.8118 | 25.38 | 46.69 | 4.288 |

## slimgpt gen 30%: ten-direction composites

Each direction translated by the system named in the row; pair rows use the
matching pair model for each direction.

| System | COMET | BLEU | chrF++ | MetricX-24 |
| :--- | ---: | ---: | ---: | ---: |
| dense ALMA-7B | 0.8475 | 30.32 | 51.13 | 2.937 |
| alma-7b-slimgpt30-gen-multi | 0.8227 | 25.94 | 47.53 | 3.815 |
| alma-7b-slimgpt30-gen-multi-lora | 0.8312 | 27.48 | 48.54 | 3.474 |
| pair models | 0.8287 | 26.78 | 48.26 | 3.614 |
| pair models + own-pair LoRA | 0.8353 | 28.04 | 49.23 | 3.371 |

## slimgpt gen 40%: ten-direction composites

Each direction translated by the system named in the row; pair rows use the
matching pair model for each direction.

| System | COMET | BLEU | chrF++ | MetricX-24 |
| :--- | ---: | ---: | ---: | ---: |
| dense ALMA-7B | 0.8475 | 30.32 | 51.13 | 2.937 |
| alma-7b-slimgpt40-gen-multi | 0.7937 | 23.27 | 44.79 | 4.866 |
| alma-7b-slimgpt40-gen-multi-lora | 0.8196 | 26.16 | 47.27 | 3.915 |
| pair models | 0.8089 | 24.33 | 45.83 | 4.303 |
| pair models + own-pair LoRA | 0.8248 | 26.34 | 47.78 | 3.783 |
