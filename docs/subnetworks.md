# Subnetwork descriptors

A pruned checkpoint records what survived but not which positions it came
from. Two 50 percent subnetworks have the same shape whether they kept the
same attention heads or disjoint ones, so if the selection is not written down
at pruning time it cannot be recovered afterwards, and RQ3's overlap question
cannot be answered at all.

This file is that record. It is plain JSON with integer indices, so a pruning
script writes it in a few lines and the analysis never has to import that
script.

## Format

```json
{
  "schema_version": 1,
  "name": "prune50-de",
  "base_model": "haoranxu/ALMA-7B",
  "pruned_for": "de-en,en-de",
  "method": "structured, magnitude times activation norm",
  "notes": "calibrated on configs/calibration/pair-de.yaml",
  "components": {
    "attention_heads": {
      "total": 32,
      "kept": {"0": [0, 2, 5, 9], "1": [1, 2, 3, 8]}
    },
    "ffn_channels": {
      "total": 11008,
      "kept": {"0": [3, 17, 22], "1": [1, 4, 88]}
    },
    "layers": {
      "total": 32,
      "kept": [0, 1, 2, 3, 4, 5]
    }
  }
}
```

- `total` is the number of positions **per layer** for a per-layer component,
  and the total count for one that is not.
- `kept` is either an object keyed by layer index, or a flat list for a
  component that is not per layer, such as whole layers.
- Indices are into the original dense model, not into the pruned one. This is
  the part that is easy to get wrong and impossible to detect later: if two
  descriptors index into their own pruned models, their overlap is
  meaningless.
- Component names are free-form, but two descriptors are only compared on the
  components they share and on components with the same `total`, so use the
  same names across a sweep.

The loader rejects out-of-range indices, repeated indices, an empty selection,
and non-integer indices. A layer that kept nothing is a removed layer and
belongs in a `layers` component rather than as an empty entry.

## Emitting one

From a pruning script that has decided on a mask, where `head_mask[layer]` is
a boolean tensor over the dense model's heads:

```python
import json

descriptor = {
    "schema_version": 1,
    "name": "prune50-de",
    "base_model": "haoranxu/ALMA-7B",
    "pruned_for": "de-en,en-de",
    "method": "structured, magnitude times activation norm",
    "components": {
        "attention_heads": {
            "total": int(head_mask[0].numel()),
            "kept": {
                str(layer): mask.nonzero().flatten().tolist()
                for layer, mask in enumerate(head_mask)
            },
        },
        "ffn_channels": {
            "total": int(ffn_mask[0].numel()),
            "kept": {
                str(layer): mask.nonzero().flatten().tolist() for layer, mask in enumerate(ffn_mask)
            },
        },
    },
}
with open("subnetworks/prune50-de.json", "w", encoding="utf-8") as handle:
    json.dump(descriptor, handle)
```

Commit it. It is a few hundred kilobytes, it is an input to the analysis
rather than an output of it, and the overlap numbers in the report cannot be
reproduced without it.

Point the model config at it so the two stay associated:

```yaml
compression:
  subnetwork: subnetworks/prune50-de.json
```

## Comparing them

```bash
mnlp-eval overlap --subnetwork-dir subnetworks/ --out reports/
```

Writes `reports/overlap.md` and `reports/overlap.json`, comparing every pair
on every shared component.

## Reading the result

**Quote the excess over chance, not the Jaccard index.** Two independently
chosen subnetworks that each keep half of every layer already share about half
of what each keeps, which is a Jaccard index of about 0.33. Reporting that as
evidence of shared structure would be the central mistake this analysis exists
to prevent, and it is an easy one to make, because 0.33 looks like a lot.

The chance figure is computed per layer: under independent uniform selection of
`|A|` and `|B|` positions out of `n`, the expected intersection is `|A||B|/n`.
The Jaccard of an expected intersection is not exactly the expectation of the
Jaccard, but at these layer widths the difference is far below the resolution
anyone would quote.

Excess near zero means the two pruning runs agreed no more than chance.
Excess well above zero is the RQ3 finding: the pairs share structure. Excess
below zero means they actively avoided each other's units, which would be a
stranger and more interesting result still.

**Use the overlap coefficient when sparsities differ.** It divides by the
smaller selection rather than by the union, so it asks whether the sparser
subnetwork sits inside the denser one. That is the right question when
comparing an 70 percent subnetwork with a 50 percent one, where the Jaccard
index is capped well below 1 no matter how nested they are.

**Look at components separately.** "Attention heads agree, FFN channels do
not" is a finding, and an average over the two would hide it. The report keeps
one table per component for that reason.
