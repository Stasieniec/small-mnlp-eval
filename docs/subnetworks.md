# Subnetwork descriptors

A pruned checkpoint records what survived but not which positions it came
from, so unless the selection is written down at pruning time the overlap
question cannot be answered afterwards. This file is that record.

## Format

```json
{
  "schema_version": 1,
  "name": "prune50-de",
  "base_model": "haoranxu/ALMA-7B",
  "pruned_for": "de-en,en-de",
  "method": "structured, magnitude times activation norm",
  "components": {
    "attention_heads": {"total": 32, "kept": {"0": [0, 2, 5, 9], "1": [1, 2, 3, 8]}},
    "ffn_channels": {"total": 11008, "kept": {"0": [3, 17, 22], "1": [1, 4, 88]}},
    "layers": {"total": 32, "kept": [0, 1, 2, 3, 4, 5]}
  }
}
```

- `total` is the count per layer, or the total count for a component that is
  not per layer.
- `kept` is an object keyed by layer index, or a flat list for a component that
  is not per layer.
- **Indices are into the original dense model**, not into the pruned one. Two
  descriptors that each index into their own pruned model produce a
  meaningless overlap that nothing downstream can detect.
- Use the same component names across a sweep. Descriptors are only compared
  on components they share.

Rejected on load: out-of-range indices, repeated indices, non-integer indices,
and an empty selection. A layer that kept nothing is a removed layer and
belongs in a `layers` component.

## Emitting one

```python
descriptor = {
    "schema_version": 1,
    "name": "prune50-de",
    "base_model": "haoranxu/ALMA-7B",
    "pruned_for": "de-en,en-de",
    "components": {
        "attention_heads": {
            "total": int(head_mask[0].numel()),
            "kept": {
                str(layer): mask.nonzero().flatten().tolist()
                for layer, mask in enumerate(head_mask)
            },
        },
    },
}
```

Commit it, and point the model config at it with `compression.subnetwork`. The
overlap numbers cannot be reproduced without it.

## Comparing

```bash
mnlp-eval overlap --subnetwork-dir subnetworks/ --out reports/
```

**Report the excess over chance, not the Jaccard index.** Two independently
chosen subnetworks that each keep half of every layer share about half of what
each keeps, a Jaccard index of about 0.33. Under independent uniform selection
of `|A|` and `|B|` positions out of `n`, the expected intersection is
`|A||B|/n`, computed per layer. The Jaccard of an expected intersection is not
exactly the expectation of the Jaccard, but the difference is far below the
resolution worth quoting at these widths.

Excess near zero means the two selections agreed no more than chance. Above
zero means they share structure; below zero means they avoided each other.

The overlap coefficient divides by the smaller selection instead, which is the
right measure when the two subnetworks have different sparsities: it asks
whether the sparser one sits inside the denser one.

Components are compared separately, since attention heads and FFN channels can
disagree and an average over the two would hide it.
