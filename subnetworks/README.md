# Subnetwork descriptors

One JSON file per pruning run, listing the units it kept. Written at pruning
time, because the selection cannot be recovered from the saved checkpoint
afterwards.

Commit them. They are inputs to the overlap analysis rather than outputs of
it, and the numbers in the report cannot be reproduced without them.

```bash
mnlp-eval overlap --subnetwork-dir subnetworks/ --out reports/
```

Format, emitting snippet, and how to read the result: [../docs/subnetworks.md](../docs/subnetworks.md).
