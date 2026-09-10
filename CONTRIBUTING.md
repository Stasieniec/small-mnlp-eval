# Contributing

## Setup

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e ".[gen,surface,dev]"
pre-commit install
```

The COMET and MetricX environments are needed only for neural metrics. See
[docs/environments.md](docs/environments.md).

## Before opening a pull request

```bash
.venv/bin/ruff format src tests scripts recipes
.venv/bin/ruff check src tests scripts recipes
.venv/bin/mypy
.venv/bin/python -m pytest tests/
.venv/bin/python scripts/check_style.py
```

CI runs all five. The test suite needs no GPU, network or model weights.

## Comments

A comment records a decision, a constraint or a trap: a pin that exists for a
reason, a value taken from an upstream script, a failure mode that is easy to
reintroduce. It does not restate the code.

## Changing the evaluation protocol

Anything that changes what a run means invalidates comparisons with existing
runs:

- editing a prompt template changes its fingerprint, so old and new runs stop
  being comparable, which is correct but must be deliberate;
- a new `ModelSpec` or `DecodeSpec` field that affects the numbers belongs in
  run identity, with a test asserting so;
- changing a `DecodeSpec` or `BenchSpec` default rebases every future run
  against different settings. Say so in the pull request.

Keep [docs/protocol.md](docs/protocol.md) current with the code.

## Tests

Every module needs tests that run without a GPU, network or model weights.
`tests/stubs.py` holds a stub translator that makes the whole pipeline
testable. Mark anything needing hardware or the Hub with `@pytest.mark.gpu`,
`@pytest.mark.network` or `@pytest.mark.slow`; CI excludes those.
`@pytest.mark.torch` means torch but no GPU and no download, and a second CI
job installs a CPU wheel and runs it.

When you add a metric, write the test that catches its most flattering possible
failure. When you add a stage, ask what two concurrent Slurm array tasks do to
it.
