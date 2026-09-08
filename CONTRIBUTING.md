# Contributing

## Setup

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e ".[gen,surface,dev]"
pre-commit install
```

See [docs/environments.md](docs/environments.md) for the COMET and MetricX
environments, which are needed only for neural metrics.

## Before opening a pull request

```bash
.venv/bin/ruff format src tests scripts recipes
.venv/bin/ruff check src tests scripts recipes
.venv/bin/mypy
.venv/bin/python -m pytest tests/
.venv/bin/python scripts/check_style.py
```

CI runs all five. The test suite needs no GPU, no network and no model weights,
so it takes seconds.

## House style

**No emojis and no em-dashes**, anywhere: code, comments, docstrings, docs,
configs, commit messages. `scripts/check_style.py` enforces this and CI runs
it. Replace an em-dash with a comma, a colon, or two sentences.

**No AI agents as authors or co-authors.** Do not add `Co-Authored-By` trailers
naming an AI tool, and do not credit one in a pull request description.

**Comments explain why, not what.** The code says what it does. A comment earns
its place by recording a decision, a constraint, or a trap: a dependency pin
that exists for a reason, a value taken from an upstream script, a failure mode
that is easy to reintroduce.

**Write prose plainly.** No filler, no hedging, no restating the obvious.

## Adding a model

You almost certainly do not need to change the framework. See
[docs/plugging-in-a-model.md](docs/plugging-in-a-model.md).

## Changing the evaluation protocol

Anything that changes what a run means needs care, because it invalidates
comparisons with existing runs:

- editing a prompt template changes its fingerprint, so old and new runs stop
  being comparable, which is correct but must be deliberate
- adding a field to `ModelSpec` or `DecodeSpec` that affects the numbers means
  adding it to run identity, and a test in `tests/test_config.py` asserting so
- changing a default in `DecodeSpec` or `BenchSpec` silently rebases every
  future run against different settings; say so in the pull request

[docs/protocol.md](docs/protocol.md) records the protocol and why each part is
the way it is. Keep it current.

## Tests

Every new module needs tests that run without a GPU, network or model weights.
`tests/stubs.py` holds a stub translator that makes the whole pipeline testable.
Mark anything that genuinely needs hardware or the Hub with `@pytest.mark.gpu`,
`@pytest.mark.network` or `@pytest.mark.slow`; CI excludes those.

Prefer a test that would have caught a real bug. Several tests here exist
because they did: signature objects that were not JSON serialisable, a
language-identification branch that reported a different key set, and a
truncation counter that conflated two unrelated failure modes.
