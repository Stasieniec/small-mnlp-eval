"""Loaders for models that need their own code.

Deliberately outside the framework package. Compression work churns while the
evaluation framework should not, so nothing in `mnlp_eval` imports anything
here. A model spec reaches this code through a dotted `entrypoint`, and the
framework has no build-time dependency on it.

See docs/plugging-in-a-model.md.
"""
