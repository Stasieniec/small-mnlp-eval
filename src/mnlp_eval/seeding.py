"""Seeding, so that a rerun of the same run identity gives the same output."""

from __future__ import annotations

import os
import random

__all__ = ["seed_everything"]


def seed_everything(seed: int, *, deterministic_algorithms: bool = False) -> None:
    """Seed Python, NumPy and torch.

    Beam search with ``do_sample=False`` is already deterministic given the
    same padding, so seeding mostly guards against a stray sampling path and
    against any data-order randomness a custom loader might introduce.

    ``deterministic_algorithms`` additionally forces deterministic CUDA
    kernels. It is off by default because it is measurably slower and would
    distort the efficiency measurements, which are the other half of what this
    framework reports.
    """
    random.seed(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic_algorithms:
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:  # pragma: no cover
        pass
