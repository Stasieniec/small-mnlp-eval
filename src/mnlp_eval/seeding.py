"""Seeding helpers."""

from __future__ import annotations

import os
import random

__all__ = ["seed_everything"]


def seed_everything(seed: int, *, deterministic_algorithms: bool = False) -> None:
    """Seed Python, NumPy and torch.

    ``deterministic_algorithms`` also forces deterministic CUDA kernels. Off by
    default since it slows down generation and would affect the timings.
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
