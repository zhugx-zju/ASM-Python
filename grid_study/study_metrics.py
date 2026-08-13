"""Numerical metrics used by forward grid-study workflows."""

from __future__ import annotations

import numpy as np


def relative_linf(a: np.ndarray, b: np.ndarray) -> float:
    """Return the infinity-norm error relative to ``b``."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    denominator = float(np.max(np.abs(b)))
    return float(np.max(np.abs(a - b)) / (denominator + 1e-15))


__all__ = ["relative_linf"]
