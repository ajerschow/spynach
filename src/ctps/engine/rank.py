"""Thresholds and ranking, in CCCP.m's order: discard after the phase cycle, after the gradients, after diffusion.

Author: Alexej Jerschow.
"""

from __future__ import annotations

import numpy as np


def survivors(
    pc: np.ndarray, grad: np.ndarray, diff: np.ndarray, pulse: np.ndarray, epsilon: float, cutoff: float
) -> np.ndarray:
    """Row indices that pass ``|pc| >= epsilon``, ``|pc| grad >= cutoff`` and ``|pc| grad diff pulse >= cutoff``."""
    a = np.abs(pc)
    keep = a >= epsilon
    keep &= a * grad >= cutoff
    keep &= a * grad * diff * pulse >= cutoff
    return np.flatnonzero(keep)


def ranking(amplitude: np.ndarray, wanted: np.ndarray | None = None) -> np.ndarray:
    """Indices by decreasing amplitude; among equal amplitudes wanted pathways first, then enumeration order."""
    if wanted is None:
        return np.argsort(-amplitude, kind="stable")
    return np.lexsort((~wanted, -amplitude))
