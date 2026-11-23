"""The array backend of the numeric engine.

The engine calls linear algebra through this module only, so that another array library (JAX, PyTorch) can
take the place of NumPy and SciPy later without touching the engine. NumPy and SciPy are the one backend now.

Author: Alexej Jerschow.
"""

from __future__ import annotations

import numpy as np
import scipy.linalg

xp = np


def unitary(hamiltonian: np.ndarray, time: float) -> np.ndarray:
    """``exp(−i H t)`` of a Hermitian matrix, by diagonalisation."""
    values, vectors = np.linalg.eigh(hamiltonian)
    return (vectors * np.exp(-1j * values * time)) @ vectors.conj().T


def exponential(generator: np.ndarray, time: float) -> np.ndarray:
    """``exp(L t)`` of a general matrix."""
    return scipy.linalg.expm(generator * time)
