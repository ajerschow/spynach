"""The array backend of the numeric engine.

The engine calls linear algebra through this module only, so that another array library (JAX, PyTorch) can
take the place of NumPy and SciPy later without touching the engine. NumPy and SciPy are the one backend now.

Author: Alexej Jerschow.
"""

from __future__ import annotations

import numpy as np
import scipy.integrate
import scipy.linalg

xp = np


def unitary(hamiltonian: np.ndarray, time: float) -> np.ndarray:
    """``exp(−i H t)`` of a Hermitian matrix, by diagonalisation."""
    values, vectors = np.linalg.eigh(hamiltonian)
    return (vectors * np.exp(-1j * values * time)) @ vectors.conj().T


def exponential(generator: np.ndarray, time: float) -> np.ndarray:
    """``exp(L t)`` of a general matrix."""
    return scipy.linalg.expm(generator * time)


def integrate(generator, size: int, start: float, end: float, rtol: float = 1e-11, atol: float = 1e-13) -> np.ndarray:
    """The propagator ``P`` of ``dP/dt = G(t) P`` from ``start`` to ``end``, for a function ``G`` returning a matrix."""

    def rhs(t: float, flat: np.ndarray) -> np.ndarray:
        return (generator(t) @ flat.reshape(size, size)).reshape(-1)

    initial = np.eye(size, dtype=complex).reshape(-1)
    solution = scipy.integrate.solve_ivp(rhs, (start, end), initial, method="DOP853", rtol=rtol, atol=atol)
    if not solution.success:
        raise RuntimeError(f"the integration of the equation of motion failed: {solution.message}")
    return solution.y[:, -1].reshape(size, size)


def unitary_eigensystem(u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The eigenvalues and orthonormal eigenvectors of a unitary matrix (also when eigenvalues coincide)."""
    triangular, vectors = scipy.linalg.schur(u, output="complex")
    return np.diag(triangular), vectors
