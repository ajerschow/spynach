"""The array backend of the numeric engine.

The engine calls linear algebra through this module only, so that another array library (JAX, PyTorch) can
take the place of NumPy and SciPy later without touching the engine. NumPy and SciPy are the one backend now.

Author: Alexej Jerschow.
"""

from __future__ import annotations

import numpy as np
import scipy.integrate
import scipy.linalg
import scipy.sparse
import scipy.sparse.csgraph

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


def block_eigensystems(hamiltonian) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """A sparse Hermitian matrix split into the blocks it does not connect: ``(indices, eigenvalues, eigenvectors)``
    of every block. A conserved quantity that is diagonal in the basis (the total ``Iz``) shows as such blocks."""
    hamiltonian = scipy.sparse.csr_matrix(hamiltonian)
    count, labels = scipy.sparse.csgraph.connected_components(hamiltonian, directed=False)
    blocks = []
    for label in range(count):
        indices = np.flatnonzero(labels == label)
        values, vectors = np.linalg.eigh(hamiltonian[indices][:, indices].toarray())
        blocks.append((indices, values, vectors))
    return blocks


def evolve_in_blocks(rho: np.ndarray, blocks, time: float) -> np.ndarray:
    """``exp(−iHt) ρ exp(+iHt)`` from the block eigensystems of ``H``."""
    out = np.empty_like(rho)
    phased = [(indices, vectors, np.exp(-1j * values * time)) for indices, values, vectors in blocks]
    for rows, left, left_phase in phased:
        partial = left.conj().T @ rho[rows]  # the rows of this block, all columns
        for columns, right, right_phase in phased:
            inner = (partial[:, columns] @ right) * left_phase[:, None] * right_phase.conj()[None, :]
            out[np.ix_(rows, columns)] = left @ inner @ right.conj().T
    return out
