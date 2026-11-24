"""The array backend of the numeric engine.

The engine calls linear algebra through this module only, so that another array library (JAX, PyTorch) can
take the place of NumPy and SciPy later without touching the engine. NumPy and SciPy are the one backend now.

Author: Alexej Jerschow.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import scipy.integrate
import scipy.interpolate
import scipy.linalg
import scipy.optimize
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


def correlate(values: np.ndarray, vectors: np.ndarray, next_values: np.ndarray, next_vectors: np.ndarray) -> np.ndarray:
    """The order that pairs the eigenstates of one matrix with those of the next along a parameter.

    ``order[k]`` is the index of the next eigenstate that continues state ``k``. The eigenvectors are the
    columns. A pair scores by the overlap of the vectors and by the closeness of the real and of the imaginary
    parts of the eigenvalues, each scaled to the largest value in its row as in SpinDynamica; the assignment
    with the highest total score is taken.
    """

    def scaled(table: np.ndarray) -> np.ndarray:
        return table / (table.max(axis=1, keepdims=True) + 1e-10)

    difference = values[:, None] - next_values[None, :]
    overlap = np.abs(vectors.conj().T @ next_vectors)
    score = 2 - scaled(np.abs(difference.real)) - scaled(np.abs(difference.imag)) + scaled(overlap)
    rows, columns = scipy.optimize.linear_sum_assignment(score, maximize=True)
    order = np.empty(len(values), dtype=int)
    order[rows] = columns
    return order


def adiabatic_transport(hamiltonians: Sequence[np.ndarray]):
    """The eigenstates of a Hamiltonian followed along a time grid: ``(initial, final, energies)``.

    ``initial`` and ``final`` hold the eigenvectors at the first and at the last time as columns, state ``k`` of
    one continuing into state ``k`` of the other, with the phase of every vector chosen so that it has a real
    positive overlap with its predecessor (parallel transport, which carries the geometric phase).
    ``energies[i, k]`` is the eigenvalue of state ``k`` at time ``i``.
    """
    values, vectors = np.linalg.eigh(hamiltonians[0])
    initial, energies = vectors, [values]
    for hamiltonian in hamiltonians[1:]:
        next_values, next_vectors = np.linalg.eigh(hamiltonian)
        order = correlate(values, vectors, next_values, next_vectors)
        next_values, next_vectors = next_values[order], next_vectors[:, order]
        overlap = (vectors.conj() * next_vectors).sum(axis=0)
        vectors = next_vectors * np.exp(-1j * np.angle(overlap))[None, :]
        values = next_values
        energies.append(values)
    return initial, vectors, np.array(energies)


def integrate_samples(samples: np.ndarray, times: Sequence[float]) -> np.ndarray:
    """The integral over time of sampled functions (one per column), through a cubic spline."""
    times = np.asarray(times, dtype=float)
    if len(times) < 4:
        return np.trapezoid(samples, times, axis=0)
    return scipy.interpolate.CubicSpline(times, samples, axis=0).integrate(times[0], times[-1])
