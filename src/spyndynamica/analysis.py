"""Analysis: bounds on transformation amplitudes and eigensystems followed along a parameter.

Author: Alexej Jerschow.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import sympy as sp

from spyndynamica import backend
from spyndynamica.liouville import Superoperator
from spyndynamica.operators import Operator


def amplitude_bounds(rho: Operator, targets: Any, *, normalization: Any = None, values: Any = None) -> np.ndarray:
    """The smallest and the largest amplitude that any unitary transformation of ``rho`` can have along a
    target operator: ``(min, max)``, or one such row per target (SpinDynamica's ``TransformationAmplitudeBounds``).

    The amplitude is ``(target|U ρ U†)/(target|target)`` as in :func:`~spyndynamica.amplitudes`. With the
    eigenvalues of both operators in ascending order the largest amplitude is ``Σ λρ λQ / Σ λQ²``, reached by
    the unitary that maps the eigenstates of one onto those of the other in that order; the smallest pairs
    them in opposite order (Sørensen's bound). Both operators must be Hermitian. The bounds are divided by
    ``normalization`` when one is given.
    """
    single = isinstance(targets, Operator)

    def spectrum(operator: Operator) -> np.ndarray:
        matrix = operator.numeric(values)
        if not np.allclose(matrix, matrix.conj().T, atol=1e-12 * max(1.0, np.abs(matrix).max())):
            raise ValueError("the bounds hold for Hermitian operators only.")
        return np.linalg.eigvalsh(matrix)

    norm = 1.0 if normalization is None else float(sp.N(sp.sympify(normalization)))
    if norm == 0:
        raise ValueError("the normalization may not be zero.")
    state = spectrum(rho)
    rows = []
    for target in [targets] if single else targets:
        observed = spectrum(target)
        scale = norm * (observed @ observed)
        rows.append([state @ observed[::-1] / scale, state @ observed / scale])
    out = np.array(rows)
    return out[0] if single else out


@dataclass(frozen=True)
class CorrelatedEigensystems:
    """Eigenvalues and eigenvectors followed along a parameter.

    ``values`` are the parameter values, ``eigenvalues[i, k]`` the eigenvalue of state ``k`` at value ``i`` and
    ``eigenvectors[i, k]`` its eigenvector: state ``k`` at one value continues into state ``k`` at the next.
    """

    values: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray

    def tracks(self) -> np.ndarray:
        """One row of ``(value, eigenvalue)`` pairs per state, the form in which SpinDynamica lists them."""
        pairs = np.broadcast_to(self.values[:, None], self.eigenvalues.shape)
        return np.stack([pairs.T, self.eigenvalues.T], axis=-1)


def _matrices(generator: Any, parameter: Any, values: Sequence[Any] | None, background: Any, basis: Any):
    if isinstance(generator, (Operator, Superoperator)):
        if parameter is None or values is None:
            raise ValueError("give the parameter and its values.")
        if background is not None:
            generator = generator + background
        return [generator.numeric({parameter: value}, basis) for value in values]
    if background is not None or basis is not None:
        raise ValueError("background and basis go with an operator or a superoperator.")
    if parameter is not None:
        function = sp.lambdify(parameter, sp.Matrix(generator), "numpy")
        return [np.array(function(float(sp.N(value))), dtype=complex) for value in values]
    return [np.array(matrix, dtype=complex) for matrix in generator]


def correlated_eigensystems(
    generator: Any,
    parameter: Any = None,
    values: Sequence[Any] | None = None,
    *,
    background: Any = None,
    basis: Any = None,
) -> CorrelatedEigensystems:
    """The eigenvalues and eigenvectors of a generator that depends on a parameter, with every eigenstate
    followed from one parameter value to the next (SpinDynamica's ``CorrelatedEigensystems``).

    ``generator`` is an operator or a superoperator whose coefficients contain the symbol ``parameter``, a
    matrix of expressions in it, or, without a parameter, a list of numeric matrices. ``values`` are the
    values of the parameter. ``background`` is added to an operator or superoperator, and ``basis`` is the
    state basis (operator basis for a superoperator) of the matrices and of the eigenvectors.

    At the first value the states are in ascending order of the eigenvalues (for matrices that are not
    Hermitian: of their real parts, then of their imaginary parts). From there on a state continues into the
    state of the next value that it overlaps with most and whose eigenvalue is closest, so a line stays with
    its state through a crossing, and at an avoided crossing it follows the eigenvalue. Overlap and closeness
    count the same, as in SpinDynamica: two values placed symmetrically about an exact crossing leave the
    choice open, and a finer or shifted grid settles it. The phase of an eigenvector is arbitrary.
    """
    matrices = _matrices(generator, parameter, values, background, basis)
    if not matrices:
        raise ValueError("there are no matrices to diagonalise.")
    hermitian = all(np.allclose(m, m.conj().T, atol=1e-12 * max(1.0, np.abs(m).max())) for m in matrices)

    def eigensystem(matrix: np.ndarray):
        if hermitian:
            return np.linalg.eigh(matrix)
        found, vectors = np.linalg.eig(matrix)
        order = np.lexsort((found.imag.round(9), found.real.round(9)))
        return found[order], vectors[:, order]

    current, vectors = eigensystem(matrices[0])
    eigenvalues, eigenvectors = [current], [vectors.T]
    for matrix in matrices[1:]:
        found, new = eigensystem(matrix)
        order = backend.correlate(current, vectors, found, new)
        current, vectors = found[order], new[:, order]
        eigenvalues.append(current)
        eigenvectors.append(vectors.T)
    points = np.arange(len(matrices)) if values is None else np.array([complex(sp.N(v)) for v in values]).real
    return CorrelatedEigensystems(np.asarray(points, dtype=float), np.array(eigenvalues), np.array(eigenvectors))
