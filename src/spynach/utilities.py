"""Small numerical helpers shared across the kernel (ports of ``clean_up.m``, ``hdot.m``).

Everything here is storage-agnostic: a ``Matrix`` is a dense ``np.ndarray`` or a
sparse ``csr_array``, and :func:`chop`, :func:`as_matrix` and :func:`nnz` go through
:mod:`spynach.backend` so that they work on device arrays as well.

Spinach authors, by original file (headers of Spinach 2.13; see docs/spinach-credits.md):
  clean_up.m: Ilya Kuprov, Alex Nevzorov
  hdot.m: Ilya Kuprov
  propagator.m: Ilya Kuprov, Luke Edwards
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sps

from spynach.backend import get_backend
from spynach.kernels import matlab_round  # noqa: F401  (kept here: the name Spinach users know)
from spynach.tolerances import Tolerances

Matrix = np.ndarray | sps.csr_array


def nnz(matrix: Matrix) -> int:
    return int(matrix.count_nonzero()) if sps.issparse(matrix) else int(np.count_nonzero(matrix))
    # GPU: count_nonzero exists for cupyx sparse matrices and cupy arrays; route through the backend then


def clean_up(tols: Tolerances, matrix: Matrix, nonzero_tol: float) -> Matrix:
    """Round elements to a grid of ``nonzero_tol`` and choose sparse or dense storage.

    Rounding removes elements smaller than half the tolerance. Matrices with
    any dimension below ``tols.small_matrix``, or with density above
    ``tols.dense_matrix``, are returned dense (``np.ndarray``); large matrices
    below that density are returned sparse (``csr_array``).

    Spinach authors (clean_up.m): Ilya Kuprov, Alex Nevzorov.
    """
    if nonzero_tol == 0 or np.isnan(nonzero_tol):
        return matrix
    if nonzero_tol < 0:
        raise ValueError("nonzero_tol parameter must be a positive real number.")
    # GPU: this is called inside the propagator Taylor loop on device arrays; the rounding
    # works through the array module and the sparse/dense switch through the backend's csr()
    be = get_backend()
    scale = 1 / nonzero_tol
    if sps.issparse(matrix):
        matrix = sps.csr_array(matrix, copy=True)
        matrix.data = be.round_array(matrix.data, scale, nonzero_tol)
        matrix.eliminate_zeros()
    else:
        matrix = be.round_array(np.asarray(matrix), scale, nonzero_tol)
    return _choose_storage(tols, matrix)


def _choose_storage(tols: Tolerances, matrix: Matrix) -> Matrix:
    """The sparse-or-dense decision of ``clean_up``, applied to an already rounded matrix."""
    density = nnz(matrix) / (matrix.shape[0] * matrix.shape[1])
    if sps.issparse(matrix):
        if (nnz(matrix) > 0 and min(matrix.shape) < tols.small_matrix) or density > tols.dense_matrix:
            return matrix.toarray()
        return matrix
    if density < tols.dense_matrix and min(matrix.shape) > tols.small_matrix:
        return sps.csr_array(matrix)
    return matrix


def matmul_clean(tols: Tolerances, A: Matrix, B: Matrix, nonzero_tol: float) -> Matrix:
    """``clean_up(tols, A @ B, nonzero_tol)``; two sparse factors are multiplied and rounded in one threaded
    pass where the compiled kernel applies (:mod:`spynach.kernels`), with the same result.

    Spinach authors (clean_up.m, fused with the product of the propagator's squarings): Ilya Kuprov, Alex
      Nevzorov.
    Spinach authors (propagator.m, the squarings): Ilya Kuprov, Luke Edwards.
    """
    if sps.issparse(A) and sps.issparse(B) and nonzero_tol > 0 and not np.isnan(nonzero_tol):
        product = get_backend().spgemm_round(A, B, 1 / nonzero_tol, nonzero_tol)
        if product is not None:
            return _choose_storage(tols, product)
    return clean_up(tols, A @ B, nonzero_tol)


def matmul(A: Matrix, X: np.ndarray) -> Matrix:
    """``A @ X``; a CSR matrix times a dense stack goes through the threaded kernel where it applies."""
    if sps.issparse(A) and not sps.issparse(X) and A.format == "csr":
        product = get_backend().csr_times_dense(A, np.asarray(X))
        if product is not None:
            return product
    return A @ X


def chop(x: Matrix, tol: float) -> Matrix:
    """Round the elements of a vector, stack, or matrix to a grid of ``tol``.

    :func:`clean_up` without the storage decision: sparse stays sparse (zeros
    dropped), dense stays dense. Used on state vectors and reduced generators
    in :mod:`spynach.propagation`.

    Spinach authors (clean_up.m, the rounding of clean_up.m without the storage decision): Ilya Kuprov, Alex
      Nevzorov.
    """
    be = get_backend()
    if be.issparse(x):
        x = be.csr(x, copy=True)
        x.data = be.round_array(x.data, tol, tol, divide=True)
        x.eliminate_zeros()
        return x
    return be.round_array(be.dense(x), tol, tol, divide=True)


def as_matrix(x: Matrix) -> Matrix:
    """Normalise the result of mixed dense/sparse arithmetic to ``ndarray`` or ``csr_array``."""
    be = get_backend()
    return be.csr(x) if be.issparse(x) else be.dense(x)


def hdot(a: Matrix, b: Matrix) -> complex:
    """Hilbert-Schmidt inner product ``sum(conj(a) * b)``.

    Spinach authors (hdot.m): Ilya Kuprov.
    """
    if a.shape != b.shape:
        raise ValueError("the two inputs must have identical dimensions.")
    if sps.issparse(a):
        return complex(a.conj().multiply(b).sum())
    if sps.issparse(b):
        return complex(b.multiply(np.conj(a)).sum())
    return complex(np.sum(np.conj(a) * b))


def dagger(matrix: Matrix) -> Matrix:
    return matrix.conj().T
