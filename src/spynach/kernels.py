"""Compiled kernels for the sparse linear algebra of the propagators (Numba), with NumPy/SciPy fallbacks.

Spinach's time goes into sparse matrix products: the scaled-and-squared Taylor exponential multiplies
sparse matrices dozens of times and rounds each product to a grid (``clean_up``), and the Krylov-type
stepping applies sparse matrices to dense state stacks. SciPy does both single-threaded, in separate
passes (product, copy, rounding, removal of zeros). The kernels here do the same arithmetic in the same
order, threaded over the rows of the left factor, with the rounding fused into the product so that
no intermediate matrix is built:

- :func:`spgemm_round`: ``A @ B`` with every element rounded to a grid and the zeros dropped, as
  ``clean_up`` does after ``A @ B`` (the product of the squaring loop of ``propagator`` and of ``expmint``);
- :func:`round_array`: the rounding alone, for dense arrays and the ``data`` of sparse ones;
- :func:`csr_times_dense`: ``A @ X`` for a CSR matrix and a dense stack, threaded;
- :func:`taylor_term`: one term of the reordered Taylor series of ``step``, fused with the running sum and
  the convergence test.

Numba is optional (the ``fast`` extra). Without it, and for small problems where thread start-up costs more
than it saves, every function runs the plain NumPy/SciPy code that the kernels reproduce; ``HAVE_NUMBA`` tells
which is in use, and ``SPYNACH_NUMBA=0`` switches the compiled path off. The rounding is Spinach's (MATLAB's)
round half away from zero, applied separately to the real and imaginary parts.

Each compiled row accumulates its products in the order of SciPy's ``csr_matmat`` (the entries of the row of
``A`` in storage order, then the entries of the row of ``B``), so results agree with SciPy to the last bit
except where the compiler fuses a multiply-add; a rounded element can then differ by one grid step in the
rare case that the sum sits within an ulp of a rounding boundary.

Spinach authors, by original file (headers of Spinach 2.13; see docs/spinach-credits.md):
  clean_up.m: Ilya Kuprov, Alex Nevzorov
  propagator.m: Ilya Kuprov, Luke Edwards
  step.m: Ilya Kuprov, Luke Edwards, Anupama Acharya, one contributor listed only by email address
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import scipy.sparse as sps

try:
    if os.environ.get("SPYNACH_NUMBA", "1") == "0":
        raise ImportError("compiled kernels switched off by SPYNACH_NUMBA=0")
    import numba
    from numba import njit, prange

    HAVE_NUMBA = True
except ImportError:  # pragma: no cover - exercised by the fallback tests through monkeypatching
    HAVE_NUMBA = False

    def njit(*args, **kwargs):  # noqa: D103 - a no-op decorator, so the module imports without Numba
        def wrap(f):
            return f

        return wrap(args[0]) if args and callable(args[0]) else wrap

    prange = range

#: Problem sizes (stored elements of the sparse factor, or array elements) below which the NumPy/SciPy path
#: is used even when Numba is present: thread start-up and the first-call compilation cost more than they save.
MIN_NNZ = 200_000
MIN_ELEMENTS = 500_000


# -- the reference implementations (also the fallbacks) ------------------------------------------------------------


def matlab_round(values: np.ndarray) -> np.ndarray:
    """Round half away from zero, separately on real and imaginary parts, as MATLAB ``round`` does."""
    if np.iscomplexobj(values):
        return matlab_round(values.real) + 1j * matlab_round(values.imag)
    return np.sign(values) * np.floor(np.abs(values) + 0.5)


def _round_numpy(values: np.ndarray, scale: float, tol: float, divide: bool) -> np.ndarray:
    scaled = values / scale if divide else scale * values
    return tol * matlab_round(scaled)


# -- compiled kernels ----------------------------------------------------------------------------------------------


@njit(cache=True, inline="always")
def _rnd(x):  # pragma: no cover - compiled
    """sign(x) * floor(|x| + 0.5), the same IEEE operations as ``matlab_round`` (signed zeros included)."""
    if x > 0.0:
        return np.floor(x + 0.5)
    if x < 0.0:
        return -np.floor(-x + 0.5)
    return x * 0.0


@njit(cache=True, parallel=True)
def _round_kernel(flat, scale, tol, divide):  # pragma: no cover - compiled
    """In place: ``flat[i] = tol * round(flat[i] * scale)`` (or ``/ scale``), complex128 data."""
    for i in prange(flat.size):
        z = flat[i]
        if divide:
            re, im = z.real / scale, z.imag / scale
        else:
            re, im = scale * z.real, scale * z.imag
        flat[i] = complex(tol * _rnd(re), tol * _rnd(im))


@njit(cache=True, parallel=True)
def _round_kernel_real(flat, scale, tol, divide):  # pragma: no cover - compiled
    for i in prange(flat.size):
        x = flat[i] / scale if divide else scale * flat[i]
        flat[i] = tol * _rnd(x)


@njit(cache=True)
def _row_blocks(indptr_a, indices_a, indptr_b, nblocks):  # pragma: no cover - compiled
    """Row ranges of roughly equal work (the number of scalar products), so threads finish together."""
    n = indptr_a.size - 1
    work = np.zeros(n + 1, dtype=np.int64)
    for i in range(n):
        w = 0
        for jj in range(indptr_a[i], indptr_a[i + 1]):
            j = indices_a[jj]
            w += indptr_b[j + 1] - indptr_b[j]
        work[i + 1] = work[i] + w + 1
    bounds = np.empty(nblocks + 1, dtype=np.int64)
    bounds[0] = 0
    total = work[n]
    row = 0
    for b in range(1, nblocks):
        target = total * b // nblocks
        while row < n and work[row] < target:
            row += 1
        bounds[b] = row
    bounds[nblocks] = n
    return bounds


@njit(cache=True, nogil=True)
def _spgemm_block(
    indptr_a, indices_a, data_a, indptr_b, indices_b, data_b, ncols, scale, tol, divide, first, last
):  # pragma: no cover - compiled
    """Gustavson product of rows ``first:last`` of A with B, rounded to the grid, zeros dropped.

    Returns ``(counts, cols, vals)``: the kept elements per row, and their column indices (sorted within each
    row) and values, row after row. Releases the GIL, so blocks run on separate threads.
    """
    acc = np.zeros(ncols, dtype=np.complex128)
    mark = np.full(ncols, -1, dtype=np.int64)
    touched = np.empty(ncols, dtype=np.int32)
    counts = np.zeros(last - first, dtype=np.int64)
    cap = 1 << 16
    cols = np.empty(cap, dtype=np.int32)
    vals = np.empty(cap, dtype=np.complex128)
    fill = 0
    for i in range(first, last):
        ntouch = 0
        for jj in range(indptr_a[i], indptr_a[i + 1]):
            j = indices_a[jj]
            v = data_a[jj]
            for kk in range(indptr_b[j], indptr_b[j + 1]):
                k = indices_b[kk]
                if mark[k] != i:
                    mark[k] = i
                    touched[ntouch] = k
                    ntouch += 1
                    acc[k] = v * data_b[kk]
                else:
                    acc[k] += v * data_b[kk]
        row = np.sort(touched[:ntouch])
        if fill + ntouch > cap:
            while fill + ntouch > cap:
                cap *= 2
            cols2 = np.empty(cap, dtype=np.int32)
            vals2 = np.empty(cap, dtype=np.complex128)
            cols2[:fill] = cols[:fill]
            vals2[:fill] = vals[:fill]
            cols, vals = cols2, vals2
        kept = 0
        for t in range(ntouch):
            k = row[t]
            z = acc[k]
            if divide:
                re, im = z.real / scale, z.imag / scale
            else:
                re, im = scale * z.real, scale * z.imag
            z = complex(tol * _rnd(re), tol * _rnd(im))
            if z.real != 0.0 or z.imag != 0.0:
                cols[fill + kept] = k
                vals[fill + kept] = z
                kept += 1
        fill += kept
        counts[i - first] = kept
    return counts, cols[:fill], vals[:fill]


@njit(cache=True, parallel=True)
def _csr_times_dense_kernel(indptr, indices, data, x, out):  # pragma: no cover - compiled
    k = x.shape[1]
    for i in prange(indptr.size - 1):
        for c in range(k):
            out[i, c] = 0.0
        for jj in range(indptr[i], indptr[i + 1]):
            j = indices[jj]
            v = data[jj]
            for c in range(k):
                out[i, c] += v * x[j, c]


@njit(cache=True, parallel=True)
def _taylor_term_kernel(indptr, indices, data, term, coef, rho, out, eps):  # pragma: no cover - compiled
    """``out = coef * (A @ term)``, ``rho += out``, and whether any ``|out| > eps``, in one pass."""
    k = term.shape[1]
    n = indptr.size - 1
    flags = np.zeros(n, dtype=np.int8)
    for i in prange(n):
        big = False
        for c in range(k):
            s = 0.0j
            for jj in range(indptr[i], indptr[i + 1]):
                s += data[jj] * term[indices[jj], c]
            s = coef * s
            out[i, c] = s
            rho[i, c] = rho[i, c] + s
            if abs(s) > eps:
                big = True
        if big:
            flags[i] = 1
    return flags.any()


# -- public functions ----------------------------------------------------------------------------------------------


def _use_compiled(size: int, minimum: int) -> bool:
    return HAVE_NUMBA and size >= minimum


def round_array(values: np.ndarray, scale: float, tol: float, *, divide: bool = False) -> np.ndarray:
    """``tol * matlab_round(scale * values)`` (``values / scale`` with ``divide``) as a new array.

    ``clean_up`` rounds with the multiplication (``scale = 1 / tol``), ``chop`` with the division; the two
    differ in the last bit and are kept apart because the results are rounded to the grid.

    Spinach authors (clean_up.m, the rounding, as a compiled kernel): Ilya Kuprov, Alex Nevzorov.
    """
    values = np.asarray(values)
    if not (_use_compiled(values.size, MIN_ELEMENTS) and values.dtype in (np.complex128, np.float64)):
        return _round_numpy(values, scale, tol, divide)
    out = np.array(values, copy=True, order="C")
    flat = out.reshape(-1)
    (_round_kernel if np.iscomplexobj(out) else _round_kernel_real)(flat, float(scale), float(tol), bool(divide))
    return out


def spgemm_round(A: sps.csr_array, B: sps.csr_array, scale: float, tol: float, *, divide: bool = False):
    """``A @ B`` rounded to the grid ``tol`` (as ``tol * matlab_round(scale * (A @ B))``) with zeros removed.

    Returns a ``csr_array`` with sorted column indices, or ``None`` when the compiled path does not apply
    (no Numba, or a problem below the size thresholds), in which case the caller runs the SciPy code.

    Spinach authors (clean_up.m, product and rounding in one pass): Ilya Kuprov, Alex Nevzorov.
    Spinach authors (propagator.m, the squarings): Ilya Kuprov, Luke Edwards.
    """
    if not _use_compiled(max(A.nnz, B.nnz), MIN_NNZ) or A.shape[1] != B.shape[0]:
        return None
    if A.dtype not in (np.float64, np.complex128) or B.dtype not in (np.float64, np.complex128):
        return None
    complex_in = np.iscomplexobj(A.data) or np.iscomplexobj(B.data)
    A = sps.csr_array(A)
    B = sps.csr_array(B)
    n = A.shape[0]
    nthreads = numba.get_num_threads()
    nblocks = min(max(1, n), 4 * nthreads)
    ia, ib = A.indptr.astype(np.int64, copy=False), B.indptr.astype(np.int64, copy=False)
    bounds = _row_blocks(ia, A.indices, ib, nblocks)
    args = (
        ia,
        A.indices,
        A.data.astype(np.complex128, copy=False),
        ib,
        B.indices,
        B.data.astype(np.complex128, copy=False),
        B.shape[1],
        float(scale),
        float(tol),
        bool(divide),
    )
    ranges = [(int(bounds[b]), int(bounds[b + 1])) for b in range(nblocks) if bounds[b + 1] > bounds[b]]
    with ThreadPoolExecutor(max_workers=nthreads) as pool:  # the blocks release the GIL
        parts = list(pool.map(lambda r: _spgemm_block(*args, r[0], r[1]), ranges))
    counts = np.concatenate([p[0] for p in parts]) if parts else np.zeros(0, dtype=np.int64)
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(counts, out=indptr[1:])
    nnz = int(indptr[-1])
    indices = np.concatenate([p[1] for p in parts]) if parts else np.zeros(0, dtype=np.int32)
    data = np.concatenate([p[2] for p in parts]) if parts else np.zeros(0, dtype=np.complex128)
    if not complex_in:
        data = data.real.copy()
    wide = nnz >= np.iinfo(np.int32).max or max(n, B.shape[1]) >= np.iinfo(np.int32).max
    index_dtype = np.int64 if wide else np.int32
    result = sps.csr_array(
        (data, indices.astype(index_dtype, copy=False), indptr.astype(index_dtype)), shape=(n, B.shape[1])
    )
    result.has_sorted_indices = True
    result.has_canonical_format = True
    return result


def csr_times_dense(A: sps.csr_array, X: np.ndarray):
    """``A @ X`` for a CSR matrix and a dense stack ``(n, k)`` (a vector is a stack of one column).

    Returns ``None`` when the compiled path does not apply; the caller then uses ``A @ X``.
    """
    if not _use_compiled(A.nnz * (X.shape[1] if X.ndim == 2 else 1), MIN_NNZ):
        return None
    if A.dtype not in (np.float64, np.complex128) or X.dtype not in (np.float64, np.complex128):
        return None
    vector = X.ndim == 1
    X2 = np.ascontiguousarray(X.reshape(-1, 1) if vector else X, dtype=np.complex128)
    out = np.empty((A.shape[0], X2.shape[1]), dtype=np.complex128)
    _csr_times_dense_kernel(A.indptr, A.indices, A.data.astype(np.complex128, copy=False), X2, out)
    if not (np.iscomplexobj(A.data) or np.iscomplexobj(X)):
        out = out.real.copy()
    return out[:, 0] if vector else out


def taylor_term(A: sps.csr_array, term: np.ndarray, coef: complex, rho: np.ndarray, eps: float):
    """One term of the Taylor series: returns ``(coef * (A @ term), any(|new term| > eps))`` and adds the new
    term to ``rho`` in place; ``None`` when the compiled path does not apply (the caller uses NumPy).

    Spinach authors (step.m, one term of subfunction reordered_taylor, fused): Ilya Kuprov, Luke Edwards,
      Anupama Acharya, one contributor listed only by email address.
    """
    if not _use_compiled(A.nnz * (term.shape[1] if term.ndim == 2 else 1), MIN_NNZ):
        return None
    if A.dtype not in (np.float64, np.complex128) or term.dtype != np.complex128 or rho.dtype != np.complex128:
        return None
    vector = term.ndim == 1
    term2 = np.ascontiguousarray(term.reshape(-1, 1) if vector else term)
    rho2 = rho.reshape(-1, 1) if vector else rho
    if not (rho2.flags.c_contiguous and rho2.flags.writeable) or np.shares_memory(rho2, term2):
        return None  # the sum is accumulated in place, which needs its own buffer
    out = np.empty_like(term2)
    big = _taylor_term_kernel(
        A.indptr, A.indices, A.data.astype(np.complex128, copy=False), term2, complex(coef), rho2, out, eps
    )
    return (out[:, 0] if vector else out), bool(big)
