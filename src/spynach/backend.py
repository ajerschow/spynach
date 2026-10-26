"""Array backend adapter: the one place the kernel's hot loops touch NumPy/SciPy by name.

The propagation code (:mod:`spynach.propagation`, :mod:`spynach.reduction`,
:func:`spynach.utilities.clean_up`) asks this module for its array module,
sparse constructors and host/device transfers instead of importing
``numpy``/``scipy.sparse`` directly. Today there is one backend, NumPy +
SciPy on the CPU, and every transfer is a no-op. A GPU backend (CuPy and
``cupyx.scipy.sparse``, see ``porting/gpu-notes.md``) plugs in here by
providing the same attributes; nothing in the kernel is expected to
change except what ``get_backend()`` returns.

Construction code (basis, superoperators, Hamiltonians, relaxation) stays
on the CPU, as in Spinach: it is index arithmetic, not linear algebra.

Lines marked ``# GPU:`` elsewhere in the package are the points where the
device backend will matter (transfers, exponentials, dense/sparse
decisions).

Spinach authors, by original file (headers of Spinach 2.13; see docs/spinach-credits.md):
  cheap_norm.m: Ilya Kuprov
  clean_up.m: Ilya Kuprov, Alex Nevzorov
  evolution.m: Ilya Kuprov, Luke Edwards, Ohad Levinkron
  krylov.m: Ilya Kuprov
  propagator.m: Ilya Kuprov, Luke Edwards
  step.m: Ilya Kuprov, Luke Edwards, Anupama Acharya, one contributor listed only by email address
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sps

from spynach import kernels


@dataclass(frozen=True)
class NumpyBackend:
    """CPU backend: NumPy arrays and SciPy ``csr_array`` sparse matrices."""

    name: str = "numpy"
    xp: Any = np  # array module (CuPy would go here)

    @staticmethod
    def issparse(x: Any) -> bool:
        return sps.issparse(x)

    @staticmethod
    def csr(x: Any, copy: bool = False) -> sps.csr_array:
        """Sparse matrix in CSR form (``cupyx.scipy.sparse.csr_matrix`` on a GPU backend)."""
        return sps.csr_array(x, copy=copy)

    @staticmethod
    def eye(n: int, dtype: Any = complex) -> sps.csr_array:
        return sps.eye_array(n, dtype=dtype, format="csr")

    @staticmethod
    def dense(x: Any, dtype: Any = None) -> np.ndarray:
        """Dense array on the device, from a dense or sparse input."""
        return np.asarray(x.toarray() if sps.issparse(x) else x, dtype=dtype)

    @staticmethod
    def to_device(x: Any) -> Any:
        """Move a dense or sparse array to the compute device. No-op on the CPU."""
        return x

    @staticmethod
    def to_host(x: Any) -> Any:
        """Move an array back to host memory (NumPy/SciPy). No-op on the CPU."""
        return x

    @staticmethod
    def expm(A: Any) -> np.ndarray:
        """Dense matrix exponential of a small matrix. Stays on the host even for GPU backends,
        as in Spinach (``propagator.m`` uses ``expm`` only below ``small_matrix``)."""
        return sla.expm(np.asarray(A))

    @staticmethod
    def nnz(x: Any) -> int:
        return int(x.count_nonzero()) if sps.issparse(x) else int(np.count_nonzero(x))

    @staticmethod
    def round_array(values: Any, scale: float, tol: float, *, divide: bool = False) -> Any:
        """``tol * matlab_round(scale * values)`` (``values / scale`` with ``divide``): the rounding grid of
        ``clean_up`` and ``chop``. Threaded Numba kernel for large arrays, NumPy otherwise."""
        return kernels.round_array(values, scale, tol, divide=divide)

    @staticmethod
    def spgemm_round(A: Any, B: Any, scale: float, tol: float) -> Any:
        """``A @ B`` for two CSR matrices, rounded to the grid ``tol`` with zeros dropped, in one threaded pass
        (``cupyx`` sparse products on a GPU backend); ``None`` where the compiled kernel does not apply."""
        return kernels.spgemm_round(A, B, scale, tol)

    @staticmethod
    def csr_times_dense(A: Any, X: Any) -> Any:
        """``A @ X`` for a CSR matrix and a dense stack, threaded; ``None`` where the kernel does not apply."""
        return kernels.csr_times_dense(A, X)

    @staticmethod
    def taylor_term(A: Any, term: Any, coef: complex, rho: Any, eps: float) -> Any:
        """One fused term of the Taylor series of ``step``: ``(coef * (A @ term), any(|.| > eps))``, adding the
        term to ``rho`` in place; ``None`` where the kernel does not apply."""
        return kernels.taylor_term(A, term, coef, rho, eps)

    @staticmethod
    def norm1(x: Any) -> float:
        """Maximum absolute column sum (Spinach ``cheap_norm`` for ordinary matrices)."""
        if x.shape[1] == 0:
            return 0.0
        return float(np.max(abs(x).sum(axis=0)))


_BACKEND: NumpyBackend = NumpyBackend()


def get_backend() -> NumpyBackend:
    """The active array backend. Only NumPy exists so far."""
    return _BACKEND


def use(name: str) -> None:
    """Select the array backend by name. ``'numpy'`` is the only one implemented;
    ``'cupy'`` is reserved for the GPU backend described in ``porting/gpu-notes.md``."""
    global _BACKEND
    if name == "numpy":
        _BACKEND = NumpyBackend()
    elif name == "cupy":
        raise NotImplementedError("the CuPy backend is not implemented yet; see porting/gpu-notes.md.")
    else:
        raise ValueError(f"unknown backend {name!r}; use 'numpy'.")
