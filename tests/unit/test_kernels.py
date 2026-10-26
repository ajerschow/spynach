"""The compiled kernels (spynach.kernels) reproduce the NumPy/SciPy code they replace, and the code runs
without Numba: rounding to the grid, the fused sparse product, the threaded sparse-times-dense product and the
fused Taylor term, each against the plain implementation, on matrices with empty rows, exact cancellations,
unsorted indices and real or complex entries."""

import numpy as np
import pytest
import scipy.sparse as sps

from spynach import kernels
from spynach.propagation import _reordered_taylor, propagator
from spynach.system import SpinSystem  # noqa: F401  (imported for the type of the tolerance carrier below)
from spynach.tolerances import Tolerances
from spynach.utilities import chop, clean_up, matlab_round, matmul, matmul_clean

numba = pytest.importorskip("numba")


@pytest.fixture
def compiled(monkeypatch):
    """Send every problem, however small, through the compiled kernels."""
    monkeypatch.setattr(kernels, "MIN_NNZ", 0)
    monkeypatch.setattr(kernels, "MIN_ELEMENTS", 0)


def random_sparse(n, m, per_row, seed, *, complex_values=True, empty_rows=()):
    rng = np.random.default_rng(seed)
    rows = np.repeat(np.arange(n), per_row)
    cols = rng.integers(0, m, rows.size)
    vals = rng.standard_normal(rows.size) * 0.4
    if complex_values:
        vals = vals + 1j * rng.standard_normal(rows.size) * 0.4
    keep = ~np.isin(rows, empty_rows)
    return sps.csr_array(sps.coo_array((vals[keep], (rows[keep], cols[keep])), shape=(n, m)))


def reference_product(A, B, tol, scale=None):
    C = sps.csr_array(A @ B, copy=True)
    C.data = tol * matlab_round((1 / tol if scale is None else scale) * C.data)
    C.eliminate_zeros()
    C.sort_indices()
    return C


@pytest.mark.parametrize("divide", [False, True])
@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
def test_round_array_is_matlab_round(compiled, divide, dtype):
    tol = 1e-3
    values = np.array([0.5, 1.5, 2.5, -0.5, -2.5, 0.0, -0.0, 0.4999, -0.4999, 123.4567, 1e-9, -1e-9]) * tol
    if dtype is np.complex128:
        values = values + 1j * values[::-1]
    scale = tol if divide else 1 / tol
    ours = kernels.round_array(values, scale, tol, divide=divide)
    expected = tol * matlab_round(values / scale if divide else scale * values)
    np.testing.assert_array_equal(ours, expected)  # -0.0 and +0.0 compare equal, as they do everywhere downstream
    assert ours.dtype == values.dtype and ours.shape == values.shape


def test_round_array_keeps_shape_and_leaves_input_alone(compiled):
    values = np.random.default_rng(0).standard_normal((7, 5)) + 0j
    original = values.copy()
    ours = kernels.round_array(values, 1e4, 1e-4)
    np.testing.assert_array_equal(values, original)
    np.testing.assert_array_equal(ours, 1e-4 * matlab_round(1e4 * values))


@pytest.mark.parametrize("complex_values", [True, False])
def test_spgemm_round_matches_scipy(compiled, complex_values):
    tol = 1e-6
    A = random_sparse(300, 250, 6, 1, complex_values=complex_values, empty_rows=(0, 17, 299))
    B = random_sparse(250, 280, 5, 2, complex_values=complex_values, empty_rows=(3, 100))
    ours = kernels.spgemm_round(A, B, 1 / tol, tol)
    expected = reference_product(A, B, tol)
    assert ours.shape == expected.shape and ours.dtype == expected.dtype
    assert ours.has_sorted_indices
    difference = (ours - expected).tocsr()
    # same arithmetic in the same order: the elements agree, apart from the odd one that a fused multiply-add
    # moves across a rounding boundary, by exactly one grid step
    flipped = np.abs(difference.data) > 1e-12
    assert flipped.sum() <= 1e-4 * max(expected.nnz, 1)
    assert np.all(np.abs(difference.data[flipped]) <= 1.0001 * tol * np.sqrt(2))


def test_spgemm_round_drops_cancellations_and_handles_unsorted_input(compiled):
    # [[1, 1], [0, 1]] @ [[1, 0], [-1, 1]] has an exact zero in the first row: the (0, 0) element 1 - 1
    A = sps.csr_array(np.array([[1.0, 1.0], [0.0, 1.0]]) + 0j)
    B = sps.csr_array(np.array([[1.0, 0.0], [-1.0, 1.0]]) + 0j)
    ours = kernels.spgemm_round(A, B, 1e10, 1e-10)
    np.testing.assert_array_equal(ours.toarray(), [[0, 1], [-1, 1]])
    assert ours.nnz == 3
    # a matrix whose column indices are not sorted within a row
    shuffled = sps.csr_array((np.array([2.0, 3.0]) + 0j, np.array([1, 0]), np.array([0, 2, 2])), shape=(2, 2))
    out = kernels.spgemm_round(shuffled, B, 1e10, 1e-10)
    np.testing.assert_allclose(out.toarray(), shuffled.toarray() @ B.toarray())


def test_spgemm_round_declines_what_it_cannot_do(monkeypatch):
    A = random_sparse(20, 20, 3, 0)
    monkeypatch.setattr(kernels, "MIN_NNZ", 200_000)
    assert kernels.spgemm_round(A, A, 1e10, 1e-10) is None  # below the size threshold
    monkeypatch.setattr(kernels, "MIN_NNZ", 0)
    assert kernels.spgemm_round(A.astype(np.complex64), A, 1e10, 1e-10) is None  # a dtype it does not handle


def test_matmul_clean_equals_clean_up_of_the_product(compiled):
    tols = Tolerances()
    A = random_sparse(400, 400, 4, 3)
    B = random_sparse(400, 400, 4, 4)
    ours = matmul_clean(tols, A, B, tols.prop_chop)
    expected = clean_up(tols, A @ B, tols.prop_chop)
    assert sps.issparse(ours) == sps.issparse(expected)
    np.testing.assert_allclose(sps.csr_array(ours).toarray(), sps.csr_array(expected).toarray(), atol=2e-10)


def test_csr_times_dense_matches_scipy(compiled):
    A = random_sparse(500, 400, 7, 5, empty_rows=(2, 3))
    rng = np.random.default_rng(6)
    X = rng.standard_normal((400, 9)) + 1j * rng.standard_normal((400, 9))
    np.testing.assert_allclose(matmul(A, X), A @ X, rtol=1e-13, atol=1e-13)
    x = X[:, 0]
    np.testing.assert_allclose(matmul(A, x), A @ x, rtol=1e-13, atol=1e-13)
    assert matmul(A, x).shape == (500,)
    real = random_sparse(500, 400, 7, 5, complex_values=False)
    assert matmul(real, X.real).dtype == np.float64
    np.testing.assert_allclose(matmul(real, X.real), real @ X.real, rtol=1e-13, atol=1e-13)
    # a non-contiguous stack
    np.testing.assert_allclose(matmul(A, X[:, ::2]), A @ X[:, ::2], rtol=1e-13, atol=1e-13)


def test_taylor_term_matches_the_plain_series(compiled, monkeypatch):
    A = random_sparse(300, 300, 5, 7) * 0.02
    A = sps.csr_array(A + A.conj().T)
    rng = np.random.default_rng(8)
    rho = rng.standard_normal((300, 4)) + 1j * rng.standard_normal((300, 4))
    fused = _reordered_taylor([A], rho.copy(), 1.7, 3)
    monkeypatch.setattr(kernels, "HAVE_NUMBA", False)
    plain = _reordered_taylor([A], rho.copy(), 1.7, 3)
    np.testing.assert_allclose(fused, plain, rtol=1e-12, atol=1e-13)
    assert np.abs(plain - rho).max() > 1e-3  # the series moved the state, so the comparison means something


def test_taylor_term_keeps_the_start_vector(compiled):
    """The first term of the series is the start vector itself; accumulating in place must not alter it."""
    A = sps.csr_array(random_sparse(200, 200, 4, 9) * 0.01)
    rho = np.random.default_rng(10).standard_normal((200, 2)) + 0j
    before = rho.copy()
    _reordered_taylor([A], rho, 0.5, 2)
    np.testing.assert_array_equal(rho, before)


def test_propagator_with_kernels_matches_the_fallback(compiled, monkeypatch):
    class Carrier:
        tols = Tolerances()

    n = 700
    H = random_sparse(n, n, 5, 11)
    H = sps.csr_array(H + H.conj().T)
    fast = propagator(Carrier(), H, 0.05)
    monkeypatch.setattr(kernels, "HAVE_NUMBA", False)
    slow = propagator(Carrier(), H, 0.05)
    assert sps.issparse(fast) == sps.issparse(slow)
    np.testing.assert_allclose(sps.csr_array(fast).toarray(), sps.csr_array(slow).toarray(), atol=1e-8)


def test_chop_goes_through_the_kernel_with_the_division(compiled):
    x = np.random.default_rng(12).standard_normal((50, 3)) * 1e-6 + 0j
    np.testing.assert_array_equal(chop(x, 1e-9), 1e-9 * matlab_round(x / 1e-9))
    S = sps.csr_array(x[:, :2])
    np.testing.assert_array_equal(chop(S, 1e-9).toarray(), 1e-9 * matlab_round(S.toarray() / 1e-9))


def test_everything_runs_without_numba(monkeypatch):
    """With Numba absent the helpers use the NumPy and SciPy code and the kernels decline."""
    monkeypatch.setattr(kernels, "HAVE_NUMBA", False)
    monkeypatch.setattr(kernels, "MIN_NNZ", 0)
    monkeypatch.setattr(kernels, "MIN_ELEMENTS", 0)
    A = random_sparse(100, 100, 4, 13)
    assert kernels.spgemm_round(A, A, 1e10, 1e-10) is None
    assert kernels.csr_times_dense(A, np.ones((100, 2), dtype=complex)) is None
    assert (
        kernels.taylor_term(A, np.ones((100, 2), dtype=complex), 1j, np.zeros((100, 2), dtype=complex), 1e-16) is None
    )
    tols = Tolerances()
    np.testing.assert_array_equal(
        sps.csr_array(matmul_clean(tols, A, A, 1e-10)).toarray(),
        sps.csr_array(clean_up(tols, A @ A, 1e-10)).toarray(),
    )
    values = np.array([0.5, -1.5, 2.5]) * 1e-3
    np.testing.assert_array_equal(kernels.round_array(values, 1e3, 1e-3), 1e-3 * matlab_round(1e3 * values))
