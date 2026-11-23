"""The fast paths of the numeric engine give the results of the plain ones."""

import time

import numpy as np
import pytest
import sympy as sp

import spyndynamica as sd
from spyndynamica import SpinSystem, backend

TOL = dict(rtol=0, atol=1e-10)


def chain(n):
    s = SpinSystem(n)
    h = s.zero()
    for j in range(1, n + 1):
        h = h + 2 * sp.pi * 100 * j * s.I(j, "z")
    for j in range(1, n):
        h = h + 2 * sp.pi * 7 * s.dot(j, j + 1)
    return s, h


def test_sparse_matrices_are_the_kronecker_products():
    s = SpinSystem([sp.Rational(1, 2), 1, sp.Rational(3, 2)])
    op = 2 * s.I(1, "x") @ s.T(2, 2, 1) + sp.I * s.I(3, "+") - 3 * s.unity() + s.I(1, "y") @ s.I(3, "z")
    dense = np.zeros((24, 24), dtype=complex)
    for key, value in op.terms.items():
        block = np.ones((1, 1), dtype=complex)
        for loc, label in zip(s.locals, key):
            block = np.kron(loc.numeric(label), block)
        dense += complex(value) * block
    assert np.allclose(op.sparse().toarray(), dense, **TOL) and np.allclose(op.numeric(), dense, **TOL)
    a = sp.Symbol("a", real=True)
    assert np.allclose((a * op).sparse({a: 2.5}).toarray(), 2.5 * dense, **TOL)
    assert s.zero().sparse().nnz == 0 and op.sparse().nnz < 24 * 24 / 4


def test_numeric_pulses_are_the_symbolic_rotations():
    flip, phase = sp.symbols("flip phase", real=True)
    for system in (SpinSystem(2), SpinSystem([sp.Rational(1, 2), 1]), SpinSystem(["3/2"])):
        labels = [label for label, _ in system.spins]
        rho = system.I(None, "z") + 2 * system.I(labels[0], "x")
        for spins, values in ((None, {flip: 0.7, phase: 0.3}), ([labels[-1]], {flip: 2.1, phase: -1.2})):
            exact = rho.rotate(flip, phase, spins=spins).numeric(values)
            assert np.allclose(sd.propagate(rho, [sd.Pulse(flip, phase, spins=spins)], values=values), exact, **TOL)
        for name in ("x", "y", "-x", "-y"):
            exact = rho.rotate(sp.pi / 3, name).numeric()
            assert np.allclose(sd.propagate(rho, [sd.Pulse(sp.pi / 3, name)]), exact, **TOL)
        superoperator = sd.rotation_superoperator(system, sp.pi / 3, "y")
        assert np.allclose(sd.propagate(rho, [superoperator]), sd.propagate(rho, [sd.Pulse(sp.pi / 3, "y")]), **TOL)


def test_sparse_method_agrees_with_dense():
    s, h = chain(4)
    i = s.I
    events = [
        sd.Pulse(sp.pi / 2, "x"),
        (h, 0.003),
        sd.Repeat([sd.Pulse(sp.pi, "y", spins=[1, 3]), sd.Delay(0.001), (h + 2 * sp.pi * 30 * i(2, "x"), 0.002)], 3),
    ]
    background = 2 * sp.pi * 15 * s.dot(1, 4)
    rho = i(None, "z")
    dense = sd.propagate(rho, events, background=background, method="dense")
    sparse = sd.propagate(rho, events, background=background, method="sparse")
    assert np.allclose(dense, sparse, **TOL)
    targets = [i(1, "x"), i(2, "y") @ i(3, "z"), i(4, "+")]
    a = sd.amplitudes(rho, targets, events, background=background, method="dense")
    b = sd.amplitudes(rho, targets, events, background=background, method="sparse")
    reference = [np.vdot(t.numeric(), dense) / np.vdot(t.numeric(), t.numeric()) for t in targets]
    assert np.allclose(a, b, **TOL) and np.allclose(a, reference, **TOL)
    assert np.allclose(sd.propagate(rho, events, order="right_to_left", method="sparse"),
                       sd.propagate(rho, events, order="right_to_left", method="dense"), **TOL)  # fmt: skip
    relaxation = sd.phenomenological_relaxation(s, 1.0)
    with pytest.raises(ValueError, match="Hamiltonians and ideal pulses only"):
        sd.propagate(rho, events, background=relaxation, method="sparse")
    with pytest.raises(ValueError, match="method must be"):
        sd.propagate(rho, events, method="fast")
    # 'auto' falls back to the dense engine when relaxation is present
    small = [sd.Pulse(sp.pi / 2, "x"), (h, 0.001)]
    assert np.allclose(sd.propagate(rho, small, background=relaxation),
                       sd.propagate(rho, small, background=relaxation, method="dense"), **TOL)  # fmt: skip


def test_blocks_follow_the_conserved_quantities():
    s, h = chain(5)
    blocks = backend.block_eigensystems(h.sparse())
    # a Hamiltonian of shifts and scalar couplings conserves the total Iz: one block per value of M
    assert sorted(len(indices) for indices, _, _ in blocks) == sorted([1, 5, 10, 10, 5, 1])
    assert len(backend.block_eigensystems((h + s.I(1, "x")).sparse())) == 1
    rho = (s.I(None, "x") + s.I(2, "z") @ s.I(4, "y")).numeric()
    u = backend.unitary(h.numeric(), 0.004)
    assert np.allclose(backend.evolve_in_blocks(rho, blocks, 0.004), u @ rho @ u.conj().T, **TOL)


def test_ten_spins():
    s, h = chain(10)
    assert s.dimension >= sd.dynamics.SPARSE_FROM
    start = time.perf_counter()
    values = sd.amplitudes(s.I(None, "z"), [s.I(1, "x"), s.I(1, "y")], [sd.Pulse(sp.pi / 2, "x"), (h, 0.01)])
    elapsed = time.perf_counter() - start
    # the same numbers as for a chain of 6 with the dense engine: the first spin sees its neighbour only weakly
    small_system, small_h = chain(6)
    small = sd.amplitudes(
        small_system.I(None, "z"),
        [small_system.I(1, "x"), small_system.I(1, "y")],
        [sd.Pulse(sp.pi / 2, "x"), (small_h, 0.01)],
        method="dense",
    )
    assert np.allclose(values, small, atol=1e-6)
    assert elapsed < 30, f"ten spins took {elapsed:.1f} s"
