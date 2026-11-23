"""The remaining Hilbert-space bases and operator utilities."""

import numpy as np
import pytest
import sympy as sp

import spyndynamica as sd
from spyndynamica import SpinSystem
from tests.spyndynamica import reference

TOL = dict(rtol=0, atol=1e-10)
SYSTEMS = {
    "two_half": lambda: SpinSystem(2),
    "three_half": lambda: SpinSystem(3),
    "half_one": lambda: SpinSystem([sp.Rational(1, 2), 1]),
    "one_one": lambda: SpinSystem([1, 1]),
}


def stored(key):
    return reference.get("11_bases_and_operators", key)


def number(matrix):
    return np.array(sp.N(matrix, 30).tolist(), dtype=complex)


@pytest.mark.parametrize("group", list(SYSTEMS))
def test_angular_momentum_basis(group):
    s = SYSTEMS[group]()
    basis = s.angular_momentum_basis()
    kets = number(basis.kets_matrix)
    assert np.allclose(kets.conj().T @ kets, np.eye(s.dimension), **TOL)
    labels = [label for label, _ in s.spins]
    total = sum((s.dot(a, b) for a in labels for b in labels), s.zero())
    j2, jz = total.numeric(basis=basis), s.I(None, "z").numeric(basis=basis)
    expected_j2 = [float(j * (j + 1)) for j, _, _ in basis.quantum_numbers]
    assert np.allclose(j2, np.diag(expected_j2), **TOL)
    assert np.allclose(jz, np.diag([float(m) for _, m, _ in basis.quantum_numbers]), **TOL)
    # SpinDynamica's kets span the same (J, M) subspaces
    theirs = stored(f"{group}/angular_momentum_kets").T
    assert np.allclose(stored(f"{group}/total_spin_squared"), theirs.conj().T @ total.numeric() @ theirs, **TOL)
    their_j2 = np.diag(stored(f"{group}/total_spin_squared")).real
    their_m = np.diag(stored(f"{group}/total_Iz")).real
    for j, m, _ in set(basis.quantum_numbers):
        mine = kets[:, [n for n, q in enumerate(basis.quantum_numbers) if q[:2] == (j, m)]]
        columns = np.isclose(their_j2, float(j * (j + 1))) & np.isclose(their_m, float(m))
        assert columns.sum() == mine.shape[1]
        assert np.allclose(mine @ mine.conj().T, theirs[:, columns] @ theirs[:, columns].conj().T, **TOL), (j, m)
    lowering = s.I(None, "-").numeric(basis=basis)
    assert np.all(lowering.real >= -1e-12), "Condon-Shortley phases: the lowering operator has no negative elements"


def test_eigenbasis_transformed_basis_and_quantum_numbers():
    s = SpinSystem(2)
    i = s.I
    ham = 2 * sp.pi * 30 * i(1, "z") - 2 * sp.pi * 10 * i(2, "z") + 2 * sp.pi * 12 * s.dot(1, 2)
    basis = s.eigenbasis(ham)
    diagonal = ham.numeric(basis=basis)
    assert np.allclose(diagonal, np.diag(basis.eigenvalues), atol=1e-9) and np.all(np.diff(basis.eigenvalues) > 0)
    theirs = stored("two_half_bases/eigenbasis_kets").T
    their_h = stored("two_half_bases/hamiltonian_in_eigenbasis")
    assert np.allclose(their_h, np.diag(np.diag(their_h)), atol=1e-9)
    assert np.allclose(np.sort(np.diag(their_h).real), basis.eigenvalues, atol=1e-9)
    mine = number(basis.kets_matrix)
    for value, column in zip(np.diag(their_h).real, theirs.T):
        own = mine[:, np.argmin(np.abs(basis.eigenvalues - value))]
        assert np.isclose(abs(np.vdot(own, column)), 1.0, atol=1e-9)
    st = s.singlet_triplet()
    matched = s.eigenbasis(s.dot(1, 2), target=st)
    assert np.isclose(abs(np.vdot(number(matched.kets_matrix)[:, 0], number(st.kets_matrix)[:, 0])), 1.0)
    assert np.allclose(matched.eigenvalues, [-0.75, 0.25, 0.25, 0.25])
    turned = s.zeeman.transformed(s.rotation(sp.pi / 2, "y", spins=[1]))
    assert np.allclose(number(turned.kets_matrix).T, stored("two_half_bases/transformed_basis_kets"), **TOL)
    assert np.allclose(i(1, "z").numeric(basis=turned), stored("two_half_bases/I1z_in_transformed_basis"), **TOL)
    numbers = [k.magnetic_quantum_numbers()[0] for k in s.zeeman.kets]
    assert np.allclose(np.array(numbers, dtype=float), stored("two_half_bases/magnetic_quantum_numbers"), **TOL)
    spin_1 = [k.magnetic_quantum_numbers([1])[0] for k in s.zeeman.kets]
    assert np.allclose(np.array(spin_1, dtype=float), stored("two_half_bases/magnetic_quantum_numbers_spin_1"), **TOL)
    assert st.ket("T0").magnetic_quantum_numbers([1]) == [sp.Rational(1, 2), -sp.Rational(1, 2)]


def test_normalisation_projection_and_shift_superoperator():
    s = SpinSystem(2)
    i = s.I
    mixed = i(1, "x") @ i(2, "x") + 2 * i(1, "z") @ i(2, "z") + 3 * i(1, "x") + i(1, "+") @ i(2, "-") + 5 * i(2, "z")
    g = "two_half_bases"
    assert np.allclose(mixed.normalized().numeric(), stored(f"{g}/normalized"), **TOL)
    assert np.allclose(mixed.normalized(3).numeric(), stored(f"{g}/normalized_to_3"), **TOL)
    assert np.allclose(mixed.normalized(i(1, "x")).numeric(), stored(f"{g}/normalized_to_operator"), **TOL)
    assert mixed.normalized().norm() == 1 or abs(float(mixed.normalized().norm()) - 1) < 1e-12
    shift = sd.shift_superoperator(i(1, "x"), i(2, "z"))
    assert np.allclose(shift.numeric(basis=sd.default_operator_basis(s)), stored(f"{g}/shift_superoperator"), **TOL)
    assert np.allclose(shift(mixed).numeric(), stored(f"{g}/shift_superoperator_applied"), **TOL)
    assert shift(mixed).equals(5 * i(1, "x"))
    # D6 in DIVERGENCES.md: documented in SpinDynamica but not implemented there
    assert mixed.projection(i(1, "z") @ i(2, "z")).equals(2 * i(1, "z") @ i(2, "z"))
    onto = i(1, "x") + i(2, "z")
    assert mixed.projection(onto).equals(4 * onto) and (mixed - mixed.projection(onto)).bracket(onto) == 0
