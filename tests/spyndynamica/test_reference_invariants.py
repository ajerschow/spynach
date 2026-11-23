"""Checks of the SpinDynamica reference that do not depend on SpinDynamica.

A fixture records what SpinDynamica returns. These tests hold the fixtures against results established
independently: angular momentum algebra, group properties of rotations, SymPy's Wigner and Clebsch-Gordan
routines, and the defining formulas of the superoperators. A disagreement found here is recorded in
``porting/spyndynamica/DIVERGENCES.md`` and the port follows the correct result.
"""

import numpy as np
import pytest
import sympy
from sympy.physics.quantum.cg import CG
from sympy.physics.quantum.spin import Rotation

from tests.spyndynamica import reference as fixtures

TOL = dict(rtol=0, atol=1e-12)
SPINS = {"one_half": {1: 0.5}, "two_half": {1: 0.5, 2: 0.5}, "spin1": {1: 1.0}, "half_one": {1: 0.5, 2: 1.0},
         "spin32": {1: 1.5}}  # fmt: skip


def op(group, name):
    return fixtures.get("02_operators", f"{group}/{name}")


def dagger(a):
    return np.conj(np.swapaxes(a, -1, -2))


# ------------------------------------------------------------------------------------------- bases
@pytest.mark.parametrize("group", ["one_half", "two_half", "three_half", "spin1", "half_one", "spin32"])
def test_zeeman_basis_is_the_unit_basis_with_the_first_spin_fastest(group):
    kets = fixtures.get("01_bases", f"{group}/zeeman_kets")
    assert np.allclose(kets, np.eye(len(kets)), **TOL)
    m = fixtures.get("01_bases", f"{group}/zeeman_quantum_numbers")[:, :, 1]
    # every spin runs from +I downwards, and the first spin changes fastest
    assert np.all(m[0] == m.max(axis=0))
    if m.shape[1] > 1:
        assert m[0, 0] != m[1, 0] and np.all(m[0, 1:] == m[1, 1:])
    assert np.allclose(fixtures.get("01_bases", f"{group}/Iz_total_diagonal"), m.sum(axis=1), **TOL)


def test_singlet_triplet_basis():
    u = fixtures.get("01_bases", "singlet_triplet/transformation_st_from_zeeman")
    back = fixtures.get("01_bases", "singlet_triplet/transformation_zeeman_from_st")
    assert np.allclose(u @ dagger(u), np.eye(4), **TOL)
    assert np.allclose(back, dagger(u), **TOL)
    # eigenstates of I1.I2: the singlet (-3/4) first, then the triplet (+1/4) with M = 1, 0, -1
    scalar = fixtures.get("01_bases", "singlet_triplet/I1I2_in_st")
    assert np.allclose(scalar, np.diag([-0.75, 0.25, 0.25, 0.25]), **TOL)
    kets = fixtures.get("01_bases", "singlet_triplet/kets_in_zeeman")
    iz = np.diag(fixtures.get("01_bases", "two_half/Iz_total_diagonal"))
    assert np.allclose([k.conj() @ iz @ k for k in kets], [0, 1, 0, -1], **TOL)


# ------------------------------------------------------------------------------------------- operators
@pytest.mark.parametrize("group", list(SPINS))
def test_angular_momentum_algebra(group):
    for spin, quantum_number in SPINS[group].items():
        x, y, z = (op(group, f"I{spin}{c}") for c in "xyz")
        plus, minus = op(group, f"I{spin}plus"), op(group, f"I{spin}minus")
        for a in (x, y, z):
            assert np.allclose(a, dagger(a), **TOL)
        assert np.allclose(x @ y - y @ x, 1j * z, **TOL)
        assert np.allclose(y @ z - z @ y, 1j * x, **TOL)
        assert np.allclose(z @ x - x @ z, 1j * y, **TOL)
        assert np.allclose(plus, x + 1j * y, **TOL) and np.allclose(minus, x - 1j * y, **TOL)
        casimir = x @ x + y @ y + z @ z
        assert np.allclose(casimir, quantum_number * (quantum_number + 1) * np.eye(len(x)), **TOL)


def test_two_spin_products_and_scalars():
    g = "two_half"
    assert np.allclose(op(g, "I1xI2z"), op(g, "I1x") @ op(g, "I2z"), **TOL)
    assert np.allclose(op(g, "I1x") @ op(g, "I2z"), op(g, "I2z") @ op(g, "I1x"), **TOL)
    scalar = sum(op(g, f"I1{c}") @ op(g, f"I2{c}") for c in "xyz")
    assert np.allclose(op(g, "I1_dot_I2"), scalar, **TOL)
    assert np.allclose(op(g, "commutator_I1x_I1y"), 1j * op(g, "I1z"), **TOL)
    assert np.allclose(op(g, "Ix_total"), op(g, "I1x") + op(g, "I2x"), **TOL)
    assert np.allclose(op(g, "I1alpha"), 0.5 * np.eye(4) + op(g, "I1z"), **TOL)
    assert np.allclose(op(g, "I2beta"), 0.5 * np.eye(4) - op(g, "I2z"), **TOL)
    # the permutation operator exchanges the spins
    p = op(g, "permutation_12")
    assert np.allclose(p @ op(g, "I1x") @ dagger(p), op(g, "I2x"), **TOL)
    assert np.allclose(p, 0.5 * np.eye(4) + 2 * scalar, **TOL)
    # scalar properties
    assert np.isclose(op(g, "trace_I1zI1z"), 1.0) and np.isclose(op(g, "trace_unity"), 4.0)
    assert np.isclose(op(g, "norm_I1x"), np.linalg.norm(op(g, "I1x")))
    assert np.isclose(op(g, "norm_I1plus"), np.linalg.norm(op(g, "I1plus")))


def test_single_transition_operators():
    g = "two_half"
    x, y, z = (op(g, f"st_{c}_12") for c in "xyz")
    assert np.allclose(x @ y - y @ x, 1j * z, **TOL)
    expected = np.zeros((4, 4))
    expected[0, 1] = expected[1, 0] = 0.5
    assert np.allclose(x, expected, **TOL)
    assert np.allclose(op(g, "coherence_1_2"), np.outer(np.eye(4)[0], np.eye(4)[1]), **TOL)
    assert np.allclose(op(g, "shift_ket1_bra2"), op(g, "coherence_1_2"), **TOL)
    assert np.allclose(op(g, "population_2"), np.diag([0, 1, 0, 0]), **TOL)


def test_cyclic_permutation_direction():
    """SpinPermutationOperator[{1, 2, 3}] is documented as 1 -> 2, 2 -> 3, 3 -> 1."""
    cycle = op("three_half", "permutation_cycle_123")
    assert np.allclose(cycle @ dagger(cycle), np.eye(8), **TOL)
    inverse = op("three_half", "permutation_cycle_132")
    assert np.allclose(cycle @ inverse, np.eye(8), **TOL)
    # a product basis state |m1 m2 m3> with the first spin fastest: index = b1 + 2 b2 + 4 b3
    state = np.zeros(8)
    state[1] = 1.0  # spin 1 down, spins 2 and 3 up
    moved = cycle @ state
    assert np.allclose(moved, np.eye(8)[2], **TOL), "the state of spin 1 moves to spin 2"
    # {{1, 2}, {2, 3}}: the exchange of 2 and 3 comes first, then that of 1 and 2
    p12 = op("three_half", "permutation_12")
    p23 = inverse @ p12  # (132)(12) = (23) when products act right to left
    composite = op("three_half", "permutation_12_after_23")
    assert np.allclose(composite @ dagger(composite), np.eye(8), **TOL)
    assert np.allclose(composite, p12 @ p23, **TOL) or np.allclose(composite, p12 @ dagger(cycle) @ p12, **TOL)


RANK1 = {1: lambda p, z, m: -p / np.sqrt(2), 0: lambda p, z, m: z, -1: lambda p, z, m: m / np.sqrt(2)}


@pytest.mark.parametrize("group, prefix, spin", [("one_half", "spin1_", 1), ("spin1", "", 1), ("spin32", "", 1),
                                                 ("half_one", "spin2_", 2), ("two_half", "spin2_", 2)])  # fmt: skip
def test_rank_one_tensor_operators(group, prefix, spin):
    plus, z, minus = (op(group, f"I{spin}{c}") for c in ("plus", "z", "minus"))
    for q, formula in RANK1.items():
        assert np.allclose(op(group, f"{prefix}T1_{q}"), formula(plus, z, minus), **TOL), q


@pytest.mark.parametrize("group, prefix, spin, ranks", [("spin1", "", 1, (1, 2)), ("spin32", "", 1, (1, 2, 3)),
                                                        ("half_one", "spin2_", 2, (1, 2))])  # fmt: skip
def test_tensor_operators_obey_the_racah_commutators(group, prefix, spin, ranks):
    plus, z, minus = (op(group, f"I{spin}{c}") for c in ("plus", "z", "minus"))
    for k in ranks:
        for q in range(-k, k + 1):
            t = op(group, f"{prefix}T{k}_{q}")
            assert np.allclose(z @ t - t @ z, q * t, **TOL)
            if q < k:
                raised = op(group, f"{prefix}T{k}_{q + 1}")
                assert np.allclose(plus @ t - t @ plus, np.sqrt(k * (k + 1) - q * (q + 1)) * raised, **TOL), (k, q)
            # T(k, -q) = (-1)^q T(k, q)^dagger
            assert np.allclose(op(group, f"{prefix}T{k}_{-q}"), (-1) ** q * dagger(t), **TOL), (k, q)


def test_two_spin_tensor_operators():
    g = "two_half"
    plus = op(g, "I1plus") + op(g, "I2plus")
    z = op(g, "I1z") + op(g, "I2z")
    for k in (0, 1, 2):
        for q in range(-k, k + 1):
            t = op(g, f"pair_T{k}_{q}")
            assert np.allclose(z @ t - t @ z, q * t, **TOL)
            if q < k:
                raised = op(g, f"pair_T{k}_{q + 1}")
                assert np.allclose(plus @ t - t @ plus, np.sqrt(k * (k + 1) - q * (q + 1)) * raised, **TOL), (k, q)
    scalar = op(g, "I1_dot_I2")
    assert np.allclose(op(g, "pair_T0_0"), -scalar / np.sqrt(3), **TOL)
    t20 = (3 * op(g, "I1zI2z") - scalar) / np.sqrt(6)
    assert np.allclose(op(g, "pair_T2_0"), t20, **TOL)


# ------------------------------------------------------------------------------------------- rotations
def rot(group, name):
    return fixtures.get("03_rotations", f"{group}/{name}")


ANGLES = (np.pi / 5, np.pi / 3, 2 * np.pi / 7)


def sympy_d(j, beta):
    j = sympy.Rational(str(j))
    ms = [j - i for i in range(int(2 * j) + 1)]
    return np.array([[complex(Rotation.d(j, m1, m2, beta).doit()) for m2 in ms] for m1 in ms]).real


@pytest.mark.parametrize("name, j", [("d_half", 0.5), ("d_1", 1), ("d_3half", 1.5), ("d_2", 2)])
def test_reduced_wigner_matrices_against_sympy(name, j):
    value = rot("wigner", f"{name}_symbolic")
    for parameters, matrix in value.points:
        assert np.allclose(matrix, sympy_d(j, parameters["b3"]), **TOL)


@pytest.mark.parametrize("name, j", [("D_half", 0.5), ("D_1", 1), ("D_3half", 1.5), ("D_2", 2)])
def test_wigner_matrices_are_the_zyz_product(name, j):
    a, b, c = ANGLES
    m = np.arange(j, -j - 1, -1)
    expected = np.exp(-1j * a * m)[:, None] * sympy_d(j, b) * np.exp(-1j * c * m)[None, :]
    matrix = rot("wigner", name)
    assert np.allclose(matrix, expected, **TOL)
    assert np.allclose(matrix @ dagger(matrix), np.eye(len(m)), **TOL)


def test_wigner_elements_rows_and_products():
    d2 = sympy_d(2, np.pi / 3)
    assert np.allclose(rot("wigner", "d_2_row_1"), d2[1], **TOL)
    assert np.allclose(rot("wigner", "d_2_column_1"), d2[:, 1], **TOL)
    assert np.isclose(rot("wigner", "D_2_element_1_m2"), rot("wigner", "D_2")[1, 4])

    def big_d(angles):
        a, b, c = angles
        m = np.arange(1, -2, -1)
        return np.exp(-1j * a * m)[:, None] * sympy_d(1, b) * np.exp(-1j * c * m)[None, :]

    second = (np.pi / 7, np.pi / 4, np.pi / 9)
    assert np.allclose(rot("wigner", "D_1_product_of_two"), big_d(ANGLES) @ big_d(second), **TOL)
    assert np.allclose(rot("wigner", "D_1_numeric"), big_d((0.3, 1.1, -0.7)), **TOL)


@pytest.mark.parametrize("name", ["half_half", "one_half", "one_one", "three_half_one", "two_one", "two_two"])
def test_clebsch_gordan_against_sympy(name):
    table = fixtures.cases("03_rotations")[f"clebsch_gordan/{name}"]["value"]
    numbers = fixtures.array(table)
    for row in numbers:
        j1, m1, j2, m2, j3, m3 = (sympy.nsimplify(x) for x in row[:6].real)
        assert np.isclose(row[6], float(CG(j1, m1, j2, m2, j3, m3).doit()), **TOL), row


def test_euler_matrix_is_rz_ry_rz():
    def rz(t):
        return np.array([[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0], [0, 0, 1]])

    def ry(t):
        return np.array([[np.cos(t), 0, np.sin(t)], [0, 1, 0], [-np.sin(t), 0, np.cos(t)]])

    a, b, c = ANGLES
    assert np.allclose(rot("euler", "matrix"), rz(a) @ ry(b) @ rz(c), **TOL)
    for parameters, matrix in rot("euler", "matrix_symbolic").points:
        expected = rz(parameters["a3"]) @ ry(parameters["b3"]) @ rz(parameters["c3"])
        assert np.allclose(matrix, expected, **TOL)
    assert np.allclose(rot("euler", "rotate_vector"), rz(a) @ ry(b) @ rz(c) @ np.array([1, 2, 3]), **TOL)


def expm_hermitian(h, angle):
    w, v = np.linalg.eigh(h)
    return (v * np.exp(-1j * angle * w)) @ dagger(v)


@pytest.mark.parametrize("group, spins", [("one_half", (1,)), ("two_half", (1, 2)), ("spin1", (1,)), ("spin32", (1,))])
def test_rotation_operators_are_exponentials_of_the_angular_momentum(group, spins):
    x, y, z = (sum(op(group, f"I{s}{c}") for s in spins) for c in "xyz")
    assert np.allclose(rot(group, "R_x_90"), expm_hermitian(x, np.pi / 2), **TOL)
    assert np.allclose(rot(group, "R_y_90"), expm_hermitian(y, np.pi / 2), **TOL)
    assert np.allclose(rot(group, "R_minus_x_90"), expm_hermitian(-x, np.pi / 2), **TOL)
    assert np.allclose(rot(group, "R_z_60"), expm_hermitian(z, np.pi / 3), **TOL)
    phase = np.pi / 4
    assert np.allclose(
        rot(group, "R_phase_pi4_120"), expm_hermitian(np.cos(phase) * x + np.sin(phase) * y, 2 * np.pi / 3), **TOL
    )
    theta, phi = np.pi / 4, np.pi / 5
    axis = np.sin(theta) * np.cos(phi) * x + np.sin(theta) * np.sin(phi) * y + np.cos(theta) * z
    assert np.allclose(rot(group, "R_polar"), expm_hermitian(axis, np.pi / 3), **TOL)
    a, b, c = ANGLES
    euler = expm_hermitian(z, a) @ expm_hermitian(y, b) @ expm_hermitian(z, c)
    assert np.allclose(rot(group, "R_euler"), euler, **TOL)
    for parameters, matrix in rot(group, "R_euler_symbolic").points:
        a3, b3, c3 = (parameters[k] for k in ("a3", "b3", "c3"))
        assert np.allclose(matrix, expm_hermitian(z, a3) @ expm_hermitian(y, b3) @ expm_hermitian(z, c3), **TOL)
    for parameters, matrix in rot(group, "R_flip_phase_symbolic").points:
        a3, b3 = parameters["a3"], parameters["b3"]
        assert np.allclose(matrix, expm_hermitian(np.cos(a3) * x + np.sin(a3) * y, b3), **TOL)


@pytest.mark.parametrize("group, name, operator", [("one_half", "I1z", "I1z"), ("one_half", "I1x", "I1x"),
                                                   ("one_half", "I1plus", "I1plus"), ("spin1", "I1z", "I1z"),
                                                   ("spin1", "T20", "T2_0"), ("spin32", "I1z", "I1z")])  # fmt: skip
def test_rotated_operators_are_sandwiches(group, name, operator):
    x, y, z = (op(group, f"I1{c}") for c in "xyz")
    a = op(group, operator)

    def sandwich(r):
        return r @ a @ dagger(r)

    assert np.allclose(rot(group, f"{name}_by_x_90"), sandwich(expm_hermitian(x, np.pi / 2)), **TOL)
    assert np.allclose(rot(group, f"{name}_by_y_90"), sandwich(expm_hermitian(y, np.pi / 2)), **TOL)
    assert np.allclose(rot(group, f"{name}_by_z_60"), sandwich(expm_hermitian(z, np.pi / 3)), **TOL)
    assert np.allclose(rot(group, f"{name}_by_euler"), sandwich(rot(group, "R_euler")), **TOL)
    for parameters, matrix in rot(group, f"{name}_by_flip_symbolic").points:
        assert np.allclose(matrix, sandwich(expm_hermitian(x, parameters["b3"])), **TOL)


def test_two_spin_rotated_operators():
    g = "two_half"
    x1 = op(g, "I1x")
    r1 = expm_hermitian(x1, np.pi / 2)
    assert np.allclose(rot(g, "R_spin1_only_x_90"), r1, **TOL)
    assert np.allclose(rot(g, "R_all_x_90"), rot(g, "R_x_90"), **TOL)
    assert np.allclose(rot(g, "I1zI2z_spin1_by_x_90"), r1 @ op(g, "I1zI2z") @ dagger(r1), **TOL)
    both = rot(g, "R_x_90")
    assert np.allclose(rot(g, "I1xI2z_both_by_x_90"), both @ op(g, "I1xI2z") @ dagger(both), **TOL)
    euler = rot(g, "R_euler")
    assert np.allclose(rot(g, "T21_both_by_euler"), euler @ op(g, "pair_T2_1") @ dagger(euler), **TOL)
    # a rank-2 tensor operator transforms with the Wigner matrix: R T2q R+ = sum_q' T2q' D2_q'q
    d2 = rot("wigner", "D_2")
    components = [op(g, f"pair_T2_{q}") for q in (2, 1, 0, -1, -2)]
    expected = sum(components[i] * d2[i, 1] for i in range(5))
    assert np.allclose(rot(g, "T21_both_by_euler"), expected, **TOL)


# ------------------------------------------------------------------------------------------- Liouville space
def lv(group, name):
    return fixtures.get("04_liouville", f"{group}/{name}")


def gram(basis):
    flat = basis.reshape(len(basis), -1)
    return flat.conj() @ flat.T


BASES = [
    "cartesian",
    "cartesian_sorted",
    "shift_and_polarization",
    "shift_and_polarization_sorted",
    "shift_and_z",
    "shift_and_z_default",
    "singlet_triplet",
    "singlet_triplet_unsorted",
    "zeeman_ket_bra",
    "zeeman_ket_bra_sorted",
    "zeeman_ket_bra_canonical",
    "spherical_tensor",
    "spherical_tensor_sorted",
]


@pytest.mark.parametrize("name", BASES)
def test_operator_bases_are_orthonormal_and_complete(name):
    basis = lv("two_half_bases", name)
    assert basis.shape == (16, 4, 4)
    assert np.allclose(gram(basis), np.eye(16), **TOL)


OTHER_BASES = [
    ("one_half", "default_basis_operators"),
    ("one_half", "cartesian"),
    ("one_half", "spherical_tensor"),
    ("spin1", "default_basis_operators"),
    ("spin1", "zeeman_ket_bra"),
    ("spin1", "spherical_tensor"),
    ("half_one", "default_basis_operators"),
    ("half_one", "spherical_tensor"),
]


@pytest.mark.parametrize("group, name", OTHER_BASES)
def test_operator_bases_of_other_spin_systems_are_orthonormal(group, name):
    basis = lv(group, name)
    assert np.allclose(gram(basis), np.eye(len(basis)), **TOL)
    assert len(basis) == basis.shape[1] ** 2


def test_operator_vectors_brackets_and_amplitudes():
    g = "two_half_vectors"
    x1, plus = op("two_half", "I1x"), op("two_half", "I1plus")
    cart = lv("two_half_bases", "cartesian")

    def components(basis, a):
        return np.einsum("kij,ij->k", basis.conj(), a)

    assert np.allclose(lv(g, "I1x_in_cartesian"), components(cart, x1), **TOL)
    assert np.allclose(lv(g, "I1xI2z_in_cartesian"), components(cart, op("two_half", "I1xI2z")), **TOL)
    assert np.allclose(lv(g, "I1x_in_default"), components(lv("two_half_bases", "shift_and_z_default"), x1), **TOL)
    # D1 in DIVERGENCES.md: the documented option form returns the vector in the default basis
    assert np.allclose(lv(g, "option_form_is_ignored"), lv(g, "I1x_in_default"), **TOL)
    assert not np.allclose(lv(g, "option_form_is_ignored"), lv(g, "I1x_in_cartesian"), **TOL)
    assert np.allclose(
        lv(g, "I1plus_in_zeeman_ket_bra"), components(lv("two_half_bases", "zeeman_ket_bra"), plus), **TOL
    )
    assert np.allclose(
        lv(g, "I1_dot_I2_in_spherical"),
        components(lv("two_half_bases", "spherical_tensor"), op("two_half", "I1_dot_I2")),
        **TOL,
    )

    def bracket(a, b):
        return np.trace(dagger(a) @ b)

    assert np.isclose(lv(g, "bracket_I1x_I1x"), bracket(x1, x1))
    assert np.isclose(lv(g, "bracket_I1plus_I1x"), bracket(plus, x1))
    assert np.isclose(lv(g, "bracket_I1x_I1plus"), bracket(x1, plus))
    y1, z1 = op("two_half", "I1y"), op("two_half", "I1z")
    assert np.isclose(lv(g, "bracket_with_superoperator"), bracket(y1, z1 @ x1 - x1 @ z1))
    assert np.isclose(lv(g, "amplitude_same"), 1.0) and np.isclose(lv(g, "amplitude_mixture"), 2.0)
    assert np.isclose(lv(g, "amplitude_scaled_target"), 0.5)
    assert np.isclose(lv(g, "amplitude_I1x_to_I1plus"), bracket(plus, x1) / bracket(plus, plus))
    assert np.isclose(lv(g, "amplitude_orthogonal"), 0.0)
    for parameters, value in lv(g, "amplitude_symbolic").points:
        assert np.isclose(value, -np.sin(parameters["b3"]))


def representation(basis, action):
    """``S_ij = (B_i | S B_j)`` for a superoperator given as a function of an operator matrix."""
    images = np.array([action(b) for b in basis])
    return np.einsum("iab,jab->ij", basis.conj(), images)


def superoperator_actions():
    g = "two_half"
    x1, z1, plus1, minus1 = (op(g, n) for n in ("I1x", "I1z", "I1plus", "I1minus"))
    z2, minus2 = op(g, "I2z"), op(g, "I2minus")
    ham = 2 * np.pi * (100 * z1 - 40 * z2 + 7 * op(g, "I1_dot_I2"))
    order = np.subtract.outer(*(np.diag(op(g, "Iz_total")),) * 2)
    order1 = np.subtract.outer(*(np.diag(z1),) * 2)
    rx = expm_hermitian(x1, np.pi / 2)
    euler = rot(g, "R_euler")
    p = op(g, "permutation_12")
    unit_z1 = z1 / np.linalg.norm(z1)

    def lindblad(a, b):
        return lambda r: a @ r @ b - 0.5 * (b @ a @ r + r @ b @ a)

    return {
        "unity": lambda r: r,
        "commutation_I1z": lambda r: z1 @ r - r @ z1,
        "commutation_I1x": lambda r: x1 @ r - r @ x1,
        "commutation_hamiltonian": lambda r: ham @ r - r @ ham,
        "left_I1x": lambda r: x1 @ r,
        "right_I1x": lambda r: r @ x1,
        "left_I1plus": lambda r: plus1 @ r,
        "right_I1plus": lambda r: r @ plus1,
        "double_commutation_I1z_I1z": lambda r: z1 @ (z1 @ r - r @ z1) - (z1 @ r - r @ z1) @ z1,
        "double_commutation_I1x_I2z": lambda r: x1 @ (z2 @ r - r @ z2) - (z2 @ r - r @ z2) @ x1,
        "lindbladian_I1minus": lindblad(minus1, plus1),
        "lindbladian_I1plus_I2minus": lindblad(plus1, minus2),
        "projection_I1z": lambda r: unit_z1 * np.trace(dagger(unit_z1) @ r),
        "filter_order_0": lambda r: np.where(order == 0, r, 0),
        "filter_orders_plus_minus_1": lambda r: np.where(np.abs(order) == 1, r, 0),
        "filter_spin1_order_plus_1": lambda r: np.where(order1 == 1, r, 0),
        "rotation_spin1_x_90": lambda r: rx @ r @ dagger(rx),
        "rotation_both_euler": lambda r: euler @ r @ dagger(euler),
        "permutation_12": lambda r: p @ r @ dagger(p),
    }


@pytest.mark.parametrize("group, basis", [("two_half_superoperators", "shift_and_z_default"),
                                          ("two_half_superoperators_cartesian", "cartesian")])  # fmt: skip
@pytest.mark.parametrize("name", list(superoperator_actions()))
def test_superoperator_matrices_follow_their_definitions(group, basis, name):
    operators = lv("two_half_bases", basis)
    expected = representation(operators, superoperator_actions()[name])
    assert np.allclose(lv(group, name), expected, rtol=0, atol=1e-10)


def test_superoperators_in_an_explicit_basis():
    g = "two_half_superoperators_explicit_basis"
    actions = superoperator_actions()
    assert np.allclose(
        lv(g, "commutation_I1z_in_zeeman_ket_bra"),
        representation(lv("two_half_bases", "zeeman_ket_bra"), actions["commutation_I1z"]),
        **TOL,
    )
    assert np.allclose(
        lv(g, "lindbladian_I1minus_in_spherical"),
        representation(lv("two_half_bases", "spherical_tensor"), actions["lindbladian_I1minus"]),
        **TOL,
    )
    assert np.allclose(
        lv(g, "commutation_hamiltonian_direct"), lv("two_half_superoperators", "commutation_hamiltonian"), atol=1e-9
    )


# ------------------------------------------------------------------------------------------- tables
def test_isotope_table():
    names = fixtures.get("07_tables", "isotopes/names")
    numbers = fixtures.get("07_tables", "isotopes/mass_gamma_spin_abundance")
    assert len(names) == len(numbers) and len(set(names)) == len(names)
    row = dict(zip(names, numbers, strict=True))
    assert np.isclose(row["1H"][1], 2.67522128e8) and row["1H"][2] == 0.5
    assert row["15N"][1] < 0 and row["2H"][2] == 1.0
    assert np.all((2 * numbers[:, 2]) % 1 == 0) and np.all((numbers[:, 3] >= 0) & (numbers[:, 3] <= 1))
    assert np.isclose(fixtures.get("07_tables", "isotopes/gamma_1H"), row["1H"][1])
    # the Larmor frequency is -gamma B0
    assert np.isclose(fixtures.get("07_tables", "isotopes/larmor_1H_9p4T"), -row["1H"][1] * 9.4)
    # the dipolar coupling is -(mu0/4 pi) gamma1 gamma2 hbar / r^3
    constant_names = fixtures.get("07_tables", "constants/names")
    constants = dict(zip(constant_names, fixtures.get("07_tables", "constants/values"), strict=True))
    hbar, mu0 = constants["ℏ"], constants["μ0"]
    expected = -(mu0 / (4 * np.pi)) * row["1H"][1] * row["13C"][1] * hbar / (1.1e-10) ** 3
    assert np.isclose(fixtures.get("07_tables", "isotopes/dipolar_1H_13C_1p1A"), expected, rtol=1e-9)


def test_orientation_sets():
    names = fixtures.get("07_tables", "orientations/tabulated_names")
    summaries = fixtures.get("07_tables", "orientations/tabulated_summaries")
    assert len(names) == len(summaries) == 23
    for name, summary in zip(names, summaries, strict=True):
        count = int("".join(ch for ch in name if ch.isdigit()))
        assert summary[0] == count, name
        assert np.isclose(summary[4], 1.0, atol=1e-6), f"the weights of {name} sum to {summary[4]}"
    for key in ("leboct10", "zcw50", "repulsion100", "polytope12", "step_beta_4", "step_alpha_beta_3_4"):
        table = fixtures.get("07_tables", f"orientations/{key}")
        assert table.shape[1] == 4 and np.isclose(table[:, 3].sum(), 1.0, atol=1e-6), key


# ------------------------------------------------------------------------------------------- propagation
def pr(group, name):
    return fixtures.get("05_propagation", f"{group}/{name}")


def two_spin_hamiltonians():
    g = "two_half"
    z1, z2 = op(g, "I1z"), op(g, "I2z")
    strong = 2 * np.pi * (100 * z1 - 40 * z2 + 7 * op(g, "I1_dot_I2"))
    weak = 2 * np.pi * (100 * z1 - 40 * z2 + 7 * op(g, "I1zI2z"))
    return strong, weak


def evolve(rho, h, t):
    u = expm_hermitian(h, t)
    return u @ rho @ dagger(u)


def amplitudes(rho, targets):
    return np.array([np.trace(dagger(b) @ rho) / np.trace(dagger(b) @ b) for b in targets])


def two_spin_targets():
    g = "two_half"
    x, y, z = op(g, "I1x"), op(g, "I1y"), op(g, "I1z")
    return [x, y, z, op(g, "I1xI2z"), y @ op(g, "I2z")]


def test_free_evolution_is_exp_minus_i_h_t():
    strong, weak = two_spin_hamiltonians()
    x1 = op("two_half", "I1x")
    assert np.allclose(pr("two_half", "propagator_strong_10ms"), expm_hermitian(strong, 0.01), atol=1e-10)
    assert np.allclose(pr("two_half", "free_strong_I1x_10ms"), evolve(x1, strong, 0.01), atol=1e-10)
    assert np.allclose(pr("two_half", "free_weak_I1x_10ms"), evolve(x1, weak, 0.01), atol=1e-10)


def test_event_sequences_and_chronology():
    g = "two_half"
    strong, weak = two_spin_hamiltonians()
    x, y, z1 = op(g, "Ix_total"), op(g, "I1y") + op(g, "I2y"), op(g, "I1z")
    rho = evolve(z1, x, np.pi / 2)
    assert np.allclose(pr(g, "rotation_then_free"), evolve(rho, strong, 0.004), atol=1e-10)
    echo = evolve(evolve(evolve(rho, weak, 0.004), y, np.pi), weak, 0.004)
    assert np.allclose(pr(g, "spin_echo"), echo, atol=1e-10)
    forward = evolve(evolve(rho, strong, 0.002), weak, 0.005)
    assert np.allclose(pr(g, "two_different_delays"), forward, atol=1e-10)
    u_forward = expm_hermitian(weak, 0.005) @ expm_hermitian(strong, 0.002)
    assert np.allclose(pr(g, "propagator_two_delays"), u_forward, atol=1e-10)
    targets = two_spin_targets()
    after = evolve(rho, weak, 0.004)
    assert np.allclose(pr(g, "amplitudes_after_rotation_and_delay"), amplitudes(after, targets), atol=1e-10)
    assert np.allclose(pr(g, "amplitudes_with_background"), amplitudes(after, targets), atol=1e-10)
    assert np.isclose(pr(g, "amplitude_spin_echo"), amplitudes(echo, [op(g, "I1y")])[0], atol=1e-10)


def test_chronology_follows_the_global_setting_only():
    """D2 in DIVERGENCES.md: the option Chronology is ignored, $Chronology is what counts."""
    g, rl = "two_half", "two_half_right_to_left"
    strong, weak = two_spin_hamiltonians()
    x, y, z1 = op(g, "Ix_total"), op(g, "I1y") + op(g, "I2y"), op(g, "I1z")
    # right to left: the last event of the list acts first
    backward = evolve(evolve(evolve(z1, weak, 0.005), strong, 0.002), x, np.pi / 2)
    assert np.allclose(pr(rl, "two_different_delays"), backward, atol=1e-10)
    rho = evolve(z1, x, np.pi / 2)
    echo = evolve(evolve(evolve(rho, weak, 0.004), y, np.pi), weak, 0.004)
    assert np.allclose(pr(rl, "spin_echo"), echo, atol=1e-10)
    u_backward = expm_hermitian(strong, 0.002) @ expm_hermitian(weak, 0.005)
    assert np.allclose(pr(rl, "propagator_two_delays"), u_backward, atol=1e-10)
    assert np.isclose(pr(rl, "amplitude"), amplitudes(backward, [op(g, "I1y")])[0], atol=1e-10)
    # the option changes nothing, in either direction
    assert np.allclose(pr(g, "chronology_option_is_ignored"), pr(g, "two_different_delays"), atol=1e-12)
    assert np.allclose(pr(g, "chronology_option_is_ignored_propagator"), pr(g, "propagator_two_delays"), atol=1e-12)
    forward = evolve(evolve(rho, strong, 0.002), weak, 0.005)
    assert np.isclose(pr(g, "chronology_option_is_ignored_amplitude"), amplitudes(forward, [op(g, "I1y")])[0])
    assert np.allclose(pr(rl, "option_left_to_right_is_ignored"), pr(rl, "two_different_delays"), atol=1e-12)
    assert not np.allclose(backward, forward, atol=1e-3)


def test_phenomenological_relaxation():
    g = "two_half_relaxation"
    basis = lv("two_half_bases", "shift_and_z_default")
    z = {1: op("two_half", "I1z"), 2: op("two_half", "I2z")}

    r = pr(g, "relaxation_superoperator")
    assert np.allclose(r, r.T.conj(), atol=1e-12) and np.all(np.linalg.eigvalsh(r) <= 1e-12)
    rates = np.sort(-np.linalg.eigvalsh(r))
    assert np.isclose(rates[0], 0.0), "the unity operator does not relax"

    def rate(a):
        """The decay rate of an operator that the superoperator maps onto itself."""
        vec = np.einsum("kij,ij->k", basis.conj(), a)
        return -(vec.conj() @ r @ vec) / (vec.conj() @ vec)

    # longitudinal magnetisation decays at 1/T1 and transverse magnetisation at 1/T2
    assert np.isclose(rate(z[1]), 1 / 2.0) and np.isclose(rate(op("two_half", "I1x")), 1 / 0.2)
    single = pr(g, "relaxation_superoperator_single_time")
    vec = np.einsum("kij,ij->k", basis.conj(), z[1])
    assert np.isclose(-(vec.conj() @ single @ vec) / (vec.conj() @ vec), 1 / 0.5)
    assert np.allclose(pr(g, "decay_I1z_500ms"), np.exp(-0.5 / 2.0) * z[1], atol=1e-9)
    strong, weak = two_spin_hamiltonians()
    commutator = representation(basis, lambda a: strong @ a - a @ strong)
    assert np.allclose(pr(g, "liouvillian"), -1j * commutator + r, atol=1e-9)


def test_pulses_on_one_spin():
    g = "one_half_pulses"
    x, y, z = (op("one_half", f"I1{c}") for c in "xyz")
    targets = [x, y, z]
    assert np.allclose(pr(g, "pulse_90x_on_Iz"), amplitudes(evolve(z, x, np.pi / 2), targets), atol=1e-10)
    assert np.allclose(pr(g, "pulse_90y_on_Iz"), amplitudes(evolve(z, y, np.pi / 2), targets), atol=1e-10)
    axis = np.cos(np.pi / 4) * x + np.sin(np.pi / 4) * y
    assert np.allclose(pr(g, "pulse_90_phase_pi4_on_Iz"), amplitudes(evolve(z, axis, np.pi / 2), targets), atol=1e-10)
    offset, nutation = 2 * np.pi * 300, 2 * np.pi * 1000
    finite = evolve(z, offset * z + nutation * x, (np.pi / 2) / nutation)
    assert np.allclose(pr(g, "pulse_finite_off_resonance"), amplitudes(finite, targets), atol=1e-9)
    rho = evolve(z, x, np.pi / 2)
    rho = evolve(rho, offset * z, 0.001)
    rho = evolve(rho, y, np.pi)
    rho = evolve(rho, offset * z, 0.001)
    assert np.allclose(pr(g, "pulse_delay_sequence"), amplitudes(rho, targets), atol=1e-9)
    assert np.allclose(pr(g, "repeated_pulse"), amplitudes(evolve(z, x, np.pi / 2), targets), atol=1e-10)
    for parameters, value in pr(g, "symbolic_flip_angle").points:
        a3, b3 = parameters["a3"], parameters["b3"]
        expected = amplitudes(evolve(z, np.cos(a3) * x + np.sin(a3) * y, b3), targets)
        assert np.allclose(value, expected, atol=1e-12)
