"""Coupled tensor operators, the spherical component filter, user-defined and product bases, and geometry."""

import warnings

import numpy as np
import pytest
import scipy.linalg
import sympy as sp

import spyndynamica as sd
from spyndynamica import SpinSystem
from tests.spyndynamica import reference

FIXTURE = "12_bases_filters_geometry"
TOL = dict(rtol=0, atol=1e-10)
HALF = sp.Rational(1, 2)


def stored(key):
    return reference.get(FIXTURE, key)


def number(matrix):
    return np.array(sp.N(matrix, 30).tolist(), dtype=complex)


def dense(rows, count, dim):
    """The operators stored as rows ``(operator, row, column, value)``."""
    out = np.zeros((count, dim, dim), dtype=complex)
    for n, row, column, value in np.asarray(rows):
        out[int(n.real) - 1, int(row.real) - 1, int(column.real) - 1] = value
    return out


def commutator(a, b):
    return a @ b - b @ a


# ------------------------------------------------------------------------------------------- tensor operators
TENSOR_SYSTEMS = {
    "three_half_tensors": lambda: SpinSystem(3),
    "half_one_half_tensors": lambda: SpinSystem([HALF, 1, HALF]),
}


@pytest.mark.parametrize("group", list(TENSOR_SYSTEMS))
def test_spherical_tensor_basis_of_three_spins(group):
    s = TENSOR_SYSTEMS[group]()
    basis = sd.spherical_tensor_basis(s)
    mine = np.array([op.numeric() for op in basis.operators])
    assert np.allclose(mine, dense(stored(f"{group}/spherical_tensor"), len(basis), s.dimension), **TOL)
    assert np.allclose(np.array(basis.indices, dtype=float), stored(f"{group}/spherical_tensor_indices"), **TOL)
    # independent of SpinDynamica: orthonormal and complete, and every operator has its rank and its order
    assert basis.complete and basis.check() == "orthonormal"
    jx, jy, jz = (s.I(None, c).numeric() for c in "xyz")
    for (rank, order), t in zip(basis.indices, mine):
        assert np.allclose(commutator(jz, t), order * t, **TOL)
        casimir = sum(commutator(j, commutator(j, t)) for j in (jx, jy, jz))
        assert np.allclose(casimir, rank * (rank + 1) * t, **TOL)
    by_rank = sd.spherical_tensor_basis(s, sort="rank")
    assert by_rank.indices == sorted(by_rank.indices) and sorted(by_rank.indices) == sorted(basis.indices)
    with pytest.raises(ValueError, match="sort"):
        sd.spherical_tensor_basis(s, sort="other")


def test_sort_values_of_the_tensor_basis_in_spindynamica():
    # D8 in DIVERGENCES.md: the documented order by rank is rejected, and every accepted value sorts alike
    assert stored("tensor_basis_sorting/rank_then_order_is_rejected") == "True"
    assert stored("tensor_basis_sorting/spherical_rank_same_as_default") == "True"
    assert stored("tensor_basis_sorting/unsorted_same_as_default") == "True"


def test_coupled_tensor_operators():
    s = SpinSystem(3)
    t = s.T
    group = "three_half_tensors"
    cases = {
        "T_history_1_1_1_component_0": t([1, 2, 3], 1, 0, history=(1, 1)),
        "T_history_1_0_1_component_1": t([1, 2, 3], 1, 1, history=(0, 1)),
        "T_history_1_2_1_component_m1": t([1, 2, 3], 1, -1, history=(2, 1)),
        "T_history_1_2_2_component_1": t([1, 2, 3], 2, 1, history=(2, 2)),
        "T_history_1_1_0_component_0": t([1, 2, 3], 0, 0, history=(1, 0)),
        "T_rank_3_component_2": t([1, 2, 3], 3, 2),
        "T_rank_1_component_0_index_2": t([1, 2, 3], 1, 0, history=(1, 1)),
        "T_spins_1_3_rank_2_component_0": t([1, 3], 2, 0),
    }
    for name, operator in cases.items():
        assert np.allclose(operator.numeric(), stored(f"{group}/{name}"), **TOL), name
    assert sd.rank_histories([1, 1, 1]) == [(0, 1), (1, 0), (1, 1), (1, 2), (2, 1), (2, 2), (2, 3)]
    for rank, order, name in [(1, 0, "T_rank_1_component_0_all"), (2, 1, "T_rank_2_component_1_all")]:
        histories = [h for h in sd.rank_histories([1, 1, 1]) if h[-1] == rank]
        theirs = stored(f"{group}/{name}")
        assert len(histories) == len(theirs)
        for history, matrix in zip(histories, theirs):
            assert np.allclose(t([1, 2, 3], rank, order, history=history).numeric(), matrix, **TOL)
    mixed = SpinSystem([HALF, 1, HALF])
    g = "half_one_half_tensors"
    op = mixed.T([1, 2, 3], 3, -2, ranks=(1, 2, 1), history=(2, 3))
    assert np.allclose(op.numeric(), stored(f"{g}/T_ranks_1_2_1_history_1_2_3"), **TOL)
    op = mixed.T([1, 2, 3], 2, 1, ranks=(1, 2, 1), history=(3, 2))
    assert np.allclose(op.numeric(), stored(f"{g}/T_ranks_1_2_1_history_1_3_2"), **TOL)
    # the pair operators of before are the same special case
    pair = SpinSystem(2)
    assert pair.T([1, 2], 0, 0).equals(-pair.dot(1, 2) / sp.sqrt(3))
    assert pair.T([2, 1], 2, 2).equals(pair.I(1, "+") @ pair.I(2, "+") / 2)
    assert s.T([2], 1, 0).equals(s.I(2, "z"))


@pytest.mark.parametrize("history", [(0, 1), (1, 1), (2, 1), (1, 2), (2, 2), (2, 3), (1, 0)])
def test_coupled_tensor_operators_obey_the_racah_commutators(history):
    s = SpinSystem(3)
    rank = history[-1]
    jz, jplus, jminus = (s.I(None, c) for c in ("z", "+", "-"))
    for q in range(rank, -rank - 1, -1):
        t = s.T([1, 2, 3], rank, q, history=history)
        assert (jz @ t - t @ jz).equals(q * t)
        for step, ladder in ((1, jplus), (-1, jminus)):
            image = ladder @ t - t @ ladder
            if abs(q + step) > rank:
                assert not image.simplify()
            else:
                factor = sp.sqrt(rank * (rank + 1) - q * (q + step))
                assert image.equals(factor * s.T([1, 2, 3], rank, q + step, history=history))


def test_coupled_tensor_operator_arguments():
    s = SpinSystem(3)
    with pytest.raises(ValueError, match="3 ways"):
        s.T([1, 2, 3], 1, 0)
    with pytest.raises(ValueError, match="cannot be coupled"):
        s.T([1, 2, 3], 4, 0)
    with pytest.raises(ValueError, match="not a sequence"):
        s.T([1, 2, 3], 1, 0, history=(3, 1))
    with pytest.raises(ValueError, match="distinct"):
        s.T([1, 1], 1, 0)
    with pytest.raises(ValueError, match="list of spins"):
        s.T(1, 1, 0, history=(1,))
    with pytest.raises(ValueError, match="exceeds"):
        s.T([1, 2], 1, 2)


# ------------------------------------------------------------------------------------------- spherical filter
def spin_matrices(j):
    m = np.arange(j, -j - 1, -1)
    plus = np.diag(np.sqrt(j * (j + 1) - m[1:] * (m[1:] + 1)), 1)
    return (plus + plus.T) / 2, (plus - plus.T) / 2j, np.diag(m)


def projection_by_integration(s, rank, to, source, nodes=8):
    """``(2λ + 1)/(8π²) ∫ D*(to, source; Ω) R(Ω) dΩ`` on row-flattened matrices, by exact quadrature."""
    jy, jz = s.I(None, "y").numeric(), np.diag(s.I(None, "z").numeric()).real
    ly = spin_matrices(rank)[1]
    x, weights = np.polynomial.legendre.leggauss(nodes)
    angles = 2 * np.pi * np.arange(nodes) / nodes
    out = np.zeros((s.dimension**2, s.dimension**2), dtype=complex)
    for cosine, weight in zip(x, weights):
        beta = np.arccos(cosine)
        middle = scipy.linalg.expm(-1j * beta * jy)
        small = scipy.linalg.expm(-1j * beta * ly)[rank - to, rank - source]
        for alpha in angles:
            for gamma in angles:
                r = np.exp(-1j * alpha * jz)[:, None] * middle * np.exp(-1j * gamma * jz)[None, :]
                wigner = np.exp(-1j * to * alpha) * small * np.exp(-1j * source * gamma)
                out += weight * np.conj(wigner) * np.kron(r, r.conj())
    return out * (2 * rank + 1) / (2 * nodes**2)


def filter_spec(name):
    return [int(part.replace("m", "-")) for part in name.split("_")[1:]]


def generic(s):
    i = s.I
    if len(s.spins) == 2:
        return (
            3 * i(1, "z") + 2 * i(2, "x") - i(2, "y") + i(1, "x") @ i(2, "z") + 5 * i(1, "y") @ i(2, "y")
            + 4 * i(1, "+") @ i(2, "+") + 7 * i(1, "z") @ i(2, "-") + 2 * s.unity() + i(1, "x") @ i(2, "y")
        )  # fmt: skip
    return (
        i(1, "z") + 2 * i(2, "x") - 3 * i(3, "y") + i(1, "x") @ i(2, "z") + 5 * i(1, "y") @ i(3, "y")
        + 4 * i(2, "+") @ i(3, "+") + 7 * i(1, "z") @ i(2, "-") @ i(3, "x") + 2 * i(1, "+") @ i(2, "+") @ i(3, "-")
        + 6 * i(1, "x") @ i(2, "y") @ i(3, "z") + i(1, "+") @ i(2, "+") @ i(3, "+") + 3 * i(2, "z") @ i(3, "x")
    )  # fmt: skip


def applied_cases(group):
    return [k.split("/")[1] for k in reference.cases(FIXTURE) if k.startswith(f"{group}/applied")]


def test_spherical_component_filter_of_two_spins():
    s = SpinSystem(2)
    cartesian = sd.cartesian_product_basis(s)
    for name in ("matrix_1_0_0", "matrix_2_1_m1", "matrix_0_0_0"):
        mine = sd.spherical_component_filter(s, *filter_spec(name)).numeric(basis=cartesian)
        assert np.allclose(mine, stored(f"two_half_filter/{name}"), **TOL)
    rho = generic(s)
    for name in applied_cases("two_half_filter"):
        spec = filter_spec(name)
        f = sd.spherical_component_filter(s, *spec)
        assert np.allclose(f(rho).numeric(), stored(f"two_half_filter/{name}"), **TOL), name
        assert np.allclose(f.numeric(), projection_by_integration(s, *spec), **TOL), name
    assert stored("two_half_filter/rank_3_is_null") == "NullSuperoperator[]"
    assert not sd.spherical_component_filter(s, 3, 0).dyads
    # the projectors onto all components of all ranks add up to unity
    total = sum(
        sd.spherical_component_filter(s, rank, q).numeric() for rank in range(3) for q in range(-rank, rank + 1)
    )
    assert np.allclose(total, np.eye(16), **TOL)
    antisymmetric = (s.I(1, "x") @ s.I(2, "y") - s.I(1, "y") @ s.I(2, "x")) / 2
    assert sd.spherical_component_filter(s, 1, 0)(rho).equals(3 * s.I(1, "z") + antisymmetric)
    with pytest.raises(ValueError, match="do not belong"):
        sd.spherical_component_filter(s, 1, 2)


# D9 in DIVERGENCES.md: SpinDynamica pairs the wrong tensors when the two components differ and several
# tensors have the rank
MISPAIRED = {"applied_1_1_0", "applied_2_m2_2", "applied_2_0_1"}


def test_spherical_component_filter_of_three_spins():
    s = SpinSystem(3)
    rho = generic(s)
    flat = rho.numeric().reshape(-1)
    names = applied_cases("three_half_filter")
    assert MISPAIRED < set(names)
    for name in names:
        spec = filter_spec(name)
        f = sd.spherical_component_filter(s, *spec)
        integral = projection_by_integration(s, *spec)
        assert np.allclose(f.numeric(), integral, **TOL), name
        mine = f(rho).numeric()
        assert np.allclose(mine.reshape(-1), integral @ flat, **TOL), name
        theirs = stored(f"three_half_filter/{name}")
        assert np.allclose(mine, theirs, **TOL) != (name in MISPAIRED), name


def test_spherical_component_filter_of_other_spins():
    # SpinDynamica has the filter for spins-1/2 only; the port's holds for any spins
    s = SpinSystem([1, HALF])
    for spec in [(0, 0, 0), (1, 1, -1), (2, 0, 1), (3, -2, 3), (2, 2, 2)]:
        f = sd.spherical_component_filter(s, *spec)
        assert np.allclose(f.numeric(), projection_by_integration(s, *spec), **TOL), spec
    assert sd.spherical_component_filter(s, 2, 1).numeric().shape == (36, 36)


# ------------------------------------------------------------------------------------------- state bases
def test_defined_basis():
    s = SpinSystem(2)
    up, down = HALF, -HALF
    k = s.ket
    kets = [
        k(up, up),
        (k(up, down) + k(down, up)) / sp.sqrt(2),
        k(down, down),
        (k(up, down) - k(down, up)) / sp.sqrt(2),
    ]
    basis = sd.define_basis(kets, labels=["a", "b", "c", "d"])
    g = "defined_bases"
    assert stored(f"{g}/labels") == ["a", "b", "c", "d"] and basis.names == ("a", "b", "c", "d")
    assert np.allclose(number(basis.kets_matrix).T, stored(f"{g}/kets"), **TOL)
    assert np.allclose(s.I(1, "z").numeric(basis=basis), stored(f"{g}/I1z_in_defined_basis"), **TOL)
    assert np.allclose(s.I(1, "x").numeric(basis=basis), stored(f"{g}/I1x_in_defined_basis"), **TOL)
    assert basis.ket("d").vector == kets[3].vector and basis.ket(2).vector == kets[1].vector
    zeeman = [k(up, up), k(down, up), k(up, down), k(down, down)]
    checks = {
        "check_orthonormal": zeeman,
        "check_same_norm": [2 * ket for ket in zeeman],
        "check_different_norms": [zeeman[0], 2 * zeeman[1], zeeman[2], zeeman[3]],
        "check_not_orthogonal": [zeeman[0], zeeman[2] + zeeman[1], zeeman[2], zeeman[3]],
        "check_incomplete_orthonormal": [zeeman[0], zeeman[3]],
    }
    expected = {
        "check_orthonormal": "orthonormal",
        "check_same_norm": "orthogonal, same norm",
        "check_different_norms": "orthogonal, different norms",
        "check_not_orthogonal": "not orthogonal",
        "check_incomplete_orthonormal": "orthonormal",
    }
    for name, candidates in checks.items():
        found = sd.check_basis(candidates)
        assert found == expected[name]
        assert (found == "orthonormal") == (stored(f"{g}/{name}") == "True"), name
    assert stored(f"{g}/defective_is_not_defined") == "True"
    with pytest.raises(ValueError, match="different norms"):
        sd.define_basis(checks["check_different_norms"])
    unchecked = sd.define_basis(checks["check_different_norms"], check=False)
    assert unchecked.kets_matrix[1, 1] == 2
    with pytest.raises(ValueError, match="4 kets"):
        sd.define_basis(checks["check_incomplete_orthonormal"])
    with pytest.raises(ValueError, match="label"):
        sd.define_basis(zeeman, labels=["a"])


def test_product_basis():
    s = SpinSystem(3)
    pair, third = SpinSystem([(1, HALF), (2, HALF)]), SpinSystem([(3, HALF)])
    g = "product_bases"
    st_z = sd.product_basis(pair.singlet_triplet(), third.zeeman, system=s)
    assert np.allclose(number(st_z.kets_matrix).T, stored(f"{g}/singlet_triplet_12_zeeman_3"), **TOL)
    z_st = sd.product_basis(third.zeeman, pair.singlet_triplet(), system=s)
    assert np.allclose(number(z_st.kets_matrix).T, stored(f"{g}/zeeman_3_singlet_triplet_12"), **TOL)
    outer = SpinSystem([(1, HALF), (3, HALF)])
    split = sd.product_basis(outer.singlet_triplet(), SpinSystem([(2, HALF)]).zeeman, system=s)
    assert np.allclose(number(split.kets_matrix).T, stored(f"{g}/singlet_triplet_13_zeeman_2"), **TOL)
    singles = [SpinSystem([(n, HALF)]).zeeman for n in (2, 3, 1)]
    assert np.allclose(
        number(sd.product_basis(*singles, system=s).kets_matrix).T, stored(f"{g}/three_zeeman_bases"), **TOL
    )
    assert np.allclose(s.I(3, "z").numeric(basis=st_z), stored(f"{g}/I3z_in_product_basis"), **TOL)
    assert np.allclose(s.dot(1, 2).numeric(basis=st_z), stored(f"{g}/I1I2_in_product_basis"), **TOL)
    # independent: the kets are Kronecker products, with the last spin as the slowest index
    kron = np.kron(np.eye(2), number(pair.singlet_triplet().kets_matrix))
    assert np.allclose(number(st_z.kets_matrix), kron, **TOL)
    assert np.allclose(s.dot(1, 2).numeric(basis=st_z), np.diag([-0.75, 0.25, 0.25, 0.25] * 2), **TOL)
    # without a spin system the spins are joined in the order of the bases
    joined = sd.product_basis(third.zeeman, pair.singlet_triplet())
    assert [label for label, _ in joined.system.spins] == [3, 1, 2]
    assert np.allclose(
        number(joined.kets_matrix), np.kron(number(pair.singlet_triplet().kets_matrix), np.eye(2)), **TOL
    )
    with pytest.raises(ValueError, match="share"):
        sd.product_basis(pair.zeeman, pair.zeeman)
    with pytest.raises(ValueError, match="are not the spins"):
        sd.product_basis(pair.zeeman, system=s)


# ------------------------------------------------------------------------------------------- operator bases
def test_defined_operator_bases():
    s = SpinSystem(2)
    i = s.I
    cartesian = sd.cartesian_product_basis(s)
    g = "defined_operator_bases"

    def matrices(basis):
        return np.array([op.numeric() for op in basis.operators])

    reverse = sd.define_operator_basis(cartesian.operators[::-1])
    assert np.allclose(matrices(reverse), stored(f"{g}/from_operators"), **TOL)
    commutation = sd.commutation(i(1, "x")).numeric(basis=reverse)
    assert np.allclose(commutation, stored(f"{g}/commutation_in_basis_from_operators"), **TOL)
    order = [15, *range(15)]
    reordered = cartesian.reordered(order)
    assert np.allclose(matrices(reordered), stored(f"{g}/reordered"), **TOL)
    turned = cartesian.transformed(sd.rotation_superoperator(s, sp.pi / 2, "x"))
    assert np.allclose(matrices(turned), stored(f"{g}/transformed"), **TOL)
    both = cartesian.transformed(sd.rotation_superoperator(s, sp.pi / 3, "y", spins=[1]), order)
    assert np.allclose(matrices(both), stored(f"{g}/transformed_reordered"), **TOL)
    # D10 in DIVERGENCES.md: with an ordering SpinDynamica stores the transformation matrix of the inverse
    # ordering, so its vectors and superoperator matrices do not belong to its own basis operators
    theirs = stored(f"{g}/transformed_reordered")
    operator = i(1, "z") + 2 * i(1, "x") @ i(2, "y")
    brackets = np.einsum("iab,jab->ij", theirs.conj(), matrices(cartesian))
    vector = np.einsum("iab,ab->i", theirs.conj(), operator.numeric())
    z = i(1, "z").numeric()
    commutation = np.array([[np.vdot(a, z @ b - b @ z) for b in theirs] for a in theirs])
    assert np.allclose(number(both.transformation(cartesian)), brackets, **TOL)
    assert np.allclose(number(both.vector(operator)).reshape(-1), vector, **TOL)
    assert np.allclose(sd.commutation(i(1, "z")).numeric(basis=both), commutation, **TOL)
    inverse = np.argsort(order)
    assert not np.allclose(stored(f"{g}/transformation_matrix"), brackets, **TOL)
    assert np.allclose(stored(f"{g}/transformation_matrix"), brackets[inverse][inverse], **TOL)
    assert np.allclose(stored(f"{g}/vector_in_transformed_reordered"), vector[inverse][inverse], **TOL)
    their_matrix = stored(f"{g}/commutation_in_transformed_reordered")
    assert np.allclose(their_matrix, commutation[np.ix_(inverse[inverse], inverse[inverse])], **TOL)
    assert both.express(both.vector(operator)).equals(operator)
    assert both.check() == "orthonormal" and both.complete
    assert stored(f"{g}/non_unitary_is_rejected") == "True"
    with pytest.raises(ValueError, match="not unitary"):
        cartesian.transformed(sd.commutation(i(1, "z")))
    assert len(cartesian.transformed(sd.commutation(i(1, "z")), check=False)) == 16
    with pytest.raises(ValueError, match="each of them once"):
        cartesian.reordered([0, 0, 1])
    checks = {
        "check_orthonormal": (list(cartesian.operators), "orthonormal"),
        "check_same_norm": ([2 * op for op in cartesian.operators], "orthogonal, same norm"),
        "check_different_norms": ([i(1, "z"), 2 * i(2, "z")], "orthogonal, different norms"),
        "check_not_orthogonal": ([i(1, "z"), i(1, "z") + i(2, "z")], "not orthogonal"),
        "check_incomplete_orthonormal": ([i(1, "z"), i(2, "z")], "orthonormal"),
    }
    for name, (operators, expected) in checks.items():
        found = sd.check_operator_basis(operators)
        assert found == expected
        assert (found == "orthonormal") == (stored(f"{g}/{name}") == "True"), name
    with pytest.raises(ValueError, match="same norm"):
        sd.define_operator_basis(checks["check_same_norm"][0])
    with pytest.warns(UserWarning, match="incomplete"):
        partial = sd.define_operator_basis([i(1, "z"), i(2, "z")])
    assert not partial.complete and len(partial) == 2
    assert np.allclose(number(partial.vector(3 * i(2, "z") + i(1, "x"))).reshape(-1), [0, 3], **TOL)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sd.define_operator_basis(cartesian.operators)


# ------------------------------------------------------------------------------------------- geometry
ANGLES = (0.3, 0.7, -0.4)
QA, QB, QC = (sd.polar_to_quaternion(*a) for a in ((0.3, 0.7, -0.4), (1.3, 0.2, 2.4), (0.5, 1.7, 0.4)))


def euler(angles):
    return np.array(sd.euler_matrix(*angles).evalf(30).tolist(), dtype=float)


def rodrigues(xi, axis):
    n = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
    cross = np.array([[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]])
    return np.eye(3) + np.sin(xi) * cross + (1 - np.cos(xi)) * cross @ cross


def geometry(name):
    return stored(f"geometry/{name}")


def test_axis_systems():
    cases = {
        "axis_system_z": ([1, 1, 1],),
        "axis_system_x_z": ([1, 2, 0], [1, 1, 1]),
        "axis_system_x_y_z": ([0, 2, 0], [-3, 0, 0], [0, 0, 1]),
        "axis_system_z_along_x": ([2, 0, 0],),
        "axis_system_z_against_x": ([-2, 0, 0],),
        "axis_system_x_parallel_to_z": ([1, 1, 1], [2, 2, 2]),
        "axis_system_default": (),
    }
    for name, axes in cases.items():
        system = sd.axis_system(*axes)
        assert np.allclose(system, geometry(name), **TOL), name
        assert np.allclose(system @ system.T, np.eye(3), **TOL) and np.isclose(np.linalg.det(system), 1)
        if axes:
            z = np.asarray(axes[-1], dtype=float)
            assert np.allclose(system[2], z / np.linalg.norm(z), **TOL)
    x, z = np.array([1.0, 2.0, 0.0]), np.array([1.0, 1.0, 1.0])
    system = sd.axis_system(x, z)
    assert np.isclose(np.linalg.det([system[0], x, z]), 0) and system[0] @ x > 0  # x axis in the plane of x and z
    assert np.allclose(sd.axis_system([x, z]), system, **TOL)
    assert np.allclose(sd.axis_system([0, 0, 0]), np.eye(3))
    with pytest.raises(ValueError, match="right-handed"):
        sd.axis_system([1, 0, 0], [0, 1, 0], [0, 0, -1])
    with pytest.raises(ValueError, match="one, two or three"):
        sd.axis_system(x, x, x, x)


def test_axes_to_euler_and_rotate_euler():
    a = sd.axis_system([1, 2, 0], [1, 1, 1])
    b = sd.axis_system([0, 1, 1], [-1, 2, 0.5])
    assert np.allclose(b, geometry("system_B"), **TOL)
    cases = {
        "axes_to_euler_z": ([[1, 1, 1]], None),
        "axes_to_euler_system": (a, None),
        "axes_to_euler_x_z": ([[1, 2, 0], [1, 1, 1]], None),
        "axes_to_euler_two_systems": (a, b),
        "axes_to_euler_two_systems_reversed": (b, a),
        "axes_to_euler_z_parallel": ([[0, 1, 0], [0, 0, 1]], None),
        "axes_to_euler_z_antiparallel": ([[0, 1, 0], [0, 0, -1]], None),
    }
    for name, (system2, system1) in cases.items():
        angles = sd.axes_to_euler(system2, system1)
        assert np.allclose(angles, geometry(name), **TOL), name
        # independent: the rotation about the axes of system 1 brings every axis of system 2 onto its partner
        first = np.eye(3) if system1 is None else sd.axis_system(system1)
        second = sd.axis_system(system2)
        about_first = first.T @ euler(angles) @ first
        assert np.allclose(second @ about_first.T, first, **TOL), name
        about_second = second.T @ euler(angles) @ second
        assert np.allclose(second @ about_second.T, first, **TOL), name
        assert all(-np.pi <= angle < np.pi for angle in angles)
    assert np.allclose(sd.axes_to_euler([1, 1, 1]), geometry("axes_to_euler_z"), **TOL)
    assert sd.axes_to_euler() == (0.0, 0.0, 0.0)

    vectors = [[1, 2, 3], [0, 0, 1]]
    assert np.allclose(sd.rotate_euler(vectors, ANGLES), geometry("rotate_euler_list"), **TOL)
    assert np.allclose(sd.rotate_euler([1, 2, 3], ANGLES), euler(ANGLES) @ [1, 2, 3], **TOL)
    # D7 in DIVERGENCES.md: about an axis system SpinDynamica applies the inverse of the documented rotation
    omega = sd.axes_to_euler(a, b)
    assert np.allclose(sd.rotate_euler(a, omega, a), b, **TOL)
    assert not np.allclose(geometry("rotate_euler_superimpose_fails"), b, atol=0.1)
    documented = a.T @ euler(ANGLES) @ a
    theirs = a @ euler(ANGLES) @ a.T
    assert np.allclose(sd.rotate_euler([1, 2, 3], ANGLES, a), documented @ [1, 2, 3], **TOL)
    assert np.allclose(geometry("rotate_euler_about_axes"), theirs @ [1, 2, 3], **TOL)
    assert np.allclose(geometry("rotate_euler_superimpose_fails"), a @ (a @ euler(omega) @ a.T).T, **TOL)
    only_z = sd.axis_system([1, 1, 1])
    assert np.allclose(geometry("rotate_euler_about_z_only"), only_z @ euler(ANGLES) @ only_z.T @ [1, 2, 3], **TOL)
    # the documented rotation: γ about the z axis of the system, then β about its y axis, then α about its z axis
    steps = rodrigues(ANGLES[0], a[2]) @ rodrigues(ANGLES[1], a[1]) @ rodrigues(ANGLES[2], a[2])
    assert np.allclose(documented, steps, **TOL)
    assert np.allclose(
        sd.rotate_euler([1, 2, 3], ANGLES, [[1, 1, 1]]), only_z.T @ euler(ANGLES) @ only_z @ [1, 2, 3], **TOL
    )
    with pytest.raises(ValueError, match="three components"):
        sd.rotate_euler([1, 2], ANGLES)


def test_polar_rotations():
    matrix = np.array(sd.polar_rotation_matrix(*ANGLES).evalf(30).tolist(), dtype=float)
    assert np.allclose(matrix, geometry("polar_rotation_matrix"), **TOL)
    xi, theta, phi = ANGLES
    axis = sd.polar_to_vector(1, (theta, phi))
    assert np.allclose(matrix, rodrigues(xi, axis), **TOL) and np.allclose(matrix @ axis, axis, **TOL)
    symbolic = geometry("polar_rotation_matrix_symbolic")
    for parameters, value in symbolic.points:
        at_point = sd.polar_rotation_matrix(*(parameters[name] for name in ("a3", "b3", "c3")))
        assert np.allclose(np.array(at_point.evalf(30).tolist(), dtype=float), value, **TOL)
    x, t, p = sp.symbols("xi theta phi", real=True)
    exact = sd.polar_rotation_matrix(x, t, p)
    assert sp.simplify(exact * sp.Matrix([sp.sin(t) * sp.cos(p), sp.sin(t) * sp.sin(p), sp.cos(t)])) == sp.Matrix(
        [sp.sin(t) * sp.cos(p), sp.sin(t) * sp.sin(p), sp.cos(t)]
    )
    assert np.allclose(sd.rotate_polar([1, 2, 3], xi, (theta, phi)), geometry("rotate_polar"), **TOL)
    assert np.allclose(sd.rotate_polar([[1, 2, 3], [0, 0, 1]], xi, (theta, phi)), geometry("rotate_polar_list"), **TOL)
    about = sd.rotate_polar([1, 2, 3], 0.3, [1, 1, 1])
    assert np.allclose(about, geometry("rotate_polar_about_vector"), **TOL)
    assert np.allclose(about, rodrigues(0.3, [1, 1, 1]) @ [1, 2, 3], **TOL)
    r, (theta, phi) = sd.vector_to_polar([1, 2, 3])
    theirs = geometry("vector_to_polar")
    assert np.allclose([r, theta, phi], [theirs[0], *theirs[1]], **TOL)
    assert np.allclose(sd.polar_to_vector(r, (theta, phi)), [1, 2, 3], **TOL)
    down = geometry("vector_to_polar_along_z")
    assert np.allclose(
        [sd.vector_to_polar([0, 0, -2])[0], *sd.vector_to_polar([0, 0, -2])[1]], [down[0], *down[1]], **TOL
    )
    assert np.allclose(sd.polar_to_vector(2, (0.3, 0.4)), geometry("polar_to_vector"), **TOL)


def test_quaternions():
    assert np.allclose([QA, QB, QC], geometry("quaternions"), **TOL)
    assert np.allclose(sd.polar_to_quaternion(*ANGLES), geometry("polar_to_quaternion"), **TOL)
    assert np.allclose(sd.quaternion_to_polar(QB), geometry("quaternion_to_polar"), **TOL)
    assert np.allclose(sd.quaternion_to_polar(QB), (1.3, 0.2, 2.4), **TOL)
    assert np.allclose(sd.quaternion_to_polar([1, 0, 0, 0]), geometry("quaternion_to_polar_identity"), **TOL)
    about_z = [np.cos(0.2), 0, 0, np.sin(0.2)]
    assert np.allclose(sd.quaternion_to_polar(about_z), geometry("quaternion_to_polar_about_z"), **TOL)
    assert np.allclose(sd.quaternion_to_euler(QA), geometry("quaternion_to_euler"), **TOL)
    assert np.allclose(sd.quaternion_to_euler(QB), geometry("quaternion_to_euler_b"), **TOL)
    assert np.allclose(sd.quaternion_to_euler([1, 0, 0, 0]), geometry("quaternion_to_euler_identity"), **TOL)
    assert np.allclose(sd.quaternion_product(QA, QB), geometry("quaternion_product"), **TOL)
    assert np.allclose(sd.quaternion_product(QC, QA, QB), geometry("quaternion_product_of_three"), **TOL)
    assert np.allclose(sd.quaternion_product(QA), geometry("quaternion_product_of_one"), **TOL)

    # independent: quaternions, polar angles and Euler angles describe the same rotation matrices
    def matrix(q):
        xi, theta, phi = sd.quaternion_to_polar(q)
        return rodrigues(xi, sd.polar_to_vector(1, (theta, phi))) if xi else np.eye(3)

    for q in (QA, QB, QC, about_z):
        assert np.allclose(euler(sd.quaternion_to_euler(q)), matrix(q), **TOL)
        assert np.isclose(np.linalg.norm(q), 1)
    assert np.allclose(matrix(sd.quaternion_product(QA, QB)), matrix(QA) @ matrix(QB), **TOL)
    assert np.allclose(matrix(sd.quaternion_product(QC, QA, QB)), matrix(QC) @ matrix(QA) @ matrix(QB), **TOL)
    with pytest.raises(ValueError, match="needs quaternions"):
        sd.quaternion_product()


def test_distances_and_angles():
    a, b, c, d = [1, 0.2, 0], [0, 0, 0], [0, 1.5, 0], [-0.5, 1.7, 0.8]
    assert np.isclose(sd.distance([1, 2, 3], [0, 1, 5]), geometry("distance"))
    assert np.isclose(sd.subtended_angle([1, 0, 0], [1, 1, 0]), geometry("subtended_angle_of_two_vectors"))
    at_point = sd.subtended_angle(a, [0, 0.5, 0], [-0.3, 1, 0.1])
    assert np.isclose(at_point, geometry("subtended_angle_at_a_point"))
    assert np.isclose(sd.subtended_angle(a, b, c, d), geometry("subtended_angle_of_four_points"))
    assert np.isclose(sd.bond_angle(a, b, [-0.3, 1, 0.1]), geometry("bond_angle"))
    assert np.isclose(sd.torsion_angle(a, b, c, d), geometry("torsion_angle"))
    mirrored = [-0.5, 1.7, -0.8]
    assert np.isclose(sd.torsion_angle(a, b, c, mirrored), geometry("torsion_angle_other_side"))
    assert np.isclose(sd.torsion_angle([1, 0, 0], b, [0, 1, 0], [-1, 1, 0]), geometry("torsion_angle_trans"))
    assert np.isclose(sd.solid_angle(np.eye(3)), geometry("solid_angle_octant"))
    triangle = [[1, 0.2, 0], [0, 2, 0.3], [0.1, 0, 3]]
    assert np.isclose(sd.solid_angle(triangle), geometry("solid_angle"))
    # independent: the angle at a point from the scalar product, the torsion as a rotation about the middle bond
    u, v = np.array(a) - [0, 0.5, 0], np.array([-0.3, 1, 0.1]) - [0, 0.5, 0]
    assert np.isclose(at_point, np.arccos(u @ v / np.linalg.norm(u) / np.linalg.norm(v)))
    assert np.isclose(sd.torsion_angle(a, b, c, mirrored), -sd.torsion_angle(a, b, c, d))
    assert np.isclose(abs(sd.torsion_angle([1, 0, 0], b, [0, 1, 0], [-1, 1, 0])), np.pi)
    assert np.isclose(sd.torsion_angle([1, 0, 0], b, [0, 1, 0], [1, 1, 0]), 0)
    axis = np.array([0.0, 1.0, 0.0])  # the bond from b to c

    def across(vector):
        """The direction of a vector seen along the bond."""
        rest = np.asarray(vector, dtype=float) - (np.asarray(vector, dtype=float) @ axis) * axis
        return rest / np.linalg.norm(rest)

    near, far = across(np.array(a) - b), across(np.array(d) - c)
    assert np.allclose(sd.rotate_polar(near, sd.torsion_angle(a, b, c, d), axis), far)
    # the eight octants fill the sphere; four points are two triangles (SpinDynamica leaves that form unevaluated)
    assert np.isclose(8 * sd.solid_angle(np.eye(3)), 4 * np.pi)
    assert geometry("solid_angle_of_four_points_is_not_evaluated") == "True"
    square = [[1, 0, 1], [0, 1, 1], [-1, 0, 1], [0, -1, 1]]
    assert np.isclose(sd.solid_angle(square), 2 * sd.solid_angle(square[:3]))
    with pytest.raises(ValueError, match="three or four"):
        sd.solid_angle([[1, 0, 0], [0, 1, 0]])
