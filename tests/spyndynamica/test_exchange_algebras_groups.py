"""Polarization levels, extended Liouville space, operator algebras, permutation groups."""

import math

import numpy as np
import pytest
import sympy as sp

import spyndynamica as sd
from spyndynamica import ExtendedOperatorBasis, KineticExchange, SiteDependent, SpinSystem
from tests.spyndynamica import reference

FIXTURE = "13_polarization_exchange_algebras"
TOL = dict(rtol=0, atol=1e-10)
HALF = sp.Rational(1, 2)


def stored(key):
    return reference.get(FIXTURE, key)


def number(matrix):
    return np.array(sp.N(sp.Matrix(matrix), 30).tolist(), dtype=complex)


def flat(matrix):
    return number(matrix).reshape(-1)


# ------------------------------------------------------------------------------------------- polarization
def test_polarization_level_operators():
    s = SpinSystem(2)
    g = "two_half_polarization"
    assert np.allclose(sd.polarization_level(s).numeric(), stored(f"{g}/level_default"), **TOL)
    assert np.allclose(sd.polarization_level(s, "z").numeric(), stored(f"{g}/level_z"), **TOL)
    assert np.allclose(sd.polarization_level(s, "x", spins=[1]).numeric(), stored(f"{g}/level_spin_1_x"), **TOL)
    assert np.allclose(sd.polarization_level(s, "y", spins=2).numeric(), stored(f"{g}/level_spin_2_y"), **TOL)
    p = sp.Rational(3, 10)
    one = sd.polarized(s, p, "z", spins=[1])
    assert one.amplitude(sd.polarization_level(s, "z", spins=[1])) == p
    assert np.isclose(stored(f"{g}/level_of_polarized_spin"), 0.3)
    both = sd.polarized(s, {1: p, 2: sp.Rational(1, 10)})
    assert both.amplitude(sd.polarization_level(s)) == sp.Rational(1, 5)
    assert np.isclose(stored(f"{g}/mean_level_of_two_spins"), 0.2)
    assert np.allclose(sd.singlet_polarization_level(s).numeric(), stored(f"{g}/singlet_level"), **TOL)
    assert sd.singlet_polarized(s, p).amplitude(sd.singlet_polarization_level(s)) == p
    assert np.isclose(stored(f"{g}/singlet_level_of_singlet_polarized"), 0.3)
    # independent: the level is <I_axis>/I, and for a singlet pair -(4/3) <I1.I2>
    rho = both.numeric()
    assert np.isclose(np.trace(rho @ s.I(1, "z").numeric()).real / 0.5, 0.3)
    singlet = s.singlet_triplet().ket("S0")
    pure = singlet @ singlet.dagger()
    assert pure.amplitude(sd.singlet_polarization_level(s)) == 1
    assert np.isclose(-4 / 3 * np.trace(pure.numeric() @ s.dot(1, 2).numeric()).real, 1.0)


def test_polarization_levels_of_other_systems():
    s = SpinSystem([1, HALF])
    g = "one_half_polarization"
    assert np.allclose(sd.polarization_level(s, spins=[1]).numeric(), stored(f"{g}/level_spin_1"), **TOL)
    assert np.allclose(sd.polarization_level(s).numeric(), stored(f"{g}/level_all"), **TOL)
    rho = s.unity() / 6 + s.I(1, "z") / 8
    level = rho.amplitude(sd.polarization_level(s, spins=[1]))
    assert np.isclose(float(level), stored(f"{g}/level_of_polarized_spin_1"))
    assert np.isclose(float(level), np.trace(rho.numeric() @ s.I(1, "z").numeric()).real / 1.0)

    four = SpinSystem(4)
    g = "four_half_polarization"
    assert np.allclose(
        sd.singlet_polarization_level(four, (1, 2)).numeric(), stored(f"{g}/singlet_level_pair_1_2"), **TOL
    )
    assert np.allclose(
        sd.singlet_polarization_level(four, [(3, 4)]).numeric(), stored(f"{g}/singlet_level_pair_3_4"), **TOL
    )
    # D11 in DIVERGENCES.md: SpinDynamica refuses pairs that are separate and accepts pairs that share a spin
    assert stored(f"{g}/separate_pairs_are_rejected") == "True"
    shared = stored(f"{g}/pairs_that_share_a_spin")
    assert np.allclose(shared, -(four.dot(1, 2) + four.dot(2, 3)).numeric() / 4, **TOL)
    with pytest.raises(ValueError, match="do not share"):
        sd.singlet_polarization_level(four, [(1, 2), (2, 3)])
    pairs = [(1, 2), (3, 4)]
    level = sd.singlet_polarization_level(four, pairs)
    assert level.equals(sd.singlet_polarization_level(four, (1, 2)) + sd.singlet_polarization_level(four, (3, 4)))
    assert sd.singlet_polarized(four, sp.Rational(3, 10), pairs).amplitude(level) == sp.Rational(3, 10)
    with pytest.raises(ValueError, match="spins-1/2"):
        sd.singlet_polarization_level(s, (1, 2))
    with pytest.raises(ValueError, match="more than two"):
        sd.singlet_polarization_level(four)


# ------------------------------------------------------------------------------------------- exchange
def test_site_dependent_operators_and_extended_vectors():
    s = SpinSystem(1)
    x, z = s.I(1, "x"), s.I(1, "z")
    g = "exchange_two_sites"
    basis = ExtendedOperatorBasis([1, 2], sd.default_operator_basis(s))
    assert np.allclose(stored(f"{g}/sites"), basis.sites) and len(basis) == 8
    assert np.allclose([op.numeric() for op in basis.basis.operators], stored(f"{g}/basis_operators"), **TOL)
    sdo = SiteDependent({1: z, 2: 3 * x})
    doubled = SiteDependent({1: z, 2: 2 * z})
    cases = {
        "vector_site_dependent": sdo,
        "vector_operator": z,
        "vector_one_site": SiteDependent({2: z}),
        "vector_sum_with_operator": sdo + x,
        "vector_product": sdo @ sdo,
        "vector_multiple": 2 * sdo,
        "vector_number": 3,
        "vector_exponential": doubled.map(lambda op: s.from_matrix(op.matrix().exp())),
    }
    for name, operator in cases.items():
        assert np.allclose(flat(basis.vector(operator)), stored(f"{g}/{name}"), **TOL), name
    cartesian = sd.cartesian_product_basis(s)
    other = SiteDependent({1: z, 2: 2 * x})
    in_cartesian = ExtendedOperatorBasis([1, 2], cartesian).vector(other)
    assert np.allclose(flat(in_cartesian), stored(f"{g}/vector_in_cartesian_basis"), **TOL)
    three = ExtendedOperatorBasis([2, 5, 1], cartesian)
    assert np.allclose(flat(three.vector(other)), stored(f"{g}/vector_at_three_sites"), **TOL)
    # independent: the vector is the concatenation of the brackets with the basis operators, site after site
    brackets = [b.bracket(op) for op in (z, 3 * x) for b in basis.basis.operators]
    assert list(basis.vector(sdo)) == brackets
    back = basis.split(basis.vector(sdo + x))
    assert back[1].equals(z + x) and back[2].equals(4 * x) and back.sites == (1, 2)
    assert (x @ sdo)[1].equals(x @ z) and (sdo - x)[2].equals(2 * x) and (sdo / 3)[2].equals(x)
    with pytest.raises(ValueError, match="same sites"):
        sdo + SiteDependent({1: z})
    with pytest.raises(ValueError, match="not among the sites"):
        basis.vector(SiteDependent({7: z}))
    with pytest.raises(TypeError, match="operators, or they hold superoperators"):
        SiteDependent({1: z, 2: sd.commutation(z)})
    with pytest.raises(ValueError, match="distinct"):
        ExtendedOperatorBasis([1, 1], cartesian)


def test_site_dependent_superoperators_and_extended_matrices():
    s = SpinSystem(1)
    x, y, z = s.I(1)
    g = "exchange_two_sites"
    basis = ExtendedOperatorBasis([1, 2], sd.default_operator_basis(s))
    sds = sd.commutation(SiteDependent({1: 10 * z, 2: -20 * z}))
    doubled = SiteDependent({1: z, 2: 2 * z})
    cases = {
        "matrix_site_dependent_commutation": sds,
        "matrix_commutation": sd.commutation(z),
        "matrix_multiple": -sp.I * sds,
        "matrix_sum": sds + 2 * sds,
        "matrix_product": sds @ sds,
        "matrix_product_with_superoperator": sds @ sd.commutation(x),
        "matrix_number": 3,
        "matrix_one_site": SiteDependent({2: sd.commutation(z)}),
        "matrix_two_superoperators": SiteDependent({1: sd.commutation(z), 2: sd.lindbladian(s.I(1, "-"))}),
        "matrix_double_commutation": sd.double_commutation(doubled, x),
        "matrix_double_commutation_both": sd.double_commutation(doubled, SiteDependent({1: x, 2: y})),
        "kinetic_superoperator": KineticExchange([1, 2], {(1, 2): 7}),
        "kinetic_superoperator_two_rules": KineticExchange([1, 2], {(1, 2): 7, (2, 1): 3}),
    }
    for name, superoperator in cases.items():
        theirs = stored(f"{g}/{name}")
        assert np.allclose(basis.numeric(superoperator), theirs, **TOL), name
        assert np.allclose(number(basis.matrix(superoperator)), theirs, **TOL), name
    # independent: one block per site, each the matrix of that site's superoperator in the basis
    blocks = basis.numeric(sds)
    assert np.allclose(blocks[:4, :4], sd.commutation(10 * z).numeric(basis=basis.basis), **TOL)
    assert np.allclose(blocks[4:, 4:], sd.commutation(-20 * z).numeric(basis=basis.basis), **TOL)
    assert np.allclose(blocks[:4, 4:], 0) and np.allclose(blocks[4:, :4], 0)
    assert sds(x)[2].equals(-20 * sp.I * y)
    # what SpinDynamica leaves undone: sums with plain superoperators and with the exchange
    assert stored(f"{g}/sum_with_superoperator_is_rejected") == "True"
    assert stored(f"{g}/sum_with_exchange_stays_unevaluated") == "True"
    exchange = KineticExchange([1, 2], {(1, 2): 7})
    both = sd.commutation(z) + sds
    assert np.allclose(basis.numeric(both), basis.numeric(sd.commutation(z)) + blocks, **TOL)
    generator = -sp.I * sds + exchange
    assert np.allclose(basis.numeric(generator), -1j * blocks + basis.numeric(exchange), **TOL)
    assert np.allclose(basis.numeric(2 * exchange - sds), 2 * basis.numeric(exchange) - blocks, **TOL)
    k = sp.Symbol("k", positive=True)
    symbolic = basis.numeric(-sp.I * sds + KineticExchange([1, 2], {(1, 2): k}), {k: 7})
    assert np.allclose(symbolic, basis.numeric(generator), **TOL)
    with pytest.raises(TypeError, match="expected superoperators"):
        basis.matrix(SiteDependent({1: z}))


def test_kinetic_exchange_matrix():
    g = "exchange_two_sites"
    k = sd.kinetic_exchange_matrix
    assert np.allclose(number(k([1, 2], {(1, 2): 7})), stored(f"{g}/kinetic_matrix_from_rule"), **TOL)
    assert np.allclose(number(k([1, 2], {(1, 2): 7, (2, 1): 3})), stored(f"{g}/kinetic_matrix_two_rules"), **TOL)
    assert np.allclose(number(k([1, 2], [[0, 3], [5, 0]])), stored(f"{g}/kinetic_matrix_from_matrix"), **TOL)
    assert np.allclose(number(k([1, 2], [[-2, 3], [2, -3]])), stored(f"{g}/kinetic_matrix_already_balanced"), **TOL)
    unbalanced = k([1, 2], [[0, 3], [5, 0]], balance=False)
    assert np.allclose(number(unbalanced), stored(f"{g}/kinetic_matrix_not_balanced"), **TOL)
    assert np.allclose(
        number(KineticExchange([1, 2], {(1, 2): 7}).matrix), stored(f"{g}/kinetic_matrix_of_superoperator"), **TOL
    )
    a, b = sp.symbols("a b", positive=True)
    symbolic = stored(f"{g}/kinetic_matrix_symbolic_not_balanced")
    for parameters, value in symbolic.points:
        at_point = k([1, 2], {(1, 2): a, (2, 1): b}, balance=False).subs({a: parameters["a3"], b: parameters["b3"]})
        assert np.allclose(number(at_point), value, **TOL)
    # SpinDynamica cannot balance symbolic rate constants; a rate from site 1 to site 2 stands in row 2, column 1
    assert stored(f"{g}/kinetic_matrix_symbolic_is_rejected") == "True"
    assert k([1, 2], {(1, 2): a, (2, 1): b}) == sp.Matrix([[-a, b], [a, -b]])
    assert stored(f"{g}/kinetic_matrix_conflict_is_rejected") == "True"
    with pytest.raises(ValueError, match="neither adds up to zero"):
        k([1, 2], [[1, 3], [2, -3]])
    with pytest.raises(ValueError, match="2 rows"):
        k([1, 2], [[1, 3, 0]])
    with pytest.raises(ValueError, match="not a pair of the sites"):
        k([1, 2], {(1, 3): 1})
    assert (
        KineticExchange([1, 2], {(1, 2): 7}).balanced
        and not KineticExchange([1, 2], unbalanced, balance=False).balanced
    )

    g = "exchange_three_sites"
    rates = {(1, 2): 1, (2, 3): 2, (3, 2): 5}
    matrix = k([1, 2, 3], rates)
    assert np.allclose(number(matrix), stored(f"{g}/kinetic_matrix"), **TOL)
    assert all(sum(matrix[:, column]) == 0 for column in range(3))
    s = SpinSystem(2)
    basis = ExtendedOperatorBasis([1, 2, 3], sd.default_operator_basis(s))
    exchange = basis.numeric(KineticExchange([1, 2, 3], rates))
    assert np.allclose(exchange.shape, stored(f"{g}/kinetic_superoperator_dimensions"))
    corner = [0, 16, 32]
    assert np.allclose(exchange[np.ix_(corner, corner)], stored(f"{g}/kinetic_superoperator_corner"), **TOL)
    assert np.allclose(exchange, np.kron(number(matrix), np.eye(16)), **TOL)
    partial = SiteDependent({1: s.I(1, "z"), 3: s.dot(1, 2)})
    assert np.allclose(flat(basis.vector(partial)), stored(f"{g}/vector_two_of_three_sites"), **TOL)
    hamiltonians = SiteDependent({1: s.I(1, "z"), 2: 2 * s.I(2, "z"), 3: s.I(None, "z")})
    diagonal = np.diag(basis.numeric(sd.commutation(hamiltonians)))
    assert np.allclose(diagonal, stored(f"{g}/matrix_site_dependent_commutation_diagonal"), **TOL)
    # sites in another order than the basis has them
    swapped = basis.numeric(KineticExchange([3, 1, 2], {(1, 2): 1, (2, 3): 2, (3, 2): 5}))
    assert np.allclose(swapped, exchange, **TOL)


def test_two_site_exchange_follows_the_bloch_mcconnell_equations():
    """Independent of SpinDynamica: the transverse magnetisation of two exchanging sites."""
    s = SpinSystem(1)
    z, minus = s.I(1, "z"), s.I(1, "-")
    operator_basis = sd.default_operator_basis(s)
    basis = ExtendedOperatorBasis(["A", "B"], operator_basis)
    omega = {"A": 2 * np.pi * 40.0, "B": -2 * np.pi * 25.0}
    component = int(np.argmax(np.abs(flat(operator_basis.vector(minus)))))
    for rate in (5.0, 80.0, 3000.0):
        hamiltonians = SiteDependent({site: value * z for site, value in omega.items()})
        generator = basis.numeric(
            -sp.I * sd.commutation(hamiltonians) + KineticExchange(["A", "B"], {("A", "B"): rate})
        )
        pick = [component, component + len(operator_basis)]
        assert np.allclose(np.delete(generator[pick], pick, axis=1), 0)  # the coherence couples to itself only
        # [I-, Iz] = I-, so the coherence of site j evolves at +i omega_j
        expected = np.array([[1j * omega["A"] - rate, rate], [rate, 1j * omega["B"] - rate]])
        assert np.allclose(generator[np.ix_(pick, pick)], expected, **TOL)
    fast = np.linalg.eigvals(expected)
    slow_mode = fast[np.argmax(fast.real)]
    assert np.isclose(slow_mode.imag, (omega["A"] + omega["B"]) / 2, rtol=1e-3)  # fast exchange: the mean frequency
    # the exchange conserves the total of every operator component
    unity = flat(basis.vector(1))
    assert np.allclose(unity @ basis.numeric(KineticExchange(["A", "B"], {("A", "B"): 7.0})), 0, **TOL)


# ------------------------------------------------------------------------------------------- algebras
def rank(vectors):
    vectors = np.array(vectors)
    return int(np.linalg.matrix_rank(vectors, tol=1e-9)) if vectors.size else 0


def same_span(operators, matrices):
    mine = [op.numeric().reshape(-1) for op in operators]
    theirs = [np.asarray(m, dtype=complex).reshape(-1) for m in matrices]
    return rank(mine) == rank(theirs) == rank(mine + theirs)


def commutes(a, b):
    return np.allclose(a @ b, b @ a, **TOL)


def lie_basis(s):
    return [s.I(spin, c) for spin in (1, 2) for c in "xyz"]


def test_commutant_and_bicommutant():
    s = SpinSystem(2)
    i = s.I
    g = "two_half_algebras"
    total_z, coupling = i(None, "z"), s.dot(1, 2)
    cases = {
        "commutant_Iz": sd.commutant([total_z]),
        "commutant_scalar_coupling_and_Iz": sd.commutant([coupling, total_z]),
        "commutant_of_one_operator": sd.commutant(coupling),
        "commutant_within": sd.commutant([i(1, "z")], lie_basis(s)),
        "commutant_in_cartesian_basis": sd.commutant([total_z], basis=sd.cartesian_product_basis(s)),
        "bicommutant_Iz": sd.bicommutant([total_z]),
        "bicommutant_scalar_coupling_and_Iz": sd.bicommutant([coupling, total_z]),
    }
    for name, operators in cases.items():
        assert same_span(operators, stored(f"{g}/{name}")), name
    # independent: everything commutes, and the dimension is that of the null space of the commutation matrix
    for generators in ([total_z], [coupling, total_z], [coupling], [i(1, "x") @ i(2, "y")]):
        found = sd.commutant(generators)
        for c in found:
            assert all(commutes(c.numeric(), q.numeric()) for q in generators)
        stacked = np.vstack([sd.commutation(q).numeric() for q in generators])
        assert len(found) == 16 - np.linalg.matrix_rank(stacked) == rank([c.numeric().reshape(-1) for c in found])
    assert same_span(sd.bicommutant([total_z]), [np.eye(4), total_z.numeric(), (total_z @ total_z).numeric()])
    # floating-point coefficients give the same space, in exact numbers
    floats = sd.commutant([2.5 * total_z + 0.1 * coupling])
    assert same_span(floats, [c.numeric() for c in sd.commutant([coupling, total_z])])
    assert all(not sp.Matrix(list(c.terms.values())).has(sp.Float) for c in floats)
    three = SpinSystem(3)
    generators = [three.I(None, "z"), three.dot(1, 2) + three.dot(2, 3) + three.dot(1, 3)]
    found = sd.commutant(generators)
    assert len(found) == stored("three_half_algebras/commutant_count")
    assert same_span(found, stored("three_half_algebras/commutant"))
    with pytest.raises(ValueError, match="must be numbers"):
        sd.commutant([sp.Symbol("w") * total_z])
    with pytest.raises(TypeError, match="operator or a list"):
        sd.commutant([])


def test_lie_algebra():
    s = SpinSystem(2)
    i = s.I
    g = "two_half_algebras"
    cases = {
        "lie_algebra_total_x_y": [i(None, "x"), i(None, "y")],
        "lie_algebra_spin_1_x_y": [i(1, "x"), i(1, "y")],
        "lie_algebra_weak_coupling": [i(1, "x"), i(2, "x"), i(1, "z") @ i(2, "z"), i(1, "y"), i(2, "y")],
        "lie_algebra_scalar_coupling": [i(None, "x"), i(None, "y"), s.dot(1, 2)],
        "lie_algebra_real_field": [i(1, "+"), i(1, "z")],
    }
    dimensions = {"lie_algebra_total_x_y": 3, "lie_algebra_spin_1_x_y": 3, "lie_algebra_weak_coupling": 15,
                  "lie_algebra_scalar_coupling": 4, "lie_algebra_real_field": 3}  # fmt: skip
    for name, generators in cases.items():
        algebra = sd.lie_algebra(generators)
        theirs = stored(f"{g}/{name}")
        assert len(algebra) == len(theirs) == dimensions[name], name
        assert same_span(algebra, theirs), name
        # closed under -i [A, B], over the real numbers
        matrices = [op.numeric() for op in algebra]
        real = [np.concatenate([m.reshape(-1).real, m.reshape(-1).imag]) for m in matrices]
        for a in matrices:
            for b in matrices:
                c = -1j * (a @ b - b @ a)
                assert rank(real + [np.concatenate([c.reshape(-1).real, c.reshape(-1).imag])]) == len(real)
    # the generators come first, as in SpinDynamica
    theirs = stored(f"{g}/lie_algebra_spin_1_x_y")
    assert np.allclose([op.numeric() for op in sd.lie_algebra(cases["lie_algebra_spin_1_x_y"])], theirs, **TOL)
    complex_field = sd.lie_algebra([i(1, "+"), i(1, "z")], field="C")
    assert len(complex_field) == len(stored(f"{g}/lie_algebra_complex_field")) == 2
    three = SpinSystem(3)
    generators = [three.I(1, "x"), three.I(1, "y"), three.dot(1, 2), three.dot(2, 3)]
    assert len(sd.lie_algebra(generators)) == stored("three_half_algebras/lie_algebra_count") == 63
    assert len(sd.lie_algebra([0.5 * i(1, "x"), 2.0 * i(1, "y")])) == 3
    with pytest.raises(ValueError, match="traceless"):
        sd.lie_algebra([s.unity() + i(1, "z")])
    with pytest.raises(ValueError, match="field"):
        sd.lie_algebra([i(1, "z")], field="Q")


def test_abelian_subalgebras():
    s = SpinSystem(2)
    i = s.I
    g = "two_half_algebras"
    within = lie_basis(s)
    cases = {
        "abelian_subalgebra_Iz": ([i(None, "z")], None),
        "abelian_subalgebra_two": ([i(1, "z"), i(2, "z")], None),
        "abelian_subalgebra_within": ([i(1, "z")], within),
    }
    for name, (operators, space) in cases.items():
        found = sd.abelian_subalgebra(operators, space)
        theirs = stored(f"{g}/{name}")
        assert len(found) == len(theirs), name
        matrices = [op.numeric() for op in found]
        assert all(commutes(a, b) for a in matrices for b in matrices)
        assert all(np.allclose(m, m.conj().T, **TOL) for m in matrices), "Hermitian, as documented"
        assert all(found[n].equals(op) for n, op in enumerate(operators))
        assert sd.is_maximal_abelian(found, space)
        assert all(abs(np.trace(m)) < 1e-10 for m in matrices)
        if space is not None:
            assert rank([m.reshape(-1) for m in matrices] + [b.numeric().reshape(-1) for b in space]) == len(space)
    # SpinDynamica's sets are abelian as well, but not always Hermitian
    theirs = stored(f"{g}/abelian_subalgebra_Iz")
    assert all(commutes(a, b) for a in theirs for b in theirs)
    assert not all(np.allclose(m, m.conj().T) for m in theirs)
    booleans = {
        "maximal_abelian_three": sd.is_maximal_abelian([i(1, "z"), i(2, "z"), i(1, "z") @ i(2, "z")]),
        "maximal_abelian_two": sd.is_maximal_abelian([i(1, "z"), i(2, "z")]),
        "maximal_abelian_within": sd.is_maximal_abelian([i(1, "z"), i(2, "x")], within),
        "maximal_abelian_within_one": sd.is_maximal_abelian([i(1, "z")], within),
    }
    for name, value in booleans.items():
        assert str(value) == stored(f"{g}/{name}"), name
    assert not sd.is_maximal_abelian([i(1, "z"), i(1, "x")])
    with pytest.raises(ValueError, match="do not commute"):
        sd.abelian_subalgebra([i(1, "z"), i(1, "x")])
    with pytest.raises(ValueError, match="Hermitian"):
        sd.abelian_subalgebra([i(1, "+")])


def test_inaccessible_operators():
    s = SpinSystem(2)
    i = s.I
    g = "two_half_algebras"
    assert sd.inaccessible(i(1, "z"), i(2, "z"), [i(1, "x"), i(1, "y")]) is True
    assert stored(f"{g}/inaccessible_other_spin") == "True"
    targets = [i(1, "x"), i(1, "y"), i(2, "z"), i(1, "z") @ i(2, "z"), i(1, "x") @ i(2, "z"), i(1, "y") @ i(2, "z")]
    hamiltonians = [i(1, "x"), i(1, "z") @ i(2, "z")]
    found = sd.inaccessible(i(1, "z"), targets, hamiltonians)
    assert [str(value) for value in found] == stored(f"{g}/inaccessible_list")
    singlet = [s.dot(1, 2), i(1, "z") - i(2, "z"), i(None, "x")]
    symmetric = [i(None, "x"), i(None, "y"), s.dot(1, 2)]
    assert [str(v) for v in sd.inaccessible(i(None, "z"), singlet, symmetric)] == stored(
        f"{g}/inaccessible_singlet_order"
    )
    difference = i(1, "z") - i(2, "z")
    asymmetric = [i(1, "x") - i(2, "x"), s.dot(1, 2), difference]
    reached = sd.inaccessible(difference, [s.dot(1, 2), i(None, "z")], asymmetric)
    assert [str(v) for v in reached] == stored(f"{g}/accessible_singlet_order")
    # independent: evolution under a random sequence of the Hamiltonians reaches exactly the accessible targets
    rng = np.random.default_rng(3)
    rho = i(1, "z").numeric()
    for _ in range(40):
        h = hamiltonians[rng.integers(2)].numeric()
        values, vectors = np.linalg.eigh(h)
        u = vectors @ np.diag(np.exp(-1j * values * rng.uniform(0.3, 2.5))) @ vectors.conj().T
        rho = u @ rho @ u.conj().T
        overlaps = [abs(np.trace(t.numeric().conj().T @ rho)) for t in targets]
        assert all(overlap < 1e-10 for overlap, blocked in zip(overlaps, found) if blocked)
    assert all(overlap > 1e-3 for overlap, blocked in zip(overlaps, found) if not blocked)


# ------------------------------------------------------------------------------------------- groups
def test_spin_permutation_groups():
    s = SpinSystem(3)
    g = "three_half_groups"
    assert sd.permutation_group("SymmetricGroup", 3) == [
        (),
        ((1, 2),),
        ((1, 3),),
        ((2, 3),),
        ((1, 2, 3),),
        ((1, 3, 2),),
    ]
    cases = {
        "symmetric_group": sd.spin_permutation_group(s, ("SymmetricGroup", 3)),
        "cyclic_group": sd.spin_permutation_group(s, ("CyclicGroup", 3)),
        "alternating_group": sd.spin_permutation_group(s, ("AlternatingGroup", 3)),
        "symmetric_group_of_spins_2_3": sd.spin_permutation_group(s, ("SymmetricGroup", 2), spins=[2, 3]),
    }
    for name, operators in cases.items():
        assert np.allclose([op.numeric() for op in operators], stored(f"{g}/{name}"), **TOL), name
    assert np.allclose(cases["symmetric_group"][1].numeric(), stored(f"{g}/element_2"), **TOL)
    assert cases["symmetric_group"][0].equals(s.unity())
    explicit = sd.spin_permutation_group(s, [[], [1, 2, 3], [[1, 3, 2]]])
    assert all(a.equals(b) for a, b in zip(explicit, cases["cyclic_group"]))
    four = SpinSystem(4)
    assert len(sd.permutation_group("SymmetricGroup", 4)) == stored("four_half_groups/symmetric_group_count") == 24
    dihedral = sd.spin_permutation_group(four, ("DihedralGroup", 4))
    theirs = stored("four_half_groups/dihedral_group")
    assert len(dihedral) == len(theirs) == 8
    mine = {tuple(op.numeric().real.astype(int).reshape(-1)) for op in dihedral}
    assert mine == {tuple(np.asarray(m).real.astype(int).reshape(-1)) for m in theirs}
    # a group: closed under multiplication
    for a in dihedral:
        for b in dihedral:
            assert tuple((a @ b).numeric().real.astype(int).reshape(-1)) in mine
    with pytest.raises(ValueError, match="unknown group"):
        sd.permutation_group("MonsterGroup", 3)
    with pytest.raises(ValueError, match="distinct spins"):
        sd.permutation_group("SymmetricGroup", 3, [1, 2])


ADAPTED = {
    "three_half": (lambda: SpinSystem(3), None, None),
    "three_half_cyclic": (lambda: SpinSystem(3), ("CyclicGroup", 3), None),
    "three_half_two_spins": (lambda: SpinSystem(3), ("SymmetricGroup", 2), [2, 3]),
    "four_half": (lambda: SpinSystem(4), None, None),
    "four_half_cyclic": (lambda: SpinSystem(4), ("CyclicGroup", 4), None),
    "four_half_dihedral": (lambda: SpinSystem(4), ("DihedralGroup", 4), None),
    "two_spin1": (lambda: SpinSystem([1, 1]), None, None),
    "three_spin1": (lambda: SpinSystem([1, 1, 1]), None, None),
}
# the cases that SpinDynamica has (it fails for the dihedral group of four spins)
ADAPTED_STORED = {
    "three_half": ["three_half_groups/symmetry_adapted_basis", "three_half_groups/symmetry_adapted_basis_symmetric"],
    "three_half_cyclic": ["three_half_groups/symmetry_adapted_basis_cyclic",
                          "three_half_groups/symmetry_adapted_basis_from_elements"],
    "four_half": ["four_half_groups/symmetry_adapted_basis"],
    "four_half_cyclic": ["four_half_groups/symmetry_adapted_basis_cyclic"],
    "two_spin1": ["two_spin1_groups/symmetry_adapted_basis"],
}  # fmt: skip


@pytest.mark.parametrize("name", list(ADAPTED))
def test_symmetry_adapted_basis(name):
    make, group, spins = ADAPTED[name]
    s = make()
    basis = sd.symmetry_adapted_basis(s, group, spins)
    kets = number(basis.kets_matrix)
    assert np.allclose(kets.conj().T @ kets, np.eye(s.dimension), **TOL)
    operators = sd.spin_permutation_group(s, group or ("SymmetricGroup", len(s.spins)), spins)
    numbers = basis.quantum_numbers
    blocks = {}
    for n, (irrep, _dim, j, m, copy, _partner) in enumerate(numbers):
        blocks.setdefault((irrep, j, m, copy), []).append(n)
    in_block = np.zeros((s.dimension, s.dimension), dtype=bool)
    for members in blocks.values():
        in_block[np.ix_(members, members)] = True
    represented = np.array([kets.conj().T @ op.numeric() @ kets for op in operators])
    assert np.allclose(represented[:, ~in_block], 0, atol=1e-8), "every group operator is block diagonal"
    matrices = {}
    for (irrep, _j, _m, _copy), members in blocks.items():
        block = represented[:, members][:, :, members]
        character = np.trace(block, axis1=1, axis2=2)
        assert np.isclose(np.sum(np.abs(character) ** 2), len(operators), atol=1e-6), "irreducible"
        assert len(members) == numbers[members[0]][1]
        if irrep in matrices:
            assert np.allclose(block, matrices[irrep], atol=1e-7), "equivalent representations have equal matrices"
        matrices[irrep] = block
    characters = [np.trace(block, axis1=1, axis2=2) for block in matrices.values()]
    for a, first in enumerate(characters):
        for second in characters[a + 1 :]:
            assert abs(np.vdot(first, second)) < 1e-6, "different representations are inequivalent"
    j2 = sum(s.I(None, c).numeric() @ s.I(None, c).numeric() for c in "xyz")
    assert np.allclose(
        kets.conj().T @ j2 @ kets, np.diag([float(j * (j + 1)) for _, _, j, _, _, _ in numbers]), atol=1e-8
    )
    assert np.allclose(
        kets.conj().T @ s.I(None, "z").numeric() @ kets, np.diag([float(q[3]) for q in numbers]), atol=1e-8
    )
    keys = [(q[1], q[0], -q[2], q[4], -q[3], q[5]) for q in numbers]
    assert keys == sorted(keys), "ordered by dimension, representation, J, copy, M"
    # SpinDynamica's kets span the same spaces: representation by representation, and (J, M) by (J, M)
    for key in ADAPTED_STORED.get(name, []):
        theirs = stored(key).T
        assert np.allclose(theirs.conj().T @ theirs, np.eye(s.dimension), atol=1e-9), key
        spaces = {}
        for n, (irrep, _dim, j, m, _copy, _partner) in enumerate(numbers):
            spaces.setdefault((irrep, j, m), []).append(n)
        counts = dict.fromkeys(spaces, 0)
        for column in theirs.T:
            weights = {space: np.linalg.norm(kets[:, members].conj().T @ column) for space, members in spaces.items()}
            best = max(weights, key=weights.get)
            assert np.isclose(weights[best], 1.0, atol=1e-8), key
            counts[best] += 1
        assert counts == {space: len(members) for space, members in spaces.items()}, key


def test_symmetry_adapted_basis_of_spindynamica_depends_on_the_draw():
    # D12 in DIVERGENCES.md: for some random draws SpinDynamica returns kets that are no basis
    defective = stored("three_half_groups_other_draw/symmetry_adapted_basis_is_defective")
    norms = np.linalg.norm(defective, axis=1)
    assert sorted(np.round(norms, 12)) == [0, 0, 1, 1, 1, 1, 1, 1]
    # the port draws from a fixed seed: the same basis every time
    first = number(sd.symmetry_adapted_basis(SpinSystem(3)).kets_matrix)
    assert np.array_equal(first, number(sd.symmetry_adapted_basis(SpinSystem(3)).kets_matrix))


def test_symmetry_adapted_basis_arguments():
    with pytest.raises(ValueError, match="same quantum number"):
        sd.symmetry_adapted_basis(SpinSystem([HALF, 1]))
    s = SpinSystem(2)
    basis = sd.symmetry_adapted_basis(s)
    assert [q[:4] for q in basis.quantum_numbers] == [(0, 1, 1, 1), (0, 1, 1, 0), (0, 1, 1, -1), (1, 1, 0, 0)]
    singlet = number(basis.kets_matrix)[:, 3]
    assert np.isclose(abs(np.vdot(singlet, flat(s.singlet_triplet().ket("S0").vector))), 1.0)


def hook_dimension(partition):
    """The dimension of the representation of the symmetric group with this Young diagram."""
    columns = [sum(1 for row in partition if row > c) for c in range(partition[0])]
    hooks = [partition[r] - c + columns[c] - r - 1 for r in range(len(partition)) for c in range(partition[r])]
    return math.factorial(sum(partition)) // math.prod(hooks)


@pytest.mark.parametrize("key, spins, j", [("three_half", 3, HALF), ("four_half", 4, HALF), ("three_one", 3, 1),
                                           ("two_three_half", 2, sp.Rational(3, 2)), ("four_one", 4, 1)])  # fmt: skip
def test_dual_pairing(key, spins, j):
    pairing = sd.dual_pairing(spins, j)
    theirs = {}
    for rows, ranks in stored(f"dual_pairing/{key}"):
        content = {}
        for label, value in ranks:
            multiplicity = int(label.split("×")[0]) if "×" in label else 1
            content[sp.nsimplify(float(value))] = multiplicity
        theirs[tuple(int(r) for r in rows)] = content
    assert pairing == theirs
    # independent: the dimensions add up to the Hilbert space, and the symmetry-adapted basis shows the same pairs
    dimension = sum(
        hook_dimension(p) * count * (2 * big + 1) for p, content in pairing.items() for big, count in content.items()
    )
    assert dimension == (2 * j + 1) ** spins
    basis = sd.symmetry_adapted_basis(SpinSystem([j] * spins))
    highest = sorted((q[1], q[2]) for q in basis.quantum_numbers if q[2] == q[3] and q[5] == 0)
    expected = sorted(
        (hook_dimension(p), big)
        for p, content in pairing.items()
        for big, count in content.items()
        for _ in range(count)
    )
    assert highest == expected


def test_dual_pairing_arguments():
    assert sd.dual_pairing(3, 0.5) == {(3,): {sp.Rational(3, 2): 1}, (2, 1): {HALF: 1}}
    assert sd.dual_pairing(6, HALF)[(3, 3)] == {0: 1} and (2, 2, 2) not in sd.dual_pairing(6, HALF)
    assert sd.dual_pairing(4, sp.Rational(3, 2))[(2, 2)] == {4: 1, 2: 2, 0: 1}
    with pytest.raises(ValueError, match="spin quantum number"):
        sd.dual_pairing(3, 0.3)
