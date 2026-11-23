"""The symbolic core against the SpinDynamica fixtures (bases, operators, rotations) and against itself."""

import itertools

import numpy as np
import pytest
import sympy as sp

import spyndynamica as sd
from spyndynamica import SpinSystem, commutator
from tests.spyndynamica import reference

TOL = dict(rtol=0, atol=1e-12)
SYSTEMS = {
    "one_half": lambda: SpinSystem(1),
    "two_half": lambda: SpinSystem(2),
    "three_half": lambda: SpinSystem(3),
    "spin1": lambda: SpinSystem([1]),
    "half_one": lambda: SpinSystem([sp.Rational(1, 2), 1]),
    "spin32": lambda: SpinSystem(["3/2"]),
}
A3, B3, C3 = sp.symbols("a3 b3 c3", real=True)
PHI, THETA = sp.symbols("phi theta", real=True)
ANGLES = (sp.pi / 5, sp.pi / 3, 2 * sp.pi / 7)


def check(fixture: str, key: str, value, variables=()):
    """Compare an operator, a matrix or a scalar of the port with a stored case, exactly where the case is exact."""
    case = reference.cases(fixture)[key]["value"]

    def numeric(v, values=None):
        if isinstance(v, sd.Operator):
            return v.numeric(values)
        v = sp.sympify(v)
        if values:
            v = v.subs(values)
        return np.array(sp.N(v, 30).tolist() if hasattr(v, "tolist") else sp.N(v, 30), dtype=complex)

    if case["type"] == "symbolic":
        symbols = {str(v): v for v in variables}
        assert set(case["variables"]) == set(symbols), key
        for point in case["points"]:
            values = {
                symbols[name]: sp.sympify(text) for name, text in zip(case["variables"], point["parameters_exact"])
            }
            expected = reference.array(point["value"])
            assert np.allclose(numeric(value, values).reshape(expected.shape), expected, **TOL), key
        return
    expected = reference.array(case)
    assert np.allclose(numeric(value).reshape(expected.shape), expected, **TOL), key
    if "exact" in case and len(case["exact"]) < 600 and isinstance(value, (sd.Operator, sp.MatrixBase)):
        exact = value.matrix() if isinstance(value, sd.Operator) else value
        parsed = sp.Matrix(reference.exact(case)).reshape(*exact.shape)
        # exact against exact, decided at 40 digits (simplify does not always reduce exp(i pi/3) against radicals)
        assert max(abs(sp.N(d, 40)) for d in (exact - parsed)) < 1e-35, key


# ------------------------------------------------------------------------------------------- bases
@pytest.mark.parametrize("group", list(SYSTEMS))
def test_zeeman_basis_order(group):
    s = SYSTEMS[group]()
    stored = reference.get("01_bases", f"{group}/zeeman_quantum_numbers")[:, :, 1]
    assert np.allclose(np.array(s.quantum_numbers, dtype=float), stored, **TOL)
    assert s.dimension == int(reference.get("01_bases", f"{group}/dimension"))
    assert list(s.zeeman.labels) == [int(x) for x in reference.get("01_bases", f"{group}/labels")]
    diagonal = np.diag(s.I(None, "z").numeric()).real
    assert np.allclose(diagonal, reference.get("01_bases", f"{group}/Iz_total_diagonal"), **TOL)
    kets = np.array([np.array(k.vector.tolist(), dtype=float).ravel() for k in s.zeeman.kets])
    assert np.allclose(kets, reference.get("01_bases", f"{group}/zeeman_kets"), **TOL)


def test_singlet_triplet_basis():
    s = SpinSystem(2)
    st = s.singlet_triplet()
    g = "singlet_triplet"
    kets = np.array([np.array(k.vector.evalf().tolist(), dtype=complex).ravel() for k in st.kets])
    assert np.allclose(kets, reference.get("01_bases", f"{g}/kets_in_zeeman"), **TOL)
    check("01_bases", f"{g}/transformation_st_from_zeeman", st.transformation(s.zeeman))
    check("01_bases", f"{g}/transformation_zeeman_from_st", s.zeeman.transformation(st))
    check("01_bases", f"{g}/I1z_in_st", s.I(1, "z").matrix(st))
    check("01_bases", f"{g}/I1I2_in_st", s.dot(1, 2).matrix(st))
    check("01_bases", f"{g}/Ix_total_in_st", s.I(None, "x").matrix(st))
    assert st.ket("S0").vector == st.ket(1).vector
    with pytest.raises(ValueError, match="two spins-1/2"):
        SpinSystem(3).singlet_triplet()


# ------------------------------------------------------------------------------------------- operators
def components(s, spin, prefix):
    names = {"x": "x", "y": "y", "z": "z", "plus": "+", "minus": "-"}
    return {f"{prefix}{name}": s.I(spin, c) for name, c in names.items()}


def tensors(s, spin, ranks, prefix):
    return {f"{prefix}T{k}_{q}": s.T(spin, k, q) for k in ranks for q in range(k, -k - 1, -1)}


def operator_cases(group):
    s = SYSTEMS[group]()
    i = s.I
    if group == "one_half":
        return {
            **components(s, 1, "I1"),
            "I1alpha": i(1, "alpha"),
            "I1beta": i(1, "beta"),
            "I1_phase_pi3": i(1, sp.pi / 3),
            "I1_polar_pi3_pi4": i(1, (sp.pi / 3, sp.pi / 4)),
            "I1_phase_symbolic": (i(1, PHI), (PHI,)),
            "I1_polar_symbolic": (i(1, (THETA, PHI)), (THETA, PHI)),
            "unity": s.unity(),
            "null": s.zero(),
            **tensors(s, 1, (1,), "spin1_"),
        }
    if group == "two_half":
        z = s.zeeman
        return {
            **components(s, 1, "I1"),
            **components(s, 2, "I2"),
            "I1alpha": i(1, "alpha"),
            "I2beta": i(2, "beta"),
            "Ix_total": i(None, "x"),
            "Iz_total": i(None, "z"),
            "Iz_list": i([1, 2], "z"),
            "I1xI2z": i(1, "x") @ i(2, "z"),
            "I1yI2x": i(1, "y") @ i(2, "x"),
            "I1zI2z": i(1, "z") @ i(2, "z"),
            "I1plusI2minus": i(1, "+") @ i(2, "-"),
            "I1_dot_I2": s.dot(1, 2),
            "I1x_I1y_product": i(1, "x") @ i(1, "y"),
            "commutator_I1x_I1y": commutator(i(1, "x"), i(1, "y")),
            "commutator_I1xI2z_I1y": commutator(i(1, "x") @ i(2, "z"), i(1, "y")),
            "unity": s.unity(),
            "unity_spin1": s.unity(),
            "st_x_12": s.single_transition(1, 2, "x"),
            "st_y_12": s.single_transition(1, 2, "y"),
            "st_z_12": s.single_transition(1, 2, "z"),
            "st_plus_23": s.single_transition(2, 3, "+"),
            "st_minus_14": s.single_transition(1, 4, "-"),
            "st_unity_23": s.single_transition(2, 3, "u"),
            "population_2": s.population(2),
            "coherence_1_2": s.coherence(1, 2),
            "coherence_3_1": s.coherence(3, 1),
            "shift_ket1_bra2": s.shift(z.ket(1), z.ket(2)),
            "projection_ket3": s.projector(z.ket(3)),
            "projection_spin1_alpha": i(1, "alpha"),
            "permutation_12": s.permutation([1, 2]),
            **tensors(s, 1, (1,), "spin1_"),
            **tensors(s, 2, (1,), "spin2_"),
            **tensors(s, [1, 2], (0, 1, 2), "pair_"),
        }
    if group == "three_half":
        cycle = s.permutation([1, 2, 3])
        return {
            "I3y": i(3, "y"),
            "I2plus": i(2, "+"),
            "I1xI2yI3z": i(1, "x") @ i(2, "y") @ i(3, "z"),
            "I1_dot_I3": s.dot(1, 3),
            "permutation_12": s.permutation([1, 2]),
            "permutation_cycle_123": cycle,
            "permutation_cycle_132": s.permutation([1, 3, 2]),
            "permutation_12_after_23": s.permutation([[1, 2], [2, 3]]),
            "permuted_I1z_by_cycle_123": cycle @ i(1, "z") @ cycle.dagger(),
        }
    if group == "spin1":
        return {
            **components(s, 1, "I1"),
            "I1z_squared": i(1, "z") @ i(1, "z"),
            "I1_dot_I1": s.dot(1, 1),
            **tensors(s, 1, (1, 2), ""),
        }
    if group == "half_one":
        return {
            **components(s, 1, "I1"),
            **components(s, 2, "I2"),
            "I1zI2z": i(1, "z") @ i(2, "z"),
            "I1_dot_I2": s.dot(1, 2),
            "I1plusI2minus": i(1, "+") @ i(2, "-"),
            **tensors(s, 2, (1, 2), "spin2_"),
        }
    return {**components(s, 1, "I1"), "I1_dot_I1": s.dot(1, 1), **tensors(s, 1, (1, 2, 3), "")}


SCALARS = {"trace_", "norm_", "coherence_order_", "spin_product_rank_", "commute_"}


@pytest.mark.parametrize("group", list(SYSTEMS))
def test_operator_matrices(group):
    built = operator_cases(group)
    stored = [k.split("/", 1)[1] for k in reference.cases("02_operators") if k.startswith(group + "/")]
    matrices = [name for name in stored if not any(name.startswith(prefix) for prefix in SCALARS)]
    assert sorted(matrices) == sorted(built), "every stored operator has a counterpart in the port"
    for name, value in built.items():
        value, variables = value if isinstance(value, tuple) else (value, ())
        check("02_operators", f"{group}/{name}", value, variables)


def test_scalar_properties():
    s = SpinSystem(2)
    i = s.I

    def stored(name):
        return reference.get("02_operators", f"two_half/{name}")

    assert (i(1, "z") @ i(1, "z")).trace() == sp.nsimplify(float(stored("trace_I1zI1z")))
    assert s.unity().trace() == 4
    assert np.isclose(float(i(1, "x").norm()), stored("norm_I1x"))
    assert np.isclose(float((i(1, "x") @ i(2, "z")).norm()), stored("norm_I1xI2z"))
    assert i(1, "+").norm() == sp.sqrt(2) and np.isclose(float(sp.sqrt(2)), stored("norm_I1plus"))
    assert (i(1, "+") @ i(2, "+")).coherence_orders() == [int(stored("coherence_order_I1plusI2plus"))]
    assert (i(1, "+") @ i(2, "-")).coherence_orders() == [int(stored("coherence_order_I1plusI2minus"))]
    assert i(1, "x").coherence_orders() == [int(x) for x in stored("coherence_order_I1x")]
    per_spin = [int(x) for x in np.ravel(stored("coherence_order_per_spin"))]
    mixed = i(1, "+") @ i(2, "-")
    assert [mixed.coherence_orders([1])[0], mixed.coherence_orders([2])[0]] == per_spin
    assert (i(1, "x") @ i(2, "y")).spin_product_rank() == int(stored("spin_product_rank_I1xI2y"))
    assert i(1, "x").spin_product_rank() == int(stored("spin_product_rank_I1x"))
    assert i(1, "z").commutes_with(i(2, "x")) is (stored("commute_I1z_I2x") == "True")
    assert i(1, "z").commutes_with(i(1, "x")) is (stored("commute_I1z_I1x") == "True")
    assert np.isclose(float(SpinSystem([1]).I(1, "z").norm()), reference.get("02_operators", "spin1/norm_I1z"))
    components = (i(1, "x") @ i(2, "x")).coherence_components()
    assert sorted(components) == [-2, 0, 2]
    assert sum(components.values(), s.zero()).equals(i(1, "x") @ i(2, "x"))


# ------------------------------------------------------------------------------------------- rotations
def test_wigner_clebsch_gordan_and_euler():
    g = "wigner"
    for name, j in (("d_half", sp.Rational(1, 2)), ("d_1", 1), ("d_3half", sp.Rational(3, 2)), ("d_2", 2)):
        check("03_rotations", f"{g}/{name}_symbolic", sd.wigner_d(j, B3), (B3,))
    for name, j in (("D_half", sp.Rational(1, 2)), ("D_1", 1), ("D_3half", sp.Rational(3, 2)), ("D_2", 2)):
        check("03_rotations", f"{g}/{name}", sd.wigner_D(j, *ANGLES))
    check("03_rotations", f"{g}/d_2_element_1_0_symbolic", sd.wigner_d(2, B3)[1, 2], (B3,))
    check("03_rotations", f"{g}/d_2_row_1", sd.wigner_d(2, sp.pi / 3)[1, :])
    check("03_rotations", f"{g}/d_2_column_1", sd.wigner_d(2, sp.pi / 3)[:, 1])
    check("03_rotations", f"{g}/D_2_element_1_m2", sd.wigner_D(2, *ANGLES)[1, 4])
    product = sd.wigner_D(1, *ANGLES) * sd.wigner_D(1, sp.pi / 7, sp.pi / 4, sp.pi / 9)
    check("03_rotations", f"{g}/D_1_product_of_two", product)
    check("03_rotations", f"{g}/D_1_numeric", sd.wigner_D(1, 0.3, 1.1, -0.7))
    for name in ("half_half", "one_half", "one_one", "three_half_one", "two_one", "two_two"):
        for row in reference.get("03_rotations", f"clebsch_gordan/{name}"):
            j1, m1, j2, m2, j3, m3 = row[:6].real
            assert np.isclose(float(sd.clebsch_gordan(j1, m1, j2, m2, j3, m3)), row[6].real, **TOL)
    assert sd.clebsch_gordan(1, 1, 1, -1, 2, 0) == 1 / sp.sqrt(6)
    assert sd.clebsch_gordan("1/2", "1/2", "1/2", "-1/2", 1, 1) == 0
    check("03_rotations", "euler/matrix_symbolic", sd.euler_matrix(A3, B3, C3), (A3, B3, C3))
    check("03_rotations", "euler/matrix", sd.euler_matrix(*ANGLES))
    check("03_rotations", "euler/rotate_vector", sd.euler_matrix(*ANGLES) * sp.Matrix([1, 2, 3]))


ROTATION_GROUPS = {"one_half": [1], "two_half": [1, 2], "spin1": [1], "spin32": [1]}


@pytest.mark.parametrize("group", list(ROTATION_GROUPS))
def test_rotation_operators(group):
    s = SYSTEMS[group]()
    spins = ROTATION_GROUPS[group]
    half = sp.pi / 2
    built = {
        "R_x_90": s.rotation(half, "x", spins=spins),
        "R_y_90": s.rotation(half, "y", spins=spins),
        "R_minus_x_90": s.rotation(half, "-x", spins=spins),
        "R_z_60": s.rotation(sp.pi / 3, "z", spins=spins),
        "R_phase_pi4_120": s.rotation(2 * sp.pi / 3, sp.pi / 4, spins=spins),
        "R_polar": s.rotation(sp.pi / 3, axis=(sp.pi / 4, sp.pi / 5), spins=spins),
        "R_euler": s.rotation_euler(*ANGLES, spins=spins),
        "R_flip_phase_symbolic": (s.rotation(B3, A3, spins=spins), (A3, B3)),
        "R_euler_symbolic": (s.rotation_euler(A3, B3, C3, spins=spins), (A3, B3, C3)),
    }
    for name, value in built.items():
        value, variables = value if isinstance(value, tuple) else (value, ())
        check("03_rotations", f"{group}/{name}", value, variables)
    if group == "two_half":
        check("03_rotations", f"{group}/R_spin1_only_x_90", s.rotation(half, "x", spins=[1]))
        check("03_rotations", f"{group}/R_all_x_90", s.rotation(half, "x"))


def rotated_cases():
    out = []
    one, two, spin1, spin32 = (SYSTEMS[g]() for g in ("one_half", "two_half", "spin1", "spin32"))
    out += [("one_half", "I1z", one.I(1, "z"), [1]), ("one_half", "I1x", one.I(1, "x"), [1])]
    out += [("one_half", "I1plus", one.I(1, "+"), [1])]
    out += [("two_half", "I1zI2z_spin1", two.I(1, "z") @ two.I(2, "z"), [1])]
    out += [("two_half", "I1xI2z_both", two.I(1, "x") @ two.I(2, "z"), [1, 2])]
    out += [("two_half", "T21_both", two.T([1, 2], 2, 1), [1, 2])]
    out += [("spin1", "I1z", spin1.I(1, "z"), [1]), ("spin1", "T20", spin1.T(1, 2, 0), [1])]
    out += [("spin1", "T22", spin1.T(1, 2, 2), [1]), ("spin32", "I1z", spin32.I(1, "z"), [1])]
    return out


@pytest.mark.parametrize("group, name, operator, spins", rotated_cases(), ids=lambda v: v if isinstance(v, str) else "")
def test_rotated_operators(group, name, operator, spins):
    half = sp.pi / 2
    check("03_rotations", f"{group}/{name}_by_x_90", operator.rotate(half, "x", spins=spins))
    check("03_rotations", f"{group}/{name}_by_y_90", operator.rotate(half, "y", spins=spins))
    check("03_rotations", f"{group}/{name}_by_z_60", operator.rotate(sp.pi / 3, "z", spins=spins))
    check("03_rotations", f"{group}/{name}_by_euler", operator.rotate_euler(*ANGLES, spins=spins))
    check("03_rotations", f"{group}/{name}_by_flip_symbolic", operator.rotate(B3, "x", spins=spins), (B3,))


# ------------------------------------------------------------------------------------------- the algebra itself
@pytest.mark.parametrize("group", list(SYSTEMS))
def test_products_agree_with_matrix_products(group):
    s = SYSTEMS[group]()
    labels = [label for label, _ in s.spins]
    pool = [s.I(label, c) for label in labels for c in ("x", "y", "z", "+")]
    pool += [s.T(labels[-1], k, q) for k in (1, 2) if k <= 2 * s.spins[-1][1] for q in (k, 0)]
    pool = pool[:10]
    for a, b in itertools.product(pool, repeat=2):
        assert np.allclose((a @ b).numeric(), a.numeric() @ b.numeric(), **TOL)
        assert np.allclose(a.dagger().numeric(), a.numeric().conj().T, **TOL)
    a, b, c = pool[0], pool[1], pool[-1]
    triple = (a @ b) @ c
    assert triple.equals(a @ (b @ c))
    assert s.from_matrix(triple.matrix()).equals(triple)
    assert np.isclose(complex(triple.trace()), np.trace(triple.numeric()))
    assert np.isclose(complex(a.bracket(b)), np.trace(a.numeric().conj().T @ b.numeric()))


def test_rotation_is_conjugation_by_the_rotation_operator():
    for group, spins in ROTATION_GROUPS.items():
        s = SYSTEMS[group]()
        r = s.rotation(sp.Rational(7, 10), sp.Rational(3, 10), spins=spins)
        a = s.I(spins[0], "z") + 2 * s.I(spins[-1], "x")
        sandwich = r.numeric() @ a.numeric() @ r.numeric().conj().T
        assert np.allclose(a.rotate(sp.Rational(7, 10), sp.Rational(3, 10), spins=spins).numeric(), sandwich, **TOL)


def test_product_operator_evolution():
    """The results of the planning prototype: pulse, shift and weak coupling, spin echo."""
    s = SpinSystem(2)
    i = s.I
    omega, j, t = sp.symbols("Omega1 J t", real=True)
    tau = sp.symbols("tau", positive=True)
    h = omega * i(1, "z") + 2 * sp.pi * j * (i(1, "z") @ i(2, "z"))
    assert commutator(i(1, "x"), i(1, "y")).equals(sp.I * i(1, "z"))
    rho1 = i(1, "z").rotate(sp.pi / 2, "x")
    assert rho1.equals(-i(1, "y"))
    rho2 = rho1.evolve(h, t)
    expected = [
        (i(1, "x"), sp.sin(omega * t) * sp.cos(sp.pi * j * t)),
        (i(1, "y"), -sp.cos(omega * t) * sp.cos(sp.pi * j * t)),
        (2 * i(1, "x") @ i(2, "z"), sp.cos(omega * t) * sp.sin(sp.pi * j * t)),
        (2 * i(1, "y") @ i(2, "z"), sp.sin(omega * t) * sp.sin(sp.pi * j * t)),
    ]
    total = s.zero()
    for target, amplitude in expected:
        assert sp.simplify(rho2.amplitude(target) - amplitude) == 0
        total = total + amplitude * target
    assert rho2.equals(total)
    echo = rho1.evolve(h, tau).rotate(sp.pi, "y").evolve(h, tau)
    assert sp.simplify(echo.amplitude(i(1, "y")) + sp.cos(2 * sp.pi * j * tau)) == 0
    # the closed form agrees with the matrix exponential
    values = {omega: 2 * sp.pi * 100, j: 7, t: sp.Rational(3, 1000)}
    hm = h.numeric(values)
    w, v = np.linalg.eigh(hm)
    u = (v * np.exp(-1j * w * 0.003)) @ v.conj().T
    assert np.allclose(rho2.numeric(values), u @ rho1.numeric() @ u.conj().T, **TOL)
    # the prototype listed these in another basis order; here the first spin changes fastest
    assert np.allclose(np.diag(hm).real, [325.15, -325.15, 303.16, -303.16], atol=5e-3)
    strong = omega * i(1, "z") + 2 * sp.pi * j * s.dot(1, 2)
    with pytest.raises(NotImplementedError, match="commute"):
        rho1.evolve(strong, t)
    with pytest.raises(NotImplementedError, match="spins-1/2"):
        one = SpinSystem([1])
        one.I(1, "x").evolve(one.I(1, "z"), t)


def test_kets_bras_and_input_checks():
    s = SpinSystem(2)
    up, down = sp.Rational(1, 2), -sp.Rational(1, 2)
    a, b = s.ket(up, up), s.ket(down, up)
    assert a.dagger() @ a == 1 and a.dagger() @ b == 0
    assert (s.I(1, "-") @ a).vector == b.vector
    assert (a.dagger() @ s.I(1, "+")).vector == b.dagger().vector
    assert (a @ b.dagger()).equals(s.coherence(1, 2))
    assert (a + b).normalized().norm() == 1
    assert repr(a) == "|1/2, 1/2>" and "I1x" in str(s.I(1, "x")) and s.I(1, "x")._repr_latex_() == "$I_{1x}$"
    assert sd.spins_of(s.I(1, "x") @ s.I(2, "z")) == [1, 2]
    named = SpinSystem({"H": 0.5, "D": 1})
    assert named.dimension == 6 and str(named.I("D", "z")) == "TD(1,0)"
    with pytest.raises(ValueError, match="no spin"):
        s.I(3, "x")
    with pytest.raises(ValueError, match="different spin systems"):
        s.I(1, "x") + SpinSystem(2).I(1, "x")
    with pytest.raises(TypeError, match="use @"):
        s.I(1, "x") * s.I(1, "y")
    with pytest.raises(ValueError, match="not a spin quantum number"):
        SpinSystem([0.3])
    with pytest.raises(ValueError, match="rank"):
        s.T(1, 2, 0)
    with pytest.raises(ValueError, match="same quantum number"):
        named.permutation(["H", "D"])
