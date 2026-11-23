"""Liouville space and the numeric engine against the SpinDynamica fixtures and against themselves."""

import numpy as np
import pytest
import sympy as sp

import spyndynamica as sd
from spyndynamica import SpinSystem
from tests.spyndynamica import reference

TOL = dict(rtol=0, atol=1e-11)
HALF = sp.Rational(1, 2)
SYSTEMS = {
    "one_half": lambda: SpinSystem(1),
    "two_half": lambda: SpinSystem(2),
    "three_half": lambda: SpinSystem(3),
    "spin1": lambda: SpinSystem([1]),
    "half_one": lambda: SpinSystem([HALF, 1]),
    "spin32": lambda: SpinSystem(["3/2"]),
}
B3, A3 = sp.symbols("b3 a3", real=True)


def lv(key):
    return reference.get("04_liouville", key)


def pr(key):
    return reference.get("05_propagation", key)


def stored_basis(key, dim):
    """The basis operators of a case, stored as matrices or as rows (operator, row, column, value)."""
    value = lv(key)
    if value.ndim == 3:
        return value
    out = np.zeros((dim * dim, dim, dim), dtype=complex)
    for k, i, j, element in value:
        out[int(k.real) - 1, int(i.real) - 1, int(j.real) - 1] = element
    return out


def matrices(basis):
    return np.array([op.numeric() for op in basis.operators])


# (system, constructor, fixture case, same order as SpinDynamica)
BASES = [
    ("one_half", sd.default_operator_basis, "one_half/default_basis_operators", True),
    ("one_half", sd.cartesian_product_basis, "one_half/cartesian", True),
    ("one_half", sd.spherical_tensor_basis, "one_half/spherical_tensor", True),
    ("two_half", sd.cartesian_product_basis, "two_half_bases/cartesian", True),
    ("two_half", sd.cartesian_product_basis, "two_half_bases/cartesian_sorted", True),
    ("two_half", sd.shift_and_polarization_basis, "two_half_bases/shift_and_polarization", True),
    ("two_half", sd.shift_and_z_basis, "two_half_bases/shift_and_z", True),
    ("two_half", sd.default_operator_basis, "two_half_bases/shift_and_z_default", True),
    ("two_half", sd.singlet_triplet_operator_basis, "two_half_bases/singlet_triplet", True),
    (
        "two_half",
        lambda s: sd.singlet_triplet_operator_basis(s, sort=False),
        "two_half_bases/singlet_triplet_unsorted",
        True,
    ),
    ("two_half", sd.zeeman_ket_bra_basis, "two_half_bases/zeeman_ket_bra", True),
    ("two_half", sd.zeeman_ket_bra_basis, "two_half_bases/zeeman_ket_bra_canonical", True),
    ("two_half", lambda s: sd.zeeman_ket_bra_basis(s, sort=True), "two_half_bases/zeeman_ket_bra_sorted", True),
    ("two_half", sd.spherical_tensor_basis, "two_half_bases/spherical_tensor", True),
    ("three_half", sd.cartesian_product_basis, "three_half_bases/cartesian", True),
    ("three_half", sd.shift_and_polarization_basis, "three_half_bases/shift_and_polarization", True),
    ("three_half", sd.default_operator_basis, "three_half_bases/shift_and_z_default", True),
    ("three_half", sd.zeeman_ket_bra_basis, "three_half_bases/zeeman_ket_bra", True),
    ("three_half", lambda s: sd.zeeman_ket_bra_basis(s, sort=True), "three_half_bases/zeeman_ket_bra_sorted", True),
    ("spin1", sd.zeeman_ket_bra_basis, "spin1/zeeman_ket_bra", True),
    ("spin1", sd.spherical_tensor_basis, "spin1/spherical_tensor", True),
    ("spin1", sd.default_operator_basis, "spin1/default_basis_operators", False),
    ("half_one", sd.spherical_tensor_basis, "half_one/spherical_tensor", True),
    ("half_one", sd.default_operator_basis, "half_one/default_basis_operators", False),
    ("spin32", sd.spherical_tensor_basis, "spin32_bases/spherical_tensor", True),
    ("spin32", lambda s: sd.zeeman_ket_bra_basis(s, sort=True), "spin32_bases/zeeman_ket_bra_sorted", True),
    ("spin32", sd.default_operator_basis, "spin32_bases/default_basis_operators", False),
]


@pytest.mark.parametrize("group, build, key, same_order", BASES, ids=[b[2] for b in BASES])
def test_operator_bases(group, build, key, same_order):
    system = SYSTEMS[group]()
    basis = build(system)
    mine, theirs = matrices(basis), stored_basis(key, system.dimension)
    gram = np.einsum("kij,lij->kl", mine.conj(), mine)
    assert np.allclose(gram, np.eye(len(basis)), **TOL)
    if same_order:
        assert np.allclose(mine, theirs, **TOL)
        return
    # the same operators, phases included; the sequence within one coherence order is the port's own
    overlap = np.einsum("kij,lij->kl", theirs.conj(), mine)
    match = np.argmax(np.abs(overlap), axis=1)
    assert sorted(match) == list(range(len(basis)))
    assert np.allclose(mine[match], theirs, **TOL)
    orders = [op.coherence_orders() for op in basis.operators]
    assert all(len(o) == 1 for o in orders) and [o[0] for o in orders] == sorted(o[0] for o in orders)


def test_basis_indices_transformations_and_vectors():
    s = SpinSystem(2)
    i = s.I
    spherical = sd.spherical_tensor_basis(s)
    assert np.allclose(np.array(spherical.indices, dtype=float), lv("two_half_bases/spherical_tensor_indices"), **TOL)
    by_rank = sd.spherical_tensor_basis(SpinSystem([1]), sort="rank")
    assert by_rank.indices == [(k, q) for k in range(3) for q in range(-k, k + 1)]
    cart, default = sd.cartesian_product_basis(s), sd.default_operator_basis(s)

    def number(m):
        return np.array(sp.N(m, 30).tolist(), dtype=complex)

    assert np.allclose(
        number(cart.transformation(default)), lv("two_half_bases/transformation_cartesian_from_default"), **TOL
    )
    assert np.allclose(
        number(spherical.transformation(cart)), lv("two_half_bases/transformation_spherical_from_cartesian"), **TOL
    )
    g = "two_half_vectors"
    for key, op, basis in (
        ("I1x_in_cartesian", i(1, "x"), cart),
        ("I1xI2z_in_cartesian", i(1, "x") @ i(2, "z"), cart),
        ("I1x_in_default", i(1, "x"), default),
        ("I1_dot_I2_in_spherical", s.dot(1, 2), spherical),
        ("I1plus_in_zeeman_ket_bra", i(1, "+"), sd.zeeman_ket_bra_basis(s)),
    ):
        assert np.allclose(number(basis.vector(op)).ravel(), lv(f"{g}/{key}"), **TOL), key
        assert basis.express(basis.vector(op)).equals(op)
    assert i(1, "x").bracket(i(1, "x")) == sp.nsimplify(float(lv(f"{g}/bracket_I1x_I1x")))
    assert np.isclose(complex(i(1, "+").bracket(i(1, "x"))), lv(f"{g}/bracket_I1plus_I1x"))
    assert np.isclose(
        complex(i(1, "y").bracket(sd.commutation(i(1, "z"))(i(1, "x")))), lv(f"{g}/bracket_with_superoperator")
    )
    mixture = 2 * i(1, "x") @ i(2, "z") + 3 * i(1, "y")
    assert mixture.amplitude(i(1, "x") @ i(2, "z")) == sp.nsimplify(float(lv(f"{g}/amplitude_mixture")))
    assert (i(1, "x") @ i(2, "z")).amplitude(2 * i(1, "x") @ i(2, "z")) == HALF
    assert np.isclose(complex(i(1, "x").amplitude(i(1, "+"))), lv(f"{g}/amplitude_I1x_to_I1plus"))
    assert np.isclose(complex(i(1, "x").amplitude(i(1, "-"))), lv(f"{g}/amplitude_I1x_to_I1minus"))
    symbolic = i(1, "z").rotate(B3, "x", spins=[1]).amplitude(i(1, "y"))
    for parameters, value in lv(f"{g}/amplitude_symbolic").points:
        assert np.isclose(float(symbolic.subs(B3, parameters["b3"])), value.real)


def two_spin_superoperators(s):
    i = s.I
    ham = 2 * sp.pi * 100 * i(1, "z") - 2 * sp.pi * 40 * i(2, "z") + 2 * sp.pi * 7 * s.dot(1, 2)
    return {
        "unity": sd.unity_superoperator(s),
        "commutation_I1z": sd.commutation(i(1, "z")),
        "commutation_I1x": sd.commutation(i(1, "x")),
        "commutation_hamiltonian": sd.commutation(ham),
        "left_I1x": sd.left_multiplication(i(1, "x")),
        "right_I1x": sd.right_multiplication(i(1, "x")),
        "left_I1plus": sd.left_multiplication(i(1, "+")),
        "right_I1plus": sd.right_multiplication(i(1, "+")),
        "double_commutation_I1z_I1z": sd.double_commutation(i(1, "z"), i(1, "z")),
        "double_commutation_I1x_I2z": sd.double_commutation(i(1, "x"), i(2, "z")),
        "lindbladian_I1minus": sd.lindbladian(i(1, "-")),
        "lindbladian_I1plus_I2minus": sd.lindbladian(i(1, "+"), i(2, "-")),
        "projection_I1z": sd.projection(i(1, "z")),
        "filter_order_0": sd.coherence_filter(s, [0]),
        "filter_orders_plus_minus_1": sd.coherence_filter(s, [-1, 1]),
        "filter_spin1_order_plus_1": sd.coherence_filter(s, [1], spins=[1]),
        "rotation_spin1_x_90": sd.rotation_superoperator(s, sp.pi / 2, "x", spins=[1]),
        "rotation_both_euler": sd.rotation_euler_superoperator(s, sp.pi / 5, sp.pi / 3, 2 * sp.pi / 7),
        "permutation_12": sd.permutation_superoperator(s, [1, 2]),
    }


SUPEROPERATOR_GROUPS = [
    ("two_half_superoperators", sd.default_operator_basis),
    ("two_half_superoperators_cartesian", sd.cartesian_product_basis),
]


@pytest.mark.parametrize("group, basis", SUPEROPERATOR_GROUPS)
def test_superoperator_matrices(group, basis):
    s = SpinSystem(2)
    built = two_spin_superoperators(s)
    stored = [k.split("/", 1)[1] for k in reference.cases("04_liouville") if k.startswith(group + "/")]
    assert sorted(stored) == sorted(built)
    opbasis = basis(s)
    for name, superoperator in built.items():
        assert np.allclose(superoperator.numeric(basis=opbasis), lv(f"{group}/{name}"), rtol=0, atol=1e-9), name


def test_superoperators_in_other_bases_and_systems():
    s = SpinSystem(2)
    i = s.I
    g = "two_half_superoperators_explicit_basis"
    assert np.allclose(
        sd.commutation(i(1, "z")).numeric(basis=sd.zeeman_ket_bra_basis(s)),
        lv(f"{g}/commutation_I1z_in_zeeman_ket_bra"),
        **TOL,
    )
    assert np.allclose(
        sd.lindbladian(i(1, "-")).numeric(basis=sd.spherical_tensor_basis(s)),
        lv(f"{g}/lindbladian_I1minus_in_spherical"),
        **TOL,
    )
    one = SpinSystem(1)
    default = sd.default_operator_basis(one)
    assert np.allclose(sd.commutation(one.I(1, "z")).numeric(basis=default), lv("one_half/commutation_I1z"), **TOL)
    assert np.allclose(sd.lindbladian(one.I(1, "-")).numeric(basis=default), lv("one_half/lindbladian_I1minus"), **TOL)
    assert np.allclose(
        sd.rotation_superoperator(one, sp.pi / 2, "x").numeric(basis=default), lv("one_half/rotation_x_90"), **TOL
    )

    def in_stored_basis(superoperator, key):
        """The matrix in SpinDynamica's own operator basis, built from its stored basis operators."""
        dim = superoperator.system.dimension
        vectors = stored_basis(key, dim).reshape(dim * dim, -1).T
        return vectors.conj().T @ superoperator.numeric() @ vectors

    d = SpinSystem([1])
    key = "spin1/default_basis_operators"
    t20 = d.T(1, 2, 0)
    assert np.allclose(in_stored_basis(sd.commutation(d.I(1, "z")), key), lv("spin1/commutation_I1z"), **TOL)
    assert np.allclose(in_stored_basis(sd.double_commutation(t20, t20), key), lv("spin1/double_commutation_T20"), **TOL)
    assert np.allclose(in_stored_basis(sd.coherence_filter(d, [0]), key), lv("spin1/filter_order_0"), **TOL)
    hd = SpinSystem([HALF, 1])
    zz = hd.I(1, "z") @ hd.I(2, "z")
    stored = lv("half_one/commutation_I1zI2z")
    assert np.allclose(in_stored_basis(sd.commutation(zz), "half_one/default_basis_operators"), stored, **TOL)


def test_superoperator_algebra():
    s = SpinSystem(2)
    i = s.I
    rho = i(1, "x") + 2 * i(1, "y") @ i(2, "z") + 3 * s.unity()
    a = sd.commutation(i(1, "z")) + 2 * sd.lindbladian(i(2, "-")) - sd.projection(i(1, "x")) / 3
    b = sd.rotation_superoperator(s, sp.pi / 3, "y") @ sd.coherence_filter(s, [0, 1])
    for superoperator in (a, b, a @ b, b @ a, a @ a):
        image = superoperator(rho)
        flat = superoperator.numeric() @ rho.numeric().reshape(-1)
        assert np.allclose(image.numeric().reshape(-1), flat, **TOL)
    assert (a @ rho).equals(a(rho))
    exact = np.array(sp.N(sd.commutation(i(1, "z")).matrix(), 30).tolist(), dtype=complex)
    assert np.allclose(exact, lv("two_half_superoperators/commutation_I1z"), **TOL)
    assert sd.lindbladian(i(1, "-"))(s.unity()).equals(-2 * i(1, "z"))
    with pytest.raises(ValueError, match="spins-1/2"):
        sd.cartesian_product_basis(SpinSystem([1]))
    with pytest.raises(NotImplementedError, match="one or two spins"):
        sd.spherical_tensor_basis(SpinSystem(3))
    with pytest.raises(NotImplementedError, match="spins-1/2"):
        sd.phenomenological_relaxation(SpinSystem([1]), 1.0)
    with pytest.raises(ValueError, match="2 T1"):
        sd.phenomenological_relaxation(s, 1.0, 3.0)
    assert sd.default_operator_basis(SpinSystem([2])).name == "Zeeman ket-bra"


# ------------------------------------------------------------------------------------------- propagation
def two_spin_setup():
    s = SpinSystem(2)
    i = s.I
    ham = 2 * sp.pi * 100 * i(1, "z") - 2 * sp.pi * 40 * i(2, "z") + 2 * sp.pi * 7 * s.dot(1, 2)
    weak = 2 * sp.pi * 100 * i(1, "z") - 2 * sp.pi * 40 * i(2, "z") + 2 * sp.pi * 7 * (i(1, "z") @ i(2, "z"))
    pi2x = sd.rotation_superoperator(s, sp.pi / 2, "x")
    piy = sd.rotation_superoperator(s, sp.pi, "y")
    targets = [i(1, "x"), i(1, "y"), i(1, "z"), i(1, "x") @ i(2, "z"), i(1, "y") @ i(2, "z")]
    return s, i, ham, weak, pi2x, piy, targets


def test_propagation_of_two_spins():
    s, i, ham, weak, pi2x, piy, targets = two_spin_setup()
    g = "two_half"
    z = i(1, "z")
    two = [pi2x, (ham, 0.002), (weak, 0.005)]
    assert np.allclose(sd.propagate(i(1, "x"), [(ham, 0.01)]), pr(f"{g}/free_strong_I1x_10ms"), **TOL)
    assert np.allclose(sd.propagate(i(1, "x"), [(weak, 0.01)]), pr(f"{g}/free_weak_I1x_10ms"), **TOL)
    assert np.allclose(sd.propagator([(ham, 0.01)]), pr(f"{g}/propagator_strong_10ms"), **TOL)
    assert np.allclose(sd.propagate(z, [pi2x, (ham, 0.004)]), pr(f"{g}/rotation_then_free"), **TOL)
    echo = [pi2x, (weak, 0.004), piy, (weak, 0.004)]
    assert np.allclose(sd.propagate(z, echo), pr(f"{g}/spin_echo"), **TOL)
    assert np.allclose(sd.propagate(z, two), pr(f"{g}/two_different_delays"), **TOL)
    assert np.allclose(sd.propagator(two[1:]), pr(f"{g}/propagator_two_delays"), **TOL)
    assert np.allclose(
        sd.amplitudes(z, targets, [pi2x, (weak, 0.004)]), pr(f"{g}/amplitudes_after_rotation_and_delay"), **TOL
    )
    with_background = sd.amplitudes(z, targets, [pi2x, (None, 0.004)], background=weak)
    assert np.allclose(with_background, pr(f"{g}/amplitudes_with_background"), **TOL)
    assert np.isclose(sd.amplitudes(z, i(1, "y"), echo), pr(f"{g}/amplitude_spin_echo"), **TOL)
    # the weak-coupling cases also follow from the closed form of the symbolic core
    symbolic = z.rotate(sp.pi / 2, "x").evolve(weak, sp.Rational(4, 1000))
    assert np.allclose(symbolic.numeric(), sd.propagate(z, [pi2x, (weak, 0.004)]), **TOL)
    # right to left: an argument of the call (SpinDynamica has only the global setting, see DIVERGENCES.md D2)
    rl = "two_half_right_to_left"
    assert np.allclose(sd.propagate(z, two, order="right_to_left"), pr(f"{rl}/two_different_delays"), **TOL)
    assert np.allclose(sd.propagate(z, echo[::-1], order="right_to_left"), pr(f"{rl}/spin_echo"), **TOL)
    assert np.allclose(sd.propagator(two[1:], order="right_to_left"), pr(f"{rl}/propagator_two_delays"), **TOL)
    assert np.isclose(sd.amplitudes(z, i(1, "y"), two, order="right_to_left"), pr(f"{rl}/amplitude"), **TOL)


def test_propagation_with_relaxation():
    s, i, ham, weak, pi2x, _, targets = two_spin_setup()
    g = "two_half_relaxation"
    default = sd.default_operator_basis(s)
    relax = sd.phenomenological_relaxation(s, 2.0, 0.2)
    assert np.allclose(relax.numeric(basis=default), pr(f"{g}/relaxation_superoperator"), **TOL)
    per_spin = sd.phenomenological_relaxation(s, {1: (2.0, 0.2), 2: (1.0, 0.5)})
    assert np.allclose(per_spin.numeric(basis=default), pr(f"{g}/relaxation_superoperator_per_spin"), **TOL)
    single = sd.phenomenological_relaxation(s, 0.5)
    assert np.allclose(single.numeric(basis=default), pr(f"{g}/relaxation_superoperator_single_time"), **TOL)
    assert np.allclose(sd.liouvillian(ham, relax).numeric(basis=default), pr(f"{g}/liouvillian"), rtol=0, atol=1e-9)
    assert np.allclose(sd.propagate(i(1, "x"), [(ham, 0.05)], background=relax), pr(f"{g}/decay_I1x_50ms"), **TOL)
    assert np.allclose(sd.propagate(i(1, "z"), [(relax, 0.5)]), pr(f"{g}/decay_I1z_500ms"), **TOL)
    values = sd.amplitudes(i(1, "z"), targets, [pi2x, (weak, 0.02)], background=relax)
    assert np.allclose(values, pr(f"{g}/amplitudes_with_relaxation"), **TOL)
    t1, t2 = sp.symbols("T1 T2", positive=True)
    symbolic = sd.phenomenological_relaxation(s, t1, t2)
    assert symbolic(i(1, "z")).equals(-i(1, "z") / t1) and symbolic(i(1, "+") @ i(2, "z")).equals(
        -(1 / t2 + 1 / t1) * i(1, "+") @ i(2, "z")
    )
    assert np.allclose(symbolic.numeric({t1: 2.0, t2: 0.2}), relax.numeric(), **TOL)
    with pytest.raises(ValueError, match="not unitary"):
        sd.propagator([(ham, 0.01)], background=relax)
    step = sd.superpropagator([(ham, 0.05)], background=relax)
    assert np.allclose((step @ i(1, "x").numeric().reshape(-1)).reshape(4, 4), pr(f"{g}/decay_I1x_50ms"), **TOL)


def test_pulses_delays_and_repetition():
    s = SpinSystem(1)
    x, y, z = s.I(1)
    g = "one_half_pulses"
    targets = [x, y, z]
    offset = 2 * sp.pi * 300 * z
    assert np.allclose(sd.amplitudes(z, targets, [sd.Pulse(sp.pi / 2, "x")]), pr(f"{g}/pulse_90x_on_Iz"), **TOL)
    assert np.allclose(sd.amplitudes(z, targets, [sd.Pulse(sp.pi / 2, "y")]), pr(f"{g}/pulse_90y_on_Iz"), **TOL)
    assert np.allclose(
        sd.amplitudes(z, targets, [sd.Pulse(sp.pi / 2, sp.pi / 4)]), pr(f"{g}/pulse_90_phase_pi4_on_Iz"), **TOL
    )
    finite = [sd.Pulse(sp.pi / 2, "x", nutation_frequency=2 * sp.pi * 1000)]
    assert np.allclose(
        sd.amplitudes(z, targets, finite, background=offset), pr(f"{g}/pulse_finite_off_resonance"), **TOL
    )
    by_duration = [sd.Pulse(sp.pi / 2, "x", duration=sp.Rational(1, 4000))]
    assert np.allclose(
        sd.amplitudes(z, targets, by_duration, background=offset), pr(f"{g}/pulse_finite_off_resonance"), **TOL
    )
    sequence = [sd.Pulse(sp.pi / 2, "x"), sd.Delay(0.001), sd.Pulse(sp.pi, "y"), sd.Delay(0.001)]
    assert np.allclose(sd.amplitudes(z, targets, sequence, background=offset), pr(f"{g}/pulse_delay_sequence"), **TOL)
    repeated = [sd.Repeat([sd.Pulse(sp.pi / 6, "x")], 3)]
    assert np.allclose(sd.amplitudes(z, targets, repeated), pr(f"{g}/repeated_pulse"), **TOL)
    for parameters, value in pr(f"{g}/symbolic_flip_angle").points:
        got = sd.amplitudes(z, targets, [sd.Pulse(B3, A3)], values={A3: parameters["a3"], B3: parameters["b3"]})
        assert np.allclose(got, value, **TOL)
    with pytest.raises(TypeError, match="not an event"):
        sd.propagate(z, ["pulse"])
    with pytest.raises(ValueError, match="order must be"):
        sd.propagate(z, [sd.Pulse(1)], order="backwards")


def test_trajectory():
    s, i, ham, weak, pi2x, _, targets = two_spin_setup()
    times, values = sd.trajectory(i(1, "z"), targets, [pi2x, (ham, 0.01)], points=20)
    assert values.shape == (5, 22) and times[0] == times[1] == 0 and np.isclose(times[-1], 0.01)
    assert np.allclose(values[:, 0], [0, 0, 1, 0, 0], **TOL) and np.allclose(values[:, 1], [0, -1, 0, 0, 0], **TOL)
    for n in (7, 21):
        expected = sd.amplitudes(i(1, "z"), targets, [pi2x, (ham, times[n])])
        assert np.allclose(values[:, n], expected, **TOL)
    relax = sd.phenomenological_relaxation(s, 2.0, 0.2)
    times, decay = sd.trajectory(i(1, "x"), i(1, "x"), [sd.Delay(0.1)], points=10, background=relax)
    assert np.allclose(decay, np.exp(-times / 0.2), **TOL)
