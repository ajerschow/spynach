"""Density operators, secularisation, thermalised relaxation, amplitude tables and maxima."""

import math

import numpy as np
import pytest
import sympy as sp

import spyndynamica as sd
from spyndynamica import SpinSystem
from tests.spyndynamica import reference

TOL = dict(rtol=0, atol=1e-11)


def stored(key):
    return reference.get("09_states", key)


def pair():
    s = SpinSystem(2)
    i = s.I
    hlab = -2 * sp.pi * 400e6 * i(1, "z") - 2 * sp.pi * 100e6 * i(2, "z") + 2 * sp.pi * 7 * s.dot(1, 2)
    return s, i, hlab


def test_thermal_equilibrium():
    s, i, hlab = pair()
    g = "two_half"
    assert np.allclose(sd.thermal_equilibrium(hlab, 300).numeric(), stored(f"{g}/thermal_300K"), rtol=0, atol=1e-15)
    assert np.allclose(
        sd.thermal_equilibrium(hlab, 300).numeric(), stored(f"{g}/thermal_list_syntax"), rtol=0, atol=1e-15
    )
    assert np.allclose(sd.thermal_equilibrium(hlab, 0.01).numeric(), stored(f"{g}/thermal_10mK"), **TOL)
    exact = sd.thermal_equilibrium(hlab, 0.01, high_temperature=False)
    assert np.allclose(exact.numeric(), stored(f"{g}/thermal_10mK_exact"), **TOL)
    assert np.isclose(complex(exact.trace()), 1.0) and np.isclose(
        complex(sd.thermal_equilibrium(hlab, 300).trace()), 1.0
    )
    # at high temperature the two agree; the polarization of spin 1 is hbar omega / (2 kB T)
    hot = sd.thermal_equilibrium(hlab, 300, high_temperature=False).numeric()
    assert np.allclose(hot, sd.thermal_equilibrium(hlab, 300).numeric(), rtol=0, atol=1e-9)
    k = sd.constants()
    temperature, omega = sp.symbols("T omega", positive=True)
    symbolic = sd.thermal_equilibrium(-omega * i(1, "z"), temperature)
    expected = k["hbar"] / k["kB"] / 4 * omega / temperature
    assert (
        sp.simplify(symbolic.terms[("z", "e")] / expected - 1).evalf(12).is_zero
        or abs(float((symbolic.terms[("z", "e")] / expected).subs({omega: 1, temperature: 1})) - 1) < 1e-12
    )


def test_polarized_states():
    s, i, _ = pair()
    g = "two_half"
    p = sp.Rational(3, 10)
    assert np.allclose(sd.polarized(s, p, "z", spins=[1]).numeric(), stored(f"{g}/polarized_spin_1_z"), **TOL)
    each = sd.polarized(s, {1: (p, "x"), 2: (sp.Rational(1, 10), "z")})
    assert np.allclose(each.numeric(), stored(f"{g}/polarized_each"), **TOL)
    # D5 in DIVERGENCES.md: SpinDynamica's short form returns the fully polarized state for any level and axis
    full = sd.polarized(s, 1, "z").numeric()
    for case in ("polarized_all_z", "polarized_all_x", "polarized_full"):
        assert np.allclose(stored(f"{g}/{case}"), full, **TOL)
    all_z = sd.polarized(s, p, "z")
    assert all_z.equals((s.unity() / 2 + p * i(1, "z")) @ (s.unity() / 2 + p * i(2, "z")))
    assert all_z.trace() == 1 and all_z.amplitude(i(1, "z")) == p / 2
    for case, level in (("singlet_polarized", sp.Rational(4, 10)), ("singlet_polarized_full", 1),
                        ("singlet_polarized_negative", -sp.Rational(1, 3))):  # fmt: skip
        assert np.allclose(sd.singlet_polarized(s, level).numeric(), stored(f"{g}/{case}"), **TOL)
    st = s.singlet_triplet()
    assert sd.singlet_polarized(s, 1).equals(st.ket("S0") @ st.ket("S0").dagger())
    three = SpinSystem(3)
    assert sd.singlet_polarized(three, p, [2, 3]).trace() == 1
    with pytest.raises(ValueError, match="give the spin pairs"):
        sd.singlet_polarized(three, p)
    with pytest.raises(NotImplementedError, match="spins-1/2"):
        sd.polarized(SpinSystem([1]), p)


def test_secularize():
    s, i, _ = pair()
    g = "two_half"
    mixed = i(1, "x") @ i(2, "x") + 2 * i(1, "z") @ i(2, "z") + 3 * i(1, "x") + i(1, "+") @ i(2, "-") + 5 * i(2, "z")
    zeeman = 2 * sp.pi * 100 * i(1, "z") + 2 * sp.pi * 30 * i(2, "z")
    assert np.allclose(sd.secularize(mixed, zeeman).numeric(), stored(f"{g}/secularize_hamiltonian"), **TOL)
    assert np.allclose(sd.secularize(mixed, zeeman, 500).numeric(), stored(f"{g}/secularize_threshold"), **TOL)
    homonuclear = sd.secularize(mixed, [[1, 2]])
    assert np.allclose(homonuclear.numeric(), stored(f"{g}/secularize_homonuclear"), **TOL)
    assert np.allclose(sd.secularize(mixed, [[1], [2]]).numeric(), stored(f"{g}/secularize_heteronuclear"), **TOL)
    assert sd.secularize(mixed, [[1], [2]]).equals(2 * i(1, "z") @ i(2, "z") + 5 * i(2, "z"))
    assert homonuclear.commutes_with(i(None, "z")) and not mixed.commutes_with(i(None, "z"))
    # the secular part of a dipolar coupling of unlike spins is its zz term
    dipolar = 3 * i(1, "z") @ i(2, "z") - s.dot(1, 2)
    assert sd.secularize(dipolar, [[1], [2]]).equals(2 * i(1, "z") @ i(2, "z"))


def test_thermalized_relaxation():
    s, i, hlab = pair()
    g = "two_half"
    relaxation = sd.phenomenological_relaxation(s, 2.0, 0.2)
    basis = sd.default_operator_basis(s)
    for case, temperature, options in (("300K", 300, {}), ("10mK", 0.01, {}),
                                       ("10mK_exact", 0.01, {"high_temperature": False})):  # fmt: skip
        thermalized = sd.thermalize(relaxation, hlab, temperature, **options)
        assert np.allclose(thermalized.numeric(basis=basis), stored(f"{g}/thermalized_relaxation_{case}"), **TOL), case
    # the equilibrium state is stationary, and any state of trace one relaxes to it
    equilibrium = sd.thermal_equilibrium(hlab, 0.01)
    thermalized = sd.thermalize(relaxation, hlab, 0.01)
    assert np.allclose(thermalized(equilibrium).numeric(), 0, **TOL)
    final = sd.propagate(i(1, "x") + s.unity() / 4, [(thermalized, 200.0)])
    assert np.allclose(final, equilibrium.numeric(), atol=1e-9)


def test_amplitude_tables_and_maxima():
    one = SpinSystem(1)
    x, y, z = one.I(1)
    beta, tau = sp.symbols("beta tau", real=True)
    flip = [sd.rotation_superoperator(one, beta, "x")]
    grid, table = sd.amplitude_table(z, [x, y, z], [sd.Pulse(beta, "x")], beta, np.linspace(0, math.pi, 5))
    reference_table = stored("one_half/table_flip_angle")
    assert np.allclose(grid, reference_table[0, :, 0].real, **TOL) and np.allclose(
        table, reference_table[:, :, 1], **TOL
    )
    grid, column = sd.amplitude_table(z, y, flip, beta, [0, sp.Rational(1, 4), sp.Rational(1, 2), sp.Rational(3, 4), 1])
    assert np.allclose(np.c_[grid, column], stored("one_half/table_step"), **TOL)
    delay = [(2 * sp.pi * 100 * z, tau)]
    grid, table = sd.amplitude_table(x, [x, y, z], delay, tau, np.linspace(0, 0.01, 5))
    assert np.allclose(table, stored("one_half/table_delay")[:, :, 1], **TOL)
    maximum, best = sd.maximize_amplitude(z, -y, [sd.Pulse(beta, "x")], {beta: 1.0})
    theirs = stored("one_half/maximize_flip_angle")
    assert np.isclose(maximum, theirs[0].real, atol=1e-9) and np.isclose(best[beta], theirs[1].real, atol=1e-5)
    events = [sd.Pulse(beta, "x"), (2 * sp.pi * 100 * z, tau)]
    maximum, best = sd.maximize_amplitude(z, x, events, {beta: 1.0, tau: 0.001}, positive=True)
    assert np.isclose(maximum, 1.0, atol=1e-8) and best[tau] > 0
    assert np.isclose(math.sin(best[beta]) * math.sin(2 * math.pi * 100 * best[tau]), 1.0, atol=1e-8)
    assert np.isclose(stored("one_half/maximize_two_parameters")[0].real, 1.0, atol=1e-9)
    maximum, best = sd.maximize_amplitude(z, -y, [sd.Pulse(beta, "x")], {beta: 1.0}, bounds={beta: (0, 1.2)})
    assert np.isclose(best[beta], 1.2) and np.isclose(maximum, math.sin(1.2))
