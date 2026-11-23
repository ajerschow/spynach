"""Time-dependent Hamiltonians, ensemble averages, orientation sets and spherical tensors."""

import math

import numpy as np
import pytest
import sympy as sp

import spyndynamica as sd
from spyndynamica import SpinSystem, TimeDependent
from tests.spyndynamica import reference

TOL = dict(rtol=0, atol=1e-11)
ODE = dict(rtol=0, atol=5e-7)  # SpinDynamica integrates with NDSolve's default tolerances
T = sp.Symbol("t", real=True)
ANGLES = (sp.pi / 5, sp.pi / 3, 2 * sp.pi / 7)


def stored(key):
    return reference.get("08_time_dependence", key)


def one_spin():
    s = SpinSystem(1)
    x, y, z = s.I(1)
    return s, x, y, z


def test_time_dependent_hamiltonians():
    s, x, y, z = one_spin()
    field = 2 * sp.pi * 50 * (x * sp.cos(2 * sp.pi * 300 * T) + y * sp.sin(2 * sp.pi * 300 * T))
    rf = field + 2 * sp.pi * 280 * z
    g = "one_half"
    assert np.allclose(sd.propagate(z, [(TimeDependent(rf, T), 0.004)]), stored(f"{g}/rotating_field"), **ODE)
    values = sd.amplitudes(z, [x, y, z], [(TimeDependent(rf, T), 0.004)])
    assert np.allclose(values, stored(f"{g}/rotating_field_amplitudes"), **ODE)
    # t is the time since the start of the sequence, not of the event
    delayed = [(2 * sp.pi * 100 * z, 0.001), sd.Delay(0.003, TimeDependent(rf, T))]
    assert np.allclose(sd.amplitudes(z, [x, y, z], delayed), stored(f"{g}/rotating_field_after_a_delay"), **ODE)
    on_background = sd.amplitudes(z, [x, y, z], [(TimeDependent(field, T), 0.004)], background=2 * sp.pi * 280 * z)
    assert np.allclose(on_background, stored(f"{g}/rotating_field_on_background"), **ODE)
    chirp = 2 * sp.pi * 40 * x + 2 * sp.pi * (-200 + 40000 * T) * z
    assert np.allclose(sd.amplitudes(z, [x, y, z], [(TimeDependent(chirp, T), 0.01)]), stored(f"{g}/chirp"), **ODE)
    # exact: in the frame rotating with the field the Hamiltonian is constant
    tilted = 2 * sp.pi * 50 * x + 2 * sp.pi * (280 - 300) * z
    frame = sd.propagate(z, [(tilted, 0.004)])
    back = sd.propagator([(2 * sp.pi * 300 * z, 0.004)])
    assert np.allclose(sd.propagate(z, [(TimeDependent(rf, T), 0.004)]), back @ frame @ back.conj().T, atol=1e-9)
    # with relaxation the same equation is integrated in Liouville space (SpinDynamica fails on this input)
    relaxation = sd.phenomenological_relaxation(s, 0.05, 0.01)
    slow = sd.amplitudes(z, [x, y, z], [(TimeDependent(chirp, T), 0.01)], background=relaxation)
    steps = [(chirp.subs(T, (n + 0.5) * 1e-6), 1e-6) for n in range(10000)]
    assert np.allclose(slow, sd.amplitudes(z, [x, y, z], steps, background=relaxation), atol=1e-6)
    times, track = sd.trajectory(z, [x, y, z], [(TimeDependent(rf, T), 0.004)], points=4)
    assert np.allclose(track[:, -1], stored(f"{g}/rotating_field_amplitudes"), **ODE) and len(times) == 5
    amplitude = sp.Symbol("w1", real=True)
    scaled = sd.propagate(z, [(TimeDependent(amplitude * rf, T), 0.004)], values={amplitude: 1})
    assert np.allclose(scaled, stored(f"{g}/rotating_field"), **ODE)


def test_ensemble_averages():
    s, x, y, z = one_spin()
    offset = sp.Symbol("offset", real=True)
    events = [sd.Pulse(sp.pi / 2, "x"), (offset * z, 0.002)]

    def amplitudes(value):
        return sd.amplitudes(z, [x, y, z], events, values={offset: value})

    offsets = [2 * math.pi * 50, 2 * math.pi * 100, 2 * math.pi * 200]
    assert np.allclose(sd.ensemble_average(amplitudes, offsets), stored("one_half/average_over_offsets"), **TOL)
    weighted = sd.ensemble_average(amplitudes, offsets, [1 / 2, 1 / 3, 1 / 6], workers=2)
    assert np.allclose(weighted, stored("one_half/weighted_average_over_offsets"), **TOL)
    # a powder average of a signal: an axially symmetric shift anisotropy
    angles, weights = sd.orientations("ZCW50")

    def signal(alpha, beta, gamma):
        shift = 2 * math.pi * 500 * (3 * math.cos(beta) ** 2 - 1) / 2
        return sd.signal_1d(shift * z, points=32, bandwidth=2 * math.pi * 4000, line_broadening=2 * math.pi * 20)

    powder = sd.ensemble_average(signal, angles, weights)
    t = powder.times
    shifts = 2 * math.pi * 500 * (3 * np.cos(angles[:, 1]) ** 2 - 1) / 2
    expected = (weights[:, None] * np.exp((1j * shifts[:, None] - 2 * math.pi * 10) * t[None, :])).sum(axis=0)
    assert powder.peaks is None and np.allclose(powder.values, expected, **TOL)
    assert np.isclose(powder.values[0], 1.0)
    with pytest.raises(ValueError, match="one weight per sample"):
        sd.ensemble_average(amplitudes, offsets, [1.0])
    with pytest.raises(ValueError, match="no samples"):
        sd.ensemble_average(amplitudes, [])


def test_orientation_sets():
    names = reference.get("07_tables", "orientations/tabulated_names")
    assert sd.orientation_schemes() == sorted(names)
    summaries = reference.get("07_tables", "orientations/tabulated_summaries")
    for name, summary in zip(names, summaries):
        angles, weights = sd.orientations(name)
        table = np.c_[angles, weights]
        assert np.allclose(table, stored(f"orientations/{name}"), **TOL)
        assert len(table) == summary[0] and np.allclose(table.sum(axis=0), summary[1:5], atol=1e-9)
        assert np.isclose(weights.sum(), 1.0, atol=1e-6)
    angles, weights = sd.step_beta(4)
    assert np.allclose(np.c_[angles, weights], reference.get("07_tables", "orientations/step_beta_4"), **TOL)
    angles, weights = sd.step_alpha_beta(3, 4)
    assert np.allclose(np.c_[angles, weights], reference.get("07_tables", "orientations/step_alpha_beta_3_4"), **TOL)
    # the sets integrate the rank-2 spherical harmonic to zero, as a powder average must
    for name in ("ZCW538", "REPULSION376", "Leboct85"):
        angles, weights = sd.orientations(name)
        assert abs((weights * (3 * np.cos(angles[:, 1]) ** 2 - 1)).sum()) < 2e-3, name
    with pytest.raises(ValueError, match="unknown orientational sampling set"):
        sd.orientations("ZCW51")


def test_spherical_tensors():
    cart = [[1, 2, 3], [4, 5, 6], [7, 8, 10]]
    symmetric = [[1, 2, 3], [2, 5, 6], [3, 6, 10]]
    for rank in (0, 1, 2):
        components = sd.cartesian_to_spherical(rank, cart)
        assert np.allclose(components, stored(f"tensors/cartesian_to_rank_{rank}")[1], **TOL)
        assert np.allclose(sd.spherical_to_cartesian(components), stored(f"tensors/rank_{rank}_to_cartesian"), **TOL)
    parts = sum(sd.spherical_to_cartesian(sd.cartesian_to_spherical(rank, cart)) for rank in (0, 1, 2))
    assert np.allclose(parts, cart, **TOL)
    vector = sd.cartesian_to_spherical(1, [1, 2, 3])
    assert np.allclose(vector, stored("tensors/vector_to_rank_1")[1], **TOL)
    tensor = sd.cartesian_to_spherical(2, symmetric)
    assert np.allclose(tensor, stored("tensors/rank_2_symmetric")[1], **TOL)
    assert np.allclose(sd.transform_spherical_tensor(tensor, ANGLES), stored("tensors/rank_2_rotated")[1], **TOL)
    assert np.allclose(sd.transform_spherical_tensor(vector, ANGLES), stored("tensors/rank_1_rotated")[1], **TOL)
    twice = sd.transform_spherical_tensor(tensor, ANGLES, (sp.pi / 7, sp.pi / 4, sp.pi / 9))
    assert np.allclose(twice, stored("tensors/rank_2_rotated_twice")[1], **TOL)
    # the transformation is the passive rotation of the Cartesian tensor: A' = R^T A R
    r = np.array(sp.N(sd.euler_matrix(*ANGLES)).tolist(), dtype=float)
    rotated = sd.cartesian_to_spherical(2, r.T @ np.array(symmetric) @ r)
    assert np.allclose(sd.transform_spherical_tensor(tensor, ANGLES), rotated, **TOL)
    assert np.allclose(sd.cartesian_to_spherical(0, 5.0), [5.0])
    with pytest.raises(ValueError, match="rank-1 part only"):
        sd.cartesian_to_spherical(2, [1, 2, 3])
    with pytest.raises(ValueError, match="1, 3 or 5"):
        sd.spherical_to_cartesian([1, 2])
