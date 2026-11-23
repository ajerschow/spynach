"""Pulse sequence objects, shaped pulses, stored propagators, normalisation, time origin and the 2D transform."""

import math

import numpy as np
import pytest
import sympy as sp

import spyndynamica as sd
from spyndynamica import Delay, Pulse, PulseSequence, SpinSystem
from tests.spyndynamica import reference

TOL = dict(rtol=0, atol=1e-10)
ODE = dict(rtol=0, atol=5e-7)
TAU, T = sp.symbols("tau t", real=True)


def stored(key):
    return reference.get("10_sequences", key)


def top_level_signal(offset):
    """A function at the top level of a module, as process pools need."""
    return np.array([math.cos(offset), math.sin(offset)])


def test_pulse_sequences_and_pulse_options():
    s = SpinSystem(1)
    x, y, z = s.I(1)
    offset = 2 * sp.pi * 300 * z
    echo = [Pulse(sp.pi / 2, "x"), Delay(0.001), Pulse(sp.pi, "y"), Delay(0.001)]

    def amp(events, **options):
        return sd.amplitudes(z, [x, y, z], events, **options)

    g = "one_half"
    assert np.allclose(amp([PulseSequence(echo)], background=offset), stored(f"{g}/sequence"), **TOL)
    assert np.allclose(amp(PulseSequence(echo), background=offset), stored(f"{g}/sequence"), **TOL)
    listed = amp([PulseSequence(echo), Pulse(sp.pi / 3, "x")], background=offset)
    assert np.allclose(listed, stored(f"{g}/sequence_in_a_list"), **TOL)
    shifted = amp([PulseSequence(echo, phase_shift=sp.pi / 2)], background=offset)
    assert np.allclose(shifted, stored(f"{g}/sequence_phase_shift"), **TOL)
    ratio = PulseSequence(
        [Pulse(sp.pi / 2, "x"), Delay(("auto", 1)), Pulse(sp.pi / 3, "y"), Delay(("auto", 2))], duration=0.004
    )
    assert np.allclose(amp([ratio], background=offset), stored(f"{g}/sequence_total_duration"), **TOL)
    explicit = [Pulse(sp.pi / 2, "x"), Delay(0.004 / 3), Pulse(sp.pi / 3, "y"), Delay(0.008 / 3)]
    assert np.allclose(amp([ratio], background=offset), amp(explicit, background=offset), **TOL)
    one = PulseSequence([Pulse(sp.pi / 2, "x"), Delay(0.001), Pulse(sp.pi / 3, "y"), Delay("auto")], duration=0.004)
    assert np.allclose(amp([one], background=offset), stored(f"{g}/sequence_one_automatic_delay"), **TOL)
    repeated = sd.Repeat([PulseSequence([Pulse(sp.pi / 5, "x"), Delay(0.0005)])], 3)
    assert np.allclose(amp([repeated], background=offset), stored(f"{g}/sequence_repeated"), **TOL)
    assert np.allclose(amp([Pulse(sp.pi / 3, "x", phase_shift=sp.pi / 2)]), stored(f"{g}/pulse_phase_shift"), **TOL)
    assert np.allclose(amp([Pulse(sp.pi / 3, "x", phase_shift=sp.pi / 2)]), amp([Pulse(sp.pi / 3, "y")]), **TOL)
    weak = Pulse(sp.pi / 2, "x", amplitude_factor=0.9)
    assert np.allclose(amp([weak]), stored(f"{g}/pulse_amplitude_factor"), **TOL)
    finite = Pulse(sp.pi / 2, "x", amplitude_factor=0.9, nutation_frequency=2 * sp.pi * 1000)
    assert np.allclose(amp([finite], background=offset), stored(f"{g}/pulse_amplitude_factor_finite"), **TOL)
    assert np.allclose(amp([Pulse(sp.pi / 2, "x")], normalization=2), stored(f"{g}/normalization_factor"), **TOL)
    with pytest.raises(ValueError, match="need the total duration"):
        amp([PulseSequence([Delay("auto")])])
    with pytest.raises(ValueError, match="longer than the total duration"):
        amp([PulseSequence([Delay(0.01), Delay("auto")], duration=0.004)])
    two = SpinSystem(2)
    i = two.I
    targets = [i(1, "y"), i(2, "y"), i(1, "z"), i(2, "z")]
    on_one = PulseSequence([Pulse(sp.pi / 2, "x"), Delay(0.001)], spins=[2])
    values = sd.amplitudes(i(None, "z"), targets, [on_one], background=2 * sp.pi * 100 * i(2, "z"))
    assert np.allclose(values, stored("two_half/sequence_on_one_spin"), **TOL)


def test_shaped_pulses_stored_propagators_and_time_origin():
    s = SpinSystem(1)
    x, y, z = s.I(1)
    g = "one_half"

    def amp(events, **options):
        return sd.amplitudes(z, [x, y, z], events, **options)

    gaussian = sd.ShapedPulse(0.01, 2 * sp.pi * 80 * sp.exp(-((TAU / 0.002) ** 2)), tau=TAU)
    assert np.allclose(amp([gaussian]), stored(f"{g}/shaped_pulse_gaussian"), **ODE)
    # on resonance the flip angle is the area of the shape
    area = 2 * math.pi * 80 * 0.002 * math.sqrt(math.pi) * math.erf(2.5)
    assert np.allclose(amp([gaussian]), [0, -math.sin(area), math.cos(area)], atol=1e-8)
    general = sd.ShapedPulse(0.01, 2 * sp.pi * 60, sp.pi / 4, 2 * sp.pi * 40, tau=TAU)
    background = 2 * sp.pi * 25 * z
    assert np.allclose(amp([general], background=background), stored(f"{g}/shaped_pulse_phase_and_frequency"), **ODE)
    # a constant shape is a rectangular pulse off resonance by the frequency offset
    tilted = 2 * sp.pi * 60 * s.I(1, sp.pi / 4) + 2 * sp.pi * (25 - 40) * z
    assert np.allclose(amp([general], background=background), amp([(tilted, 0.01)]), atol=1e-8)
    assert np.allclose(amp([sd.ShapedPulse(0.01, 2 * sp.pi * 80, tau=TAU, amplitude_factor=0.5)]),
                       amp([(2 * sp.pi * 40 * x, 0.01)]), atol=1e-8)  # fmt: skip
    offset = 2 * sp.pi * 300 * z
    echo = [Pulse(sp.pi / 2, "x"), Delay(0.001), Pulse(sp.pi, "y"), Delay(0.001)]
    block = sd.precalculate(echo, background=offset)
    assert block.unitary and np.isclose(block.duration, 0.002)
    assert np.allclose(amp([block, Pulse(sp.pi / 3, "x")], background=offset), stored(f"{g}/precalculated"), **TOL)
    relaxation = sd.phenomenological_relaxation(s, 0.5, 0.1)
    lossy = sd.precalculate(echo, background=[offset, relaxation])
    assert not lossy.unitary
    assert np.allclose(amp([lossy]), amp(echo, background=[offset, relaxation]), **TOL)
    field = 2 * sp.pi * 50 * (x * sp.cos(2 * sp.pi * 300 * T) + y * sp.sin(2 * sp.pi * 300 * T)) + 2 * sp.pi * 280 * z
    late = amp([(sd.TimeDependent(field, T), 0.003)], start_time=0.001)
    assert np.allclose(late, stored(f"{g}/initial_time_point"), **ODE)
    assert not np.allclose(late, amp([(sd.TimeDependent(field, T), 0.003)]), atol=1e-3)


def test_filtration_events_normalised_signals_and_the_two_dimensional_transform():
    s = SpinSystem(2)
    i = s.I
    zz = (2 * sp.pi * 10 * i(1, "z") @ i(2, "z"), 0.025)
    events = [Pulse(sp.pi / 2, "x"), zz, Pulse(sp.pi / 2, "x"), sd.coherence_filter(s, [-2, 2]), Pulse(sp.pi / 3, "y")]
    targets = [i(1, "x") @ i(2, "x"), i(1, "z") @ i(2, "z"), i(1, "y") @ i(2, "y")]
    assert np.allclose(sd.amplitudes(i(None, "z"), targets, events), stored("two_half/filtration_event"), **TOL)
    # the filter keeps the double-quantum part of the antiphase term the two pulses make
    before = sd.propagate(i(None, "z"), events[:3])
    kept = sd.propagate(i(None, "z"), events[:4])
    orders = np.subtract.outer(np.diag(i(None, "z").numeric()), np.diag(i(None, "z").numeric()))
    assert np.allclose(kept, np.where(np.abs(orders) == 2, before, 0), **TOL) and np.abs(kept).max() > 0.2
    one_spin = sd.amplitudes(
        i(None, "x"), [i(1, "x"), i(2, "x"), i(1, "-"), i(2, "+")], [sd.coherence_filter(s, [1], spins=[1])]
    )
    assert np.allclose(one_spin, stored("two_half/filtration_one_spin"), **TOL)
    signal = sd.signal_1d(
        2 * sp.pi * 30 * i(1, "z"), points=17, bandwidth=2 * sp.pi * 200, line_broadening=None, normalization=2
    )
    assert np.allclose(signal.values, stored("two_half/signal_normalization_factor")[:, 1], **TOL)
    grid = np.array([[np.exp(-(j + 2 * k) / 5) * np.exp(1j * (j - k)) for k in range(4)] for j in range(4)])
    assert np.allclose(sd.ft2(grid), stored("one_half/ft_two_dimensional"), **TOL)
    assert np.allclose(sd.ft2(grid, zero_fill=False), stored("one_half/ft_two_dimensional_no_zero_fill"), **TOL)


def test_random_orientations_and_process_pools():
    angles, weights = sd.random_orientations(20000, seed=1)
    assert angles.shape == (20000, 3) and np.isclose(weights.sum(), 1.0)
    assert abs((weights * (3 * np.cos(angles[:, 1]) ** 2 - 1)).sum()) < 0.03 and abs(np.cos(angles[:, 1]).mean()) < 0.02
    beta_only, _ = sd.random_orientations(10, "beta", seed=2)
    assert np.all(beta_only[:, [0, 2]] == 0)
    assert np.array_equal(sd.random_orientations(5, seed=3)[0], sd.random_orientations(5, seed=3)[0])
    with pytest.raises(ValueError, match="angles is"):
        sd.random_orientations(5, "gamma")
    offsets = [0.1, 0.7, 1.9, 2.5]
    serial = sd.ensemble_average(top_level_signal, offsets)
    assert np.allclose(sd.ensemble_average(top_level_signal, offsets, processes=2), serial, **TOL)
    assert np.allclose(sd.ensemble_average(top_level_signal, offsets, workers=2), serial, **TOL)
