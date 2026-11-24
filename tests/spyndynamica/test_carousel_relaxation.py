"""The carousel average of the COMPUTE method and signals under time-dependent Hamiltonians with relaxation."""

import math

import numpy as np
import pytest
import sympy as sp

import spyndynamica as sd
from spyndynamica import Periodic, SpinSystem, TimeDependent
from tests.spyndynamica import reference

ODE = dict(rtol=0, atol=5e-6)  # SpinDynamica integrates with NDSolve's default tolerances
T = sp.Symbol("t", real=True)
PERIOD = sp.Rational(1, 1000)
KW = dict(points=65, bandwidth=2 * sp.pi * 8000)
LOG2_DIM = 2  # of the two-spin system, see divergence D13


def stored(key):
    return reference.get("15_carousel_relaxation", key)


def spinning_pair():
    """Two spins under a rotation of 1 kHz: modulated shifts and a modulated dipolar coupling."""
    s = SpinSystem(2)
    i = s.I
    w = 2 * sp.pi * 1000
    shift_1 = 300 + 400 * sp.cos(w * T) + 150 * sp.cos(2 * w * T + sp.Rational(1, 2))
    shift_2 = -200 + 250 * sp.cos(w * T + 1)
    coupling = 120 * sp.cos(w * T + sp.Rational(3, 10)) + 60 * sp.cos(2 * w * T)
    h = 2 * sp.pi * (shift_1 * i(1, "z") + shift_2 * i(2, "z") + coupling * (3 * i(1, "z") @ i(2, "z") - s.dot(1, 2)))
    return s, h


def relaxation(s):
    return sd.phenomenological_relaxation(s, {1: (0.05, 0.004), 2: (0.03, 0.006)})


def average_over_starts(h, **kw):
    """The signal averaged over the eight sampling points of the period as the start of the acquisition."""
    shifted = (Periodic(h.subs(T, T + sp.Rational(j, 8000)), T, PERIOD) for j in range(8))
    return sum(sd.signal_1d(g, method="direct", **kw).values for g in shifted) / 8


def from_peaks(peaks, times):
    return (peaks[:, 0, None] * np.exp((1j * peaks[:, 1, None] - peaks[:, 2, None]) * times)).sum(axis=0)


def test_carousel_average():
    s, h = spinning_pair()
    spinning = Periodic(h, T, PERIOD)
    kw = dict(line_broadening=2 * math.pi * 5, **KW)
    carousel = sd.signal_1d(spinning, carousel=True, **kw)
    assert carousel.peaks is not None
    # independent of SpinDynamica: the explicit average over the start of the acquisition within the period
    assert np.allclose(carousel.values, average_over_starts(h, **kw), rtol=0, atol=1e-9)
    assert np.allclose(sd.signal_1d(spinning, carousel=True, method="compute", **kw).values, carousel.values)
    # the average differs from the signal of a single start
    assert np.abs(carousel.values - sd.signal_1d(spinning, **kw).values).max() > 0.05
    fid = stored("carousel/fid")
    assert np.allclose(carousel.times, fid[:, 0].real, rtol=0, atol=1e-12)
    assert np.allclose(carousel.values, fid[:, 1], **ODE)
    # SpinDynamica's COMPUTE in Liouville space divides by log2 of the dimension once more (divergence D13)
    assert np.allclose(carousel.values, LOG2_DIM * stored("carousel/fid_in_liouville_space")[:, 1], **ODE)
    assert np.allclose(carousel.values, from_peaks(stored("carousel/peaks"), carousel.times), **ODE)
    # the lines are differences of Floquet frequencies plus multiples of the rotation frequency
    base = carousel.peaks[:, 1].real % (2 * math.pi * 1000)
    assert len(np.unique(np.round(base, 4))) <= 16 and np.allclose(carousel.peaks[:, 2].real, math.pi * 5)
    # other initial states and observables
    from_x = sd.signal_1d(spinning, carousel=True, initial=s.I(1, "x"), preparation=None, **kw)
    assert np.allclose(from_x.values, stored("carousel/fid_from_I1x")[:, 1], **ODE)
    assert np.allclose(
        from_x.values, average_over_starts(h, initial=s.I(1, "x"), preparation=None, **kw), rtol=0, atol=1e-9
    )
    spin_2 = sd.signal_1d(spinning, carousel=True, observable=s.I(2, "-"), **kw)
    assert np.allclose(spin_2.values, stored("carousel/fid_observing_spin_2")[:, 1], **ODE)


def test_carousel_average_of_a_single_spin_has_real_sidebands():
    # for one spin with a modulated shift the carousel-averaged sideband amplitudes are |a_k|^2: real, positive
    s = SpinSystem(1)
    z = s.I(1, "z")
    h = 2 * sp.pi * (300 + 900 * sp.cos(2 * sp.pi * 1000 * T + sp.Rational(2, 5))) * z
    signal = sd.signal_1d(Periodic(h, T, PERIOD), carousel=True, line_broadening=None, **KW)
    amplitudes = signal.peaks[:, 0]
    assert np.allclose(amplitudes.imag, 0, atol=1e-10) and (amplitudes.real > 0).all()
    assert np.isclose(amplitudes.sum(), 1)
    # Bessel functions: a_k = J_k(900 / 1000), sampled at eight points per period (aliased orders add up)
    import scipy.special

    orders = np.round((signal.peaks[:, 1].real / (2 * math.pi) - 300) / 1000).astype(int)
    alias = np.arange(-3, 4) * 8
    expected = [abs((scipy.special.jv(k + alias, 0.9) * np.exp(0.4j * alias)).sum()) ** 2 for k in orders]
    assert np.allclose(amplitudes.real, expected, atol=1e-9)
    single = sd.signal_1d(Periodic(h, T, PERIOD), line_broadening=None, **KW)
    assert np.abs(single.peaks[:, 0].imag).max() > 0.01


def test_carousel_average_needs_the_compute_method():
    s, h = spinning_pair()
    spinning = Periodic(h, T, PERIOD)
    assert stored("carousel/direct_method_is_rejected") == "True"
    with pytest.raises(ValueError, match="needs the COMPUTE method"):
        sd.signal_1d(spinning, carousel=True, method="direct", **KW)
    with pytest.raises(ValueError, match="needs the COMPUTE method"):
        sd.signal_1d(TimeDependent(h, T), carousel=True, **KW)
    with pytest.raises(ValueError, match="needs the COMPUTE method"):
        sd.signal_1d(h.subs(T, 0), carousel=True, **KW)
    with pytest.raises(ValueError, match="divides the period"):
        sd.signal_1d(spinning, carousel=True, points=9, dwell=0.00013)
    pulse = [(2 * sp.pi * 2500 * s.I(None, "x"), sp.Rational(1, 10000))]
    with pytest.warns(UserWarning, match="finite duration"):
        sd.signal_1d(spinning, carousel=True, preparation=pulse, **KW)


def test_periodic_hamiltonian_with_relaxation():
    s, h = spinning_pair()
    spinning = Periodic(h, T, PERIOD)
    kw = dict(background=relaxation(s), line_broadening=None, **KW)
    compute = sd.signal_1d(spinning, **kw)
    direct = sd.signal_1d(spinning, method="direct", **kw)
    assert compute.peaks is not None and direct.peaks is None
    assert np.allclose(compute.values, direct.values, rtol=0, atol=1e-10)
    # a generator without a stated period is stepped through, with the same result
    plain = sd.signal_1d(TimeDependent(h, T), **kw)
    assert plain.peaks is None and np.allclose(plain.values, direct.values, rtol=0, atol=1e-10)
    # independent of the signal code: the events of the engine, sample by sample
    q = (-sp.I / 2 * s.I(None, "-")).numeric()
    for n in (1, 7, 20):
        state = sd.propagate(-s.I(None, "y"), [(TimeDependent(h, T), n / 8000)], background=relaxation(s))
        assert np.isclose(compute.values[n], np.vdot(q, state) / np.vdot(q, q), rtol=0, atol=1e-9)
    # SpinDynamica's values are smaller by log2 of the dimension, here 2 (divergence D13)
    for case in ("compute_fid", "automatic_method_fid"):
        assert np.allclose(compute.values, LOG2_DIM * stored(f"relaxation/{case}")[:, 1], **ODE), case
    theirs = from_peaks(stored("relaxation/compute_peaks"), compute.times)
    assert np.allclose(compute.values, LOG2_DIM * theirs, **ODE)
    # the lines decay: every rate lies between the slowest and the fastest rate of the relaxation model
    rates = compute.peaks[:, 2].real
    assert rates.min() > 1 / 0.006 - 1e-6 and rates.max() < 1 / 0.004 + 1 / 0.006 + 1e-6
    # the carousel average in Liouville space
    carousel = sd.signal_1d(spinning, carousel=True, **kw)
    assert np.allclose(carousel.values, average_over_starts(h, **kw), rtol=0, atol=1e-10)
    assert np.allclose(carousel.values, LOG2_DIM * stored("relaxation/carousel_fid")[:, 1], **ODE)
    theirs = from_peaks(stored("relaxation/carousel_peaks"), carousel.times)
    assert np.allclose(carousel.values, LOG2_DIM * theirs, **ODE)


def test_acquisition_that_starts_later():
    # the generator acts from time zero; the acquisition starts at `start`
    s, h = spinning_pair()
    for background in (None, relaxation(s)):
        kw = dict(background=background, line_broadening=None, start=0.0003, **KW)
        compute = sd.signal_1d(Periodic(h, T, PERIOD), **kw)
        direct = sd.signal_1d(TimeDependent(h, T), **kw)
        assert np.allclose(compute.values, direct.values, rtol=0, atol=1e-9)
        state = sd.propagate(-s.I(None, "y"), [(TimeDependent(h, T), 0.0003 + 5 / 8000)], background=background)
        q = (-sp.I / 2 * s.I(None, "-")).numeric()
        assert np.isclose(compute.values[5], np.vdot(q, state) / np.vdot(q, q), rtol=0, atol=1e-9)
        assert np.isclose(compute.times[0], 0.0003)


def test_compute_signals_are_normalised_like_the_other_methods():
    # divergence D13: a signal that starts at 1 by definition starts at 4/3, 1/4 or 2 times its value in
    # SpinDynamica's COMPUTE method; here every method gives the same signal
    g = "normalization"
    assert np.isclose(stored(f"{g}/four_spins_direct"), 1)
    assert np.isclose(stored(f"{g}/four_spins_compute"), 4 / 3)
    assert np.isclose(stored(f"{g}/four_spins_compute_in_liouville_space"), 1 / 4)
    assert np.isclose(stored(f"{g}/four_spins_direct_observing_spin_1"), -0.5j)
    assert np.isclose(stored(f"{g}/four_spins_compute_observing_spin_1"), -1j)
    s = SpinSystem(4)
    h = 2 * sp.pi * (300 + 400 * sp.cos(2 * sp.pi * 1000 * T)) * s.I(None, "z")
    kw = dict(points=16, bandwidth=2 * sp.pi * 8000, line_broadening=None)
    for observable, first in ((None, 1), (s.I(1, "-"), -0.5j)):
        for options in (dict(), dict(method="direct"), dict(carousel=True)):
            signal = sd.signal_1d(Periodic(h, T, PERIOD), observable=observable, **options, **kw)
            assert np.isclose(signal.values[0], first), options
    relaxing = sd.signal_1d(Periodic(h, T, PERIOD), background=sd.phenomenological_relaxation(s, 1.0), **kw)
    assert np.isclose(relaxing.values[0], 1)
