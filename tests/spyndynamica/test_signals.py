"""Signals, the Fourier transform, lineshapes and the isotope data against the SpinDynamica fixtures."""

import math

import numpy as np
import pytest
import sympy as sp

import spyndynamica as sd
from spyndynamica import SpinSystem
from tests.spyndynamica import reference

TOL = dict(rtol=0, atol=1e-11)
SW = 2 * sp.pi * 200
LB = 2 * math.pi * 2


def stored(key):
    return reference.get("06_signals", key)


def ab():
    s = SpinSystem(2)
    i = s.I
    return s, i, 2 * sp.pi * 30 * i(1, "z") - 2 * sp.pi * 10 * i(2, "z") + 2 * sp.pi * 12 * s.dot(1, 2)


def same_peaks(mine, theirs):
    """The same lines whatever their order: compare the signals they stand for at a few times."""
    t = np.linspace(0, 0.3, 7)

    def signal(peaks):
        return (peaks[:, 0, None] * np.exp((1j * peaks[:, 1, None] - peaks[:, 2, None]) * t)).sum(axis=0)

    return len(mine) == len(theirs) and np.allclose(signal(mine), signal(np.asarray(theirs, dtype=complex)), **TOL)


def test_signal_of_an_ab_system():
    s, i, ham = ab()
    g = "two_half"
    kw = dict(points=17, bandwidth=SW, line_broadening=LB)
    signal = sd.signal_1d(ham, **kw)
    fid = stored(f"{g}/ab_fid")
    assert np.allclose(signal.times, fid[:, 0].real, **TOL) and np.allclose(signal.values, fid[:, 1], **TOL)
    assert same_peaks(signal.peaks, stored(f"{g}/ab_peaks"))
    direct = sd.signal_1d(ham, method="direct", **kw)
    assert direct.peaks is None and np.allclose(direct.values, stored(f"{g}/ab_fid_direct")[:, 1], **TOL)
    assert np.allclose(sd.signal_1d(None, background=ham, system=s, **kw).values, fid[:, 1], **TOL)
    assert np.allclose(
        sd.signal_1d(ham, points=17, dwell=sp.Rational(1, 200), line_broadening=LB).values, fid[:, 1], **TOL
    )
    # line broadening: none, a full width at half height, and automatic (1% at the last sample)
    assert same_peaks(sd.signal_1d(ham, **{**kw, "line_broadening": None}).peaks, stored(f"{g}/ab_peaks_no_broadening"))
    automatic = sd.signal_1d(ham, points=17, bandwidth=SW)
    assert same_peaks(automatic.peaks, stored(f"{g}/ab_peaks_automatic_broadening"))
    assert np.isclose(
        abs(automatic.values[-1]) / abs(sd.signal_1d(ham, **{**kw, "line_broadening": None}).values[-1]), 0.01
    )
    # observables and preparation
    assert same_peaks(sd.signal_1d(ham, observable=[1], **kw).peaks, stored(f"{g}/ab_peaks_observable_spin_1"))
    assert same_peaks(sd.signal_1d(ham, observable=i(1, "x"), **kw).peaks, stored(f"{g}/ab_peaks_observable_I1x"))
    unprepared = sd.signal_1d(ham, initial=i(1, "x"), preparation=None, **kw)
    assert same_peaks(unprepared.peaks, stored(f"{g}/ab_peaks_from_I1x_without_preparation"))
    # relaxation: the Liouvillian is diagonalised, or propagated step by step
    relaxation = sd.phenomenological_relaxation(s, 1.0, 0.05)
    with_relaxation = dict(points=17, bandwidth=SW, line_broadening=None, background=relaxation)
    assert np.allclose(sd.signal_1d(ham, **with_relaxation).values, stored(f"{g}/ab_fid_with_relaxation")[:, 1], **TOL)
    stepwise = sd.signal_1d(ham, method="direct", **with_relaxation).values
    assert np.allclose(stepwise, stored(f"{g}/ab_fid_with_relaxation_direct")[:, 1], **TOL)


def test_one_and_three_spins():
    one = SpinSystem(1)
    shift = 2 * sp.pi * 30 * one.I(1, "z")
    kw = dict(points=17, bandwidth=SW, line_broadening=LB)
    signal = sd.signal_1d(shift, **kw)
    # a Hamiltonian omega Iz gives one line at +omega with unit amplitude
    assert np.allclose(signal.peaks, [[1, 2 * math.pi * 30, LB / 2]], **TOL)
    assert np.allclose(signal.peaks, stored("one_half/peaks"), **TOL)
    assert np.allclose(signal.values, stored("one_half/fid")[:, 1], **TOL)
    late = sd.signal_1d(shift, start=0.01, **kw)
    assert np.allclose(late.times, stored("one_half/fid_start_time")[:, 0].real, **TOL)
    assert np.allclose(late.values, stored("one_half/fid_start_time")[:, 1], **TOL)
    s = SpinSystem(3)
    i = s.I
    abx = 2 * sp.pi * (30 * i(1, "z") - 10 * i(2, "z") + 70 * i(3, "z") + 12 * s.dot(1, 2))
    abx = abx + 2 * sp.pi * (5 * i(1, "z") @ i(3, "z") + 3 * i(2, "z") @ i(3, "z"))
    kw = dict(points=33, bandwidth=2 * sp.pi * 400, line_broadening=LB)
    signal = sd.signal_1d(abx, **kw)
    assert np.allclose(signal.values, stored("three_half/abx_fid")[:, 1], **TOL)
    assert np.allclose(
        sd.signal_1d(abx, method="direct", **kw).values, stored("three_half/abx_fid_direct")[:, 1], **TOL
    )
    assert len(signal.peaks) == int(stored("three_half/abx_number_of_peaks"))


def test_number_of_points_and_peak_frequencies_differ_from_spindynamica():
    """D3 and the frequency grid in DIVERGENCES.md: what SpinDynamica returns and what the port returns."""
    s, i, ham = ab()
    g = "two_half"
    # an even request loses its last point in SpinDynamica, an odd one is kept
    assert int(stored(f"{g}/points_sw_16")) == 15 and int(stored(f"{g}/points_sw_17")) == 17
    assert int(stored(f"{g}/points_sw_16_not_forced")) == 16
    assert np.allclose(stored(f"{g}/timing_sw_16"), [0, 0.07, 0.005]) and np.allclose(
        stored(f"{g}/timing_sw_17"), [0, 0.08, 0.005]
    )
    assert np.allclose(stored(f"{g}/timing_range_step_16_points"), [0, 0.07, 0.005])
    assert np.allclose(stored(f"{g}/timing_range_step_17_points"), [0, 0.08, 0.005])
    assert sd.signal_1d(ham, points=16, bandwidth=SW).points == 16
    # by default SpinDynamica moves every peak to the nearest point of the frequency grid
    exact, snapped = stored(f"{g}/ab_peaks"), stored(f"{g}/ab_peaks_snapped_to_grid")
    grid = 2 * math.pi * 200 / 34
    assert np.allclose(snapped[:, 1] / grid, np.round(snapped[:, 1] / grid), atol=1e-9)
    assert np.abs(np.sort(exact[:, 1]) - np.sort(snapped[:, 1])).max() > 1.0
    assert same_peaks(sd.signal_1d(ham, points=17, bandwidth=SW, line_broadening=LB).peaks, exact)


def test_fourier_transform():
    s, i, ham = ab()
    signal = sd.signal_1d(ham, points=17, bandwidth=SW, line_broadening=LB)
    for case, options in (("ab_spectrum", {}),):
        frequencies, spectrum = sd.ft(signal, **options)
        reference_spectrum = stored(f"two_half/{case}")
        assert np.allclose(frequencies, reference_spectrum[:, 0].real, **TOL)
        assert np.allclose(spectrum, reference_spectrum[:, 1], **TOL)
    data = [1.0, 0.5, 0.25, 0.125]
    assert np.allclose(sd.ft(data)[1], stored("fourier/list"), **TOL)
    assert np.allclose(sd.ft(data, zero_fill=False)[1], stored("fourier/list_no_zero_fill"), **TOL)
    assert np.allclose(sd.ft(data, zero_fill=False, first_point=1)[1], stored("fourier/list_first_point_1"), **TOL)
    assert np.allclose(sd.ft(data, zero_fill=4)[1], stored("fourier/list_zero_fill_4"), **TOL)
    spiral = [1.0, 0.5j, -0.25, -0.125j]
    for case, options in (("pairs", {}), ("pairs_no_zero_fill", {"zero_fill": False}), ("signal_data", {})):
        frequencies, spectrum = sd.ft(spiral, dwell=0.1, **options)
        assert np.allclose(frequencies, stored(f"fourier/{case}")[:, 0].real, **TOL), case
        assert np.allclose(spectrum, stored(f"fourier/{case}")[:, 1], **TOL), case
    # a rotation at +2.5 Hz shows at +2.5 Hz
    frequencies, spectrum = sd.ft(spiral, dwell=0.1)
    assert frequencies[np.argmax(np.abs(spectrum))] == 2.5
    # D4 in DIVERGENCES.md: for an odd number of points without zero filling SpinDynamica's values sit one point
    # too low on its frequency axis; the port returns the transform at the stated frequencies
    odd = [1.0, 0.5j, -0.25, -0.125j, 0.0625]
    frequencies, spectrum = sd.ft(odd, zero_fill=False)
    weights = np.array(odd) * np.array([0.5, 1, 1, 1, 1])
    direct = np.array([(weights * np.exp(-2j * np.pi * f * np.arange(5))).sum() / 5 for f in frequencies])
    assert np.allclose(spectrum, direct, **TOL)
    assert np.allclose(np.roll(spectrum, -1), stored("fourier/list_odd_length"), **TOL)
    frequencies, spectrum = sd.ft(signal, zero_fill=False)
    shifted = stored("two_half/ab_spectrum_no_zero_fill")
    assert np.allclose(frequencies, shifted[:, 0].real, **TOL) and np.allclose(
        np.roll(spectrum, -1), shifted[:, 1], **TOL
    )


def test_signal_objects_and_line_broadening():
    flat = sd.Signal(0.0, 0.1, 4, data=np.ones(4))
    assert np.allclose(flat.broadened(2 * math.pi).values, stored("fourier/apply_line_broadening")[:, 1], **TOL)
    assert np.allclose(sd.apply_line_broadening(flat, 2 * math.pi).values, flat.broadened(2 * math.pi).values)
    assert np.allclose(flat.broadened("auto").values, stored("fourier/signal_data_automatic_broadening")[:, 1], **TOL)
    spiral = sd.Signal(0.0, 0.1, 4, data=np.array([1.0, 0.5j, -0.25, -0.125j]))
    assert np.allclose(spiral.broadened(2 * math.pi).values, stored("fourier/signal_data_broadened")[:, 1], **TOL)
    lines = np.array([[1 + 1j, 4 * math.pi, 3], [0.5, -2 * math.pi, 1]])
    peaks = sd.Signal(0.0, 0.1, 4, peaks=lines)
    t = peaks.times
    expected = (1 + 1j) * np.exp((4j * math.pi - 3) * t) + 0.5 * np.exp((-2j * math.pi - 1) * t)
    assert np.allclose(peaks.values, expected, **TOL)
    # SpinDynamica expands a list of peaks after moving them to its frequency grid (here 2 pi / 0.8 rad/s)
    grid = 2 * math.pi / 0.8
    snapped = lines.copy()
    snapped[:, 1] = np.round(lines[:, 1].real / grid) * grid
    on_grid = sd.Signal(0.0, 0.1, 4, peaks=snapped)
    assert np.allclose(on_grid.values, stored("fourier/signal_peaks_expanded")[:, 1], **TOL)
    assert np.allclose(peaks.times, stored("fourier/signal_peaks_expanded")[:, 0].real, **TOL)
    assert flat.broadened(None) is flat
    s, i, ham = ab()
    with pytest.raises(ValueError, match="dwell time or the bandwidth"):
        sd.signal_1d(ham, points=8)
    with pytest.raises(ValueError, match="method must be"):
        sd.signal_1d(ham, points=8, dwell=0.01, method="compute")
    with pytest.raises(ValueError, match="give the spin system"):
        sd.signal_1d(None, points=8, dwell=0.01)


def test_lineshapes():
    w = np.array([-3, -0.5, 0, 0.3, 2])
    line, two = (2, 0.5, 0.3), [(2, 0.5, 0.3), (1, -1, 0.2)]
    assert np.allclose(sd.lorentzian(w, line), stored("lineshapes/lorentzian"), **TOL)
    assert np.allclose(sd.lorentzian(w, (2 + 1j, 0.5, 0.3)), stored("lineshapes/lorentzian_complex_amplitude"), **TOL)
    assert np.allclose(sd.lorentzian(w, (0.5, 0.3)), stored("lineshapes/lorentzian_unit"), **TOL)
    assert np.allclose(sd.lorentzian(w, two), stored("lineshapes/lorentzian_two_lines"), **TOL)
    assert np.allclose(sd.absorption(w, line), stored("lineshapes/absorption"), **TOL)
    assert np.allclose(sd.dispersion(w, line), stored("lineshapes/dispersion"), **TOL)
    assert np.allclose(sd.gaussian(w, line), stored("lineshapes/gaussian"), **TOL)
    assert np.allclose(sd.gaussian(w, two), stored("lineshapes/gaussian_two_lines"), **TOL)
    assert np.allclose(sd.lorentzian(w, line).real, sd.absorption(w, line)) and np.allclose(
        sd.lorentzian(w, line).imag, -sd.dispersion(w, line)
    )
    grid = np.linspace(-40, 40, 400001)
    assert np.isclose(np.trapezoid(sd.gaussian(grid, (1, 0.5, 0.3)), grid), 1.0, atol=1e-9)
    assert np.isclose(sd.gaussian(0.8, (1, 0.5, 0.3)) / sd.gaussian(0.5, (1, 0.5, 0.3)), 0.5)


def test_isotopes_and_interactions():
    def table(key):
        return reference.get("07_tables", f"isotopes/{key}")

    names = table("names")
    assert sd.isotopes() == names
    numbers = table("mass_gamma_spin_abundance")
    for name, (_, gamma, spin, abundance) in zip(names, numbers.real):
        assert sd.magnetogyric_ratio(name) == gamma and float(sd.spin_quantum_number(name)) == spin
        assert sd.natural_abundance(name) == abundance
    assert sd.spin_quantum_number("13C") == sp.Rational(1, 2) == sp.nsimplify(float(table("spin_13C")))
    assert float(sd.spin_quantum_number(2)) == float(table("spin_by_mass_number_2"))
    assert sd.magnetogyric_ratio("1H") == float(table("gamma_1H")) and sd.magnetogyric_ratio("15N") == float(
        table("gamma_15N")
    )
    assert sd.natural_abundance("13C") == float(table("abundance_13C")) and sd.isotope(13) == table("isotope_13")
    assert np.isclose(sd.larmor_frequency("1H", 9.4), table("larmor_1H_9p4T"), rtol=1e-14)
    assert np.isclose(sd.larmor_frequency("15N", 9.4), table("larmor_15N_9p4T"), rtol=1e-14)
    assert np.isclose(sd.dipolar_coupling("1H", 2e-10), table("dipolar_1H_1H_2A"), rtol=1e-12)
    assert np.isclose(sd.dipolar_coupling("1H", "13C", 1.1e-10), table("dipolar_1H_13C_1p1A"), rtol=1e-12)
    assert np.isclose(sd.dipolar_coupling("1H", "15N", 1e-10), table("dipolar_1H_15N_1A"), rtol=1e-12)
    constants = sd.constants()
    stored_names = reference.get("07_tables", "constants/names")
    stored_values = reference.get("07_tables", "constants/values")
    ascii_names = {"ℏ": "hbar", "μ0": "mu0", "ϵ0": "epsilon0"}
    assert {ascii_names.get(n, n): v for n, v in zip(stored_names, stored_values)} == constants
    with pytest.warns(UserWarning, match="shared by"):
        assert sd.isotope(3) == "3H"
    with pytest.raises(ValueError, match="unknown isotope"):
        sd.isotope("1Q")
    # a spin system from isotope data
    pair = SpinSystem({"H": sd.spin_quantum_number("1H"), "D": sd.spin_quantum_number("2H")})
    assert pair.dimension == 6
