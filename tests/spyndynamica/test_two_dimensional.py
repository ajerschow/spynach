"""Two-dimensional data: the transform with its options, reflection and combination of spectra."""

import math

import numpy as np
import pytest
import sympy as sp

import spyndynamica as sd
from spyndynamica import Signal, SpinSystem
from tests.spyndynamica import reference

TOL = dict(rtol=0, atol=1e-12)
GRID = np.array([[np.exp(-(j + 2 * k) / 5) * np.exp(1j * (j - k)) for k in range(6)] for j in range(4)])
OTHER = np.array([[np.exp(-(2 * j + k) / 7) * np.exp(1j * (j + 2 * k)) for k in range(6)] for j in range(4)])
LIST = np.array([1.0, 0.5j, -0.25, -0.125j, 0.3, 0.2])


def stored(key):
    return reference.get("16_two_dimensional", key)


def dft(samples, frequencies, dwell=1.0, first_point=0.5):
    """The transform by its definition, a sum over the samples."""
    samples = np.array(samples, dtype=complex)
    samples[0] *= first_point
    t = dwell * np.arange(len(samples))
    return np.array([(samples * np.exp(-2j * math.pi * f * t)).sum() for f in frequencies]) / len(samples)


def test_two_dimensional_transform_with_options_per_dimension():
    assert np.allclose(sd.ft2(GRID), stored("transform/grid"), **TOL)
    # (direct, indirect): the direct dimension is along the rows
    wide = sd.ft2(GRID, zero_fill=(4, False))
    assert wide.shape == (4, 24) and np.allclose(wide, stored("transform/zero_fill_per_dimension"), **TOL)
    factors = sd.ft2(GRID, zero_fill=False, first_point=(0.5, 1.0))
    assert np.allclose(factors, stored("transform/first_point_per_dimension"), **TOL)
    # independent of SpinDynamica: the double sum of the definition
    f1 = sd.frequency_axis(4, 0.1, zero_fill=False)
    f2 = sd.frequency_axis(6, 0.05, zero_fill=4)
    direct = np.array([dft(row, f2, 0.05, 0.5) for row in GRID])
    both = np.array([dft(column, f1, 0.1, 1.0) for column in direct.T]).T
    assert np.allclose(sd.ft2(GRID, zero_fill=(4, False), first_point=(0.5, 1.0)), both, **TOL)
    with pytest.raises(ValueError, match="matrix"):
        sd.ft2(LIST)
    with pytest.raises(ValueError, match="zero_fill"):
        sd.ft2(GRID, zero_fill=0)


def test_frequency_axes():
    assert np.allclose(sd.frequency_axis(4, 0.1, zero_fill=False), [-2.5, 0, 2.5, 5])
    assert np.allclose(sd.frequency_axis(5, zero_fill=False), [-0.4, -0.2, 0, 0.2, 0.4])
    frequencies, spectrum = sd.ft(LIST, dwell=0.05)
    assert np.array_equal(frequencies, sd.frequency_axis(6, 0.05)) and len(frequencies) == 12
    assert np.allclose(spectrum, dft(LIST, frequencies, 0.05), **TOL)
    # SpinDynamica's embedded coordinates of two-dimensional data are the frequencies divided by the product
    # of the sampling rates, 10 Hz and 20 Hz here (divergence D14); the values are those of the plain transform
    for case, fill in (("sample_intervals", False), ("triples", False), ("sample_intervals_zero_filled", True)):
        theirs = stored(f"transform/{case}")
        f1, f2 = sd.frequency_axis(4, 0.1, fill), sd.frequency_axis(6, 0.05, fill)
        mesh = np.array([(a, b) for a in f1 for b in f2])
        assert np.allclose(theirs[:, :2].real * 200, mesh, **TOL), case
        assert np.allclose(theirs[:, 2], sd.ft2(GRID, zero_fill=fill).reshape(-1), **TOL), case


def test_reflection_of_spectra():
    assert np.allclose(sd.reflect_spectrum(LIST), stored("reflection/list"), **TOL)
    assert np.allclose(sd.reflect_spectrum(LIST), stored("reflection/pairs")[:, 1], **TOL)
    assert np.allclose(sd.reflect_spectrum(GRID), stored("reflection/matrix"), **TOL)
    assert np.allclose(sd.reflect_spectrum(GRID).reshape(-1), stored("reflection/triples")[:, 2], **TOL)
    # independent: the transform by its definition at the frequencies with the opposite sign
    signal = np.exp((2j * math.pi * 1.5 - 0.4) * np.arange(8) * 0.1) + 0.5j * np.exp(-6j * math.pi * np.arange(8) * 0.1)
    for fill in (False, True, 3):
        frequencies, spectrum = sd.ft(signal, dwell=0.1, zero_fill=fill)
        reflected = sd.reflect_spectrum(spectrum)
        assert np.allclose(reflected[:-1], dft(signal, -frequencies[:-1], 0.1), **TOL)
        assert reflected[-1] == spectrum[-1]  # the last point is the Nyquist frequency
    # an odd number of points has a symmetric axis; SpinDynamica leaves that call unevaluated
    assert stored("reflection/odd_list_is_unevaluated") == "True"
    assert np.array_equal(sd.reflect_spectrum(LIST[:5]), LIST[:5][::-1])
    assert np.allclose(sd.reflect_spectrum(sd.reflect_spectrum(GRID)), GRID)
    # two dimensions: only the indirect one is reflected
    planes = sd.ft2(GRID)
    assert np.allclose(sd.reflect_spectrum(planes)[:, 3], sd.reflect_spectrum(planes[:, 3]))
    with pytest.raises(ValueError, match="list or a matrix"):
        sd.reflect_spectrum(np.zeros((2, 2, 2)))


def test_combination_of_data():
    assert np.allclose(
        sd.combine_data([GRID, OTHER], [sp.Rational(1, 4), 0.75]), stored("combination/matrices_weighted")
    )
    assert np.allclose(sd.combine_data([GRID, OTHER]), stored("combination/matrices_mean"), **TOL)
    assert np.allclose(sd.combine_data([LIST, LIST[::-1]], [2, -1]), stored("combination/lists"), **TOL)
    assert np.allclose(sd.combine_complementary_spectra(GRID, OTHER), stored("reflection/complementary"), **TOL)
    assert np.allclose(sd.combine_complementary_spectra(GRID, OTHER), (GRID + sd.reflect_spectrum(OTHER)) / 2)
    # SpinDynamica documents the list form of CombineComplementarySpectra and defines the form with two arguments
    assert stored("reflection/complementary_list_form_is_unevaluated") == "True"
    # its combination of data with coordinates works for two points only
    assert stored("combination/pairs_with_coordinates_is_rejected") == "True"
    assert np.allclose(stored("combination/two_pairs_with_coordinates")[:, 1], LIST[:2] / 2, **TOL)
    with pytest.raises(ValueError, match="number of weights"):
        sd.combine_data([GRID, OTHER], [1])
    with pytest.raises(ValueError, match="different shapes"):
        sd.combine_data([GRID, LIST])
    with pytest.raises(ValueError, match="nothing to combine"):
        sd.combine_data([])


def test_combination_of_signals():
    a = Signal(0.0, 0.1, 8, peaks=np.array([[1.0, 3.0, 0.5], [0.5j, -2.0, 0.25]], dtype=complex))
    b = Signal(0.0, 0.1, 8, peaks=np.array([[0.25, 1.0, 0.125]], dtype=complex))
    combined = sd.combine_data([a, b], [0.25, 0.75])
    assert combined.peaks is not None and len(combined.peaks) == 3
    assert np.allclose(combined.values, 0.25 * a.values + 0.75 * b.values, **TOL)
    # SpinDynamica makes a new signal of the weighted peaks, which moves their frequencies to its grid of
    # 2π / (2 × 8 × 0.1) rad/s again, whatever the signals that went in (see DigitalFrequencyResolution)
    theirs = stored("combination/signals_with_peaks")
    theirs, ours = (peaks[np.argsort(peaks[:, 1].real)] for peaks in (theirs, combined.peaks))
    assert np.allclose(ours[:, [0, 2]], theirs[:, [0, 2]], **TOL)
    grid = 2 * math.pi / 1.6
    assert np.allclose(theirs[:, 1], np.round(ours[:, 1].real / grid) * grid, **TOL)
    snapped = Signal(0.0, 0.1, 8, peaks=theirs)
    assert np.allclose(snapped.values, stored("combination/signals_with_peaks_fid")[:, 1], **TOL)
    c = Signal(0.0, 0.1, 4, data=np.array([1.0, 0.5j, -0.25, -0.125j]))
    d = Signal(0.0, 0.1, 4, data=np.array([0.5, 0.25, 0.125, 0.0625]))
    sampled = sd.combine_data([c, d], [0.25, 0.75])
    assert sampled.peaks is None
    assert np.allclose(sampled.values, stored("combination/signals_with_data")[:, 1], **TOL)
    # peaks and samples together give samples
    mixed = sd.combine_data([Signal(0.0, 0.1, 4, peaks=a.peaks), c])
    assert mixed.peaks is None and np.allclose(mixed.values, (a.values[:4] + c.values) / 2, **TOL)
    with pytest.raises(ValueError, match="same start"):
        sd.combine_data([a, c])
    with pytest.raises(ValueError, match="signals combine with signals"):
        sd.combine_data([a, LIST])


def test_pure_absorption_from_echo_and_antiecho():
    # two spins-1/2 with a coupling: evolution for t1, then detection of what is left along x or along y.
    # Cosine and sine modulation in t1 combine to the echo and antiecho signals, with opposite ω1 frequencies
    s = SpinSystem(2)
    i = s.I
    h = 2 * sp.pi * (30 * i(1, "z") - 20 * i(2, "z") + 6 * i(1, "z") @ i(2, "z"))
    points, dwell = 16, 0.01
    fid = sd.signal_1d(h, points=points, dwell=dwell, line_broadening=2 * math.pi * 4).values
    t1 = dwell * np.arange(points)
    # the components along x and y after t1, with the same decay as in t2, each detected after a z filter
    along = np.array([sd.amplitudes(i(None, "x"), [i(None, "x"), i(None, "y")], [(h, t)]).real for t in t1])
    along = along * np.exp(-math.pi * 4 * t1)[:, None]
    cosine, sine = along[:, 0, None] * fid[None, :], along[:, 1, None] * fid[None, :]
    echo, antiecho = sd.ft2(cosine - 1j * sine), sd.ft2(cosine + 1j * sine)
    absorption = sd.combine_complementary_spectra(antiecho, echo)
    # each part alone has a phase twist; in the combination the indirect dimension is in pure absorption:
    # the real part of its transform times the spectrum of the direct dimension
    indirect = sd.ft(along[:, 0] + 1j * along[:, 1])[1]
    expected = indirect.real[:, None] * sd.ft(fid)[1][None, :]
    assert np.allclose(absorption[:-1], expected[:-1], **TOL)  # the last row is the Nyquist frequency
    assert np.abs(indirect.imag).max() > 0.3 * np.abs(indirect.real).max()
    assert not np.allclose(antiecho[:-1], expected[:-1], atol=1e-3)
