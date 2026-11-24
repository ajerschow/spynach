"""Signals: free-induction decays from a spin system, line broadening, Fourier transform and lineshapes.

A signal is the amplitude of an observable in the evolving density operator, ``(Q|ρ(t)) / (Q|Q)``, as in
SpinDynamica's ``Signal1D``: with the default observable ``−(i/2) I−`` a Hamiltonian ``ω Iz`` gives
``exp(+iωt)``. Frequencies and line widths are in rad/s, times in seconds; the Fourier transform returns its
frequency axis in Hz, as SpinDynamica's ``FT`` does.

Author: Alexej Jerschow.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import sympy as sp

from spyndynamica import backend
from spyndynamica.dynamics import Periodic, TimeDependent, _Engine, _state, propagate
from spyndynamica.liouville import rotation_superoperator
from spyndynamica.operators import Operator
from spyndynamica.system import SpinSystem


@dataclass(frozen=True)
class Signal:
    """A sampled time-domain signal: ``points`` samples from ``start`` in steps of ``dwell``.

    It holds either the sampled ``data`` or a list of ``peaks``, one row ``(amplitude, frequency, decay rate)``
    per line, which stand for ``Σ a exp((iω − λ)(t − start))``. ``values`` is the sampled signal in both cases.
    """

    start: float
    dwell: float
    points: int
    data: np.ndarray | None = None
    peaks: np.ndarray | None = None

    @property
    def times(self) -> np.ndarray:
        return self.start + self.dwell * np.arange(self.points)

    @property
    def values(self) -> np.ndarray:
        if self.data is not None:
            return np.asarray(self.data, dtype=complex)
        t = self.dwell * np.arange(self.points)
        amplitude, omega, decay = (self.peaks[:, k] for k in range(3))
        return (amplitude[:, None] * np.exp((1j * omega - decay)[:, None] * t[None, :])).sum(axis=0)

    def broadened(self, fwhh: Any = "auto") -> Signal:
        """The signal with a Lorentzian line broadening of full width at half height ``fwhh`` (rad/s).

        ``'auto'`` brings the envelope down to 1% at the last sample.
        """
        if fwhh is None:
            return self
        if isinstance(fwhh, str):
            last = self.dwell * (self.points - 1)
            rate = math.log(100.0) / last if last > 0 else 0.0
        else:
            rate = float(fwhh) / 2
        if self.peaks is not None:
            peaks = np.array(self.peaks, dtype=complex)
            peaks[:, 2] += rate
            return replace(self, peaks=peaks)
        return replace(self, data=self.values * np.exp(-rate * self.dwell * np.arange(self.points)))


def apply_line_broadening(signal: Signal, fwhh: Any = "auto") -> Signal:
    """SpinDynamica's ``ApplyLineBroadening``: :meth:`Signal.broadened`."""
    return signal.broadened(fwhh)


def _merge(peaks: np.ndarray, tolerance: float = 1e-9) -> np.ndarray:
    """Lines of the same frequency and decay rate added up, lines of no amplitude dropped."""
    scale = max(1.0, float(np.abs(peaks[:, 1:]).max(initial=0.0)))
    keys = np.round(np.stack([peaks[:, 1].real, peaks[:, 2].real], axis=1) / (tolerance * scale))
    merged: dict[tuple[float, float], list] = {}
    for key, row in zip(map(tuple, keys), peaks):
        if key in merged:
            merged[key][0] += row[0]
        else:
            merged[key] = [row[0], row[1], row[2]]
    out = np.array([row for row in merged.values() if abs(row[0]) > 1e-12], dtype=complex).reshape(-1, 3)
    return out[np.argsort(-out[:, 1].real, kind="stable")]


def signal_1d(
    generator: Any = None,
    *,
    points: int,
    dwell: Any = None,
    bandwidth: Any = None,
    start: float = 0.0,
    system: SpinSystem | None = None,
    initial: Operator | None = None,
    preparation: Any = "default",
    observable: Any = None,
    background: Any = None,
    line_broadening: Any = "auto",
    method: str = "auto",
    values: Mapping[Any, Any] | None = None,
    normalization: Any = None,
    carousel: bool = False,
) -> Signal:
    """A free-induction decay (SpinDynamica's ``Signal1D``).

    ``generator`` (a Hamiltonian or a Liouvillian) and ``background`` act during the acquisition of ``points``
    samples, ``dwell`` apart (or ``2π / bandwidth`` with the bandwidth in rad/s), from the time ``start`` after
    the preparation. ``initial`` is the density operator before the preparation (default: ``Iz`` of all spins),
    ``preparation`` a list of events (default: a 90° pulse about x; ``None`` for none) and ``observable`` an
    operator or a list of spins (default: all spins), which stands for ``−(i/2) I−`` of those spins.
    ``line_broadening`` is a full width at half height in rad/s, ``'auto'`` (1% at the last sample) or ``None``.
    The signal is divided by ``normalization`` when one is given.

    ``method='diagonalization'`` returns the signal as a list of peaks from the eigenvalues of the generator;
    ``'direct'`` propagates step by step and returns the samples; ``'auto'`` diagonalises.

    A :class:`~spyndynamica.TimeDependent` generator is propagated step by step. A
    :class:`~spyndynamica.Periodic` one uses the COMPUTE method when the dwell time divides its period
    (``'auto'`` or ``'compute'``): the propagators of one period give the whole signal as a list of peaks.
    Both work with relaxation in ``background``, in Liouville space.

    ``carousel=True`` (SpinDynamica's ``CarouselAverage``, "γ-COMPUTE") averages the signal of the COMPUTE
    method over the time within the period at which the acquisition starts, in steps of the dwell time. For a
    rotating sample that is the average over the angle γ of the rotor, provided the state at the start of the
    acquisition does not depend on γ.
    """
    varying = generator if isinstance(generator, TimeDependent) else None
    parts = [None if varying else generator] + (
        list(background) if isinstance(background, (list, tuple)) else [background]
    )
    parts = [p for p in parts if p is not None]
    if varying is not None and system is None:
        system = varying.operator.system
    if system is None:
        candidates = [p for p in (*parts, initial, observable) if hasattr(p, "system")]
        if not candidates:
            raise ValueError("give the spin system when no operator is passed.")
        system = candidates[0].system
    if (dwell is None) == (bandwidth is None):
        raise ValueError("give either the dwell time or the bandwidth.")
    engine = _Engine(system, [(None, 1.0)], parts, values, "left_to_right")
    step = (
        float(sp.N(dwell)) if dwell is not None else 2 * math.pi / float(sp.N(sp.sympify(bandwidth).subs(values or {})))
    )
    rho = system.I(None, "z") if initial is None else initial
    if isinstance(preparation, str):
        preparation = [rotation_superoperator(system, sp.pi / 2, "x")]
    state = propagate(rho, preparation, values=values) if preparation else _state(rho, values)
    if observable is None or isinstance(observable, (list, tuple)):
        observable = -sp.I / 2 * system.I(list(observable) if observable else None, "-")
    q = observable.numeric(values)
    norm = np.vdot(q, q) * (1.0 if normalization is None else float(sp.N(sp.sympify(normalization))))
    dim = system.dimension
    interval = (0.0, None, None, None, None)  # the engine adds the background, which holds every generator here
    if start and varying is None:
        state = engine.advance(state, interval, float(start))
    if method not in ("auto", "direct", "diagonalization", "compute"):
        raise ValueError("method must be 'auto', 'direct', 'diagonalization' or 'compute'.")
    if carousel and (method == "direct" or method == "diagonalization" or not isinstance(varying, Periodic)):
        raise ValueError("carousel=True needs the COMPUTE method: a Periodic generator and method 'auto' or 'compute'.")
    if varying is not None:
        if method == "diagonalization" or (method == "compute" and not isinstance(varying, Periodic)):
            raise ValueError("a time-dependent generator needs method='direct', or 'compute' when it is Periodic.")
        constant = engine._hamiltonian(None)
        function = varying.function(values)
        if engine.liouville:
            eye = np.eye(dim)
            size = dim * dim

            def generate(t: float) -> np.ndarray:
                h = function(t) + constant
                return -1j * (np.kron(h, eye) - np.kron(eye, h.T)) + engine.r0
        else:
            size = dim

            def generate(t: float) -> np.ndarray:
                return -1j * (function(t) + constant)

        period = float(sp.N(sp.sympify(varying.period).subs(values or {}))) if isinstance(varying, Periodic) else None
        per_period = round(period / step) if period else 0
        matched = bool(period) and per_period >= 1 and abs(per_period * step - period) < 1e-9 * period
        if (method == "compute" or carousel) and not matched:
            raise ValueError("the COMPUTE method needs a dwell time that divides the period.")
        t0 = float(start)
        if t0:
            first = backend.integrate(generate, size, 0.0, t0)
            state = (
                (first @ state.reshape(-1)).reshape(dim, dim) if engine.liouville else first @ state @ first.conj().T
            )
        if method == "direct" or not matched:
            signal = _direct_time_dependent(
                generate, size, state, q, norm, t0, step, points, per_period if matched else 0
            )
            return signal.broadened(line_broadening)
        if carousel and preparation and _duration(system, preparation, values) > 0:
            warnings.warn(
                "the preparation has a finite duration: the carousel average holds only when the state at the "
                "start of the acquisition does not depend on the rotor angle.",
                stacklevel=2,
            )
        partial = [np.eye(size, dtype=complex)]
        for j in range(per_period):
            partial.append(backend.integrate(generate, size, t0 + j * step, t0 + (j + 1) * step) @ partial[-1])
        compute = _compute_liouville if engine.liouville else _compute
        return compute(partial, state, q, norm, t0, step, points, carousel).broadened(line_broadening)
    if method == "compute":
        raise ValueError("the COMPUTE method needs a Periodic generator.")
    if method == "direct":
        samples = np.empty(points, dtype=complex)
        advance = engine.superpropagator(interval, step) if engine.liouville else engine.unitary(interval, step)
        for n in range(points):
            samples[n] = np.vdot(q, state) / norm
            if engine.liouville:
                state = (advance @ state.reshape(-1)).reshape(dim, dim)
            else:
                state = advance @ state @ advance.conj().T
        return Signal(float(start), step, points, data=samples).broadened(line_broadening)
    hamiltonian = engine._hamiltonian(None)
    if not engine.liouville:
        energies, vectors = np.linalg.eigh(hamiltonian)
        rho_e = vectors.conj().T @ state @ vectors
        q_e = vectors.conj().T @ q @ vectors
        amplitude = (q_e.conj() * rho_e / norm).reshape(-1)
        omega = (energies[None, :] - energies[:, None]).reshape(-1)
        peaks = np.stack([amplitude, omega.astype(complex), np.zeros_like(amplitude)], axis=1)
    else:
        eye = np.eye(dim)
        generator_matrix = -1j * (np.kron(hamiltonian, eye) - np.kron(eye, hamiltonian.T)) + engine.r0
        rates, right = np.linalg.eig(generator_matrix)
        weights = np.linalg.solve(right, state.reshape(-1))
        amplitude = (q.reshape(-1).conj() @ right) * weights / norm
        peaks = np.stack([amplitude, rates.imag.astype(complex), -rates.real.astype(complex)], axis=1)
    return Signal(float(start), step, points, peaks=_merge(peaks)).broadened(line_broadening)


def _duration(system: SpinSystem, events: Any, values: Mapping[Any, Any] | None) -> float:
    """The total duration of a list of events."""
    return sum(step[0] for step in _Engine(system, events, [], values, "left_to_right").steps)


def _direct_time_dependent(generate, size, state, q, norm, start, step, points, per_period) -> Signal:
    """Step by step under a time-dependent generator, in Hilbert space (``size`` the dimension of the states) or
    in Liouville space (its square); the propagators of one period are reused when ``per_period`` dwell times
    make a period."""
    hilbert = size == len(q)
    cache: dict[int, np.ndarray] = {}
    samples = np.empty(points, dtype=complex)
    for n in range(points):
        samples[n] = np.vdot(q, state) / norm
        key = n % per_period if per_period else n
        if key not in cache or not per_period:
            cache[key] = backend.integrate(generate, size, start + n * step, start + (n + 1) * step)
        u = cache[key]
        state = u @ state @ u.conj().T if hilbert else (u @ state.reshape(-1)).reshape(state.shape)
    return Signal(start, step, points, data=samples)


def _orders(n: int, period: float) -> np.ndarray:
    """The sideband frequencies in the order of the discrete Fourier transform of ``n`` points."""
    return np.fft.fftfreq(n, d=1.0 / n) * 2 * math.pi / period


def _compute(partial, state, q, norm, start, step, points, carousel) -> Signal:
    """The COMPUTE method (Edén, Lee and Levitt, J. Magn. Reson. A 120, 56 (1996)) for a periodic Hamiltonian.

    ``partial`` holds the propagators from the start of the acquisition to the ``n`` sampling points of one
    period and to its end. In the eigenbasis of the propagator of the full period the signal is a sum of lines
    at the differences of the Floquet frequencies plus multiples of the rotation frequency, with amplitudes
    from one discrete Fourier transform.

    With ``carousel`` the signal is averaged over the ``n`` sampling points of the period as the start of the
    acquisition (Hohwy, Bildsøe and Nielsen, J. Magn. Reson. 136, 6 (1999)): the transformed density operator
    becomes periodic like the observable, and the average is the correlation of the two, a product of their
    Fourier coefficients.
    """
    dim = len(q)
    n = len(partial) - 1
    period = n * step
    phases, vectors = backend.unitary_eigensystem(partial[n])
    epsilon = -np.angle(phases) / period  # U(T) = X exp(-i epsilon T) X^+
    difference = epsilon[:, None] - epsilon[None, :]
    observed = np.empty((n, dim, dim), dtype=complex)
    excited = np.empty((n, dim, dim), dtype=complex)
    for j in range(n):
        u = partial[j] @ vectors
        modulation = np.exp(1j * difference * j * step)
        observed[j] = (u.conj().T @ q @ u).conj() * modulation
        excited[j] = (u.conj().T @ state @ u) / modulation
    # f(j) = sum_k a_k exp(+2 pi i j k / n) needs the forward transform
    amplitudes = np.fft.fft(observed, axis=0) / n
    if carousel:
        # the average over the start j0 of q(j0 + m) r(j0) is sum_k q_k r_(-k) exp(2 pi i m k / n)
        amplitudes = amplitudes * np.roll(np.fft.fft(excited, axis=0)[::-1], 1, axis=0) / n
    else:
        amplitudes = amplitudes * excited[0]
    rows = []
    for a, sideband in zip(amplitudes, _orders(n, period)):
        omega = -difference + sideband
        rows.append(
            np.stack(
                [a.reshape(-1) / norm, omega.reshape(-1).astype(complex), np.zeros(dim * dim, dtype=complex)], axis=1
            )
        )
    return Signal(start, step, points, peaks=_merge(np.concatenate(rows)))


def _compute_liouville(partial, state, q, norm, start, step, points, carousel) -> Signal:
    """The COMPUTE method in Liouville space, for a periodic Hamiltonian with relaxation.

    ``partial`` holds the propagators of row-flattened density matrices. The propagator of the full period has
    the eigenvalues ``exp(z T)``, with right eigenvectors ``X``; a line has the frequency ``Im z`` plus a
    multiple of the rotation frequency and the decay rate ``−Re z``.
    """
    n = len(partial) - 1
    period = n * step
    multipliers, right = np.linalg.eig(partial[n])
    if np.linalg.cond(right) > 1e12:
        raise RuntimeError("the propagator of one period has no usable eigenvectors: use method='direct'.")
    z = np.log(multipliers) / period
    rho = state.reshape(-1)
    bra = q.reshape(-1).conj()
    index = np.arange(n)[:, None] * step
    observed = np.array([bra @ partial[j] @ right for j in range(n)]) * np.exp(-z[None, :] * index)
    amplitudes = np.fft.fft(observed, axis=0) / n
    if carousel:
        back = np.array([np.linalg.solve(partial[j] @ right, rho) for j in range(n)])  # X^-1 A_j^-1 rho
        excited = back * np.exp(z[None, :] * index)
        amplitudes = amplitudes * np.roll(np.fft.fft(excited, axis=0)[::-1], 1, axis=0) / n
    else:
        amplitudes = amplitudes * np.linalg.solve(right, rho)[None, :]
    rows = [
        np.stack([a / norm, (z.imag + sideband).astype(complex), (-z.real).astype(complex)], axis=1)
        for a, sideband in zip(amplitudes, _orders(n, period))
    ]
    return Signal(start, step, points, peaks=_merge(np.concatenate(rows)))


def ft(
    signal: Signal | Sequence[complex],
    *,
    dwell: float | None = None,
    zero_fill: int | bool = True,
    first_point: float = 0.5,
    line_broadening: Any = None,
) -> tuple[np.ndarray, np.ndarray]:
    """The Fourier transform ``S(f) = (1/N) Σ s(t_n) exp(−2πi f t_n)`` as ``(frequencies, spectrum)``.

    The frequencies are in Hz, ascending, with zero at the middle (for an even number of points the positive
    side has one point more); in units of the sampling rate when no ``dwell`` is known. The first sample is
    multiplied by ``first_point``; ``zero_fill`` is the factor by which the signal is padded (``True`` is 2,
    ``False`` is 1), and ``N`` is the number of samples before padding.
    """
    if isinstance(signal, Signal):
        if line_broadening is not None:
            signal = signal.broadened(line_broadening)
        data, dwell = signal.values, signal.dwell
    else:
        data = np.array(signal, dtype=complex)
        if line_broadening is not None:
            data = Signal(0.0, dwell or 1.0, len(data), data=data).broadened(line_broadening).values
    count = len(data)
    data = data.copy()
    data[0] *= first_point
    total = count * _fill(zero_fill)
    index = np.arange(total)
    order = np.argsort(np.where(index > total // 2, index - total, index), kind="stable")
    return frequency_axis(count, dwell, zero_fill), (np.fft.fft(data, n=total) / count)[order]


def frequency_axis(points: int, dwell: float | None = None, zero_fill: int | bool = True) -> np.ndarray:
    """The frequencies (Hz, ascending) of the spectrum that :func:`ft` or :func:`ft2` makes of ``points``
    samples ``dwell`` apart; in units of the sampling rate when no ``dwell`` is given."""
    total = points * _fill(zero_fill)
    index = np.arange(total)
    index = np.sort(np.where(index > total // 2, index - total, index))
    return index / (total * (dwell or 1.0))


def _fill(zero_fill: int | bool) -> int:
    factor = 2 if zero_fill is True else (1 if zero_fill is False else int(zero_fill))
    if factor < 1:
        raise ValueError("zero_fill is True, False or a whole number from 1.")
    return factor


def _pair(value: Any) -> tuple[Any, Any]:
    return tuple(value) if isinstance(value, (tuple, list)) else (value, value)


def ft2(data: Any, *, zero_fill: Any = True, first_point: Any = 0.5) -> np.ndarray:
    """The two-dimensional Fourier transform of a matrix of samples, with the conventions of :func:`ft` along
    both dimensions.

    The rows belong to the indirect dimension (``t1``) and the columns to the direct one (``t2``):
    ``data[j, k] = s(t1_j, t2_k)``. ``zero_fill`` and ``first_point`` are one value for both dimensions or a
    pair ``(direct, indirect)``, in SpinDynamica's order. :func:`frequency_axis` gives the two frequency axes.
    """
    out = np.array(data, dtype=complex)
    if out.ndim != 2:
        raise ValueError("ft2 takes a matrix of samples.")
    fills, factors = _pair(zero_fill), _pair(first_point)
    for axis, fill, factor in ((1, fills[0], factors[0]), (0, fills[1], factors[1])):
        out = np.apply_along_axis(lambda row, f=fill, p=factor: ft(row, zero_fill=f, first_point=p)[1], axis, out)
    return out


def embed_coordinates(values: Any, *axes: Any) -> np.ndarray:
    """Data with their coordinates as rows ``(x, value)`` or ``(x1, x2, value)``, the form in which
    SpinDynamica passes data with ``EmbedCoordinates -> True``.

    ``axes`` are the times or frequencies of each dimension of ``values``; for a matrix the first axis belongs
    to the rows (the indirect dimension) and changes slowest. The result is a complex array.
    """
    values = np.asarray(values, dtype=complex)
    if len(axes) != values.ndim or any(len(axis) != size for axis, size in zip(axes, values.shape)):
        raise ValueError("one axis per dimension of the data is needed, each as long as that dimension.")
    mesh = np.meshgrid(*(np.asarray(axis, dtype=float) for axis in axes), indexing="ij")
    return np.stack([*(m.reshape(-1) for m in mesh), values.reshape(-1)], axis=1)


def split_coordinates(rows: Any) -> tuple[np.ndarray, ...]:
    """The axes and the values of data with embedded coordinates: ``(x, values)`` for rows ``(x, value)``,
    ``(x1, x2, values)`` with a matrix of values for rows ``(x1, x2, value)``. The inverse of
    :func:`embed_coordinates`; the rows must be a full grid in that order."""
    rows = np.asarray(rows, dtype=complex)
    if rows.ndim != 2 or rows.shape[1] not in (2, 3):
        raise ValueError("data with coordinates are rows (x, value) or (x1, x2, value).")
    axes = [np.unique(rows[:, k].real) for k in range(rows.shape[1] - 1)]
    shape = tuple(len(axis) for axis in axes)
    if int(np.prod(shape)) != len(rows):
        raise ValueError("the coordinates are not a full grid.")
    values = rows[:, -1].reshape(shape)
    if not np.allclose(embed_coordinates(values, *axes)[:, :-1], rows[:, :-1]):
        raise ValueError("the rows are not in the order of a grid with ascending coordinates.")
    return (*axes, values)


def ft_coordinates(rows: Any, *, zero_fill: Any = True, first_point: Any = 0.5) -> np.ndarray:
    """The Fourier transform of data with embedded time coordinates, as data with embedded frequencies
    (SpinDynamica's ``FT`` with ``EmbedCoordinates -> True``).

    Rows ``(t, value)`` give rows ``(f, value)``, rows ``(t1, t2, value)`` rows ``(f1, f2, value)``, with the
    sampling intervals read from the times, which must be evenly spaced, and the frequencies in Hz. The
    options are those of :func:`ft` and :func:`ft2`.
    """
    *axes, values = split_coordinates(rows)
    dwells = []
    for axis in axes:
        steps = np.diff(axis)
        if len(axis) < 2 or not np.allclose(steps, steps[0]):
            raise ValueError("the times must be evenly spaced, with two or more points per dimension.")
        dwells.append(float(steps[0]))
    if len(axes) == 1:
        frequencies, spectrum = ft(values, dwell=dwells[0], zero_fill=zero_fill, first_point=first_point)
        return embed_coordinates(spectrum, frequencies)
    fills = _pair(zero_fill)
    spectrum = ft2(values, zero_fill=zero_fill, first_point=first_point)
    indirect = frequency_axis(len(axes[0]), dwells[0], fills[1])
    direct = frequency_axis(len(axes[1]), dwells[1], fills[0])
    return embed_coordinates(spectrum, indirect, direct)


def reflect_spectrum(spectrum: Any) -> np.ndarray:
    """A spectrum reflected about zero frequency (SpinDynamica's ``ReflectSpectrum``).

    A two-dimensional spectrum is reflected about ``ω1 = 0``, the zero of the indirect dimension (the rows).
    With an even number of points the frequency axis of :func:`ft` has one positive point more than negative
    ones; that last point stays where it is.
    """
    out = np.asarray(spectrum)
    if out.ndim not in (1, 2):
        raise ValueError("a spectrum is a list or a matrix of values.")
    reflected = out[::-1]
    return np.roll(reflected, -1, axis=0) if len(out) % 2 == 0 else reflected.copy()


def combine_complementary_spectra(first: Any, second: Any) -> np.ndarray:
    """Half the sum of ``first`` and the reflection of ``second`` about ``ω1 = 0``
    (SpinDynamica's ``CombineComplementarySpectra``).

    Two spectra whose peaks have opposite ``ω1`` frequencies and the same real amplitudes, as the echo and
    antiecho parts of a two-dimensional experiment, give a spectrum in pure absorption.
    """
    return combine_data([first, reflect_spectrum(second)])


def combine_data(datasets: Sequence[Any], weights: Sequence[Any] | None = None) -> Any:
    """The linear combination of data sets with the ``weights`` (SpinDynamica's ``CombineData``); their mean
    when no weights are given.

    The data sets are arrays of one shape, or signals with the same sampling. Signals that all hold peaks
    give a signal with the weighted peaks of all of them; otherwise the samples are combined.
    """
    datasets = list(datasets)
    if not datasets:
        raise ValueError("there is nothing to combine.")
    weights = [1 / len(datasets)] * len(datasets) if weights is None else [complex(sp.N(w)) for w in weights]
    if len(weights) != len(datasets):
        raise ValueError("the number of weights must be the number of data sets.")
    if any(isinstance(d, Signal) for d in datasets):
        if not all(isinstance(d, Signal) for d in datasets):
            raise ValueError("signals combine with signals, arrays with arrays.")
        head = datasets[0]
        sampling = {(round(d.start, 15), round(d.dwell, 15), d.points) for d in datasets}
        if len(sampling) != 1:
            raise ValueError("the signals must have the same start, dwell time and number of points.")
        if all(d.peaks is not None for d in datasets):
            peaks = [np.array(d.peaks, dtype=complex) * np.array([w, 1, 1]) for d, w in zip(datasets, weights)]
            return Signal(head.start, head.dwell, head.points, peaks=_merge(np.concatenate(peaks)))
        return Signal(head.start, head.dwell, head.points, data=sum(w * d.values for d, w in zip(datasets, weights)))
    arrays = [np.asarray(d) for d in datasets]
    if len({a.shape for a in arrays}) != 1:
        raise ValueError("the data sets have different shapes and cannot be combined.")
    return sum(w * a for a, w in zip(arrays, weights))


def _lines(lines: Any) -> np.ndarray:
    rows = np.atleast_2d(np.array(lines, dtype=complex))
    if rows.shape[1] == 2:
        rows = np.concatenate([np.ones((len(rows), 1)), rows], axis=1)
    return rows


def lorentzian(w: Any, lines: Any) -> np.ndarray:
    """``Σ a / (λ + i (w − w0))`` for lines ``(a, w0, λ)`` or ``(w0, λ)``; ``λ`` is the half width at half height."""
    w = np.asarray(w, dtype=float)
    return sum(a / (lam + 1j * (w - w0)) for a, w0, lam in _lines(lines))


def absorption(w: Any, lines: Any) -> np.ndarray:
    """``Σ a λ / (λ² + (w − w0)²)``, the real part of :func:`lorentzian` for real amplitudes."""
    w = np.asarray(w, dtype=float)
    return sum(a * lam / (lam**2 + (w - w0) ** 2) for a, w0, lam in _lines(lines)).real


def dispersion(w: Any, lines: Any) -> np.ndarray:
    """``Σ a (w − w0) / (λ² + (w − w0)²)``. This is minus the imaginary part of :func:`lorentzian`."""
    w = np.asarray(w, dtype=float)
    return sum(a * (w - w0) / (lam**2 + (w - w0) ** 2) for a, w0, lam in _lines(lines)).real


def gaussian(w: Any, lines: Any) -> np.ndarray:
    """``Σ a exp(−ln2 (w − w0)²/σ²) / (σ sqrt(π/ln2))`` for lines ``(a, w0, σ)``: unit area, half width ``σ``."""
    w = np.asarray(w, dtype=float)
    ln2 = math.log(2.0)
    total = sum(a * np.exp(-ln2 * (w - w0) ** 2 / s**2) / (s * math.sqrt(math.pi / ln2)) for a, w0, s in _lines(lines))
    return total.real
