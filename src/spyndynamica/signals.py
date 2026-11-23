"""Signals: free-induction decays from a spin system, line broadening, Fourier transform and lineshapes.

A signal is the amplitude of an observable in the evolving density operator, ``(Q|ρ(t)) / (Q|Q)``, as in
SpinDynamica's ``Signal1D``: with the default observable ``−(i/2) I−`` a Hamiltonian ``ω Iz`` gives
``exp(+iωt)``. Frequencies and line widths are in rad/s, times in seconds; the Fourier transform returns its
frequency axis in Hz, as SpinDynamica's ``FT`` does.

Author: Alexej Jerschow.
"""

from __future__ import annotations

import math
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
) -> Signal:
    """A free-induction decay (SpinDynamica's ``Signal1D``).

    ``generator`` (a Hamiltonian or a Liouvillian) and ``background`` act during the acquisition of ``points``
    samples, ``dwell`` apart (or ``2π / bandwidth`` with the bandwidth in rad/s), from the time ``start`` after
    the preparation. ``initial`` is the density operator before the preparation (default: ``Iz`` of all spins),
    ``preparation`` a list of events (default: a 90° pulse about x; ``None`` for none) and ``observable`` an
    operator or a list of spins (default: all spins), which stands for ``−(i/2) I−`` of those spins.
    ``line_broadening`` is a full width at half height in rad/s, ``'auto'`` (1% at the last sample) or ``None``.

    ``method='diagonalization'`` returns the signal as a list of peaks from the eigenvalues of the generator;
    ``'direct'`` propagates step by step and returns the samples; ``'auto'`` diagonalises.

    A :class:`~spyndynamica.TimeDependent` generator is propagated step by step. A
    :class:`~spyndynamica.Periodic` one uses the COMPUTE method when the dwell time divides its period
    (``'auto'`` or ``'compute'``): the propagators of one period give the whole signal as a list of peaks.
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
    norm = np.vdot(q, q)
    dim = system.dimension
    interval = (0.0, None, None, None, None)  # the engine adds the background, which holds every generator here
    if start:
        state = engine.advance(state, interval, float(start))
    if method not in ("auto", "direct", "diagonalization", "compute"):
        raise ValueError("method must be 'auto', 'direct', 'diagonalization' or 'compute'.")
    if varying is not None:
        if engine.liouville:
            raise NotImplementedError("a time-dependent generator with relaxation is not available in signal_1d.")
        if method == "diagonalization" or (method == "compute" and not isinstance(varying, Periodic)):
            raise ValueError("a time-dependent generator needs method='direct', or 'compute' when it is Periodic.")
        constant = engine._hamiltonian(None)
        function = varying.function(values)

        def minus_i_h(t: float) -> np.ndarray:
            return -1j * (function(t) + constant)

        period = float(sp.N(sp.sympify(varying.period).subs(values or {}))) if isinstance(varying, Periodic) else None
        per_period = round(period / step) if period else 0
        matched = bool(period) and per_period >= 1 and abs(per_period * step - period) < 1e-9 * period
        if method == "compute" and not matched:
            raise ValueError("the COMPUTE method needs a dwell time that divides the period.")
        if method == "direct" or not matched:
            return _direct_time_dependent(
                minus_i_h, state, q, norm, float(start), step, points, per_period if matched else 0
            ).broadened(line_broadening)
        if start:
            raise NotImplementedError("the COMPUTE method starts at time zero.")
        return _compute(minus_i_h, state, q, norm, step, points, per_period).broadened(line_broadening)
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


def _direct_time_dependent(minus_i_h, state, q, norm, start, step, points, per_period) -> Signal:
    """Step by step under a time-dependent Hamiltonian; the propagators of one period are reused when
    ``per_period`` dwell times make a period."""
    dim = len(q)
    cache: dict[int, np.ndarray] = {}
    samples = np.empty(points, dtype=complex)
    for n in range(points):
        samples[n] = np.vdot(q, state) / norm
        key = n % per_period if per_period else n
        if key not in cache or not per_period:
            cache[key] = backend.integrate(minus_i_h, dim, start + n * step, start + (n + 1) * step)
        u = cache[key]
        state = u @ state @ u.conj().T
    return Signal(start, step, points, data=samples)


def _compute(minus_i_h, state, q, norm, step, points, per_period) -> Signal:
    """The COMPUTE method (Edén, Lee and Levitt, J. Magn. Reson. A 120, 56 (1996)) for a periodic Hamiltonian.

    The propagators to the ``per_period`` sampling points of one period are calculated once. In the eigenbasis
    of the propagator of the full period the signal is a sum of lines at the differences of the Floquet
    frequencies plus multiples of the rotation frequency, with amplitudes from one discrete Fourier transform.
    """
    dim = len(q)
    n = per_period
    period = n * step
    partial = [np.eye(dim, dtype=complex)]
    for j in range(n):
        partial.append(backend.integrate(minus_i_h, dim, j * step, (j + 1) * step) @ partial[-1])
    phases, vectors = backend.unitary_eigensystem(partial[n])
    epsilon = -np.angle(phases) / period  # U(T) = X exp(-i epsilon T) X^+
    rho = vectors.conj().T @ state @ vectors
    difference = epsilon[:, None] - epsilon[None, :]
    table = np.empty((n, dim, dim), dtype=complex)
    for j in range(n):
        u = partial[j] @ vectors
        table[j] = (u.conj().T @ q @ u).conj() * rho * np.exp(1j * difference * j * step)
    amplitudes = np.fft.fft(table, axis=0) / n  # f(j) = sum_k a_k exp(+2 pi i j k / n) needs the forward transform
    orders = np.fft.fftfreq(n, d=1.0 / n)
    rotation = 2 * math.pi / period
    rows = []
    for k_index, k in enumerate(orders):
        a = amplitudes[k_index] / norm
        omega = -difference + k * rotation
        rows.append(
            np.stack([a.reshape(-1), omega.reshape(-1).astype(complex), np.zeros(dim * dim, dtype=complex)], axis=1)
        )
    return Signal(0.0, step, points, peaks=_merge(np.concatenate(rows)))


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
    factor = 2 if zero_fill is True else (1 if zero_fill is False else int(zero_fill))
    total = count * factor
    spectrum = np.fft.fft(data, n=total) / count
    index = np.arange(total)
    index = np.where(index > total // 2, index - total, index)
    order = np.argsort(index, kind="stable")
    frequencies = index[order] / (total * (dwell or 1.0))
    return frequencies, spectrum[order]


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
