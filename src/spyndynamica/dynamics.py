"""Events and propagation: the numeric engine for sequences of time-independent events.

A sequence is a list of events:

- a :class:`~spyndynamica.liouville.Superoperator`, acting instantaneously (a rotation, a filter);
- ``(generator, duration)``, where the generator is a Hamiltonian (an operator), a Liouvillian or dissipator
  (a superoperator), or ``None`` for the background alone;
- :class:`Pulse`, :class:`Delay` and :class:`Repeat`.

``background`` is a Hamiltonian, a superoperator or a list of both that acts during every event with a
duration. The engine works in Hilbert space when everything is unitary and in Liouville space otherwise.
Symbols are given numbers through ``values``. The order of the events is an argument of each call
(``order='left_to_right'`` by default); there is no global setting.

Author: Alexej Jerschow.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import scipy.sparse
import sympy as sp

from spyndynamica import backend
from spyndynamica.liouville import Superoperator
from spyndynamica.operators import Operator, rotation_spec
from spyndynamica.system import SpinSystem


@dataclass(frozen=True)
class Pulse:
    """A rectangular pulse of flip angle ``flip`` and ``phase`` (``'x'``, ``'y'``, ``'-x'``, ``'-y'`` or an angle) on
    ``spins`` (default: all).

    Without ``nutation_frequency`` and ``duration`` the pulse is ideal: an instantaneous rotation. With one of
    them (rad/s, s) it has the duration ``flip / nutation_frequency`` and the background acts during it.
    ``phase_shift`` is added to the phase. ``amplitude_factor`` scales the radiofrequency field: the flip angle
    of an ideal pulse, the nutation frequency of a finite one (whose duration stays the nominal one).
    """

    flip: Any
    phase: Any = "x"
    spins: Any = None
    nutation_frequency: Any = None
    duration: Any = None
    phase_shift: Any = 0
    amplitude_factor: Any = 1


@dataclass(frozen=True)
class Delay:
    """An interval of ``duration`` (s) under the background, plus ``generator`` if one is given.

    Inside a :class:`PulseSequence` with a total duration, ``Delay("auto")`` takes the time that is left, and
    several ``Delay(("auto", weight))`` share it in the ratio of their weights.
    """

    duration: Any
    generator: Any = None


@dataclass(frozen=True)
class ShapedPulse:
    """A shaped radiofrequency pulse of ``duration`` on ``spins`` (default: all).

    ``amplitude`` (the nutation frequency in rad/s), ``phase`` and ``frequency`` (an offset of the
    radiofrequency in rad/s) are expressions in the local time ``tau``, whose origin is at the fraction
    ``origin`` of the pulse (the centre by default). The field is
    ``amplitude (Ix cos phase + Iy sin phase) − frequency Iz``, the Hamiltonian in the frame of the pulse.
    """

    duration: Any
    amplitude: Any
    phase: Any = 0
    frequency: Any = 0
    tau: Any = None
    spins: Any = None
    origin: Any = sp.Rational(1, 2)
    amplitude_factor: Any = 1


@dataclass(frozen=True)
class PulseSequence:
    """A list of events as one object (SpinDynamica's ``PulseSequence``).

    ``spins`` are the spins of the pulses that name none, ``phase_shift`` is added to the phase of every
    pulse, and ``duration`` is the total duration, which the automatic delays fill (see :class:`Delay`).
    """

    events: Sequence[Any]
    spins: Any = None
    duration: Any = None
    phase_shift: Any = 0


@dataclass(frozen=True)
class Precalculated:
    """The stored propagator of a set of events, made by :func:`precalculate` and used as one event."""

    matrix: Any
    unitary: bool
    duration: float = 0.0


@dataclass(frozen=True)
class TimeDependent:
    """A Hamiltonian that depends on time: an operator whose coefficients contain the symbol ``t``.

    Use it as the generator of an event, ``(TimeDependent(H, t), duration)``, or as the generator of a
    :class:`Delay`. ``t`` is the time since the start of the sequence. The equation of motion is integrated
    numerically.
    """

    operator: Operator
    t: Any

    def function(self, values: Mapping[Any, Any] | None):
        """``H(t)`` as a function returning a NumPy matrix."""
        parts = []
        for key, coefficient in self.operator.terms.items():
            if values:
                coefficient = coefficient.subs(values)
            single = Operator(self.operator.system, {key: sp.S.One}).numeric()
            parts.append((sp.lambdify(self.t, coefficient, "numpy"), single))

        def hamiltonian(time: float) -> np.ndarray:
            return sum(complex(f(time)) * matrix for f, matrix in parts)

        return hamiltonian


@dataclass(frozen=True)
class Periodic(TimeDependent):
    """A time-dependent Hamiltonian with the period ``period``: ``H(t + period) = H(t)``.

    As the generator of :func:`~spyndynamica.signal_1d` it selects the COMPUTE method, which needs the dwell
    time to divide the period. In an event it behaves like :class:`TimeDependent`.
    """

    period: Any = None


@dataclass(frozen=True)
class Repeat:
    """``events`` repeated ``count`` times."""

    events: Sequence[Any]
    count: int


def _number(value: Any, values: Mapping[Any, Any] | None) -> float:
    value = sp.sympify(value)
    if values:
        value = value.subs(values)
    out = complex(sp.N(value, 30))
    if abs(out.imag) > 1e-12 * max(1.0, abs(out.real)):
        raise ValueError(f"expected a real number, got {value}.")
    return out.real


_PHASES = {"x": 0, "y": sp.pi / 2, "-x": sp.pi, "-y": 3 * sp.pi / 2}


def _shift(pulse: Any, shift: Any, spins: Any) -> Any:
    """A pulse with the phase shift and the default spins of the sequence it belongs to."""
    if isinstance(pulse, (Pulse, ShapedPulse)):
        changes: dict[str, Any] = {}
        if pulse.spins is None and spins is not None:
            changes["spins"] = spins
        if shift != 0:
            if isinstance(pulse, Pulse):
                changes["phase_shift"] = pulse.phase_shift + shift
            else:
                changes["phase"] = sp.sympify(pulse.phase) + shift
        return replace(pulse, **changes) if changes else pulse
    return pulse


def _event_duration(event: Any, values: Mapping[Any, Any] | None) -> float:
    if isinstance(event, Pulse):
        if event.duration is not None:
            return _number(event.duration, values)
        return (
            _number(event.flip, values) / _number(event.nutation_frequency, values) if event.nutation_frequency else 0.0
        )
    if isinstance(event, (Delay, ShapedPulse)):
        return _number(event.duration, values)
    if isinstance(event, Precalculated):
        return event.duration
    if isinstance(event, tuple) and len(event) == 2:
        return _number(event[1], values)
    return 0.0


def _expand(events: Iterable[Any], values: Mapping[Any, Any] | None = None) -> list[Any]:
    out: list[Any] = []
    for event in events:
        if isinstance(event, Repeat):
            out += _expand(event.events, values) * int(event.count)
        elif isinstance(event, PulseSequence):
            inner = [_shift(e, sp.sympify(event.phase_shift), event.spins) for e in _expand(event.events, values)]
            automatic = [e for e in inner if isinstance(e, Delay) and not _fixed(e)]
            if automatic:
                if event.duration is None:
                    raise ValueError("automatic delays need the total duration of the pulse sequence.")
                fixed = sum(_event_duration(e, values) for e in inner if e not in automatic)
                weights = [1.0 if isinstance(e.duration, str) else float(e.duration[1]) for e in automatic]
                left = _number(event.duration, values) - fixed
                if left < -1e-15:
                    raise ValueError("the events are longer than the total duration of the pulse sequence.")
                shares = iter(left * w / sum(weights) for w in weights)
                inner = [
                    replace(e, duration=next(shares)) if isinstance(e, Delay) and not _fixed(e) else e for e in inner
                ]
            out += inner
        else:
            out.append(event)
    return out


def _fixed(delay: Delay) -> bool:
    return not (isinstance(delay.duration, str) or (isinstance(delay.duration, tuple) and delay.duration[0] == "auto"))


class _Engine:
    """The events of one call as numeric steps (see :meth:`_step`)."""

    def __init__(
        self, system: SpinSystem, events, background, values, order: str, sparse: bool = False, start_time: float = 0.0
    ):
        if order not in ("left_to_right", "right_to_left"):
            raise ValueError("order must be 'left_to_right' or 'right_to_left'.")
        self.system = system
        self.sparse = sparse
        self._blocks: dict[int, list] = {}
        self.values = values
        self.dim = system.dimension
        self.start_time = float(start_time)
        events = _expand(
            events if isinstance(events, (list, tuple)) and not self._is_pair(events) else [events], values
        )
        if order == "right_to_left":
            events = events[::-1]
        parts = background if isinstance(background, (list, tuple)) else [background]
        self.h0, self.r0 = self._generators(parts)
        self.steps = []
        clock = self.start_time
        for event in events:
            self.steps.append(self._step(event, clock))
            clock += self.steps[-1][0]
        self.liouville = self.r0 is not None or any(step[2] is not None or step[4] is not None for step in self.steps)

    @staticmethod
    def _is_pair(event: Any) -> bool:
        return (
            isinstance(event, tuple)
            and len(event) == 2
            and (event[0] is None or isinstance(event[0], (Operator, Superoperator, TimeDependent)))
            and not isinstance(event[1], (Operator, Superoperator, Pulse, Delay, Repeat, tuple, list))
        )

    def _generators(self, parts: Iterable[Any]) -> tuple[np.ndarray | None, np.ndarray | None]:
        h = r = None
        for part in parts:
            if part is None:
                continue
            if isinstance(part, Operator):
                matrix = part.sparse(self.values) if self.sparse else part.numeric(self.values)
                h = matrix if h is None else h + matrix
            elif isinstance(part, Superoperator):
                r = part.numeric(self.values) if r is None else r + part.numeric(self.values)
            else:
                raise TypeError(f"a generator is an operator or a superoperator, not {type(part).__name__}.")
        return h, r

    def _step(self, event: Any, clock: float = 0.0):
        """``(duration, hamiltonian, dissipator, unitary, superoperator[, hamiltonian(t)])``; the unitary and the
        superoperator are instantaneous, the function of time is there for a time-dependent Hamiltonian."""
        if isinstance(event, Precalculated):
            if event.unitary:
                return (0.0, None, None, event.matrix, None)
            return (0.0, None, None, None, event.matrix)
        if isinstance(event, ShapedPulse):
            duration = _number(event.duration, self.values)
            substitute = (lambda e: sp.sympify(e).subs(self.values)) if self.values else sp.sympify
            factor = substitute(event.amplitude_factor)
            functions = [
                sp.lambdify(event.tau, substitute(e), "numpy") for e in (event.amplitude, event.phase, event.frequency)
            ]
            ix, iy, iz = (self.system.I(event.spins, c).numeric() for c in "xyz")
            origin = clock + float(sp.N(event.origin)) * duration
            scale = float(sp.N(factor))

            def shaped(t: float) -> np.ndarray:
                tau = t - origin
                angle = float(functions[1](tau))
                field = scale * float(functions[0](tau)) * (math.cos(angle) * ix + math.sin(angle) * iy)
                return field - float(functions[2](tau)) * iz

            return (duration, None, None, None, None, shaped)
        if isinstance(event, Pulse):
            phase = _PHASES.get(event.phase, event.phase) + sp.sympify(event.phase_shift)
            if event.nutation_frequency is None and event.duration is None:
                return (0.0, None, None, self._rotation(replace(event, phase=phase, phase_shift=0)), None)
            else:
                flip = _number(event.flip, self.values)
                if event.duration is not None:
                    duration = _number(event.duration, self.values)
                    nutation = flip / duration
                else:
                    nutation = _number(event.nutation_frequency, self.values)
                    duration = flip / nutation
                nutation = nutation * _number(event.amplitude_factor, self.values)
                rf = nutation * self.system.I(event.spins, phase).numeric(self.values)
                return (duration, rf, None, None, None)
        if isinstance(event, Delay):
            event = (event.generator, event.duration)
        if isinstance(event, Superoperator):
            if len(event.sandwiches) == 1 and not event.dyads:
                left, right = (op.numeric(self.values) for op in event.sandwiches[0])
                if np.allclose(right, left.conj().T, atol=1e-12) and np.allclose(
                    left @ right, np.eye(self.dim), atol=1e-12
                ):
                    return (0.0, None, None, left, None)
            return (0.0, None, None, None, event.numeric(self.values))
        if self._is_pair(event):
            if isinstance(event[0], TimeDependent):
                return (_number(event[1], self.values), None, None, None, None, event[0].function(self.values))
            h, r = self._generators([event[0]])
            return (_number(event[1], self.values), h, r, None, None)
        raise TypeError(f"not an event: {event!r}.")

    def _rotation(self, pulse: Pulse) -> np.ndarray:
        """The unitary of an ideal pulse, built from the rotation matrices of the single spins."""
        substitute = (lambda a: sp.sympify(a).subs(self.values)) if self.values else sp.sympify
        phase = pulse.phase if isinstance(pulse.phase, str) else substitute(pulse.phase)
        spec = rotation_spec(substitute(pulse.flip) * substitute(pulse.amplitude_factor), phase)
        indices = set(self.system._indices(pulse.spins))
        out = np.ones((1, 1), dtype=complex)
        for index, loc in enumerate(self.system.locals):
            single = loc.rotation_numeric(spec) if index in indices else np.eye(loc.dim, dtype=complex)
            out = np.kron(single, out)  # the first spin changes fastest
        return out

    def _hamiltonian(self, h):
        parts = [part for part in (h, self.h0) if part is not None]
        if not parts:
            if self.sparse:
                return scipy.sparse.csr_matrix((self.dim, self.dim), dtype=complex)
            return np.zeros((self.dim, self.dim), dtype=complex)
        return parts[0] if len(parts) == 1 else parts[0] + parts[1]

    @staticmethod
    def time_dependent(step) -> bool:
        return len(step) > 5

    def unitary(self, step, duration: float | None = None, start: float = 0.0) -> np.ndarray:
        """The Hilbert-space propagator of one step (of part of its duration, from the time ``start``)."""
        if step[3] is not None:
            return step[3]
        duration = step[0] if duration is None else duration
        if self.time_dependent(step):
            h0 = self._hamiltonian(None)
            return backend.integrate(lambda t: -1j * (step[5](t) + h0), self.dim, start, start + duration)
        return backend.unitary(self._hamiltonian(step[1]), duration)

    def superpropagator(self, step, duration: float | None = None, start: float = 0.0) -> np.ndarray:
        """The propagator of one step on row-flattened density matrices."""
        if step[4] is not None:
            return step[4]
        if step[3] is not None:
            return np.kron(step[3], step[3].conj())
        duration = step[0] if duration is None else duration
        dissipators = [part for part in (step[2], self.r0) if part is not None]
        if self.time_dependent(step):
            eye = np.eye(self.dim)
            h0 = self._hamiltonian(None)
            constant = sum(dissipators) if dissipators else 0.0

            def generator(t: float) -> np.ndarray:
                h = step[5](t) + h0
                return -1j * (np.kron(h, eye) - np.kron(eye, h.T)) + constant

            return backend.integrate(generator, self.dim**2, start, start + duration)
        h = self._hamiltonian(step[1])
        if not dissipators:
            u = backend.unitary(h, duration)
            return np.kron(u, u.conj())
        eye = np.eye(self.dim)
        generator = -1j * (np.kron(h, eye) - np.kron(eye, h.T)) + sum(dissipators)
        return backend.exponential(generator, duration)

    def advance(self, rho: np.ndarray, step, duration: float | None = None, start: float = 0.0) -> np.ndarray:
        if self.liouville:
            return (self.superpropagator(step, duration, start) @ rho.reshape(-1)).reshape(self.dim, self.dim)
        if self.sparse and step[3] is None:
            blocks = self._blocks.get(id(step))
            if blocks is None:
                blocks = self._blocks[id(step)] = backend.block_eigensystems(self._hamiltonian(step[1]))
            return backend.evolve_in_blocks(rho, blocks, step[0] if duration is None else duration)
        u = self.unitary(step, duration, start)
        return u @ rho @ u.conj().T


def _system(*candidates: Any) -> SpinSystem:
    for candidate in candidates:
        if isinstance(candidate, (Operator, Superoperator)):
            return candidate.system
        if isinstance(candidate, (list, tuple)):
            try:
                return _system(*candidate)
            except ValueError:
                continue
    raise ValueError("no operator to take the spin system from.")


def _state(rho: Any, values: Mapping[Any, Any] | None) -> np.ndarray:
    return rho.numeric(values) if isinstance(rho, Operator) else np.asarray(rho, dtype=complex)


def propagator(events, *, system: SpinSystem | None = None, background=None, values=None, order="left_to_right"):
    """The unitary propagator of the events as a NumPy matrix (Zeeman basis). Needs events without relaxation."""
    engine = _Engine(system or _system(events, background), events, background, values, order)
    if engine.liouville:
        raise ValueError("the events are not unitary; use superpropagator.")
    out, now = np.eye(engine.dim, dtype=complex), 0.0
    for step in engine.steps:
        out = engine.unitary(step, start=now) @ out
        now += step[0]
    return out


def superpropagator(events, *, system: SpinSystem | None = None, background=None, values=None, order="left_to_right"):
    """The propagator of the events in Liouville space, acting on row-flattened density matrices."""
    engine = _Engine(system or _system(events, background), events, background, values, order)
    out, now = np.eye(engine.dim**2, dtype=complex), 0.0
    for step in engine.steps:
        out = engine.superpropagator(step, start=now) @ out
        now += step[0]
    return out


SPARSE_FROM = 128  # the Hilbert-space dimension from which method="auto" uses the sparse path


def _engine(
    system: SpinSystem, events, background, values, order: str, method: str, start_time: float = 0.0
) -> _Engine:
    """The engine for a call: sparse when asked for (or large) and the events are plain Hamiltonian evolution."""
    if method not in ("auto", "dense", "sparse"):
        raise ValueError("method must be 'auto', 'dense' or 'sparse'.")
    if method == "sparse" or (method == "auto" and system.dimension >= SPARSE_FROM):
        engine = _Engine(system, events, background, values, order, sparse=True, start_time=start_time)
        if not engine.liouville and not any(engine.time_dependent(step) for step in engine.steps):
            return engine
        if method == "sparse":
            raise ValueError("the sparse method takes Hamiltonians and ideal pulses only.")
    return _Engine(system, events, background, values, order, start_time=start_time)


def propagate(
    rho, events, *, background=None, values=None, order="left_to_right", method="auto", start_time: float = 0.0
) -> np.ndarray:
    """The density matrix after the events (a NumPy matrix in the Zeeman basis).

    ``method='dense'`` diagonalises every Hamiltonian as a whole. ``'sparse'`` keeps the Hamiltonians sparse,
    splits each into the blocks that it does not connect (the subspaces of a conserved quantity such as the
    total ``Iz``) and diagonalises the blocks, which is what large spin systems need. ``'auto'`` chooses by
    the dimension. The sparse method takes Hamiltonians and ideal pulses only.
    """
    system = _system(rho, events, background)
    engine = _engine(system, events, background, values, order, method, start_time)
    state, now = _state(rho, values), engine.start_time
    for step in engine.steps:
        state = engine.advance(state, step, start=now)
        now += step[0]
    return state


def _amplitudes(state: np.ndarray, targets: Sequence[np.ndarray]) -> np.ndarray:
    return np.array([np.vdot(t, state) / np.vdot(t, t) for t in targets])


def amplitudes(
    rho,
    targets,
    events,
    *,
    background=None,
    values=None,
    order="left_to_right",
    method="auto",
    start_time: float = 0.0,
    normalization: Any = None,
):
    """The transformation amplitudes ``(target|ρ_final)/(target|target)`` of ``rho`` into one target or a list.

    ``start_time`` is the time at which the sequence starts (it matters for time-dependent Hamiltonians), and
    the result is divided by ``normalization`` when one is given.
    """
    single = isinstance(targets, Operator)
    state = propagate(
        rho, events, background=background, values=values, order=order, method=method, start_time=start_time
    )
    if normalization is not None:
        state = state / _number(normalization, values)
    out = []
    for target in [targets] if single else targets:
        matrix = target.sparse(values)
        out.append(matrix.conj().multiply(state).sum() / (np.abs(matrix.data) ** 2).sum())
    return complex(out[0]) if single else np.array(out)


def trajectory(rho, targets, events, *, points: int = 32, background=None, values=None, order="left_to_right"):
    """The amplitudes of ``targets`` during the events: ``(times, values)`` with one row of values per target.

    Every event with a duration is sampled at ``points`` equal steps. An instantaneous event adds one sample at
    the time it acts, so a pulse shows as a step.
    """
    system = _system(rho, events, background)
    engine = _Engine(system, events, background, values, order)
    single = isinstance(targets, Operator)
    numeric = [t.numeric(values) for t in ([targets] if single else targets)]
    state = _state(rho, values)
    times, samples, now = [0.0], [_amplitudes(state, numeric)], 0.0
    for step in engine.steps:
        if step[0] == 0:
            state = engine.advance(state, step)
            times.append(now)
            samples.append(_amplitudes(state, numeric))
            continue
        dt = step[0] / points
        fixed = not engine.time_dependent(step)
        if fixed:
            advance = engine.superpropagator(step, dt) if engine.liouville else engine.unitary(step, dt)
        for _ in range(points):
            if not fixed:
                state = engine.advance(state, step, dt, now)
            elif engine.liouville:
                state = (advance @ state.reshape(-1)).reshape(engine.dim, engine.dim)
            else:
                state = advance @ state @ advance.conj().T
            now += dt
            times.append(now)
            samples.append(_amplitudes(state, numeric))
    out = np.array(samples).T
    return np.array(times), (out[0] if single else out)


def precalculate(events, *, system: SpinSystem | None = None, background=None, values=None) -> Precalculated:
    """The propagator of ``events`` as one event that can be reused (SpinDynamica's ``Precalculate``).

    It holds for the background it was calculated with, and it does not advance the time of the sequence.
    """
    system = system or _system(events, background)
    engine = _Engine(system, events, background, values, "left_to_right")
    duration = sum(step[0] for step in engine.steps)
    if engine.liouville:
        return Precalculated(
            superpropagator(events, system=system, background=background, values=values), False, duration
        )
    return Precalculated(propagator(events, system=system, background=background, values=values), True, duration)
