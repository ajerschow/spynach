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

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
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
    """

    flip: Any
    phase: Any = "x"
    spins: Any = None
    nutation_frequency: Any = None
    duration: Any = None


@dataclass(frozen=True)
class Delay:
    """An interval of ``duration`` (s) under the background, plus ``generator`` if one is given."""

    duration: Any
    generator: Any = None


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


def _expand(events: Iterable[Any]) -> list[Any]:
    out: list[Any] = []
    for event in events:
        if isinstance(event, Repeat):
            out += _expand(event.events) * int(event.count)
        else:
            out.append(event)
    return out


class _Engine:
    """The events of one call as numeric steps (see :meth:`_step`)."""

    def __init__(self, system: SpinSystem, events, background, values, order: str, sparse: bool = False):
        if order not in ("left_to_right", "right_to_left"):
            raise ValueError("order must be 'left_to_right' or 'right_to_left'.")
        self.system = system
        self.sparse = sparse
        self._blocks: dict[int, list] = {}
        self.values = values
        self.dim = system.dimension
        events = _expand(events if isinstance(events, (list, tuple)) and not self._is_pair(events) else [events])
        if order == "right_to_left":
            events = events[::-1]
        parts = background if isinstance(background, (list, tuple)) else [background]
        self.h0, self.r0 = self._generators(parts)
        self.steps = [self._step(event) for event in events]
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

    def _step(self, event: Any):
        """``(duration, hamiltonian, dissipator, unitary, superoperator[, hamiltonian(t)])``; the unitary and the
        superoperator are instantaneous, the function of time is there for a time-dependent Hamiltonian."""
        if isinstance(event, Pulse):
            if event.nutation_frequency is None and event.duration is None:
                return (0.0, None, None, self._rotation(event), None)
            else:
                flip = _number(event.flip, self.values)
                if event.duration is not None:
                    duration = _number(event.duration, self.values)
                    nutation = flip / duration
                else:
                    nutation = _number(event.nutation_frequency, self.values)
                    duration = flip / nutation
                phase = {"x": 0, "y": sp.pi / 2, "-x": sp.pi, "-y": 3 * sp.pi / 2}.get(event.phase, event.phase)
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
        spec = rotation_spec(substitute(pulse.flip), phase)
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


def _engine(system: SpinSystem, events, background, values, order: str, method: str) -> _Engine:
    """The engine for a call: sparse when asked for (or large) and the events are plain Hamiltonian evolution."""
    if method not in ("auto", "dense", "sparse"):
        raise ValueError("method must be 'auto', 'dense' or 'sparse'.")
    if method == "sparse" or (method == "auto" and system.dimension >= SPARSE_FROM):
        engine = _Engine(system, events, background, values, order, sparse=True)
        if not engine.liouville and not any(engine.time_dependent(step) for step in engine.steps):
            return engine
        if method == "sparse":
            raise ValueError("the sparse method takes Hamiltonians and ideal pulses only.")
    return _Engine(system, events, background, values, order)


def propagate(rho, events, *, background=None, values=None, order="left_to_right", method="auto") -> np.ndarray:
    """The density matrix after the events (a NumPy matrix in the Zeeman basis).

    ``method='dense'`` diagonalises every Hamiltonian as a whole. ``'sparse'`` keeps the Hamiltonians sparse,
    splits each into the blocks that it does not connect (the subspaces of a conserved quantity such as the
    total ``Iz``) and diagonalises the blocks, which is what large spin systems need. ``'auto'`` chooses by
    the dimension. The sparse method takes Hamiltonians and ideal pulses only.
    """
    system = _system(rho, events, background)
    engine = _engine(system, events, background, values, order, method)
    state, now = _state(rho, values), 0.0
    for step in engine.steps:
        state = engine.advance(state, step, start=now)
        now += step[0]
    return state


def _amplitudes(state: np.ndarray, targets: Sequence[np.ndarray]) -> np.ndarray:
    return np.array([np.vdot(t, state) / np.vdot(t, t) for t in targets])


def amplitudes(rho, targets, events, *, background=None, values=None, order="left_to_right", method="auto"):
    """The transformation amplitudes ``(target|ρ_final)/(target|target)`` of ``rho`` into one target or a list."""
    single = isinstance(targets, Operator)
    state = propagate(rho, events, background=background, values=values, order=order, method=method)
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
