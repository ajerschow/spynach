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
import sympy as sp

from spyndynamica import backend
from spyndynamica.liouville import Superoperator, rotation_superoperator
from spyndynamica.operators import Operator
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
    """The events of one call as numeric steps ``(duration, hamiltonian, dissipator, instantaneous)``."""

    def __init__(self, system: SpinSystem, events, background, values, order: str):
        if order not in ("left_to_right", "right_to_left"):
            raise ValueError("order must be 'left_to_right' or 'right_to_left'.")
        self.system = system
        self.values = values
        self.dim = system.dimension
        events = _expand(events if isinstance(events, (list, tuple)) and not self._is_pair(events) else [events])
        if order == "right_to_left":
            events = events[::-1]
        parts = background if isinstance(background, (list, tuple)) else [background]
        self.h0, self.r0 = self._generators(parts)
        self.steps = [self._step(event) for event in events]
        self.liouville = self.r0 is not None or any(step[2] is not None or step[4] for step in self.steps)

    @staticmethod
    def _is_pair(event: Any) -> bool:
        return (
            isinstance(event, tuple)
            and len(event) == 2
            and (event[0] is None or isinstance(event[0], (Operator, Superoperator)))
            and not isinstance(event[1], (Operator, Superoperator, Pulse, Delay, Repeat, tuple, list))
        )

    def _generators(self, parts: Iterable[Any]) -> tuple[np.ndarray | None, np.ndarray | None]:
        h = r = None
        for part in parts:
            if part is None:
                continue
            if isinstance(part, Operator):
                h = part.numeric(self.values) if h is None else h + part.numeric(self.values)
            elif isinstance(part, Superoperator):
                r = part.numeric(self.values) if r is None else r + part.numeric(self.values)
            else:
                raise TypeError(f"a generator is an operator or a superoperator, not {type(part).__name__}.")
        return h, r

    def _step(self, event: Any):
        """``(duration, hamiltonian, dissipator, unitary, superoperator)``; the last two are instantaneous."""
        if isinstance(event, Pulse):
            if event.nutation_frequency is None and event.duration is None:
                event = rotation_superoperator(self.system, event.flip, event.phase, spins=event.spins)
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
            h, r = self._generators([event[0]])
            return (_number(event[1], self.values), h, r, None, None)
        raise TypeError(f"not an event: {event!r}.")

    def _hamiltonian(self, h: np.ndarray | None) -> np.ndarray:
        total = np.zeros((self.dim, self.dim), dtype=complex)
        for part in (h, self.h0):
            if part is not None:
                total = total + part
        return total

    def unitary(self, step, duration: float | None = None) -> np.ndarray:
        """The Hilbert-space propagator of one step (of part of its duration)."""
        if step[3] is not None:
            return step[3]
        return backend.unitary(self._hamiltonian(step[1]), step[0] if duration is None else duration)

    def superpropagator(self, step, duration: float | None = None) -> np.ndarray:
        """The propagator of one step on row-flattened density matrices."""
        if step[4] is not None:
            return step[4]
        if step[3] is not None:
            return np.kron(step[3], step[3].conj())
        duration = step[0] if duration is None else duration
        dissipators = [part for part in (step[2], self.r0) if part is not None]
        h = self._hamiltonian(step[1])
        if not dissipators:
            u = backend.unitary(h, duration)
            return np.kron(u, u.conj())
        eye = np.eye(self.dim)
        generator = -1j * (np.kron(h, eye) - np.kron(eye, h.T)) + sum(dissipators)
        return backend.exponential(generator, duration)

    def advance(self, rho: np.ndarray, step, duration: float | None = None) -> np.ndarray:
        if self.liouville:
            return (self.superpropagator(step, duration) @ rho.reshape(-1)).reshape(self.dim, self.dim)
        u = self.unitary(step, duration)
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
    out = np.eye(engine.dim, dtype=complex)
    for step in engine.steps:
        out = engine.unitary(step) @ out
    return out


def superpropagator(events, *, system: SpinSystem | None = None, background=None, values=None, order="left_to_right"):
    """The propagator of the events in Liouville space, acting on row-flattened density matrices."""
    engine = _Engine(system or _system(events, background), events, background, values, order)
    out = np.eye(engine.dim**2, dtype=complex)
    for step in engine.steps:
        out = engine.superpropagator(step) @ out
    return out


def propagate(rho, events, *, background=None, values=None, order="left_to_right") -> np.ndarray:
    """The density matrix after the events (a NumPy matrix in the Zeeman basis)."""
    system = _system(rho, events, background)
    engine = _Engine(system, events, background, values, order)
    state = _state(rho, values)
    for step in engine.steps:
        state = engine.advance(state, step)
    return state


def _amplitudes(state: np.ndarray, targets: Sequence[np.ndarray]) -> np.ndarray:
    return np.array([np.vdot(t, state) / np.vdot(t, t) for t in targets])


def amplitudes(rho, targets, events, *, background=None, values=None, order="left_to_right"):
    """The transformation amplitudes ``(target|ρ_final)/(target|target)`` of ``rho`` into one target or a list."""
    single = isinstance(targets, Operator)
    numeric = [t.numeric(values) for t in ([targets] if single else targets)]
    out = _amplitudes(propagate(rho, events, background=background, values=values, order=order), numeric)
    return complex(out[0]) if single else out


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
        advance = engine.superpropagator(step, dt) if engine.liouville else engine.unitary(step, dt)
        for _ in range(points):
            if engine.liouville:
                state = (advance @ state.reshape(-1)).reshape(engine.dim, engine.dim)
            else:
                state = advance @ state @ advance.conj().T
            now += dt
            times.append(now)
            samples.append(_amplitudes(state, numeric))
    out = np.array(samples).T
    return np.array(times), (out[0] if single else out)
