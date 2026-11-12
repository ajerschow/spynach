"""Pulse programs: phase cycles, gradients and per-pathway k-vectors run as one branch per coherence pathway.

A :class:`Program` is a list of blocks (:class:`Pulse`, :class:`Delay`, :class:`Gradient`, :class:`Evolve`,
:class:`Homospoil`, :class:`Block`, :class:`Acquire`) with the phase lists of a phase cycle written on the
pulses and the receiver, as on a spectrometer. Instead of running the cycle, :meth:`Program.run` keeps one
branch of the density matrix per coherence transfer pathway that survives the cycle (worked out by the
``ctps`` module, which is only asked for the phase-cycle weights) and carries, per branch, the gradient
k-vector and the diffusion integral b = ∫|k(t)|² dt. Inside an order block the gradient term
``G(t)·r Σ_X γ_X F_z,X`` is the scalar ``γ* G(t)·r`` with ``γ* = Σ_X γ_X p_X``, so a gradient never touches
the state vector: it only advances ``k``, and at detection the average over a uniform sample is the
envelope ``sinc(k_z z_max) · 2J₁(k_⊥ x_max)/(k_⊥ x_max)``, free diffusion the factor ``exp(-D b)`` (exact for
any gradient shape, Bloch–Torrey). A read gradient during acquisition moves ``k`` linearly, so every
pathway echoes at its own time. The explicit phase cycle (``mode='explicit'``) is run from the same
program as the reference; there the gradients are represented without a spatial grid either, on a lattice
of Fourier modes along the gradient axis: the state is one vector per wavevector ``n k0``, with ``k0 =
γ_ref A0`` for the base gradient area ``A0`` of which every gradient is an integer multiple; a gradient of
``m`` units moves the order-``p`` block of mode ``n`` to mode ``n + m p`` (times ``γ*/γ_ref``), and
detection weights every mode with the sample envelope. That representation holds the whole state at each
wavevector, so it is also where mean-field terms (the distant dipolar field) can act. No Spinach counterpart;
see ``docs/pulse-programs.md``. Beta: the pulse programs, the phase-cycle selection and the gradient selection
are new and their interface may still change.

Conventions: coherence orders follow the textbook (and ``ctps``) convention, in which the detected coherence
has order ``-1``: a pulse of phase ``φ`` is ``cos φ Lx + sin φ Ly``, which multiplies a transfer ``p → p'`` of
the pulsed isotope by ``exp(-i Δp φ)``, and the receiver phase ``φ_rec`` multiplies the signal by
``exp(-i φ_rec)``; phases are in degrees and lists shorter than the cycle repeat cyclically (CCCP conventions,
Jerschow & Müller 1998). Spinach labels the basis the other way round: its ``L+`` coil state picks the ``+1``
component, and its signal is the complex conjugate of the textbook one, so a program's order ``p`` is the
basis order ``-p`` of :func:`spynach.coherence` and the receiver acts on Spinach's signal as ``exp(+i φ_rec)``.
Gradients are in T/m with their real duration, ``k`` in rad/m.

Author: Alexej Jerschow.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
from scipy.special import j1

from spynach.ddf import DipolarFieldSpec, dipolar_field
from spynach.filters import homospoil, isotope_orders, leakage, orders_present
from spynach.operators import operator
from spynach.propagation import evolution, iserstep, propagator, step
from spynach.states import state
from spynach.system import SpinSystem
from spynach.utilities import Matrix

PhaseSpec = Sequence[float] | tuple[str, int]
_DEGREE_BASES = (1, 2, 4, 8, 12, 16, 24, 36, 72, 360)


@dataclass(frozen=True)
class Pulse:
    """An ideal pulse of flip ``angle`` (rad) on ``spins``: an isotope, a list of isotopes, or 0-based spin numbers.

    ``phase`` is the phase list in degrees, cycled with the program, or ``("cog", nu)`` for a cogwheel
    winding number (the program then needs ``cycle_length``). A list of isotopes becomes consecutive pulses
    with the same phases, as ``Lx_F1 + Lx_F2`` in the HSQC port; spin numbers must share one isotope.

    Author: Alexej Jerschow.
    """

    spins: str | Sequence[str] | Sequence[int]
    angle: float
    phase: PhaseSpec = (0.0,)
    name: str = ""


@dataclass(frozen=True)
class Delay:
    """Free evolution for ``duration`` (s) under ``'full'`` ``L`` or ``'relaxation'`` (``1j (R + K)``, NOESY mixing).

    ``orders`` restricts the coherence orders of the period, ``{"1H": [-2, 2]}``, as a filter would (textbook
    convention: the detected order is ``-1``, the opposite sign of :func:`spynach.coherence`).
    """

    duration: float
    generator: str = "full"
    orders: dict[str, Sequence[int]] | None = None


@dataclass(frozen=True)
class Gradient:
    """A rectangular gradient of ``duration`` (s) with amplitudes ``x``, ``y``, ``z`` in T/m.

    Author: Alexej Jerschow.
    """

    duration: float
    z: float = 0.0
    x: float = 0.0
    y: float = 0.0

    @property
    def amplitude(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=float)


@dataclass(frozen=True)
class Evolve:
    """An indirect dimension: ``npoints`` increments of ``timestep`` (s) under ``L``; the state becomes a stack."""

    timestep: float
    npoints: int
    name: str = "t1"


@dataclass(frozen=True)
class Refocus:
    """The second half of a refocused indirect dimension.

    Column ``k`` of the last :class:`Evolve` evolves ``k`` steps of ``timestep`` (Spynach's ``'refocus'`` output),
    so ``Evolve(dt, n), Pulse(..., pi), Refocus(dt)`` is ``tau/2 - 180 - tau/2`` incremented as one dimension; a
    second :class:`Evolve` would evolve every column by every second-half length, an ``n x n`` grid.
    """

    timestep: float


@dataclass(frozen=True)
class Homospoil:
    """A homospoil gradient: every transverse pathway is dropped (``keep='zqc'`` keeps zero-quantum coherences)."""

    keep: str | None = None


@dataclass(frozen=True)
class Block:
    """A group of blocks whose pulses all receive the extra phase list ``phase`` (block phase cycling).

    Author: Alexej Jerschow.
    """

    blocks: Sequence[Any]
    phase: PhaseSpec = (0.0,)


@dataclass(frozen=True)
class Acquire:
    """Detection of ``spins`` (an isotope): ``npoints`` points ``timestep`` (s) apart, the receiver phases in degrees.

    ``coil`` replaces the default ``L+`` of the isotope; ``gradient`` is a constant read gradient during
    the acquisition (a :class:`Gradient` whose duration is ignored).
    """

    spins: str
    timestep: float
    npoints: int
    receiver: PhaseSpec = (0.0,)
    coil: np.ndarray | None = None
    gradient: Gradient | None = None


@dataclass(frozen=True)
class Sample:
    """A cylindrical sample: ``z_max`` half the active length, ``x_max`` the radius (m)."""

    z_max: float = 5e-3
    x_max: float = 2.5e-3


def sample_envelope(k: np.ndarray, sample: Sample) -> np.ndarray:
    """Average of ``exp(-i k·r)`` over the sample for k-vectors ``(..., 3)`` in rad/m: the signal envelope.

    Author: Alexej Jerschow.
    """
    k = np.asarray(k, dtype=float)
    u = k[..., 2] * sample.z_max
    f_z = np.sinc(u / np.pi)
    v = np.hypot(k[..., 0], k[..., 1]) * sample.x_max
    safe = np.where(v > 0, v, 1.0)
    f_xy = np.where(v > 0, 2 * j1(safe) / safe, 1.0)
    return f_z * f_xy


def ramp_integral(k0: np.ndarray, k1: np.ndarray, duration: float) -> float:
    """``∫|k(t)|² dt`` over a linear ramp from ``k0`` to ``k1`` lasting ``duration``.

    Author: Alexej Jerschow.
    """
    return float(duration * (k0 @ k0 + k0 @ k1 + k1 @ k1) / 3)


# ----------------------------------------------------------------------------------------------------------
# Expansion of a program into pulses, periods and the integer phase cycle


@dataclass
class _PulseStep:
    isotope: str
    spins: Any
    angle: float
    phases: np.ndarray  # (N,) integers in units of 2π / base
    name: str


@dataclass
class _Period:
    segments: list[tuple] = field(default_factory=list)  # ("hold", τ) or ("ramp", area (3,), δ)
    orders: dict[str, set[int]] | None = None  # restrictions from Delay.orders
    spoiled: bool = False
    evolve: bool = False

    @property
    def area(self) -> np.ndarray:
        total = np.zeros(3)
        for segment in self.segments:
            if segment[0] == "ramp":
                total += segment[1]
        return total

    @property
    def duration(self) -> float:
        return float(sum(s[1] if s[0] == "hold" else s[2] for s in self.segments))

    def restrict(self, orders: dict[str, Sequence[int]]) -> None:
        self.orders = self.orders or {}
        for isotope, values in orders.items():
            values = {int(v) for v in np.atleast_1d(values)}
            self.orders[isotope] = self.orders[isotope] & values if isotope in self.orders else values


@dataclass
class _Flat:
    pulses: list[_PulseStep]
    events: list[tuple]
    periods: list[_Period]
    acquire: Acquire
    base: int
    steps: int
    receiver: np.ndarray  # (N,) integers in units of 2π / base

    def phases_rad(self, m: int) -> list[float]:
        return [2 * math.pi * int(p.phases[m]) / self.base for p in self.pulses]

    def receiver_rad(self, m: int) -> float:
        return 2 * math.pi * int(self.receiver[m]) / self.base


def _phase_ints(spec: PhaseSpec, cycle_length: int | None, what: str) -> tuple[np.ndarray, int]:
    """A phase list as integers on the finest grid that holds it exactly: ``(ints, base)``."""
    if isinstance(spec, tuple) and len(spec) == 2 and isinstance(spec[0], str):
        if spec[0] != "cog":
            raise ValueError(f"{what}: a phase tuple must be ('cog', nu).")
        if not cycle_length:
            raise ValueError(f"{what}: cogwheel phases need Program(cycle_length=N).")
        N = int(cycle_length)
        return (int(spec[1]) * np.arange(N)) % N, N
    values = np.atleast_1d(np.asarray(spec, dtype=float))
    if values.size == 0:
        raise ValueError(f"{what}: a phase list needs at least one entry.")
    for base in _DEGREE_BASES:
        scaled = values * base / 360
        if np.all(np.abs(scaled - np.round(scaled)) < 1e-9):
            return np.round(scaled).astype(int) % base, base
    raise ValueError(f"{what}: phases must lie on a 1° grid; use ('cog', nu) for cogwheel cycles.")


def _flatten(blocks: Sequence[Any], outer: list[PhaseSpec]) -> list[tuple[Any, list[PhaseSpec]]]:
    out = []
    for block in blocks:
        if isinstance(block, Block):
            out.extend(_flatten(block.blocks, [*outer, block.phase]))
        else:
            out.append((block, outer))
    return out


def _expand(program: Program, spin_system: SpinSystem) -> _Flat:
    items = _flatten(program.blocks, [])
    acquires = [b for b, _ in items if isinstance(b, Acquire)]
    if len(acquires) != 1 or not isinstance(items[-1][0], Acquire):
        raise ValueError("a program ends with exactly one Acquire block.")
    acquire = acquires[0]
    if acquire.spins not in spin_system.isotopes:
        raise ValueError(f"Acquire: isotope {acquire.spins!r} is not in the spin system.")
    isotopes = list(spin_system.isotopes)
    # Phase lists as integers on a common base
    raw: list[tuple[list[tuple[np.ndarray, int]], Any]] = []
    bases = []
    for block, outer in items:
        if isinstance(block, Pulse):
            lists = [_phase_ints(block.phase, program.cycle_length, f"pulse {block.name or block.spins}")]
            lists += [_phase_ints(p, program.cycle_length, "block phase") for p in outer]
            raw.append((lists, block))
            bases += [b for _, b in lists]
    receiver_ints, receiver_base = _phase_ints(acquire.receiver, program.cycle_length, "receiver")
    bases.append(receiver_base)
    base = int(np.lcm.reduce(bases)) if bases else 1
    lengths = [len(ints) for lists, _ in raw for ints, _ in lists] + [len(receiver_ints)]
    if program.cycle_length:
        lengths.append(int(program.cycle_length))
    steps = int(np.lcm.reduce(lengths))

    def expand(ints: np.ndarray, own_base: int) -> np.ndarray:
        return np.resize(ints, steps) * (base // own_base)

    # Pulses, events and periods
    pulses: list[_PulseStep] = []
    events: list[tuple] = []
    periods: list[_Period] = [_Period()]
    pulse_phases = iter(raw)
    for block, _outer in items:
        if isinstance(block, Pulse):
            lists, _ = next(pulse_phases)
            total = np.zeros(steps, dtype=int)
            for ints, own_base in lists:
                total = (total + expand(ints, own_base)) % base
            for isotope, spins in _pulse_targets(block, spin_system):
                pulses.append(_PulseStep(isotope, spins, float(block.angle), total, block.name))
                events.append(("pulse", len(pulses) - 1))
                periods.append(_Period())
        elif isinstance(block, Delay):
            if block.generator not in ("full", "relaxation"):
                raise ValueError("Delay.generator must be 'full' or 'relaxation'.")
            periods[-1].segments.append(("hold", float(block.duration)))
            if block.orders:
                unknown = set(block.orders) - set(isotopes)
                if unknown:
                    raise ValueError(f"Delay.orders names isotopes not in the system: {sorted(unknown)}.")
                periods[-1].restrict(block.orders)
            events.append(("delay", block))
        elif isinstance(block, Gradient):
            area = block.amplitude * float(block.duration)
            periods[-1].segments.append(("ramp", area, float(block.duration)))
            events.append(("gradient", block))
        elif isinstance(block, Evolve):
            periods[-1].evolve = True
            events.append(("evolve", block))
        elif isinstance(block, Refocus):
            if not any(p.evolve for p in periods):
                raise ValueError("Refocus needs an Evolve block before it.")
            events.append(("refocus", block))
        elif isinstance(block, Homospoil):
            if block.keep not in (None, "zqc"):
                raise ValueError("Homospoil.keep must be None or 'zqc'.")
            periods[-1].spoiled = True
            events.append(("homospoil", block))
        elif isinstance(block, Acquire):
            events.append(("acquire", block))
        else:
            raise TypeError(f"unknown program block {block!r}.")
    return _Flat(pulses, events, periods, acquire, base, steps, expand(receiver_ints, receiver_base))


def _pulse_targets(pulse: Pulse, spin_system: SpinSystem) -> list[tuple[str, Any]]:
    """``(isotope, spins argument of operator())`` for each consecutive pulse a Pulse block stands for."""
    spins = pulse.spins
    if isinstance(spins, str):
        if spins not in spin_system.isotopes:
            raise ValueError(f"Pulse: isotope {spins!r} is not in the spin system.")
        return [(spins, spins)]
    spins = list(spins)
    if all(isinstance(s, str) for s in spins):
        return [t for s in spins for t in _pulse_targets(replace(pulse, spins=s), spin_system)]
    numbers = [int(s) for s in spins]
    if any(n < 0 or n >= spin_system.nspins for n in numbers):
        raise ValueError("Pulse: spin numbers must be within the system bounds.")
    found = {spin_system.isotopes[n] for n in numbers}
    if len(found) != 1:
        raise ValueError("Pulse: spin numbers must belong to one isotope; use a list of isotopes otherwise.")
    return [(found.pop(), numbers)]


# ----------------------------------------------------------------------------------------------------------
# Selection: the pathways that survive, with their weights and kinematics


@dataclass
class Selection:
    """The coherence transfer pathways of a program that survive its phase cycle, with weights and k-vectors.

    ``orders[i, n, q]`` is the order of isotope ``isotopes[n]`` in period ``q`` of pathway ``i`` (textbook
    convention, the detected order is ``-1``);
    ``pc`` the complex phase-cycle weight; ``k[i, q]`` the k-vector (rad/m) at the end of period ``q``;
    ``b[i, q]`` the diffusion integral up to the end of period ``q`` (indirect dimensions excluded);
    ``envelope[i]`` the largest sample envelope over the acquisition window; ``echo_time[i]`` when the
    pathway refocuses during acquisition (``nan`` when it does not); ``kept[i]`` whether the run
    propagates it (envelope above ``crush_tol``). ``result`` is the ``ctps.PathwayResult`` behind it.
    """

    isotopes: tuple[str, ...]
    orders: np.ndarray
    pc: np.ndarray
    k: np.ndarray
    b: np.ndarray
    envelope: np.ndarray
    echo_time: np.ndarray
    kept: np.ndarray
    result: Any = None
    leakage: dict[str, float] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.pc)

    @property
    def n_periods(self) -> int:
        return self.orders.shape[2]

    @property
    def k0(self) -> np.ndarray:
        """k-vector at the start of acquisition (after every gradient before it), ``(npaths, 3)``."""
        return self.k[:, -1]

    def pathway(self, i: int) -> str:
        return "; ".join(
            f"{iso} " + ",".join(f"{p:+d}" if p else "0" for p in self.orders[i, n])
            for n, iso in enumerate(self.isotopes)
        )

    def table(self) -> list[dict[str, Any]]:
        return [
            {
                "pathway": self.pathway(i),
                "weight": float(abs(self.pc[i])),
                "phase_deg": float(np.degrees(np.angle(self.pc[i])) % 360),
                "k0": float(np.linalg.norm(self.k0[i])),
                "envelope": float(self.envelope[i]),
                "echo_time": float(self.echo_time[i]),
                "kept": bool(self.kept[i]),
            }
            for i in range(len(self))
        ]

    def __repr__(self) -> str:
        lines = [f"Selection: {len(self)} pathways pass the phase cycle, {int(self.kept.sum())} kept"]
        for row in self.table():
            echo = "" if np.isnan(row["echo_time"]) else f", echo at {row['echo_time'] * 1e3:.3g} ms"
            lines.append(
                f"  {row['pathway']}: |pc| {row['weight']:.3g}, phase {row['phase_deg']:.0f}°, "
                f"|k0| {row['k0']:.3g} rad/m, envelope {row['envelope']:.3g}{echo}"
                + ("" if row["kept"] else " (crushed)")
            )
        if self.leakage:
            lines.append("  leakage: " + ", ".join(f"{k}: {v:.2g}" for k, v in self.leakage.items()))
        return "\n".join(lines)

    def _repr_html_(self) -> str:
        head = (
            "<tr><th>pathway</th><th>|pc|</th><th>ψ / °</th><th>|k₀| / rad m⁻¹</th>"
            "<th>envelope</th><th>echo / ms</th></tr>"
        )
        rows = []
        for row in self.table():
            style = "" if row["kept"] else " style='opacity:0.5'"
            echo = "" if np.isnan(row["echo_time"]) else f"{row['echo_time'] * 1e3:.3g}"
            rows.append(
                f"<tr{style}><td><code>{row['pathway']}</code></td><td>{row['weight']:.3g}</td>"
                f"<td>{row['phase_deg']:.0f}</td><td>{row['k0']:.3g}</td><td>{row['envelope']:.3g}</td>"
                f"<td>{echo}</td></tr>"
            )
        note = f"<p>{len(self)} pathways pass the phase cycle, {int(self.kept.sum())} kept (crushed ones greyed).</p>"
        return f"<div>{note}<table>{head}{''.join(rows)}</table></div>"


@dataclass
class PathwayFids:
    """The FID of every kept pathway: ``fids[i]`` has the shape of the program's FID; ``sum()`` is the FID."""

    fids: np.ndarray
    selection: Selection
    index: np.ndarray  # rows of the selection, in the order of fids

    def sum(self) -> np.ndarray:
        return self.fids.sum(axis=0)

    @property
    def orders(self) -> np.ndarray:
        return self.selection.orders[self.index]

    @property
    def echo_time(self) -> np.ndarray:
        return self.selection.echo_time[self.index]

    def pathway(self, i: int) -> str:
        return self.selection.pathway(int(self.index[i]))


def _import_ctps():
    try:
        import ctps  # noqa: PLC0415
    except ImportError as error:  # pragma: no cover - ctps ships with spynach
        raise ImportError(
            "pulse programs derive their pathways with the ctps module, which is installed with spynach "
            "(pip install spynach); it could not be imported."
        ) from error
    return ctps


# ----------------------------------------------------------------------------------------------------------
# The program


@dataclass(frozen=True)
class Program:
    """A pulse program: blocks in order, the sample, diffusion, and how to run it.

    ``mode`` is ``'pathways'`` (one branch per surviving pathway, the default) or ``'explicit'`` (the phase
    cycle step by step, gradients on the Fourier-mode lattice: all gradients collinear, their areas integer
    multiples of ``gradient_unit``, by default the smallest area present);
    ``epsilon`` is the threshold on the phase-cycle weight ``|pc|``, ``crush_tol`` on the sample envelope
    during acquisition below which a pathway is not propagated; ``rho0`` the initial state (default ``Lz`` of
    the detected isotope, ``'cheap'``); ``max_order`` caps ``|p|`` per isotope (an ``int`` or
    ``{isotope: int}``) to keep the enumeration small; ``cycle_length`` fixes the cycle length (needed
    by cogwheel phases); ``check_leakage`` measures, per order block met, how much of the generator leaves it;
    ``field`` is a :class:`spynach.ddf.DipolarFieldSpec`, the distant dipolar field of the sample, integrated
    on the lattice of explicit mode (``mode='explicit'`` is then required; ``Evolve`` is not available).

    A program is a pulse sequence in Spynach's sense: ``program(spin_system, parameters, H, R, K)`` runs it,
    so ``spynach.liquid(spin_system, program, parameters, "nmr")`` works as for the built-in sequences.

    Author: Alexej Jerschow.
    """

    blocks: Sequence[Any]
    sample: Sample = Sample()
    diffusion: float = 0.0
    mode: str = "pathways"
    epsilon: float = 1e-6
    crush_tol: float = 1e-3
    rho0: np.ndarray | None = None
    max_order: int | dict[str, int] | None = None
    cycle_length: int | None = None
    check_leakage: bool = False
    gradient_unit: float | None = None
    field: DipolarFieldSpec | None = None

    def __post_init__(self) -> None:
        if self.mode not in ("pathways", "explicit"):
            raise ValueError("Program.mode must be 'pathways' or 'explicit'.")
        if self.field is not None and self.mode != "explicit":
            raise ValueError("the dipolar field needs mode='explicit' (Program.explicit()).")
        object.__setattr__(self, "blocks", tuple(self.blocks))

    def explicit(self) -> Program:
        """A copy that runs the phase cycle step by step."""
        return replace(self, mode="explicit")

    def __call__(self, spin_system: SpinSystem, parameters: dict[str, Any], H: Matrix, R: Matrix, K: Matrix):
        return self.run(spin_system, H, R, K)

    # -- selection ----------------------------------------------------------------------------------------

    def steps(self, spin_system: SpinSystem) -> int:
        """Number of steps of the phase cycle."""
        return _expand(self, spin_system).steps

    def _rho0(self, spin_system: SpinSystem, flat: _Flat) -> np.ndarray:
        if self.rho0 is not None:
            return np.asarray(self.rho0, dtype=complex).reshape(-1)
        return np.asarray(state(spin_system, "Lz", flat.acquire.spins, "cheap"), dtype=complex)

    def _coil(self, spin_system: SpinSystem, flat: _Flat) -> np.ndarray:
        if flat.acquire.coil is not None:
            return np.asarray(flat.acquire.coil, dtype=complex).reshape(-1)
        return np.asarray(state(spin_system, "L+", flat.acquire.spins, "cheap"), dtype=complex)

    def to_ctps(self, spin_system: SpinSystem):
        """The program as ``(ctps.Sequence, ctps.PhaseCycle)``, for ``ctps.simulate`` and its plots.

        Orders come from the basis (capped by ``max_order``), the start from ``rho0``, the end from the coil;
        each gradient is ctps's delta of the same area at the start of its period, the read gradient is left out.
        """
        ctps = _import_ctps()
        flat = _expand(self, spin_system)
        rho0, coil = self._rho0(spin_system, flat), self._coil(spin_system, flat)
        isotopes = list(dict.fromkeys(spin_system.isotopes))
        per_state = _program_orders(spin_system)
        n_periods = len(flat.periods)
        nuclei, allowed = [], {}
        for iso in isotopes:
            n = spin_system.isotopes.index(iso)
            nuclei.append(
                ctps.Nucleus(iso, gamma=float(spin_system.gammas[n]), spin=(int(spin_system.mults[n]) - 1) / 2)
            )
            present = set(int(p) for p in np.unique(per_state[iso]))
            cap = self.max_order.get(iso) if isinstance(self.max_order, dict) else self.max_order
            if cap is not None:
                present = {p for p in present if abs(p) <= int(cap)}
            p_max = max(abs(p) for p in present)
            grid = ctps.AllowedOrders(p_max, n_periods, np.zeros((2 * p_max + 1, n_periods)))
            for q, period in enumerate(flat.periods):
                if q == 0:
                    values = set(-int(p) for p in orders_present(spin_system, rho0, iso))
                elif q == n_periods - 1:
                    values = set(-int(p) for p in orders_present(spin_system, coil, iso))
                else:
                    values = set(present)
                    if period.spoiled:
                        values &= {0}
                    if period.orders and iso in period.orders:
                        values &= period.orders[iso]
                if not values:
                    raise ValueError(f"no coherence order of {iso} is allowed in period {q}.")
                grid.only(q, sorted(values))
            allowed[iso] = grid
        pulses = tuple(ctps.Pulse(p.isotope, "free", label=p.name) for p in flat.pulses)
        periods = tuple(ctps.Period(period.duration, tuple(float(a) for a in period.area)) for period in flat.periods)
        sequence = ctps.Sequence(
            tuple(nuclei),
            pulses,
            periods,
            allowed,
            sample=ctps.Sample(self.sample.z_max, self.sample.x_max),
            diffusion=self.diffusion,
        )
        cycle = ctps.PhaseCycle(
            tuple(tuple(int(v) for v in p.phases) for p in flat.pulses),
            tuple(int(v) for v in flat.receiver),
            flat.base,
        )
        return sequence, cycle

    def select(self, spin_system: SpinSystem) -> Selection:
        """The pathways that pass the phase cycle, with weights, k-vectors, envelopes and echo times."""
        ctps = _import_ctps()
        flat = _expand(self, spin_system)
        sequence, cycle = self.to_ctps(spin_system)
        result = ctps.simulate(sequence, cycle, epsilon=self.epsilon, cutoff=0.0)
        isotopes = tuple(n.name for n in sequence.nuclei)
        orders = np.stack([result.per_nucleus[iso].astype(int) for iso in isotopes], axis=1)
        selection = _kinematics(self, spin_system, flat, isotopes, orders, result.pc)
        selection.result = result
        return selection

    # -- running ------------------------------------------------------------------------------------------

    def run(
        self,
        spin_system: SpinSystem,
        H: Matrix,
        R: Matrix,
        K: Matrix,
        *,
        output: str = "fid",
        selection: Selection | None = None,
    ) -> np.ndarray | PathwayFids | ModeFids:
        """Run the program; ``output='fid'`` the summed FID, ``'pathways'`` a :class:`PathwayFids` per kept pathway
        (pathways mode), ``'modes'`` a :class:`ModeFids` per lattice mode (explicit mode).

        The FID has the shape ``(npoints,)``, or ``(npoints, n1)`` with an :class:`Evolve` block, ``(npoints,
        n1, n2)`` with two, as the built-in 2-D sequences.
        """
        if output not in ("fid", "pathways", "modes"):
            raise ValueError("output must be 'fid', 'pathways' or 'modes'.")
        if spin_system.basis is None or spin_system.basis.formalism != "sphten-liouv":
            raise NotImplementedError("pulse programs run in the sphten-liouv formalism.")
        flat = _expand(self, spin_system)
        if self.mode == "explicit":
            if output == "pathways":
                raise ValueError("per-pathway output needs mode='pathways'.")
            return _run_explicit(self, spin_system, flat, H, R, K, output)
        if output == "modes":
            raise ValueError("per-mode output needs mode='explicit'.")
        selection = selection if selection is not None else self.select(spin_system)
        return _run_pathways(self, spin_system, flat, selection, H, R, K, output)


# ----------------------------------------------------------------------------------------------------------
# Kinematics: k-vectors, diffusion integrals, envelopes and echo times per pathway


def _program_orders(spin_system: SpinSystem) -> dict[str, np.ndarray]:
    """Coherence order of every basis state per isotope in the program's (textbook) convention: minus Spinach's."""
    return {iso: -orders for iso, orders in isotope_orders(spin_system).items()}


def _gammas(spin_system: SpinSystem, isotopes: Sequence[str]) -> np.ndarray:
    return np.array([float(spin_system.gammas[spin_system.isotopes.index(iso)]) for iso in isotopes])


def _advance(k: np.ndarray, b: float, period: _Period, gamma_star: float) -> tuple[np.ndarray, float]:
    """``k`` and ``b`` after the segments of a period for a branch of effective gyromagnetic ratio ``gamma_star``."""
    for segment in period.segments:
        if segment[0] == "hold":
            b += segment[1] * float(k @ k)
        else:
            k1 = k + gamma_star * segment[1]
            b += ramp_integral(k, k1, segment[2])
            k = k1
    return k, b


def _kinematics(
    program: Program,
    spin_system: SpinSystem,
    flat: _Flat,
    isotopes: tuple[str, ...],
    orders: np.ndarray,
    pc: np.ndarray,
) -> Selection:
    gammas = _gammas(spin_system, isotopes)
    npaths, _, n_periods = orders.shape
    k = np.zeros((npaths, n_periods, 3))
    b = np.zeros((npaths, n_periods))
    gamma_star = np.einsum("n,inq->iq", gammas, orders)
    for i in range(npaths):
        kk, bb = np.zeros(3), 0.0
        for q, period in enumerate(flat.periods):
            kk, bb = _advance(kk, bb, period, gamma_star[i, q])
            k[i, q], b[i, q] = kk, bb
    acquire = flat.acquire
    t = np.arange(acquire.npoints) * acquire.timestep
    k0 = k[:, -1]
    g = gamma_star[:, -1][:, None] * (acquire.gradient.amplitude if acquire.gradient else np.zeros(3))[None, :]
    kt = k0[:, None, :] + g[:, None, :] * t[None, :, None]
    envelope = np.abs(sample_envelope(kt, program.sample)).max(axis=1)
    echo = np.full(npaths, np.nan)
    gg = np.einsum("ia,ia->i", g, g)
    for i in range(npaths):
        if gg[i] > 0:
            t_e = -(k0[i] @ g[i]) / gg[i]
            residual = np.linalg.norm(k0[i] + g[i] * t_e)
            if 0 <= t_e <= t[-1] and residual * max(program.sample.z_max, program.sample.x_max) < 1e-6:
                echo[i] = t_e
        elif np.linalg.norm(k0[i]) * max(program.sample.z_max, program.sample.x_max) < 1e-6:
            echo[i] = 0.0
    kept = (np.abs(pc) >= program.epsilon) & (envelope >= program.crush_tol)
    return Selection(isotopes, orders, np.asarray(pc), k, b, envelope, echo, kept)


# ----------------------------------------------------------------------------------------------------------
# The branch runner


@dataclass
class _Branch:
    orders: np.ndarray  # (nisotopes,) current order of each isotope
    candidates: np.ndarray  # rows of the selection consistent with the history so far
    k: np.ndarray  # (3,)
    b: np.ndarray  # (ncols,) diffusion integral per column
    rho: np.ndarray  # (nstates, ncols)


def _run_pathways(
    program: Program,
    spin_system: SpinSystem,
    flat: _Flat,
    selection: Selection,
    H: Matrix,
    R: Matrix,
    K: Matrix,
    output: str,
) -> np.ndarray | PathwayFids:
    L = H + 1j * R + 1j * K
    L_relax = 1j * R + 1j * K
    isotopes = selection.isotopes
    gammas = _gammas(spin_system, isotopes)
    per_state = _program_orders(spin_system)
    masks = {iso: per_state[iso] for iso in isotopes}
    rho0 = program._rho0(spin_system, flat)
    coil = program._coil(spin_system, flat)
    kept = np.flatnonzero(selection.kept)
    orders = selection.orders
    leak: dict[str, float] = {}

    def block_mask(values: np.ndarray) -> np.ndarray:
        mask = np.ones(spin_system.basis.nstates, dtype=bool)
        for n, iso in enumerate(isotopes):
            mask &= masks[iso] == values[n]
        return mask

    def check_leakage(generator: Matrix, values: np.ndarray, label: str) -> None:
        if not program.check_leakage:
            return
        key = f"{label} " + ",".join(f"{iso}:{v:+d}" for iso, v in zip(isotopes, values))
        if key not in leak:  # the filters label orders the Spinach way, hence the sign
            leak[key] = leakage(spin_system, generator, {iso: -int(v) for iso, v in zip(isotopes, values)})

    # Period 0: one branch per combination of orders present in rho0 that some kept pathway starts from
    branches: list[_Branch] = []
    starts = {tuple(orders[i, :, 0]) for i in kept}
    for values in sorted(starts):
        values = np.array(values)
        rho = rho0 * block_mask(values)
        if np.abs(rho).max() <= spin_system.tols.zte_tol:
            continue
        candidates = np.array([i for i in kept if tuple(orders[i, :, 0]) == tuple(values)])
        branches.append(_Branch(values, candidates, np.zeros(3), np.zeros(1), rho.reshape(-1, 1)))
    period = 0
    ncols_shape: list[int] = []

    def stacked(generator_fn) -> None:
        """Apply ``generator_fn(stack) -> stack`` to all branches at once."""
        if not branches:
            return
        counts = [b.rho.shape[1] for b in branches]
        out = generator_fn(np.hstack([b.rho for b in branches]))
        out = out.reshape(out.shape[0], -1)
        for b, piece in zip(branches, np.split(out, np.cumsum(counts)[:-1], axis=1)):
            b.rho = piece

    for kind, payload in flat.events:
        if kind == "pulse":
            pulse = flat.pulses[payload]
            Lx = operator(spin_system, "Lx", pulse.spins)
            stacked(lambda stack, Lx=Lx, angle=pulse.angle: step(spin_system, Lx, stack, angle))
            n_iso = isotopes.index(pulse.isotope)
            children: list[_Branch] = []
            for b in branches:
                for p_new in np.unique(orders[b.candidates, n_iso, period + 1]):
                    sub = b.candidates[orders[b.candidates, n_iso, period + 1] == p_new]
                    values = b.orders.copy()
                    values[n_iso] = p_new
                    rho = b.rho * (masks[pulse.isotope] == p_new)[:, None]
                    if np.abs(rho).max() <= spin_system.tols.zte_tol:
                        continue
                    children.append(_Branch(values, sub, b.k.copy(), b.b.copy(), rho))
            branches = children
            period += 1
        elif kind == "delay":
            generator = L if payload.generator == "full" else L_relax
            for b in branches:
                check_leakage(generator, b.orders, payload.generator)
            stacked(lambda stack, G=generator, t=payload.duration: evolution(spin_system, G, stack, t, 1, "final"))
            for b in branches:
                b.b = b.b + payload.duration * float(b.k @ b.k)
        elif kind == "gradient":
            for b in branches:
                check_leakage(L, b.orders, "full")
            stacked(lambda stack, t=payload.duration: evolution(spin_system, L, stack, t, 1, "final"))
            for b in branches:
                gamma_star = float(gammas @ b.orders)
                k1 = b.k + gamma_star * payload.amplitude * payload.duration
                b.b = b.b + ramp_integral(b.k, k1, payload.duration)
                b.k = k1
        elif kind == "evolve":
            t = np.arange(payload.npoints) * payload.timestep
            for b in branches:
                check_leakage(L, b.orders, "full")
                columns = [
                    evolution(spin_system, L, b.rho[:, c], payload.timestep, payload.npoints - 1, "trajectory")
                    for c in range(b.rho.shape[1])
                ]
                b.rho = np.hstack(columns)
                b.b = np.repeat(b.b, payload.npoints) + np.tile(t * float(b.k @ b.k), len(columns))
            ncols_shape.append(payload.npoints)
        elif kind == "refocus":
            n_last = ncols_shape[-1]
            for b in branches:
                check_leakage(L, b.orders, "full")
                b.rho = _refocus(spin_system, L, b.rho, payload.timestep, n_last)
                b.b = b.b + np.tile(np.arange(n_last) * payload.timestep, b.rho.shape[1] // n_last) * float(b.k @ b.k)
        elif kind == "homospoil":
            flag = "keep" if payload.keep == "zqc" else "destroy"
            for b in branches:
                b.rho = homospoil(spin_system, b.rho.T, flag).T
            branches = [b for b in branches if np.abs(b.rho).max() > spin_system.tols.zte_tol]
        elif kind == "acquire":
            acquire = payload
            t = np.arange(acquire.npoints) * acquire.timestep
            read = acquire.gradient.amplitude if acquire.gradient else np.zeros(3)
            fids, index = [], []
            for b in branches:
                if b.candidates.size != 1:
                    raise RuntimeError("a branch ends with more than one candidate pathway; this is a bug.")
                i = int(b.candidates[0])
                gamma_star = float(gammas @ b.orders)
                g = gamma_star * read
                kt = b.k[None, :] + g[None, :] * t[:, None]
                envelope = sample_envelope(kt, program.sample)
                if np.abs(envelope).max() < program.crush_tol:
                    continue
                b_t = float(b.k @ b.k) * t + float(b.k @ g) * t**2 + float(g @ g) * t**3 / 3
                # Spinach's signal is the conjugate of the textbook one: the textbook weight pc applies conjugated
                weight = np.conj(selection.pc[i]) * envelope * np.exp(-program.diffusion * b_t)
                check_leakage(L, b.orders, "full")
                fid = evolution(spin_system, L, b.rho, acquire.timestep, acquire.npoints - 1, "observable", coil=coil)
                fid = fid.reshape(acquire.npoints, -1) * weight[:, None]
                if program.diffusion:
                    fid = fid * np.exp(-program.diffusion * b.b)[None, :]
                fids.append(fid.reshape(acquire.npoints, *ncols_shape))
                index.append(i)
            selection.leakage.update(leak)
            shape = (acquire.npoints, *ncols_shape)
            if output == "pathways":
                stack = np.stack(fids) if fids else np.zeros((0, *shape), dtype=complex)
                return PathwayFids(stack, selection, np.array(index, dtype=int))
            return np.sum(fids, axis=0) if fids else np.zeros(shape, dtype=complex)
    raise RuntimeError("the program has no Acquire block.")  # pragma: no cover - _expand checks this


def _refocus(spin_system: SpinSystem, L: Matrix, rho: np.ndarray, timestep: float, n_last: int) -> np.ndarray:
    """``evolution(..., 'refocus')`` on the last indirect dimension of a stack whose columns run that index fastest."""
    nstates, ncols = rho.shape
    blocks = rho.reshape(nstates, ncols // n_last, n_last)
    out = np.stack(
        [evolution(spin_system, L, blocks[:, j], timestep, 0, "refocus") for j in range(blocks.shape[1])], axis=1
    )
    return out.reshape(nstates, ncols)


# ----------------------------------------------------------------------------------------------------------
# The explicit runner: the phase cycle step by step, gradients on a Fourier-mode lattice


@dataclass
class _Lattice:
    """The wavevector lattice of an explicit run: modes ``n k0 axis`` for ``n = -half .. half``.

    ``ratio[s]`` is ``γ*/γ_ref`` of basis state ``s`` (``γ* = Σ γ_X p_X``, ``γ_ref`` that of the acquired isotope):
    a gradient of ``m`` units moves state ``s`` of mode ``n`` to mode ``n + m ratio[s]``.
    """

    axis: np.ndarray
    unit: float
    k0: float
    half: int
    ratio: np.ndarray

    @property
    def modes(self) -> np.ndarray:
        return np.arange(-self.half, self.half + 1)

    @property
    def size(self) -> int:
        return 2 * self.half + 1

    def units(self, area: np.ndarray, label: str) -> int:
        """The integer number of units of a gradient area (a vector), ``0`` for no gradient."""
        area = np.asarray(area, dtype=float)
        norm = float(np.linalg.norm(area))
        if norm == 0:
            return 0
        if np.linalg.norm(np.cross(area, self.axis)) > 1e-9 * norm:
            raise ValueError(
                f"explicit mode puts every gradient on one axis: {label} is not collinear with the first gradient."
            )
        m = float(area @ self.axis) / self.unit
        if abs(m - round(m)) > 1e-9 * max(1.0, abs(m)):
            raise ValueError(
                f"explicit mode needs gradient areas that are integer multiples of one unit "
                f"({self.unit:.6g} T s/m): {label} is {m:.6g} units; set Program(gradient_unit=...)."
            )
        return int(round(m))

    def shifts(self, m: int, rho: np.ndarray, tol: float) -> np.ndarray:
        """Integer mode shifts per basis state for ``m`` units; non-integer shifts of populated states raise."""
        raw = m * self.ratio
        rounded = np.round(raw)
        off = np.abs(raw - rounded) > 1e-9
        if np.any(off) and np.abs(rho[off]).max() > tol:
            raise NotImplementedError(
                "a gradient during coherences of a second isotope moves them by a non-integer number of modes "
                "(the gyromagnetic ratios are not commensurate); explicit mode cannot represent this."
            )
        return rounded.astype(int)

    def envelope(self, sample: Sample) -> np.ndarray:
        """The sample envelope of every mode, ``(size,)``."""
        return sample_envelope(self.modes[:, None] * self.k0 * self.axis[None, :], sample)


def _lattice(program: Program, spin_system: SpinSystem, flat: _Flat, extra: int = 0) -> _Lattice | None:
    """The lattice of an explicit run, ``None`` without gradients; ``extra`` modes are added on each side."""
    acquire = flat.acquire
    areas: list[tuple[str, np.ndarray]] = []
    for q, period in enumerate(flat.periods):
        for segment in period.segments:
            if segment[0] == "ramp":
                areas.append((f"the gradient in period {q}", np.asarray(segment[1], dtype=float)))
    read = acquire.gradient.amplitude * acquire.timestep if acquire.gradient is not None else np.zeros(3)
    areas.append(("the read gradient (per dwell)", read))
    nonzero = [(label, a) for label, a in areas if np.linalg.norm(a) > 0]
    if not nonzero:
        return None
    axis = nonzero[0][1] / np.linalg.norm(nonzero[0][1])
    for label, a in nonzero:
        if np.linalg.norm(np.cross(a, axis)) > 1e-9 * np.linalg.norm(a):
            raise ValueError(
                f"explicit mode puts every gradient on one axis: {label} is not collinear with the first gradient."
            )
    unit = float(program.gradient_unit) if program.gradient_unit else min(abs(float(a @ axis)) for _, a in nonzero)
    if unit <= 0:
        raise ValueError("gradient_unit must be positive.")
    gamma_ref = float(spin_system.gammas[spin_system.isotopes.index(acquire.spins)])
    per_state = _program_orders(spin_system)
    gamma_star = sum(
        float(spin_system.gammas[spin_system.isotopes.index(iso)]) * orders for iso, orders in per_state.items()
    )
    lattice = _Lattice(axis, unit, gamma_ref * unit, 0, np.asarray(gamma_star, dtype=float) / gamma_ref)
    total = sum(abs(lattice.units(a, label)) for label, a in areas[:-1])
    total += abs(lattice.units(read, areas[-1][0])) * (acquire.npoints - 1)
    reach = float(np.abs(lattice.ratio).max()) * total
    lattice.half = int(math.ceil(reach - 1e-9)) + int(extra)
    return lattice


def _shift_modes(rho: np.ndarray, shifts: np.ndarray, tol: float, strict: bool = True) -> np.ndarray:
    """``rho`` of shape ``(nstates, nmodes, ncols)`` with state ``s`` moved ``shifts[s]`` modes up the lattice.

    Amplitude pushed off the lattice is a bug when ``strict`` (the lattice covers every gradient) and the
    truncation of the harmonics a dipolar field made otherwise.
    """
    out = np.zeros_like(rho)
    size = rho.shape[1]
    for d in np.unique(shifts):
        rows = shifts == d
        d = int(d)
        if abs(d) >= size:
            lost = rho[rows]
        elif d >= 0:
            out[rows, d:, :] = rho[rows, : size - d, :]
            lost = rho[rows, size - d :, :]
        else:
            out[rows, : size + d, :] = rho[rows, -d:, :]
            lost = rho[rows, :-d, :]
        if strict and lost.size and np.abs(lost).max() > tol:
            raise RuntimeError("a gradient pushed amplitude beyond the mode lattice; this is a bug.")
    return out


@dataclass
class ModeFids:
    """The FID of every mode of an explicit run: ``fids[i]`` is the contribution of the mode ``modes[i]``, at
    wavevector ``modes[i] * k0`` (rad/m) along ``axis``, weighted by its sample ``envelope[i]``; ``signals[i]``
    is the same signal without the envelope; ``sum()`` is the FID.

    Author: Alexej Jerschow.
    """

    fids: np.ndarray
    signals: np.ndarray
    modes: np.ndarray
    k0: float
    axis: np.ndarray
    envelope: np.ndarray

    def sum(self) -> np.ndarray:
        return self.fids.sum(axis=0)


def _run_explicit(
    program: Program, spin_system: SpinSystem, flat: _Flat, H: Matrix, R: Matrix, K: Matrix, output: str
) -> np.ndarray | ModeFids:
    """The phase cycle step by step: every pulse at its phase, the receiver factor applied, averaged over the cycle.

    The receiver multiplies the textbook signal by ``exp(-i φ_rec)``; Spinach's ``L+`` coil gives its complex
    conjugate, so here the factor is ``exp(+i φ_rec)``. Gradients put the state on the Fourier-mode lattice
    (:class:`_Lattice`): the state is ``(nstates, nmodes, ncols)``, every linear step sees the modes as more
    columns, a gradient moves each coherence-order block along the mode axis, and acquisition weights every
    mode with its sample envelope. With a :class:`DipolarFieldSpec` every delay, gradient and dwell is
    integrated by :func:`iserstep` with the mean-field generator of :mod:`spynach.ddf` on the lattice (the
    gradient's own duration evolves at the wavevectors before the shift; diffusion is in the generator).
    """
    L = H + 1j * R + 1j * K
    L_relax = 1j * R + 1j * K
    rho0 = program._rho0(spin_system, flat)
    coil = program._coil(spin_system, flat)
    tol = spin_system.tols.zte_tol
    spec = program.field
    lattice = _lattice(program, spin_system, flat, extra=spec.nmodes if spec is not None else 0)
    size = lattice.size if lattice else 1
    nstates = rho0.size
    modes = lattice.modes if lattice else np.zeros(1, dtype=int)
    k0 = lattice.k0 if lattice else 0.0
    axis = lattice.axis if lattice else np.array([0.0, 0.0, 1.0])
    envelope = lattice.envelope(program.sample) if lattice else np.ones(1)
    D = program.diffusion
    acquire = flat.acquire
    read_units = 0
    if lattice and acquire.gradient is not None:
        read_units = lattice.units(acquire.gradient.amplitude * acquire.timestep, "the read gradient")
    fields: dict[str, Any] = {}
    if spec is not None:
        if any(kind in ("evolve", "refocus") for kind, _ in flat.events):
            raise NotImplementedError(
                "Evolve and Refocus are not available with a dipolar field (read the build-up during acquisition)."
            )
        angle = math.acos(min(1.0, abs(float(axis[2]))))
        for key, L0 in (("full", L), ("relaxation", L_relax)):
            fields[key] = dipolar_field(
                spin_system,
                L0,
                spec.isotope,
                spec.tau_d,
                angle=angle,
                demagnetising=spec.demagnetising,
                nmodes=lattice.half if lattice else 0,
                k0=k0,
                rho_eq=rho0,
                diffusion=D,
                weights=spec.weights,
            )
        sub_dt = float(spec.timestep) if spec.timestep else acquire.timestep
        if sub_dt <= 0:
            raise ValueError("the field's timestep must be positive.")

    def advance(rho: np.ndarray, duration: float, key: str) -> np.ndarray:
        """``duration`` of evolution under the dipolar field, in sub-steps of the spec's timestep."""
        generator = fields[key]
        nsub = max(1, int(math.ceil(duration / sub_dt - 1e-9)))
        h = duration / nsub
        vec = generator.stack(rho[:, :, 0])
        for _ in range(nsub):
            vec = iserstep(spin_system, generator, 0.0, spec.method, vec, h)
        return generator.columns(vec)[:, :, None]

    def hold(rho: np.ndarray, duration: float) -> np.ndarray:
        """Free diffusion for ``duration`` at fixed wavevectors."""
        if D and lattice:
            rho = rho * np.exp(-D * (modes * k0) ** 2 * duration)[None, :, None]
        return rho

    def ramp(rho: np.ndarray, m: int, duration: float, held: bool = False) -> np.ndarray:
        """A gradient of ``m`` units lasting ``duration``: the diffusion weight of the ramp (less the weight of a
        hold at the starting wavevectors when ``held``, the generator having applied it), then the mode shift."""
        shifts = lattice.shifts(m, np.abs(rho).max(axis=(1, 2)), tol)
        if D:
            n0 = modes[None, :]
            n1 = n0 + shifts[:, None]
            b = duration * k0**2 * (n0**2 + n0 * n1 + n1**2) / 3
            if held:
                b = b - duration * k0**2 * n0**2
            rho = rho * np.exp(-D * b)[:, :, None]
        return _shift_modes(rho, shifts, tol, strict=spec is None)

    total = None
    for m_step in range(flat.steps):
        phases = flat.phases_rad(m_step)
        rho = np.zeros((nstates, size, 1), dtype=complex)
        rho[:, size // 2, 0] = rho0
        shape: list[int] = []
        for kind, payload in flat.events:
            ncols = rho.shape[2]
            if kind == "pulse":
                pulse = flat.pulses[payload]
                phi = phases[payload]
                op = math.cos(phi) * operator(spin_system, "Lx", pulse.spins) + math.sin(phi) * operator(
                    spin_system, "Ly", pulse.spins
                )
                rho = step(spin_system, op, rho.reshape(nstates, -1), pulse.angle).reshape(nstates, size, ncols)
            elif kind == "delay":
                if spec is not None:
                    rho = advance(rho, payload.duration, payload.generator)
                    continue
                generator = L if payload.generator == "full" else L_relax
                flat_rho = evolution(spin_system, generator, rho.reshape(nstates, -1), payload.duration, 1, "final")
                rho = hold(flat_rho.reshape(nstates, size, ncols), payload.duration)
            elif kind == "gradient":
                units = lattice.units(payload.amplitude * payload.duration, "a gradient")
                if spec is not None:
                    rho = advance(rho, payload.duration, "full")
                    rho = ramp(rho, units, payload.duration, held=True) if units else rho
                    continue
                flat_rho = evolution(spin_system, L, rho.reshape(nstates, -1), payload.duration, 1, "final")
                rho = flat_rho.reshape(nstates, size, ncols)
                rho = ramp(rho, units, payload.duration) if units else hold(rho, payload.duration)
            elif kind == "evolve":
                t = np.arange(payload.npoints) * payload.timestep
                columns = rho.reshape(nstates, -1)
                flat_rho = np.hstack(
                    [
                        evolution(spin_system, L, columns[:, c], payload.timestep, payload.npoints - 1, "trajectory")
                        for c in range(columns.shape[1])
                    ]
                )
                rho = flat_rho.reshape(nstates, size, ncols * payload.npoints)
                if D and lattice:
                    weight = np.exp(-D * (modes[:, None] * k0) ** 2 * np.tile(t, ncols)[None, :])
                    rho = rho * weight[None, :, :]
                shape.append(payload.npoints)
            elif kind == "refocus":
                n_last = shape[-1]
                flat_rho = _refocus(spin_system, L, rho.reshape(nstates, -1), payload.timestep, n_last)
                rho = flat_rho.reshape(nstates, size, ncols)
                if D and lattice:
                    t = np.tile(np.arange(n_last) * payload.timestep, ncols // n_last)
                    rho = rho * np.exp(-D * (modes[:, None] * k0) ** 2 * t[None, :])[None, :, :]
            elif kind == "homospoil":
                flag = "keep" if payload.keep == "zqc" else "destroy"
                rho = homospoil(spin_system, rho.reshape(nstates, -1).T, flag).T.reshape(nstates, size, ncols)
            elif kind == "acquire":
                n, dt = payload.npoints, payload.timestep
                if spec is not None:
                    signals = np.zeros((n, size, ncols), dtype=complex)
                    for i in range(n):
                        signals[i] = np.einsum("s,smc->mc", np.conj(coil), rho)
                        if i == n - 1:
                            break
                        rho = advance(rho, dt, "full")
                        if read_units:
                            rho = ramp(rho, read_units, dt, held=True)
                elif read_units:
                    signals = np.zeros((n, size, ncols), dtype=complex)
                    P = propagator(spin_system, L, dt)
                    for i in range(n):
                        signals[i] = np.einsum("s,smc->mc", np.conj(coil), rho)
                        if i == n - 1:
                            break
                        rho = (P @ rho.reshape(nstates, -1)).reshape(nstates, size, ncols)
                        rho = ramp(rho, read_units, dt)
                else:
                    signals = evolution(spin_system, L, rho.reshape(nstates, -1), dt, n - 1, "observable", coil=coil)
                    signals = signals.reshape(n, size, ncols)
                    if D and lattice:
                        t = np.arange(n) * dt
                        signals = signals * np.exp(-D * (modes[None, :] * k0) ** 2 * t[:, None])[:, :, None]
                signals = np.moveaxis(signals, 1, 0).reshape(size, n, *shape) * np.exp(1j * flat.receiver_rad(m_step))
                total = signals if total is None else total + signals
    signals = total / flat.steps
    fids = signals * envelope.reshape(-1, *([1] * (signals.ndim - 1)))
    if output == "modes":
        return ModeFids(fids, signals, modes, k0, axis, envelope)
    return fids.sum(axis=0)
