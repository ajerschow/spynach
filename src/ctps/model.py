"""Data model: nuclei, pulses, gradients, periods, allowed coherence orders, phase cycles, results.

Conventions follow Jerschow & Müller, J. Magn. Reson. 134, 17 (1998): a sequence of ``n``
pulses has ``n + 1`` periods numbered ``0..n`` (period 0 before the first pulse, period
``n`` the detection period); coherence orders change only at pulses; ``Δp_j = p_j - p_{j-1}``
is the order jump at pulse ``j`` for the nucleus the pulse acts on; a pulse phase ``φ``
multiplies the signal by ``exp(-i Δp φ)`` and the receiver phase ``φ_rec`` by
``exp(-i φ_rec)`` (Eq. 12). Gradient areas ``h`` are ``∫G dt`` per period and axis in T·s/m;
gyromagnetic ratios are in rad/s/T; delays are in seconds.

Author: Alexej Jerschow.
"""

from __future__ import annotations

import math
from collections.abc import Sequence as SequenceABC
from dataclasses import dataclass, field
from typing import Any

import numpy as np

GAMMA = {  # rad/s/T, the common cases; any nucleus can be given explicitly
    "1H": 2.6752218744e8,
    "2H": 4.10662791e7,
    "13C": 6.728284e7,
    "15N": -2.7116e7,
    "19F": 2.51815e8,
    "23Na": 7.0808493e7,
    "31P": 1.08394e8,
}

PULSE_KINDS = ("free", "perfect_180", "imperfect_180", "small_flip", "cccp_1998")


@dataclass(frozen=True)
class Nucleus:
    """A spin species: ``gamma`` in rad/s/T, ``spin`` the quantum number (1/2 by default)."""

    name: str
    gamma: float | None = None
    spin: float = 0.5
    n_spins: int = 1  # coupled spins of this species whose orders add up (bounds the transfer weights)

    def __post_init__(self) -> None:
        if self.n_spins < 1:
            raise ValueError("n_spins must be at least 1.")
        if self.gamma is None:
            if self.name not in GAMMA:
                raise ValueError(f"no gyromagnetic ratio known for {self.name!r}; pass gamma=")
            object.__setattr__(self, "gamma", GAMMA[self.name])
        if self.spin <= 0 or round(2 * self.spin) != 2 * self.spin:
            raise ValueError("spin must be a positive multiple of 1/2.")

    @property
    def p_max(self) -> int:
        """Largest coherence order of a single such spin, ``2 * spin``."""
        return int(round(2 * self.spin))


@dataclass(frozen=True)
class Pulse:
    """One rf pulse on one nucleus.

    ``kind``: ``"free"`` (the order change is governed by the allowed orders only, as for
    excitation and mixing pulses in CCCP.m), ``"perfect_180"`` (exact inversion ``p -> -p``),
    ``"imperfect_180"`` (deviation ``beta`` from pi, axis tilt ``epsilon``) or ``"small_flip"``
    (flip angle ``beta``), the last two weighted after Jerschow, J. Chem. Phys. 113, 979 (2000);
    ``"cccp_1998"`` is the legacy CCCP.m model with parameter ``epsilon`` (parity only).
    ``time`` is the pulse's position in seconds from the start (used by flow weighting only).
    """

    nucleus: str
    kind: str = "free"
    beta: float = 0.0
    epsilon: float = 0.0
    time: float | None = None
    label: str = ""

    def __post_init__(self) -> None:
        if self.kind not in PULSE_KINDS:
            raise ValueError(f"pulse kind must be one of {PULSE_KINDS}, not {self.kind!r}.")


@dataclass(frozen=True)
class Period:
    """A free-precession period: its ``duration`` (s) and the gradient areas ``(hx, hy, hz)`` in T·s/m.

    The gradient is treated as a delta function at the start of the period (1998 Eq. 41);
    ``duration`` is then the delay to the next gradient. Period 0 precedes the first pulse.
    """

    duration: float = 0.0
    gradient: tuple[float, float, float] = (0.0, 0.0, 0.0)

    @staticmethod
    def with_gradient(duration: float, axis: str, amplitude: float, length: float) -> Period:
        """A period whose gradient of ``amplitude`` (T/m) lasts ``length`` seconds along ``axis``."""
        area = [0.0, 0.0, 0.0]
        area["xyz".index(axis)] = amplitude * length
        return Period(duration, tuple(area))


@dataclass(frozen=True)
class Sample:
    """Sample geometry for gradient attenuation: ``z_max`` half the active length, ``x_max`` the radius (m)."""

    z_max: float = 5e-3
    x_max: float = 2.5e-3


class AllowedOrders:
    """Allowed coherence orders of one nucleus in every period, with optional weights.

    ``weights[p + p_max, period]`` is 0 for a forbidden order and otherwise the empirical
    weight the pathway receives for passing through it (1 unless set), as CCCP.m's
    ``allow_paths`` entries. ``standard(p_max, n_periods)`` allows every order in the
    intermediate periods, only ``start`` before the first pulse and only ``end`` in the
    detection period.
    """

    def __init__(self, p_max: int, n_periods: int, weights: np.ndarray | None = None):
        self.p_max = int(p_max)
        self.n_periods = int(n_periods)
        shape = (2 * self.p_max + 1, self.n_periods)
        if weights is None:
            self.weights = np.ones(shape)
        else:
            self.weights = np.array(weights, dtype=float)
            if self.weights.shape != shape:
                raise ValueError(f"weights must have shape {shape} (orders x periods), not {self.weights.shape}.")

    @classmethod
    def standard(
        cls, p_max: int, n_periods: int, start: SequenceABC[int] = (0,), end: SequenceABC[int] = (-1,)
    ) -> AllowedOrders:
        orders = cls(p_max, n_periods)
        orders.only(0, start)
        orders.only(n_periods - 1, end)
        return orders

    def only(self, period: int, orders: SequenceABC[int], weight: float = 1.0) -> AllowedOrders:
        """Allow only ``orders`` in ``period``."""
        self.weights[:, period] = 0.0
        for p in orders:
            self.weights[p + self.p_max, period] = weight
        return self

    def forbid(self, period: int, orders: SequenceABC[int]) -> AllowedOrders:
        for p in orders:
            self.weights[p + self.p_max, period] = 0.0
        return self

    def weight(self, period: int, orders: SequenceABC[int], weight: float) -> AllowedOrders:
        """Give ``orders`` in ``period`` an empirical weight (relaxation, abundance, ...)."""
        for p in orders:
            self.weights[p + self.p_max, period] = weight
        return self

    def orders(self, period: int) -> np.ndarray:
        """Allowed orders of a period, ascending."""
        return np.flatnonzero(self.weights[:, period]) - self.p_max

    def copy(self) -> AllowedOrders:
        return AllowedOrders(self.p_max, self.n_periods, self.weights.copy())


@dataclass(frozen=True)
class Sequence:
    """A pulse sequence: nuclei, pulses in order, the ``n + 1`` periods, allowed orders per nucleus."""

    nuclei: tuple[Nucleus, ...]
    pulses: tuple[Pulse, ...]
    periods: tuple[Period, ...]
    allowed: dict[str, AllowedOrders]
    sample: Sample = Sample()
    diffusion: float = 0.0  # m^2/s
    flow: float = 0.0  # maximum velocity along z, m/s, for a uniform velocity distribution

    def __post_init__(self) -> None:
        if len(self.periods) != len(self.pulses) + 1:
            raise ValueError(f"{len(self.pulses)} pulses need {len(self.pulses) + 1} periods, got {len(self.periods)}.")
        names = [n.name for n in self.nuclei]
        if len(set(names)) != len(names):
            raise ValueError("nucleus names must be unique.")
        for pulse in self.pulses:
            if pulse.nucleus not in names:
                raise ValueError(f"pulse on unknown nucleus {pulse.nucleus!r}; nuclei are {names}.")
        for name in names:
            if name not in self.allowed:
                raise ValueError(f"no allowed orders given for nucleus {name!r}.")
            if self.allowed[name].n_periods != len(self.periods):
                raise ValueError(
                    f"allowed orders of {name!r} cover {self.allowed[name].n_periods} periods, not {len(self.periods)}."
                )

    @classmethod
    def simple(
        cls,
        n_pulses: int,
        p_max: int,
        *,
        start: SequenceABC[int] = (0,),
        end: SequenceABC[int] = (-1,),
        orders: SequenceABC[int] | None = None,
        periods: dict[int, SequenceABC[int]] | None = None,
        nucleus: str = "1H",
        spin: float = 0.5,
    ) -> Sequence:
        """A phase-cycling problem without gradients: one nucleus, ``n_pulses`` free pulses, orders up to ``p_max``.

        ``orders`` restricts every intermediate period to the given set (for example the odd
        orders and zero); ``periods`` overrides the allowed set of single periods by number.
        """
        allowed = AllowedOrders.standard(p_max, n_pulses + 1, start=start, end=end)
        if orders is not None:
            for period in range(1, n_pulses):
                allowed.only(period, orders)
        for period, values in (periods or {}).items():
            allowed.only(period, values)
        return cls(
            (Nucleus(nucleus, spin=spin),),
            tuple(Pulse(nucleus) for _ in range(n_pulses)),
            tuple(Period() for _ in range(n_pulses + 1)),
            {nucleus: allowed},
        )

    @property
    def n_pulses(self) -> int:
        return len(self.pulses)

    @property
    def n_periods(self) -> int:
        return len(self.periods)

    def nucleus(self, name: str) -> Nucleus:
        return next(n for n in self.nuclei if n.name == name)

    def gradient_areas(self) -> np.ndarray:
        """``(n_periods, 3)`` array of gradient areas, the matrix ``ĥ`` of 1998 Eq. 27."""
        return np.array([p.gradient for p in self.periods], dtype=float)

    def durations(self) -> np.ndarray:
        return np.array([p.duration for p in self.periods], dtype=float)

    def pulse_times(self) -> np.ndarray:
        """Time of each pulse from the start: ``Pulse.time`` if given, else from the period durations."""
        times = np.cumsum(self.durations())[:-1]
        given = [p.time for p in self.pulses]
        return np.array([t if t is not None else d for t, d in zip(given, times)], dtype=float)

    def _repr_html_(self) -> str:
        pulses = "".join(
            f"<tr><td>{j + 1}</td><td>{p.nucleus}</td><td>{p.kind}</td>"
            f"<td>{p.beta:g}</td><td>{p.epsilon:g}</td><td>{p.label}</td></tr>"
            for j, p in enumerate(self.pulses)
        )
        periods = "".join(
            f"<tr><td>{i}</td><td>{p.duration * 1e3:g}</td>"
            f"<td>{', '.join(f'{a}: {v:g}' for a, v in zip('xyz', p.gradient) if v) or ''}</td></tr>"
            for i, p in enumerate(self.periods)
        )
        grids = ""
        for name, allowed in self.allowed.items():
            head = "".join(f"<th>{i}</th>" for i in range(allowed.n_periods))
            rows = ""
            for p in range(allowed.p_max, -allowed.p_max - 1, -1):
                cells = ""
                for i in range(allowed.n_periods):
                    w = allowed.weights[p + allowed.p_max, i]
                    style = f"text-align:center;background:{'#2a78d6' if w else 'transparent'};opacity:{w if w else 1}"
                    cells += f"<td style='{style}'>{'' if w else '·'}</td>"
                rows += f"<tr><th>{p:+d}</th>{cells}</tr>"
            grids += (
                f"<p>allowed orders of {name} (blue = allowed; period across, order down)</p>"
                f"<table><tr><th></th>{head}</tr>{rows}</table>"
            )
        return (
            f"<div><b>Sequence</b>: {', '.join(n.name for n in self.nuclei)}, {self.n_pulses} pulses, "
            f"{self.n_periods} periods; D = {self.diffusion:g} m²/s; sample z_max {self.sample.z_max * 1e3:g} mm, "
            f"x_max {self.sample.x_max * 1e3:g} mm"
            f"<table><tr><th>pulse</th><th>nucleus</th><th>kind</th><th>β</th><th>ε</th><th>label</th></tr>{pulses}</table>"
            f"<table><tr><th>period</th><th>duration / ms</th><th>gradient areas / T·s/m</th></tr>{periods}</table>"
            f"{grids}</div>"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "nuclei": [{"name": n.name, "gamma": n.gamma, "spin": n.spin, "n_spins": n.n_spins} for n in self.nuclei],
            "pulses": [
                {
                    "nucleus": p.nucleus,
                    "kind": p.kind,
                    "beta": p.beta,
                    "epsilon": p.epsilon,
                    "time": p.time,
                    "label": p.label,
                }
                for p in self.pulses
            ],
            "periods": [{"duration": p.duration, "gradient": list(p.gradient)} for p in self.periods],
            "allowed": {name: {"p_max": a.p_max, "weights": a.weights.tolist()} for name, a in self.allowed.items()},
            "sample": {"z_max": self.sample.z_max, "x_max": self.sample.x_max},
            "diffusion": self.diffusion,
            "flow": self.flow,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Sequence:
        periods = tuple(Period(p["duration"], tuple(p["gradient"])) for p in data["periods"])
        return cls(
            nuclei=tuple(Nucleus(**n) for n in data["nuclei"]),
            pulses=tuple(Pulse(**p) for p in data["pulses"]),
            periods=periods,
            allowed={
                name: AllowedOrders(a["p_max"], len(periods), np.array(a["weights"]))
                for name, a in data["allowed"].items()
            },
            sample=Sample(**data.get("sample", {})),
            diffusion=data.get("diffusion", 0.0),
            flow=data.get("flow", 0.0),
        )


@dataclass(frozen=True)
class PhaseCycle:
    """Phase lists in units of ``2π / base`` per pulse, plus the receiver list, as CCCP's ``ph`` and ``base``.

    Lists shorter than the longest are repeated cyclically (CCCP.m). ``base`` is the number
    of phase steps per turn (4 means 0, 90, 180, 270 degrees); the receiver may have its own
    ``receiver_base``. ``steps`` is the cycle length ``N``.
    """

    phases: tuple[tuple[int, ...], ...]
    receiver: tuple[int, ...]
    base: int = 4
    receiver_base: int | None = None
    winding: tuple[int, ...] | None = None  # (N, nu_1..nu_n, nu_rec) when built as a cogwheel cycle

    def __post_init__(self) -> None:
        if self.base < 1:
            raise ValueError("base must be at least 1.")
        if any(len(ph) == 0 for ph in self.phases) or len(self.receiver) == 0:
            raise ValueError("every phase list needs at least one entry.")

    @classmethod
    def explicit(cls, phases: SequenceABC[SequenceABC[int]], receiver: SequenceABC[int], base: int = 4) -> PhaseCycle:
        return cls(tuple(tuple(int(v) for v in ph) for ph in phases), tuple(int(v) for v in receiver), base)

    @classmethod
    def degrees(cls, phases: SequenceABC[SequenceABC[float]], receiver: SequenceABC[float]) -> PhaseCycle:
        """Phase lists in degrees; the base becomes 360 (or a divisor that represents every value exactly)."""
        values = [v for ph in phases for v in ph] + list(receiver)
        for base in (4, 8, 12, 16, 24, 36, 72, 360):
            if all(abs(v * base / 360 - round(v * base / 360)) < 1e-9 for v in values):
                break
        return cls(
            tuple(tuple(int(round(v * base / 360)) % base for v in ph) for ph in phases),
            tuple(int(round(v * base / 360)) % base for v in receiver),
            base,
        )

    @classmethod
    def cogwheel(cls, N: int, nu: SequenceABC[int], nu_rec: int) -> PhaseCycle:
        """``COG_N(nu_1, ..., nu_n; nu_rec)``: pulse ``i`` has phase ``2π nu_i m / N`` at step ``m``.

        The receiver has ``2π nu_rec m / N``.
        """
        steps = np.arange(N)
        phases = tuple(tuple(int(v) for v in (n * steps) % N) for n in nu)
        return cls(
            phases, tuple(int(v) for v in (nu_rec * steps) % N), N, N, (int(N), *[int(n) for n in nu], int(nu_rec))
        )

    @classmethod
    def nested(cls, levels: SequenceABC[SequenceABC[int]], receiver: SequenceABC[int], base: int = 4) -> PhaseCycle:
        """Nested (mixed-radix) cycle: each inner list is repeated as a block for the lists before it.

        ``levels[i]`` is the phase list of pulse ``i``; pulse 0 steps fastest. The receiver list
        is given for the full length. Equivalent to CCCP's cyclic expansion when the lists'
        lengths are the nested counts.
        """
        phases = []
        block = 1
        for level in levels:
            level = [int(v) for v in level]
            phases.append(tuple(np.repeat(level, block)))
            block *= len(level)
        total = block
        phases = [tuple(np.tile(ph, total // len(ph))) for ph in phases]
        return cls(tuple(phases), tuple(int(v) for v in receiver), base)

    @property
    def n_pulses(self) -> int:
        return len(self.phases)

    @property
    def steps(self) -> int:
        return int(np.lcm.reduce([len(ph) for ph in self.phases] + [len(self.receiver)]))

    def phase_matrix(self) -> np.ndarray:
        """``(n_pulses, N)`` pulse phases in radians, lists expanded cyclically to ``N`` steps."""
        n = self.steps
        return np.array([2 * math.pi * np.resize(ph, n) / self.base for ph in self.phases], dtype=float)

    def receiver_phases(self) -> np.ndarray:
        """``(N,)`` receiver phases in radians."""
        base = self.receiver_base or self.base
        return 2 * math.pi * np.resize(self.receiver, self.steps) / base

    def integer_matrix(self) -> tuple[np.ndarray, np.ndarray, int]:
        """Phases as integers on a common grid: ``(pulse phases (n, N), receiver (N,), fullbase)``."""
        base_rec = self.receiver_base or self.base
        fullbase = int(np.lcm(self.base, base_rec))
        n = self.steps
        phases = np.array([np.resize(ph, n) * (fullbase // self.base) for ph in self.phases], dtype=np.int64)
        receiver = np.resize(self.receiver, n) * (fullbase // base_rec)
        return phases % fullbase, receiver.astype(np.int64) % fullbase, fullbase

    def cog_string(self) -> str:
        if self.winding is None:
            return f"explicit cycle, {self.steps} steps, base {self.base}"
        N, *nu, rec = self.winding
        return f"COG{N}({','.join(str(v) for v in nu)};{rec})"

    def to_dict(self) -> dict[str, Any]:
        return {
            "phases": [list(ph) for ph in self.phases],
            "receiver": list(self.receiver),
            "base": self.base,
            "receiver_base": self.receiver_base,
            "winding": list(self.winding) if self.winding else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PhaseCycle:
        return cls(
            tuple(tuple(ph) for ph in data["phases"]),
            tuple(data["receiver"]),
            data.get("base", 4),
            data.get("receiver_base"),
            tuple(data["winding"]) if data.get("winding") else None,
        )

    def _repr_html_(self) -> str:
        n = self.steps
        rows = "".join(
            f"<tr><th>pulse {i}</th>{''.join(f'<td>{v}</td>' for v in np.resize(ph, n))}</tr>"
            for i, ph in enumerate(self.phases)
        )
        rec = "".join(f"<td>{v}</td>" for v in np.resize(self.receiver, n))
        return (
            f"<div><b>{self.cog_string()}</b>, {n} steps, phases in units of 360/{self.base} degrees"
            f"<table><tr><th></th>{''.join(f'<th>{m}</th>' for m in range(n))}</tr>{rows}"
            f"<tr><th>receiver</th>{rec}</tr></table></div>"
        )


@dataclass
class PathwayResult:
    """Ranked pathways of a simulation with their factored weights.

    ``orders``: ``(k, n_periods)`` total coherence order per period (``int8``); ``per_nucleus``:
    the same per nucleus name; ``pc`` the complex phase-cycle factor ``s / N`` (1998 Eq. 17);
    ``grad``, ``diff``, ``pulse`` the real attenuation factors; ``total = pc * grad * diff * pulse``;
    ``phase`` the signal phase ``arg(total)`` in degrees, the ``ψ`` column of the 1998 tables; ``wanted`` marks
    the pathways the user asked for; ``n_enumerated`` is the number of pathways before any threshold.
    """

    orders: np.ndarray
    per_nucleus: dict[str, np.ndarray]
    pc: np.ndarray
    grad: np.ndarray
    diff: np.ndarray
    pulse: np.ndarray
    wanted: np.ndarray
    n_enumerated: int
    sequence: Sequence
    cycle: PhaseCycle
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def total(self) -> np.ndarray:
        return self.pc * self.grad * self.diff * self.pulse

    @property
    def amplitude(self) -> np.ndarray:
        return np.abs(self.total)

    @property
    def phase(self) -> np.ndarray:
        """Signal phase ``arg(total)`` in degrees modulo 360, the ``ψ`` column of the 1998 tables."""
        return np.degrees(np.angle(self.total)) % 360

    def __len__(self) -> int:
        return len(self.orders)

    def __getitem__(self, index: int) -> Pathway:
        return Pathway(self, int(index))

    def top(self, k: int) -> PathwayResult:
        """The ``k`` strongest pathways."""
        keep = np.arange(min(k, len(self)))
        return self.select(keep)

    def select(self, index: np.ndarray) -> PathwayResult:
        return PathwayResult(
            self.orders[index],
            {name: o[index] for name, o in self.per_nucleus.items()},
            self.pc[index],
            self.grad[index],
            self.diff[index],
            self.pulse[index],
            self.wanted[index],
            self.n_enumerated,
            self.sequence,
            self.cycle,
            dict(self.extra),
        )

    def leakage(self, threshold: float = 0.01) -> int:
        """Number of unwanted pathways with an amplitude above ``threshold`` times the strongest wanted one."""
        if len(self) == 0:
            return 0
        reference = self.amplitude[self.wanted].max() if self.wanted.any() else self.amplitude.max()
        return int(np.sum(~self.wanted & (self.amplitude > threshold * reference)))

    def table(self) -> list[dict[str, Any]]:
        rows = []
        for i in range(len(self)):
            rows.append(
                {
                    "rank": i + 1,
                    "pathway": self.orders[i].tolist(),
                    "wanted": bool(self.wanted[i]),
                    "total": float(self.amplitude[i]),
                    "phase_deg": float(self.phase[i]),
                    "pc": float(abs(self.pc[i])),
                    "grad": float(self.grad[i]),
                    "diff": float(self.diff[i]),
                    "pulse": float(self.pulse[i]),
                }
            )
        return rows

    def _repr_html_(self) -> str:
        head = (
            "<tr><th>#</th><th>pathway</th><th>total</th><th>ψ / °</th>"
            "<th>pc</th><th>grad</th><th>diff</th><th>pulse</th></tr>"
        )
        rows = []
        for r in self.table()[:50]:
            mark = " style='font-weight:bold'" if r["wanted"] else ""
            path = ",".join(f"{p:+d}" if p else "0" for p in r["pathway"])
            rows.append(
                f"<tr{mark}><td>{r['rank']}</td><td><code>{path}</code></td><td>{r['total']:.4g}</td>"
                f"<td>{r['phase_deg']:.0f}</td><td>{r['pc']:.4g}</td><td>{r['grad']:.4g}</td>"
                f"<td>{r['diff']:.4g}</td><td>{r['pulse']:.4g}</td></tr>"
            )
        more = f"<p>{len(self) - 50} more pathways not shown</p>" if len(self) > 50 else ""
        wanted = int(self.wanted.sum())
        summary = (
            f"<p>{self.n_enumerated} pathways enumerated, {len(self)} kept; {wanted} wanted; "
            f"{self.leakage(0.01)} unwanted pathways above 1% of the strongest wanted one.</p>"
        )
        return f"<div>{summary}<table>{head}{''.join(rows)}</table>{more}</div>"


class Pathway:
    """One pathway of a result, with its factored weights."""

    def __init__(self, result: PathwayResult, index: int):
        self.result = result
        self.index = index

    @property
    def orders(self) -> np.ndarray:
        return self.result.orders[self.index]

    @property
    def factors(self) -> dict[str, float]:
        r, i = self.result, self.index
        return {
            "pc": float(abs(r.pc[i])),
            "grad": float(r.grad[i]),
            "diff": float(r.diff[i]),
            "pulse": float(r.pulse[i]),
        }

    @property
    def amplitude(self) -> float:
        return float(self.result.amplitude[self.index])

    def __repr__(self) -> str:
        path = ",".join(str(p) for p in self.orders)
        f = self.factors
        return (
            f"Pathway({path}: total={self.amplitude:.4g}, pc={f['pc']:.4g}, grad={f['grad']:.4g}, "
            f"diff={f['diff']:.4g}, pulse={f['pulse']:.4g})"
        )

    def _repr_html_(self) -> str:
        f = self.factors

        def bar(v: float) -> str:
            width = max(2, int(120 * (1 + math.log10(max(v, 1e-6)) / 6)))
            return f"<div style='background:#2a78d6;height:0.8em;width:{width}px'></div>"

        bars = "".join(f"<tr><td>{name}</td><td>{bar(v)}</td><td>{v:.4g}</td></tr>" for name, v in f.items())
        per_nucleus = "".join(
            f"<tr><td>{name}</td><td><code>{','.join(str(p) for p in o[self.index])}</code></td></tr>"
            for name, o in self.result.per_nucleus.items()
        )
        return (
            f"<div><b>Pathway {self.index + 1}</b>: <code>{','.join(str(p) for p in self.orders)}</code>, "
            f"total {self.amplitude:.4g}, phase {self.result.phase[self.index]:.0f}°"
            f"<table>{per_nucleus}</table>"
            f"<table><tr><th>factor</th><th>log scale</th><th>value</th></tr>{bars}</table></div>"
        )
