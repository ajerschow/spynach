"""Spin system description and its construction (port of ``create.m``).

Spinach keeps everything in one nested ``spin_system`` struct that functions
copy, modify, and return. spynach keeps that functional style with frozen
dataclasses: :func:`create` builds a :class:`SpinSystem`, and later stages
(:func:`spynach.basis`, :func:`spynach.assume`) return modified copies.

Spin indices are 0-based throughout spynach (Spinach uses 1-based indices).

Ported so far: isotopes, magnet, user tolerances, Zeeman interactions as
scalars, tensors, or eigenvalues with Euler angles, couplings as scalars,
tensors, or eigenvalues with Euler angles, atomic coordinates (point
dipolar couplings and the proximity graph), and the chemical subsystem
description (``inter['chem']``: parts, concentrations, first-order
exchange rates), and the relaxation settings (``inter['relaxation']`` with
the ``'damp'``, ``'t1_t2'``, ``'redfield'``, ``'SRFK'`` and ``'SRSK'``
theories, ``rlx_keep``, ``equilibrium``, ``tau_c``, ``temperature``,
``damp_rate``, ``r1_rates``/``r2_rates``, ``rlx_dfs``, ``srfk_tau_c``,
``srfk_mdepth``, ``srsk_sources``). Every other ``sys`` or ``inter``
field raises ``NotImplementedError`` rather than being ignored.

Spinach authors, by original file (headers of Spinach 2.13; see docs/spinach-credits.md):
  create.m: Ilya Kuprov, Hannah Hogben, Konstantin Pervushin, Luke Edwards
  dilute.m: Ilya Kuprov, Luke Edwards, Bud Macaulay
  dipolar.m: Ilya Kuprov
  kill_spin.m: Ilya Kuprov, Luke Edwards
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from spynach.conventions import euler2dcm
from spynach.isotopes import particle_type, spin
from spynach.tolerances import Tolerances, tolerances

FREE_ELECTRON_G = 2.00231930436092  # Spinach tols.freeg

SYS_FIELDS = {"isotopes", "magnet", "tols", "labels", "enable", "disable"}
# Spinach sys.enable / sys.disable tokens that have a meaning in spynach
ENABLE_TOKENS = {"krylov"}
DISABLE_TOKENS = {"zte", "pt", "dss", "merge", "krylov", "trajlevel"}
INTER_FIELDS = {
    "zeeman",
    "coupling",
    "chem",
    "coordinates",
    "relaxation",
    "rlx_keep",
    "equilibrium",
    "tau_c",
    "temperature",
    "damp_rate",
    "r1_rates",
    "r2_rates",
    "rlx_dfs",
    "srfk_tau_c",
    "srfk_mdepth",
    "srsk_sources",
}
RELAXATION_THEORIES = {"damp", "t1_t2", "redfield", "SRFK", "SRSK"}
UNPORTED_THEORIES = {"lindblad", "nottingham", "weizmann", "naka-zwan"}
ZEEMAN_FIELDS = {"scalar", "matrix", "eigs", "euler"}
COUPLING_FIELDS = {"scalar", "matrix", "eigs", "euler"}
CHEM_FIELDS = {"parts", "concs", "rates"}


@dataclass(frozen=True, eq=False)
class Basis:
    """Basis set information (Spinach ``spin_system.bas``).

    ``basis`` has one row per basis state and one column per spin. In
    ``zeeman-hilb`` it holds Zeeman level numbers (1 = highest projection);
    in ``sphten-liouv`` the linear spherical tensor index ``l*l + l - m`` of
    each spin (0 = unit operator), see :mod:`spynach.basis`.
    """

    formalism: str
    approximation: str
    basis: np.ndarray
    tot_proj: np.ndarray | None = None  # (nstates,) total coherence order (sphten-liouv)
    tot_cord: np.ndarray | None = None  # (nstates,) spin correlation order (sphten-liouv)
    # Row hashing for superoperator assembly (sphten-liouv): hash_table[spin, index] holds a random
    # 64-bit key per (spin, tensor index); row_hash is their sum over the columns of a row, and
    # sort_order/sorted_hash let a row be found from its hash by bisection (operators.superop)
    hash_table: np.ndarray | None = None
    row_hash: np.ndarray | None = None
    sort_order: np.ndarray | None = None
    sorted_hash: np.ndarray | None = None
    # Superoperator triplets by (opspec bytes, side), filled by operators.superop (Spinach's op_cache);
    # a frozen dataclass may still mutate the dictionary it holds
    op_cache: dict = field(default_factory=dict, repr=False, compare=False)

    @property
    def nstates(self) -> int:
        return self.basis.shape[0]


@dataclass(frozen=True, eq=False)
class Assumptions:
    """Interaction truncation levels (Spinach ``inter.*.strength``), set by :func:`spynach.assume`."""

    name: str
    zeeman: tuple[str, ...]  # per spin: 'full', 'secular', or 'ignore'
    coupling: tuple[tuple[str, ...], ...]  # per spin pair: 'strong', 'secular', 'zz', 'ignore', ...


@dataclass(frozen=True, eq=False)
class Chemistry:
    """Chemical subsystems (Spinach ``spin_system.chem``).

    Every spin belongs to exactly one part; a system without ``inter['chem']``
    is one part holding all spins at unit concentration. ``rates`` is the
    first-order exchange matrix in Hz, ``dc/dt = rates @ c``: ``rates[i, j]``
    is the flow from part ``j`` into part ``i`` and every column sums to zero.
    """

    parts: tuple[tuple[int, ...], ...]  # 0-based spin indices of each part, ascending
    concs: np.ndarray  # (nparts,) initial concentrations
    rates: np.ndarray | None  # (nparts, nparts) exchange rates, or None

    @property
    def nparts(self) -> int:
        return len(self.parts)


@dataclass(frozen=True, eq=False)
class Relaxation:
    """Relaxation settings (Spinach ``spin_system.rlx``), read by :func:`spynach.relaxation`.

    ``theories`` is the list from ``inter['relaxation']`` (empty: no
    relaxation). ``keep`` is the retention policy (``'diagonal'``, ``'kite'``,
    ``'secular'``, ``'labframe'``), ``equilibrium`` the thermalisation
    (``'zero'``, ``'IME'``, ``'dibari'``), ``dfs`` what to do with dynamic
    frequency shifts (``'ignore'`` or ``'keep'``). ``tau_c`` holds one array
    of one to three rotational correlation times (s) per chemical
    subsystem; ``r1_rates``/``r2_rates`` one scalar or 3x3 tensor (Hz) per
    spin; ``damp_rate`` a scalar or 3x3 tensor (Hz).

    Scalar relaxation of the first kind (``'SRFK'``): ``srfk_tau_c`` holds
    ``(weight, tau)`` pairs of the exponential components of the coupling
    modulation correlation function (s), ``srfk_mdepth`` the root mean
    square modulation depths (Hz) of the scalar couplings in an
    ``nspins x nspins`` layout, ``None`` where a pair is not modulated.
    Scalar relaxation of the second kind (``'SRSK'``): ``srsk_sources``
    lists the 0-based indices of the fast-relaxing source spins.
    """

    theories: tuple[str, ...] = ()
    keep: str = "labframe"
    equilibrium: str = "zero"
    dfs: str = "ignore"
    temperature: float = 298.0
    tau_c: tuple[np.ndarray, ...] = ()
    damp_rate: np.ndarray | None = None
    r1_rates: tuple[np.ndarray, ...] = ()
    r2_rates: tuple[np.ndarray, ...] = ()
    srfk_tau_c: tuple[tuple[float, float], ...] = ()
    srfk_mdepth: tuple[tuple[float | None, ...], ...] = ()
    srsk_sources: tuple[int, ...] = ()


@dataclass(frozen=True, eq=False)
class SpinSystem:
    isotopes: tuple[str, ...]
    labels: tuple[str | None, ...]
    types: tuple[str, ...]  # 'S' for spins; 'C', 'V', 'T' for bosonic modes
    mults: np.ndarray  # (nspins,) int multiplicities
    gammas: np.ndarray  # (nspins,) magnetogyric ratios, rad/(s*T)
    magnet: float  # Tesla
    basefrqs: np.ndarray  # (nspins,) carrier frequencies -gamma*B0, rad/s
    zeeman_matrix: tuple[np.ndarray, ...]  # 3x3 lab-frame Zeeman tensors, rad/s
    coupling_matrix: tuple[tuple[np.ndarray | None, ...], ...]  # 3x3 coupling tensors, rad/s; None if absent
    coordinates: tuple[np.ndarray | None, ...]  # (3,) positions in Angstrom, None if not given
    proxmatrix: np.ndarray  # (nspins, nspins) bool: pairs closer than prox_cutoff
    tols: Tolerances
    chem: Chemistry
    rlx: Relaxation = Relaxation()
    enable: frozenset[str] = frozenset()  # sys['enable'] tokens, e.g. 'krylov'
    disable: frozenset[str] = frozenset()  # sys['disable'] tokens: 'zte', 'pt', 'dss', 'merge', 'krylov', 'trajlevel'
    basis: Basis | None = None
    assumptions: Assumptions | None = None

    @property
    def nspins(self) -> int:
        return len(self.isotopes)


def create(sys: dict[str, Any], inter: dict[str, Any]) -> SpinSystem:
    """Build a spin system from Spinach-style ``sys`` and ``inter`` dictionaries.

    ``sys['isotopes']``: list of isotope names; ``sys['magnet']``: field in Tesla;
    optional ``sys['tols']``: tolerance overrides; optional ``sys['labels']``;
    optional ``sys['enable']`` (``'krylov'``) and ``sys['disable']`` (``'zte'``,
    ``'pt'``, ``'dss'``, ``'merge'``, ``'krylov'``, ``'trajlevel'``) switch
    the trajectory-level reductions and the Krylov propagation path as in
    Spinach; other tokens are not ported.

    ``inter['zeeman']['scalar']``: per-spin chemical shift in ppm (nuclei) or
    isotropic g-value (electrons), ``None`` for none. ``inter['zeeman']['matrix']``:
    per-spin 3x3 tensors in the same units. ``inter['coupling']['scalar']``:
    nspins x nspins nested list of couplings in Hz (``None`` or 0 for none);
    ``inter['coupling']['matrix']``: same layout with 3x3 tensors in Hz.

    Spinach authors (create.m): Ilya Kuprov, Hannah Hogben, Konstantin Pervushin, Luke Edwards.
    """
    _reject_unported(sys, SYS_FIELDS, "sys")
    _reject_unported(inter, INTER_FIELDS, "inter")
    isotopes = tuple(sys["isotopes"])
    nspins = len(isotopes)
    if nspins == 0:
        raise ValueError("sys['isotopes'] must not be empty.")
    tols = tolerances(sys.get("tols"))
    labels = tuple(sys.get("labels", [None] * nspins))
    if len(labels) != nspins:
        raise ValueError("sys['labels'] must have one entry per spin.")
    enable = frozenset(sys.get("enable", []))
    disable = frozenset(sys.get("disable", []))
    if "gpu" in enable:
        raise NotImplementedError(
            "sys['enable'] 'gpu' is not available yet: the GPU backend is planned, see porting/gpu-notes.md"
        )
    if enable - ENABLE_TOKENS:
        raise NotImplementedError(f"sys['enable'] token(s) {sorted(enable - ENABLE_TOKENS)} not ported yet")
    if disable - DISABLE_TOKENS:
        raise NotImplementedError(f"sys['disable'] token(s) {sorted(disable - DISABLE_TOKENS)} not ported yet")

    gammas = np.empty(nspins)
    mults = np.empty(nspins, dtype=np.int64)
    for n, name in enumerate(isotopes):
        gammas[n], mults[n] = spin(name)
    magnet = float(sys["magnet"])
    basefrqs = -gammas * magnet

    zeeman = _zeeman_tensors(isotopes, basefrqs, inter.get("zeeman"))
    chem = _chemistry(isotopes, inter.get("chem"))
    coordinates = _coordinates(nspins, inter.get("coordinates"))
    dipolar, proxmatrix = _dipolar(coordinates, gammas, chem, tols)
    coupling = _coupling_tensors(mults, tols, inter.get("coupling"), dipolar)
    _check_no_couplings_between_parts(chem, coupling)
    rlx = _relaxation_settings(nspins, chem, inter)

    return SpinSystem(
        isotopes=isotopes,
        labels=labels,
        types=tuple(particle_type(name) for name in isotopes),
        mults=mults,
        gammas=gammas,
        magnet=magnet,
        basefrqs=basefrqs,
        zeeman_matrix=zeeman,
        coupling_matrix=coupling,
        coordinates=coordinates,
        proxmatrix=proxmatrix,
        tols=tols,
        chem=chem,
        rlx=rlx,
        enable=enable,
        disable=disable,
    )


RELAXATION_OPTIONS = (
    "rlx_keep", "equilibrium", "tau_c", "damp_rate", "r1_rates", "r2_rates", "rlx_dfs",
    "srfk_tau_c", "srfk_mdepth", "srsk_sources",
)  # fmt: skip


def _relaxation_settings(nspins: int, chem: Chemistry, inter: dict[str, Any]) -> Relaxation:
    """Relaxation inputs, validated as in the relaxation section of create.m, one helper per theory."""
    theories = tuple(inter.get("relaxation", []))
    temperature = float(inter.get("temperature", 298.0))
    if temperature < 0:
        raise ValueError("inter['temperature'] must be non-negative.")
    if not theories:
        for name in RELAXATION_OPTIONS:
            if name in inter:
                raise ValueError(f"inter['{name}'] requires a relaxation theory in inter['relaxation'].")
        return Relaxation(temperature=temperature)
    keep, equilibrium, dfs = _relaxation_policies(theories, inter)
    tau_c = _tau_c(theories, chem, inter)
    damp_rate = _damp_rate(theories, inter)
    r1_rates, r2_rates = _t1_t2_rates(theories, nspins, inter)
    srfk_tau_c, srfk_mdepth = _srfk_settings(theories, nspins, inter)
    srsk_sources = _srsk_sources(theories, nspins, inter)
    return Relaxation(
        theories=theories,
        keep=keep,
        equilibrium=equilibrium,
        dfs=dfs,
        temperature=temperature,
        tau_c=tau_c,
        damp_rate=damp_rate,
        r1_rates=r1_rates,
        r2_rates=r2_rates,
        srfk_tau_c=srfk_tau_c,
        srfk_mdepth=srfk_mdepth,
        srsk_sources=srsk_sources,
    )


def _relaxation_policies(theories: tuple[str, ...], inter: dict[str, Any]) -> tuple[str, str, str]:
    """Theories ported, then the retention, equilibrium and dynamic frequency shift policies."""
    unported = set(theories) & UNPORTED_THEORIES
    if unported:
        raise NotImplementedError(
            f"relaxation theor{'y' if len(unported) == 1 else 'ies'} {sorted(unported)} not ported yet"
        )
    if set(theories) - RELAXATION_THEORIES:
        raise ValueError("unrecognised relaxation theory specification.")
    if "rlx_keep" not in inter:
        raise ValueError("relaxation superoperator term retention policy must be specified in inter['rlx_keep'].")
    if inter["rlx_keep"] not in ("diagonal", "kite", "secular", "labframe"):
        raise ValueError("allowed values for inter['rlx_keep'] are 'diagonal', 'kite', 'secular' and 'labframe'.")
    if "equilibrium" not in inter:
        raise ValueError("relaxation destination must be specified in inter['equilibrium'].")
    if inter["equilibrium"] not in ("zero", "IME", "dibari"):
        raise ValueError("allowed values for inter['equilibrium'] are 'zero', 'IME' and 'dibari'.")
    dfs = inter.get("rlx_dfs", "ignore")
    if dfs not in ("keep", "ignore"):
        raise ValueError("allowed values for inter['rlx_dfs'] are 'keep' and 'ignore'.")
    return inter["rlx_keep"], inter["equilibrium"], dfs


def _tau_c(theories: tuple[str, ...], chem: Chemistry, inter: dict[str, Any]) -> tuple[np.ndarray, ...]:
    """Rotational correlation times of Redfield theory, one array of one to three values per substance."""
    if "redfield" not in theories:
        if "tau_c" in inter:
            raise ValueError("inter['tau_c'] requires Redfield relaxation theory.")
        return ()
    if "tau_c" not in inter:
        raise ValueError("correlation time(s) must be specified in inter['tau_c'] with Redfield theory.")
    tau_c = tuple(np.atleast_1d(np.asarray(t, dtype=float)) for t in inter["tau_c"])
    if len(tau_c) != chem.nparts:
        raise ValueError("the number of elements in inter['tau_c'] must match the number of chemical subsystems.")
    for t in tau_c:
        if t.ndim != 1 or not 1 <= t.size <= 3 or np.any(t <= 0):
            raise ValueError("elements of inter['tau_c'] must be lists of one to three positive correlation times.")
    return tau_c


def _damp_rate(theories: tuple[str, ...], inter: dict[str, Any]) -> np.ndarray | None:
    """Non-selective damping rate, a scalar or a 3x3 tensor."""
    if "damp" not in theories:
        return None
    if "damp_rate" not in inter:
        raise ValueError("damping rate or tensor must be specified in inter['damp_rate'] with non-selective damping.")
    damp_rate = np.asarray(inter["damp_rate"], dtype=float)
    if damp_rate.shape not in ((), (3, 3)):
        raise ValueError("inter['damp_rate'] must be a scalar or a 3x3 matrix.")
    return damp_rate


def _t1_t2_rates(
    theories: tuple[str, ...], nspins: int, inter: dict[str, Any]
) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
    """R1 and R2 of the extended T1/T2 model, one scalar or tensor per spin."""
    if "t1_t2" not in theories:
        return (), ()
    if "r1_rates" not in inter or "r2_rates" not in inter:
        raise ValueError(
            "R1 and R2 rates must be specified in inter['r1_rates'] and inter['r2_rates'] with extended T1,T2 theory."
        )
    return _rate_list(inter["r1_rates"], nspins, "r1_rates"), _rate_list(inter["r2_rates"], nspins, "r2_rates")


def _srfk_settings(
    theories: tuple[str, ...], nspins: int, inter: dict[str, Any]
) -> tuple[tuple[tuple[float, float], ...], tuple[tuple[float | None, ...], ...]]:
    """Scalar relaxation of the first kind: ``(weight, tau)`` components and the modulation depth matrix."""
    if "SRFK" not in theories:
        for name in ("srfk_tau_c", "srfk_mdepth"):
            if name in inter:
                raise ValueError(f"inter['{name}'] requires SRFK relaxation theory.")
        return (), ()
    if "srfk_tau_c" not in inter:
        raise ValueError("SRFK requires modulation correlation time to be specified in inter['srfk_tau_c'] variable.")
    if "srfk_mdepth" not in inter:
        raise ValueError("SRFK requires modulation depths to be specified in inter['srfk_mdepth'] variable.")
    components = []
    for component in inter["srfk_tau_c"]:
        pair = np.asarray(component, dtype=float).reshape(-1)
        if pair.size != 2 or not np.all(np.isreal(component)):
            raise ValueError("inter['srfk_tau_c'] must be a list of 2-element [weight, tau] pairs.")
        if pair[1] < 0:
            raise ValueError("correlation times in inter['srfk_tau_c'] must be non-negative.")
        components.append((float(pair[0]), float(pair[1])))
    depths = _check_square(inter["srfk_mdepth"], nspins, "inter['srfk_mdepth']")
    srfk_mdepth = tuple(
        tuple(None if depths[n][k] is None else float(depths[n][k]) for k in range(nspins)) for n in range(nspins)
    )
    return tuple(components), srfk_mdepth


def _srsk_sources(theories: tuple[str, ...], nspins: int, inter: dict[str, Any]) -> tuple[int, ...]:
    """Scalar relaxation of the second kind: the 0-based indices of the fast-relaxing source spins."""
    if "SRSK" not in theories:
        if "srsk_sources" in inter:
            raise ValueError("inter['srsk_sources'] requires SRSK relaxation theory.")
        return ()
    if "srsk_sources" not in inter:
        raise ValueError("SRSK requires source spin list in inter['srsk_sources'] variable.")
    sources = np.atleast_1d(np.asarray(inter["srsk_sources"]))
    if sources.ndim != 1 or sources.size == 0 or not np.issubdtype(sources.dtype, np.integer):
        raise ValueError("inter['srsk_sources'] must be a list of 0-based spin indices.")
    if np.any(sources < 0) or np.any(sources >= nspins) or len(set(sources.tolist())) != sources.size:
        raise ValueError(f"inter['srsk_sources'] must hold unique spin indices between 0 and {nspins - 1}.")
    return tuple(int(k) for k in sources)


def _rate_list(values: Sequence[Any], nspins: int, name: str) -> tuple[np.ndarray, ...]:
    _check_length(values, nspins, f"inter['{name}']")
    out = []
    for value in values:
        if callable(value):
            raise NotImplementedError(f"orientation functions in inter['{name}'] are not ported yet.")
        rate = np.asarray(value, dtype=float)
        if rate.shape not in ((), (3, 3)):
            raise ValueError(f"elements of inter['{name}'] must be scalars or 3x3 matrices.")
        out.append(rate)
    return tuple(out)


def _chemistry(isotopes: tuple[str, ...], chem: dict[str, Any] | None) -> Chemistry:
    """Chemical subsystems, validated as in the chemistry section of create.m."""
    nspins = len(isotopes)
    if chem is None:
        return Chemistry(parts=(tuple(range(nspins)),), concs=np.ones(1), rates=None)
    _reject_unported(chem, CHEM_FIELDS, "inter['chem']")
    if "rates" in chem and "parts" not in chem:
        raise ValueError("subsystem identifiers (inter['chem']['parts']) must be provided.")
    if "rates" in chem and "concs" not in chem:
        raise ValueError("initial concentrations (inter['chem']['concs']) must be provided.")

    if "parts" in chem:
        parts = tuple(tuple(sorted(int(n) for n in part)) for part in chem["parts"])
        for part in parts:
            if not part or len(set(part)) != len(part) or min(part) < 0 or max(part) >= nspins:
                raise ValueError(
                    "elements of inter['chem']['parts'] must be non-empty lists of unique spin indices "
                    f"between 0 and {nspins - 1}."
                )
        assigned = [n for part in parts for n in part]
        if len(set(assigned)) != len(assigned):
            raise ValueError("a given spin can only belong to one chemical subsystem in inter['chem']['parts'].")
        if set(assigned) != set(range(nspins)):
            raise ValueError("every spin must belong to one of the chemical subsystems in inter['chem']['parts'].")
    else:
        parts = (tuple(range(nspins)),)

    if "concs" in chem:
        concs = np.atleast_1d(np.asarray(chem["concs"], dtype=float))
        if concs.ndim != 1 or np.any(concs < 0):
            raise ValueError("inter['chem']['concs'] must be a vector of non-negative real numbers.")
        if concs.size != len(parts):
            raise ValueError("the number of initial concentrations must be equal to the number of chemical species.")
    elif len(parts) == 1:
        concs = np.ones(1)
    else:
        raise ValueError("concentrations (inter['chem']['concs']) must be provided.")

    rates = None
    if "rates" in chem:
        rates = np.asarray(chem["rates"], dtype=float)
        if rates.ndim != 2 or rates.shape[0] != rates.shape[1]:
            raise ValueError("inter['chem']['rates'] must be a real square matrix.")
        if rates.shape[0] != len(parts):
            raise ValueError(
                "both dimensions of inter['chem']['rates'] must be equal to the number of chemical subsystems."
            )
        if np.any(np.abs(rates.sum(axis=0)) > 10 * np.finfo(float).eps * np.linalg.norm(rates, 1)):
            raise ValueError("inter['chem']['rates'] violates conservation of matter: column sums must be negligible.")
        # First-order exchange moves spin states between parts, so the parts must match spin for spin
        sequences = {tuple(isotopes[n] for n in part) for part in parts}
        if len({len(part) for part in parts}) != 1:
            raise ValueError("exchange mode: all chemical subsystems must have the same number of spins.")
        if len(sequences) != 1:
            raise ValueError("exchange mode: isotope sequences in all chemical subsystems must be the same.")
    return Chemistry(parts=parts, concs=concs, rates=rates)


def _check_no_couplings_between_parts(chem: Chemistry, coupling: tuple[tuple[np.ndarray | None, ...], ...]) -> None:
    for a, part_a in enumerate(chem.parts):
        for b, part_b in enumerate(chem.parts):
            if a != b and any(coupling[n][k] is not None for n in part_a for k in part_b):
                raise ValueError("couplings detected between spins in different chemical species.")


def _reject_unported(spec: dict[str, Any], supported: set[str], name: str) -> None:
    unported = set(spec) - supported
    if unported:
        raise NotImplementedError(f"{name} field(s) {sorted(unported)} not ported yet; supported: {sorted(supported)}")


def _zeeman_tensors(
    isotopes: tuple[str, ...], basefrqs: np.ndarray, zeeman: dict[str, Any] | None
) -> tuple[np.ndarray, ...]:
    """Lab-frame Zeeman tensors in rad/s, as in the Zeeman section of create.m."""
    nspins = len(isotopes)
    if zeeman is None:
        return tuple(np.eye(3) * basefrqs[n] for n in range(nspins))
    _reject_unported(zeeman, ZEEMAN_FIELDS, "inter['zeeman']")
    tensors = [np.zeros((3, 3)) for _ in range(nspins)]
    if "eigs" in zeeman:
        if "euler" not in zeeman:
            raise ValueError("inter['zeeman']['euler'] must be set together with inter['zeeman']['eigs'].")
        _check_length(zeeman["eigs"], nspins, "inter['zeeman']['eigs']")
        _check_length(zeeman["euler"], nspins, "inter['zeeman']['euler']")
        for n, (eigs, euler) in enumerate(zip(zeeman["eigs"], zeeman["euler"])):
            if (eigs is None) != (euler is None):
                raise ValueError("inter['zeeman']['eigs'] and ['euler'] must have identical non-empty patterns.")
            if eigs is not None:
                tensors[n] = tensors[n] + _rotated_diagonal(eigs, euler, "inter['zeeman']")
    if "matrix" in zeeman:
        _check_length(zeeman["matrix"], nspins, "inter['zeeman']['matrix']")
        for n, user in enumerate(zeeman["matrix"]):
            if user is not None and np.linalg.norm(np.asarray(user, dtype=float), 2) > 0:
                tensors[n] = tensors[n] + np.asarray(user, dtype=float)
    if "scalar" in zeeman:
        _check_length(zeeman["scalar"], nspins, "inter['zeeman']['scalar']")
        for n, user in enumerate(zeeman["scalar"]):
            if user is not None and abs(user) > 0:
                tensors[n] = tensors[n] + np.eye(3) * float(user)
    out = []
    for n, name in enumerate(isotopes):
        if name[0] == "E":
            # Electrons: g-tensor in Bohr magneton units
            out.append(tensors[n] * basefrqs[n] / FREE_ELECTRON_G)
        else:
            # Nuclei: chemical shielding in ppm
            out.append((np.eye(3) + 1e-6 * tensors[n]) * basefrqs[n])
    return tuple(out)


def _coupling_tensors(
    mults: np.ndarray, tols: Tolerances, coupling: dict[str, Any] | None, dipolar: list[list[np.ndarray]]
) -> tuple[tuple[np.ndarray | None, ...], ...]:
    """Coupling tensors in rad/s, as in the coupling section of create.m, starting from the dipolar ones."""
    nspins = len(mults)
    tensors = [[dipolar[n][k].copy() for k in range(nspins)] for n in range(nspins)]
    if coupling is not None:
        _reject_unported(coupling, COUPLING_FIELDS, "inter['coupling']")
        if "eigs" in coupling:
            if "euler" not in coupling:
                raise ValueError("inter['coupling']['euler'] must be set together with inter['coupling']['eigs'].")
            eigs = _check_square(coupling["eigs"], nspins, "inter['coupling']['eigs']")
            euler = _check_square(coupling["euler"], nspins, "inter['coupling']['euler']")
            for n in range(nspins):
                for k in range(nspins):
                    value = eigs[n][k]
                    if value is not None and np.linalg.norm(np.asarray(value, dtype=float)) > tols.inter_cutoff:
                        tensors[n][k] = tensors[n][k] + 2 * np.pi * _rotated_diagonal(
                            value, euler[n][k], "inter['coupling']"
                        )
        if "matrix" in coupling:
            user = _check_square(coupling["matrix"], nspins, "inter['coupling']['matrix']")
            for n in range(nspins):
                for k in range(nspins):
                    value = user[n][k]
                    if value is not None and np.linalg.norm(np.asarray(value, dtype=float), 2) > tols.inter_cutoff:
                        tensors[n][k] = tensors[n][k] + 2 * np.pi * np.asarray(value, dtype=float)
        if "scalar" in coupling:
            user = _check_square(coupling["scalar"], nspins, "inter['coupling']['scalar']")
            for n in range(nspins):
                for k in range(nspins):
                    value = user[n][k]
                    if value is not None and abs(value) > tols.inter_cutoff:
                        tensors[n][k] = tensors[n][k] + 2 * np.pi * np.eye(3) * float(value)
    # Drop insignificant tensors and couplings to spin-zero particles
    return tuple(
        tuple(
            None
            if (np.linalg.norm(tensors[n][k], 2) < 2 * np.pi * tols.inter_cutoff or mults[n] == 1 or mults[k] == 1)
            else tensors[n][k]
            for k in range(nspins)
        )
        for n in range(nspins)
    )


def _rotated_diagonal(eigs: Any, euler: Any, name: str) -> np.ndarray:
    """``S diag(eigs) S'`` with ``S`` the DCM of the Euler angles (identity if ``None``), symmetrised."""
    eigs = np.asarray(eigs, dtype=float).reshape(-1)
    if eigs.size != 3:
        raise ValueError(f"non-empty elements of {name}['eigs'] must be real 3-vectors.")
    if euler is None:
        S = np.eye(3)
    else:
        euler = np.asarray(euler, dtype=float).reshape(-1)
        if euler.size != 3:
            raise ValueError(f"non-empty elements of {name}['euler'] must be real 3-vectors.")
        S = euler2dcm(euler)
    component = S @ np.diag(eigs) @ S.T
    return (component + component.T) / 2


def _coordinates(nspins: int, coordinates: Sequence[Any] | None) -> tuple[np.ndarray | None, ...]:
    if coordinates is None:
        return tuple([None] * nspins)
    _check_length(coordinates, nspins, "inter['coordinates']")
    out = []
    for xyz in coordinates:
        if xyz is None or np.size(xyz) == 0:
            out.append(None)
            continue
        xyz = np.asarray(xyz, dtype=float).reshape(-1)
        if xyz.size != 3:
            raise ValueError("non-empty elements of inter['coordinates'] must be real 3-vectors.")
        out.append(xyz)
    return tuple(out)


def _dipolar(
    coordinates: tuple[np.ndarray | None, ...], gammas: np.ndarray, chem: Chemistry, tols: Tolerances
) -> tuple[list[list[np.ndarray]], np.ndarray]:
    """Point dipolar coupling tensors (rad/s, traceless) and the proximity matrix (port of ``dipolar.m``).

    Pairs within one chemical subsystem, both with coordinates, closer than
    ``tols.prox_cutoff`` Angstrom are proximate and get the dipolar tensor
    ``A (1 - 3 n n^T)`` with ``A = mu0 hbar g1 g2 / (8 pi r^3)``. Without
    coordinates the proximity matrix is the identity, as in Spinach.

    Spinach authors (dipolar.m): Ilya Kuprov.
    """
    nspins = len(coordinates)
    tensors = [[np.zeros((3, 3)) for _ in range(nspins)] for _ in range(nspins)]
    if all(xyz is None for xyz in coordinates):
        return tensors, np.eye(nspins, dtype=bool)
    proxmatrix = np.zeros((nspins, nspins), dtype=bool)
    for part in chem.parts:
        for n in part:
            for k in part:
                if n == k or coordinates[n] is None or coordinates[k] is None:
                    continue
                dv = coordinates[k] - coordinates[n]
                distance = float(np.linalg.norm(dv))
                if distance >= tols.prox_cutoff:
                    continue
                if distance < 0.5:
                    raise ValueError(f"collision detected between spin {n} and spin {k}.")
                proxmatrix[n, k] = True
                ort = dv / distance
                A = 0.5 * gammas[n] * gammas[k] * tols.hbar * tols.mu0 / (4 * np.pi * (distance * 1e-10) ** 3)
                D = A * (np.eye(3) - 3 * np.outer(ort, ort))
                tensors[n][k] = tensors[n][k] + (D - np.eye(3) * np.trace(D) / 3)
    return tensors, proxmatrix


def _check_length(values: Sequence[Any], nspins: int, name: str) -> None:
    if len(values) != nspins:
        raise ValueError(f"{name} must have one entry per spin ({nspins}).")


def _check_square(values: Sequence[Sequence[Any]], nspins: int, name: str) -> Sequence[Sequence[Any]]:
    if len(values) != nspins or any(len(row) != nspins for row in values):
        raise ValueError(f"{name} must be a {nspins} x {nspins} nested list.")
    return values


def kill_spin(spin_system: SpinSystem, hit_list: Sequence[int]) -> SpinSystem:
    """Remove the spins in ``hit_list`` (0-based) from a spin system (port of ``kill_spin.m``).

    Interaction tensors, coordinates, the proximity matrix, chemical parts
    and per-spin relaxation rates are reindexed; the basis is dropped and
    must be built again with :func:`spynach.basis`.

    Spinach authors (kill_spin.m): Ilya Kuprov, Luke Edwards.
    """
    hit = sorted({int(n) for n in hit_list})
    if any(n < 0 or n >= spin_system.nspins for n in hit):
        raise ValueError("hit_list must contain spin numbers within the system bounds.")
    keep = [n for n in range(spin_system.nspins) if n not in hit]
    new_index = {old: new for new, old in enumerate(keep)}
    chem = spin_system.chem
    parts = tuple(tuple(new_index[n] for n in part if n in new_index) for part in chem.parts)
    rlx = spin_system.rlx
    return replace(
        spin_system,
        isotopes=tuple(spin_system.isotopes[n] for n in keep),
        labels=tuple(spin_system.labels[n] for n in keep),
        types=tuple(spin_system.types[n] for n in keep),
        mults=spin_system.mults[keep],
        gammas=spin_system.gammas[keep],
        basefrqs=spin_system.basefrqs[keep],
        zeeman_matrix=tuple(spin_system.zeeman_matrix[n] for n in keep),
        coupling_matrix=tuple(tuple(spin_system.coupling_matrix[n][k] for k in keep) for n in keep),
        coordinates=tuple(spin_system.coordinates[n] for n in keep),
        proxmatrix=spin_system.proxmatrix[np.ix_(keep, keep)],
        chem=replace(chem, parts=parts),
        rlx=replace(
            rlx,
            r1_rates=tuple(rlx.r1_rates[n] for n in keep) if rlx.r1_rates else (),
            r2_rates=tuple(rlx.r2_rates[n] for n in keep) if rlx.r2_rates else (),
            srfk_mdepth=tuple(tuple(rlx.srfk_mdepth[n][k] for k in keep) for n in keep) if rlx.srfk_mdepth else (),
            srsk_sources=tuple(keep.index(k) for k in rlx.srsk_sources if k in keep),
        ),
        basis=None,
        assumptions=None,
    )


def dilute(spin_system: SpinSystem, isotope: str, tuples: int = 1) -> list[SpinSystem]:
    """Isotopomer subsystems of a dilute isotope (port of ``dilute.m``).

    Returns one spin system per combination of ``tuples`` spins of
    ``isotope``, with every other spin of that isotope removed; the
    natural-abundance spectrum is the sum over the subsystems.

    Spinach authors (dilute.m): Ilya Kuprov, Luke Edwards, Bud Macaulay.
    """
    from itertools import combinations

    dilute_spins = [n for n, iso in enumerate(spin_system.isotopes) if iso == isotope]
    if len(dilute_spins) < tuples:
        raise ValueError(f"not enough {isotope} in the system to make the selection.")
    return [
        kill_spin(spin_system, sorted(set(dilute_spins) - set(combo))) for combo in combinations(dilute_spins, tuples)
    ]
