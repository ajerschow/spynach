"""The distant dipolar field: the mean-field generator of a spin system in its own dipolar field, without a grid.

The magnetisation of a liquid sample creates a dipolar field at every point of the sample (the distant dipolar
field, DDF, or demagnetising field). When the magnetisation varies along one direction ``s`` only, as it does
after gradients on one axis, that field is local (Deville, Bernier and Delrieux, Phys. Rev. B 19, 5666 (1979)):
in the rotating frame and secular form, ``B_d(s) = μ0 Δ [ M(s)/3 − M_z(s) ẑ ]`` with ``Δ = (3 cos²θ − 1)/2``
for the angle ``θ`` between ``s`` and ``B0``, so the field vanishes at the magic angle; a helix along ``B0``
acts like a stack of thin discs (``−(2/3) μ0 M_z`` along ``z``), one across it like a stack of slabs.
The uniform part of the magnetisation sees the Lorentz field minus the demagnetising field of the sample
shape, which for an axially symmetric sample has the same form with ``Δ0 = (3 N_z − 1)/2`` for the
demagnetising factor ``N_z`` along ``B0`` (a sphere, ``1/3``, has none; a long cylinder along ``B0`` has
``−1/2``, a thin disc across it ``+1``, the limits of the modulated modes).

The state is expanded in Fourier modes along ``s``, ``ρ(s) = Σ_n ρ_n exp(i n k0 s)``, the lattice of
``spynach.program``'s explicit mode, and the field is the mean-field Hamiltonian per mode

    H_d(n) = −γ F·B_d(n) = sign(γ) (Δ_n / τ_d) [ (2/3) m_z(n) F_z − (1/3) (m_x(n) F_x + m_y(n) F_y) ]

with the dipolar demagnetising time ``τ_d = 1/(|γ| μ0 M0)`` (water at 9.4 T and 300 K: about 0.1 s), the
total spin ``F`` of the isotope, and ``m(n) = <F>_n / <F_z>_eq`` the Fourier coefficients of the magnetisation
in units of its equilibrium value, read from the state. A product in ``s`` is a convolution in ``n``:
``dρ_n/dt = −i L0 ρ_n − i Σ_m L_d(m) ρ_{n−m}``. This is what :class:`DipolarField` returns as a generator for
:func:`spynach.iserstep`, on the flattened mode stack, and what :class:`spynach.program.Program` integrates
when it is given a :class:`DipolarFieldSpec` (explicit mode): the modes couple only here, so a lattice of
``nmodes`` harmonics beyond the ones the gradients reach is the only truncation. The uniform mode alone
(``nmodes=0``) is the demagnetising shift without gradients. Literature: Warren et al., Science 262, 2005
(1993) (CRAZED); Lee, Richter, Warren et al., J. Chem. Phys. 105, 874 (1996); Jeener, J. Chem. Phys. 112,
5091 (2000); Levitt, Concepts Magn. Reson. 8, 77 (1996); Bowtell, Bowley and Glover, J. Magn. Reson. 88, 643
(1990) (multiple spin echoes). No Spinach counterpart; see ``docs/distant-dipolar-field.md``.

Limits: one modulation axis, one isotope producing and feeling the field, the secular form (the field from a
second isotope would need its own frame), and the uniform mode by the demagnetising factor only.

Author: Alexej Jerschow.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sps

from spynach.operators import operator
from spynach.probe import curie_magnetisation
from spynach.states import state
from spynach.system import SpinSystem
from spynach.utilities import Matrix, hdot


@dataclass
class DipolarField:
    """The state-dependent generator of a spin system in its distant dipolar field; call it as ``L(t, rho)``.

    Made by :func:`dipolar_field`. ``L0`` is the generator without the field, ``Cx``, ``Cy``, ``Cz`` the
    commutation superoperators of the total spin of the isotope (the field acts on every spin of it alike),
    ``coils`` the ``Lx``, ``Ly``, ``Lz`` states that read the magnetisation (weighted by the relative number
    densities of the spins), ``mz_eq`` its equilibrium value, ``half`` the half-width of the mode lattice
    (modes ``−half .. half``), ``k0`` the base wavevector (rad/m) and ``diffusion`` the diffusion coefficient
    (m²/s) that damps mode ``n`` at ``D (n k0)²``. ``rho`` is the flattened mode stack, mode-major: build it
    with :meth:`stack`, read it back with :meth:`columns`, detect it with :meth:`observe`.

    Author: Alexej Jerschow.
    """

    spin_system: SpinSystem
    isotope: str
    tau_d: float
    angle: float
    demagnetising: float
    L0: Matrix
    Cx: Matrix
    Cy: Matrix
    Cz: Matrix
    coils: tuple[np.ndarray, np.ndarray, np.ndarray]
    mz_eq: float
    half: int
    k0: float
    diffusion: float
    nstates: int = field(init=False)
    sign: float = field(init=False)

    def __post_init__(self) -> None:
        self.nstates = self.coils[2].size
        gamma = float(self.spin_system.gammas[self.spin_system.isotopes.index(self.isotope)])
        self.sign = 1.0 if gamma >= 0 else -1.0

    @property
    def size(self) -> int:
        """The number of modes, ``2 half + 1``."""
        return 2 * self.half + 1

    @property
    def modes(self) -> np.ndarray:
        return np.arange(-self.half, self.half + 1)

    @property
    def delta(self) -> np.ndarray:
        """``Δ_n`` of every mode: ``(3 cos²θ − 1)/2`` for ``n ≠ 0``, ``(3 N_z − 1)/2`` for the uniform mode."""
        delta = np.full(self.size, (3 * np.cos(self.angle) ** 2 - 1) / 2)
        delta[self.half] = (3 * self.demagnetising - 1) / 2
        return delta

    def stack(self, rho: Matrix) -> np.ndarray:
        """The flattened mode stack: a single state goes into mode 0, an ``(nstates, size)`` array as it is."""
        rho = np.asarray(rho.toarray() if sps.issparse(rho) else rho, dtype=complex)
        if rho.ndim == 2 and rho.shape == (self.nstates, self.size):
            return rho.T.reshape(-1)
        rho = rho.reshape(-1)
        if rho.size != self.nstates:
            raise ValueError(f"rho has {rho.size} elements, the spin system {self.nstates} states.")
        out = np.zeros((self.size, self.nstates), dtype=complex)
        out[self.half] = rho
        return out.reshape(-1)

    def columns(self, rho: np.ndarray) -> np.ndarray:
        """The modes of a flattened stack as the columns of an ``(nstates, size)`` array."""
        rho = np.asarray(rho).reshape(-1)
        if rho.size != self.nstates * self.size:
            raise ValueError(f"the mode stack has {rho.size} elements, expected {self.nstates} x {self.size}.")
        return rho.reshape(self.size, self.nstates).T

    def observe(self, coil: Matrix, rho: np.ndarray, envelope: np.ndarray | None = None) -> complex:
        """``Σ_n envelope[n] <coil|ρ_n>``; the default envelope detects the uniform mode alone (a long sample)."""
        cols = self.columns(rho)
        coil = np.asarray(coil.toarray() if sps.issparse(coil) else coil, dtype=complex).reshape(-1)
        signals = np.conj(coil) @ cols
        if envelope is None:
            return complex(signals[self.half])
        return complex(np.asarray(envelope, dtype=complex).reshape(-1) @ signals)

    def magnetisation(self, rho: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The Fourier coefficients ``(m_x, m_y, m_z)`` of the magnetisation of the isotope, in units of ``<F_z>_eq``,
        one ``(size,)`` array each (complex; ``m(−n)`` is the conjugate of ``m(n)`` for a Hermitian state)."""
        cols = self.columns(rho)
        return tuple(np.conj(c) @ cols / self.mz_eq for c in self.coils)

    def __call__(self, t: float, rho: np.ndarray) -> Matrix:
        mx, my, mz = self.magnetisation(rho)
        coefficient = self.sign * self.delta / self.tau_d
        az, ax, ay = 2 / 3 * coefficient * mz, -coefficient * mx / 3, -coefficient * my / 3
        tol = self.spin_system.tols.zte_tol
        if self.size == 1:
            return self.L0 + az[0] * self.Cz + ax[0] * self.Cx + ay[0] * self.Cy
        L = sps.kron(sps.eye_array(self.size), sps.csr_array(self.L0), format="csr")
        if self.diffusion:
            rates = -1j * self.diffusion * (self.modes * self.k0) ** 2
            L = L + sps.kron(sps.diags_array(rates), sps.eye_array(self.nstates), format="csr")
        # GPU: the convolution over modes, one shifted block per mode that carries magnetisation
        for j in np.flatnonzero(np.maximum(np.abs(az), np.maximum(np.abs(ax), np.abs(ay))) > tol):
            shift = sps.eye_array(self.size, k=-int(self.modes[j]))
            L = L + sps.kron(shift, az[j] * self.Cz + ax[j] * self.Cx + ay[j] * self.Cy, format="csr")
        return L


@dataclass(frozen=True)
class DipolarFieldSpec:
    """The distant dipolar field of a pulse program (``Program(field=..., mode='explicit')``).

    ``isotope`` produces and feels the field, ``tau_d`` is its dipolar demagnetising time (s) at the
    magnetisation of the program's initial state ``rho0``, ``demagnetising`` the factor ``N_z`` of the sample
    shape along ``B0`` for the uniform mode (default a sphere, no uniform field), ``nmodes`` the harmonics kept
    beyond the modes the gradients reach, ``method`` and ``timestep`` (s, default the acquisition dwell) those of
    :func:`spynach.iserstep` for every delay, gradient and dwell, and ``weights`` the relative number densities
    of the spins of the isotope (default equal). The angle of the field follows the gradient axis of the program.

    Author: Alexej Jerschow.
    """

    isotope: str
    tau_d: float
    demagnetising: float = 1 / 3
    nmodes: int = 8
    method: str = "LG4"
    timestep: float | None = None
    weights: Sequence[float] | None = None


def dipolar_field(
    spin_system: SpinSystem,
    L0: Matrix,
    isotope: str,
    tau_d: float,
    *,
    angle: float = 0.0,
    demagnetising: float = 1 / 3,
    nmodes: int = 0,
    k0: float = 0.0,
    rho_eq: Matrix | None = None,
    diffusion: float = 0.0,
    weights: Sequence[float] | None = None,
) -> DipolarField:
    """The generator ``L(t, rho)`` of ``spin_system`` in the distant dipolar field of ``isotope``, for :func:`iserstep`.

    ``L0`` is the Liouvillian without the field (``H + 1j * R``, sphten-liouv), ``tau_d`` the dipolar
    demagnetising time (s) at the magnetisation of the reference state ``rho_eq`` (default the ``Lz`` state of
    the isotope; :func:`ddf_time` computes it from the concentration), ``angle`` that of the modulation axis to
    ``B0`` (rad), ``demagnetising`` the factor ``N_z`` of the sample shape for the uniform mode (default a sphere:
    no uniform field), ``nmodes`` the half-width of the mode lattice (``0``: the uniform mode alone), ``k0`` the
    base wavevector (rad/m) and ``diffusion`` the coefficient (m²/s) that damps mode ``n`` at ``D (n k0)²``;
    ``weights`` are the relative number densities of the spins of the isotope (default equal), the field being
    read from the weighted magnetisation and acting on every spin of the isotope.
    """
    if spin_system.basis is None or spin_system.basis.formalism != "sphten-liouv":
        raise NotImplementedError("the distant dipolar field runs in the sphten-liouv formalism.")
    if isotope not in spin_system.isotopes:
        raise ValueError(f"{isotope!r} is not in the spin system.")
    if not np.isfinite(tau_d) or tau_d <= 0:
        raise ValueError("tau_d must be a positive time (s).")
    if int(nmodes) != nmodes or nmodes < 0:
        raise ValueError("nmodes must be a non-negative integer.")
    if diffusion < 0:
        raise ValueError("diffusion must be non-negative.")
    spins = [n for n, iso in enumerate(spin_system.isotopes) if iso == isotope]
    if weights is None:
        coils = tuple(
            np.asarray(state(spin_system, label, isotope), dtype=complex).reshape(-1) for label in "Lx Ly Lz".split()
        )
    else:
        w = np.asarray(weights, dtype=float).reshape(-1)
        if w.size != len(spins) or np.any(w < 0) or w.sum() == 0:
            raise ValueError(f"weights must be non-negative, one per spin of {isotope} ({len(spins)}), not all zero.")
        coils = tuple(
            sum(
                float(wn) * np.asarray(state(spin_system, label, [n]), dtype=complex).reshape(-1)
                for wn, n in zip(w, spins)
            )
            for label in "Lx Ly Lz".split()
        )
    rho_eq = coils[2] if rho_eq is None else rho_eq
    mz_eq = float(np.real(hdot(coils[2], np.asarray(rho_eq.toarray() if sps.issparse(rho_eq) else rho_eq).reshape(-1))))
    if mz_eq == 0:
        raise ValueError("the reference state has no longitudinal magnetisation on the isotope.")
    Cx, Cy, Cz = (operator(spin_system, label, isotope) for label in ("Lx", "Ly", "Lz"))
    return DipolarField(
        spin_system,
        isotope,
        float(tau_d),
        float(angle),
        float(demagnetising),
        L0,
        Cx,
        Cy,
        Cz,
        coils,
        mz_eq,
        int(nmodes),
        float(k0),
        float(diffusion),
    )


def ddf_time(spin_system: SpinSystem, isotope: str, *, concentration: float, temperature: float | None = None) -> float:
    """The dipolar demagnetising time ``τ_d = 1 / (|γ| μ0 M0)`` (s) of ``isotope`` at ``concentration`` (mol/L of that
    nucleus) in the field of the spin system at ``temperature`` (default ``inter['temperature']``), ``M0`` the Curie
    magnetisation; water (110 mol/L of 1H) at 9.4 T and 300 K gives about 0.1 s."""
    gamma = float(spin_system.gammas[spin_system.isotopes.index(isotope)]) if isotope in spin_system.isotopes else 0.0
    M0 = curie_magnetisation(spin_system, isotope, concentration=concentration, temperature=temperature)
    return float(1 / (abs(gamma) * spin_system.tols.mu0 * M0))
