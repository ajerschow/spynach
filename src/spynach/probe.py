"""Radiation damping: the feedback of the probe on the spins, as a state-dependent generator for :func:`iserstep`.

The transverse magnetisation of the isotope the probe is tuned to induces a current in the coil, and the field of
that current rotates the magnetisation back towards equilibrium (Bloembergen and Pound, Phys. Rev. 95, 8 (1954)).
In the rotating frame the feedback is a Hamiltonian term ``w_x Ix + w_y Iy`` on that isotope whose coefficients
follow the state:

    w_x + 1j w_y = -1j * rate * cos(phase) * exp(1j * phase) * (<Ix> + 1j <Iy>) / <Iz>_eq

``rate`` is the radiation damping rate 1/T_rd (s^-1) at the equilibrium magnetisation ``<Iz>_eq`` of the reference
state, and ``phase`` the tuning phase of the probe (0 on resonance; the detuned tank circuit reduces the feedback
by cos(phase) and rotates it by phase). Since ``d<I>/dt = w x <I>`` for ``H = w . I``, the term drives ``<Iz>``
towards ``+<Iz>_eq`` and ``<Ix>^2 + <Iy>^2`` grows at ``rate`` times itself when the magnetisation is inverted:
with no other terms, ``m_z(t) = tanh(rate * (t - t0))`` for a magnetisation of equilibrium length. The feedback
is unitary, so the trace, Hermiticity and purity of the density matrix are preserved; relaxation and the
coherent Hamiltonian come from ``L0``, the usual Liouvillian ``H + 1j * R``. The probe is taken to respond
instantaneously (its ring-down time 2Q/omega_0 much shorter than T_rd, T2 and the inverse offsets), which holds
for every NMR probe up to Q of a few thousand; the strong-coupling regime of very high Q is not modelled.

An inhomogeneous line is a set of isochromats, copies of the spin system at different offsets that share the
coil: the ensemble state is the flattened stack of their Liouville vectors, the generator is block diagonal, and
the feedback uses the weighted average magnetisation. This module has no Spinach counterpart.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sps

from spynach.isotopes import spin
from spynach.operators import operator
from spynach.states import state
from spynach.system import SpinSystem
from spynach.utilities import Matrix, hdot

__all__ = ["RadiationDamping", "line_weights", "radiation_damping", "rd_phase", "rd_rate"]


@dataclass
class RadiationDamping:
    """The state-dependent generator of a spin system under radiation damping; call it as ``L(t, rho)``.

    Made by :func:`radiation_damping`. ``L0`` is the generator without the feedback (for an isochromat
    ensemble, block diagonal over the isochromats with their offsets added), ``Cx`` and ``Cy`` the feedback
    operators of the tuned isotope (commutation superoperators in Liouville space, operators in Hilbert space),
    ``coils`` the Lx, Ly and Lz detection states, ``mz_eq`` the equilibrium magnetisation of the reference
    state, ``offsets`` (Hz) and ``weights`` the isochromats.
    """

    spin_system: SpinSystem
    isotope: str
    rate: float
    phase: float
    L0: Matrix
    Cx: Matrix
    Cy: Matrix
    coils: tuple[Matrix, Matrix, Matrix]
    mz_eq: float
    offsets: np.ndarray
    weights: np.ndarray
    nstates: int = field(init=False)

    def __post_init__(self) -> None:
        self.nstates = self.coils[2].shape[0]

    @property
    def nisochromats(self) -> int:
        return self.offsets.size

    @property
    def gain(self) -> complex:
        """The complex feedback gain ``rate * cos(phase) * exp(1j * phase)`` (s^-1)."""
        return self.rate * np.cos(self.phase) * np.exp(1j * self.phase)

    def stack(self, rho: Matrix) -> np.ndarray:
        """The ensemble state: ``rho`` replicated over the isochromats and flattened (Liouville space)."""
        rho = np.asarray(rho.toarray() if sps.issparse(rho) else rho, dtype=complex).reshape(-1)
        if rho.size != self.nstates:
            raise ValueError(f"rho has {rho.size} elements, the spin system {self.nstates} states.")
        return np.tile(rho, self.nisochromats)

    def columns(self, rho: np.ndarray) -> np.ndarray:
        """The isochromat states of an ensemble state as the columns of an ``(nstates, nisochromats)`` array."""
        rho = np.asarray(rho).reshape(-1)
        if rho.size != self.nstates * self.nisochromats:
            raise ValueError(
                f"the ensemble state has {rho.size} elements, expected {self.nstates} x {self.nisochromats}."
            )
        return rho.reshape(self.nisochromats, self.nstates).T

    def observe(self, coil: Matrix, rho: Matrix) -> complex:
        """``<coil|rho>`` averaged over the isochromats with their weights (``hdot`` for a single state)."""
        if self.nisochromats == 1 and not (isinstance(rho, np.ndarray) and rho.ndim == 1 and rho.size != self.nstates):
            return hdot(coil, rho)
        cols = self.columns(rho)
        coil = np.asarray(coil.toarray() if sps.issparse(coil) else coil, dtype=complex).reshape(-1)
        return complex(self.weights @ (np.conj(coil) @ cols))

    def magnetisation(self, rho: Matrix) -> tuple[float, float, float]:
        """``(<Ix>, <Iy>, <Iz>) / <Iz>_eq`` of the tuned isotope, averaged over the isochromats."""
        lx, ly, lz = self.coils
        return tuple(float(np.real(self.observe(c, rho))) / self.mz_eq for c in (lx, ly, lz))

    def __call__(self, t: float, rho: Matrix) -> Matrix:
        mx, my, _ = self.magnetisation(rho)
        w = -1j * self.gain * (mx + 1j * my)
        return self.L0 + w.real * self.Cx + w.imag * self.Cy


def line_weights(offsets: Sequence[float], fwhm: float, shape: str = "gaussian") -> np.ndarray:
    """Normalised weights of isochromats at ``offsets`` (Hz) under a line of full width ``fwhm`` at half maximum."""
    offsets = np.asarray(offsets, dtype=float)
    if fwhm <= 0:
        raise ValueError("fwhm must be positive.")
    if shape == "gaussian":
        sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
        weights = np.exp(-0.5 * (offsets / sigma) ** 2)
    elif shape == "lorentzian":
        weights = 1 / (1 + (2 * offsets / fwhm) ** 2)
    else:
        raise ValueError("shape must be 'gaussian' or 'lorentzian'.")
    return weights / weights.sum()


def radiation_damping(
    spin_system: SpinSystem,
    L0: Matrix,
    isotope: str,
    rate: float,
    *,
    phase: float = 0.0,
    rho_eq: Matrix | None = None,
    offsets: Sequence[float] | None = None,
    weights: Sequence[float] | None = None,
) -> RadiationDamping:
    """The generator ``L(t, rho)`` of ``spin_system`` under radiation damping on ``isotope``, for :func:`iserstep`.

    ``L0`` is the Liouvillian without the probe (``H + 1j * R`` in Liouville space, the Hamiltonian in Hilbert
    space), ``rate`` the radiation damping rate 1/T_rd in s^-1 at the magnetisation of the reference state
    ``rho_eq`` (default: the ``Lz`` state of the isotope, the high-temperature equilibrium with unit polarisation;
    pass :func:`spynach.equilibrium` or any other state to define the rate against it, and start from a state
    scaled the same way: a hyperpolarised sample is a larger ``<Iz>`` and damps proportionally faster), and
    ``phase`` the tuning phase in radians (:func:`rd_phase`). A negative ``rate`` or a phase beyond 90 degrees is
    active feedback. ``offsets`` (Hz) make an isochromat ensemble with ``weights`` (default equal; see
    :func:`line_weights`), Liouville space only: build the ensemble state with ``.stack(rho)``, read it back with
    ``.columns(rho)`` and detect with ``.observe(coil, rho)``. The returned object is the callable to pass as
    ``L`` to :func:`iserstep`, and ``.magnetisation(rho)`` gives ``(<Ix>, <Iy>, <Iz>) / <Iz>_eq``.
    """
    if spin_system.basis is None:
        raise ValueError("basis set information is missing, run basis() before calling this function.")
    if isotope not in spin_system.isotopes:
        raise ValueError(f"{isotope!r} is not in the spin system.")
    if not np.isfinite(rate) or not np.isfinite(phase):
        raise ValueError("rate and phase must be finite numbers.")
    formalism = spin_system.basis.formalism
    Cx = operator(spin_system, "Lx", isotope)
    Cy = operator(spin_system, "Ly", isotope)
    coils = tuple(state(spin_system, label, isotope) for label in ("Lx", "Ly", "Lz"))
    if rho_eq is None:
        rho_eq = coils[2]
    mz_eq = float(np.real(hdot(coils[2], rho_eq)))
    if mz_eq == 0:
        raise ValueError("the reference state has no longitudinal magnetisation on the tuned isotope.")
    if offsets is None:
        offsets_arr = np.zeros(1)
        weights_arr = np.ones(1)
    else:
        if formalism != "sphten-liouv":
            raise NotImplementedError("isochromat ensembles need the sphten-liouv formalism.")
        offsets_arr = np.asarray(offsets, dtype=float).reshape(-1)
        if offsets_arr.size == 0:
            raise ValueError("offsets must not be empty.")
        weights_arr = np.ones(offsets_arr.size) if weights is None else np.asarray(weights, dtype=float).reshape(-1)
        if weights_arr.size != offsets_arr.size or np.any(weights_arr < 0) or weights_arr.sum() == 0:
            raise ValueError("weights must be non-negative, one per offset, and not all zero.")
        weights_arr = weights_arr / weights_arr.sum()
        if offsets_arr.size > 1 or offsets_arr[0] != 0:
            K = offsets_arr.size
            Lz_comm = operator(spin_system, "Lz", isotope)
            L0 = sps.kron(sps.eye_array(K), sps.csr_array(L0), format="csr") + sps.kron(
                sps.diags_array(2 * np.pi * offsets_arr), sps.csr_array(Lz_comm), format="csr"
            )
            Cx = sps.kron(sps.eye_array(K), Cx, format="csr")
            Cy = sps.kron(sps.eye_array(K), Cy, format="csr")
    return RadiationDamping(
        spin_system, isotope, float(rate), float(phase), L0, Cx, Cy, coils, mz_eq, offsets_arr, weights_arr
    )


AVOGADRO = 6.02214076e23


def rd_rate(
    spin_system: SpinSystem,
    isotope: str,
    *,
    eta: float,
    Q: float,
    concentration: float,
    temperature: float | None = None,
) -> float:
    """The radiation damping rate 1/T_rd (s^-1) of ``isotope`` from the probe and the sample, SI throughout.

    ``1/T_rd = mu0 * eta * Q * |gamma| * M0 / 2`` (Bloembergen and Pound) with the filling factor ``eta``, the
    quality factor ``Q`` and the Curie magnetisation ``M0 = n gamma^2 hbar^2 I(I+1) B0 / (3 k T)`` of the isotope
    at ``concentration`` (mol/L of that nucleus) in the field of the spin system at ``temperature`` (default
    ``inter['temperature']``). Water at 11.7 T with Q = 200 and eta = 0.5 gives about 600 s^-1, T_rd of 1.6 ms.
    """
    if eta <= 0 or Q <= 0:
        raise ValueError("eta, Q and concentration must be positive.")
    gamma, _ = spin(isotope)
    M0 = curie_magnetisation(spin_system, isotope, concentration=concentration, temperature=temperature)
    return float(spin_system.tols.mu0 * eta * Q * abs(gamma) * M0 / 2)


def curie_magnetisation(
    spin_system: SpinSystem, isotope: str, *, concentration: float, temperature: float | None = None
) -> float:
    """The Curie magnetisation ``M0 = n gamma^2 hbar^2 I(I+1) B0 / (3 k T)`` (A/m) of ``isotope`` at ``concentration``
    (mol/L of that nucleus) in the field of the spin system at ``temperature`` (default ``inter['temperature']``)."""
    if isotope not in spin_system.isotopes:
        raise ValueError(f"{isotope!r} is not in the spin system.")
    if concentration <= 0:
        raise ValueError("eta, Q and concentration must be positive.")
    tols = spin_system.tols
    temperature = spin_system.rlx.temperature if temperature is None else float(temperature)
    if temperature <= 0:
        raise ValueError("temperature must be positive.")
    gamma, mult = spin(isotope)
    I = (mult - 1) / 2
    n = concentration * 1e3 * AVOGADRO
    return float(n * gamma**2 * tols.hbar**2 * I * (I + 1) * spin_system.magnet / (3 * tols.kbol * temperature))


def rd_phase(spin_system: SpinSystem, isotope: str, *, Q: float, detuning: float) -> float:
    """The tuning phase (radians) of a probe of quality factor ``Q`` detuned by ``detuning`` (Hz) from the
    Larmor frequency of ``isotope``: ``atan(2 Q detuning / nu0)``."""
    if isotope not in spin_system.isotopes:
        raise ValueError(f"{isotope!r} is not in the spin system.")
    if Q <= 0:
        raise ValueError("Q must be positive.")
    gamma, _ = spin(isotope)
    nu0 = abs(gamma) * spin_system.magnet / (2 * np.pi)
    if nu0 == 0:
        raise ValueError("the Larmor frequency is zero: set the magnet field.")
    return float(np.arctan(2 * Q * detuning / nu0))
