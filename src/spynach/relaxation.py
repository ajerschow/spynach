"""Relaxation superoperators and thermal equilibrium (ports of ``relaxation.m``, ``equilibrium.m``, ``thermalize.m``).

Ported theories (``inter['relaxation']``):

* ``'redfield'``: Bloch-Redfield-Wangsness theory for isotropic,
  axial or rhombic rotational diffusion, in the diagonalisation-free form
  of Kuprov (2011): every pair of irreducible components of the
  anisotropic Hamiltonian contributes ``-w A int_0^T exp(iL0 t) C exp(-iD t) dt``,
  evaluated as a block of a single matrix exponential (auxiliary matrix
  formalism of Goodwin and Kuprov, 2015, :func:`expmint`);
* ``'t1_t2'``: the extended T1/T2 model, a diagonal superoperator with
  per-spin longitudinal and transverse rates summed over the spins each
  basis state correlates;
* ``'damp'``: non-selective damping of every state except the unit state;
* ``'SRFK'``: scalar relaxation of the first kind, a Redfield integral over
  the scalar couplings modulated by chemical exchange (``rlx_scalar.m``):
  ``-w H1 int_0^T exp(iH0 t) H1 exp(-iH0 t) exp(-t/tau) dt`` per component
  of the modulation correlation function;
* ``'SRSK'``: scalar relaxation of the second kind, Abragam's R1 and R2
  contributions to the spins scalar-coupled to a fast-relaxing source
  spin, added through the extended T1/T2 model.

Then the retention policy ``rlx_keep`` (``'diagonal'``, ``'kite'``,
``'secular'``, ``'labframe'``), the dynamic frequency shift policy, and the
thermalisation ``equilibrium`` (``'zero'``, ``'IME'``, ``'dibari'``) are
applied. Everything is ``sphten-liouv`` only, except ``'damp'`` and
:func:`equilibrium`, which also work in Hilbert space.

Spinach authors, by original file (headers of Spinach 2.13; see docs/spinach-credits.md):
  corrfun.m: Ilya Kuprov
  equilibrium.m: Luke Edwards, Ilya Kuprov
  expmint.m: Luke Edwards, David Goodwin, Ilya Kuprov
  redfield_integral_serial.m: Ilya Kuprov
  relaxation.m: Ilya Kuprov, Luke Edwards, Alexander Karabanov, Walter Kockenberger
  rlx_scalar.m: Ilya Kuprov
  thermalize.m: Ilya Kuprov, Filip Vestergaard Jensen
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import replace

import numpy as np
import scipy.sparse as sps

from spynach.assume import assume
from spynach.conventions import euler2dcm
from spynach.hamiltonian import hamiltonian, orientation
from spynach.operators import lin2lm
from spynach.propagation import propagator, step
from spynach.states import state
from spynach.system import SpinSystem
from spynach.utilities import Matrix, clean_up, matmul_clean, nnz

log = logging.getLogger("spynach")


def relaxation(spin_system: SpinSystem, euler_angles: Sequence[float] | None = None) -> Matrix:
    """Relaxation superoperator ``R`` (Hz) for the theories in ``inter['relaxation']``.

    ``euler_angles`` select the orientation for anisotropic rate tensors
    and for the thermal equilibrium of anisotropic systems. The result is
    a dissipative generator: the Liouvillian is ``H + 1j * R + 1j * K``.

    Spinach authors (relaxation.m): Ilya Kuprov, Luke Edwards, Alexander Karabanov, Walter Kockenberger.
    """
    if spin_system.basis is None:
        raise ValueError("basis set information is missing, run basis() before calling this function.")
    rlx = spin_system.rlx
    formalism = spin_system.basis.formalism
    dim = spin_system.basis.nstates if formalism == "sphten-liouv" else int(np.prod(spin_system.mults))
    R: Matrix = sps.csr_array((dim, dim), dtype=complex)
    if not rlx.theories:
        return R
    if formalism != "sphten-liouv" and set(rlx.theories) != {"damp"}:
        raise NotImplementedError(
            "Redfield, T1/T2, SRFK and SRSK relaxation theories are only available in the sphten-liouv formalism."
        )

    if "t1_t2" in rlx.theories:
        R = R + _t1_t2(spin_system, euler_angles)
    if "redfield" in rlx.theories:
        R = R + _redfield(spin_system)
        max_rate = np.abs(_diag(R)).max() if dim else 0.0
        for tau in rlx.tau_c:
            if max_rate > 0 and 1 / max_rate < tau.max():
                raise ValueError("T1,2>>tau_c validity condition violation in Redfield theory")
    if "SRFK" in rlx.theories:
        R = R + _srfk(spin_system)
    if "SRSK" in rlx.theories:
        R = R + _srsk(spin_system, R)

    if rlx.dfs == "ignore":
        R = R.real if sps.issparse(R) else np.real(R)
    R = _keep(spin_system, R)

    if "damp" in rlx.theories:
        if euler_angles is not None:
            ort = np.array([0, 0, 1.0]) @ euler2dcm(euler_angles)
            rate = float(ort @ np.atleast_2d(rlx.damp_rate) @ ort) if rlx.damp_rate.ndim == 2 else float(rlx.damp_rate)
        else:
            rate = float(np.mean(np.diag(rlx.damp_rate))) if rlx.damp_rate.ndim == 2 else float(rlx.damp_rate)
            if rlx.damp_rate.ndim == 2:
                log.warning("damping rate anisotropy ignored")
        if formalism == "zeeman-hilb":
            R = R - (rate / 2) * sps.eye_array(dim, dtype=complex, format="csr")
        else:
            # Liouville space: every state but the unit state is damped
            damping = -rate * np.ones(dim)
            damping[0] = 0
            R = R + sps.diags_array(damping.astype(complex), format="csr")

    if rlx.equilibrium == "IME":
        if formalism != "sphten-liouv":
            raise NotImplementedError("IME thermalisation is only available in Liouville space.")
        H_left = _lab_frame_left(spin_system, euler_angles)
        R = thermalize(spin_system, R, "IME", rho_eq=equilibrium(spin_system, H_left))
    elif rlx.equilibrium == "dibari":
        if rlx.temperature == 0:
            raise ValueError("DiBari-Levitt thermalization cannot be used with the high-temperature approximation.")
        R = thermalize(
            spin_system,
            R,
            "dibari",
            H_left=_lab_frame_left(spin_system, euler_angles),
            temperature=rlx.temperature,
        )
    return _complex(clean_up(spin_system.tols, R, spin_system.tols.rlx_zero))


def equilibrium(
    spin_system: SpinSystem, I: Matrix | None = None, Q: list | None = None, euler_angles: Sequence[float] | None = None
) -> np.ndarray | Matrix:
    """Thermal equilibrium state at ``inter['temperature']`` (port of ``equilibrium.m``).

    ``I`` is the lab-frame Hamiltonian as a *left-side product*
    superoperator in Liouville space (``hamiltonian(assume(spin_system, 'labframe'),
    'left')``) or the Hamiltonian matrix in Hilbert space; with ``Q`` and
    ``euler_angles`` the anisotropic part at that orientation is added.
    Without ``I`` the isotropic lab-frame Hamiltonian is built here. The
    Liouville-space result has unit coefficient on the unit state; the
    Hilbert-space density matrix has unit trace.

    Spinach authors (equilibrium.m): Luke Edwards, Ilya Kuprov.
    """
    if spin_system.basis is None:
        raise ValueError("basis set information is missing, run basis() before calling this function.")
    temperature = spin_system.rlx.temperature
    if temperature == 0:
        raise ValueError("absolute zero temperature is not supported.")
    formalism = spin_system.basis.formalism
    if I is None:
        I = hamiltonian(assume(spin_system, "labframe"), "left" if formalism == "sphten-liouv" else "comm")
    if Q is not None:
        if euler_angles is None:
            raise ValueError("euler_angles must be given with Q.")
        I = I + orientation(Q, euler_angles)
    beta = spin_system.tols.hbar / (spin_system.tols.kbol * temperature)
    if formalism == "sphten-liouv":
        unit = np.zeros(I.shape[1], dtype=complex)
        unit[0] = 1
        if np.abs(I @ unit).sum() < 1e-10:
            raise ValueError("I and Q must be left side product superoperators, not commutation superoperators.")
        rho = step(spin_system, I, unit, -1j * beta)
        if np.any(np.isnan(rho)):
            raise ValueError("numerical accuracy issues, temperature too low - switch to Hilbert space.")
        return rho / rho[0]
    if formalism != "zeeman-hilb":
        raise NotImplementedError(f"formalism {formalism!r} not ported yet.")
    norm = float(abs(beta) * abs(I).sum(axis=0).max())
    n_squarings = max(0, int(np.ceil(np.log2(norm)))) if norm > 0 else 0
    rho = propagator(spin_system, I, -1j * beta / 2**n_squarings)
    rho = rho / _trace(rho)
    for _ in range(n_squarings):
        rho = rho @ rho
        rho = clean_up(spin_system.tols, rho / _trace(rho), spin_system.tols.prop_chop)
    return rho


def thermalize(
    spin_system: SpinSystem,
    R: Matrix,
    method: str,
    *,
    rho_eq: np.ndarray | None = None,
    H_left: Matrix | None = None,
    temperature: float | None = None,
) -> Matrix:
    """Make the thermal equilibrium the stationary state of ``R`` (port of ``thermalize.m``).

    ``method='IME'`` (inhomogeneous master equation) needs ``rho_eq``:
    ``R -> R - (R rho_eq) u^T`` with ``u`` the unit state, so that
    ``R rho_eq = 0``. ``method='dibari'`` (DiBari-Levitt) needs ``H_left``,
    the left-side product superoperator of the lab-frame Hamiltonian
    (Spinach's ``HLSPS``), and the ``temperature`` in K:
    ``R -> R expm(-H_left beta)`` with ``beta = hbar / (k T)``. Spinach's
    signature is ``thermalize(spin_system, R, HLSPS, T, rho_eq, method)``.

    Spinach authors (thermalize.m): Ilya Kuprov, Filip Vestergaard Jensen.
    """
    if spin_system.basis is None or spin_system.basis.formalism != "sphten-liouv":
        raise NotImplementedError("thermalize() is only ported for the sphten-liouv formalism.")
    if R.ndim != 2 or R.shape[0] != R.shape[1]:
        raise ValueError("R must be a square matrix.")
    dim = R.shape[0]
    unit = np.zeros(dim, dtype=complex)
    unit[0] = 1
    if np.linalg.norm(R @ unit) > 1e-10:
        raise ValueError("R appears to be thermalized already.")
    if method == "IME":
        if rho_eq is None:
            raise ValueError("rho_eq must be given for the IME method.")
        drive = np.asarray(R @ np.asarray(rho_eq, dtype=complex)).reshape(-1)
        correction = sps.csr_array((drive, (np.arange(dim), np.zeros(dim, dtype=int))), shape=(dim, dim))
        return R - correction
    if method == "dibari":
        if H_left is None or temperature is None:
            raise ValueError("H_left and temperature must be given for the DiBari-Levitt method.")
        if np.abs(H_left @ unit).sum() < 1e-8:
            raise ValueError("H_left appears to be a commutation superoperator.")
        beta = spin_system.tols.hbar / (spin_system.tols.kbol * temperature)
        return R @ propagator(spin_system, H_left, 1j * beta)
    raise ValueError("method must be 'IME' or 'dibari'.")


def expmint(spin_system: SpinSystem, A: Matrix, B: Matrix, C: Matrix, T: float) -> Matrix:
    """``int_0^T exp(-iA t) B exp(-iC t) dt`` evaluated as a block of one exponential (port of ``expmint.m``).

    The auxiliary matrix ``[[-A, iB], [0, -C]]`` is exponentiated with
    :func:`propagator` over ``T``; the result is ``P^H Q`` with ``P`` the
    upper-left and ``Q`` the upper-right block.

    Spinach authors (expmint.m): Luke Edwards, David Goodwin, Ilya Kuprov.
    """
    dim = B.shape[0]
    if T == 0 or nnz(B) == 0:
        return sps.csr_array((dim, dim), dtype=complex)
    aux = sps.block_array([[-sps.csr_array(A), 1j * sps.csr_array(B)], [None, -sps.csr_array(C)]], format="csr")
    aux = propagator(spin_system, aux, T)
    aux = sps.csr_array(aux) if sps.issparse(aux) else np.asarray(aux)
    P = aux[:dim, :dim].conj().T
    Q = aux[:dim, dim:]
    return matmul_clean(spin_system.tols, P, Q, spin_system.tols.prop_chop)


def _redfield(spin_system: SpinSystem) -> Matrix:
    """The BRW relaxation integral (``redfield_integral_serial.m``) for rotational diffusion.

    Spinach authors (redfield_integral_serial.m): Ilya Kuprov.
    """
    tols = spin_system.tols
    labframe = assume(spin_system, "labframe")
    L0, Q = hamiltonian(labframe, "comm", aniso=True)
    dim = L0.shape[0]
    R: Matrix = sps.csr_array((dim, dim), dtype=complex)
    basis = spin_system.basis.basis
    species_states = [basis[:, list(part)].any(axis=1) for part in spin_system.chem.parts]
    # Spinach's loop structure and names. Q[rank - 1][k][m] is the anisotropic Hamiltonian
    # component that multiplies the Wigner matrix element D^rank(k, m); k and m (and p, q for
    # the second component) are 0-based positions in the component list, +rank first.
    # Rotational diffusion correlates component (k, m) with (p, q) through the weights and
    # decay rates of _corrfun; every nonzero weight contributes
    #     R -= weight * A * int_0^T exp(-iB t) C exp(-iD t) dt
    # with A = Q[k][m], B = L0 (the isotropic Liouvillian, rounded to the integration
    # tolerance), C = Q[p][q]^H and D = B shifted by the decay rate, so that the matrix
    # exponential carries the decay of the correlation function; T is where that decay
    # reaches tols.rlx_integration. Each term is restricted to the states of one substance.
    for rank, Q_rank in enumerate(Q, start=1):
        size = 2 * rank + 1
        for k in range(size):
            for m in range(size):
                if nnz(Q_rank[k][m]) == 0:
                    continue
                for p in range(size):
                    for q in range(size):
                        if nnz(Q_rank[p][q]) == 0:
                            continue
                        for s, (weights, rates) in enumerate(_corrfun(spin_system, rank, k, m, p, q)):
                            mask = species_states[s]
                            for weight, rate in zip(weights, rates):
                                if abs(weight) == 0:
                                    continue
                                upper_limit = -1.5 * (1 / rate) * np.log(1 / tols.rlx_integration)
                                B = clean_up(tols, L0, tols.rlx_integration / abs(upper_limit))
                                A = Q_rank[k][m]
                                C = Q_rank[p][q].conj().T
                                D = B - 1j * rate * sps.eye_array(dim, dtype=complex, format="csr")
                                A, B, C, D = (_restrict(X, mask) for X in (A, B, C, D))
                                R_int = -weight * (A @ expmint(spin_system, B, C, D, upper_limit))
                                R = R + clean_up(tols, R_int, 1e-2 * tols.rlx_zero)
    return R


def _corrfun(spin_system: SpinSystem, rank: int, k: int, m: int, p: int, q: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Weights and decay rates of the rotational diffusion correlation functions per chemical subsystem (``corrfun.m``).

    Indices ``k, m, p, q`` are 0-based positions in the ``(2*rank + 1)``
    component lists (``+rank`` first).

    Spinach authors (corrfun.m): Ilya Kuprov.
    """
    out = []
    for tau in spin_system.rlx.tau_c:
        if tau.size == 1:
            D = 1 / (6 * tau[0])
            weights = np.array([(1 / (2 * rank + 1)) * (k == p) * (m == q)], dtype=float)
            rates = np.array([-rank * (rank + 1) * D])
        elif tau.size == 2:
            D_ax, D_eq = 1 / (6 * tau[0]), 1 / (6 * tau[1])
            weights = np.array([(1 / (2 * rank + 1)) * (k == p) * (m == q)], dtype=float)
            proj = rank - m  # projection quantum number of component m
            rates = np.array([-(rank * (rank + 1) * D_eq + proj**2 * (D_ax - D_eq))])
        else:
            Dxx, Dyy, Dzz = 1 / (6 * tau)
            mean = np.mean([Dxx, Dyy, Dzz])
            if min(abs(Dxx - Dyy), abs(Dyy - Dzz), abs(Dzz - Dxx)) < 1e-6 * mean:
                raise ValueError("the three rotational correlation times must be different.")
            if rank != 2:
                raise ValueError("rhombic rotational diffusion only implemented for rank 2.")
            delta = np.sqrt(Dxx**2 + Dyy**2 + Dzz**2 - Dxx * Dyy - Dxx * Dzz - Dyy * Dzz)
            rates = -np.array(
                [
                    4 * Dxx + Dyy + Dzz,
                    Dxx + 4 * Dyy + Dzz,
                    Dxx + Dyy + 4 * Dzz,
                    2 * Dxx + 2 * Dyy + 2 * Dzz - 2 * delta,
                    2 * Dxx + 2 * Dyy + 2 * Dzz + 2 * delta,
                ]
            )
            lam_p = np.sqrt(2 / 3) * (Dxx + Dyy - 2 * Dzz + 2 * delta) / (Dxx - Dyy)
            lam_m = np.sqrt(2 / 3) * (Dxx + Dyy - 2 * Dzz - 2 * delta) / (Dxx - Dyy)
            h = np.zeros((5, 5))
            h[0, 1] = h[0, 3] = 1 / np.sqrt(2)
            h[1, 1], h[1, 3] = -1 / np.sqrt(2), 1 / np.sqrt(2)
            h[2, 4], h[2, 0] = 1 / np.sqrt(2), -1 / np.sqrt(2)
            h[3, 0] = h[3, 4] = 1 / np.sqrt(2 + lam_m**2)
            h[3, 2] = lam_m / np.sqrt(2 + lam_m**2)
            h[4, 0] = h[4, 4] = 1 / np.sqrt(2 + lam_p**2)
            h[4, 2] = lam_p / np.sqrt(2 + lam_p**2)
            weights = np.array([(1 / 5) * (k == p) * h[j, m] * h[j, q] for j in range(5)])
        out.append((weights, rates))
    return out


def _srfk(spin_system: SpinSystem) -> Matrix:
    """Scalar relaxation of the first kind (the SRFK block of ``relaxation.m`` and ``rlx_scalar.m``).

    ``H0`` is the lab-frame Hamiltonian commutation superoperator; ``H1``
    the one of the modulated scalar couplings alone, each
    ``2*pi*mdepth`` (rad/s) times the unit tensor, with every other
    interaction removed.

    Spinach authors (relaxation.m, a branch of the original): Ilya Kuprov, Luke Edwards, Alexander Karabanov,
      Walter Kockenberger.
    Spinach authors (rlx_scalar.m, a branch of the original): Ilya Kuprov.
    """
    rlx, tols = spin_system.rlx, spin_system.tols
    log.info("scalar relaxation of the first kind...")
    H0 = hamiltonian(assume(spin_system, "labframe"), "comm")
    nspins = spin_system.nspins
    modulated = tuple(
        tuple(
            None if rlx.srfk_mdepth[n][k] is None else 2 * np.pi * rlx.srfk_mdepth[n][k] * np.eye(3)
            for k in range(nspins)
        )
        for n in range(nspins)
    )
    local = replace(spin_system, coupling_matrix=modulated)
    H1 = hamiltonian(assume(local, "labframe", "couplings"), "comm")
    H0 = (H0 + H0.conj().T) / 2
    H1 = (H1 + H1.conj().T) / 2
    dim = H0.shape[0]
    RS: Matrix = sps.csr_array((dim, dim), dtype=complex)
    log.info("integrating the SRFK component...")
    for weight, tau in rlx.srfk_tau_c:
        if weight == 0 or tau == 0:
            continue
        upper_limit = 2 * tau * np.log(1 / tols.rlx_integration)
        H0c = clean_up(tols, H0, 1e-2 / upper_limit)
        C = H0c + (1j / tau) * sps.eye_array(dim, dtype=complex, format="csr")
        RS = RS - weight * (H1 @ expmint(spin_system, H0c, H1.conj().T, C, upper_limit))
    max_rate = np.abs(_diag(RS)).max() if dim else 0.0
    for _, tau in rlx.srfk_tau_c:
        if max_rate > 0 and 1 / max_rate < tau:
            log.info("1/max(diag(R)) = %g, srfk_tau_c = %g", 1 / max_rate, tau)
            raise ValueError("T1,2>>srfk_tau_c validity condition violation in Redfield theory")
    log.info("SRFK terms have been incorporated.")
    return RS


def _srsk(spin_system: SpinSystem, R: Matrix) -> Matrix:
    """Scalar relaxation of the second kind (the SRSK block of ``relaxation.m``).

    The T1 and T2 of each source spin are read from ``R`` as built so far;
    Abragam's expressions give R1 and R2 contributions to every spin of a
    different isotope that is scalar-coupled to it, and those rates enter
    through a recursive call with the extended T1/T2 model.

    Spinach authors (relaxation.m): Ilya Kuprov, Luke Edwards, Alexander Karabanov, Walter Kockenberger.
    """
    rlx = spin_system.rlx
    log.info("scalar relaxation of the second kind...")
    nspins = spin_system.nspins
    R1 = np.zeros(nspins)
    R2 = np.zeros(nspins)
    for k in rlx.srsk_sources:
        Lz_k = state(spin_system, ["Lz"], [k])
        Lp_k = state(spin_system, ["L+"], [k])
        Lz_k = Lz_k / np.linalg.norm(Lz_k)
        Lp_k = Lp_k / np.linalg.norm(Lp_k)
        T1k = -1 / np.vdot(Lz_k, R @ Lz_k).real
        T2k = -1 / np.vdot(Lp_k, R @ Lp_k).real
        Sk = (spin_system.mults[k] - 1) / 2
        log.info("SRSK from spin %d (%s, T1=%g s, T2=%g s):", k, spin_system.isotopes[k], T1k, T2k)
        for n in range(nspins):
            if n in rlx.srsk_sources or spin_system.isotopes[n] == spin_system.isotopes[k]:
                continue
            A = sum(
                np.trace(C) / 3
                for C in (spin_system.coupling_matrix[n][k], spin_system.coupling_matrix[k][n])
                if C is not None
            )
            if (2 * np.pi / T1k) < 10 * abs(A) or (2 * np.pi / T2k) < 10 * abs(A):
                raise ValueError("SRSK theory is not applicable: source spin relaxation is too slow")
            delta_omega = spin_system.basefrqs[n] - spin_system.basefrqs[k]
            R1add = (2 / 3) * A**2 * Sk * (Sk + 1) * (T2k / (1 + delta_omega**2 * T2k**2))
            R2add = (1 / 3) * A**2 * Sk * (Sk + 1) * (T1k + T2k / (1 + delta_omega**2 * T2k**2))
            R1[n] += R1add
            R2[n] += R2add
            if R1add > spin_system.tols.liouv_zero or R2add > spin_system.tols.liouv_zero:
                log.info("  > onto spin %d (%s): +%g Hz to R1, +%g Hz to R2", n, spin_system.isotopes[n], R1add, R2add)
    local = replace(
        spin_system,
        rlx=replace(
            rlx,
            theories=("t1_t2",),
            r1_rates=tuple(np.float64(r) for r in R1),
            r2_rates=tuple(np.float64(r) for r in R2),
        ),
    )
    log.info("recursive call for SRSK relaxation terms...")
    RS = relaxation(local)
    log.info("SRSK relaxation theory terms have been incorporated.")
    return RS


def _t1_t2(spin_system: SpinSystem, euler_angles: Sequence[float] | None) -> sps.csr_array:
    """Extended T1/T2 model: ``-diag(sum of R1 over longitudinal spins + sum of R2 over transverse spins)``."""
    rlx = spin_system.rlx
    rates = []
    for tensors in (rlx.r1_rates, rlx.r2_rates):
        values = np.zeros(spin_system.nspins)
        for n, rate in enumerate(tensors):
            if rate.ndim == 2:
                if euler_angles is None:
                    raise ValueError("Euler angles must be specified with anisotropic T1/T2 relaxation theory.")
                ort = np.array([0, 0, 1.0]) @ euler2dcm(euler_angles)
                values[n] = ort @ rate @ ort
            else:
                values[n] = rate
        rates.append(values)
    ranks, projs = lin2lm(spin_system.basis.basis)
    longitudinal = (ranks != 0) & (projs == 0)
    transverse = projs != 0
    diagonal = longitudinal @ rates[0] + transverse @ rates[1]
    return sps.diags_array((-diagonal).astype(complex), format="csr")


def _keep(spin_system: SpinSystem, R: Matrix) -> Matrix:
    """Retention policy ``rlx_keep``."""
    keep = spin_system.rlx.keep
    if keep == "labframe":
        return R
    dim = R.shape[0]
    if keep == "diagonal":
        diagonal = _diag(R).copy()
        diagonal[0] = 0  # the unit state is never relaxed
        return sps.diags_array(diagonal.astype(complex), format="csr")
    rows, cols, vals = _find(R)
    _, projs = lin2lm(spin_system.basis.basis)
    if keep == "kite":
        longitudinal = np.flatnonzero(np.abs(projs).sum(axis=1) == 0)
        mask = (np.isin(rows, longitudinal) & np.isin(cols, longitudinal)) | (rows == cols)
    elif keep == "secular":
        group = _frequency_groups(projs @ spin_system.basefrqs, spin_system.basefrqs)
        mask = group[rows] == group[cols]
    else:
        raise ValueError("invalid value of the inter.rlx_keep parameter.")
    return sps.csr_array((vals[mask], (rows[mask], cols[mask])), shape=(dim, dim))


def _frequency_groups(frequencies: np.ndarray, basefrqs: np.ndarray) -> np.ndarray:
    """Index of the Zeeman frequency of every state, frequencies equal to within rounding sharing one.

    The frequency of a state is a sum of base frequencies, and two sums with the same value can differ
    in the last bits (``w_C + w_F - w_F`` and ``w_C``), so exact equality, as in Spinach, would split a
    group and drop secular elements. Distinct Zeeman frequencies differ by a sizeable fraction of a
    base frequency, far above the ``1e-9`` relative tolerance.
    """
    if frequencies.size == 0:
        return np.zeros(0, dtype=np.int64)
    tolerance = 1e-9 * max(float(np.abs(basefrqs).max(initial=0.0)), 1.0)
    order = np.argsort(frequencies, kind="stable")
    group = np.empty(frequencies.size, dtype=np.int64)
    group[order] = np.cumsum(np.concatenate([[0], np.diff(frequencies[order]) > tolerance]))
    return group


def _lab_frame_left(spin_system: SpinSystem, euler_angles: Sequence[float] | None) -> Matrix:
    labframe = assume(spin_system, "labframe")
    if euler_angles is None:
        return hamiltonian(labframe, "left")
    H, Q = hamiltonian(labframe, "left", aniso=True)
    return H + orientation(Q, euler_angles)


def _restrict(X: Matrix, mask: np.ndarray) -> Matrix:
    """Zero the rows and columns outside ``mask`` (states of other chemical subsystems)."""
    if mask.all():
        return X
    keep = sps.diags_array(mask.astype(float), format="csr")
    return keep @ X @ keep


def _diag(R: Matrix) -> np.ndarray:
    return np.asarray(R.diagonal()).reshape(-1)


def _find(R: Matrix) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coo = sps.coo_array(R)
    return coo.row, coo.col, coo.data


def _trace(rho: Matrix) -> complex:
    return complex(rho.trace()) if sps.issparse(rho) else complex(np.trace(rho))


def _complex(R: Matrix) -> Matrix:
    return R.astype(complex) if sps.issparse(R) else np.asarray(R, dtype=complex)
