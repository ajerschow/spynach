"""Simulation contexts (ports of ``liquid.m``, ``crystal.m``, ``powder.m``, ``singlerot.m`` and ``fourdif.m``).

A context builds the Hamiltonian, relaxation, and kinetics for a physical
setting and hands them to a pulse sequence function with the signature
``sequence(spin_system, parameters, H, R, K)``. ``singlerot`` works in the
Fokker-Planck space ``rotor phase (x) spin``: the state vector is the stack
of the spin state at every rotor phase grid point (spin index fastest, see
``porting/design-notes.md``), ``parameters['spc_dim']`` is the number of
rotor phases and ``parameters['spn_dim']`` the spin state space dimension.
"""

from __future__ import annotations

from collections.abc import Callable
from importlib import resources
from typing import Any

import numpy as np
import scipy.sparse as sps

from spynach.assume import assume
from spynach.conventions import wigner
from spynach.hamiltonian import frqoffset, hamiltonian, orientation
from spynach.kinetics import kinetics
from spynach.relaxation import equilibrium, relaxation
from spynach.system import SpinSystem
from spynach.utilities import Matrix, clean_up, nnz

PulseSequence = Callable[[SpinSystem, dict[str, Any], Matrix, Matrix, Matrix], Any]


def liquid(spin_system: SpinSystem, pulse_sequence: PulseSequence, parameters: dict[str, Any], assumptions: str) -> Any:
    """Run ``pulse_sequence(spin_system, parameters, H, R, K)`` in the isotropic liquid context.

    ``parameters`` must contain ``'spins'`` (isotopes of the working channels)
    and may contain ``'offset'`` (Hz per channel). ``K`` is
    :func:`spynach.kinetics` in Liouville space and zero in Hilbert space,
    where exchange cannot be described.
    """
    parameters = _defaults(spin_system, parameters)
    spin_system = assume(spin_system, assumptions)
    H = hamiltonian(spin_system)
    H = frqoffset(spin_system, H, parameters)
    R, K = _dissipators(spin_system, H.shape[0], None)
    parameters["spc_dim"], parameters["spn_dim"] = 1, H.shape[0]
    return pulse_sequence(spin_system, parameters, H, R, K)


def crystal(
    spin_system: SpinSystem, pulse_sequence: PulseSequence, parameters: dict[str, Any], assumptions: str
) -> Any:
    """Run a pulse sequence for a single orientation, ``parameters['orientation']`` (ZYZ Euler angles, rad).

    The Hamiltonian is the isotropic part plus the anisotropic components
    rotated to that orientation (:func:`spynach.orientation`), Hermitised.
    """
    parameters = _defaults(spin_system, parameters)
    if "orientation" not in parameters:
        raise ValueError("system orientation must be specified in parameters['orientation'] as three Euler angles.")
    angles = np.asarray(parameters["orientation"], dtype=float).reshape(-1)
    if angles.size != 3 or not np.all(np.isfinite(angles)):
        raise ValueError("parameters['orientation'] must be three finite Euler angles.")
    spin_system = assume(spin_system, assumptions)
    # Spinach's names: I is the isotropic Hamiltonian, Q the anisotropic irreducible components
    # that orientation() contracts with the Wigner matrices of the Euler angles
    I, Q = hamiltonian(spin_system, aniso=True)
    I = frqoffset(spin_system, I, parameters)
    H = _hermitise(I + orientation(Q, angles))
    R, K = _dissipators(spin_system, H.shape[0], angles)
    parameters["spc_dim"], parameters["spn_dim"] = 1, H.shape[0]
    return pulse_sequence(spin_system, parameters, H, R, K)


def powder(spin_system: SpinSystem, pulse_sequence: PulseSequence, parameters: dict[str, Any], assumptions: str) -> Any:
    """Powder average of a pulse sequence over the spherical grid ``parameters['grid']``.

    The grid names are Spinach's (``'rep_2ang_6400pts_sph'``, ``'leb_2ang_rank_23'``,
    ...; see :func:`load_grid`). With ``parameters['sum_up']`` true (the
    default) the weighted sum over orientations is returned; otherwise the
    list of per-orientation results. Orientations run serially; Spinach
    uses a parallel pool and computes the same sum. ``parameters['progress']``,
    if set, is called as ``progress(done, total)`` after each orientation.
    """
    parameters = _defaults(spin_system, parameters)
    parameters.setdefault("sum_up", True)
    if not parameters.get("grid"):
        raise ValueError("spherical averaging grid must be specified in parameters['grid'].")
    if callable(parameters.get("rho0")):
        raise NotImplementedError("orientation-dependent initial states (callable rho0) are not ported yet.")
    grid = load_grid(parameters["grid"])
    spin_system = assume(spin_system, assumptions)
    I, Q = hamiltonian(spin_system, aniso=True)
    I = frqoffset(spin_system, I, parameters)
    K = (
        kinetics(spin_system)
        if spin_system.basis.formalism == "sphten-liouv"
        else sps.csr_array(I.shape, dtype=complex)
    )
    parameters["spc_dim"], parameters["spn_dim"] = 1, I.shape[0]
    progress = _progress_hook(parameters)
    n_orientations = len(grid["alphas"])
    results = []
    for done, (alpha, beta, gamma) in enumerate(zip(grid["alphas"], grid["betas"], grid["gammas"]), start=1):
        local = dict(parameters)
        local.pop("progress", None)
        local["current_angles"] = np.array([alpha, beta, gamma])
        H = _hermitise(I + orientation(Q, [alpha, beta, gamma]))
        R = relaxation(spin_system, [alpha, beta, gamma])
        results.append(pulse_sequence(spin_system, local, H, R, K))
        if progress is not None:
            progress(done, n_orientations)
    if not parameters["sum_up"]:
        return results
    answer = grid["weights"][0] * results[0]
    for weight, result in zip(grid["weights"][1:], results[1:]):
        answer = answer + weight * result
    return answer


def singlerot(
    spin_system: SpinSystem, pulse_sequence: PulseSequence, parameters: dict[str, Any], assumptions: str
) -> Any:
    """Magic angle spinning (or any single-axis rotation) in Fokker-Planck form (port of ``singlerot.m``).

    Required parameters: ``'rate'`` (Hz, spinning rate), ``'axis'`` (the
    rotor axis in the lab frame, e.g. ``[1, 1, 1]`` for the magic angle),
    ``'max_rank'`` (the rotor phase grid has ``2*max_rank + 1`` points, so
    sidebands up to that order are resolved), and ``'grid'`` (spherical
    averaging grid, ``'single_crystal'`` for one orientation). The
    generator handed to the sequence is the block-diagonal stack of the
    Hamiltonians at the rotor phases plus the rotor-turning term
    ``i 2 pi rate (D_phi (x) 1)`` with a Fourier spectral derivative;
    ``R`` and ``K`` are replicated over the phases, ``rho0`` is averaged
    over them (powder) or placed at phase zero (single crystal), ``coil``
    is replicated. Liouville space only; the sequence runs for every
    orientation and the results are weighted and summed (``'sum_up'``).
    ``parameters['progress']``, if set, is called as ``progress(done, total)``
    after each orientation.
    """
    parameters = _defaults(spin_system, parameters)
    parameters.setdefault("sum_up", True)
    for key in ("rate", "axis", "max_rank", "grid"):
        if key not in parameters:
            raise ValueError(f"parameters['{key}'] must be present.")
    if spin_system.basis.formalism != "sphten-liouv":
        raise NotImplementedError("singlerot() is only ported for the sphten-liouv formalism.")
    if callable(parameters.get("rho0")):
        raise NotImplementedError("orientation-dependent initial states (callable rho0) are not ported yet.")
    max_rank = int(parameters["max_rank"])
    if max_rank < 0 or max_rank != parameters["max_rank"]:
        raise ValueError("parameters['max_rank'] must be a non-negative integer.")
    axis = np.asarray(parameters["axis"], dtype=float).reshape(-1)
    if axis.size != 3:
        raise ValueError("parameters['axis'] must be a vector of three real numbers.")
    grid = load_grid(parameters["grid"])

    spin_system = assume(spin_system, assumptions)
    I, Q = hamiltonian(spin_system, aniso=True)
    I = frqoffset(spin_system, I, parameters)
    R = relaxation(spin_system)
    K = kinetics(spin_system)
    # Rotor direction in the lab frame: polar angles of the axis
    rotor_phi = float(np.arctan2(axis[1], axis[0]))
    rotor_theta = float(np.pi / 2 - np.arctan2(axis[2], np.hypot(axis[0], axis[1])))

    spn_dim = I.shape[0]
    spc_dim = 2 * max_rank + 1
    parameters["spc_dim"], parameters["spn_dim"] = spc_dim, spn_dim
    rotor_phases, d_dphi = fourdif(spc_dim, 1)
    M = 2 * np.pi * float(parameters["rate"]) * sps.kron(sps.csr_array(d_dphi), sps.eye_array(spn_dim), format="csr")
    eye_space = sps.eye_array(spc_dim, format="csr")
    R_fp, K_fp = sps.kron(eye_space, R, format="csr"), sps.kron(eye_space, K, format="csr")
    if "rho0" in parameters:
        space_part = np.zeros(spc_dim) if parameters["grid"] == "single_crystal" else np.full(spc_dim, 1 / spc_dim)
        if parameters["grid"] == "single_crystal":
            space_part[0] = 1
        parameters["rho0"] = np.kron(space_part, np.asarray(parameters["rho0"], dtype=complex))
    if "coil" in parameters:
        parameters["coil"] = np.kron(np.ones(spc_dim), np.asarray(parameters["coil"], dtype=complex))

    D_lab2rot = [wigner(rank, rotor_phi, rotor_theta, 0) for rank in (1, 2)]
    D_rotor = [[wigner(rank, 0, 0, phase) for rank in (1, 2)] for phase in rotor_phases]
    progress = _progress_hook(parameters)
    n_orientations = len(grid["alphas"])
    results = []
    for done, (alpha, beta, gamma) in enumerate(zip(grid["alphas"], grid["betas"], grid["gammas"]), start=1):
        D_mol2rot = [wigner(rank, alpha, beta, gamma) for rank in (1, 2)]
        blocks = []
        for n in range(spc_dim):
            H = I
            for r, Q_rank in enumerate(Q):
                D = D_lab2rot[r] @ D_rotor[n][r] @ D_mol2rot[r]
                for k in range(2 * r + 3):
                    for m in range(2 * r + 3):
                        block = Q_rank[k][m]
                        if nnz(block) > 0:
                            H = H + D[k, m] * block
            blocks.append(sps.csr_array(H))
        G = clean_up(spin_system.tols, sps.block_diag(blocks, format="csr") + 1j * M, spin_system.tols.liouv_zero)
        local = dict(parameters)
        local.pop("progress", None)
        results.append(pulse_sequence(spin_system, local, G, R_fp, K_fp))
        if progress is not None:
            progress(done, n_orientations)
    if not parameters["sum_up"]:
        return results
    answer = grid["weights"][0] * results[0]
    for weight, result in zip(grid["weights"][1:], results[1:]):
        answer = answer + weight * result
    return answer


def fourdif(npoints: int, order: int) -> tuple[np.ndarray, np.ndarray]:
    """Fourier spectral differentiation on a periodic grid (port of ``fourdif.m``, Weideman and Reddy).

    Returns the grid ``2 pi k / N`` and the dense ``N x N`` matrix of the
    ``order``-th derivative, exact for trigonometric polynomials the grid
    resolves.
    """
    N, m = int(npoints), int(order)
    if N < 1 or m < 1:
        raise ValueError("N and m must be positive integers.")
    x = 2 * np.pi * np.arange(N) / N
    h = 2 * np.pi / N
    kk = np.arange(1, N)
    n1, n2 = (N - 1) // 2, -(-(N - 1) // 2)
    if m == 1:
        if N % 2 == 0:
            topc = 1 / np.tan(np.arange(1, n2 + 1) * h / 2)
            col1 = np.concatenate([[0], 0.5 * (-1.0) ** kk * np.concatenate([topc, -topc[:n1][::-1]])])
        else:
            topc = 1 / np.sin(np.arange(1, n2 + 1) * h / 2)
            col1 = np.concatenate([[0], 0.5 * (-1.0) ** kk * np.concatenate([topc, topc[:n1][::-1]])])
        row1 = -col1
    elif m == 2:
        if N % 2 == 0:
            topc = 1 / np.sin(np.arange(1, n2 + 1) * h / 2) ** 2
            col1 = np.concatenate(
                [[-(np.pi**2) / 3 / h**2 - 1 / 6], -0.5 * (-1.0) ** kk * np.concatenate([topc, topc[:n1][::-1]])]
            )
        else:
            angles = np.arange(1, n2 + 1) * h / 2
            topc = 1 / np.sin(angles) / np.tan(angles)
            col1 = np.concatenate(
                [[-(np.pi**2) / 3 / h**2 + 1 / 12], -0.5 * (-1.0) ** kk * np.concatenate([topc, -topc[:n1][::-1]])]
            )
        row1 = col1
    else:
        N1 = (N - 1) // 2
        N2 = np.array([-N / 2] * ((N + 1) % 2)) * ((m + 1) % 2)
        mwave = 1j * np.concatenate([np.arange(N1 + 1), N2, np.arange(-N1, 0)])
        col1 = np.real(np.fft.ifft(mwave**m * np.fft.fft(np.concatenate([[1], np.zeros(N - 1)]))))
        if m % 2 == 0:
            row1 = col1
        else:
            col1 = np.concatenate([[0], col1[1:]])
            row1 = -col1
    from scipy.linalg import toeplitz

    return x, toeplitz(col1, row1)


def load_grid(name: str) -> dict[str, np.ndarray]:
    """Spherical averaging grid by its Spinach name: arrays ``alphas``, ``betas``, ``gammas`` (rad) and ``weights``.

    The grids are Spinach's ``kernel/grids`` files converted to NumPy
    archives (``porting/tools/gen_grids.py``): Lebedev (``leb_*``),
    repulsion (``rep_*``), icosahedral (``icos_*``) and ``single_crystal``.
    """
    path = resources.files("spynach").joinpath("grids", f"{name}.npz")
    if not path.is_file():
        raise ValueError(f"unknown spherical grid {name!r}; see spynach/grids for the available names.")
    with resources.as_file(path) as file, np.load(file) as data:
        return {key: np.array(data[key]) for key in ("alphas", "betas", "gammas", "weights")}


def _progress_hook(parameters: dict[str, Any]):
    """Return ``parameters['progress']`` when it is a ``progress(done, total)`` callable."""
    progress = parameters.get("progress")
    if progress is None:
        return None
    if not callable(progress):
        raise TypeError("parameters['progress'] must be a callable, progress(done, total).")
    return progress


def _defaults(spin_system: SpinSystem, parameters: dict[str, Any]) -> dict[str, Any]:
    if spin_system.basis is None:
        raise ValueError("basis set information is missing, run basis() before calling this function.")
    if "spins" not in parameters or not parameters["spins"]:
        raise ValueError("working spins must be specified in parameters['spins'].")
    parameters = dict(parameters)
    parameters.setdefault("decouple", [])
    parameters.setdefault("rframes", [])
    parameters.setdefault("offset", np.zeros(len(parameters["spins"])))
    parameters.setdefault("needs", [])
    if parameters["rframes"]:
        raise NotImplementedError("additional rotating frame transformations are not ported yet.")
    unported = set(parameters["needs"]) - {"rho_eq", "iso_eq"}
    if unported:
        raise NotImplementedError(f"parameters['needs'] {sorted(unported)} not ported yet.")
    if parameters["needs"] and "rho0" in parameters:
        raise ValueError("thermal equilibrium request conflicts with user-specified initial condition.")
    if parameters["needs"]:
        # 'rho_eq' (liquid) and 'iso_eq' (singlerot): the thermal state of the isotropic lab-frame Hamiltonian
        parameters["rho0"] = equilibrium(spin_system)
    return parameters


def _dissipators(spin_system: SpinSystem, dim: int, angles: Any) -> tuple[Matrix, Matrix]:
    R = relaxation(spin_system) if angles is None else relaxation(spin_system, angles)
    if spin_system.basis.formalism == "sphten-liouv":
        return R, kinetics(spin_system)
    return R, sps.csr_array((dim, dim), dtype=complex)


def _hermitise(H: Matrix) -> Matrix:
    H = (H + H.conj().T) / 2
    return sps.csr_array(H) if sps.issparse(H) else np.asarray(H)
