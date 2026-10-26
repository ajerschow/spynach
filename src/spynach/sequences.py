"""Pulse sequences (ports of ``experiments/acquire.m`` and ``experiments/nmr_liquids/{cosy,noesy,hsqc}.m``).

Every sequence has Spinach's signature, ``sequence(spin_system, parameters,
H, R, K)``, is handed to a context (:func:`spynach.liquid`, ...) and forms
the Liouvillian as ``L = H + 1j * R + 1j * K``. Pulses are ideal rotations
by :func:`spynach.step` with the pulse operator and a flip angle; the
indirect dimension of a 2-D experiment is a stack of state vectors from
the ``'trajectory'`` output of :func:`spynach.evolution`, so the second
dimension is propagated for all `t1` increments at once. 2-D FIDs are
``(npoints[1], npoints[0])`` arrays: rows run along the direct dimension
``t2``, columns along ``t1``, as in Spinach.

Spinach authors, by original file (headers of Spinach 2.13; see docs/spinach-credits.md):
  acquire.m: Ilya Kuprov, Luke Edwards
  cosy.m: Gareth Charnock, Ilya Kuprov
  hsqc.m: Ilya Kuprov, Luke Edwards
  noesy.m: Ilya Kuprov, Luke Edwards, Hannah Hogben
"""

from __future__ import annotations

from typing import Any

import numpy as np

from spynach.filters import coherence, decouple, homospoil
from spynach.operators import operator
from spynach.propagation import evolution, step
from spynach.states import state
from spynach.system import SpinSystem
from spynach.utilities import Matrix


def acquire(spin_system: SpinSystem, parameters: dict[str, Any], H: Matrix, R: Matrix, K: Matrix) -> np.ndarray:
    """Free induction decay: evolve ``parameters['rho0']`` and detect ``parameters['coil']``.

    Required parameters: ``'sweep'`` (Hz, sets the dwell time 1/sweep),
    ``'npoints'``, ``'rho0'``, ``'coil'``. Optional: ``'decouple'``
    (isotopes to decouple analytically, Liouville space), ``'dead_time'``
    (s, free evolution before acquisition), ``'homodec_oper'`` with
    ``'homodec_pwr'`` (Hz, a homodecoupling operator added to the
    Liouvillian). Returns ``npoints`` complex points.

    Spinach authors (acquire.m): Ilya Kuprov, Luke Edwards.
    """
    for key in ("sweep", "npoints", "rho0", "coil"):
        if key not in parameters:
            raise ValueError(f"parameters['{key}'] must be present.")
    L = H + 1j * R + 1j * K
    rho0 = parameters["rho0"]
    L, rho0 = decouple(spin_system, L, rho0, parameters.get("decouple", []))
    if "homodec_oper" in parameters:
        if "homodec_pwr" not in parameters:
            raise ValueError("homodecoupling power must be specified in parameters['homodec_pwr'].")
        spc_dim = parameters.get("spc_dim", 1)
        oper = parameters["homodec_oper"]
        if spc_dim > 1:
            import scipy.sparse as sps

            oper = sps.kron(sps.eye_array(spc_dim), sps.csr_array(oper), format="csr")
        L = L + 2 * np.pi * parameters["homodec_pwr"] * oper
    if "dead_time" in parameters:
        rho0 = step(spin_system, L, rho0, parameters["dead_time"])
    return evolution(
        spin_system,
        L,
        rho0,
        1 / parameters["sweep"],
        parameters["npoints"] - 1,
        "observable",
        coil=parameters["coil"],
    )


def cosy(spin_system: SpinSystem, parameters: dict[str, Any], H: Matrix, R: Matrix, K: Matrix) -> np.ndarray:
    """Phase-sensitive COSY with a second pulse of ``parameters['angle']`` (port of ``cosy.m``).

    Parameters: ``'sweep'`` (one value for both dimensions), ``'npoints'``
    (two values), ``'spins'`` (one isotope), ``'angle'`` (rad; ``pi/2`` for
    COSY-90, ``pi/4`` for COSY-45). Returns a ``(npoints[1], npoints[0])``
    FID; process with ``fft2``.

    Spinach authors (cosy.m): Gareth Charnock, Ilya Kuprov.
    """
    _check_2d(spin_system, parameters, nspins=1, sweeps=1, extra=("angle",))
    L = H + 1j * R + 1j * K
    isotope = parameters["spins"][0]
    rho = state(spin_system, "Lz", isotope, "cheap")
    coil = state(spin_system, "L+", isotope, "cheap")
    Lx = operator(spin_system, "Lx", isotope)
    rho = step(spin_system, Lx, rho, np.pi / 2)
    rho_stack = evolution(spin_system, L, rho, 1 / parameters["sweep"], parameters["npoints"][0] - 1, "trajectory")
    rho_stack = coherence(spin_system, rho_stack.T, [(isotope, 1)]).T
    rho_stack = step(spin_system, Lx, rho_stack, parameters["angle"])
    return evolution(
        spin_system,
        L,
        rho_stack,
        1 / parameters["sweep"],
        parameters["npoints"][1] - 1,
        "observable",
        coil=coil,
    )


def noesy(
    spin_system: SpinSystem, parameters: dict[str, Any], H: Matrix, R: Matrix, K: Matrix
) -> dict[str, np.ndarray]:
    """States-TPPI NOESY (port of ``noesy.m``).

    Parameters: ``'sweep'`` and ``'npoints'`` (two values each), ``'spins'``
    (one isotope), ``'tmix'`` (s), ``'rho0'`` (usually the thermal state,
    ``needs=['rho_eq']``), optional ``'decouple'`` and ``'oldschool'``
    (keep the Hamiltonian during mixing instead of a homospoil and
    relaxation only). Returns ``{'cos': fid, 'sin': fid}`` for the States
    processing of the indirect dimension.

    Spinach authors (noesy.m): Ilya Kuprov, Luke Edwards, Hannah Hogben.
    """
    _check_2d(spin_system, parameters, nspins=1, sweeps=2, extra=("tmix", "rho0"))
    oldschool = bool(parameters.get("oldschool", False))
    timestep = 1 / np.asarray(parameters["sweep"], dtype=float)
    isotope = parameters["spins"][0]
    coil = state(spin_system, "L+", isotope, "cheap")
    Lx = operator(spin_system, "Lx", isotope)
    Ly = operator(spin_system, "Ly", isotope)
    rho0 = parameters["rho0"]
    H, rho0 = decouple(spin_system, H, rho0, parameters.get("decouple", []))
    L = H + 1j * R + 1j * K
    rho = step(spin_system, Lx, rho0, np.pi / 2)
    second = ((Lx, np.pi / 2), (Ly, np.pi / 2), (Lx, -np.pi / 2), (Ly, -np.pi / 2))
    fids = []
    for op2, angle2 in second:
        rho_stack = evolution(spin_system, L, rho, timestep[0], parameters["npoints"][0] - 1, "trajectory")
        rho_stack = step(spin_system, op2, rho_stack, angle2)
        if oldschool:
            rho_stack = evolution(spin_system, L, rho_stack, parameters["tmix"], 1, "final")
        else:
            rho_stack = homospoil(spin_system, rho_stack.T, "destroy").T
            rho_stack = evolution(spin_system, 1j * R + 1j * K, rho_stack, parameters["tmix"], 1, "final")
        rho_stack = step(spin_system, Ly, rho_stack, np.pi / 2)
        fids.append(
            evolution(
                spin_system,
                L,
                rho_stack,
                timestep[1],
                parameters["npoints"][1] - 1,
                "observable",
                coil=coil,
            )
        )
    return {"cos": fids[0] - fids[2], "sin": fids[1] - fids[3]}


def hsqc(spin_system: SpinSystem, parameters: dict[str, Any], H: Matrix, R: Matrix, K: Matrix) -> dict[str, np.ndarray]:
    """Echo-antiecho HSQC (port of ``hsqc.m``).

    Parameters: ``'sweep'``, ``'npoints'`` (two values each, F1 then F2),
    ``'spins'`` (two isotopes: the indirect heteronucleus, then the
    detected one), ``'J'`` (Hz, the one-bond coupling setting the INEPT
    delays), ``'decouple_f1'`` and ``'decouple_f2'`` (isotope lists);
    optional ``'rho0'`` and ``'coil'`` (defaults: ``Lz`` and ``L+`` of the
    detected isotope). Returns ``{'pos': fid, 'neg': fid}`` for the
    echo-antiecho processing of the indirect dimension.

    Spinach authors (hsqc.m): Ilya Kuprov, Luke Edwards.
    """
    _check_2d(spin_system, parameters, nspins=2, sweeps=2, extra=("J", "decouple_f1", "decouple_f2"))
    if parameters["spins"][0] == parameters["spins"][1]:
        raise ValueError("parameters['spins'] must specify two different isotopes.")
    if parameters["spins"][0] in parameters["decouple_f1"]:
        raise ValueError("parameters['decouple_f1'] must not contain the active indirect isotope.")
    L = H + 1j * R + 1j * K
    timestep = 1 / np.asarray(parameters["sweep"], dtype=float)
    delta = abs(1 / (2 * parameters["J"]))
    f1, f2 = parameters["spins"]
    rho0 = parameters.get("rho0", state(spin_system, "Lz", f2, "cheap"))
    coil = parameters.get("coil", state(spin_system, "L+", f2, "cheap"))
    Lx_F1, Lx_F2, Ly_F2 = (
        operator(spin_system, "Lx", f1),
        operator(spin_system, "Lx", f2),
        operator(spin_system, "Ly", f2),
    )

    # INEPT: proton magnetisation to antiphase heteronuclear coherence
    rho = step(spin_system, Lx_F2, rho0, np.pi / 2)
    rho = evolution(spin_system, L, rho, delta / 2, 1, "final")
    rho = step(spin_system, Lx_F2 + Lx_F1, rho, np.pi)
    rho = evolution(spin_system, L, rho, delta / 2, 1, "final")
    rho = step(spin_system, Ly_F2, rho, np.pi / 2)
    rho = step(spin_system, Lx_F1, rho, np.pi / 2) - step(spin_system, Lx_F1, rho, -np.pi / 2)

    # Indirect dimension with a refocusing pulse on the decoupled channels in the middle
    rho_stack = evolution(spin_system, L, rho, timestep[0] / 2, parameters["npoints"][0] - 1, "trajectory")
    for isotope in parameters["decouple_f1"]:
        Lp = operator(spin_system, "L+", isotope)
        rho_stack = step(spin_system, (Lp + Lp.conj().T) / 2, rho_stack, np.pi)
    rho_stack = evolution(spin_system, L, rho_stack, timestep[0] / 2, parameters["npoints"][0] - 1, "refocus")

    # Echo and antiecho coherence pathways, back transfer, decoupled acquisition
    fids = {}
    for key, order in (("pos", 1), ("neg", -1)):
        stack = coherence(spin_system, rho_stack.T, [(f2, 0), (f1, order)]).T
        stack = step(spin_system, Lx_F1 + Lx_F2, stack, np.pi / 2)
        stack = evolution(spin_system, L, stack, delta / 2, 1, "final")
        stack = step(spin_system, Lx_F1 + Lx_F2, stack, np.pi)
        stack = evolution(spin_system, L, stack, delta / 2, 1, "final")
        L_dec, stack = decouple(spin_system, L, stack, parameters["decouple_f2"])
        fids[key] = evolution(
            spin_system,
            L_dec,
            stack,
            timestep[1],
            parameters["npoints"][1] - 1,
            "observable",
            coil=coil,
        )
    return fids


def _check_2d(
    spin_system: SpinSystem, parameters: dict[str, Any], nspins: int, sweeps: int, extra: tuple[str, ...]
) -> None:
    if spin_system.basis is None or spin_system.basis.formalism != "sphten-liouv":
        raise NotImplementedError("this sequence is only available for the sphten-liouv formalism.")
    for key in ("sweep", "npoints", "spins") + extra:
        if key not in parameters:
            raise ValueError(f"parameters['{key}'] must be present.")
    if np.size(parameters["sweep"]) != sweeps:
        raise ValueError(f"parameters['sweep'] should have exactly {sweeps} element(s).")
    if np.size(parameters["npoints"]) != 2:
        raise ValueError("parameters['npoints'] should have exactly two elements.")
    if len(parameters["spins"]) != nspins:
        raise ValueError(f"parameters['spins'] should have exactly {nspins} element(s).")
    if any(iso not in spin_system.isotopes for iso in parameters["spins"]):
        raise ValueError("parameters['spins'] refers to an isotope that is not present in the system.")
