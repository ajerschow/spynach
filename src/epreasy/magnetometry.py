"""Magnetometry: EasySpin's curry.

``curry`` computes the magnetic moment along the field and the static
magnetic susceptibility of a spin system for lists of fields and
temperatures, averaged over a powder or for crystal orientations, from the
thermal populations of the eigenstates (the operator method, with the
susceptibility as a numerical field derivative) or from the partition
function.
"""

from __future__ import annotations

import numpy as np

from .constants import avogadro, bmagn, boltzm, mu0, planck
from .grids import sphgrid
from .hamiltonian import ham, ham_ez
from .resfields import _crystal_orientations
from .rotations import erot, vec2ang
from .spinsystem import validatespinsys
from .symmetry import hamsymm

__all__ = ["curry"]

_KEYWORDS = ("mu", "mumol", "muBM", "mueff", "muvec", "chi", "chimol", "chimolT", "1/chimol")


def curry(system: dict, exp: dict, opt: dict | None = None):  # noqa: C901
    """Magnetic moment and susceptibility for the fields ``exp["Field"]`` (mT) and temperatures
    ``exp["Temperature"]`` (K), each an array of shape (nFields, nTemperatures).

    ``opt["Output"]`` names the quantities, space-separated (default ``'muBM chimol'``): ``mu`` (J/T per
    centre), ``mumol`` (per mole), ``muBM`` (Bohr magnetons), ``muvec`` (the three lab-frame components),
    ``chi``, ``chimol`` (m^3/mol in SI), ``chimolT``, ``1/chimol``, ``mueff``. One keyword returns the array,
    several a tuple in that order. ``opt["Units"]``: ``'SI'`` or ``'CGS'``; ``opt["Method"]``: ``'operator'`` or
    ``'partitionfunction'``; ``opt["GridSize"]`` (default 10); ``opt["Spins"]`` (electron spins whose moment
    is taken); ``opt["deltaB"]`` (mT, the derivative step). ``exp["SampleFrame"]``/``MolFrame`` make it a crystal.
    """
    if isinstance(system, (list, tuple)):
        raise ValueError("curry does not support calculations with multiple components.")
    sys = validatespinsys(system)
    tip = float(system.get("TIP", 0.0))
    opt = dict(opt or {})
    exp = dict(exp)
    for obsolete in ("Orientations", "CrystalOrientation"):
        if obsolete in exp:
            raise ValueError(f"Exp.{obsolete} is no longer supported, use Exp.SampleFrame instead.")
    if exp.get("lightBeam"):
        raise ValueError("Photoselection (via Exp.lightBeam) is not supported.")
    if "Field" not in exp:
        exp["Field"] = 0.0
    B = np.atleast_1d(np.asarray(exp["Field"], dtype=float)).ravel() / 1e3  # T
    n_fields = B.size
    if "Temperature" not in exp:
        raise ValueError("Exp.Temperature is missing.")
    T = np.atleast_1d(np.asarray(exp["Temperature"], dtype=float)).ravel()
    n_temps = T.size
    if np.any(T < 0):
        raise ValueError("Negative temperatures are not possible.")
    zero_temp = np.any(T == 0)
    if exp.get("Ordering") is not None:
        raise NotImplementedError("Partially ordered samples are not supported in curry().")
    if exp.get("SampleRotation") is not None:
        raise NotImplementedError("Exp.SampleRotation is not ported yet.")
    crystal = exp.get("MolFrame") is not None or bool(exp.get("CrystalSymmetry"))
    if "nKnots" in opt:
        raise ValueError("Options.nKnots is obsolete. Use Options.GridSize instead, e.g. Options.GridSize = 91.")
    grid_size = int(opt.get("GridSize", 10))
    dB = float(opt.get("deltaB", 1.0)) * 1e-3  # T
    output = opt.get("Output", "muBM chimol")
    keywords = str(output).split()
    for word in keywords:
        if word not in _KEYWORDS:
            raise ValueError(f"'{word}' keyword in Opt.Output is not known.")
    calc_mu = any(k in ("mu", "mumol", "muBM", "muvec") for k in keywords)
    calc_vec = "muvec" in keywords
    calc_chi = any(k in ("mueff", "chi", "chimol", "chimolT", "1/chimol") for k in keywords)
    units = str(opt.get("Units", "SI")).upper()
    if units not in ("SI", "CGS"):
        raise ValueError(f"Unknown units '{units}' in Opt.Units. Use either 'SI' or 'CGS'.")
    cgs = units == "CGS"
    method = str(opt.get("Method", "operator")).lower()
    if method not in ("operator", "partitionfunction"):
        raise ValueError("Opt.Method can be 'operator' or 'partitionfunction'!")
    operator_method = method == "operator"
    spins = opt.get("Spins")
    if spins is not None and np.size(spins) > 0:
        spins = [int(s) for s in np.atleast_1d(spins)]
        if any(s < 1 or s > sys["nElectrons"] for s in spins):
            raise ValueError(f"Opt.Spins must contain electron spin indices between 1 and {sys['nElectrons']}.")
        if not operator_method:
            raise ValueError("To use Opt.Spins, you must use Opt.Method = 'operator'.")
    else:
        spins = None

    H0, muxM, muyM, muzM = ham(sys)
    if spins is not None:
        muxM, muyM, muzM = ham_ez(sys, None, spins)
    H0 = H0 * 1e6 * planck  # J
    c = 1e6 * 1e3 * planck  # MHz/mT -> J/T
    muxM, muyM, muzM = muxM * c, muyM * c, muzM * c

    # Orientations
    if not crystal:
        symmetry = opt.get("GridSymmetry") or None
        if symmetry is None:
            symmetry, frame = hamsymm(sys)
        else:
            frame = np.eye(3)
        if opt.get("GridFrame") is not None:
            frame = np.asarray(opt["GridFrame"], dtype=float)
        grid = sphgrid(symmetry, grid_size)
        phi, theta = vec2ang(frame @ grid.vecs)
        orientations = np.column_stack([phi, theta, np.zeros_like(phi)])
        weights = grid.weights / 4 / np.pi
        n_sites = 1
    else:
        sample_frame = exp.get("SampleFrame")
        exp_cr = {
            "MolFrame": exp.get("MolFrame") if exp.get("MolFrame") is not None else [0, 0, 0],
            "SampleFrame": sample_frame,
            "CrystalSymmetry": exp.get("CrystalSymmetry"),
        }
        orientations, n_orientations, n_sites, _ = _crystal_orientations(exp_cr, {})
        weights = np.full(orientations.shape[0], 1.0 / n_sites)
    n_orientations = orientations.shape[0]
    with np.errstate(divide="ignore"):  # T = 0 gives an infinite beta; the ground state alone is populated
        beta = 1 / T / boltzm  # (nT,)

    muz = np.zeros((n_fields, n_temps))
    mux = np.zeros((n_fields, n_temps))
    muy = np.zeros((n_fields, n_temps))
    chizz = np.zeros((n_fields, n_temps))

    def populations_of(E):
        with np.errstate(invalid="ignore"):  # 0 * inf at T = 0; the ground state is set below
            pop = np.exp(-np.outer(E - E[0], beta))  # (nStates, nT)
        if zero_temp:
            pop[0, T == 0] = 1.0
        return pop

    def calcmu(V, muOp, pop):
        expect = np.real(np.einsum("ij,jk,ki->i", V.conj().T, muOp, V))
        return (expect @ pop) / pop.sum(axis=0)

    for iOri in range(n_orientations):
        R = erot(orientations[iOri])
        xL, yL, zL = R[0], R[1], R[2]
        muOpzL = zL[0] * muxM + zL[1] * muyM + zL[2] * muzM
        if calc_vec:
            muOpxL = xL[0] * muxM + xL[1] * muyM + xL[2] * muzM
            muOpyL = yL[0] * muxM + yL[1] * muyM + yL[2] * muzM
        w = weights[iOri]
        for iB in range(n_fields):
            if operator_method:
                E, V = np.linalg.eigh(H0 - B[iB] * muOpzL)
                pop = populations_of(E)
                muz_ = calcmu(V, muOpzL, pop)
                if calc_vec:
                    mux_ = calcmu(V, muOpxL, pop)
                    muy_ = calcmu(V, muOpyL, pop)
                if calc_chi:
                    E2, V2 = np.linalg.eigh(H0 - (B[iB] + dB) * muOpzL)
                    pop2 = populations_of(E2)
                    chizz_ = (calcmu(V2, muOpzL, pop2) - muz_) / dB
            else:
                E1 = np.linalg.eigvalsh(H0 - (B[iB] - dB) * muOpzL)
                E3 = np.linalg.eigvalsh(H0 - (B[iB] + dB) * muOpzL)
                Emin = min(E1.min(), E3.min())

                def lnZ(E, Emin=Emin):
                    return np.log(np.sum(np.exp(-np.outer(E - Emin, beta)), axis=0))

                lnZ1, lnZ3 = lnZ(E1), lnZ(E3)
                if calc_mu:
                    muz_ = (lnZ3 - lnZ1) / (2 * dB) / beta
                if calc_chi:
                    E2 = np.linalg.eigvalsh(H0 - B[iB] * muOpzL)
                    chizz_ = (lnZ3 - 2 * lnZ(E2) + lnZ1) / dB**2 / beta
            if calc_mu:
                muz[iB] += w * muz_
            if calc_vec:
                mux[iB] += w * mux_
                muy[iB] += w * muy_
            if calc_chi:
                chizz[iB] += w * chizz_

    chizz = chizz * mu0 + tip / avogadro  # single-centre SI, with the temperature-independent part
    if cgs:
        muz, mux, muy = muz / 1e-3, mux / 1e-3, muy / 1e-3
        chizz = chizz / (4 * np.pi * 1e-6)
        muB = bmagn / 1e-3
        kB = boltzm / 1e-7
        c_eff = 3 * kB / muB**2
    else:
        muB = bmagn
        kB = boltzm
        c_eff = 3 * kB / muB**2 / mu0
    Tmat = np.broadcast_to(T[None, :], (n_fields, n_temps))
    outputs = []
    for word in keywords:
        if word == "mu":
            outputs.append(muz)
        elif word == "mumol":
            outputs.append(muz * avogadro)
        elif word == "muBM":
            outputs.append(muz / muB)
        elif word == "muvec":
            outputs.append(np.stack([mux, muy, muz]))
        elif word == "chi":
            outputs.append(chizz)
        elif word == "chimol":
            outputs.append(chizz * avogadro)
        elif word == "chimolT":
            outputs.append(chizz * avogadro * Tmat)
        elif word == "1/chimol":
            outputs.append(1 / (chizz * avogadro))
        else:
            outputs.append(np.sqrt(chizz * Tmat * c_eff))
    return outputs[0] if len(outputs) == 1 else tuple(outputs)
