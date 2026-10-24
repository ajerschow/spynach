"""Resonance fields of cw EPR: EasySpin's ``resfields`` (adaptive spline modelling of the energy level
diagram) and ``resfields_eig`` (the exact eigenfield method), with the helpers they use.

``resfields(sys, exp, opt)`` takes a spin system dict, an experiment dict
(``mwFreq`` in GHz, ``Range`` or ``CenterSweep`` in mT, ``Temperature``,
``SampleFrame``, ``MolFrame``, ``mwMode``, ``lightBeam``) and an options dict
(``Transitions``, ``Threshold``, ``Method``/``Hybrid``, ``HybridCoreNuclei``,
``FuzzLevel``, ...), and returns ``Resonances(pos, int, wid, trans)``: line
positions (mT) and intensities as (transitions x orientations) arrays, the
strain widths (FWHM, mT) when strains are given, and the level pairs. The
orientations are those of ``Exp.SampleFrame`` (one lab-frame orientation per
row, crystal symmetry P1); ``pepper`` supplies grid orientations through the
private ``resfields_oriented``.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any, NamedTuple

import numpy as np
import scipy.linalg as sla

from .constants import boltzm, nmagn, planck
from .grids import sphgrid
from .nucdata import nucstring2list
from .rotations import ang2vec, erot, eulang, letter2vec, vec2ang
from .spinops import sop
from .spinsystem import validatespinsys


class Resonances(NamedTuple):
    pos: np.ndarray  # (nTransitions, nOrientations), mT; NaN where out of range
    int: np.ndarray | None  # same shape
    wid: np.ndarray | None  # same shape, FWHM in mT, when strains are present
    trans: np.ndarray  # (nTransitions, 2), 1-based level pairs u < v


class MwMode(NamedTuple):
    xi1: float
    xik: float
    nB1: np.ndarray
    nk: np.ndarray
    nB0: np.ndarray
    linear: bool
    unpolarized: bool
    circular: bool
    circ_sense: float
    parallel: bool


# ----------------------------------------------------------------------------- option handling
def adddefaults(user: dict | None, defaults: dict) -> dict:
    """``defaults`` updated by ``user``; a user key that matches a default only up to case is an error."""
    out = dict(defaults)
    if not user:
        return out
    for name, value in user.items():
        exact = name in defaults
        similar = [d for d in defaults if d.lower() == name.lower()]
        if not exact and similar:
            raise ValueError(
                f"Supplied field '{name}' is similar to '{similar[0]}'. Please correct spelling or remove field."
            )
        out[name] = value
    return out


def _mw_mode(mw_mode) -> MwMode:
    """EasySpin's ``p_excitationgeometry``: the microwave field and propagation directions in the lab frame."""
    if mw_mode is None or mw_mode == "":
        mw_mode = "perpendicular"
    linear = unpolarized = circular = parallel = False
    circ_sense = np.nan
    if isinstance(mw_mode, str):
        linear = True
        if mw_mode == "perpendicular":
            k, alpha = "y", -np.pi / 2
        elif mw_mode == "parallel":
            k, alpha, parallel = "y", np.pi, True
        else:
            raise ValueError(f"Unrecognized Exp.mwMode: '{mw_mode}'.")
    elif isinstance(mw_mode, (list, tuple)) and len(mw_mode) == 2:
        k, mode2 = mw_mode
        if isinstance(mode2, str):
            if mode2 == "unpolarized":
                unpolarized = True
            elif mode2 == "circular+":
                circular, circ_sense = True, 1.0
            elif mode2 == "circular-":
                circular, circ_sense = True, -1.0
            else:
                raise ValueError(f"Unrecognized 2nd element in Exp.mwMode: '{mode2}'")
            alpha = 0.0
        else:
            linear, alpha = True, float(mode2)
    else:
        raise ValueError("Exp.mwMode must be 'perpendicular' or 'parallel' or a 2-element list [k, alpha].")
    if isinstance(k, str):
        p, t = vec2ang(letter2vec(k))
        k_ori = (p[0], t[0])
    else:
        k = np.asarray(k, dtype=float).ravel()
        if k.size == 3:
            p, t = vec2ang(k)
            k_ori = (p[0], t[0])
        elif k.size == 2:
            k_ori = (k[0], k[1])
        elif k.size == 1:
            k_ori = (0.0, k[0])
        else:
            raise ValueError("k (first element in mwMode) must be a letter, a 3-vector, or an array with two angles.")
    R = erot([k_ori[0], k_ori[1], alpha])
    nB1, nk = R[0], R[2]
    nB0 = np.array([0.0, 0.0, 1.0])
    return MwMode(float(nB1 @ nB0), float(nk @ nB0), nB1, nk, nB0, linear, unpolarized, circular, circ_sense, parallel)


def _crystal_orientations(exp: dict, opt: dict) -> tuple[np.ndarray, int, int, bool]:
    """EasySpin's ``p_crystalorientations`` for symmetry P1: lab-frame Euler angles per sample orientation."""
    mol_frame = np.atleast_2d(np.asarray(exp["MolFrame"], dtype=float))
    if mol_frame.shape[1] != 3:
        raise ValueError("Exp.MolFrame requires three numbers per row (three Euler angles).")
    symmetry = exp.get("CrystalSymmetry", 1)
    if symmetry not in (1, "P1", None):
        raise NotImplementedError("Crystal symmetry other than P1 (site transformations) is not ported yet.")
    if exp.get("SampleRotation"):
        raise NotImplementedError("Exp.SampleRotation is not ported yet.")
    if opt.get("Sites"):
        raise NotImplementedError("Opt.Sites needs crystal symmetry, which is not ported yet.")
    sample_frame = exp.get("SampleFrame")
    if sample_frame is None or np.size(sample_frame) == 0:
        R_L2S = [np.eye(3)]
    else:
        sample_frame = np.atleast_2d(np.asarray(sample_frame, dtype=float))
        R_L2S = [erot(row) for row in sample_frame]
    R_S2M = erot(mol_frame[0])
    xyzM_S = R_S2M.T @ np.eye(3)
    angles = np.array([eulang(R.T @ xyzM_S, nocheck=True) for R in R_L2S])
    return angles, angles.shape[0], 1, False


def photoselect(tdm, ori, k, alpha: float) -> np.ndarray:
    """Photoselection weight (0..1) of a spin centre: transition dipole ``tdm`` (letter, vector or angles) in the
    molecular frame, lab-frame orientations ``ori`` (rows ``[phi, theta]`` integrate over chi; ``[phi, theta, chi]``
    do not), light propagation ``k`` and polarisation angle ``alpha`` (``nan`` for unpolarised light)."""
    if isinstance(tdm, str):
        tdm_mol = letter2vec(tdm)
    else:
        tdm = np.asarray(tdm, dtype=float).ravel()
        if tdm.size == 3:
            tdm_mol = tdm / np.linalg.norm(tdm)
        elif tdm.size == 2:
            tdm_mol = ang2vec(tdm[0], tdm[1]).ravel()
        else:
            raise ValueError(
                "tdm (first input) must be a letter designating a direction, a 3-vector, or an array with two angles."
            )
    ori = np.atleast_2d(np.asarray(ori, dtype=float))
    if ori.shape[1] not in (2, 3):
        raise ValueError("ori (second input) must consist of an array with two or three Euler angles on each row.")
    integrate_chi = ori.shape[1] == 2
    if isinstance(k, str):
        p, t = vec2ang(letter2vec(k))
        k_ori = (p[0], t[0])
    else:
        k = np.asarray(k, dtype=float).ravel()
        if k.size == 3:
            p, t = vec2ang(k)
            k_ori = (p[0], t[0])
        elif k.size == 2:
            k_ori = (k[0], k[1])
        else:
            raise ValueError(
                "k (third input) must be a letter designating a direction, a 3-vector, or an array with two angles."
            )
    unpolarized = np.isnan(alpha)
    R = erot([k_ori[0], k_ori[1], 0.0 if unpolarized else alpha])
    p_lab, k_lab = R[0], R[2]
    weight = np.zeros(ori.shape[0])
    for i, row in enumerate(ori):
        if integrate_chi:
            tdm_lab = erot([row[0], row[1], 0.0]) @ tdm_mol
            if unpolarized:
                weight_k = (tdm_lab[0] ** 2 + tdm_lab[1] ** 2) * (k_lab[0] ** 2 + k_lab[1] ** 2) / 2 + tdm_lab[
                    2
                ] ** 2 * k_lab[2] ** 2
                weight[i] = (1 - weight_k) / 2
            else:
                weight[i] = (tdm_lab[0] ** 2 + tdm_lab[1] ** 2) * (p_lab[0] ** 2 + p_lab[1] ** 2) / 2 + tdm_lab[
                    2
                ] ** 2 * p_lab[2] ** 2
        else:
            tdm_lab = erot(row) @ tdm_mol
            weight[i] = (1 - abs(tdm_lab @ k_lab) ** 2) / 2 if unpolarized else abs(tdm_lab @ p_lab) ** 2
    return weight


def nucspinrmv(system: dict, rmvidx) -> dict:
    """Remove the nuclei with the (1-based) numbers ``rmvidx`` from an unprocessed spin system dict."""
    new = dict(system)
    rmv = [int(i) for i in np.atleast_1d(rmvidx)]
    if not rmv:
        return new
    if np.size(new.get("nn", [])) > 0 and np.any(np.asarray(new["nn"]) != 0):
        raise ValueError("nucspinrmv does not work if Sys.nn is given.")
    nucs = nucstring2list(new.get("Nucs", ""))
    n_nuc = len(nucs)
    if any(i > n_nuc or i <= 0 for i in rmv):
        raise ValueError(f"There are only {n_nuc} nuclei in the spin system. Index out of range.")
    keep = np.array([i + 1 not in rmv for i in range(n_nuc)])
    kept = [n for n, k in zip(nucs, keep) if k]
    if kept:
        new["Nucs"] = ",".join(kept)
    else:
        new.pop("Nucs", None)

    def rows(name, full_flag):
        if name not in new:
            return
        v = (
            np.atleast_2d(np.asarray(new[name], dtype=float))
            if np.ndim(new[name]) < 2
            else np.asarray(new[name], dtype=float)
        )
        if full_flag:
            v = v[np.repeat(keep, 3)]
        else:
            v = v[keep] if v.shape[0] == n_nuc else v
        if v.size == 0:
            del new[name]
        else:
            new[name] = v

    full_sigma = new.get("fullsigma", "sigma" in new and np.shape(np.atleast_2d(new["sigma"]))[0] == 3 * n_nuc)
    rows("sigma", full_sigma)
    if full_sigma:
        new.pop("sigmaFrame", None)
    else:
        rows("sigmaFrame", False)
    for name in ("n", "gnscale"):
        if name in new:
            v = np.asarray(new[name], dtype=float).ravel()[keep]
            if v.size == 0:
                del new[name]
            else:
                new[name] = v
    full_a = new.get("fullA", "A" in new and np.shape(np.atleast_2d(new["A"]))[0] == 3 * n_nuc)
    for name in ("A", "AFrame", "A_"):
        rows(name, full_a and name == "A")
    if full_a:
        new.pop("AFrame", None)
    full_q = new.get("fullQ", "Q" in new and np.shape(np.atleast_2d(new["Q"]))[0] == 3 * n_nuc)
    rows("Q", full_q)
    if full_q:
        new.pop("QFrame", None)
    else:
        rows("QFrame", False)
    return new


# ----------------------------------------------------------------------------- Hamiltonian data
def gethamdata(B: float, H0: np.ndarray, muzL: np.ndarray, u: np.ndarray, v: np.ndarray, n_levels: int):
    """Eigenpairs of ``H0 - B muzL`` (sorted), ``dE/dB`` and the transition energies ``E[v] - E[u]``."""
    E, V = np.linalg.eigh(H0 - B * muzL)
    if n_levels < E.size:
        E, V = E[:n_levels], V[:, :n_levels]
    if B == 0:
        dE = np.abs(np.diff(E))
        tol = 1e3 * np.finfo(float).eps * (dE.max() if dE.size else 0.0)
        blk = np.cumsum(np.concatenate([[1], (dE > tol).astype(int)]))
        GG = V.conj().T @ (-muzL) @ V
        GG = (GG + GG.conj().T) / 2
        VV = np.zeros_like(GG)
        for k in range(1, blk.max() + 1):
            ix = np.nonzero(blk == k)[0]
            _, vk = np.linalg.eigh(GG[np.ix_(ix, ix)])
            VV[np.ix_(ix, ix)] = vk
        V = V @ VV
    dEdB = np.real(np.einsum("ij,ij->j", V.conj(), (-muzL) @ V))
    return V, E, dEdB, E[v] - E[u]


def getdstrainops(sys: dict) -> tuple[bool, list, list]:
    """Hamiltonian derivatives with respect to D and E, premultiplied with the D strain widths."""
    use = bool(np.any(sys["DStrain"]))
    dHdD: list[Any] = [0] * sys["nElectrons"]
    dHdE: list[Any] = [0] * sys["nElectrons"]
    if not use:
        return use, dHdD, dHdE
    spins = sys["Spins"]
    for iEl in range(sys["nElectrons"]):
        if not np.any(sys["DStrain"][iEl, :2]):
            continue
        Sx, Sy, Sz = (sop(spins, np.array([[iEl + 1, c]])) for c in (1, 2, 3))
        if np.any(sys["DFrame"][iEl]):
            R = erot(sys["DFrame"][iEl])
            SxD2 = np.linalg.matrix_power(R[0, 0] * Sx + R[0, 1] * Sy + R[0, 2] * Sz, 2)
            SyD2 = np.linalg.matrix_power(R[1, 0] * Sx + R[1, 1] * Sy + R[1, 2] * Sz, 2)
            SzD2 = np.linalg.matrix_power(R[2, 0] * Sx + R[2, 1] * Sy + R[2, 2] * Sz, 2)
        else:
            SxD2, SyD2, SzD2 = Sx @ Sx, Sy @ Sy, Sz @ Sz
        dHdD_ = (2 * SzD2 - SxD2 - SyD2) / 3
        dHdE_ = SxD2 - SyD2
        deltaD, deltaE = sys["DStrain"][iEl, 0], sys["DStrain"][iEl, 1]
        rDE = sys["DStrainCorr"][iEl]
        if rDE != 0:
            R12 = rDE * deltaD * deltaE
            L, Vc = np.linalg.eigh(np.array([[deltaD**2, R12], [R12, deltaE**2]]))
            L = np.sqrt(L)
            dHdD[iEl] = L[0] * (Vc[0, 0] * dHdD_ + Vc[0, 1] * dHdE_)
            dHdE[iEl] = L[1] * (Vc[1, 0] * dHdD_ + Vc[1, 1] * dHdE_)
        else:
            dHdD[iEl] = deltaD * dHdD_
            dHdE[iEl] = deltaE * dHdE_
    return use, dHdD, dHdE


# ----------------------------------------------------------------------------- cubic roots
_EPS_NEWTON = 1e-11
_MAX_NEWTON = 11


def cubicsolve(C: np.ndarray, loop_fields: bool) -> tuple[np.ndarray, np.ndarray]:
    """Roots in [0, 1] of the cubics ``C[:, 0] t^3 + ... + C[:, 3]`` (rows), as EasySpin's MEX.

    Returns ``(roots, counts)`` with the roots NaN-padded to (N, 3) and sorted ascending. With
    ``loop_fields`` false a Newton search is tried first on monotonic cubics."""
    a, b, c, d = (np.asarray(C[:, k], float) for k in range(4))
    N = a.size
    with np.errstate(all="ignore"):
        mono = (c * (3 * a + 2 * b + c) > 0) & (c * a < 0) if not loop_fields else np.zeros(N, bool)
        same = (a + b + c + d) * d > 0
        yex = d + c * (2 * a + b) / (3 * a + 2 * b)
        noroot = np.where(mono, same, same & (yex * d > 0))
        active = ~noroot
        roots = np.full((N, 3), np.nan)
        count = np.zeros(N, np.int64)
        zero_d = active & (d == 0)
        roots[zero_d, 0] = 0.0
        count[zero_d] = 1
        active &= ~zero_d
        found = np.zeros(N, bool)
        running = active & mono
        t = np.where(running, -d / (a + b + c), 0.0)
        for _ in range(_MAX_NEWTON):
            if not running.any():
                break
            f = ((a * t + b) * t + c) * t + d
            fp = (3 * a * t + 2 * b) * t + c
            dt = np.where(running, f / fp, 0.0)
            t = t - dt
            out = running & ((t < 0) | (t > 1))
            conv = running & ~out & (np.abs(dt) < _EPS_NEWTON)
            found |= conv
            running &= ~(out | conv)
        roots[found, 0] = t[found]
        count[found] = 1
        ex = active & ~found
        cand = np.full((N, 3), np.nan)
        cub = ex & (a != 0)
        aa, bb, cc = b / a, c / a, d / a
        Q = (aa * aa - 3 * bb) / 9
        R = (2 * aa**3 - 9 * aa * bb + 27 * cc) / 54
        QQQ = Q**3
        three = cub & (R * R < QQQ)
        theta = np.arccos(np.clip(R / np.sqrt(QQQ), -1, 1))
        Q2 = -2 * np.sqrt(Q)
        cand[three, 0] = (Q2 * np.cos(theta / 3) - aa / 3)[three]
        cand[three, 1] = (Q2 * np.cos((theta + 2 * np.pi) / 3) - aa / 3)[three]
        cand[three, 2] = (Q2 * np.cos((theta - 2 * np.pi) / 3) - aa / 3)[three]
        one = cub & ~three
        A = -np.power(np.abs(R) + np.sqrt(R * R - QQQ), 1 / 3.0)
        A = np.where(R < 0, -A, A)
        Bq = np.where(A == 0, 0.0, Q / A)
        cand[one, 0] = (A + Bq - aa / 3)[one]
        quad = ex & (a == 0) & (b != 0)
        disc = c * c - 4 * b * d
        qok = quad & (disc >= 0)
        Qq = -(c + np.where(c < 0, -1.0, 1.0) * np.sqrt(disc)) / 2
        cand[qok, 0] = (Qq / b)[qok]
        cand[qok, 1] = (d / Qq)[qok]
        lin = ex & (a == 0) & (b == 0) & (c != 0)
        cand[lin, 0] = (-d / c)[lin]
        if np.any(ex & (a == 0) & (b == 0) & (c == 0) & (d == 0)):
            raise ValueError("All coefficients of the cubic spline are zero!")
        valid = (cand >= 0) & (cand <= 1)
        cand[~valid] = np.nan
        packed = np.sort(cand, axis=1)
        roots[ex] = packed[ex]
        count[ex] = valid.sum(axis=1)[ex]
    return roots, count


# ----------------------------------------------------------------------------- resfields
_SPLINE_M = np.array([[2, -2, 1, 1], [-3, 3, -2, -1], [0, 0, 1, 0], [1, 0, 0, 0]], dtype=float)

_DEFAULT_EXP = {
    "mwFreq": np.nan, "Range": np.nan, "CenterSweep": np.nan, "Temperature": np.nan, "mwMode": "perpendicular",
    "SampleFrame": [0, 0, 0], "CrystalSymmetry": 1, "MolFrame": [0, 0, 0], "SampleRotation": None,
}  # fmt: skip
_DEFAULT_OPT = {
    "Transitions": None, "Threshold": 1e-4, "Hybrid": 0, "HybridCoreNuclei": [], "FuzzLevel": 1e-10,
    "TPSGridSize": 4, "TPSGridSymm": "D2h", "Freq2Field": True, "maxSegments": 2000, "ModellingAccuracy": 2e-6,
    "RediagLimit": 0.95, "Sparse": False, "nLevels": None, "Gradient": 1, "Intensity": 1,
    "HybridHFIThreshold": 0.02, "HybridIntThreshold": 0.005, "HybridOnlyHFI": 0, "Verbosity": 0, "Sites": None,
    "Method": None,
}  # fmt: skip


def _prepare_exp(exp: dict) -> tuple[dict, float]:
    exp = adddefaults(exp, _DEFAULT_EXP)
    if "CrystalOrientation" in exp:
        raise ValueError("Exp.CrystalOrientation is no longer supported, use Exp.SampleFrame/Exp.MolFrame instead.")
    if "Mode" in exp:
        raise ValueError("Exp.Mode is no longer supported. Use Exp.mwMode instead.")
    if exp["mwFreq"] is None or np.isnan(exp["mwFreq"]):
        raise ValueError("Exp.mwFreq is missing!")
    if exp["CenterSweep"] is not None and not np.any(np.isnan(np.atleast_1d(exp["CenterSweep"]))):
        cs = np.asarray(exp["CenterSweep"], dtype=float).ravel()
        exp["Range"] = np.array([cs[0] - cs[1] / 2, cs[0] + cs[1] / 2])
        if exp["Range"][0] < 0:
            raise ValueError("Lower field limit from Exp.CenterSweep cannt be negative.")
    if exp["Range"] is None or np.any(np.isnan(np.atleast_1d(exp["Range"]))):
        raise ValueError("Exp.Range/Exp.CenterSweep is missing!")
    rng = np.asarray(exp["Range"], dtype=float).ravel()
    if rng.size != 2 or rng[1] <= rng[0] or not np.all(np.isfinite(rng)):
        raise ValueError("Exp.Range is not valid!")
    if np.any(rng < 0):
        raise ValueError("Negative magnetic fields in Exp.Range are not possible.")
    exp["Range"] = rng
    return exp, float(exp["mwFreq"]) * 1e3


def _init_state(sys: dict, n_core: int):
    """EasySpin's handling of Sys.initState: a density matrix in the uncoupled basis, or populations."""
    spec = sys.get("initState")
    if spec is None:
        return None, None
    if isinstance(spec, (tuple, list)) and len(spec) == 2 and isinstance(spec[1], str):
        state, basis = np.asarray(spec[0], dtype=complex), spec[1]
    elif isinstance(spec, str):
        raise NotImplementedError(f"Sys.initState = '{spec}' shorthand is not ported yet.")
    else:
        state, basis = np.asarray(spec, dtype=complex), "uncoupled"
    if basis not in ("uncoupled", "eigen", "zerofield"):
        raise NotImplementedError(f"Sys.initState in the '{basis}' basis is not ported yet.")
    n_el_states = int(np.prod(2 * sys["S"] + 1))
    if state.ndim == 2 and state.shape[0] == state.shape[1]:
        if state.shape[0] not in (n_el_states, n_core):
            raise ValueError(
                f"The density matrix in Sys.initState must have dimensions of nxn with n = {n_el_states} or {n_core}."
            )
        if state.shape[0] == n_el_states and n_core > n_el_states:
            state = np.kron(state, np.eye(n_core // n_el_states)) / (n_core // n_el_states)
    else:
        state = state.ravel()
        if state.size not in (n_el_states, n_core):
            raise ValueError(f"The population vector in Sys.initState must have {n_el_states} or {n_core} elements.")
        if state.size == n_el_states and state.size != n_core:
            state = np.kron(state, np.ones(n_core // n_el_states)) / (n_core // n_el_states)
        state = np.diag(state)
    return state, basis


def resfields(system: dict, exp: dict, opt: dict | None = None) -> Resonances:
    """Resonance fields, intensities, strain widths and transitions of a cw EPR experiment (module docstring)."""
    exp_full = adddefaults(exp, _DEFAULT_EXP)
    if "lightBeam" in exp:
        exp_full["lightBeam"] = exp["lightBeam"]
    if "lightScatter" in exp:
        exp_full["lightScatter"] = exp["lightScatter"]
    opt = adddefaults(opt, _DEFAULT_OPT)
    angles_M2L, n_ori, n_sites, average_over_chi = _crystal_orientations(exp_full, opt)
    return resfields_oriented(system, exp, opt, angles_M2L, average_over_chi, n_sites)


def resfields_oriented(  # noqa: C901
    system: dict, exp: dict, opt: dict | None, angles_M2L: np.ndarray, average_over_chi: bool, n_sites: int = 1
) -> Resonances:
    """The resfields computation for given lab-frame orientations ``angles_M2L`` (rows ``[phi, theta, chi]``)."""
    sys = validatespinsys(system)
    sys = dict(sys)
    sys.setdefault("gAStrainCorr", 1)
    corr = sys["gAStrainCorr"]
    if not np.isscalar(corr) or corr == 0 or not np.isfinite(corr):
        raise ValueError("Sys.gAStrainCorr must be a single number, either +1 or -1.")
    sys["gAStrainCorr"] = np.sign(corr)
    if sys["nElectrons"] > 1 and np.any(sys["AStrain"]):
        raise ValueError("AStrain is not supported in spin systems with more than one electron spin.")
    if np.any(sys["gStrain"]) or np.any(sys["AStrain"]):
        if sys["fullg"]:
            raise ValueError("gStrain and AStrain are not supported when full g matrices are given!")
        if np.any(sys["gFrame"]):
            raise ValueError("g strain cannot be used with tilted g tensors.")
        if sys["nNuclei"] > 0 and np.any(sys["AFrame"][0]):
            raise ValueError("A strain cannot be used with a tilted A tensor.")
        if np.any(sys["DStrain"]):
            raise ValueError("D strain and g/A strain cannot be used at the same time.")

    exp_full, mw_freq = _prepare_exp(exp)
    exp_full["lightBeam"] = exp.get("lightBeam", "")
    exp_full["lightScatter"] = exp.get("lightScatter", 0)
    mode = _mw_mode(exp_full["mwMode"])
    opt = adddefaults(opt, _DEFAULT_OPT)
    if opt.get("Sparse"):
        raise NotImplementedError("Opt.Sparse is not ported; dense matrices are used.")
    if opt["Method"] is not None:
        opt["Hybrid"] = opt["Method"] == "hybrid"
    threshold = np.atleast_1d(np.asarray(opt["Threshold"], dtype=float))
    if threshold.size > 2 or np.any(threshold < 0) or np.any(threshold >= 1):
        raise ValueError("Options.Threshold must be a number >=0 and <1.")
    pre_threshold = threshold[0]
    post_threshold = threshold[1] if threshold.size == 2 else pre_threshold
    compute_freq2field = bool(opt["Freq2Field"])
    n_orientations = angles_M2L.shape[0]

    use_photoselection = bool(exp_full["lightBeam"]) and exp_full["lightScatter"] < 1
    if use_photoselection:
        if sys.get("tdm") is None:
            raise ValueError("To include photoselection weights, Sys.tdm must be given.")
        beam = exp_full["lightBeam"]
        if isinstance(beam, str):
            alpha = {"perpendicular": -np.pi / 2, "parallel": np.pi, "unpolarized": np.nan}.get(beam)
            if alpha is None:
                raise ValueError(
                    "Unknown string in Exp.lightBeam. Use '', 'perpendicular', 'parallel' or 'unpolarized'."
                )
            beam = (np.array([0.0, 1.0, 0.0]), alpha)
        elif not (isinstance(beam, (list, tuple)) and len(beam) == 2):
            raise ValueError("Exp.lightBeam should be a 2-element list [k, alpha].")

    strains_present = bool(
        np.any(sys["HStrain"]) or np.any(sys["DStrain"]) or np.any(sys["gStrain"]) or np.any(sys["AStrain"])
    )
    compute_strains = strains_present
    compute_intensities = bool(opt["Intensity"])

    # Core system and perturbational nuclei
    core = sys
    hfi_strength = np.zeros(0)
    if sys["nNuclei"] > 0:
        if sys["fullA"]:
            flat = sys["A"].ravel(order="F")
            max_hf = np.array([flat[3 * k : 3 * k + 3].max() for k in range(sys["nNuclei"])])
        else:
            max_hf = np.abs(sys["A"]).max(axis=1)
        hfi_strength = max_hf * (0.5 + sys["I"]) / mw_freq
    n_perturb = 0
    perturb_idx: list[int] = []
    if sys["nNuclei"] >= 1 and opt["Hybrid"]:
        core_nuclei = (
            [int(i) for i in np.atleast_1d(opt["HybridCoreNuclei"])] if opt["HybridCoreNuclei"] is not None else []
        )
        if any(i > sys["nNuclei"] for i in core_nuclei):
            raise ValueError("Opt.HybridCoreNuclei is incorrect!")
        perturb_idx = [i + 1 for i in range(sys["nNuclei"]) if i + 1 not in core_nuclei]
        n_perturb = len(perturb_idx)
        core_in = nucspinrmv({k: v for k, v in system.items() if k != "lwpp"}, perturb_idx)
        core = validatespinsys(core_in)
        Hhfi: dict[tuple[int, int], tuple] = {}
        Hzeem: list[tuple] = []
        Hquad: list[Any] = []
        n_perturb_transitions = []
        for ii, iNuc in enumerate(perturb_idx):
            I = sys["I"][iNuc - 1]
            Ix, Iy, Iz = sop(I, "x", "y", "z")
            n_perturb_transitions.append(int(round(2 * I + 1)) ** 2)
            for iEl in range(sys["nElectrons"]):
                idxE = slice(3 * iEl, 3 * iEl + 3)
                if sys["fullA"]:
                    A = sys["A"][3 * (iNuc - 1) : 3 * iNuc, idxE]
                else:
                    R_A2M = erot(sys["AFrame"][iNuc - 1, idxE]).T
                    A = R_A2M @ np.diag(sys["A"][iNuc - 1, idxE]) @ R_A2M.T
                Hhfi[(iEl, ii)] = tuple(A[r, 0] * Ix + A[r, 1] * Iy + A[r, 2] * Iz for r in range(3))
            if not opt["HybridOnlyHFI"]:
                pre = -nmagn / planck / 1e9 * sys["gn"][iNuc - 1]
                Hzeem.append((pre * Ix, pre * Iy, pre * Iz))
                Hq = np.zeros_like(Ix)
                if I >= 1:
                    R_Q2M = erot(sys["QFrame"][iNuc - 1]).T
                    Q = R_Q2M @ np.diag(sys["Q"][iNuc - 1]) @ R_Q2M.T
                    Ivec = (Ix, Iy, Iz)
                    for c1 in range(3):
                        for c2 in range(3):
                            Hq = Hq + Q[c1, c2] * Ivec[c1] @ Ivec[c2]
                Hquad.append(Hq)
        S_ops = [
            tuple(sop(core["Spins"], np.array([[iEl + 1, c]])) for c in (1, 2, 3)) for iEl in range(sys["nElectrons"])
        ]

    from .hamiltonian import ham

    kH0, kmuxM, kmuyM, kmuzM = ham(core)
    n_core = kH0.shape[0]
    n_levels = n_core

    # Populations
    init_state, init_basis = _init_state(sys, n_core)
    compute_nonequi = init_state is not None
    temperature = exp_full["Temperature"]
    if compute_nonequi:
        compute_boltzmann = False
    elif temperature is None:
        compute_boltzmann = False
    else:
        if np.size(temperature) != 1:
            raise ValueError("If given, Exp.Temperature must be a single number.")
        if np.isinf(temperature):
            raise ValueError("If given, Exp.Temperature must be a finite value.")
        compute_boltzmann = not np.isnan(temperature)
    if compute_nonequi:
        opt["FuzzLevel"] = 0 if "FuzzLevel" not in (opt or {}) else opt["FuzzLevel"]

    if opt["FuzzLevel"] > 0 and (core["nNuclei"] > 1 or core["nElectrons"] > 1):
        noise = 1 + opt["FuzzLevel"] * (2 * np.random.default_rng().random(kH0.shape) - 1)
        noise = (noise + noise.T) / 2
        kH0, kmuxM, kmuyM, kmuzM = kH0 * noise, kmuxM * noise, kmuyM * noise, kmuzM * noise

    if compute_nonequi and init_basis == "zerofield":
        ZF_E, ZF_V = np.linalg.eigh(kH0)
        if np.unique(ZF_E).size != ZF_E.size:
            raise ValueError(
                "Degenerate energy levels detected at zero-field. This prevents unambiguous assignment of the "
                "provided sublevel populations to the zero-field states. Please provide the non-equilibrium "
                "state using the full density matrix."
            )
        init_state = ZF_V @ init_state @ ZF_V.conj().T
    else:
        ZF_E = np.linalg.eigvalsh(kH0)
    max_zfs = ZF_E[-1] - ZF_E[0]
    loop_fields = max_zfs / mw_freq > 1 - 1e-6

    # Transition pre-selection
    rng_B = exp_full["Range"]
    if opt["Transitions"] is not None and np.size(opt["Transitions"]) > 0:
        if isinstance(opt["Transitions"], str):
            if opt["Transitions"] != "all":
                raise ValueError("Options.Transitions must be 'all' or a nx2 array of enery level indices.")
            n_states_ = n_core if opt["nLevels"] is None else int(opt["nLevels"])
            transitions = np.array(list(combinations(range(1, n_states_ + 1), 2)), dtype=int)
            post_threshold = 0.0
        else:
            tr = np.atleast_2d(np.asarray(opt["Transitions"], dtype=int))
            if tr.shape[1] != 2:
                raise ValueError("Options.Transitions must be a nx2 array of energy level indices.")
            transitions = np.sort(tr, axis=1)
            transitions = transitions[~np.any(transitions > n_levels, axis=1)]
    else:
        if pre_threshold == 0 or (core["nElectrons"] == 1 and core["nNuclei"] == 0):
            max_rates = np.ones((n_core, n_core))
        else:
            if n_orientations > 1:
                grid = sphgrid(opt["TPSGridSymm"], opt["TPSGridSize"])
                phi, theta = grid.phi, grid.theta
            else:
                phi, theta = np.array([angles_M2L[0, 0]]), np.array([angles_M2L[0, 1]])
            ExM, EyM, EzM = -kmuxM, -kmuyM, -kmuzM
            st, ct, sp, cp = np.sin(theta), np.cos(theta), np.sin(phi), np.cos(phi)
            centerB = rng_B.mean()
            max_rates = np.zeros((n_core, n_core))
            for i in range(theta.size):
                kmuzL = st[i] * (cp[i] * kmuxM + sp[i] * kmuyM) + ct[i] * kmuzM
                _, Vecs = np.linalg.eigh(kH0 - centerB * kmuzL)
                ExyM = cp[i] * ExM + sp[i] * EyM
                if mode.parallel:
                    EzL = st[i] * ExyM + ct[i] * EzM
                    rate = np.abs(Vecs.conj().T @ EzL @ Vecs) ** 2
                else:
                    EyL = -sp[i] * ExM + cp[i] * EyM
                    ExL = ct[i] * ExyM - st[i] * EzM
                    rate = (np.abs(Vecs.conj().T @ ExL @ Vecs) ** 2 + np.abs(Vecs.conj().T @ EyL @ Vecs) ** 2) / 2
                max_rates = np.maximum(max_rates, rate)
        keep = np.triu(np.ones((n_core, n_core), bool), 1)
        if hfi_strength.size and hfi_strength.max() < 0.5 and pre_threshold > 0:
            n_el_states = int(np.prod(2 * core["S"] + 1))
            nuclear = np.kron(np.eye(n_el_states), np.ones((n_core // n_el_states, n_core // n_el_states))).astype(bool)
            keep &= ~nuclear
        u_all, v_all = np.nonzero(keep.T)  # column-major order of find(keepidx): (u, v) with v outer
        u_all, v_all = v_all, u_all
        rates = max_rates[u_all, v_all]
        order = np.argsort(-rates, kind="stable")
        thr = pre_threshold * rates.max()
        n_tr = int(np.sum(rates > thr))
        transitions = np.column_stack([u_all, v_all])[order[:n_tr]] + 1
    if transitions.shape[0] == 0:
        raise ValueError("No transitions selected! Decrease Opt.Threshold.")
    if np.any(transitions > n_core):
        raise ValueError("Level index in Options.Transitions is out of range.")
    u = transitions[:, 0] - 1
    v = transitions[:, 1] - 1
    n_transitions = u.size

    # Strain preparations
    use_g_strain = use_a_strain = False
    simple_g_strain = True
    if compute_strains:
        h_strain2 = core["HStrain"] ** 2
        use_d_strain, dHdD, dHdE = getdstrainops(core)
        use_g_strain = bool(np.any(core["gStrain"]))
        g_strain_matrix = []
        if use_g_strain:
            simple_g_strain = core["nElectrons"] == 1
            for iEl in range(core["nElectrons"]):
                m = np.diag(core["gStrain"][iEl] / core["g"][iEl]) * mw_freq
                if np.any(core["gFrame"][iEl]):
                    R = erot(core["gFrame"][iEl]).T
                    m = R @ m @ R.T
                g_strain_matrix.append(m)
            if not simple_g_strain:
                kS = [
                    tuple(sop(core["Spins"], np.array([[iEl + 1, c]])) for c in (1, 2, 3))
                    for iEl in range(core["nElectrons"])
                ]
        else:
            g_strain_matrix = [np.zeros((3, 3)) for _ in range(core["nElectrons"])]
        use_a_strain = core["nNuclei"] > 0 and bool(np.any(core["AStrain"]))
        if use_a_strain:
            a_strain = np.diag(core["AStrain"])
            R = erot(core["AFrame"][0, :3]).T
            a_strain = R @ a_strain @ R.T
            _, Vecs = np.linalg.eigh(kH0 - rng_B.mean() * kmuzM)
            Iz1 = sop(core["Spins"], np.array([[2, 3]]))
            mI = np.real(np.einsum("ij,ij->j", Vecs.conj(), Iz1 @ Vecs))
            mI_tr = (mI[u] + mI[v]) / 2
            gA2 = [
                (
                    np.repeat(g_strain_matrix[e][:, :, None], n_transitions, axis=2)
                    + sys["gAStrainCorr"] * mI_tr[None, None, :] * a_strain[:, :, None]
                )
                ** 2
                for e in range(sys["nElectrons"])
            ]
        else:
            gA2 = [
                np.repeat((g_strain_matrix[e] ** 2)[:, :, None], n_transitions, axis=2)
                for e in range(sys["nElectrons"])
            ]
    pol_full = np.prod(2 * sys["I"] + 1)

    # Data over orientations: per transition and orientation, lists of resonances in discovery order
    level_accuracy = mw_freq * opt["ModellingAccuracy"]
    max_slope = max(
        np.linalg.eigvalsh(-kmuxM).max(), np.linalg.eigvalsh(-kmuyM).max(), np.linalg.eigvalsh(-kmuzM).max()
    )
    records: list[list[list[dict]]] = [[[] for _ in range(n_orientations)] for _ in range(n_transitions)]
    if compute_boltzmann:
        boltzmann_pre = 1e6 * planck / boltzm / temperature

    for iOri in range(n_orientations):
        R = erot(angles_M2L[iOri])
        xLab_M, yLab_M, zLab_M = R[0], R[1], R[2]
        kmuzL = zLab_M[0] * kmuxM + zLab_M[1] * kmuyM + zLab_M[2] * kmuzM
        kmuxL = xLab_M[0] * kmuxM + xLab_M[1] * kmuyM + xLab_M[2] * kmuzM
        kmuyL = yLab_M[0] * kmuxM + yLab_M[1] * kmuyM + yLab_M[2] * kmuzM
        if use_g_strain and not simple_g_strain:
            kSzL = [
                zLab_M[0] * kS[e][0] + zLab_M[1] * kS[e][1] + zLab_M[2] * kS[e][2] for e in range(sys["nElectrons"])
            ]
        if compute_strains:
            linewidth_squared = float(h_strain2 @ zLab_M**2)
        if use_photoselection:
            ori = angles_M2L[iOri, :2] if average_over_chi else angles_M2L[iOri, :3]
            photo_weight = photoselect(sys["tdm"], ori, beam[0], beam[1])[0]
            photo_weight = (1 - exp_full["lightScatter"]) * photo_weight + exp_full["lightScatter"]
        else:
            photo_weight = 1.0

        # Spline model by iterative bisection
        knots = [rng_B[0], rng_B[1]]
        data = [gethamdata(B, kH0, kmuzL, u, v, n_levels) for B in knots]  # (V, E, dEdB, deltaE)
        converged = [False]
        n_segments = 1
        while not all(converged) and n_segments < opt["maxSegments"]:
            s = converged.index(False)
            dB = knots[s + 1] - knots[s]
            V0, E0, dEdB0, dE0 = data[s]
            V1, E1, dEdB1, dE1 = data[s + 1]
            if E1[-1] - E1[0] > mw_freq:
                if loop_fields:
                    possible = np.abs((dE0 + dE1) / 2 - mw_freq) <= max_slope * dB
                else:
                    possible = (dE0 - mw_freq) * (dE1 - mw_freq) <= 0
            else:
                possible = np.zeros(n_transitions, bool)
            if not np.any(possible):
                converged[s] = True
                continue
            newB = (knots[s] + knots[s + 1]) / 2
            new = gethamdata(newB, kH0, kmuzL, u, v, n_levels)
            include = np.zeros(n_core, bool)
            include[u[possible]] = True
            include[v[possible]] = True
            deviation = 2 * ((E0 + E1) / 2 + dB / 8 * (dEdB0 - dEdB1) - new[1])
            conv = bool(np.abs(deviation[include]).max() <= level_accuracy)
            knots.insert(s + 1, newB)
            data.insert(s + 1, new)
            converged[s : s + 1] = [conv, conv]
            n_segments += 1
        knots_arr = np.asarray(knots)
        dBs = np.diff(knots_arr)
        stability = np.array(
            [np.abs(np.sum(np.conj(data[s + 1][0]) * data[s][0], axis=0)) for s in range(n_segments)]
        ).T  # (nCore, nSeg)
        coeffs = [
            _SPLINE_M @ np.vstack([data[s][1], data[s + 1][1], dBs[s] * data[s][2], dBs[s] * data[s + 1][2]])
            for s in range(n_segments)
        ]

        for iTrans in range(n_transitions):
            for s in range(n_segments):
                dEpoly = coeffs[s][:, v[iTrans]] - coeffs[s][:, u[iTrans]]
                dEpoly = dEpoly.copy()
                dEpoly[3] -= mw_freq
                roots, count = cubicsolve(dEpoly[None, :], loop_fields)
                zeros = roots[0, : count[0]]
                if zeros.size == 0:
                    continue
                res_fields = knots_arr[s] + dBs[s] * zeros
                for iReson, B_res in enumerate(res_fields):
                    rec: dict[str, Any] = {"pos": B_res}
                    uu, vv = u[iTrans], v[iTrans]
                    energies = None
                    if stability[uu, s] < opt["RediagLimit"] or stability[vv, s] < opt["RediagLimit"]:
                        energies, Vectors_ = np.linalg.eigh(kH0 - B_res * kmuzL)
                        U, V = Vectors_[:, uu], Vectors_[:, vv]
                    else:
                        z = zeros[iReson]
                        Ua, Ub = data[s][0][:, uu], data[s + 1][0][:, uu]
                        idx = np.argmax(np.abs(Ua))
                        phase = Ua[idx] / Ub[idx]
                        U = Ua * (1 - z) + z * phase / abs(phase) * Ub
                        U = U / np.linalg.norm(U)
                        Va, Vb = data[s][0][:, vv], data[s + 1][0][:, vv]
                        idx = np.argmax(np.abs(Va))
                        phase = Va[idx] / Vb[idx]
                        V = Va * (1 - z) + z * phase / abs(phase) * Vb
                        V = V / np.linalg.norm(V)
                        if compute_boltzmann:
                            t = z
                            energies = np.array([t**3, t**2, t, 1]) @ coeffs[s]
                    if compute_freq2field:
                        dBdE = 1 / abs(np.real((V - U).conj() @ (-kmuzL) @ (V + U)))
                        if dBdE > 1e5:
                            raise ValueError(
                                f"1/g factor diverges because d(Ev-Eu)/dB is almost zero for transition "
                                f"between levels u={uu + 1} and v={vv + 1}."
                            )
                    else:
                        dBdE = 1.0
                    if compute_intensities:
                        mu_L = np.array([V.conj() @ kmuxL @ U, V.conj() @ kmuyL @ U, V.conj() @ kmuzL @ U])
                        norm2 = float(np.linalg.norm(mu_L) ** 2)
                        if average_over_chi:
                            if mode.linear:
                                rate = (
                                    (1 - mode.xi1**2) * norm2 + (3 * mode.xi1**2 - 1) * abs(mode.nB0 @ mu_L) ** 2
                                ) / 2
                            elif mode.unpolarized:
                                rate = (
                                    (1 + mode.xik**2) * norm2 + (1 - 3 * mode.xik**2) * abs(mode.nB0 @ mu_L) ** 2
                                ) / 4
                            else:
                                rate = (
                                    (1 + mode.xik**2) * norm2 + (1 - 3 * mode.xik**2) * abs(mode.nB0 @ mu_L) ** 2
                                ) / 2 - mode.circ_sense * mode.xik * (mode.nB0 @ np.cross(1j * mu_L, np.conj(mu_L)))
                        else:
                            if mode.linear:
                                rate = abs(mode.nB1 @ mu_L) ** 2
                            elif mode.unpolarized:
                                rate = (norm2 - abs(mode.nk @ mu_L) ** 2) / 2
                            else:
                                rate = (norm2 - abs(mode.nk @ mu_L) ** 2) - mode.circ_sense * (
                                    mode.nk @ np.cross(1j * mu_L, np.conj(mu_L))
                                )
                        rate = float(np.real(rate))
                        if abs(rate) < 1e-10:
                            rate = 0.0
                        if compute_boltzmann:
                            pops = np.exp(-boltzmann_pre * (energies - energies[0]))
                            pops = pops / pops.sum()
                            polarization = pops[uu] - pops[vv]
                            if polarization < 0:
                                raise ValueError(
                                    f"Negative thermal polarization for transition {uu + 1}<->{vv + 1}: "
                                    f"{polarization:f}"
                                )
                            if n_perturb > 0:
                                polarization /= pol_full
                        elif compute_nonequi:
                            if init_basis == "eigen":
                                pu, pv = init_state[uu, uu], init_state[vv, vv]
                            else:
                                pu, pv = U.conj() @ init_state @ U, V.conj() @ init_state @ V
                            polarization = float(np.real(pu - pv))
                            if n_perturb > 0:
                                polarization /= pol_full
                        else:
                            polarization = 1.0 / pol_full
                        rec["int"] = dBdE * rate * polarization * photo_weight
                    if compute_strains:

                        def m(Op, U=U, V=V):
                            return np.real((V - U).conj() @ Op @ (V + U))

                        lw2 = linewidth_squared
                        if use_d_strain:
                            for iEl in range(core["nElectrons"]):
                                if not np.isscalar(dHdD[iEl]):
                                    lw2 += abs(m(dHdD[iEl])) ** 2 + abs(m(dHdE[iEl])) ** 2
                        if use_g_strain or use_a_strain:
                            if simple_g_strain:
                                g2 = gA2[0][:, :, iTrans]
                            else:
                                g2 = sum(abs(m(kSzL[e])) * gA2[e][:, :, iTrans] for e in range(sys["nElectrons"]))
                            lw2 += float(zLab_M @ g2 @ zLab_M)
                        rec["wid"] = dBdE * np.sqrt(lw2)
                    if n_perturb > 0:
                        Su = np.array(
                            [[np.real(U.conj() @ S_ops[e][c] @ U) for e in range(sys["nElectrons"])] for c in range(3)]
                        )
                        Sv = np.array(
                            [[np.real(V.conj() @ S_ops[e][c] @ V) for e in range(sys["nElectrons"])] for c in range(3)]
                        )
                        shifts, amps = [], []
                        for ii in range(n_perturb):
                            Hu = sum(
                                Su[0, e] * Hhfi[(e, ii)][0] + Su[1, e] * Hhfi[(e, ii)][1] + Su[2, e] * Hhfi[(e, ii)][2]
                                for e in range(sys["nElectrons"])
                            )
                            Hv = sum(
                                Sv[0, e] * Hhfi[(e, ii)][0] + Sv[1, e] * Hhfi[(e, ii)][1] + Sv[2, e] * Hhfi[(e, ii)][2]
                                for e in range(sys["nElectrons"])
                            )
                            if not opt["HybridOnlyHFI"]:
                                Hc = Hquad[ii] + B_res * (
                                    zLab_M[0] * Hzeem[ii][0] + zLab_M[1] * Hzeem[ii][1] + zLab_M[2] * Hzeem[ii][2]
                                )
                                Hu, Hv = Hu + Hc, Hv + Hc
                            Hu, Hv = (Hu + Hu.conj().T) / 2, (Hv + Hv.conj().T) / 2
                            dEu, Vu = np.linalg.eigh(Hu)
                            dEv, Vv = np.linalg.eigh(Hv)
                            M = np.abs(Vu.conj().T @ Vv) ** 2  # the Mims matrix
                            n = dEu.size
                            vidx, uidx = np.meshgrid(
                                np.arange(n), np.arange(n), indexing="ij"
                            )  # column-major: v fastest
                            shifts.append(-dBdE * (dEv[vidx.ravel(order="F")] - dEu[uidx.ravel(order="F")]))
                            amps.append(M.ravel(order="F"))
                        rec["shifts"], rec["amps"] = shifts, amps
                    records[iTrans][iOri].append(rec)
                if not loop_fields:
                    break

    # Assemble arrays: rows per transition = max number of resonances over orientations
    rows_u, rows_v, pos_rows, int_rows, wid_rows, shift_rows, amp_rows = [], [], [], [], [], [], []
    for iTrans in range(n_transitions):
        n_rows = max(1, max(len(records[iTrans][o]) for o in range(n_orientations)))
        for k in range(n_rows):
            pos = np.full(n_orientations, np.nan)
            ints = np.full(n_orientations, np.nan)
            wids = np.full(n_orientations, np.nan)
            sh = [np.zeros((n_orientations, n_perturb_transitions[ii])) for ii in range(n_perturb)] if n_perturb else []
            am = [np.zeros((n_orientations, n_perturb_transitions[ii])) for ii in range(n_perturb)] if n_perturb else []
            for o in range(n_orientations):
                if k < len(records[iTrans][o]):
                    rec = records[iTrans][o][k]
                    pos[o] = rec["pos"]
                    ints[o] = rec.get("int", np.nan)
                    wids[o] = rec.get("wid", np.nan)
                    for ii in range(n_perturb):
                        sh[ii][o] = rec["shifts"][ii]
                        am[ii][o] = rec["amps"][ii]
            rows_u.append(u[iTrans] + 1)
            rows_v.append(v[iTrans] + 1)
            pos_rows.append(pos)
            int_rows.append(ints)
            wid_rows.append(wids)
            shift_rows.append(sh)
            amp_rows.append(am)
    Pdat = np.array(pos_rows)
    Idat = np.array(int_rows) if compute_intensities else None
    Wdat = np.array(wid_rows) if compute_strains else None
    Trans = np.column_stack([rows_u, rows_v])

    # Post-selection
    remove = np.zeros(Pdat.shape[0], bool)
    if compute_intensities:
        max_int = np.nanmax(np.abs(Idat), axis=1, initial=-np.inf)
        max_int = np.where(np.isfinite(max_int), max_int, -np.inf)
        abs_max = max_int.max()
        remove |= max_int < abs_max * post_threshold
    remove |= np.all(np.isnan(Pdat), axis=1)
    keep_rows = ~remove
    Pdat, Trans = Pdat[keep_rows], Trans[keep_rows]
    if Idat is not None:
        Idat = Idat[keep_rows]
    if Wdat is not None:
        Wdat = Wdat[keep_rows]
    shift_rows = [r for r, k in zip(shift_rows, keep_rows) if k]
    amp_rows = [r for r, k in zip(amp_rows, keep_rows) if k]
    if Pdat.shape[0] == 0:
        import warnings

        warnings.warn(
            f"No resonance fields at {exp_full['mwFreq']:g} GHz between {rng_B[0]:g} and {rng_B[1]:g} mT. "
            "Check field range and spectrometer frequency.",
            stacklevel=2,
        )
        return Resonances(Pdat, Idat, Wdat, Trans)

    # Perturbational nuclei: post-selection and combination
    if n_perturb > 0:
        n_rows = Pdat.shape[0]
        pP = [np.array([r[ii] for r in shift_rows]) for ii in range(n_perturb)]  # (nRows, nOri, nSub_ii)
        pI = [np.array([r[ii] for r in amp_rows]) for ii in range(n_perturb)]
        for ii in range(n_perturb):
            total = pI[ii].sum(axis=(0, 1))
            keep_sub = total >= total.max() * opt["HybridIntThreshold"]
            pP[ii], pI[ii] = pP[ii][:, :, keep_sub], pI[ii][:, :, keep_sub]
        P_comb, I_comb = pP[0], pI[0]
        for ii in range(1, n_perturb):
            n2, n1 = pP[ii].shape[2], P_comb.shape[2]
            idx2, idx1 = np.meshgrid(np.arange(n2), np.arange(n1), indexing="ij")
            idx2, idx1 = idx2.ravel(order="F"), idx1.ravel(order="F")
            P_comb = P_comb[:, :, idx1] + pP[ii][:, :, idx2]
            I_comb = I_comb[:, :, idx1] * pI[ii][:, :, idx2]
        n_sub = P_comb.shape[2]

        def rearrange(x):
            return np.transpose(x, (0, 2, 1)).reshape(n_rows * n_sub, n_orientations, order="F")

        expand = lambda x: np.repeat(x[:, :, None], n_sub, axis=2)  # noqa: E731
        Pdat = rearrange(expand(Pdat) + P_comb)
        Idat = rearrange(expand(Idat) * I_comb)
        if Wdat is not None:
            Wdat = rearrange(expand(Wdat))
        Trans = np.transpose(np.repeat(Trans[:, :, None], n_sub, axis=2), (0, 2, 1)).reshape(
            n_rows * n_sub, 2, order="F"
        )

    # Sort transitions lexicographically (stable, as MATLAB's sortrows)
    order = np.lexsort((Trans[:, 1], Trans[:, 0]))
    Trans, Pdat = Trans[order], Pdat[order]
    if Idat is not None:
        Idat = Idat[order]
    if Wdat is not None:
        Wdat = Wdat[order]
    return Resonances(Pdat, Idat, Wdat, Trans)


# ----------------------------------------------------------------------------- eigenfield method
def resfields_eig(system: dict, exp: dict, opt: dict | None = None):
    """Resonance fields by the eigenfield method (Belford et al., J. Magn. Reson. 11, 251 (1973)): all fields at which a
    transition of energy ``mwFreq`` exists, without a field sweep. Returns the fields (mT) per orientation (one array
    for a single orientation, a list otherwise); with ``opt["Intensities"]`` true also the intensities."""
    sys = validatespinsys(system)
    if "Detection" in exp:
        raise ValueError("Exp.Detection is obsolete. Use Exp.mwMode instead.")
    defaults = {
        "mwFreq": np.nan, "Range": [0, np.finfo(float).max], "mwMode": "perpendicular", "Temperature": np.nan,
        "SampleFrame": [0, 0, 0], "CrystalSymmetry": 1, "MolFrame": [0, 0, 0], "SampleRotation": None,
    }  # fmt: skip
    exp_full = adddefaults(exp, defaults)
    if np.isnan(exp_full["mwFreq"]):
        raise ValueError("Parameters.mwFreq missing!")
    rng_B = np.asarray(exp_full["Range"], dtype=float).ravel()
    if rng_B.size != 2 or rng_B[1] <= rng_B[0] or np.any(rng_B < 0) or not np.all(np.isfinite(rng_B)):
        raise ValueError("Parameters.Range is not valid!")
    mw_mode = exp_full["mwMode"] or "perpendicular"
    if mw_mode not in ("perpendicular", "parallel"):
        raise ValueError(f"Unknown value '{mw_mode}' for field mwMode.")
    parallel = mw_mode == "parallel"
    if exp_full["Temperature"] is not None and not np.isnan(exp_full["Temperature"]):
        raise ValueError("Thermal equilibrium populations (Exp.Temperature) not implemented.")
    if sys.get("initState") is not None:
        raise ValueError("Non-equilibrium populations (Sys.initState) not implemented.")
    mw_freq = exp_full["mwFreq"] * 1e3
    opt = adddefaults(
        opt,
        {
            "Freq2Field": True,
            "Threshold": 0,
            "RejectionRatio": 1e-8,
            "Intensities": False,
            "Sites": None,
            "Verbosity": 0,
        },
    )
    orientations, n_ori, _, average_over_chi = _crystal_orientations(exp_full, opt)
    compute_intensities = bool(opt["Intensities"]) or opt["Threshold"] != 0
    from .hamiltonian import ham

    H0, mux, muy, muz = ham(sys)
    n = H0.shape[0]
    A = np.kron(np.eye(n), H0) - np.kron(np.conj(H0), np.eye(n)) + mw_freq * np.eye(n * n)
    simple = bool(np.all(np.linalg.eigvals(A).real > 0))
    fields_out, ints_out = [], []
    for iOri in range(n_ori):
        R = erot(orientations[iOri])
        xLab, yLab, zLab = R[0], R[1], R[2]
        muzL = zLab[0] * mux + zLab[1] * muy + zLab[2] * muz
        B = -np.kron(np.conj(muzL), np.eye(n)) + np.kron(np.eye(n), muzL)
        if compute_intensities:
            if simple:
                BB = np.linalg.solve(A, B)
                w, Vecs = np.linalg.eig(BB)
                with np.errstate(divide="ignore", invalid="ignore"):
                    fields = 1 / w
            else:
                w, Vecs = sla.eig(A, B)
                fields = w
            order = np.argsort(fields.real)
            fields, Vecs = fields[order], Vecs[:, order]
            ok = (
                (np.abs(fields.imag) < opt["RejectionRatio"] * np.abs(fields.real))
                & (fields.real > 0)
                & np.isfinite(fields)
                & (fields.real > rng_B[0])
                & (fields.real < rng_B[1])
            )
            if not np.any(ok):
                fields_out.append(np.zeros(0))
                ints_out.append(np.zeros(0))
                continue
            fields_k = fields.real[ok]
            Vecs = Vecs[:, ok]
            Vecs = Vecs / np.sqrt(np.sum(np.abs(Vecs) ** 2, axis=0))
            # EasySpin's reshape(mux.',[],1) is the row-major vectorisation; with vec(|u><v|) it gives trace(mu |u><v|)
            mux_vec, muy_vec, muz_vec = mux.ravel(), muy.ravel(), muz.ravel()
            if parallel:
                muzL_vec = zLab[0] * mux_vec + zLab[1] * muy_vec + zLab[2] * muz_vec
                rate = np.abs(muzL_vec @ Vecs) ** 2
            else:
                muxL_vec = xLab[0] * mux_vec + xLab[1] * muy_vec + xLab[2] * muz_vec
                if average_over_chi:
                    muyL_vec = yLab[0] * mux_vec + yLab[1] * muy_vec + yLab[2] * muz_vec
                    rate = (np.abs(muxL_vec @ Vecs) ** 2 + np.abs(muyL_vec @ Vecs) ** 2) / 2
                else:
                    rate = np.abs(muxL_vec @ Vecs) ** 2
            polarization = 1 / np.prod(2 * sys["I"] + 1)
            if opt["Freq2Field"]:
                dBdE = np.empty(fields_k.size)
                for k in range(fields_k.size):
                    Vm = Vecs[:, k].reshape(n, n, order="F")
                    dBdE[k] = 1 / abs(np.trace(-muzL @ (Vm @ Vm.conj().T - Vm.conj().T @ Vm)))
            else:
                dBdE = np.ones(fields_k.size)
            ints = polarization * np.real(rate * dBdE)
            sel = ints >= opt["Threshold"] * ints.max()
            fields_out.append(fields_k[sel])
            ints_out.append(ints[sel])
        else:
            if simple:
                with np.errstate(divide="ignore", invalid="ignore"):
                    fields = 1 / np.linalg.eigvals(np.linalg.solve(A, B))
            else:
                fields = sla.eigvals(A, B)
            fields = np.sort_complex(fields)
            ok = (
                (np.abs(fields.imag) < opt["RejectionRatio"] * np.abs(fields.real))
                & (fields.real > 0)
                & np.isfinite(fields)
                & (fields.real > rng_B[0])
                & (fields.real < rng_B[1])
            )
            fields_out.append(np.sort(fields.real[ok]))
    if n_ori == 1:
        fields_out, ints_out = fields_out[0], (ints_out[0] if ints_out else None)
    return (fields_out, ints_out) if compute_intensities else fields_out
