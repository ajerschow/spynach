"""Resonance fields by perturbation theory: EasySpin's ``resfields_perturb``.

Second-order perturbation treatment of the general spin Hamiltonian in an
arbitrary coordinate system (M. Iwasaki, J. Magn. Reson. 16, 417 (1974)) for
one electron spin and any number of nuclei: the field of every allowed
transition ``mS <-> mS-1`` with every nuclear sublevel combination, in closed
form, with the first-order (``PerturbOrder=1``) or second-order (``2``)
corrections. No diagonalisation, so it is the fast method of ``pepper`` for
systems with many nuclei; it requires non-zero hyperfine couplings and
supports H, g, A and D strains (D strain for integer spins).

EasySpin authors, by original file (contributors to the file up to EasySpin 6.0.15, see
  docs/easyspin-credits.md):
  resfields_perturb.m: Stefan Stoll, Claudia Tait, Joscha Nehrkorn, AlaCodon (GitHub user), Matthew Krzyaniak
"""

from __future__ import annotations

from itertools import product

import numpy as np

from .constants import bmagn, boltzm, planck
from .resfields import Resonances, _crystal_orientations, _mw_mode, _prepare_exp, adddefaults, photoselect
from .rotations import erot
from .spinsystem import validatespinsys

_DEFAULT_EXP = {
    "mwFreq": np.nan, "Range": np.nan, "CenterSweep": np.nan, "Temperature": np.nan, "mwMode": "",
    "SampleFrame": [0, 0, 0], "CrystalSymmetry": 1, "MolFrame": [0, 0, 0], "SampleRotation": None,
}  # fmt: skip


def resfields_perturb(system: dict, exp: dict, opt: dict | None = None) -> Resonances:
    """Resonance fields, intensities, widths and transitions by perturbation theory for the orientations of
    ``Exp.SampleFrame`` (crystal symmetry P1); ``opt["PerturbOrder"]`` is 1 or 2 (default).

    EasySpin authors (resfields_perturb.m): Stefan Stoll, Claudia Tait, Joscha Nehrkorn, AlaCodon (GitHub
      user), Matthew Krzyaniak.
    """
    exp_full = adddefaults(exp, _DEFAULT_EXP)
    opt = dict(opt or {})
    angles, n_ori, n_sites, average = _crystal_orientations(exp_full, opt)
    return resfields_perturb_oriented(system, exp, opt, angles, average)


def resfields_perturb_oriented(  # noqa: C901
    system: dict, exp: dict, opt: dict | None, orientations: np.ndarray, average_over_chi: bool
) -> Resonances:
    """The perturbation-theory computation for given lab-frame orientations (rows ``[phi, theta, chi]``).

    EasySpin authors (resfields_perturb.m, the per-orientation part): Stefan Stoll, Claudia Tait, Joscha
      Nehrkorn, AlaCodon (GitHub user), Matthew Krzyaniak.
    """
    sys = validatespinsys(system)
    S = float(sys["S"][0])
    high_spin = S > 0.5
    if sys["nElectrons"] != 1:
        raise ValueError(
            f"Perturbation theory available only for systems with 1 electron. Yours has {sys['nElectrons']}."
        )
    if high_spin and np.any(sys["DStrain"]) and S % 1 != 0:
        raise ValueError(
            "D strain not supported for half-integer spins with perturbation theory. "
            "Use matrix diagonalization or remove Sys.DStrain."
        )
    if np.any(sys["DStrain"]) and np.any(sys["DFrame"]):
        raise ValueError("D stain cannot be used with tilted D tensors.")
    if sys["nNuclei"] > 1 and np.any(sys["nn"] != 0):
        raise ValueError("Perturbation theory not available for nuclear-nuclear couplings (Sys.nn).")
    corr = np.sign(np.atleast_1d(sys.get("gAStrainCorr", 1))[0]) if sys.get("gAStrainCorr") is not None else 1.0
    if sys["fullg"]:
        g = sys["g"][:3]
    else:
        R = erot(sys["gFrame"][0]).T
        g = R @ np.diag(sys["g"][0]) @ R.T
    if high_spin:
        if sys["fullD"]:
            D = sys["D"][:3]
        else:
            R = erot(sys["DFrame"][0]).T
            D = R @ np.diag(sys["D"][0]) @ R.T
        D = D - np.eye(3) * np.trace(D) / 3
    n_trans = int(round(2 * S))
    I = sys["I"]
    n_nuc = sys["nNuclei"]
    n_nuc_states = (2 * I + 1).astype(int) if n_nuc > 0 else np.array([1])
    if n_nuc > 0 and not sys["fullA"] and np.any(sys["A"] == 0):
        raise ValueError("All hyperfine coupling constants must be non-zero.")
    A, mI_lists = [], []
    for k in range(n_nuc):
        if sys["fullA"]:
            A.append(sys["A"][3 * k : 3 * k + 3, :3])
        else:
            R = erot(sys["AFrame"][k, :3]).T
            A.append(R @ np.diag(sys["A"][k, :3]) @ R.T)
        mI_lists.append(np.arange(-I[k], I[k] + 0.5))

    exp_full, mw_freq = _prepare_exp({**exp, "mwMode": exp.get("mwMode", "")})
    if "SearchRange" in exp:
        exp_full["Range"] = np.asarray(exp["SearchRange"], dtype=float)
    mode = _mw_mode(exp_full["mwMode"] or "perpendicular")
    temperature = exp_full["Temperature"]
    if temperature is not None and np.size(temperature) > 1:
        raise ValueError("Exp.Temperature must be a single number.")
    if temperature is not None and np.isinf(temperature):
        raise ValueError("If given, Exp.Temperature must have a finite value.")
    light_beam = exp.get("lightBeam", "")
    light_scatter = exp.get("lightScatter", 0)
    use_photo = bool(light_beam) and light_scatter < 1
    if use_photo:
        if sys.get("tdm") is None:
            raise ValueError("To include photoselection weights, Sys.tdm must be given.")
        if isinstance(light_beam, str):
            alpha = {"perpendicular": -np.pi / 2, "parallel": np.pi, "unpolarized": np.nan}.get(light_beam)
            if alpha is None:
                raise ValueError(
                    "Unknown string in Exp.lightBeam. Use '', 'perpendicular', 'parallel' or 'unpolarized'."
                )
            beam = (np.array([0.0, 1.0, 0.0]), alpha)
        else:
            beam = light_beam
    opt = dict(opt or {})
    order = int(opt.get("PerturbOrder", 2))
    if order not in (1, 2):
        raise ValueError("Opt.PerturbOrder must be either 1 or 2.")
    if opt.get("ImmediateBinning"):
        raise NotImplementedError("Opt.ImmediateBinning is not ported.")
    second = order == 2
    E0 = mw_freq
    detA, invA, trAA = [], [], []
    for k in range(n_nuc):
        detA.append(np.linalg.det(A[k]))
        invA.append(np.linalg.inv(A[k]))
        trAA.append(np.trace(A[k].T @ A[k]))
    if high_spin:
        trDD = np.trace(D @ D)
    II1 = I * (I + 1)
    idxn = np.array(list(product(*[range(n) for n in n_nuc_states]))) if n_nuc > 0 else np.zeros((1, 0), int)
    n_nuc_trans = idxn.shape[0]
    if temperature is not None and not np.isnan(temperature):
        pops = np.exp(-planck * np.arange(2 * S, -0.5, -1) * mw_freq * 1e6 / boltzm / temperature)
        pops = pops / pops.sum()
        polarization = np.diff(pops)[::-1]
    else:
        polarization = np.ones(n_trans)
    gg = g @ g.T
    trgg = np.trace(gg)
    mS_list = np.arange(S, -S + 0.5, -1)
    c = bmagn / 2 * np.sqrt(S * (S + 1) - mS_list * (mS_list - 1)) / planck / 1e9
    c2 = c**2
    n_ori = orientations.shape[0]
    intensity = np.zeros((n_trans, n_ori))
    vecs = np.zeros((3, n_ori))
    geff = np.zeros(n_ori)
    B_final = np.zeros((n_trans, n_ori, n_nuc_trans))
    for iOri in range(n_ori):
        R_L2M = erot(orientations[iOri]).T
        n0 = R_L2M @ mode.nB0
        vecs[:, iOri] = n0
        geff[iOri] = np.linalg.norm(g.T @ n0)
        u = g.T @ n0 / geff[iOri]
        pre_ori = 1e6 * planck / (geff[iOri] * bmagn)
        if use_photo:
            ori = orientations[iOri, :2] if average_over_chi else orientations[iOri, :3]
            photo = photoselect(sys["tdm"], ori, beam[0], beam[1])[0]
            photo = (1 - light_scatter) * photo + light_scatter
        else:
            photo = 1.0
        gu2 = np.linalg.norm(g @ u) ** 2
        if average_over_chi:
            if mode.linear:
                rate = c2 / 2 * (1 - mode.xi1**2) * (trgg - gu2)
            elif mode.unpolarized:
                rate = c2 / 4 * (1 + mode.xik**2) * (trgg - gu2)
            else:
                rate = c2 / 2 * (1 + mode.xik**2) * (
                    trgg - gu2
                ) + mode.circ_sense * 2 * c2 * mode.xik**2 * np.linalg.det(g) / np.linalg.norm(g.T @ n0)
        else:
            if mode.linear:
                nB1 = R_L2M @ mode.nB1
                rate = c2 * np.linalg.norm(np.cross(g.T @ nB1, u)) ** 2
            elif mode.unpolarized:
                nk = R_L2M @ mode.nk
                rate = c2 / 2 * (trgg - gu2 - np.linalg.norm(np.cross(g.T @ nk, u)) ** 2)
            else:
                nk = R_L2M @ mode.nk
                rate = c2 * (
                    trgg - gu2 - np.linalg.norm(np.cross(g.T @ nk, u)) ** 2
                ) + mode.circ_sense * 2 * c2 * np.linalg.det(g) * mode.xik / np.linalg.norm(g.T @ n0)
        dBdE = (planck / bmagn * 1e9) / geff[iOri]
        intensity[:, iOri] = polarization * rate * dBdE * photo
        if high_spin:
            Du = D @ u
            uDu = u @ Du
            uDDu = Du @ Du
            D1sq = uDDu - uDu**2
            D2sq = 2 * trDD + uDu**2 - 4 * uDDu
        for imS, mS in enumerate(mS_list):
            E1D = -uDu / 2 * (3 - 6 * mS) if high_spin else 0.0
            E1A = np.zeros((n_nuc_trans, n_nuc))
            E2A = np.zeros((n_nuc_trans, n_nuc))
            E2DA = np.zeros((n_nuc_trans, n_nuc))
            kvecs = []
            for k in range(n_nuc):
                K = A[k] @ u
                nK = np.linalg.norm(K)
                kvecs.append(K / nK)
                E1A[:, k] = (mI_lists[k] * nK)[idxn[:, k]]
            E2D = 0.0
            if second:
                if high_spin:
                    x = D1sq * (4 * S * (S + 1) - 3 * (8 * mS**2 - 8 * mS + 3)) - D2sq / 4 * (
                        2 * S * (S + 1) - 3 * (2 * mS**2 - 2 * mS + 1)
                    )
                    E2D = -x / (2 * E0)
                for k in range(n_nuc):
                    k_ = kvecs[k]
                    Ak = A[k].T @ k_
                    kAu = Ak @ u
                    kAAk = np.linalg.norm(Ak) ** 2
                    A1sq = kAAk - kAu**2
                    A2 = detA[k] * (u @ invA[k] @ k_)
                    A3 = trAA[k] - np.linalg.norm(A[k] @ u) ** 2 - kAAk + kAu**2
                    mI = mI_lists[k]
                    x = A1sq * mI**2 - A2 * (1 - 2 * mS) * mI + A3 / 2 * (II1[k] - mI**2)
                    E2A[:, k] = (x / (2 * E0))[idxn[:, k]]
                    if high_spin:
                        DA = Du @ Ak - uDu * kAu
                        E2DA[:, k] = (-(DA * (3 - 6 * mS) * mI) / E0)[idxn[:, k]]
            if second:
                B_final[imS, iOri] = (E0 - E1D - E2D - np.sum(E1A + E2A + E2DA, axis=1)) * pre_ori
            else:
                B_final[imS, iOri] = (E0 - E1D - np.sum(E1A, axis=1)) * pre_ori
    # Positions: rows ordered mS block outer, nuclear combination inner; columns orientations
    B = B_final.transpose(0, 2, 1).reshape(n_trans * n_nuc_trans, n_ori) * 1e3
    n_sub = int(np.prod(n_nuc_states))
    Int = np.repeat(intensity, n_sub, axis=0) / n_sub
    Int = Int[::-1]  # EasySpin's flipud
    Wid = None
    if np.any(sys["HStrain"]):
        lw2 = sys["HStrain"] ** 2 @ vecs**2
        lw = np.sqrt(lw2) * 1e6 * planck / geff / bmagn * 1e3
        Wid = np.tile(lw, (n_nuc_trans * n_trans, 1))
    if np.any(sys["gStrain"]) or np.any(sys["AStrain"]):
        if np.any(sys["gFrame"]):
            raise ValueError("g strain cannot be used with tilted g tensors.")
        if n_nuc > 0 and np.any(sys["AFrame"][0]):
            raise ValueError("A strain cannot be used with a tilted A tensor.")
        g_strain = np.diag(sys["gStrain"][0] / sys["g"][0]) * mw_freq if np.any(sys["gStrain"]) else np.zeros((3, 3))
        base = Wid if Wid is not None else 0.0
        if np.any(sys["AStrain"]) and n_nuc > 0:
            Rp = erot(sys["AFrame"][0, :3]).T
            a_strain = Rp @ np.diag(sys["AStrain"]) @ Rp.T
            mI1 = np.arange(-I[0], I[0] + 0.5)
            lw2 = np.array(
                [[vecs[:, o] @ (g_strain + corr * m * a_strain) ** 2 @ vecs[:, o] for o in range(n_ori)] for m in mI1]
            )
            wid_gA = np.sqrt(lw2) * planck * 1e6 / geff / bmagn * 1e3
            idx = np.tile(np.arange(mI1.size), n_trans * n_nuc_trans // mI1.size)
            Wid = np.sqrt(wid_gA[idx] ** 2 + base**2)
        else:
            lw2 = np.array([vecs[:, o] @ g_strain**2 @ vecs[:, o] for o in range(n_ori)])
            wid_gA = np.sqrt(lw2) * planck * 1e6 / geff / bmagn * 1e3
            Wid = np.sqrt(np.tile(wid_gA**2, (n_trans * n_nuc_trans, 1)) + base**2)
    elif np.any(sys["DStrain"]):
        if np.any(sys["DFrame"]):
            raise ValueError("Cannot use D/E strain with tilted D tensor.")
        x, y, z = vecs
        mS_all = np.arange(S, -S - 0.5, -1)
        mSS = (mS_all**2 - S * (S + 1) / 3)[:, None]
        dHdD = mSS * ((3 * z**2 - 1) / 2)[None, :]
        dHdE = mSS * (3 / 2 * (x**2 - y**2))[None, :]
        deltaD, deltaE = sys["DStrain"][0, 0], sys["DStrain"][0, 1]
        rDE = sys["DStrainCorr"][0]
        if rDE != 0:
            R12 = rDE * deltaD * deltaE
            L, Vc = np.linalg.eigh(np.array([[deltaD**2, R12], [R12, deltaE**2]]))
            L = np.sqrt(L)
            dHdD, dHdE = L[0] * (Vc[0, 0] * dHdD + Vc[0, 1] * dHdE), L[1] * (Vc[1, 0] * dHdD + Vc[1, 1] * dHdE)
        else:
            dHdD, dHdE = dHdD * deltaD, dHdE * deltaE
        lwD = np.diff(dHdD, axis=0) * ((planck / bmagn * 1e9) / geff)[None, :]
        lwE = np.diff(dHdE, axis=0) * ((planck / bmagn * 1e9) / geff)[None, :]
        base = Wid if Wid is not None else 0.0
        Wid = np.sqrt(np.tile(lwD**2 + lwE**2, (n_nuc_trans, 1)) + base**2)
    lower = np.arange(1, n_sub + 1)
    trans = []
    for _ in range(n_trans):
        upper = lower + n_sub
        trans.append(np.column_stack([lower, upper]))
        lower = upper
    return Resonances(B, Int, Wid, np.vstack(trans))
