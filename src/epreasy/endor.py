"""ENDOR: EasySpin's endorfrq, endorfrq_perturb, orisel and salt.

``endorfrq`` gives the nuclear transition frequencies (MHz) and ENDOR
intensities of a spin system at a fixed field by matrix diagonalisation,
with EPR transition pre-selection, hyperfine enhancement, orientation
selection by the excitation width, Boltzmann or non-equilibrium
populations and pulse-ENDOR flip angles; ``endorfrq_perturb`` the
first- and second-order perturbation expressions of Iwasaki; ``orisel``
the orientation-selection weights of an EPR excitation; ``salt`` the ENDOR
spectrum of a powder or crystal with the interpolation, projection and
summation of ``pepper``.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from .components import compisoloop
from .constants import bmagn, boltzm, nmagn, planck
from .garlic import Spectrum
from .grids import sphgrid
from .hamiltonian import ham, ham_ez, ham_nz
from .interp import gridinterp
from .kernels import lisum1i, projecttriangles, projectzones
from .lineshapes import convspec, gaussian
from .nucdata import nucstring2list
from .resfields import _crystal_orientations, _init_state, adddefaults, nucspinrmv
from .rotations import ang2vec, erot, vec2ang
from .spinsystem import validatespinsys
from .symmetry import hamsymm

__all__ = ["EndorResonances", "anisosubsys", "endorfrq", "endorfrq_perturb", "orisel", "salt"]


class EndorResonances(NamedTuple):
    """ENDOR frequencies ``pos`` (MHz) and intensities ``int``, transitions-by-orientations; level pairs
    ``trans`` (1-based, empty for perturbation theory); ``info`` with the selectivity."""

    pos: np.ndarray
    int: np.ndarray | None
    trans: np.ndarray
    info: dict


_DEFAULT_EXP = {
    "mwFreq": np.nan, "Temperature": np.nan, "ExciteWidth": np.inf, "Field": np.nan, "SampleFrame": [0, 0, 0],
    "CrystalSymmetry": 1, "MolFrame": [0, 0, 0], "SampleRotation": None, "Range": None, "rf": 0,
}  # fmt: skip
_DEFAULT_OPT = {
    "Transitions": None, "Threshold": 1e-3, "Nuclei": None, "Intensity": "on", "Enhancement": "off",
    "OriWeights": None, "OriThreshold": 1e-4, "TPSGridSize": 4, "TPSGridSymm": "D2h", "Sites": None,
}  # fmt: skip


def _lab_orientations(exp_full: dict, opt: dict, orientations, average_over_chi: bool):
    """Orientations handed over by salt (grid, chi-averaged; or crystal), or the crystal orientations of Exp."""
    if orientations is not None:
        return np.asarray(orientations, dtype=float), average_over_chi
    angles, _, _, average_over_chi = _crystal_orientations(exp_full, opt)
    return angles, average_over_chi


# ----------------------------------------------------------------------------- endorfrq
def endorfrq(
    system: dict, exp: dict, opt: dict | None = None, *, _orientations=None, _average_over_chi: bool = True
) -> EndorResonances:  # noqa: C901
    """ENDOR frequencies (MHz) and intensities of ``system`` at ``exp["Field"]`` (mT) by matrix diagonalisation.

    ``exp``: ``Field``, ``mwFreq`` (GHz, needed with ``ExciteWidth``), ``ExciteWidth`` (MHz FWHM; orientation
    selection), ``Temperature`` (K), ``Range`` (MHz, drops transitions out of range), ``rf`` (pulse ENDOR flip
    angle), ``SampleFrame``, ``MolFrame``. ``opt``: ``Transitions``, ``Threshold``, ``Nuclei`` (1-based, which
    nuclei's Zeeman terms detect), ``Intensity``, ``Enhancement`` (``'on'`` adds the electron Zeeman term to the
    detection operator), ``OriWeights``/``OriThreshold`` (skip orientations).
    """
    sys = validatespinsys(system)
    if sys["nNuclei"] == 0:
        raise ValueError("The system doesn't contain any nuclei.")
    if "Field" not in exp or exp["Field"] is None:
        raise ValueError("Exp.Field is missing.")
    exp_full = adddefaults(exp, _DEFAULT_EXP)
    field = float(exp_full["Field"])
    mw_freq = float(exp_full["mwFreq"]) * 1e3  # GHz -> MHz
    opt_full = adddefaults(opt or {}, _DEFAULT_OPT)
    if "nTransitions" in opt_full:
        raise ValueError("Options.nTransitions is obsolete. Please remove from code!")
    n_nuclei = sys["nNuclei"]
    nuclei = opt_full["Nuclei"]
    nuclei = (
        list(range(1, n_nuclei + 1))
        if nuclei is None or np.size(nuclei) == 0
        else [int(k) for k in np.atleast_1d(nuclei)]
    )
    if any(k < 1 or k > n_nuclei for k in nuclei):
        raise ValueError("Opt.Nuclei is out of range.")
    compute_intensities = {"off": False, "on": True}[opt_full["Intensity"]]
    enhancement = {"off": False, "on": True}[opt_full["Enhancement"]]
    init_state, init_basis = _init_state(sys, sys["nStates"])
    non_equi = init_state is not None
    temperature = exp_full["Temperature"]
    if non_equi:
        boltzmann = False
    else:
        if np.isinf(temperature):
            raise ValueError("If given, Params.Temperature must be a finite value.")
        boltzmann = not np.isnan(temperature)
    angles_M2L, average_over_chi = _lab_orientations(exp_full, opt_full, _orientations, _average_over_chi)
    n_orientations = angles_M2L.shape[0]
    ori_weights = opt_full["OriWeights"]
    use_ori_weights = ori_weights is not None and np.size(ori_weights) > 0 and opt_full["OriThreshold"] > 0
    if use_ori_weights:
        ori_weights = np.asarray(ori_weights, dtype=float).ravel()
        if ori_weights.size != n_orientations:
            raise ValueError(
                f"Opt.OriWeights has wrong number of elements ({ori_weights.size}). It must have one entry for "
                f"each of the {n_orientations} orientations!"
            )
    ori_selection = np.isfinite(exp_full["ExciteWidth"]) and exp_full["ExciteWidth"] > 0
    if ori_selection and np.isnan(mw_freq):
        raise ValueError("Exp.mwFreq is missing! It is needed, since Exp.ExciteWidth is given.")
    h_strain = np.asarray(sys["HStrain"], dtype=float).ravel()[:3]

    H0, muxM, muyM, muzM = ham(sys)
    n_states = sys["nStates"]
    if non_equi and init_basis == "zerofield":
        zf_energies, zf_states = np.linalg.eigh(H0)
        if np.unique(np.round(zf_energies, 9)).size != zf_energies.size:
            raise ValueError(
                "Degenerate energy levels detected at zero-field. This prevents unambiguous assignment of the "
                "provided sublevel populations to the zero-field states. Please provide the non-equilibrium state "
                "using the full density matrix."
            )
        init_state = zf_states @ init_state @ zf_states.conj().T

    # Transition pre-selection: NMR transitions by their nuclear Zeeman transition rates on a coarse grid
    if opt_full["Transitions"] is None or np.size(opt_full["Transitions"]) == 0:
        grid = sphgrid(opt_full["TPSGridSymm"], int(opt_full["TPSGridSize"]))
        smuxM, smuyM, smuzM = ham_nz(sys, None, nuclei)
        rates = np.zeros((n_states, n_states))
        max_e = np.zeros((n_states, n_states))
        min_e = np.full((n_states, n_states), np.inf)
        for phi, theta, w in zip(grid.phi, grid.theta, grid.weights, strict=True):
            st, ct, sp, cp = np.sin(theta), np.cos(theta), np.sin(phi), np.cos(phi)
            E, Vs = np.linalg.eigh(H0 - field * (st * (cp * muxM + sp * muyM) + ct * muzM))
            smupM = cp * smuxM + sp * smuyM
            rates += (
                w
                * (
                    np.abs(Vs.conj().T @ (ct * smupM - st * smuzM) @ Vs) ** 2
                    + np.abs(Vs.conj().T @ (-sp * smuxM + cp * smuyM) @ Vs) ** 2
                )
                / 2
            )
            EE = np.abs(E[:, None] - E[None, :])
            max_e = np.maximum(max_e, EE)
            min_e = np.minimum(min_e, EE)
        rng = exp_full["Range"]
        if rng is not None and np.size(rng) == 2 and not np.any(np.isnan(rng)):
            rates[(max_e < rng[0]) | (min_e > rng[1])] = 0
        n_s_states = int(np.prod(2 * sys["S"] + 1))
        electronic = ~np.kron(np.eye(n_s_states, dtype=bool), np.ones((n_states // n_s_states,) * 2, dtype=bool))
        lower = np.tril(np.ones((n_states, n_states), dtype=bool))
        keep = ~(lower | electronic)
        uu, vv = np.nonzero(keep.T)  # column-major order, as MATLAB's find
        u_all, v_all = vv, uu
        tr = rates.T[keep.T]
        n_transitions = int(np.sum(tr > opt_full["Threshold"] * tr.max())) if tr.size else 0
        order = np.argsort(-tr, kind="stable")[:n_transitions]
        transitions = np.column_stack([u_all[order], v_all[order]]) + 1
    else:
        transitions = np.sort(np.atleast_2d(np.asarray(opt_full["Transitions"], dtype=int)), axis=1)
    u = transitions[:, 0] - 1
    v = transitions[:, 1] - 1
    n_transitions = u.size

    pdat = np.full((n_transitions, n_orientations), np.nan)
    idat = np.zeros((n_transitions, n_orientations)) if compute_intensities else None
    if compute_intensities:
        DxM, DyM, DzM = ham_nz(sys, None, nuclei)
        if enhancement:
            ExM_, EyM_, EzM_ = ham_ez(sys)
            DxM, DyM, DzM = DxM + ExM_, DyM + EyM_, DzM + EzM_
        if ori_selection:
            ExM, EyM, EzM = ham_ez(sys, None, [1])
            lw_excite2 = float(exp_full["ExciteWidth"]) ** 2
        if boltzmann:
            boltzmann_pre = -1e6 * planck / boltzm / temperature
    max_epr, min_epr, min_ex_width = -np.inf, np.inf, np.inf
    rf_angle = float(exp_full["rf"] or 0)
    for iOri in range(n_orientations):
        if use_ori_weights and ori_weights[iOri] < opt_full["OriThreshold"]:
            continue
        R = erot(angles_M2L[iOri])
        xLab, yLab, zLab = R[0], R[1], R[2]
        H = H0 - field * (zLab[0] * muxM + zLab[1] * muyM + zLab[2] * muzM)
        if compute_intensities:
            E0, Vs = np.linalg.eigh(H)
        else:
            E0 = np.linalg.eigvalsh(H)
        pdat[:, iOri] = E0[v] - E0[u]
        if not compute_intensities:
            continue
        DyL = yLab[0] * DxM + yLab[1] * DyM + yLab[2] * DzM
        if average_over_chi:
            DxL = xLab[0] * DxM + xLab[1] * DyM + xLab[2] * DzM
            endor_int = (np.abs(Vs.conj().T @ DxL @ Vs) ** 2 + np.abs(Vs.conj().T @ DyL @ Vs) ** 2) / 2
            if ori_selection:
                ExL = xLab[0] * ExM + xLab[1] * EyM + xLab[2] * EzM
                EyL = yLab[0] * ExM + yLab[1] * EyM + yLab[2] * EzM
                epr_int = 2 * np.pi * (np.abs(Vs.conj().T @ ExL @ Vs) ** 2 + np.abs(Vs.conj().T @ EyL @ Vs) ** 2) / 2
        else:
            endor_int = np.abs(Vs.conj().T @ DyL @ Vs) ** 2
            if ori_selection:
                ExL = xLab[0] * ExM + xLab[1] * EyM + xLab[2] * EzM
                epr_int = np.abs(Vs.conj().T @ ExL @ Vs) ** 2
        if rf_angle > 0:
            endor_int = (1 - np.cos(np.sqrt(endor_int) * rf_angle)) / 2
        if ori_selection:
            lw2 = float(np.sum((h_strain * zLab) ** 2)) + lw_excite2
            dE = E0[None, :] - E0[:, None]  # positive in the upper triangle
            populations = None
            if boltzmann:
                populations = np.exp(boltzmann_pre * (E0 - E0[0]))
            elif non_equi:
                if init_basis == "eigen":
                    populations = np.real(np.diag(init_state))
                else:
                    populations = np.real(np.einsum("ij,jk,ki->i", Vs.conj().T, init_state, Vs))
            polarizations = (
                np.ones_like(dE) if populations is None else np.abs(populations[:, None] - populations[None, :])
            )
            epr_int = np.sqrt(2 / np.pi) * polarizations * epr_int * np.exp(-2 / lw2 * (mw_freq - np.abs(dE)) ** 2)
            sum_tp = epr_int.sum(axis=0) - np.diag(epr_int)
            excitation = sum_tp[u] + sum_tp[v] - 2 * epr_int[u, v]
            big = dE[dE >= 0.3 * mw_freq]
            if big.size:
                min_epr, max_epr = min(min_epr, big.min()), max(max_epr, big.max())
            min_ex_width = min(min_ex_width, np.sqrt(lw2))
        else:
            excitation = 1.0
        if boltzmann:
            populations = np.exp(boltzmann_pre * (E0 - E0[0]))
            nuc_pol = (populations[u] - populations[v]) / populations.sum()
        else:
            nuc_pol = 1.0
        idat[:, iOri] = nuc_pol * excitation * endor_int[u, v]
    selectivity = (max_epr - min_epr) / min_ex_width if ori_selection else 0.0
    return EndorResonances(pdat, idat, transitions, {"Selectivity": selectivity})


# ----------------------------------------------------------------------------- endorfrq_perturb
def endorfrq_perturb(
    system: dict, exp: dict, opt: dict | None = None, *, _orientations=None, _average_over_chi: bool = True
) -> EndorResonances:  # noqa: C901
    """ENDOR frequencies and intensities by first- or second-order perturbation theory (Iwasaki 1974).

    One electron spin; ``opt["PerturbOrder"]`` 1 or 2 (default 2); ``opt["Nuclei"]`` selects the nuclei. The
    transitions are ordered by nucleus, then by the mI of the lower level, then by mS; ``trans`` is empty."""
    sys = validatespinsys(system)
    if sys["nElectrons"] != 1:
        raise ValueError("Perturbation theory available only for systems with 1 electron.")
    if np.any(sys["AStrain"]):
        raise ValueError("A strain not supported with perturbation theory. Use matrix diagonalization.")
    if np.any(sys["DStrain"]):
        raise ValueError("D strain not supported with perturbation theory. Use matrix diagonalization.")
    if sys["nNuclei"] > 1 and np.any(sys["nn"] != 0):
        raise ValueError(
            "Nuclear-nuclear couplings not supported with perturbation theory. Use matrix diagonalization."
        )
    if "Field" not in exp or exp["Field"] is None:
        raise ValueError("Exp.Field is missing.")
    exp_full = adddefaults(exp, _DEFAULT_EXP)
    if exp.get("mwMode") == "parallel":
        raise ValueError("Parallel mode ENDOR not supported with perturbation theory. Use matrix diagonalization.")
    if np.isfinite(exp_full["Temperature"]):
        raise ValueError(
            "ENDOR temperature effects are not supported with perturbation theory. Use matrix diagonalization."
        )
    opt = dict(opt or {})
    if opt.get("Transitions") is not None and np.size(opt["Transitions"]) > 0:
        raise ValueError("ENDOR perturbation theory does not support Opt.Transitions.")
    order = int(opt.get("PerturbOrder", 2))
    if order not in (1, 2):
        raise ValueError("Opt.PerturbOrder must be either 1 or 2.")
    second = order == 2
    field = float(exp_full["Field"])
    S = float(sys["S"][0])
    high_spin = S > 0.5
    SS1 = S * (S + 1)
    mS = np.arange(-S, S + 1)
    mS2 = mS**2
    n_mS = mS.size
    if sys["fullg"]:
        g = sys["g"][:3, :3]
    else:
        R = erot(sys["gFrame"][0]).T
        g = R @ np.diag(sys["g"][0]) @ R.T
    if high_spin:
        R = erot(sys["DFrame"][0]).T
        D = R @ np.diag(sys["D"][0]) @ R.T
        trDD = np.trace(D @ D)
    I = sys["I"]
    n_nuclei = sys["nNuclei"]
    nuI = -sys["gn"] * nmagn * field / 1e3 / planck / 1e6
    A, P, mI = [], [], []
    for k in range(n_nuclei):
        mI.append(np.arange(-I[k], I[k] + 1))
        if sys["fullA"]:
            A.append(sys["A"][3 * k : 3 * k + 3, :3])
        else:
            R = erot(sys["AFrame"][k]).T
            A.append(R @ np.diag(sys["A"][k, :3]) @ R.T)
        if sys["fullQ"]:
            Pk = sys["Q"][3 * k : 3 * k + 3, :3]
        else:
            R = erot(sys["QFrame"][k]).T
            Pk = R @ np.diag(sys["Q"][k, :3]) @ R.T
        P.append(Pk - np.eye(3) * np.trace(Pk) / 3)
    nuclei = opt.get("Nuclei")
    nuclei = (
        list(range(1, n_nuclei + 1))
        if nuclei is None or np.size(nuclei) == 0
        else [int(k) for k in np.atleast_1d(nuclei)]
    )
    if any(k < 1 or k > n_nuclei for k in nuclei):
        raise ValueError("Opt.Nuclei is out of range.")
    excite = float(exp_full["ExciteWidth"])
    ori_selection = np.isfinite(excite)
    lw_excite2 = excite**2
    mw_freq = float(exp_full["mwFreq"]) * 1e3
    h_strain = np.asarray(sys["HStrain"], dtype=float).ravel()[:3]
    angles_M2L, _ = _lab_orientations(exp_full, opt, _orientations, _average_over_chi)
    n_orientations = angles_M2L.shape[0]
    has_q = I > 0.5
    detA = [np.linalg.det(a) for a in A]
    invA = [np.linalg.inv(a) if abs(d) > 0 else np.zeros((3, 3)) for a, d in zip(A, detA, strict=True)]
    trAA = [np.trace(a.T @ a) for a in A]
    II1 = I * (I + 1)
    n_mI = [m.size for m in mI]
    # all combinations of nuclear state indices, first nucleus slowest (allcombinations ..., 'i')
    states = (
        np.array(np.meshgrid(*[np.arange(n) for n in n_mI], indexing="ij")).reshape(n_nuclei, -1).T
        if n_nuclei
        else np.zeros((1, 0), int)
    )
    endor_freqs = [np.zeros((n_orientations, n_mS * (n_mI[k] - 1))) for k in range(n_nuclei)]
    total_weight = [np.zeros((n_orientations, n_mI[k] - 1)) for k in range(n_nuclei)]
    max_epr, min_epr, min_ex_width = -np.inf, np.inf, np.inf
    for iOri in range(n_orientations):
        h = erot(angles_M2L[iOri])[2]
        geff = np.linalg.norm(g @ h)
        E0 = geff * bmagn * field * 1e-3 / planck / 1e6
        uvec = g @ h / geff
        if high_spin:
            Du = D @ uvec
            uDu = uvec @ Du
            E1D = -uDu / 2 * (SS1 - 3 * mS2)
            if second:
                uDDu = Du @ Du
                D1sq = uDDu - uDu**2
                D2sq = 2 * trDD + uDu**2 - 4 * uDDu
                E2D = -(D1sq * mS * (4 * SS1 - 8 * mS2 - 1) - D2sq / 4 * mS * (2 * SS1 - 2 * mS2 - 1)) / (2 * E0)
            else:
                E2D = np.zeros(n_mS)
        else:
            E1D = np.zeros(n_mS)
            E2D = np.zeros(n_mS)
        electron_energies = mS * E0 + E1D + E2D
        shifts = []
        for k in range(n_nuclei):
            mI_ = mI[k]
            Ah = A[k] @ g / geff @ h
            shift = np.zeros((n_mS, mI_.size))
            for imS, mS_ in enumerate(mS):
                hf_field = mS_ * Ah
                larmor = nuI[k] * h
                Kh = hf_field + larmor
                normKh = np.linalg.norm(Kh)
                kvec = Kh / normKh
                E1A = mI_ * normKh
                strong = np.linalg.norm(hf_field) > np.linalg.norm(larmor)
                if strong and mS_ < 0:
                    E1A = -E1A
                if has_q[k]:
                    kPk = kvec @ P[k] @ kvec
                    E1Q = -kPk / 2 * (II1[k] - 3 * mI_**2)
                else:
                    kPk = 0.0
                    E1Q = 0.0
                if second:
                    Ak = A[k].T @ kvec
                    kAu = Ak @ uvec
                    kAAk = Ak @ Ak
                    A1sq = kAAk - kAu**2
                    A2 = detA[k] * (uvec @ invA[k] @ kvec)
                    A3 = trAA[k] - np.linalg.norm(A[k] @ uvec) ** 2 - kAAk + kAu**2
                    E2A = (A1sq * mS_ * mI_**2 - A2 * (SS1 - mS_**2) * mI_ + A3 / 2 * mS_ * (II1[k] - mI_**2)) / (
                        2 * E0
                    )
                    if strong and mS_ < 0:
                        E2A = -E2A
                    if high_spin:
                        DA = Du @ Ak - uDu * kAu
                        E2DA = -DA * (SS1 - 3 * mS_**2) * mI_ / E0
                    else:
                        E2DA = 0.0
                    if has_q[k]:
                        kPPk = kvec @ P[k] @ P[k] @ kvec
                        P1sq = kPPk - kPk**2
                        P2sq = 2 * np.trace(P[k]) ** 2 + kPk**2 - 4 * kPPk
                        E2Q = -(
                            P1sq * mI_ * (4 * II1[k] - 8 * mI_**2 - 1) - P2sq / 4 * mI_ * (2 * II1[k] - 2 * mI_**2 - 1)
                        ) / (2 * normKh)
                    else:
                        E2Q = 0.0
                    shift[imS] = E1Q + E1A + E2A + E2DA + E2Q
                else:
                    shift[imS] = E1Q + E1A
            shifts.append(shift)
            dE = np.abs(np.diff(shift, axis=1))  # (nmS, nmI-1)
            endor_freqs[k][iOri] = dE.ravel(order="F")  # mI of the lower level outer, mS inner
        if ori_selection:
            total = np.zeros((states.shape[0], n_mS))
            for imS in range(n_mS):
                total[:, imS] = electron_energies[imS] + sum(shifts[k][imS, states[:, k]] for k in range(n_nuclei))
            epr = np.diff(total, axis=1)
            lw2 = float(np.sum((h_strain * h) ** 2)) + lw_excite2
            excite_factor = np.exp(-2 / lw2 * (epr - mw_freq) ** 2)
            min_epr, max_epr = min(min_epr, epr.min()), max(max_epr, epr.max())
            min_ex_width = min(min_ex_width, np.sqrt(lw2))
            for k in range(n_nuclei):
                for imI in range(n_mI[k] - 1):
                    sel = (states[:, k] == imI) | (states[:, k] == imI + 1)
                    # EasySpin indexes the (states x EPR transitions) factor matrix with this logical vector, which
                    # reaches its first column only: for S > 1/2 the lowest EPR transition alone weights the line
                    total_weight[k][iOri, imI] = excite_factor[sel, 0].sum()
    selectivity = (max_epr - min_epr) / min_ex_width if ori_selection else 0.0
    pos = np.vstack([endor_freqs[k - 1].T for k in nuclei]) if nuclei else np.zeros((0, n_orientations))
    ints = []
    n_nuc_states = 2 * I + 1
    for k in nuclei:
        two_i = int(round(2 * I[k - 1]))
        element = np.sqrt(np.arange(1, two_i + 1) * np.arange(two_i, 0, -1)) / 2
        pre = sys["gn"][k - 1] * nmagn / planck / 1e6 / 1e3
        pre = (pre * element) ** 2
        new = np.repeat((pre / n_nuc_states[k - 1])[:, None], n_orientations, axis=1)
        if ori_selection:
            new = new * total_weight[k - 1].T
        idx = np.repeat(np.arange(two_i), n_mS)
        ints.append(new[idx])
    intensities = (np.vstack(ints) if ints else np.zeros((0, n_orientations))) * np.prod(2 * I + 1)
    return EndorResonances(pos, intensities, np.zeros((0, 2), int), {"Selectivity": selectivity})


# ----------------------------------------------------------------------------- orisel
def orisel(system: dict, exp: dict, opt: dict | None = None, *, per_transition: bool = False):
    """Orientation-selection weights of an EPR excitation at ``exp["Field"]`` (mT), ``exp["mwFreq"]`` (GHz) with
    the excitation width ``exp["ExciteWidth"]`` (MHz FWHM) and ``Sys.HStrain``: one weight per orientation of
    ``exp["Orientations"]`` (rows ``[phi, theta]``) or of the grid (``opt["GridSize"]``, ``GridSymmetry``).
    ``per_transition=True`` returns ``(weights, transitions)`` with one column per level pair."""
    sys = validatespinsys(system)
    h_strain = np.asarray(sys.get("HStrain", [0, 0, 0]), dtype=float).ravel()
    if h_strain.size == 1:
        h_strain = np.repeat(h_strain, 3)
    for key in ("Field", "mwFreq", "ExciteWidth"):
        if key not in exp:
            raise ValueError(f"Params.{key} is missing!")
    excite = float(exp["ExciteWidth"])
    if np.any(h_strain[:3] + excite <= 0):
        raise ValueError("Sys.HStrain or Exp.ExciteWidth must be positive!")
    mw_freq = float(exp["mwFreq"]) * 1e3
    field = float(exp["Field"])
    opt = dict(opt or {})
    averaged = opt.get("AveragedIntensity", True)
    oris = exp.get("Orientations")
    if oris is not None and np.size(oris) > 0:
        oris = np.atleast_2d(np.asarray(oris, dtype=float))
        if oris.shape[0] not in (2, 3) and oris.shape[1] in (2, 3):
            oris = oris.T
        phi, theta = oris[0], oris[1]
    else:
        symmetry = opt.get("GridSymmetry") or None
        if symmetry is None:
            symmetry, frame = hamsymm(sys)
        else:
            frame = np.eye(3)
        grid = sphgrid(symmetry, int(opt.get("GridSize", 46)))
        phi, theta = vec2ang(frame @ grid.vecs)
    n_orientations = phi.size
    H0, muxM, muyM, muzM = ham(sys)
    N = H0.shape[0]
    u, v = np.triu_indices(N, 1)
    order = np.lexsort((v, u))  # nchoosek order: (1,2), (1,3), ...
    u, v = u[order], v[order]
    transitions = np.column_stack([u, v]) + 1
    n_transitions = u.size
    if not np.isfinite(excite):
        weights = np.ones((n_orientations, n_transitions))
    else:
        fac = 1 / np.sqrt(2 * np.log(2))
        vec = ang2vec(phi, theta).T
        gamma = fac * np.sqrt(np.sum(vec**2 * h_strain[:3] ** 2, axis=1) + excite**2)
        weights = np.zeros((n_orientations, n_transitions))
        for k in range(n_orientations):
            R = erot([phi[k], theta[k], 0.0])
            xLab, yLab, zLab = R[0], R[1], R[2]
            muxL = xLab[0] * muxM + xLab[1] * muyM + xLab[2] * muzM
            muyL = yLab[0] * muxM + yLab[1] * muyM + yLab[2] * muzM
            muzL = zLab[0] * muxM + zLab[1] * muyM + zLab[2] * muzM
            E, V = np.linalg.eigh(H0 - field * muzL)
            if averaged:
                rate = (np.abs(V.conj().T @ muxL @ V) ** 2 + np.abs(V.conj().T @ muyL @ V) ** 2) / 2
            else:
                rate = np.abs(V.conj().T @ muxL @ V) ** 2
            xi = (np.abs(E[v] - E[u]) - mw_freq) / gamma[k]
            weights[k] = rate[u, v] * np.exp(-2 * xi**2)
    if per_transition:
        return weights, transitions
    return weights.sum(axis=1)


def anisosubsys(system: dict, lw=None) -> dict:
    """The sub-system responsible for the anisotropy of the EPR spectrum: nuclei whose splitting is below the
    line width ``lw`` (default ``Sys.HStrain``) are removed and their splitting added to ``HStrain``."""
    sub = validatespinsys(system)
    if lw is None:
        lw = np.asarray(system.get("HStrain", [0, 0, 0]), dtype=float).ravel()
    lw = np.asarray(lw, dtype=float).ravel()
    if lw.size == 1:
        lw = np.repeat(lw, 3)
    if sub["nNuclei"] == 0:
        return sub
    if sub["fullA"]:
        amax = np.array([np.abs(sub["A"][3 * k : 3 * k + 3]).max() for k in range(sub["nNuclei"])])
    else:
        amax = np.max(np.abs(sub["A"][: sub["nNuclei"], :3]), axis=1)
    splitting = 2 * sub["I"] * amax
    rmv = splitting < lw.min()
    splitting = np.where(rmv, splitting, 0.0)
    reduced = {
        k: v for k, v in system.items() if k != "processed"
    }  # the raw input, since derived keys do not revalidate
    reduced["HStrain"] = np.sqrt(lw**2 + np.sum(splitting**2))
    reduced = nucspinrmv(reduced, [int(k) + 1 for k in np.nonzero(rmv)[0]])
    return validatespinsys(reduced)


# ----------------------------------------------------------------------------- salt
_SALT_EXP = {
    "Field": np.nan, "mwFreq": np.nan, "Range": None, "CenterSweep": None, "nPoints": 1024, "Temperature": np.nan,
    "ExciteWidth": np.inf, "Ordering": None, "SampleRotation": None, "SampleFrame": None, "CrystalSymmetry": "",
    "MolFrame": None, "Harmonic": 0,
}  # fmt: skip


def salt(system: dict, exp: dict, opt: dict | None = None) -> Spectrum:  # noqa: C901
    """ENDOR spectrum of a powder or crystal: ``Spectrum(x, spec, info)`` with ``x`` in MHz.

    ``exp``: ``Field`` (mT), ``mwFreq`` (GHz), ``Range`` or ``CenterSweep`` (MHz; automatic if absent),
    ``nPoints``, ``Temperature``, ``ExciteWidth`` (MHz), ``SampleFrame``, ``MolFrame``, ``Harmonic``.
    ``opt``: ``Method`` (``'matrix'``, ``'perturb1'``, ``'perturb2'``), ``GridSize``, ``GridSymmetry``,
    ``GridFrame``, ``separate`` (``''``, ``'components'``, ``'transitions'``, ``'orientations'``), ``Transitions``,
    ``Threshold``, ``Nuclei``, ``Intensity``, ``Enhancement``, ``OriPreSelect``, ``OriThreshold``, ``OriWeights``.
    """
    if isinstance(system, (list, tuple)):
        return compisoloop(salt, system, exp, opt)
    sys = validatespinsys(system)
    if any(not any(ch.isdigit() for ch in n) for n in nucstring2list(system.get("Nucs", ""))):
        raise NotImplementedError(
            "Isotope mixtures (elements without mass number in Sys.Nucs) are not ported; give isotopes."
        )
    if np.any(sys["n"] > 1):
        raise ValueError("salt does not support sets of equivalent nuclei (Sys.n>1).")
    if "ExciteWidth" in system:
        raise ValueError("You gave Sys.ExciteWidth, but it should be Exp.ExciteWidth. Please correct.")
    lw_endor = np.asarray(sys["lwEndor"], dtype=float).ravel()
    if lw_endor.size == 1:
        lw_endor = np.append(lw_endor, 0.0)
    convolution = bool(np.any(lw_endor > 0))
    opt_full = dict(opt or {})
    methods = {"matrix": 0, "perturb": 2, "perturb1": 1, "perturb2": 2}
    method_name = opt_full.get("Method", "matrix")
    if method_name not in methods:
        raise ValueError(f"Unknown value '{method_name}' for Opt.Method.")
    perturb_order = methods[method_name]
    exp_full = adddefaults(exp, _SALT_EXP)
    for obsolete, hint in (
        ("Orientations", "use Exp.SampleFrame instead"),
        ("CrystalOrientation", "use Exp.SampleFrame instead"),
    ):
        if obsolete in exp_full and exp_full[obsolete] is not None:
            raise ValueError(f"Exp.{obsolete} is no longer supported, {hint}.")
    if exp_full.get("lightBeam"):
        raise ValueError("Photoselection (via Exp.lightBeam) is not supported.")
    if "HStrain" in exp_full:
        raise ValueError("You gave Exp.HStrain, but it should be Sys.HStrain. Please correct.")
    if np.isnan(exp_full["Field"]):
        raise ValueError("Experiment.Field is missing!")
    if np.isnan(exp_full["mwFreq"]) and np.isfinite(exp_full["ExciteWidth"]):
        raise ValueError("Experiment.mwFreq is missing! It is needed, since Exp.ExciteWidth is given.")
    if exp_full["Ordering"] is not None:
        raise NotImplementedError("Partially ordered samples are not implemented in salt.")
    if exp_full["SampleRotation"] is not None:
        raise NotImplementedError("Exp.SampleRotation is not ported yet.")
    auto_range = exp_full["Range"] is None and exp_full["CenterSweep"] is None
    if exp_full["CenterSweep"] is not None:
        cs = np.asarray(exp_full["CenterSweep"], dtype=float)
        if cs[1] <= 0:
            raise ValueError("Sweep range in Exp.CenterSweep must be positive.")
        exp_full["Range"] = cs[0] + np.array([-1, 1]) * cs[1] / 2
        if exp_full["Range"][0] < 0:
            raise ValueError(f"Start value resulting from Exp.CenterSweep is negative ({exp_full['Range'][0]:g}).")
    if exp_full["Range"] is not None:
        rng = np.asarray(exp_full["Range"], dtype=float).ravel()
        if rng.size != 2 or rng[1] <= rng[0] or np.any(~np.isfinite(rng)) or np.any(rng < 0):
            raise ValueError("Experiment.Range is not valid!")
        exp_full["Range"] = rng
    field = float(exp_full["Field"])
    if exp_full["mwFreq"] == 0 and not sys["fullg"] and np.ptp(sys["g"][0]) == 0:
        exp_full["mwFreq"] = np.mean(sys["g"][0]) * bmagn * field * 1e-3 / planck / 1e9
    crystal = exp_full["MolFrame"] is not None or bool(exp_full["CrystalSymmetry"])
    sample_frame = exp_full["SampleFrame"]
    sample_frame = (
        np.zeros((1, 3))
        if sample_frame is None or np.size(sample_frame) == 0
        else np.atleast_2d(np.asarray(sample_frame, dtype=float))
    )
    if not crystal and sample_frame.shape[0] > 1:
        raise ValueError(
            "For disordered and partially ordered samples, only a single sample orientation (Exp.SampleFrame) "
            "can be used."
        )
    harmonic = int(exp_full["Harmonic"])
    if harmonic > 0 and not convolution:
        harmonic = 0
    n_points = int(exp_full["nPoints"])

    defaults = {
        "GridSize": [61, 0] if perturb_order else [31, 3], "GridSymmetry": "", "GridFrame": None, "separate": "",
        "Smoothing": 2, "minEffKnots": 5, "OriWeights": None, "OriThreshold": 1e-4, "IncludeOriWeights": 1,
        "OriPreSelect": None, "PaddingMultiplier": 3, "Verbosity": 0,
    }  # fmt: skip
    opt_full = adddefaults(opt_full, defaults)
    for obsolete in ("nKnots", "Symmetry", "Convolution", "Width", "nSpline", "LineShape", "Output"):
        if obsolete in opt_full:
            raise ValueError(f"Options.{obsolete} is obsolete.")
    grid_size = np.atleast_1d(np.asarray(opt_full["GridSize"], dtype=int)).tolist()
    if grid_size[0] < 2:
        raise ValueError("Opt.GridSize(1) must be 2 or larger.")
    if len(grid_size) < 2:
        grid_size.append(defaults["GridSize"][1])
    separate = opt_full["separate"] or ""
    if separate not in ("", "components", "transitions", "orientations", "sites"):
        raise ValueError(f"Unknown value '{separate}' for Opt.separate.")
    if separate == "components":
        separate = ""
    if not crystal and separate in ("orientations", "sites"):
        raise ValueError(
            f"Cannot return separate {separate} for non-crystal spectra (Opt.separate='{separate}'). "
            "Use other setting for Opt.separate."
        )
    if crystal and separate == "transitions":
        raise ValueError(
            "Cannot return separate transitions for crystal spectra (Opt.separate='transitions'). "
            "Use other setting for Opt.separate."
        )
    supplied_weights = opt_full["OriWeights"] is not None and np.size(opt_full["OriWeights"]) > 0
    if supplied_weights and opt_full["OriPreSelect"]:
        raise ValueError("You cannot use OriPreSelect and supply OriWeights simultaneously!")

    # No nuclei: an empty spectrum
    if sys["nNuclei"] == 0:
        rng = exp_full["Range"] if exp_full["Range"] is not None else np.array([0.0, 1.0])
        return Spectrum(
            np.linspace(rng[0], rng[1], n_points), np.zeros(n_points), {"Transitions": np.zeros((0, 2), int)}
        )

    # Orientation grid
    if not crystal:
        if opt_full["GridSymmetry"]:
            grid_symmetry, grid_frame = opt_full["GridSymmetry"], None
        else:
            grid_symmetry, grid_frame = hamsymm(sys)
        if opt_full["GridFrame"] is not None:
            grid_frame = np.asarray(opt_full["GridFrame"], dtype=float)
        if grid_frame is None:
            grid_frame = np.eye(3)
        grid = sphgrid(grid_symmetry, grid_size[0], triangulate=True)
        n_octants = grid.n_octants
        ori_weights = grid.weights
        phi, theta = vec2ang(grid_frame @ grid.vecs)
        angles_M2L = np.column_stack([phi, theta, np.zeros_like(phi)])
        n_orientations = phi.size
        grid_params = [n_orientations, grid_size[0], grid.closed_phi, n_octants, grid.max_phi]
    else:
        n_octants = -2
        exp_cr = {**exp_full, "SampleFrame": sample_frame}
        angles_M2L, n_orientations, _, _ = _crystal_orientations(exp_cr, {})
        ori_weights = np.full(n_orientations, 4 * np.pi)
    n_sites = 1

    # Orientation pre-selection by the excitation window of the EPR transitions
    opt_res = {
        k: v
        for k, v in (opt or {}).items()
        if k
        in (
            "Transitions",
            "Threshold",
            "Nuclei",
            "Intensity",
            "Enhancement",
            "OriThreshold",
            "TPSGridSize",
            "TPSGridSymm",
        )
    }
    if supplied_weights:
        opt_res["OriWeights"] = np.asarray(opt_full["OriWeights"], dtype=float).ravel()
    pre_select = (
        (not crystal) and bool(opt_full["OriPreSelect"]) and opt_full["OriThreshold"] > 0 and not supplied_weights
    )
    if pre_select:
        weights = orisel(anisosubsys(system), {**exp_full, "Orientations": np.column_stack([phi, theta])})
        opt_res["OriWeights"] = weights / weights.max()
    exp_res = {k: exp_full[k] for k in ("mwFreq", "Field", "Temperature", "ExciteWidth", "Range")}
    exp_res["rf"] = exp.get("rf", 0)
    if "initState" in system:
        pass
    if perturb_order:
        res = endorfrq_perturb(
            sys,
            exp_res,
            {**opt_res, "PerturbOrder": perturb_order},
            _orientations=angles_M2L,
            _average_over_chi=not crystal,
        )
    else:
        res = endorfrq(sys, exp_res, opt_res, _orientations=angles_M2L, _average_over_chi=not crystal)
    pdat, idat, transitions = res.pos, res.int, res.trans
    n_transitions = pdat.shape[0]
    if n_transitions == 0:
        rng = exp_full["Range"]
        raise ValueError(
            f"No ENDOR resonances between {rng[0] if rng is not None else 0:g} and "
            f"{rng[1] if rng is not None else 0:g} MHz "
            f"({field:g} mT). Check frequency range, magnetic field and transition selection."
        )
    anisotropic_intensities = idat is not None and idat.size > 1
    if supplied_weights and opt_full["IncludeOriWeights"] and idat is not None:
        idat = idat * opt_res["OriWeights"][None, :]
    if auto_range:
        rng = np.array([np.nanmin(pdat), np.nanmax(pdat)]) + np.array([-1, 1]) * lw_endor.max() * 2
        rng = rng + (rng[1] - rng[0]) * np.array([-1, 1]) * 0.2
        rng[0] = max(rng[0], 0)
        exp_full["Range"] = rng
    rng = exp_full["Range"]
    delta_x = (rng[1] - rng[0]) / (n_points - 1)
    x_axis = rng[0] + np.arange(n_points) * delta_x
    selectivity = res.info["Selectivity"]
    info = {"Transitions": transitions, "Selectivity": selectivity, "nOrientations": n_orientations}
    if selectivity > 0 and not crystal and grid_size[0] / opt_full["minEffKnots"] < selectivity:
        info["warning"] = (
            f"Strong orientation selection: only {grid_size[0] / selectivity:0.1f} orientations within the excitation "
            f"window. Increase Opt.GridSize ({grid_size[0]}) or Exp.ExciteWidth ({exp_full['ExciteWidth']:g} MHz)."
        )

    # Spectrum construction
    nan_in_pdat = np.any(np.isnan(pdat), axis=1)
    if np.any(nan_in_pdat):
        grid_size[1] = 1
    axial_grid = n_octants == 0
    using_grid = n_octants >= 0
    do_projection = using_grid
    if not do_projection:
        x0T = 5e4
        wT = x0T / 2.5
        template = gaussian(np.arange(2 * x0T), x0T, wT, -1)
    do_interpolation = using_grid and grid_size[1] > 1
    if do_interpolation:
        if axial_grid:
            interp_mode = ["L3", "L3"] if np.any(nan_in_pdat) else ["G3", "L3"]
        else:
            interp_mode = ["L1", "L1"] if np.any(nan_in_pdat) else ["G3", "L1"]
    else:
        grid_size[1] = 1
    nf_knots = (grid_size[0] - 1) * grid_size[1] + 1
    n_spectra = n_orientations if separate == "orientations" else n_transitions if separate == "transitions" else 1
    spec = np.zeros((n_spectra, n_points))
    if crystal:
        for o in range(n_orientations):
            this_int = idat[:, o] if anisotropic_intensities else np.ones(n_transitions)
            this_spec = lisum1i(template, x0T, wT, pdat[:, o], this_int, np.zeros(n_transitions), x_axis)
            this_spec = this_spec / n_sites / n_orientations * 2 * np.pi * ori_weights[o]
            spec[o if separate == "orientations" else 0] += this_spec
    elif not using_grid:
        for t in range(n_transitions):
            # a single intensity is used as it is (EasySpin takes the scalar Idat here)
            this_int = idat[t] if anisotropic_intensities else (idat.ravel() if idat is not None else np.ones(1))
            this_spec = lisum1i(template, x0T, wT, pdat[t], this_int, np.zeros(1), x_axis) * 2 * np.pi * ori_weights[0]
            if separate == "transitions":
                spec[t] = this_spec
            else:
                spec[0] += this_spec
    else:
        if axial_grid:
            if do_interpolation:
                fgrid = sphgrid(0, nf_knots)
                fphi, fthe = fgrid.phi, fgrid.theta
            else:
                fphi, fthe = phi, theta
            f_weights = -np.diff(np.cos(fthe)) * 4 * np.pi
        else:
            if do_interpolation:
                fgrid = sphgrid(grid_symmetry, nf_knots, triangulate=True)
                fphi, fthe = fgrid.phi, fgrid.theta
                idx_tri, f_weights = fgrid.tri_idx.T.astype(np.intp), fgrid.tri_areas
            else:
                fphi, fthe = phi, theta
                idx_tri, f_weights = grid.tri_idx.T.astype(np.intp), grid.tri_areas
            f_weights = 4 * np.pi * f_weights / f_weights.sum()
        for t in range(n_transitions):
            if do_interpolation:
                f_pos = gridinterp(pdat[t], grid_params, fphi, fthe, interp_mode[0])
                f_int = (
                    gridinterp(idat[t], grid_params, fphi, fthe, interp_mode[1])
                    if anisotropic_intensities
                    else np.ones(fthe.size)
                )
            else:
                f_pos = pdat[t]
                f_int = idat[t] if anisotropic_intensities else np.ones(fthe.size)
            if np.any(f_int < 0):
                max_int = f_int.max()
                f_int = np.where((f_int < 0) & (f_int / max_int > -1e-6), 0, f_int)
                if np.any(f_int < 0):
                    raise ValueError("Negative intensities encountered! Please report!")
            if axial_grid:
                this_spec = projectzones(f_pos, f_int, f_weights, x_axis)
            else:
                this_spec = projecttriangles(idx_tri, f_weights, f_pos, f_int, x_axis)
            this_spec = this_spec * 2 * np.pi
            if separate == "transitions":
                spec[t] = this_spec
            else:
                spec[0] += this_spec

    # Convolution with the ENDOR line shape, or differentiation
    if convolution:
        fwhm_g, fwhm_l = lw_endor[0], lw_endor[1]
        harm_l, harm_g = (harmonic, 0) if fwhm_l > 0 else (0, harmonic)
        exceeds = np.any(spec[:, 0] != 0) or np.any(spec[:, -1] != 0)
        n_pad = int(round(max(fwhm_g, fwhm_l) / delta_x * opt_full["PaddingMultiplier"])) if exceeds else 0
        if n_pad > 0:
            spec = np.hstack([np.repeat(spec[:, :1], n_pad, axis=1), spec, np.repeat(spec[:, -1:], n_pad, axis=1)])
        for fwhm, harm, alpha in ((fwhm_l, harm_l, 0), (fwhm_g, harm_g, 1)):
            if fwhm == 0:
                continue
            if fwhm > 2 * delta_x:
                widths = [0, fwhm] if spec.shape[0] > 1 else fwhm
                spec = convspec(spec, delta_x, widths, harm, alpha)
            elif harm != 0:
                kind = "Lorentzian" if alpha == 0 else "Gaussian"
                raise ValueError(f"{kind} linewidth is smaller than increment - cannot perform convolution.")
        if n_pad > 0:
            spec = spec[:, n_pad : n_pad + n_points]
    elif harmonic > 0:
        for _ in range(harmonic):
            d = np.diff(spec, axis=1) / delta_x
            spec = (np.hstack([d[:, :1], d]) + np.hstack([d, d[:, -1:]])) / 2
    out = spec[0] if spec.shape[0] == 1 and separate == "" else spec
    return Spectrum(x_axis, out, info)
