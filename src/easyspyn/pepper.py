"""Solid-state cw EPR spectra: EasySpin's ``pepper`` for field sweeps.

The resonance data of every transition (field, intensity, strain width)
are computed on an orientation grid chosen by the symmetry of the
Hamiltonian, interpolated to a finer grid, and projected onto the field
axis (triangles of the grid, or segments for axial symmetry); with
anisotropic strain widths every grid cell is summed as a Gaussian line
instead. Convolution with the residual line shape, the detection harmonic
and field modulation follow. Crystals are summed over the given sample
orientations. Methods: ``'matrix'`` (adaptive diagonalisation),
``'hybrid'``, ``'perturb1'``, ``'perturb2'`` (``'perturb'``), ``'eig'``.

EasySpin authors, by original file (contributors to the file up to EasySpin 6.0.15, see
  docs/easyspin-credits.md):
  pepper.m: Stefan Stoll, Claudia Tait, Joscha Nehrkorn
"""

from __future__ import annotations

import numpy as np

from .components import compisoloop
from .constants import bmagn, planck
from .garlic import Spectrum
from .grids import sphgrid
from .interp import gridinterp
from .kernels import lisum1i, projecttriangles, projectzones
from .lineshapes import convspec, gaussian, lorentzian
from .nucdata import nucstring2list
from .perturb import resfields_perturb_oriented
from .processing import fieldmod
from .resfields import _crystal_orientations, adddefaults, resfields_eig, resfields_oriented
from .rotations import vec2ang
from .spinsystem import validatespinsys
from .symmetry import hamsymm

_DEFAULT_EXP = {
    "CenterSweep": None, "Range": None, "mwCenterSweep": None, "mwRange": None, "nPoints": 1024, "Temperature": np.nan,
    "Harmonic": None, "mwMode": "perpendicular", "ModAmp": 0, "mwPhase": 0, "lightBeam": "", "Ordering": None,
    "SampleRotation": None, "SampleFrame": None, "CrystalSymmetry": "", "MolFrame": None,
}  # fmt: skip
_DEFAULT_OPT = {
    "Verbosity": 0, "GridSymmetry": "", "GridFrame": None, "Method": "matrix", "separate": "", "Smoothing": 2,
    "GridSizeMinimum": 10, "Debug": 0, "Intensity": "on", "BruteForce": 0, "ImmediateBinning": 0,
    "PaddingMultiplier": 3, "GridSize": [19, 4], "IsoCutoff": 1e-3,
}  # fmt: skip


def _round_half_up(x):
    return np.floor(np.asarray(x, dtype=float) + 0.5)


def pepper(system: dict, exp: dict, opt: dict | None = None) -> Spectrum:  # noqa: C901
    """Field-swept cw EPR spectrum of a powder or a crystal: ``Spectrum(x, spec, info)``.

    ``exp``: ``mwFreq`` (GHz), ``Range`` or ``CenterSweep`` (mT; automatic for S = 1/2), ``nPoints``, ``Harmonic``,
    ``ModAmp``, ``mwPhase``, ``mwMode``, ``Temperature``, ``SampleFrame``, ``MolFrame``, ``CrystalSymmetry`` (P1),
    ``lightBeam``. ``opt``: ``Method``, ``GridSize`` (``[knots, interpolation factor]``), ``GridSymmetry``,
    ``GridFrame``, ``separate`` (``''`` or ``'transitions'``; ``'orientations'`` for crystals), ``Threshold``,
    ``Transitions``, ``Intensity``, ``BruteForce``, ``Smoothing``, and the ``resfields`` options.

    EasySpin authors (pepper.m): Stefan Stoll, Claudia Tait, Joscha Nehrkorn.
    """
    if isinstance(system, (list, tuple)):  # several components: weighted sum of their spectra
        return compisoloop(pepper, system, exp, opt)
    sys = validatespinsys(system)
    if any(not any(ch.isdigit() for ch in n) for n in nucstring2list(system.get("Nucs", ""))):
        raise NotImplementedError(
            "Isotope mixtures (elements without mass number in Sys.Nucs) are not ported; give isotopes."
        )
    if np.any(sys["n"] > 1):
        raise ValueError("pepper does not support sets of equivalent nuclei (Sys.n>1).")
    strain_widths = bool(
        np.any(sys["HStrain"]) or np.any(sys["DStrain"]) or np.any(sys["gStrain"]) or np.any(sys["AStrain"])
    )
    convolution = bool(np.any(sys["lw"] > 0))
    lw = sys["lw"].copy()

    exp_full = adddefaults(exp, _DEFAULT_EXP)
    for obsolete in ("Orientations", "CrystalOrientation", "Mode"):
        if obsolete in exp_full:
            raise ValueError(f"Exp.{obsolete} is no longer supported.")
    if exp_full.get("mwFreq") is None:
        if exp_full.get("Field") is None:
            raise ValueError(
                "Please supply either the microwave frequency in Exp.mwFreq (for field sweeps) "
                "or the magnetic field in Exp.Field (for frequency sweeps)."
            )
        raise NotImplementedError("Frequency sweeps (Exp.Field with Exp.mwRange) are not ported yet.")
    if exp_full.get("Field") is not None:
        raise ValueError(
            "Give either Exp.mwFreq (for a field sweep) or Exp.Field (for a frequency sweep), but not both."
        )
    mw_freq = float(exp_full["mwFreq"])
    if mw_freq <= 0:
        raise ValueError("Uninterpretable microwave frequency in Exp.mwFreq.")
    if exp_full["CenterSweep"] is None and exp_full["Range"] is None:
        if sys["nElectrons"] == 1 and sys["S"][0] == 0.5:
            I = sys["I"]
            if I.size:
                if sys["fullA"]:
                    Amax = np.abs(sys["A"]).max(axis=1).reshape(-1, 3).max(axis=1)
                else:
                    Amax = np.abs(sys["A"]).max(axis=1)
            else:
                Amax = np.zeros(0)
            hf = float(np.sum(I * Amax)) * 1e6
            gvals = np.linalg.eigvalsh(sys["g"][:3]) if sys["fullg"] else sys["g"].ravel()
            minB = planck * (mw_freq * 1e9 - hf) / bmagn / gvals.max() / 1e-3
            maxB = planck * (mw_freq * 1e9 + hf) / bmagn / gvals.min() / 1e-3
            centre = (maxB + minB) / 2
            sweep = maxB - minB + 3 * lw.max()
            if sweep == 0:
                sweep = 5 * lw.max()
            if sweep == 0:
                sweep = 10
            exp_full["CenterSweep"] = [centre, 1.25 * sweep]
        else:
            raise ValueError(
                "Cannot automatically determine field range. Please provide either Exp.CenterSweep or Exp.Range."
            )
    if exp_full["CenterSweep"] is not None:
        cs = np.asarray(exp_full["CenterSweep"], dtype=float)
        exp_full["Range"] = cs[0] + np.array([-1, 1]) * cs[1] / 2
        if exp_full["Range"][0] < 0:
            raise ValueError("Lower field limit from Exp.CenterSweep cannot be negative.")
    rng = np.asarray(exp_full["Range"], dtype=float).ravel()
    if rng[1] <= rng[0] or not np.all(np.isfinite(rng)):
        raise ValueError("Exp.Range is not valid!")
    if np.any(rng < 0):
        raise ValueError("Negative magnetic fields in Exp.Range are not possible.")
    n_points = int(exp_full["nPoints"])
    if n_points < 2:
        raise ValueError("Problem with Exp.nPoints. Needs to be a number not smaller than 2.")
    harmonic = exp_full["Harmonic"]
    no_broadening = not strain_widths and not convolution
    if harmonic is None or (np.isscalar(harmonic) and np.isnan(harmonic)):
        harmonic = 0 if no_broadening else 1
    harmonic = int(harmonic)
    if harmonic not in (-1, 0, 1, 2):
        raise ValueError("Exp.Harmonic must be either 0, 1 or 2.")
    if no_broadening and harmonic != 0:
        raise ValueError(f"No broadening given. Cannot compute spectrum with Exp.Harmonic={harmonic}.")
    mod_amp = float(exp_full["ModAmp"])
    if mod_amp < 0 or np.isnan(mod_amp):
        raise ValueError("Exp.ModAmp must be either a single positive number or zero.")
    if mod_amp > 0:
        if harmonic < 1:
            raise ValueError("With field modulation (Exp.ModAmp), Exp.Harmonic=0 does not work.")
        mod_harmonic, conv_harmonic, deriv_harmonic = harmonic, 0, 0
    else:
        mod_harmonic = 0
        conv_harmonic, deriv_harmonic = (harmonic, 0) if convolution else (0, harmonic)
    mw_phase = float(exp_full["mwPhase"])
    mw_mode = exp_full["mwMode"]
    if isinstance(mw_mode, str) and mw_mode and mw_mode not in ("perpendicular", "parallel"):
        raise ValueError("Exp.mwMode must be either 'perpendicular' or 'parallel'.")
    non_equi = sys.get("initState") is not None
    temperature = exp_full["Temperature"]
    if not non_equi and np.size(temperature) != 1:
        raise ValueError("If given, Exp.Temperature must be a single number.")

    # Sample type
    if exp_full["Ordering"] is not None:
        raise NotImplementedError("Partially ordered samples (Exp.Ordering) are not ported yet.")
    if exp_full["SampleRotation"]:
        raise NotImplementedError("Exp.SampleRotation is not ported yet.")
    crystal = exp_full["MolFrame"] is not None or bool(exp_full["CrystalSymmetry"])
    if crystal:
        if not exp_full["CrystalSymmetry"]:
            exp_full["CrystalSymmetry"] = "P1"
        if exp_full["MolFrame"] is None:
            exp_full["MolFrame"] = [0, 0, 0]
    if exp_full["SampleFrame"] is None or np.size(exp_full["SampleFrame"]) == 0:
        exp_full["SampleFrame"] = np.zeros((1, 3))
    sample_frame = np.atleast_2d(np.asarray(exp_full["SampleFrame"], dtype=float))
    if sample_frame.shape[1] != 3:
        raise ValueError(
            f"Exp.SampleFrame must be a Nx3 array, yours is {sample_frame.shape[0]}x{sample_frame.shape[1]}."
        )
    if not crystal and sample_frame.shape[0] > 1:
        raise ValueError(
            "For disordered and partially ordered samples, only a single sample orientation (Exp.SampleFrame) "
            "can be used."
        )

    # Options
    opt_full = adddefaults(opt, _DEFAULT_OPT)
    for obsolete, hint in (
        ("Convolution", ""),
        ("Width", ""),
        ("nKnots", "Use Options.GridSize"),
        ("Symmetry", "Use Options.GridSymmetry"),
        ("nSpline", "Use a second number in Options.GridSize"),
        ("Perturb", "Use Opt.Method"),
        ("Output", "Use Options.separate"),
    ):
        if obsolete in opt_full:
            raise ValueError(f"Options.{obsolete} is obsolete. {hint}".strip())
    separate = opt_full["separate"] or ""
    if separate not in ("", "components", "transitions", "orientations", "sites"):
        raise ValueError(f"Unknown value '{separate}' for field separate.")
    if separate == "components":
        separate = ""  # one component: the summed spectrum
    methods = {"eig": 1, "matrix": 2, "perturb": 3, "perturb1": 4, "perturb2": 5, "hybrid": 6}
    if opt_full["Method"] not in methods:
        raise ValueError(f"Unknown value '{opt_full['Method']}' for field Method.")
    method = methods[opt_full["Method"]]
    use_eigenfields = method == 1
    perturbation = method in (3, 4, 5)
    if opt_full["ImmediateBinning"]:
        raise NotImplementedError("Opt.ImmediateBinning is not ported.")
    if perturbation and non_equi:
        raise ValueError("Perturbation theory not available for systems with non-equilibrium populations.")
    grid_size = np.atleast_1d(np.asarray(opt_full["GridSize"], dtype=float)).astype(int)
    if grid_size[0] < opt_full["GridSizeMinimum"]:
        raise ValueError(f"Options.GridSize must not be less than {opt_full['GridSizeMinimum']}.")
    if grid_size.size < 2:
        grid_size = np.array([grid_size[0], 4])
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
    if opt_full["Intensity"] not in ("on", "off"):
        raise ValueError("Options.Intensity must be 'on' or 'off'.")
    anisotropic_intensities = opt_full["Intensity"] == "on"

    # Orientation grid
    if not crystal:
        if opt_full["GridSymmetry"]:
            grid_symmetry, grid_frame = opt_full["GridSymmetry"], None
        else:
            grid_symmetry, grid_frame = hamsymm(sys)
            if non_equi and grid_symmetry == "Dinfh":
                grid_symmetry = "D2h"
        if opt_full["GridFrame"] is not None:
            grid_frame = np.asarray(opt_full["GridFrame"], dtype=float)
        if grid_frame is None:
            grid_frame = np.eye(3)
        grid = sphgrid(grid_symmetry, int(grid_size[0]), triangulate=True)
        n_octants = grid.n_octants
        ori_weights = grid.weights
        phi, theta = vec2ang(grid_frame @ grid.vecs)
        angles_M2L = np.column_stack([phi, theta, np.zeros_like(phi)])
        n_orientations = phi.size
        grid_params = [n_orientations, int(grid_size[0]), grid.closed_phi, n_octants, grid.max_phi]
        average_over_chi = True
    else:
        n_octants = -2
        exp_cr = {**exp_full, "SampleFrame": sample_frame}
        angles_M2L, n_orientations, _, average_over_chi = _crystal_orientations(exp_cr, {})
        ori_weights = np.full(n_orientations, 4 * np.pi)
    n_sites = 1

    # Resonance data
    exp_res = {
        k: v
        for k, v in exp_full.items()
        if k not in ("nPoints", "Harmonic", "ModAmp", "mwPhase", "Ordering", "CenterSweep", "mwCenterSweep", "mwRange")
    }
    exp_res["Range"] = rng
    exp_res["mwMode"] = mw_mode
    opt_res = {
        k: v
        for k, v in (opt or {}).items()
        if k
        in (
            "Transitions",
            "Threshold",
            "Hybrid",
            "HybridCoreNuclei",
            "FuzzLevel",
            "Freq2Field",
            "maxSegments",
            "ModellingAccuracy",
            "RediagLimit",
            "nLevels",
            "Gradient",
            "HybridHFIThreshold",
            "HybridIntThreshold",
            "HybridOnlyHFI",
            "Sites",
        )
    }
    opt_res["Intensity"] = 1
    if use_eigenfields:
        pdat_list, idat_list = [], []
        for o in range(n_orientations):
            exp_e = {**exp_res, "Range": [0, 1e8]}
            B_e, I_e = resfields_eig(
                system,
                exp_e,
                {"Intensities": True},
                _orientations=angles_M2L[o : o + 1],
                _average_over_chi=average_over_chi,
            )
            pdat_list.append(np.asarray(B_e))
            idat_list.append(np.asarray(I_e))
        wdat = None
        transitions = np.zeros((0, 2), int)
    else:
        exp_search = {**exp_res, "Range": np.clip(rng + 0.2 * (rng[1] - rng[0]) * np.array([-1, 1]), 0, None)}
        if method in (2, 6):
            if method == 6:
                opt_res["Hybrid"] = 1
            # resfields in EasySpin 6.0.15 ignores Exp.SearchRange and searches Exp.Range only
            res = resfields_oriented(system, exp_res, opt_res, angles_M2L, average_over_chi, n_sites)
        else:
            opt_res["PerturbOrder"] = 1 if method == 4 else 2
            res = resfields_perturb_oriented(
                system, {**exp_search, "SearchRange": exp_search["Range"]}, opt_res, angles_M2L, average_over_chi
            )
        pdat, idat, wdat, transitions = res.pos, res.int, res.wid, res.trans
        if wdat is not None and wdat.size and np.nanmax(wdat) <= 0:
            wdat = None
    anisotropic_widths = wdat is not None and wdat.size > 0 and np.nanmax(wdat) > 0
    if not anisotropic_intensities and not use_eigenfields:
        idat = np.full_like(idat, np.nanmean(idat))
    n_transitions = 0 if use_eigenfields else pdat.shape[0]

    x_axis = np.linspace(rng[0], rng[1], n_points)
    delta_x = x_axis[1] - x_axis[0]
    brute_force = use_eigenfields or bool(opt_full["BruteForce"])
    axial_grid = n_octants == 0
    anisotropic_spectrum = n_octants >= 0

    if not brute_force:
        nan_in_pdat = np.any(np.isnan(pdat), axis=1)
        do_interpolation = anisotropic_spectrum and grid_size[1] > 1
        if do_interpolation:
            if axial_grid:
                interp_mode = ["L3", "L3", "L3"] if np.any(nan_in_pdat) else ["G3", "L3", "L3"]
            else:
                interp_mode = ["L1", "L1", "L1"] if np.any(nan_in_pdat) else ["G3", "L1", "L1"]
        else:
            grid_size[1] = 1
        nf_knots = (grid_size[0] - 1) * grid_size[1] + 1
        do_projection = anisotropic_spectrum and not anisotropic_widths
        do_summation = not do_projection
        if do_summation:
            use_gaussian_template = True
            if not anisotropic_widths:
                if lw[0] > 0:
                    this_wid = lw[0]
                    lw[0] = 0
                    convolution = lw[1] > 0
                else:
                    use_gaussian_template = False
                    this_wid = lw[1]
                    lw[1] = 0
                    convolution = False
                if not convolution:
                    deriv_harmonic += conv_harmonic
                    conv_harmonic = 0
            if use_gaussian_template:
                x0T = 5e4
                wT = x0T / 2.5
                template = gaussian(np.arange(2 * x0T), x0T, wT, -1)
            else:
                x0T = 1e5
                wT = x0T / 20
                template = lorentzian(np.arange(2 * x0T), x0T, wT, -1, mw_phase)
        if separate == "orientations":
            n_spectra = n_orientations
        elif separate == "transitions":
            n_spectra = n_transitions
        else:
            n_spectra = 1
        spec = np.zeros((n_spectra, n_points))
        if crystal:
            if n_transitions > 0:
                for o in range(n_orientations):
                    this_pos = pdat[:, o]
                    this_int = idat[:, o] if anisotropic_intensities else np.ones(n_transitions)
                    this_w = wdat[:, o] if anisotropic_widths else np.full(n_transitions, this_wid)
                    this_spec = lisum1i(template, x0T, wT, this_pos, this_int, this_w, x_axis)
                    this_spec = this_spec / n_sites / n_orientations * 2 * np.pi * ori_weights[o]
                    if separate == "orientations":
                        spec[o] += this_spec
                    else:
                        spec[0] += this_spec
        elif not anisotropic_spectrum:
            for t in range(n_transitions):
                this_int = idat[t] if anisotropic_intensities else np.ones(1)
                this_w = wdat[t] if anisotropic_widths else np.full(1, this_wid)
                this_spec = lisum1i(template, x0T, wT, pdat[t], this_int, this_w, x_axis) * 2 * np.pi * ori_weights[0]
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
                    f_wid = (
                        gridinterp(wdat[t], grid_params, fphi, fthe, interp_mode[2])
                        if anisotropic_widths
                        else np.zeros(fthe.size)
                    )
                else:
                    f_pos = pdat[t]
                    f_int = idat[t] if anisotropic_intensities else np.ones(fthe.size)
                    f_wid = wdat[t] if anisotropic_widths else np.zeros(fthe.size)
                if not non_equi and np.any(f_int < 0):
                    raise ValueError("Negative intensities encountered! Please report!")
                if np.any(f_wid < 0):
                    raise ValueError("Negative widths encountered! Please report!")
                if do_projection:
                    if axial_grid:
                        this_spec = projectzones(f_pos, f_int, f_weights, x_axis)
                    else:
                        this_spec = projecttriangles(idx_tri, f_weights, f_pos, f_int, x_axis)
                else:
                    if axial_grid:
                        f_pos_c = (f_pos[:-1] + f_pos[1:]) / 2
                        f_int_c = f_weights * (f_int[:-1] + f_int[1:]) / 2
                        f_spread = np.abs(f_pos[:-1] - f_pos[1:])
                        f_wid_m = (f_wid[:-1] + f_wid[1:]) / 2
                        c1, c2 = 1.57246, 18.6348
                    else:
                        sorted_pos = np.sort(f_pos[idx_tri - 1], axis=0)
                        f_pos_c = sorted_pos.mean(axis=0)
                        f_int_c = f_weights * f_int[idx_tri - 1].mean(axis=0)
                        f_spread = sorted_pos[2] - sorted_pos[0]
                        f_wid_m = f_wid[idx_tri - 1].mean(axis=0)
                        c1, c2 = 2.8269, 42.6843
                    with np.errstate(divide="ignore", invalid="ignore"):
                        lam = f_wid_m / f_spread
                        gam = 1 / np.sqrt(c1 * lam**2 + c2 * lam**4)
                    gam[~np.isfinite(gam)] = 0
                    f_wid_c = f_wid_m * (1 + opt_full["Smoothing"] * gam)
                    this_spec = lisum1i(template, x0T, wT, f_pos_c, f_int_c, f_wid_c, x_axis)
                this_spec = this_spec * 2 * np.pi
                if separate == "transitions":
                    spec[t] = this_spec
                else:
                    spec[0] += this_spec
    else:
        if separate:
            raise ValueError("Cannot return separate subspectra when constructing stick spectrum.")
        spec = np.zeros((1, n_points))
        prefactor = (n_points - 1) / (rng[1] - rng[0])
        for o in range(n_orientations):
            if use_eigenfields:
                this_p, amps = pdat_list[o], idat_list[o]
            else:
                this_p, amps = pdat[:, o], idat[:, o]
            idx_pos = _round_half_up(1 + prefactor * (this_p - rng[0]))
            keep = (idx_pos >= 1) & (idx_pos <= n_points) & np.isfinite(idx_pos)
            weights = ori_weights[o] * (amps[keep] if anisotropic_intensities else np.ones(keep.sum()))
            spec[0] += np.bincount(idx_pos[keep].astype(int) - 1, weights=weights, minlength=n_points)
        spec = spec * 2 * np.pi / delta_x

    # Looping transitions in separate output: combine the branches of the same level pair
    if separate == "transitions" and not crystal:
        uniq, inverse = np.unique(transitions, axis=0, return_inverse=True)
        if uniq.shape[0] < transitions.shape[0]:
            new = np.zeros((uniq.shape[0], n_points))
            for k, row in enumerate(np.ravel(inverse)):
                new[row] += spec[k]
            spec, transitions = new, uniq

    # Convolution with the residual line shape, or differentiation
    if convolution:
        fwhm_g, fwhm_l = lw[0], lw[1]
        if fwhm_l > 0:
            harm_l, phase_l, harm_g, phase_g = conv_harmonic, mw_phase, 0, 0.0
        else:
            harm_l, phase_l, harm_g, phase_g = 0, 0.0, conv_harmonic, mw_phase
        exceeds = np.any(spec[:, 0] != 0) or np.any(spec[:, -1] != 0)
        n_pad = int(round(max(fwhm_g, fwhm_l) / delta_x * opt_full["PaddingMultiplier"])) if exceeds else 0
        if n_pad > 0:
            spec = np.hstack([np.repeat(spec[:, :1], n_pad, axis=1), spec, np.repeat(spec[:, -1:], n_pad, axis=1)])
        for fwhm, harm, phase, alpha in ((fwhm_l, harm_l, phase_l, 0), (fwhm_g, harm_g, phase_g, 1)):
            if fwhm == 0:
                continue
            if fwhm > 2 * delta_x:
                widths = [0, fwhm] if spec.shape[0] > 1 else fwhm
                spec = convspec(spec, delta_x, widths, harm, alpha, phase)
            elif harm != 0:
                kind = "Lorentzian" if alpha == 0 else "Gaussian"
                raise ValueError(
                    f"{kind} linewidth (FWHM {fwhm:g} mT) is smaller than 2 increments "
                    f"(2x{delta_x:g} = {2 * delta_x:g} mT) "
                    "- cannot do convolution. Increase linewidth, or increment number of points in Exp.nPoints."
                )
        if n_pad > 0:
            spec = spec[:, n_pad : n_pad + n_points]
    elif deriv_harmonic > 0:
        for _ in range(deriv_harmonic):
            d = np.diff(spec, axis=1) / delta_x
            spec = (np.hstack([d[:, :1], d]) + np.hstack([d, d[:, -1:]])) / 2
    if mod_amp > 0:
        spec = np.vstack([fieldmod(x_axis, row, mod_amp, mod_harmonic) for row in spec])
    out = spec[0] if spec.shape[0] == 1 and separate == "" else spec
    return Spectrum(x_axis, out, {"Transitions": transitions, "nSites": n_sites, "nOrientations": n_orientations})
