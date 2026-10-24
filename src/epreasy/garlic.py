"""Isotropic and fast-motion cw EPR spectra: EasySpin's ``garlic`` with ``fastmotion``, ``equivcouple``.

For one electron spin S = 1/2 and any number of nuclei, the line positions
of each nuclear group come from the Breit-Rabi fixed-point iteration (or a
perturbation expansion), equivalent nuclei (``Sys.n``) are combined into
sets of effective spins, the shifts of all groups are combined, and the
spectrum is built by binning (then convolution), by a Lorentzian template
or by explicit summation. In the fast-motion regime (``Sys.tcorr``) every
line gets its own Lorentzian width from the Redfield coefficients A, B, C
of the anisotropies of g and A (and the quadrupole).
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from .components import compisoloop
from .constants import bmagn, boltzm, nmagn, planck
from .kernels import lisum1i
from .lineshapes import convspec, lorentzian
from .processing import fieldmod
from .resfields import adddefaults
from .rotations import erot
from .spinsystem import validatespinsys
from .units import unitconvert


class Spectrum(NamedTuple):
    x: np.ndarray  # field axis (mT) or frequency axis (GHz)
    spec: np.ndarray
    info: dict


def equivcouple(I: float, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Effective spins ``F`` and their multiplicities ``N`` from ``n`` equivalent spins ``I`` (Clebsch-Gordan
    decomposition; Freed & Fraenkel, J. Chem. Phys. 39, 326 (1963), eq. 4.32)."""
    if n == 1:
        return np.array([float(I)]), np.array([1])
    F = np.arange(I * n, -0.5, -1.0)
    F = F[F >= 0]
    E = np.ones(int(round(2 * I + 1)))
    row = np.array([1.0])
    for _ in range(n):
        row = np.convolve(row, E)
    row = np.concatenate([[0.0], row])
    N = np.diff(row)[: F.size]
    return F, N.astype(int)


def _all_mI(I: np.ndarray) -> np.ndarray:
    """All combinations of mI values, one line per row, the first nucleus varying slowest."""
    grids = np.meshgrid(*[np.arange(-i, i + 0.5) for i in I], indexing="ij")
    return np.column_stack([g.ravel() for g in grids]) if len(I) else np.zeros((1, 0))


def fastmotion(system: dict, field: float, tcorr: float, domain: str = "field"):
    """Lorentzian FWHM line widths (mT, or MHz with ``domain='freq'``) of the fast-motion regime for all mI
    combinations, from the anisotropies of g and A and the quadrupole couplings: ``(lw, mI)``."""
    if domain not in ("field", "freq"):
        raise ValueError("Fourth input must be either 'field' or 'freq'.")
    sys = validatespinsys(system)
    if sys["nElectrons"] != 1 or sys["S"][0] != 0.5:
        raise ValueError("Fast-motion linewidths can only be calculated for systems with S=1/2.")
    if tcorr < 1e-14:
        raise ValueError("Correlation time too small!")
    I = sys["I"]
    n_nuc = sys["nNuclei"]
    g = sys["g"][:3] if sys["fullg"] else (lambda R: R @ np.diag(sys["g"][0]) @ R.T)(erot(sys["gFrame"][0]).T)
    aniso_g = g.max() != g.min() if sys["fullg"] else sys["g"][0].max() != sys["g"][0].min()
    aniso_A = False
    for k in range(n_nuc):
        a = np.diag(sys["A"][3 * k : 3 * k + 3, :3]) if sys["fullA"] else sys["A"][k, :3]
        aniso_A = aniso_A or a.max() != a.min()
    if not aniso_g and not aniso_A and not np.any(sys["Q"]):
        raise ValueError("Either g or A must be anisotropic, or Q must be present.")
    g0 = np.trace(g) / 3
    g1 = (g - np.eye(3) * g0).ravel(order="F")
    A0 = np.zeros(n_nuc)
    A1 = np.zeros((9, n_nuc))
    for k in range(n_nuc):
        if sys["fullA"]:
            A = sys["A"][3 * k : 3 * k + 3, :3]
        else:
            R = erot(sys["AFrame"][k, :3]).T
            A = R @ np.diag(sys["A"][k, :3]) @ R.T
        A0[k] = np.trace(A) / 3
        A1[:, k] = (A - np.eye(3) * A0[k]).ravel(order="F")
    A1 = A1 * 1e6 * 2 * np.pi
    cc = bmagn * field * 1e-3 / (planck / 2 / np.pi)
    omega0 = g0 * cc
    j0 = tcorr
    j1 = tcorr / (1 + omega0**2 * tcorr**2)
    if n_nuc > 0:
        AA = A1.T @ A1
        gA = cc * (g1 @ A1)
    else:
        AA, gA = np.zeros((0, 0)), np.zeros(0)
    gg = cc**2 * (g1 @ g1)
    # Q is taken as given, like EasySpin: a two-element [e2qQ/h, eta] row is not converted to principal values
    Q_raw = np.atleast_2d(np.asarray(system.get("Q", np.zeros((n_nuc, 3))), dtype=float))
    PP = np.sum(Q_raw**2, axis=1) * (1e6 * 2 * np.pi) ** 2 if n_nuc else np.zeros(0)
    diagAA = np.diag(AA) if n_nuc else np.zeros(0)
    Acoef = gg * (2 / 15 * j0 + 1 / 10 * j1) + np.sum(I * (I + 1) * diagAA) * (1 / 20 * j0 + 7 / 60 * j1)
    Bcoef = gA * (4 / 15 * j0 + 1 / 5 * j1)
    Ccoef = diagAA * (1 / 12 * j0 - 1 / 60 * j1)
    Dcoef = AA * (4 / 15 * j0 + 1 / 10 * j1)
    Dcoef = Dcoef - np.tril(Dcoef)

    def convert(c):
        c = np.asarray(c, dtype=float) / 1e6 / np.pi
        return unitconvert(c, "MHz->mT", g0) if domain == "field" else c

    Acoef, Bcoef, Ccoef, Dcoef = (convert(c) for c in (Acoef, Bcoef, Ccoef, Dcoef))
    if n_nuc > 0:
        II = I * (I + 1)
        qA = convert(np.sum(j0 / 20 * PP * II * (II - 1)))
        qC = convert(j0 / 20 * PP * I * (I + 1) * 2)
        qE = convert(j0 / 20 * PP * (-3))
        mI = _all_mI(I)
        lw = np.array(
            [
                Acoef
                + np.sum(Bcoef * m)
                + np.sum(Ccoef * m**2)
                + np.sum(Dcoef * np.outer(m, m))
                + qA
                + np.sum(qC * m**2)
                + np.sum(qE * m**4)
                for m in mI
            ]
        )
    else:
        mI = np.zeros((1, 0))
        lw = np.array([float(Acoef)])
    return lw, mI


def _combine_shifts(shifts: list[np.ndarray]) -> np.ndarray:
    out = np.zeros(1)
    for s in shifts:
        out = np.add.outer(out, s).ravel()
    return out


def _combine_amplitudes(amps: list[np.ndarray]) -> np.ndarray:
    out = np.ones(1)
    for a in amps:
        out = np.multiply.outer(out, a).ravel()
    return out


def _round_half_up(x: np.ndarray) -> np.ndarray:
    return np.floor(x + 0.5)


def garlic(system: dict, exp: dict, opt: dict | None = None) -> Spectrum:  # noqa: C901
    """Isotropic or fast-motion cw EPR spectrum of an S = 1/2 system: ``Spectrum(x, spec, info)``.

    ``exp``: ``mwFreq`` (GHz) with ``Range``/``CenterSweep`` (mT, or automatic) for a field sweep, or ``Field``
    (mT) with ``mwRange``/``mwCenterSweep`` (GHz) for a frequency sweep; ``nPoints``, ``Harmonic``, ``ModAmp``,
    ``mwPhase``, ``mwMode``, ``Temperature``. ``opt``: ``Method`` (``'exact'`` Breit-Rabi, ``'perturb1'``..
    ``'perturb5'``), ``AccumMethod`` (``'binning'``, ``'template'``, ``'explicit'``), ``Stretch``, ``Accuracy``.
    """
    if isinstance(system, (list, tuple)):  # several components: weighted sum of their spectra
        return compisoloop(garlic, system, exp, opt)
    sys = validatespinsys(system)
    opt = dict(opt or {})
    if sys["nElectrons"] != 1 or sys["S"][0] != 0.5:
        raise ValueError("Only systems with electron spin S=1/2 are supported.")
    if np.any(sys["nn"] != 0) if sys["nNuclei"] > 1 else False:
        raise ValueError("garlic does not support nuclear-nuclear couplings (Sys.nn).")
    tcorr = 10.0 ** sys["logtcorr"] if "logtcorr" in sys else sys.get("tcorr", 0)
    tcorr = float(np.asarray(tcorr).ravel()[0]) if np.size(tcorr) else 0.0
    fast_motion = tcorr != 0
    if fast_motion:
        if tcorr <= 0:
            raise ValueError("Problem with System.tcorr: must be positive!")
        if tcorr < 1e-13:
            raise ValueError(f"Correlation time too small ({tcorr:f} seconds).")
        if np.any(sys["n"] > 1):
            raise ValueError("Cannot treat equivalent nuclei in fast-motion regime! Please rewrite spin system!")
    convolution = bool(np.any(sys["lw"] > 0)) or fast_motion

    exp = adddefaults(
        exp,
        {
            "Harmonic": None,
            "nPoints": 1024,
            "ModAmp": 0,
            "mwMode": "perpendicular",
            "mwPhase": 0,
            "Temperature": np.nan,
        },
    )
    for obsolete in ("Orientations", "CrystalOrientation", "Mode"):
        if obsolete in exp:
            raise ValueError(f"Exp.{obsolete} is no longer supported.")
    if exp.get("lightBeam"):
        raise ValueError("Photoselection (via Exp.lightBeam) is not supported.")
    if exp.get("Ordering") is not None:
        raise ValueError("Partial ordering (via Exp.Ordering) is not supported.")
    field_sweep = exp.get("mwFreq") is not None
    if not field_sweep and exp.get("Field") is None:
        raise ValueError(
            "Please supply either the microwave frequency in Exp.mwFreq (for field sweeps) "
            "or the magnetic field in Exp.Field (for frequency sweeps)."
        )
    if field_sweep and exp.get("Field") is not None:
        raise ValueError(
            "Give either Exp.mwFreq (for a field sweep) or Exp.Frequency (for a frequency sweep), but not both."
        )
    if field_sweep:
        mw_freq_ghz = float(exp["mwFreq"])
        if mw_freq_ghz <= 0:
            raise ValueError("Uninterpretable microwave frequency in Exp.mwFreq.")
    else:
        field_mT = float(exp["Field"])
        if field_mT <= 0:
            raise ValueError("Uninterpretable magnetic field in Exp.Field.")
    mw_phase = float(exp["mwPhase"]) if field_sweep else -float(exp["mwPhase"])
    auto_range = False
    if field_sweep:
        if exp.get("CenterSweep") is not None:
            cs = np.asarray(exp["CenterSweep"], dtype=float)
        elif exp.get("Range") is not None:
            rng = np.asarray(exp["Range"], dtype=float)
            if rng[0] >= rng[1] or np.any(rng < 0):
                raise ValueError("Invalid sweep range!")
            cs = np.array([rng.mean(), rng[1] - rng[0]])
        else:
            auto_range = True
        if not auto_range:
            sweep_range = cs[0] + np.array([-1, 1]) / 2 * cs[1]
    else:
        if exp.get("mwCenterSweep") is not None:
            cs = np.asarray(exp["mwCenterSweep"], dtype=float)
        elif exp.get("mwRange") is not None:
            rng = np.asarray(exp["mwRange"], dtype=float)
            if rng[0] >= rng[1] or np.any(rng < 0):
                raise ValueError("Invalid sweep range!")
            cs = np.array([rng.mean(), rng[1] - rng[0]])
        else:
            auto_range = True
        if not auto_range:
            sweep_range = cs[0] + np.array([-1, 1]) / 2 * cs[1]
    n_points = int(exp["nPoints"])
    if n_points < 2:
        raise ValueError("Problem with Exp.nPoints. Needs to be a number not smaller than 2.")
    harmonic = exp["Harmonic"]
    no_broadening = not fast_motion and not np.any(sys["lw"]) and not np.any(sys["lwpp"])
    if harmonic is None or (np.isscalar(harmonic) and np.isnan(harmonic)):
        harmonic = 1 if (field_sweep and not no_broadening) else 0
    harmonic = int(harmonic)
    if harmonic not in (-1, 0, 1, 2):
        raise ValueError("Exp.Harmonic must be either 0, 1 or 2.")
    if no_broadening and harmonic != 0:
        raise ValueError(f"No broadening given. Cannot compute spectrum with Exp.Harmonic={harmonic}.")
    if exp["mwMode"] == "perpendicular":
        parallel = False
    elif exp["mwMode"] == "parallel":
        parallel = True
    else:
        raise ValueError("Exp.mwMode must be either 'perpendicular' or 'parallel'.")
    temperature = exp["Temperature"]
    if temperature is not None and np.isfinite(temperature) is False and not np.isnan(temperature):
        raise ValueError("If given, Exp.Temperature must be a finite value.")
    mod_amp = float(exp["ModAmp"])
    if mod_amp < 0 or np.isnan(mod_amp):
        raise ValueError("Exp.ModAmp must be either a single positive number or zero.")
    if mod_amp > 0:
        if not field_sweep:
            raise ValueError("Exp.ModAmp cannot be used with frequency sweeps.")
        if harmonic < 1:
            raise ValueError("With field modulation (Exp.ModAmp), Exp.Harmonic=0 does not work.")
        mod_harmonic, conv_harmonic = harmonic, 0
    else:
        mod_harmonic = 0
        conv_harmonic = harmonic if convolution else 0  # garlic has no differentiation step

    stretch = opt.get("Stretch", 0.25)
    max_iter = int(opt.get("MaxIterations", 15))
    accuracy = opt.get("Accuracy", 1e-12)
    method = opt.get("Method", "exact")
    orders = {"exact": 0, "perturb": 5, "perturb1": 1, "perturb2": 2, "perturb3": 3, "perturb4": 4, "perturb5": 5}
    if method not in orders:
        raise ValueError(f"Unknown method '{method}'.")
    perturb_order = orders[method]
    accum = opt.get("AccumMethod") or ("explicit" if fast_motion else "binning")

    giso = np.mean(np.linalg.eigvals(sys["g"][:3]).real) if sys["fullg"] else np.mean(sys["g"][0])
    if field_sweep:
        mw_freq = mw_freq_ghz * 1e9
        central = mw_freq * planck / (bmagn * giso) * 1e3  # mT
    else:
        central = bmagn * giso * field_mT * 1e-3 / planck  # Hz
    if fast_motion:
        if field_sweep:
            fm_lw, _ = fastmotion(sys, central, tcorr, "field")  # the validated system, like EasySpin
        else:
            fm_lw, _ = fastmotion(sys, field_mT, tcorr, "freq")
            fm_lw = fm_lw / 1e3
        if np.all(fm_lw == 0):
            raise ValueError(
                "All calculated fast-motion linewidths for rotational diffusion are zero. "
                "Please provide full tensors, and not just isotropic averages."
            )
        if np.any(fm_lw <= 0):
            raise ValueError("Some calculated fast-motion linewidths are negative. They must be positive.")

    n_nuc = sys["nNuclei"]
    if n_nuc > 0:
        I_all, gn, n_all = sys["I"], sys["gn"], sys["n"].astype(int)
        if sys["fullA"]:
            a_all = (
                np.array([np.mean(np.linalg.eigvals(sys["A"][3 * k : 3 * k + 3, :3]).real) for k in range(n_nuc)]) * 1e6
            )
        else:
            a_all = np.mean(sys["A"][:, :3], axis=1) * 1e6
        if np.any(I_all == 0):
            raise ValueError("Nuclei with spin 0 present. Cannot compute spectrum.")
        if np.any(a_all == 0):
            raise ValueError("Nuclei with hyperfine coupling 0 present. Cannot compute spectrum.")
        a_all = a_all * planck  # J
    shifts: list[np.ndarray] = []
    amplitudes: list[np.ndarray] = []
    for k in range(n_nuc):
        if n_all[k] == 0:
            shifts.append(np.zeros(1))
            amplitudes.append(np.ones(1))
            continue
        F, nn = equivcouple(I_all[k], n_all[k])
        n_lines = (2 * I_all[k] + 1) ** n_all[k]
        aiso = abs(a_all[k])
        positions, intensities = [], []
        if field_sweep:
            gammae = giso * bmagn
            h = planck
            for Fi, mult in zip(F, nn):
                if mult == 0:
                    continue
                mI = np.arange(-Fi, Fi + 0.5)
                if perturb_order == 0:
                    B_ = np.zeros(mI.size)
                    gamman = 0.0
                    for _ in range(max_iter):
                        nu = mw_freq + gamman / h * B_
                        qf = 1 - (aiso / (2 * h * nu)) ** 2
                        Bnew = (
                            aiso
                            / (gammae + gamman)
                            / qf
                            * (-mI + np.sqrt(mI**2 + qf * ((h * nu / aiso) ** 2 - (Fi + 0.5) ** 2)))
                        )
                        if np.any(np.isnan(Bnew)):
                            raise ValueError(
                                "Hyperfine coupling too large compared to microwave frequency. "
                                "Cannot compute resonance field positions. Use pepper() instead."
                            )
                        rel_change = np.inf if np.all(B_ == 0) else np.max(np.abs(1 - Bnew / B_))
                        B_ = Bnew
                        gamman = gn[k] * nmagn
                        if rel_change < accuracy:
                            break
                    else:
                        raise ValueError(f"Breit-Rabi solver didn't converge after {max_iter} iterations!")
                    positions.extend(B_)
                    intensities.extend([mult / n_lines] * mI.size)
                else:
                    pre = aiso * (aiso / (giso * bmagn + gn[k] * nmagn) / 2) ** np.arange(1, 5)
                    for m in mI:
                        c = np.zeros(6)
                        c[5] = (
                            pre[3]
                            * m
                            * (
                                1
                                + 6 * (Fi - 1) * Fi * (Fi + 1) * (Fi + 2)
                                - 10 * (-3 + 2 * Fi * (Fi + 1)) * m**2
                                + 14 * m**4
                            )
                        )
                        c[4] = pre[2] * (Fi - 2 * Fi**3 - Fi**4 + 6 * (Fi**2 + Fi - 1) * m**2 - 5 * m**4)
                        c[3] = pre[1] * m * (1 - 2 * Fi * (Fi + 1) + 2 * m**2)
                        c[2] = pre[0] * (Fi * (Fi + 1) - m**2)
                        c[1] = aiso * m - mw_freq * planck
                        c[0] = giso * bmagn
                        c = c[: perturb_order + 1]
                        dc = np.polyder(c)
                        Bb = -c[1] / c[0]
                        dB = Bb
                        for _ in range(max_iter):
                            if abs(dB / Bb) < accuracy:
                                break
                            dB = np.polyval(c, Bb) / np.polyval(dc, Bb)
                            Bb = Bb - dB
                        else:
                            raise ValueError("Newton-Raphson has convergence problem!")
                        positions.append(Bb)
                        intensities.append(mult / n_lines)
            positions = np.asarray(positions) * 1e3  # T -> mT
        else:
            B_ = field_mT * 1e-3
            gebB = giso * bmagn * B_
            gnbB = gn[k] * nmagn * B_
            for Fi, mult in zip(F, nn):
                if mult == 0:
                    continue
                mI = np.arange(-Fi, Fi + 0.5)
                if perturb_order == 0:
                    alpha = (gebB + gnbB) / aiso / (Fi + 0.5)
                    E1 = (
                        -aiso / 4
                        - gnbB * (mI + 0.5)
                        + (Fi + 0.5) * aiso / 2 * np.sqrt(1 + 2 * (mI + 0.5) / (Fi + 0.5) * alpha + alpha**2)
                    )
                    E2 = (
                        -aiso / 4
                        - gnbB * (mI - 0.5)
                        - (Fi + 0.5) * aiso / 2 * np.sqrt(1 + 2 * (mI - 0.5) / (Fi + 0.5) * alpha + alpha**2)
                    )
                    positions.extend((E1 - E2) / planck)
                    intensities.extend([mult / n_lines] * mI.size)
                else:
                    pre = aiso * (aiso / (giso * bmagn * B_ + gn[k] * nmagn * B_) / 2) ** np.arange(1, 5)
                    for m in mI:
                        c = np.array([
                            giso * bmagn * B_,
                            aiso * m,
                            pre[0] * (Fi * (Fi + 1) - m**2),
                            pre[1] * m * (1 - 2 * Fi * (Fi + 1) + 2 * m**2),
                            pre[2] * (Fi - 2 * Fi**3 - Fi**4 + 6 * (Fi**2 + Fi - 1) * m**2 - 5 * m**4),
                            pre[3]
                            * m
                            * (1 + 6 * (Fi - 1) * Fi * (Fi + 1) * (Fi + 2) - 10 * (-3 + 2 * Fi * (Fi + 1)) * m**2
                                          + 14 * m**4),
                        ])  # fmt: skip
                        positions.append(np.sum(c[: perturb_order + 1]) / planck)
                        intensities.append(mult / n_lines)
            positions = np.asarray(positions)
        shifts.append(positions - central)
        amplitudes.append(np.asarray(intensities, dtype=float))
    pos_max = central + sum(s.max() for s in shifts) if shifts else central
    pos_min = central + sum(s.min() for s in shifts) if shifts else central

    if fast_motion:
        lorentz_lw = fm_lw + sys["lw"][1]
        gauss_lw = sys["lw"][0]
        max_lw = max(fm_lw.max(), sys["lw"][0])
    else:
        lorentz_lw = np.array([sys["lw"][1]])
        gauss_lw = sys["lw"][0]
        max_lw = sys["lw"].max()
    if auto_range:
        if field_sweep:
            pos_range = (pos_max - pos_min) * stretch
            sweep_range = np.array([pos_min, pos_max]) + np.array([-1, 1]) * max(5 * max_lw, pos_range, 1.0)
            sweep_range[0] = max(sweep_range[0], 0)
        else:
            pos_range = (pos_max - pos_min) * stretch
            sweep_range = np.array([pos_min, pos_max]) + np.array([-1, 1]) * max(5 * max_lw / 1e3, pos_range, 1e6)
            sweep_range[0] = max(sweep_range[0], 0)
            sweep_range = sweep_range / 1e9
    positions = _combine_shifts(shifts) + central if shifts else np.array([central])
    intensities = _combine_amplitudes(amplitudes) if amplitudes else np.ones(1)
    if not field_sweep:
        positions = positions / 1e9  # Hz -> GHz
    if parallel:
        intensities = np.zeros_like(intensities)
    else:
        rate = (8 * np.pi**2) * giso**2 * (bmagn / planck / 1e9 / 2) ** 2
        intensities = intensities * rate
        if field_sweep:
            intensities = intensities * planck / (giso * bmagn) * 1e9
        if temperature is not None and np.isfinite(temperature):
            dE = planck * mw_freq if field_sweep else bmagn * giso * field_mT * 1e-3
            e = np.exp(-dE / boltzm / temperature)
            pops = np.array([1, e]) / (1 + e)
            intensities = intensities * (pops[0] - pops[1])

    x_axis = np.linspace(sweep_range[0], sweep_range[1], n_points)
    if accum == "template":
        if np.all(lorentz_lw == 0):
            raise ValueError("Cannot use templated linshape accumulation with zero Lorentzian linewidth.")
        if mw_phase != 0:
            raise ValueError("Cannot use templated lineshape accumulation with non-zero Exp.mwPhase.")
        dx_fine = min(x_axis[1] - x_axis[0], lorentz_lw.min() / 5)
        n_fine = int(round((sweep_range[1] - sweep_range[0]) / dx_fine + 1))
        x_fine = np.linspace(sweep_range[0], sweep_range[1], n_fine)
        dx_fine = x_fine[1] - x_fine[0]
        x0T = 1e5
        wT = x0T / 20
        xT = np.arange(2 * x0T)
        template = lorentzian(xT, x0T, wT, harmonic - 1)
        widths = np.broadcast_to(lorentz_lw, positions.shape) if lorentz_lw.size == 1 else lorentz_lw
        # lisum1i scales amplitudes for an integral template only; each derivative of the template needs a factor
        # wT/lw (the correction of EasySpin's development branch; the 6.0.15 release lacks it)
        scaled = intensities * (wT / widths) ** harmonic
        spec = lisum1i(template, x0T, wT, positions, scaled, widths, x_fine)
        conv_harmonic = 0
        lorentz_lw = np.zeros(1)
    elif accum == "explicit":
        if np.all(lorentz_lw == 0):
            raise ValueError("Cannot use axplicit accumulation with zero Lorentzian linewidth.")
        dx_fine = min(x_axis[1] - x_axis[0], lorentz_lw.min() / 5)
        n_fine = int(round((sweep_range[1] - sweep_range[0]) / dx_fine + 1))
        x_fine = np.linspace(sweep_range[0], sweep_range[1], n_fine)
        dx_fine = x_fine[1] - x_fine[0]
        spec = np.zeros(n_fine)
        widths = np.broadcast_to(lorentz_lw, positions.shape) if lorentz_lw.size == 1 else lorentz_lw
        for p, a, w in zip(positions, intensities, widths):
            spec = spec + a * lorentzian(x_fine, p, w, conv_harmonic, mw_phase)
        mw_phase = 0.0
        conv_harmonic = 0
        lorentz_lw = np.zeros(1)
    elif accum == "binning":
        if fast_motion:
            raise ValueError("Cannot use delta binning (Options.AccumMethod='binning' in the fast-motion regime.")
        x_fine = x_axis
        dx_fine = x_fine[1] - x_fine[0]
        idx = 1 + _round_half_up((positions - sweep_range[0]) / (sweep_range[1] - sweep_range[0]) * (n_points - 1))
        in_range = (idx >= 1) & (idx <= n_points)
        spec = np.bincount(idx[in_range].astype(int) - 1, weights=intensities[in_range], minlength=n_points) / dx_fine
    else:
        raise ValueError(f"Unknown method '{accum}' in Options.AccumMethod.")

    if convolution:
        fwhm_l = float(lorentz_lw.ravel()[0]) if lorentz_lw.size == 1 else 0.0
        fwhm_g = gauss_lw
        if fwhm_l > 0:
            harm_l, phase_l, harm_g, phase_g = conv_harmonic, mw_phase, 0, 0.0
        else:
            harm_l, phase_l, harm_g, phase_g = 0, 0.0, conv_harmonic, mw_phase
        unit = "mT"
        if not field_sweep:
            fwhm_l, fwhm_g, unit = fwhm_l / 1e3, fwhm_g / 1e3, "GHz"
        if fwhm_l != 0:
            if fwhm_l > 2 * dx_fine:
                spec = convspec(spec, dx_fine, fwhm_l, harm_l, 0, phase_l)
            elif harm_l != 0:
                raise ValueError(
                    f"Lorentzian linewidth (FWHM {fwhm_l:g} {unit}) is smaller than 2 increments - cannot do "
                    "convolution. Increase linewidth or decrease increment."
                )
        if fwhm_g != 0:
            if fwhm_g > 2 * dx_fine:
                spec = convspec(spec, dx_fine, fwhm_g, harm_g, 1, phase_g)
            elif harm_g != 0:
                raise ValueError(
                    f"Gaussian linewidth (FWHM {fwhm_g:g} {unit}) is smaller than 2 increments - cannot do "
                    "convolution. Increase linewidth or decrease increment."
                )
    if x_fine.size != x_axis.size:
        spec = np.interp(x_axis, x_fine, spec)
    if field_sweep and mod_amp > 0:
        spec = fieldmod(x_axis, spec, mod_amp, mod_harmonic)
    return Spectrum(x_axis, spec, {"resfields": positions, "intensities": intensities})
