"""Pulse EPR: EasySpin's saffron (fast engine) with its helpers larmorfrq, apowin, fdaxis and exponfit.

``saffron`` simulates ESEEM, HYSCORE and Mims ENDOR signals of a spin
system in the ideal-pulse picture of EasySpin's fast engine: for every
orientation, the electron spin manifolds are diagonalised (analytically for
S = 1/2 with all nuclei weakly coupled, numerically otherwise), the nuclear
sub-Hamiltonians of both manifolds give the nuclear frequencies and the
overlap matrix, and the echo modulation of the predefined experiments
(``'2pESEEM'``, ``'3pESEEM'``, ``'4pESEEM'``, ``'HYSCORE'``, ``'MimsENDOR'``)
or of a user-defined sequence (coherence transfer pathways found by
``sf_pathways``) is accumulated in a frequency-domain buffer (the C kernel
``sf_peaks``, here vectorised) and transformed to the time domain, or
evolved directly in the time domain. Real rectangular pulses are included
by numerical propagation over an offset distribution; the product rule
treats nuclei in separate subspaces. The thyme engine for shaped pulses and
``spidyan`` are not ported.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import expm

from .components import compisoloop
from .constants import bmagn, nmagn, planck
from .fitting import basecorr
from .garlic import Spectrum
from .grids import sphgrid
from .hamiltonian import ham
from .lineshapes import convspec
from .nucdata import nucgval, nucstring2list
from .resfields import _crystal_orientations, nucspinrmv
from .rotations import ang2vec, erot, vec2ang
from .spinops import sop
from .spinsystem import validatespinsys
from .symmetry import hamsymm

__all__ = ["apowin", "exponfit", "fdaxis", "larmorfrq", "saffron", "sf_pathways"]

_EXPERIMENTS = {"2pESEEM": 1, "3pESEEM": 2, "4pESEEM": 3, "HYSCORE": 4, "MimsENDOR": 5}
_INC_SCHEMES = {
    (1,): 1,
    (1, 1): 2,
    (1, -1): 3,
    (1, 2): 11,
    (1, 2, 1): 12,
    (1, 2, 2): 13,
    (1, 1, 2): 14,
    (1, 2, 2, 1): 15,
    (1, 2, -2, 1): 16,
    (1, 1, 2, 2): 17,
}


# ----------------------------------------------------------------------------- helpers
def larmorfrq(nucs, fields) -> np.ndarray:
    """Nuclear Larmor frequencies (MHz) of the isotopes in ``nucs`` at the fields ``fields`` (mT):
    an (nFields, nNuclei) array."""
    gn = np.abs(np.atleast_1d(nucgval(nucs)))
    pre = gn * nmagn / 1e3 / planck / 1e6
    fields = np.atleast_1d(np.asarray(fields, dtype=float)).ravel()
    return np.outer(fields, pre)


def apowin(kind: str, n: int, alpha: float | None = None) -> np.ndarray:
    """Apodisation window of ``n`` points: ``'ham'``, ``'han'``, ``'bla'``, ``'bar'``, ``'con'``, ``'cos'``,
    ``'wel'`` (no parameter), ``'kai'``, ``'gau'``, ``'exp'`` (with ``alpha``); a trailing ``'+'`` or
    ``'-'`` gives the right or left half."""
    if not isinstance(kind, str) or len(kind) not in (3, 4):
        raise ValueError("The first argument must be a 3- or 4-character string!")
    if len(kind) == 4:
        if kind[3] == "+":
            xmin, xmax = 0.0, 1.0
        elif kind[3] == "-":
            xmin, xmax = -1.0, 0.0
        else:
            raise ValueError("Wrong 4th character in Type. Should be + or -.")
    else:
        xmin, xmax = -1.0, 1.0
    if n <= 0 or n != int(n):
        raise ValueError("n must be a positive integer!")
    x = np.linspace(xmin, xmax, int(n))
    base = kind[:3]
    needs_alpha = base in ("kai", "gau", "exp")
    if needs_alpha and alpha is None:
        raise ValueError("Wrong number of input arguments")
    if not needs_alpha and alpha is not None:
        raise ValueError("Wrong number of input arguments")
    if base == "ham":
        w = 0.54 + 0.46 * np.cos(np.pi * x)
    elif base == "kai":
        from scipy.special import i0

        w = i0(alpha * np.sqrt(1 - x**2)) / i0(alpha)
    elif base == "gau":
        w = np.exp(-2 * x**2 / alpha**2)
    elif base == "exp":
        w = np.exp(-alpha * np.abs(x))
    elif base == "han":
        w = 0.5 + 0.5 * np.cos(np.pi * x)
    elif base == "bla":
        w = 0.42 + 0.5 * np.cos(np.pi * x) + 0.08 * np.cos(2 * np.pi * x)
    elif base == "bar":
        w = 1 - np.abs(x)
    elif base == "con":
        w = (1 - x**2) ** 2
    elif base == "cos":
        w = np.cos(np.pi * x / 2)
    elif base == "wel":
        w = 1 - x**2
    else:
        raise ValueError("Unknown apodization window specified!")
    if xmax - xmin == 2:
        w = (w + w[::-1]) / 2
    return w / w.max()


def fdaxis(*args) -> np.ndarray:
    """Frequency axis of an FFT: ``fdaxis(dt, n)`` or ``fdaxis(time_axis)``, centred (zero at index n//2)."""
    if len(args) == 1:
        t = np.asarray(args[0], dtype=float).ravel()
        dt, n = t[1] - t[0], t.size
    elif len(args) == 2:
        dt, n = float(args[0]), int(args[1])
    else:
        raise ValueError("Wrong number of input arguments!")
    return 1 / (2 * dt) * 2 / n * (np.arange(n) - int(n / 2))


def _exp_residuals(x, t, y, include_offset):
    A = np.exp(np.outer(t, x))
    if include_offset:
        A = np.column_stack([A, np.ones(t.size)])
    c = np.linalg.lstsq(A, y, rcond=None)[0]
    res = y - A @ c
    n = x.size
    J = np.zeros((t.size, n))
    AA = A.T @ A
    for j in range(n):
        jj = -t * A[:, j]
        b = c[j] * (A.T @ jj)
        b[j] -= jj @ res
        J[:, j] = c[j] * jj - A @ np.linalg.solve(AA, b)
    return res, J, c


def _mexpfit(t, y, k_guess, include_offset):
    """Variable-projection Levenberg-Marquardt fit of a sum of exponentials (EasySpin's mexpfit2)."""
    x = -np.sort(-np.asarray(k_guess, dtype=float).ravel())
    n = x.size
    if np.any(x >= 0):
        raise ValueError("k must be strictly negative")
    min_grad, tol_step, max_evals, mu0 = 1e-11, 1e-11, 10000, 1e-3
    res, J, _ = _exp_residuals(x, t, y, include_offset)
    A = J.T @ J
    g = J.T @ res
    F = res @ res / 2
    gnorm = np.linalg.norm(g, np.inf)
    mu = mu0 * np.max(np.diag(A))
    nu = 2.0
    n_evals = 1
    mok = 0
    stop = 0
    while not stop:
        if gnorm <= min_grad:
            stop = 1
        elif n_evals == max_evals:
            stop = 3
        else:
            step = np.linalg.solve(A + mu * np.eye(n), -g)
            nstep = np.linalg.norm(step)
            nx = tol_step + np.linalg.norm(x)
            if nstep <= tol_step * nx:
                stop = 2
            elif nstep >= nx / np.finfo(float).eps:
                stop = 4
        if stop:
            break
        bad = np.nonzero(x + step >= 0)[0]
        if bad.size:
            step = np.min(-0.99 * x[bad] / step[bad]) * step
            mu = 10 * mu
        xnew = x + step
        step = xnew - x
        dL = step @ (mu * step - g) / 2
        res_new, J_new, _ = _exp_residuals(xnew, t, y, include_offset)
        n_evals += 1
        F_new = res_new @ res_new / 2
        dF = F - F_new
        if dL > 0 and dF > 0:
            mok += 1
            mu = mu * max(1 / 3, 1 - (2 * dF / dL - 1) ** 3)
            nu = 2.0
            x, F, J, res = xnew, F_new, J_new, res_new
            A = J.T @ J
            g = J.T @ res
            gnorm = np.linalg.norm(g, np.inf)
        else:
            mu = mu * nu
            nu = 2 * nu
            if mok > n and nu > 8:
                stop = 5
    x = -np.sort(-x)
    _, _, c = _exp_residuals(x, t, y, include_offset)
    if include_offset:
        c = np.concatenate([c[n:], c[:n]])
    return x, c


def exponfit(x, y, n_exponents: int = 1, noconst: bool = False):
    """Fit one or two decaying exponentials (with an offset unless ``noconst``) to ``y(x)``, column by
    column: ``(k, c, fitted)`` with decay rates ``k``, coefficients ``c`` (offset first) and the fit."""
    if n_exponents not in (1, 2):
        raise ValueError("Number of exponents must be 1 or 2.")
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float)
    flipped = y.ndim == 1
    Y = y[:, None] if flipped else y
    scale = np.max(np.abs(x))
    xs = x / scale
    n_cols = Y.shape[1]
    k = np.zeros((n_exponents, n_cols))
    c = np.zeros((n_exponents + 1, n_cols))
    fitted = np.zeros_like(Y)
    for col in range(n_cols):
        ys = Y[:, col].copy()
        recovery = ys[0] < ys[-1]
        if recovery:
            ys = -ys
        C = ys[-1]
        A = ys[0] - C
        dt = xs[1] - xs[0]
        integral = np.sum(ys - C) * dt
        k_guess = A / integral if integral != 0 else 0.0
        if k_guess < 0 or integral == 0:
            k_guess = 1 / (xs.size / 2 * dt)
        k_start = -k_guess if n_exponents == 1 else np.array([0.8, 1.3]) * -k_guess
        kk, cc = _mexpfit(xs, ys, np.atleast_1d(k_start), not noconst)
        if noconst:
            c[0, col] = 0
            c[1:, col] = cc
        else:
            c[:, col] = cc
        k[:, col] = kk
        if recovery:
            c[:, col] = -c[:, col]
        fitted[:, col] = c[0, col] + sum(c[1 + e, col] * np.exp(k[e, col] * xs) for e in range(n_exponents))
    k = -k / scale
    if flipped:
        return k[:, 0], c[:, 0], fitted[:, 0]
    return k.T, c.T, fitted


# ----------------------------------------------------------------------------- pathways
def sf_pathways(t, inc) -> np.ndarray:
    """Coherence transfer pathways (rows; 1 = alpha, 2 = beta, 3 = +1, 4 = -1) that refocus at the end of a
    sequence with ideal pulses, initial delays ``t`` (us) and incrementation scheme ``inc`` (0 constant,
    +-k incremented or decremented along dimension k)."""
    t = np.asarray(t, dtype=float).ravel()
    inc = np.asarray(inc, dtype=int).ravel()
    if inc.size != t.size:
        raise ValueError("inc (first input) and t (second input) must have the same number of entries.")
    n = t.size
    if n <= 1:
        return np.array([[4]])
    N = n - 1
    grids = np.meshgrid(*[np.arange(1, 5)] * N, indexing="ij")
    pathways = np.stack([g.ravel(order="F") for g in grids], axis=1)[:, ::-1]
    pathways = np.column_stack([pathways, np.full(pathways.shape[0], 4)])
    order = np.array([0, 0, 1, -1])
    co = order[pathways - 1]
    keep = []
    for p in range(pathways.shape[0]):
        nonzero = co[p][co[p] != 0]
        flipping = np.sum(np.abs(np.diff(nonzero)) == 2) > 0
        t0 = co[p] @ t
        if np.max(np.abs(inc)) > 0:
            incdec = np.array([np.sum(co[p][np.abs(inc) == d]) for d in range(1, np.max(np.abs(inc)) + 1)])
            refocusing = flipping and t0 == 0 and np.all(incdec == 0)
        else:
            refocusing = flipping and t0 == 0
        keep.append(bool(refocusing))
    return pathways[np.array(keep, dtype=bool)]


def _pathwayparser(ctp: np.ndarray, with_pulses: bool):
    PropL = np.array([[1, 3, 1, 3], [4, 2, 4, 2], [1, 3, 1, 3], [4, 2, 4, 2]])
    PropR = np.array([[1, 3, 3, 1], [4, 2, 2, 4], [4, 2, 2, 4], [1, 3, 3, 1]])
    ctp = np.asarray(ctp, dtype=int)
    free_l = PropL[ctp - 1, ctp - 1]
    free_r = PropR[ctp - 1, ctp - 1]
    if not with_pulses:
        return free_l, free_r, None, None
    before = np.column_stack([np.full(ctp.shape[0], 2), ctp[:, :-1]])
    puls_l = PropL[ctp - 1, before - 1]
    puls_r = PropR[ctp - 1, before - 1]
    return free_l, free_r, puls_l, puls_r


# ----------------------------------------------------------------------------- the peak kernel
def _bins(nu, dt, n):
    x = np.fmod(-nu * dt, 1.0) * n
    idx = np.where(x >= 0, np.floor(x), np.ceil(x))
    idx = np.where(idx < 0, idx + n, idx)
    return idx.astype(np.intp)


def _sf_peaks(scheme: int, buff: np.ndarray, dt, free_l, free_r, Ea, Eb, G, D, *blocks):  # noqa: C901
    """Add the peaks of one pathway to the frequency-domain buffer ``buff`` (in place), as sf_peaks."""
    E = {1: Ea, 2: Eb}
    dt = np.atleast_1d(np.asarray(dt, dtype=float))
    free_l = [int(v) for v in np.atleast_1d(free_l)]
    free_r = [int(v) for v in np.atleast_1d(free_r)]
    n = Ea.size
    n_free = {1: 1, 2: 2, 3: 2, 11: 2, 12: 3, 13: 3, 14: 3, 15: 4, 17: 4}.get(scheme)
    if n_free is None:
        raise NotImplementedError(f"Incrementation scheme {scheme} is not supported.")
    if len(free_l) != n_free or len(blocks) != 2 * (n_free - 1):
        raise ValueError("Wrong number of evolution intervals or transfer matrices.")
    El = [E[k] for k in free_l]
    Er = [E[k] for k in free_r]
    n1 = buff.shape[0]
    n2 = buff.shape[1] if buff.ndim == 2 else 1
    if scheme == 1:
        amp = D.T * G  # D[k,j]*G[j,k] as a (j,k) array
        nu = El[0][:, None] - Er[0][None, :]
        np.add.at(buff, _bins(nu, dt[0], n1), amp)
        return
    T1l, T1r = blocks[0], blocks[1]
    if scheme in (2, 3, 11):
        # Amp[i,j,k,l] = T1l[i,j] G[j,k] T1r[k,l] D[l,i]
        amp = np.einsum("ij,jk,kl,li->ijkl", T1l, G, T1r, D)
        if scheme == 2:
            nu = (El[1][:, None, None, None] - Er[1][None, None, None, :]) + (
                El[0][None, :, None, None] - Er[0][None, None, :, None]
            )
            np.add.at(buff, _bins(nu, dt[0], n1), amp)
        elif scheme == 3:
            nu = (
                -El[1][:, None, None, None]
                + El[0][None, :, None, None]
                - Er[0][None, None, :, None]
                + Er[1][None, None, None, :]
            )
            np.add.at(buff, _bins(nu, dt[0], n1), amp)
        else:
            nu1 = El[0][None, :, None, None] - Er[0][None, None, :, None] + 0 * amp.real
            nu2 = El[1][:, None, None, None] - Er[1][None, None, None, :] + 0 * amp.real
            np.add.at(buff, (_bins(nu1, dt[0], n1), _bins(nu2, dt[1], n2)), amp)
        return
    if scheme in (12, 14):
        T2l, T2r = blocks[2], blocks[3]
        # Amp = D[n,i] T2l[i,j] T1l[j,k] G[k,l] T1r[l,m] T2r[m,n]  (indices i..n)
        amp = np.einsum("ni,ij,jk,kl,lm,mn->ijklmn", D, T2l, T1l, G, T1r, T2r)
        Ei = lambda arr, axis: np.expand_dims(arr, tuple(a for a in range(6) if a != axis))  # noqa: E731
        if scheme == 12:
            nu1 = (Ei(El[0], 2) - Ei(Er[0], 3)) + (Ei(El[2], 0) - Ei(Er[2], 5))
            nu2 = Ei(El[1], 1) - Ei(Er[1], 4)
        else:
            nu1 = (Ei(El[0], 2) - Ei(Er[0], 3)) + (Ei(El[1], 1) - Ei(Er[1], 4))
            nu2 = Ei(El[2], 0) - Ei(Er[2], 5)
        nu1 = nu1 + 0 * amp.real
        nu2 = nu2 + 0 * amp.real
        np.add.at(buff, (_bins(nu1, dt[0], n1), _bins(nu2, dt[1], n2)), amp)
        return
    T1l, T1r = blocks[0], blocks[3]
    T2l, T3l, T2r, T3r = blocks[1], blocks[2], blocks[4], blocks[5]
    # Amp = D[p,i] T3l[i,j] T2l[j,k] T1l[k,l] G[l,m] T1r[m,n] T2r[n,o] T3r[o,p]
    for i in range(n):
        for p in range(n):
            left = np.einsum("j,jk,kl->jkl", T3l[i], T2l, T1l)  # (j,k,l)
            right = np.einsum("lm,mn,no,o->lmno", G, T1r, T2r, T3r[:, p])  # (l,m,n,o)
            amp = D[p, i] * np.einsum("jkl,lmno->jklmno", left, right)
            Ei = lambda arr, axis: np.expand_dims(arr, tuple(a for a in range(6) if a != axis))  # noqa: E731
            if scheme == 15:
                nu1 = (Ei(El[0], 2) - Ei(Er[0], 3)) + (El[3][i] - Er[3][p])
                nu2 = (Ei(El[1], 1) - Ei(Er[1], 4)) + (Ei(El[2], 0) - Ei(Er[2], 5))
            else:
                nu1 = (Ei(El[0], 2) - Ei(Er[0], 3)) + (Ei(El[1], 1) - Ei(Er[1], 4))
                nu2 = (Ei(El[2], 0) - Ei(Er[2], 5)) + (El[3][i] - Er[3][p])
            nu1 = nu1 + 0 * amp.real
            nu2 = nu2 + 0 * amp.real
            np.add.at(buff, (_bins(nu1, dt[0], n1), _bins(nu2, dt[1], n2)), amp)


def _sf_evolve(scheme, n_points, dt, inc_l, inc_r, Ea, Eb, G, D, *blocks):
    """Time-domain evolution of one pathway (sf_evolve) for the schemes [1], [1 1] and [1 2]."""
    E = {1: Ea, 2: Eb}
    dt = np.atleast_1d(np.asarray(dt, dtype=float))
    n_points = np.atleast_1d(np.asarray(n_points, dtype=int))
    inc_l = [int(v) for v in np.atleast_1d(inc_l)]
    inc_r = [int(v) for v in np.atleast_1d(inc_r)]
    if scheme == 1:
        rho = G.ravel(order="F")
        UUt = (
            np.exp(-2j * np.pi * E[inc_l[0]] * dt[0])[:, None]
            * np.conj(np.exp(-2j * np.pi * E[inc_r[0]] * dt[0]))[None, :]
        ).ravel(order="F")
        signal = np.zeros(n_points[0], dtype=complex)
        det = D.T.ravel(order="F")
        for k in range(n_points[0]):
            signal[k] = det @ rho
            rho = UUt * rho
        return signal
    if scheme == 2:
        T1l, T1r = blocks
        UUleft = np.exp(-2j * np.pi * E[inc_l[1]] * dt[0])[:, None] * np.exp(-2j * np.pi * E[inc_l[0]] * dt[0])[None, :]
        UUright = np.exp(2j * np.pi * E[inc_r[0]] * dt[0])[:, None] * np.exp(2j * np.pi * E[inc_r[1]] * dt[0])[None, :]
        signal = np.zeros(n_points[0], dtype=complex)
        T1l, T1r = T1l.copy(), T1r.copy()
        for k in range(n_points[0]):
            final = T1l @ G @ T1r
            signal[k] = np.sum(D.T * final)
            T1l = UUleft * T1l
            T1r = UUright * T1r
        return signal
    if scheme == 11:
        T1l, T1r = blocks
        UUt1 = (
            np.exp(-2j * np.pi * E[inc_l[0]] * dt[0])[:, None]
            * np.conj(np.exp(-2j * np.pi * E[inc_r[0]] * dt[0]))[None, :]
        )
        UUt2 = (
            np.exp(-2j * np.pi * E[inc_l[1]] * dt[1])[:, None]
            * np.conj(np.exp(-2j * np.pi * E[inc_r[1]] * dt[1]))[None, :]
        )
        signal = np.zeros((n_points[0], n_points[1]), dtype=complex)
        rho0 = G.copy()
        for k1 in range(n_points[0]):
            final = T1l @ rho0 @ T1r
            for k2 in range(n_points[1]):
                signal[k1, k2] = np.sum(D.T * final)
                final = UUt2 * final
            rho0 = UUt1 * rho0
        return signal
    raise NotImplementedError("Time-domain evolution for this experiment not supported.")


# ----------------------------------------------------------------------------- saffron
def _parse_custom_sequence(exp: dict):
    """Flip, phase, tp (us), delays and the increment scheme of a user-defined sequence of ideal pulses."""
    seq = exp["Sequence"]
    pulses = [s for s in seq if isinstance(s, dict)]
    delays = [s for s in seq if not isinstance(s, dict)]
    flags = [isinstance(s, dict) for s in seq]
    if any(flags[k] and flags[k + 1] for k in range(len(flags) - 1)):
        raise NotImplementedError(
            "Two or more pulses are not separated by an interpulse delay: the thyme engine is not ported."
        )
    n_pulses = len(pulses)
    flip = np.zeros(n_pulses)
    phase = np.ones(n_pulses)
    tp = np.zeros(n_pulses)
    for k, p in enumerate(pulses):
        if p.get("Type", "rectangular") not in ("rectangular", "rectangular/none"):
            raise NotImplementedError(
                "The fast algorithm only supports monochromatic rectangular pulses; "
                "shaped pulses need the thyme engine, which is not ported."
            )
        if "Flip" not in p:
            raise ValueError("Flip angles must be provided for every pulse.")
        flip[k] = float(p["Flip"]) / (np.pi / 2)
        if "Phase" in p:
            phase[k] = np.mod(phase[k] + float(p["Phase"]) / (np.pi / 2), 4)
        if p.get("tp", 0):
            tp[k] = float(p["tp"])
    t = np.array([float(d) for d in delays])
    out = {"Flip": flip, "Phase": phase, "tp": tp, "t": t}
    if "nPoints" in exp:
        n_points = np.atleast_1d(np.asarray(exp["nPoints"], dtype=int))
        n_dims = n_points.size
        if n_dims > 2:
            raise NotImplementedError("More than 2 indirect dimensions need the thyme engine, which is not ported.")
        inc = np.zeros(t.size, dtype=int)
        dt = np.zeros(n_dims)
        for d in range(1, n_dims + 1):
            rows = exp.get(f"Dim{d}")
            if rows is None:
                raise ValueError(f"You requested a {n_dims}-dimensional experiment, but Exp.Dim{d} is missing.")
            if isinstance(rows, (list, tuple)) and len(rows) == 2 and isinstance(rows[0], str):
                rows = [rows]
            for spec, value in rows:
                for item in spec.split(","):
                    item = item.strip()
                    kind, index = item[0], item[1:].split(".")[0]
                    if kind == "p":
                        raise NotImplementedError(
                            "Changing pulse parameters along an indirect dimension needs the thyme engine, "
                            "which is not ported."
                        )
                    inc[int(index) - 1] = d
                    if dt[d - 1] == 0:
                        dt[d - 1] = float(value)
                    elif dt[d - 1] != float(value):
                        raise NotImplementedError("Non-linear increments need the thyme engine, which is not ported.")
        out.update(Inc=inc, dt=dt, nPoints=n_points)
    return out


def saffron(system: dict, exp: dict, opt: dict | None = None) -> Spectrum:  # noqa: C901
    """Pulse EPR (ESEEM, HYSCORE, Mims ENDOR) by EasySpin's fast engine: ``Spectrum(x, signal, info)``.

    ``exp``: ``Sequence`` (``'2pESEEM'``, ``'3pESEEM'``, ``'4pESEEM'``, ``'HYSCORE'``, ``'MimsENDOR'``, or a
    list of pulse dicts ``{"Flip": angle, "Phase": phase, "tp": length}`` alternating with delays in us),
    ``Field`` (mT), ``dt``, ``nPoints``, ``tau``, ``T``, ``t1``, ``t2``, ``Dim1``/``Dim2`` (``[["d2", dt]]``),
    ``Filter``, ``Pathways``, ``mwFreq`` with ``ExciteWidth`` (orientation selection), ``Range``, ``tprf``
    (ENDOR), ``SampleFrame``, ``MolFrame``. ``opt``: ``GridSize``, ``GridSymmetry``, ``Nuclei``,
    ``ProductRule``, ``TimeDomain``, ``Expand``, ``EndorMethod``, ``OriThreshold``, ``Transitions``,
    ``Window``, ``ZeroFillFactor``, ``nOffsets``, ``lwOffset``. ``x`` is the time axis (a tuple of two for
    2-D experiments) or the rf axis; ``info`` holds ``td``, ``fd`` and the frequency axis.
    """
    if isinstance(system, (list, tuple)):
        return compisoloop(saffron, system, exp, opt, needs_range=False)
    opt = dict(opt or {})
    exp = dict(exp)
    if any(not any(ch.isdigit() for ch in n) for n in nucstring2list(system.get("Nucs", ""))):
        raise NotImplementedError(
            "Isotope mixtures (elements without mass number in Sys.Nucs) are not ported; give isotopes."
        )
    if opt.get("SimulationMode", "fast") != "fast":
        raise NotImplementedError("Opt.SimulationMode='thyme' (shaped pulses, explicit propagation) is not ported.")
    for key in ("DetWindow", "ResonatorFrequency", "ResonatorQL", "FrequencyResponse", "DetSequence", "DetOperator"):
        if key in exp:
            raise NotImplementedError(f"Exp.{key} needs the thyme engine, which is not ported.")
    for key in ("initState", "eqState"):
        if system.get(key) is not None:
            raise ValueError(f"Sys.{key} is specific to spidyan and cannot be used with saffron. Please remove.")
    sys = validatespinsys(system)
    if np.any(sys["gStrain"]) or np.any(sys["AStrain"]) or np.any(sys["DStrain"]):
        raise ValueError("saffron does not support Sys.gStrain, Sys.AStrain, or Sys.DStrain.")
    if exp.get("Temperature") is not None:
        raise ValueError("Exp.Temperature is not supported for pulse EPR simulations.")
    if exp.get("lightBeam"):
        raise ValueError("Photoselection (via Exp.lightBeam) is not supported.")
    if "Field" not in exp:
        raise ValueError("Exp.Field is missing. Give a magnetic field in mT.")
    if "Sequence" not in exp:
        raise ValueError("Exp.Sequence is missing. Provide a pre-defined or user-defined pulse sequence.")
    if exp.get("Ordering") is not None:
        raise NotImplementedError("Partially ordered samples are not implemented in saffron.")
    if exp.get("SampleRotation") is not None:
        raise NotImplementedError("Exp.SampleRotation is not ported yet.")
    if "HStrain" in exp:
        raise ValueError(
            "You gave Exp.HStrain, but it should be Sys.HStrain (in the system, not the experiment structure)."
        )
    field = float(exp["Field"])
    if sys["nElectrons"] > 1:
        raise NotImplementedError(
            "Fast mode does not work for more than one electron spin; the thyme engine is not ported."
        )
    if np.any(sys["n"] != 1):
        raise ValueError("Fast mode does not work for equivalent nuclei.")
    if sys["nNuclei"] > 1 and np.size(sys.get("nn", [])) and np.any(np.asarray(sys["nn"]) != 0):
        raise ValueError("Fast mode does not work in the presence of nuclear-nuclear couplings.")
    if "nKnots" in opt:
        raise ValueError("Options.nKnots is obsolete. Use Options.GridSize instead, e.g. Options.GridSize = 91.")
    grid_size = np.atleast_1d(np.asarray(opt.get("GridSize", 31), dtype=int))
    if grid_size.size > 1:
        raise ValueError("Only one number allowed in Opt.GridSize. saffron does not support interpolation.")
    grid_size = int(grid_size[0])
    if grid_size < 7:
        raise ValueError(f"Opt.GridSize must be at least 7. You gave {grid_size}.")
    T1 = float(np.atleast_1d(system.get("T1", 0))[0]) if system.get("T1") is not None else 0.0
    T2 = float(np.atleast_1d(system.get("T2", 0))[0]) if system.get("T2") is not None else 0.0
    if np.size(system.get("T1", 0)) > 1 or np.size(system.get("T2", 0)) > 1:
        raise ValueError(
            "The fast algorithm does not support transition selective relaxation times, "
            "please provide a single value for Sys.T1 and Sys.T2."
        )
    T1 = np.inf if T1 == 0 else T1
    T2 = np.inf if T2 == 0 else T2
    if T1 <= 0 or T2 <= 0:
        raise ValueError("T1 and T2 in Sys.T1 and Sys.T2 must be positive, in microseconds.")

    # Orientations
    crystal = exp.get("MolFrame") is not None or bool(exp.get("CrystalSymmetry"))
    if not crystal:
        symmetry = opt.get("GridSymmetry") or None
        frame = None
        if symmetry is None:
            symmetry, frame = hamsymm(sys)
        if opt.get("GridFrame") is not None:
            frame = np.asarray(opt["GridFrame"], dtype=float)
        if frame is None:
            frame = np.eye(3)
        grid = sphgrid(symmetry, grid_size)
        phi, theta = vec2ang(frame @ grid.vecs)
        orientations = np.column_stack([phi, theta, np.zeros_like(phi)])
        weights = grid.weights
    else:
        exp_cr = {
            "MolFrame": exp.get("MolFrame") if exp.get("MolFrame") is not None else [0, 0, 0],
            "SampleFrame": exp.get("SampleFrame"),
            "CrystalSymmetry": exp.get("CrystalSymmetry"),
        }
        orientations, _, _, _ = _crystal_orientations(exp_cr, {})
        weights = np.full(orientations.shape[0], 4 * np.pi)
    if not crystal and exp.get("SampleFrame") is not None and np.size(exp["SampleFrame"]):
        sf = np.atleast_2d(np.asarray(exp["SampleFrame"], dtype=float))
        if sf.shape[0] > 1:
            raise ValueError("For disordered samples, only a single sample orientation (Exp.SampleFrame) can be used.")
    n_orientations = orientations.shape[0]

    # Pulse sequence
    predefined = isinstance(exp["Sequence"], str)
    is_endor = False
    custom = None
    if predefined:
        if "Filter" in exp:
            raise ValueError("Exp.Filter can only be used with custom sequences.")
        if exp["Sequence"] not in _EXPERIMENTS:
            raise ValueError(f"Exp.Sequence '{exp['Sequence']}' not recognized.")
        experiment_id = _EXPERIMENTS[exp["Sequence"]]
        if "tp" in exp and np.any(np.asarray(exp["tp"]) != 0):
            raise ValueError("You cannot use predefined sequences (Exp.Sequence) with real pulses (Exp.tp).")
        n_intervals, n_dims, scheme, n_pathways, pathway_prefactor, tau_required = {
            1: (2, 1, 2, 1, np.array([0.5]), False),
            2: (3, 1, 1, 2, np.array([1 / 8, 1 / 8]), True),
            3: (4, 1, 2, 2, np.array([-1 / 8, -1 / 8]), True),
            4: (4, 2, 11, 2, np.array([-1 / 8, -1 / 8]), True),
            5: (3, 1, 0, 2, np.array([1 / 8, 1 / 8]), True),
        }[experiment_id]
        is_endor = experiment_id == 5
        tau = exp.get("tau", 0 if not tau_required else None)
        if tau is None:
            raise ValueError("Exp.tau is missing.")
        if np.size(tau) != 1:
            raise ValueError("Exp.tau must contain a single value (in us).")
        tau = float(np.asarray(tau).ravel()[0])
        if tau_required and tau == 0:
            raise ValueError("Exp.tau must be larger than 0.")
        tp = np.zeros(n_intervals)
        ideal = np.ones(n_intervals, bool)
    else:
        experiment_id = -1
        if not np.isinf(T1) or not np.isinf(T2):
            raise ValueError("Relaxation times T1 and T2 for custom sequences not supported.")
        custom = _parse_custom_sequence(exp)
        flip, phase, tp, t = custom["Flip"], custom["Phase"], custom["tp"], custom["t"]
        n_intervals = flip.size
        if "Inc" not in custom:
            raise ValueError("Fast sequence engine: the incrementation scheme (Exp.nPoints with Exp.Dim1) is missing.")
        inc = custom["Inc"]
        if inc.size != n_intervals:
            raise ValueError("Exp.Inc must contain the same number of elements as Exp.Flip (one delay per pulse).")
        inc_scheme = tuple(int(v) for v in inc[inc != 0])
        if inc_scheme not in _INC_SCHEMES:
            raise ValueError("Unsupported incrementation scheme in Exp.Inc.")
        scheme = _INC_SCHEMES[inc_scheme]
        n_dims = int(np.max(np.abs(inc)))
        if t.size != n_intervals:
            raise ValueError("Exp.t must contain the same number of elements as Exp.Flip.")
        ideal = tp == 0
    real_pulse = ~ideal
    if not is_endor:
        dt = (
            np.atleast_1d(np.asarray(custom["dt"] if custom is not None else exp.get("dt"), dtype=float))
            if (custom is not None or "dt" in exp)
            else None
        )
        if dt is None:
            raise ValueError("Exp.dt is missing.")
        if dt.size == 1:
            dt = np.repeat(dt, n_dims)
        elif dt.size != n_dims:
            raise ValueError(f"Exp.dt needs either 1 or {n_dims} elements, one per dimension. You gave {dt.size}.")
    n_points_in = custom["nPoints"] if custom is not None else exp.get("nPoints")
    if n_points_in is None:
        n_points = np.array([1001]) if is_endor else (np.array([512]) if n_dims == 1 else np.array([256, 256]))
    else:
        n_points = np.atleast_1d(np.asarray(n_points_in, dtype=int))
        if n_points.size == 1:
            n_points = np.repeat(n_points, n_dims)
        elif n_points.size != n_dims:
            raise ValueError(
                f"Exp.nPoints needs either 1 or {n_dims} elements, one per dimension. You gave {n_points.size}."
            )

    # Pathways of a user-defined sequence
    if not predefined:
        for k in range(n_intervals):
            d = abs(inc[k])
            t_range = (
                np.array([t[k], t[k]])
                if d == 0
                else t[k] + np.sign(inc[k]) * np.array([0, (n_points[d - 1] - 1) * dt[d - 1]])
            )
            if np.any(t_range < 0):
                raise ValueError(f"Negative delay after pulse {k + 1}.")
        if exp.get("Pathways") is not None:
            code = {"a": 1, "b": 2, "+": 3, "-": 4}
            pathway_list = np.array(
                [[code[c] for c in row] for row in np.atleast_1d(exp["Pathways"])], dtype=int
            ).reshape(-1, n_intervals)
        else:
            pathway_list = sf_pathways(t, inc)
        if pathway_list.shape[0] == 0:
            raise ValueError("Sorry, no focused echo with this sequence and timings.")
        filt = exp.get("Filter")
        if filt:
            if not isinstance(filt, str) or len(filt) != n_intervals:
                raise ValueError(
                    f"Exp.Filter must be a string of {n_intervals} characters "
                    "from '0', '1', 'a', 'b', '+', '-' and '.'."
                )
            for k, ch in enumerate(filt):
                col = pathway_list[:, k]
                keep = {
                    "0": (col == 1) | (col == 2),
                    "1": (col == 3) | (col == 4),
                    "a": col == 1,
                    "b": col == 2,
                    "+": col == 3,
                    "-": col == 4,
                    ".": col != 0,
                    "*": col != 0,
                }.get(ch)
                if keep is None:
                    raise ValueError(f"Exp.Filter({k + 1}) = '{ch}' is invalid.")
                pathway_list = pathway_list[keep]
            if pathway_list.shape[0] == 0:
                raise ValueError(
                    "Exp.Filter is too restrictive: no echo at detection point left after applying the filter."
                )
        free_l, free_r, puls_l, puls_r = _pathwayparser(pathway_list, True)
        n_pathways = pathway_list.shape[0]
        pathway_prefactor = np.ones(n_pathways, dtype=complex)
        for ip in range(n_intervals):
            if not ideal[ip]:
                continue
            theta_ = flip[ip] * np.pi / 2
            c, s = np.cos(theta_ / 2), np.sin(theta_ / 2)
            for q in range(n_pathways):
                pl = {1: c, 2: c, 3: -1j * s * (-1j) ** phase[ip], 4: -1j * s * (1j) ** phase[ip]}[int(puls_l[q, ip])]
                pr = {1: c, 2: c, 3: 1j * s * (1j) ** phase[ip], 4: 1j * s * (-1j) ** phase[ip]}[int(puls_r[q, ip])]
                pathway_prefactor[q] = pl * pathway_prefactor[q] * pr
        keep = np.abs(pathway_prefactor) >= 1e-6
        pathway_list, free_l, free_r, puls_l, puls_r, pathway_prefactor = (
            pathway_list[keep],
            free_l[keep],
            free_r[keep],
            puls_l[keep],
            puls_r[keep],
            pathway_prefactor[keep],
        )
        n_pathways = pathway_list.shape[0]
        idx_inc_l = free_l[:, inc != 0]
        idx_inc_r = free_r[:, inc != 0]
    if is_endor:
        lw_endor = np.atleast_1d(np.asarray(sys["lwEndor"], dtype=float))
        if np.all(lw_endor == 0):
            raise ValueError("Positive ENDOR line width in Sys.lwEndor needed.")
        lw_endor = float(lw_endor[0])
        if "Range" not in exp:
            raise ValueError("Frequency range (Exp.Range) must be given for an ENDOR experiment.")
        tprf = float(exp.get("tprf", 20))
    T_delay = float(exp.get("T", 0))
    t1_delay, t2_delay = float(exp.get("t1", 0)), float(exp.get("t2", 0))
    orientation_selection = "mwFreq" in exp
    if orientation_selection and "ExciteWidth" not in exp:
        raise ValueError(
            "Orientation selection: Exp.ExciteWidth (in MHz) missing. It should be about the inverse of the "
            "first pulse length (100MHz for 10ns). If you don't want orientation selection, set it to a very "
            "large number (1e6) or remove the microwave frequency."
        )
    if "ExciteWidth" in exp and "mwFreq" not in exp:
        raise ValueError("Exp.ExciteWidth is given, but Exp.mwFreq is missing. Please give Exp.mwFreq.")
    excite_width = float(exp.get("ExciteWidth", np.inf))
    mw_freq = float(exp.get("mwFreq", 0))

    # Nuclei
    n_nuc = sys["nNuclei"]
    if "Nuclei" in opt:
        shf = [int(k) for k in np.atleast_1d(opt["Nuclei"])]
        if not shf:
            raise ValueError("Opt.Nuclei must contain the indices of nuclei to include in the simulation.")
        if any(k > n_nuc or k < 0 for k in shf):
            raise ValueError("Index in Opt.Nuclei out of range.")
    else:
        shf = list(range(1, n_nuc + 1))
        if shf and np.all(ideal) and "ExciteWidth" in exp:
            if sys["fullA"]:
                strong = [np.abs(sys["A"][3 * k : 3 * k + 3, :3]).max() > excite_width for k in range(n_nuc)]
            else:
                strong = [np.abs(sys["A"][k, :3]).max() > excite_width for k in range(n_nuc)]
            shf = [k + 1 for k in range(n_nuc) if not strong[k]]
    two_manifolds = sys["nElectrons"] == 1 and sys["S"][0] == 0.5 and len(shf) == n_nuc
    product_rule = bool(opt.get("ProductRule", 0))
    if n_nuc == 1:
        product_rule = False
    if is_endor:
        endor_method = opt.get("EndorMethod")
        if endor_method is None:
            endor_method = 0 if (len(shf) == 1 or product_rule) else 1
        endor_method = int(endor_method)
        if endor_method not in (0, 1, 2):
            raise ValueError("Unknown setting for Opt.EndorMethod. Must be 0, 1, or 2.")
        if endor_method == 0 and len(shf) > 1 and not product_rule:
            raise ValueError("Opt.EndorMethod=0 gives incorrect results for multiple nuclei.")
    expand = int(opt.get("Expand", 4 if n_dims == 1 else 2))
    if expand < 0 or expand > 8:
        raise ValueError("Opt.Expand must be an integer between 0 and 8.")
    if np.any(real_pulse) and product_rule:
        raise ValueError("saffron: Cannot apply product rule and real pulses at the same time.")
    ori_threshold = float(opt.get("OriThreshold", 0.005))
    window = opt.get("Window", "ham+")
    zero_fill = int(opt.get("ZeroFillFactor", 2))
    time_domain = bool(opt.get("TimeDomain", 0))
    if not predefined and product_rule:
        raise NotImplementedError("Product rule for user-defined experiments not implemented.")

    # Core electronic Hamiltonian
    if not two_manifolds:
        core = nucspinrmv({k: v for k, v in system.items() if k != "processed"}, shf)
        core["lw"] = 0
        core.pop("A_", None)
        core_sys = validatespinsys(core)
        H0, mux, muy, muz = ham(core_sys)
        Sx = sop(core_sys["Spins"], "x1")
        Sy = sop(core_sys["Spins"], "y1")
        Sz = sop(core_sys["Spins"], "z1")
    # Nuclear sub-Hamiltonians
    I = sys["I"][[k - 1 for k in shf]] if shf else np.zeros(0)
    nuc_hams: list[dict] = []
    if shf:
        n_spaces = len(shf) if product_rule else 1
        nuc_hams = [
            {k: 0.0 for k in ("Hnzx", "Hnzy", "Hnzz", "Hhfx", "Hhfy", "Hhfz", "Hnq", "Ix", "Iy", "Iz")}
            for _ in range(n_spaces)
        ]
        for ii, nuc in enumerate(shf):
            if product_rule:
                ops = [sop([I[ii]], f"{c}1") for c in "xyz"]
                target = nuc_hams[ii]
            else:
                ops = [sop(I, f"{c}{ii + 1}") for c in "xyz"]
                target = nuc_hams[0]
            Ix, Iy, Iz = ops
            target["Ix"] = target["Ix"] + Ix
            target["Iy"] = target["Iy"] + Iy
            target["Iz"] = target["Iz"] + Iz
            pre = -sys["gn"][nuc - 1] * nmagn / 1e3 / planck / 1e6
            target["Hnzx"] = target["Hnzx"] + pre * Ix
            target["Hnzy"] = target["Hnzy"] + pre * Iy
            target["Hnzz"] = target["Hnzz"] + pre * Iz
            if sys["fullA"]:
                A = sys["A"][3 * (nuc - 1) : 3 * nuc, :3]
            else:
                R = erot(sys["AFrame"][nuc - 1, :3]).T
                A = R @ np.diag(sys["A"][nuc - 1, :3]) @ R.T
                A = (A + A.T) / 2
            target["Hhfx"] = target["Hhfx"] + A[0, 0] * Ix + A[0, 1] * Iy + A[0, 2] * Iz
            target["Hhfy"] = target["Hhfy"] + A[1, 0] * Ix + A[1, 1] * Iy + A[1, 2] * Iz
            target["Hhfz"] = target["Hhfz"] + A[2, 0] * Ix + A[2, 1] * Iy + A[2, 2] * Iz
            if sys["I"][nuc - 1] >= 1:
                if sys["fullQ"]:
                    Q = sys["Q"][3 * (nuc - 1) : 3 * nuc, :3]
                else:
                    R = erot(sys["QFrame"][nuc - 1, :3]).T
                    Q = R @ np.diag(sys["Q"][nuc - 1, :3]) @ R.T
                    Q = (Q + Q.T) / 2
                target["Hnq"] = (
                    target["Hnq"]
                    + Ix @ (Q[0, 0] * Ix + Q[0, 1] * Iy + Q[0, 2] * Iz)
                    + Iy @ (Q[1, 0] * Ix + Q[1, 1] * Iy + Q[1, 2] * Iz)
                    + Iz @ (Q[2, 0] * Ix + Q[2, 1] * Iy + Q[2, 2] * Iz)
                )
    n_spaces = len(nuc_hams)

    # Orientation pre-selection for S = 1/2 from g alone
    if orientation_selection and two_manifolds:
        zlab = ang2vec(orientations[:, 0], orientations[:, 1])
        gmat = sys["g"][:3, :3] if sys["fullg"] else np.diag(sys["g"][0, :3])
        geff = np.linalg.norm(gmat @ zlab, axis=0)
        nu = geff * bmagn * field / 1e3 / planck / 1e6
        lw = np.diag(sys["HStrain"][:3]) @ zlab
        lw2 = np.sum(lw**2, axis=0) + excite_width**2
        g_ori_weight = np.exp(-((nu - 1000 * mw_freq) ** 2) / lw2)
    if two_manifolds:
        gmat = (
            sys["g"][:3, :3]
            if sys["fullg"]
            else (lambda R: R @ np.diag(sys["g"][0, :3]) @ R.T)(erot(sys["gFrame"][0]).T)
        )

    # Buffers
    if is_endor:
        rng = np.asarray(exp["Range"], dtype=float).ravel()
        rf = np.linspace(rng[0], rng[1], n_points[0])
        endorspc = np.zeros(n_points[0], dtype=complex)
    else:
        n_points_f = (2**expand) * n_points
        shape = tuple(int(v) for v in n_points_f)
        if not time_domain:
            if product_rule:
                pathway_buff = [[np.zeros(shape, dtype=complex) for _ in range(n_spaces)] for _ in range(n_pathways)]
            else:
                buff = np.zeros(shape, dtype=complex)
        total_td = np.zeros(tuple(int(v) for v in n_points), dtype=complex)
    if np.any(real_pulse):
        n_offsets = int(opt.get("nOffsets", 291))
        lw_offset = float(opt.get("lwOffset", 100))
        offsets = np.linspace(-1, 1, n_offsets) * lw_offset * 2
        offset_weight = np.exp(-((offsets / lw_offset) ** 2))
        offset_weight = offset_weight / offset_weight.sum()
    else:
        offsets, offset_weight = np.zeros(1), np.ones(1)
    n_skipped = 0

    for iOri in range(n_orientations):
        R = erot(orientations[iOri])
        yLab, zLab = R[1], R[2]
        if two_manifolds:
            ori_weight = g_ori_weight[iOri] if orientation_selection else 1.0
            if ori_weight < ori_threshold:
                n_skipped += 1
                continue
            q = gmat.T @ zLab
            q = q / np.linalg.norm(q)
            manifold_S = {1: -0.5 * q, 2: 0.5 * q}
            transitions = np.array([[1, 2]])
            ori_weights_t = np.array([ori_weight])
            involved = [1, 2]
        else:
            muzL = zLab[0] * mux + zLab[1] * muy + zLab[2] * muz
            H = H0 - field * muzL
            eE, eV = np.linalg.eigh(H)
            SyLab = yLab[0] * Sx + yLab[1] * Sy + yLab[2] * Sz
            SyLab = np.abs(eV.conj().T @ SyLab @ eV)
            max_syy = SyLab.max()
            if orientation_selection:
                dE = eE[:, None] - eE[None, :] - mw_freq * 1e3
                SyLab = SyLab * np.exp(-((dE / excite_width) ** 2))
            SyLab[SyLab < ori_threshold * max_syy] = 0
            if opt.get("Transitions") is not None:
                rmv = np.ones_like(SyLab)
                for tr in np.atleast_2d(np.asarray(opt["Transitions"], dtype=int)):
                    rmv[tr[0] - 1, tr[1] - 1] = 0
                    rmv[tr[1] - 1, tr[0] - 1] = 0
                SyLab[rmv != 0] = 0
            # [v,u,w] = find(tril(SyLab,-1)): rows v > columns u in column-major order
            low = np.tril(SyLab, -1)
            u_idx, v_idx = np.nonzero(low.T)
            ori_weights_t = low.T[u_idx, v_idx]
            transitions = np.column_stack([u_idx, v_idx]) + 1
            if transitions.shape[0] == 0:
                n_skipped += 1
                continue
            involved = sorted(set(transitions.ravel().tolist()))
            manifold_S = {
                m: np.real(
                    np.array(
                        [
                            eV[:, m - 1].conj() @ Sx @ eV[:, m - 1],
                            eV[:, m - 1].conj() @ Sy @ eV[:, m - 1],
                            eV[:, m - 1].conj() @ Sz @ eV[:, m - 1],
                        ]
                    )
                )
                for m in involved
            }
        n_transitions = transitions.shape[0]
        manifold_V: dict = {}
        manifold_E: dict = {}
        for s in range(n_spaces):
            nh = nuc_hams[s]
            Hnuc = field * (zLab[0] * nh["Hnzx"] + zLab[1] * nh["Hnzy"] + zLab[2] * nh["Hnzz"]) + nh["Hnq"]
            for m in involved:
                S = manifold_S[m]
                Hm = Hnuc + S[0] * nh["Hhfx"] + S[1] * nh["Hhfy"] + S[2] * nh["Hhfz"]
                EE, VV = np.linalg.eigh((Hm + Hm.conj().T) / 2)
                manifold_V[(m, s)] = VV
                manifold_E[(m, s)] = np.real(EE)
        for iT in range(n_transitions):
            b_, a_ = transitions[iT, 0], transitions[iT, 1]
            pathway_td = [[None] * n_spaces for _ in range(n_pathways)] if product_rule else None
            for s in range(n_spaces):
                Ea, Eb = manifold_E[(a_, s)], manifold_E[(b_, s)]
                Ma, Mb = manifold_V[(a_, s)], manifold_V[(b_, s)]
                M = Ma.conj().T @ Mb
                Mt = M.conj().T
                n_states = Ea.size
                if is_endor:
                    IyLab = yLab[0] * nuc_hams[s]["Ix"] + yLab[1] * nuc_hams[s]["Iy"] + yLab[2] * nuc_hams[s]["Iz"]
                for iOff in range(offsets.size):
                    pulse_sub: dict = {}
                    for k in range(n_intervals):
                        if real_pulse[k]:
                            nu1 = (np.pi / 2 * flip[k]) / tp[k] / 2 / np.pi
                            Hpulse = np.block(
                                [
                                    [np.diag(Ea + offsets[iOff] / 2), M * nu1 / 2j],
                                    [-Mt * nu1 / 2j, np.diag(Eb - offsets[iOff] / 2)],
                                ]
                            )
                            P = expm(-2j * np.pi * tp[k] * Hpulse)
                            ia, ib = slice(0, n_states), slice(n_states, 2 * n_states)
                            pulse_sub[k] = {1: P[ia, ia], 2: P[ib, ib], 3: P[ia, ib], 4: P[ib, ia]}
                    if is_endor or not product_rule:
                        prefactor = weights[iOri] * ori_weights_t[iT]
                    else:
                        prefactor = 1.0
                    if not is_endor and not time_domain and product_rule:
                        for ip in range(n_pathways):
                            pathway_buff[ip][s][...] = 0
                    prefactor = prefactor * offset_weight[iOff]
                    if not predefined:
                        for ip in range(n_pathways):
                            blocks_l, blocks_r = [], []
                            left = prefactor * pathway_prefactor[ip]
                            right = 1.0
                            i_block = 0
                            G = None
                            for k in range(n_intervals):
                                if ideal[k]:
                                    pl, pr = int(puls_l[ip, k]), int(puls_r[ip, k])
                                    if pl == 3:
                                        left = M @ left if not np.isscalar(left) else M * left
                                    elif pl == 4:
                                        left = Mt @ left if not np.isscalar(left) else Mt * left
                                    if pr == 3:
                                        right = right @ Mt if not np.isscalar(right) else right * Mt
                                    elif pr == 4:
                                        right = right @ M if not np.isscalar(right) else right * M
                                else:
                                    PL = pulse_sub[k][int(puls_l[ip, k])]
                                    PR = pulse_sub[k][int(puls_r[ip, k])].conj().T
                                    left = PL @ left if not np.isscalar(left) else PL * left
                                    right = right @ PR if not np.isscalar(right) else right * PR
                                tk = t[k]
                                if tk > 0:
                                    El_ = Ea if free_l[ip, k] == 1 else Eb
                                    Er_ = Ea if free_r[ip, k] == 1 else Eb
                                    left = np.exp(-2j * np.pi * El_ * tk)[:, None] * (
                                        left if not np.isscalar(left) else left * np.eye(n_states)
                                    )
                                    right = (right if not np.isscalar(right) else right * np.eye(n_states)) * np.exp(
                                        2j * np.pi * Er_ * tk
                                    )[None, :]
                                if inc[k] != 0:
                                    if i_block == 0:
                                        G = (
                                            left @ right
                                            if not (np.isscalar(left) or np.isscalar(right))
                                            else np.asarray(left * right)
                                        )
                                        if np.isscalar(G) or G.ndim == 0:
                                            G = G * np.eye(n_states)
                                    else:
                                        blocks_l.append(np.eye(n_states) if np.isscalar(left) else left)
                                        blocks_r.append(np.eye(n_states) if np.isscalar(right) else right)
                                    i_block += 1
                                    left, right = 1.0, 1.0
                            if inc[-1] != 0:
                                Dm = M
                            else:
                                Dm = (
                                    (right if not np.isscalar(right) else right * np.eye(n_states))
                                    @ M
                                    @ (left if not np.isscalar(left) else left * np.eye(n_states))
                                )
                            args = (idx_inc_l[ip], idx_inc_r[ip], Ea, Eb, G, Dm, *blocks_l, *blocks_r)
                            if time_domain:
                                total_td = total_td + _sf_evolve(scheme, n_points, dt, *args)
                            else:
                                _sf_peaks(scheme, buff, dt, *args)
                    elif experiment_id == 5:
                        Q_ = np.exp(-2j * np.pi * Ea * tau)[:, None] * np.conj(np.exp(-2j * np.pi * Eb * tau))[None, :]
                        G_ = prefactor * Q_ * M
                        D_ = np.conj(Q_) * M
                        G1, D1 = G_ @ Mt, D_ @ Mt
                        G2, D2 = Mt @ G_, Mt @ D_
                        if T_delay != 0:
                            qa = np.exp(-2j * np.pi * Ea * T_delay)
                            G1 = (qa[:, None] * np.conj(qa)[None, :]) * G1
                            qb = np.exp(-2j * np.pi * Eb * T_delay)
                            G2 = (qb[:, None] * np.conj(qb)[None, :]) * G2
                        G1 = np.diag(np.diag(G1))
                        G2 = np.diag(np.diag(G2))
                        trace1 = np.trace(G1 @ D1)
                        trace2 = np.trace(G2 @ D2)
                        nu1m = np.abs(Ea[:, None] - Ea[None, :])
                        nu2m = np.abs(Eb[:, None] - Eb[None, :])
                        Iy1 = Ma.conj().T @ IyLab @ Ma
                        Iy2 = Mb.conj().T @ IyLab @ Mb
                        gamma_bw = (1 / tprf) / (2 * np.sqrt(np.log(2)))
                        theta_rf = np.pi
                        gam = lw_endor / np.sqrt(2 * np.log(2))
                        pre = np.sqrt(2 / np.pi) / gam
                        if endor_method == 2:
                            for irf, f in enumerate(rf):
                                BW1 = np.exp(-(((nu1m - f) / gamma_bw) ** 2))
                                BW2 = np.exp(-(((nu2m - f) / gamma_bw) ** 2))
                                if np.any(BW1 > 0.01):
                                    P = expm(-1j * theta_rf * (Iy1 * BW1))
                                    G1_ = np.diag(np.diag(P @ G1 @ P.conj().T))
                                    endorspc[irf] += trace1 - np.trace(G1_ @ D1)
                                if np.any(BW2 > 0.01):
                                    P = expm(-1j * theta_rf * (Iy2 * BW2))
                                    G2_ = np.diag(np.diag(P @ G2 @ P.conj().T))
                                    endorspc[irf] += trace2 - np.trace(G2_ @ D2)
                        elif endor_method == 0:
                            for j in range(n_states - 1):
                                i = j + 1
                                G1_ = G1.copy()
                                G1_[i, i], G1_[j, j] = G1[j, j], G1[i, i]
                                G2_ = G2.copy()
                                G2_[i, i], G2_[j, j] = G2[j, j], G2[i, i]
                                ampl = [trace1 - np.trace(G1_ @ D1), trace2 - np.trace(G2_ @ D2)]
                                freq = [Ea[i] - Ea[j], Eb[i] - Eb[j]]
                                endorspc += ampl[0] * pre * np.exp(-2 * ((rf - freq[0]) / gam) ** 2)
                                endorspc += ampl[1] * pre * np.exp(-2 * ((rf - freq[1]) / gam) ** 2)
                        else:
                            for i in range(n_states):
                                for j in range(i + 1, n_states):
                                    f1_, f2_ = nu1m[i, j], nu2m[i, j]
                                    BW1 = np.exp(-(((nu1m - f1_) / gamma_bw) ** 2))
                                    BW2 = np.exp(-(((nu2m - f2_) / gamma_bw) ** 2))
                                    Pa = expm(-1j * theta_rf * (Iy1 * BW1))
                                    Pb = expm(-1j * theta_rf * (Iy2 * BW2))
                                    G1_ = np.diag(np.diag(Pa @ G1 @ Pa.conj().T))
                                    G2_ = np.diag(np.diag(Pb @ G2 @ Pb.conj().T))
                                    sig1 = trace1 - np.trace(G1_ @ D1)
                                    sig2 = trace2 - np.trace(G2_ @ D2)
                                    endorspc += sig1 * pre * np.exp(-(((rf - f1_) / gam) ** 2))
                                    endorspc += sig2 * pre * np.exp(-(((rf - f2_) / gam) ** 2))
                    else:
                        Q_ = np.exp(-2j * np.pi * Ea * tau)[:, None] * np.conj(np.exp(-2j * np.pi * Eb * tau))[None, :]
                        if experiment_id == 1:
                            G = prefactor * M
                            Dm = M
                            if tau > 0:
                                G = Q_ * G
                                Dm = np.conj(Q_) * Dm
                            calls = [((1, 2), (2, 1), G, Dm, Mt, Mt)]
                        else:
                            G_ = prefactor * Q_ * M
                            D_ = np.conj(Q_) * M
                            if experiment_id == 4:
                                G1, G2 = G_ @ Mt, Mt @ G_
                                D1, D2 = Mt @ D_, D_ @ Mt
                                if t1_delay != 0:
                                    qa = np.exp(-2j * np.pi * t1_delay * Ea)
                                    G1 = (qa[:, None] * np.conj(qa)[None, :]) * G1
                                    qb = np.exp(-2j * np.pi * t1_delay * Eb)
                                    G2 = (qb[:, None] * np.conj(qb)[None, :]) * G2
                                if t2_delay != 0:
                                    qb = np.exp(2j * np.pi * t2_delay * Eb)
                                    D1 = (qb[:, None] * np.conj(qb)[None, :]) * D1
                                    qa = np.exp(2j * np.pi * t2_delay * Ea)
                                    D2 = (qa[:, None] * np.conj(qa)[None, :]) * D2
                                calls = [((1, 2), (1, 2), G1, D1, Mt, M), ((2, 1), (2, 1), G2, D2, M, Mt)]
                            elif experiment_id == 3:
                                G1, D1 = G_ @ Mt, Mt @ D_
                                G2, D2 = Mt @ G_, D_ @ Mt
                                if T_delay != 0:
                                    qa = np.exp(-2j * np.pi * T_delay * Ea)
                                    G1 = (qa[:, None] * np.conj(qa)[None, :]) * G1
                                    qb = np.exp(-2j * np.pi * T_delay * Eb)
                                    G2 = (qb[:, None] * np.conj(qb)[None, :]) * G2
                                    qb = np.exp(2j * np.pi * T_delay * Eb)
                                    D1 = (qb[:, None] * np.conj(qb)[None, :]) * D1
                                    qa = np.exp(2j * np.pi * T_delay * Ea)
                                    D2 = (qa[:, None] * np.conj(qa)[None, :]) * D2
                                calls = [((1, 2), (1, 2), G1, D1, Mt, M), ((2, 1), (2, 1), G2, D2, M, Mt)]
                            else:
                                G1, D1 = G_ @ Mt, D_ @ Mt
                                G2, D2 = Mt @ G_, Mt @ D_
                                if T_delay != 0:
                                    qa = np.exp(-2j * np.pi * T_delay * Ea)
                                    G1 = (qa[:, None] * np.conj(qa)[None, :]) * G1
                                    qb = np.exp(-2j * np.pi * T_delay * Eb)
                                    G2 = (qb[:, None] * np.conj(qb)[None, :]) * G2
                                calls = [((1,), (1,), G1, D1), ((2,), (2,), G2, D2)]
                        for ip, call in enumerate(calls):
                            il, ir, Gc, Dc, *bl = call
                            if time_domain:
                                td_ = _sf_evolve(scheme, n_points, dt, il, ir, Ea, Eb, Gc, Dc, *bl)
                                if product_rule:
                                    pathway_td[ip][s] = td_
                                else:
                                    total_td = total_td + td_
                            elif product_rule:
                                _sf_peaks(scheme, pathway_buff[ip][s], dt, il, ir, Ea, Eb, Gc, Dc, *bl)
                            else:
                                _sf_peaks(scheme, buff, dt, il, ir, Ea, Eb, Gc, Dc, *bl)
            if product_rule and not is_endor:
                for ip in range(n_pathways):
                    td_ = 1.0
                    for s in range(n_spaces):
                        if time_domain:
                            this = pathway_td[ip][s]
                        else:
                            pb = pathway_buff[ip][s]
                            if n_dims == 1:
                                this = np.fft.ifft(pb) * pb.size
                                this = this[: n_points[0]]
                            else:
                                this = _ifft2_partial(pb, n_points) * pb.size
                        td_ = td_ * this
                    total_td = total_td + weights[iOri] * ori_weights_t[iT] * td_

    # Post-processing
    info: dict = {}
    eq_trace = float(np.prod(2 * sys["I"] + 1))
    if is_endor:
        spc = np.real(endorspc) / eq_trace / n_pathways
        if endor_method == 2:
            spc = convspec(spc, rf[1] - rf[0], lw_endor)
        info["fd"] = spc
        info["td"] = None
        return Spectrum(rf, spc, info)
    if product_rule or time_domain:
        td = total_td
    else:
        if n_dims == 1:
            td = np.fft.ifft(buff) * buff.size
            td = td[: n_points[0]]
        else:
            td = _ifft2_partial(buff, n_points) * buff.size
    td = td / eq_trace / n_pathways
    if not predefined:
        td = td / np.max(np.abs(pathway_prefactor))
    if experiment_id == -1:
        axes = [np.arange(n_points[d]) * dt[d] for d in range(n_dims)]
    elif experiment_id == 1:
        axes = [np.arange(n_points[0]) * dt[0] + tau]
    elif experiment_id == 2:
        axes = [np.arange(n_points[0]) * dt[0] + tau + T_delay]
    elif experiment_id == 3:
        axes = [np.arange(n_points[0]) * dt[0] + T_delay]
    else:
        axes = [np.arange(n_points[0]) * dt[0] + t1_delay, np.arange(n_points[1]) * dt[1] + t2_delay]
    t1_axis = axes[0]
    decay_added = False
    if not np.isinf(T1) or not np.isinf(T2):
        tdecay = None
        if experiment_id == 1 and not np.isinf(T2):
            tdecay = np.exp(-2 * t1_axis / T2)
        elif experiment_id == 2 and not np.isinf(T1):
            tdecay = np.exp(-2 * tau / T2) * np.exp(-t1_axis / T1)
        elif experiment_id == 4 and not np.isinf(T1):
            tdecay = np.exp(-2 * tau / T2) * np.outer(np.exp(-t1_axis / T1), np.exp(-axes[1] / T1))
        if tdecay is not None:
            td = td * tdecay
            decay_added = True
    if n_dims == 1:
        if decay_added:
            _, _, fit = exponfit(t1_axis, np.real(td), 1)
            tdx = td - fit
        else:
            tdx = td
        tdx = tdx - tdx.mean()
        tdx = tdx * apowin(window, tdx.size)
        fd = np.fft.fftshift(np.fft.fft(tdx, zero_fill * tdx.size))
        info["f"] = fdaxis(dt[0], fd.size)
    else:
        tdx = (
            basecorr(np.real(td), None, [2, 2] if decay_added else [0, 0])[0]
            + 1j * basecorr(np.imag(td), None, [2, 2] if decay_added else [0, 0])[0]
        )
        w1 = apowin(window, n_points[0])
        w2 = apowin(window, n_points[1])
        fd = np.fft.fftshift(np.fft.fftn(tdx * np.outer(w1, w2), s=tuple(zero_fill * n_points), axes=(0, 1)))
        info["f1"] = fdaxis(dt[0], fd.shape[0])
        info["f2"] = fdaxis(dt[1], fd.shape[1])
    if np.max(np.abs(fd)) < 1e-300:
        fd = fd * 0
    info["fd"] = fd
    info["td"] = td
    info["nSkippedOrientations"] = n_skipped
    x_out = t1_axis if n_dims == 1 else (axes[0], axes[1])
    return Spectrum(x_out, td, info)


def _ifft2_partial(spc, n):
    td = np.fft.ifft(spc, axis=0)
    td = np.fft.ifft(td[: n[0], :], axis=1)
    return td[:, : n[1]]
