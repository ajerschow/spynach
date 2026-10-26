"""Line shapes: gaussian, lorentzian, lshape (pseudo-Voigt), voigtian, convspec.

All shapes are area-normalised for ``diff=0``; ``diff`` is the derivative order
(``-1`` the integral from minus infinity), ``phase`` mixes absorption and
dispersion (``pi/2`` is pure dispersion). Each function returns the
absorption shape, and ``(absorption, dispersion)`` with ``dispersion=True``.

EasySpin authors, by original file (contributors to the file up to EasySpin 6.0.15, see
  docs/easyspin-credits.md):
  convspec.m: Stefan Stoll
  gaussian.m: Stefan Stoll
  lorentzian.m: Stefan Stoll, Claudia Tait
  lshape.m: Stefan Stoll
  voigtian.m: Stefan Stoll
"""

from __future__ import annotations

import numpy as np
from scipy.special import dawsn, erf


def _hermite(x: np.ndarray, n: int) -> np.ndarray:
    """Physicists' Hermite polynomial H_n(x) by recursion, as in EasySpin."""
    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return 2 * x
    return 2 * x * _hermite(x, n - 1) - 2 * (n - 1) * _hermite(x, n - 2)


def _gpoly(x: np.ndarray, n: int) -> np.ndarray:
    """G_n(x) of Barakat, J. Quant. Spectrosc. Radiat. Transfer 11, 1729 (1971)."""
    if n == -1:
        return np.zeros_like(x)
    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return 2 * x
    return 2 * x * _gpoly(x, n - 1) - 2 * n * _gpoly(x, n - 2)


def _dawson_integral(x: np.ndarray) -> np.ndarray:
    """The integral of the Dawson function from 0 to x (even in x).

    24-point Gauss-Legendre quadrature on a fixed grid of step 1/8 from the origin, plus the remaining piece up to
    each abscissa; the integrand is smooth on that scale, so the rule is exact to rounding."""
    x = np.asarray(x, dtype=float)
    ax = np.abs(x).ravel()
    step = 0.125
    nodes, weights = np.polynomial.legendre.leggauss(24)

    def panels(lo, hi):
        mid, half = (hi + lo) / 2, (hi - lo) / 2
        return half * (weights[None, :] * dawsn(mid[:, None] + half[:, None] * nodes[None, :])).sum(axis=1)

    n_grid = int(np.floor(ax.max() / step)) + 1 if ax.size else 1
    grid = step * np.arange(n_grid)
    cumulative = np.concatenate([[0.0], np.cumsum(panels(grid[:-1], grid[1:]))]) if n_grid > 1 else np.zeros(1)
    k = np.minimum(np.floor(ax / step).astype(int), n_grid - 1)
    out = cumulative[k] + panels(grid[k], ax)
    return out.reshape(x.shape)


def _dawson_deriv(x: np.ndarray, n: int) -> np.ndarray:
    """n-th derivative of the Dawson function."""
    return (-1) ** n * (_hermite(x, n) * dawsn(x) - _gpoly(x, n - 1))


def gaussian(x, x0: float, fwhm: float, diff: int = 0, phase: float = 0.0, *, dispersion: bool = False):
    """Area-normalised Gaussian absorption (and dispersion) line shape or its derivatives.

    EasySpin authors (gaussian.m): Stefan Stoll.
    """
    x = np.asarray(x, dtype=float)
    if fwhm <= 0:
        raise ValueError("3rd input (fwhm, full width at half maximum) must be a real and positive number.")
    if diff < -1 or diff != int(diff):
        raise ValueError("4th input (diff, lineshape derivative) must be -1, 0, 1, 2, etc.")
    diff = int(diff)
    nonzero_phase = np.mod(phase, 2 * np.pi) != 0
    calc_dispersion = nonzero_phase or dispersion
    sig = fwhm / np.sqrt(2 * np.log(2)) / 2
    k = (x - x0) / sig / np.sqrt(2)
    if diff == -1:
        yabs = (1 + erf(k)) / 2
        # EasySpin evaluates the integral of the Dawson function as (k^2/2) 2F2(1,1;3/2,2;-k^2) with the Symbolic
        # Toolbox; here it is integrated numerically from the origin (Gauss-Legendre per interval, exact to rounding)
        pre = np.sqrt(2 / np.pi) * 2 / np.sqrt(np.pi) / sig / 2 * sig * np.sqrt(2)
        ydisp = pre * _dawson_integral(k) if calc_dispersion else None
    else:
        yabs = (
            np.sqrt(2 / np.pi) / 2 / sig * (-1 / sig) ** diff * 2 ** (-diff / 2) * _hermite(k, diff) * np.exp(-(k**2))
        )
        ydisp = (
            np.sqrt(2 / np.pi) * 2 / np.sqrt(np.pi) / sig / 2 * (1 / np.sqrt(2) / sig) ** diff * _dawson_deriv(k, diff)
            if calc_dispersion
            else None
        )
    if nonzero_phase:
        y1 = yabs * np.cos(phase) + ydisp * np.sin(phase)
        y2 = -yabs * np.sin(phase) + ydisp * np.cos(phase)
    else:
        y1, y2 = yabs, ydisp
    return (y1, y2) if dispersion else y1


def lorentzian(x, x0: float, fwhm: float, diff: int = 0, phase: float = 0.0, *, dispersion: bool = False):
    """Area-normalised Lorentzian absorption (and dispersion) line shape or its derivatives (``diff`` -1..2).

    EasySpin authors (lorentzian.m): Stefan Stoll, Claudia Tait.
    """
    x = np.asarray(x, dtype=float)
    if fwhm <= 0:
        raise ValueError("3rd input (fwhm, full width at half maximum) must be a real and positive number.")
    if diff not in (-1, 0, 1, 2):
        raise ValueError("4th input (diff, lineshape derivative) must be -1, 0, 1, or 2.")
    gamma = fwhm / np.sqrt(3)
    pre = 2 / np.pi / np.sqrt(3)
    z = (x - x0) / gamma
    d = 1 + 4 / 3 * z**2
    if diff == -1:
        yabs = np.arctan(2 / np.sqrt(3) * z) / np.pi + 0.5
        ydisp = (1 / 2 / np.pi) * np.log(3 + 4 * z**2)
    elif diff == 0:
        yabs = pre / gamma / d
        ydisp = pre**2 * np.pi / gamma * z / d
    elif diff == 1:
        yabs = -8 / 3 * pre / gamma**2 * z / d**2
        ydisp = pre**2 * np.pi / gamma**2 * (1 - 4 / 3 * z**2) / d**2
    else:
        yabs = 8 / 3 * pre / gamma**3 * (4 * z**2 - 1) / d**3
        ydisp = pre**2 * np.pi / gamma**3 * 2 * 4 / 3 * z * (4 / 3 * z**2 - 3) / d**3
    if phase != 0:
        y1 = yabs * np.cos(phase) + ydisp * np.sin(phase)
        y2 = -yabs * np.sin(phase) + ydisp * np.cos(phase)
    else:
        y1, y2 = yabs, ydisp
    return (y1, y2) if dispersion else y1


def lshape(x, x0: float, fwhm, diff: int = 0, alpha: float = 1.0, phase: float = 0.0):
    """Pseudo-Voigt line shape ``alpha*Gaussian + (1-alpha)*Lorentzian``; ``fwhm`` is one width or two.

    EasySpin authors (lshape.m): Stefan Stoll.
    """
    fwhm = np.atleast_1d(np.asarray(fwhm, dtype=float))
    if np.any(fwhm <= 0):
        raise ValueError("fwhm must be positive and real!")
    if fwhm.size > 2:
        raise ValueError("fwhm must contain 1 or 2 elements!")
    fwhm_g, fwhm_l = (fwhm[0], fwhm[1]) if fwhm.size == 2 else (fwhm[0], fwhm[0])
    gauss = 0.0
    if alpha != 0:
        ga, gd = gaussian(x, x0, fwhm_g, diff, dispersion=True)
        gauss = ga * np.cos(phase) + gd * np.sin(phase) if phase != 0 else ga
    lorentz = 0.0
    if alpha != 1:
        la, ld = lorentzian(x, x0, fwhm_l, diff, dispersion=True)
        lorentz = la * np.cos(phase) + ld * np.sin(phase) if phase != 0 else la
    return alpha * gauss + (1 - alpha) * lorentz


def voigtian(x, x0: float, fwhm_gl, diff: int = 0, phase: float = 0.0, *, dispersion: bool = False):
    """Voigt line shape: the convolution of a Gaussian and a Lorentzian of widths ``fwhm_gl = [fwhmG, fwhmL]``.

    EasySpin authors (voigtian.m): Stefan Stoll.
    """
    x = np.asarray(x, dtype=float)
    fwhm_gl = np.asarray(fwhm_gl, dtype=float).ravel()
    if fwhm_gl.size != 2:
        raise ValueError("fwhmGL (3rd input) must contain two numbers!")
    if np.any(fwhm_gl < 0):
        raise ValueError("Both numbers in fwhmGL (3rd input) must be non-negative.")
    if np.all(fwhm_gl == 0):
        raise ValueError("At least one value in fwhmGL (3rd input) must be nonzero.")
    fwhm_g, fwhm_l = fwhm_gl
    x0center = x[int(np.ceil(x.size / 2)) - 1]
    if fwhm_g > fwhm_l:
        diff_g, phase_g, x0_g, diff_l, phase_l, x0_l = diff, phase, x0, 0, 0.0, x0center
    else:
        diff_g, phase_g, x0_g, diff_l, phase_l, x0_l = 0, 0.0, x0center, diff, phase, x0
    yg = ygq = yl = ylq = None
    if fwhm_g > 0:
        yg, ygq = (
            gaussian(x, x0_g, fwhm_g, diff_g, phase_g, dispersion=True)
            if dispersion
            else (gaussian(x, x0_g, fwhm_g, diff_g, phase_g), None)
        )
    if fwhm_l > 0:
        yl, ylq = (
            lorentzian(x, x0_l, fwhm_l, diff_l, phase_l, dispersion=True)
            if dispersion
            else (lorentzian(x, x0_l, fwhm_l, diff_l, phase_l), None)
        )
    if fwhm_l > 0 and fwhm_g > 0:
        dx = x[1] - x[0]
        y1 = np.convolve(yg, yl, mode="same") * dx
        if dispersion:
            y2 = (np.convolve(yg, ylq, mode="same") if fwhm_l > fwhm_g else np.convolve(ygq, yl, mode="same")) * dx
    elif fwhm_l > 0:
        y1, y2 = yl, ylq
    else:
        y1, y2 = yg, ygq
    return (y1, y2) if dispersion else y1


def _expand(value, n: int) -> np.ndarray:
    v = np.atleast_1d(np.asarray(value, dtype=float)).ravel()
    return np.concatenate([v, np.full(n - v.size, v[0])]) if v.size < n else v[:n]


def convspec(spec, steps, fwhm, deriv=0, alpha=1.0, phase=0.0):
    """Convolve a spectrum (any dimension) with a line shape along every axis.

    ``steps`` is the abscissa increment, ``fwhm`` the width, ``deriv`` the
    derivative (0, 1, 2 or -1), ``alpha`` the Gaussian fraction (1 Gaussian, 0
    Lorentzian), ``phase`` the phase, each either one value for all axes or
    one per axis; a width of 0 skips that axis.

    EasySpin authors (convspec.m): Stefan Stoll.
    """
    spec = np.asarray(spec)
    if np.any(np.isnan(spec)):
        import warnings

        warnings.warn("The input data array (spec) contains NaN entries.", stacklevel=2)
    sz_original = spec.shape
    sz = tuple(n for n in sz_original if n != 1)
    if len(sz) == 0:
        sz = (spec.size,)
    work = spec.reshape(sz)
    n_dims = len(sz)
    steps = _expand(steps, n_dims)
    fwhm = _expand(fwhm, n_dims)
    deriv = _expand(deriv, n_dims)
    alpha = _expand(alpha, n_dims)
    phase = _expand(phase, n_dims)
    if np.any(~np.isin(deriv, [-1, 0, 1, 2])):
        raise ValueError("deriv must be 0, 1 or 2.")
    if np.any(alpha < 0) or np.any(alpha > 1):
        raise ValueError("alpha must satisfy 0 <= alpha <= 1.")
    NN = [2 * n + 1 for n in sz]
    mid = [int(np.round(n / 2)) + 1 for n in NN]  # 1-based line centre
    fwhm = fwhm / steps
    steps = np.where(fwhm == 0, 1.0, steps)
    if np.any((fwhm < 1) & (fwhm > 0) & (deriv == 1)):
        raise ValueError(
            "The linewidth is smaller than the increment. The convolution with such a narrow derivative "
            "lineshape does not work. Either increase linewidth or increase axis resolution."
        )
    decays = []
    for i in range(n_dims):
        if fwhm[i] > 0:
            line = lshape(np.arange(1, NN[i] + 1), mid[i], fwhm[i], int(deriv[i]), alpha[i], phase[i])
            line = np.concatenate([line[mid[i] - 1 :], line[: mid[i] - 1]])
            decays.append(np.fft.ifft(line))
        else:
            decays.append(np.ones(NN[i]) / NN[i])
    outer = decays[0]
    for d in decays[1:]:
        outer = np.multiply.outer(outer, d)
    axes = tuple(range(n_dims))
    out = np.fft.fftn(np.fft.ifftn(work, s=NN, axes=axes) * outer, axes=axes)
    out = np.prod(NN) / np.prod(steps**deriv) * out
    out = out[tuple(slice(0, n) for n in sz)].reshape(sz_original)
    return np.real(out) if np.isrealobj(spec) else out
