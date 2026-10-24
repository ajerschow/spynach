"""Interpolation over the orientation grid: EasySpin's esspline1d, esspline2d and gridinterp.

Resonance data computed on a coarse orientation grid are interpolated to a
finer one before the spectrum is projected: along the quarter meridian for
axial grids, on the rectangle of (theta, phi) that the triangular grid is
"rectified" to otherwise. ``G3`` is the global cubic spline, ``L3`` the local
cubic (Fritsch-Carlson slopes, axial grids only), ``L1`` linear.
"""

from __future__ import annotations

import numpy as np


def esspline_coeffs(y: np.ndarray, endcon: int = 0) -> np.ndarray:
    """Cubic spline coefficients of the rows of ``y`` over ``x = 1..n``: array (4, d, n-1) of ``c4, c3, c2, c1``.

    ``endcon`` 0 is not-a-knot, 1 zero end slopes, 2 periodic (EasySpin's esspline1d).
    """
    y = np.atleast_2d(np.asarray(y, dtype=float))
    d, n = y.shape
    if n < 2:
        raise ValueError("Interpolation not possible: n < 2.")
    dy = np.diff(y, axis=1)
    if n == 2:
        if endcon != 1:
            return np.stack([np.zeros((d, 1)), np.zeros((d, 1)), dy, y[:, :1]])
        dy2 = np.diff(np.hstack([np.zeros((d, 1)), dy, np.zeros((d, 1))]), axis=1)  # (d, 2)
        return np.stack([np.diff(dy2, axis=1), (2 * dy2[:, :1] - dy2[:, 1:]), np.zeros((d, 1)), y[:, :1]])
    if n == 3 and endcon == 0:
        c3 = np.diff(dy, axis=1) / 2  # (d, 1)
        c2 = dy[:, :1] - c3
        return np.stack([np.zeros((d, 2)), np.repeat(c3, 2, axis=1), np.hstack([c2, c2 + 2 * c3]), y[:, :2]])
    A = np.zeros((n, n))
    idx = np.arange(1, n - 1)
    A[idx, idx - 1] = 1
    A[idx, idx + 1] = 1
    A[idx, idx] = 4
    A[0, 0] = 1
    A[n - 1, n - 1] = 1
    b = np.zeros((n, d))
    b[1 : n - 1] = 3 * (dy[:, : n - 2] + dy[:, 1 : n - 1]).T
    if endcon == 0:
        A[0, 1] = 2
        A[n - 1, n - 2] = 2
        b[0] = 0.5 * (5 * dy[:, 0] + dy[:, 1])
        b[n - 1] = 0.5 * (dy[:, n - 3] + 5 * dy[:, n - 2])
    elif endcon == 1:
        pass
    elif endcon == 2:
        A[0, n - 1] = -1
        A[n - 1, [0, 1, n - 2, n - 1]] = [2, 1, 1, 2]
        b[n - 1] = 3 * (dy[:, 0] + dy[:, n - 2])
    else:
        raise ValueError("endcon must be 0, 1 or 2.")
    slp = np.linalg.solve(A, b).T  # (d, n)
    c4 = slp[:, : n - 1] + slp[:, 1:] - 2 * dy
    c3 = dy - slp[:, : n - 1] - c4
    return np.stack([c4, c3, slp[:, : n - 1], y[:, : n - 1]])


def esspline_eval(coeffs: np.ndarray, xs: np.ndarray) -> np.ndarray:
    """Evaluate spline coefficients (4, d, npp) at the 1-based abscissae ``xs``: array (d, len(xs))."""
    npp = coeffs.shape[2]
    xs = np.asarray(xs, dtype=float).ravel()
    index = np.minimum(np.trunc(xs).astype(np.intp), npp)
    t = xs - index
    c = coeffs[:, :, index - 1]  # (4, d, m)
    return ((c[0] * t + c[1]) * t + c[2]) * t + c[3]


def esspline1d(y, endcon: int = 0, xx=None) -> np.ndarray:
    """Cubic spline interpolation of the rows of ``y`` over ``1..n`` at ``xx`` (a count of points, or abscissae)."""
    y = np.atleast_2d(np.asarray(y, dtype=float))
    n = y.shape[1]
    coeffs = esspline_coeffs(y, endcon)
    if xx is None:
        return coeffs
    xs = np.linspace(1, n, int(xx)) if np.ndim(xx) == 0 else np.asarray(xx, dtype=float).ravel()
    out = esspline_eval(coeffs, xs)
    return out[0] if y.shape[0] == 1 else out


def esspline2d(data, ri, ci, endcon=(0, 0)) -> np.ndarray:
    """Tensor-product cubic spline of a 2-D array at 1-based (row, column) abscissae ``ri``, ``ci``."""
    data = np.asarray(data, dtype=float)
    R, C = data.shape
    ri = np.asarray(ri, dtype=float)
    ci = np.asarray(ci, dtype=float).reshape(ri.shape)
    # along rows (functions of the column index), then the coefficient sequences across rows
    cb = esspline_coeffs(data, endcon[1])  # (4, R, C-1): [b, r, jp]
    seqs = cb.transpose(0, 2, 1).reshape(4 * (C - 1), R)
    K = esspline_coeffs(seqs, endcon[0]).reshape(4, 4, C - 1, R - 1)  # [a, b, jp, ip]
    ip = np.minimum(np.trunc(ri).astype(np.intp), R - 1)
    jp = np.minimum(np.trunc(ci).astype(np.intp), C - 1)
    tr, tc = ri - ip, ci - jp
    k = K[:, :, jp - 1, ip - 1]  # (4, 4, ...)
    inner = ((k[:, 0] * tc + k[:, 1]) * tc + k[:, 2]) * tc + k[:, 3]  # (4, ...)
    return ((inner[0] * tr + inner[1]) * tr + inner[2]) * tr + inner[3]


def _fast_linear(y: np.ndarray, xx: np.ndarray) -> np.ndarray:
    k = np.minimum(np.maximum(1 + np.floor(xx - 1).astype(np.intp), 1), y.size - 1)
    return y[k - 1] + (xx - k) * (y[k] - y[k - 1])


def rectify(y: np.ndarray, grid_size: int, n_octants: int, periodic: bool, cubic: bool) -> np.ndarray:
    """Triangularly gridded data (one value per knot) to a (theta, phi) rectangle, interpolated along each theta row."""
    y = np.asarray(y, dtype=float).ravel()
    full_sphere = n_octants == 8
    if full_sphere:
        nr, nc, dlen = 2 * grid_size - 1, 4 * (grid_size - 1) + 1, 4
    else:
        nr, nc, dlen = grid_size, n_octants * (grid_size - 1) + 1, n_octants
    z = np.zeros((nr, nc))
    z[0, :] = y[0]
    idx = 1  # 0-based start of the current row's knots
    length = dlen + 1 - int(periodic)

    def row(idx, length):
        pts = np.concatenate([y[idx : idx + length], y[idx : idx + 1]]) if periodic else y[idx : idx + length]
        if cubic:
            return esspline1d(pts, 2 if periodic else 1, nc)
        return _fast_linear(pts, np.linspace(1, length, nc))

    for ir in range(1, grid_size - 1):
        z[ir, :] = row(idx, length)
        idx += length
        length += dlen
    z[grid_size - 1, :] = (
        np.concatenate([y[idx : idx + length], y[idx : idx + 1]]) if periodic else y[idx : idx + length]
    )
    if full_sphere:
        idx += length
        length -= dlen
        for ir in range(grid_size, 2 * grid_size - 2):
            z[ir, :] = row(idx, length)
            idx += length
            length -= dlen
        z[2 * grid_size - 2, :] = y[idx]
    return z


def _fritsch_carlson_row(y: np.ndarray, factor: int) -> np.ndarray:
    n = y.size
    x = np.arange(0, 1, 1 / factor)
    X = np.column_stack([x**3, x**2, x, np.ones(factor)])
    H = np.array([[2, -2, 1, 1], [-3, 3, -2, -1], [0, 0, 1, 0], [1, 0, 0, 0]], dtype=float)
    delta = np.diff(y)
    k = np.nonzero(np.sign(delta[: n - 2]) * np.sign(delta[1 : n - 1]) > 0)[0]
    tangents = np.zeros(n)
    if k.size:
        dmax = np.maximum(np.abs(delta[k]), np.abs(delta[k + 1]))
        dmin = np.minimum(np.abs(delta[k]), np.abs(delta[k + 1]))
        tangents[k + 1] = 2 * dmin * dmax / (delta[k] + delta[k + 1])
    Cm = np.vstack([y[:-1], y[1:], tangents[:-1], tangents[1:]])
    yii = X @ H @ Cm  # (factor, n-1)
    return np.concatenate([yii.ravel(order="F"), [y[n - 1]]])


def gridinterp(y, grid_params, phi, theta, interp_type: str = "G3") -> np.ndarray:
    """Interpolate data given on the knots of a spherical grid (one value per knot, rows of ``y``) to the
    orientations ``phi``, ``theta``. ``grid_params`` is ``[nOrientations, GridSize, closedPhi, nOctants, maxPhi]``."""
    y = np.atleast_2d(np.asarray(y, dtype=float))
    n_ori, grid_size, closed_phi, n_octants, max_phi = grid_params
    grid_size, n_octants = int(grid_size), int(n_octants)
    periodic = not bool(closed_phi)
    glob = interp_type[0].upper() == "G"
    cubic = interp_type[1] == "3"
    if y.shape[1] != int(n_ori):
        raise ValueError("Wrong number of points. Not a valid data set!")
    phi = np.asarray(phi, dtype=float).ravel()
    theta = np.asarray(theta, dtype=float).ravel()
    if n_octants == 0:
        m, n = y.shape
        factor = int(round((phi.size - 1) / (n - 1)))
        iphi = np.linspace(1, n, (n - 1) * factor + 1)
        if not cubic:
            out = np.vstack([np.interp(iphi, np.arange(1, n + 1), row) for row in y])
        elif glob:
            out = np.atleast_2d(esspline1d(y, 1, iphi))
        else:
            out = np.vstack([_fritsch_carlson_row(row, factor) for row in y])
        return out[0] if m == 1 else out
    rows = []
    for row in y:
        z = rectify(row, grid_size, n_octants, periodic, cubic)
        if n_octants < 8:
            iphi = 1 + n_octants * (grid_size - 1) * (phi / max_phi)
            ithe = 1 + (grid_size - 1) * (theta / (np.pi / 2))
        else:
            iphi = 1 + 4 * (grid_size - 1) * (phi / max_phi)
            ithe = 1 + 2 * (grid_size - 1) * (theta / np.pi)
        if not cubic:
            rows.append(_bilinear(z, iphi, ithe))
        elif glob:
            rows.append(esspline2d(z, ithe, iphi))
        else:
            raise ValueError(f"Local cubic interpolator for {n_octants} (oct) not available!")
    out = np.vstack(rows)
    return out[0] if y.shape[0] == 1 else out


def _bilinear(z: np.ndarray, xi: np.ndarray, yi: np.ndarray) -> np.ndarray:
    """MATLAB's interp2(z, xi, yi, 'linear') with 1-based coordinates (xi columns, yi rows)."""
    nr, nc = z.shape
    j = np.clip(np.floor(xi).astype(np.intp), 1, nc - 1)
    i = np.clip(np.floor(yi).astype(np.intp), 1, nr - 1)
    tx, ty = xi - j, yi - i
    z00, z01, z10, z11 = z[i - 1, j - 1], z[i - 1, j], z[i, j - 1], z[i, j]
    out = (1 - ty) * ((1 - tx) * z00 + tx * z01) + ty * ((1 - tx) * z10 + tx * z11)
    outside = (xi < 1) | (xi > nc) | (yi < 1) | (yi > nr)
    return np.where(outside, np.nan, out)
