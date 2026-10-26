"""Spectrum-building kernels: EasySpin's C functions lisum1i, projecttriangles and projectzones.

Each kernel has a pure NumPy version and, when Numba is installed (the
``fast`` extra), a compiled loop that reproduces the MEX bit for bit at the
same speed; the NumPy versions are 5-12x slower and serve where Numba is
unavailable (the browser page). ``lisum1i`` adds line shapes to a spectrum
by interpolating from an *integrated* template, ``projecttriangles`` and
``projectzones`` project the resonance positions of triangles (or
segments, for axial grids) of an orientation grid onto the field axis.

EasySpin authors, by original file (contributors to the file up to EasySpin 6.0.15, see
  docs/easyspin-credits.md):
  lisum1i.c: Stefan Stoll
  projecttriangles.c: Stefan Stoll
  projectzones.c: Stefan Stoll
"""

from __future__ import annotations

import numpy as np

try:
    from numba import njit

    HAVE_NUMBA = True
except ImportError:  # pragma: no cover
    HAVE_NUMBA = False

    def njit(*args, **kwargs):
        def wrap(f):
            return f

        return wrap


def _ragged(starts: np.ndarray, counts: np.ndarray):
    """Flat indices start_i .. start_i+count_i-1 for every i, and the owner i of each entry."""
    counts = np.maximum(counts, 0)
    owner = np.repeat(np.arange(starts.size), counts)
    offsets = np.cumsum(counts) - counts
    within = np.arange(counts.sum()) - np.repeat(offsets, counts)
    return starts[owner] + within, owner


# ----------------------------------------------------------------------------- lisum1i
@njit(cache=True)
def _lisum1i_loop(T, PosT, WidT, Pos, Amp, Wid, x):  # pragma: no cover - compiled
    PosT = PosT - 1.0
    nT = T.size
    nx = x.size
    y = np.zeros(nx)
    delta = x[1] - x[0]
    beta = delta
    for k in range(Pos.size):
        if np.isnan(Pos[k]) or np.isnan(Amp[k]) or np.isnan(Wid[k]):
            continue
        if np.isinf(Pos[k]) or np.isinf(Amp[k]) or np.isinf(Wid[k]):
            continue
        WidS = Wid[k]
        if WidS < 0:
            WidS = 0.0
        if WidS < delta / 100:
            WidS = delta / 100
        alpha = WidT / WidS * beta
        left = (Pos[k] - x[0]) / delta - PosT / alpha - 0.5
        right = left + (nT - 1) / alpha
        if left >= nx - 1 or right <= 0:
            continue
        first = 0 if left < 0 else int(np.ceil(left))
        last = nx - 1 if right > nx - 1 else int(np.ceil(right))
        Amplitude = Amp[k] / beta
        pT = alpha * (first - left)
        idxT = int(pT - alpha)
        remT = pT - alpha - idxT
        ValueOld = T[0] if idxT < 0 else T[idxT] * (1 - remT) + T[idxT + 1] * remT
        idx = first
        while idx < last:
            idxT = int(pT)
            remT = pT - idxT
            Value = T[idxT] * (1 - remT) + T[idxT + 1] * remT
            y[idx] += Amplitude * (Value - ValueOld)
            idx += 1
            pT += alpha
            ValueOld = Value
        idxT = int(pT)
        remT = pT - idxT
        Value = T[idxT] * (1 - remT) + T[idxT + 1] * remT if idxT < nT - 2 else T[nT - 1]
        y[idx] += Amplitude * (Value - ValueOld)
    return y


def lisum1i_numpy(T, PosT, WidT, Pos, Amp, Wid, x, chunk: int = 3_000_000):
    """Pure NumPy lisum1i (ragged vectorisation, processed in chunks of ``chunk`` points).

    EasySpin authors (lisum1i.c, the C kernel, in NumPy): Stefan Stoll.
    """
    T = np.asarray(T, dtype=float)
    x = np.asarray(x, dtype=float)
    PosT = PosT - 1.0
    nT = T.size
    nx = x.size
    delta = x[1] - x[0]
    beta = delta
    Pos, Amp, Wid = (np.asarray(v, float).ravel() for v in (Pos, Amp, Wid))
    ok = np.isfinite(Pos) & np.isfinite(Amp) & np.isfinite(Wid)
    with np.errstate(divide="ignore", invalid="ignore"):
        WidS = np.maximum(np.where(Wid < 0, 0.0, Wid), delta / 100)
        alpha = WidT / WidS * beta
        left = (Pos - x[0]) / delta - PosT / alpha - 0.5
        right = left + (nT - 1) / alpha
    ok &= ~((left >= nx - 1) | (right <= 0))
    ok &= np.isfinite(left) & np.isfinite(right)
    first = np.where(left < 0, 0, np.ceil(np.where(ok, left, 0))).astype(np.intp)
    last = np.where(right > nx - 1, nx - 1, np.ceil(np.where(ok, right, 0))).astype(np.intp)
    A = Amp / beta
    q = alpha * (first - left) - alpha
    idx0 = np.where(ok, q, 0).astype(np.intp)
    rem0 = q - idx0
    idx0c = np.clip(idx0, 0, nT - 2)
    V0 = np.where(idx0 < 0, T[0], T[idx0c] * (1 - rem0) + T[idx0c + 1] * rem0)
    m = last - first + 1
    lines = np.nonzero(ok)[0]
    y = np.zeros(nx)
    if lines.size == 0:
        return y
    bounds = np.searchsorted(np.cumsum(m[lines]), np.arange(chunk, m[lines].sum() + chunk, chunk))
    start = 0
    for stop in np.unique(np.append(bounds, lines.size)):
        L = lines[start:stop]
        start = stop
        if L.size == 0:
            continue
        idx, owner = _ragged(first[L], m[L])
        line = L[owner]
        j = idx - first[line]
        pT = alpha[line] * (idx - left[line])
        idxT = pT.astype(np.intp)
        rem = pT - idxT
        idxTc = np.minimum(idxT, nT - 2)
        V = T[idxTc] * (1 - rem) + T[idxTc + 1] * rem
        is_last = j == m[line] - 1
        V = np.where(is_last & (idxT >= nT - 2), T[nT - 1], V)
        Vprev = np.empty_like(V)
        Vprev[1:] = V[:-1]
        starts = j == 0
        Vprev[starts] = V0[line[starts]]
        y += np.bincount(idx, weights=A[line] * (V - Vprev), minlength=nx)
    return y


def lisum1i(T, PosT, WidT, Pos, Amp, Wid, x):
    """Accumulate line shapes into a spectrum from an integrated template ``T`` (EasySpin's lisum1i).

    ``T`` is the integral of the line shape on the index axis ``0..len(T)-1``
    with centre ``PosT`` (1-based, as in EasySpin) and width ``WidT``;
    ``Pos``, ``Amp`` and ``Wid`` are the lines in the units of the axis ``x``.

    EasySpin authors (lisum1i.c): Stefan Stoll.
    """
    args = (
        np.ascontiguousarray(T, dtype=float), float(PosT), float(WidT),
        np.ascontiguousarray(np.ravel(Pos), dtype=float), np.ascontiguousarray(np.ravel(Amp), dtype=float),
        np.ascontiguousarray(np.ravel(Wid), dtype=float), np.ascontiguousarray(x, dtype=float),
    )  # fmt: skip
    return _lisum1i_loop(*args) if HAVE_NUMBA else lisum1i_numpy(*args)


# ----------------------------------------------------------------------------- projecttriangles
@njit(cache=True)
def _projecttriangles_loop(tri, areas, fun, amp, x):  # pragma: no cover - compiled
    nx = x.size
    spec = np.zeros(nx)
    iso = amp.size == 1
    Amplitude = amp[0]
    delta = x[1] - x[0]
    for iTri in range(tri.shape[1]):
        i1 = tri[0, iTri] - 1
        i2 = tri[1, iTri] - 1
        i3 = tri[2, iTri] - 1
        left = fun[i1]
        middle = fun[i2]
        right = fun[i3]
        Area = areas[iTri]
        if np.isnan(left) or np.isnan(middle) or np.isnan(right):
            continue
        if right < left:
            left, right = right, left
        if middle < left:
            left, middle = middle, left
        elif middle > right:
            right, middle = middle, right
        left = (left - x[0]) / delta
        middle = (middle - x[0]) / delta
        right = (right - x[0]) / delta
        Width = right - left
        Width1 = middle - left
        Width2 = right - middle
        if not iso:
            Amplitude = (amp[i1] + amp[i2] + amp[i3]) / 3
        first1 = int(left)
        last1 = int(middle)
        first2 = last1
        last2 = int(right)
        if Width1 > 0 and first1 < nx and last1 >= 0:
            f0 = (2 * Amplitude * Area / Width) / Width1 / delta
            if first1 == last1:
                spec[first1] += f0 * Width1 * Width1 / 2
            else:
                if first1 >= 0:
                    spec[first1] += f0 * (first1 + 1 - left) / 2 * (first1 + 1 - left)
                else:
                    first1 = -1
                if last1 < nx:
                    spec[last1] += f0 * ((last1 + middle) / 2 - left) * (middle - last1)
                else:
                    last1 = nx
                l2 = left - 0.5
                for idx in range(first1 + 1, last1):
                    spec[idx] += f0 * (idx - l2)
        if Width2 > 0 and first2 < nx and last2 >= 0:
            f0 = (2 * Amplitude * Area / Width) / Width2 / delta
            if first2 == last2:
                spec[first2] += f0 * Width2 * Width2 / 2
            else:
                if first2 >= 0:
                    spec[first2] += f0 * (first2 + 1 - middle) * (right - (middle + first2 + 1) / 2)
                else:
                    first2 = -1
                if last2 < nx:
                    spec[last2] += f0 * (right - last2) * (right - last2) / 2
                else:
                    last2 = nx
                r2 = right - 0.5
                for idx in range(first2 + 1, last2):
                    spec[idx] += f0 * (r2 - idx)
        if Width1 == 0 and Width2 == 0:
            first1 = int(left)
            if 0 <= first1 < nx:
                spec[first1] += Amplitude * Area / delta
    return spec


def projecttriangles_numpy(tri, areas, fun, amp, x):
    """Pure NumPy projecttriangles.

    EasySpin authors (projecttriangles.c, the C kernel, in NumPy): Stefan Stoll.
    """
    x = np.asarray(x, dtype=float)
    nx = x.size
    delta = x[1] - x[0]
    i = np.asarray(tri).astype(np.intp) - 1
    fun = np.asarray(fun, dtype=float)
    amp = np.asarray(amp, dtype=float).ravel()
    P = np.sort(fun[i], axis=0)
    ok = ~np.isnan(P).any(axis=0)
    left, middle, right = ((np.where(ok, P[k], 0.0) - x[0]) / delta for k in range(3))
    Amp = amp[i].mean(axis=0) if amp.size > 1 else np.full(left.shape, amp[0])
    areas = np.asarray(areas, dtype=float).ravel()
    Width, W1, W2 = right - left, middle - left, right - middle
    spec = np.zeros(nx)
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.where(W1 > 0, (2 * Amp * areas / Width) / W1 / delta, 0.0)
        f2 = np.where(W2 > 0, (2 * Amp * areas / Width) / W2 / delta, 0.0)
    first1, last1, last2 = (v.astype(np.intp) for v in (left, middle, right))
    first2 = last1

    def half(first, last, f0, W, first_w, last_w, ramp_sign, ramp_ref):
        mask = ok & (W > 0) & (first < nx) & (last >= 0)
        same = mask & (first == last)
        spec[:] += np.bincount(first[same], weights=f0[same] * W[same] ** 2 / 2, minlength=nx)
        diff = mask & ~same
        fb = diff & (first >= 0)
        spec[:] += np.bincount(first[fb], weights=f0[fb] * first_w[fb], minlength=nx)
        lb = diff & (last < nx)
        spec[:] += np.bincount(last[lb], weights=f0[lb] * last_w[lb], minlength=nx)
        lo_eff = np.where(first >= 0, first, -1) + 1
        hi_eff = np.where(last < nx, last, nx)
        sel = np.nonzero(diff)[0]
        if sel.size:
            idx, owner = _ragged(lo_eff[sel], hi_eff[sel] - lo_eff[sel])
            t = sel[owner]
            spec[:] += np.bincount(idx, weights=f0[t] * ramp_sign * (idx - ramp_ref[t]), minlength=nx)

    half(
        first1,
        last1,
        f1,
        W1,
        (first1 + 1 - left) ** 2 / 2,
        ((last1 + middle) / 2 - left) * (middle - last1),
        1.0,
        left - 0.5,
    )
    half(
        first2,
        last2,
        f2,
        W2,
        (first2 + 1 - middle) * (right - (middle + first2 + 1) / 2),
        (right - last2) ** 2 / 2,
        -1.0,
        right - 0.5,
    )
    zero = ok & (W1 == 0) & (W2 == 0)
    b = first1[zero]
    inr = (b >= 0) & (b < nx)
    spec += np.bincount(b[inr], weights=(Amp * areas / delta)[zero][inr], minlength=nx)
    return spec


def projecttriangles(tri, areas, fun, amp, x):
    """Project the values ``fun`` at the knots of a triangulation ``tri`` ((3, N), 1-based) onto the axis ``x``,
    with triangle weights ``areas`` and knot amplitudes ``amp`` (one value for all, or one per knot).

    EasySpin authors (projecttriangles.c): Stefan Stoll.
    """
    args = (
        np.ascontiguousarray(tri, dtype=np.intp), np.ascontiguousarray(np.ravel(areas), dtype=float),
        np.ascontiguousarray(np.ravel(fun), dtype=float), np.ascontiguousarray(np.ravel(amp), dtype=float),
        np.ascontiguousarray(x, dtype=float),
    )  # fmt: skip
    return _projecttriangles_loop(*args) if HAVE_NUMBA else projecttriangles_numpy(*args)


# ----------------------------------------------------------------------------- projectzones
@njit(cache=True)
def _projectzones_loop(pos, amp, seg_weights, x):  # pragma: no cover - compiled
    nx = x.size
    spec = np.zeros(nx)
    delta = x[1] - x[0]
    aniso = amp.size != 1
    meanAmp = amp[0] / delta
    for iSeg in range(seg_weights.size):
        left = pos[iSeg]
        right = pos[iSeg + 1]
        if np.isnan(left) or np.isnan(right):
            continue
        if left > right:
            left, right = right, left
        left = (left - x[0]) / delta
        right = (right - x[0]) / delta
        first = int(left)
        last = int(right)
        if first >= nx or last < 0:
            continue
        if aniso:
            meanAmp = (amp[iSeg] + amp[iSeg + 1]) / 2 / delta
        if first == last:
            spec[first] += meanAmp * seg_weights[iSeg]
        else:
            Height = meanAmp * seg_weights[iSeg] / (right - left)
            if first >= 0:
                spec[first] += Height * (first + 1 - left)
            else:
                first = -1
            if last < nx:
                spec[last] += Height * (right - last)
            else:
                last = nx
            for idx in range(first + 1, last):
                spec[idx] += Height
    return spec


def projectzones_numpy(pos, amp, seg_weights, x):
    """Pure NumPy projectzones.

    EasySpin authors (projectzones.c, the C kernel, in NumPy): Stefan Stoll.
    """
    x = np.asarray(x, dtype=float)
    nx = x.size
    delta = x[1] - x[0]
    pos = np.asarray(pos, dtype=float).ravel()
    amp = np.asarray(amp, dtype=float).ravel()
    w = np.asarray(seg_weights, dtype=float).ravel()
    lo, hi = pos[:-1], pos[1:]
    ok = ~(np.isnan(lo) | np.isnan(hi))
    left = (np.minimum(lo, hi) - x[0]) / delta
    right = (np.maximum(lo, hi) - x[0]) / delta
    left, right = np.where(ok, left, 0.0), np.where(ok, right, 0.0)
    first, last = np.trunc(left).astype(np.intp), np.trunc(right).astype(np.intp)
    ok &= ~((first >= nx) | (last < 0))
    mean_amp = (amp[:-1] + amp[1:]) / 2 / delta if amp.size != 1 else np.full(lo.size, amp[0] / delta)
    spec = np.zeros(nx)
    same = ok & (first == last)
    spec += np.bincount(first[same], weights=(mean_amp * w)[same], minlength=nx)
    diff = ok & ~same
    with np.errstate(divide="ignore", invalid="ignore"):
        height = np.where(diff, mean_amp * w / (right - left), 0.0)
    fb = diff & (first >= 0)
    spec += np.bincount(first[fb], weights=(height * (first + 1 - left))[fb], minlength=nx)
    lb = diff & (last < nx)
    spec += np.bincount(last[lb], weights=(height * (right - last))[lb], minlength=nx)
    lo_eff = np.where(first >= 0, first, -1) + 1
    hi_eff = np.where(last < nx, last, nx)
    sel = np.nonzero(diff)[0]
    if sel.size:
        idx, owner = _ragged(lo_eff[sel], hi_eff[sel] - lo_eff[sel])
        spec += np.bincount(idx, weights=height[sel][owner], minlength=nx)
    return spec


def projectzones(pos, amp, seg_weights, x):
    """Project the segments between consecutive positions ``pos`` (an axial grid) onto the axis ``x``, with
    segment weights ``seg_weights`` (one fewer than positions) and amplitudes ``amp`` (one, or one per position).

    EasySpin authors (projectzones.c): Stefan Stoll.
    """
    args = (
        np.ascontiguousarray(np.ravel(pos), dtype=float), np.ascontiguousarray(np.ravel(amp), dtype=float),
        np.ascontiguousarray(np.ravel(seg_weights), dtype=float), np.ascontiguousarray(x, dtype=float),
    )  # fmt: skip
    if args[2].size != args[0].size - 1:
        raise ValueError("Wrong number of segment weights. Should be nPeaks-1")
    return _projectzones_loop(*args) if HAVE_NUMBA else projectzones_numpy(*args)
