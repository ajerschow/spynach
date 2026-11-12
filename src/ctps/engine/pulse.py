"""Coherence transfer weights of non-ideal pulses, after Jerschow, J. Chem. Phys. 113, 979 (2000).

A rotation keeps the rank ``l`` of a spherical tensor and mixes its order ``m`` (Eq. 2); a
single spin ``j`` has ``l ≤ 2j``, and the largest rank gives the upper bound on every transfer
(Eq. 19, verified here for ``l ≤ 8``). Leading-order transfer weights for ``p -> p'``:

* small flip angle ``β`` (Eq. 11-13): ``U = |f^l_{p'p}| (β/2)^|p - p'|`` with the coefficients
  ``f`` of Eq. 13;
* near ``π``, ``β`` the deviation from ``π`` and ``ε`` the axis tilt (``tan ε = Δω/ω₁``, Eq. 10),
  Eq. 14-16 and 20: ``U = |f^l_{p', -p}| |u|^|p + p'|`` with ``|u| = sqrt((β/2)² + ε²)``, so an
  inverted order passes with weight 1, a non-inverted one with ``f |u|^(2|p|)``, and ``p -> 0``
  leaks at first order;
* coupled spins (Eq. 21/22, Table III): the maximum over partitions of ``p`` and ``p'`` among the
  spins, all jumps of one sign, of the product of single-spin coefficients;
* a perfect ``π`` pulse inverts exactly (Eq. 3).

``exact=True`` replaces the leading terms by the full rotation matrix ``exp(-iβ n·J)`` with
``n = (0, cos ε, sin ε)`` (Eq. 8), the paper's convention (its Cayley-Klein form was checked
against this matrix to 1e-15). Errata of the paper noted while verifying it: after Eq. 2 the
last phase factor is ``exp(-imγ)``; below Eq. 7 the pairing is ``t' = l + m' - t``; in Section V
``Δm`` means ``Σ_i |m_i - m_{i-1}|`` as Table IV uses it. The legacy CCCP.m model
(``kind="cccp_1998"``: ``(1 - ε²/2)^|p|`` inverted, ``ε^|p|`` non-inverted) is kept for parity
with the MATLAB program only.

Author: Alexej Jerschow.
"""

from __future__ import annotations

import itertools
from functools import cache
from math import factorial, sqrt

import numpy as np

from ctps.model import Sequence


def f_coefficient(l: int, mp: int, m: int) -> float:
    """``f^l_{m'm}`` of Eq. 13: the coefficient of ``(β/2)^|m - m'|`` in ``d^l_{m'm}(β)`` at small ``β``."""
    l, mp, m = int(l), int(mp), int(m)
    if abs(m) > l or abs(mp) > l:
        return 0.0
    if m >= mp:
        return sqrt(factorial(l - mp) * factorial(l + m) / (factorial(l + mp) * factorial(l - m))) / factorial(m - mp)
    sign = -1.0 if (mp - m) % 2 else 1.0
    return (
        sign * sqrt(factorial(l + mp) * factorial(l - m) / (factorial(l - mp) * factorial(l + m))) / factorial(mp - m)
    )


def wigner_d(l: int, beta: float) -> np.ndarray:
    """``d^l_{m'm}(β)`` as a ``(2l+1, 2l+1)`` array indexed ``[m' + l, m + l]`` (Wigner's formula, ``exp(-iβJ_y)``)."""
    d = np.zeros((2 * l + 1, 2 * l + 1))
    c, s = np.cos(beta / 2), np.sin(beta / 2)
    for mp in range(-l, l + 1):
        for m in range(-l, l + 1):
            total = 0.0
            for k in range(max(0, m - mp), min(l + m, l - mp) + 1):
                total += (
                    (-1) ** (k - m + mp)
                    * c ** (2 * l + m - mp - 2 * k)
                    * s ** (mp - m + 2 * k)
                    / (factorial(l + m - k) * factorial(k) * factorial(mp - m + k) * factorial(l - mp - k))
                )
            d[mp + l, m + l] = total * sqrt(factorial(l + mp) * factorial(l - mp) * factorial(l + m) * factorial(l - m))
    return d


def rotation_matrix(l: int, beta: float, epsilon: float = 0.0) -> np.ndarray:
    """``|<l m'| exp(-iβ n·J) |l m>|`` with ``n = (0, cos ε, sin ε)`` (Eq. 8).

    A ``(2l+1, 2l+1)`` array indexed ``[m' + l, m + l]``.
    """
    from scipy.linalg import expm

    m = np.arange(l, -l - 1, -1, dtype=float)  # basis order l, l-1, ..., -l
    jz = np.diag(m)
    jp = np.zeros((2 * l + 1, 2 * l + 1))
    for i in range(1, 2 * l + 1):  # J+ |m> = sqrt(l(l+1) - m(m+1)) |m+1>
        jp[i - 1, i] = sqrt(l * (l + 1) - m[i] * (m[i] + 1))
    jy = (jp - jp.T) / 2j
    u = expm(-1j * beta * (np.cos(epsilon) * jy + np.sin(epsilon) * jz))
    return np.abs(u)[::-1, ::-1]  # reindex to [m' + l, m + l]


@cache
def _partition_bound(l: int, n_spins: int, mp: int, m: int) -> float:
    """Eq. 21/22: the largest ``Π |f|`` over partitions of ``m`` and ``m'`` among ``n_spins`` spins of rank ``l``.

    Only partitions whose single-spin jumps all have one sign count.
    """
    if n_spins == 1:
        return abs(f_coefficient(l, mp, m))
    best = 0.0
    values = range(-l, l + 1)
    for parts in itertools.product(values, repeat=n_spins):
        if sum(parts) != m:
            continue
        for parts_p in itertools.product(values, repeat=n_spins):
            if sum(parts_p) != mp:
                continue
            jumps = [b - a for a, b in zip(parts, parts_p)]
            if any(j > 0 for j in jumps) and any(j < 0 for j in jumps):
                continue
            product = 1.0
            for a, b in zip(parts, parts_p):
                product *= abs(f_coefficient(l, b, a))
            best = max(best, product)
    return best


def transfer_matrix(
    kind: str, p_max: int, *, l: int = 1, n_spins: int = 1, beta: float = 0.0, epsilon: float = 0.0, exact: bool = False
) -> np.ndarray:
    """Weights ``U[p' + p_max, p + p_max]`` of the transfer ``p -> p'`` for one pulse of the given kind."""
    size = 2 * p_max + 1
    orders = np.arange(-p_max, p_max + 1)
    U = np.zeros((size, size))
    if kind == "free":
        U[:] = 1.0
    elif kind == "perfect_180":
        for i, p in enumerate(orders):
            if abs(p) <= p_max:
                U[p_max - p, i] = 1.0
    elif kind == "cccp_1998":
        for i, p in enumerate(orders):
            U[p_max - p, i] = (1 - epsilon**2 / 2) ** abs(p)
            if p != 0:
                U[i, i] = epsilon ** abs(p)
    elif kind in ("small_flip", "imperfect_180"):
        if exact:
            if n_spins != 1:
                raise ValueError(
                    "exact rotation matrices are for a single spin; use the leading-order bound for coupled spins."
                )
            angle = beta if kind == "small_flip" else np.pi + beta
            R = rotation_matrix(l, angle, epsilon)
            for i, p in enumerate(orders):
                for k, pp in enumerate(orders):
                    if abs(p) <= l and abs(pp) <= l:
                        U[k, i] = R[pp + l, p + l]
        else:
            u = beta / 2 if kind == "small_flip" else sqrt((beta / 2) ** 2 + epsilon**2)
            for i, p in enumerate(orders):
                for k, pp in enumerate(orders):
                    source = p if kind == "small_flip" else -p
                    power = abs(source - pp)
                    coefficient = _partition_bound(l, n_spins, pp, source)
                    U[k, i] = coefficient * u**power if coefficient else 0.0
    else:
        raise ValueError(f"unknown pulse kind {kind!r}.")
    return U


def pulse_factor(sequence: Sequence, per_nucleus: dict[str, np.ndarray], exact: bool = False) -> np.ndarray:
    """Product over pulses of the transfer weight of the order jump each pathway makes (Eq. 23)."""
    k = len(next(iter(per_nucleus.values())))
    factor = np.ones(k)
    for j, pulse in enumerate(sequence.pulses):
        if pulse.kind in ("free", "perfect_180"):
            continue
        nucleus = sequence.nucleus(pulse.nucleus)
        allowed = sequence.allowed[pulse.nucleus]
        U = transfer_matrix(
            pulse.kind,
            allowed.p_max,
            l=nucleus.p_max,
            n_spins=nucleus.n_spins,
            beta=pulse.beta,
            epsilon=pulse.epsilon,
            exact=exact,
        )
        orders = per_nucleus[pulse.nucleus]
        before = orders[:, j].astype(int) + allowed.p_max
        after = orders[:, j + 1].astype(int) + allowed.p_max
        factor *= U[after, before]  # GPU: gather
    return factor
