"""Slow-motion cw EPR: EasySpin's chili with its stochastic Liouville equation machinery.

``chili`` solves the stochastic Liouville equation in the basis of Wigner
rotation functions (the K-symmetrised LjKKM basis of the Freed programs)
times the spin Liouville space, with Brownian rotational diffusion
(isotropic, axial or rhombic), an orientational potential, MOMD averaging
over a director grid, and Heisenberg exchange. The spectrum follows from the
Lanczos tridiagonalisation and the continued-fraction expansion of the
resolvent, or from a direct solve, the eigenvalue method, conjugate gradients
or BiCGSTAB. This is EasySpin's general Liouvillian code (``Opt.LiouvMethod =
'general'``); the C kernel of the ``'fast'`` method for S = 1/2 with up to two
nuclei is not ported, since it reproduces the general method to the accuracy
of the basis truncation.

Also here: ``wigner3j`` (exact, by integer arithmetic), ``wigner6j``,
``wignerd`` (matrices and single elements), ``tensor_cart2sph`` and
``tensor_sph2cart``.
"""

from __future__ import annotations

import math
from fractions import Fraction
from functools import cache

import numpy as np
import scipy.sparse as sps
from scipy.integrate import dblquad, quad, tplquad
from scipy.linalg import expm

from .components import compisoloop
from .constants import bmagn, boltzm, nmagn, planck
from .garlic import Spectrum, garlic
from .grids import sphgrid
from .lineshapes import convspec
from .nucdata import nucstring2list
from .processing import fieldmod
from .resfields import nucspinrmv
from .rotations import erot, vec2ang
from .spinops import sop
from .spinsystem import validatespinsys
from .units import unitconvert

__all__ = ["chili", "tensor_cart2sph", "tensor_sph2cart", "wigner3j", "wigner6j", "wignerd"]


# ----------------------------------------------------------------------------- angular momentum algebra
def _half(x) -> int:
    v = int(round(2 * float(x)))
    if abs(2 * float(x) - v) > 1e-9:
        raise ValueError("Nonphysical parameters. All parameters must be integers or half-integers.")
    return v


@cache
def _wigner3j_exact(j1d: int, j2d: int, j3d: int, m1d: int, m2d: int, m3d: int) -> float:
    """3j symbol from doubled arguments, exact: sign and the square root of a rational number."""
    if m1d + m2d + m3d != 0:
        return 0.0
    if not (j1d + j2d >= j3d and j2d + j3d >= j1d and j3d + j1d >= j2d):
        return 0.0
    if abs(m1d) > j1d or abs(m2d) > j2d or abs(m3d) > j3d:
        return 0.0
    if (j1d - m1d) % 2 or (j2d - m2d) % 2 or (j3d - m3d) % 2:
        raise ValueError("Nonphysical parameter. m must be one of -j, -j+1, ..., j-1, j.")
    if (j1d + j2d + j3d) % 2:
        return 0.0
    f = math.factorial
    j1pm1, j1mm1 = (j1d + m1d) // 2, (j1d - m1d) // 2
    j2pm2, j2mm2 = (j2d + m2d) // 2, (j2d - m2d) // 2
    j3pm3, j3mm3 = (j3d + m3d) // 2, (j3d - m3d) // 2
    a, b, c = (j1d + j2d - j3d) // 2, (j1d - j2d + j3d) // 2, (-j1d + j2d + j3d) // 2
    J = (j1d + j2d + j3d) // 2
    pref2 = Fraction(f(a) * f(b) * f(c) * f(j1pm1) * f(j1mm1) * f(j2pm2) * f(j2mm2) * f(j3pm3) * f(j3mm3), f(J + 1))
    tmin = max(0, (j2d - j3d - m1d) // 2, (j1d - j3d + m2d) // 2)
    tmax = min(a, j1mm1, j2pm2)
    s = Fraction(0)
    for t in range(tmin, tmax + 1):
        den = (
            f(t)
            * f(a - t)
            * f(j1mm1 - t)
            * f(j2pm2 - t)
            * f((j3d - j2d + m1d) // 2 + t)
            * f((j3d - j1d - m2d) // 2 + t)
        )
        s += Fraction((-1) ** t, den)
    if s == 0:
        return 0.0
    sign = -1 if ((j1d - j2d - m3d) // 2) % 2 else 1
    sign *= 1 if s > 0 else -1
    return sign * math.sqrt(float(pref2 * s * s))


def wigner3j(*args) -> float:
    """Wigner 3j symbol ``(j1 j2 j3; m1 m2 m3)``: six numbers, three pairs ``[j, m]``, or ``j`` and ``m``
    arrays (EasySpin's call forms). Exact for any rank (integer arithmetic)."""
    if len(args) == 6:
        vals = args
    elif len(args) == 3:
        vals = (args[0][0], args[1][0], args[2][0], args[0][1], args[1][1], args[2][1])
    elif len(args) == 2:
        j, m = args
        vals = (j[0], j[1], j[2], m[0], m[1], m[2])
    elif len(args) == 1:
        jm = np.asarray(args[0], dtype=float)
        if jm.shape != (2, 3):
            raise ValueError("If all J and M are supplied in an array, the array must be 2x3.")
        vals = (*jm[0], *jm[1])
    else:
        raise ValueError("Wrong number of parameters!")
    d = tuple(_half(v) for v in vals)
    if any(v < 0 for v in d[:3]):
        raise ValueError("Nonphysical parameter. j must satisfy j>=0.")
    return _wigner3j_exact(*d)


def wigner6j(*args) -> float:
    """Wigner 6j symbol ``{j1 j2 j3; j4 j5 j6}`` (six numbers, three pairs, two triples or one 6-array)."""
    if len(args) == 6:
        vals = args
    elif len(args) == 3:
        vals = (args[0][0], args[1][0], args[2][0], args[0][1], args[1][1], args[2][1])
    elif len(args) == 2:
        vals = (*args[0], *args[1])
    elif len(args) == 1:
        vals = tuple(np.asarray(args[0], dtype=float).ravel())
    else:
        raise ValueError("Wrong number of parameters!")
    j = [_half(v) / 2 for v in vals]
    j1, j2, j3, j4, j5, j6 = j

    def tri(a, b, c):
        return a + b >= c and b + c >= a and c + a >= b

    if not (tri(j1, j2, j3) and tri(j1, j5, j6) and tri(j4, j2, j6) and tri(j4, j5, j3)):
        return 0.0
    f = math.factorial

    def fi(x):
        return f(int(round(x)))

    alpha = sorted([j1 + j2 + j4 + j5, j1 + j3 + j4 + j6, j2 + j3 + j5 + j6])
    beta = sorted([j1 + j2 + j3, j1 + j5 + j6, j2 + j4 + j6, j3 + j4 + j5])
    A1, A2, A3 = alpha
    B1, B2, B3, B4 = beta

    def bino(a, b):
        return math.comb(int(round(a)), int(round(b)))

    pre = Fraction(fi(A1 - B2) * fi(A1 - B3) * fi(A1 - B4), fi(A1 - B1) * fi(B1 + 1))
    pre *= Fraction(fi(A2 - B1) * fi(A2 - B3) * fi(A2 - B4), fi(A2 - B2) * fi(B2 + 1))
    pre *= Fraction(fi(A3 - B1) * fi(A3 - B2) * fi(A3 - B4), fi(A3 - B3) * fi(B3 + 1))
    pre_sqrt = math.sqrt(float(pre * fi(B4 + 1)))
    sumfactor = (-1) ** int(round(B4)) * bino(A1 - B1, A1 - B4) * bino(A2 - B2, A2 - B4) * bino(A3 - B3, A3 - B4)
    if B4 >= A1:
        return pre_sqrt * sumfactor
    t = Fraction(1)
    s = Fraction(1)
    for n in range(int(round(B4)) + 1, int(round(A1)) + 1):
        t = -t * Fraction(n + 1, n - int(round(B4))) * Fraction(int(round(A1)) - n + 1, n - int(round(B1)))
        t = (
            t
            * Fraction(int(round(A2)) - n + 1, n - int(round(B2)))
            * Fraction(int(round(A3)) - n + 1, n - int(round(B3)))
        )
        s += t
    return pre_sqrt * sumfactor * float(s)


def _jacobip(n, a, b, x):
    P0 = np.ones_like(x)
    if n == 0:
        return P0
    P1 = (a - b) / 2 + (1 + (a + b) / 2) * x
    if n == 1:
        return P1
    a2b2 = a**2 - b**2
    ab = a + b
    for N in range(2, n + 1):
        P = (
            (2 * N + ab - 1) * ((2 * N + ab) * (2 * N + ab - 2) * x + a2b2) * P1
            - 2 * (N + a - 1) * (N + b - 1) * (2 * N + ab) * P0
        ) / (2 * N * (N + ab) * (2 * N + ab - 2))
        P0, P1 = P1, P
    return P1


def _littled(J: float, m1: float, m2: float, beta):
    """The reduced Wigner element d^J_{m1 m2}(beta) by the Jacobi polynomial formula, vectorised in beta."""
    beta = -np.asarray(beta, dtype=float)
    cands = [J + m2, J - m2, J + m1, J - m1]
    idx = int(np.argmin(cands))
    k = cands[idx]
    if idx == 0:
        mu, nu, lam = m1 - m2, -m1 - m2, m1 - m2
    elif idx == 1:
        mu, nu, lam = m2 - m1, m1 + m2, 0
    elif idx == 2:
        mu, nu, lam = m2 - m1, -m1 - m2, 0
    else:
        mu, nu, lam = m1 - m2, m1 + m2, m1 - m2
    k, mu, nu, lam = int(round(k)), int(round(mu)), int(round(nu)), int(round(lam))
    q_enum = list(range(int(round(2 * J - k)), int(round(2 * J - 2 * k - mu)), -1)) + list(range(1, nu + 1))
    q_denom = list(range(1, k + mu + 1)) + list(range(k + 1, k + nu + 1))
    q = 1.0
    for e, d in zip(sorted(q_enum), sorted(q_denom), strict=True):
        q *= e / d
    return (
        (-1) ** lam * math.sqrt(q) * np.sin(beta / 2) ** mu * np.cos(beta / 2) ** nu * _jacobip(k, mu, nu, np.cos(beta))
    )


def wignerd(J, *angles, phase: str = "-"):
    """Wigner rotation matrix ``D^J(alpha, beta, gamma)`` (or ``d^J(beta)`` with one angle) in EasySpin's
    convention (phase ``'-'``), ordered m = J .. -J; with ``J = [J, m1, m2]`` the single element, vectorised
    over the angles."""
    sign = {"-": -1, "+": 1}[phase]
    Jarr = np.atleast_1d(np.asarray(J, dtype=float))
    if len(angles) == 1:
        alpha, beta, gamma = None, angles[0], None
    elif len(angles) == 3:
        alpha, beta, gamma = angles
    else:
        raise ValueError("Give beta, or alpha, beta and gamma.")
    if Jarr.size == 1:
        Jv = float(Jarr[0])
        n = int(round(2 * Jv + 1))
        beta = float(beta)
        if beta == 0 and (alpha is None or (float(alpha) == 0 and float(gamma) == 0)):
            return np.eye(n)
        v = np.sqrt(np.arange(1, n) * np.arange(n - 1, 0, -1)) / 2
        Jy = np.diag(v, 1) - np.diag(v, -1)
        d = expm(sign * beta * Jy)
        if alpha is None:
            return d
        mz = np.arange(Jv, -Jv - 1, -1)
        return np.outer(np.exp(1j * sign * float(alpha) * mz), np.exp(1j * sign * float(gamma) * mz)) * d
    if Jarr.size != 3:
        raise ValueError("First input must be either J or [J m1 m2].")
    Jv, m1, m2 = (float(x) for x in Jarr)
    d = _littled(Jv, m1, m2, sign * np.asarray(beta, dtype=float))
    if alpha is None:
        return d
    return np.exp(sign * 1j * (np.asarray(alpha, dtype=float) * m1 + np.asarray(gamma, dtype=float) * m2)) * d


def tensor_cart2sph(Tc):
    """Irreducible spherical components ``(T0, T1[3], T2[5])`` of a scalar, a 3-vector of principal values or
    a 3x3 cartesian tensor (Mehring's convention; components ordered +q .. -q)."""
    Tc = np.asarray(Tc, dtype=float)
    T1 = np.zeros(3, dtype=complex)
    T2 = np.zeros(5, dtype=complex)
    if Tc.size == 1:
        T0 = -math.sqrt(1 / 3) * 3 * float(Tc.ravel()[0])
    elif Tc.size == 3:
        x, y, z = Tc.ravel()
        T0 = -math.sqrt(1 / 3) * (x + y + z)
        T2[0] = T2[4] = 0.5 * (x - y)
        T2[2] = math.sqrt(2 / 3) * (z - 0.5 * (x + y))
    elif Tc.size == 9:
        T = Tc.reshape(3, 3)
        T0 = -math.sqrt(1 / 3) * (T[0, 0] + T[1, 1] + T[2, 2])
        T1[0] = -0.5 * (T[2, 0] - T[0, 2] + 1j * (T[2, 1] - T[1, 2]))
        T1[2] = -0.5 * (T[2, 0] - T[0, 2] - 1j * (T[2, 1] - T[1, 2]))
        T1[1] = -(1j / math.sqrt(2)) * (T[0, 1] - T[1, 0])
        T2[0] = 0.5 * ((T[0, 0] - T[1, 1]) + 1j * (T[0, 1] + T[1, 0]))
        T2[4] = 0.5 * ((T[0, 0] - T[1, 1]) - 1j * (T[0, 1] + T[1, 0]))
        T2[1] = -0.5 * ((T[0, 2] + T[2, 0]) + 1j * (T[1, 2] + T[2, 1]))
        T2[3] = 0.5 * ((T[0, 2] + T[2, 0]) - 1j * (T[1, 2] + T[2, 1]))
        T2[2] = math.sqrt(2 / 3) * (T[2, 2] - 0.5 * (T[0, 0] + T[1, 1]))
    else:
        raise ValueError("Cartesian tensor has wrong size. It needs to be 1x1, 3x1, 1x3, or 3x3.")
    return T0, T1, T2


def tensor_sph2cart(T0, T1, T2) -> np.ndarray:
    """The cartesian 3x3 tensor from its spherical components (the inverse of ``tensor_cart2sph``)."""
    T1 = np.asarray(T1, dtype=complex).ravel()
    T2 = np.asarray(T2, dtype=complex).ravel()
    if T1.size != 3 or T2.size != 5:
        raise ValueError("T1 needs 3 and T2 needs 5 components.")
    t1 = {q: T1[1 - q] for q in (-1, 0, 1)}
    t2 = {q: T2[2 - q] for q in (-2, -1, 0, 1, 2)}
    T = np.zeros((3, 3), dtype=complex)
    T[0, 2] = 0.5 * t1[1] + 0.5 * t1[-1] - 0.5 * t2[1] + 0.5 * t2[-1]
    T[2, 0] = -0.5 * t1[1] - 0.5 * t1[-1] - 0.5 * t2[1] + 0.5 * t2[-1]
    T[1, 2] = -0.5j * t1[1] + 0.5j * t1[-1] + 0.5j * t2[1] + 0.5j * t2[-1]
    T[2, 1] = 0.5j * t1[1] - 0.5j * t1[-1] + 0.5j * t2[1] + 0.5j * t2[-1]
    T[0, 0] = -T0 / math.sqrt(3) + 0.5 * t2[2] - t2[0] / math.sqrt(6) + 0.5 * t2[-2]
    T[1, 1] = -T0 / math.sqrt(3) - 0.5 * t2[2] - t2[0] / math.sqrt(6) - 0.5 * t2[-2]
    T[2, 2] = -T0 / math.sqrt(3) + math.sqrt(2 / 3) * t2[0]
    T[0, 1] = 1j / math.sqrt(2) * t1[0] - 0.5j * t2[2] + 0.5j * t2[-2]
    T[1, 0] = -1j / math.sqrt(2) * t1[0] - 0.5j * t2[2] + 0.5j * t2[-2]
    return T


def _istotensor(a, b):
    """Rank-0, 1 and 2 irreducible spherical tensor operators from two vector operators (or vectors)."""

    def mul(x, y):  # operator product for matrices, plain product with scalars
        if np.isscalar(x) or np.isscalar(y):
            return x * y
        return x @ y

    ax, ay, az = a
    bx, by, bz = b
    xx, yy, zz = mul(ax, bx), mul(ay, by), mul(az, bz)
    xz, zx, yz, zy, xy, yx = mul(ax, bz), mul(az, bx), mul(ay, bz), mul(az, by), mul(ax, by), mul(ay, bx)
    T0 = -(1 / math.sqrt(3)) * (xx + yy + zz)
    T1 = [-0.5 * ((xz - zx) + 1j * (yz - zy)), -(1j / math.sqrt(2)) * (yx - xy), -0.5 * ((xz - zx) - 1j * (yz - zy))]
    T2 = [
        0.5 * ((xx - yy) + 1j * (xy + yx)),
        -0.5 * ((xz + zx) + 1j * (yz + zy)),
        math.sqrt(2 / 3) * (zz - 0.5 * (xx + yy)),
        0.5 * ((xz + zx) - 1j * (yz + zy)),
        0.5 * ((xx - yy) - 1j * (xy + yx)),
    ]
    return T0, T1, T2


def _hil2liouv(H):
    """Hilbert-space operator to its Liouville-space commutator superoperator, kron(1, H) - kron(H^T, 1)."""
    H = sps.csr_array(H)
    n = H.shape[0]
    eye = sps.identity(n, format="csr")
    return (sps.kron(eye, H) - sps.kron(H.T, eye)).tocsr()


# ----------------------------------------------------------------------------- orientational basis
def _generateoribasis(even_lmax: int, odd_lmax: int, mmax: int, kmax: int, jkmin: int, even_k: bool):
    """The K-symmetrised LjKKM basis in the Freed ordering: columns L, jK, K, M."""
    delta_k = 2 if even_k else 1
    if even_k:
        kmax = kmax // 2 * 2
    Llist = sorted(list(range(0, even_lmax + 1, 2)) + list(range(1, odd_lmax + 1, 2)))
    rows = []
    for L in Llist:
        parity = 1 if L % 2 == 0 else -1
        for jK in range(jkmin, 2, 2):
            for K in range(0, min(L, kmax) + 1, delta_k):
                if K == 0 and parity != jK:
                    continue
                mmx = min(L, mmax)
                for M in range(-mmx, mmx + 1):
                    rows.append((L, jK, K, M))
    return np.array(rows, dtype=int).reshape(-1, 4)


@cache
def _jjj(rank: int, L1: int, MK1: int, L2: int, MK2: int) -> float:
    return _wigner3j_exact(2 * L1, 2 * rank, 2 * L2, 2 * MK1, -2 * MK1 - 2 * MK2, 2 * MK2)


def _pqorder(spins):
    """(p, q) = (m1 - m2, m1 + m2) of every spin for the Liouville product basis in descending mm order."""
    spins = np.asarray(spins, dtype=float)
    n_states = (2 * spins + 1).astype(int)
    n1 = int(np.prod(n_states))
    n2 = 1
    m = np.zeros((n1, spins.size))
    for k, s in enumerate(spins):
        n1 //= n_states[k]
        m_ = np.tile(np.repeat(np.arange(s, -s - 1, -1), n1), n2)
        m[:, k] = m_
        n2 *= n_states[k]
    pq = np.zeros((m.shape[0] ** 2, 2 * spins.size))
    for k in range(spins.size):
        # column-major Liouville index: the row index m1 varies fastest, as in hil2liouv's kron(1, H) - kron(H^T, 1)
        m1 = np.tile(m[:, k], m.shape[0])
        m2 = np.repeat(m[:, k], m.shape[0])
        pq[:, 2 * k] = m1 - m2
        pq[:, 2 * k + 1] = m1 + m2
    return pq


# ----------------------------------------------------------------------------- interactions
def _magint(sys: dict, spin_ops, center_field_mT: float, include_nz: bool, explicit_sweep: bool):
    """Spherical tensor operators (T) and interaction tensor components (F, Hz) of every term."""
    n_el = sys["nElectrons"]
    n_nuc = sys["nNuclei"]
    B0 = (0.0, 0.0, 1.0) if explicit_sweep else (0.0, 0.0, center_field_mT / 1e3)
    T0, T1, T2, F0, F1, F2, field_dep = [], [], [], [], [], [], []

    def add(a, b, cart, dep):
        t0, t1, t2 = _istotensor(a, b)
        f0, f1, f2 = tensor_cart2sph(cart)
        T0.append(t0)
        T1.append(t1)
        T2.append(t2)
        F0.append(f0)
        F1.append(f1)
        F2.append(f2)
        field_dep.append(dep)

    for e in range(n_el):
        g = sys["g"][3 * e : 3 * e + 3, :3] if sys["fullg"] else np.diag(sys["g"][e, :3])
        R = erot(sys["gFrame"][e]).T
        g = R @ g @ R.T
        add(B0, spin_ops[e], g * bmagn / planck, True)
    for e in range(n_el):
        for n in range(n_nuc):
            if sys["fullA"]:
                A = sys["A"][3 * n : 3 * n + 3, 3 * e : 3 * e + 3] * 1e6
            else:
                A = np.diag(sys["A"][n, 3 * e : 3 * e + 3]) * 1e6
            frame = sys["AFrame"][n, 3 * e : 3 * e + 3]
            if np.any(frame):
                R = erot(frame).T
                A = R @ A @ R.T
            add(spin_ops[e], spin_ops[n_el + n], A, False)
    for e in range(n_el):
        if sys["S"][e] < 1:
            continue
        D = sys["D"][3 * e : 3 * e + 3, :3] if sys["fullD"] else np.diag(sys["D"][e, :3])
        D = D * 1e6
        if np.any(sys["DFrame"][e]):
            R = erot(sys["DFrame"][e]).T
            D = R @ D @ R.T
        add(spin_ops[e], spin_ops[e], D, False)
    if n_el > 1:
        c = 0
        for e1 in range(n_el):
            for e2 in range(e1 + 1, n_el):
                J = sys["ee"][3 * c : 3 * c + 3, :3] if sys["fullee"] else np.diag(sys["ee"][c, :3])
                if np.any(sys["eeFrame"][c]):
                    R = erot(sys["eeFrame"][c]).T
                    J = R @ np.diag(sys["ee"][c, :3]) @ R.T
                add(spin_ops[e1], spin_ops[e2], J * 1e6, False)
                c += 1
    if include_nz:
        for n in range(n_nuc):
            add(B0, spin_ops[n_el + n], np.array(-sys["gn"][n] * nmagn / planck), True)
    F1a = np.array(F1).reshape(-1, 3)
    F2a = np.array(F2).reshape(-1, 5)
    symmetry = {
        "nobetatilts": bool(np.allclose(F2a[:, [1, 3]], 0)),
        "tensorsCollinear": bool(np.all(np.abs(F2a.imag) == 0)),
        "axialSystem": bool(np.allclose(F2a[:, 0], 0)),
    }
    return (T0, T1, T2), (np.array(F0), F1a, F2a), symmetry, np.array(field_dep, dtype=bool)


def _rbos(T, F, angles, field_dep):
    """Rotational basis operators in Liouville space: rank-0 scalar, rank-1 (3x3) and rank-2 (5x5) grids of
    sparse superoperators, split into field-dependent (B) and field-independent (G) parts."""
    T0, T1, T2 = T
    F0, F1, F2 = F
    D1 = wignerd(1, *angles)
    D2 = wignerd(2, *angles)
    n_terms = F0.size
    n = T0[0].shape[0]
    zero = sps.csr_array((n, n), dtype=complex)

    def accumulate(dep):
        Q0 = zero.copy()
        for k in range(n_terms):
            if field_dep[k] == dep:
                Q0 = Q0 + np.conj(F0[k]) * T0[k]
        Q1 = None
        if np.any(F1):
            Q1 = [[zero.copy() for _ in range(3)] for _ in range(3)]
            for mp in range(3):
                for mq in range(3):
                    acc = zero.copy()
                    for m in range(3):
                        for k in range(n_terms):
                            if field_dep[k] == dep:
                                acc = acc + D1[m, mp] * (-1) * np.conj(F1[k, mq]) * T1[k][m]
                    Q1[mp][mq] = acc
        Q2 = [[zero.copy() for _ in range(5)] for _ in range(5)]
        for mp in range(5):
            for mq in range(5):
                acc = zero.copy()
                for m in range(5):
                    for k in range(n_terms):
                        if field_dep[k] == dep:
                            acc = acc + D2[m, mp] * np.conj(F2[k, mq]) * T2[k][m]
                Q2[mp][mq] = acc
        return Q0, Q1, Q2

    Q0B, Q1B, Q2B = accumulate(True)
    Q0G, Q1G, Q2G = accumulate(False)

    def liouv(Q0, Q1, Q2):
        L0 = _hil2liouv(Q0)
        L1 = [[_hil2liouv(q) for q in row] for row in Q1] if Q1 is not None else None
        L2 = [[_hil2liouv(q) for q in row] for row in Q2]
        return L0, L1, L2

    return liouv(Q0B, Q1B, Q2B), liouv(Q0G, Q1G, Q2G)


def _liouvhamiltonian_full(basis, Q0, Q1, Q2):
    """liouvhamiltonian as EasySpin assembles it: rank 0 fully, ranks 1 and 2 from the upper triangle."""
    L, jK, K, M = basis[:, 0], basis[:, 1], basis[:, 2], basis[:, 3]
    n_ori = L.size
    n_spin = Q0.shape[0]
    n_total = n_ori * n_spin
    Q0c = sps.coo_array(Q0)
    rows, cols, vals = [], [], []
    for b in range(n_ori):
        factor = (
            (-1.0) ** (K[b] - M[b])
            * (2 * L[b] + 1)
            * _jjj(0, L[b], -M[b], L[b], M[b])
            * _jjj(0, L[b], -K[b], L[b], K[b])
        )
        rows.append(Q0c.row + b * n_spin)
        cols.append(Q0c.col + b * n_spin)
        vals.append(factor * Q0c.data)
    H0 = sps.coo_array(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n_total, n_total)
    ).tocsr()
    H = H0
    for rank, Q in ((1, Q1), (2, Q2)):
        if Q is None:
            continue
        Qd = [[q.toarray() for q in row] for row in Q]
        rows, cols, vals = [], [], []
        for b1 in range(n_ori):
            L1, jK1, K1, M1 = L[b1], jK[b1], K[b1], M[b1]
            k1zero = K1 == 0
            for b2 in range(0 if rank == 1 else b1, n_ori):
                L2 = L[b2]
                if abs(L1 - L2) > rank:
                    if rank == 2 and L2 > L1:
                        break
                    continue
                M2 = M[b2]
                if abs(M1 - M2) > rank:
                    continue
                K2 = K[b2]
                if abs(K1 - K2) > rank:
                    continue
                jK2 = jK[b2]
                NL = math.sqrt((2 * L1 + 1) * (2 * L2 + 1))
                k2zero = K2 == 0
                if rank == 1:
                    jjjM, jjjKa = _jjj(1, L1, M1, L2, M2), _jjj(1, L1, K1, L2, K2)
                else:
                    jjjM, jjjKa = _jjj(2, L1, M1, L2, -M2), _jjj(2, L1, K1, L2, -K2)
                idxM, idxKa, idxKb = rank - (M1 - M2), rank - (K1 - K2), rank - (-K1 + K2)
                pre = (
                    (1 / (2 * math.sqrt((1 + k1zero) * (1 + k2zero))))
                    * np.conj(np.sqrt(complex(jK1)))
                    * np.sqrt(complex(jK2))
                    * NL
                    * jjjM
                )
                block = (
                    pre
                    * jjjKa
                    * ((-1.0) ** (K1 - M1) * Qd[idxM][idxKa] + jK1 * jK2 * (-1.0) ** (K2 - M1) * Qd[idxM][idxKb])
                )
                if K1 + K2 <= rank:
                    jjjKb = _jjj(1, L1, K1, L2, -K2) if rank == 1 else _jjj(2, L1, K1, L2, K2)
                    idxKc, idxKd = rank - (-K1 - K2), rank - (K1 + K2)
                    block = block + pre * jjjKb * (
                        jK1 * (-1.0) ** (L2 - M1) * Qd[idxM][idxKc]
                        + jK2 * (-1.0) ** (L2 + K1 + K2 - M1) * Qd[idxM][idxKd]
                    )
                r, c = np.nonzero(block)
                if r.size == 0:
                    continue
                rows.append(r + b1 * n_spin)
                cols.append(c + b2 * n_spin)
                vals.append(block[r, c])
        if not rows:
            continue
        Hr = sps.coo_array(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n_total, n_total)
        ).tocsr()
        if rank == 2:
            Hr = sps.triu(Hr)
        Hr = Hr + sps.triu(Hr, k=1).T
        H = H + Hr
    return H.tocsr()


def _xlmk(pot_L, pot_M, pot_K, pot_lambda, R):
    """Expansion coefficients X^L_MK of the potential-dependent diffusion operator (chili_xlmk)."""
    L = np.asarray(pot_L, dtype=int).ravel()
    M = np.asarray(pot_M, dtype=int).ravel()
    K = np.asarray(pot_K, dtype=int).ravel()
    lam = np.asarray(pot_lambda, dtype=complex).ravel()
    keep = lam != 0
    L, M, K, lam = L[keep], M[keep], K[keep], lam[keep]
    if lam.size == 0:
        return []
    idx = (M != 0) | (K != 0)
    lam = np.concatenate([lam, np.conj(lam[idx]) * (-1.0) ** (K[idx] - M[idx])])
    L = np.concatenate([L, L[idx]])
    M = np.concatenate([M, -M[idx]])
    K = np.concatenate([K, -K[idx]])
    n_terms = lam.size
    max_lpot = int(L.max())
    all_m_zero = bool(np.all(M == 0))
    lam_tab = [np.zeros((2 * l + 1, 2 * l + 1), dtype=complex) for l in range(max_lpot + 1)]
    for p in range(n_terms):
        lam_tab[L[p]][M[p] + L[p], K[p] + L[p]] = lam[p]

    def lamf(l, m, k):
        return lam_tab[l][m + l, k + l]

    R = np.asarray(R, dtype=float).ravel()
    if R.size == 1:
        R = np.repeat(R, 3)
    Rz = R[2]
    Rp = (R[0] + R[1]) / 2
    Rd = (R[0] - R[1]) / 4

    def cm(l, k):
        return math.sqrt(l * (l + 1) - k * (k - 1))

    def cp(l, k):
        return math.sqrt(l * (l + 1) - k * (k + 1))

    max_lx = 2 * max_lpot
    X = [np.zeros((2 * l + 1, 2 * l + 1), dtype=complex) for l in range(max_lx + 1)]
    for Lx in range(max_lx + 1):
        m_range = [0] if all_m_zero else range(-Lx, Lx + 1)
        for Mx in m_range:
            for Kx in range(-Lx, Lx + 1):
                A = 0.0
                if Lx <= max_lpot:
                    if Rd != 0:
                        if Kx + 2 <= Lx:
                            A += Rd * cm(Lx, Kx + 1) * cm(Lx, Kx + 2) * lamf(Lx, Mx, Kx + 2)
                        if Kx - 2 >= -Lx:
                            A += Rd * cp(Lx, Kx - 1) * cp(Lx, Kx - 2) * lamf(Lx, Mx, Kx - 2)
                    A += lamf(Lx, Mx, Kx) * (Rp * (Lx * (Lx + 1) - Kx**2) + Rz * Kx**2)
                B = 0.0
                for p1 in range(n_terms):
                    L1, M1, K1, lam1 = L[p1], M[p1], K[p1], lam[p1]
                    for p2 in range(n_terms):
                        L2, M2, K2, lam2 = L[p2], M[p2], K[p2], lam[p2]
                        if M1 - Mx + M2 != 0:
                            continue
                        if abs(L1 - L2) > Lx or Lx > L1 + L2:
                            continue
                        if M1 == 0 and M2 == 0 and Mx == 0 and (L1 + L2 + Lx) % 2:
                            continue
                        B_ = 0.0
                        if Rd != 0:
                            if K1 - Kx + K2 + 2 == 0 and abs(K1 + 1) <= L1 and abs(K2 + 1) <= L2:
                                cpp = cp(L1, K1) * cp(L2, K2)
                                if cpp != 0:
                                    B_ += Rd * cpp * wigner3j(L1, Lx, L2, K1 + 1, -Kx, K2 + 1)
                            if K1 - Kx + K2 - 2 == 0 and abs(K1 - 1) <= L1 and abs(K2 - 1) <= L2:
                                cmm = cm(L1, K1) * cm(L2, K2)
                                if cmm != 0:
                                    B_ += Rd * cmm * wigner3j(L1, Lx, L2, K1 - 1, -Kx, K2 - 1)
                        if K1 - Kx + K2 == 0:
                            if Rz != 0 and K1 != 0 and K2 != 0:
                                B_ += Rz * K1 * K2 * wigner3j(L1, Lx, L2, K1, -Kx, K2)
                            if Rp != 0 and abs(K1 + 1) <= L1 and abs(K2 - 1) <= L2:
                                cpm = cp(L1, K1) * cm(L2, K2)
                                if cpm != 0:
                                    B_ += Rp * cpm * wigner3j(L1, Lx, L2, K1 + 1, -Kx, K2 - 1)
                        if B_ == 0:
                            continue
                        B += lam1 * lam2 * wigner3j(L1, Lx, L2, M1, -Mx, M2) * B_
                X[Lx][Mx + Lx, Kx + Lx] = -0.5 * A - (2 * Lx + 1) / 4 * (-1.0) ** (Kx - Mx) * B
    return X


def _diffsuperop(basis: np.ndarray, R, XLMK, all_m_zero: bool) -> sps.csr_array:
    """The diffusion superoperator in the LjKKM basis for a diagonal diffusion tensor, with the potential."""
    L, jK, K, M = basis[:, 0], basis[:, 1], basis[:, 2], basis[:, 3]
    n = L.size
    R = np.asarray(R, dtype=float).ravel()
    if R.size == 1:
        R = np.repeat(R, 3)
    Rz, Rp, Rd = R[2], (R[0] + R[1]) / 2, (R[0] - R[1]) / 4
    include_potential = len(XLMK) > 0
    if not include_potential and Rd == 0:
        return sps.diags((Rp * (L * (L + 1) - K**2) + Rz * K**2).astype(float)).tocsr()

    def Np(l, k):
        return math.sqrt((l * (l + 1) - k * (k + 1)) * (l * (l + 1) - (k + 1) * (k + 2)))

    def Nm(l, k):
        return math.sqrt((l * (l + 1) - k * (k - 1)) * (l * (l + 1) - (k - 1) * (k - 2)))

    rows, cols, vals = [], [], []
    for b1 in range(n):
        L1, jK1, K1, M1 = L[b1], jK[b1], K[b1], M[b1]
        for b2 in range(b1, n):
            if L[b2] != L1 or M[b2] != M1 or jK[b2] != jK1:
                continue
            L2, K2 = L[b2], K[b2]
            if K1 != K2 and K1 != K2 + 2 and K1 != K2 - 2:
                continue
            val = Rp * L2 * (L2 + 1) + (Rz - Rp) * K2**2 if K1 == K2 else 0.0
            val2 = Np(L2, K2) * (K1 == K2 + 2) + Nm(L2, K2) * (
                (K1 == K2 - 2) + jK1 * (-1.0) ** (L1 + K1) * (-K1 == K2 - 2)
            )
            val += Rd * val2 / math.sqrt((1 + (K1 == 0)) * (1 + (K2 == 0)))
            if val == 0:
                continue
            rows.append(b1)
            cols.append(b2)
            vals.append(val)
            if b1 != b2:
                rows.append(b2)
                cols.append(b1)
                vals.append(np.conj(val))
    Gamma = sps.coo_array((np.array(vals, dtype=complex), (rows, cols)), shape=(n, n)).tocsr()
    if not include_potential:
        return Gamma
    lx_max = len(XLMK) - 1

    def X(l, m, k):
        return XLMK[l][m + l, k + l]

    rows, cols, vals = [], [], []
    for b1 in range(n):
        L1, jK1, K1, M1 = L[b1], jK[b1], K[b1], M[b1]
        for b2 in range(b1, n):
            L2, M2, K2, jK2 = L[b2], M[b2], K[b2], jK[b2]
            if all_m_zero and M1 != M2:
                continue
            val = 0.0
            for Lx in range(abs(L1 - L2), min(lx_max, L1 + L2) + 1):
                if abs(M1 - M2) > Lx:
                    continue
                v = 0.0
                if abs(K1 - K2) <= Lx:
                    X1 = X(Lx, M1 - M2, K1 - K2) + X(Lx, M1 - M2, -K1 + K2) * jK1 * jK2 * (-1.0) ** (Lx + K1 + K2)
                    if X1 != 0:
                        v += X1 * wigner3j(L1, Lx, L2, -K1, K1 - K2, K2)
                if abs(K1 + K2) <= Lx:
                    X2 = X(Lx, M1 - M2, -K1 - K2) * jK1 * (-1.0) ** (K1 + Lx + L2) + X(Lx, M1 - M2, K1 + K2) * jK2 * (
                        -1
                    ) ** (L2 + K2)
                    if X2 != 0:
                        v += X2 * wigner3j(L1, Lx, L2, -K1, K1 + K2, -K2)
                if v == 0:
                    continue
                val += wigner3j(L1, Lx, L2, -M1, M1 - M2, M2) * v
            if val == 0:
                continue
            pre_jk = (
                np.conj(np.sqrt(complex(jK1)))
                * np.sqrt(complex(jK2))
                / 2
                / math.sqrt((1 + (K1 == 0)) * (1 + (K2 == 0)))
            )
            pre_l = math.sqrt((2 * L1 + 1) * (2 * L2 + 1))
            val = (-1.0) ** (K1 - M1) * pre_l * pre_jk * val
            rows.append(b1)
            cols.append(b2)
            vals.append(val)
            if b1 != b2:
                rows.append(b2)
                cols.append(b1)
                vals.append(np.conj(val))
    if rows:
        Gamma = Gamma + sps.coo_array((np.array(vals, dtype=complex), (rows, cols)), shape=(n, n)).tocsr()
    return Gamma.tocsr()


def _eqpopvec(basis: np.ndarray, pot_L, pot_M, pot_K, pot_lambda, tolerances=(1e-6, 1e-4, 1e-4)):
    """The vector of sqrt(Peq) in the LjKKM basis under an orientational potential (chili_eqpopvec)."""
    L, jK, K, M = basis[:, 0], basis[:, 1], basis[:, 2], basis[:, 3]
    n = L.size
    Lp = np.asarray(pot_L, dtype=int).ravel()
    Mp = np.asarray(pot_M, dtype=int).ravel()
    Kp = np.asarray(pot_K, dtype=int).ravel()
    lam = np.asarray(pot_lambda, dtype=complex).ravel()
    keep = (lam != 0) & ~((Lp == 0) & (Mp == 0) & (Kp == 0))
    Lp, Mp, Kp, lam = Lp[keep], Mp[keep], Kp[keep], lam[keep]
    threshold, abstol, reltol = tolerances
    if lam.size == 0:
        idx0 = np.nonzero((L == 0) & (M == 0) & (K == 0))[0]
        if idx0.size != 1:
            raise ValueError("Exactly one orientational basis function with L=M=K=0 is allowed.")
        v = np.zeros(n)
        v[idx0[0]] = 1.0
        return v
    zero_mp, zero_kp = bool(np.all(Mp == 0)), bool(np.all(Kp == 0))
    even_lp, even_mp, even_kp = bool(np.all(Lp % 2 == 0)), bool(np.all(Mp % 2 == 0)), bool(np.all(Kp % 2 == 0))

    def U(a, b, c):
        u = 0.0
        for p in range(lam.size):
            if Kp[p] == 0 and Mp[p] == 0:
                u = u - wignerd([Lp[p], Mp[p], Kp[p]], b) * lam[p].real
            else:
                u = u - 2 * np.real(wignerd([Lp[p], Mp[p], Kp[p]], a, b, c) * lam[p])
        return u

    def int_b(f):
        return quad(f, 0, np.pi, epsabs=abstol, epsrel=reltol, limit=200)[0]

    def int_ab(f):  # f(a, b)
        return dblquad(lambda b, a: f(a, b), 0, 2 * np.pi, 0, np.pi, epsabs=abstol, epsrel=reltol)[0]

    def int_bc(f):  # f(b, c)
        return dblquad(lambda c, b: f(b, c), 0, np.pi, 0, 2 * np.pi, epsabs=abstol, epsrel=reltol)[0]

    def int_abc(f):  # f(a, b, c)
        return tplquad(lambda c, b, a: f(a, b, c), 0, 2 * np.pi, 0, np.pi, 0, 2 * np.pi, epsabs=abstol, epsrel=reltol)[
            0
        ]

    if zero_mp and zero_kp:
        Z = (2 * np.pi) ** 2 * int_b(lambda b: np.exp(-U(0, b, 0)) * np.sin(b))
    elif zero_mp:
        Z = 2 * np.pi * int_bc(lambda b, c: np.exp(-U(0, b, c)) * np.sin(b))
    elif zero_kp:
        Z = 2 * np.pi * int_ab(lambda a, b: np.exp(-U(a, b, 0)) * np.sin(b))
    else:
        Z = int_abc(lambda a, b, c: np.exp(-U(a, b, c)) * np.sin(b))
    sqrtZ = math.sqrt(Z)
    sqrt_peq = np.zeros(n, dtype=complex)
    cache: dict = {}
    for b in range(n):
        L_, M_, K_, jK_ = int(L[b]), int(M[b]), int(K[b]), int(jK[b])
        if zero_mp:
            if M_ != 0 or (even_lp and L_ % 2) or (even_kp and K_ % 2) or jK_ != 1:
                continue
            key = (L_, M_, K_)
            if key not in cache:
                if zero_kp:
                    if K_ != 0:
                        continue
                    cache[key] = (2 * np.pi) ** 2 * int_b(
                        lambda bb, L_=L_: wignerd([L_, 0, 0], bb) * np.exp(-U(0, bb, 0) / 2) / sqrtZ * np.sin(bb)
                    )
                else:
                    cache[key] = (
                        2
                        * np.pi
                        * int_bc(
                            lambda bb, cc, L_=L_, K_=K_: (
                                np.cos(K_ * cc)
                                * wignerd([L_, 0, K_], bb)
                                * np.exp(-U(0, bb, cc) / 2)
                                / sqrtZ
                                * np.sin(bb)
                            )
                        )
                    )
            value = cache[key]
        elif zero_kp:
            if K_ != 0 or (even_lp and L_ % 2) or (even_mp and M_ % 2):
                continue
            key = (L_, M_, K_)
            if key not in cache:
                cache[key] = (
                    2
                    * np.pi
                    * int_ab(
                        lambda aa, bb, L_=L_, M_=M_: (
                            np.cos(M_ * aa) * wignerd([L_, M_, 0], bb) * np.exp(-U(aa, bb, 0) / 2) / sqrtZ * np.sin(bb)
                        )
                    )
                )
            value = cache[key]
        else:
            key = (L_, M_, K_)
            if key not in cache:
                re = int_abc(
                    lambda aa, bb, cc, L_=L_, M_=M_, K_=K_: np.real(
                        np.conj(wignerd([L_, M_, K_], aa, bb, cc)) * np.exp(-U(aa, bb, cc) / 2) / sqrtZ * np.sin(bb)
                    )
                )
                im = int_abc(
                    lambda aa, bb, cc, L_=L_, M_=M_, K_=K_: np.imag(
                        np.conj(wignerd([L_, M_, K_], aa, bb, cc)) * np.exp(-U(aa, bb, cc) / 2) / sqrtZ * np.sin(bb)
                    )
                )
                cache[key] = re + 1j * im
            value = cache[key]
        if abs(np.imag(value)) < 1e-5 * abs(np.real(value)):
            value = np.real(value)
        value = math.sqrt((2 * L_ + 1) / (8 * np.pi**2)) * value * math.sqrt(2 / (1 + (K_ == 0)))
        if abs(value) >= threshold:
            sqrt_peq[b] = value
    return sqrt_peq if np.any(np.imag(sqrt_peq)) else np.real(sqrt_peq)


# ----------------------------------------------------------------------------- solvers
def _lanczos(A, b, z, lentz: bool = True, threshold: float = 1e-6):
    """Spectral function b^T (A + z)^-1 b of the complex-symmetric A by Lanczos tridiagonalisation and the
    continued fraction (modified Lentz left to right, or right to left): ``(spec, converged, changes)``."""
    N = b.size
    alpha = np.zeros(N, dtype=complex)
    beta = np.zeros(N, dtype=complex)
    changes = []
    interval = min(10, int(math.ceil(N / 20)))
    z = np.asarray(z, dtype=complex)
    if lentz:
        tiny = 1e-30
        spec = np.full(z.shape, tiny, dtype=complex)
        C = spec.copy()
        D = np.zeros(z.shape, dtype=complex)
    else:
        z_test = np.linspace(z[0], z[-1], 201)
        old = np.inf
    converged = False
    q = b / np.sqrt(b @ b)  # pseudo-norm, no conjugation
    bq = 0.0
    k = 0
    for k in range(N):
        y = A @ q
        alpha[k] = q @ y
        y = y - alpha[k] * q - bq
        beta[k] = np.sqrt(y @ y)
        bq = beta[k] * q
        if beta[k] != 0:
            q = y / beta[k]
        if lentz:
            bk = alpha[k] + z
            a = 1.0 if k == 0 else -(beta[k - 1] ** 2)
            D = 1 / (bk + a * D)
            C = bk + a / C
            delta = C * D
            spec = spec * delta
            if (k + 1) % interval == 0:
                changes.append(np.max(np.abs(delta - 1)))
                converged = changes[-1] < threshold
        elif (k + 1) % interval == 0:
            s = _contfracspec(z_test, alpha, beta, k + 1)
            re = np.real(s)
            changes.append(np.max(np.abs(re - old)) / np.max(re))
            converged = changes[-1] < threshold
            old = re
        if converged:
            break
        if beta[k] == 0:  # the Krylov space is exhausted: the continued fraction is exact
            changes.append(0.0)
            converged = True
            break
    if not lentz:
        spec = _contfracspec(z, alpha, beta, k + 1)
    return spec, converged, np.array(changes)


def _contfracspec(z, alpha, beta, k):
    spec = np.full(np.shape(z), np.inf, dtype=complex)
    for m in range(k - 1, -1, -1):
        spec = z + alpha[m] - beta[m] ** 2 / spec
    return 1 / spec


def _conjgrad(A, b, shift):
    """Conjugate gradients with the Lanczos tridiagonal (diagonal T0, superdiagonal T1) from the CG scalars."""
    x = np.zeros_like(b)
    r = b - A @ x
    p = r.copy()
    rr = r @ r
    alpha, beta = [], []
    err = np.inf
    for step in range(b.size):
        rr_old = rr
        rr = r @ r
        beta_ = 0.0 if step == 0 else rr / rr_old
        if step > 0:
            p = r + beta_ * p
        Ap = A @ p + shift * p
        alpha_ = rr / (p @ Ap)
        x = x + alpha_ * p
        r = r - alpha_ * Ap
        alpha.append(alpha_)
        beta.append(beta_)
        err = np.sum(np.abs(r))
        if err < 1e-8:
            break
    steps = len(alpha)
    T0 = np.zeros(steps, dtype=complex)
    T1 = np.zeros(steps, dtype=complex)
    for k in range(1, steps):
        t1 = np.sqrt(beta[k]) / alpha[k - 1]
        T1[k - 1] = t1 * np.sign(np.real(t1))
        T0[k] = 1 / alpha[k] + beta[k] / alpha[k - 1]
    T0[0] = 1 / alpha[0]
    T0 = T0 - shift
    return x, T0, T1, err, steps


# ----------------------------------------------------------------------------- chili
def _process_dynamics(system: dict, field_sweep: bool):
    fields = [k for k in ("logtcorr", "tcorr", "Diff", "logDiff") if k in system]
    if len(fields) > 1:
        raise ValueError("Only one of Sys.tcorr, Sys.logtcorr, Sys.Diff, and Sys.logDiff is allowed.")
    if not fields:
        raise ValueError("One of Sys.tcorr, Sys.logtcorr, Sys.Diff, and Sys.logDiff is required.")
    key = fields[0]
    val = np.asarray(system[key], dtype=float).ravel()
    if key == "Diff":
        if np.any(val < 0):
            raise ValueError("Sys.Diff cannot be negative.")
        R = val
    elif key == "logDiff":
        R = 10.0**val
    elif key == "tcorr":
        R = 1 / 6 / val
    else:
        if np.any(val >= 0):
            raise ValueError("Sys.logtcorr must be negative.")
        R = 1 / 6 / 10.0**val
    if np.any(R < 0):
        raise ValueError("Negative diffusion rate or correlation times are not possible.")
    if R.size == 1:
        R = np.repeat(R, 3)
    elif R.size == 2:
        R = np.array([R[0], R[0], R[1]])
    elif R.size != 3:
        raise ValueError("Sys.Diff must have 1, 2 or 3 elements (isotropic, axial, rhombic).")
    lw = np.asarray(system.get("lw", 0), dtype=float).ravel()
    if "lwpp" in system:
        lwpp = np.asarray(system["lwpp"], dtype=float).ravel()
        lw = np.array([lwpp[0] * math.sqrt(2 * math.log(2)), lwpp[1] * math.sqrt(3) if lwpp.size > 1 else 0.0])
    if lw.size == 1:
        lw = np.append(lw, 0.0)
    if lw[1] != 0:
        lorentz_hz = lw[1] * 28 * 1e6 if field_sweep else lw[1] * 1e6
        T2 = 1 / lorentz_hz / np.pi
    else:
        T2 = np.inf
    return R, lw, T2


def chili(system: dict, exp: dict, opt: dict | None = None) -> Spectrum:  # noqa: C901
    """Slow-motion cw EPR spectrum by the stochastic Liouville equation: ``Spectrum(x, spec, info)``.

    ``system``: the spin system with ``tcorr``/``logtcorr``/``Diff``/``logDiff`` (1, 2 or 3 values),
    ``DiffFrame``, ``lw``/``lwpp`` (Gaussian by convolution, Lorentzian through T2), ``Exchange`` (1/us),
    ``Potential`` (rows ``[L, M, K, lambda]``). ``exp``: ``mwFreq`` with ``Range``/``CenterSweep`` (field
    sweep) or ``Field`` with ``mwRange``/``mwCenterSweep`` (frequency sweep), ``nPoints``, ``Harmonic``,
    ``ModAmp``, ``mwPhase``, ``Temperature``, ``SampleFrame`` (one director orientation), ``Ordering``.
    ``opt``: ``LLMK`` (``[evenLmax, oddLmax, Mmax, Kmax]``), ``evenK``, ``jKmin``, ``highField``, ``pImax``,
    ``pImaxall``, ``MpSymm``, ``GridSize``, ``GridSymmetry``, ``GridFrame``, ``PostConvNucs``, ``Solver``
    (``'L'`` Lanczos, ``'\\\\'`` direct, ``'E'`` eigenvalues, ``'C'`` conjugate gradients, ``'B'`` BiCGSTAB),
    ``FieldSweepMethod`` (``'approxlin'``, ``'approxinv'``, ``'explicit'``), ``Lentz``, ``Threshold``,
    ``Rescale``, ``IncludeNZI``, ``separate`` (``''`` or ``'components'``).
    """
    if isinstance(system, (list, tuple)):
        return compisoloop(chili, system, exp, opt)
    opt = dict(opt or {})
    exp = dict(exp)
    if any(not any(ch.isdigit() for ch in n) for n in nucstring2list(system.get("Nucs", ""))):
        raise NotImplementedError(
            "Isotope mixtures (elements without mass number in Sys.Nucs) are not ported; give isotopes."
        )
    if opt.get("LiouvMethod", "") == "fast":
        raise NotImplementedError(
            "Opt.LiouvMethod='fast' (EasySpin's C kernel) is not ported; the general method is used."
        )
    for obsolete, hint in (
        ("LLKM", "Use Opt.LLMK instead; M and K are swapped"),
        ("nKnots", "Use Opt.GridSize"),
        ("Symmetry", "Use Opt.GridSymmetry"),
        ("Output", "Use Opt.separate"),
        ("deltaK", "Use Opt.evenK"),
        ("pSmin", "Use Opt.highField"),
    ):
        if obsolete in opt:
            raise ValueError(f"Opt.{obsolete} is obsolete. {hint}.")
    separate = opt.get("separate", "") or ""
    if separate in ("transitions", "sites", "orientations"):
        raise ValueError(f"chili does not support Opt.separate='{separate}'.")
    if "lambda" in system:
        raise ValueError("Sys.lambda is obsolete. Use Sys.Potential = [L M K lambda; ...] instead.")
    sys = validatespinsys(system)
    if sys["nNuclei"] > 1 and np.size(sys.get("nn", [])) and np.any(np.asarray(sys["nn"]) != 0):
        raise ValueError("chili does not support nuclear-nuclear couplings (Sys.nn).")
    if np.any(sys["HStrain"]) or np.any(sys["gStrain"]) or np.any(sys["AStrain"]) or np.any(sys["DStrain"]):
        raise ValueError("chili does not support strains (HStrain, gStrain, AStrain, DStrain).")
    field_sweep = "mwFreq" in exp
    if field_sweep and "Field" in exp:
        raise ValueError(
            "Give either Exp.mwFreq (for a field sweep) or Exp.Field (for a frequency sweep), but not both."
        )
    if not field_sweep and "Field" not in exp:
        raise ValueError(
            "Please supply either the microwave frequency in Exp.mwFreq (for field sweeps) or the magnetic field "
            "in Exp.Field (for frequency sweeps)."
        )
    R, lw, T2 = _process_dynamics(system, field_sweep)
    convolution = lw[0] > 0
    exchange = float(system.get("Exchange", 0)) * 2 * np.pi * 1e6
    if exchange != 0:
        raise NotImplementedError(
            "Heisenberg exchange (Sys.Exchange) belongs to EasySpin's fast method and is not ported."
        )
    if "lightBeam" in exp and exp["lightBeam"]:
        raise ValueError("Photoselection (via Exp.lightBeam) is not supported.")
    if np.any(sys["n"] != 1) and not opt.get("PostConvNucs"):
        raise ValueError(
            "chili cannot handle systems with nuclei with Sys.n > 1 unless these nuclei are treated using "
            "post-convolution (Opt.PostConvNucs)."
        )

    # Potential
    potential = (
        np.atleast_2d(np.asarray(system.get("Potential", np.zeros((0, 4))), dtype=complex)).reshape(-1, 4)
        if np.size(system.get("Potential", []))
        else np.zeros((0, 4), dtype=complex)
    )
    pot_L, pot_M, pot_K, pot_lam = (
        potential[:, 0].real.astype(int),
        potential[:, 1].real.astype(int),
        potential[:, 2].real.astype(int),
        potential[:, 3],
    )
    keep = (pot_lam != 0) & ~((pot_L == 0) & (pot_M == 0) & (pot_K == 0))
    pot_L, pot_M, pot_K, pot_lam = pot_L[keep], pot_M[keep], pot_K[keep], pot_lam[keep]
    use_potential = pot_lam.size > 0
    if use_potential:
        if np.any(pot_L < 0):
            raise ValueError("L values of potential coefficients must be nonnegative.")
        if np.any(np.abs(pot_K) > pot_L) or np.any(np.abs(pot_M) > pot_L):
            raise ValueError("L, M and K values of potential coefficients must satisfy |M|<=L and |K|<=L.")
        if np.any(pot_K < 0):
            raise ValueError("Only potential terms with nonnegative values of K are allowed.")
        if np.any(pot_M[pot_K == 0] < 0):
            raise ValueError("For potential terms with K=0, M must be nonnegative.")
        if np.any(np.imag(pot_lam[(pot_K == 0) & (pot_M == 0)]) != 0):
            raise ValueError("Potential coefficients for M=K=0 must be real-valued.")

    # Experiment
    n_points = int(exp.get("nPoints", 1024))
    if n_points < 2:
        raise ValueError("Problem with Exp.nPoints. Needs to be a number >= 2.")
    harmonic = exp.get("Harmonic")
    mw_phase = float(exp.get("mwPhase", 0))
    temperature = exp.get("Temperature", np.nan)
    temperature = np.nan if temperature is None else float(temperature)
    if not np.isnan(temperature) and (np.isinf(temperature) or temperature < 0):
        raise ValueError("Problem with Exp.Temperature. If given, Exp.Temperature must be a positive value.")
    mod_amp = float(exp.get("ModAmp", 0))
    if exp.get("mwMode", "perpendicular") == "parallel":
        raise ValueError("chili does not support parallel-mode spectra.")
    if field_sweep:
        mw_freq = float(exp["mwFreq"])
        if mw_freq <= 0:
            raise ValueError("Uninterpretable microwave frequency in Exp.mwFreq.")
        if exp.get("CenterSweep") is not None:
            center_sweep = np.asarray(exp["CenterSweep"], dtype=float)
        elif exp.get("Range") is not None:
            rng = np.asarray(exp["Range"], dtype=float)
            center_sweep = np.array([rng.mean(), rng[1] - rng[0]])
        else:
            if sys["nElectrons"] != 1 or sys["S"][0] != 0.5:
                raise ValueError("Cannot automatically determine sweep range for this spin system.")
            hf = (
                float(np.sum(sys["I"] * np.max(np.abs(sys["A"][: sys["nNuclei"], :3]), axis=1))) * 1e6
                if sys["nNuclei"]
                else 0.0
            )
            gvals = np.linalg.eigvalsh(sys["g"][:3]) if sys["fullg"] else sys["g"].ravel()
            minB = planck * (mw_freq * 1e9 - hf) / bmagn / gvals.max() / 1e-3
            maxB = planck * (mw_freq * 1e9 + hf) / bmagn / gvals.min() / 1e-3
            center_sweep = np.array([(maxB + minB) / 2, 1.25 * max(maxB - minB, 5)])
        center_field = center_sweep[0]
        rng = center_field + np.array([-1, 1]) / 2 * center_sweep[1]
        if np.any(rng < 0) or rng[1] <= rng[0]:
            raise ValueError("Invalid sweep range! Check Exp.CenterSweep or Exp.Range.")
    else:
        center_field = float(exp["Field"])
        if center_field <= 0:
            raise ValueError("Uninterpretable magnetic field in Exp.Field.")
        if exp.get("mwCenterSweep") is not None:
            mw_cs = np.asarray(exp["mwCenterSweep"], dtype=float)
        elif exp.get("mwRange") is not None:
            mwr = np.asarray(exp["mwRange"], dtype=float)
            mw_cs = np.array([mwr.mean(), mwr[1] - mwr[0]])
        else:
            raise ValueError("Either Exp.mwRange or Exp.mwCenterSweep need to be given.")
        mw_range = mw_cs[0] + np.array([-1, 1]) / 2 * mw_cs[1]
        if np.any(mw_range < 0) or mw_range[1] <= mw_range[0]:
            raise ValueError("Invalid sweep range! Check Exp.mwCenterSweep or Exp.mwRange.")
    if harmonic is None or (np.isscalar(harmonic) and np.isnan(harmonic)):
        harmonic = 1 if field_sweep else 0
    harmonic = int(harmonic)
    if harmonic not in (-1, 0, 1, 2):
        raise ValueError("Exp.Harmonic must be 0, 1 or 2.")
    if mod_amp < 0 or np.isnan(mod_amp):
        raise ValueError("Exp.ModAmp must be either a single positive number or zero.")
    if mod_amp > 0:
        if not field_sweep:
            raise ValueError("Exp.ModAmp cannot be used with frequency sweeps.")
        if harmonic < 1:
            raise ValueError("With field modulation (Exp.ModAmp), Exp.Harmonic=0 does not work.")
        mod_harmonic, conv_harmonic, deriv_harmonic = harmonic, 0, 0
    else:
        mod_harmonic = 0
        conv_harmonic, deriv_harmonic = (harmonic, 0) if convolution else (0, harmonic)
    ordering = exp.get("Ordering")
    if ordering is not None:
        if np.isscalar(ordering):
            lam_ord = float(ordering)

            def ordering_fn(alpha, beta, gamma, lam_ord=lam_ord):
                return np.exp(lam_ord * (3 * np.cos(beta) ** 2 - 1) / 2) * np.ones_like(gamma)

        elif callable(ordering):
            ordering_fn = ordering
        else:
            raise ValueError("Exp.Ordering must be either a single number or a function handle.")
    sample_frame = exp.get("SampleFrame")
    integrate_over_grid = (sample_frame is None or np.size(sample_frame) == 0) and use_potential

    # Options
    if "LLMK" not in opt:
        if use_potential:
            raise ValueError("Sys.Potential is given. Please provide basis set information (Opt.LLMK etc).")
        opt["LLMK"] = [14, 7, 2, 6]
    llmk = np.asarray(opt["LLMK"], dtype=float).ravel()
    if llmk.size != 4 or np.any(llmk < 0) or np.any(llmk % 1):
        raise ValueError("Opt.LLMK must be a 4-element array with non-negative integers.")
    llmk = llmk.astype(int)
    if llmk[0] % 2:
        raise ValueError("Opt.LLMK(1) must be an even number.")
    if llmk[1] > 0 and llmk[1] % 2 == 0:
        raise ValueError("Opt.LLMK(2) must be an odd integer.")
    max_l = max(llmk[0], llmk[1])
    if llmk[2] > max_l or llmk[3] > max_l:
        raise ValueError("Opt.LLMK(3) and Opt.LLMK(4) must be smaller or equal to the maximum L.")
    high_field = bool(opt.get("highField", False))
    ps_min = 1 if high_field else -1
    grid_size = np.atleast_1d(np.asarray(opt.get("GridSize", [19, 0]), dtype=int))
    if grid_size.size > 1 and grid_size[1] != 0:
        raise ValueError("chili cannot interpolate orientations. Set Opt.GridSize(2) to zero.")
    grid_symmetry = opt.get("GridSymmetry", "Dinfh") or "Dinfh"
    field_sweep_method = opt.get("FieldSweepMethod") or "approxlin"
    if field_sweep_method not in ("explicit", "approxinv", "approxlin"):
        raise ValueError(f"Unknown setting '{field_sweep_method}' in Opt.FieldSweepMethod.")
    explicit_sweep = field_sweep and field_sweep_method == "explicit"
    solver = opt.get("Solver") or ("\\" if explicit_sweep else "L")
    if solver not in ("L", "\\", "E", "C", "B"):
        raise ValueError("Unknown method in Options.Solver. Must be 'L', '\\', 'E', 'C', or 'B'.")
    if explicit_sweep and solver != "\\":
        raise ValueError("For an explicit field sweep, use Opt.Solver='\\'.")
    rescale = bool(opt.get("Rescale", True))
    threshold = float(opt.get("Threshold", 1e-6))
    lentz = bool(opt.get("Lentz", True))
    include_nzi = bool(opt.get("IncludeNZI", True))
    mp_symm = bool(opt.get("MpSymm", False))
    post_conv = opt.get("PostConvNucs")
    post_conv = (
        sorted(set(int(k) for k in np.atleast_1d(post_conv))) if post_conv is not None and np.size(post_conv) else []
    )
    full_system = system
    if post_conv:
        if any(k < 1 or k > sys["nNuclei"] for k in post_conv):
            raise ValueError(f"Opt.PostConvNucs must contain indices of nuclei (1 to {sys['nNuclei']}).")
        reduced = nucspinrmv({k: v for k, v in system.items() if k != "processed"}, post_conv)
        sys = validatespinsys(reduced)
    if np.any(sys["n"] != 1):
        raise ValueError(
            "chili cannot handle systems with nuclei with Sys.n > 1 unless these nuclei are treated using "
            "post-convolution (Opt.PostConvNucs)."
        )

    # Spin operators and interactions
    spins = sys["Spins"]
    spin_ops = [
        [sop(spins, [[k + 1, c + 1]], sparse=True).astype(complex) for c in range(3)] for k in range(len(spins))
    ]
    sdet = sum(spin_ops[e][0] for e in range(sys["nElectrons"]))
    T, F, symmetry, field_dep = _magint(sys, spin_ops, center_field, include_nzi, explicit_sweep)
    if np.all(F[1] == 0) and np.all(F[2] == 0):
        raise ValueError("This is an isotropic spin system. chili cannot calculate a slow-motion spectrum.")

    # Basis
    jk_min = opt.get("jKmin")
    if jk_min is None:
        jk_min = 1 if symmetry["tensorsCollinear"] else -1
    jk_min = int(jk_min)
    if jk_min not in (-1, 1):
        raise ValueError("Opt.jKmin must be either -1 or +1.")
    even_k = opt.get("evenK")
    even_k = symmetry["nobetatilts"] if even_k is None else bool(even_k)
    I = sys["I"]
    n_nuc = sys["nNuclei"]
    p_imax = opt.get("pImax")
    if n_nuc == 0:
        p_imax_arr = np.zeros(0)
        p_imaxall = 0
    else:
        p_imax_arr = (
            2 * I
            if p_imax is None or np.size(p_imax) == 0
            else np.minimum(
                np.broadcast_to(np.asarray(p_imax, dtype=float).ravel(), (n_nuc,))
                if np.size(p_imax) in (1, n_nuc)
                else None,
                2 * I,
            )
        )
        if p_imax_arr is None:
            raise ValueError("Opt.pImax must contain one entry for every nucleus or just a single number.")
        if np.any(p_imax_arr < 0):
            raise ValueError("Every element in Opt.pImax must be 0 or larger.")
        p_imaxall = opt.get("pImaxall")
        p_imaxall = float(np.sum(p_imax_arr)) if p_imaxall is None else min(float(p_imaxall), float(np.sum(2 * I)))
    basis = _generateoribasis(llmk[0], llmk[1], llmk[2], llmk[3], jk_min, even_k)
    n_ori = basis.shape[0]
    n_states = sys["nStates"]
    n_spin = n_states**2
    pq = _pqorder(spins)
    keep = np.ones(n_spin, bool)
    for e in range(sys["nElectrons"]):
        keep &= pq[:, 2 * e] >= ps_min
    pi_sum = np.zeros(n_spin)
    for n in range(n_nuc):
        pI = pq[:, 2 * sys["nElectrons"] + 2 * n]
        pi_sum += pI
        keep &= np.abs(pI) <= p_imax_arr[n]
    keep &= np.abs(pi_sum) <= p_imaxall
    keep_full = np.tile(keep, n_ori)
    if mp_symm:
        psum = np.sum(pq[:, 0::2], axis=1)
        keep_mp = (psum[None, :] - basis[:, 3][:, None]) == 1
        keep_full &= keep_mp.ravel()
    basis_size = int(keep_full.sum())

    # Diffusion operator and starting vector
    XLMK = _xlmk(pot_L, pot_M, pot_K, pot_lam, R) if use_potential else []
    gamma = _diffsuperop(basis, R, XLMK, bool(np.all(pot_M == 0)))
    gamma = sps.kron(gamma, sps.identity(n_spin, format="csr"), format="csr")
    gamma = gamma[keep_full][:, keep_full]
    sqrt_peq = _eqpopvec(basis, pot_L, pot_M, pot_K, pot_lam, opt.get("PeqTol") or (1e-6, 1e-4, 1e-4))
    sdet_vec = np.asarray(sdet.todense()).ravel(order="F")  # column-major, as MATLAB's SdetOp(:)
    start = np.kron(sqrt_peq, sdet_vec / np.linalg.norm(sdet_vec))[keep_full]
    norm_peq = np.linalg.norm(sqrt_peq) ** 2
    info = {"nBasis": basis_size, "nOriBasis": n_ori, "normPeq": norm_peq}
    if norm_peq < 0.99:
        info["warning"] = (
            f"The norm of the equilibrium population vector in this truncated basis is {norm_peq:0.3g}. "
            "It should be close to 1. The basis might be too small."
        )
    start = start / np.linalg.norm(start)

    # Field and frequency axes
    gavg = (
        float(
            np.mean(
                [np.mean(np.linalg.eigvals(sys["g"][3 * e : 3 * e + 3, :3]).real) for e in range(sys["nElectrons"])]
            )
        )
        if sys["fullg"]
        else float(np.mean(sys["g"][:, :3]))
    )
    if field_sweep:
        B_ = np.linspace(rng[0], rng[1], n_points)
        if field_sweep_method == "explicit":
            B0 = B_
            nu = np.array([mw_freq])
        elif field_sweep_method == "approxinv":
            B0 = np.array([center_field])
            nu = mw_freq * center_field / B_
        else:
            B0 = np.array([center_field])
            nu = mw_freq - unitconvert(B_ - center_field, "mT->MHz", gavg) / 1e3
        x_axis = B_
    else:
        B0 = np.array([center_field])
        nu = np.linspace(mw_range[0], mw_range[1], n_points)
        x_axis = nu
    B0_T = B0 / 1e3
    omega0 = 2 * np.pi * nu * 1e9 + 1j / T2

    # Orientations of the director
    if integrate_over_grid:
        if grid_size[0] == 1:
            phi, theta, weights = np.zeros(1), np.zeros(1), np.array([4 * np.pi])
        else:
            grid = sphgrid(grid_symmetry, int(grid_size[0]))
            vecs = grid.vecs if opt.get("GridFrame") is None else np.asarray(opt["GridFrame"], dtype=float) @ grid.vecs
            phi, theta = vec2ang(vecs)
            weights = grid.weights
    else:
        if sample_frame is not None and np.size(sample_frame) > 0:
            sf = np.asarray(sample_frame, dtype=float).ravel()
            phi, theta = np.array([-sf[2]]), np.array([-sf[1]])
        else:
            phi, theta = np.zeros(1), np.zeros(1)
        weights = np.array([4 * np.pi])
    n_orientations = phi.size
    if ordering is not None:
        ow = np.asarray(
            [float(np.asarray(ordering_fn(0.0, -t, -p)).ravel()[0]) for p, t in zip(phi, theta, strict=True)]
        )
        if np.any(ow < 0):
            raise ValueError("User-supplied orientation distribution gives negative values!")
        if np.all(ow == 0):
            raise ValueError("User-supplied orientation distribution is all-zero.")
        weights = weights * ow
    weights = 4 * np.pi * weights / weights.sum()

    # Orientation loop
    spec = np.zeros(nu.size if not explicit_sweep else n_points, dtype=complex)
    scale = 1.0
    for iOri in range(n_orientations):
        (Q0B, Q1B, Q2B), (Q0G, Q1G, Q2G) = _rbos(T, F, (phi[iOri], theta[iOri], 0.0), field_dep)
        if explicit_sweep:
            HB = _liouvhamiltonian_full(basis, Q0B, Q1B, Q2B)[keep_full][:, keep_full]
            HG = _liouvhamiltonian_full(basis, Q0G, Q1G, Q2G)[keep_full][:, keep_full]
        else:
            Q0 = Q0B + Q0G
            Q1 = [[Q1B[i][j] + Q1G[i][j] for j in range(3)] for i in range(3)] if Q1B is not None else None
            Q2 = [[Q2B[i][j] + Q2G[i][j] for j in range(5)] for i in range(5)]
            H = _liouvhamiltonian_full(basis, Q0, Q1, Q2)[keep_full][:, keep_full]
        this_spec = np.zeros_like(spec)
        for iB in range(B0.size):
            if explicit_sweep:
                H = B0_T[iB] * HB + HG
            Lmat = (2j * np.pi * H + gamma).tocsr()
            if np.any(np.isnan(Lmat.data)):
                raise ValueError("Liouvillian matrix contains NaN entries! Please report.")
            if rescale:
                scale = float(np.max(np.abs(Lmat.data)))
                Lmat = Lmat / scale
                omega = omega0 / scale
            else:
                scale = 1.0
                omega = omega0
            if solver == "L":
                if use_potential:
                    diff = Lmat - Lmat.T
                    asym = np.max(np.abs(diff.data)) if diff.nnz else 0.0
                    if asym > 1e-10 * np.max(np.abs(Lmat.data)):
                        raise ValueError("L is not complex symmetric - cannot use Lanczos method.")
                s, converged, changes = _lanczos(Lmat, start.astype(complex), -1j * omega, lentz, threshold)
                if not converged:
                    s = np.ones(omega.shape, dtype=complex)
                    info["warning"] = (
                        f"Tridiagonalization did not converge to within {threshold:g} after {basis_size} steps! "
                        f"Increase Opt.LLMK (current settings {llmk.tolist()})"
                    )
                this_spec = this_spec + s if not explicit_sweep else this_spec
                if explicit_sweep:
                    this_spec[iB] = s[0]
            elif solver == "\\":
                eye = sps.identity(Lmat.shape[0], format="csc", dtype=complex)
                if explicit_sweep:
                    Q = (Lmat - 1j * omega[0] * eye).tocsc()
                    this_spec[iB] = np.conj(start) @ sps.linalg.spsolve(Q, start.astype(complex))
                else:
                    for k, om in enumerate(omega):
                        Q = (Lmat - 1j * om * eye).tocsc()
                        this_spec[k] = np.conj(start) @ sps.linalg.spsolve(Q, start.astype(complex))
            elif solver == "E":
                Lam, U = np.linalg.eig(Lmat.toarray())
                amp = (np.conj(start) @ U) * np.linalg.solve(U, start.astype(complex))
                this_spec = np.sum(amp[:, None] / (Lam[:, None] - 1j * omega[None, :]), axis=0)
            elif solver == "B":
                eye = sps.identity(Lmat.shape[0], format="csr", dtype=complex)
                for k, om in enumerate(omega):
                    Q = (Lmat - 1j * om * eye).tocsr()
                    u, _ = sps.linalg.bicgstab(Q, start.astype(complex), rtol=threshold, maxiter=Lmat.shape[0])
                    this_spec[k] = np.conj(start) @ u
            else:
                cg_shift = 1e-6 + 1e-6j
                _, alpha, beta, _, _ = _conjgrad(Lmat, start.astype(complex), cg_shift)
                this_spec = _contfracspec(-1j * omega, alpha, beta, alpha.size)
        spec = spec + this_spec * weights[iOri]
    spec = spec / scale
    spec = spec * 1e10 / 2
    if high_field:
        spec = spec / 2
    if not field_sweep:
        spec = spec * 1e3
        mw_phase = -mw_phase
    spec = np.real(np.exp(1j * mw_phase) * spec)

    # Post-convolution with the isotropic spectrum of the remaining nuclei
    if post_conv:
        full = validatespinsys(full_system)
        pc_sys = {
            "g": float(np.mean(full["g"][:, :3])),
            "Nucs": ",".join(nucstring2list(full_system["Nucs"])[k - 1] for k in post_conv),
        }
        pc_sys["A"] = [float(np.mean(full["A"][k - 1, :3])) for k in post_conv]
        if "n" in full_system:
            pc_sys["n"] = [int(np.atleast_1d(full_system["n"])[k - 1]) for k in post_conv]
        if field_sweep:
            pc_exp = {
                "Range": rng.tolist(),
                "mwFreq": float(unitconvert(np.mean(rng), "mT->MHz", pc_sys["g"]) / 1e3),
                "Harmonic": 0,
                "nPoints": n_points,
            }
            dx = (rng[1] - rng[0]) / (n_points - 1)
        else:
            pc_exp = {"mwRange": mw_range.tolist(), "Field": center_field, "Harmonic": 0, "nPoints": n_points}
            dx = (mw_range[1] - mw_range[0]) * 1e3 / (n_points - 1)
        pc_sys["lw"] = dx / 12
        spec_pc = garlic(pc_sys, pc_exp).spec
        spec_pc = spec_pc / spec_pc.sum()
        full = np.convolve(
            spec, spec_pc, mode="full"
        )  # MATLAB's conv(...,'same') centring, one point later than NumPy's
        start = spec_pc.size // 2
        spec = full[start : start + spec.size]

    # Temperature, convolution, modulation, derivatives
    if np.isfinite(temperature):
        dE = planck * mw_freq * 1e9 if field_sweep else bmagn * gavg * center_field * 1e-3
        e = np.exp(-dE / boltzm / temperature)
        spec = spec * (1 - e) / (1 + e)
    if convolution:
        fwhm = lw[0] if field_sweep else lw[0] / 1e3
        dx = x_axis[1] - x_axis[0]
        spec = convspec(spec, dx, fwhm, conv_harmonic, 1)
    out = spec
    if field_sweep and mod_amp > 0:
        out = fieldmod(x_axis, out, mod_amp, mod_harmonic)
    if deriv_harmonic > 0:
        dx = x_axis[1] - x_axis[0]
        for _ in range(deriv_harmonic):
            d = np.diff(out) / dx
            out = (np.concatenate([d[:1], d]) + np.concatenate([d, d[-1:]])) / 2
    return Spectrum(x_axis, out, info)
