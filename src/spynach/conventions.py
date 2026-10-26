"""Rotation and tensor conventions (ports of ``euler2dcm.m``, ``mat2sphten.m``, ``wigner.m``, ``eeqq2nqi.m``).

Euler angles are Spinach's active ZYZ convention: ``euler2dcm(alpha,
beta, gamma)`` is ``Rz(alpha) Ry(beta) Rz(gamma)``, counter-clockwise.
Irreducible spherical components follow Mueller's convention, with the
projection index running from ``+l`` down to ``-l``.

Spinach authors, by original file (headers of Spinach 2.13; see docs/spinach-credits.md):
  eeqq2nqi.m: Ilya Kuprov
  euler2dcm.m: Ilya Kuprov
  mat2sphten.m: Ilya Kuprov
  wigner.m: Ilya Kuprov
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import scipy.linalg as sla


def euler2dcm(angles: Sequence[float]) -> np.ndarray:
    """Direction cosine matrix of ZYZ Euler angles ``(alpha, beta, gamma)`` in radians.

    Spinach authors (euler2dcm.m): Ilya Kuprov.
    """
    alpha, beta, gamma = (float(a) for a in angles)
    ca, sa, cb, sb, cg, sg = np.cos(alpha), np.sin(alpha), np.cos(beta), np.sin(beta), np.cos(gamma), np.sin(gamma)
    r_alpha = np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]])
    r_beta = np.array([[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]])
    r_gamma = np.array([[cg, -sg, 0], [sg, cg, 0], [0, 0, 1]])
    return r_alpha @ r_beta @ r_gamma


def mat2sphten(M: np.ndarray | None) -> tuple[float, np.ndarray, np.ndarray]:
    """Isotropic part, rank-1 components ``[+1, 0, -1]`` and rank-2 components ``[+2, ..., -2]`` of a 3x3 tensor.

    Spinach authors (mat2sphten.m): Ilya Kuprov.
    """
    M = np.zeros((3, 3)) if M is None else np.asarray(M, dtype=float)
    if M.shape != (3, 3):
        raise ValueError("the argument must be a real 3x3 matrix.")
    rank0 = np.trace(M) / 3
    rank1 = np.array(
        [
            -0.5 * (M[2, 0] - M[0, 2] - 1j * (M[2, 1] - M[1, 2])),
            (1j / np.sqrt(2)) * (M[0, 1] - M[1, 0]),
            -0.5 * (M[2, 0] - M[0, 2] + 1j * (M[2, 1] - M[1, 2])),
        ]
    )
    rank2 = np.array(
        [
            0.5 * (M[0, 0] - M[1, 1] - 1j * (M[0, 1] + M[1, 0])),
            -0.5 * (M[0, 2] + M[2, 0] - 1j * (M[1, 2] + M[2, 1])),
            (1 / np.sqrt(6)) * (2 * M[2, 2] - M[0, 0] - M[1, 1]),
            0.5 * (M[0, 2] + M[2, 0] + 1j * (M[1, 2] + M[2, 1])),
            0.5 * (M[0, 0] - M[1, 1] + 1j * (M[0, 1] + M[1, 0])),
        ]
    )
    return float(rank0), rank1, rank2


def wigner(rank: int, alpha: float, beta: float, gamma: float) -> np.ndarray:
    """Wigner rotation matrix ``D^rank(alpha, beta, gamma)``, rows and columns from ``+rank`` down to ``-rank``.

    Ranks 1 and 2 use closed forms (as ``wigner.m``); higher ranks use the
    exponentials of the spin operators of multiplicity ``2*rank + 1``.

    Spinach authors (wigner.m): Ilya Kuprov.
    """
    if rank < 0 or rank != int(rank):
        raise ValueError("rank must be a non-negative integer.")
    if rank == 1:
        egp1, egm1 = np.exp(-1j * gamma), np.exp(1j * gamma)
        eap1, eam1 = np.exp(-1j * alpha), np.exp(1j * alpha)
        sb, cb = np.sin(beta), np.cos(beta)
        return np.array(
            [
                [eap1 * egp1 * (0.5 + cb / 2), -eap1 * sb / np.sqrt(2), eap1 * egm1 * (0.5 - cb / 2)],
                [egp1 * sb / np.sqrt(2), cb, -egm1 * sb / np.sqrt(2)],
                [eam1 * egp1 * (0.5 - cb / 2), eam1 * sb / np.sqrt(2), eam1 * egm1 * (0.5 + cb / 2)],
            ]
        )
    if rank == 2:
        egp2, egp1, egm1, egm2 = np.exp(-2j * gamma), np.exp(-1j * gamma), np.exp(1j * gamma), np.exp(2j * gamma)
        eap2, eap1, eam1, eam2 = np.exp(-2j * alpha), np.exp(-1j * alpha), np.exp(1j * alpha), np.exp(2j * alpha)
        sb, cb = np.sin(beta), np.cos(beta)
        cbh4, sbh4 = np.cos(beta / 2) ** 4, np.sin(beta / 2) ** 4
        s38, s32 = np.sqrt(3 / 8), np.sqrt(3 / 2)
        return np.array(
            [
                [
                    eap2 * cbh4 * egp2,
                    eap2 * (-sb * (1 + cb) / 2) * egp1,
                    eap2 * s38 * sb**2,
                    eap2 * (-sb * (1 - cb) / 2) * egm1,
                    eap2 * sbh4 * egm2,
                ],
                [
                    eap1 * (sb * (cb + 1) / 2) * egp2,
                    eap1 * (2 * cb - 1) * (1 + cb) / 2 * egp1,
                    eap1 * (-s32 * sb * cb),
                    eap1 * (2 * cb + 1) * (1 - cb) / 2 * egm1,
                    eap1 * (sb * (cb - 1) / 2) * egm2,
                ],
                [
                    s38 * sb**2 * egp2,
                    s32 * sb * cb * egp1,
                    (3 * cb**2 - 1) / 2,
                    -s32 * sb * cb * egm1,
                    s38 * sb**2 * egm2,
                ],
                [
                    -eam1 * (sb * (cb - 1) / 2) * egp2,
                    eam1 * (2 * cb + 1) * (1 - cb) / 2 * egp1,
                    eam1 * (s32 * sb * cb),
                    eam1 * (2 * cb - 1) * (1 + cb) / 2 * egm1,
                    eam1 * (-sb * (cb + 1) / 2) * egm2,
                ],
                [
                    eam2 * sbh4 * egp2,
                    eam2 * sb * (1 - cb) / 2 * egp1,
                    eam2 * s38 * sb**2,
                    eam2 * sb * (1 + cb) / 2 * egm1,
                    eam2 * cbh4 * egm2,
                ],
            ]
        )
    from spynach.operators import pauli  # noqa: PLC0415  (operators imports system, which imports this module)

    ops = pauli(2 * rank + 1)
    Lz, Ly = ops["z"].toarray(), ops["y"].toarray()
    return sla.expm(-1j * Lz * alpha) @ sla.expm(-1j * Ly * beta) @ sla.expm(-1j * Lz * gamma)


def eeqq2nqi(eeqq: float, eta: float, spin: float, euler: Sequence[float]) -> np.ndarray:
    """Quadrupolar coupling tensor (Hz, traceless) from the quadrupolar constant ``eeqq`` (Hz),
    asymmetry ``eta``, spin quantum number and Euler angles of the principal frame (port of ``eeqq2nqi.m``).

    Spinach authors (eeqq2nqi.m): Ilya Kuprov.
    """
    if spin < 1 or spin != round(2 * spin) / 2:
        raise ValueError("quadrupolar coupling needs a spin of at least 1.")
    R = euler2dcm(euler)
    principal = np.diag([-(1 - eta), -(1 + eta), 2]) * eeqq / (4 * spin * (2 * spin - 1))
    return R @ principal @ R.T
