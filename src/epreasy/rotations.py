"""Rotations and orientations: erot, eulang, ang2vec, vec2ang, letter2vec.

EasySpin's Euler angles ``[alpha, beta, gamma]`` describe a *passive*
rotation (a coordinate transformation) about z, y', z'' in that order;
``erot`` returns the matrix ``Rp`` with ``vec2 = Rp @ vec1`` and
``mat2 = Rp @ mat1 @ Rp.T`` taking vectors and matrices from frame 1 to
frame 2. Spynach's ``euler2dcm`` is Spinach's *active* ZYZ rotation; the two
are transposes of each other for the same angles.

EasySpin authors, by original file (contributors to the file up to EasySpin 6.0.15, see
  docs/easyspin-credits.md):
  ang2vec.m: Stefan Stoll
  erot.m: Stefan Stoll
  eulang.m: Stefan Stoll, Claudia Tait
  letter2vec.m: Stefan Stoll
  vec2ang.m: Stefan Stoll
"""

from __future__ import annotations

import numpy as np

_LETTERS = {
    "x": (1, 0, 0),
    "y": (0, 1, 0),
    "z": (0, 0, 1),
    "xy": (1, 1, 0),
    "xz": (1, 0, 1),
    "yz": (0, 1, 1),
    "xyz": (1, 1, 1),
}


def erot(*angles) -> np.ndarray:
    """Passive rotation matrix from Euler angles (radians): ``erot([a, b, g])`` or ``erot(a, b, g)``.

    EasySpin authors (erot.m): Stefan Stoll.
    """
    if len(angles) == 1:
        angles = np.asarray(angles[0], dtype=float).ravel()
        if angles.size != 3:
            raise ValueError(f"Three angles expected, got {angles.size} elements.")
        alpha, beta, gamma = angles
    elif len(angles) == 3:
        alpha, beta, gamma = (float(a) for a in angles)
    else:
        raise TypeError("erot takes a 3-element array or three angles.")
    if np.isnan(alpha) or np.isnan(beta) or np.isnan(gamma):
        raise ValueError("At least one of the angles is NaN. Angles must be numbers.")
    sa, ca = np.sin(alpha), np.cos(alpha)
    sb, cb = np.sin(beta), np.cos(beta)
    sg, cg = np.sin(gamma), np.cos(gamma)
    return np.array(
        [
            [cg * cb * ca - sg * sa, cg * cb * sa + sg * ca, -cg * sb],
            [-sg * cb * ca - cg * sa, -sg * cb * sa + cg * ca, sg * sb],
            [sb * ca, sb * sa, cb],
        ]
    )


def eulang(R, nocheck: bool = False) -> np.ndarray:
    """Euler angles ``[alpha, beta, gamma]`` (radians, ``beta >= 0``) of a rotation matrix ``R``.

    A nearly orthogonal matrix is orthogonalised by SVD; a far-from-orthogonal
    one or a negative determinant raises. ``nocheck=True`` skips the checks.

    EasySpin authors (eulang.m): Stefan Stoll, Claudia Tait.
    """
    R = np.asarray(R, dtype=float)
    if not nocheck:
        if R.shape != (3, 3) or np.iscomplexobj(R):
            raise ValueError("eulang: Rotation matrix must be a real-valued 3x3 matrix.")
        orthogonality_error = np.linalg.norm(R.T @ R - np.eye(3))
        if orthogonality_error > 1e-2:
            raise ValueError(f"eulang: Rotation matrix is not orthogonal, deviation is {orthogonality_error:f}.")
        if orthogonality_error > 1e-6:
            U, _, Vt = np.linalg.svd(R)
            R = U @ Vt
        if np.linalg.det(R) < 0:
            raise ValueError("eulang: Rotation matrix has negative determinant. Change the signs in one column or row.")
    limit = 1e-8
    if abs(R[2, 2] - 1) <= limit:
        alpha, beta, gamma = np.arctan2(R[0, 1], R[1, 1]), 0.0, 0.0
    elif abs(R[2, 2] + 1) <= limit:
        alpha, beta, gamma = np.arctan2(-R[0, 1], R[1, 1]), np.pi, 0.0
    else:
        alpha = np.arctan2(R[2, 1], R[2, 0])
        beta = np.arctan2(np.hypot(R[2, 0], R[2, 1]), R[2, 2])
        gamma = np.arctan2(R[1, 2], -R[0, 2])
    thr = -1e-8
    if alpha < thr:
        alpha += 2 * np.pi
    if gamma < thr:
        gamma += 2 * np.pi
    if beta < 0:
        beta = -beta
        if alpha < 0:
            alpha, gamma = alpha + np.pi, gamma + np.pi
        else:
            alpha, gamma = alpha - np.pi, gamma - np.pi
    return np.array([alpha, beta, gamma])


def letter2vec(s: str) -> np.ndarray:
    """Unit vector for a direction letter: ``'x'``, ``'xy'``, ``'-z'``, ``'xyz'``, ...

    EasySpin authors (letter2vec.m): Stefan Stoll.
    """
    if not isinstance(s, str):
        raise TypeError("Input argument must be a string.")
    sign = -1.0 if s.startswith("-") else 1.0
    key = s[1:] if s.startswith("-") else s
    if key not in _LETTERS:
        raise ValueError(f"Unknown value '{s}' for direction.")
    n = sign * np.array(_LETTERS[key], dtype=float)
    return n / np.linalg.norm(n)


def ang2vec(phi, theta) -> np.ndarray:
    """Unit vectors (3, n) from polar angles ``phi`` (azimuth) and ``theta`` (from z), radians; one may be scalar.

    EasySpin authors (ang2vec.m): Stefan Stoll.
    """
    phi = np.asarray(phi, dtype=float).ravel()
    theta = np.asarray(theta, dtype=float).ravel()
    if phi.size != 1 and theta.size != 1 and phi.size != theta.size:
        raise ValueError("phi and theta must have the same number of elements!")
    phi, theta = np.broadcast_arrays(phi, theta)
    st = np.sin(theta)
    return np.vstack([st * np.cos(phi), st * np.sin(phi), np.cos(theta)])


def vec2ang(v) -> tuple[np.ndarray, np.ndarray]:
    """Polar angles ``(phi, theta)`` of vectors given as a direction letter, a 3-vector, or a (3, n) or (n, 3) array.

    EasySpin authors (vec2ang.m): Stefan Stoll.
    """
    if isinstance(v, str):
        v = letter2vec(v)
    v = np.asarray(v, dtype=float)
    if v.ndim == 1:
        v = v.reshape(3, 1)
    if v.shape[0] != 3 and v.shape[1] == 3:
        v = v.T
    if v.shape[0] != 3:
        raise ValueError("At least one dimension of the input array must be 3.")
    x, y, z = v
    phi = np.arctan2(y, x)
    theta = np.pi / 2 - np.arctan2(z, np.hypot(x, y))
    return phi, theta
