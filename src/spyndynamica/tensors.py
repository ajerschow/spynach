"""Irreducible spherical tensors: conversion from and to Cartesian tensors, and rotation between frames.

A spherical tensor of rank ``k`` is the list of its ``2k + 1`` components from ``m = +k`` down, as in
SpinDynamica's ``SphericalTensor``. The conventions are SpinDynamica's: a vector has ``T(1, ±1) = ∓(x ± iy)/√2``
and ``T(1, 0) = z``; a matrix ``A`` has ``T(0, 0) = −Tr A/√3``, ``T(2, 0) = (3 A_zz − Tr A)/√6``,
``T(2, ±1) = ∓[(A_xz + A_zx) ± i(A_yz + A_zy)]/2`` and ``T(2, ±2) = [(A_xx − A_yy) ± i(A_xy + A_yx)]/2``.

Author: Alexej Jerschow.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import sympy as sp

from spyndynamica._local import wigner_matrix


def cartesian_to_spherical(rank: int, cartesian: Any) -> np.ndarray:
    """The rank-``rank`` spherical tensor of a scalar, a 3-vector or a 3 × 3 matrix (``CartesianToSpherical``)."""
    a = np.asarray(cartesian, dtype=complex)
    if a.ndim == 0:
        if rank != 0:
            raise ValueError("a scalar has a rank-0 part only.")
        return np.array([a])
    if a.shape == (3,):
        if rank != 1:
            raise ValueError("a vector has a rank-1 part only.")
        x, y, z = a
        return np.array([-(x + 1j * y) / np.sqrt(2), z, (x - 1j * y) / np.sqrt(2)])
    if a.shape != (3, 3):
        raise ValueError("expected a scalar, a 3-vector or a 3 x 3 matrix.")
    if rank == 0:
        return np.array([-np.trace(a) / np.sqrt(3)])
    if rank == 1:
        zx, zy, xy = a[2, 0] - a[0, 2], a[2, 1] - a[1, 2], a[0, 1] - a[1, 0]
        return np.array([-(zx + 1j * zy) / 2, -1j * xy / np.sqrt(2), -(zx - 1j * zy) / 2])
    if rank == 2:
        xz, yz = a[0, 2] + a[2, 0], a[1, 2] + a[2, 1]
        dd, xy = a[0, 0] - a[1, 1], a[0, 1] + a[1, 0]
        zero = (3 * a[2, 2] - np.trace(a)) / np.sqrt(6)
        return np.array([(dd + 1j * xy) / 2, -(xz + 1j * yz) / 2, zero, (xz - 1j * yz) / 2, (dd - 1j * xy) / 2])
    raise ValueError("the rank is 0, 1 or 2.")


def spherical_to_cartesian(components: Sequence[complex]) -> np.ndarray:
    """The 3 × 3 matrix of a spherical tensor of rank 0, 1 or 2 (``SphericalToCartesian``): the isotropic, the
    antisymmetric or the symmetric traceless part."""
    t = np.asarray(components, dtype=complex)
    rank = (len(t) - 1) // 2
    if len(t) not in (1, 3, 5):
        raise ValueError("expected 1, 3 or 5 components.")
    if rank == 0:
        return -t[0] / np.sqrt(3) * np.eye(3, dtype=complex)
    if rank == 1:
        zx, zy, xy = -(t[0] + t[2]), 1j * (t[0] - t[2]), 1j * np.sqrt(2) * t[1]
        return np.array([[0, xy, -zx], [-xy, 0, -zy], [zx, zy, 0]]) / 2
    dd, xy = t[0] + t[4], -1j * (t[0] - t[4])
    xz, yz = -(t[1] - t[3]), 1j * (t[1] + t[3])
    zz = np.sqrt(6) * t[2] / 3
    xx, yy = (dd - zz) / 2, (-dd - zz) / 2
    return np.array([[xx, xy / 2, xz / 2], [xy / 2, yy, yz / 2], [xz / 2, yz / 2, zz]])


def transform_spherical_tensor(components: Sequence[complex], *euler_angles: Sequence[Any]) -> np.ndarray:
    """The components in a new frame (``TransformSphericalTensor``): ``A'(m) = Σ A(m') D(m', m)``.

    ``euler_angles`` are the angles ``(α, β, γ)`` that rotate the old frame into the new one. Several sets are
    applied one after the other, the first one first.
    """
    out = np.asarray(components, dtype=complex)
    rank = sp.Rational(len(out) - 1, 2)
    for angles in euler_angles:
        d = np.array(sp.N(wigner_matrix(rank, *(sp.sympify(a) for a in angles)), 30).tolist(), dtype=complex)
        out = d.T @ out
    return out
