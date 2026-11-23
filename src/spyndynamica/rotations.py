"""Wigner matrices, Clebsch-Gordan coefficients, Euler and polar rotations, axis systems and quaternions in
SpinDynamica's conventions.

Indices of the Wigner matrices run from ``+j`` down. ``wigner_D(j, α, β, γ)`` belongs to the rotation
``Rz(α) Ry(β) Rz(γ)`` and has the elements ``exp(−i m α) d(β) exp(−i m' γ)``; a tensor operator transforms as
``R T(k, q) R† = Σ T(k, q') D(q', q)``.

Author: Alexej Jerschow.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import sympy as sp
from sympy.physics.quantum.cg import CG

from spyndynamica._local import reduced_wigner, wigner_matrix


def wigner_d(j: Any, beta: Any) -> sp.Matrix:
    """The reduced Wigner matrix ``d^j(β)`` (SpinDynamica's ``Wignerd[j][β]``)."""
    return sp.Matrix(reduced_wigner(sp.Rational(sp.nsimplify(j, rational=True)), sp.sympify(beta)))


def wigner_D(j: Any, alpha: Any, beta: Any, gamma: Any) -> sp.Matrix:  # noqa: N802 - the name of the matrix
    """The Wigner matrix ``D^j(α, β, γ)`` (SpinDynamica's ``WignerD[j][{α, β, γ}]``)."""
    return wigner_matrix(sp.Rational(sp.nsimplify(j, rational=True)), *(sp.sympify(a) for a in (alpha, beta, gamma)))


def clebsch_gordan(j1: Any, m1: Any, j2: Any, m2: Any, j3: Any, m3: Any) -> sp.Expr:
    """``<j1 m1, j2 m2|j3 m3>`` (SpinDynamica's ``CG[{j3, m3}, {j1, m1}, {j2, m2}]``)."""
    args = [sp.Rational(sp.nsimplify(a, rational=True)) for a in (j1, m1, j2, m2, j3, m3)]
    return CG(*args).doit()


def euler_matrix(alpha: Any, beta: Any, gamma: Any) -> sp.Matrix:
    """The 3 × 3 matrix of ``Rz(α) Ry(β) Rz(γ)`` (SpinDynamica's ``EulerRotationMatrix``)."""

    def rz(t):
        return sp.Matrix([[sp.cos(t), -sp.sin(t), 0], [sp.sin(t), sp.cos(t), 0], [0, 0, 1]])

    def ry(t):
        return sp.Matrix([[sp.cos(t), 0, sp.sin(t)], [0, 1, 0], [-sp.sin(t), 0, sp.cos(t)]])

    return rz(alpha) * ry(beta) * rz(gamma)


def polar_rotation_matrix(xi: Any, theta: Any, phi: Any) -> sp.Matrix:
    """The 3 × 3 matrix of a rotation through ``ξ`` about the axis with the polar angles ``(θ, φ)``
    (SpinDynamica's ``PolarRotationMatrix``)."""
    return euler_matrix(phi, theta, xi) * euler_matrix(0, -sp.sympify(theta), -sp.sympify(phi))


# ====================================================================================== geometry (numeric)
def _euler(alpha: float, beta: float, gamma: float) -> np.ndarray:
    ca, sa, cb, sb, cg, sg = np.cos(alpha), np.sin(alpha), np.cos(beta), np.sin(beta), np.cos(gamma), np.sin(gamma)
    return np.array(
        [
            [ca * cb * cg - sa * sg, -ca * cb * sg - sa * cg, ca * sb],
            [sa * cb * cg + ca * sg, -sa * cb * sg + ca * cg, sa * sb],
            [-sb * cg, sb * sg, cb],
        ]
    )


def _vector(v: Any) -> np.ndarray:
    out = np.array([float(sp.N(x)) for x in v], dtype=float)
    if out.shape != (3,):
        raise ValueError("expected a vector with three components.")
    return out


def _unit(v: Any) -> np.ndarray:
    v = _vector(v)
    return v / np.linalg.norm(v)


def _vectors(obj: Any) -> np.ndarray:
    vectors = np.asarray(obj)
    if vectors.dtype == object:  # exact SymPy numbers
        vectors = np.vectorize(lambda x: float(sp.N(x)), otypes=[float])(vectors)
    if vectors.shape[-1] != 3:
        raise ValueError("expected a vector, or an array of vectors, with three components.")
    return vectors.astype(float)


_EX, _EZ = np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])


def axis_system(*axes: Any) -> np.ndarray:
    """A right-handed orthonormal axis system, with the axes ``x``, ``y``, ``z`` as the rows
    (SpinDynamica's ``AxisSystem``).

    ``axis_system(z)`` has its third axis along ``z`` and its first axis in the plane of the laboratory ``x``
    axis and ``z``; ``axis_system(x, z)`` has its first axis in the plane of ``x`` and ``z``;
    ``axis_system(x, y, z)`` normalises three vectors that already are a right-handed orthogonal set. Without
    arguments it is the laboratory system. A list of one to three vectors is taken as these arguments.
    """
    if len(axes) == 1 and np.ndim(axes[0]) == 2:
        axes = tuple(axes[0])
    if not axes:
        return np.eye(3)
    if len(axes) == 1:
        z = _vector(axes[0])
        if not z.any():
            return np.eye(3)
        z = z / np.linalg.norm(z)
        if np.allclose(z, _EX):
            return axis_system(-_EZ, z)
        if np.allclose(z, -_EX):
            return axis_system(_EZ, z)
        return axis_system(_EX, z)
    if len(axes) == 2:
        x, z = _vector(axes[0]), _vector(axes[1])
        if not x.any() or np.allclose(np.cross(x, z), 0, atol=1e-12 * np.linalg.norm(x) * max(np.linalg.norm(z), 1)):
            return axis_system(z)
        z = z / np.linalg.norm(z)
        y = _unit(np.cross(z, x))
        return np.array([np.cross(y, z), y, z])
    if len(axes) == 3:
        system = np.array([_unit(a) for a in axes])
        if not np.allclose(system @ system.T, np.eye(3), atol=1e-10) or np.linalg.det(system) < 0:
            raise ValueError("the three vectors are not a right-handed orthogonal set.")
        return system
    raise ValueError("an axis system is given by one, two or three vectors.")


def _round_angle(angle: float) -> float:
    """The angle brought into ``[−π, π)``, as SpinDynamica returns Euler angles."""
    return float(np.mod(angle + np.pi, 2 * np.pi) - np.pi)


def _matrix_to_euler(m: np.ndarray) -> tuple[float, float, float]:
    """The Euler angles of the rotation matrix ``m``, with SpinDynamica's choices where they are not unique."""
    if abs(m[2, 2] - 1) < 1e-10:
        angles = (np.arctan2(-m[0, 1], m[0, 0]), 0.0, 0.0)
    elif abs(m[2, 2] + 1) < 1e-10:
        angles = (np.arctan2(-m[0, 1], -m[0, 0]), np.pi, 0.0)
    else:
        angles = (np.arctan2(m[1, 2], m[0, 2]), np.arccos(m[2, 2]), np.arctan2(m[2, 1], -m[2, 0]))
    return tuple(_round_angle(a) for a in angles)  # type: ignore[return-value]


def _as_system(system: Any) -> np.ndarray:
    if system is None:
        return np.eye(3)
    return axis_system(system) if np.ndim(system) == 1 else axis_system(*system)


def axes_to_euler(system2: Any = None, system1: Any = None) -> tuple[float, float, float]:
    """The Euler angles ``(α, β, γ)`` of the rotation that brings every axis of ``system2`` onto the
    corresponding axis of ``system1`` (SpinDynamica's ``AxesToEuler``).

    The rotation is ``Rz(α) Ry(β) Rz(γ)`` about the axes of either of the two systems. Each system is a vector
    (its z axis) or a list of one to three vectors as for :func:`axis_system`; ``system1`` defaults to the
    laboratory system. With the principal axis system of an interaction as ``system2`` and a reference frame
    as ``system1`` these are the angles used in spherical tensor expressions of the Hamiltonian. The angles
    are numbers in ``[−π, π)``; symbolic axes are not supported.
    """
    return _matrix_to_euler(_as_system(system2) @ _as_system(system1).T)


def rotate_euler(obj: Any, angles: Sequence[Any], axes: Any = None) -> np.ndarray:
    """A vector, or an array of vectors, rotated by ``Rz(α) Ry(β) Rz(γ)`` (SpinDynamica's ``RotateEuler``).

    The rotations are about the laboratory axes, or about the axes of ``axes`` (given as for
    :func:`axis_system`): by ``γ`` about its z axis, then by ``β`` about its y axis, then by ``α`` about its z
    axis. With ``angles = axes_to_euler(a, b)`` and ``axes=a`` the system ``a`` is turned onto ``b``.
    """
    rotation = _euler(*(float(sp.N(a)) for a in angles))
    if axes is not None:
        frame = _as_system(axes).T  # the axes as columns
        rotation = frame @ rotation @ frame.T
    return _vectors(obj) @ rotation.T


def vector_to_polar(vector: Any) -> tuple[float, tuple[float, float]]:
    """``(r, (θ, φ))``: the length and the polar angles of a vector (SpinDynamica's ``VectorToPolar``)."""
    v = _vector(vector)
    r = float(np.linalg.norm(v))
    unit = v / r
    phi = 0.0 if abs(abs(unit[2]) - 1) < 1e-10 else float(np.arctan2(unit[1], unit[0]))
    return r, (float(np.arccos(unit[2])), phi)


def polar_to_vector(r: Any, direction: Sequence[Any]) -> np.ndarray:
    """The vector of length ``r`` with the polar angles ``direction = (θ, φ)`` (SpinDynamica's ``PolarToVector``)."""
    theta, phi = (float(sp.N(a)) for a in direction)
    return float(sp.N(r)) * np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])


def _direction(direction: Sequence[Any]) -> tuple[float, float]:
    if len(direction) == 3:
        return vector_to_polar(direction)[1]
    theta, phi = (float(sp.N(a)) for a in direction)
    return theta, phi


def rotate_polar(obj: Any, xi: Any, direction: Sequence[Any]) -> np.ndarray:
    """A vector, or an array of vectors, rotated through ``ξ`` about an axis (SpinDynamica's ``RotatePolar``).

    ``direction`` is the axis: its polar angles ``(θ, φ)`` or a vector with three components.
    """
    theta, phi = _direction(direction)
    xi = float(sp.N(xi))
    rotation = _euler(phi, theta, xi) @ _euler(0.0, -theta, -phi)
    return _vectors(obj) @ rotation.T


def polar_to_quaternion(xi: Any, theta: Any, phi: Any) -> np.ndarray:
    """The unit quaternion ``(cos ξ/2, n sin ξ/2)`` of the rotation through ``ξ`` about the axis ``n`` with the
    polar angles ``(θ, φ)`` (SpinDynamica's ``PolarToQuaternion``)."""
    xi, theta, phi = (float(sp.N(a)) for a in (xi, theta, phi))
    n = np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
    return np.concatenate([[np.cos(xi / 2)], np.sin(xi / 2) * n])


def quaternion_to_polar(quaternion: Sequence[Any]) -> tuple[float, float, float]:
    """``(ξ, θ, φ)``: the angle and the polar angles of the axis of a unit quaternion
    (SpinDynamica's ``QuaternionToPolar``)."""
    q = np.array([float(sp.N(x)) for x in quaternion], dtype=float)
    transverse = float(np.hypot(q[1], q[2]))
    xi = 2 * float(np.arctan2(np.linalg.norm(q[1:]), q[0]))
    theta = float(np.arctan2(transverse, q[3])) if q[1:].any() else 0.0
    phi = float(np.arctan2(q[2], q[1])) if transverse else 0.0
    return xi, theta, phi


def quaternion_to_euler(quaternion: Sequence[Any]) -> tuple[float, float, float]:
    """The Euler angles ``(α, β, γ)`` of the rotation of a unit quaternion (SpinDynamica's ``QuaternionToEuler``)."""
    xi, theta, phi = quaternion_to_polar(quaternion)
    rotation = _euler(phi, theta, xi) @ _euler(0.0, -theta, -phi)
    return _matrix_to_euler(rotation)


def quaternion_product(*quaternions: Sequence[Any]) -> np.ndarray:
    """The quaternion of rotations applied one after the other, the last argument first
    (SpinDynamica's ``QuaternionProduct[q2, q1]``); the result is normalised."""
    if not quaternions:
        raise ValueError("a product needs quaternions.")
    out = np.array([float(sp.N(x)) for x in quaternions[-1]], dtype=float)
    for quaternion in reversed(quaternions[:-1]):
        q = np.array([float(sp.N(x)) for x in quaternion], dtype=float)
        scalar = q[0] * out[0] - q[1:] @ out[1:]
        out = np.concatenate([[scalar], q[0] * out[1:] + out[0] * q[1:] + np.cross(q[1:], out[1:])])
        out = out / np.linalg.norm(out)
    return out
