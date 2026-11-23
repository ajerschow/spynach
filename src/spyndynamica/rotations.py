"""Wigner matrices, Clebsch-Gordan coefficients and Euler rotation matrices in SpinDynamica's conventions.

Indices of the Wigner matrices run from ``+j`` down. ``wigner_D(j, α, β, γ)`` belongs to the rotation
``Rz(α) Ry(β) Rz(γ)`` and has the elements ``exp(−i m α) d(β) exp(−i m' γ)``; a tensor operator transforms as
``R T(k, q) R† = Σ T(k, q') D(q', q)``.

Author: Alexej Jerschow.
"""

from __future__ import annotations

from typing import Any

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
