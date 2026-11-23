"""Single-spin operator algebras: the basis operators of one spin, their exact matrices and their products.

An operator of a spin system is a sum of products of single-spin basis operators, one per spin. For a
spin-1/2 the basis is Cartesian, ``e`` (unity), ``x``, ``y``, ``z``; for a higher spin it is ``e`` and the
irreducible spherical tensor operators ``(k, q)`` with ``1 <= k <= 2I``. Both are orthogonal under the trace,
so products, adjoints and expansions follow from exact matrices by projection, computed once and cached.

The tensor operators follow SpinDynamica's ``opT``: ``T(1, ±1) = ∓I±/√2``, ``T(1, 0) = Iz``,
``T(2, 0) = (3 Iz² − I(I+1))/√6``, ``T(2, ±1) = ∓(I± Iz + Iz I±)/2``, ``T(2, ±2) = I±²/2``, and for ranks above 2
``<m1|T(k, q)|m2> = <k q, I m2|I m1> (−1)^k / sqrt(Σ_m <k 0, I m|I m>²)`` (so that ``T(k, 0)`` has unit norm:
SpinDynamica normalises the ranks above 2 differently from ranks 1 and 2).

Author: Alexej Jerschow.
"""

from __future__ import annotations

from functools import cache
from typing import Any

import numpy as np
import sympy as sp
from sympy.physics.quantum.cg import CG
from sympy.physics.quantum.spin import Rotation

Label = Any  # "e", "x", "y", "z" or (k, q)
Expansion = tuple[tuple[Label, sp.Expr], ...]
HALF = sp.Rational(1, 2)


def tidy(c: Any) -> sp.Expr:
    """A coefficient in expanded form; exactly zero when it vanishes."""
    c = sp.expand(sp.sympify(c))
    if c == 0:
        return sp.S.Zero
    if c.is_number and not c.is_Rational and abs(complex(sp.N(c, 30))) < 1e-25 and sp.simplify(c) == 0:
        return sp.S.Zero
    return c


def quantum_number(value: Any) -> sp.Rational:
    """A spin quantum number as an exact rational (``0.5``, ``"3/2"`` and ``Rational(3, 2)`` are all accepted)."""
    out = sp.nsimplify(value, rational=True) if not isinstance(value, str) else sp.Rational(value)
    if not (out.is_Rational and out > 0 and (2 * out).is_Integer):
        raise ValueError(f"{value!r} is not a spin quantum number.")
    return sp.Rational(out)


def spin_matrices(spin: sp.Rational) -> dict[str, sp.Matrix]:
    """``Ix``, ``Iy``, ``Iz``, ``I+``, ``I-`` and unity of one spin, with ``m`` running from ``+I`` down."""
    dim = int(2 * spin + 1)
    m = [spin - i for i in range(dim)]
    plus = sp.zeros(dim, dim)
    for i in range(dim - 1):
        plus[i, i + 1] = sp.sqrt(spin * (spin + 1) - m[i + 1] * (m[i + 1] + 1))
    minus = plus.T
    return {
        "e": sp.eye(dim),
        "z": sp.diag(*m),
        "+": plus,
        "-": minus,
        "x": (plus + minus) / 2,
        "y": (plus - minus) / (2 * sp.I),
    }


def tensor_matrix(spin: sp.Rational, k: int, q: int) -> sp.Matrix:
    """The matrix of the tensor operator ``T(k, q)`` of one spin (SpinDynamica's ``opT``)."""
    s = spin_matrices(spin)
    dim = int(2 * spin + 1)
    if k == 0:
        return s["e"]
    if k == 1:
        return {1: -s["+"] / sp.sqrt(2), 0: s["z"], -1: s["-"] / sp.sqrt(2)}[q]
    if k == 2:
        return {
            2: s["+"] * s["+"] / 2,
            1: -(s["+"] * s["z"] + s["z"] * s["+"]) / 2,
            0: (3 * s["z"] * s["z"] - spin * (spin + 1) * s["e"]) / sp.sqrt(6),
            -1: (s["-"] * s["z"] + s["z"] * s["-"]) / 2,
            -2: s["-"] * s["-"] / 2,
        }[q]
    m = [spin - i for i in range(dim)]
    reduced = (-1) ** k / sp.sqrt(sum(CG(k, 0, spin, mm, spin, mm).doit() ** 2 for mm in m))
    out = sp.zeros(dim, dim)
    for i, m1 in enumerate(m):
        for j, m2 in enumerate(m):
            if m1 - m2 == q:
                out[i, j] = sp.simplify(CG(k, q, spin, m2, spin, m1).doit() * reduced)
    return out


@cache
def reduced_wigner(j: sp.Rational, beta: sp.Expr) -> sp.ImmutableMatrix:
    """The reduced Wigner matrix ``d^j(β)``, rows and columns from ``+j`` down."""
    j = sp.Rational(j)
    ms = [j - i for i in range(int(2 * j + 1))]
    b = sp.Dummy("b", real=True)
    rows = [[Rotation.d(j, m1, m2, b).doit() for m2 in ms] for m1 in ms]
    return sp.ImmutableMatrix(rows).subs(b, beta)


def wigner_matrix(j: sp.Rational, alpha: Any, beta: Any, gamma: Any) -> sp.Matrix:
    """``D^j(α, β, γ)`` of the rotation ``Rz(α) Ry(β) Rz(γ)``: elements ``exp(−i m α) d(β) exp(−i m' γ)``."""
    j = sp.Rational(j)
    ms = [j - i for i in range(int(2 * j + 1))]
    d = reduced_wigner(j, sp.sympify(beta))
    left = sp.diag(*[sp.exp(-sp.I * m * alpha) for m in ms])
    right = sp.diag(*[sp.exp(-sp.I * m * gamma) for m in ms])
    return left * d * right


def rotation_matrix(j: sp.Rational, rotation: tuple) -> sp.Matrix:
    """The Wigner matrix of a rotation given as ``("euler", α, β, γ)`` or ``("axis", angle, θ, φ)``.

    The axis has polar angles ``θ``, ``φ``; the rotation about it is ``Rz(φ) Ry(θ) Rz(angle) Ry(−θ) Rz(−φ)``.
    """
    kind, *p = rotation
    if kind == "euler":
        return wigner_matrix(j, *p)
    angle, theta, phi = p
    if theta == 0:
        return wigner_matrix(j, angle, 0, 0)
    if theta == sp.pi / 2:
        # an axis in the xy plane: Rz(φ − π/2) Ry(angle) Rz(π/2 − φ)
        return wigner_matrix(j, phi - sp.pi / 2, angle, sp.pi / 2 - phi)
    out = wigner_matrix(j, phi, theta, angle) * wigner_matrix(j, 0, -theta, -phi)
    return out.applyfunc(lambda c: sp.simplify(c.rewrite(sp.cos)))


class Local:
    """The operator basis of one spin: labels, exact matrices, products, adjoints and expansions."""

    def __init__(self, spin: sp.Rational):
        self.spin = spin
        self.dim = int(2 * spin + 1)
        self.cartesian = spin == HALF
        if self.cartesian:
            s = spin_matrices(spin)
            self.matrices = {label: sp.ImmutableMatrix(s[label]) for label in ("e", "x", "y", "z")}
        else:
            self.matrices = {"e": sp.ImmutableMatrix(sp.eye(self.dim))}
            for k in range(1, self.dim):
                for q in range(k, -k - 1, -1):
                    self.matrices[(k, q)] = sp.ImmutableMatrix(tensor_matrix(spin, k, q))
        self.labels = tuple(self.matrices)
        self.spin_matrices = {name: sp.ImmutableMatrix(mat) for name, mat in spin_matrices(spin).items()}
        self.norm2 = {label: tidy((mat.H * mat).trace()) for label, mat in self.matrices.items()}
        self._adjoints = {label: mat.H for label, mat in self.matrices.items()}
        self._products: dict[tuple[Label, Label], Expansion] = {}
        self._numeric: dict[Label, np.ndarray] = {}

    def expand(self, matrix: sp.Matrix) -> Expansion:
        """A matrix of this spin as a sum of basis operators."""
        out = []
        for label in self.labels:
            c = tidy((self._adjoints[label] * matrix).trace() / self.norm2[label])
            if c != 0:
                out.append((label, c))
        return tuple(out)

    def product(self, a: Label, b: Label) -> Expansion:
        """The product of two basis operators as a sum of basis operators."""
        if a == "e":
            return ((b, sp.S.One),)
        if b == "e":
            return ((a, sp.S.One),)
        key = (a, b)
        if key not in self._products:
            self._products[key] = self.expand(self.matrices[a] * self.matrices[b])
        return self._products[key]

    @cache  # noqa: B019 - one instance per spin quantum number, kept for the session
    def dagger(self, label: Label) -> Expansion:
        if label == "e" or self.cartesian:
            return ((label, sp.S.One),)
        k, q = label
        return (((k, -q), sp.Integer((-1) ** q)),)

    def order(self, label: Label) -> int | None:
        """The coherence order of a basis operator; ``None`` for ``x`` and ``y``, which mix +1 and −1."""
        if label in ("e", "z"):
            return 0
        if self.cartesian:
            return None
        return int(label[1])

    def numeric(self, label: Label) -> np.ndarray:
        if label not in self._numeric:
            self._numeric[label] = np.array(self.matrices[label].evalf(30).tolist(), dtype=complex)
        return self._numeric[label]

    @cache  # noqa: B019
    def rotation_images(self, rotation: tuple) -> dict[Label, Expansion]:
        """``R B R†`` for every basis operator ``B``, for a rotation as in :func:`rotation_matrix`."""
        images: dict[Label, Expansion] = {"e": (("e", sp.S.One),)}
        if self.cartesian:
            r = cartesian_rotation(rotation)
            for col, label in enumerate("xyz"):
                images[label] = tuple(
                    (new, tidy(r[row, col])) for row, new in enumerate("xyz") if tidy(r[row, col]) != 0
                )
            return images
        for k in range(1, self.dim):
            big_d = rotation_matrix(k, rotation)
            for col, q in enumerate(range(k, -k - 1, -1)):
                terms = []
                for row, q_new in enumerate(range(k, -k - 1, -1)):
                    c = tidy(big_d[row, col])
                    if c != 0:
                        terms.append(((k, q_new), c))
                images[(k, q)] = tuple(terms)
        return images

    def rotation_operator(self, rotation: tuple) -> Expansion:
        """The rotation operator ``R`` itself as a sum of basis operators."""
        matrix = rotation_matrix(self.spin, rotation)
        if self.cartesian:
            matrix = matrix.applyfunc(lambda c: sp.simplify(c.rewrite(sp.cos)))
        return self.expand(matrix)


def cartesian_rotation(rotation: tuple) -> sp.Matrix:
    """The 3 × 3 matrix that rotates the vector ``(Ix, Iy, Iz)``: column ``a`` is the image of component ``a``."""

    def rz(t):
        return sp.Matrix([[sp.cos(t), -sp.sin(t), 0], [sp.sin(t), sp.cos(t), 0], [0, 0, 1]])

    def ry(t):
        return sp.Matrix([[sp.cos(t), 0, sp.sin(t)], [0, 1, 0], [-sp.sin(t), 0, sp.cos(t)]])

    kind, *p = rotation
    if kind == "euler":
        alpha, beta, gamma = p
        return rz(alpha) * ry(beta) * rz(gamma)
    angle, theta, phi = p
    n = sp.Matrix([sp.sin(theta) * sp.cos(phi), sp.sin(theta) * sp.sin(phi), sp.cos(theta)])
    cross = sp.Matrix([[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]])
    out = sp.cos(angle) * sp.eye(3) + sp.sin(angle) * cross + (1 - sp.cos(angle)) * (n * n.T)
    return out.applyfunc(sp.trigsimp) if theta not in (0, sp.pi / 2) else out


@cache
def local(spin: sp.Rational) -> Local:
    """The operator basis of a spin of quantum number ``spin`` (one shared instance per quantum number)."""
    return Local(spin)
