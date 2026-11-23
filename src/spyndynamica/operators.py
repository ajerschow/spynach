"""Symbolic spin operators: sums of products of single-spin basis operators with symbolic coefficients.

An :class:`Operator` is a map from terms to coefficients. A term is a tuple with one single-spin basis label
per spin (see :mod:`spyndynamica._local`), so products, commutators, adjoints, traces and rotations are exact
algebra on the labels, and SymPy only handles the scalar coefficients. Matrices are derived from the terms
(:meth:`Operator.matrix`, :meth:`Operator.numeric`), never the other way round in the algebra.

Author: Alexej Jerschow.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from itertools import product as cartesian
from typing import TYPE_CHECKING, Any

import numpy as np
import scipy.sparse
import sympy as sp

from spyndynamica._local import Expansion, Label, tidy

if TYPE_CHECKING:
    from spyndynamica.system import Basis, SpinSystem

Term = tuple[Label, ...]


def _add(terms: dict[Term, sp.Expr], key: Term, coefficient: sp.Expr) -> None:
    value = terms.get(key, sp.S.Zero) + coefficient
    terms[key] = value


def _cleaned(terms: Mapping[Term, Any]) -> dict[Term, sp.Expr]:
    out = {}
    for key, value in terms.items():
        value = tidy(value)
        if value != 0:
            out[key] = value
    return out


def rotation_spec(angle: Any, phase: Any = "x", axis: Any = None) -> tuple:
    """A rotation by ``angle`` about an axis in the xy plane at ``phase`` (``'x'``, ``'y'``, ``'-x'``, ``'-y'``
    or an angle), about ``'z'``, or about the axis with polar angles ``axis = (θ, φ)``."""
    angle = sp.sympify(angle)
    if axis is not None:
        theta, phi = (sp.sympify(a) for a in axis)
        return ("axis", angle, theta, phi)
    if isinstance(phase, str):
        named = {"x": 0, "y": sp.pi / 2, "-x": sp.pi, "-y": 3 * sp.pi / 2}
        if phase == "z":
            return ("axis", angle, sp.S.Zero, sp.S.Zero)
        if phase == "-z":
            return ("axis", -angle, sp.S.Zero, sp.S.Zero)
        if phase not in named:
            raise ValueError(f"unknown phase {phase!r}; use 'x', 'y', '-x', '-y', 'z', '-z' or an angle.")
        return ("axis", angle, sp.pi / 2, sp.sympify(named[phase]))
    return ("axis", angle, sp.pi / 2, sp.sympify(phase))


class Operator:
    """An operator of a :class:`~spyndynamica.SpinSystem`.

    Build operators from the spin system (``s.I(1, "x")``, ``s.T(1, 2, 0)``, ``s.unity()``) and combine them with
    ``+``, ``-``, ``*`` (a scalar) and ``@`` (the operator product). Coefficients are SymPy expressions.
    """

    __slots__ = ("system", "terms")
    __hash__ = None  # type: ignore[assignment]

    def __init__(self, system: SpinSystem, terms: Mapping[Term, Any] | None = None):
        self.system = system
        self.terms: dict[Term, sp.Expr] = _cleaned(terms or {})

    # ------------------------------------------------------------------ linear structure
    def _same(self, other: Operator) -> None:
        if other.system is not self.system:
            raise ValueError("the operators belong to different spin systems.")

    def __add__(self, other: Any) -> Operator:
        if not isinstance(other, Operator):
            if other == 0:
                return self
            other = sp.sympify(other) * self.system.unity()
        self._same(other)
        terms = dict(self.terms)
        for key, value in other.terms.items():
            _add(terms, key, value)
        return Operator(self.system, terms)

    __radd__ = __add__

    def __neg__(self) -> Operator:
        return Operator(self.system, {k: -v for k, v in self.terms.items()})

    def __sub__(self, other: Any) -> Operator:
        return self + (-other if isinstance(other, Operator) else -sp.sympify(other))

    def __rsub__(self, other: Any) -> Operator:
        return (-self) + other

    def __mul__(self, scalar: Any) -> Operator:
        if isinstance(scalar, Operator):
            raise TypeError("use @ for the product of two operators; * multiplies by a scalar.")
        scalar = sp.sympify(scalar)
        return Operator(self.system, {k: scalar * v for k, v in self.terms.items()})

    __rmul__ = __mul__

    def __truediv__(self, scalar: Any) -> Operator:
        return self * (1 / sp.sympify(scalar))

    def __bool__(self) -> bool:
        return bool(self.terms)

    # ------------------------------------------------------------------ products
    def __matmul__(self, other: Any):
        from spyndynamica.system import Ket

        if isinstance(other, Ket):
            return Ket(self.system, self.matrix() * other.vector)
        if not isinstance(other, Operator):
            return NotImplemented
        self._same(other)
        locals_ = self.system.locals
        terms: dict[Term, sp.Expr] = {}
        for ka, ca in self.terms.items():
            for kb, cb in other.terms.items():
                factors = [loc.product(a, b) for loc, a, b in zip(locals_, ka, kb)]
                if any(not f for f in factors):
                    continue
                if all(len(f) == 1 for f in factors):
                    c = ca * cb
                    for f in factors:
                        c = c * f[0][1]
                    _add(terms, tuple(f[0][0] for f in factors), c)
                    continue
                for combo in cartesian(*factors):
                    c = ca * cb
                    for _, factor in combo:
                        c = c * factor
                    _add(terms, tuple(label for label, _ in combo), c)
        return Operator(self.system, terms)

    def commutator(self, other: Operator) -> Operator:
        """``[self, other]``."""
        return self @ other - other @ self

    def commutes_with(self, other: Operator) -> bool:
        return not self.commutator(other).simplify()

    def dagger(self) -> Operator:
        """The adjoint. Symbols are conjugated unless they were declared real."""
        locals_ = self.system.locals
        terms: dict[Term, sp.Expr] = {}
        for key, value in self.terms.items():
            c = sp.conjugate(value)
            new = []
            for loc, label in zip(locals_, key):
                ((image, factor),) = loc.dagger(label)
                new.append(image)
                c = c * factor
            _add(terms, tuple(new), c)
        return Operator(self.system, terms)

    # ------------------------------------------------------------------ scalars
    def trace(self) -> sp.Expr:
        return tidy(self.terms.get(self.system.identity_term, sp.S.Zero) * self.system.dimension)

    def bracket(self, other: Operator) -> sp.Expr:
        """The Liouville bracket ``(self|other) = Tr(self† other)``."""
        self._same(other)
        total = sp.S.Zero
        for key, value in self.terms.items():
            if key in other.terms:
                total += sp.conjugate(value) * other.terms[key] * self.system.term_norm2(key)
        return tidy(total)

    def norm(self) -> sp.Expr:
        """``sqrt(Tr(A† A))``."""
        return sp.sqrt(self.bracket(self))

    def normalized(self, norm: Any = 1) -> Operator:
        """The operator scaled to the norm ``norm``, a number or another operator whose norm is taken."""
        wanted = norm.norm() if isinstance(norm, Operator) else sp.sympify(norm)
        return self * (wanted / self.norm())

    def projection(self, onto: Operator) -> Operator:
        """The part of the operator along ``onto``: ``onto (onto|self)/(onto|onto)``."""
        return onto * (onto.bracket(self) / onto.bracket(onto))

    def amplitude(self, target: Operator) -> sp.Expr:
        """How much of ``target`` this operator contains: ``(target|self) / (target|target)``."""
        return sp.simplify(target.bracket(self) / target.bracket(target))

    def coherence_components(self) -> dict[int, Operator]:
        """The parts of the operator by total coherence order."""
        return self._order_components(range(len(self.system.spins)))

    def coherence_orders(self, spins: Iterable[Any] | None = None) -> list[int]:
        """The coherence orders the operator contains, of all spins or summed over ``spins`` only."""
        indices = range(len(self.system.spins)) if spins is None else [self.system.index(s) for s in spins]
        return sorted(self._order_components(indices))

    def _order_components(self, indices: Iterable[int]) -> dict[int, Operator]:
        indices = list(indices)
        locals_ = self.system.locals
        half = sp.Rational(1, 2)
        split = {
            "x": ((1, (("x", half), ("y", sp.I / 2))), (-1, (("x", half), ("y", -sp.I / 2)))),
            "y": ((1, (("x", -sp.I / 2), ("y", half))), (-1, (("x", sp.I / 2), ("y", half)))),
        }
        parts: dict[int, dict[Term, sp.Expr]] = {}
        for key, value in self.terms.items():
            options = []
            for i, label in enumerate(key):
                order = locals_[i].order(label)
                if i not in indices:
                    options.append(((0, ((label, sp.S.One),)),))
                elif order is None:
                    options.append(split[label])
                else:
                    options.append(((order, ((label, sp.S.One),)),))
            for choice in cartesian(*options):
                total = sum(order for order, _ in choice)
                for combo in cartesian(*(expansion for _, expansion in choice)):
                    c = value
                    for _, factor in combo:
                        c = c * factor
                    _add(parts.setdefault(total, {}), tuple(label for label, _ in combo), c)
        out = {order: Operator(self.system, terms) for order, terms in parts.items()}
        return {order: op for order, op in out.items() if op.simplify()}

    def spin_product_rank(self) -> int | list[int]:
        """The number of spins a term involves; a list when the terms differ."""
        ranks = sorted({sum(label != "e" for label in key) for key in self.terms})
        return ranks[0] if len(ranks) == 1 else ranks

    # ------------------------------------------------------------------ coefficients
    def map_coefficients(self, function: Callable[[sp.Expr], Any]) -> Operator:
        return Operator(self.system, {k: function(v) for k, v in self.terms.items()})

    def simplify(self) -> Operator:
        """The operator with every coefficient simplified (terms that vanish are dropped)."""
        return self.map_coefficients(lambda c: sp.simplify(sp.trigsimp(c)))

    def subs(self, *args: Any, **kwargs: Any) -> Operator:
        return self.map_coefficients(lambda c: c.subs(*args, **kwargs))

    def equals(self, other: Operator) -> bool:
        """Whether the two operators are the same after simplification."""
        return not (self - other).simplify()

    # ------------------------------------------------------------------ rotations and evolution
    def _map_spins(self, images: Mapping[int, Mapping[Label, Expansion]]) -> Operator:
        terms: dict[Term, sp.Expr] = {}
        for key, value in self.terms.items():
            factors = [images[i][label] if i in images else ((label, sp.S.One),) for i, label in enumerate(key)]
            for combo in cartesian(*factors):
                c = value
                for _, factor in combo:
                    c = c * factor
                _add(terms, tuple(label for label, _ in combo), c)
        return Operator(self.system, terms)

    def _rotated(self, spec: tuple, spins: Iterable[Any] | None) -> Operator:
        indices = range(len(self.system.spins)) if spins is None else [self.system.index(s) for s in spins]
        images = {i: self.system.locals[i].rotation_images(spec) for i in indices}
        return self._map_spins(images)

    def rotate(self, angle: Any, phase: Any = "x", *, spins: Iterable[Any] | None = None, axis=None) -> Operator:
        """``R A R†`` for the rotation ``R = exp(−i angle n·I)`` of ``spins`` (default: all spins).

        The axis ``n`` lies in the xy plane at ``phase`` (``'x'``, ``'y'``, ``'-x'``, ``'-y'`` or an angle), is
        ``'z'``, or has the polar angles ``axis = (θ, φ)``.
        """
        return self._rotated(rotation_spec(angle, phase, axis), spins)

    def rotate_euler(self, alpha: Any, beta: Any, gamma: Any, *, spins: Iterable[Any] | None = None) -> Operator:
        """``R A R†`` for ``R = Rz(α) Ry(β) Rz(γ)`` (zyz Euler angles) acting on ``spins``."""
        spec = ("euler", sp.sympify(alpha), sp.sympify(beta), sp.sympify(gamma))
        return self._rotated(spec, spins)

    def evolve(self, hamiltonian: Operator, time: Any) -> Operator:
        """``exp(−i H t) A exp(+i H t)`` in closed form.

        Available when the spins that ``H`` involves are spins-1/2 and the terms of ``H`` commute with one
        another (chemical shifts and weak couplings). Other Hamiltonians need the numeric engine.
        """
        self._same(hamiltonian)
        system = self.system
        pieces = [Operator(system, {key: value}) for key, value in hamiltonian.terms.items()]
        for key in hamiltonian.terms:
            if any(label != "e" and not loc.cartesian for loc, label in zip(system.locals, key)):
                raise NotImplementedError("closed-form evolution needs spins-1/2 in the Hamiltonian.")
        for i, a in enumerate(pieces):
            for b in pieces[i + 1 :]:
                if a.commutator(b):
                    raise NotImplementedError(
                        "closed-form evolution needs a Hamiltonian whose terms commute (weak coupling); "
                        "use the numeric engine for strong coupling."
                    )
        time = sp.sympify(time)
        rho = self
        for key, value in hamiltonian.terms.items():
            rank = sum(label != "e" for label in key)
            if rank == 0:
                continue
            # G = 2^(rank-1) x (product operator) has G^2 = 1/4, and every product operator either commutes
            # or anticommutes with it: exp(-i a G) A exp(i a G) = A cos a - i [G, A] sin a in the second case
            generator = Operator(system, {key: sp.Integer(2) ** (rank - 1)})
            angle = value * time / sp.Integer(2) ** (rank - 1)
            out = Operator(system)
            for term, coefficient in rho.terms.items():
                piece = Operator(system, {term: coefficient})
                turned = generator.commutator(piece)
                out = out + (piece if not turned else sp.cos(angle) * piece - sp.I * sp.sin(angle) * turned)
            rho = out
        return rho

    # ------------------------------------------------------------------ matrices
    def matrix(self, basis: Basis | None = None) -> sp.Matrix:
        """The exact matrix in the Zeeman basis, or in ``basis``."""
        system = self.system
        out = sp.zeros(system.dimension, system.dimension)
        for key, value in self.terms.items():
            factors = [loc.matrices[label] for loc, label in zip(system.locals, key)]
            out += value * sp.kronecker_product(*reversed(factors))  # the first spin changes fastest
        if basis is not None:
            out = basis.kets_matrix.H * out * basis.kets_matrix
        return out.applyfunc(tidy)

    def sparse(self, values: Mapping[Any, Any] | None = None) -> scipy.sparse.csr_matrix:
        """The matrix in the Zeeman basis as a SciPy sparse matrix, with ``values`` substituted for the symbols."""
        system = self.system
        out = scipy.sparse.csr_matrix((system.dimension, system.dimension), dtype=complex)
        for key, value in self.terms.items():
            c = complex(sp.N(value.subs(values) if values else value, 30))
            block = scipy.sparse.identity(1, dtype=complex, format="csr")
            for loc, label in zip(system.locals, key):
                block = scipy.sparse.kron(loc.sparse(label), block, format="csr")  # the first spin changes fastest
            out = out + c * block
        return out

    def numeric(self, values: Mapping[Any, Any] | None = None, basis: Basis | None = None) -> np.ndarray:
        """The matrix as a complex NumPy array, with ``values`` substituted for the symbols."""
        out = self.sparse(values).toarray()
        if basis is not None:
            v = np.array(basis.kets_matrix.evalf(30).tolist(), dtype=complex)
            out = v.conj().T @ out @ v
        return out

    # ------------------------------------------------------------------ display
    def _sorted_terms(self) -> list[tuple[Term, sp.Expr]]:
        def rank(item):
            key = item[0]
            return (sum(label != "e" for label in key), [str(label) for label in key])

        return sorted(self.terms.items(), key=rank)

    def __str__(self) -> str:
        if not self.terms:
            return "0"
        parts = []
        for key, value in self._sorted_terms():
            name = self.system.term_name(key)
            if name == "1":
                parts.append(f"({value})" if value.is_Add else str(value))
            elif value == 1:
                parts.append(name)
            elif value == -1:
                parts.append(f"-{name}")
            else:
                parts.append(f"({value})*{name}" if value.is_Add else f"{value}*{name}")
        return " + ".join(parts).replace("+ -", "- ")

    def __repr__(self) -> str:
        return f"<Operator {self}>"

    def _repr_latex_(self) -> str:
        if not self.terms:
            return "$0$"
        parts = []
        for key, value in self._sorted_terms():
            name = self.system.term_name(key, latex=True)
            coefficient = sp.latex(value)
            if name == "1":
                parts.append(coefficient)
            elif value == 1:
                parts.append(name)
            elif value == -1:
                parts.append(f"-{name}")
            else:
                parts.append((rf"\left({coefficient}\right)" if value.is_Add else coefficient) + r"\," + name)
        return "$" + " + ".join(parts).replace("+ -", "- ") + "$"


def commutator(a: Operator, b: Operator) -> Operator:
    """``[a, b] = a @ b − b @ a``."""
    return a.commutator(b)
