"""Liouville space: operator bases and superoperators.

A :class:`Superoperator` is a sum of sandwiches ``ρ → L ρ R`` and of dyads ``ρ → K (B|ρ)`` built from symbolic
operators, so it acts on an operator exactly, composes with ``@`` and has a matrix in any orthonormal
:class:`OperatorBasis`: ``S_ij = (B_i | S B_j)`` with the bracket ``(A|B) = Tr(A† B)``.

The operator bases are SpinDynamica's. Their operators are the same; the order is the same where SpinDynamica's
follows from a rule, and is stated in each function where it comes from Mathematica's internal sorting instead.

Author: Alexej Jerschow.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Mapping, Sequence
from functools import cached_property
from itertools import product as cartesian
from typing import Any

import numpy as np
import sympy as sp

from spyndynamica._local import HALF
from spyndynamica.operators import Operator, rotation_spec
from spyndynamica.system import SpinSystem, _describe_gram, rank_histories


# ====================================================================================== operator bases
class OperatorBasis:
    """An orthonormal basis of operator space, a list of operators in a fixed order."""

    def __init__(self, system: SpinSystem, name: str, operators: Sequence[Operator]):
        self.system = system
        self.name = name
        self.operators = tuple(operators)
        if not 0 < len(self.operators) <= system.dimension**2:
            raise ValueError(f"an operator basis of {system!r} has at most {system.dimension**2} operators.")

    def __repr__(self) -> str:
        return f"<{self.name} operator basis of {self.system!r}>"

    def __len__(self) -> int:
        return len(self.operators)

    @property
    def complete(self) -> bool:
        """Whether the basis spans the whole operator space."""
        return len(self.operators) == self.system.dimension**2

    def check(self) -> str:
        """``'orthonormal'``, ``'orthogonal, same norm'``, ``'orthogonal, different norms'`` or
        ``'not orthogonal'`` (SpinDynamica's ``CheckOperatorBasis``)."""
        return _describe_gram(self.vectors.conj().T @ self.vectors)

    def reordered(self, order: Sequence[int], name: str | None = None) -> OperatorBasis:
        """The basis with operator ``order[i]`` of this one in place ``i``; ``order`` counts from 0."""
        order = [int(n) for n in order]
        if sorted(order) != list(range(len(self))):
            raise ValueError(f"the order must hold the integers from 0 to {len(self) - 1}, each of them once.")
        return OperatorBasis(self.system, name or f"reordered {self.name}", [self.operators[n] for n in order])

    def transformed(
        self, superoperator: Superoperator, order: Sequence[int] | None = None, *, check: bool = True, name=None
    ) -> OperatorBasis:
        """The basis of the operators ``superoperator(B)`` of this one, optionally reordered as by
        :meth:`reordered` (SpinDynamica's ``DefineOperatorBasis[new, basis, superoperator, ordering]``).

        With ``check`` (the default) the superoperator must be unitary, so that the new basis is orthonormal.
        """
        if check:
            matrix = superoperator.numeric(basis=self)
            if not np.allclose(matrix.conj().T @ matrix, np.eye(len(self)), rtol=0, atol=1e-10):
                raise ValueError("the superoperator is not unitary (check=False skips the test).")
        source = self if order is None else self.reordered(order)
        operators = [superoperator(b) for b in source.operators]
        return OperatorBasis(self.system, name or f"transformed {self.name}", operators)

    @cached_property
    def vectors(self) -> np.ndarray:
        """The basis operators as the columns of a matrix (each matrix flattened row by row)."""
        return np.array([op.numeric().reshape(-1) for op in self.operators]).T

    def vector(self, operator: Operator) -> sp.Matrix:
        """The components ``(B_k | operator)`` (SpinDynamica's ``OperatorVectorRepresentation``)."""
        return sp.Matrix([b.bracket(operator) for b in self.operators])

    def express(self, components: Iterable[Any]) -> Operator:
        """The operator with these components."""
        return sum((sp.sympify(c) * b for c, b in zip(components, self.operators)), self.system.zero())

    def transformation(self, other: OperatorBasis) -> sp.Matrix:
        """``(self_i | other_j)`` (SpinDynamica's ``OperatorBasisTransformationMatrix[self, other]``)."""
        return sp.Matrix(len(self), len(other), lambda i, j: self.operators[i].bracket(other.operators[j]))


def check_operator_basis(operators: Sequence[Operator]) -> str:
    """Whether operators are fit to be an operator basis (SpinDynamica's ``CheckOperatorBasis``):
    ``'orthonormal'``, ``'orthogonal, same norm'``, ``'orthogonal, different norms'`` or ``'not orthogonal'``."""
    vectors = np.array([op.numeric().reshape(-1) for op in operators]).T
    return _describe_gram(vectors.conj().T @ vectors)


def define_operator_basis(
    operators: Sequence[Operator], *, check: bool = True, name: str = "user-defined"
) -> OperatorBasis:
    """An operator basis with the given operators (SpinDynamica's ``DefineOperatorBasis``).

    With ``check`` (the default) the operators must be orthonormal under ``(A|B) = Tr(A† B)``. Fewer operators
    than the operator space has dimensions are accepted, with a warning: matrices in such a basis are
    projections. To reorder or transform an existing basis use ``basis.reordered`` and ``basis.transformed``.
    """
    operators = list(operators)
    if not operators:
        raise ValueError("an operator basis needs operators.")
    if check and (found := check_operator_basis(operators)) != "orthonormal":
        raise ValueError(f"the operators are {found}; an operator basis is orthonormal (check=False skips the test).")
    basis = OperatorBasis(operators[0].system, name, operators)
    if not basis.complete:
        warnings.warn(
            "the operator basis is incomplete: matrix representations of superoperator products may be unreliable.",
            stacklevel=2,
        )
    return basis


def _half_only(system: SpinSystem, what: str) -> None:
    if any(value != HALF for _, value in system.spins):
        raise ValueError(f"the {what} is defined for spins-1/2.")


def _letters(system: SpinSystem, index: int) -> dict[str, Operator]:
    spin = system.spins[index][0]
    out = {name: system.I(spin, name) for name in ("x", "y", "z", "+", "-")}
    out["e"] = system.unity()
    if system.locals[index].cartesian:
        out["a"], out["b"] = system.I(spin, "alpha"), system.I(spin, "beta")
    return out


def _product(factors: Sequence[Operator]) -> Operator:
    out = factors[0]
    for factor in factors[1:]:
        out = out @ factor
    return out


def _unit(system: SpinSystem, row: int, column: int) -> Operator:
    """``|row><column|`` of the Zeeman basis."""
    matrix = sp.zeros(system.dimension, system.dimension)
    matrix[row, column] = 1
    return system.from_matrix(matrix)


def _order(system: SpinSystem, row: int, column: int) -> sp.Expr:
    return sum(system.quantum_numbers[row]) - sum(system.quantum_numbers[column])


def zeeman_ket_bra_basis(system: SpinSystem, sort: bool = False) -> OperatorBasis:
    """The operators ``|r><s|`` of the Zeeman basis, with ``r`` as the slow index (SpinDynamica's canonical order).

    ``sort=True`` orders them by coherence order, then by the ket and by the bra with ``m`` ascending, as
    SpinDynamica's ``Sorted -> True`` does.
    """
    pairs = list(cartesian(range(system.dimension), repeat=2))
    if sort:
        states = system.quantum_numbers
        pairs.sort(key=lambda p: (_order(system, *p), states[p[0]], states[p[1]]))
    return OperatorBasis(system, "Zeeman ket-bra", [_unit(system, r, c) for r, c in pairs])


def cartesian_product_basis(system: SpinSystem) -> OperatorBasis:
    """The normalised Cartesian product operators of spins-1/2, by the number of spins involved and then by spin
    and component (``1/2, I1x, I1y, I1z, I2x, ..., 2 I1x I2x, 2 I1x I2y, ...`` for two spins)."""
    _half_only(system, "Cartesian product operator basis")
    n = len(system.spins)
    terms = list(cartesian("exyz", repeat=n))
    terms.sort(key=lambda t: (sum(c != "e" for c in t), [(i, c) for i, c in enumerate(t) if c != "e"]))
    operators = []
    for term in terms:
        rank = sum(c != "e" for c in term)
        operators.append(Operator(system, {term: sp.Integer(2) ** rank / sp.sqrt(2) ** n}))
    return OperatorBasis(system, "Cartesian product", operators)


def shift_and_polarization_basis(system: SpinSystem, sort: bool = True) -> OperatorBasis:
    """Products of ``I+``, ``I-``, ``Iα``, ``Iβ`` of spins-1/2.

    Sorted (the default) by coherence order and then spin by spin in the order ``-``, ``+``, ``α``, ``β``;
    ``sort=False`` gives the ket-bra order of the Zeeman basis with the first spin as the slow index.
    """
    _half_only(system, "shift and polarization operator basis")
    n = len(system.spins)
    letters = [_letters(system, i) for i in range(n)]
    symbol = {("a", "a"): "a", ("a", "b"): "+", ("b", "a"): "-", ("b", "b"): "b"}
    states = list(cartesian("ab", repeat=n))
    words = [tuple(symbol[(k, b)] for k, b in zip(ket, bra)) for ket in states for bra in states]
    if sort:
        rank = {"-": 0, "+": 1, "a": 2, "b": 3}
        words.sort(key=lambda w: (w.count("+") - w.count("-"), [rank[c] for c in w]))
    operators = [_product([letters[i][c] for i, c in enumerate(word)]) for word in words]
    return OperatorBasis(system, "shift and polarization", operators)


def _single_spin_shift_and_z(spin: sp.Rational) -> list[tuple[int, sp.Matrix]]:
    """The orthonormal shift-and-z operators of one spin as ``(order, matrix)``: negative orders, zero, positive.

    Powers of ``Iz`` and the symmetrised products ``I+^p Iz^i + Iz^i I+^p`` are orthonormalised in that
    sequence, as SpinDynamica does; the operators of negative order are the transposes in reverse.
    """
    from spyndynamica._local import spin_matrices

    s = spin_matrices(spin)
    dim = int(2 * spin + 1)
    raw: list[tuple[int, sp.Matrix]] = [(0, s["z"] ** i) for i in range(dim)]
    for p in range(1, dim):
        plus = s["+"] ** p
        raw += [(p, plus * s["z"] ** i + s["z"] ** i * plus) for i in range(dim - p)]
    done: list[tuple[int, sp.Matrix]] = []
    for order, matrix in raw:
        for _, other in done:
            matrix = matrix - (other.H * matrix).trace() * other
        matrix = (matrix / sp.sqrt((matrix.H * matrix).trace())).applyfunc(sp.simplify)
        done.append((order, matrix))
    zero = [item for item in done if item[0] == 0]
    positive = [item for item in done if item[0] > 0]
    negative = [(-order, matrix.T) for order, matrix in reversed(positive)]
    return negative + zero + positive


def shift_and_z_basis(system: SpinSystem, sort: bool = True) -> OperatorBasis:
    """Orthonormal products of shift operators and powers of ``Iz``: SpinDynamica's default operator basis.

    For a spin-1/2 the single-spin operators are ``I-``, ``1/√2``, ``√2 Iz``, ``I+``. ``sort=False`` is the plain
    product order with the last spin as the slow index. Sorted (the default) is by coherence order; for
    spins-1/2 the operators of one order then follow SpinDynamica's sequence (products of two spin operators,
    of three, ..., then single-spin operators, then unity; within each by spin and by ``-``, ``+``, ``z``). For
    higher spins the sequence within one coherence order is the product order, which is not SpinDynamica's
    (that one comes from Mathematica's sorting of expressions).
    """
    singles = [_single_spin_shift_and_z(value) for _, value in system.spins]
    items = []
    for position, combo in enumerate(cartesian(*reversed(singles))):
        combo = tuple(reversed(combo))  # one (order, matrix) per spin, the first spin fastest
        operator = _product([system._single(i, matrix) for i, (_, matrix) in enumerate(combo)])
        items.append((sum(order for order, _ in combo), position, combo, operator))
    if sort:
        if all(loc.cartesian for loc in system.locals):

            def key(item):
                factors = []
                for i, (order, matrix) in enumerate(item[2]):
                    if order != 0:
                        factors.append((i, 0 if order < 0 else 1))
                    elif matrix[0, 0] != matrix[1, 1]:
                        factors.append((i, 2))
                kind = (0, len(factors)) if len(factors) > 1 else ((1, 0) if factors else (2, 0))
                return (item[0], kind, factors)

            items.sort(key=key)
        else:
            items.sort(key=lambda item: (item[0], item[1]))
    return OperatorBasis(system, "shift and z", [item[3] for item in items])


def singlet_triplet_operator_basis(system: SpinSystem, sort: bool = True) -> OperatorBasis:
    """The ket-bra products of the singlet-triplet states of two spins-1/2.

    Sorted (the default) by coherence order in SpinDynamica's sequence; ``sort=False`` is the ket-bra order of
    ``S0, T+1, T0, T-1``.
    """
    st = system.singlet_triplet()
    k = {name: st.ket(name) for name in st.names}
    if not sort:
        pairs = [(a, b) for a in st.names for b in st.names]
    else:
        s0, t1, t0, tm = "S0", "T+1", "T0", "T-1"
        pairs = [(t1, tm), (t1, s0), (t1, t0), (s0, tm), (t0, tm), (s0, s0), (t1, t1), (t0, t0), (tm, tm), (s0, t0)]
        pairs += [(t0, s0), (tm, s0), (tm, t0), (s0, t1), (t0, t1), (tm, t1)]
        pairs.reverse()
    return OperatorBasis(system, "singlet-triplet", [k[a] @ k[b].dagger() for a, b in pairs])


def _tensor_multiplets(system: SpinSystem) -> list[tuple[int, tuple, tuple]]:
    """Every irreducible tensor of the spin system as ``(rank, scheme, history)``, in SpinDynamica's sequence:
    the rank of the first spin is the slowest index, and for one set of single-spin ranks the intermediate
    ranks ascend."""
    labels = [label for label, _ in system.spins]
    out = []
    for ranks in cartesian(*(range(int(2 * value) + 1) for _, value in system.spins)):
        scheme = tuple((labels[i], k) for i, k in enumerate(ranks) if k)
        if len(scheme) < 2:
            histories: list[tuple[int, ...]] = [(scheme[0][1],) if scheme else (0,)]
        else:
            histories = rank_histories([k for _, k in scheme])
        out += [(history[-1], scheme, history) for history in histories]
    return out


def _tensor_component(system: SpinSystem, scheme: tuple, history: tuple, mu: int) -> Operator:
    """The normalised component ``mu`` of one irreducible tensor."""
    if not scheme:
        op = system.unity()
    elif len(scheme) == 1:
        op = system.T(scheme[0][0], scheme[0][1], mu)
    else:
        op = system._coupled(scheme, history, mu)
    return (op / op.norm()).map_coefficients(sp.radsimp)


def spherical_tensor_basis(system: SpinSystem, sort: str = "coherence") -> OperatorBasis:
    """The normalised irreducible spherical tensor operators of the spin system.

    Every combination of single-spin ranks is coupled from left to right to every possible total rank (see
    ``SpinSystem.T``). ``sort='coherence'`` (the default, and SpinDynamica's only order) sorts by coherence
    order and then by rank, ``sort='rank'`` by rank and then by order. Operators with the same rank and order
    follow SpinDynamica: by the single-spin ranks with the first spin as the slowest index, then by the
    intermediate ranks. ``basis.indices`` lists ``(rank, order)``.
    """
    if sort not in ("coherence", "rank"):
        raise ValueError("sort is 'coherence' or 'rank'.")
    memo = system.__dict__.setdefault("_tensor_bases", {})
    if sort not in memo:
        specs = [
            (rank, mu, position, scheme, history)
            for position, (rank, scheme, history) in enumerate(_tensor_multiplets(system))
            for mu in range(rank, -rank - 1, -1)
        ]
        first, second = (1, 0) if sort == "coherence" else (0, 1)
        specs.sort(key=lambda spec: (spec[first], spec[second], spec[2]))
        operators = [_tensor_component(system, scheme, history, mu) for _, mu, _, scheme, history in specs]
        basis = OperatorBasis(system, "spherical tensor", operators)
        basis.indices = [(rank, mu) for rank, mu, *_ in specs]  # type: ignore[attr-defined]
        memo[sort] = basis
    return memo[sort]


def default_operator_basis(system: SpinSystem) -> OperatorBasis:
    """SpinDynamica's default: the shift-and-z basis, or the Zeeman ket-bra basis when a spin exceeds 3/2."""
    if max(value for _, value in system.spins) > sp.Rational(3, 2):
        return zeeman_ket_bra_basis(system)
    return shift_and_z_basis(system)


# ====================================================================================== superoperators
class Superoperator:
    """A linear map on operators: a sum of sandwiches ``L ρ R`` and of dyads ``K (B|ρ)``.

    Call it on an operator, add and scale it, and compose it with ``@`` (``(A @ B)(ρ) = A(B(ρ))``).
    """

    __slots__ = ("system", "sandwiches", "dyads")
    __hash__ = None  # type: ignore[assignment]

    def __init__(self, system: SpinSystem, sandwiches: Iterable[tuple[Operator, Operator]] = (), dyads=()):
        self.system = system
        self.sandwiches = [(left, right) for left, right in sandwiches if left and right]
        self.dyads = [(ket, bra) for ket, bra in dyads if ket and bra]

    def __call__(self, rho: Operator) -> Operator:
        out = self.system.zero()
        for left, right in self.sandwiches:
            out = out + left @ rho @ right
        for ket, bra in self.dyads:
            out = out + bra.bracket(rho) * ket
        return out

    def __add__(self, other: Superoperator) -> Superoperator:
        if not isinstance(other, Superoperator):
            return NotImplemented
        return Superoperator(self.system, self.sandwiches + other.sandwiches, self.dyads + other.dyads)

    def __neg__(self) -> Superoperator:
        return self * -1

    def __sub__(self, other: Superoperator) -> Superoperator:
        return self + (-other)

    def __mul__(self, scalar: Any) -> Superoperator:
        return Superoperator(
            self.system,
            [(scalar * left, right) for left, right in self.sandwiches],
            [(scalar * k, b) for k, b in self.dyads],
        )

    __rmul__ = __mul__

    def __truediv__(self, scalar: Any) -> Superoperator:
        return self * (1 / sp.sympify(scalar))

    def __matmul__(self, other: Any):
        if isinstance(other, Operator):
            return self(other)
        if not isinstance(other, Superoperator):
            return NotImplemented
        sandwiches = [(l1 @ l2, r2 @ r1) for l1, r1 in self.sandwiches for l2, r2 in other.sandwiches]
        dyads = [(l1 @ k @ r1, b) for l1, r1 in self.sandwiches for k, b in other.dyads]
        dyads += [(k, left.dagger() @ b @ right.dagger()) for k, b in self.dyads for left, right in other.sandwiches]
        dyads += [(b1.bracket(k2) * k1, b2) for k1, b1 in self.dyads for k2, b2 in other.dyads]
        return Superoperator(self.system, sandwiches, dyads)

    def numeric(self, values: Mapping[Any, Any] | None = None, basis: OperatorBasis | None = None) -> np.ndarray:
        """The matrix as a NumPy array: on row-flattened matrices by default, in ``basis`` when one is given."""
        dim = self.system.dimension
        out = np.zeros((dim * dim, dim * dim), dtype=complex)
        for left, right in self.sandwiches:
            out += np.kron(left.numeric(values), right.numeric(values).T)
        for ket, bra in self.dyads:
            out += np.outer(ket.numeric(values).reshape(-1), bra.numeric(values).conj().reshape(-1))
        if basis is not None:
            out = basis.vectors.conj().T @ out @ basis.vectors
        return out

    def matrix(self, basis: OperatorBasis | None = None) -> sp.Matrix:
        """The exact matrix ``(B_i | S B_j)`` in ``basis`` (default: the default operator basis of the system)."""
        basis = basis or default_operator_basis(self.system)
        images = [self(b) for b in basis.operators]
        return sp.Matrix(len(basis), len(basis), lambda i, j: basis.operators[i].bracket(images[j]))


def unity_superoperator(system: SpinSystem) -> Superoperator:
    return Superoperator(system, [(system.unity(), system.unity())])


def left_multiplication(a: Operator) -> Superoperator:
    """``ρ → a ρ``."""
    return Superoperator(a.system, [(a, a.system.unity())])


def right_multiplication(a: Operator) -> Superoperator:
    """``ρ → ρ a``."""
    return Superoperator(a.system, [(a.system.unity(), a)])


def commutation(a: Operator) -> Superoperator:
    """``ρ → [a, ρ]``."""
    return left_multiplication(a) - right_multiplication(a)


def double_commutation(a: Operator, b: Operator) -> Superoperator:
    """``ρ → [a, [b, ρ]]``."""
    return commutation(a) @ commutation(b)


def lindbladian(a: Operator, b: Operator | None = None) -> Superoperator:
    """``ρ → a ρ b − (b a ρ + ρ b a)/2``, with ``b = a†`` by default."""
    b = a.dagger() if b is None else b
    ba = b @ a
    half = sp.Rational(1, 2)
    return Superoperator(a.system, [(a, b), (-half * ba, a.system.unity()), (-half * a.system.unity(), ba)])


def projection(a: Operator) -> Superoperator:
    """The projector onto the normalised operator: ``ρ → â (â|ρ)``."""
    unit = a / a.norm()
    return Superoperator(a.system, dyads=[(unit, unit)])


def shift_superoperator(to: Operator, source: Operator) -> Superoperator:
    """``|to)(source|`` with both operators normalised: ``ρ → t̂ (ŝ|ρ)`` (SpinDynamica's ``ShiftSuperoperator``)."""
    return Superoperator(to.system, dyads=[(to / to.norm(), source / source.norm())])


def spherical_component_filter(system: SpinSystem, rank: int, to: int, source: int | None = None) -> Superoperator:
    """Takes the component ``source`` of every irreducible tensor of rank ``rank`` to its component ``to`` and
    removes everything else (SpinDynamica's ``SphericalComponentFiltrationSuperoperator[{rank, to, source}]``).

    It is ``Σ |T(rank, to))(T(rank, source)|`` over the tensors of that rank in the spherical tensor operator
    basis, which equals ``(2 rank + 1)/(8π²) ∫ D*(to, source; Ω) R(Ω) dΩ`` over all rotations. ``source``
    defaults to ``to``: the projector onto that component. SpinDynamica has it for spins-1/2 only.
    """
    rank, to = int(rank), int(to)
    source = to if source is None else int(source)
    if rank < 0 or abs(to) > rank or abs(source) > rank:
        raise ValueError(f"components {to} and {source} do not belong to rank {rank}.")
    dyads = [
        (_tensor_component(system, scheme, history, to), _tensor_component(system, scheme, history, source))
        for total, scheme, history in _tensor_multiplets(system)
        if total == rank
    ]
    return Superoperator(system, dyads=dyads)


def coherence_filter(system: SpinSystem, orders: Iterable[int], spins: Any = None) -> Superoperator:
    """Keeps the coherence orders ``orders`` (of all spins, or summed over ``spins``) and removes the others."""
    indices = system._indices(spins)
    values = [sum(state[i] for i in indices) for state in system.quantum_numbers]
    projectors = {}
    for m in sorted(set(values)):
        projectors[m] = system.from_matrix(sp.diag(*[1 if v == m else 0 for v in values]))
    wanted = set(orders)
    pairs = [(projectors[a], projectors[b]) for a in projectors for b in projectors if a - b in wanted]
    return Superoperator(system, pairs)


def rotation_superoperator(
    system: SpinSystem, angle: Any, phase: Any = "x", *, spins: Any = None, axis: Any = None
) -> Superoperator:
    """``ρ → R ρ R†`` for the rotation of ``Operator.rotate``."""
    r = system._rotation(rotation_spec(angle, phase, axis), spins)
    return Superoperator(system, [(r, r.dagger())])


def rotation_euler_superoperator(system: SpinSystem, alpha: Any, beta: Any, gamma: Any, *, spins: Any = None):
    """``ρ → R ρ R†`` for ``R = Rz(α) Ry(β) Rz(γ)``."""
    r = system.rotation_euler(alpha, beta, gamma, spins=spins)
    return Superoperator(system, [(r, r.dagger())])


def permutation_superoperator(system: SpinSystem, cycles: Sequence[Any]) -> Superoperator:
    """``ρ → P ρ P†`` for the spin permutation ``P``."""
    p = system.permutation(cycles)
    return Superoperator(system, [(p, p.dagger())])


def phenomenological_relaxation(system: SpinSystem, t1: Any, t2: Any = None) -> Superoperator:
    """Uncorrelated relaxation of spins-1/2: longitudinal operators of a spin decay at ``1/T1``, transverse ones
    at ``1/T2``, and the rates of a product add.

    ``t1`` and ``t2`` apply to all spins (``t2`` defaults to ``t1``); ``t1`` may also be a mapping
    ``{spin: T}`` or ``{spin: (T1, T2)}``, and spins it leaves out do not relax.
    """
    if isinstance(t1, Mapping):
        times = {spin: (v if isinstance(v, (tuple, list)) else (v, v)) for spin, v in t1.items()}
    else:
        times = {spin: (t1, t1 if t2 is None else t2) for spin, _ in system.spins}
    out = Superoperator(system)
    unity = system.unity()
    for spin, (long, trans) in times.items():
        if not system.locals[system.index(spin)].cartesian:
            raise NotImplementedError("the phenomenological relaxation model is available for spins-1/2.")
        r1, r2 = 1 / sp.sympify(long), 1 / sp.sympify(trans)
        if (r1 - 2 * r2).is_positive:
            raise ValueError("T2 cannot exceed 2 T1.")
        x, y, z = (2 * system.I(spin, c) for c in "xyz")
        # the part of rho along Iz of this spin is (rho - X rho X - Y rho Y + Z rho Z)/4, the transverse part
        # (rho - Z rho Z)/2
        out = out + Superoperator(
            system,
            [
                ((-r1 / 4 - r2 / 2) * unity, unity),
                ((r1 / 4) * x, x),
                ((r1 / 4) * y, y),
                ((-r1 / 4 + r2 / 2) * z, z),
            ],
        )
    return out


def liouvillian(hamiltonian: Operator | None, *dissipators: Superoperator) -> Superoperator:
    """``−i [H, ·]`` plus the dissipators: the generator ``L`` of ``dρ/dt = L ρ``."""
    system = hamiltonian.system if hamiltonian is not None else dissipators[0].system
    out = Superoperator(system) if hamiltonian is None else commutation(hamiltonian) * (-sp.I)
    for dissipator in dissipators:
        out = out + dissipator
    return out
