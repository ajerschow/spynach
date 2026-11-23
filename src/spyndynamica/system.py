"""Spin systems, Hilbert-space bases, kets and bras.

A :class:`SpinSystem` is an explicit object (SpinDynamica keeps one global spin system; here several may
coexist). It builds the operators (``s.I``, ``s.T``, ``s.rotation``), the states (``s.ket``) and the bases
(``s.zeeman``, ``s.singlet_triplet()``). In the Zeeman basis every spin runs from ``m = +I`` down and the first
spin changes fastest, as in SpinDynamica: for two spins-1/2 the order is αα, βα, αβ, ββ.

Author: Alexej Jerschow.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from functools import cached_property
from itertools import product as cartesian
from typing import Any

import sympy as sp
from sympy.physics.quantum.cg import CG

from spyndynamica._local import HALF, Local, local, quantum_number, tensor_matrix, tidy
from spyndynamica.operators import Operator, Term, rotation_spec

_COMPONENTS = {"x": "x", "y": "y", "z": "z", "+": "+", "-": "-"}
_POLARIZATION = {"alpha": 1, "α": 1, "beta": -1, "β": -1}


class SpinSystem:
    """A set of spins with labels and spin quantum numbers.

    ``SpinSystem(2)`` is two spins-1/2 labelled 1 and 2; ``SpinSystem([1/2, 1])`` gives the quantum numbers of
    spins 1, 2, ...; ``SpinSystem({"H": 1/2, "D": 1})`` and ``SpinSystem([("H", 1/2), ("D", 1)])`` also give
    the labels.
    """

    def __init__(self, spins: int | Sequence[Any] | Mapping[Any, Any]):
        if isinstance(spins, int) and not isinstance(spins, bool):
            pairs = [(i + 1, HALF) for i in range(spins)]
        elif isinstance(spins, Mapping):
            pairs = [(label, quantum_number(value)) for label, value in spins.items()]
        elif all(isinstance(item, (tuple, list)) for item in spins):
            pairs = [(label, quantum_number(value)) for label, value in spins]
        else:
            pairs = [(i + 1, quantum_number(value)) for i, value in enumerate(spins)]
        if not pairs:
            raise ValueError("a spin system needs at least one spin.")
        if len({label for label, _ in pairs}) != len(pairs):
            raise ValueError("the spin labels must be distinct.")
        self.spins: tuple[tuple[Any, sp.Rational], ...] = tuple(pairs)
        self.locals: tuple[Local, ...] = tuple(local(value) for _, value in pairs)
        self.dimension = int(sp.prod([loc.dim for loc in self.locals]))
        self.identity_term: Term = tuple("e" for _ in pairs)
        self._index = {label: i for i, (label, _) in enumerate(pairs)}

    def __repr__(self) -> str:
        return "SpinSystem([" + ", ".join(f"({label!r}, {value})" for label, value in self.spins) + "])"

    # ------------------------------------------------------------------ bookkeeping
    def index(self, label: Any) -> int:
        """The position of the spin with this label."""
        try:
            return self._index[label]
        except KeyError:
            raise ValueError(f"no spin {label!r} in {self!r}.") from None

    def _indices(self, spins: Any) -> list[int]:
        if spins is None:
            return list(range(len(self.spins)))
        if isinstance(spins, (list, tuple, set, frozenset)):
            return [self.index(s) for s in spins]
        return [self.index(spins)]

    def term_norm2(self, key: Term) -> sp.Expr:
        out = sp.S.One
        for loc, label in zip(self.locals, key):
            out = out * loc.norm2[label]
        return out

    def term_name(self, key: Term, latex: bool = False) -> str:
        names = []
        for (spin, _), label in zip(self.spins, key):
            if label == "e":
                continue
            if isinstance(label, str):
                names.append(f"I_{{{spin}{label}}}" if latex else f"I{spin}{label}")
            else:
                k, q = label
                names.append(f"T^{{({spin})}}_{{{k},{q}}}" if latex else f"T{spin}({k},{q})")
        return (r"\," if latex else " ").join(names) or "1"

    @cached_property
    def quantum_numbers(self) -> tuple[tuple[sp.Rational, ...], ...]:
        """``(m1, m2, ...)`` of every Zeeman state, in the order of the basis."""
        ranges = [[value - i for i in range(loc.dim)] for (_, value), loc in zip(self.spins, self.locals)]
        return tuple(tuple(reversed(combo)) for combo in cartesian(*reversed(ranges)))

    # ------------------------------------------------------------------ operators
    def zero(self) -> Operator:
        return Operator(self)

    def unity(self) -> Operator:
        return Operator(self, {self.identity_term: sp.S.One})

    def _single(self, index: int, matrix: sp.Matrix) -> Operator:
        terms = {}
        for label, value in self.locals[index].expand(matrix):
            key = list(self.identity_term)
            key[index] = label
            terms[tuple(key)] = value
        return Operator(self, terms)

    def I(self, spin: Any = None, component: Any = None):  # noqa: E743, N802 - the name of the operator
        """The angular momentum operator of ``spin``, of a list of spins (their sum) or of all spins.

        ``component`` is ``'x'``, ``'y'``, ``'z'``, ``'+'``, ``'-'``; ``'alpha'`` or ``'beta'`` (the polarization
        operators of a spin-1/2); an angle ``φ`` (the component in the xy plane at that phase); or polar angles
        ``(θ, φ)``. Without a component the three Cartesian components are returned.
        """
        if component is None:
            return tuple(self.I(spin, c) for c in "xyz")
        total = self.zero()
        for index in self._indices(spin):
            total = total + self._component(index, component)
        return total

    def _component(self, index: int, component: Any) -> Operator:
        matrices = self.locals[index].spin_matrices
        if isinstance(component, str):
            if component in _COMPONENTS:
                return self._single(index, matrices[component])
            if component in _POLARIZATION:
                if not self.locals[index].cartesian:
                    raise ValueError("polarization operators are defined for spins-1/2.")
                return self._single(index, matrices["e"] / 2 + _POLARIZATION[component] * matrices["z"])
            raise ValueError(f"unknown component {component!r}.")
        if isinstance(component, (tuple, list)):
            theta, phi = (sp.sympify(a) for a in component)
            n = (sp.sin(theta) * sp.cos(phi), sp.sin(theta) * sp.sin(phi), sp.cos(theta))
        else:
            phi = sp.sympify(component)
            n = (sp.cos(phi), sp.sin(phi), sp.S.Zero)
        return self._single(index, n[0] * matrices["x"] + n[1] * matrices["y"] + n[2] * matrices["z"])

    def dot(self, a: Any, b: Any) -> Operator:
        """The scalar product ``I_a · I_b``."""
        return sum((self.I(a, c) @ self.I(b, c) for c in "xyz"), self.zero())

    def T(self, spins: Any, k: int, q: int) -> Operator:  # noqa: N802 - the name of the operator
        """The spherical tensor operator ``T(k, q)`` of one spin, or of a pair of spins (``k`` = 0, 1, 2).

        The pair operator couples the rank-1 operators of the two spins,
        ``T(k, q) = Σ <1 q1, 1 q2|k q> T(1, q1) T(1, q2)``.
        """
        k, q = int(k), int(q)
        if abs(q) > k:
            raise ValueError(f"|q| = {abs(q)} exceeds the rank {k}.")
        if isinstance(spins, (list, tuple)):
            if len(spins) != 2 or k > 2:
                raise ValueError("a coupled tensor operator takes two spins and a rank of 0, 1 or 2.")
            a, b = spins
            total = self.zero()
            for q1 in (1, 0, -1):
                q2 = q - q1
                if abs(q2) <= 1:
                    total = total + CG(1, q1, 1, q2, k, q).doit() * (self.T(a, 1, q1) @ self.T(b, 1, q2))
            return total
        index = self.index(spins)
        loc = self.locals[index]
        if k > 2 * loc.spin:
            raise ValueError(f"a spin {loc.spin} has no tensor operator of rank {k}.")
        return self._single(index, tensor_matrix(loc.spin, k, q))

    def _rotation(self, spec: tuple, spins: Any) -> Operator:
        out = self.unity()
        for index in self._indices(spins):
            terms = {}
            for label, value in self.locals[index].rotation_operator(spec):
                key = list(self.identity_term)
                key[index] = label
                terms[tuple(key)] = value
            out = out @ Operator(self, terms)
        return out

    def rotation(self, angle: Any, phase: Any = "x", *, spins: Any = None, axis: Any = None) -> Operator:
        """The rotation operator ``exp(−i angle n·I)`` of ``spins`` (default: all), axis as in ``Operator.rotate``."""
        return self._rotation(rotation_spec(angle, phase, axis), spins)

    def rotation_euler(self, alpha: Any, beta: Any, gamma: Any, *, spins: Any = None) -> Operator:
        """The rotation operator ``Rz(α) Ry(β) Rz(γ)`` of ``spins`` (default: all)."""
        return self._rotation(("euler", sp.sympify(alpha), sp.sympify(beta), sp.sympify(gamma)), spins)

    def from_matrix(self, matrix: Any, basis: Basis | None = None) -> Operator:
        """The operator with this matrix in the Zeeman basis (or in ``basis``)."""
        matrix = sp.Matrix(matrix)
        if matrix.shape != (self.dimension, self.dimension):
            raise ValueError(f"expected a {self.dimension} x {self.dimension} matrix.")
        if basis is not None:
            matrix = basis.kets_matrix * matrix * basis.kets_matrix.H
        terms: dict[Term, sp.Expr] = {}

        def peel(block: sp.Matrix, index: int, tail: tuple, factor: sp.Expr) -> None:
            # the last spin is the slowest index: split it off and expand it in its own basis
            if index < 0:
                c = tidy(factor * block[0, 0])
                if c != 0:
                    terms[tail] = c
                return
            loc = self.locals[index]
            size = block.shape[0] // loc.dim
            for label in loc.labels:
                adjoint = loc.matrices[label].H
                rest = sp.zeros(size, size)
                for a in range(loc.dim):
                    for b in range(loc.dim):
                        if adjoint[b, a] != 0:
                            rest += adjoint[b, a] * block[a * size : (a + 1) * size, b * size : (b + 1) * size]
                if not rest.is_zero_matrix:
                    peel(rest, index - 1, (label, *tail), factor / loc.norm2[label])

        peel(matrix, len(self.spins) - 1, (), sp.S.One)
        return Operator(self, terms)

    def permutation(self, cycles: Sequence[Any]) -> Operator:
        """The operator that permutes spins. ``[1, 2, 3]`` moves the state of spin 1 to spin 2, of 2 to 3 and of
        3 to 1; a list of cycles is applied from right to left."""
        if cycles and isinstance(cycles[0], (list, tuple)):
            out = self.unity()
            for cycle in cycles:
                out = out @ self.permutation(cycle)
            return out
        indices = [self.index(s) for s in cycles]
        if len({self.spins[i][1] for i in indices}) > 1:
            raise ValueError("only spins of the same quantum number can be permuted.")
        target = dict(zip(indices, indices[1:] + indices[:1]))
        position = {state: n for n, state in enumerate(self.quantum_numbers)}
        matrix = sp.zeros(self.dimension, self.dimension)
        for column, state in enumerate(self.quantum_numbers):
            moved = list(state)
            for source, destination in target.items():
                moved[destination] = state[source]
            matrix[position[tuple(moved)], column] = 1
        return self.from_matrix(matrix)

    def shift(self, ket: Ket, bra: Bra | Ket) -> Operator:
        """``|ket><bra|`` with both states normalised."""
        bra = bra.dagger() if isinstance(bra, Ket) else bra
        return ket.normalized() @ bra.normalized()

    def projector(self, ket: Ket) -> Operator:
        """``|ket><ket|`` for the normalised ket."""
        return self.shift(ket, ket)

    def population(self, state: Any, basis: Basis | None = None) -> Operator:
        """The population operator ``|r><r|`` of a state of ``basis`` (default: the Zeeman basis)."""
        ket = (basis or self.zeeman).ket(state)
        return ket @ ket.dagger()

    def coherence(self, r: Any, s: Any, basis: Basis | None = None) -> Operator:
        """The coherence operator ``|r><s|`` of two states of ``basis`` (default: the Zeeman basis)."""
        basis = basis or self.zeeman
        return basis.ket(r) @ basis.ket(s).dagger()

    def single_transition(self, r: Any, s: Any, component: str, basis: Basis | None = None) -> Operator:
        """A single-transition operator of the states ``r`` and ``s``: ``'x'``, ``'y'``, ``'z'``, ``'+'``, ``'-'``
        or ``'u'`` (the unity operator of the pair)."""
        rs, sr = self.coherence(r, s, basis), self.coherence(s, r, basis)
        rr, ss = self.population(r, basis), self.population(s, basis)
        table = {"x": (rs + sr) / 2, "y": (rs - sr) / (2 * sp.I), "z": (rr - ss) / 2, "+": rs, "-": sr, "u": rr + ss}
        if component not in table:
            raise ValueError(f"unknown component {component!r}.")
        return table[component]

    # ------------------------------------------------------------------ states and bases
    def ket(self, *m: Any) -> Ket:
        """The Zeeman product ket with the magnetic quantum numbers ``m1, m2, ...`` of the spins."""
        if len(m) != len(self.spins):
            raise ValueError(f"expected {len(self.spins)} magnetic quantum numbers.")
        state = tuple(sp.Rational(sp.nsimplify(value, rational=True)) for value in m)
        if state not in self.quantum_numbers:
            raise ValueError(f"{m} is not a state of {self!r}.")
        vector = sp.zeros(self.dimension, 1)
        vector[self.quantum_numbers.index(state)] = 1
        return Ket(self, vector)

    @cached_property
    def zeeman(self) -> Basis:
        """The Zeeman product basis."""
        return Basis(self, "Zeeman", sp.eye(self.dimension))

    def angular_momentum_basis(self) -> Basis:
        """The coupled basis: eigenkets of the total angular momentum and of its z component.

        The spins are coupled one after the other with Clebsch-Gordan coefficients. The kets are ordered by
        ``J`` (descending), then by the intermediate angular momenta, then by ``M`` (descending);
        ``basis.quantum_numbers`` lists ``(J, M, intermediate J)``. SpinDynamica's ``AngularMomentumBasis`` spans
        the same ``(J, M)`` subspaces in another order.
        """
        vectors, numbers = _coupled(self)
        basis = Basis(self, "angular momentum", sp.Matrix.hstack(*vectors))
        basis.quantum_numbers = numbers  # type: ignore[attr-defined]
        return basis

    def eigenbasis(self, operator: Operator, target: Basis | None = None, values=None) -> Basis:
        """A basis of eigenkets of a Hermitian operator, by ascending eigenvalue (numeric).

        With ``target`` the eigenkets are put in the order in which they match the kets of that basis best.
        """
        import numpy as np

        eigenvalues, vectors = np.linalg.eigh(operator.numeric(values))
        if target is not None:
            reference = np.array(target.kets_matrix.evalf(30).tolist(), dtype=complex)
            overlap = np.abs(reference.conj().T @ vectors)
            order = []
            for row in range(len(eigenvalues)):
                free = [c for c in np.argsort(-overlap[row]) if c not in order]
                order.append(free[0])
            eigenvalues, vectors = eigenvalues[order], vectors[:, order]
        basis = Basis(self, "eigenbasis", sp.Matrix(vectors.tolist()))
        basis.eigenvalues = eigenvalues  # type: ignore[attr-defined]
        return basis

    def singlet_triplet(self) -> Basis:
        """The singlet-triplet basis of two spins-1/2, in SpinDynamica's order ``S0, T+1, T0, T-1``."""
        if [value for _, value in self.spins] != [HALF, HALF]:
            raise ValueError("the singlet-triplet basis is defined for a system of two spins-1/2.")
        up, down = HALF, -HALF
        root = sp.sqrt(2)
        kets = [
            (self.ket(up, down) - self.ket(down, up)) / root,
            self.ket(up, up),
            (self.ket(up, down) + self.ket(down, up)) / root,
            self.ket(down, down),
        ]
        return Basis(self, "singlet-triplet", sp.Matrix.hstack(*(k.vector for k in kets)), ("S0", "T+1", "T0", "T-1"))


def _coupled(system: SpinSystem) -> tuple[list[sp.Matrix], list[tuple]]:
    """The kets of the coupled basis: the spins are coupled one after the other with Clebsch-Gordan coefficients."""
    dims = [loc.dim for loc in system.locals]
    first = system.spins[0][1]
    # every state: (J, M, history of intermediate J, vector over the spins coupled so far)
    states = [(first, first - i, (first,), sp.eye(dims[0])[:, i]) for i in range(dims[0])]
    for index in range(1, len(dims)):
        spin = system.spins[index][1]
        size = states[0][3].shape[0]
        coupled = []
        for history in dict.fromkeys(state[2] for state in states):
            members = {state[1]: state[3] for state in states if state[2] == history}
            j1 = history[-1]
            total = j1 + spin
            while total >= abs(j1 - spin):
                for step in range(int(2 * total) + 1):
                    m = total - step
                    vector = sp.zeros(size * dims[index], 1)
                    for k in range(dims[index]):
                        m2 = spin - k
                        m1 = m - m2
                        if m1 in members:
                            c = CG(j1, m1, spin, m2, total, m).doit()
                            if c != 0:
                                unit = sp.zeros(dims[index], 1)
                                unit[k] = 1
                                vector += c * sp.kronecker_product(unit, members[m1])  # the first spin fastest
                    coupled.append((total, m, (*history, total), vector))
                total -= 1
        states = coupled
    states.sort(key=lambda state: (-state[0], state[2], -state[1]))
    return [state[3] for state in states], [(state[0], state[1], state[2]) for state in states]


class Basis:
    """An orthonormal basis of the Hilbert space: its kets are the columns of ``kets_matrix`` (Zeeman
    components). States are referred to by their number (from 1) or by their name."""

    def __init__(self, system: SpinSystem, name: str, kets_matrix: sp.Matrix, names: Sequence[str] | None = None):
        self.system = system
        self.name = name
        self.kets_matrix = sp.ImmutableMatrix(kets_matrix)
        self.names = tuple(names) if names else None

    def __repr__(self) -> str:
        return f"<{self.name} basis of {self.system!r}>"

    @property
    def labels(self) -> tuple[int, ...]:
        return tuple(range(1, self.system.dimension + 1))

    def position(self, state: Any) -> int:
        if self.names and state in self.names:
            return self.names.index(state)
        if isinstance(state, int) and 1 <= state <= self.system.dimension:
            return state - 1
        raise ValueError(f"{state!r} is not a state of the {self.name} basis.")

    def ket(self, state: Any) -> Ket:
        return Ket(self.system, sp.Matrix(self.kets_matrix[:, self.position(state)]))

    @property
    def kets(self) -> list[Ket]:
        return [self.ket(n) for n in self.labels]

    def transformed(self, operator: Operator, name: str | None = None) -> Basis:
        """The basis whose kets are ``operator`` applied to the kets of this one (``TransformedBasis``)."""
        return Basis(self.system, name or f"transformed {self.name}", operator.matrix() * self.kets_matrix)

    def transformation(self, other: Basis) -> sp.Matrix:
        """The matrix of overlaps ``<self_i|other_j>`` (SpinDynamica's ``BasisTransformationMatrix[self, other]``)."""
        return (self.kets_matrix.H * other.kets_matrix).applyfunc(tidy)


class _State:
    __slots__ = ("system", "vector")
    __hash__ = None  # type: ignore[assignment]

    def __init__(self, system: SpinSystem, vector: sp.Matrix):
        self.system = system
        self.vector = sp.ImmutableMatrix(vector.applyfunc(tidy))

    def _new(self, vector: sp.Matrix):
        return type(self)(self.system, vector)

    def __add__(self, other):
        if type(other) is not type(self) or other.system is not self.system:
            return NotImplemented
        return self._new(self.vector + other.vector)

    def __sub__(self, other):
        if type(other) is not type(self) or other.system is not self.system:
            return NotImplemented
        return self._new(self.vector - other.vector)

    def __neg__(self):
        return self._new(-self.vector)

    def __mul__(self, scalar: Any):
        return self._new(sp.sympify(scalar) * self.vector)

    __rmul__ = __mul__

    def __truediv__(self, scalar: Any):
        return self._new(self.vector / sp.sympify(scalar))

    def norm(self) -> sp.Expr:
        return sp.sqrt(tidy(sum(sp.conjugate(c) * c for c in self.vector)))

    def normalized(self):
        return self / self.norm()

    def _text(self, left: str, right: str) -> str:
        parts = []
        for value, state in zip(self.vector, self.system.quantum_numbers):
            if value != 0:
                name = left + ", ".join(str(m) for m in state) + right
                parts.append(name if value == 1 else f"({value})*{name}")
        return " + ".join(parts) or "0"


class Ket(_State):
    """A state vector, with its components in the Zeeman basis."""

    def magnetic_quantum_numbers(self, spins: Any = None) -> list[sp.Rational]:
        """The magnetic quantum numbers ``M`` (of all spins, or summed over ``spins``) that the ket contains."""
        indices = self.system._indices(spins)
        states = self.system.quantum_numbers
        return sorted({sum(states[n][i] for i in indices) for n, c in enumerate(self.vector) if c != 0}, reverse=True)

    def dagger(self) -> Bra:
        return Bra(self.system, self.vector.conjugate())

    def __matmul__(self, other: Any):
        if isinstance(other, Bra):
            return self.system.from_matrix(self.vector * other.vector.T)
        return NotImplemented

    def __repr__(self) -> str:
        return self._text("|", ">")


class Bra(_State):
    """The adjoint of a ket; ``vector`` holds the conjugated components."""

    def dagger(self) -> Ket:
        return Ket(self.system, self.vector.conjugate())

    def __matmul__(self, other: Any):
        if isinstance(other, Ket):
            return tidy((self.vector.T * other.vector)[0, 0])
        if isinstance(other, Operator):
            return Bra(self.system, (self.vector.T * other.matrix()).T)
        return NotImplemented

    def __repr__(self) -> str:
        return self._text("<", "|")


def spins_of(operator: Operator) -> list[Any]:
    """The labels of the spins an operator involves."""
    system = operator.system
    involved = {i for key in operator.terms for i, label in enumerate(key) if label != "e"}
    return [system.spins[i][0] for i in sorted(involved)]


__all__ = ["Basis", "Bra", "Ket", "SpinSystem", "spins_of"]
