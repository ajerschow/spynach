"""Spin permutation groups, symmetry-adapted bases and the pairing of permutation and rotation symmetry.

A permutation is a list of cycles of spin labels, as for ``SpinSystem.permutation``; a group is the list of its
permutations. :func:`symmetry_adapted_basis` builds kets that belong to irreducible representations of the
group and to a total angular momentum at the same time, which is possible because permutations of equal spins
commute with rotations.

Author: Alexej Jerschow.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import sympy as sp
from sympy.combinatorics import named_groups

from spyndynamica.operators import Operator
from spyndynamica.system import Basis, SpinSystem

TOLERANCE = 1e-8
_NAMED = {
    "SymmetricGroup": named_groups.SymmetricGroup,
    "AlternatingGroup": named_groups.AlternatingGroup,
    "CyclicGroup": named_groups.CyclicGroup,
    "DihedralGroup": named_groups.DihedralGroup,
}
Permutation = tuple[tuple[Any, ...], ...]


def _order(permutation: Permutation) -> tuple:
    """SpinDynamica lists the elements in Mathematica's order of their cycles: fewer cycles first, then
    shorter cycles, then by the labels."""
    return (len(permutation), [(len(cycle), list(map(str, cycle))) for cycle in permutation])


def permutation_group(name: str, degree: int, spins: Sequence[Any] | None = None) -> list[Permutation]:
    """The permutations of a named group as tuples of cycles, the identity (no cycle) first.

    ``name`` is ``'SymmetricGroup'``, ``'AlternatingGroup'``, ``'CyclicGroup'`` or ``'DihedralGroup'`` and
    ``degree`` the number of objects it permutes. The objects are ``1, 2, ...`` or the labels ``spins``.
    """
    if name not in _NAMED:
        raise ValueError(f"unknown group {name!r}; known are {sorted(_NAMED)}.")
    degree = int(degree)
    labels = list(range(1, degree + 1)) if spins is None else list(spins)
    if len(labels) != degree or len(set(labels)) != degree:
        raise ValueError(f"a group of degree {degree} needs {degree} distinct spins.")
    elements = []
    for element in _NAMED[name](degree).elements:
        cycles = [cycle for cycle in element.cyclic_form if len(cycle) > 1]
        # every cycle starts at its smallest position, and the cycles are in the order of those positions
        elements.append(tuple(tuple(labels[i] for i in cycle) for cycle in sorted(cycles)))
    return sorted(elements, key=_order)


def _permutations(system: SpinSystem, group: Any, spins: Sequence[Any] | None) -> list[Permutation]:
    if isinstance(group, tuple) and len(group) == 2 and isinstance(group[0], str):
        name, degree = group
        if spins is None:
            spins = [label for label, _ in system.spins][: int(degree)]
        return permutation_group(name, degree, spins)
    out = []
    for element in group:
        cycles = list(element)
        if cycles and not isinstance(cycles[0], (list, tuple)):
            cycles = [cycles]
        out.append(tuple(tuple(cycle) for cycle in cycles if len(cycle) > 1))
    return out


def spin_permutation_group(system: SpinSystem, group: Any, spins: Sequence[Any] | None = None) -> list[Operator]:
    """The spin permutation operators of a group (SpinDynamica's ``SpinPermutationGroup``).

    ``group`` is ``(name, degree)`` as for :func:`permutation_group` (acting on ``spins``, by default on the
    first spins of the system) or a list of permutations, each a cycle or a list of cycles of spin labels.
    """
    return [_operator(system, element) for element in _permutations(system, group, spins)]


def _operator(system: SpinSystem, permutation: Permutation) -> Operator:
    return system.permutation([list(cycle) for cycle in permutation]) if permutation else system.unity()


# ====================================================================================== symmetry-adapted basis
def _null(matrix: np.ndarray) -> np.ndarray:
    """An orthonormal basis of the null space, as columns."""
    if matrix.shape[1] == 0:
        return matrix.reshape(0, 0)
    _, values, vh = np.linalg.svd(matrix, full_matrices=True)
    rank = int(np.sum(values > TOLERANCE))
    return vh[rank:].conj().T


def _fix_phase(vector: np.ndarray) -> np.ndarray:
    """The vector with its first sizeable component real and positive."""
    pivot = vector[np.argmax(np.abs(vector) > 0.5 * np.abs(vector).max())]
    return vector * (abs(pivot) / pivot)


def _half(value: float) -> sp.Rational:
    """A multiple of 1/2 as an exact number."""
    return sp.Rational(int(round(2 * value)), 2)


class _Irreps:
    """The irreducible representations met so far, each with the matrices that every copy is brought to."""

    def __init__(self, order: int, rng: np.random.Generator):
        self.order = order
        self.rng = rng
        self.matrices: list[np.ndarray] = []  # per irrep: one matrix per group element

    def identify(self, matrices: np.ndarray) -> tuple[int, np.ndarray]:
        """The number of the irrep with these matrices and the unitary that turns them into the stored ones."""
        dim = matrices.shape[1]
        character = np.trace(matrices, axis1=1, axis2=2)
        for number, reference in enumerate(self.matrices):
            if reference.shape[1] != dim:
                continue
            if not np.allclose(np.trace(reference, axis1=1, axis2=2), character, atol=1e-6):
                continue
            for _ in range(20):
                x = self.rng.normal(size=(dim, dim)) + 1j * self.rng.normal(size=(dim, dim))
                # the intertwiner: matrices[g] @ t = t @ reference[g], a multiple of a unitary matrix
                t = np.einsum("gab,bc,gdc->ad", matrices, x, reference.conj())
                scale = np.sqrt(np.trace(t.conj().T @ t).real / dim)
                if scale > 1e-6 * self.order:
                    return number, t / scale
            raise RuntimeError("no intertwiner found between equivalent representations.")
        self.matrices.append(matrices)
        return len(self.matrices) - 1, np.eye(dim)


def symmetry_adapted_basis(system: SpinSystem, group: Any = None, spins: Sequence[Any] | None = None) -> Basis:
    """A basis of kets that belong to irreducible representations of a spin permutation group and are
    eigenkets of the total angular momentum and of its z component (SpinDynamica's ``SymmetryAdaptedBasis``).

    ``group`` is given as for :func:`spin_permutation_group`; the default is the symmetric group of all spins.
    In the basis every group operator is block diagonal, each block is an irreducible representation, and
    equivalent representations have the same matrices. The kets are ordered by the dimension of the
    representation, the representation, ``J`` (descending), the copy and ``M`` (descending);
    ``basis.quantum_numbers`` lists ``(irrep, dimension, J, M, copy, partner)`` for every ket. The spins must
    have the same quantum number. The basis is numeric; which kets span a representation that occurs more than
    once, and the kets within a representation of dimension above 1, are a choice.
    """
    labels = [label for label, _ in system.spins]
    if len({value for _, value in system.spins}) != 1:
        raise ValueError("a symmetry-adapted basis needs spins of the same quantum number.")
    if group is None:
        group = ("SymmetricGroup", len(labels))
    elements = [_operator(system, p).numeric() for p in _permutations(system, group, spins)]
    order = len(elements)
    jz = np.diag(system.I(None, "z").numeric()).real
    jplus, jminus = system.I(None, "+").numeric(), system.I(None, "-").numeric()
    rng = np.random.default_rng(0)
    irreps = _Irreps(order, rng)
    found = []  # (irrep, dimension, J, copy, partner, {M: ket})
    for m in sorted(set(np.round(jz, 6)), reverse=True):
        if m < 0:
            break
        columns = np.flatnonzero(np.isclose(jz, m))
        # highest-weight kets of J = m: annihilated by the raising operator
        highest = np.zeros((system.dimension, 0), dtype=complex)
        null = _null(jplus[:, columns])
        if null.size:
            highest = np.zeros((system.dimension, null.shape[1]), dtype=complex)
            highest[columns] = null
        size = highest.shape[1]
        if size == 0:
            continue
        representation = np.array([highest.conj().T @ g @ highest for g in elements])
        seed = rng.normal(size=(size, size)) + 1j * rng.normal(size=(size, size))
        seed = seed + seed.conj().T
        generic = sum(d @ seed @ d.conj().T for d in representation) / order
        values, vectors = np.linalg.eigh(generic)
        copies: dict[int, int] = {}
        start = 0
        while start < size:
            stop = start + 1
            while stop < size and abs(values[stop] - values[start]) < 1e-7:
                stop += 1
            block = vectors[:, start:stop]
            number, turn = irreps.identify(np.array([block.conj().T @ d @ block for d in representation]))
            kets = highest @ block @ turn
            kets = kets * (_fix_phase(kets[:, 0]) @ kets[:, 0].conj())  # a common phase leaves the matrices alone
            copy = copies.get(number, 0)
            copies[number] = copy + 1
            for partner in range(stop - start):
                ladder = {m: kets[:, partner]}
                value = m
                while value > -m + 1e-9:
                    lowered = jminus @ ladder[value]
                    value -= 1
                    ladder[value] = lowered / np.linalg.norm(lowered)
                found.append((number, stop - start, m, copy, partner, ladder))
            start = stop
    rows = []
    for number, dim, j, copy, partner, ladder in found:
        for value, ket in ladder.items():
            rows.append(((dim, number, -j, copy, -value, partner), (number, dim, j, value, copy, partner), ket))
    rows.sort(key=lambda row: row[0])
    if len(rows) != system.dimension:
        raise RuntimeError("the decomposition into irreducible representations is incomplete.")
    kets = np.array([row[2] for row in rows]).T
    kets = np.where(np.abs(kets) < 1e-12, 0, kets)
    basis = Basis(system, "symmetry-adapted", sp.Matrix(kets.tolist()))
    basis.quantum_numbers = [  # type: ignore[attr-defined]
        (number, dim, _half(j), _half(m), copy, partner) for _, (number, dim, j, m, copy, partner), _ in rows
    ]
    return basis


# ====================================================================================== dual pairing
def _partitions(n: int, largest: int | None = None):
    largest = n if largest is None else largest
    if n == 0:
        yield ()
        return
    for first in range(min(n, largest), 0, -1):
        for rest in _partitions(n - first, first):
            yield (first, *rest)


def dual_pairing(spins: int, j: Any) -> dict[tuple[int, ...], dict[sp.Rational, int]]:
    """Which total angular momenta go with which irreducible representation of the symmetric group in a
    system of ``spins`` equal spins ``j`` (SpinDynamica's ``DualPairing``).

    The result maps every partition of ``spins`` with at most ``2j + 1`` rows (the Young diagram of the
    representation, as row lengths) to ``{J: multiplicity}``: the rotation behaviour of the kets that
    transform under that representation. For three spins-1/2 it is ``{(3,): {3/2: 1}, (2, 1): {1/2: 1}}``.
    """
    spins = int(spins)
    j = sp.nsimplify(j, rational=True)
    if spins < 1 or (2 * j).is_integer is not True or j <= 0:
        raise ValueError("expected a positive number of spins and a spin quantum number j = 1/2, 1, 3/2, ...")
    dim = int(2 * j + 1)
    q = sp.Symbol("q")
    # the complete homogeneous symmetric polynomials of the variables q^(2m), m = -j ... j, times q^(2 j k)
    t = sp.Symbol("t")
    generating = sp.series(sp.prod([1 / (1 - q ** (2 * k) * t) for k in range(dim)]), t, 0, spins + 1).removeO()
    h = [sp.Poly(sp.expand(generating.coeff(t, k)), q) for k in range(spins + 1)]
    zero = sp.Poly(0, q)
    out = {}
    for partition in _partitions(spins):
        if len(partition) > dim:
            continue
        size = len(partition)
        # Jacobi-Trudi: the Schur polynomial is det h(partition_i - i + k)
        orders = [[partition[a] - a + b for b in range(size)] for a in range(size)]
        matrix = sp.Matrix([[(h[k] if 0 <= k <= spins else zero).as_expr() for k in row] for row in orders])
        schur = sp.Poly(sp.expand(matrix.det(method="berkowitz")), q)
        # the exponent of q is 2 (M + j spins); the multiplicity of J is the step of the coefficient at M = J
        coefficients = {power[0]: int(c) for power, c in schur.terms()}
        shift = int(2 * j * spins)
        content = {}
        big = j * spins
        while big >= 0:
            count = coefficients.get(int(2 * big) + shift, 0) - coefficients.get(int(2 * big) + shift + 2, 0)
            if count:
                content[sp.nsimplify(big)] = count
            big -= 1
        out[partition] = content
    return out


__all__ = ["dual_pairing", "permutation_group", "spin_permutation_group", "symmetry_adapted_basis"]
