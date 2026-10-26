"""Spin operators and superoperators.

Ports ``pauli.m``, ``irr_sph_ten.m``, ``human2opspec.m``, ``operator.m``,
``ist_product_table.m`` and ``superop.m``. As in Spinach, every operator is
assembled from single-spin irreducible spherical tensors T(l,m), which keeps
one code path for any multiplicity; an operator specification ("opspec")
lists the linear tensor index ``l*l + l - m`` for each spin, 0 meaning the
unit operator.

In ``zeeman-hilb`` an opspec becomes a Kronecker product of tensors. In
``sphten-liouv`` it becomes a superoperator in the restricted basis: the
product of two spherical tensors on one spin is a known linear combination
of spherical tensors on that spin (the structure coefficients in
:func:`ist_product_table`), so the action of an operator on a product basis
state factorises over spins and never needs the full Liouville space.

Spinach authors, by original file (headers of Spinach 2.13; see docs/spinach-credits.md):
  human2opspec.m: Ilya Kuprov, Luke Edwards
  irr_sph_ten.m: Ilya Kuprov, Hannah Hogben
  ist_product_table.m: Hannah Hogben, Ilya Kuprov
  lin2lm.m: Ilya Kuprov
  lm2lin.m: Ilya Kuprov
  operator.m: Luke Edwards, Ilya Kuprov
  pauli.m: Ilya Kuprov
  superop.m: Ilya Kuprov, Hannah Hogben
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from functools import cache

import numpy as np
import scipy.sparse as sps

from spynach.system import SpinSystem

SQRT2 = np.sqrt(2)


@cache
def pauli(mult: int) -> dict[str, sps.csr_array]:
    """Spin operators ``'u'`` (unit), ``'p'``, ``'m'``, ``'z'``, ``'x'``, ``'y'`` for a multiplicity.

    Basis order is descending projection, m = s, s-1, ..., -s.

    Spinach authors (pauli.m): Ilya Kuprov.
    """
    if mult < 1:
        raise ValueError("mult must be a positive real integer.")
    s = (mult - 1) / 2
    prjs = np.arange(mult - 1, -1, -1) - s
    plus = np.sqrt(s * (s + 1) - prjs[1:] * (prjs[1:] + 1))
    minus = np.sqrt(s * (s + 1) - prjs[:-1] * (prjs[:-1] - 1))
    ops = {
        "u": sps.eye_array(mult, dtype=complex, format="csr"),
        "p": sps.csr_array(sps.diags_array(plus.astype(complex), offsets=1, shape=(mult, mult))),
        "m": sps.csr_array(sps.diags_array(minus.astype(complex), offsets=-1, shape=(mult, mult))),
        "z": sps.csr_array(sps.diags_array(prjs.astype(complex), offsets=0, shape=(mult, mult))),
    }
    ops["x"] = (ops["p"] + ops["m"]) / 2
    ops["y"] = (ops["p"] - ops["m"]) / 2j
    return ops


@cache
def irr_sph_ten(mult: int, rank: int) -> tuple[sps.csr_array, ...]:
    """Irreducible spherical tensor operators T(rank, m) for m = rank, rank-1, ..., -rank.

    Spinach authors (irr_sph_ten.m): Ilya Kuprov, Hannah Hogben.
    """
    if not 0 <= rank < mult:
        raise ValueError("k must be an integer between 0 and mult-1.")
    if rank == 0:
        return (sps.eye_array(mult, dtype=complex, format="csr"),)
    ops = pauli(mult)
    tensors = [((-1) ** rank) * (2 ** (-rank / 2)) * _matrix_power(ops["p"], rank)]
    for n in range(1, 2 * rank + 1):
        q = rank - n + 1
        prev = tensors[-1]
        tensors.append((1 / np.sqrt((rank + q) * (rank - q + 1))) * (ops["m"] @ prev - prev @ ops["m"]))
    return tuple(sps.csr_array(t) for t in tensors)


def lin2lm(index):
    """Linear spherical tensor index to (l, m): index = l*l + l - m. Works on ints and arrays.

    Spinach authors (lin2lm.m): Ilya Kuprov.
    """
    if isinstance(index, (int, np.integer)):
        rank = int(np.sqrt(index))
        return rank, rank * rank + rank - index
    index = np.asarray(index)
    rank = np.floor(np.sqrt(index)).astype(np.int64)
    return rank, rank * rank + rank - index


def lm2lin(rank: int, proj: int) -> int:
    if rank < 0 or abs(proj) > rank:
        raise ValueError("unacceptable projection number.")
    return rank * rank + rank - proj


def human2opspec(
    spin_system: SpinSystem, operators: str | Sequence[str], spins: str | int | Sequence[int]
) -> tuple[list[np.ndarray], np.ndarray]:
    """Parse a human-readable operator description into opspecs and coefficients.

    Three forms, as in Spinach (but with 0-based spin indices):

    * ``('Lz', '13C')``: sum of single-spin operators over all spins of that
      isotope; ``'all'``, ``'electrons'``, and ``'nuclei'`` are also accepted.
    * ``('Lz', [0, 2])``: sum over the listed spins.
    * ``(['Lz', 'L+'], [0, 1])``: product operator Lz(0) L+(1).

    Operator labels: ``'E'``, ``'Lz'``, ``'Lx'``, ``'Ly'``, ``'L+'``, ``'L-'``, ``'Tl,m'``.

    Spinach authors (human2opspec.m): Ilya Kuprov, Luke Edwards.
    """
    if isinstance(operators, str):
        specs: list[np.ndarray] = []
        coeffs: list[np.ndarray] = []
        for n in _spin_numbers(spin_system, spins):
            sub_specs, sub_coeffs = human2opspec(spin_system, [operators], [n])
            specs += sub_specs
            coeffs.append(sub_coeffs)
        return specs, np.concatenate(coeffs)
    operators = list(operators)
    if isinstance(spins, (str, int, np.integer)):
        raise TypeError("a product operator needs a sequence of spin indices, one per operator.")
    spins = [int(n) for n in spins]
    if len(operators) != len(spins):
        raise ValueError("spins and operators lists should have the same number of elements.")
    _check_spin_indices(spin_system, spins)

    stack = np.zeros((1, spin_system.nspins), dtype=np.int64)
    coeffs_arr = np.ones(1, dtype=complex)
    for label, n in zip(operators, spins):
        if spin_system.types[n] != "S":
            raise NotImplementedError("operators on bosonic modes are not ported yet.")
        if label == "E":
            stack[:, n] = 0
        elif label == "L+":
            stack[:, n] = 1
            coeffs_arr = -SQRT2 * coeffs_arr
        elif label == "L-":
            stack[:, n] = 3
            coeffs_arr = SQRT2 * coeffs_arr
        elif label == "Lz":
            stack[:, n] = 2
        elif label in ("Lx", "Ly"):
            stack_a, stack_b = stack.copy(), stack.copy()
            stack_a[:, n], stack_b[:, n] = 1, 3
            stack = np.vstack([stack_a, stack_b])
            factors = np.array([-SQRT2, SQRT2]) / 2 if label == "Lx" else np.array([-SQRT2, -SQRT2]) / 2j
            coeffs_arr = np.kron(factors, coeffs_arr)
        elif match := re.fullmatch(r"T([+-]?\d+),([+-]?\d+)", label):
            rank, proj = int(match[1]), int(match[2])
            if rank < 0 or abs(proj) > rank or lm2lin(rank, proj) + 1 > spin_system.mults[n] ** 2:
                raise ValueError("invalid irreducible spherical tensor indices.")
            stack[:, n] = lm2lin(rank, proj)
        elif label.startswith(("CT", "ZL")):
            raise NotImplementedError(f"operator {label!r} not ported yet.")
        else:
            raise ValueError(f"unrecognized operator or state specification {label!r}.")
    return list(stack), coeffs_arr


OPERATOR_TYPES = ("left", "right", "comm", "acomm")


def operator(
    spin_system: SpinSystem,
    operators: str | Sequence[str],
    spins: str | int | Sequence[int],
    operator_type: str = "comm",
) -> sps.csr_array:
    """Operator from a human-readable description; see :func:`human2opspec`.

    In ``zeeman-hilb`` the result is the Hilbert-space matrix and
    ``operator_type`` is ignored, as in Spinach. In ``sphten-liouv`` it is a
    superoperator in the restricted basis: ``'left'`` (``Q rho``),
    ``'right'`` (``rho Q``), ``'comm'`` (``[Q, rho]``, the default) or
    ``'acomm'`` (``{Q, rho}``).

    Spinach authors (operator.m): Luke Edwards, Ilya Kuprov.
    """
    if spin_system.basis is None:
        raise ValueError("basis set information is missing, run basis() before calling this function.")
    if operator_type not in OPERATOR_TYPES:
        raise ValueError(f"operator_type must be one of {OPERATOR_TYPES}.")
    opspecs, coeffs = human2opspec(spin_system, operators, spins)
    formalism = spin_system.basis.formalism
    if formalism == "zeeman-hilb":
        dim = int(np.prod(spin_system.mults))
        total = sps.csr_array((dim, dim), dtype=complex)
        for spec, coeff in zip(opspecs, coeffs):
            term = sps.csr_array(np.array([[coeff]], dtype=complex))
            for mult, index in zip(spin_system.mults, spec):
                rank, proj = lin2lm(int(index))
                term = sps.kron(term, irr_sph_ten(int(mult), rank)[rank - proj], format="csr")
            total = total + term
        return total
    if formalism != "sphten-liouv":
        raise NotImplementedError(f"formalism {formalism!r} not ported yet.")
    rows, cols, vals = [], [], []
    for spec, coeff in zip(opspecs, coeffs):
        r, c, v = superop(spin_system, spec, operator_type)
        rows.append(r)
        cols.append(c)
        vals.append(coeff * v)
    nstates = spin_system.basis.nstates
    # Duplicate entries are summed, as MATLAB's sparse() does
    return sps.csr_array(
        sps.coo_array((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(nstates, nstates))
    )


@cache
def ist_product_table(mult: int) -> tuple[np.ndarray, np.ndarray]:
    """Structure coefficients of the spherical tensors of one spin (port of ``ist_product_table.m``).

    With ``Tn`` the tensor of linear index ``n`` and ``Bn = Tn / ||Tn||`` the
    normalised basis state, ``left[n, m, k] = <Bk, Tn Bm>`` and
    ``right[n, m, k] = <Bk, Bm Tn>`` (Frobenius products): the expansion of
    the unnormalised operator acting on a basis state in basis states.
    Elements below 1e-14 are rounding residue and are set to zero. The
    spin-1/2 tables are hard-coded, as in Spinach, so that they are exact.

    Spinach authors (ist_product_table.m): Hannah Hogben, Ilya Kuprov.
    """
    if mult < 2:
        raise ValueError("mult must be a real integer greater than 1.")
    if mult == 2:
        return _SPIN_HALF_LEFT, _SPIN_HALF_RIGHT
    tensors = np.array([t.toarray() for rank in range(mult) for t in irr_sph_ten(mult, rank)])
    norms = np.sqrt(np.einsum("nij,nij->n", tensors.conj(), tensors).real)
    normalised = tensors / norms[:, None, None]
    left = np.empty((mult * mult,) * 3, dtype=complex)
    right = np.empty_like(left)
    for n in range(mult * mult):
        left[n] = norms[n] * np.einsum("kij,mij->mk", normalised.conj(), normalised[n] @ normalised)
        right[n] = norms[n] * np.einsum("kij,mij->mk", normalised.conj(), normalised @ normalised[n])
    for table in (left, right):
        table[np.abs(table) < 1e-14] = 0
    return np.real_if_close(left), np.real_if_close(right)


# ist_product_table.m, spin-1/2: [n, m, k] with pages k; index order E, T(1,+1), T(1,0), T(1,-1)
_SPIN_HALF_LEFT = np.transpose(
    np.array(
        [
            [[1.0, 0, 0, 0], [0, 0, 0, -0.5], [0, 0, 0.5, 0], [0, -0.5, 0, 0]],
            [[0, 1.0, 0, 0], [0.5, 0, -0.5, 0], [0, 0.5, 0, 0], [0, 0, 0, 0]],
            [[0, 0, 1.0, 0], [0, 0, 0, -0.5], [0.5, 0, 0, 0], [0, 0.5, 0, 0]],
            [[0, 0, 0, 1.0], [0, 0, 0, 0], [0, 0, 0, -0.5], [0.5, 0, 0.5, 0]],
        ]
    ),
    (1, 2, 0),
)
_SPIN_HALF_RIGHT = np.transpose(
    np.array(
        [
            [[1.0, 0, 0, 0], [0, 0, 0, -0.5], [0, 0, 0.5, 0], [0, -0.5, 0, 0]],
            [[0, 1.0, 0, 0], [0.5, 0, 0.5, 0], [0, -0.5, 0, 0], [0, 0, 0, 0]],
            [[0, 0, 1.0, 0], [0, 0, 0, 0.5], [0.5, 0, 0, 0], [0, -0.5, 0, 0]],
            [[0, 0, 0, 1.0], [0, 0, 0, 0], [0, 0, 0, 0.5], [0.5, 0, -0.5, 0]],
        ]
    ),
    (1, 2, 0),
)
_SPIN_HALF_LEFT.flags.writeable = _SPIN_HALF_RIGHT.flags.writeable = False


def superop(spin_system: SpinSystem, opspec: np.ndarray, side: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Superoperator of one opspec in the ``sphten-liouv`` basis as COO triplets (port of ``superop.m``).

    ``side`` is ``'left'``, ``'right'``, ``'comm'`` or ``'acomm'``. Returns
    ``(rows, cols, vals)`` with ``A[row, col] = val`` for ``A`` acting on
    column vectors; duplicates may occur and are to be summed.

    Spinach authors (superop.m): Ilya Kuprov, Hannah Hogben.
    """
    if side == "comm":
        rows_l, cols_l, vals_l = _superop_side(spin_system, opspec, "left", commutator=True)
        rows_r, cols_r, vals_r = _superop_side(spin_system, opspec, "right", commutator=True)
        return np.concatenate([rows_l, rows_r]), np.concatenate([cols_l, cols_r]), np.concatenate([vals_l, -vals_r])
    if side == "acomm":
        rows_l, cols_l, vals_l = _superop_side(spin_system, opspec, "left")
        rows_r, cols_r, vals_r = _superop_side(spin_system, opspec, "right")
        return np.concatenate([rows_l, rows_r]), np.concatenate([cols_l, cols_r]), np.concatenate([vals_l, vals_r])
    if side not in ("left", "right"):
        raise ValueError("invalid side specification.")
    return _superop_side(spin_system, opspec, side)


OP_CACHE_LIMIT = 1024  # entries per basis, first in first out (an entry is a few MB at 10^4 states)


def _superop_side(
    spin_system: SpinSystem, opspec: np.ndarray, side: str, commutator: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Triplets of one side, cached on the basis since sequences and Hamiltonians ask for the same ones repeatedly."""
    cache = spin_system.basis.op_cache
    key = (np.asarray(opspec, dtype=np.int64).tobytes(), side, commutator)
    hit = cache.get(key)
    if hit is None:
        hit = _superop_side_build(spin_system, opspec, side, commutator)
        if len(cache) >= OP_CACHE_LIMIT:
            del cache[next(iter(cache))]
        cache[key] = hit
    return hit


def _superop_side_build(
    spin_system: SpinSystem, opspec: np.ndarray, side: str, commutator: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    basis = spin_system.basis.basis
    nstates = basis.shape[0]
    opspec = np.asarray(opspec)
    if np.any(opspec + 1 > spin_system.mults**2):
        raise ValueError("physically impossible state requested in opspec.")
    active = np.flatnonzero(opspec)
    if active.size == 0:
        idx = np.arange(nstates)
        return idx, idx, np.ones(nstates, dtype=complex)

    # Paths on each active spin: (input tensor, output tensor, coefficient), then all combinations
    ins, outs, coeffs = [], [], []
    for n in active:
        table = ist_product_table(int(spin_system.mults[n]))[0 if side == "left" else 1]
        page = table[opspec[n]]  # page[m, k]: input state m goes to output state k
        m, k = np.nonzero(page)
        ins.append(m)
        outs.append(k)
        coeffs.append(page[m, k])
    grids = np.meshgrid(*(np.arange(len(x)) for x in ins), indexing="ij")
    choice = [g.ravel() for g in grids]
    path_in = np.stack([ins[i][c] for i, c in enumerate(choice)], axis=1)
    path_out = np.stack([outs[i][c] for i, c in enumerate(choice)], axis=1)
    path_coeff = np.prod(np.stack([coeffs[i][c] for i, c in enumerate(choice)], axis=1), axis=1)
    if commutator:
        # Paths from or into the all-unit pattern on the active spins cancel between the two sides
        keep = (path_in.sum(axis=1) != 0) & (path_out.sum(axis=1) != 0)
        path_in, path_out, path_coeff = path_in[keep], path_out[keep], path_coeff[keep]

    # Every path maps the rows carrying the input pattern on the active spins onto the rows with the
    # same passive pattern and the output pattern: the target row hash is the source hash shifted by
    # the key difference on the active spins, and the target row is found by bisection
    bas = spin_system.basis
    table, row_hash, order, sorted_hash = bas.hash_table, bas.row_hash, bas.sort_order, bas.sorted_hash
    active_cols = basis[:, active]
    rows, cols, vals = [], [], []
    for pin, pout, coeff in zip(path_in, path_out, path_coeff):
        src = np.flatnonzero((active_cols == pin).all(axis=1))
        if src.size == 0:
            continue
        # Hash arithmetic is modulo 2**64 by design; array operations wrap silently
        shift = np.zeros(1, dtype=np.uint64)
        for a, i, o in zip(active, pin, pout):
            shift = shift + table[a, o] - table[a, i]
        target = row_hash[src] + shift
        pos = np.searchsorted(sorted_hash, target)
        pos[pos == sorted_hash.size] = 0
        found = sorted_hash[pos] == target
        if not found.any():
            continue  # the output pattern is not in the restricted basis for any of these rows
        rows.append(order[pos[found]])
        cols.append(src[found])
        vals.append(np.full(int(found.sum()), coeff, dtype=complex))
    if not rows:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64), np.zeros(0, dtype=complex)
    return np.concatenate(rows), np.concatenate(cols), np.concatenate(vals)


def _spin_numbers(spin_system: SpinSystem, spins: str | int | Sequence[int]) -> list[int]:
    if isinstance(spins, str):
        if spins == "all":
            numbers = list(range(spin_system.nspins))
        elif spins == "electrons":
            numbers = [n for n, iso in enumerate(spin_system.isotopes) if iso.startswith("E")]
        elif spins == "nuclei":
            numbers = [n for n, iso in enumerate(spin_system.isotopes) if not iso.startswith("E")]
        else:
            numbers = [n for n, iso in enumerate(spin_system.isotopes) if iso == spins]
        if not numbers:
            raise ValueError("no such spins in the system.")
        return numbers
    numbers = [int(spins)] if isinstance(spins, (int, np.integer)) else [int(n) for n in spins]
    if not numbers:
        raise ValueError("spin list cannot be empty.")
    _check_spin_indices(spin_system, numbers)
    return numbers


def _check_spin_indices(spin_system: SpinSystem, numbers: list[int]) -> None:
    if any(n < 0 or n >= spin_system.nspins for n in numbers):
        raise ValueError(f"spin indices must be between 0 and {spin_system.nspins - 1}.")
    if len(set(numbers)) != len(numbers):
        raise ValueError("spin list must not have any repetitions.")


def _matrix_power(matrix: sps.csr_array, power: int) -> sps.csr_array:
    result = matrix
    for _ in range(power - 1):
        result = result @ matrix
    return result
