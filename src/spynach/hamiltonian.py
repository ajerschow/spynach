"""Hamiltonian and its irreducible components (ports of ``hamiltonian.m``, ``orientation.m`` and ``frqoffset.m``).

As in Spinach, one term table serves both formalisms: in ``zeeman-hilb``
the terms are Hilbert-space operators, in ``sphten-liouv`` they are
superoperators of the requested ``operator_type`` (``'comm'`` by default,
giving the commutation superoperator ``[H, .]``). Each term carries an
isotropic coefficient and, for the anisotropic part, the coefficients of
its contribution to the rank-1 and rank-2 irreducible spherical components
of the interaction (``ist_coeff``) together with those components
(``irr_comp``, from :func:`spynach.conventions.mat2sphten`). The isotropic
Hamiltonian is ``sum(iso * op)``; the anisotropic part at a given
orientation is ``sum_{r,k,m} D^r(k, m) Q[r][k][m]`` with Wigner matrices
``D^r`` and ``Q[r][k][m] = sum_n ist_coeff[n, r, k] irr_comp[n, r, m] op_n``.

Spinach authors, by original file (headers of Spinach 2.13; see docs/spinach-credits.md):
  frqoffset.m: Ilya Kuprov
  hamiltonian.m: Ilya Kuprov, Luke Edwards, Hannah Hogben, Dmitry Savostyanov
  orientation.m: Ilya Kuprov
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import scipy.sparse as sps

from spynach.conventions import mat2sphten, wigner
from spynach.operators import human2opspec, operator, superop
from spynach.system import SpinSystem
from spynach.utilities import Matrix, as_matrix, clean_up, nnz

SQRT23 = np.sqrt(2 / 3)
SQRT18 = np.sqrt(1 / 8)
RANK1, RANK2 = slice(0, 3), slice(3, 8)  # columns of the 8-element coefficient rows: [+1 0 -1 | +2 +1 0 -1 -2]

# Two-spin product operators of each bilinear coupling strength (hamiltonian.m). The inner
# dictionaries map a position in the 8-element irreducible component row to the coefficient
# of that operator in that component: positions 0, 1, 2 are the rank-1 components
# T(1,+1), T(1,0), T(1,-1) and positions 3 to 7 the rank-2 components T(2,+2) ... T(2,-2)
# (the RANK1 and RANK2 slices above). For example Lz Lz enters only T(2,0) with sqrt(2/3),
# and L+ L- enters T(1,0) with -sqrt(1/8) and T(2,0) with -sqrt(2/3)/4.
_PAIR_TERMS: dict[str, dict[tuple[str, str], dict[int, float]]] = {
    "strong": {
        ("Lz", "Lz"): {5: SQRT23},
        ("L+", "L-"): {1: -SQRT18, 5: -SQRT23 / 4},
        ("L-", "L+"): {1: SQRT18, 5: -SQRT23 / 4},
        ("Lz", "L+"): {0: 0.5, 4: -0.5},
        ("L+", "Lz"): {0: -0.5, 4: -0.5},
        ("Lz", "L-"): {2: 0.5, 6: 0.5},
        ("L-", "Lz"): {2: -0.5, 6: 0.5},
        ("L+", "L+"): {3: 0.5},
        ("L-", "L-"): {7: 0.5},
    },
    "z*": {("Lz", "Lz"): {5: SQRT23}, ("Lz", "L+"): {0: 0.5, 4: -0.5}, ("Lz", "L-"): {2: 0.5, 6: 0.5}},
    "*z": {("Lz", "Lz"): {5: SQRT23}, ("L+", "Lz"): {0: -0.5, 4: -0.5}, ("L-", "Lz"): {2: -0.5, 6: 0.5}},
    "secular": {
        ("Lz", "Lz"): {5: SQRT23},
        ("L+", "L-"): {1: -SQRT18, 5: -SQRT23 / 4},
        ("L-", "L+"): {1: SQRT18, 5: -SQRT23 / 4},
    },
    "zz": {("Lz", "Lz"): {5: SQRT23}},
    "z+": {("Lz", "L+"): {0: 0.5, 4: -0.5}},
    "+z": {("L+", "Lz"): {0: -0.5, 4: -0.5}},
    "T(L,+1)": {("Lz", "L+"): {0: 0.5, 4: -0.5}, ("L+", "Lz"): {0: -0.5, 4: -0.5}},
    "z-": {("Lz", "L-"): {2: 0.5, 6: 0.5}},
    "-z": {("L-", "Lz"): {2: -0.5, 6: 0.5}},
    "T(L,-1)": {("Lz", "L-"): {2: 0.5, 6: 0.5}, ("L-", "Lz"): {2: -0.5, 6: 0.5}},
    "ignore": {},
}
_ZEEMAN_TERMS = {
    "full": {("L+",): {0: -0.5, 4: -0.5}, ("Lz",): {5: SQRT23}, ("L-",): {2: -0.5, 6: 0.5}},
    "secular": {("Lz",): {5: SQRT23}},
    "ignore": {},
}
_QUAD_TERMS = {
    "strong": {
        ("T2,2",): {3: 1.0},
        ("T2,1",): {4: 1.0},
        ("T2,0",): {5: 1.0},
        ("T2,-1",): {6: 1.0},
        ("T2,-2",): {7: 1.0},
    },
    "secular": {("T2,0",): {5: 1.0}},
    "ignore": {},
}


class _Term:
    """One row of the Hamiltonian descriptor: an operator with its coefficients."""

    __slots__ = ("ops", "spins", "iso", "ist", "irr")

    def __init__(self, ops: tuple[str, ...], spins: tuple[int, ...]):
        self.ops, self.spins = ops, spins
        self.iso = 0.0 + 0j
        self.ist = np.zeros(8, dtype=complex)
        self.irr = np.zeros(8, dtype=complex)


def hamiltonian(
    spin_system: SpinSystem, operator_type: str = "comm", aniso: bool = False
) -> Matrix | tuple[Matrix, list]:
    """Isotropic Hamiltonian in rad/s, and optionally its anisotropic irreducible components.

    Interaction strengths are those set by :func:`spynach.assume`. In
    ``sphten-liouv`` the result is the superoperator of ``operator_type``
    (``'left'``, ``'right'``, ``'comm'``, ``'acomm'``); in ``zeeman-hilb``
    it is the Hamiltonian matrix and ``operator_type`` is ignored.

    With ``aniso=True`` the return value is ``(H, Q)``, where ``Q[r-1][k][m]``
    for ranks ``r = 1, 2`` holds the 3x3 and 5x5 arrays of operators that
    :func:`orientation` contracts with Wigner matrices (Spinach's second
    output). Quadratic (quadrupolar, zero-field) couplings
    ``inter['coupling']['matrix'][n][n]`` only enter through ``Q``.

    Spinach authors (hamiltonian.m): Ilya Kuprov, Luke Edwards, Hannah Hogben, Dmitry Savostyanov.
    """
    if spin_system.assumptions is None:
        raise ValueError("assumption information is missing, run assume() before calling this function.")
    if spin_system.basis is None:
        raise ValueError("basis set information is missing, run basis() before calling this function.")
    tols = spin_system.tols
    terms: dict[tuple[tuple[str, ...], tuple[int, ...]], _Term] = {}

    def term(ops: Sequence[str], spins: Sequence[int]) -> _Term:
        key = (tuple(ops), tuple(spins))
        if key not in terms:
            terms[key] = _Term(*key)
        return terms[key]

    def add_coefficients(table: dict, spins: Sequence[int], irr: np.ndarray) -> None:
        for ops, coefficients in table.items():
            row = term(ops, spins)
            for index, value in coefficients.items():
                row.ist[index] += value
            row.irr = irr

    for n in range(spin_system.nspins):
        strength = spin_system.assumptions.zeeman[n]
        if strength not in _ZEEMAN_TERMS:
            raise ValueError(f"unknown strength specification for the Zeeman interaction of spin {n}")
        if strength != "ignore":
            zeeman_iso = np.trace(spin_system.zeeman_matrix[n]) / 3
            if strength == "secular":
                zeeman_iso -= spin_system.basefrqs[n]  # rotating frame: subtract the carrier frequency
            if abs(zeeman_iso) > tols.liouv_zero:
                term(["Lz"], [n]).iso += zeeman_iso
        if aniso:
            _, lam, phi = mat2sphten(spin_system.zeeman_matrix[n])
            if np.linalg.norm(lam) > tols.liouv_zero or np.linalg.norm(phi) > tols.liouv_zero:
                add_coefficients(_ZEEMAN_TERMS[strength], [n], np.concatenate([lam, phi]))
            quad = spin_system.coupling_matrix[n][n]
            if quad is not None and np.linalg.norm(quad, 2) > 2 * np.pi * tols.inter_cutoff:
                iso_quad, lam_quad, phi_quad = mat2sphten(quad)
                if abs(iso_quad) > 1e-6 * np.linalg.norm(phi_quad) or np.linalg.norm(lam_quad) > 1e-6 * np.linalg.norm(
                    phi_quad
                ):
                    raise ValueError("quadratic couplings must be traceless and symmetric.")
                strength_quad = spin_system.assumptions.coupling[n][n]
                if strength_quad not in _QUAD_TERMS:
                    raise ValueError(f"unknown strength specification for the quadratic coupling of spin {n}")
                add_coefficients(_QUAD_TERMS[strength_quad], [n], np.concatenate([np.zeros(3), phi_quad]))

    for n, k in _coupled_pairs(spin_system):
        strength = spin_system.assumptions.coupling[n][k]
        if strength not in _PAIR_TERMS:
            raise ValueError(f"unknown strength specification for the bilinear coupling between spins {n} and {k}")
        coupling_iso = np.trace(spin_system.coupling_matrix[n][k]) / 3
        if abs(coupling_iso) > 2 * np.pi * tols.inter_cutoff:
            if strength in ("strong", "secular"):
                term(["Lz", "Lz"], [n, k]).iso += coupling_iso
                term(["L+", "L-"], [n, k]).iso += coupling_iso / 2
                term(["L-", "L+"], [n, k]).iso += coupling_iso / 2
            elif strength in ("z*", "*z", "zz"):
                term(["Lz", "Lz"], [n, k]).iso += coupling_iso
        if aniso:
            _, lam, phi = mat2sphten(spin_system.coupling_matrix[n][k])
            if np.linalg.norm(lam) > tols.liouv_zero or np.linalg.norm(phi) > tols.liouv_zero:
                add_coefficients(_PAIR_TERMS[strength], [n, k], np.concatenate([lam, phi]))

    # Drop rows that contribute to neither part, then build every operator once
    rows = [
        t
        for t in terms.values()
        if abs(t.iso) >= tols.liouv_zero
        or (np.abs(t.irr).sum() >= tols.liouv_zero and np.abs(t.ist).sum() >= tols.liouv_zero)
    ]
    liouville = spin_system.basis.formalism == "sphten-liouv"
    dim = spin_system.basis.nstates if liouville else int(np.prod(spin_system.mults))
    # Every operator once, as COO triplets; sums of terms are then one sparse construction each,
    # which sums duplicate entries as MATLAB's sparse() does
    opers = [_triplets(spin_system, t, operator_type, liouville) for t in rows]

    def assemble(coefficients: np.ndarray) -> Matrix:
        parts = [(c * v, r, k) for c, (r, k, v) in zip(coefficients, opers) if abs(c) > tols.liouv_zero]
        if not parts:
            return clean_up(tols, sps.csr_array((dim, dim), dtype=complex), tols.liouv_zero)
        vals = np.concatenate([x[0] for x in parts])
        rws = np.concatenate([x[1] for x in parts])
        cls = np.concatenate([x[2] for x in parts])
        return clean_up(tols, sps.csr_array(sps.coo_array((vals, (rws, cls)), shape=(dim, dim))), tols.liouv_zero)

    H = assemble(np.array([t.iso for t in rows], dtype=complex))
    if not aniso:
        return H
    ist = np.array([t.ist for t in rows], dtype=complex).reshape(len(rows), 8)
    irr = np.array([t.irr for t in rows], dtype=complex).reshape(len(rows), 8)
    Q = []
    for rank, block in ((1, RANK1), (2, RANK2)):
        size = 2 * rank + 1
        Q.append([[assemble(ist[:, block][:, k] * irr[:, block][:, m]) for m in range(size)] for k in range(size)])
    return H, Q


def _triplets(
    spin_system: SpinSystem, t: _Term, operator_type: str, liouville: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """COO triplets of one term's operator, summing the opspecs of composite labels (Lx, Ly)."""
    if liouville:
        opspecs, coeffs = human2opspec(spin_system, list(t.ops), list(t.spins))
        rows, cols, vals = [], [], []
        for spec, coeff in zip(opspecs, coeffs):
            r, c, v = superop(spin_system, spec, operator_type)
            rows.append(r)
            cols.append(c)
            vals.append(coeff * v)
        return np.concatenate(rows), np.concatenate(cols), np.concatenate(vals)
    coo = sps.coo_array(operator(spin_system, list(t.ops), list(t.spins), operator_type))
    return coo.row, coo.col, coo.data


def orientation(Q: list, euler_angles: Sequence[float]) -> Matrix:
    """Anisotropic part of the Hamiltonian at the orientation of ZYZ Euler angles in rad (port of ``orientation.m``).

    Spinach authors (orientation.m): Ilya Kuprov.
    """
    euler_angles = np.asarray(euler_angles, dtype=float).reshape(-1)
    if euler_angles.size != 3:
        raise ValueError("euler_angles must be a three-element real vector.")
    H: Any = 0
    for rank, Q_rank in enumerate(Q, start=1):
        D = wigner(rank, *euler_angles)
        for k in range(2 * rank + 1):
            for m in range(2 * rank + 1):
                block = Q_rank[k][m]
                if block is not None and nnz(block) > 0:
                    H = H + D[k, m] * block
    if isinstance(H, int):
        dim = next(b.shape[0] for Q_rank in Q for row in Q_rank for b in row if b is not None)
        return sps.csr_array((dim, dim), dtype=complex)
    return as_matrix((H + H.conj().T) / 2)


def frqoffset(spin_system: SpinSystem, H: Matrix, parameters: dict[str, Any]) -> Matrix:
    """Add rotating-frame offsets ``parameters['offset']`` (Hz) on ``parameters['spins']`` (isotopes).

    Spinach authors (frqoffset.m): Ilya Kuprov.
    """
    spins = list(parameters["spins"])
    offsets = np.atleast_1d(np.asarray(parameters["offset"], dtype=float))
    if not spins:
        raise ValueError("parameters['spins'] cannot be empty.")
    if any(s not in spin_system.isotopes for s in spins):
        raise ValueError("parameters['spins'] refers to a spin that is not present in the system.")
    if offsets.size != len(spins):
        raise ValueError("parameters['offset'] must have the same number of elements as parameters['spins'].")
    unique: dict[str, float] = {}
    for isotope, offset in zip(spins, offsets):
        if unique.setdefault(isotope, offset) != offset:
            raise ValueError("offsets on channels referring to the same spin must be the same.")
    for isotope, offset in unique.items():
        if offset != 0:
            H = H + 2 * np.pi * offset * operator(spin_system, "Lz", isotope)
    return as_matrix(H)


def _coupled_pairs(spin_system: SpinSystem) -> list[tuple[int, int]]:
    """Spin pairs with a significant coupling tensor, excluding quadratic (n == k) terms."""
    cutoff = 2 * np.pi * spin_system.tols.inter_cutoff
    return [
        (n, k)
        for k in range(spin_system.nspins)
        for n in range(spin_system.nspins)
        if n != k
        and spin_system.coupling_matrix[n][k] is not None
        and np.linalg.norm(spin_system.coupling_matrix[n][k], 2) > cutoff
    ]
