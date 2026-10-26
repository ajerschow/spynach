"""Basis set selection (port of ``basis.m``).

Ported: the ``'zeeman-hilb'`` formalism (Hilbert space, Zeeman product basis,
no truncation) and the ``'sphten-liouv'`` formalism (Liouville space,
product basis of irreducible spherical tensors) with the ``'none'``,
``'IK-0'``, ``'IK-1'`` and ``'IK-2'`` state-space restrictions and the
``projections``, ``longitudinal``, ``zero_quantum`` and ``manual`` filters.
Permutation symmetry (``sym_group``), ``'zeeman-liouv'``, ``'IK-DNP'`` and
``'IK-SBS'`` are not ported yet.

In ``sphten-liouv`` the basis descriptor is an integer matrix with one row
per basis state and one column per spin, holding the linear index
``l*l + l - m`` of the single-spin spherical tensor on that spin (0 is the
unit operator). Basis states are orthonormal: row ``(i_0, ..., i_N)`` is the
operator ``kron(T_i0, ..., T_iN)`` divided by its Frobenius norm. Rows are
the lexicographically sorted set of unique states, so the unit state is
always first. Chemical subsystems (``spin_system.chem.parts``) never share a row:
the basis is a direct sum over substances with one common unit state.

Spinach authors, by original file (headers of Spinach 2.13; see docs/spinach-credits.md):
  basis.m: Ilya Kuprov
  dfpt.m: Ilya Kuprov, Stefan Stoll
  prune_subgraphs.m: Ilya Kuprov
"""

from __future__ import annotations

from dataclasses import replace
from itertools import combinations
from typing import Any

import numpy as np

from spynach.operators import lin2lm
from spynach.system import Basis, SpinSystem

BAS_FIELDS = {
    "formalism",
    "approximation",
    "connectivity",
    "inter_level",
    "prox_level",
    "projections",
    "longitudinal",
    "zero_quantum",
    "manual",
    "sym_group",
    "sym_spins",
}
RETIRED = {
    "level": "bas['level'] has been renamed bas['inter_level'].",
    "space_level": "bas['space_level'] has been renamed bas['prox_level'].",
    "longitudinals": "bas['longitudinals'] has been renamed bas['longitudinal'], with one list per chemical substance.",
}
APPROXIMATIONS = {"none", "IK-0", "IK-1", "IK-2", "IK-DNP", "IK-SBS"}


def basis(spin_system: SpinSystem, bas: dict[str, Any]) -> SpinSystem:
    """Return a copy of ``spin_system`` with the basis described by ``bas``.

    ``bas['formalism']`` is ``'zeeman-hilb'`` (with ``approximation='none'``)
    or ``'sphten-liouv'``. For ``'sphten-liouv'``, ``bas['approximation']`` is
    one of

    * ``'none'``: complete basis of every chemical substance;
    * ``'IK-0'``: all spin correlations up to order ``bas['inter_level']``;
    * ``'IK-1'``: correlations up to ``inter_level`` between spins connected
      through the coupling graph, and up to ``bas['prox_level']`` between
      spatially proximate spins;
    * ``'IK-2'``: every spin correlated with all of its directly coupled
      neighbours, plus proximity correlations up to ``prox_level``.

    ``IK-1`` and ``IK-2`` need ``bas['connectivity']``, ``'scalar_couplings'``
    (isotropic parts build the coupling graph) or ``'full_tensors'``. Without
    coordinates (not ported) no spins are proximate, so ``prox_level`` has
    no effect. Optional filters, one entry per chemical substance:
    ``bas['projections']`` (coherence orders to keep, the unit state is always
    kept), ``bas['longitudinal']`` (spins, as 0-based numbers or isotope
    names, that only carry longitudinal states), ``bas['zero_quantum']``
    (spins whose total coherence order must be zero). ``bas['manual']`` is a
    boolean matrix of extra subgraphs (rows) over all spins.

    Spinach authors (basis.m): Ilya Kuprov.
    """
    _grumble(spin_system, bas)
    formalism = bas["formalism"]
    if formalism == "zeeman-hilb":
        return replace(
            spin_system,
            basis=Basis(formalism=formalism, approximation="none", basis=_zeeman_levels(spin_system.mults)),
        )
    descriptor = _sphten_basis(spin_system, bas)
    ranks, projs = lin2lm(descriptor)
    hash_table, row_hash, sort_order, sorted_hash = row_hashes(descriptor, spin_system.mults)
    return replace(
        spin_system,
        basis=Basis(
            formalism=formalism,
            approximation=bas["approximation"],
            basis=descriptor,
            tot_proj=projs.sum(axis=1),
            tot_cord=np.count_nonzero(ranks, axis=1),
            hash_table=hash_table,
            row_hash=row_hash,
            sort_order=sort_order,
            sorted_hash=sorted_hash,
        ),
    )


def row_hashes(descriptor: np.ndarray, mults: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Additive 64-bit hashes of the basis rows, for finding a row from a modified copy of another.

    ``hash_table[spin, index]`` is a fixed pseudo-random key; the hash of a
    row is the sum (modulo 2**64) of the keys of its entries, so changing
    the tensor indices on a few spins changes the hash by a known amount.
    Collisions between distinct rows have probability of order
    ``nstates**2 / 2**64`` and are checked for.
    """
    nspins = descriptor.shape[1]
    width = int(np.max(mults)) ** 2
    rng = np.random.default_rng(0x5EED_5EED)
    hash_table = rng.integers(0, 2**63 - 1, size=(nspins, width), dtype=np.int64).astype(np.uint64) * np.uint64(
        2
    ) + np.uint64(1)
    hash_table[:, 0] = 0  # the unit operator contributes nothing, so the all-unit row hashes to zero
    row_hash = np.zeros(descriptor.shape[0], dtype=np.uint64)
    for n in range(nspins):
        row_hash += hash_table[n, descriptor[:, n]]
    sort_order = np.argsort(row_hash, kind="stable")
    sorted_hash = row_hash[sort_order]
    if np.any(sorted_hash[1:] == sorted_hash[:-1]):
        raise RuntimeError("basis row hash collision; this should not happen, please report it.")
    return hash_table, row_hash, sort_order, sorted_hash


def _zeeman_levels(mults: np.ndarray) -> np.ndarray:
    """Level number of every spin in every Zeeman product state; spin 0 varies slowest."""
    grids = np.meshgrid(*(np.arange(1, m + 1) for m in mults), indexing="ij")
    return np.stack([g.ravel() for g in grids], axis=1)


def _sphten_basis(spin_system: SpinSystem, bas: dict[str, Any]) -> np.ndarray:
    """The sphten-liouv branch of basis.m: subgraphs per substance, their product bases, union."""
    nspins, mults, parts = spin_system.nspins, spin_system.mults, spin_system.chem.parts
    approximation = bas["approximation"]

    # State lists of individual spins, reduced to m = 0 where only longitudinal states are kept
    state_lists = [np.arange(m * m) for m in mults]
    for s, entries in enumerate(bas.get("longitudinal") or []):
        for entry in entries:
            for n in _spins_in_substance(spin_system, s, entry, "longitudinal"):
                state_lists[n] = state_lists[n][lin2lm(state_lists[n])[1] == 0]

    # Coupling and proximity graphs (every spin connected and proximate to itself)
    if approximation in ("IK-1", "IK-2"):
        conmatrix = _connectivity(spin_system, bas["connectivity"])
        proxmatrix = spin_system.proxmatrix | spin_system.proxmatrix.T | np.eye(nspins, dtype=bool)

    # Subgraphs (boolean rows over all spins) and the substance each belongs to
    subgraphs: list[np.ndarray] = []
    substance: list[int] = []
    manual = np.asarray(bas["manual"], dtype=bool) if "manual" in bas else np.zeros((0, nspins), dtype=bool)
    for s, part in enumerate(parts):
        spins = np.array(part)
        local = len(spins)
        if approximation == "none":
            coupling_sg = np.ones((1, local), dtype=bool)
            proximity_sg = np.zeros((0, local), dtype=bool)
        elif approximation == "IK-0":
            level = min(local, bas["inter_level"])
            coupling_sg = np.zeros((0, local), dtype=bool)
            if level:
                groups = np.array(list(combinations(range(local), level)))
                coupling_sg = np.zeros((len(groups), local), dtype=bool)
                coupling_sg[np.arange(len(groups))[:, None], groups] = True
            proximity_sg = np.zeros((0, local), dtype=bool)
        elif approximation == "IK-1":
            coupling_sg = dfpt(conmatrix[np.ix_(spins, spins)], min(local, bas["inter_level"]))
            proximity_sg = dfpt(proxmatrix[np.ix_(spins, spins)], min(local, bas["prox_level"]))
        else:  # IK-2
            coupling_sg = np.unique(conmatrix[np.ix_(spins, spins)], axis=0)
            proximity_sg = dfpt(proxmatrix[np.ix_(spins, spins)], min(local, bas["prox_level"]))
        inside = manual[:, spins]
        crosses = (inside.sum(axis=1) != manual.sum(axis=1)) & inside.any(axis=1)
        if crosses.any():
            raise ValueError(
                f"row {np.flatnonzero(crosses)[0]} of bas['manual'] crosses chemical substance boundaries."
            )
        manual_sg = inside[inside.sum(axis=1) == manual.sum(axis=1)]

        rows = np.vstack([coupling_sg, proximity_sg, manual_sg])
        rows[:, mults[spins] == 1] = False  # spin-zero particles carry no states
        rows = rows[rows.any(axis=1)]
        rows = prune_subgraphs(np.unique(rows, axis=0))
        for row in rows:
            full = np.zeros(nspins, dtype=bool)
            full[spins] = row
            subgraphs.append(full)
            substance.append(s)

    # Zero-quantum spin sets per substance
    zq_spins = np.zeros((len(parts), nspins), dtype=bool)
    for s, entries in enumerate(bas.get("zero_quantum") or []):
        for entry in entries:
            zq_spins[s, _spins_in_substance(spin_system, s, entry, "zero_quantum")] = True
    projections = bas.get("projections") or [None] * len(parts)

    # Product basis of every subgraph, filtered, embedded into the full spin index
    blocks = [np.zeros((1, nspins), dtype=np.int64)]
    for mask, s in zip(subgraphs, substance):
        involved = np.flatnonzero(mask)
        grids = np.meshgrid(*(state_lists[n] for n in involved), indexing="ij")
        local_spec = np.stack([g.ravel() for g in grids], axis=1)
        if projections[s] is not None and len(projections[s]):
            keep = np.isin(lin2lm(local_spec)[1].sum(axis=1), projections[s])
            keep[0] = True
            local_spec = local_spec[keep]
        zq = zq_spins[s, involved]
        if zq.any():
            local_spec = local_spec[lin2lm(local_spec[:, zq])[1].sum(axis=1) == 0]
        block = np.zeros((len(local_spec), nspins), dtype=np.int64)
        block[:, involved] = local_spec
        blocks.append(block)

    # Unique rows in lexicographic order; the all-zero unit state sorts first
    return np.unique(np.vstack(blocks), axis=0)


def dfpt(conmatrix: np.ndarray, max_size: int) -> np.ndarray:
    """Connected subgraphs of ``max_size`` spins in a coupling graph (port of ``dfpt.m``).

    Grows every subgraph by every neighbour, one spin at a time; isolated
    subgraphs are kept as they are. Returns a boolean matrix with one
    subgraph per row.

    Spinach authors (dfpt.m): Ilya Kuprov, Stefan Stoll.
    """
    nspins = conmatrix.shape[0]
    subgraphs = np.arange(nspins)[:, None]
    for _ in range(2, max_size + 1):
        grown = []
        for subgraph in subgraphs:
            neighbours = conmatrix[:, subgraph].any(axis=1)
            neighbours[subgraph] = False
            neighbours = np.flatnonzero(neighbours)
            if len(neighbours) == 0:
                grown.append(np.append(subgraph, subgraph[-1])[None, :])
            else:
                grown.append(np.column_stack([np.tile(subgraph, (len(neighbours), 1)), neighbours]))
        subgraphs = np.unique(np.sort(np.vstack(grown), axis=1), axis=0)
    out = np.zeros((len(subgraphs), nspins), dtype=bool)
    out[np.arange(len(subgraphs))[:, None], subgraphs] = True
    return out


def prune_subgraphs(subgraphs: np.ndarray) -> np.ndarray:
    """Drop subgraphs that are strict subsets of another subgraph (port of ``prune_subgraphs.m``).

    Spinach authors (prune_subgraphs.m): Ilya Kuprov.
    """
    if subgraphs.shape[0] < 2 or subgraphs.shape[1] < 2:
        return subgraphs
    counts = subgraphs.sum(axis=1)
    common = subgraphs.astype(np.int64) @ subgraphs.T.astype(np.int64)
    supersets = (common == counts[:, None]) & (counts[None, :] > counts[:, None])
    return subgraphs[~supersets.any(axis=1)]


def _connectivity(spin_system: SpinSystem, connectivity: str) -> np.ndarray:
    """Symmetric coupling graph with self-connections, from tensor norms above ``inter_cutoff``."""
    nspins = spin_system.nspins
    norm = (lambda t: abs(np.trace(t) / 3)) if connectivity == "scalar_couplings" else (lambda t: np.linalg.norm(t, 2))
    cutoff = 2 * np.pi * spin_system.tols.inter_cutoff
    conmatrix = np.eye(nspins, dtype=bool)
    for n in range(nspins):
        for k in range(nspins):
            tensor = spin_system.coupling_matrix[n][k]
            if tensor is not None and norm(tensor) > cutoff:
                conmatrix[n, k] = conmatrix[k, n] = True
    return conmatrix


def _spins_in_substance(spin_system: SpinSystem, s: int, entry: Any, field: str) -> list[int]:
    """Resolve a filter entry (isotope name or 0-based spin numbers) within substance ``s``."""
    part = spin_system.chem.parts[s]
    if isinstance(entry, str):
        spins = [n for n in part if spin_system.isotopes[n] == entry]
        if not spins:
            raise ValueError(f"no {entry} spins in substance {s}.")
        return spins
    spins = [int(n) for n in np.atleast_1d(entry)]
    if not all(n in part for n in spins):
        raise ValueError(f"bas['{field}'][{s}] refers to spins outside substance {s}.")
    return spins


def _grumble(spin_system: SpinSystem, bas: dict[str, Any]) -> None:
    """Validate ``bas`` as ``basis.m`` does, with Spinach's messages; raise on unported options."""
    formalism, approximation = _check_fields(bas)
    if formalism == "zeeman-hilb":
        _check_zeeman_hilb(spin_system, bas, approximation)
    else:
        _check_sphten_liouv(spin_system, bas, approximation)
        _check_filters(spin_system, bas)
    _check_option_scope(bas, approximation)


def _check_fields(bas: dict[str, Any]) -> tuple[str, str]:
    """Field names, formalism and approximation present and ported; returns both."""
    for name, message in RETIRED.items():
        if name in bas:
            raise ValueError(message)
    unported = set(bas) - BAS_FIELDS
    if unported:
        raise NotImplementedError(f"bas field(s) {sorted(unported)} not ported yet")
    if "formalism" not in bas:
        raise ValueError("basis formalism must be specified in bas['formalism'].")
    formalism = bas["formalism"]
    if formalism not in ("zeeman-hilb", "sphten-liouv"):
        raise NotImplementedError(f"formalism {formalism!r} not ported yet; 'zeeman-hilb' and 'sphten-liouv' are")
    if "sym_group" in bas or "sym_spins" in bas:
        raise NotImplementedError("permutation symmetry (bas['sym_group']) not ported yet.")
    if "approximation" not in bas:
        raise ValueError(f"approximation level must be specified in bas['approximation'] for {formalism} formalism.")
    return formalism, bas["approximation"]


def _check_zeeman_hilb(spin_system: SpinSystem, bas: dict[str, Any], approximation: str) -> None:
    if approximation != "none":
        raise ValueError("bas['approximation'] should be set to 'none' in zeeman-hilb formalism.")
    for name in ("projections", "longitudinal", "zero_quantum"):
        if name in bas:
            raise ValueError(f"bas['{name}'] option is only available for sphten-liouv formalism.")
    if spin_system.chem.nparts > 1:
        raise ValueError("chemical reaction modelling is only available for sphten-liouv formalism.")


def _check_sphten_liouv(spin_system: SpinSystem, bas: dict[str, Any], approximation: str) -> None:
    """Approximation level and the connectivity and tracing depths it needs."""
    nspins = spin_system.nspins
    if np.any(spin_system.mults > 16):
        raise ValueError("multiplicities above 16 are not supported by sphten-liouv formalism.")
    if approximation not in APPROXIMATIONS:
        raise ValueError("unrecognized approximation - see the basis preparation section of the manual.")
    if approximation in ("IK-DNP", "IK-SBS"):
        raise NotImplementedError(f"approximation {approximation!r} not ported yet.")
    if approximation in ("IK-1", "IK-2"):
        if "connectivity" not in bas:
            raise ValueError("connectivity type must be specified in bas['connectivity'].")
        if bas["connectivity"] not in ("scalar_couplings", "full_tensors"):
            raise ValueError("unknown connectivity type - see the basis preparation section of the manual.")
    if approximation in ("IK-0", "IK-1"):
        if "inter_level" not in bas:
            raise ValueError("connectivity tracing depth must be specified in bas['inter_level'].")
        _check_level(bas["inter_level"], nspins, "inter_level")
    if approximation in ("IK-1", "IK-2"):
        if "prox_level" not in bas:
            raise ValueError("proximity tracing depth must be specified in bas['prox_level'].")
        _check_level(bas["prox_level"], nspins, "prox_level")


def _check_filters(spin_system: SpinSystem, bas: dict[str, Any]) -> None:
    """``manual``, ``projections``, ``longitudinal`` and ``zero_quantum`` (sphten-liouv only)."""
    nspins, nparts = spin_system.nspins, spin_system.chem.nparts
    if "manual" in bas:
        manual = np.asarray(bas["manual"])
        if manual.ndim != 2 or manual.shape[1] != nspins:
            raise ValueError(
                "the number of columns in bas['manual'] must be equal to the number of spins in the system."
            )
    if "projections" in bas:
        if len(bas["projections"]) != nparts:
            raise ValueError("bas['projections'] must be a list with one element per chemical substance.")
        for entry in bas["projections"]:
            if entry is not None and any(int(p) != p for p in np.atleast_1d(entry)):
                raise ValueError("elements of bas['projections'] must be empty or lists of integers.")
    for name in ("longitudinal", "zero_quantum"):
        if name not in bas:
            continue
        if len(bas[name]) != nparts:
            raise ValueError(f"bas['{name}'] must be a list with one element per chemical substance.")
        for entries in bas[name]:
            for entry in entries:
                if isinstance(entry, str):
                    if entry not in spin_system.isotopes:
                        raise ValueError(f"bas['{name}'] refers to spins that are not present in the system.")
                elif any(int(n) != n or n < 0 or n >= nspins for n in np.atleast_1d(entry)):
                    raise ValueError(
                        f"numeric entries in bas['{name}'] must be spin numbers between 0 and {nspins - 1}."
                    )


def _check_option_scope(bas: dict[str, Any], approximation: str) -> None:
    """Options that only some approximation levels accept."""
    if "inter_level" in bas and approximation not in ("IK-0", "IK-1", "IK-DNP", "IK-SBS"):
        raise ValueError("bas['inter_level'] is only applicable to IK-0,1,2,DNP,SBS basis sets.")
    if "prox_level" in bas and approximation not in ("IK-1", "IK-2"):
        raise ValueError("bas['prox_level'] is only applicable to IK-1,2 basis sets.")
    if "connectivity" in bas and approximation not in ("IK-1", "IK-2", "IK-SBS"):
        raise ValueError("bas['connectivity'] is only applicable to IK-1,2,SBS basis sets.")


def _check_level(level: Any, nspins: int, name: str) -> None:
    if not isinstance(level, (int, np.integer)) or level < 1:
        raise ValueError(f"bas['{name}'] must be a positive integer.")
    if level > nspins:
        raise ValueError(f"bas['{name}'] cannot be greater than the number of spins in the system.")
