"""Trajectory-level state space reduction (ports of ``reduce.m``, ``zte.m``, ``path_trace.m``, ``binpack.m``).

These reductions are exact for the trajectory at hand: zero track
elimination drops states that a short Krylov-type trial propagation never
populates (Kuprov 2008), and path tracing splits the generator into
non-interacting subspaces (Hogben et al. 2010) and keeps the populated
ones. Permutation symmetry is not ported yet.

Projectors are sparse ``(nstates, k)`` selection matrices, as in Spinach:
the reduced generator is ``P.conj().T @ L @ P`` and the reduced state is
``P.conj().T @ rho``.

Spinach authors, by original file (headers of Spinach 2.13; see docs/spinach-credits.md):
  binpack.m: Ilya Kuprov
  path_trace.m: Ilya Kuprov, Matthew Krzystyniak
  reduce.m: Ilya Kuprov, Matthew Krzystyniak, Luke Edwards
  zte.m: Ilya Kuprov
"""

from __future__ import annotations

import logging

import numpy as np
import scipy.sparse as sps
from scipy.sparse.csgraph import connected_components

from spynach.backend import get_backend
from spynach.propagation import step
from spynach.system import SpinSystem
from spynach.utilities import Matrix

log = logging.getLogger("spynach")


def reduce(spin_system: SpinSystem, L: Matrix, rho: np.ndarray) -> list[sps.csc_array]:
    """Projectors onto the subspaces that the trajectory of ``rho`` under ``L`` can reach.

    Liouville-space formalisms only: zero track elimination followed by path
    tracing. A state stack ``(nstates, k)`` is represented by the mean of
    its absolute values. ``'trajlevel'`` in ``sys['disable']`` switches the
    reduction off (one projector, the identity).

    Spinach authors (reduce.m): Ilya Kuprov, Matthew Krzystyniak, Luke Edwards.
    """
    if L.ndim != 2 or L.shape[0] != L.shape[1]:
        raise ValueError("L must be a square matrix.")
    if spin_system.basis is None or spin_system.basis.formalism != "sphten-liouv":
        raise NotImplementedError("reduce() is only ported for the sphten-liouv formalism.")
    dim = L.shape[0]
    if "trajlevel" in spin_system.disable:
        log.warning("trajectory level state space restriction disabled by user.")
        return [sps.eye_array(dim, format="csc", dtype=complex)]
    rho = np.asarray(rho)
    if rho.ndim == 2 and rho.shape[1] > 1:
        rho = np.mean(np.abs(rho), axis=1)
    rho = rho.reshape(-1)
    projector = zte(spin_system, L, rho)
    projectors = []
    L_red = projector.conj().T @ L @ projector
    rho_red = projector.conj().T @ rho
    for part in path_trace(spin_system, L_red, rho_red):
        projectors.append(sps.csc_array(projector @ part))
    return projectors


def zte(spin_system: SpinSystem, L: Matrix, rho: np.ndarray) -> sps.csc_array:
    """Zero track elimination projector (port of ``zte.m``).

    Runs up to ``tols.zte_nsteps`` steps of length ``1/||L||_1`` starting
    from ``rho``, stopping when the set of states with amplitude above
    ``tols.zte_tol`` stops growing, and drops the states never reached.
    Skipped (identity returned) when disabled, when ``rho`` is denser than
    ``tols.zte_maxden``, or when its 1-norm is below ``tols.zte_tol``.

    Spinach authors (zte.m): Ilya Kuprov.
    """
    rho = np.asarray(rho).reshape(-1)
    dim = L.shape[0]
    if L.shape[1] != rho.shape[0]:
        raise ValueError("Liouvillian and state vector dimensions must be consistent.")
    tols = spin_system.tols
    identity = sps.eye_array(dim, format="csc", dtype=complex)
    if "zte" in spin_system.disable:
        log.warning("zero track elimination disabled, basis left unchanged.")
        return identity
    if np.count_nonzero(rho) / rho.size > tols.zte_maxden:
        log.info("too few zeros in the state vector, basis left unchanged.")
        return identity
    if np.sum(np.abs(rho)) < tols.zte_tol:
        log.info("state vector norm below drop tolerance, basis left unchanged.")
        return identity
    norm = get_backend().norm1(L)
    timestep = 1 / norm if norm > 0 else 1.0
    # GPU: the trial propagation is a handful of step() calls; L stays wherever step() puts it
    trajectory = np.zeros((dim, tols.zte_nsteps), dtype=complex)
    trajectory[:, 0] = rho
    reached = np.abs(rho) > tols.zte_tol
    for n in range(1, tols.zte_nsteps):
        trajectory[:, n] = step(spin_system, L, trajectory[:, n - 1], timestep)
        previous = reached.sum()
        reached |= np.abs(trajectory[:, n]) > tols.zte_tol
        if reached.sum() == previous:
            break
    keep = np.flatnonzero(reached)
    log.debug("zero track elimination: state space dimension reduced from %d to %d", dim, keep.size)
    return _selector(dim, keep)


def path_trace(spin_system: SpinSystem, L: Matrix, rho: np.ndarray | None) -> list[sps.csc_array]:
    """Projectors onto the populated non-interacting subspaces of ``L`` (port of ``path_trace.m``).

    Connected components of the graph of elements above ``tols.liouv_zero``
    are kept when ``rho`` has 1-norm above ``tols.subs_drop`` in them, and
    small ones are merged into batches of at most ``tols.merge_dim`` states
    (``'merge'`` in ``sys['disable']`` keeps them separate). Skipped below
    ``tols.merge_dim`` states or when ``'pt'`` is disabled.

    Spinach authors (path_trace.m): Ilya Kuprov, Matthew Krzystyniak.
    """
    tols = spin_system.tols
    dim = L.shape[0]
    if "pt" in spin_system.disable:
        log.warning("path tracing disabled by the user.")
        return [sps.eye_array(dim, format="csc", dtype=complex)]
    if dim < tols.merge_dim:
        return [sps.eye_array(dim, format="csc", dtype=complex)]
    graph = sps.csr_array((abs(sps.csr_array(L)) > tols.liouv_zero).astype(np.int8))
    graph = graph + graph.T + sps.eye_array(dim, format="csr", dtype=np.int8)
    n_components, labels = connected_components(graph, directed=False)
    subspaces = []
    for n in range(n_components):
        members = np.flatnonzero(labels == n)
        if rho is None or np.sum(np.abs(np.asarray(rho).reshape(-1)[members])) > tols.subs_drop:
            subspaces.append(members)
    log.debug("path tracing: %d non-interacting subspaces, %d populated", n_components, len(subspaces))
    if "merge" not in spin_system.disable:
        sizes = [members.size for members in subspaces]
        subspaces = [np.concatenate([subspaces[i] for i in batch]) for batch in binpack(sizes, tols.merge_dim)]
    return [_selector(dim, members) for members in subspaces]


def binpack(sizes: list[int], bin_size: int) -> list[list[int]]:
    """First-fit batching of ``sizes`` into bins of at most ``bin_size`` (port of ``binpack.m``).

    Items larger than a bin get a bin of their own; the rest are packed in
    order, each bin taking the longest prefix of the remaining items that
    fits.

    Spinach authors (binpack.m): Ilya Kuprov.
    """
    index = list(range(len(sizes)))
    bins = [[i] for i in index if sizes[i] > bin_size]
    remaining = [i for i in index if sizes[i] <= bin_size]
    while remaining:
        total, batch = 0, []
        for i in remaining:
            total += sizes[i]
            if total > bin_size:
                break
            batch.append(i)
        bins.append(batch)
        remaining = [i for i in remaining if i not in batch]
    return bins


def _selector(dim: int, columns: np.ndarray) -> sps.csc_array:
    """Columns ``columns`` of the ``dim x dim`` identity."""
    return sps.csc_array(
        (np.ones(columns.size, dtype=complex), (columns, np.arange(columns.size))), shape=(dim, columns.size)
    )
