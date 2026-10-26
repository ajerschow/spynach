"""Chemical kinetics superoperator (port of the exchange branch of ``kinetics.m``).

Ported: first-order exchange between chemical subsystems (``inter['chem']
['rates']``). Magnetisation flux and radical-pair kinetics are refused by
:func:`spynach.create`. The superoperator is dissipative and enters the
Liouvillian as ``L = H + 1j * R + 1j * K``, as in Spinach.

Spinach authors, by original file (headers of Spinach 2.13; see docs/spinach-credits.md):
  equilibrate.m: Ilya Kuprov
  kinetics.m: Ilya Kuprov, Luke Edwards, Hannah Hogben
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sps

from spynach.system import SpinSystem


def kinetics(spin_system: SpinSystem) -> sps.csr_array:
    """Exchange superoperator ``K`` in the ``sphten-liouv`` basis, in Hz.

    For every non-zero rate ``rates[i, j]`` (flow from part ``j`` into part
    ``i``) the states living on part ``j`` are mapped onto the identical
    states of part ``i``: ``K[states_i, states_j] += rates[i, j]``. The
    diagonal rates give the drain on each part's own states. The unit
    state is never touched. With no rates the result is the zero matrix.

    Spinach authors (kinetics.m): Ilya Kuprov, Luke Edwards, Hannah Hogben.
    """
    if spin_system.basis is None:
        raise ValueError("basis set information is missing, run basis() before calling this function.")
    if spin_system.basis.formalism != "sphten-liouv":
        raise NotImplementedError("chemical reaction treatment is only available for sphten-liouv formalism.")
    basis = spin_system.basis.basis
    nstates = basis.shape[0]
    rates = spin_system.chem.rates
    if rates is None:
        return sps.csr_array((nstates, nstates), dtype=complex)
    rows, cols, vals = [], [], []
    for i, j in zip(*np.nonzero(rates)):
        part_i, part_j = list(spin_system.chem.parts[i]), list(spin_system.chem.parts[j])
        states_i = np.flatnonzero(basis[:, part_i].any(axis=1))
        states_j = np.flatnonzero(basis[:, part_j].any(axis=1))
        if not np.array_equal(basis[np.ix_(states_i, part_i)], basis[np.ix_(states_j, part_j)]):
            raise ValueError(
                "spin systems on either side of the reaction arrow have different topologies or basis sets."
            )
        rows.append(states_i)
        cols.append(states_j)
        vals.append(np.full(len(states_i), rates[i, j], dtype=complex))
    return sps.csr_array(
        sps.coo_array((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(nstates, nstates))
    )


def equilibrate(K: np.ndarray, c0: np.ndarray) -> np.ndarray:
    """Equilibrium concentrations of the first-order kinetics ``dc/dt = K c`` (port of ``equilibrate.m``).

    ``K`` must conserve matter (zero column sums). Each connected group of
    reactions is solved separately: the total concentration of the group is
    kept and ``K c = 0`` is imposed. The result keeps the shape of ``c0``.

    Spinach authors (equilibrate.m): Ilya Kuprov.
    """
    from scipy.sparse.csgraph import connected_components

    K = np.asarray(K, dtype=float)
    if K.ndim != 2 or K.shape[0] != K.shape[1]:
        raise ValueError("K must be a real square matrix.")
    if not np.all(np.abs(K.sum(axis=0)) < 10 * np.finfo(float).eps):
        raise ValueError("K violates conservation of matter: column sums must be zero.")
    c0 = np.asarray(c0, dtype=float)
    if c0.ndim != 1 or np.any(c0 < 0):
        raise ValueError("c0 must be a vector of non-negative real numbers.")
    if c0.size != K.shape[1]:
        raise ValueError("dimension mismatch between K and c0")
    if np.linalg.norm(c0) == 0:
        return c0.copy()
    n_groups, labels = connected_components((K != 0) | (K != 0).T, directed=False)
    if n_groups > 1:
        c = np.zeros_like(c0)
        for group in range(n_groups):
            members = np.flatnonzero(labels == group)
            c[members] = equilibrate(K[np.ix_(members, members)], c0[members])
        return c
    rate_scale = np.abs(K).max()
    if rate_scale > 0:
        K = K / rate_scale
    A = np.vstack([np.ones((1, K.shape[1])), K])
    b = np.concatenate([[c0.sum()], np.zeros(K.shape[0])])
    if np.linalg.cond(A) > 1 / np.sqrt(np.finfo(float).eps):
        raise ValueError("kinetic matrix does not constrain the equilibrium.")
    return np.linalg.lstsq(A, b, rcond=None)[0]
