"""Spin states (port of ``state.m``).

In ``zeeman-hilb`` a state is the operator itself (a density matrix). In
``sphten-liouv`` it is a dense 1-D complex vector of coefficients in the
orthonormal basis of :mod:`spynach.basis`; its first element is the
coefficient of the unit state ``E / sqrt(dim)``.

Spinach authors, by original file (headers of Spinach 2.13; see docs/spinach-credits.md):
  state.m: Dmitry Savostyanov, Luke Edwards, Ilya Kuprov
  stateinfo.m: Ilya Kuprov, Konstantin Pervushin
  trajan.m: Gareth Charnock, Ilya Kuprov, Luke Edwards
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
import scipy.sparse as sps

from spynach.operators import human2opspec, lin2lm, operator, superop
from spynach.system import SpinSystem

log = logging.getLogger("spynach")

METHODS = ("exact", "cheap", "chem")


def state(
    spin_system: SpinSystem,
    states: str | Sequence[str],
    spins: str | int | Sequence[int],
    method: str = "exact",
) -> sps.csr_array | np.ndarray:
    """State from a human-readable description, same forms as :func:`spynach.operator`.

    ``zeeman-hilb``: the (unnormalised) operator, e.g. ``state(spin_system, 'L+', '1H')``
    is the sum of L+ over all protons; ``method`` is ignored.

    ``sphten-liouv``: a vector, by one of three methods as in Spinach:

    * ``'exact'`` (default): the operator applied to the unit state, i.e. the
      expansion of ``Op / sqrt(dim)`` in the orthonormal basis (a spin-1/2
      ``Lz`` has coefficient 1/2 on its ``T(1,0)`` state);
    * ``'cheap'``: the coefficient of each operator placed directly on its
      basis state (``Lz`` gives 1, ``L+`` gives -sqrt(2)); errors if the
      state is not in the basis;
    * ``'chem'``: ``'exact'`` with each term multiplied by the concentration
      of the chemical subsystem its spins belong to.

    Spinach authors (state.m): Dmitry Savostyanov, Luke Edwards, Ilya Kuprov.
    """
    if spin_system.basis is None:
        raise ValueError("basis set information is missing, run basis() before calling this function.")
    if method not in METHODS:
        raise ValueError(f"unknown state generation method; use one of {METHODS}.")
    formalism = spin_system.basis.formalism
    if formalism == "zeeman-hilb":
        return operator(spin_system, states, spins)
    if formalism != "sphten-liouv":
        raise NotImplementedError(f"formalism {formalism!r} not ported yet.")

    basis = spin_system.basis.basis
    nstates = basis.shape[0]
    if method == "exact":
        return operator(spin_system, states, spins, "left")[:, [0]].toarray().ravel()

    opspecs, coeffs = human2opspec(spin_system, states, spins)
    rho = np.zeros(nstates, dtype=complex)
    if method == "cheap":
        for spec, coeff in zip(opspecs, coeffs):
            matches = np.flatnonzero((basis == spec).all(axis=1))
            if matches.size == 0:
                raise ValueError("the requested state is not present in the basis.")
            rho[matches[0]] += coeff
        return rho
    for spec, coeff in zip(opspecs, coeffs):
        active = set(np.flatnonzero(spec))
        species = [s for s, part in enumerate(spin_system.chem.parts) if active <= set(part)]
        if len(species) != 1:
            raise ValueError("the spin state requested crosses chemical species boundaries.")
        rows, cols, vals = superop(spin_system, spec, "left")
        on_unit = cols == 0
        np.add.at(rho, rows[on_unit], coeff * spin_system.chem.concs[species[0]] * vals[on_unit])
    return rho


TRAJAN_PROPERTIES = ("correlation_order", "coherence_order", "total_each_spin", "local_each_spin")


def trajan(spin_system: SpinSystem, traj: np.ndarray, property: str) -> tuple[np.ndarray, list[str]]:
    """Trajectory analysis (the numbers behind Spinach's ``trajan``): amplitudes of state classes along a trajectory.

    ``traj`` is a ``(nstates, npoints)`` array of ``sphten-liouv`` state
    vectors. The unit state is projected out, and for every class the
    2-norm of the trajectory restricted to that class is returned, one
    row per class, with the legend text of each row:

    * ``'correlation_order'``: number of correlated spins, 1-spin, 2-spin, ...;
    * ``'coherence_order'``: total coherence order of the state;
    * ``'total_each_spin'``: every state touching spin ``n``;
    * ``'local_each_spin'``: states on spin ``n`` alone.

    Spinach's ``'level_populations'`` (Zeeman-basis populations) is not ported.
    :func:`spynach.plotting.plot_trajan` draws the result.

    Spinach authors (trajan.m): Gareth Charnock, Ilya Kuprov, Luke Edwards.
    """
    if spin_system.basis is None or spin_system.basis.formalism != "sphten-liouv":
        raise NotImplementedError("trajectory analysis is only available for the sphten-liouv formalism.")
    if property == "level_populations":
        raise NotImplementedError("the 'level_populations' property of trajan is not ported yet.")
    if property not in TRAJAN_PROPERTIES:
        raise ValueError("unknown property.")
    traj = np.asarray(traj.toarray() if sps.issparse(traj) else traj)
    if traj.ndim == 1:
        traj = traj[:, None]
    basis = spin_system.basis.basis
    if traj.ndim != 2 or traj.shape[0] != basis.shape[0]:
        raise ValueError("trajectory must be a (nstates, npoints) array.")
    traj = traj.copy()
    traj[0, :] = 0  # the unit state is projected out
    if property == "correlation_order":
        orders = (basis != 0).sum(axis=1)
        classes = [o for o in np.unique(orders) if o != 0]
        masks = [orders == o for o in classes]
        labels = [f"{o}-spin" for o in classes]
    elif property == "coherence_order":
        _, projs = lin2lm(basis)
        orders = projs.sum(axis=1)
        classes = list(np.unique(orders))
        masks = [orders == o for o in classes]
        labels = [f"{o:+d}" if o else "0" for o in classes]
    else:
        nspins = spin_system.nspins
        on_spin = basis != 0
        if property == "total_each_spin":
            masks = [on_spin[:, n] for n in range(nspins)]
        else:
            masks = [on_spin[:, n] & (basis.sum(axis=1) == basis[:, n]) for n in range(nspins)]
        labels = [spin_system.labels[n] or f"{spin_system.isotopes[n]} ({n})" for n in range(nspins)]
    result = np.stack([np.sqrt((np.abs(traj[mask, :]) ** 2).sum(axis=0)) for mask in masks])
    return result, labels


def stateinfo(
    spin_system: SpinSystem, rho: np.ndarray, npops: int
) -> list[tuple[tuple[tuple[int, int], ...], complex, int]]:
    """The ``npops`` largest coefficients of a ``sphten-liouv`` state vector (port of ``stateinfo.m``).

    Returns ``[(state, coefficient, index), ...]`` in descending magnitude,
    where ``state`` holds ``(l, m)`` of every spin (``(0, 0)`` for a spin
    that is not involved) and ``index`` the 0-based basis row, and logs
    them as Spinach reports them.

    Spinach authors (stateinfo.m): Ilya Kuprov, Konstantin Pervushin.
    """
    if spin_system.basis is None or spin_system.basis.formalism != "sphten-liouv":
        raise NotImplementedError("state analysis is only available for sphten-liouv formalism.")
    rho = np.asarray(rho.toarray() if sps.issparse(rho) else rho).reshape(-1)
    if not isinstance(npops, (int, np.integer)) or npops < 1:
        raise ValueError("npops parameter must be a positive integer.")
    if npops > rho.size:
        raise ValueError("number of populations exceeds state space dimension.")
    log.info("state vector 2-norm: %g", np.linalg.norm(rho))
    order = np.argsort(-np.abs(rho), kind="stable")[:npops]
    ranks, projs = lin2lm(spin_system.basis.basis[order])
    log.info("%d most populated basis states (state, coeff, number)", npops)
    out = []
    for row, index in enumerate(order):
        lm = tuple((int(ranks[row, k]), int(projs[row, k])) for k in range(spin_system.nspins))
        text = "".join("  ....  " if l == 0 else f" ({l},{m:+d}) " for l, m in lm)
        value = rho[index]
        coefficient = f"{value.real:+5.3e}" if value.imag == 0 else f"{value.real:+5.3e}{value.imag:+5.3e}i"
        log.info("%s    %s    %d", text, coefficient, index)
        out.append((lm, complex(rho[index]), int(index)))
    return out
