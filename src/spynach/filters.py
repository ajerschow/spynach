"""Analytical state filters (ports of ``correlation.m``, ``coherence.m``, ``homospoil.m`` and ``decouple.m``).

Ported for ``sphten-liouv``, where correlation and coherence orders are
labels of the basis states and filtering means zeroing coefficients.
The Hilbert-space branches (which need projection superoperators) are not
ported yet.

State arrays may hold several vectors or a Fokker-Planck stack: the last
axis must be the spin state space (``(..., nstates)``), which is spynach's
layout for composite spaces with the spin index varying fastest.

Spinach authors, by original file (headers of Spinach 2.13; see docs/spinach-credits.md):
  coherence.m: Ilya Kuprov, Luke Edwards
  correlation.m: Ilya Kuprov, Luke Edwards
  decouple.m: Ilya Kuprov
  homospoil.m: Ilya Kuprov
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import scipy.sparse as sps

from spynach.operators import lin2lm
from spynach.system import SpinSystem
from spynach.utilities import clean_up


def correlation(
    spin_system: SpinSystem, rho: np.ndarray, orders: int | Sequence[int], spins: str | Sequence[int] = "all"
) -> np.ndarray:
    """Keep only the states whose spin correlation order over ``spins`` is in ``orders``.

    ``spins`` is ``'all'``, an isotope name, or a list of 0-based spin
    numbers; ``orders`` are non-negative integers.

    Spinach authors (correlation.m): Ilya Kuprov, Luke Edwards.
    """
    _check(spin_system, rho)
    orders = np.atleast_1d(np.asarray(orders))
    if orders.ndim != 1 or np.any(orders < 0) or np.any(orders != orders.astype(int)):
        raise ValueError("orders must be a vector of non-negative integers.")
    selected = _spins(spin_system, spins)
    present = np.count_nonzero(spin_system.basis.basis[:, selected], axis=1)
    return _apply(rho, np.isin(present, orders))


def coherence(spin_system: SpinSystem, rho: np.ndarray, spec: Sequence[tuple[Any, Any]]) -> np.ndarray:
    """Keep only the states whose coherence orders match every entry of ``spec``.

    ``spec`` is a list of ``(spins, orders)`` pairs, e.g.
    ``[("13C", [1, -1]), ("1H", -1)]``: the total coherence order over the
    named spins (``'all'``, ``'electrons'``, ``'nuclei'``, an isotope, or
    0-based spin numbers) must be one of ``orders``.

    Spinach authors (coherence.m): Ilya Kuprov, Luke Edwards.
    """
    _check(spin_system, rho)
    _, projections = lin2lm(spin_system.basis.basis)
    mask = np.ones(spin_system.basis.nstates, dtype=bool)
    for entry in spec:
        if len(entry) != 2:
            raise ValueError("each element of spec must be a (spins, orders) pair.")
        selected = _spins(spin_system, entry[0])
        orders = np.atleast_1d(np.asarray(entry[1]))
        if np.any(orders != orders.astype(int)):
            raise ValueError("coherence orders in spec must be integers.")
        mask &= np.isin(projections[:, selected].sum(axis=1), orders)
    return _apply(rho, mask)


def _check(spin_system: SpinSystem, rho: np.ndarray) -> None:
    if spin_system.basis is None:
        raise ValueError("basis set information is missing, run basis() before calling this function.")
    if spin_system.basis.formalism != "sphten-liouv":
        raise NotImplementedError("analytical order selection is only ported for the sphten-liouv formalism.")
    if np.shape(rho)[-1] != spin_system.basis.nstates:
        raise ValueError("the last dimension of rho must be the dimension of the spin state space.")


def _apply(rho: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = np.array(rho, copy=True)
    out[..., ~mask] = 0
    return out


def _spins(spin_system: SpinSystem, spins: str | Sequence[int]) -> list[int]:
    isotopes = spin_system.isotopes
    if isinstance(spins, str):
        if spins == "all":
            return list(range(spin_system.nspins))
        if spins == "electrons":
            return [n for n, iso in enumerate(isotopes) if iso.startswith("E")]
        if spins == "nuclei":
            return [n for n, iso in enumerate(isotopes) if not iso.startswith("E")]
        if spins not in isotopes:
            raise ValueError(
                "spins must be 'all', 'electrons', 'nuclei', an isotope present in the system, or spin numbers."
            )
        return [n for n, iso in enumerate(isotopes) if iso == spins]
    numbers = [int(n) for n in np.atleast_1d(spins)]
    if any(n < 0 or n >= spin_system.nspins for n in numbers):
        raise ValueError("spin numbers must be within the system bounds.")
    return numbers


def homospoil(spin_system: SpinSystem, rho: np.ndarray, zqc_flag: str) -> np.ndarray:
    """Homospoil gradient: destroy all coherences (port of ``homospoil.m``).

    ``zqc_flag='keep'`` zeroes every state with a non-zero Zeeman frequency
    (``sum(basefrq * m)``), keeping zero-quantum coherences; ``'destroy'``
    zeroes every state with a non-zero coherence order on any spin,
    keeping only longitudinal states and the unit state.

    Spinach authors (homospoil.m): Ilya Kuprov.
    """
    _check(spin_system, rho)
    _, projections = lin2lm(spin_system.basis.basis)
    if zqc_flag == "keep":
        mask = np.abs(projections @ spin_system.basefrqs) <= 1e-6
    elif zqc_flag == "destroy":
        mask = np.abs(projections).sum(axis=1) == 0
    else:
        raise ValueError("the available values for zqc_flag are 'keep' and 'destroy'.")
    return _apply(rho, mask)


def decouple(
    spin_system: SpinSystem, L: Any, rho: np.ndarray | None, spins: Sequence[Any]
) -> tuple[Any, np.ndarray | None]:
    """Analytical decoupling: freeze and depopulate ``spins`` (port of ``decouple.m``, ``sphten-liouv``).

    ``spins`` is a list of isotope names or 0-based spin numbers (empty:
    nothing happens). Every basis state that involves one of them is
    removed from the dynamics: its rows and columns of the generator
    ``L`` and its entries of the state ``rho`` are zeroed. Both ``L`` and
    ``rho`` may span a Fokker-Planck space (``spc_dim`` copies of the spin
    state space, spin index fastest).

    Spinach authors (decouple.m): Ilya Kuprov.
    """
    spins = list(spins)
    if not spins:
        return L, rho
    if spin_system.basis is None or spin_system.basis.formalism != "sphten-liouv":
        raise NotImplementedError("analytical decoupling is only ported for the sphten-liouv formalism.")
    if all(isinstance(s, str) for s in spins):
        if any(s not in spin_system.isotopes for s in spins):
            raise ValueError("the system does not contain the spins specified.")
        mask = np.array([iso in spins for iso in spin_system.isotopes])
    else:
        numbers = [int(s) for s in spins]
        if any(n < 0 or n >= spin_system.nspins for n in numbers):
            raise ValueError("spin numbers must be within the system bounds.")
        mask = np.zeros(spin_system.nspins, dtype=bool)
        mask[numbers] = True
    spn_dim = spin_system.basis.nstates
    zero_mask = spin_system.basis.basis[:, mask].any(axis=1)
    if L is not None:
        spc_dim = L.shape[0] // spn_dim
        keep = np.tile(~zero_mask, spc_dim).astype(float)
        projector = sps.diags_array(keep, format="csr")
        L = projector @ L @ projector
        L = clean_up(
            spin_system.tols, sps.csr_array(L) if sps.issparse(L) else np.asarray(L), spin_system.tols.liouv_zero
        )
    if rho is not None:
        rho = np.array(rho, dtype=complex)
        spc_dim = rho.shape[0] // spn_dim
        rho[np.tile(zero_mask, spc_dim)] = 0
    return L, rho
