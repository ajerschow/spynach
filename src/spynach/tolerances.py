"""Numerical tolerances and thresholds (port of the defaults in ``tolerances.m``).

Spinach keeps these in ``spin_system.tols``. Only the "safe default" set is
ported; the ``'paranoia'`` and ``'cowboy'`` presets are not. Individual
values can be overridden through ``sys['tols']`` in :func:`spynach.create`.

Spinach authors, by original file (headers of Spinach 2.13; see docs/spinach-credits.md):
  tolerances.m: Ilya Kuprov, Hannah Hogben
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Any


@dataclass(frozen=True)
class Tolerances:
    inter_cutoff: float = 1e-10  # interaction norm below which it is dropped (Hz)
    liouv_zero: float = 1e-10  # rounding grid for Hamiltonian/Liouvillian elements
    rlx_zero: float = 1e-10  # rounding grid for relaxation superoperators
    prop_chop: float = 1e-10  # rounding grid for propagators and Taylor terms
    subs_drop: float = 1e-10
    irrep_drop: float = 1e-10
    stst_tol: float = 1e-8
    zte_nsteps: int = 32
    zte_tol: float = 1e-24
    zte_maxden: float = 0.5
    prox_cutoff: float = 100
    krylov_tol: int = 10000  # matrix dimension above which evolution() uses Krylov
    basis_hush: int = 256
    merge_dim: int = 1000
    dense_matrix: float = 0.15  # density above which sparse matrices are made full
    small_matrix: int = 200  # dimension below which matrices are kept full
    rlx_integration: float = 1e-4
    dP_method: str = "auxmat"
    dd_ncells: int = 2
    cache_mem: int = 365
    # Physical constants, as Spinach keeps them in tols (CODATA 2018)
    hbar: float = 6.62607015e-34 / (2 * 3.141592653589793)
    kbol: float = 1.380649e-23
    freeg: float = 2.00231930436092
    mu0: float = 1.25663706127e-6
    muB: float = 9.2740100657e-24


def tolerances(overrides: dict[str, Any] | None = None) -> Tolerances:
    """Default tolerances, with user overrides (Spinach ``sys.tols``) applied.

    Spinach authors (tolerances.m): Ilya Kuprov, Hannah Hogben.
    """
    if not overrides:
        return Tolerances()
    known = {f.name for f in fields(Tolerances)}
    unknown = set(overrides) - known
    if unknown:
        raise ValueError(f"unknown tolerance(s) in sys['tols']: {sorted(unknown)}")
    return replace(Tolerances(), **overrides)
