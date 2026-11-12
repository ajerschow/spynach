"""Phase-cycle factor and cogwheel selection.

1998 Eq. 12/16/17: ``ψ_m = -Δp·φ_m - φ_rec,m`` and ``s = Σ_m exp(-iψ_m)``; the tables report
``|s| / N``. The integer form evaluates the same sum through a table of the ``fullbase``
roots of unity, so that exact zeros are exact (a design choice of this port; CCCP++ uses
precomputed sines and cosines per pathway). Cogwheel cycles ``COG_N(ν; ν_rec)`` select a
pathway iff ``Σ_j ν_j Δp_j + ν_rec ≡ 0 (mod N)`` (Jerschow & Kumar 2003 Eq. 4), the receiver of
a wanted pathway being ``ν_rec = -Σ ν_j Δp⁰_j mod N``.

Author: Alexej Jerschow.
"""

from __future__ import annotations

import numpy as np

from ctps.model import PhaseCycle


def phase_cycle_factor(dp: np.ndarray, cycle: PhaseCycle, integer: bool = True) -> np.ndarray:
    """Complex ``s / N`` for every row of ``dp`` (``(k, n_pulses)`` order jumps)."""
    if integer:
        phases, receiver, fullbase = cycle.integer_matrix()
        # GPU: integer GEMM, modulo, gather, row mean
        index = (dp.astype(np.int64) @ phases + receiver[None, :]) % fullbase
        table = np.exp(-2j * np.pi * np.arange(fullbase) / fullbase)
        return table[index].mean(axis=1)
    psi = dp.astype(float) @ cycle.phase_matrix() + cycle.receiver_phases()[None, :]
    return np.exp(-1j * psi).mean(axis=1)  # GPU: complex exp and row mean


def cogwheel_selects(dp: np.ndarray, N: int, nu: np.ndarray, nu_rec: int) -> np.ndarray:
    """True where ``Σ ν_j Δp_j + ν_rec ≡ 0 (mod N)``: the pathway survives the cogwheel cycle."""
    return (dp.astype(np.int64) @ np.asarray(nu, dtype=np.int64) + int(nu_rec)) % int(N) == 0


def receiver_winding(dp_wanted: np.ndarray, N: int, nu: np.ndarray) -> int:
    """``ν_rec`` that lets the wanted pathway (``(n_pulses,)`` jumps) through: ``-Σ ν_j Δp⁰_j mod N``."""
    return int((-np.asarray(dp_wanted, dtype=np.int64) @ np.asarray(nu, dtype=np.int64)) % int(N))
