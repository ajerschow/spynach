"""Isotope database: multiplicity and magnetogyric ratio (port of ``spin.m``).

Spinach authors, by original file (headers of Spinach 2.13; see docs/spinach-credits.md):
  spin.m: Matthew Krzystyniak, Andreas Biternas, Ilya Kuprov
"""

from __future__ import annotations

import re

from spynach._isotope_data import ISOTOPES, NO_DATA

ELECTRON_GAMMA = -1.76085963023e11  # rad/(s*T), CODATA 2018


def spin(name: str) -> tuple[float, int]:
    """Return ``(gamma, multiplicity)`` for an isotope name such as ``'1H'`` or ``'E'``.

    Special names follow Spinach: ``'G'`` ghost spin, ``'E'`` electron,
    ``'E<n>'`` electron with multiplicity n, ``'C<n>'``, ``'V<n>'``, ``'T<n>'``
    bosonic modes (cavity, phonon, transmon) with n levels and zero gamma.

    Spinach authors (spin.m): Matthew Krzystyniak, Andreas Biternas, Ilya Kuprov.
    """
    if not isinstance(name, str):
        raise TypeError("isotope specification must be a character string.")
    if re.fullmatch(r"E\d+", name):
        mult = int(name[1:])
        if mult < 2:
            raise ValueError("electrons need at least two energy levels.")
        return ELECTRON_GAMMA, mult
    if re.fullmatch(r"[CVT]\d+", name):
        mult = int(name[1:])
        if mult < 3:
            raise ValueError("bosonic modes need at least three energy levels.")
        return 0.0, mult
    if name in NO_DATA:
        raise ValueError(f"{name} - no data available in the current NMR literature.")
    if name not in ISOTOPES:
        raise ValueError(f"{name} - unknown isotope.")
    mult, gamma = ISOTOPES[name]
    return float(gamma), mult


def particle_type(name: str) -> str:
    """Spinach particle type: ``'S'`` for spins, ``'C'``/``'V'``/``'T'`` for bosonic modes."""
    return name[0] if re.fullmatch(r"[CVT]\d+", name) else "S"
