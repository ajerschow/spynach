"""Isotope data, physical constants, Larmor frequencies and dipolar couplings, as in SpinDynamica.

The isotope table and the constants (CODATA 2010) are those of SpinDynamica 3.10.2. Frequencies and couplings
are in rad/s, fields in tesla, distances in metres.

Author: Alexej Jerschow.
"""

from __future__ import annotations

import json
import math
import warnings
from functools import cache
from pathlib import Path
from typing import Any

import sympy as sp


@cache
def _data() -> dict[str, Any]:
    return json.loads((Path(__file__).parent / "data" / "isotopes.json").read_text(encoding="utf-8"))


@cache
def constants() -> dict[str, float]:
    """The physical constants in SI units: ``c``, ``h``, ``hbar``, ``mu0``, ``epsilon0``, ``kB``, ``Nav``, ``e``,
    ``me``, ``mp``, ``mn``."""
    return dict(_data()["constants"])


def isotopes() -> list[str]:
    """The names of the isotopes in the table, such as ``'13C'``."""
    return [row["name"] for row in _data()["isotopes"]]


def isotope(which: str | int) -> str:
    """The name of an isotope given as a name or as a mass number (the first one when several share it)."""
    rows = _data()["isotopes"]
    if isinstance(which, str):
        if which not in isotopes():
            raise ValueError(f"unknown isotope {which!r}.")
        return which
    matches = [row["name"] for row in rows if row["mass_number"] == which]
    if not matches:
        raise ValueError(f"no isotope of mass number {which} in the table.")
    if len(matches) > 1:
        warnings.warn(f"mass number {which} is shared by {matches}; using {matches[0]}.", stacklevel=2)
    return matches[0]


def _row(which: str | int) -> dict[str, Any]:
    name = isotope(which)
    return next(row for row in _data()["isotopes"] if row["name"] == name)


def spin_quantum_number(which: str | int) -> sp.Rational:
    """The nuclear spin quantum number of the ground state."""
    return sp.Rational(_row(which)["spin"])


def magnetogyric_ratio(which: str | int) -> float:
    """The magnetogyric ratio in rad/(s T)."""
    return float(_row(which)["gamma"])


def natural_abundance(which: str | int) -> float:
    """The natural abundance as a fraction."""
    return float(_row(which)["abundance"])


def larmor_frequency(which: str | int, field: float) -> float:
    """The Larmor frequency ``−γ B0`` in rad/s."""
    return -magnetogyric_ratio(which) * field


def dipolar_coupling(first: str | int, second: str | int | float, distance: float | None = None) -> float:
    """The dipolar coupling ``−(μ0/4π) γ1 γ2 ħ / r³`` in rad/s; ``dipolar_coupling('1H', r)`` is for a like pair."""
    if distance is None:
        second, distance = first, second
    k = constants()
    gammas = magnetogyric_ratio(first) * magnetogyric_ratio(second)
    return -(k["mu0"] / (4 * math.pi)) * gammas * k["hbar"] / distance**3
