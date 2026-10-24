"""Unit conversions of EasySpin: unitconvert, mt2mhz, mhz2mt."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .constants import bmagn, boltzm, clight, evolt, gfree, planck


def _conversions(g: np.ndarray) -> dict[str, Callable[[np.ndarray], np.ndarray]]:
    return {
        "cm^-1->eV": lambda v: v * 100 * clight * planck / evolt,
        "cm^-1->K": lambda v: v * 100 * clight * planck / boltzm,
        "cm^-1->mT": lambda v: v / g * (planck / bmagn / 1e-3) * 100 * clight,
        "cm^-1->MHz": lambda v: v * 100 * clight / 1e6,
        "eV->cm^-1": lambda v: v * evolt / 100 / clight / planck,
        "eV->K": lambda v: v * evolt / boltzm,
        "eV->mT": lambda v: v / g / bmagn / 1e-3 * evolt,
        "eV->MHz": lambda v: v * evolt / planck / 1e6,
        "K->cm^-1": lambda v: v * boltzm / 100 / clight / planck,
        "K->eV": lambda v: v * boltzm / evolt,
        "K->mT": lambda v: v / g / bmagn / 1e-3 * boltzm,
        "K->MHz": lambda v: v * boltzm / planck / 1e6,
        "mT->cm^-1": lambda v: v * g / (planck / bmagn / 1e-3) / 100 / clight,
        "mT->eV": lambda v: v * g * bmagn * 1e-3 / evolt,
        "mT->K": lambda v: v * g * bmagn * 1e-3 / boltzm,
        "mT->MHz": lambda v: v * g * (1e-3 * bmagn / planck / 1e6),
        "MHz->cm^-1": lambda v: v * 1e6 / 100 / clight,
        "MHz->eV": lambda v: v * 1e6 * planck / evolt,
        "MHz->K": lambda v: v * 1e6 * planck / boltzm,
        "MHz->mT": lambda v: v / g * (planck / bmagn / 1e-3) * 1e6,
    }


def unitconvert(value, units: str, g=gfree):
    """Convert ``value`` between ``cm^-1``, ``eV``, ``K``, ``mT`` and ``MHz``.

    ``units`` is ``'<from>-><to>'``, e.g. ``'cm^-1->MHz'``. Conversions into or
    out of ``mT`` use the g factor ``g`` (default: the free electron). ``value``
    and ``g`` may be arrays of the same shape.
    """
    value = np.asarray(value, dtype=float)
    table = _conversions(np.asarray(g, dtype=float))
    if units in table:
        return table[units](value)
    for name in table:
        if name.lower() == units.lower():
            raise ValueError(f"You provided: {units}. Did you mean {name}?")
    raise ValueError("Unknown unit conversion specified.")


def mt2mhz(value_mT, g=gfree):
    """Magnetic field (mT) to frequency (MHz) for g factor ``g``."""
    return unitconvert(value_mT, "mT->MHz", g)


def mhz2mt(value_MHz, g=gfree):
    """Frequency (MHz) to magnetic field (mT) for g factor ``g``."""
    return unitconvert(value_MHz, "MHz->mT", g)
