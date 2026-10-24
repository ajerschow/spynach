"""Nuclear isotope data: EasySpin's ``nucdata`` and its wrappers.

The table ``data/isotopedata.txt`` is EasySpin's, kept verbatim (nuclear
moments from N. Stone, IAEA INDC(NDS)-0658, 2014). It differs from the
isotope table of ``spynach`` (which comes from Spinach); each package keeps
its own so that both match their originals.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files
from typing import NamedTuple

import numpy as np


@dataclass(frozen=True)
class Isotope:
    protons: int
    nucleons: int
    radioactive: bool
    element: str
    name: str
    spin: float
    gn: float
    abundance: float  # percent
    qm: float  # barn; NaN when not measured

    @property
    def symbol(self) -> str:
        return f"{self.nucleons}{self.element}"


class NucData(NamedTuple):
    spin: np.ndarray
    gn: np.ndarray
    qm: np.ndarray
    abund: np.ndarray  # fraction between 0 and 1
    nucs: list[str]


_TABLE: dict[str, Isotope] | None = None


def isotopes() -> dict[str, Isotope]:
    """The full isotope table, keyed by symbol (``'14N'``), read once."""
    global _TABLE
    if _TABLE is None:
        table: dict[str, Isotope] = {}
        text = files("epreasy").joinpath("data/isotopedata.txt").read_text(encoding="utf-8")
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("%"):
                continue
            cols = line.split()
            iso = Isotope(
                protons=int(cols[0]),
                nucleons=int(cols[1]),
                radioactive=cols[2] == "*",
                element=cols[3],
                name=cols[4],
                spin=float(cols[5]),
                gn=float(cols[6]),
                abundance=float(cols[7]),
                qm=float("nan") if cols[8].lower() == "nan" else float(cols[8]),
            )
            table[iso.symbol] = iso
        _TABLE = table
    return _TABLE


def nucstring2list(nuc_string) -> list[str]:
    """Split EasySpin's comma-separated isotope string into a list (``'14N,1H'`` -> ``['14N', '1H']``)."""
    if isinstance(nuc_string, (list, tuple)):
        return [str(n) for n in nuc_string]
    if nuc_string is None or nuc_string == "":
        return []
    if not isinstance(nuc_string, str):
        raise TypeError("Isotope list must be a string, not numeric.")
    if ";" in nuc_string:
        raise ValueError(f"Use , and not ; in nuclear spin string. You gave '{nuc_string}'.")
    text = "".join(nuc_string.split())
    if text.endswith(","):
        text = text[:-1]
    return [part for part in text.split(",") if part]


def nucdata(isotopes_spec) -> NucData:
    """Spin, g factor, quadrupole moment (barn) and natural abundance (fraction) of nuclei.

    ``isotopes_spec`` is an isotope symbol (``'63Cu'``), a comma-separated list
    (``'14N,14N,1H'``) or a list of symbols. Symbols are case-sensitive.
    """
    nucs = nucstring2list(isotopes_spec)
    table = isotopes()
    for k, nuc in enumerate(nucs, start=1):
        if nuc not in table:
            element = "".join(ch for ch in nuc if ch.isalpha())
            available = [iso.symbol for iso in table.values() if iso.element == element]
            if available:
                raise ValueError(
                    f"Problem in isotopes list, entry {k} ('{nuc}'): Please specify one of {', '.join(available)}."
                )
            raise ValueError(f"Problem in isotopes list, entry {k} ('{nuc}'): Unknown element '{element}'.")
    spin = np.array([table[n].spin for n in nucs], dtype=float)
    gn = np.array([table[n].gn for n in nucs], dtype=float)
    qm = np.array([table[n].qm for n in nucs], dtype=float)
    abund = np.array([table[n].abundance / 100 for n in nucs], dtype=float)
    return NucData(spin, gn, qm, abund, nucs)


def nucspin(isotopes_spec) -> np.ndarray:
    """Nuclear spin quantum numbers."""
    return nucdata(isotopes_spec).spin


def nucgval(isotopes_spec) -> np.ndarray:
    """Nuclear g factors."""
    return nucdata(isotopes_spec).gn


def nucqmom(isotopes_spec) -> np.ndarray:
    """Electric quadrupole moments in barn (NaN where not measured)."""
    return nucdata(isotopes_spec).qm


def nucabund(isotopes_spec) -> np.ndarray:
    """Natural abundances as fractions between 0 and 1."""
    return nucdata(isotopes_spec).abund
