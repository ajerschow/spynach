"""Interaction truncation assumptions (port of ``assume.m``).

Ported: the ``'nmr'``, ``'esr'``, ``'deer'``, ``'deer-zz'`` and ``'labframe'``
assumption sets and the ``'zeeman'``/``'couplings'`` retention options.

Spinach authors, by original file (headers of Spinach 2.13; see docs/spinach-credits.md):
  assume.m: Ilya Kuprov
"""

from __future__ import annotations

from dataclasses import replace

from spynach.system import Assumptions, SpinSystem


def assume(spin_system: SpinSystem, assumptions: str, retention: str | None = None) -> SpinSystem:
    """Return a copy of ``spin_system`` with interaction strengths set for ``assumptions``.

    ``'nmr'``: rotating frame for all spins, secular Zeeman terms, secular
    couplings between spins of the same isotope, and weak (zz) couplings
    between different isotopes. ``'esr'``/``'deer'``: rotating frame for
    electrons, lab frame for nuclei, secular electron couplings, weak and
    pseudosecular hyperfine couplings, all nuclear terms (``'deer-zz'``:
    weak inter-electron couplings). ``'labframe'``: no assumptions, every
    term of every interaction.

    ``retention='zeeman'`` drops all couplings; ``'couplings'`` drops all
    Zeeman interactions.

    Spinach authors (assume.m): Ilya Kuprov.
    """
    if any(t != "S" for t in spin_system.types):
        raise ValueError("Cavities, phonons, and transmons are not allowed under this assumption set.")
    isotopes = spin_system.isotopes
    nspins = spin_system.nspins
    electron = [iso.startswith("E") for iso in isotopes]
    if assumptions == "nmr":
        zeeman = ["secular"] * nspins
        coupling = [["secular" if isotopes[n] == isotopes[k] else "zz" for k in range(nspins)] for n in range(nspins)]
    elif assumptions in ("esr", "deer", "deer-zz"):
        zeeman = ["secular" if electron[n] else "full" for n in range(nspins)]
        coupling = []
        for n in range(nspins):
            row = []
            for k in range(nspins):
                if electron[n] and not electron[k]:
                    row.append("z*")
                elif electron[k] and not electron[n]:
                    row.append("*z")
                elif electron[n] and electron[k]:
                    row.append("zz" if assumptions == "deer-zz" and n != k else "secular")
                else:
                    row.append("strong")
            coupling.append(row)
    elif assumptions == "labframe":
        zeeman = ["full"] * nspins
        coupling = [["strong"] * nspins for _ in range(nspins)]
    else:
        raise NotImplementedError(
            f"assumption set {assumptions!r} not ported yet; 'nmr', 'esr', 'deer', 'deer-zz', 'labframe' are"
        )
    if retention == "couplings":
        zeeman = ["ignore"] * nspins
    elif retention == "zeeman":
        coupling = [["ignore"] * nspins for _ in range(nspins)]
    elif retention is not None:
        raise ValueError("retention must be 'zeeman', 'couplings', or None.")
    return replace(
        spin_system,
        assumptions=Assumptions(name=assumptions, zeeman=tuple(zeeman), coupling=tuple(tuple(row) for row in coupling)),
    )
