"""Input and output: COG notation, spectrometer phase lists, result files.

The legacy loaders for CCCP.m example scripts and CCCP++ ``.par`` files are added once the
original sources are available in the container (they define the exact formats).

Author: Alexej Jerschow.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

from ctps.model import PathwayResult, PhaseCycle

_COG = re.compile(r"^\s*COG\s*(\d+)\s*[\(\{]\s*([-\d,\s]+?)\s*;\s*(-?\d+)\s*[\)\}]\s*$")


def parse_cog(text: str) -> PhaseCycle:
    """``COG23(0,3,4;14)`` or ``COG23{0,3,4;14}`` -> the cogwheel cycle (pulse winding numbers, then the receiver)."""
    match = _COG.match(text)
    if not match:
        raise ValueError(f"not a COG string: {text!r}; expected like COG23(0,3,4;14)")
    N = int(match.group(1))
    nu = [int(v) for v in match.group(2).replace(" ", "").split(",") if v != ""]
    return PhaseCycle.cogwheel(N, nu, int(match.group(3)))


def bruker_phase_program(cycle: PhaseCycle, *, receiver: int = 31, start: int = 1) -> str:
    """Phase lists in Bruker pulse-program syntax: ``ph1=(base) 0 1 2 3`` per pulse and ``ph31`` for the receiver.

    A base other than 4 is written with the ``(n)`` prefix Bruker uses for ``n`` steps per turn.
    """
    n = cycle.steps
    prefix = "" if cycle.base == 4 else f"({cycle.base}) "
    lines = []
    for i, ph in enumerate(cycle.phases):
        values = " ".join(str(v) for v in (list(ph) * (n // len(ph)))[:n])
        lines.append(f"ph{start + i}={prefix}{values}")
    base_rec = cycle.receiver_base or cycle.base
    prefix_rec = "" if base_rec == 4 else f"({base_rec}) "
    values = " ".join(str(v) for v in (list(cycle.receiver) * (n // len(cycle.receiver)))[:n])
    lines.append(f"ph{receiver}={prefix_rec}{values}")
    return "\n".join(lines) + "\n"


def phase_table(cycle: PhaseCycle, *, degrees: bool = True) -> list[dict]:
    """One row per step with every pulse phase and the receiver, in degrees (or base units), for any console."""
    n = cycle.steps
    rows = []
    phases = cycle.phase_matrix()
    receiver = cycle.receiver_phases()
    for m in range(n):
        row = {"step": m}
        for i in range(cycle.n_pulses):
            row[f"pulse_{i + 1}"] = (
                round(float(phases[i, m]) * 180 / 3.141592653589793, 6)
                if degrees
                else int(cycle.phases[i][m % len(cycle.phases[i])])
            )
        row["receiver"] = (
            round(float(receiver[m]) * 180 / 3.141592653589793, 6)
            if degrees
            else int(cycle.receiver[m % len(cycle.receiver)])
        )
        rows.append(row)
    return rows


def write_csv(result: PathwayResult, path: str | Path) -> None:
    rows = result.table()
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["rank"])
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "pathway": " ".join(str(p) for p in row["pathway"])})


def write_json(result: PathwayResult, path: str | Path) -> None:
    payload = {
        "sequence": result.sequence.to_dict(),
        "cycle": result.cycle.to_dict(),
        "n_enumerated": result.n_enumerated,
        "pathways": result.table(),
    }
    Path(path).write_text(json.dumps(payload, indent=1), encoding="utf-8")
