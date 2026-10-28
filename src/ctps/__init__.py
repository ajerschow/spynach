"""ctps: coherence transfer pathway selection by phase cycling and pulsed field gradients.

A port of CCCP.m (Jerschow & Müller, J. Magn. Reson. 134, 17 (1998)) and CCCP++ (Jerschow &
Kumar, J. Magn. Reson. 160, 59 (2003)) with the non-ideal pulse treatment of Jerschow, J. Chem.
Phys. 113, 979 (2000) and an optimiser for phase cycles. Independent of ``spynach``.
"""

from ctps.model import (
    AllowedOrders,
    Nucleus,
    PathwayResult,
    Period,
    PhaseCycle,
    Pulse,
    Sample,
    Sequence,
)
from ctps.optimize import CycleSolution, SearchResult, coverage, find_cycle, hcl_cycle
from ctps.simulate import simulate

__all__ = [
    "AllowedOrders",
    "CycleSolution",
    "SearchResult",
    "coverage",
    "find_cycle",
    "hcl_cycle",
    "Nucleus",
    "PathwayResult",
    "Period",
    "PhaseCycle",
    "Pulse",
    "Sample",
    "Sequence",
    "simulate",
]
__version__ = "0.5.0"
