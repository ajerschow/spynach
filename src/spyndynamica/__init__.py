"""SpynDynamica: a Python port of SpinDynamica (Malcolm Levitt's Mathematica package for spin dynamics).

The symbolic core: spin systems, operators as sums of product and spherical tensor operators with symbolic
coefficients, rotations, bases, kets and exact matrices. See ``docs/spyndynamica.md`` and
``porting/spyndynamica/PLAN.md``. The package imports NumPy and SymPy only, and nothing of spynach.
"""

from spyndynamica.operators import Operator, commutator
from spyndynamica.rotations import clebsch_gordan, euler_matrix, wigner_D, wigner_d
from spyndynamica.system import Basis, Bra, Ket, SpinSystem, spins_of

__version__ = "0.0.1"

__all__ = [
    "Basis",
    "Bra",
    "Ket",
    "Operator",
    "SpinSystem",
    "clebsch_gordan",
    "commutator",
    "euler_matrix",
    "spins_of",
    "wigner_D",
    "wigner_d",
]
