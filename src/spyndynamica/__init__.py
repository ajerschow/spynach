"""SpynDynamica: a Python port of SpinDynamica (Malcolm Levitt's Mathematica package for spin dynamics).

The symbolic core (spin systems, operators as sums of product and spherical tensor operators with symbolic
coefficients, rotations, bases, kets, exact matrices), Liouville space (operator bases, superoperators) and
the numeric engine for sequences of time-independent events. See ``docs/spyndynamica.md`` and
``porting/spyndynamica/PLAN.md``. The package imports NumPy, SciPy and SymPy only, and nothing of spynach.

SpinDynamica is the work of Malcolm H. Levitt and co-authors. Its terms of use
(https://www.spindynamica.soton.ac.uk/about/termsofuse/) state that it is open-source and freely available
for use and ask that published work cite C. Bengs, M. H. Levitt, "SpinDynamica: Symbolic and numerical
magnetic resonance in a Mathematica environment", Magn. Reson. Chem. (2017), doi:10.1002/mrc.4642. Work
that uses this port should cite that publication.
"""

from spyndynamica.dynamics import Delay, Pulse, Repeat, amplitudes, propagate, propagator, superpropagator, trajectory
from spyndynamica.liouville import (
    OperatorBasis,
    Superoperator,
    cartesian_product_basis,
    coherence_filter,
    commutation,
    default_operator_basis,
    double_commutation,
    left_multiplication,
    lindbladian,
    liouvillian,
    permutation_superoperator,
    phenomenological_relaxation,
    projection,
    right_multiplication,
    rotation_euler_superoperator,
    rotation_superoperator,
    shift_and_polarization_basis,
    shift_and_z_basis,
    singlet_triplet_operator_basis,
    spherical_tensor_basis,
    unity_superoperator,
    zeeman_ket_bra_basis,
)
from spyndynamica.operators import Operator, commutator
from spyndynamica.rotations import clebsch_gordan, euler_matrix, wigner_D, wigner_d
from spyndynamica.system import Basis, Bra, Ket, SpinSystem, spins_of

__version__ = "0.0.1"

__all__ = [
    "Basis",
    "Bra",
    "Delay",
    "Ket",
    "Operator",
    "OperatorBasis",
    "Pulse",
    "Repeat",
    "SpinSystem",
    "Superoperator",
    "amplitudes",
    "cartesian_product_basis",
    "clebsch_gordan",
    "coherence_filter",
    "commutation",
    "commutator",
    "default_operator_basis",
    "double_commutation",
    "euler_matrix",
    "left_multiplication",
    "lindbladian",
    "liouvillian",
    "permutation_superoperator",
    "phenomenological_relaxation",
    "projection",
    "propagate",
    "propagator",
    "right_multiplication",
    "rotation_euler_superoperator",
    "rotation_superoperator",
    "shift_and_polarization_basis",
    "shift_and_z_basis",
    "singlet_triplet_operator_basis",
    "spherical_tensor_basis",
    "spins_of",
    "superpropagator",
    "trajectory",
    "unity_superoperator",
    "wigner_D",
    "wigner_d",
    "zeeman_ket_bra_basis",
]
