"""spynach: a Python reimplementation of the Spinach spin dynamics kernel."""

__version__ = "0.3.1"  # keep equal to [project].version in pyproject.toml (tests/unit/test_version.py)

from spynach.assume import assume
from spynach.basis import basis
from spynach.contexts import crystal, liquid, load_grid, powder, singlerot
from spynach.filters import coherence, correlation, decouple, homospoil
from spynach.hamiltonian import frqoffset, hamiltonian, orientation
from spynach.kinetics import equilibrate, kinetics
from spynach.operators import operator
from spynach.processing import apodisation
from spynach.propagation import evolution, krylov, propagator, step
from spynach.reduction import path_trace, reduce, zte
from spynach.relaxation import equilibrium, relaxation, thermalize
from spynach.sequences import acquire, cosy, hsqc, noesy
from spynach.states import state, stateinfo, trajan
from spynach.system import Chemistry, SpinSystem, create, dilute, kill_spin
from spynach.utilities import hdot

__all__ = [
    "Chemistry",
    "SpinSystem",
    "acquire",
    "apodisation",
    "assume",
    "basis",
    "coherence",
    "correlation",
    "cosy",
    "create",
    "crystal",
    "decouple",
    "dilute",
    "equilibrate",
    "equilibrium",
    "evolution",
    "frqoffset",
    "hamiltonian",
    "hdot",
    "homospoil",
    "hsqc",
    "kill_spin",
    "kinetics",
    "krylov",
    "liquid",
    "load_grid",
    "noesy",
    "operator",
    "orientation",
    "path_trace",
    "powder",
    "propagator",
    "reduce",
    "relaxation",
    "singlerot",
    "state",
    "stateinfo",
    "step",
    "thermalize",
    "trajan",
    "zte",
]
