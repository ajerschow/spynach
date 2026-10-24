"""epreasy: EasySpin's EPR simulation functions in Python, beside Spynach.

Spin systems are dicts with EasySpin's field names; units are EasySpin's
(mT, MHz); spin numbers are 1-based as in EasySpin. See ``docs/epreasy.md``
and ``docs/coming-from-easyspin.md``.
"""

from .constants import amu, avogadro, bmagn, boltzm, clight, echarge, evolt, gfree, hbar, mu0, nmagn, planck
from .grids import SphGrid, gridparam, sphgrid
from .hamiltonian import ham, ham_ee, ham_ez, ham_hf, ham_nn, ham_nq, ham_nz, ham_zf, levels
from .lineshapes import convspec, gaussian, lorentzian, lshape, voigtian
from .nucdata import isotopes, nucabund, nucdata, nucgval, nucqmom, nucspin, nucstring2list
from .resfields import Resonances, cubicsolve, nucspinrmv, photoselect, resfields, resfields_eig
from .rotations import ang2vec, erot, eulang, letter2vec, vec2ang
from .spinops import sop, stev
from .spinsystem import hsdim, spinvec, validatespinsys
from .units import mhz2mt, mt2mhz, unitconvert

__version__ = "0.1.0"

__all__ = [
    "amu", "avogadro", "bmagn", "boltzm", "clight", "echarge", "evolt", "gfree", "hbar", "mu0", "nmagn", "planck",
    "ham", "ham_ee", "ham_ez", "ham_hf", "ham_nn", "ham_nq", "ham_nz", "ham_zf", "levels",
    "SphGrid", "gridparam", "sphgrid",
    "convspec", "gaussian", "lorentzian", "lshape", "voigtian",
    "Resonances", "cubicsolve", "nucspinrmv", "photoselect", "resfields", "resfields_eig",
    "isotopes", "nucabund", "nucdata", "nucgval", "nucqmom", "nucspin", "nucstring2list",
    "ang2vec", "erot", "eulang", "letter2vec", "vec2ang",
    "sop", "stev",
    "hsdim", "spinvec", "validatespinsys",
    "mhz2mt", "mt2mhz", "unitconvert",
]  # fmt: skip
