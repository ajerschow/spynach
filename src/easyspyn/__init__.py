"""easyspyn: EasySpin's EPR simulation functions in Python, beside Spynach.

Spin systems are dicts with EasySpin's field names; units are EasySpin's
(mT, MHz); spin numbers are 1-based as in EasySpin. See ``docs/easyspyn.md``
and ``docs/coming-from-easyspin.md``.
"""

from .constants import amu, avogadro, bmagn, boltzm, clight, echarge, evolt, gfree, hbar, mu0, nmagn, planck
from .endor import EndorResonances, anisosubsys, endorfrq, endorfrq_perturb, orisel, salt
from .fitting import (
    FitResult,
    addnoise,
    basecorr,
    deriv,
    esfit,
    esfit_genetic,
    esfit_grid,
    esfit_levmar,
    esfit_montecarlo,
    esfit_simplex,
    esfit_swarm,
    hccm,
    jacobianest,
    rescaledata,
)
from .garlic import Spectrum, equivcouple, fastmotion, garlic
from .grids import SphGrid, gridparam, sphgrid
from .hamiltonian import ham, ham_ee, ham_ez, ham_hf, ham_nn, ham_nq, ham_nz, ham_zf, levels
from .interp import esspline1d, esspline2d, gridinterp
from .kernels import lisum1i, projecttriangles, projectzones
from .lineshapes import convspec, gaussian, lorentzian, lshape, voigtian
from .magnetometry import curry
from .nucdata import isotopes, nucabund, nucdata, nucgval, nucqmom, nucspin, nucstring2list
from .pepper import pepper
from .perturb import resfields_perturb
from .processing import fieldmod
from .pulsed import apowin, exponfit, fdaxis, larmorfrq, saffron, sf_pathways
from .resfields import Resonances, cubicsolve, nucspinrmv, photoselect, resfields, resfields_eig
from .rotations import ang2vec, erot, eulang, letter2vec, vec2ang
from .slowmotion import chili, tensor_cart2sph, tensor_sph2cart, wigner3j, wigner6j, wignerd
from .spinops import sop, stev
from .spinsystem import hsdim, spinvec, validatespinsys
from .symmetry import hamsymm
from .units import mhz2mt, mt2mhz, unitconvert

__version__ = "0.3.0"

__all__ = [
    "amu", "avogadro", "bmagn", "boltzm", "clight", "echarge", "evolt", "gfree", "hbar", "mu0", "nmagn", "planck",
    "ham", "ham_ee", "ham_ez", "ham_hf", "ham_nn", "ham_nq", "ham_nz", "ham_zf", "levels",
    "Spectrum", "equivcouple", "fastmotion", "garlic",
    "EndorResonances", "anisosubsys", "endorfrq", "endorfrq_perturb", "orisel", "salt", "curry",
    "chili", "tensor_cart2sph", "tensor_sph2cart", "wigner3j", "wigner6j", "wignerd",
    "apowin", "exponfit", "fdaxis", "larmorfrq", "saffron", "sf_pathways",
    "FitResult", "addnoise", "basecorr", "deriv", "esfit", "esfit_genetic", "esfit_grid", "esfit_levmar",
    "esfit_montecarlo", "esfit_simplex", "esfit_swarm", "hccm", "jacobianest", "rescaledata",
    "SphGrid", "gridparam", "sphgrid",
    "esspline1d", "esspline2d", "gridinterp",
    "lisum1i", "projecttriangles", "projectzones",
    "fieldmod", "hamsymm", "pepper", "resfields_perturb",
    "convspec", "gaussian", "lorentzian", "lshape", "voigtian",
    "Resonances", "cubicsolve", "nucspinrmv", "photoselect", "resfields", "resfields_eig",
    "isotopes", "nucabund", "nucdata", "nucgval", "nucqmom", "nucspin", "nucstring2list",
    "ang2vec", "erot", "eulang", "letter2vec", "vec2ang",
    "sop", "stev",
    "hsdim", "spinvec", "validatespinsys",
    "mhz2mt", "mt2mhz", "unitconvert",
]  # fmt: skip
