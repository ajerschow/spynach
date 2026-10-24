"""Physical constants with EasySpin's names and values.

EasySpin exposes each constant as a function (``bmagn()``); here they are
module attributes (``epreasy.bmagn``). Values are those of EasySpin 6.0.15
(CODATA 2022 where not exact by the 2019 SI redefinition).
"""

from math import pi

bmagn = 9.2740100657e-24  # Bohr magneton, J/T
nmagn = 5.0507837393e-27  # nuclear magneton, J/T
planck = 6.62607015e-34  # Planck constant, J s (exact)
hbar = planck / (2 * pi)  # reduced Planck constant, J s
gfree = 2.00231930436092  # free-electron g factor
clight = 299792458.0  # speed of light in vacuum, m/s (exact)
mu0 = 1.25663706127e-6  # vacuum permeability, N/A^2
boltzm = 1.380649e-23  # Boltzmann constant, J/K (exact)
echarge = 1.602176634e-19  # elementary charge, C (exact)
evolt = echarge  # electron volt, J (exact)
amu = 1.66053906892e-27  # atomic mass unit, kg
avogadro = 6.02214076e23  # Avogadro constant, 1/mol (exact)
