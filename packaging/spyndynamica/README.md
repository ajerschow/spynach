# SpynDynamica: SpinDynamica in Python

A Python port of [SpinDynamica](https://www.spindynamica.soton.ac.uk) 3.10.2, the Mathematica package for
spin dynamics and NMR by Malcolm H. Levitt and co-authors. Spin operators are sums of product and spherical
tensor operators with symbolic coefficients (SymPy), and a numeric engine (NumPy, SciPy) propagates sequences
of pulses and delays.

    pip install spyndynamica

Nothing to install? The examples run in the browser, cell by cell:
https://ajerschow.github.io/spynach/try.html#ex-sd-cosy

```python
import sympy as sp
from spyndynamica import SpinSystem

s = SpinSystem(2)
I = s.I
Omega, J, t = sp.symbols("Omega J t", real=True)
H = Omega * I(1, "z") + 2 * sp.pi * J * (I(1, "z") @ I(2, "z"))

rho = I(1, "z").rotate(sp.pi / 2, "x", spins=[1]).evolve(H, t)
print(rho)
print(rho.amplitude(2 * I(1, "x") @ I(2, "z")))
```

What is ported:

- spin systems of any spin quantum numbers, product and spherical tensor operators, commutators, rotations,
  closed-form evolution, the Zeeman, singlet-triplet, coupled and user-defined bases, kets and bras;
- Liouville space: operator bases, superoperators, phenomenological relaxation, coherence-order and spherical
  rank filters;
- the numeric engine: pulses, delays, shaped pulses and pulse sequences, time-dependent and periodic
  Hamiltonians, adiabatic events, trajectories, signals with their Fourier transform in one and two dimensions,
  the COMPUTE method with the carousel average, ensemble and powder averages;
- density operators and polarization levels, chemical exchange in extended Liouville space, bounds on
  transformation amplitudes, eigenvalues followed along a parameter;
- commutants, Lie algebras, spin permutation groups and symmetry-adapted bases, and three-dimensional pictures
  of tensors and frames.

The conventions are those of SpinDynamica, and the results are compared with SpinDynamica's own output on
about 980 reference cases. Names follow Python usage (`signal_1d` for `Signal1D`); the table of names and the
list of differences are in
https://github.com/ajerschow/spynach/blob/main/docs/coming-from-spindynamica.md. Where SpinDynamica 3.10.2
does something else than it documents, the port follows the documentation and records the case in
https://github.com/ajerschow/spynach/blob/main/porting/spyndynamica/DIVERGENCES.md.

SpynDynamica is part of the [Spynach](https://github.com/ajerschow/spynach) project, which also ports the
Spinach spin dynamics kernel and EasySpin; it uses none of their code and is installed on its own. It is an
early release; corrections and bug reports are welcome on GitHub.

- Guides: https://github.com/ajerschow/spynach/blob/main/docs/spyndynamica.md (operators, bases, superoperators),
  https://github.com/ajerschow/spynach/blob/main/docs/spyndynamica-simulations.md (the numeric engine),
  https://github.com/ajerschow/spynach/blob/main/docs/spyndynamica-symmetry.md (algebras, symmetry, pictures),
  https://github.com/ajerschow/spynach/blob/main/docs/spyndynamica-adiabatic.md (trajectories, level diagrams,
  adiabatic events)
- Figures of the three-dimensional pictures need Plotly: `pip install "spyndynamica[plot]"`

Credit and citation: SpinDynamica is the work of Malcolm H. Levitt and co-authors. Its
[terms of use](https://www.spindynamica.soton.ac.uk/about/termsofuse/) state that it is open-source and freely
available for use, and ask that published work cite C. Bengs, M. H. Levitt, *SpinDynamica: Symbolic and
numerical magnetic resonance in a Mathematica environment*, Magn. Reson. Chem. (2017), doi:10.1002/mrc.4642.
Work that uses SpynDynamica should cite that publication. The port is under the MIT licence (`LICENSE`, which
carries the SpinDynamica notice); no SpinDynamica source is part of it.
