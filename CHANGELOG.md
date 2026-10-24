# Changelog

Releases are tagged `vX.Y.Z` on GitHub; the version is `spynach.__version__`.

## Unreleased

- `epreasy`, a separate package beside Spynach that ports EasySpin (Stefan
  Stoll and co-workers, MIT) to Python, imported as `epreasy` and depending
  on `spynach` (never the reverse; a test enforces it). This first step is
  the foundation: spin systems as dicts with EasySpin's fields
  (`validatespinsys`), `sop` and `stev`, `ham` with the field split and its
  components, `levels`, `erot`/`eulang`, units, constants and EasySpin's
  isotope table; verified against EasySpin 6.0.15 on five spin systems
  (`oracle/matlab/easyspin/`). Guides `docs/epreasy.md` and
  `docs/coming-from-easyspin.md`.
- `epreasy` resonance-field layer: `resfields` (EasySpin's adaptive spline
  modelling of the energy level diagram, matrix and hybrid methods, H/g/A/D
  strains, Boltzmann and non-equilibrium populations, excitation modes,
  photoselection; crystal symmetry P1), `resfields_eig` (the eigenfield
  method), `cubicsolve` vectorised over all cubics, the line shapes
  `gaussian`, `lorentzian`, `lshape`, `voigtian`, `convspec`, and `sphgrid`
  with its triangulation; verified against EasySpin on a nitroxide, Cu(II)
  with two 14N, Mn(II) and a radical pair.
- `epreasy` simulators `pepper` (cw EPR of powders and crystals: matrix,
  hybrid, perturbation and eigenfield methods, orientation interpolation
  and triangle projection, strain summation, convolution, derivatives,
  field modulation) and `garlic` (isotropic solutions: Breit-Rabi and
  perturbation positions, equivalent nuclei, fast-motion widths, frequency
  sweeps), with `fastmotion`, `equivcouple`, `hamsymm`, `gridinterp`,
  `fieldmod` and the MEX kernels `lisum1i`, `projecttriangles`,
  `projectzones` in NumPy with an optional Numba path (`spynach-spin[fast]`)
  that is bit-exact with EasySpin's C code; verified against EasySpin on 32
  `pepper` and 14 `garlic` spectra.
- Spynach in the browser (contributed by Moritz Zaiss, MRsources): `try.html` runs
  the kernel in Pyodide in a web worker (`worker.js`), with no install, and
  draws spectra, FIDs and 2D contours from an editor (`show_spectrum`,
  `show_fid`, `show_image`). GitHub Pages serves it beside the project page.
- `powder` and `singlerot` take an optional `parameters["progress"]`, called
  as `progress(done, total)` after each orientation; Spinach has no
  equivalent.

## 0.2.0 - 2026-10-19

The Spinach port is unchanged in what it computes; this release adds a
module beside it and work on the figures and documentation.

- `ctps`, a separate module beside Spynach (`import ctps`, no dependency on
  `spynach`): coherence transfer pathway selection, a port of CCCP.m and
  CCCP++. It enumerates the pathways a sequence allows, weighted by phase
  cycle, gradients, diffusion and pulses, and has the non-ideal pulse model
  of J. Chem. Phys. 113, 979 (2000), a cogwheel cycle search by exact integer
  arithmetic (`find_cycle`, `coverage`, `hcl_cycle`), the DOTCOPS crusher
  objective (`crusher_scheme`) and an interactive pathway explorer. It can be
  removed by deleting its paths (`src/ctps/README.md`).
- New guides: `ctps` and `ctps-examples` (double-quantum filter, stimulated
  echo, HSQC, MQMAS cogwheel), and `beyond-the-port`, which collects what
  Spynach has and Spinach does not; `coming-from-spinach` points to it.
- `figure_png` (Matplotlib fallback) draws subplots, shapes, text
  annotations, dashed lines, markers and opacity, so static copies of
  multi-panel figures match the interactive ones.
- Notebooks stay below the 1 MB that nbviewer can fetch: embedded PNG copies
  at screen resolution (the figures in `docs/figures/` at twice that),
  `plot_2d` contour data stored as float32, the 2D spectra of the
  pulse-sequences guide binned with `max_points=128`, and a test that fails
  when a notebook reaches 1 MB.
- `playground/` for scratch notebooks (ignored except its README); the
  landing-page image is `assets/Spynoza.jpg`.

## 0.1.0 - 2026-10-18

First tagged release. Everything listed is verified against Spinach 2.13
fixtures (`docs/parity.md` lists every case, tolerance and measured
difference).

- Hilbert space (`zeeman-hilb`): spin systems, operators and states,
  isotropic Hamiltonians, propagators, pulse-acquire in the `liquid` context.
- Liouville space (`sphten-liouv`): basis with `IK-0`, `IK-1` and `IK-2`
  state-space restriction, projection, longitudinal, zero-quantum and manual
  filters, chemical subsystems and exchange kinetics, commutation and
  product superoperators, coherence and correlation filters, propagation with
  zero track elimination, path tracing, destination state screening and
  Krylov stepping, trajectory analysis (`trajan`, `stateinfo`).
- Anisotropic interactions: shielding and coupling tensors from eigenvalues
  and Euler angles, dipolar couplings from coordinates, quadrupolar couplings
  (`eeqq2nqi`), the irreducible components and `orientation`, the `crystal`
  and `powder` contexts with Spinach's spherical grids.
- Relaxation: Redfield theory for rotational diffusion with the `labframe`,
  `secular`, `kite` and `diagonal` retention policies, T1/T2 and damping
  models, thermal equilibrium with the inhomogeneous master equation and the
  DiBari-Levitt thermalisation, scalar relaxation of the first and second
  kind.
- Magic angle spinning in Fokker-Planck form (`singlerot`).
- Pulse sequences: `acquire`, `cosy`, `noesy`, `hsqc` with isotopomer
  dilution, analytical decoupling, `homospoil`, `apodisation`.
- Plotting: `plot_1d`, `plot_2d`, `plot_fid`, `plot_time_series`,
  `plot_trajan` as interactive Plotly figures, `figure_png` for static copies.
- Verification tooling: the oracle fixture reader and `compare`, 323 parity
  tests over 45 fixture cases, the parity summary generated from the suite,
  and the timed and profiled Spinach-vs-Spynach comparison protocol with HTML
  reports.
- Documentation: seven guides whose code runs in the test suite, each with a
  generated notebook, full-size example scripts, and contributed notebooks in
  `user_py/`.

Not in this release: a PyPI distribution (the name `spynach` on PyPI belongs
to an unrelated project), distribution of the Spinach fixtures, and the GPU
backend.
