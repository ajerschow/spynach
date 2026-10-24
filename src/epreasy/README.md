# epreasy: EasySpin's EPR functions in Python

A port of [EasySpin](https://easyspin.org) (Stefan Stoll and contributors, MIT
licence), the MATLAB toolbox for EPR spectral simulation, to Python. It lives
beside Spynach in this repository, is imported as `epreasy`, and depends on
`spynach` (for the oracle reader and comparison, and later for Liouville-space
propagation); nothing in `spynach` depends on it, and a test enforces that.
The Spinach port is the core of this repository: `epreasy` adapts to it and
changes nothing in it.

The user guide is [`docs/epreasy.md`](../../docs/epreasy.md), the comparison
with EasySpin [`docs/coming-from-easyspin.md`](../../docs/coming-from-easyspin.md)
(notebooks under `docs/notebooks/`); the tests are in `tests/epreasy/`.

## What is here

- `epreasy.spinsystem`: `validatespinsys`, `spinvec`, `hsdim`. A spin system
  is a dict with EasySpin's field names.
- `epreasy.spinops`: `sop` (EasySpin's operator syntax) and `stev` (extended
  Stevens operators).
- `epreasy.hamiltonian`: `ham` with the field split `H = H0 - B·mu`, the
  components `ham_ez`, `ham_nz`, `ham_hf`, `ham_zf`, `ham_nq`, `ham_ee`,
  `ham_nn`, and `levels`.
- `epreasy.resfields`: `resfields` (adaptive spline modelling of the energy
  level diagram; matrix and hybrid methods, strains, populations, excitation
  modes, photoselection), `resfields_eig` (the eigenfield method),
  `cubicsolve`, `photoselect`, `nucspinrmv`.
- `epreasy.lineshapes`: `gaussian`, `lorentzian`, `lshape`, `voigtian`,
  `convspec`.
- `epreasy.grids`: `sphgrid` (the SOPHE grid with its triangulation),
  `gridparam`.
- `epreasy.pepper`: `pepper`, cw EPR spectra of powders and crystals by the
  matrix, hybrid, perturbation and eigenfield methods, with orientation
  interpolation, triangle projection, strain summation, convolution,
  derivatives and field modulation; `epreasy.perturb` holds
  `resfields_perturb`.
- `epreasy.garlic`: `garlic` (isotropic cw spectra: Breit-Rabi and
  perturbation positions, equivalent nuclei, fast-motion widths, frequency
  sweeps), `fastmotion`, `equivcouple`.
- `epreasy.kernels`: `lisum1i`, `projecttriangles`, `projectzones`, the MEX
  kernels in NumPy with a Numba path (`fast` extra) that is bit-exact with
  the MEX.
- `epreasy.interp`: `gridinterp`, `esspline1d`, `esspline2d`;
  `epreasy.symmetry`: `hamsymm`; `epreasy.processing`: `fieldmod`;
  `epreasy.components`: the loop over component systems
  (`pepper([sys1, sys2], exp)`).
- `epreasy.fitting`: `esfit` without the GUI (EasySpin-style and custom
  models, global fits, scaling, baselines, masks, uncertainties), the
  algorithms `esfit_simplex`, `esfit_levmar`, `esfit_montecarlo`,
  `esfit_genetic`, `esfit_grid`, `esfit_swarm` (and SciPy's least squares
  for `lsqnonlin`), `jacobianest`, `hccm`, and the data helpers
  `rescaledata`, `basecorr`, `deriv`, `addnoise`.
- `epreasy.rotations`: `erot`, `eulang`, `ang2vec`, `vec2ang`, `letter2vec`.
- `epreasy.units`, `epreasy.constants`: `unitconvert`, `mt2mhz`, `mhz2mt`,
  the physical constants as module attributes.
- `epreasy.nucdata`: EasySpin's isotope table (`data/isotopedata.txt`, kept
  verbatim) and `nucdata`, `nucspin`, `nucgval`, `nucqmom`, `nucabund`.
- `epreasy.oracle`: the fixture root (`$EPREASY_FIXTURES`, default
  `/easyspin-fixtures`) for the parity tests against EasySpin.

## Not here (yet, or on purpose)

- The simulators `salt`, `chili`, `saffron`, `spidyan`, `curry`: the next
  phases. In `pepper` and `garlic`: frequency sweeps of powders, isotope
  mixtures, partial ordering, sample rotations, immediate binning. In
  `esfit`: the window.
- Orbital angular momentum (`Sys.L`, `soc`, `gL`, `CF*`; `ham_so`, `ham_cf`,
  `ham_oz`), higher-order Zeeman terms (`Sys.Ham...`, `ham_ezho`), tilted
  frames of Stevens-operator terms (`Sys.B?Frame`, needs `wignerd`), crystal
  symmetries beyond P1 (`sitetransforms`) and sample rotations, sparse
  matrices in `resfields`.
- EasySpin's GUIs (`esfit`'s window, `eprconvert`), MD-trajectory methods
  (`cardamom`, `mdhmm`), Bruker file readers.

## Conventions

EasySpin's throughout, so that an EasySpin script maps line by line: spin
systems are dicts with EasySpin's field names; fields in mT, energies and
couplings in MHz, magnetic moments in MHz/mT; Euler angles are EasySpin's
passive rotations (`erot`); spin and level numbers are 1-based (`'x2'` is the
second spin). This differs from `spynach`, which uses Spinach's conventions
(T, Hz, rad/s, active rotations, 0-based spins); the two packages are not
mixed in one call, and conversions will live on the `epreasy` side.

Numbers follow MATLAB's shapes: a scalar or a flat list is one row
(`g=[2.0088, 2.0061, 2.0027]` is one electron's principal values; `A=[16, 86]`
is one axial hyperfine tensor), a list of lists is one row per spin, and a
full tensor is a 3x3 block per spin.

## Reference data

`oracle/matlab/easyspin/` holds the MATLAB side: `run_easyspin_fixtures.m`
(settings block: EasySpin install, version, fixture root), the dump scripts
in `dumps/` and `lib/es_meta.m`; `oracle/run_easyspin_fixtures.sh` starts it
in batch mode. The installed EasySpin release is only read. Fixtures go to
`<root>/<version>/<case>/` with `data.h5` (the shared `oracle_write.m`
layout) and `meta.json`, whose `inputs` the parity tests read back to rebuild
each case; `porting/EASYSPIN_FIXTURES.md` lists the cases.

## Rolling it back

The package touches nothing in `src/spynach`. To remove it:

```
git rm -r src/epreasy tests/epreasy oracle/matlab/easyspin oracle/run_easyspin_fixtures.sh \
          docs/epreasy.md docs/coming-from-easyspin.md docs/notebooks/epreasy.ipynb \
          docs/notebooks/coming-from-easyspin.ipynb docs/figures/epreasy-*.png porting/EASYSPIN_FIXTURES.md
```

then revert the one-line mentions in `README.md`, `docs/README.md`,
`docs/beyond-the-port.md`, `CHANGELOG.md`, `porting/ROADMAP.md`, the
`epreasy` package-data line in `pyproject.toml` and the EasySpin section of
`LICENSE`, and run the test suite.

## Licence

EasySpin is distributed under the MIT licence (copyright Stefan Stoll and
other contributors); its notice is reproduced in `LICENSE` at the repository
root, as that licence requires for the translated functions and the isotope
table. This port is released with the repository under the MIT licence.
