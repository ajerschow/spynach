# EPReasy: EasySpin's EPR functions in Python

A port of [EasySpin](https://easyspin.org) (Stefan Stoll and contributors, MIT
licence), the MATLAB toolbox for EPR spectral simulation, to Python. It lives
beside Spynach in this repository, is imported as `epreasy`, and depends on
`spynach` (for the oracle reader and comparison, and later for Liouville-space
propagation); nothing in `spynach` depends on it, and a test enforces that.
The Spinach port is the core of this repository: `epreasy` adapts to it and
changes nothing in it.

The user guides are [`docs/epreasy.md`](../../docs/epreasy.md) (foundation) and
[`docs/epreasy-simulators.md`](../../docs/epreasy-simulators.md) (simulators), the comparison
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
- `epreasy.endor`: `endorfrq` (ENDOR frequencies and intensities by matrix
  diagonalisation: transition pre-selection, enhancement, orientation
  selection, populations, flip angles), `endorfrq_perturb` (Iwasaki's first
  and second order), `orisel`, `anisosubsys`, and `salt` (powder and
  crystal ENDOR spectra).
- `epreasy.magnetometry`: `curry` (magnetic moments and susceptibilities,
  SI and CGS, operator and partition-function methods).
- `epreasy.slowmotion`: `chili` (the stochastic Liouville equation in the
  K-symmetrised Wigner basis: Brownian diffusion, orientational potentials,
  MOMD, Lanczos/continued-fraction and the other solvers, post-convolution;
  EasySpin's general Liouvillian method), `wigner3j`, `wigner6j`,
  `wignerd`, `tensor_cart2sph`, `tensor_sph2cart`.
- `epreasy.pulsed`: `saffron` (pulse EPR with EasySpin's fast engine:
  ideal and rectangular pulses, the predefined ESEEM, HYSCORE and Mims
  ENDOR experiments and custom sequences, coherence transfer pathways,
  orientation selection, product rule, relaxation decays, time-domain
  evolution), `sf_pathways`, `larmorfrq`, `apowin`, `fdaxis`, `exponfit`.
- `epreasy.rotations`: `erot`, `eulang`, `ang2vec`, `vec2ang`, `letter2vec`.
- `epreasy.units`, `epreasy.constants`: `unitconvert`, `mt2mhz`, `mhz2mt`,
  the physical constants as module attributes.
- `epreasy.nucdata`: EasySpin's isotope table (`data/isotopedata.txt`, kept
  verbatim) and `nucdata`, `nucspin`, `nucgval`, `nucqmom`, `nucabund`.
- `epreasy.oracle`: the fixture root (`$EPREASY_FIXTURES`, default
  `/easyspin-fixtures`) for the parity tests against EasySpin.

## Not here (yet, or on purpose)

- The simulator `spidyan` and `saffron`'s thyme engine (shaped pulses,
  explicit propagation, resonators, transient detection). In `pepper`
  and `garlic`: frequency sweeps of powders, isotope mixtures, partial
  ordering, sample rotations, immediate binning. In `salt` and `curry`:
  partial ordering, sample rotations. In `esfit`: the window. In `chili`:
  the fast C kernel (`chili_lm`), Heisenberg exchange, the LMK basis option,
  user-supplied starting vectors.
- Orbital angular momentum (`Sys.L`, `soc`, `gL`, `CF*`; `ham_so`, `ham_cf`,
  `ham_oz`), higher-order Zeeman terms (`Sys.Ham...`, `ham_ezho`), tilted
  frames of Stevens-operator terms (`Sys.B?Frame`, needs `wignerd`), crystal
  symmetries beyond P1 (`sitetransforms`) and sample rotations, sparse
  matrices in `resfields`.
- EasySpin's GUIs (`esfit`'s window, `eprconvert`), MD-trajectory methods
  (`cardamom`, `mdhmm`), Bruker file readers.

## Conventions

Integer index arrays, repeat counts and `bincount` bins are `np.intp`, not
`np.int64`: the browser playground runs the package on 32-bit WebAssembly,
where `intp` is int32 and NumPy refuses to cast int64 counts to it.

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

## On PyPI

EPReasy is published on PyPI as `easyspyn` (`pip install easyspyn`; the import name stays `epreasy`; the
distribution needs only NumPy and SciPy, with `easyspyn[fast]` adding Numba). `python scripts/build_easyspyn.py`
builds it into `dist/easyspyn/` from a staged copy of this package and `packaging/easyspyn/`; the version there
is kept equal to the repository's by `tests/unit/test_version.py`.

Releases go out from GitHub through PyPI's Trusted Publishing, with no token: publishing a GitHub release tagged
`easyspyn-vX.Y.Z` (the version in `packaging/easyspyn/pyproject.toml`) runs
`.github/workflows/publish-easyspyn.yml`, which tests and builds that commit and, once the reviewer of the `pypi`
environment approves, uploads it. One-time setup: a pending publisher on PyPI (project `easyspyn`, owner
`ajerschow`, repository `spynach`, workflow `publish-easyspyn.yml`, environment `pypi`) and an environment named
`pypi` in the repository settings. Spynach's own `vX.Y.Z` releases do not start it. A version can be uploaded only
once: bump it in `packaging/easyspyn/pyproject.toml`, `src/epreasy/__init__.py`, `pyproject.toml`,
`CITATION.cff` and the CHANGELOG (the version test lists what disagrees) before the next release.

## Rolling it back

The package touches nothing in `src/spynach`. To remove it:

```
git rm -r src/epreasy tests/epreasy oracle/matlab/easyspin oracle/run_easyspin_fixtures.sh \
          docs/epreasy.md docs/epreasy-simulators.md docs/coming-from-easyspin.md docs/notebooks/epreasy.ipynb \
          docs/notebooks/epreasy-simulators.ipynb docs/notebooks/coming-from-easyspin.ipynb \
          docs/figures/epreasy-*.png porting/EASYSPIN_FIXTURES.md
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
