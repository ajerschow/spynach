# EasySpyn: EasySpin's EPR functions in Python

A port of [EasySpin](https://easyspin.org) (Stefan Stoll and contributors, MIT
licence), the MATLAB toolbox for EPR spectral simulation, to Python. It lives
beside Spynach in this repository, is imported as `easyspyn`, and depends on
`spynach` (for the oracle reader and comparison, and later for Liouville-space
propagation); nothing in `spynach` depends on it, and a test enforces that.
The Spinach port is the core of this repository: `easyspyn` adapts to it and
changes nothing in it.

The user guides are [`docs/easyspyn.md`](../../docs/easyspyn.md) (foundation) and
[`docs/easyspyn-simulators.md`](../../docs/easyspyn-simulators.md) (simulators), the comparison
with EasySpin [`docs/coming-from-easyspin.md`](../../docs/coming-from-easyspin.md)
(notebooks under `docs/notebooks/`); the tests are in `tests/easyspyn/`.

## What is here

- `easyspyn.spinsystem`: `validatespinsys`, `spinvec`, `hsdim`. A spin system
  is a dict with EasySpin's field names.
- `easyspyn.spinops`: `sop` (EasySpin's operator syntax) and `stev` (extended
  Stevens operators).
- `easyspyn.hamiltonian`: `ham` with the field split `H = H0 - B·mu`, the
  components `ham_ez`, `ham_nz`, `ham_hf`, `ham_zf`, `ham_nq`, `ham_ee`,
  `ham_nn`, and `levels`.
- `easyspyn.resfields`: `resfields` (adaptive spline modelling of the energy
  level diagram; matrix and hybrid methods, strains, populations, excitation
  modes, photoselection), `resfields_eig` (the eigenfield method),
  `cubicsolve`, `photoselect`, `nucspinrmv`.
- `easyspyn.lineshapes`: `gaussian`, `lorentzian`, `lshape`, `voigtian`,
  `convspec`.
- `easyspyn.grids`: `sphgrid` (the SOPHE grid with its triangulation),
  `gridparam`.
- `easyspyn.pepper`: `pepper`, cw EPR spectra of powders and crystals by the
  matrix, hybrid, perturbation and eigenfield methods, with orientation
  interpolation, triangle projection, strain summation, convolution,
  derivatives and field modulation; `easyspyn.perturb` holds
  `resfields_perturb`.
- `easyspyn.garlic`: `garlic` (isotropic cw spectra: Breit-Rabi and
  perturbation positions, equivalent nuclei, fast-motion widths, frequency
  sweeps), `fastmotion`, `equivcouple`.
- `easyspyn.kernels`: `lisum1i`, `projecttriangles`, `projectzones`, the MEX
  kernels in NumPy with a Numba path (installed by default, `EASYSPYN_NUMBA=0` switches
  it off) that is bit-exact with the MEX.
- `easyspyn.interp`: `gridinterp`, `esspline1d`, `esspline2d`;
  `easyspyn.symmetry`: `hamsymm`; `easyspyn.processing`: `fieldmod`;
  `easyspyn.components`: the loop over component systems
  (`pepper([sys1, sys2], exp)`).
- `easyspyn.fitting`: `esfit` without the GUI (EasySpin-style and custom
  models, global fits, scaling, baselines, masks, uncertainties), the
  algorithms `esfit_simplex`, `esfit_levmar`, `esfit_montecarlo`,
  `esfit_genetic`, `esfit_grid`, `esfit_swarm` (and SciPy's least squares
  for `lsqnonlin`), `jacobianest`, `hccm`, and the data helpers
  `rescaledata`, `basecorr`, `deriv`, `addnoise`.
- `easyspyn.endor`: `endorfrq` (ENDOR frequencies and intensities by matrix
  diagonalisation: transition pre-selection, enhancement, orientation
  selection, populations, flip angles), `endorfrq_perturb` (Iwasaki's first
  and second order), `orisel`, `anisosubsys`, and `salt` (powder and
  crystal ENDOR spectra).
- `easyspyn.magnetometry`: `curry` (magnetic moments and susceptibilities,
  SI and CGS, operator and partition-function methods).
- `easyspyn.slowmotion`: `chili` (the stochastic Liouville equation in the
  K-symmetrised Wigner basis: Brownian diffusion, orientational potentials,
  MOMD, Lanczos/continued-fraction and the other solvers, post-convolution;
  EasySpin's general Liouvillian method), `wigner3j`, `wigner6j`,
  `wignerd`, `tensor_cart2sph`, `tensor_sph2cart`.
- `easyspyn.pulsed`: `saffron` (pulse EPR with EasySpin's fast engine:
  ideal and rectangular pulses, the predefined ESEEM, HYSCORE and Mims
  ENDOR experiments and custom sequences, coherence transfer pathways,
  orientation selection, product rule, relaxation decays, time-domain
  evolution), `sf_pathways`, `larmorfrq`, `apowin`, `fdaxis`, `exponfit`.
- `easyspyn.rotations`: `erot`, `eulang`, `ang2vec`, `vec2ang`, `letter2vec`.
- `easyspyn.units`, `easyspyn.constants`: `unitconvert`, `mt2mhz`, `mhz2mt`,
  the physical constants as module attributes.
- `easyspyn.nucdata`: EasySpin's isotope table (`data/isotopedata.txt`, kept
  verbatim) and `nucdata`, `nucspin`, `nucgval`, `nucqmom`, `nucabund`.
- `easyspyn.oracle`: the fixture root (`$EASYSPYN_FIXTURES`, default
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
mixed in one call, and conversions will live on the `easyspyn` side.

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

## Credits

EasySpin's function headers name no authors, so every function that translates an EasySpin file carries a line
`EasySpin authors (pepper.m): ...` in its docstring, naming the people who authored commits to that file up to
release 6.0.15 (and any author the source names, such as John D'Errico for `jacobianest`), a block per module, and
the generated page [`docs/easyspin-credits.md`](../../docs/easyspin-credits.md).
`porting/tools/gen_easyspin_authors.py --fetch` reads the history through the `gh` CLI into
`porting/easyspin_authors.json`; `--write` writes the docstrings and the page; `tests/easyspyn/test_credits.py` keeps
them current without network access. The lists show who changed a file, not who wrote which algorithm.

## On PyPI

EasySpyn is published on PyPI as `easyspyn` (`pip install easyspyn`; the import name is the same; the
distribution needs NumPy, SciPy and Numba, which `EASYSPYN_NUMBA=0` switches off). `python scripts/build_easyspyn.py`
builds it into `dist/easyspyn/` from a staged copy of this package and `packaging/easyspyn/`; the version there
is kept equal to the repository's by `tests/unit/test_version.py`.

Releases go out from GitHub through PyPI's Trusted Publishing, with no token: publishing a GitHub release tagged
`easyspyn-vX.Y.Z` (the version in `packaging/easyspyn/pyproject.toml`) runs
`.github/workflows/publish-easyspyn.yml`, which tests and builds that commit and, once the reviewer of the `pypi`
environment approves, uploads it. One-time setup: a pending publisher on PyPI (project `easyspyn`, owner
`ajerschow`, repository `spynach`, workflow `publish-easyspyn.yml`, environment `pypi`) and an environment named
`pypi` in the repository settings. Spynach's own `vX.Y.Z` releases do not start it. A version can be uploaded only
once: bump it in `packaging/easyspyn/pyproject.toml`, `src/easyspyn/__init__.py`, `pyproject.toml`,
`CITATION.cff` and the CHANGELOG (the version test lists what disagrees) before the next release.

## Rolling it back

The package touches nothing in `src/spynach`. To remove it:

```
git rm -r src/easyspyn tests/easyspyn oracle/matlab/easyspin oracle/run_easyspin_fixtures.sh \
          docs/easyspyn.md docs/easyspyn-simulators.md docs/coming-from-easyspin.md docs/notebooks/easyspyn.ipynb \
          docs/notebooks/easyspyn-simulators.ipynb docs/notebooks/coming-from-easyspin.ipynb \
          docs/figures/easyspyn-*.png porting/EASYSPIN_FIXTURES.md
```

then revert the one-line mentions in `README.md`, `docs/README.md`,
`docs/beyond-the-port.md`, `CHANGELOG.md`, `porting/ROADMAP.md`, the
`easyspyn` package-data line in `pyproject.toml` and the EasySpin section of
`LICENSE`, and run the test suite.

## Licence

EasySpin is distributed under the MIT licence (copyright Stefan Stoll and
other contributors); its notice is reproduced in `LICENSE` at the repository
root, as that licence requires for the translated functions and the isotope
table. This port is released with the repository under the MIT licence.
