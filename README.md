# Spynach

A Python reimplementation of the [Spinach](https://spindynamics.org) spin
dynamics kernel (MIT licence, see `LICENSE`), verified stage by stage against the MATLAB
original. **Start with the [Hilbert-space guide](docs/hilbert-space.md)**; if you use
Spinach, also read [Coming from Spinach](docs/coming-from-spinach.md). Each
guide has a runnable Jupyter notebook in [`docs/notebooks/`](docs/notebooks/). The project
page at [ajerschow.github.io/spynach](https://ajerschow.github.io/spynach/) links the
guides, the notebooks (rendered by nbviewer) and the Spinach comparison reports, and
has an "Open in Google Colab" button that runs the quick-start example without a local
install ([`docs/colab/quick-start.ipynb`](docs/colab/quick-start.ipynb)), and a page that
runs Spynach in the browser through Pyodide
([try.html](https://ajerschow.github.io/spynach/try.html), contributed by Moritz Zaiss). See `HANDOFF.md` for goals and design decisions, and
`oracle/README.md` for the reference-data ("fixture") harness.

## Status

Ported and verified against Spinach 2.13 fixtures (`tests/slice/`): the
`zeeman-hilb` pulse-acquire chain for liquid-state NMR with isotropic
interactions. The `sphten-liouv` Liouville-space formalism (basis with state-space
restriction, superoperators, states, Hamiltonian and kinetics
superoperators, analytical filters, propagation with zero track
elimination, path tracing and destination state screening) runs
pulse-acquire and chemical exchange end to end; its setup is verified
against Spinach 2.13 fixtures (`tests/slice/test_liouv_setup_parity.py`)
and its propagation against Spinach 2.13 fixtures as well
(`tests/slice/test_liouv_pa_parity.py`), Hilbert space, and analytic
exchange line shapes. Anisotropic interactions (shielding and coupling
tensors, dipolar couplings from coordinates, quadrupolar couplings) with
the `crystal` and `powder` contexts are ported and verified against
Spinach 2.13 fixtures as well (`tests/slice/test_aniso_parity.py`).
Relaxation (Redfield theory, T1/T2 and damping models, thermalisation)
is ported and verified against Spinach 2.13 fixtures
(`tests/slice/test_relax_parity.py`) and the Solomon equations; scalar
relaxation of the first and second kind (`SRFK`, `SRSK`) is ported and
verified against Spinach 2.13 fixtures (`tests/slice/test_scalar_parity.py`),
Abragam's expressions and the explicit integral.
Magic angle spinning in Fokker-Planck form (`singlerot`) is ported and
verified against Spinach 2.13 fixtures (`tests/slice/test_mas_parity.py`),
rotor echoes and the fast-spinning limit.
The pulse sequence library (COSY, NOESY, HSQC with isotopomer dilution,
analytical decoupling, apodisation and 2D plotting) is ported and
verified against Spinach 2.13 fixtures (`tests/slice/test_seq_parity.py`).

```python
import numpy as np
import spynach

spin_system = spynach.create(
    {"isotopes": ["1H", "1H"], "magnet": 14.1},
    {"zeeman": {"scalar": [1.00, 1.10]}, "coupling": {"scalar": [[0, 12.0], [0, 0]]}},
)
spin_system = spynach.basis(spin_system, {"formalism": "zeeman-hilb", "approximation": "none"})
pars = {"spins": ["1H"], "offset": 630, "sweep": 200, "npoints": 1024,
        "rho0": spynach.state(spin_system, "L+", "1H"), "coil": spynach.state(spin_system, "L+", "1H")}
fid = spynach.liquid(spin_system, spynach.acquire, pars, "nmr")
fid = spynach.apodisation(fid, [("exp", 6)])     # halve the first point, exponential window
spectrum = np.fft.fftshift(np.fft.fft(fid, 4096))
```

| Spinach | Spynach | ported |
|---|---|---|
| `create.m`, `dipolar.m` | `spynach.create` | isotopes, magnet, tolerances, Zeeman and coupling tensors as scalars, matrices, or eigenvalues with Euler angles, coordinates with point dipolar couplings and the proximity graph, chemical subsystems (`inter.chem` parts, concentrations, exchange rates) |
| `basis.m` | `spynach.basis` | `zeeman-hilb`; `sphten-liouv` with `none`, `IK-0`, `IK-1`, `IK-2`, the `projections`/`longitudinal`/`zero_quantum`/`manual` filters, and chemical subsystems (no symmetry, no coordinates) |
| `operator.m`, `state.m` | `spynach.operator`, `spynach.state` | Hilbert space and `sphten-liouv` superoperators (`left`, `right`, `comm`, `acomm`); states by the `exact`, `cheap` and `chem` methods; `E`, `Lx`, `Ly`, `Lz`, `L+`, `L-`, `Tl,m` |
| `assume.m` | `spynach.assume` | `nmr`, `esr`, `deer`, `deer-zz`, `labframe`, with `zeeman`/`couplings` retention |
| `hamiltonian.m`, `orientation.m`, `frqoffset.m` | `spynach.hamiltonian`, `spynach.orientation`, `spynach.frqoffset` | isotropic part and the rank-1/rank-2 irreducible components `Q` (CSA, anisotropic couplings, quadrupolar); in `sphten-liouv` as `left`/`right`/`comm`/`acomm` superoperators |
| `euler2dcm.m`, `mat2sphten.m`, `wigner.m`, `eeqq2nqi.m` | `spynach.conventions` | the same conventions |
| `kinetics.m`, `equilibrate.m` | `spynach.kinetics`, `spynach.equilibrate` | first-order exchange between chemical subsystems; equilibrium concentrations |
| `trajan.m`, `stateinfo.m` | `spynach.trajan`, `spynach.stateinfo`, `spynach.plotting.plot_trajan` | correlation-order, coherence-order and per-spin amplitudes along a trajectory; the largest coefficients of a state (no `level_populations`) |
| `correlation.m`, `coherence.m` | `spynach.correlation`, `spynach.coherence` | `sphten-liouv` |
| `propagator.m`, `step.m`, `krylov.m` | `spynach.propagator`, `spynach.step`, `spynach.krylov` | dense Pade below `small_matrix`, scaled-squared Taylor above; Taylor action on vectors with 1-, 2-, 3-point Lie quadratures |
| `evolution.m` | `spynach.evolution` | `zeeman-hilb`: `observable`; `sphten-liouv`: `observable`, `final`, `trajectory`, `multichannel`, `refocus`, destination state screening, Krylov path |
| `reduce.m`, `zte.m`, `path_trace.m` | `spynach.reduce`, `spynach.zte`, `spynach.path_trace` | zero track elimination, path tracing with subspace merging; no symmetry |
| `relaxation.m`, `equilibrium.m`, `thermalize.m` | `spynach.relaxation`, `spynach.equilibrium`, `spynach.thermalize` | Redfield (isotropic, axial, rhombic diffusion; diagonalisation-free integral), `t1_t2`, `damp`, `SRFK` (`rlx_scalar.m`), `SRSK`; `rlx_keep`, `rlx_dfs`; `zero`/`IME`/`dibari` thermalisation |
| `liquid.m`, `crystal.m`, `powder.m`, `singlerot.m`, `fourdif.m`, `acquire.m` | `spynach.liquid`, `spynach.crystal`, `spynach.powder`, `spynach.singlerot`, `spynach.contexts.fourdif`, `spynach.acquire` | both formalisms (MAS in Liouville space); relaxation and exchange kinetics; powder averages over Spinach's grids (shipped as `spynach/grids`); MAS in Fokker-Planck form with the Fourier spectral derivative; no decoupling |
| `ft_axis.m`, `axis_1d.m`, `plot_1d.m`, `plot_2d.m`, `contspacing.m`, `apodisation.m` | `spynach.plotting`, `spynach.apodisation` | all axis units; interactive Plotly 1D and 2D contour figures; window functions except `bad-z*`; no `derivative` |
| `etc/molecules/strychnine.m` | `spynach.molecules.strychnine` | the 22-proton system (shifts, couplings, coordinates); 13C/15N not ported |
| `cosy.m`, `noesy.m`, `hsqc.m`, `decouple.m`, `homospoil.m`, `dilute.m`, `kill_spin.m` | `spynach.cosy`, `spynach.noesy`, `spynach.hsqc`, `spynach.decouple`, `spynach.homospoil`, `spynach.dilute`, `spynach.kill_spin` | COSY, States-TPPI NOESY, echo-antiecho HSQC; analytical decoupling; isotopomer sums |

Anything not ported raises `NotImplementedError`; inputs are never silently
ignored.

## Differences from Spinach

- **Spin indices are 0-based** (`operator(spin_system, ["Lz", "L+"], [0, 1])` is
  Spinach's `operator(spin_system, {'Lz','L+'}, {1,2})`). Isotope names and
  `'all'`/`'nuclei'`/`'electrons'` work as in Spinach.
- **Product vs. sum operators** are distinguished by the type of the first
  argument: a string (`"Lz"`) gives a sum over spins, a list (`["Lz", "L+"]`)
  a product operator.
- **Spin system objects are frozen dataclasses**; `basis()` and `assume()`
  return modified copies, as Spinach returns a modified struct.
- **Time series are 1-D arrays** (MATLAB returns `n x 1` columns).
- **`evolution()` runs serially**; Spinach's Kuprov-Edwards parallel split
  computes the same observable.

## Beyond the port

Add-ons originating here, with no Spinach counterpart, each with an example in the guide
[Beyond the port](docs/beyond-the-port.md):

- **`ctps`**, a separate module (`import ctps`, no dependency on `spynach`) that
  ports CCCP.m and CCCP++, the coherence transfer pathway programs: every
  pathway a sequence allows, weighted by phase cycle, gradients, diffusion and
  pulses, with an interactive pathway explorer ([docs/ctps.md](docs/ctps.md),
  `src/ctps/README.md` for how to remove it without touching Spynach).
- **Non-ideal pulses** (perfect 180°, imperfect 180°, small flip angle) after
  J. Chem. Phys. 113, 979 (2000), and a **cogwheel cycle search** by exact
  integer arithmetic (`find_cycle`, `coverage`, `hcl_cycle`) plus the DOTCOPS
  crusher objective (`crusher_scheme`), in `ctps`.
- **Interactive Plotly figures** from `spynach.plotting`, `plot_fid`,
  `plot_time_series` and `plot_trajan` with no Spinach equivalent, and
  `figure_png` for static copies.
- **Verification tooling**: `spynach.oracle.compare` for checking any result
  against a Spinach run, the generated parity page `docs/parity.md`, and the
  timed, profiled comparison protocol in `comparison/`.
- **Executable documentation**: guides whose code runs in the tests, notebooks
  generated with outputs, a Colab quick start, and contributed notebooks in
  `user_py/`.
- **Browser-based execution**: [try.html](https://ajerschow.github.io/spynach/try.html)
  runs Spynach and `ctps` in the browser through Pyodide, with nothing
  installed, and plots spectra, FIDs and 2D contours from an editor
  preloaded with the guides' examples (contributed by Moritz Zaiss).

## Accuracy notes

Every parity comparison is relative with zero absolute tolerance, at 1e-14
for operators and states, 1e-12 for Hamiltonians and 1e-9 for anything
through a propagator or an FID; `docs/parity.md`, regenerated from a run
of the suite by `oracle/parity_summary.py`, lists every fixture case,
tolerance and measured difference. To check a workflow of your own against
a Spinach run, see the section
[Checking your own workflow against Spinach](docs/coming-from-spinach.md#checking-your-own-workflow-against-spinach).

- Spin operators match Spinach to 1 ulp: MATLAB evaluates `2^(-1/2)` one ulp
  below the correctly rounded value, so Spinach's `Lz` is `0.4999999999999999`.
- Spinach rounds Hamiltonian elements to a 1e-10 rad/s grid (`clean_up`) after
  summing terms in random order, so Spinach itself reproduces only to about
  one grid step per element; FIDs therefore agree to ~1e-10 x acquisition time.
- The Taylor propagator's error scales with `tols.prop_chop` and the number of
  squarings: about 7e-8 relative at the default 1e-10 with 7 squarings.

## Development

```
PYTHONPATH=src python -m pytest -q
```

Parity tests skip when fixtures are absent (`$SPYNACH_FIXTURES`, default
`/fixtures`). A caveat for containers: when the CPU quota is below the
visible core count (the development container sees 12 cores under a 4-CPU
quota), OpenBLAS oversubscribes the quota and dense LAPACK calls run ~1000x
slower (a 256x256 `eigh` takes 10 s instead of 10 ms). Set
`OPENBLAS_NUM_THREADS` to the quota there, for tests and especially for
benchmarks; on an ordinary machine nothing needs to be set.

To test on a macOS host outside the container, install
[uv](https://docs.astral.sh/uv/) (`brew install uv`) and run
`scripts/mac-test.sh` (extra arguments go to pytest). The container's `.venv`
does not work on the host, so the script keeps a separate `.venv-mac` and
syncs it with `pyproject.toml` before every run. Source edits need no
reinstall. Point VS Code's interpreter at `.venv-mac/bin/python`.
`scripts/mac-compare.sh` runs the Spynach side of a comparison case in
the same environment (with the `compare` extra), so that both sides of a
report are timed on the same machine.

Releases are tagged `vX.Y.Z` and listed in [`CHANGELOG.md`](CHANGELOG.md);
`spynach.__version__` gives the installed version, and
`pip install "spynach @ git+https://github.com/ajerschow/spynach.git@v0.2.0"`
installs a tagged one. To release: set the version in `pyproject.toml`,
`src/spynach/__init__.py` and `CITATION.cff` (a test keeps them equal), add
the `CHANGELOG.md` entry, refresh `uv.lock`, run the checks above plus
`python docs/build_notebooks.py --check` and
`python oracle/parity_summary.py --check`, commit, tag with
`git tag -a vX.Y.Z -m "Spynach X.Y.Z"`, push with `--follow-tags`, and draft
the GitHub release from the tag with the changelog entry as its notes.

Code style is enforced by [ruff](https://docs.astral.sh/ruff/) (`pip install
-e ".[dev]"`), configured in `pyproject.toml`: 120-character lines, import
sorting, pyupgrade and bugbear checks. Before committing, run

```
ruff format && ruff check
```

Spinach's own matrix names (`I`, `Q`, `L`, `A`, `B`, ...) are kept where a
function mirrors a Spinach function, so the ambiguous-name check is off.

Every user-facing change comes with user documentation in the same
commit; see [`docs/README.md`](docs/README.md) for the conventions. After
editing a guide, regenerate its notebook with
`python docs/build_notebooks.py`; `tests/docs/` fails if it is stale.

`index.html` at the repository root is the landing page served by GitHub
Pages from the root of `main`, and `try.html` beside it runs Spynach in the
browser: `worker.js` loads Pyodide in a web worker and fetches the Python
sources from its hard-coded `SOURCE` list (a test keeps that list equal to
`src/spynach` and `src/ctps`). Both pages are served (Settings > Pages > Deploy from a branch >
`main`, `/ (root)`; `.nojekyll` keeps Pages from preprocessing the tree).
It is hand-maintained: when a guide or a comparison case is added, add
its row or card there. Reports are linked relatively, so they are served
by Pages; guides and notebooks link to GitHub and nbviewer. Its hero
figure is `assets/Spynoza.jpg`. `assets/liouvillian-strychnine*.png`
are sparsity patterns of the strychnine NOESY Liouvillian rendered by
`scripts/landing-figure.py` (needs Pillow and Matplotlib, which the
`compare` extra brings in), kept as alternatives; the script embeds the
full-matrix mask into the page when the page carries its SVG mask element.

`src/spynach/_isotope_data.py` is generated from Spinach's `spin.m` by
`porting/tools/gen_isotopes.py`; do not edit it by hand.
`src/spynach/grids/*.npz` are generated from Spinach's `kernel/grids` by
`porting/tools/gen_grids.py`. `examples/` holds full-size translations of
Spinach example scripts for side-by-side comparison with MATLAB (see
[`docs/README.md`](docs/README.md#example-scripts)); `comparison/` holds the
protocol for timed, profiled Spinach-vs-spynach runs with an HTML report
([`comparison/README.md`](comparison/README.md)). `porting/tools/bench_strychnine.py` times the
kernel on the 22-proton strychnine system (12 316 states at `IK-2`,
proximity level 3): basis 0.04 s, Hamiltonian 0.1 s, Redfield
relaxation 2.4 s, pulse-acquire 2.1 s, COSY 32 x 64 3.3 s in this
container.

## Contributors

- Florin Teleanu: Spinach scripts in `user_ml/`
  (two-site 7Li exchange at equilibrium, zero- to ultralow-field PF6⁻
  spectrum), translated in `examples/user_ml/` and used as comparison cases,
  and the project page's hero figure (`assets/Spynoza.jpg`).
- [Moritz Zaiss](https://github.com/mzaiss) (MRsources): the in-browser playground
  (`try.html`, `worker.js`, Spynach in Pyodide) and the orientation
  `progress` callback of `powder` and `singlerot` (pull request #1).

New contributors are added here and in the "With thanks" section of
`index.html` when their first contribution is merged; the CHANGELOG names
them in the release entry.

## Licence and attribution

Spynach is released under the MIT licence (`LICENSE`). It is derived from
[Spinach](https://spindynamics.org) by Ilya Kuprov and co-workers, also MIT
licensed: the kernel, molecule data, isotope table, spherical grids and
example scripts were translated from it, and Spinach's copyright and
permission notice is reproduced in `LICENSE`. Keep that notice with any
copy or substantial portion of this software.

The translation from MATLAB to Python was carried out largely by an AI
coding agent (Claude, Anthropic) under the direction of Alexej Jerschow,
who reviewed the results; commits made by the agent are marked as such in
the git history. Correctness rests on the test suite's comparison with
Spinach reference data, not on the method of authorship (see `LICENSE`).

Results obtained with Spynach should cite Spinach: H. J. Hogben,
M. Krzystyniak, G. T. P. Charnock, P. J. Hore, I. Kuprov, *Spinach: a
software library for simulation of spin dynamics in large spin systems*,
J. Magn. Reson. 208, 179 (2011), together with the method papers that the
Spinach documentation lists for the features used (restricted state
spaces, relaxation theories, Fokker-Planck contexts).

Scripts contributed by users in `user_ml/` (Spinach) and `user_py/`
(Python notebooks and scripts) remain the copyright of their authors
and are redistributed, with their agreement, under the same MIT licence
(see the README in each directory).
