# Spynach documentation

| guide | notebook | covers |
|---|---|---|
| [Hilbert space](hilbert-space.md) | [hilbert-space.ipynb](notebooks/hilbert-space.ipynb) · [nbviewer](https://nbviewer.org/github/ajerschow/spynach/blob/main/docs/notebooks/hilbert-space.ipynb) | liquid-state NMR in Hilbert space (`zeeman-hilb`): setup, the simulation pipeline, pulse sequences, tolerances, parity with Spinach |
| [Liouville space](liouville-space.md) | [liouville-space.ipynb](notebooks/liouville-space.ipynb) · [nbviewer](https://nbviewer.org/github/ajerschow/spynach/blob/main/docs/notebooks/liouville-space.ipynb) | the spherical-tensor formalism (`sphten-liouv`): basis descriptor, state-space restriction (`IK-0/1/2`), filters, chemical subsystems |
| [Solid state](solid-state.md) | [solid-state.ipynb](notebooks/solid-state.ipynb) · [nbviewer](https://nbviewer.org/github/ajerschow/spynach/blob/main/docs/notebooks/solid-state.ipynb) | anisotropic interactions: tensors from eigenvalues and Euler angles, dipolar couplings from coordinates, quadrupolar couplings, the `Q` components and `orientation`, `crystal` and `powder` contexts, grids, assumption sets |
| [Relaxation](relaxation.md) | [relaxation.ipynb](notebooks/relaxation.ipynb) · [nbviewer](https://nbviewer.org/github/ajerschow/spynach/blob/main/docs/notebooks/relaxation.ipynb) | Redfield theory for rotational diffusion, T1/T2 and damping models, retention policies, thermal equilibrium and thermalisation, transient NOE |
| [Magic angle spinning](magic-angle-spinning.md) | [magic-angle-spinning.ipynb](notebooks/magic-angle-spinning.ipynb) · [nbviewer](https://nbviewer.org/github/ajerschow/spynach/blob/main/docs/notebooks/magic-angle-spinning.ipynb) | the Fokker-Planck treatment of sample rotation: rotor phase grid and spectral derivative, the `singlerot` context, sidebands, rotor echoes |
| [Pulse sequences](pulse-sequences.md) | [pulse-sequences.ipynb](notebooks/pulse-sequences.ipynb) · [nbviewer](https://nbviewer.org/github/ajerschow/spynach/blob/main/docs/notebooks/pulse-sequences.ipynb) | writing sequences, `apodisation` and `plot_2d`, decoupling and homospoil, COSY, NOESY, HSQC with isotopomer dilution |
| [Coherence transfer pathways](ctps.md) | [ctps.ipynb](notebooks/ctps.ipynb) · [nbviewer](https://nbviewer.org/github/ajerschow/spynach/blob/main/docs/notebooks/ctps.ipynb) | the separate `ctps` module: pathway enumeration, phase cycling, gradients, diffusion, non-ideal pulses, cogwheel cycle search, the pathway explorer |
| [Coherence transfer pathways: worked examples](ctps-examples.md) | [ctps-examples.ipynb](notebooks/ctps-examples.ipynb) · [nbviewer](https://nbviewer.org/github/ajerschow/spynach/blob/main/docs/notebooks/ctps-examples.ipynb) | a cookbook for `ctps`: double-quantum filter, imperfect spin echo, stimulated echo, NOESY-sized enumeration, HSQC, MQMAS nested versus cogwheel, multi-echo trains, cogwheel plus gradients, MRS crushers; then every check of the module's tests as a runnable example |
| [Beyond the port](beyond-the-port.md) | [beyond-the-port.ipynb](notebooks/beyond-the-port.ipynb) · [nbviewer](https://nbviewer.org/github/ajerschow/spynach/blob/main/docs/notebooks/beyond-the-port.ipynb) | features with no Spinach counterpart, each with an example: `ctps`, non-ideal pulses and cogwheel search, interactive figures and `figure_png`, `compare` and the parity page, executable documentation |
| [Coming from Spinach](coming-from-spinach.md) | [coming-from-spinach.ipynb](notebooks/coming-from-spinach.ipynb) · [nbviewer](https://nbviewer.org/github/ajerschow/spynach/blob/main/docs/notebooks/coming-from-spinach.ipynb) | what carries over from Spinach and what differs: spin systems, basis, operators, Hamiltonians, parameters, calculation modalities, processing |
| [EPR spin systems with `epreasy`](epreasy.md) | [epreasy.ipynb](notebooks/epreasy.ipynb) · [nbviewer](https://nbviewer.org/github/ajerschow/spynach/blob/main/docs/notebooks/epreasy.ipynb) | the EasySpin port beside Spynach: spin systems as dicts with EasySpin's fields, `sop` and `stev`, `ham` with the field split, `levels`, `resfields`, line shapes and grids |
| [EPR simulations with `epreasy`](epreasy-simulators.md) | [epreasy-simulators.ipynb](notebooks/epreasy-simulators.ipynb) · [nbviewer](https://nbviewer.org/github/ajerschow/spynach/blob/main/docs/notebooks/epreasy-simulators.ipynb) | `pepper` for powders and crystals, `garlic` for solutions, `esfit` for fitting, `salt` for ENDOR, `curry` for magnetometry, `chili` for slow motion |
| [Coming from EasySpin](coming-from-easyspin.md) | [coming-from-easyspin.ipynb](notebooks/coming-from-easyspin.ipynb) · [nbviewer](https://nbviewer.org/github/ajerschow/spynach/blob/main/docs/notebooks/coming-from-easyspin.ipynb) | what carries over from EasySpin to `epreasy`, what differs, and what is not ported |

The notebooks are saved with their outputs, plots included. On GitHub the
guides show each figure as a static image from `docs/figures/` and the
notebooks show the static copy saved with every figure; the nbviewer links
show the interactive figures. Open the notebooks
with `jupyter lab docs/notebooks/` after `pip install -e ".[docs]"`, or in
VS Code with the Python and Jupyter extensions; the plots are interactive
in both. (In a container whose CPU quota is below its visible core count,
set `OPENBLAS_NUM_THREADS` to the quota first; see the setup section of the
[Hilbert space guide](hilbert-space.md#setup).) The project page
[ajerschow.github.io/spynach](https://ajerschow.github.io/spynach/)
links every notebook through nbviewer as well.

[`spinach-credits.md`](spinach-credits.md) is a generated table of the Spinach authors of every Spynach
function that translates or derives from one (names from the Spinach developer team page, taken from the headers
of the original functions); `porting/tools/gen_spinach_authors.py` writes it, and the same lines, into the
docstrings.

[`parity.md`](parity.md) is not a guide but a generated table: every fixture
case, tolerance and measured difference of the parity suite, written by
`oracle/parity_summary.py` from a test run together with the summary
paragraph on the project page; regenerate it after adding fixtures or
parity tests.

[`colab/quick-start.ipynb`](colab/quick-start.ipynb) is a stand-alone
notebook for [Google Colab](https://colab.research.google.com/github/ajerschow/spynach/blob/main/docs/colab/quick-start.ipynb):
it installs Spynach from GitHub and runs the project page's two-spin
example with a plot. It is the one hand-written notebook (it is not
generated from a guide and is saved without outputs, since Colab runs it);
keep its example in step with the quick start on `index.html` and in the
README. `try.html` at the repository root runs Spynach, `epreasy` and `ctps`
in the browser (Pyodide, `worker.js`) with NMR, EPR and pathway examples in an
editor, written into the page itself and run natively by
`tests/unit/test_playground_examples.py`; see the guide
[Beyond the port](beyond-the-port.md#spynach-in-the-browser). The project
site lists the guides by package on its documentation page, generated from
the table above by `scripts/build_site.py`, which files every guide under a
package; a new guide needs a row here and an entry in that script's groups.

To experiment, copy a notebook into `playground/` at the repository root
(ignored by git except for its README) and work on the copy; it uses the
same `spynach` code. The
notebooks in `docs/notebooks/` are rebuilt from the guides, so edits to them
are overwritten; if you run one there, `git restore docs/notebooks/` discards
the changed outputs.

## Example scripts

`examples/` at the repository root mirrors Spinach's `examples/` tree;
`examples/user_ml/` holds translations of the contributed scripts in
`user_ml/` (see its README). Python code contributed by users, such as
the 23Na triple-quantum build-up notebook, lives in `user_py/` at the
repository root (see its README); it is not generated from a guide and
not run by the tests.
Each script is a translation of the Spinach example of the same name at
the Spinach size (the guides above run reduced versions in seconds), and
is meant for a side-by-side comparison with MATLAB: it writes its FIDs
and spectrum as `.npy` files and its figure as `.html` into `out/` next
to itself (git-ignored), and `--compare FILE.mat` reports the normwise
relative error against a `.mat` file saved by the Spinach script (the
script's docstring gives the lines to add in MATLAB). The scripts are not
run by the test suite. For a timed and profiled comparison of both sides
with an HTML report, use the protocol in
[`comparison/README.md`](../comparison/README.md).

| script | Spinach example | full size in this container |
|---|---|---|
| `examples/nmr_liquids/noesy_strychnine.py` | `examples/nmr_liquids/noesy_strychnine.m` (22 protons, `IK-2`, Redfield, 500 ms NOESY) | 2 min 50 s at 512 x 512 points (12 316 states, 4 cores); `--npoints 64 --zerofill 256` runs in 15 s |
| `examples/relaxation_theory/maz_noesy_2.py` | `examples/relaxation_theory/maz_noesy_2.m` (methylaziridine, 7 protons + 14N, CSA tensors, 14N quadrupole, Redfield with scalar relaxation of the first and second kind, `IK-2`, 2 s NOESY) | 28 min on the 12-core Mac at 256 x 256 points (19 314 states; 34 min in this container; Spinach: 14 min on the same Mac), nearly all of it the SRFK integral, which exponentiates a 38 628-dimensional auxiliary matrix with 26 squarings as Spinach does; `--npoints` only shortens the NOESY stage |
| `examples/user_ml/li_exchange_equilibrium.py` | `user_ml/Li_exchange_equilibrium.m` (contributed: two-site 7Li exchange with quadrupolar and shift tensors, thermal state equilibrated under exchange, IME-thermalised Redfield relaxation, 2 s trajectory analysed with `trajan` and `stateinfo`) | about a second at full size (31 states: the two sites are separate chemical subsystems) |
| `examples/user_ml/zulf_pf6_1d.py` | `user_ml/ZULF_PF6_1D.m` (contributed: PF6- at 2.7 uT, adiabatic-drop state from file, selective pulse, 16 384-point ZULF spectrum) | 36 s at 16 384 points (16 384 states; state-space reduction brings the subspaces below `krylov_tol`, so the propagator path is used) |

## Conventions for contributors

**Every user-facing change ships with user documentation in the same
commit.** A newly ported function, formalism, context, pulse sequence,
option, or behaviour change is not done until the guides describe it.

1. **Markdown is the source.** Guides are `docs/<name>.md`. Every
   ```` ```python ```` block is executed by `tests/docs/test_guides.py`, in
   order and in one namespace per guide, so examples must run as written
   and show what they claim with a `print` or a plot. Verification
   belongs in `tests/`, not in the guides, so they carry no `assert`
   statements.
2. **Notebooks and figures are generated.** `python docs/build_notebooks.py`
   writes `docs/notebooks/<name>.ipynb` for every guide, runs it in a fresh
   kernel, and saves it with its outputs; each Plotly figure is saved with a
   PNG copy (what GitHub shows) and written to `docs/figures/`. Never edit
   a notebook or a file in `docs/figures/` by hand; the tests fail if a
   notebook is stale, was saved without outputs, or has no guide, and if a
   figure file is missing.
3. **Plots** go in their own blocks, use `spynach.plotting` (or Plotly
   directly), end with the figure as the last expression so the notebook
   shows it, and define no names that later blocks need: without Plotly
   installed, the tests skip them. Each plotting block is followed by an
   image line `![caption](figures/<guide>-<what>.png)` that names the file
   the builder writes for it; the guide shows the image on GitHub and the
   notebook drops the line, since the cell above shows the figure. Show the
   results a section computes: spectra, FIDs, and comparisons between them.
4. **Keep examples fast** (a few seconds per guide). If a faithful
   translation of a Spinach example is slow, reduce its size and say what
   was changed and how to restore it.
5. **Compare with Spinach.** For every ported feature, update
   [Coming from Spinach](coming-from-spinach.md): what behaves the same,
   what differs (call syntax, defaults, return types, units, numerics), and
   what is still not ported. Move items out of the "not ported" lists of
   both guides and the README status table as they are ported.
6. **Name the spin system `spin_system`**, as Spinach does, in guides,
   README, examples and tests; when several coexist in one namespace, add
   a qualifier (`spin_system_hilb`, `spin_system_ik2`, `spin_system_plain`).
   Helper functions that build one are verbs or nouns (`make_spin_system`,
   `hn_pair`), never the bare name.
7. **Split guides by theme as they grow.** When a guide passes about 500
   lines, or a new formalism, context, or experiment family arrives, give it
   its own guide (for example `liouville-space.md`, `relaxation.md`,
   `solid-state.md`) rather than appending, and add it to the table above.
   Its notebook follows automatically.
7. Use 0-based spin numbers, Spinach's field names and units, and the
   verified tolerances; link to sections instead of repeating them.
