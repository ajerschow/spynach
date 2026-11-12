# Changelog

Releases are tagged `vX.Y.Z` on GitHub; the version is `spynach.__version__`.

## Unreleased

- Pulse programs: explicit mode (`program.explicit()`, the phase cycle step by step) takes gradients, on a lattice of Fourier modes along the gradient axis instead of a spatial grid: the state is one vector per wavevector `n k0`, a gradient moves each coherence-order block along the lattice, detection weights every mode with the sample envelope; `Program(gradient_unit=...)` sets the base area (default: the smallest present), `output="modes"` returns the per-mode FIDs (`ModeFids`). All gradients must be collinear with areas on one integer grid. Checked against the pathway runner on every gradient program of the test file (inclined axis, read gradient, diffusion, indirect dimension, phase cycle, the 23Na filter) to 1e-10; guide section in `docs/pulse-programs.md`. Pathways mode is unchanged.

## 0.7.0 - 2026-11-11

- `easyspyn` data files: `eprload` reads Bruker BES3T pairs (`.DSC`/`.DTA`: 1-D to 3-D, real or complex, several values per point, `.XGF`/`.YGF`/`.ZGF` axis files, the scaling letters `n P G T c`), Bruker ESP/WinEPR pairs (`.par`/`.spc`: every axis rule, the complex and 2-D flags, EMX axes, power sweeps, WinEPR floats, the scaling keys) and text columns, as `(x, y, params)` with EasySpin's parameter parsing; the other formats raise `NotImplementedError`. `eprsave` writes BES3T pairs as EasySpin does. `exp_from_params` builds the `exp` dict of the simulators and `esfit` from a loaded file (`mwFreq`, `Range` in mT, `nPoints`, `Harmonic`, `ModAmp`, `Temperature`), so a fit of a measured spectrum is three lines. Unit tests on files written in the test from EasySpin's byte layouts (`tests/easyspyn/test_eprfiles.py`); the parity dump `esdump_eprload.m` (BES3T files from `eprsave`, hand-written ESP files, user-supplied instrument files in `oracle/matlab/easyspin/dumps/eprload_files/`) and `tests/easyspyn/test_parity_eprload.py` are prepared, pending the MATLAB run. Guide section in `docs/easyspyn-simulators.md`, mapping table in `docs/coming-from-easyspin.md`; playground example `try.html#ex-epr-eprload` fits a bundled BES3T file (`src/easyspyn/data/example_cw.DSC/.DTA`, a `pepper` spectrum with noise and the parameter layer of a cw experiment), and the playground now ships binary package data.
- Pulse programs, gradients, phase cycling (beta) (`spynach.program`, no Spinach counterpart; guide `docs/pulse-programs.md`; the interface may still change): gradients without a spatial grid. Coherence orders follow the textbook convention (the detected order is -1, the mirror image of Spinach's `L+` coil, whose signal is the complex conjugate); phase lists per pulse, nested, cogwheel (`("cog", nu)` with `cycle_length`) and block cycles. a sequence written as blocks (`Pulse` with phase lists or cogwheel winding numbers, `Delay`, `Gradient` in T/m, `Evolve`/`Refocus` for indirect dimensions, `Homospoil`, `Block` phases, `Acquire` with receiver phases and an optional read gradient) runs as one branch of the density matrix per coherence pathway that survives the phase cycle (pathways and weights from `ctps` through `Program.to_ctps`/`select`), each branch carrying its gradient k-vector and diffusion integral: gradient selection, crushers and read-out gradients are exact sample envelopes with no spatial grid, PGSE diffusion is Stejskal–Tanner with the finite-width term, per-pathway FIDs and echo times are available (`output="pathways"`), `check_leakage` measures the secular assumption, and `Program.explicit()` runs the cycle step by step as the reference. A `Program` is a sequence callable for the contexts. Kernel additions: `filters.coherence_mask`, `isotope_orders`, `orders_present`, `leakage`; `coherence` itself is unchanged. Checked against a hand-written cycle loop (nested, cogwheel, block and heteronuclear cycles), the COSY, NOESY and HSQC ports, an explicit average over sample positions and Stejskal–Tanner (`tests/unit/test_program.py`). Playground examples `try.html#ex-dqf-cosy` (the four-step cycle run as pathways or step by step, timed), `#ex-phase-cycle` (a stimulated echo under a nested 16-step cycle or the shortest cogwheel found by ctps), `#ex-gcosy` (P- or N-type gradient selection with the kept and crushed pathways), `#ex-pathway-echoes` (pathways read out as echoes under a read gradient) and `#ex-pgse` (diffusion attenuation against Stejskal-Tanner). A 23Na triple-quantum filter under slow-motion quadrupolar relaxation (guide section, `tests/unit/test_program.py`): the six-step cycle run as pathways against the explicit cycle, the first phase of the cycle selecting the transverse route (through T(3,±1)) or the longitudinal one (through T(3,0)), and the gradient-selected version (gradients 1 : 1 : 4 after the first pulse, after the third and before acquisition, no phase cycle: the longitudinal route, the mirror pathway and the magnetisation recovering during the gradients are dephased) checked pathway by pathway against the cycle; playground examples `#ex-na23-tqf-cycle` and `#ex-na23-tqf-gradients`, with the build-up curves per route against the T(3,1) tensor projection.
- Playground: the plots are drawn with plotly.js (pinned 4.1.2 from cdn.jsdelivr.net) instead of the hand-drawn canvas: zoom by box or wheel, pan, double-click reset, hover read-out, legend toggling and PNG download, in the look of the notebooks' figures; the 2-D contour plots keep Spinach's level spacing through the level-index mapping of `plot_2d`. The chord diagram stays SVG.
- Playground: the radiation damping example plots the x, y and z magnetisation against time (it showed m_z alone).
- Playground: choosing an example from the drop-down updates the URL fragment, so the address bar is a direct link to it (`try.html#ex-…`, as the docs link examples).
- Playground and `user_py/na23_tqf_buildup.ipynb`: the second 90° pulse of the 23Na build-up is applied about y (it was about x, which turns the y-type rank-3 tensor into even orders only, so the "T(3,3) after a 90° pulse" trace was the T(3,0) route, not the conversion of T(3,1)); the notebook is re-executed.
- Playground: the 23Na triple-quantum build-up of the contributed notebook `user_py/na23_tqf_buildup.ipynb` as the example `try.html#ex-na23-tqf` (group "Relaxation (spynach)": quadrupolar coupling, asymmetry, correlation time, field, temperature and preparation time in the form; the rank-3 tensor, the triple-quantum coherence after the second pulse and the Lz recovery, with the biexponential check printed).
- Code style: ruff 0.17 formats the Python blocks of the Markdown guides and lints notebooks; the guides' code blocks are reformatted accordingly (their notebooks regenerated, nothing computed changes) and the contributed 23Na notebook is lint-clean.

## 0.6.0 - 2026-11-08

- Playground: the radiation damping example has a line-broadening parameter (exponential window on both FIDs) and a shift-difference range up to 1 kHz; the pulse example's window is the same "Line broadening" control as the other 1-D examples (it was a decay time in points).
- Radiation damping (no Spinach counterpart): `spynach.radiation_damping` builds the state-dependent generator of a spin system under the feedback of the probe (rate 1/T_rd at the reference magnetisation, tuning phase; isochromat ensembles with `line_weights` for inhomogeneous lines; Liouville and Hilbert space) for `iserstep`; `rd_rate` and `rd_phase` convert Q, filling factor, concentration and detuning. Checked against the Bloembergen-Pound closed form, the classical Bloch-Maxwell equations of an ensemble, the linearised damping rate and frequency pulling of a detuned probe, basis restriction and the two formalisms (`tests/unit/test_radiation_damping.py`). The guide `docs/radiation-damping.md` adds the closed form, the detuned probe, the inhomogeneous line and a strongly coupled pair against the line-by-line model; the playground example `try.html#ex-rd` runs the pair.
- `iserstep`, the integrator for generators that depend on the state (port of Spinach's `kernel/pulses/iserstep.m`: `PWCL`, `LG2`, `LG4`, `LG4A`, `RKMK4`, `RKMK-DP5`, `RKMK-DP8`, `RKMK-RKF45`), called as `iserstep(spin_system, L, t, method, rho, dt)`; `LG4` in Liouville space only. New guide `docs/radiation-damping.md` with the Bloch-Maxwell model of Spinach's `product_quadratures_2` example, the example script `examples/fundamentals/product_quadratures_2.py`, the comparison case `product_quadratures_2`, and the fixture `iserstep_hilb_1spin` (FR-008) for parity with Spinach's own integrator test (every method to 1e-12; the comparison run agrees with Spinach to 1e-14 in the trajectory and the error table, `comparison/runs/product_quadratures_2/`). Nothing existing changes.
- Attribution: `LICENSE` names the EasySpyn port and ctps (a port of the earlier CCCP.m and CCCP++ programs) in its header; the README says how to credit EasySpyn (EasySpin publications), the Spynach and EasySpyn ports (this repository) and ctps (the CCCP papers), and lists the Claude and Alexej Jerschow collaboration among the contributors; Alexej Jerschow's homepage is linked in the credits; Florin Teleanu is acknowledged in the docstrings of the relaxation pathways module, its chord plots and its tests.
- Site: a link at the bottom of the landing, Spinach, EasySpin, Extras, Documentation and playground pages switches between the light and dark layout (`assets/theme.js`); the layout is dark by default, and the choice of light is remembered in the browser (without JavaScript the pages follow the system setting). The playground redraws its plots in the chosen layout.
- Landing page: a click on the Spynoza picture is an RF pulse: the picture rings down like a free induction decay (a damped cosine refracting the image, with slight colour fringing) while the receiver trace is drawn from the click point (WebGL, up to four at once; skipped with reduced motion or without WebGL).
- The playground opens an example in its Parameters view (the form) when it has one; "Prefer code" switches to the
  source and is remembered, and `try.html?view=code` opens the source. (`?view=params` still works.)
- The playground's help text is a popup behind an info button (`i`, next to Run and Stop; `try.html?help` opens it),
  and the line under the editor is down to "Ctrl+Enter runs. Press i for help."

## 0.5.0 - 2026-10-28

- Spynach has a PyPI distribution, `spynach` (`pip install spynach`; Python 3.11 or newer, Numba included). It holds
  `spynach` and `ctps`; `easyspyn` stays its own distribution, which `pip install "spynach[epr]"` adds (the extra pins
  the matching version, `tests/unit/test_version.py`). Before, the GitHub distribution bundled all three. A GitHub
  Actions workflow (`.github/workflows/publish-spynach.yml`) publishes it through Trusted Publishing when a release
  tagged `vX.Y.Z` is published, after the reviewer of the `pypi` environment approves; EasySpyn's releases
  (`easyspyn-vX.Y.Z`) do not start it. The project file gained the readme, classifiers and URLs shown on the PyPI page.
- Relaxation pathways (`spynach.relaxation_pathways`, new module `spynach.pathways`): the Bloch-Redfield-Wangsness
  relaxation matrix of a spin system between (collective) product operators, split by mechanism (DD, CSA, Q, JA) and
  by every cross-correlation between two mechanisms, with each element as a rate and as an exact SymPy expression in
  the interaction strengths and `J(w)`; `secular` selects the per-isotope, total-coherence or lab-frame form, `ranks`
  adds the higher-rank operators of quadrupolar spins, `coherence` keeps one coherence order. `plot_pathways` in
  `spynach.plotting` draws the result as a chord diagram (ribbon width = rate, colour = mechanisms). Checked against
  `spynach.relaxation` and textbook rates (`tests/unit/test_pathways.py`); guide `docs/relaxation-pathways.md` with
  11BF4- as the example. SymPy is a new optional dependency (`symbolic` extra, also in `test` and `docs`).
  `figure_png`'s Matplotlib fallback draws filled outlines, rotated text and equal-aspect axes.
- `spynach.plotting.chord_layout`: the chord diagram's geometry as plain lists, shared by `plot_pathways` and the
  browser playground. The playground has a relaxation example (`try.html#ex-pathways`, 11BF4- with its parameters in
  the form) and a `show_pathways(pathways)` helper that draws the diagram as SVG with the rates and expressions on
  hover; the worker loads SymPy from Pyodide's distribution the first time a run needs it.
- Landing page: the hero is a slidable gallery; the first figure is a live animation of interfering spin-phase spirals
  (`assets/hero-gallery.js`), followed by a chord diagram of the 11BF4- relaxation pathways (`assets/relaxation-chord.svg`, written by `scripts/make_hero_chord.py`; a click sends a pulse through the network), a k-space halftone and a COSY-contour animation, and the Spynoza picture. Without WebGL or with reduced motion the first slide is a still frame of the animation (`assets/spin-vortex-frame.png`).
- Link previews and icons: every page of the site carries Open Graph and Twitter card tags with a 1200 x 630 card
  made from the spiral (`assets/social-card.png`) and a tab icon (`assets/favicon.svg`, with PNG fallbacks). The tags
  are generated by `scripts/build_site.py` from each page's title and description.

## 0.4.0 - 2026-10-27

- The browser playground (`try.html`) has a Parameters view: a form of the main settings of an example, each with its
  unit and a description, toggled with "Prefer GUI" / "Prefer code" (shown only for examples that have a form). The code stays the only state: a form edit rewrites
  one number in the source, an edit in the source updates the form. Parameters are plain assignments with a
  `# GUI: label | unit | description | range` comment, read by `assets/playground-params.js`; every example has them,
  and `tests/unit/test_playground_params.py` runs every example at the ends of each parameter's range. When the code
  is rewritten under the form, a parameter the code no longer reads is greyed out, a name assigned twice is read-only,
  and if no parameter lines are left a notice offers "Restore example".

- Numba is installed by default: `pip install easyspyn` (and the git install of `spynach`) bring it, so the compiled
  kernels of both packages are used without the `fast` extra, which stays as an empty alias. It is skipped on
  interpreters other than CPython, and `EASYSPYN_NUMBA=0` (new) or `SPYNACH_NUMBA=0` force the NumPy and SciPy code.
- Quick-start instructions are prominent on the landing page and at the top of the README: `pip install easyspyn`, the git install of Spynach, and the options that need no installation (the browser playground, Colab notebooks for EPR and NMR, Binder via `.binder/requirements.txt`).

## 0.3.0 - 2026-10-26

- The EasySpin port is called EasySpyn (formerly EPReasy, `epreasy`) everywhere: the import name and the PyPI
  distribution are both `easyspyn`, the source is `src/easyspyn`, the tests `tests/easyspyn`, the guides
  `docs/easyspyn*.md`, and the fixture variable `EASYSPYN_FIXTURES` (the old `EPREASY_FIXTURES` is still read).
  This breaks `import epreasy` of the 0.2.0 PyPI release; nothing else in the repository uses the old name.
- The EasySpin contributors behind every function that EasySpyn translates are acknowledged by name, as the Spinach
  authors are in Spynach: an `EasySpin authors (pepper.m): ...` line in the docstring of each function (113 functions
  from 103 EasySpin files), a block per module, and the generated page `docs/easyspin-credits.md`. EasySpin's headers
  name no authors, so the lists are the people who authored commits to each original file up to release 6.0.15,
  plus authors the source names (John D'Errico for `jacobianest.m`).
  `porting/tools/gen_easyspin_authors.py` reads the history through the `gh` CLI; `tests/easyspyn/test_credits.py`
  keeps docstrings and page current offline. The shared code of both credits tools is in
  `porting/tools/credits_common.py`.
- The Spinach authors of every translated function are acknowledged by name: a `Spinach authors (propagator.m): ...`
  line in its docstring (and a module line for each module), and the generated page `docs/spinach-credits.md`
  (65 ports, 9 derived functions, 17 without a Spinach counterpart). Names come from the Spinach developer team
  page, plus three contributors it does not list (Stefan Stoll, Callum Musselwhite, Filip Vestergaard Jensen).
  `porting/tools/gen_spinach_authors.py` reads the headers of a Spinach checkout into
  `porting/spinach_authors.json` and writes the docstrings and the page; `tests/unit/test_credits.py` keeps them
  current without a checkout. Function names are unchanged.
- EasySpyn has a PyPI distribution, `easyspyn` (NumPy and SciPy only; `easyspyn[fast]` adds
  Numba), built by `scripts/build_easyspyn.py` from `packaging/easyspyn/`; the name is held for an initial 0.2.0
  release while the final name of the package is decided.
  A GitHub Actions workflow (`.github/workflows/publish-easyspyn.yml`) publishes it through PyPI's Trusted
  Publishing when a release tagged `easyspyn-vX.Y.Z` is made and approved.
- Threaded Numba kernels for the sparse linear algebra of `spynach` (`spynach.kernels`, the `fast`
  extra): the product of two sparse matrices with the rounding of `clean_up` fused in (the squarings of
  `propagator` and `expmint`), sparse-times-dense products, one fused term of the Taylor series of `step`, and the
  rounding itself, routed through `spynach.backend`. Results agree with the SciPy code to the last bit apart from
  elements within an ulp of a rounding boundary; below a size threshold, without Numba, and with
  `SPYNACH_NUMBA=0` the NumPy and SciPy code runs as before. Full-size strychnine NOESY: 88 s to 68 s, the
  rest being dense matrix products at BLAS speed; timings of the three long comparison cases are on the project site.
- `easyspyn` and `ctps` carry the distribution version, 0.2.0, as `spynach` does; tests keep the
  three equal to `[project].version`.
- `easyspyn`, a separate package beside Spynach that ports EasySpin (Stefan
  Stoll and co-workers, MIT) to Python, imported as `easyspyn` and depending
  on `spynach` (never the reverse; a test enforces it). This first step is
  the foundation: spin systems as dicts with EasySpin's fields
  (`validatespinsys`), `sop` and `stev`, `ham` with the field split and its
  components, `levels`, `erot`/`eulang`, units, constants and EasySpin's
  isotope table; verified against EasySpin 6.0.15 on five spin systems
  (`oracle/matlab/easyspin/`). Guides `docs/easyspyn.md` and
  `docs/coming-from-easyspin.md`.
- `easyspyn` resonance-field layer: `resfields` (EasySpin's adaptive spline
  modelling of the energy level diagram, matrix and hybrid methods, H/g/A/D
  strains, Boltzmann and non-equilibrium populations, excitation modes,
  photoselection; crystal symmetry P1), `resfields_eig` (the eigenfield
  method), `cubicsolve` vectorised over all cubics, the line shapes
  `gaussian`, `lorentzian`, `lshape`, `voigtian`, `convspec`, and `sphgrid`
  with its triangulation; verified against EasySpin on a nitroxide, Cu(II)
  with two 14N, Mn(II) and a radical pair.
- `easyspyn` simulators `pepper` (cw EPR of powders and crystals: matrix,
  hybrid, perturbation and eigenfield methods, orientation interpolation
  and triangle projection, strain summation, convolution, derivatives,
  field modulation) and `garlic` (isotropic solutions: Breit-Rabi and
  perturbation positions, equivalent nuclei, fast-motion widths, frequency
  sweeps), with `fastmotion`, `equivcouple`, `hamsymm`, `gridinterp`,
  `fieldmod` and the MEX kernels `lisum1i`, `projecttriangles`,
  `projectzones` in NumPy with an optional Numba path (`spynach-spin[fast]`)
  that is bit-exact with EasySpin's C code; verified against EasySpin on 32
  `pepper` and 14 `garlic` spectra.
- `easyspyn` fitting: `esfit` without the GUI (EasySpin-style parameter
  dicts for `pepper` and `garlic` or custom model functions, global fits,
  autoscaling, baselines, masks, weights, standard deviations and
  confidence intervals), the algorithms `esfit_simplex`, `esfit_levmar`,
  `esfit_montecarlo`, `esfit_genetic`, `esfit_grid`, `esfit_swarm` and
  SciPy's least squares for `lsqnonlin`, `jacobianest`, `hccm`, the data
  helpers `rescaledata`, `basecorr`, `deriv`, `addnoise`, and several
  component systems in `pepper` and `garlic` (`Sys.weight`); verified
  against EasySpin on 33 fits of the same data.
- `easyspyn` ENDOR and magnetometry: `endorfrq` (matrix diagonalisation
  with transition pre-selection, hyperfine enhancement, orientation
  selection, populations, flip angles), `endorfrq_perturb` (Iwasaki's
  first and second order), `orisel`, `anisosubsys`, `salt` (powder and
  crystal ENDOR spectra) and `curry` (magnetic moments and
  susceptibilities in SI and CGS by the operator or partition-function
  method); verified against EasySpin on 14 `endorfrq`, 23 `salt` and 14
  `curry` cases.
- `easyspyn` slow motion: `chili` solves the stochastic Liouville equation
  in the K-symmetrised Wigner basis (EasySpin's general Liouvillian method:
  Brownian diffusion, orientational potentials, MOMD, Lanczos with
  continued fraction and the direct, eigenvalue, conjugate-gradient and
  BiCGSTAB solvers, approximate and explicit field sweeps, frequency sweeps,
  post-convolution), with `wigner3j` (exact for every rank), `wigner6j`,
  `wignerd`, `tensor_cart2sph` and `tensor_sph2cart`; verified against
  EasySpin on 48 spectra and the helpers. The `easyspyn` guide is split:
  `docs/easyspyn.md` keeps the foundation, `docs/easyspyn-simulators.md`
  holds the simulators.
- Project site restructured: a one-screen landing page naming the Spinach
  port, the EasySpin port and the extras, with pages for each (`spinach.html`,
  `easyspin.html`, `extras.html`) and a documentation page listing the guides
  by package; the timing tables (Spinach vs Spynach from the comparison
  reports' new `summary.json`, EasySpin vs `easyspyn` from
  `scripts/easyspin-timings.py` on the fixtures, which now record EasySpin's
  wall clock) and the navigation are generated by `scripts/build_site.py`,
  checked by `tests/unit/test_site.py`; the comparison reports carry the
  site's stylesheet and navigation. The browser playground runs `easyspyn`
  too: eleven EPR examples (powder, solution, slow motion, ESEEM, HYSCORE,
  Mims ENDOR, cw ENDOR, magnetometry, energy levels, Cu(II) with four 14N, a
  fit with an iteration counter) in a grouped example list with deep links
  such as `try.html#ex-epr-hyscore`; every example is run natively by
  `tests/unit/test_playground_examples.py` and in Pyodide under Node by
  `scripts/pyodide-examples.mjs`. EasySpyn's kernels index with `np.intp`
  instead of `np.int64`, which the 32-bit WebAssembly NumPy refused.
- `easyspyn` pulse EPR: `saffron` with EasySpin's fast engine (ideal and
  rectangular pulses, the predefined 2-, 3- and 4-pulse ESEEM, HYSCORE and
  Mims ENDOR experiments, custom sequences with coherence transfer
  pathways, filters and phases, orientation selection, product rule,
  relaxation decays, frequency-histogram and time-domain evolution, crystals
  and components), with `sf_pathways`, `larmorfrq`, `apowin`, `fdaxis` and
  `exponfit`; verified against EasySpin on 69 experiments. The thyme engine
  (shaped pulses) and `spidyan` are not ported.
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
