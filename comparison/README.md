# Spinach vs Spynach comparison runs

A reusable protocol for timing, profiling and comparing a Spinach
calculation with its Spynach translation, ending in one HTML report per run.
The Spinach side runs in MATLAB on the host. The Spynach side runs on
the same host for the timings that go into a report
(`scripts/mac-compare.sh`), or in the container for development checks;
the report is built from the files both sides leave behind and its
provenance cards say where each side ran. Nothing here is executed by the test suite
except the report and runner helpers (`tests/unit/test_comparison.py`).

## Layout

```
comparison/
  README.md                      this protocol
  cases/<case>/<case>_spinach.m  MATLAB side of a case: timed stages, returns data and figures
  cases/<case>/<case>_spynach.py  spynach side: run(timer, **options) -> (data, figs)
  matlab/run_comparison.m        MATLAB runner (settings block at the top): profiler, dumps, figures
  matlab/lib/                    cmp_timer, cmp_profile, cmp_meta, cmp_json, cmp_option
  run_spinach.sh                 runs run_comparison.m in MATLAB batch mode
  run_spynach.py               spynach runner: cProfile, dumps, figures (scripts/mac-compare.sh runs it on the host)
  report.py                      builds runs/<run>/report.html from the two sides
  runs/<run>/spinach/            MATLAB outputs (git-ignored)
  runs/<run>/spynach/          spynach outputs (git-ignored)
  runs/<run>/report.html         the report
  runs/<run>/summary.json        its timings, differences and provenance as data (the project site's table)
```

**What is versioned.** `comparison/runs/` is committed except for the
large files: `data.mat`, `data.npz` and the interactive `figure_*.html`
are ignored, as are runs tagged `quick`. So every report, with its
timings, profiles, provenance, PNG figures and console logs, is in git and
a rerun shows up as a diff; the arrays behind a report live only on the
machine that made them, and the report's "interactive figure" links work
there only. Rerun the side in question to regenerate them.

Spynach's compiled kernels (`spynach.kernels`, the `fast` extra) change the timings, not the numbers, so
the long cases are kept in two runs on the same machine: `runs/<case>/` timed without Numba
(`SPYNACH_NUMBA=0`, or Numba not installed) and `runs/<case>-numba/` timed with it, both against the same
Spinach side (copy `spinach/` into the tagged directory before running the Spynach side with `--tag numba`).
`meta.json` records `numba_version` (null without the kernels) and the report's provenance card shows it;
`scripts/build_site.py` shows the faster of the two in the project page's table.

`<run>` is `<case>[-gpu][-<tag>]`: the case name, `-gpu` when the run was
made with the GPU switch, and `-<tag>` when a tag was given (use the same
switch and tag on both sides). CPU and GPU comparisons of the same case
therefore live next to each other, `runs/noesy_strychnine/` and
`runs/noesy_strychnine-gpu/`, each with its own report. Each side directory
holds:

| file | content |
|---|---|
| `timings.json` | `stages` (label, seconds, in order) and `total_s` (wall clock of the whole case) |
| `profile.json` | `by_total` and `by_self`: the top functions with `calls`, `total_s` (with callees), `self_s` |
| `meta.json` | provenance: Spinach commit and dirty flag, MATLAB version, BLAS and LAPACK libraries, computational threads, pool size; Spynach commit, Python/NumPy/SciPy versions, the BLAS each was built against (`blas`), the thread-count variables set in the environment (`thread_env`, empty for library defaults); host, platform, cores, case options |
| `data.mat` / `data.npz` | every array the case returned, same names on both sides (`-v7` MAT file, readable by SciPy) |
| `figure_<name>.png` | the figures the case returned (Spynach also writes `figure_<name>.html`, the interactive Plotly figure) |
| `console.log` | everything printed during the run, including Spinach's report output |

## Running a comparison

1. **Spinach side, on the host.** Edit the settings block at the top of
   `matlab/run_comparison.m` (repository path, Spinach checkout, cases, the
   `gpu` switch, tag, per-case options), then either

   ```
   comparison/run_spinach.sh          # MATLAB=/path/to/matlab to override the executable
   ```

   or from MATLAB: `run('<repo>/comparison/matlab/run_comparison.m')`. The
   runner puts Spinach on the path, checks that `create()` resolves to that
   checkout, runs each case under `profile on`, and writes
   `runs/<run>/spinach/`. Figures are saved from invisible figure windows,
   so batch mode works.

2. **Spynach side, on the host.**

   ```
   scripts/mac-compare.sh noesy_strychnine --report
   ```

   runs the case in `.venv-mac` (uv syncs it with the `compare` extra
   first, see `README.md`), writes `runs/noesy_strychnine/spynach/`
   and, with `--report`, builds the report. The same runner works in the
   container for development checks,

   ```
   OPENBLAS_NUM_THREADS=4 PYTHONPATH=src python comparison/run_spynach.py noesy_strychnine --report
   ```

   but a report meant to compare speed should have both sides from the
   same machine. `--gpu` selects the GPU run (`runs/noesy_strychnine-gpu/`). `--tag quick --option npoints=64 --option zerofill=256` runs a
   reduced case into `runs/noesy_strychnine-quick/`; option values are
   parsed as JSON and passed to the case's `run()` as keyword arguments.
   Without case names every case runs. The PNG figures need Kaleido or
   Matplotlib (`pip install -e ".[compare]"` in the container; the host
   script installs it); without either only the HTML figures are written.

3. **Report.** `python comparison/report.py noesy_strychnine` (a run name
   under `runs/`, or a path) writes `runs/noesy_strychnine/report.html`:
   provenance of both sides, stage timings side by side with the
   spynach/Spinach ratio, the normwise relative error
   `||spynach - Spinach||_F / ||Spinach||_F` and the largest elementwise
   difference of every array (the oracle tests' metric; FIDs of ported
   sequences agree to about 1e-10), the figures next to each other, the
   profiler tops of both sides, and links to the console logs. Either side
   may be missing; the report says so. Beside it, `summary.json` holds the
   same timings, relative differences and provenance as data, from which
   `scripts/build_site.py` renders the timing table of the project site;
   the report inlines the site's stylesheet and carries its navigation, so
   it opens from the site and standalone alike. The report embeds the PNGs, so it
   can be sent on its own.

**CPU and GPU runs.** The default on both sides is CPU. `gpu=true` in
the MATLAB settings block adds `'gpu'` to `sys.enable` of every case (the
case honours `options.gpu`), records the device in `meta.json` and writes
to `runs/<case>-gpu/`; Spinach itself falls back to the CPU when no
CUDA device is present, which `gpu_device` in the provenance table shows.
`--gpu` on `run_spynach.py` does the same for Spynach, which currently
refuses it (`porting/gpu-notes.md`): the run then leaves only `meta.json`
with the error, and the report says so in its status line. A GPU report
is still worth building for the Spinach timings alone, and the Spynach
side fills in once the backend exists. Never mix sides from different
switches in one run directory.

The two sides can run in either order, and either can be rerun alone
(for example after a Spynach optimisation) before rebuilding the report.
Keep the Spinach outputs of a run when only Spynach changes: they are the
reference.

## Reading the numbers

- **Timings** are wall clock per stage, both sides on the same machine.
  Each side runs as shipped: Spinach uses its parallel pool
  (`pool_workers` in `meta.json`) and MATLAB's multithreaded BLAS;
  Spynach is a single Python process whose dense kernels run on the
  threads of the BLAS NumPy and SciPy were built against (`blas` and
  `thread_env` in `meta.json`: Apple Accelerate with library defaults on
  the Mac, OpenBLAS pinned to the CPU quota in the container). The ratio
  is therefore throughput per machine for each implementation, not a
  per-core comparison. Runs made in the container are development
  checks: its CPU quota is a fraction of the host's and the provenance
  cards show the mismatch. Spinach's first run after starting MATLAB
  includes pool start-up and JIT warm-up inside `create`; run the cases
  twice and keep the second run when that matters.
- **Profiles**: the MATLAB `self_s` list is the kernel hot spots
  (`profile('info')` with built-ins included in the `by_total` list); the
  Python list comes from cProfile, where C extensions such as SciPy sparse
  kernels appear as built-in methods. Both include the profiler overhead.
- **Data**: `shape mismatch` means the two sides ran at different sizes
  (same `npoints`/`zerofill` options on both sides). A relative error of
  order 1 with matching shapes means a real discrepancy, usually a sign,
  transpose or axis-ordering convention; errors around 1e-8 or below are
  floating-point and ordering differences.

## Adding a case

A case is a directory `cases/<case>/` with two files that run the same
calculation with the same stage labels and return the same arrays and
figures under the same names.

**MATLAB** `cases/<case>/<case>_spinach.m`:

```matlab
function [data,figs]=<case>_spinach(timer,options)
npoints=cmp_option(options,'npoints',512);     % options come from the settings block
...                                              % the Spinach example, verbatim
timer.start('basis'); spin_system=basis(spin_system,bas); timer.stop();
timer.start('liquid/noesy'); fid=liquid(spin_system,@noesy,parameters,'nmr'); timer.stop();
data.fid_cos=fid.cos; data.fid_sin=fid.sin;      % arrays to save (dense; name them as on the Python side)
...
timer.start('plot'); figs.noesy=figure('Visible','off'); plot_2d(spin_system,...); timer.stop();
end
```

**Python** `cases/<case>/<case>_spynach.py`:

```python
from comparison.run_spynach import load_example

example = load_example("nmr_liquids/<case>")  # reuse the translation in examples/ when there is one


def run(timer, npoints=512, zerofill=2048):
    ...  # with timer("basis"): spin_system = spynach.basis(spin_system, bas)
    return {"fid_cos": ..., "fid_sin": ..., "spectrum": ...}, {"noesy": fig}
```

Rules that keep the report meaningful:

- Stage labels are identical on both sides, in the same order; a label on
  one side only shows up with a blank column. Stages do not nest; keep
  them coarse (`create`, `basis`, the context call, `processing`, `plot`)
  and let the profiles show what is inside.
- Save raw FIDs before apodisation, and the final spectrum; name arrays
  identically. MATLAB `cell` and `struct` values cannot be compared; save
  dense arrays.
- Figures: the same Spinach `plot_*` call on both sides, same name.
- Take the calculation from `examples/` (the Spinach example translated
  at the Spinach size) rather than writing it twice; the `--option`
  mechanism provides reduced sizes for quick checks.
- Add the case to `cases` in `run_comparison.m` and, if the case
  mirrors a Spinach example, mention the example in `docs/README.md`.

## Cases

| case | Spinach example | notes |
|---|---|---|
| `noesy_strychnine` | `examples/nmr_liquids/noesy_strychnine.m` | 22 protons, `IK-2`, Redfield, 500 ms NOESY; options `npoints`, `zerofill`, `gpu`; stages `create`, `basis`, `liquid/noesy`, `processing`, `plot`; arrays `fid_cos`, `fid_sin`, `spectrum`; figure `noesy` |
| `maz_noesy_2` | `examples/relaxation_theory/maz_noesy_2.m` | methylaziridine, 7 protons + 14N, DFT shielding tensors, 14N quadrupolar tensor, Redfield + SRFK + SRSK, `IK-2` prox 4 (19 314 states; the SRFK integral dominates, about half an hour in the container), 2 s NOESY at 11.75 T; options `npoints`, `zerofill`, `gpu`; stages `create`, `basis`, `liquid/noesy`, `processing`, `plot`; arrays `fid_cos`, `fid_sin`, `spectrum`; figure `noesy` |
| `li_exchange_equilibrium` | `user_ml/Li_exchange_equilibrium.m` (contributed) | two 7Li sites (quadrupolar 30 and 90 kHz, axial shift tensors) in two-site exchange at 1.93 T, Redfield secular; thermal state equilibrated under H + 1i K (1e6 steps of 10 us), R thermalised to it (IME), 2e5-step trajectory analysed with `trajan`; options `nsteps_ex`, `nsteps_eq`, `gpu`; stages `create`, `basis`, `hamiltonian`, `equilibrium`, `relaxation`, `trajectory`, `analysis`, `plot`; arrays `rho_eq`, `rho_eq_ex`, `rho_final`, `tensors`, `corr_order`, `coh_order`, `spin_total`, `unit_population`, `time_eq`; figures `tensors`, `correlation`, `coherence`, `spins` |
| `zulf_pf6_1d` | `user_ml/ZULF_PF6_1D.m` (contributed) | 31P + 6 x 19F at 2.7 uT, complete basis (16 384 states), lab frame, pulse and 16 384-point acquisition (propagator path after state-space reduction); options `npoints`, `zerofill`, `rho_file`, `gpu`; stages `create`, `basis`, `hamiltonian`, `operators`, `pulse`, `acquisition`, `processing`, `plot`; arrays `rho0`, `rho_pulse`, `fid`, `spectrum`, `coil_x`, `coil_z`; figure `spectrum` |
