# ctps: coherence transfer pathway selection

A Python port of CCCP.m (Jerschow & Müller, *J. Magn. Reson.* 134, 17 (1998)) and
CCCP++ (Jerschow & Kumar, *J. Magn. Reson.* 160, 59 (2003)), with the treatment of
non-ideal pulses of Jerschow, *J. Chem. Phys.* 113, 979 (2000) and an optimiser for
phase cycles. It lives beside Spynach in this repository, is imported as `ctps`,
and imports nothing from `spynach`.

The user guide is [`docs/ctps.md`](../../docs/ctps.md) and the cookbook
[`docs/ctps-examples.md`](../../docs/ctps-examples.md) (notebooks under
`docs/notebooks/`); the tests are in `tests/ctps/`.

## What is here

- `ctps.model`: `Nucleus`, `Pulse`, `Period`, `Sample`, `AllowedOrders`,
  `Sequence`, `PhaseCycle`, `PathwayResult`.
- `ctps.simulate`: enumerate, weight and rank pathways.
- `ctps.engine`: enumeration, phase-cycle and cogwheel factors, gradient,
  diffusion and flow attenuation, non-ideal pulse weights, ranking.
- `ctps.optimize`: cogwheel search (`find_cycle`, `coverage`) and the
  Hughes–Carravetta–Levitt construction (`hcl_cycle`).
- `ctps.plot`: the pathway explorer and companion figures.
- `ctps.io`: COG notation, Bruker phase programs, phase tables, CSV and JSON.

## Not here, on purpose

- The density-matrix pulse-damping tier of CCCP.m (AX/AMX/AMXinv propagators,
  `calcord`); Spynach's pulse programs (`spynach.program`, `docs/pulse-programs.md`)
  take its place: they convert to a `Sequence` and `PhaseCycle` with
  `Program.to_ctps`, take the phase-cycle weights from `simulate`, and propagate
  every surviving pathway exactly. The plug-in point for cheaper pulse models stays
  `Pulse.kind` with the `PulseModel` idea in `ctps.engine.pulse`.
- GPU backends (hot loops are marked `# GPU:`), a command line, a GUI, a
  YAML format, `ipywidgets` editors.
- Loaders for CCCP.m scripts and CCCP++ `.par` files, and parity fixtures
  from the original programs, until their sources are mounted at `/cccp`
  and `/cccp-pp`.

## Conventions

Jerschow & Müller 1998 throughout: `n` pulses, `n + 1` periods numbered from 0,
`Δp_j = p_j - p_{j-1}`, a pulse phase `φ` multiplies the signal by
`exp(-i Δp φ)`, the receiver by `exp(-i φ_rec)`; the reported phase `ψ` is
`arg(s)` in degrees as the 1998 tables list it. Cogwheel cycles
`COG_N(ν_1..ν_n; ν_rec)` select a pathway iff `Σ ν_j Δp_j + ν_rec ≡ 0 (mod N)`.

## Rolling it back

The module touches nothing in `src/spynach`; `spynach.program` imports it lazily
(`Program.to_ctps`, `Program.select`) and raises `ImportError` without it, the
rest of Spynach is unaffected. To remove it:

```
git rm -r src/ctps tests/ctps docs/ctps.md docs/notebooks/ctps.ipynb docs/figures/ctps-*.png
```

then revert the one-line mentions in `README.md`, `docs/README.md`,
`CHANGELOG.md` and `porting/ROADMAP.md`, and run the test suite.

## Licence

The original programs are by Alexej Jerschow (CCCP.m with N. Müller, CCCP++ with
R. Kumar); this port is released with the repository under the MIT licence
(`LICENSE` at the root) with the author's agreement.
