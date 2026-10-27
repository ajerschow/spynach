# Roadmap

Phase 1 as laid out in `HANDOFF.md`, revised 2026-10-11 after the literature
and source review in `design-notes.md`. Steps 1 and 2 are done (oracle
harness, FR-001; `zeeman-hilb` pulse-acquire slice, verified against Spinach
2.13). Step 3 is narrowed to the Liouville-space *setup*: basis, operators,
states, Hamiltonian and kinetics superoperators, verified against cheap
fixtures. Propagation becomes step 4.

## Step 3: `sphten-liouv` setup and Hamiltonians

Scope: everything up to and including the Liouvillian, nothing that
propagates. Each sub-step is a commit with tests and docs.

Status (2026-10-11): done. 3a–3f on `main` (commits `68b0626` onwards);
FR-002 fulfilled, 132 parity tests pass (agreement table in
`docs/liouville-space.md`). `IK-1`/`IK-2` with proximity graphs remain
untestable until coordinates are ported. Next: step 4.

### 3a. Spin system: chemistry fields and the generalised `Basis`

- `create()`: accept `inter['chem']` with `parts` (0-based spin index
  lists), `concs`, `rates`; validate as `create.m` does (every spin in
  exactly one part, parts sorted, `concs` required for >1 part, no
  couplings between parts, `rates` square with zero column sums and equal
  part sizes). `flux_rate/flux_type`, `rp_*` and nonlinear reactions stay
  `NotImplementedError`. Default: one part with all spins, `concs = [1]`.
- `Basis` dataclass grows: `formalism`, `approximation`, `basis` (int array
  `(nstates, nspins)`; for `zeeman-hilb` keep the current `levels`),
  `tot_proj`, `tot_cord`, cached product tables. Keep frozen dataclasses.

### 3b. `basis()` for `sphten-liouv`

- Approximations `none`, `IK-0`, `IK-1`, `IK-2` with the 2.13 field names
  `inter_level`, `prox_level`, `connectivity` (`'scalar_couplings'` and
  `'full_tensors'`), `projections`, `longitudinal`, `zero_quantum`, `manual`
  — all per substance exactly as `basis.m` (design-notes §3). `sym_group`,
  `IK-DNP`, `IK-SBS` and `zeeman-liouv` raise `NotImplementedError`; the
  old field names raise the same rename errors as Spinach.
- Helpers: `dfpt`, `prune_subgraphs`, connectivity matrix (reuse
  `hamiltonian._coupled_pairs` logic for the norm), proximity matrix
  (identity; coordinates not ported).
- Unit tests without fixtures: state counts for `none` (`Π mult²`), IK-0
  (`Σ_{j≤k} C(n,j)·3^j` for n spin-1/2 at `inter_level` k), two substances (`1 + 3 + 3`), the
  7-state example, projections keeping the unit row, longitudinal reducing
  per-spin lists to `l² + l`.

### 3c. Product tables, `operator()`, `state()`

- `ist_product_table(mult)` from `irr_sph_ten`, cached; unit-tested against
  the hard-coded spin-1/2 tables in `ist_product_table.m`.
- `operator(spin_system, ops, spins, operator_type='comm')` dispatching on
  formalism; the sphten branch is the vectorised `superop` of design-notes
  §5.3(4). `left`, `right`, `comm`, `acomm`.
- `state(spin_system, ops, spins, method='exact')` with `exact`, `cheap`, `chem`.
- Cross-formalism tests without fixtures: for complete bases,
  `Pᴴ (kron(I,H) − kron(Hᵀ,I)) P / dim` equals the sphten commutation
  superoperator, with `P` from the `sphten2zeeman` construction; `exact`
  states equal `Pᴴ vec(Op)/√dim`-type images of the Hilbert-space operators.

### 3d. `hamiltonian()`, `frqoffset()`, `kinetics()`, filters

- `hamiltonian(spin_system, operator_type='comm')` returns the isotropic
  commutation superoperator in Liouville space (same term logic, new
  `operator_type` plumbing, term table keeps slots for `Q`). `frqoffset`
  unchanged apart from dispatch.
- `kinetics(spin_system)` from `chem.rates` (design-notes §5.1); unit test: column
  sums of `K` vanish (Spinach's `test_kinetics_generator_suite.m`).
- `correlation(spin_system, rho, orders, spins)` and `coherence(spin_system, rho, spec)`.
- `liquid()` still refuses `sphten-liouv` until step 4 (it would need
  `evolution`); say so in the error message.

### 3e. Fixtures: `dump_liouv_setup.m`, FR-002

All cases are seconds in MATLAB: no propagation, only
`create → basis → operator → state → hamiltonian → frqoffset → kinetics`.
Variables per case (same HDF5 conventions as FR-001):

| path | content |
|---|---|
| `/bas/basis`, `/bas/tot_proj`, `/bas/tot_cord` | basis descriptor and derived orders |
| `/bas/lpst_<mult>`, `/bas/rpst_<mult>` | product tables for every multiplicity present |
| `/ops/<name>_<type>` | `E, Lz, Lx, Ly, L+, L-` per spin and a two-spin product, for `left/right/comm/acomm` |
| `/states/<op>_<method>` | `Lz`, `L+` on the detected isotope for `exact`, `cheap`, `chem` |
| `/ham/H`, `/ham/H_zeeman`, `/ham/H_coupling`, `/ham/H_off` | commutation superoperators |
| `/ham/H_left`, `/ham/H_right` | `hamiltonian(...,'left'/'right')` |
| `/kin/K` | `kinetics(spin_system)` where `chem.rates` exists |
| `/filters/corr_<orders>`, `/filters/coh_<spec>` | filter outputs on a dense test vector |
| `/conv/P_sphten2zeeman` | for complete bases only |

Cases (Spinach examples in parentheses; all `B0 = 14.1 T` unless the
example says otherwise, `assume 'nmr'`):

| case | system | `bas` | exercises |
|---|---|---|---|
| `liouv_ab_none` | 2×1H (reuse FR-001 `ab_2spin`) | `none` | complete basis, cross-check with Hilbert fixtures |
| `liouv_hd_none` | 1H + 2H (FR-001 `hd_2spin`) | `none` | mult-3 tables, spin-1 states |
| `liouv_inad_none` | 3×13C (`nmr_liquids/inad_three_spin.m`) | `none` | 64 states, heteronuclear sums |
| `liouv_styrene_ik0_<k>` | 8×1H (`pa_styrene.m`) | `IK-0`, `inter_level` 1, 2, 4 | correlation-order truncation, state counts |
| `liouv_styrene_ik1` | 8×1H | `IK-1`, `inter_level 3`, `prox_level 1`, `scalar_couplings` | `dfpt` on the J graph |
| `liouv_styrene_ik2_proj` | 8×1H | `IK-2`, `prox_level 1`, `projections {[1]}` | star subgraphs, coherence filter |
| `liouv_chain_ik2` | 12×1H chain, J = 10 Hz nearest neighbour (`product_quadratures_1.m`, shortened from 31) | `IK-2`, `prox_level 1` | sparse large basis, timing |
| `liouv_exch_sym` | 2×1H, 2 parts (`kinetics/exchange_symmetric.m`) | `none`, `rates ±2e3`, `concs [1 1]` | segmented basis, `kinetics`, `chem` states |
| `liouv_exch_asym` | same, `rates [-5e2 2e3; 5e2 -2e3]`, `concs [2e3 5e2]` | `none` | unequal concentrations |
| `liouv_hcn_long` | 1H, 13C, 15N with J | `none`, `longitudinal {{'15N','13C'}}`, `zero_quantum {{'13C'}}` | longitudinal and zero-quantum filters |

`liouv_chain_ik2` is the only case with a non-trivial basis size; keep it
under a second. Write the dump, add FR-002 to `FIXTURE_REQUESTS.md`, and
the parity tests in `tests/slice/test_liouv_setup_parity.py`.

### 3f. Documentation

- Keep `docs/user-guide.md` as the Hilbert-space guide (rename to
  `hilbert-space.md` when the Liouville guide lands, so the two sit side by
  side in `docs/README.md`; `coming-from-spinach.md` stays the cross-cutting
  comparison). Add `docs/liouville-space.md` with its own notebook:
  basis descriptor and state counts, operators and the four superoperator
  types, states and normalisation, Hamiltonian commutation superoperator,
  restriction (`IK-0/1/2`, filters) with the per-order norm plot and
  state-count tables, two-site exchange setup. Nothing in it propagates
  until step 4.
- `coming-from-spinach.md`: basis section (field renames, per-substance
  cells, 0-based `parts`), operators (`operator_type`), states (`method`),
  kinetics; move items out of the not-ported lists; README table.

## Step 4: propagation in Liouville space

Status (2026-10-11): `step`, `krylov`, `evolution` ('observable', 'final',
'trajectory', destination state screening), `reduce`/`zte`/`path_trace`,
`liquid` with kinetics are on `main` (commit `3fd6f81`), verified against
Hilbert space and the McConnell equations; FR-003 fulfilled, 54 parity
tests pass (agreement 1e-13 or better, table in `docs/liouville-space.md`).
Deferred: the CuPy backend (no GPU in the development container; needs a
host with CUDA to test) and the benchmarks that depend on it.

`step` (Krylov-style Taylor action on a vector), `evolution` for the
Liouville formalisms with `reduce` (ZTE, path tracing, DSS; symmetry still
refused), `krylov`, Taylor propagator with thresholding and the
`krylov_tol` switch, pulse-acquire through `liquid`/`acquire` with
`H + i·K`; fixtures `dump_liouv_pa.m` (FR-003) reusing the step-3 cases plus
exchange line shapes. Then the `spynach.backend` adapter and CuPy, with the
first benchmarks on `liouv_chain_ik2`-style systems at 500–10 000 states.

## Step 5: anisotropic interactions (decided 2026-10-11: this, then relaxation)

Status (2026-10-11): implemented on `main`. `create` takes `eigs`/`euler`
tensors and coordinates (dipolar couplings, proximity graph);
`hamiltonian(..., aniso=True)` returns `(I, Q)` and `orientation` contracts
`Q` with Wigner matrices (`spynach.conventions`); `crystal` and `powder`
contexts with Spinach's grids shipped as `spynach/grids/*.npz`;
`esr`/`deer`/`labframe` assumption sets. Verified against closed forms
(CSA, dipolar angular dependence, quadrupolar) and Hilbert/Liouville
agreement; FR-004 fulfilled, 25 parity tests pass (1e-13 or better,
table in `docs/solid-state.md`). Not ported: `pbc`, MAS and the
Fokker-Planck contexts, `needs`, callable `rho0`.

## Step 6: relaxation

Redfield/BRW (`relaxation.m`, diagonalisation-free implementation of
Kuprov 2011, auxiliary-matrix integrals of Goodwin & Kuprov 2015) with the
`inter.relaxation`, `inter.tau_c`, `inter.r1_rates`/`r2_rates` inputs,
thermalisation (`equilibrium`, `inter.temperature`), and `R` in all
contexts; fixtures from `examples/nmr_liquids` NOESY-type cases.

Status (2026-10-11): implemented on `main`. `redfield` (isotropic, axial,
rhombic diffusion via `expmint`), `t1_t2`, `damp`; `rlx_keep`, `rlx_dfs`;
`equilibrium` and `thermalize` (IME, DiBari-Levitt); `R` in every context;
`evolution` `multichannel`. Verified against the Solomon equations and
Boltzmann states (`tests/unit/test_relaxation.py`); FR-005 fulfilled,
20 parity tests pass (superoperators identical, table in
`docs/relaxation.md`). Not ported: Lindblad, Nottingham/Weizmann DNP, SRFK/SRSK,
Nakajima-Zwanzig, rate functions, bosonic dissipators.

## Step 7: magic angle spinning (decided 2026-10-11)

Status (2026-10-11): `singlerot` in Fokker-Planck form with `fourdif`
implemented on `main`, as `singlerot.m` builds it (explicit block-diagonal
generator plus the rotor term; no polyadic container needed at this
scale). Verified by exact rotor echoes, the fast-spinning limit and the
static limit against `crystal`; FR-006 fulfilled, 16 parity tests pass
(generators identical, FIDs 1e-13; table in `docs/magic-angle-spinning.md`). Not ported:
`doublerot`, `floquet`, `gridfree`, Hilbert-space `singlerot`.

## Step 8: pulse sequence library (decided 2026-10-11)

Status (2026-10-11): implemented on `main`. `cosy`, `noesy` (States-TPPI,
both mixing modes), `hsqc` (echo-antiecho) as in `experiments/nmr_liquids`;
`acquire` with `decouple`, `dead_time`, `homodec_*`; `decouple`,
`homospoil`, `dilute`/`kill_spin`; `evolution` `refocus`; `needs=['rho_eq']`;
`apodisation`; `plot_2d` with `contspacing`. Verified by COSY/NOESY/HSQC
peak positions and signs, decoupled singlets, refocus against explicit
propagation; FR-007 fulfilled, 3 parity tests pass (1e-13 or better,
table in `docs/pulse-sequences.md`). Not ported: the other sequences
(`dqf_cosy`, `tocsy`, `hmqc`, ...), soft/shaped pulses.

## Step 9: consolidation (decided 2026-10-11)

Status (2026-10-11): profiling on strychnine (22 1H, IK-2, prox 3, 12 316
states; `porting/tools/bench_strychnine.py`) showed superoperator
assembly dominating: `np.unique(axis=0)` on the basis per term and side.
Replaced by per-basis 64-bit row hashes (one sort per basis, bisection per
path), a bounded per-basis operator cache (`Basis.op_cache`, Spinach's
`op_cache`), and triplet-based assembly of `H` and the 34 `Q` blocks.
Hamiltonian 1.3 s -> 0.1 s on 7459 states; strychnine relaxation 4.9 ->
2.4 s, pulse-acquire 5.3 -> 2.1 s, COSY 6.1 -> 3.3 s. Parity unchanged
(498 tests). The guides' "not ported" lists were reconciled with the
current state. API review (38 public names, Spinach signatures with
0-based spins and dict parameters): consistent; two conventions to keep
in mind when adding functions: Liouville states are dense 1-D arrays and
stacks are `(nstates, k)`, filters take the spin index on the last axis
(hence the transposes around `coherence` in the 2-D sequences). Open:
`evolution` device sessions (`porting/gpu-notes.md`), Hamiltonian and
propagator caches.

## Later

Symmetry, `doublerot`/`floquet`/`gridfree`, more sequences and shaped
pulses, the polyadic generator container for imaging-scale problems, the
GPU backend (`porting/gpu-notes.md`), the JAX layer. Pathway-aware state-space
restriction from phase-cycle selection (exploratory, not scheduled, see
`porting/pathway-notes.md`).

# CTPS: the coherence-pathway module `ctps` is separate from this roadmap; see src/ctps/README.md.

# EPR: the EasySpin port `easyspyn` is separate from this roadmap; see src/easyspyn/README.md.
