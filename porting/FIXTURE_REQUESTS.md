# Fixture requests

Requests for a human to run MATLAB dump scripts on the host. The agent adds
entries; the human runs them and fills in the result. Fixtures are written
read-only under `/fixtures/<spinach-tag>/`.

## FR-001: zeeman-hilb pulse-acquire slice (`dump_hilb_pa.m`)

- **Status:** fulfilled 2026-10-10, `/fixtures/2.13/`, run log `run_20261010T184836Z.log`; MATLAB R2026b, Spinach `03a88b7e` (2.13), clean; `pytest tests/oracle`: 34 passed
- **Requested:** 2026-10-08, branch `oracle-harness`
- **Spinach:** clean checkout, expected `03a88b7e912ffbb3122dd24829cc5d216209c804` (no local modifications; `oracle_meta.m` records the dirty flag)
- **Needs:** MATLAB R2024b or later with Parallel Computing Toolbox (`create()` starts a pool). The shadow `oracle/matlab/shadow/isworkernode.m` sends `evolution()` down its serial branch, because the Hilbert-space `spmd` split uses `distributed` arrays, which MATLAB does not support on local process pools on Apple silicon Macs (`distributedutil.errorIfUnsupportedPlatformOrEnvironment`; this is a MATLAB restriction, not a Spinach one). Spinach's startup check rejects the shadow, so the dump sets `sys.disable={'hygiene'}`.
- **Re-running:** existing `hilb_pa_*` case directories are skipped; delete them first to regenerate.
- **Produces:** `hilb_pa_ab_2spin`, `hilb_pa_abx_3spin`, `hilb_pa_hc_2spin`, `hilb_pa_hd_2spin` (each a few MB at most)

Edit the settings block at the top of `oracle/matlab/run_fixtures.m` if the
defaults do not match the host (repository worktree, Spinach directory, fixture
output root), then either run it from a shell:

```
oracle/run_fixtures.sh            # MATLAB=/path/to/matlab to override the executable
```

or from MATLAB:

```matlab
run('<repo>/oracle/matlab/run_fixtures.m')
```

It puts Spinach (`kernel`, `etc`, `experiments`, `interfaces`) and the harness
on the path, checks that `create()` resolves to that Spinach copy, writes to
`<fixtures_root>/<spinach tag>/`, and makes the new case directories read-only.

Progress: each stage of each case prints a timestamped line, e.g.
`2026-10-09 14:02:11 [abx_3spin 2/4] propagator`. The full console output,
including any error report, is saved as `<fixtures_root>/<tag>/run_<UTC time>.log`,
and Spinach's own messages for each case go to `<case>/spinach.log`.
To follow a batch run from another terminal: `tail -f <fixtures_root>/<tag>/run_*.log`.

Then check on the Python side (with `/fixtures` mapped to `fixtures_root`):

```
SPYNACH_FIXTURES=/fixtures pytest tests/oracle -rs
```

All tests in `test_hilb_pa_fixtures.py` should pass, not skip. A failure in
`test_layout_probes` means the HDF5 layout is wrong, not Spinach; report it
before going further.

- **Result:** _(human: fixture path, MATLAB version, any errors or warnings)_

## FR-002: sphten-liouv setup slice (`dump_liouv_setup.m`)

- **Status:** fulfilled 2026-10-11, `/fixtures/2.13/`, run log `run_20261011T202010Z.log`; MATLAB R2026b, Spinach `03a88b7e` (2.13), clean; `pytest tests/slice/test_liouv_setup_parity.py tests/oracle`: 166 passed
- **Requested:** 2026-10-11, branch `main`
- **Spinach:** clean checkout, 2.13 (`03a88b7e`), as for FR-001
- **Needs:** the same host setup as FR-001 (shadow `isworkernode.m`, `sys.disable={'hygiene'}`). No propagation: every case runs in seconds, the twelve-spin chain is the largest (a few thousand states).
- **Run:** from the `main` checkout (not the `oracle-harness` worktree): check `repo_dir` in `oracle/matlab/run_fixtures.m` and run `oracle/run_fixtures.sh`. `dump_liouv_setup` is already in `dump_scripts`; the existing `hilb_pa_*` cases are skipped and the new ones are written next to them.
- **Produces:** `liouv_ab_none`, `liouv_hd_none`, `liouv_inad_none`, `liouv_styrene_ik0_1`, `liouv_styrene_ik0_2`, `liouv_styrene_ik0_4`, `liouv_styrene_ik1`, `liouv_styrene_ik2_proj`, `liouv_chain12_ik2`, `liouv_exch_sym`, `liouv_exch_asym`, `liouv_hcn_filters` (each a few MB at most)
- **Check:** `SPYNACH_FIXTURES=/fixtures pytest tests/slice/test_liouv_setup_parity.py tests/oracle -rs`; nothing should skip.
- **Result:** ran without issues (host, 2026-10-11); all twelve cases written, `hilb_pa_*` skipped.

## FR-003: sphten-liouv propagation (`dump_liouv_pa.m`)

- **Status:** fulfilled 2026-10-11, `/fixtures/2.13/`, run log `run_20261011T204823Z.log`; MATLAB R2026b, Spinach `03a88b7e` (2.13), clean; `pytest tests/slice/test_liouv_pa_parity.py`: 54 passed
- **Requested:** 2026-10-11, branch `main`
- **Spinach:** clean checkout, 2.13 (`03a88b7e`), as before
- **Needs:** the same host setup as FR-001/FR-002. Nine cases; the largest propagates 1789 states (styrene IK-1) for 512 points and should take well under a minute.
- **Run:** `oracle/run_fixtures.sh` from the `main` checkout (`dump_liouv_pa` is in `dump_scripts`; existing cases are skipped).
- **Produces:** `liouv_pa_ab`, `liouv_pa_hd`, `liouv_pa_inad_krylov` (Krylov path forced with `tols.krylov_tol=1`), `liouv_pa_hc_pt` (path tracing forced with `tols.merge_dim=2`), `liouv_pa_styrene_ik0_2`, `liouv_pa_styrene_ik1`, `liouv_pa_chain12_ik2`, `liouv_pa_exch_sym`, `liouv_pa_exch_asym`
- **Check:** `SPYNACH_FIXTURES=/fixtures pytest tests/slice/test_liouv_pa_parity.py -rs`; nothing should skip.
- **Result:** ran successfully (host, 2026-10-11); nine cases written, earlier cases skipped.

## FR-004: anisotropic interactions (`dump_aniso.m`)

- **Status:** fulfilled 2026-10-11, `/fixtures/2.13/`, run log `run_20261011T211740Z.log`; MATLAB R2026b, Spinach `03a88b7e` (2.13), clean; `pytest tests/slice/test_aniso_parity.py`: 25 passed
- **Requested:** 2026-10-11, branch `main`
- **Spinach:** clean checkout, 2.13 (`03a88b7e`), as before
- **Needs:** the same host setup. Five cases, each a few seconds (powder averages over 50- and 100-point grids with 128 points).
- **Run:** `oracle/run_fixtures.sh` from the `main` checkout (`dump_aniso` is in `dump_scripts`; existing cases are skipped).
- **Produces:** `aniso_csa_pair`, `aniso_dip_pair`, `aniso_nqi_14n`, `aniso_hc_tensors`, `aniso_esr_nitroxide`
- **Check:** `SPYNACH_FIXTURES=/fixtures pytest tests/slice/test_aniso_parity.py -rs`; nothing should skip.
- **Result:** done (host, 2026-10-11); five cases written, earlier cases skipped.

## FR-005: relaxation (`dump_relax.m`)

- **Status:** fulfilled 2026-10-11, `/fixtures/2.13/`, run log `run_20261011T213425Z.log`; MATLAB R2026b, Spinach `03a88b7e` (2.13), clean; `pytest tests/slice/test_relax_parity.py`: 20 passed
- **Requested:** 2026-10-11, branch `main`
- **Spinach:** clean checkout, 2.13 (`03a88b7e`), as before
- **Needs:** the same host setup. The dump sets `sys.disable={'hygiene','asyredf'}` so that the Redfield integral takes the serial path (same numbers as the parallel one). Five cases, seconds each.
- **Run:** `oracle/run_fixtures.sh` from the `main` checkout (`dump_relax` is in `dump_scripts`; existing cases are skipped).
- **Produces:** `relax_noe_het`, `relax_noe_hom`, `relax_csa_dd_secular`, `relax_axial_dfs`, `relax_t1t2_damp`
- **Check:** `SPYNACH_FIXTURES=/fixtures pytest tests/slice/test_relax_parity.py -rs`; nothing should skip.
- **Result:** done (host, 2026-10-11); five cases written, earlier cases skipped.

## FR-008: scalar relaxation of the first and second kind (`dump_scalar.m`)

- **Status:** fulfilled 2026-10-14, `/fixtures/2.13/`, run log `run_20261014T180739Z.log`; MATLAB R2026b, Spinach `03a88b7e` (2.13), clean; `pytest tests/slice/test_scalar_parity.py`: 9 passed
- **Requested:** 2026-10-14, branch `main`
- **Spinach:** clean checkout, 2.13 (`03a88b7e`), as before
- **Needs:** the same host setup as FR-005 (`sys.disable={'hygiene','asyredf'}`, serial Redfield path). Three cases, seconds each.
- **Run:** `oracle/run_fixtures.sh` from the `main` checkout (`dump_scalar` is in `dump_scripts`; existing cases are skipped).
- **Produces:** `scalar_srfk_pair` (SRFK alone, `examples/relaxation_theory/scalar_relaxation_1.m`), `scalar_srsk_hn` (Redfield + SRSK from a quadrupolar 14N onto a coupled proton), `scalar_srfk_srsk_3spin` (Redfield + SRFK + SRSK with CSA and a quadrupolar tensor at an orientation)
- **Check:** `SPYNACH_FIXTURES=/fixtures pytest tests/slice/test_scalar_parity.py -rs`; nothing should skip.
- **Result:** done (host, 2026-10-14); the first run stopped on a double log handle in the dump (fixed in `801ecaa`), the rerun wrote the remaining cases.

## FR-006: magic angle spinning, Fokker-Planck form (`dump_mas.m`)

- **Status:** fulfilled 2026-10-11, `/fixtures/2.13/`, run log `run_20261011T221502Z.log`; MATLAB R2026b, Spinach `03a88b7e` (2.13), clean; `pytest tests/slice/test_mas_parity.py`: 16 passed
- **Requested:** 2026-10-11, branch `main`
- **Spinach:** clean checkout, 2.13 (`03a88b7e`), as before
- **Needs:** the same host setup; `parameters.serial=1` keeps the powder loop serial. Four cases (CSA pair, dipolar pair, 2H quadrupole, single crystal off the magic angle), rotor rank 5-7 and 50-point grids, 128 points: seconds each.
- **Run:** `oracle/run_fixtures.sh` from the `main` checkout (`dump_mas` is in `dump_scripts`; existing cases are skipped).
- **Produces:** `mas_csa`, `mas_dip`, `mas_nqi`, `mas_crystal_offaxis`
- **Check:** `SPYNACH_FIXTURES=/fixtures pytest tests/slice/test_mas_parity.py -rs`; nothing should skip.
- **Result:** finished (host, 2026-10-11); four cases written, earlier cases skipped.

## FR-007: pulse sequence library (`dump_seq.m`)

- **Status:** fulfilled 2026-10-11, `/fixtures/2.13/`, run log `run_20261011T225412Z.log`; MATLAB R2026b, Spinach `03a88b7e` (2.13), clean; `pytest tests/slice/test_seq_parity.py`: 3 passed
- **Requested:** 2026-10-11, branch `main`
- **Spinach:** clean checkout, 2.13 (`03a88b7e`), as before
- **Needs:** the same host setup (`asyredf` disabled for the NOESY case). Three cases, 32-64 points per dimension: seconds to a minute each.
- **Run:** `oracle/run_fixtures.sh` from the `main` checkout (`dump_seq` is in `dump_scripts`; existing cases are skipped).
- **Produces:** `seq_cosy_derome`, `seq_noesy_pair`, `seq_hsqc_dilute`
- **Check:** `SPYNACH_FIXTURES=/fixtures pytest tests/slice/test_seq_parity.py -rs`; nothing should skip.
- **Result:** done (host, 2026-10-11); three cases written, earlier cases skipped.

## FR-008: state-dependent integrator `iserstep` (`dump_iserstep.m`)

- **Status:** fulfilled 2026-11-08, `/Users/alexej/Documents/PROG/spynach-fixtures/2.13/`, run log `run_20261108T143104Z.log`; MATLAB R2026b, Spinach `03a88b7e` (2.13), clean; `pytest tests/unit/test_iserstep.py`: 8 passed
- **Requested:** 2026-11-08, branch `main`
- **Spinach:** clean checkout, 2.13 (`03a88b7e`), as before (it holds `kernel/pulses/iserstep.m`)
- **Needs:** the same host setup. One case, one proton in Hilbert space, one step of each of the eight methods and the two-half-step DP8 reference of Spinach's `test_dynamic_iserstep_highorder.m`: under a second.
- **Run:** `oracle/run_fixtures.sh` from the `main` checkout (`dump_iserstep` is in `dump_scripts`; existing cases are skipped).
- **Produces:** `iserstep_hilb_1spin`
- **Check:** `SPYNACH_FIXTURES=/Users/alexej/Documents/PROG/spynach-fixtures pytest tests/unit/test_iserstep.py -rs`; nothing should skip.
- **Result:** done (host, 2026-11-08); one case written, earlier cases skipped. A first run failed in `oracle_meta` (`jsonencode` refuses complex inputs), so the dump records the density matrix as real and imaginary parts.
