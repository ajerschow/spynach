# Design notes: the Liouville-space kernel (`sphten-liouv`)

Written 2026-10-11 from the Spinach publications in `spinach-publications/`
(local folder, not in git; index in `Spinach_papers_index.md`) and from the
Spinach 2.13 source at `/spinach`. These notes record *why* the kernel is
built the way it is, how the design changed over the years, and which
version Spynach ports. The implementation plan derived from them is in
`ROADMAP.md`.

Paper keys: K07 = Kuprov 2007 JMR 189; K08 = Kuprov 2008 JMR 195; HHK10 =
Hogben/Hore/Kuprov 2010 JCP 132; SP11 = Hogben et al. 2011 JMR 208 (the
Spinach paper); KAR11 = Karabanov et al. 2011 JCP 135; KEK11 =
Krzystyniak/Edwards/Kuprov 2011 JMR 210; EK12 = Edwards/Kuprov 2012 JCP 136;
ED13 = Edwards et al. 2013 JMR 235; ED14 = Edwards et al. 2014 JMR 243; GK15
= Goodwin/Kuprov 2015 JCP 143; FP16 = Kuprov 2016 JMR 270; TUT18 = Kuprov
2018 MRC 56; AL19 = Allami et al. 2019 Sci. Adv. 5; DM19 = Kuprov 2019 JMR
306; AC25 = Acharya et al. 2025 Sci. Adv. 11 (`sciadv.ady9103.pdf`).

Three files in the folder are not what their names say:
`2017_Guduff_Efficient_simulation_of_ultrafast_MR_PCCP19.pdf` is Jarvis et
al. 2017 (14N indirect detection; the spindynamics.org link serves the same
wrong file), `2017_Guduff_Spatially_encoded_DOSY_ChemComm53.pdf` is Suturina
et al. 2017 (PCS), and `2024_Allami_..._MedNTD.pdf` is a ScienceDirect error
page. The ultrafast-MR content is covered by FP16 Sec. 5 and AL19.

## 1. How the design evolved

| year | paper | what it introduced | status in Spinach 2.13 |
|---|---|---|---|
| 2007 | K07 | Cluster expansion on a Pauli product basis (index 0/1/2/3 = E/σz/σ+/σ-); exact per-cluster Liouvillians "blended" into a global one; closure of the restricted space enforced by modifying H | superseded: no Hamiltonian modification, different index convention |
| 2008 | K08 | Zero track elimination (ZTE); Krylov propagation; piecewise-constant time dependence | current, applied inside `evolution` |
| 2010 | HHK10 | Irreducible spherical tensor (IST) product basis with one integer per spin; superoperators built *directly in the restricted basis* from single-spin structure coefficients; conservation-law (coherence order) screening; Liouville-space symmetry keeps only A1g; Tarjan path tracing | current |
| 2011 | SP11 | Packaging: IK-0/1/2 names, `reduce` = symmetry → ZTE → path tracing, sparse clean-up, kernel API | current (ESR-1/ESR-2 bases removed since) |
| 2011 | KAR11 | Accuracy proof: error of correlation-order truncation decays as a Gaussian in the order, governed by h/r (largest coupling / smallest relaxation rate) | current justification |
| 2011 | KEK11 | Destination state screening (reduce against the coil state, backwards) | current, default in `evolution` |
| 2012 | EK12 | Hilbert-space column split of ρ for parallel propagation; sparsity bounds for Pauli-basis H | Spynach's `evolution` runs serially; no constraint on the Liouville design |
| 2013–16 | ED13, FP16 | Fokker-Planck: classical coordinates (rotor phase, RF phase, space) as extra Kronecker factors *in front of* the spin factor | current; defines how composite state vectors are stacked |
| 2014 | ED14 | Final basis algorithm: two coupling graphs (J and dipolar/proximity), DFS enumeration of connected subgraphs, union of per-subgraph product bases; IK-1(n,k), IK-2(n) | current (`dfpt.m`, `prune_subgraphs.m`) |
| 2015 | GK15 | Auxiliary (block) matrices for integrals of exponentials | later (relaxation, derivatives) |
| 2018 | TUT18 | Recommendations: IK-2 with `prox_level=3`, 5 Å, for small molecules; IK-1(5,3) for proteins; `inter.chem.parts/rates/concs` | current (field names since renamed, see 3.1) |
| 2019 | AL19, DM19 | Polyadic (lazy Kronecker) generators, `[A⊗B]v = vec(B V Aᵀ)`; "never diagonalise, never open krons, cheap norm estimates" | current (`@polyadic`, `sys.enable={'polyadic'}`) |
| 2025 | AC25 | Segmented per-substance basis with one unit state per substance; drain/fill generators for bimolecular reactions; concentration-weighted ρ | current (`react_gen.m`) |

Takeaways: the basis representation (integer descriptor, one row per
state), the structure-coefficient construction of superoperators, and the
exact reductions (symmetry, ZTE, path tracing, DSS) have been stable since
2010–2011. What changed later is *around* them: how many species share a
basis (AC25), how classical coordinates are attached (FP16), and how
generators are stored when the kron products get large (AL19). Spynach ports
the 2.13 state of all of this, and must leave room for the FP16/AL19/AC25
extensions in the state and generator layout (Section 5).

## 2. The representation (Spinach 2.13 source)

### 2.1 Basis descriptor

- Single-spin ISTs are numbered `I = l² + l − m` (`kernel/indexing/lm2lin.m`;
  inverse `L = fix(sqrt(I))`, `M = L² + L − I`): 0 = T00 (unit), 1 = T1+1,
  2 = T10, 3 = T1−1, 4..8 = T2+2..T2−2. A spin of multiplicity `mult` has
  indices `0..mult²−1`. Spynach's `operators.lin2lm/lm2lin` already use this.
- `spin_system.bas.basis` is an `nstates × nspins` integer matrix (sparse
  double in MATLAB), one row per basis state. Row 1 is always the all-zero
  unit state. Rows are the lexicographically sorted *set* of unique rows
  (`unihash` then `sortrows`, `basis.m` lines 612–630), spin 1 most
  significant. Derived: `bas.tot_proj = Σ m` (coherence order) and
  `bas.tot_cord = number of non-zero entries` (correlation order).
- Basis states are **orthonormal**: row `(i₁…i_N)` is the operator
  `⊗_s T_{i_s}/‖T_{i_s}‖_F`. For spin-1/2, `‖T00‖ = √2`, `‖T1m‖ = 1/√2`, with
  `T(1,+1) = −L+/√2`, `T(1,0) = Lz`, `T(1,−1) = +L−/√2` (`irr_sph_ten.m`,
  which Spynach already reproduces). The unit state is `E/√dim`.
- `sphten2zeeman.m` gives the map to the Zeeman-Liouville vectorisation:
  `P(:,n) = √(Π mults) · vec(⊗T)/‖⊗T‖`, `ρ_zeeman = P ρ_sphten`. For a
  complete basis this is a cross-formalism identity usable without fixtures:
  `L_sphten = Pᴴ L_zeeman_liouv P / dim` with
  `L_zeeman_liouv = kron(I, H) − kron(Hᵀ, I)` (MATLAB column-major `vec`).

### 2.2 Structure coefficient tables (`kernel/cache/ist_product_table.m`)

With `T̂ = T/‖T‖_F` and `hdot(A,B) = Σ conj(A)∘B = Tr(AᴴB)`:

    lpst(n,m,k) = ⟨T̂_k, T_n T̂_m⟩_F      (unnormalised T_n acting from the left)
    rpst(n,m,k) = ⟨T̂_k, T̂_m T_n⟩_F      (from the right)

i.e. the action of the *unnormalised* operator `T_n` on the orthonormal
basis state `T̂_m`, expanded in orthonormal states `T̂_k`. Spin-1/2 tables
(pages are `k`; index order E, T11, T10, T1−1):

    lpst(:,:,1)=[1 0 0 0; 0 0 0 -.5; 0 0 .5 0; 0 -.5 0 0]    lpst(:,:,2)=[0 1 0 0; .5 0 -.5 0; 0 .5 0 0; 0 0 0 0]
    lpst(:,:,3)=[0 0 1 0; 0 0 0 -.5; .5 0 0 0; 0 .5 0 0]      lpst(:,:,4)=[0 0 0 1; 0 0 0 0; 0 0 0 -.5; .5 0 .5 0]
    rpst(:,:,1)=[1 0 0 0; 0 0 0 -.5; 0 0 .5 0; 0 -.5 0 0]    rpst(:,:,2)=[0 1 0 0; .5 0 .5 0; 0 -.5 0 0; 0 0 0 0]
    rpst(:,:,3)=[0 0 1 0; 0 0 0 .5; .5 0 0 0; 0 -.5 0 0]      rpst(:,:,4)=[0 0 0 1; 0 0 0 0; 0 0 0 .5; .5 0 -.5 0]

Spinach caches tables for mult > 2 as `.mat` files; Spynach computes them from
`irr_sph_ten` (dense `np.vdot`) and caches per multiplicity. Multiplicities
above 16 are refused by Spinach.

### 2.3 Superoperators (`kernel/operators/superop.m`)

`operator(spin_system, ops, spins, type)` with `type ∈ {left, right, comm
(default), acomm}` parses the request with `human2opspec` (already ported;
`L+ → −√2·T(1,+1)`, `L− → +√2·T(1,−1)`, `Lz → T(1,0)`), builds each opspec's
superoperator as COO triplets and sums duplicates into an `nstates ×
nstates` sparse matrix. `comm = leftofcomm − rightofcomm`, `acomm = left +
right`. For one opspec with active spins `a₁..a_p`:

1. For each active spin, take the table page `pt = lpst{mult}(opspec+1,:,:)`
   (or `rpst`) and list its non-zeros as (input state, output state,
   coefficient) on that spin.
2. Form all combinations across active spins (first active spin slowest);
   coefficient = product. For `leftofcomm`/`rightofcomm`, drop paths whose
   input or output pattern is all-unit on the active spins (they cancel in
   the commutator; keeping them changes nothing but the sparsity pattern).
3. For each path: find the basis rows whose active columns equal the input
   pattern and those equal to the output pattern; match them on the passive
   columns (identity on passive spins). Matched pairs give entries
   `A[row_out, row_in] = coeff`. Unmatched outputs are dropped — this is
   the state-space restriction, i.e. the orthogonal projection of the full
   superoperator onto the kept states. **`A` acts on column vectors
   normally** (Spinach's `source`/`destin` variable names read backwards).
4. Empty opspec → identity of the basis dimension.

### 2.4 States (`kernel/state.m`)

`unit` = the vector with 1 on the unit state (as an operator: `E/√dim`).

- `'exact'` (default): `operator(..., 'left') * unit`, i.e. the orthonormal
  coefficients of `Op/√dim`. Spin-1/2: `Lz` → 1/2 on T10; `L+` → −1/√2 on
  T1+1; a p-spin product gets `1/2^p`.
- `'cheap'`: coefficient placed directly on the single matching basis row
  (`Lz` → 1, `L+` → −√2); errors if the row is absent or ambiguous.
  `exact = cheap · Π_active ‖T_{lm}‖_F/√mult`.
- `'chem'`: like `exact`, each opspec multiplied by the concentration of the
  substance containing all its spins (error if spins cross substances).
  Kinetics examples use `'chem'` for `rho0` and `'exact'` for `coil`.
- Norm: Liouville vectors are a factor `√dim` smaller than zeeman-liouv's
  `vec(Op)`; `normalization_1.m` compares only within one formalism.

### 2.5 Hamiltonian, assumptions, offsets

- `hamiltonian.m` is formalism-agnostic: it builds a term table and calls
  `operator(..., type, 'xyz')` per term, so in `sphten-liouv` the default
  `type='comm'` returns the commutation superoperator `[H,·]` directly. The
  isotropic logic is exactly what `spynach/hamiltonian.py` already does
  (Zeeman full/secular/ignore; `strong/secular → LzSz + (L+S− + L−S+)/2`;
  `zz/z*/*z → LzSz`), followed by `clean_up(..., liouv_zero)`.
- Second output `Q = {3×3 cell, 5×5 cell}` holds the rank-1 and rank-2
  irreducible components, each a commutation superoperator; `orientation.m`
  contracts them with Wigner matrices. Coefficients: `irr_comp = [λ φ]` from
  `mat2sphten` (Mueller conventions) times tabulated two-spin IST expansion
  coefficients (`hamiltonian.m` ~lines 690–800; `Lz Sz → √(2/3)` on T20,
  `L+S− → −√(1/8)` on T10 and `−√(2/3)/4` on T20, etc.). Out of scope now;
  the term-table design must keep a slot for it.
- `assume.m` and `frqoffset.m` contain no formalism dependence: `'nmr'` is
  as ported; `frqoffset` adds `2π·offset·operator('Lz', isotope)`, which in
  Liouville space is `2π·offset·[Lz,·]`.

### 2.6 Filters (`correlation.m`, `coherence.m`)

Both reshape ρ to `(nstates, spatial)` and zero rows. `correlation(ρ, orders,
spins)` keeps rows whose number of non-unit entries over `spins` is in
`orders`; `coherence(ρ, {{'13C',[1 −1]}, {'1H',−1}})` keeps rows whose `Σ m`
over each spin set is in the list (AND across entries). Both are pure
descriptor operations; they replace phase cycles (ED14).

## 3. State-space restriction

### 3.1 Field names in 2.13

`basis.m` errors on the old names: `bas.level → bas.inter_level`,
`bas.space_level → bas.prox_level`, `bas.longitudinals → bas.longitudinal`
(now a cell per substance). `bas.projections` and `bas.zero_quantum` are also
cells with one entry per chemical substance. Allowed approximations:
`none, IK-0, IK-1, IK-2, IK-DNP, IK-SBS`. ESR-1/ESR-2 (SP11) no longer exist.
HANDOFF.md and the older papers use the old names; Spynach uses the 2.13 names.

### 3.2 Graphs

Only for IK-1/IK-2 (and DNP/SBS). `conmatrix(n,k) = |coupling norm| >
2π·inter_cutoff`, where the norm is `|trace/3|` for `connectivity =
'scalar_couplings'` and the 2-norm for `'full_tensors'`; OR'd with the
identity and symmetrised. `proxmatrix` is the identity without coordinates;
with coordinates `dipolar.m` sets it for pairs *within one substance* closer
than `tols.prox_cutoff` (Å) and adds the dipolar tensors to
`coupling.matrix`. The number of connected components of `conmatrix |
proxmatrix` is reported as a warning only.

### 3.3 Subgraphs per substance (`basis.m` lines ~330–520)

Per substance `s` with spins `chem.parts{s}` (local indexing):

| approximation | coupling subgraphs | proximity subgraphs |
|---|---|---|
| `none` | the whole substance | – |
| `IK-0` | all `inter_level`-subsets of the substance's spins | – |
| `IK-1` | `dfpt(conmatrix, inter_level)` | `dfpt(proxmatrix, prox_level)` |
| `IK-2` | `unique(conmatrix, 'rows')`: each spin with all its direct neighbours | `dfpt(proxmatrix, prox_level)` |

`dfpt(C, k)` (`utilities/dfpt.m`): start from single spins; at each size,
extend every subgraph by every neighbour not yet in it (isolated subgraphs are
padded with a duplicate of their last spin), sort rows, unique; repeat up to
size `k`. Result: every connected subgraph of size exactly `k` plus the
maximal components smaller than `k`. Smaller subgraphs need no listing
because each product basis includes the unit state on every spin.

Then: append rows of `bas.manual` lying inside the substance; zero the
columns of spin-0 particles; drop empty rows; `unique`; `prune_subgraphs`
(drop rows that are strict subsets of another row); embed into global spin
indexing. All substances' subgraphs are concatenated (and shuffled with
`randperm` for load balancing only; the final sort removes any effect).

### 3.4 States of a subgraph and the filters

Per-spin state lists are `0..mult²−1`, reduced to `m = 0` indices
(`l² + l`) for spins named in `bas.longitudinal{s}`. For each subgraph: the
full product of the per-spin lists (first involved spin slowest); if
`bas.projections{s}` is set, keep rows with `Σ m ∈ projections` **plus the
unit row**; if `bas.zero_quantum{s}` names spins in the subgraph, keep rows
with `Σ m = 0` over those spins. Scatter into `nspins` columns.

### 3.5 Union and order

`[zeros(1,nspins); all rows]` → unique (stable) → `sortrows`. In NumPy,
`np.unique(rows, axis=0)` gives exactly this (lexicographic, zero row first).
Example: two spin-1/2 in separate substances with `none` give 7 states
(1 + 3 + 3), not 16 — substances form a **direct sum sharing one unit
state** (AC25 calls this the segmented basis; needed so that each substance's
trace/concentration sits in its own unit-state coefficient).

### 3.6 Symmetry, ZTE, path tracing, DSS (not in step 3)

`symmetry.m` builds `bas.irrep(n).projector` from `sym_group`/`sym_spins`
(groups S2–S8A; cannot cross substances; `spsortrows`/`spunicols` MEX used
here only). `reduce(spin_system, L, ρ)` (Liouville formalisms): return `{1}`
if `'trajlevel'` disabled; symmetry projectors kept by `irrep_drop`, else
`zte`; then `path_trace` on each. `zte.m`: skip if disabled, ρ density >
`zte_maxden`, or `‖ρ‖₁ < zte_tol`; else up to `zte_nsteps` Krylov steps of
`1/cheap_norm(L)`, drop states never exceeding `zte_tol`. `path_trace.m`:
skip if disabled or `dim < merge_dim`; Tarjan components of `|L| >
liouv_zero`, drop components with projection of ρ below `subs_drop`. DSS is
in `evolution.m`: `reduce(spin_system, L', coil)` unless `'dss'` is disabled.
These are propagation-time reductions and belong with the propagation step.

## 4. Accuracy and validation (KAR11, TUT18)

- A two-spin coupling changes the correlation order by at most ±1, so norm
  flows up the orders at speed `2h` and is damped at rate `k·r` in order `k`.
  Steady state is Gaussian in `k`; the leakage above `k` is below `ξ` when
  `k > 2√(h/r)·erfc⁻¹(ξ·erfc(½√(r/h)))`. Proton NMR with J ≈ 5 Hz, r ≈ 1 Hz
  needs `k ≈ 8` for 1 %. Without relaxation the truncated simulation is
  still exact for `t ≤ (k−1)/(2h)`.
- Validation observables used by the authors, which become Spynach's
  regression tests: (i) squared norm of the projection of ρ(t) onto each
  correlation order (`correlation()` filter), log scale; (ii) convergence of
  the spectrum with `inter_level`; (iii) invariance of integrals and phases
  under truncation (K07). Reference numbers: K07 Table 1 and K08 Table 1
  (state counts and Liouvillian densities for chain systems of 6–198 spins),
  SP11 Fig. 3 (sucrose, 22 spins: IK-2 206 states; IK-1(6,3) 4906; IK-1(2,1)
  21), HHK10 Tables I–III (radical pairs, explicit state lists), TUT18
  (difluoroheptane, 16 spins: 1 924 374 → 373 957 (A1g) → 360 770 (ZTE)).

## 5. Chemistry and composite spaces: what the state layout must allow

### 5.1 Species (`inter.chem`, `kernel/kinetics/kinetics.m`)

- One spin system lists all spins of all species. `chem.parts` = list of
  spin-index sets (every spin in exactly one part; default all spins in one
  part), `chem.concs` (required when more than one part), `chem.rates`
  (`nparts × nparts`, `dc/dt = rates·c`, columns sum to zero; `rates(i,j)`
  is the flow from species j into species i), `chem.flux_rate/flux_type`
  (single-spin magnetisation flux without parts). Couplings between parts
  are an error in `create.m`; proximity/dipolar couplings are computed within
  a part only. More than one part forces `sphten-liouv`. With `rates`, all
  parts must have the same number of spins and the same sub-basis.
- `kinetics(spin_system)`: for each non-zero `rates(i,j)`, take the basis
  rows living on part j (`any(basis(:, spins_j))`) and on part i, check the
  two sub-basis blocks are identical (same rows, same order, which the
  lexicographic sort guarantees), and add `rates(i,j)` at `(rows_i, rows_j)`.
  The unit state is untouched. `liquid.m` forms `L = H + i·R + i·K`.
- Nonlinear kinetics (AC25, `react_gen.m`): concentrations are solved first;
  spin evolution is then linear in the concentration-weighted ρ, with
  constant drain/fill index maps built from the same basis matrix via a
  reactant→product spin matching table. The first coefficient of each
  substance's sub-vector is its concentration, which is why every substance
  needs its own unit state (segmented basis).

### 5.2 Classical coordinates (FP16, AL19, AC25)

- Composite space = `[periodic coordinates] ⊗ [space] ⊗ [chemistry] ⊗ [spin]`,
  spin last/fastest. Spatial generators are `M ⊗ 1_spin`, pure spin
  generators `1 ⊗ L`, gradients and coil maps `diag(coords) ⊗ S_op`. First-
  order kinetics is `rates ⊗ 1_spin` on the direct sum of species; diffusion
  and flow are finite-volume/finite-difference matrices on the spatial index.
  Rotor turning, RF phase advance, diffusion, exchange and SLE diffusion are
  the same object: a generator on a classical index kron'ed with a spin
  identity.
- The generator is kept as a short sum of unopened Kronecker products
  (polyadic): `[A⊗B]v = vec(B V Aᵀ)` in MATLAB's column-major layout; identity
  factors are skipped, pre/post multipliers are replayed. **The state vector
  is always dense and uncompressed**; everything is reduced to matvecs
  (Taylor with 1-norm estimate, Krylov).
- Observables are short sums `Σ phantom_k ⊗ spin_state_k`; the average spin
  state is the sum over classical indices; RF-phase dimensions are summed
  out at the end of a pulse.

### 5.3 Implications for Spynach

1. **Basis**: dense `int` array `(nstates, nspins)` on `Basis`, plus
   `tot_proj`, `tot_cord`, the per-substance membership, and cached product
   tables. Rows sorted lexicographically, unit row first. Substances are
   handled in `basis()` and `kinetics()` only; `operator`/`state`/
   `hamiltonian` never need to know about them.
2. **State vectors** are 1-D complex arrays of length `nstates` (Spynach
   already returns 1-D time series). A composite state is a 1-D array of
   length `n_classical·nstates` with the spin index fastest, viewed as
   `(…, nstates)` in C order; the MATLAB `vec(B V Aᵀ)` trick becomes
   `V @ B.T` on the last axis, with `A` acting on the leading axes. State
   stacks (several vectors) are `(nstates, k)` as in Spinach's column stacks.
   Keep this fixed now so that `correlation`/`coherence` (which already
   reshape to `(nstates, spatial)`) and later Fokker-Planck code share one
   convention.
3. **Generators** are `scipy.sparse.csr_array` (complex128) for everything
   in step 3. A `Polyadic` class (terms = lists of factors with identity
   placeholders, `__matmul__` on vectors via reshape, 1-norm estimate) is
   the planned matrix-free container for Fokker-Planck/kinetics later and
   is the natural GPU hook; it is not needed for the setup slice.
4. **Superoperator assembly** is index arithmetic on the basis matrix: for
   a given active-spin set, encode the active columns and the passive
   columns of every row as two integer keys (mixed radix with `mult²`),
   group rows by active key, and for each path look the output active key
   up and intersect on passive keys with `np.searchsorted`/dictionary
   lookups. This is vectorised NumPy; Numba is reserved for cases where
   profiling shows the need (large IK-1 bases). Triplets are summed into
   CSR with `coo_array(...).tocsr()`, which sums duplicates like MATLAB's
   `sparse`.
5. **Hamiltonian**: keep the term-table structure (one row per operator
   term with isotropic coefficient and slots for the rank-1/rank-2
   irreducible coefficients) so that `Q` can be added without changing the
   isotropic path; `operator_type` parameter passed through.
6. **Kinetics** is cheap to port with the basis (pure row matching) and
   gives the first multi-substance fixtures; propagation under `H + iK`
   belongs with the propagation step.
7. Deferred, in this order: propagation paths and reductions (ZTE, path
   tracing, DSS, `reduce`), symmetry, anisotropic `Q`, relaxation,
   polyadic generators and Fokker-Planck, CuPy.

## 6. Relaxation (ported 2026-10-11)

- `relaxation.m` is a dispatcher over theories; ported: `redfield`, `t1_t2`,
  `damp`. The Redfield branch (`includes/redfield_integral_serial.m`) is
  Kuprov 2011's diagonalisation-free BRW: lab-frame `[L0, Q] =
  hamiltonian(assume(labframe))` (commutation superoperators), then for
  every pair of components `Q{n}{k,m}`, `Q{n}{p,q}` and every chemical
  species and correlation-function component with weight `w` and decay
  rate `r` (`corrfun.m`: isotropic `w = δkp δmq/(2n+1)`, `r = -n(n+1)D`,
  `D = 1/(6τc)`; axial and rhombic forms), `R -= w · A · expmint(B, C, D, T)`
  with `A = Q{n}{k,m}`, `B = clean_up(L0, rlx_integration/T)`,
  `C = Q{n}{p,q}'`, `D = B - i r`, `T = -1.5 ln(1/rlx_integration)/r`.
  `expmint` (GK15 auxiliary matrix) is the upper blocks of
  `propagator([[-A, iB],[0, -C]], T)`: `R = P^H Q`. Rows/columns outside the
  species are zeroed. Validity: `1/max|diag R| > τc` else error.
- `rlx_dfs='ignore'` takes the real part; `rlx_keep` then masks: `diagonal`
  (unit-state element zeroed), `kite` (longitudinal block + diagonal),
  `secular` (same Zeeman frequency `Σ basefrq·m`), `labframe`.
- `damp`: Liouville `-rate` on every state but the unit state; Hilbert
  `-rate/2 · 1`. `t1_t2`: `-diag(Σ R1 over l≠0,m=0 spins + Σ R2 over m≠0
  spins)`; tensor rates need an orientation (`ort = [0 0 1]·R(euler)`).
- Thermalisation: `equilibrium.m` applies `exp(-βH)` (left product
  superoperator, via `step` with imaginary time) to the unit state and
  normalises its first element to 1; `thermalize.m`: IME
  `R -= (R ρeq) uᵀ`, DiBari-Levitt `R ← R·propagator(HLSPS, iβ)`.
- Contexts pass `R` per orientation (`powder`), and `evolution`'s
  `multichannel` output (`coil'*rho` for a coil stack, `(ncoils, nsteps+1, k)`)
  is what the NOE examples use.
- Verified: Solomon rates (ρI, ρS, σ) to the integral truncation 1e-4; the
  transient NOE against the 2×2 Solomon solution; equilibrium against the
  Boltzmann matrix; both thermalisations leave ρeq stationary.

## 7. Magic angle spinning (ported 2026-10-11)

`singlerot.m` does not use the polyadic class: it builds the Fokker-Planck
generator explicitly. Per powder orientation `(α,β,γ)` and rotor phase
`φ_n` (`fourdif(2·max_rank+1, 1)` grid), `H_n = I + Σ_r Σ_km D_comp(k,m)
Q{r}{k,m}` with `D_comp = D(rotor_φ, rotor_θ, 0) · D(0,0,φ_n) · D(α,β,γ)`
(no Hermitisation unless `rframes` are present), `G = clean_up(blkdiag(H_n)
+ i·2π·rate·kron(D_φ, 1), liouv_zero)`; `R`, `K` are `kron(1_φ, ·)`; `ρ0` is
`kron(1/N, ρ0)` for powders and `kron(e_1, ρ0)` for `single_crystal`; the
coil is `kron(1, coil)`. The rotor axis polar angles come from `cart2sph`
with `θ = π/2 − elevation`. `evolution`, `reduce` and `acquire` are
dimension-agnostic, so nothing else changes. The rotor-phase grid is
exact for band-limited states: with the initial state averaged over the
phase, single-spin echoes are exact to 1e-9 at rank ≥ anisotropy/rate + a
few; a single-crystal `ρ0` is a delta in `φ` and is only approximate
between echoes (Spinach shares this).

## 8. Open points to settle with fixtures

- Whether `operator()` in `sphten-liouv` reproduces Spinach's sparsity pattern
  exactly (the `kill_mask` in `superop.m`) — compare `nnz` as well as values.
- Element-level agreement of `hamiltonian()`: Spinach sums terms in random
  order and rounds to the `liouv_zero` grid (`clean_up`), so the same
  ~1e-10 rad/s per-element discrepancy seen in the Hilbert slice is expected.
- `state(..., 'exact')` for spin-1 operators (product tables for mult 3) and
  the `'chem'` weighting with unequal concentrations.
- IK-1 with a proximity graph needs `inter.coordinates` (dipolar couplings),
  which is out of scope; the fixtures use `prox_level` with no coordinates
  (proximity graph = identity) and `connectivity = 'scalar_couplings'`.

## 9. The distant dipolar field (2026-11-12)

No Spinach counterpart. The field of a one-dimensionally modulated magnetisation is local (Deville, Bernier
and Delrieux, Phys. Rev. B 19, 5666 (1979)): in the secular rotating-frame form `B_d(s) = μ0 Δ [M(s)/3 −
M_z(s) ẑ]` with `Δ = (3 cos²θ − 1)/2`; the sign follows from the macroscopic field `μ0 [M − k̂ (k̂·M)]` minus
the Lorentz-sphere field `(2/3) μ0 M`, so a helix along `B0` is a stack of discs (`−(2/3) μ0 M_z`), and the
uniform mode carries the sample-shape factor `Δ_shape = (3 N_z − 1)/2` (Lorentz minus demagnetising field, axially
symmetric sample). Representation: the Fourier-mode lattice of the explicit runner of `spynach.program` (one
state vector per wavevector, gradients as shifts of the coherence-order blocks); the field is the convolution
`dρ_n/dt = −i L0 ρ_n − i Σ_m L_d(m) ρ_{n−m}` with `L_d(m)` the commutation superoperator of the mean-field
Hamiltonian from the Fourier coefficients of the magnetisation in mode `m`, assembled as a sparse block
generator for the ported `iserstep`. Mean-field closure in the spirit of Jeener (J. Chem. Phys. 112, 5091
(2000)) and Lee, Richter, Warren et al. (J. Chem. Phys. 105, 874 (1996)): `M = n γ ħ <F>` with the ratio to
the equilibrium magnetisation and `τ_d = 1/(|γ| μ0 M0)` as the only parameter. Validation without an oracle:
the Bessel closed forms of CRAZED and of the multiple spin echoes (Bowtell, Bowley and Glover, J. Magn.
Reson. 88, 643 (1990)), a 1-D grid of positions with the local field, and the demagnetising limits. The
uniform-mode treatment matters at the few-per-cent level once the CRAZED signal is appreciable (the signal's
own magnetisation makes no field in a sphere), which the closed forms of the literature neglect.
