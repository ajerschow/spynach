# GPU acceleration: plan and markers

Status (2026-10-11): deferred. The development container has no GPU and
the only host so far is an Apple-silicon Mac (no CUDA). This file records
where GPU acceleration enters the kernel, what the code already does to
make that easy, and what is still open, so that the work can start as soon
as a target device is decided.

## Where Spinach uses the GPU (`sys.enable={'gpu'}`)

| Spinach | what runs on the device | Spynach location |
|---|---|---|
| `propagator.m` | above 500 states: the Taylor loop with sparse-sparse products and `clean_up` between terms, then the squarings; the result is gathered | `propagation.propagator` |
| `step.m` | the Taylor action of `L` (or the 2/3-point quadrature generators) on the state vector or stack | `propagation.step` / `_reordered_taylor` |
| `krylov.m` | keeps `L`, `rho`, `coil` on the device across all steps | `propagation.krylov` |
| `evolution.m` | per subspace: `P`, `rho_loc`, `coil_loc` moved to the device for the `P @ rho` loops; observables gathered | `propagation.evolution` |
| `cheap_norm.m` | `norm(A, inf)` on device arrays | `backend.norm1` |
| `clean_up.m` | rounding and sparse/dense decisions on device arrays | `utilities.clean_up` |
| `zte.m` | trial propagation through `step` | `reduction.zte` |

Spinach never puts basis construction, `superop`, `hamiltonian`,
`relaxation` or `kinetics` on the GPU: they are index arithmetic with
small dense kernels, and Spynach keeps the same split.

## What is in place

- `spynach.backend` is the single adapter: `get_backend()` returns an object
  with `xp` (array module), `issparse`, `csr`, `eye`, `dense`, `expm`,
  `nnz`, `norm1`, `to_device`, `to_host`, and the four kernel entry points
  `round_array`, `spgemm_round`, `csr_times_dense`, `taylor_term` (threaded
  Numba kernels in `spynach.kernels` on the CPU; a device backend implements
  them with its own sparse products, or returns `None` to take the generic
  path). Only `NumpyBackend` exists; the CPU transfers are no-ops. `backend.use('cupy')` is reserved.
- The hot loops go through it: `propagator` (device transfer of the
  generator, Taylor loop, squarings, host return), `step` (generators and
  state to the device, host return), `krylov`, `evolution`, `_chop`,
  `_add`, `_norm1`.
- Every line where device placement will matter is marked `# GPU:` with
  the Spinach behaviour it mirrors (`grep -rn "GPU:" src/spynach`).
- `sys['enable'] = ['gpu']` is refused with a message pointing here, so
  user scripts that set it in Spinach fail loudly rather than silently
  running on the CPU.
- dtype is a parameter everywhere (`complex` defaults, no hard-coded
  `complex128` in the loops), which is what mixed precision needs.

## What a CuPy backend has to provide

- `xp = cupy`; `csr = cupyx.scipy.sparse.csr_matrix` (CuPy has no
  `*_array` API); `eye` from `cupyx.scipy.sparse.eye`; `dense` via
  `.toarray()`/`cupy.asarray`; `to_device = cupy.asarray` for dense and
  `cupyx.scipy.sparse.csr_matrix(x)` for SciPy sparse input; `to_host =
  cupy.asnumpy` and `.get()` for sparse.
- `expm`: stays on the host (`scipy.linalg.expm` of a gathered small
  matrix), exactly as Spinach only uses `expm` below `small_matrix`.
- `utilities.clean_up` must accept device arrays: `matlab_round` is pure
  array-module arithmetic (NumPy's `np.sign`/`np.floor` dispatch to CuPy
  through `__array_function__`), but the `sps.issparse`/`sps.csr_array`
  calls there must become `backend.issparse`/`backend.csr`. This is the
  one remaining direct SciPy use on the hot path; it was left because the
  function is also used by construction code on the CPU.
- `nnz` likewise (`count_nonzero` exists on both).
- `reduction.path_trace` uses `scipy.sparse.csgraph`; it runs once per
  evolution on the host and does not need a device version.

## Construction-side notes (2026-10-11)

Superoperator assembly (`operators.superop`) is index arithmetic on
64-bit row hashes with NumPy `searchsorted`, cached per basis; Hamiltonian
and `Q` blocks are built from concatenated COO triplets. None of this
belongs on a device: it runs once per system and is already far cheaper
than propagation at the sizes where a GPU would matter.

## Open decisions

- **Target device.** CuPy requires CUDA (NVIDIA). On Apple silicon the
  options are MLX or PyTorch-MPS, neither of which has sparse complex
  matrix products; the Liouville-space loops would have to run dense,
  which is only competitive below a few thousand states. Decide the device
  before writing any backend.
- **Sparse-sparse products in the Taylor loop** (dim 500–10 000): the
  HANDOFF risk. Benchmark `cupyx` CSR·CSR against densifying early before
  settling on the propagator's device path.
- **Mixed precision:** the plan is build in double on the CPU, propagate
  in `complex64` on the device, accumulate observables in double. Needs
  the parity harness re-run at `complex64` to find where it is acceptable.
- **Transfers:** the current markers transfer per call (`propagator`,
  `step`). For `evolution` loops the state should stay on the device
  across steps, as `krylov.m` does; restructure `evolution` so that the
  per-subspace loop is one device session.
