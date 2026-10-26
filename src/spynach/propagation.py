"""Time propagation (ports of ``propagator.m``, ``step.m``, ``krylov.m`` and ``evolution.m``).

Liouville-space states are dense vectors ``(nstates,)`` or stacks
``(nstates, k)``; ``step`` applies ``expm(-1j * L * dt)`` to them without
forming the exponential, and ``evolution`` reduces the problem with
:mod:`spynach.reduction` before propagating each subspace.

Spinach authors, by original file (headers of Spinach 2.13; see docs/spinach-credits.md):
  evolution.m: Ilya Kuprov, Luke Edwards, Ohad Levinkron
  krylov.m: Ilya Kuprov
  propagator.m: Ilya Kuprov, Luke Edwards
  step.m: Ilya Kuprov, Luke Edwards, Anupama Acharya, Callum Musselwhite
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import replace

import numpy as np

from spynach.backend import get_backend
from spynach.system import SpinSystem
from spynach.utilities import Matrix, as_matrix, chop, clean_up, dagger, hdot, matmul, matmul_clean, nnz

log = logging.getLogger("spynach")


def propagator(spin_system: SpinSystem, L: Matrix, timestep: float) -> Matrix:
    """Exponential propagator ``expm(-1j * L * timestep)``.

    Below ``tols.small_matrix`` this is a dense Pade exponential. Larger
    generators use Spinach's scaled and squared Taylor series, rounding each
    term to ``tols.prop_chop`` so that sparsity is kept where it exists.

    Spinach authors (propagator.m): Ilya Kuprov, Luke Edwards.
    """
    if L.ndim != 2 or L.shape[0] != L.shape[1]:
        raise ValueError("L argument must be a square matrix.")
    if not np.isfinite(timestep):
        raise ValueError("timestep must be a finite scalar.")
    tols = spin_system.tols
    be = get_backend()
    if L.shape[0] < tols.small_matrix:
        return be.expm((-1j * timestep) * be.dense(L))  # GPU: small matrices stay on the host, as in Spinach

    # GPU: Spinach moves A to the device above 500 states and runs the Taylor loop and the
    # squarings there (sparse-sparse products); clean_up must then work on device arrays
    A = be.to_device(clean_up(tols, (-1j * timestep) * L, tols.prop_chop))
    mat_norm = be.norm1(A)
    if mat_norm > 1e9:
        raise ValueError("norm of -i*L*dt exceeds 1e9, check your L and your dt.")
    if mat_norm > 1024:
        log.warning(
            "time step greatly exceeds the timescale of system dynamics; "
            "exponentiation tolerance tightened up to eps('double')."
        )
        tols = replace(tols, prop_chop=np.finfo(float).eps)
    elif mat_norm > 16:
        log.warning("time step exceeds the timescale of system dynamics.")

    n_squarings = max(0, math.ceil(math.log2(mat_norm))) if mat_norm > 0 else 0
    if n_squarings > 0:
        A = (1 / 2**n_squarings) * A

    P: Matrix = be.eye(A.shape[0])
    next_term: Matrix = P
    n = 1
    while be.nnz(next_term) > 0:
        if be.issparse(A):
            next_term = matmul_clean(tols, (1 / n) * A, next_term, tols.prop_chop)
        else:
            next_term = clean_up(tols, ((1 / n) * next_term) @ A, tols.prop_chop)
        P = _add(P, next_term)
        n += 1
    log.debug("Taylor series converged in %d iterations.", n)

    P = clean_up(tols, P, tols.prop_chop)
    for _ in range(n_squarings):
        P = matmul_clean(tols, P, P, tols.prop_chop)  # two sparse factors: one threaded pass, rounding included
    return be.to_host(P)  # GPU: propagators are returned to the host, as Spinach gathers them


def step(spin_system: SpinSystem, L: Matrix | Sequence[Matrix], rho: Matrix, timestep: float) -> Matrix:
    """One time step under a constant generator (port of ``step.m``).

    In the Liouville formalisms ``rho`` is a vector or a stack of column
    vectors and the result is ``expm(-1j * L * dt) @ rho`` computed as a
    Taylor series in substeps of ``2 / ||L||_1``. ``L`` may also be a list
    of two or three generators sampled across the step (Lie-group
    quadratures of Rasulov et al. 2023). In ``zeeman-hilb`` ``rho`` is a
    density matrix, propagated by a dense exponential below
    ``tols.small_matrix`` and by the commutator series above it.

    Spinach authors (step.m): Ilya Kuprov, Luke Edwards, Anupama Acharya, Callum Musselwhite.
    """
    if not np.isscalar(timestep) or not np.isfinite(timestep):
        raise ValueError("timestep must be a finite scalar.")
    generators = list(L) if isinstance(L, (list, tuple)) else [L]
    if len(generators) not in (1, 2, 3):
        raise ValueError("L must be a matrix or a list of two or three matrices.")
    if timestep == 0:
        return rho
    if spin_system.basis is None:
        raise ValueError("basis set information is missing, run basis() before calling this function.")
    be = get_backend()
    norm = max(be.norm1(G) for G in generators) * abs(timestep)
    nsteps = max(1, math.ceil(norm / 2))
    if nsteps > 1e4:
        raise ValueError("either dt is too long, or L is too big: |L*dt|>1e4, check both.")
    if nsteps > 100:
        log.warning("%d substeps required, consider using evolution() here.", nsteps)

    if spin_system.basis.formalism == "zeeman-hilb":
        if len(generators) > 1:
            raise NotImplementedError("Lie-group quadratures are not ported for Hilbert space yet.")
        H = generators[0]
        if H.shape[0] < spin_system.tols.small_matrix:
            P = be.expm((-1j * timestep) * be.dense(H))
            return P @ rho @ dagger(P)
        return _commutator_series(H, rho, timestep, nsteps)

    # GPU: step.m moves L and rho to the device whenever 'gpu' is enabled; the Taylor
    # loop below is pure matvec and runs unchanged on device arrays
    rho = be.dense(rho, dtype=complex)
    scaling = float(np.abs(rho).max()) if rho.size else 0.0
    if scaling == 0:
        log.warning("all-zero state received")
        return rho
    generators = [be.to_device(G) for G in generators]
    return be.to_host(scaling * _reordered_taylor(generators, be.to_device(rho / scaling), timestep, nsteps))


def _reordered_taylor(generators: list[Matrix], rho: np.ndarray, t: float, nsteps: int) -> np.ndarray:
    """Taylor series of expm(-1j*L*t) acting on rho, in nsteps substeps (step.m ``reordered_taylor``).

    Spinach authors (step.m, subfunction reordered_taylor): Ilya Kuprov, Luke Edwards, Anupama Acharya, Callum
      Musselwhite.
    """
    eps = np.finfo(float).eps
    dt = t / nsteps
    be = get_backend()
    for _ in range(nsteps):
        next_term, k = rho, 1
        if len(generators) == 1 and be.issparse(generators[0]) and np.any(np.abs(next_term) > eps):
            # One sparse generator: each term, the running sum and the convergence test in a single threaded
            # pass (the sum is accumulated in place, so the start vector, which is also the first term, is kept)
            total = np.array(rho, dtype=complex)
            fused = True
            while True:
                step_result = be.taylor_term(generators[0], next_term, (-1j / k) * dt, total, eps)
                if step_result is None:
                    fused = False
                    break
                next_term, big = step_result
                k += 1
                if not big:
                    break
            if fused:
                rho = total
                if k > 32:
                    log.warning(
                        "loss of accuracy in the Taylor series, k=%d, use evolution() here instead of step()", k
                    )
                continue
            next_term, k = rho, 1  # the kernel does not apply to this problem: the plain loop below
        while np.any(np.abs(next_term) > eps):
            if len(generators) == 1:
                next_term = (-1j / k) * dt * matmul(generators[0], next_term)
            elif len(generators) == 2:
                a, b = generators[0] @ next_term, generators[1] @ next_term
                next_term = (-1j / 2) * (1 / k) * dt * (a + b) + (1 / 6) * (1 / k) * (t**2 / nsteps) * (
                    generators[0] @ b - generators[1] @ a
                )
            else:
                a, b, c = generators[0] @ next_term, generators[1] @ next_term, generators[2] @ next_term
                next_term = (-1j / 6) * (1 / k) * dt * (a + 4 * b + c) + (1 / 12) * (1 / k) * (t**2 / nsteps) * (
                    generators[0] @ c - generators[2] @ a
                )
            rho = rho + next_term
            k += 1
        if k > 32:
            log.warning("loss of accuracy in the Taylor series, k=%d, use evolution() here instead of step()", k)
    return rho


def _commutator_series(H: Matrix, rho: Matrix, t: float, nsteps: int) -> Matrix:
    """Hilbert-space step by the nested commutator series (step.m ``comm_series``).

    Spinach authors (step.m, subfunction comm_series): Ilya Kuprov, Luke Edwards, Anupama Acharya, Callum
      Musselwhite.
    """
    eps = np.finfo(float).eps
    scaling = get_backend().norm1(rho)
    if scaling == 0:
        log.warning("all-zero state received")
        return rho
    rho = rho / scaling
    dt = t / nsteps
    for _ in range(nsteps):
        next_term, k = rho, 1
        while nnz(next_term) > 0:
            next_term = (-1j * dt / k) * (H @ next_term - next_term @ H)
            next_term = chop(next_term, eps)
            rho = _add(rho, next_term)
            k += 1
        if k > 32:
            log.warning("loss of accuracy in the commutator series, iter=%d, use evolution() here instead of step()", k)
        rho = chop(rho, eps)
    return scaling * rho


def krylov(
    spin_system: SpinSystem,
    L: Matrix,
    rho: np.ndarray,
    timestep: float,
    nsteps: int,
    output: str,
    *,
    coil: np.ndarray | None = None,
) -> np.ndarray:
    """Propagation by repeated :func:`step` calls, for generators too large to exponentiate (port of ``krylov.m``).

    Same outputs as :func:`evolution`; ``coil`` is needed for ``'observable'`` and ``'multichannel'``.

    Spinach authors (krylov.m): Ilya Kuprov.
    """
    if spin_system.basis is None or spin_system.basis.formalism != "sphten-liouv":
        raise NotImplementedError("krylov() is only ported for the sphten-liouv formalism.")
    rho = get_backend().dense(rho, dtype=complex)  # GPU: krylov.m keeps L, rho and coil on the device across steps
    if output == "final":
        return step(spin_system, L, rho, timestep * nsteps)
    if output == "trajectory":
        answer = np.zeros((rho.shape[0], nsteps + 1), dtype=complex)
        answer[:, 0] = rho.reshape(-1)
        for n in range(nsteps):
            rho = step(spin_system, L, rho, timestep)
            answer[:, n + 1] = rho.reshape(-1)
        return answer
    if output == "observable":
        answer = np.zeros((nsteps + 1,) + rho.shape[1:], dtype=complex)
        answer[0] = _detect(coil, rho)
        for n in range(nsteps):
            rho = step(spin_system, L, rho, timestep)
            answer[n + 1] = _detect(coil, rho)
        return answer
    if output == "multichannel":
        answer = np.zeros((coil.shape[1], nsteps + 1, rho.shape[1]), dtype=complex)
        answer[:, 0, :] = coil.conj().T @ rho
        for n in range(nsteps):
            rho = step(spin_system, L, rho, timestep)
            answer[:, n + 1, :] = coil.conj().T @ rho
        return answer
    if output == "refocus":
        rho = rho.copy()
        for n in range(1, rho.shape[1]):
            rho[:, n:] = step(spin_system, L, rho[:, n:], timestep)
        return rho
    raise NotImplementedError(f"krylov output {output!r} not ported yet.")


def evolution(
    spin_system: SpinSystem,
    L: Matrix,
    rho: Matrix,
    timestep: float,
    nsteps: int,
    output: str,
    *,
    coil: Matrix | None = None,
    destination: np.ndarray | None = None,
) -> np.ndarray:
    """Time evolution under a constant generator (port of ``evolution.m``).

    Spinach's ``evolution(spin_system, L, coil, rho, dt, nsteps, output, destination)``
    with the detection state ``coil`` and the ``destination`` state as keyword
    arguments, since most calls have neither (see ``docs/coming-from-spinach.md``).

    ``zeeman-hilb``: ``output='observable'`` only; returns ``hdot(coil, rho(t))``
    at ``nsteps + 1`` equally spaced time points starting at t = 0, as a 1-D
    complex array. Spinach parallelises this case with the Kuprov-Edwards
    split over the columns of rho; the serial loop computes the same thing.

    ``sphten-liouv``: ``rho`` is a vector or a stack of column vectors. The
    problem is first reduced with :func:`spynach.reduction.reduce`, screened
    against ``destination`` or, for ``'observable'``, the coil state
    (destination state screening, unless ``'dss'`` is disabled), then each
    subspace is propagated with the exponential propagator, or with
    :func:`krylov` above ``tols.krylov_tol`` states. Outputs: ``'observable'``
    (``coil^H rho(t)``, shape ``(nsteps + 1,)`` or ``(nsteps + 1, k)``),
    ``'final'`` (the state after ``nsteps`` steps), ``'trajectory'``
    (``(nstates, nsteps + 1)``, single vector only), ``'multichannel'``
    (several coils stacked as the columns of ``coil``: ``(ncoils, nsteps + 1,
    k)``, as in Spinach), ``'refocus'`` (column ``k`` of the stack propagated
    for ``k`` steps, the second half of a constant-time or refocused
    indirect dimension; ``nsteps`` is ignored).

    Spinach authors (evolution.m): Ilya Kuprov, Luke Edwards, Ohad Levinkron.
    """
    if spin_system.basis is None:
        raise ValueError("basis set information is missing, run basis() before calling this function.")
    if isinstance(rho, (list, tuple)):
        raise NotImplementedError("state stacks as lists are not ported yet; stack column vectors in an array.")
    if spin_system.basis.formalism == "zeeman-hilb":
        return _evolution_hilbert(spin_system, L, rho, timestep, nsteps, output, coil)
    if spin_system.basis.formalism != "sphten-liouv":
        raise NotImplementedError(f"formalism {spin_system.basis.formalism!r} not ported yet.")
    if output not in _OUTPUTS:
        raise NotImplementedError(f"evolution output {output!r} not ported yet.")

    from spynach.reduction import reduce  # noqa: PLC0415  (reduction imports step from this module)

    tols = spin_system.tols
    be = get_backend()
    rho = be.dense(rho, dtype=complex)
    single = rho.ndim == 1
    rho = rho.reshape(rho.shape[0], -1)
    if output == "trajectory" and rho.shape[1] > 1:
        raise ValueError("trajectory output is not available for state vector stacks.")
    detected = output in ("observable", "multichannel")
    if detected and coil is None:
        raise ValueError("coil state must be given for the observable and multichannel outputs.")
    if coil is not None:
        coil = be.dense(coil, dtype=complex)
        coil = coil.reshape(-1, 1) if coil.ndim == 1 else coil

    # Trajectory-level reduction, run backwards from the destination or coil state where there is one
    if "dss" in spin_system.disable:
        projectors = reduce(spin_system, L, rho)
    elif destination is not None:
        projectors = reduce(spin_system, dagger(L), destination)
    elif detected:
        projectors = reduce(spin_system, dagger(L), coil)
    else:
        projectors = reduce(spin_system, L, rho)

    evolve = _OUTPUTS[output]
    answer = _result_array(output, rho, coil, nsteps)
    use_krylov = "krylov" in spin_system.enable
    for P in projectors:
        P_dag = P.conj().T
        L_loc = clean_up(tols, as_matrix(P_dag @ L @ P), tols.liouv_zero)
        rho_loc = chop(P_dag @ rho, tols.zte_tol)
        coil_loc = chop(P_dag @ coil, tols.zte_tol) if detected else None
        krylov_path = use_krylov or (L_loc.shape[0] >= tols.krylov_tol and "krylov" not in spin_system.disable)
        # GPU: evolution.m moves P, rho_loc and coil_loc to the device for the loops below
        # and gathers the observables; the propagator itself is built as above
        result = evolve(spin_system, L_loc, rho_loc, coil_loc, timestep, nsteps, krylov_path)
        # Detected outputs add up over the subspaces; state outputs are projected back into the full space
        answer += result if detected else P @ chop(result, tols.zte_tol)
    if output in ("final", "refocus"):
        answer = chop(answer, tols.zte_tol)
    if single and output in ("observable", "final", "refocus"):
        answer = answer[:, 0]
    elif single and output == "multichannel":
        answer = answer[:, :, 0]
    return answer


def _evolution_hilbert(
    spin_system: SpinSystem, H: Matrix, rho: Matrix, timestep: float, nsteps: int, output: str, coil: Matrix | None
) -> np.ndarray:
    """``hdot(coil, rho(t))`` under ``P rho P^H`` steps; the ``zeeman-hilb`` branch of :func:`evolution`."""
    if output != "observable":
        raise NotImplementedError(f"evolution output {output!r} not ported yet in zeeman-hilb; only 'observable'.")
    if coil is None:
        raise ValueError("coil state must be given for the observable output.")
    if coil.shape != rho.shape:
        raise ValueError("coil and rho must have identical dimensions.")
    P = propagator(spin_system, H, timestep)
    P_dag = dagger(P)
    answer = np.empty(nsteps + 1, dtype=complex)
    for n in range(nsteps + 1):
        answer[n] = hdot(coil, rho)
        rho = P @ rho @ P_dag
    return answer


def _result_array(output: str, rho: np.ndarray, coil: np.ndarray | None, nsteps: int) -> np.ndarray:
    """Zero array of the shape :func:`evolution` returns for a stack ``rho`` of shape ``(nstates, k)``."""
    if output == "observable":
        return np.zeros((nsteps + 1, rho.shape[1]), dtype=complex)
    if output == "multichannel":
        return np.zeros((coil.shape[1], nsteps + 1, rho.shape[1]), dtype=complex)
    if output in ("final", "refocus"):
        return np.zeros_like(rho)
    return np.zeros((rho.shape[0], nsteps + 1), dtype=complex)


# One function per evolution() output, all with the signature
# (spin_system, L, rho, coil, timestep, nsteps, krylov_path) on one reduced subspace:
# L is the reduced generator, rho the reduced stack (nstates, k), coil the reduced
# detection state(s) or None; krylov_path selects step()-based propagation over the
# exponential propagator. Detected outputs return their observable, the others the
# propagated stack, which evolution() projects back into the full space.


def _evolve_observable(spin_system, L, rho, coil, timestep, nsteps, krylov_path) -> np.ndarray:
    """``coil^H rho(t)`` at ``nsteps + 1`` points: ``(nsteps + 1, k)``."""
    if krylov_path:
        return krylov(spin_system, L, rho, timestep, nsteps, "observable", coil=coil)
    prop = propagator(spin_system, L, timestep)
    answer = np.zeros((nsteps + 1, rho.shape[1]), dtype=complex)
    for n in range(nsteps + 1):
        answer[n] = _detect(coil, rho)
        rho = matmul(prop, rho)
    return answer


def _evolve_multichannel(spin_system, L, rho, coil, timestep, nsteps, krylov_path) -> np.ndarray:
    """Every column of ``coil`` detected at every point: ``(ncoils, nsteps + 1, k)``."""
    if krylov_path:
        return krylov(spin_system, L, rho, timestep, nsteps, "multichannel", coil=coil)
    prop = propagator(spin_system, L, timestep)
    answer = np.zeros((coil.shape[1], nsteps + 1, rho.shape[1]), dtype=complex)
    for n in range(nsteps + 1):
        answer[:, n, :] = coil.conj().T @ rho
        rho = matmul(prop, rho)
    return answer


def _evolve_final(spin_system, L, rho, coil, timestep, nsteps, krylov_path) -> np.ndarray:
    """The stack after ``nsteps`` steps."""
    if krylov_path:
        return krylov(spin_system, L, rho, timestep, nsteps, "final")
    # Optimal stepping: fewer, longer steps when the propagator is cheap relative to the stack
    nsteps_opt = max(1, math.ceil(math.log2(L.shape[0] / (rho.shape[1] * math.log(2)))))
    prop = propagator(spin_system, L, timestep * nsteps / nsteps_opt)
    for _ in range(nsteps_opt):
        rho = matmul(prop, rho)
    return rho


def _evolve_refocus(spin_system, L, rho, coil, timestep, nsteps, krylov_path) -> np.ndarray:
    """Column ``k`` of the stack propagated for ``k`` steps (``nsteps`` is ignored)."""
    if krylov_path:
        return krylov(spin_system, L, rho, timestep, nsteps, "refocus")
    prop = propagator(spin_system, L, timestep)
    rho = rho.copy()
    for n in range(1, rho.shape[1]):
        rho[:, n:] = matmul(prop, rho[:, n:])
    return rho


def _evolve_trajectory(spin_system, L, rho, coil, timestep, nsteps, krylov_path) -> np.ndarray:
    """The single state vector at every point: ``(nstates, nsteps + 1)``."""
    if krylov_path:
        return krylov(spin_system, L, rho, timestep, nsteps, "trajectory")
    prop = propagator(spin_system, L, timestep)
    traj = np.zeros((L.shape[0], nsteps + 1), dtype=complex)
    for n in range(nsteps + 1):
        traj[:, n] = rho[:, 0]
        rho = matmul(prop, rho)
    return traj


_OUTPUTS = {
    "observable": _evolve_observable,
    "multichannel": _evolve_multichannel,
    "final": _evolve_final,
    "refocus": _evolve_refocus,
    "trajectory": _evolve_trajectory,
}


def _detect(coil: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """``coil' * rho`` for a coil column and a state stack ``(nstates, k)``, giving ``(k,)``."""
    return (coil.conj().T @ rho).reshape(-1)


def _add(a: Matrix, b: Matrix) -> Matrix:
    """Sum of two matrices that may be dense or sparse; dense if either is dense."""
    be = get_backend()
    if be.issparse(a) and be.issparse(b):
        return be.csr(a + b)
    return be.dense(a) + be.dense(b)
