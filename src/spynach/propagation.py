"""Time propagation (ports of ``propagator.m``, ``step.m``, ``krylov.m`` and ``evolution.m``).

Liouville-space states are dense vectors ``(nstates,)`` or stacks
``(nstates, k)``; ``step`` applies ``expm(-1j * L * dt)`` to them without
forming the exponential, and ``evolution`` reduces the problem with
:mod:`spynach.reduction` before propagating each subspace.

Spinach authors, by original file (headers of Spinach 2.13; see docs/spinach-credits.md):
  comm.m: Ilya Kuprov
  evolution.m: Ilya Kuprov, Luke Edwards, Ohad Levinkron
  iserstep.m: Ilya Kuprov, Andrew Graham, Anupama Acharya
  krylov.m: Ilya Kuprov
  propagator.m: Ilya Kuprov, Luke Edwards
  step.m: Ilya Kuprov, Luke Edwards, Anupama Acharya, Callum Musselwhite
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Sequence
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


ISERSTEP_METHODS = ("PWCL", "LG2", "LG4", "LG4A", "RKMK4", "RKMK-DP5", "RKMK-DP8", "RKMK-RKF45")


def iserstep(
    spin_system: SpinSystem, L: Callable[[float, Matrix], Matrix], t: float, method: str, rho: Matrix, timestep: float
) -> Matrix:
    """One step of ``d rho / dt = -1j * L(t, rho) @ rho`` with a state-dependent generator (port of ``iserstep.m``).

    ``L(t, rho)`` returns the evolution generator (rad/s) at time ``t`` for the state ``rho``; ``t`` is the
    time at the start of the step. The difference from :func:`step` is that the generator can depend on the
    density matrix, which is what radiation damping, the distant dipolar field and other mean-field terms
    need; a time-dependent generator alone is also fine. Spinach takes ``{L, t, method}`` as one cell array,
    this port takes the three as arguments.

    Methods: ``'PWCL'`` (the generator at the left edge, first order), ``'LG2'`` (midpoint generator, second
    order), ``'LG4'`` (left, midpoint and estimated right generators through the fourth-order Lie quadrature
    of :func:`step`), ``'LG4A'`` (the explicit fourth-order Lie-group method of Casas and Iserles, J. Phys.
    A 39, 5445 (2006), Appendix A.1), ``'RKMK4'`` (fourth-order Runge-Kutta-Munthe-Kaas) and the
    Runge-Kutta-Munthe-Kaas schemes with the Dormand-Prince 5(4) and 8(7) and Fehlberg 4(5) tableaux
    (``'RKMK-DP5'``, ``'RKMK-DP8'``, ``'RKMK-RKF45'``), whose stages are corrected by the truncated
    inverse of the differential of the exponential map. Every method exponentiates, so a unitary generator
    gives a unitary step. ``'LG4'`` needs the three-generator :func:`step`, which is Liouville-space only;
    the other methods work in both formalisms. A zero ``timestep`` returns ``rho``.

    Spinach authors (iserstep.m): Ilya Kuprov, Andrew Graham, Anupama Acharya.
    """
    if method not in ISERSTEP_METHODS:
        raise ValueError(f"method must be one of {ISERSTEP_METHODS}, not {method!r}.")
    if not callable(L):
        raise ValueError("L must be a function L(t, rho) returning the evolution generator.")
    if not np.isscalar(t) or not np.isfinite(t):
        raise ValueError("t must be a finite scalar.")
    if not np.isscalar(timestep) or not np.isfinite(timestep):
        raise ValueError("timestep must be a finite scalar.")
    if timestep == 0:
        return rho
    dt = timestep
    if method == "PWCL":
        # Left generator from the left state, assumed constant over the step
        return step(spin_system, L(t, rho), rho, dt)
    if method == "LG2":
        # Left generator, estimated midpoint state, midpoint generator over the whole step
        LL = L(t, rho)
        rho_mid = step(spin_system, LL, rho, dt / 2)
        return step(spin_system, L(t + dt / 2, rho_mid), rho, dt)
    if method == "LG4":
        # Left and midpoint generators, endpoint estimated with the midpoint generator, then the quadrature
        LL = L(t, rho)
        rho_mid = step(spin_system, LL, rho, dt / 2)
        LM = L(t + dt / 2, rho_mid)
        rho_b = step(spin_system, LM, rho, dt)
        return step(spin_system, [LL, LM, L(t + dt, rho_b)], rho, dt)
    if method == "LG4A":
        return _lie_group_4a(spin_system, L, t, rho, dt)
    if method == "RKMK4":
        LL = L(t, rho)
        rho_mid = step(spin_system, LL, rho, dt / 2)
        LMA = L(t + dt / 2, rho_mid)
        rho_mid = step(spin_system, LMA + 1j * dt * (1 / 6) * _comm(LL, LMA), rho, dt / 2)
        LMB = L(t + dt / 2, rho_mid)
        rho_right = step(spin_system, LMB + 1j * dt * (1 / 6) * _comm(LL, LMB), rho, dt)
        LR = L(t + dt, rho_right)
        LA = (1 / 6) * (LL + 2 * LMA + 2 * LMB + LR)
        LC = _comm(LL - LR, LMA + LMB) + _comm(LL, LR)
        return step(spin_system, LA + 1j * dt * (1 / 36) * LC, rho, dt)
    return _rkmk(method[5:], spin_system, L, t, rho, dt)


def _comm(A: Matrix, B: Matrix) -> Matrix:
    """Commutator ``A @ B - B @ A`` of two generators (``comm.m``).

    Spinach authors (comm.m): Ilya Kuprov.
    """
    return A @ B - B @ A


def _lie_group_4a(spin_system: SpinSystem, L, t: float, rho: Matrix, dt: float) -> Matrix:
    """The fourth-order explicit Lie-group method of Casas and Iserles, Appendix A.1 (iserstep.m, case 'LG4A').

    Spinach authors (iserstep.m, case LG4A): Ilya Kuprov, Andrew Graham, Anupama Acharya.
    """
    # Lie algebra elements are exponentiated by step() as exp(-1j * (1j/dt * u) * dt) = exp(u)
    h, scale = dt, 1j / dt
    a_fun = lambda tt, state: -1j * L(tt, state)  # noqa: E731
    k1 = h * a_fun(t, rho)
    Q1 = k1
    u2 = 0.5 * Q1
    k2 = h * a_fun(t + h / 2, step(spin_system, scale * u2, rho, dt))
    Q2 = k2 - k1
    u3 = 0.5 * Q1 + 0.25 * Q2
    k3 = h * a_fun(t + h / 2, step(spin_system, scale * u3, rho, dt))
    Q3 = k3 - k2
    u4 = Q1 + Q2
    k4 = h * a_fun(t + h, step(spin_system, scale * u4, rho, dt))
    Q4 = k4 - 2 * k2 + k1
    u5 = 0.5 * Q1 + 0.25 * Q2 + (1 / 3) * Q3 - (1 / 24) * Q4 - (1 / 48) * _comm(Q1, Q2)
    k5 = h * a_fun(t + h / 2, step(spin_system, scale * u5, rho, dt))
    Q5 = k5 - k2
    u6 = Q1 + Q2 + (2 / 3) * Q3 + (1 / 6) * Q4 - (1 / 6) * _comm(Q1, Q2)
    k6 = h * a_fun(t + h, step(spin_system, scale * u6, rho, dt))
    Q6 = k6 - 2 * k2 + k1
    v = Q1 + Q2 + (2 / 3) * Q5 + (1 / 6) * Q6 - (1 / 6) * (Q1 @ Q2 - (Q2 - Q3 + Q5 + 0.5 * Q6) @ Q1)
    return step(spin_system, scale * v, rho, dt)


# Bernoulli numbers B_0 ... B_10, for the truncated series of the inverse differential of the exponential map
_BERNOULLI = (1, -1 / 2, 1 / 6, 0, -1 / 30, 0, 1 / 42, 0, -1 / 30, 0, 5 / 66)


def _dexpinv(u: Matrix, v: Matrix, order: int) -> Matrix:
    """``dexp^{-1}_u(v)`` as the Bernoulli series in nested commutators, truncated after ``order`` terms."""
    if order > 10:
        raise ValueError(f"extend _BERNOULLI for order {order}.")
    w, ad_term = v, v
    for n in range(1, order + 1):
        ad_term = _comm(u, ad_term)
        if _BERNOULLI[n] != 0:
            w = w + (_BERNOULLI[n] / math.factorial(n)) * ad_term
    return w


def _rk_tableau(name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Stage matrix ``A``, weights ``b``, nodes ``c`` and order of the named explicit Runge-Kutta scheme."""
    if name == "DP5":
        c = np.array([0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1, 1])
        A = np.zeros((7, 7))
        A[1, 0] = 1 / 5
        A[2, :2] = [3 / 40, 9 / 40]
        A[3, :3] = [44 / 45, -56 / 15, 32 / 9]
        A[4, :4] = [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729]
        A[5, :5] = [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656]
        A[6, :6] = [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84]
        b = np.array([35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0])
        return A, b, c, 5
    if name == "RKF45":
        c = np.array([0, 1 / 4, 3 / 8, 12 / 13, 1, 1 / 2])
        A = np.zeros((6, 6))
        A[1, 0] = 1 / 4
        A[2, :2] = [3 / 32, 9 / 32]
        A[3, :3] = [1932 / 2197, -7200 / 2197, 7296 / 2197]
        A[4, :4] = [439 / 216, -8, 3680 / 513, -845 / 4104]
        A[5, :5] = [-8 / 27, 2, -3544 / 2565, 1859 / 4104, -11 / 40]
        b = np.array([16 / 135, 0, 6656 / 12825, 28561 / 56430, -9 / 50, 2 / 55])
        return A, b, c, 5
    if name == "DP8":
        c = np.array(
            [0, 1 / 18, 1 / 12, 1 / 8, 5 / 16, 3 / 8, 59 / 400, 93 / 200, 5490023248 / 9719169821, 13 / 20,
             1201146811 / 1299019798, 1, 1]
        )  # fmt: skip
        A = np.zeros((13, 13))
        A[1, 0] = 1 / 18
        A[2, :2] = [1 / 48, 1 / 16]
        A[3, 0], A[3, 2] = 1 / 32, 3 / 32
        A[4, 0], A[4, 2], A[4, 3] = 5 / 16, -75 / 64, 75 / 64
        A[5, 0], A[5, 3], A[5, 4] = 3 / 80, 3 / 16, 3 / 20
        A[6, 0], A[6, 3:6] = 29443841 / 614563906, [77736538 / 692538347, -28693883 / 1125000000, 23124283 / 1800000000]
        A[7, 0] = 16016141 / 946692911
        A[7, 3:7] = [61564180 / 158732637, 22789713 / 633445777, 545815736 / 2771057229, -180193667 / 1043307555]
        A[8, 0] = 39632708 / 573591083
        A[8, 3:8] = [
            -433636366 / 683701615, -421739975 / 2616292301, 100302831 / 723423059, 790204164 / 839813087,
            800635310 / 3783071287,
        ]  # fmt: skip
        A[9, 0] = 246121993 / 1340847787
        A[9, 3:9] = [
            -37695042795 / 15268766246, -309121744 / 1061227803, -12992083 / 490766935, 6005943493 / 2108947869,
            393006217 / 1396673457, 123872331 / 1001029789,
        ]  # fmt: skip
        A[10, 0] = -1028468189 / 846180014
        A[10, 3:10] = [
            8478235783 / 508512852, 1311729495 / 1432422823, -10304129995 / 1701304382, -48777925059 / 3047939560,
            15336726248 / 1032824649, -45442868181 / 3398467696, 3065993473 / 597172653,
        ]  # fmt: skip
        A[11, 0] = 185892177 / 718116043
        A[11, 3:11] = [
            -3185094517 / 667107341, -477755414 / 1098053517, -703635378 / 230739211, 5731566787 / 1027545527,
            5232866602 / 850066563, -4093664535 / 808688257, 3962137247 / 1805957418, 65686358 / 487910083,
        ]  # fmt: skip
        A[12, 0] = 403863854 / 491063109
        A[12, 3:12] = [
            -5068492393 / 434740067, -411421997 / 543043805, 652783627 / 914296604, 11173962825 / 925320556,
            -13158990841 / 6184727034, 3936647629 / 1978049680, -160528059 / 685178525, 248638103 / 1413531060, 0,
        ]  # fmt: skip
        b = np.array(
            [
                14005451 / 335480064, 0, 0, 0, 0, -59238493 / 1068277825, 181606767 / 758867731,
                561292985 / 797845732, -1041891430 / 1371343529, 760417239 / 1151165299, 118820643 / 751138087,
                -528747749 / 2220607170, 1 / 4,
            ]
        )  # fmt: skip
        return A, b, c, 8
    raise ValueError(f"unknown Runge-Kutta tableau {name!r}.")


def _rkmk(name: str, spin_system: SpinSystem, L, t: float, rho: Matrix, dt: float) -> Matrix:
    """Explicit Runge-Kutta-Munthe-Kaas step with the named tableau (iserstep.m, subfunction RKMK).

    Spinach authors (iserstep.m, subfunction RKMK): Ilya Kuprov, Andrew Graham, Anupama Acharya.
    """
    A, b, c, order = _rk_tableau(name)
    order_dexp = order - 1
    scale = 1j / dt
    K: list[Matrix] = []
    for n in range(len(b)):
        # Lie algebra element of this stage, the stage state it gives, and the field there
        u_n = None
        for m in range(n):
            if A[n, m] != 0:
                term = dt * A[n, m] * K[m]
                u_n = term if u_n is None else u_n + term
        rho_n = rho if u_n is None else step(spin_system, scale * u_n, rho, dt)
        k_raw = -1j * L(t + c[n] * dt, rho_n)
        K.append(k_raw if u_n is None or order_dexp == 0 else _dexpinv(u_n, k_raw, order_dexp))
    v = None
    for n in range(len(b)):
        if b[n] != 0:
            term = dt * b[n] * K[n]
            v = term if v is None else v + term
    return rho if v is None else step(spin_system, scale * v, rho, dt)


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
