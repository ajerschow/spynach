"""Finding phase cycles: cogwheel search by exact integer arithmetic, with the published reductions.

Problem (Jerschow & Kumar 2003; Levitt, Madhu & Hughes 2002; Miloushev & Palmer 2018): given
the set ``A`` of pathways the allowed orders permit, with order jumps ``Δp(u)``, and the wanted
pathways ``W``, find ``COG_N(ν; ν_rec)`` such that every wanted pathway survives with one
receiver phase and every unwanted one is rejected. In the difference set ``D = {Δp(u) - Δp(w₀)}``
relative to one wanted pathway, a cogwheel cycle rejects ``u`` iff ``ν·δ(u) ≢ 0 (mod N)`` and
keeps the wanted ones iff ``ν·δ(w) ≡ 0``; the receiver is ``ν_rec = -ν·Δp(w₀) mod N`` (2003 Eq. 4).

Reductions (Hughes, Carravetta & Levitt 2004 p. 262; Zuckerstätter 2007): pulses whose ``δ``
component is always zero need no winding number; when every pathway has the same total jump
(fixed first and last orders) a uniform shift ``ν -> ν + c`` changes nothing, so one free
winding number is fixed at 0; multiplying ``ν`` by a unit modulo ``N`` gives the same selection,
so the first nonzero free winding number is restricted to divisors of ``N`` (this covers the
sign flip); a common factor of all ``δ`` divides ``N``. The remaining candidates are tested in
batches as one integer matrix product ``D @ νᵀ mod N`` (GPU-shaped), stopping at the first
valid ``ν`` of the smallest ``N`` unless every solution is asked for.

Modes: ``minimal`` (smallest ``N`` with complete selection, up to ``max_unwanted`` tolerated
survivors), ``coverage`` (a fixed ``N``, as many unwanted pathways cancelled as possible, ties
broken by the weighted residual of Landheer & Juchem 2020 Eq. 5 when weights are given).

The Hughes–Carravetta–Levitt construction (2004 Eq. 23-26) gives a cycle at once for full
symmetric order ranges and a single wanted pathway; its ``N`` is conjectured minimal when the
winding differences share no prime factor, and is used here as the upper limit of the scan,
never to skip smaller ``N``.

Author: Alexej Jerschow.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence as SequenceABC
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field

import numpy as np

from ctps.engine import paths as paths_mod
from ctps.model import PhaseCycle, Sequence


@dataclass(frozen=True)
class CycleSolution:
    N: int
    nu: tuple[int, ...]
    nu_rec: int
    survivors: int  # unwanted pathways that survive (0 for complete selection)
    residual: float  # weighted sum of surviving unwanted pathways (0 without weights)
    margin: int  # smallest |ν·δ mod N| distance to zero among rejected pathways (robustness)

    @property
    def cycle(self) -> PhaseCycle:
        return PhaseCycle.cogwheel(self.N, self.nu, self.nu_rec)

    def cog_string(self) -> str:
        return f"COG{self.N}({','.join(str(v) for v in self.nu)};{self.nu_rec})"


@dataclass
class SearchResult:
    """What ``find_cycle`` found: the best solution, every solution if asked, and the ``N`` values scanned."""

    best: CycleSolution | None
    solutions: list[CycleSolution]
    scanned: list[int]
    n_pathways: int
    n_differences: int
    impossible: list[tuple[int, ...]] = field(default_factory=list)  # unwanted pathways with a wanted pathway's jumps

    def _repr_html_(self) -> str:
        if self.best is None:
            span = f"{self.scanned[0]}..{self.scanned[-1]}" if self.scanned else "none"
            return f"<div>No cycle found over N = {span} ({self.n_pathways} pathways).</div>"
        b = self.best
        rows = "".join(
            f"<tr><td>{s.cog_string()}</td><td>{s.survivors}</td><td>{s.residual:.3g}</td><td>{s.margin}</td></tr>"
            for s in self.solutions[:20]
        )
        return (
            f"<div><b>{b.cog_string()}</b>: {self.n_pathways} pathways, {self.n_differences} distinct differences, "
            f"{b.survivors} unwanted survivors, margin {b.margin}."
            f"<table><tr><th>cycle</th><th>survivors</th><th>residual</th><th>margin</th></tr>{rows}</table></div>"
        )


def pathway_jumps(sequence: Sequence) -> tuple[np.ndarray, np.ndarray]:
    """All pathways the sequence allows: ``(orders (k, n_periods), Δp (k, n_pulses))``."""
    parts = [paths_mod.nucleus_pathways(sequence, nucleus.name) for nucleus in sequence.nuclei]
    orders, jumps = [], []
    for block in paths_mod.combine(parts):
        orders.append(block.orders)
        jumps.append(paths_mod.delta_p(sequence, block.per_nucleus))
    return np.concatenate(orders), np.concatenate(jumps)


def difference_set(
    dp: np.ndarray, wanted_rows: np.ndarray, weights: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``(D, D_wanted, counts, weight)``: unique unwanted jump differences, wanted ones, multiplicities, weight sums."""
    reference = dp[wanted_rows[0]]
    delta = dp.astype(np.int64) - reference[None, :]
    is_wanted = np.zeros(len(dp), dtype=bool)
    is_wanted[wanted_rows] = True
    unwanted, inverse = np.unique(delta[~is_wanted], axis=0, return_inverse=True)
    counts = np.bincount(inverse.ravel(), minlength=len(unwanted))
    weight = (
        np.bincount(inverse.ravel(), weights=weights[~is_wanted], minlength=len(unwanted))
        if weights is not None
        else counts.astype(float)
    )
    d_wanted = np.unique(delta[is_wanted], axis=0)
    return unwanted, d_wanted, counts, weight


def hcl_cycle(p_wanted: SequenceABC[int], p_max: SequenceABC[int]) -> CycleSolution:
    """Hughes–Carravetta–Levitt construction (2004 Eq. 23-26) for one wanted pathway over full symmetric ranges.

    ``p_wanted`` are the orders of the intermediate periods ``1..n-1`` (period 0 at order 0 and
    detection at -1 are implied), ``p_max`` their maximum orders. Returns the predicted cycle
    ``COG_N`` with ``ν_1 = 0`` and ``ν_{l+1} = ν_l + Δν_l``.
    """
    q = [pm + 1 - abs(p) for p, pm in zip(p_wanted, p_max)]
    if any(v < 1 for v in q):
        raise ValueError("every |p_wanted| must be at most its p_max.")
    Q = math.prod(q)
    N = Q + sum(abs(2 * Q * p // ql) for p, ql in zip(p_wanted, q))
    dnu = [(1 if p >= 0 else -1) * (Q // ql) for p, ql in zip(p_wanted, q)]
    nu = [0]
    for d in dnu:
        nu.append(nu[-1] + d)
    jumps = np.diff([0, *p_wanted, -1])
    nu_rec = int((-np.dot(nu, jumps)) % N)
    return CycleSolution(N, tuple(int(v) % N for v in nu), nu_rec, 0, 0.0, 0)


def find_cycle(
    sequence: Sequence,
    wanted: SequenceABC[SequenceABC[int]],
    *,
    N: int | None = None,
    N_min: int = 2,
    N_max: int | None = None,
    max_unwanted: int = 0,
    weights: np.ndarray | None = None,
    tolerance: float = 0.0,
    all_solutions: bool = False,
    batch: int = 1 << 18,
    workers: int = 1,
) -> SearchResult:
    """Find a cogwheel cycle for ``sequence`` that selects the ``wanted`` pathways (total-order tuples).

    With ``N`` given, the fixed-length coverage mode: the ``ν`` with the fewest surviving unwanted
    pathways (then the smallest weighted residual). Otherwise the smallest ``N`` in
    ``N_min..N_max`` with at most ``max_unwanted`` survivors (or residual at most ``tolerance``
    when ``weights`` are given per pathway); ``N_max`` defaults to the HCL prediction when the
    orders allow it, else to the product of the order ranges (2003 Eq. 11). ``all_solutions``
    collects every valid ``ν`` of the winning ``N`` instead of stopping at the first.
    """
    orders, dp = pathway_jumps(sequence)
    targets = {tuple(int(v) for v in w) for w in wanted}
    wanted_rows = np.array([i for i, row in enumerate(orders) if tuple(int(v) for v in row) in targets])
    if wanted_rows.size == 0:
        raise ValueError("none of the wanted pathways is allowed by the sequence's orders.")
    D, D_wanted, counts, weight = difference_set(dp, wanted_rows, weights)
    impossible = [tuple(int(v) for v in orders[i]) for i in np.flatnonzero(~np.any(dp - dp[wanted_rows[0]], axis=1))]
    impossible = [p for p in impossible if p not in targets]
    zero_rows = ~np.any(D, axis=1)
    D, counts, weight = D[~zero_rows], counts[~zero_rows], weight[~zero_rows]

    n_pulses = dp.shape[1]
    free = np.flatnonzero(np.any(D, axis=0))  # pulses where some unwanted pathway differs from the wanted one
    total_jump_fixed = np.all(D.sum(axis=1) == 0) and np.all(D_wanted.sum(axis=1) == 0)
    common = int(np.gcd.reduce(np.abs(D).ravel())) if D.size else 1

    if N_max is None:
        N_max = _default_n_max(sequence, wanted, dp)
    candidates_N = [N] if N is not None else [n for n in range(max(2, N_min), N_max + 1)]
    scanned: list[int] = []
    solutions: list[CycleSolution] = []
    best: CycleSolution | None = None
    jobs = [
        (
            n,
            D,
            D_wanted,
            counts,
            weight,
            free,
            total_jump_fixed,
            common,
            n_pulses,
            max_unwanted,
            tolerance,
            weights is not None,
            all_solutions or N is not None,
            batch,
        )
        for n in candidates_N
    ]
    runner = _search_n
    if workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for n, found in zip(candidates_N, pool.map(runner, jobs)):
                scanned.append(n)
                if found:
                    solutions = found
                    break
    else:
        for job in jobs:
            scanned.append(job[0])
            found = runner(job)
            if found:
                solutions = found
                break
    if solutions:
        solutions.sort(key=lambda s: (s.survivors, s.residual, -s.margin, s.nu))
        reference = dp[wanted_rows[0]]
        solutions = [
            CycleSolution(s.N, s.nu, int((-np.dot(s.nu, reference)) % s.N), s.survivors, s.residual, s.margin)
            for s in solutions
        ]
        best = solutions[0]
    return SearchResult(best, solutions, scanned, len(dp), len(D), impossible)


def _default_n_max(sequence: Sequence, wanted, dp: np.ndarray) -> int:
    """HCL prediction for a single wanted pathway over full ranges; otherwise the 2003 product bound."""
    try:
        if len(wanted) == 1 and len(sequence.nuclei) == 1:
            allowed = sequence.allowed[sequence.nuclei[0].name]
            orders = list(wanted[0])
            p_max = [int(np.abs(allowed.orders(period)).max()) for period in range(1, sequence.n_periods - 1)]
            full = all(
                set(allowed.orders(period).tolist()) == set(range(-pm, pm + 1))
                for period, pm in zip(range(1, sequence.n_periods - 1), p_max)
            )
            if full and orders[0] == 0 and orders[-1] == -1:
                return hcl_cycle(orders[1:-1], p_max).N
    except (ValueError, KeyError):
        pass
    return int(np.prod([len(np.unique(dp[:, j])) for j in range(dp.shape[1])]))


def _candidates(N: int, free: np.ndarray, total_jump_fixed: bool, batch: int):
    """Yield ``(batch, k)`` arrays of winding-number candidates over the free pulses, reduced by the symmetries."""
    k = len(free)
    if k == 0:
        yield np.zeros((1, 0), dtype=np.int64)
        return
    divisors = [d for d in range(1, N) if N % d == 0]
    if total_jump_fixed and k > 1:
        # first free winding number fixed at 0 (uniform shift); the next nonzero one is a divisor of N (unit multiples)
        rest = k - 1
        heads = [0]
    else:
        rest = k
        heads = None

    def rows_for(prefix: list[int], remaining: int):
        if remaining == 0:
            yield np.array(prefix, dtype=np.int64)[None, :]
            return
        grid = (
            np.array(list(itertools.product(range(N), repeat=remaining)), dtype=np.int64)
            if N**remaining <= batch
            else None
        )
        if grid is not None:
            block = np.hstack([np.tile(np.array(prefix, dtype=np.int64), (len(grid), 1)), grid])
            for start in range(0, len(block), batch):
                yield block[start : start + batch]
        else:
            for first in range(N):
                yield from rows_for([*prefix, first], remaining - 1)

    if heads is None:
        # first nonzero winding number restricted to divisors of N
        for lead in range(rest):
            for d in divisors:
                prefix = [0] * lead + [d]
                yield from rows_for(prefix, rest - lead - 1)
        return
    for lead in range(rest):
        for d in divisors:
            prefix = [0] + [0] * lead + [d]
            yield from rows_for(prefix, rest - lead - 1)


def _search_n(job: tuple) -> list[CycleSolution]:
    (
        N,
        D,
        D_wanted,
        counts,
        weight,
        free,
        total_jump_fixed,
        common,
        n_pulses,
        max_unwanted,
        tolerance,
        weighted,
        collect_all,
        batch,
    ) = job
    if common > 1 and N % common == 0 and N != common:
        pass  # a common factor of all δ divides any usable N; the scan still tests it so results stay complete
    found: list[CycleSolution] = []
    Df = D[:, free]
    Dw = D_wanted[:, free]
    for block in _candidates(N, free, total_jump_fixed, batch):
        # GPU: integer GEMM, modulo, column reductions
        product = (Df @ block.T) % N  # (n_diff, n_cand)
        wanted_ok = np.all((Dw @ block.T) % N == 0, axis=0) if len(Dw) else np.ones(len(block), dtype=bool)
        zero = product == 0
        survivors = counts @ zero  # unwanted pathways surviving per candidate
        residual = weight @ zero if weighted else survivors.astype(float)
        if weighted:
            ok = wanted_ok & (residual <= tolerance + 1e-12)
        else:
            ok = wanted_ok & (survivors <= max_unwanted)
        if not ok.any():
            continue
        distance = np.minimum(product, N - product)
        distance = np.where(zero, N, distance)
        for c in np.flatnonzero(ok):
            nu = np.zeros(n_pulses, dtype=np.int64)
            nu[free] = block[c]
            found.append(
                CycleSolution(
                    N,
                    tuple(int(v) for v in nu),
                    0,
                    int(survivors[c]),
                    float(residual[c]),
                    int(distance[:, c].min()) if len(D) else N,
                )
            )
            if not collect_all:
                return found
    if collect_all and found:
        return found
    return found


def coverage(sequence: Sequence, wanted: SequenceABC[SequenceABC[int]], N: int, **options) -> SearchResult:
    """Fixed-``N`` mode: the cogwheel cycle cancelling the most unwanted pathways (Landheer & Juchem 2020 Table 1)."""
    orders, dp = pathway_jumps(sequence)
    targets = {tuple(int(v) for v in w) for w in wanted}
    wanted_rows = np.array([i for i, row in enumerate(orders) if tuple(int(v) for v in row) in targets])
    D, D_wanted, counts, weight = difference_set(dp, wanted_rows, options.get("weights"))
    zero_rows = ~np.any(D, axis=1)
    D, counts, weight = D[~zero_rows], counts[~zero_rows], weight[~zero_rows]
    free = np.flatnonzero(np.any(D, axis=0))
    total_jump_fixed = bool(np.all(D.sum(axis=1) == 0) and np.all(D_wanted.sum(axis=1) == 0))
    best = None
    Df, Dw = D[:, free], D_wanted[:, free]
    for block in _candidates(N, free, total_jump_fixed, options.get("batch", 1 << 18)):
        product = (Df @ block.T) % N
        wanted_ok = np.all((Dw @ block.T) % N == 0, axis=0) if len(Dw) else np.ones(len(block), dtype=bool)
        zero = product == 0
        survivors = counts @ zero
        residual = weight @ zero
        score = np.where(wanted_ok, survivors + 1e-9 * residual / max(residual.max(), 1e-300), np.inf)
        c = int(np.argmin(score))
        if np.isfinite(score[c]) and (best is None or score[c] < best[0]):
            nu = np.zeros(dp.shape[1], dtype=np.int64)
            nu[free] = block[c]
            distance = np.minimum(product[:, c], N - product[:, c])
            best = (
                score[c],
                CycleSolution(
                    N,
                    tuple(int(v) for v in nu),
                    0,
                    int(survivors[c]),
                    float(residual[c]),
                    int(distance[~zero[:, c]].min()) if (~zero[:, c]).any() else 0,
                ),
            )
    if best is None:
        return SearchResult(None, [], [N], len(dp), len(D))
    s = best[1]
    reference = dp[wanted_rows[0]]
    solution = CycleSolution(N, s.nu, int((-np.dot(s.nu, reference)) % N), s.survivors, s.residual, s.margin)
    return SearchResult(solution, [solution], [N], len(dp), len(D))


def crusher_scheme(
    sequence: Sequence,
    wanted: SequenceABC[SequenceABC[int]],
    *,
    bounds: np.ndarray | float = 1.0,
    lam: float = 1.0,
    starts: int = 200,
    seed: int = 0,
    detect_only: bool = True,
) -> dict:
    """Crusher areas that dephase every unwanted pathway as evenly and strongly as possible (DOTCOPS).

    The objective of Landheer & Juchem, MRM 81, 2209 (2019) Eq. 4-7: with ``X`` the
    ``(n_periods, 3)`` crusher areas (in units of ``bounds``) and ``D(u) = |Σ_i p_i(u) X_i|`` the
    k-space radius of pathway ``u`` (geometry-free; ``γ`` left out), maximise
    ``min_u D(u) + lam · mean_u D(u)`` over the unwanted pathways subject to ``D(w) = 0`` for every
    wanted pathway and ``|X| <= bounds`` elementwise. ``bounds`` is a scalar or a ``(n_periods, 3)``
    array; a zero bound disables a crusher (period 0 and the detection period by default).
    ``detect_only`` keeps only pathways ending at the detection order of the wanted ones
    (DOTCOPS counts pathways ending at -1). Multistart SLSQP from ``starts`` random points;
    returns the best ``areas``, ``min_D``, ``mean_D`` and the objective.
    """
    from scipy.optimize import minimize

    orders, _ = pathway_jumps(sequence)
    targets = {tuple(int(v) for v in w) for w in wanted}
    is_wanted = np.array([tuple(int(v) for v in row) in targets for row in orders])
    if not is_wanted.any():
        raise ValueError("none of the wanted pathways is allowed by the sequence's orders.")
    if detect_only:
        ends = {int(orders[i, -1]) for i in np.flatnonzero(is_wanted)}
        keep = np.array([int(row[-1]) in ends for row in orders])
        orders, is_wanted = orders[keep], is_wanted[keep]
    P_unwanted = orders[~is_wanted].astype(float)  # (n_u, n_periods)
    P_wanted = orders[is_wanted].astype(float)
    n_periods = orders.shape[1]
    limit = np.broadcast_to(np.asarray(bounds, dtype=float), (n_periods, 3)).copy()
    if np.isscalar(bounds):
        limit[0, :] = 0.0
        limit[-1, :] = 0.0
    active = np.flatnonzero(limit.ravel() > 0)
    if active.size == 0:
        raise ValueError("every crusher is disabled by the bounds.")

    def unpack(x: np.ndarray) -> np.ndarray:
        full = np.zeros(n_periods * 3)
        full[active] = x
        return full.reshape(n_periods, 3)

    def distances(X: np.ndarray) -> np.ndarray:
        return np.linalg.norm(P_unwanted @ X, axis=1)

    def objective(x: np.ndarray) -> float:
        d = distances(unpack(x))
        return -(d.min() + lam * d.mean())

    def refocus(x: np.ndarray) -> np.ndarray:
        return (P_wanted @ unpack(x)).ravel()

    rng = np.random.default_rng(seed)
    lo, hi = -limit.ravel()[active], limit.ravel()[active]
    best = None
    for _ in range(starts):
        x0 = rng.uniform(lo, hi)
        res = minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=list(zip(lo, hi)),
            constraints=[{"type": "eq", "fun": refocus}],
            options={"maxiter": 300, "ftol": 1e-10},
        )
        if not res.success:
            continue
        if best is None or res.fun < best.fun:
            best = res
    if best is None:
        raise RuntimeError("no feasible crusher scheme found; loosen the bounds or check the wanted pathways.")
    X = unpack(best.x)
    d = distances(X)
    return {"areas": X, "min_D": float(d.min()), "mean_D": float(d.mean()), "objective": float(-best.fun)}


def pathway_weights(sequence: Sequence, *, envelope: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """``(orders, g)``: every allowed pathway and its gradient x diffusion x pulse weight, in enumeration order.

    ``g`` is the weight a pathway keeps before any phase cycling, the ``g(u)`` of the weighted
    search: pass it as ``weights`` to :func:`find_cycle` so that pathways the gradients or
    the pulses already suppress need not be cancelled by the cycle (``tolerance`` then bounds
    the summed weight of the survivors).
    """
    from ctps.engine import gradient as grad_mod
    from ctps.engine import pulse as pulse_mod

    parts = [paths_mod.nucleus_pathways(sequence, nucleus.name) for nucleus in sequence.nuclei]
    orders, weights = [], []
    areas, durations = sequence.gradient_areas(), sequence.durations()
    for block in paths_mod.combine(parts):
        gamma_star = grad_mod.effective_gamma(sequence, block.per_nucleus)
        K = grad_mod.k_vectors(gamma_star, areas)
        g = grad_mod.gradient_attenuation(K[:, -1, :], sequence.sample, envelope)
        g = g * grad_mod.diffusion_attenuation(K, durations, sequence.diffusion)
        g = g * pulse_mod.pulse_factor(sequence, block.per_nucleus) * block.weight
        orders.append(block.orders)
        weights.append(g)
    return np.concatenate(orders), np.concatenate(weights)
