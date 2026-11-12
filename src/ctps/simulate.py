"""``simulate``: enumerate, weight and rank the coherence transfer pathways of a sequence under a phase cycle.

Author: Alexej Jerschow.
"""

from __future__ import annotations

from collections.abc import Sequence as SequenceABC
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from ctps.engine import gradient as grad_mod
from ctps.engine import paths as paths_mod
from ctps.engine import pulse as pulse_mod
from ctps.engine.phase import phase_cycle_factor
from ctps.engine.rank import ranking, survivors
from ctps.model import PathwayResult, PhaseCycle, Sequence


def simulate(
    sequence: Sequence,
    cycle: PhaseCycle,
    *,
    wanted: SequenceABC[SequenceABC[int]] | None = None,
    epsilon: float = 1e-6,
    cutoff: float = 1e-6,
    top: int | None = None,
    envelope: bool = True,
    exact_pulses: bool = False,
    integer: bool = True,
    chunk: int = 1 << 18,
    workers: int = 1,
) -> PathwayResult:
    """Rank every pathway the allowed orders permit by ``|pc| · grad · diff · pulse``.

    ``wanted`` lists the desired pathways as total-order tuples over the periods; ``epsilon`` is
    the threshold on ``|pc|`` after the phase cycle and ``cutoff`` the threshold on the product after
    gradients and after diffusion (CCCP.m's discard order); ``top`` keeps the strongest pathways;
    ``envelope`` selects the program's attenuation envelopes (1998 Eq. 31/34) over the exact forms;
    ``exact_pulses`` uses full rotation matrices instead of the leading-order transfer weights;
    ``integer`` evaluates the phase-cycle sum through the root-of-unity table; ``chunk`` rows are
    weighted at a time, with ``workers`` processes when more than one.
    """
    if cycle.n_pulses != sequence.n_pulses:
        raise ValueError(f"the cycle has {cycle.n_pulses} phase lists but the sequence {sequence.n_pulses} pulses.")
    parts = [paths_mod.nucleus_pathways(sequence, nucleus.name) for nucleus in sequence.nuclei]
    n_enumerated = paths_mod.combined_count(parts)
    counts = [len(p.orders) for p in parts]
    starts = list(range(0, n_enumerated, chunk))
    options = (sequence, cycle, epsilon, cutoff, envelope, exact_pulses, integer)
    if workers > 1 and len(starts) > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            pieces = list(
                pool.map(_weigh_chunk, [(parts, s, min(chunk, n_enumerated - s), counts, options) for s in starts])
            )
    else:
        pieces = [_weigh_chunk((parts, s, min(chunk, n_enumerated - s), counts, options)) for s in starts]

    def gather(key: int) -> np.ndarray:
        return np.concatenate([piece[key] for piece in pieces]) if pieces else np.zeros(0)

    orders, pc, grad, diff, pulse = (gather(k) for k in range(5))
    per_nucleus = {
        name: np.concatenate([piece[5][name] for piece in pieces])
        if pieces
        else np.zeros((0, sequence.n_periods), np.int8)
        for name in (n.name for n in sequence.nuclei)
    }
    amplitude = np.abs(pc) * grad * diff * pulse
    is_wanted = np.zeros(len(amplitude), dtype=bool)
    if wanted:
        targets = {tuple(int(v) for v in w) for w in wanted}
        is_wanted = np.array([tuple(int(v) for v in row) in targets for row in orders], dtype=bool)
    order = ranking(amplitude, is_wanted)
    if top is not None:
        order = order[:top]
    wanted_mask = is_wanted[order]
    return PathwayResult(
        orders[order],
        {name: o[order] for name, o in per_nucleus.items()},
        pc[order],
        grad[order],
        diff[order],
        pulse[order],
        wanted_mask,
        n_enumerated,
        sequence,
        cycle,
    )


def _weigh_chunk(job: tuple) -> tuple:
    """Factors of one chunk of combined pathways, thresholds applied (a module-level function so it pickles)."""
    parts, start, size, counts, (sequence, cycle, epsilon, cutoff, envelope, exact_pulses, integer) = job
    block = paths_mod.chunk_at(parts, start, size, counts)
    dp = paths_mod.delta_p(sequence, block.per_nucleus)
    pc = phase_cycle_factor(dp, cycle, integer=integer)
    gamma_star = grad_mod.effective_gamma(sequence, block.per_nucleus)
    K = grad_mod.k_vectors(gamma_star, sequence.gradient_areas())
    grad = grad_mod.gradient_attenuation(K[:, -1, :], sequence.sample, envelope)
    diff = grad_mod.diffusion_attenuation(K, sequence.durations(), sequence.diffusion)
    if sequence.flow:
        starts = np.concatenate([[0.0], np.cumsum(sequence.durations())[:-1]])
        diff = diff * grad_mod.flow_attenuation(gamma_star, sequence.gradient_areas(), starts, sequence.flow)
    pulse = pulse_mod.pulse_factor(sequence, block.per_nucleus, exact=exact_pulses) * block.weight
    keep = survivors(pc, grad, diff, pulse, epsilon, cutoff)
    return (
        block.orders[keep],
        pc[keep],
        grad[keep],
        diff[keep],
        pulse[keep],
        {name: o[keep] for name, o in block.per_nucleus.items()},
    )
