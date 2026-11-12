"""Pathway enumeration: per nucleus, then combined across nuclei in chunks.

The order of a nucleus changes only at pulses acting on it (CCCP.m build loop: columns of
the other nuclei carry the previous order forward), so each nucleus is enumerated over its
own pulses as a mixed-radix product of the allowed orders, with the period masks of the
periods a segment spans intersected and their weights multiplied. A ``perfect_180`` pulse
forces ``p -> -p``. Nuclei are combined lazily by index grids over chunks (the outer sum of
CCCP.m without materialising it).

Author: Alexej Jerschow.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from ctps.model import AllowedOrders, Sequence


@dataclass(frozen=True)
class NucleusPathways:
    """All pathways of one nucleus: ``orders (k, n_periods)`` int8 and the product of allowed-order weights."""

    name: str
    orders: np.ndarray
    weights: np.ndarray


def nucleus_pathways(sequence: Sequence, name: str) -> NucleusPathways:
    allowed: AllowedOrders = sequence.allowed[name]
    n_periods = sequence.n_periods
    own = [j for j, pulse in enumerate(sequence.pulses) if pulse.nucleus == name]
    # Segments: periods [0, own[0]], [own[0] + 1, own[1]], ..., [own[-1] + 1, n_periods - 1]
    bounds = [0, *[j + 1 for j in own], n_periods]
    segments = [range(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]

    def segment_orders(periods: range) -> tuple[np.ndarray, np.ndarray]:
        weights = np.ones(2 * allowed.p_max + 1)
        for period in periods:
            weights = weights * allowed.weights[:, period]
        keep = np.flatnonzero(weights)
        return keep - allowed.p_max, weights[keep]

    orders_so_far = np.zeros((1, 0), dtype=np.int8)
    weight_so_far = np.ones(1)
    for s, periods in enumerate(segments):
        values, values_weight = segment_orders(periods)
        if values.size == 0:
            return NucleusPathways(name, np.zeros((0, n_periods), dtype=np.int8), np.zeros(0))
        if s == 0:
            new = values[:, None].astype(np.int8)
            weight = values_weight
        else:
            pulse = sequence.pulses[own[s - 1]]
            previous = orders_so_far[:, -1]
            if pulse.kind == "perfect_180":
                # p -> -p only: keep rows whose inverted order is allowed in the next segment
                index = {int(v): i for i, v in enumerate(values)}
                target = np.array([index.get(int(-p), -1) for p in previous])
                rows = np.flatnonzero(target >= 0)
                new = np.hstack([orders_so_far[rows], (-previous[rows])[:, None].astype(np.int8)])
                weight = weight_so_far[rows] * values_weight[target[rows]]
            else:
                k = len(orders_so_far)
                new = np.hstack(
                    [np.repeat(orders_so_far, len(values), axis=0), np.tile(values, k)[:, None].astype(np.int8)]
                )
                weight = np.repeat(weight_so_far, len(values)) * np.tile(values_weight, k)
        # expand the segment's single order over all its periods
        width = len(periods)
        orders_so_far = np.hstack([new[:, :-1], np.repeat(new[:, -1:], width, axis=1)]) if width != 1 else new
        weight_so_far = weight
    return NucleusPathways(name, np.ascontiguousarray(orders_so_far), weight_so_far)


@dataclass(frozen=True)
class Chunk:
    """A block of combined pathways: total orders, per-nucleus orders, allowed-order weight, and the row index."""

    orders: np.ndarray
    per_nucleus: dict[str, np.ndarray]
    weight: np.ndarray
    start: int


def combined_count(parts: list[NucleusPathways]) -> int:
    return int(np.prod([len(p.orders) for p in parts]))


def combine(parts: list[NucleusPathways], chunk: int = 1 << 20) -> Iterator[Chunk]:
    """Yield chunks of the Cartesian product of the per-nucleus pathway sets (row-major, last nucleus fastest)."""
    counts = [len(p.orders) for p in parts]
    total = combined_count(parts)
    for start in range(0, total, chunk):
        yield chunk_at(parts, start, min(chunk, total - start), counts)


def chunk_at(parts: list[NucleusPathways], start: int, size: int, counts: list[int] | None = None) -> Chunk:
    counts = counts or [len(p.orders) for p in parts]
    flat = np.arange(start, start + size)
    indices = []
    for count in reversed(counts):
        flat, index = np.divmod(flat, count)
        indices.append(index)
    indices.reverse()
    per_nucleus = {part.name: part.orders[index] for part, index in zip(parts, indices)}
    orders = np.zeros_like(next(iter(per_nucleus.values())), dtype=np.int16)
    weight = np.ones(size)
    for part, index in zip(parts, indices):
        orders += part.orders[index]
        weight *= part.weights[index]
    return Chunk(orders.astype(np.int8), per_nucleus, weight, start)


def delta_p(sequence: Sequence, per_nucleus: dict[str, np.ndarray]) -> np.ndarray:
    """``(k, n_pulses)`` order jumps ``Δp_j = p_j - p_{j-1}`` of the nucleus each pulse acts on (1998 Eq. 12)."""
    columns = []
    for j, pulse in enumerate(sequence.pulses):
        orders = per_nucleus[pulse.nucleus]
        columns.append(orders[:, j + 1].astype(np.int16) - orders[:, j].astype(np.int16))
    return np.stack(columns, axis=1) if columns else np.zeros((len(next(iter(per_nucleus.values()))), 0), np.int16)
