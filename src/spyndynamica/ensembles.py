"""Ensemble averages and the orientational sampling sets of SpinDynamica.

``orientations(name)`` returns the Euler angles and weights of a tabulated set (``'ZCW50'``, ``'REPULSION100'``,
``'Leboct10'``, ...); ``step_beta`` and ``step_alpha_beta`` build the regular sets. ``ensemble_average``
averages any function over samples, in threads when asked.

Author: Alexej Jerschow.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np

from spyndynamica.signals import Signal


@cache
def _sets() -> dict[str, np.ndarray]:
    with np.load(Path(__file__).parent / "data" / "orientations.npz") as data:
        return {name: data[name] for name in data.files}


def orientation_schemes() -> list[str]:
    """The names of the tabulated orientational sampling sets."""
    return sorted(_sets())


def orientations(name: str) -> tuple[np.ndarray, np.ndarray]:
    """The Euler angles ``(α, β, γ)`` (one row per orientation, in radians) and the weights of a tabulated set."""
    sets = _sets()
    if name not in sets:
        raise ValueError(f"unknown orientational sampling set {name!r}; see orientation_schemes().")
    return sets[name][:, :3].copy(), sets[name][:, 3].copy()


def step_beta(count: int, beta_range: Sequence[float] = (0.0, np.pi)) -> tuple[np.ndarray, np.ndarray]:
    """``count`` orientations with ``β`` at the midpoints of equal steps over ``beta_range``, weighted by ``sin β``."""
    low, high = beta_range
    step = (high - low) / count
    beta = low + step * (np.arange(count) + 0.5)
    weights = np.sin(beta)
    angles = np.stack([np.zeros(count), beta, np.zeros(count)], axis=1)
    return angles, weights / weights.sum()


def step_alpha_beta(alpha_count: int, beta_count: int) -> tuple[np.ndarray, np.ndarray]:
    """A grid of ``α`` over a full turn and ``β`` over a half turn (midpoints of equal steps), weighted by
    ``sin β``; ``α`` is the slow index."""
    alpha = 2 * np.pi / alpha_count * (np.arange(alpha_count) + 0.5)
    beta = np.pi / beta_count * (np.arange(beta_count) + 0.5)
    a, b = (grid.reshape(-1) for grid in np.meshgrid(alpha, beta, indexing="ij"))
    weights = np.sin(b)
    return np.stack([a, b, np.zeros_like(a)], axis=1), weights / weights.sum()


def ensemble_average(
    function: Callable[..., Any],
    samples: Iterable[Any],
    weights: Sequence[float] | None = None,
    workers: int | None = None,
):
    """The weighted average of ``function(sample)`` over the samples (equal weights by default).

    A sample that is a tuple or a row of an array is passed as separate arguments. The function may return a
    number, an array or a :class:`~spyndynamica.Signal`; signals are averaged sample by sample. ``workers``
    runs the evaluations in that many threads.
    """
    items = list(samples)
    if not items:
        raise ValueError("no samples to average over.")
    w = np.full(len(items), 1.0 / len(items)) if weights is None else np.asarray(weights, dtype=float)
    if len(w) != len(items):
        raise ValueError("one weight per sample is needed.")

    def call(sample: Any):
        if isinstance(sample, (tuple, list, np.ndarray)):
            return function(*sample)
        return function(sample)

    if workers and workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(call, items))
    else:
        results = [call(item) for item in items]
    first = results[0]
    if isinstance(first, Signal):
        data = sum(weight * result.values for weight, result in zip(w, results))
        return replace(first, data=data, peaks=None)
    return sum(weight * np.asarray(result) for weight, result in zip(w, results))
