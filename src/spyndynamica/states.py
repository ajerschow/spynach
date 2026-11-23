"""Density operators and approximations: thermal equilibrium, polarized states, secularisation, thermalised
relaxation; and tables and maxima of transformation amplitudes.

Hamiltonians are in rad/s and temperatures in kelvin; ``ħ`` and ``kB`` are SpinDynamica's values
(:func:`spyndynamica.constants`).

Author: Alexej Jerschow.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import scipy.optimize
import sympy as sp

from spyndynamica.dynamics import amplitudes
from spyndynamica.interactions import constants
from spyndynamica.liouville import Superoperator
from spyndynamica.operators import Operator
from spyndynamica.system import SpinSystem


def _from_numeric(system: SpinSystem, matrix: np.ndarray) -> Operator:
    return system.from_matrix(
        sp.Matrix(matrix.shape[0], matrix.shape[1], lambda i, j: sp.sympify(complex(matrix[i, j])))
    )


def thermal_equilibrium(
    hamiltonian: Operator, temperature: Any, *, high_temperature: bool = True, values: Mapping[Any, Any] | None = None
) -> Operator:
    """The density operator at thermal equilibrium under the laboratory-frame Hamiltonian.

    In the high-temperature approximation (the default, as in SpinDynamica) it is ``(1 − ħH/(kB T))/d``, exact
    in the symbols of ``H``. With ``high_temperature=False`` it is ``exp(−ħH/(kB T))/Z``, evaluated numerically.
    """
    system = hamiltonian.system
    k = constants()
    scale = k["hbar"] / k["kB"]
    if high_temperature:
        return (system.unity() - sp.Float(scale) / sp.sympify(temperature) * hamiltonian) / system.dimension
    energies, vectors = np.linalg.eigh(hamiltonian.numeric(values))
    exponent = -scale * energies / float(sp.N(sp.sympify(temperature).subs(values or {})))
    weights = np.exp(exponent - exponent.max())
    return _from_numeric(system, (vectors * (weights / weights.sum())) @ vectors.conj().T)


def polarized(system: SpinSystem, polarization: Any, axis: Any = "z", spins: Any = None) -> Operator:
    """A density operator with spins-1/2 polarized along an axis: the product of ``1/2 + p I_axis`` over the
    spins (unpolarized spins contribute ``1/2``).

    ``polarization`` and ``axis`` apply to ``spins`` (default: all); ``polarization`` may also be a mapping
    ``{spin: p}`` or ``{spin: (p, axis)}``. The axis is a component as in ``SpinSystem.I``.
    """
    if isinstance(polarization, Mapping):
        levels = {spin: (v if isinstance(v, (tuple, list)) else (v, axis)) for spin, v in polarization.items()}
    else:
        levels = {system.spins[i][0]: (polarization, axis) for i in system._indices(spins)}
    out = system.unity()
    for spin, value in system.spins:
        if value != sp.Rational(1, 2):
            raise NotImplementedError("polarized density operators are available for spins-1/2.")
        factor = system.unity() / 2
        if spin in levels:
            level, direction = levels[spin]
            factor = factor + sp.sympify(level) * system.I(spin, direction)
        out = out @ factor
    return out


def singlet_polarized(system: SpinSystem, polarization: Any, pairs: Sequence[Any] | None = None) -> Operator:
    """A density operator with singlet polarization ``p`` of spin-1/2 pairs: ``1/4 − p I_j·I_k`` for every pair,
    ``1/2`` for the other spins. ``pairs`` is one pair or a list of pairs (default: the two spins of a pair)."""
    labels = [label for label, _ in system.spins]
    if pairs is None:
        if len(labels) != 2:
            raise ValueError("give the spin pairs for a system of more than two spins.")
        pairs = [tuple(labels)]
    elif not isinstance(pairs[0], (tuple, list)):
        pairs = [tuple(pairs)]
    paired = {spin for pair in pairs for spin in pair}
    out = system.unity()
    for a, b in pairs:
        out = out @ (system.unity() / 4 - sp.sympify(polarization) * system.dot(a, b))
    for spin in labels:
        if spin not in paired:
            out = out / 2
    return out


def polarization_level(system: SpinSystem, axis: Any = "z", spins: Any = None) -> Operator:
    """The operator whose amplitude in a density operator is the mean polarization level of ``spins`` (default:
    all) along ``axis`` (SpinDynamica's ``PolarizationLevelOperator``).

    ``ρ.amplitude(polarization_level(s, "z"))`` is the mean of ``<I_z>/I`` over the spins, for a density
    operator of unit trace. For one spin the operator is ``3 I_axis/((I + 1) d)`` with ``d`` the dimension of
    the Hilbert space; for several spins these are added. The axis is a component as in ``SpinSystem.I``.
    """
    out = system.zero()
    for index in system._indices(spins):
        spin, value = system.spins[index]
        out = out + sp.Integer(3) / ((value + 1) * system.dimension) * system.I(spin, axis)
    return out


def singlet_polarization_level(system: SpinSystem, pairs: Sequence[Any] | None = None) -> Operator:
    """The operator whose amplitude in a density operator is the mean singlet polarization level of spin-1/2
    pairs, ``−(4/3) <I_j·I_k>`` (SpinDynamica's ``SingletPolarizationLevelOperator``).

    For one pair the operator is ``−(4/d) I_j·I_k`` with ``d`` the dimension of the Hilbert space; for several
    pairs, which must not share a spin, these are added. ``pairs`` is one pair or a list of pairs (default:
    the two spins of a pair).
    """
    labels = [label for label, _ in system.spins]
    if pairs is None:
        if len(labels) != 2:
            raise ValueError("give the spin pairs for a system of more than two spins.")
        pairs = [tuple(labels)]
    elif not isinstance(pairs[0], (tuple, list)):
        pairs = [tuple(pairs)]
    used = [spin for pair in pairs for spin in pair]
    if any(len(pair) != 2 for pair in pairs) or len(set(used)) != len(used):
        raise ValueError("the pairs are pairs of distinct spins and do not share spins.")
    out = system.zero()
    for a, b in pairs:
        if any(system.spins[system.index(spin)][1] != sp.Rational(1, 2) for spin in (a, b)):
            raise ValueError("singlet polarization is defined for pairs of spins-1/2.")
        out = out - sp.Integer(4) / system.dimension * system.dot(a, b)
    return out


def secularize(operator: Operator, reference: Any, threshold: float | None = None, values=None) -> Operator:
    """The secular part of an operator.

    With a Hamiltonian as ``reference``, the elements between eigenstates of it whose eigenvalues differ (by
    more than ``threshold``, when given) are removed; the result is numeric. With a list of spin sets, one set
    per isotope (``[[1, 2]]`` for a homonuclear pair, ``[[1], [2]]`` for a heteronuclear one), the terms that
    commute with the total ``Iz`` of every set are kept, exactly.
    """
    system = operator.system
    if isinstance(reference, Operator):
        energies, vectors = np.linalg.eigh(reference.numeric(values))
        matrix = vectors.conj().T @ operator.numeric(values) @ vectors
        gap = np.abs(energies[:, None] - energies[None, :])
        limit = threshold if threshold is not None else 1e-9 * max(1.0, float(np.abs(energies).max()))
        matrix[gap > limit] = 0
        return _from_numeric(system, vectors @ matrix @ vectors.conj().T)
    out = operator
    for group in reference:
        indices = [system.index(spin) for spin in group]
        out = out._order_components(indices).get(0, system.zero())
    return out


def thermalize(
    relaxation: Superoperator, hamiltonian: Operator, temperature: Any, *, high_temperature: bool = True, values=None
) -> Superoperator:
    """A relaxation superoperator corrected so that it drives the system to thermal equilibrium:
    ``ρ → R ρ − R(ρ_eq) Tr ρ`` (SpinDynamica's ``ThermalizeSuperoperator``)."""
    system = hamiltonian.system
    equilibrium = thermal_equilibrium(hamiltonian, temperature, high_temperature=high_temperature, values=values)
    return relaxation + Superoperator(system, dyads=[(-1 * relaxation(equilibrium), system.unity())])


def amplitude_table(
    rho: Operator,
    targets: Any,
    events: Any,
    parameter: Any,
    grid: Sequence[Any],
    *,
    background=None,
    values=None,
    **options,
) -> tuple[np.ndarray, np.ndarray]:
    """Transformation amplitudes for every value of ``parameter`` in ``grid`` (``TransformationAmplitudeTable``):
    ``(grid, amplitudes)``, with one row of amplitudes per target when ``targets`` is a list."""
    points = [float(sp.N(g)) for g in grid]
    rows = []
    for g in grid:
        merged = {**(values or {}), parameter: g}
        rows.append(amplitudes(rho, targets, events, background=background, values=merged, **options))
    table = np.array(rows)
    return np.array(points), (table if isinstance(targets, Operator) else table.T)


def maximize_amplitude(
    rho: Operator,
    target: Operator,
    events: Any,
    start: Mapping[Any, float],
    *,
    bounds: Mapping[Any, tuple[float | None, float | None]] | None = None,
    positive: bool | Sequence[Any] = False,
    background=None,
    values=None,
    **options,
) -> tuple[float, dict[Any, float]]:
    """The parameters that maximise the real part of a transformation amplitude, from the starting values
    ``start = {symbol: value}`` (``MaximizeTransformationAmplitude``): ``(maximum, {symbol: value})``.

    ``positive=True`` keeps all parameters positive, a list keeps those; ``bounds = {symbol: (low, high)}``
    gives other limits. The search is local.
    """
    symbols = list(start)
    limits = []
    for symbol in symbols:
        low, high = (bounds or {}).get(symbol, (None, None))
        if low is None and (positive is True or (positive and symbol in positive)):
            low = 0.0
        limits.append((low, high))
    scale = np.array([abs(float(start[s])) or 1.0 for s in symbols])

    def negative(x: np.ndarray) -> float:
        merged = {**(values or {}), **{s: float(v) for s, v in zip(symbols, x * scale)}}
        return -amplitudes(rho, target, events, background=background, values=merged, **options).real

    scaled = [(None if lo is None else lo / sc, None if hi is None else hi / sc) for (lo, hi), sc in zip(limits, scale)]
    x0 = np.array([float(start[s]) for s in symbols]) / scale
    result = scipy.optimize.minimize(
        negative, x0, method="L-BFGS-B", bounds=scaled, options={"ftol": 1e-14, "gtol": 1e-10}
    )
    return -float(result.fun), {s: float(v) for s, v in zip(symbols, result.x * scale)}
