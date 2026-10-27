"""Spin operator matrices: EasySpin's ``sop`` and the extended Stevens operators ``stev``.

Basis order is EasySpin's ``|mS, mI, ...>`` with the first spin slowest and
``m`` descending within each spin (``+S`` first). Spin and level numbers in
operator specifications are 1-based, as in EasySpin.

EasySpin authors, by original file (contributors to the file up to EasySpin 6.0.15, see
  docs/easyspin-credits.md):
  sop.m: Stefan Stoll, Stephan Pribitzer, Peter Martin
  stev.m: Stefan Stoll, AlbanBloorRiley (GitHub user)
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import scipy.sparse as sps

from .spinsystem import spinvec

_COMPONENT_OF_INDEX = {0: "e", 1: "x", 2: "y", 3: "z", 4: "+", 5: "-"}
_SYNTAX3 = re.compile(r"^([exyzab+-])(\((\d+)(\|(\d+))?\))?(\d+)?$")

# Stevens prefactors F(k, |q|) from Ryabov, J. Magn. Reson. 140, 141 (1999), as tabulated in EasySpin
_F: dict[int, list[int]] = {
    0: [1],
    1: [2, 1],
    2: [4, 2, 1],
    3: [24, 6, 6, 1],
    4: [48, 24, 8, 4, 1],
    5: [480, 240, 240, 10, 10, 1],
    6: [2880, 1440, 360, 60, 12, 6, 1],
    7: [40320, 5040, 1680, 168, 168, 14, 14, 1],
    8: [80640, 40320, 40320, 6720, 672, 336, 16, 8, 1],
    9: [1451520, 725700, 725700, 60480, 60480, 864, 288, 18, 18, 1],
    10: [14515200, 7257600, 1209600, 604800, 86400, 2880, 360, 180, 20, 10, 1],
    11: [319334400, 79833600, 79833600, 13305600, 2661120, 23760, 7920, 1320, 1320, 22, 22, 1],
    12: [1916006400, 958003200, 958003200, 31933440, 3991680, 1995840, 31680, 15840, 1584, 264, 24, 12, 1],
}


def _spins(spins) -> np.ndarray:
    vec = spinvec(spins) if isinstance(spins, dict) else np.atleast_1d(np.asarray(spins, dtype=float)).ravel()
    if np.any(vec < 0) or np.any(np.mod(vec, 0.5)) or np.iscomplexobj(vec):
        raise ValueError("First input must contain valid spin quantum numbers (0, 1/2, 1, 3/2, etc).")
    return vec


def _single_spin_matrix(I: float, component: str, transition: list[int] | None) -> sps.csr_array:
    """The (2I+1)-dimensional matrix of one component for one spin, optionally transition-selective."""
    n = int(round(2 * I + 1))
    if transition:
        t = [k - 1 for k in transition]  # 1-based levels -> 0-based
    if component == "x":
        if transition:
            r, c, val = [t[0], t[1]], [t[1], t[0]], [0.5, 0.5]
        else:
            m = np.arange(1, n)
            dia = 0.5 * np.sqrt(m * m[::-1])
            r, c, val = np.concatenate([m - 1, m]), np.concatenate([m, m - 1]), np.concatenate([dia, dia])
    elif component == "y":
        if transition:
            r, c, val = [t[0], t[1]], [t[1], t[0]], [-0.5j, 0.5j]
        else:
            m = np.arange(1, n)
            dia = -0.5j * np.sqrt(m * m[::-1])
            r, c, val = np.concatenate([m - 1, m]), np.concatenate([m, m - 1]), np.concatenate([dia, -dia])
    elif component == "z":
        if transition:
            r, c, val = [t[0], t[1]], [t[0], t[1]], [0.5, -0.5]
        else:
            m = np.arange(1, n + 1)
            r, c, val = m - 1, m - 1, I + 1 - m
    elif component == "+":
        if transition:
            r, c, val = [t[0]], [t[1]], [1.0]
        else:
            m = np.arange(1, n)
            r, c, val = m - 1, m, np.sqrt(m * m[::-1])
    elif component == "-":
        if transition:
            r, c, val = [t[1]], [t[0]], [1.0]
        else:
            m = np.arange(1, n)
            r, c, val = m, m - 1, np.sqrt(m * m[::-1])
    elif component == "a":
        r, c, val = [0], [0], [1.0]
    elif component == "b":
        r, c, val = [1], [1], [1.0]
    elif component == "e":
        if transition:
            r, c, val = [t[0]], [t[0]], [1.0]
        else:
            r = c = np.arange(n)
            val = np.ones(n)
    else:
        raise ValueError("Unknown operator specification.")
    dtype = complex if component == "y" else float
    return sps.csr_array((np.asarray(val, dtype=dtype), (np.asarray(r), np.asarray(c))), shape=(n, n))


def _parse_spec(spec: str, n_spins: int) -> tuple[list[int], list[str], dict[int, list[int]]]:
    """Spin numbers (1-based), components and transition levels of one string specification."""
    transitions: dict[int, list[int]] = {}
    if re.fullmatch(rf"[exyzab+-]{{{n_spins}}}", spec):
        return list(range(1, n_spins + 1)), list(spec), transitions
    if re.fullmatch(rf"([exyzab+-]\d+){{1,{n_spins}}}", spec):
        tokens = re.findall(r"[exyzab+-]\d+", spec)
        spins = [int(t[1:]) for t in tokens]
        if any(s > n_spins for s in spins):
            raise ValueError(f"sop: The spin system only contains {n_spins} spins, but you requested '{spec}'.")
        if len(set(spins)) < len(spins):
            raise ValueError(f"sop: Duplicate spin index in '{spec}'.")
        return spins, [t[0] for t in tokens], transitions
    parts = spec.split(",")
    matches = [_SYNTAX3.match(p) for p in parts]
    if not all(matches):
        raise ValueError(f"sop: Cannot parse '{spec}' for the given spin system.")
    spins, coords, index_present = [], [], []
    for k, m in enumerate(matches, start=1):
        component, level1, level2, spin_index = m.group(1), m.group(3), m.group(5), m.group(6)
        coords.append(component)
        index_present.append(spin_index is not None)
        spins.append(int(spin_index) if spin_index is not None else k)
        if level1 is None:
            continue
        if component in "ab":
            raise ValueError(f"sop: 'a' or 'b' cannot be used in conjunction with levels in '{spec}'")
        if level2 is not None:
            levels = [int(level1), int(level2)]
            if component in "xyz+-" and levels[0] == levels[1]:
                raise ValueError(
                    f"sop: The component '{parts[k - 1]}' of your spin operator connects identical levels."
                )
            if component == "e" and levels[0] != levels[1]:
                raise ValueError(
                    f"sop: The component '{parts[k - 1]}' can not connect two different levels. "
                    "Use e(L1) instead of e(L1|L2)."
                )
            if levels[1] < levels[0]:
                raise ValueError(
                    f"sop: The two levels (L1|L2) in '{parts[k - 1]}' must be ascending order such that L1<=L2."
                )
        else:
            levels = [int(level1)]
            if component in "xyz+-":
                raise ValueError(
                    f"sop: The component '{parts[k - 1]}' of your operator must connect two (different) levels."
                )
        transitions[k] = levels
    if any(index_present) and not all(index_present):
        raise ValueError(
            f"sop: Please give spin indices for all components, or omit them all. '{spec}' is inconsistent."
        )
    if not any(index_present):
        if len(parts) != n_spins:
            raise ValueError(f"sop: Could not determine what '{spec}' is for the given spin system.")
        spins = list(range(1, n_spins + 1))
    if len(set(spins)) < len(spins):
        raise ValueError(f"sop: Repeated spin index in '{spec}'.")
    # transitions are keyed by position in the list; re-key by spin number
    transitions = {spins[pos - 1]: levels for pos, levels in transitions.items()}
    return spins, coords, transitions


def _build(
    spin_vec: np.ndarray, spins: list[int], coords: list[str], transitions: dict[int, list[int]]
) -> sps.csr_array:
    if any(spin_vec[s - 1] != 0.5 for s, c in zip(spins, coords) if c in "ab"):
        raise ValueError("'a' and 'b' work only for spin-1/2.")
    components = ["e"] * spin_vec.size
    for s, c in zip(spins, coords):
        if s < 1 or s > spin_vec.size:
            raise ValueError(f"Spin index {s} out of range for a system of {spin_vec.size} spins.")
        components[s - 1] = c
    op = sps.csr_array(np.ones((1, 1)))
    for k, I in enumerate(spin_vec, start=1):
        if I <= 0:
            continue
        op = sps.kron(op, _single_spin_matrix(I, components[k - 1], transitions.get(k)), format="csr")
    return op


def sop(spins, *specs, sparse: bool = False):
    """Spin operator matrices of a spin system in the ``|mS, mI, ...>`` basis.

    ``spins`` is a vector of spin quantum numbers or a spin system dict.
    Each specification in ``specs`` is either a string in EasySpin's syntax
    (``'xye'``: one component per spin; ``'x2,z3'`` or ``'x2z3'``: component
    and spin number; ``'x(1|3)'``: transition-selective between levels 1 and 3)
    or an (N, 2) array of ``[spin, component]`` rows with component 0..5 for
    e, x, y, z, +, -. One specification returns one matrix, several a tuple.
    ``sparse=True`` returns ``scipy.sparse`` CSR matrices.

    EasySpin authors (sop.m): Stefan Stoll, Stephan Pribitzer, Peter Martin.
    """
    spin_vec = _spins(spins)
    if not specs:
        raise TypeError("Not enough input arguments - at least one operator specification is needed!")
    results: list[Any] = []
    for spec in specs:
        if isinstance(spec, str):
            s, c, t = _parse_spec(spec, spin_vec.size)
        else:
            a = np.asarray(spec)
            if a.ndim != 2 or a.shape[1] != 2:
                raise ValueError("For numeric input, the array specifying spins and components must be Nx2.")
            s = [int(v) for v in a[:, 0]]
            c = [_COMPONENT_OF_INDEX[int(v)] for v in a[:, 1]]
            t = {}
        op = _build(spin_vec, s, c, t)
        results.append(op if sparse else op.toarray())
    return results[0] if len(results) == 1 else tuple(results)


def stev(spins, k: int, q: int, ispin: int = 1, *, sparse: bool = False):
    """Extended Stevens operator O_k^q for spin number ``ispin`` (1-based) of ``spins``.

    ``q >= 0`` gives the cosine tesseral operator O_k^|q|(c), ``q < 0`` the sine
    operator O_k^|q|(s); ``0 <= k <= 2S`` and ``|q| <= k`` with ``k <= 12``.

    EasySpin authors (stev.m): Stefan Stoll, AlbanBloorRiley (GitHub user).
    """
    spin_vec = _spins(spins)
    if ispin < 1 or ispin > spin_vec.size:
        raise ValueError(f"iSpin = {ispin} is out of range. It should be between 1 and {spin_vec.size}")
    S = spin_vec[ispin - 1]
    if k != int(k) or k < 0:
        raise ValueError("k must be a non-negative integer.")
    if k > 12:
        raise ValueError("k too large. Maximum supported k is 12.")
    if k > 2 * S:
        raise ValueError(f"Stevens operator with k={k} given. k must not be larger than 2*S ({2 * S:g} with S={S:g}).")
    if q != int(q) or abs(q) > k:
        raise ValueError(f"q must be an integer between -k and k ({-k} and {k}).")
    k, q = int(k), int(q)
    Jp = sop(spin_vec, np.array([[ispin, 4]]), sparse=True).astype(complex)
    Jm = sop(spin_vec, np.array([[ispin, 5]]), sparse=True).astype(complex)
    T = sps.csr_array(sps.eye(Jp.shape[0], dtype=complex, format="csr"))
    for _ in range(k):
        T = T @ Jp
    for _ in range(k - 1, abs(q) - 1, -1):
        T = Jm @ T - T @ Jm
    alpha = 1.0 if k % 2 else (0.5 if q % 2 else 1.0)
    c = (-1) ** (k - q) * alpha / _F[k][abs(q)]
    op = c / 2 * (T + T.conj().T) if q >= 0 else c / 2j * (T - T.conj().T)
    op = sps.csr_array(op)
    op.data[np.abs(op.data) < 1e-14] = 0
    op.eliminate_zeros()
    return op if sparse else op.toarray()
