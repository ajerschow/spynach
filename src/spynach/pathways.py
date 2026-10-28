"""Cross-relaxation pathways resolved by relaxation mechanism, with analytical rate expressions.

:func:`relaxation_pathways` builds the Bloch-Redfield-Wangsness (BRW)
relaxation matrix of a spin system made by :func:`spynach.create` in a small
basis of product operators, and splits every element into the contributions
of the relaxation mechanisms, dipole-dipole (``DD``), chemical shift
anisotropy (``CSA``), quadrupolar (``Q``) and anisotropic scalar coupling
(``JA``), and of the cross-correlations between them (``DD/CSA``,
``DD/DD``, ``CSA/Q``, ...). Every contribution comes as a rate (s^-1) and
as a SymPy expression in the interaction strengths and the spectral
density ``J(omega)``; :func:`spynach.plotting.plot_pathways` draws the
result as a chord diagram. This module has no Spinach counterpart.

Theory. The anisotropic Hamiltonian is written as
``H1(t) = sum_mu sum_q A^mu_q(t)^* T^mu_q``, with the rank-2 spherical
components ``A^mu_q`` of the spatial tensor of mechanism ``mu`` and its spin
tensor operators ``T^mu_q``. Isotropic rotational diffusion gives
``<A^mu_q(0)^* A^nu_q(t)> = (G_mu_nu / 5) exp(-t / tau_c)``, where
``G_mu_nu`` is the scalar product of the two rank-2 tensors in the molecular
frame (so the geometry of the cross-correlation is exact), and the BRW
master equation ``d rho / dt = -Gamma rho`` with

    Gamma rho = sum_{mu,nu} sum_q sum_{w,w'} (G_mu_nu / 5) J(w') [T^mu_{q,w}, [T^nu_{q,w'}^+, rho]]

where ``T_{q,w}`` is the part of ``T_q`` that oscillates at the frequency
``w`` of the Zeeman Hamiltonian, ``J(w) = tau_c / (1 + w^2 tau_c^2)``, and
the dynamic frequency shifts are dropped. The secular approximation keeps
the pairs ``w = w'`` (``secular='isotope'``). The double commutators are
evaluated numerically in Hilbert space with the interaction strengths set
to one; the remaining numbers (geometry and spin algebra) are recognised as
square roots of rationals where possible, which turns each matrix element
into an exact expression. The same Gamma is what :func:`spynach.relaxation`
returns (as ``-R``) for ``inter['relaxation'] = ['redfield']`` with one
isotropic correlation time; ``tests/unit/test_pathways.py`` checks the two
against each other.
"""

from __future__ import annotations

import itertools
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from functools import reduce
from typing import Any

import numpy as np
import scipy.sparse as sps

from spynach.operators import irr_sph_ten, pauli
from spynach.system import SpinSystem

KINDS = ("DD", "CSA", "Q", "JA")
SECULAR = ("isotope", "total", "none")

# Relative size below which a matrix element of a mechanism pair counts as zero
_ZERO = 1e-10


def _sympy():
    try:
        import sympy
    except ImportError as error:
        raise ImportError('relaxation pathways need SymPy: pip install -e ".[symbolic]"') from error
    return sympy


@dataclass(frozen=True, eq=False)
class Mechanism:
    """One relaxation mechanism: a rank-2 interaction modulated by rotational diffusion.

    ``strength`` is a SymPy expression in rad/s (``b_BF`` for a dipolar
    coupling, ``omega_F*dsigma_F`` for a shielding anisotropy, ``2*pi*CQ_B``
    for a quadrupolar coupling, ``2*pi*dJ_BF`` for an anisotropic scalar
    coupling) and ``value`` its number; ``shape`` holds the rank-2 spherical
    components (``+2 ... -2``) of the spatial tensor divided by ``value``, in
    the molecular frame of the spin system.
    """

    kind: str
    spins: tuple[int, ...]
    name: str
    strength: Any
    value: float
    shape: np.ndarray


@dataclass(frozen=True, eq=False)
class ProductOperator:
    """A basis operator of the pathway analysis, normalised to unit Frobenius norm.

    Spins of one group enter collectively: in a group of equivalent spins
    ``Fz`` is ``sum_k F_kz`` and ``FzFz`` is ``sum_{k<l} F_kz F_lz``.
    ``coherence`` holds the coherence order of each isotope, in the order
    of :attr:`Pathways.isotopes`.
    """

    label: str
    html: str
    latex: str
    coherence: tuple[int, ...]
    matrix: np.ndarray

    @property
    def order(self) -> int:
        """Total coherence order."""
        return sum(self.coherence)


@dataclass(frozen=True, eq=False)
class Link:
    """A cross-relaxation pathway: the element ``Gamma[source, target]`` (s^-1), resolved by mechanism class.

    ``signature`` lists the classes whose contribution is not zero by
    symmetry, whatever its size; ``rates`` and ``expressions`` hold the
    contribution of each of them.
    """

    source: int
    target: int
    rate: float
    rates: dict[str, float]
    expressions: dict[str, Any]
    signature: tuple[str, ...]


@dataclass(eq=False)
class Pathways:
    """The relaxation matrix of :func:`relaxation_pathways`, split by mechanism class.

    ``terms[cls][(strengths, frequency)]`` is the matrix of numerical
    coefficients that multiplies the product of interaction strengths
    ``strengths`` and the spectral density ``J(frequency)`` (both SymPy
    expressions) in class ``cls``, so that
    ``Gamma = sum coefficient * strengths * J(frequency)`` and
    ``d<O_i>/dt = -sum_j Gamma[i, j] <O_j>``.
    """

    operators: list[ProductOperator]
    mechanisms: list[Mechanism]
    isotopes: tuple[str, ...]
    groups: tuple[tuple[int, ...], ...]
    symbols: dict[Any, float]
    latex_names: dict[Any, str]
    frequencies: dict[Any, float]
    tau_c: float
    secular: str
    terms: dict[str, dict[tuple[Any, Any], np.ndarray]]
    _values: dict = field(default_factory=dict, repr=False)

    @property
    def classes(self) -> list[str]:
        """Mechanism classes present: auto-correlated kinds first, then cross-correlations."""
        return sorted(self.terms, key=_class_order)

    @property
    def labels(self) -> list[str]:
        return [op.label for op in self.operators]

    def index(self, label: str) -> int:
        """Position of the operator with this plain-text label."""
        for n, op in enumerate(self.operators):
            if op.label == label:
                return n
        raise KeyError(f"no operator {label!r}; the basis holds {self.labels}")

    def spectral_density(self, frequency: Any) -> float:
        """``J(w) = tau_c / (1 + w^2 tau_c^2)`` (s) at a frequency expression of this analysis."""
        w = self.frequencies[frequency]
        return self.tau_c / (1 + (w * self.tau_c) ** 2)

    def rate_matrix(self, classes: str | Iterable[str] | None = None) -> np.ndarray:
        """``Gamma`` (s^-1) in the operator basis, from all classes or the ones given."""
        selected = self._select(classes)
        n = len(self.operators)
        gamma = np.zeros((n, n))
        for cls in selected:
            for (strengths, frequency), coeff in self.terms[cls].items():
                gamma += coeff * self._value(strengths) * self.spectral_density(frequency)
        return gamma

    def expression(self, source: int | str, target: int | str, classes: str | Iterable[str] | None = None):
        """``Gamma[source, target]`` as a SymPy expression, from all classes or the ones given."""
        sympy = _sympy()
        i, j = (self.index(x) if isinstance(x, str) else x for x in (source, target))
        return sympy.Add(*(self._class_expression(cls, i, j) for cls in self._select(classes)))

    def links(self, *, auto: bool = False) -> list[Link]:
        """The nonzero off-diagonal elements ``Gamma[i, j]``, ``i < j``, largest first (``auto`` adds ``i == j``)."""
        n = len(self.operators)
        per_class = {cls: self.rate_matrix(cls) for cls in self.classes}
        out = []
        for i in range(n):
            for j in range(i if auto else i + 1, n):
                present = [
                    cls
                    for cls in self.classes
                    if any(abs(c[i, j]) > _ZERO or abs(c[j, i]) > _ZERO for c in self.terms[cls].values())
                ]
                if not present:
                    continue
                rates = {cls: float(per_class[cls][i, j]) for cls in present}
                expressions = {cls: self._class_expression(cls, i, j) for cls in present}
                out.append(Link(i, j, sum(rates.values()), rates, expressions, tuple(present)))
        return sorted(out, key=lambda link: -abs(link.rate))

    def table(self, *, auto: bool = False, latex: bool = True, digits: int = 4) -> str:
        """Markdown table of the pathways with each class's rate and expression (LaTeX, or SymPy's text form)."""
        sympy = _sympy()
        rows = ["| pathway | mechanism | rate (s⁻¹) | expression |", "|---|---|---:|---|"]
        for link in self.links(auto=auto):
            a, b = self.operators[link.source], self.operators[link.target]
            name = f"${a.latex} \\leftrightarrow {b.latex}$" if latex else f"{a.label} <-> {b.label}"
            if link.source == link.target:
                name = f"${a.latex}$ (auto)" if latex else f"{a.label} (auto)"
            for cls in link.signature:
                expr = link.expressions[cls]
                text = f"${sympy.latex(expr, symbol_names=self.latex_names)}$" if latex else str(expr)
                rows.append(f"| {name} | {cls} | {link.rates[cls]:.{digits}g} | {text} |")
                name = ""
        return "\n".join(rows)

    def states(self, spin_system: SpinSystem) -> np.ndarray:
        """The basis operators as unit vectors of the ``sphten-liouv`` basis of ``spin_system``, one per row.

        ``spin_system`` is the system of the analysis after :func:`spynach.basis`; with a full basis
        ``-(V.conj() @ R @ V.T)`` is the lab-frame ``Gamma`` from ``R = spynach.relaxation(spin_system)``.
        A restricted basis keeps only the projections, which are then not of unit length.
        """
        if spin_system.basis is None or spin_system.basis.formalism != "sphten-liouv":
            raise ValueError("states() needs a spin system with a sphten-liouv basis, run basis() first.")
        rows = []
        for row in spin_system.basis.basis:
            matrix = np.ones((1, 1))
            for n, index in enumerate(row):
                rank = int(np.sqrt(index))
                proj = rank * rank + rank - int(index)
                matrix = np.kron(matrix, irr_sph_ten(int(spin_system.mults[n]), rank)[rank - proj].toarray())
            rows.append((matrix / np.linalg.norm(matrix)).ravel())
        if np.size(rows[0]) != self.operators[0].matrix.size:
            raise ValueError("spin_system is not the system of this analysis.")
        return np.array([np.array(rows).conj() @ op.matrix.ravel() for op in self.operators])

    def _select(self, classes: str | Iterable[str] | None) -> list[str]:
        if classes is None:
            return self.classes
        selected = [classes] if isinstance(classes, str) else list(classes)
        unknown = set(selected) - set(self.terms)
        if unknown:
            raise KeyError(f"unknown mechanism class(es) {sorted(unknown)}; present: {self.classes}")
        return selected

    def _value(self, expr: Any) -> float:
        if expr not in self._values:
            self._values[expr] = float(expr.subs(self.symbols))
        return self._values[expr]

    def _class_expression(self, cls: str, i: int, j: int):
        sympy = _sympy()
        J = sympy.Function("J")
        brackets: dict[Any, list] = {}
        for (strengths, frequency), coeff in self.terms[cls].items():
            if abs(coeff[i, j]) > _ZERO:
                # the number in the strengths (4 of (2 pi CQ)^2, say) joins the coefficient before it is recognised
                number, symbols = strengths.as_coeff_Mul()
                brackets.setdefault(symbols, []).append(_exact(coeff[i, j] * float(number)) * J(frequency))
        parts = [strengths * sympy.factor_terms(sympy.Add(*terms)) for strengths, terms in brackets.items()]
        return sympy.Add(*parts)


def relaxation_pathways(
    spin_system: SpinSystem,
    *,
    tau_c: float | None = None,
    groups: Sequence[Sequence[int]] | None = None,
    ranks: dict[str, Sequence[int]] | None = None,
    max_order: int = 2,
    max_per_group: int = 1,
    coherence: int | Sequence[int] | None = None,
    secular: str = "isotope",
    mechanisms: Sequence[str] = KINDS,
) -> Pathways:
    """Relaxation matrix of a spin system between product operators, split by mechanism, with expressions.

    The mechanisms are read from the interactions of ``spin_system`` (made
    by :func:`spynach.create`; no basis needed): a ``DD`` mechanism for every
    proximate pair with coordinates, a ``CSA`` mechanism for every
    anisotropic Zeeman tensor, a ``Q`` mechanism for every quadratic
    coupling of a spin with itself, and a ``JA`` mechanism for the rank-2
    part of a bilinear coupling tensor beyond the dipolar one. Antisymmetric
    (rank-1) parts are ignored. Equal interaction strengths share a symbol.
    ``mechanisms`` selects the kinds to include.

    ``tau_c`` is the isotropic rotational correlation time (s), by default
    the single one of ``inter['tau_c']``.

    The basis: ``groups`` lists the spins treated collectively (default:
    spins of the same isotope and isotropic chemical shift), and the
    operators are products over groups of either the unit operator or a
    symmetrised sum over the group's spins of at most ``max_per_group``
    single-spin operators, with at most ``max_order`` active spins in all.
    The single-spin operators are ``z``, ``+``, ``-`` for rank 1, and
    ``T(l, m)`` for the ranks ``l`` listed in ``ranks``, a dictionary keyed by
    group letter or isotope (default rank 1 only). ``coherence`` keeps only
    operators of the given total coherence order(s), for instance ``0`` for
    the zero-quantum space.

    ``secular``: ``'isotope'`` keeps the pairs of components with equal
    frequencies (the usual high-field secular approximation; only operators
    with the same coherence order of every isotope are connected);
    ``'total'`` also keeps the pairs with equal total coherence order, which
    connects for example ``B+F-`` with ``BzFz``: these terms oscillate at the
    difference of the Larmor frequencies and are secular only when the
    Larmor frequencies are close; ``'none'`` keeps everything (the
    lab-frame Redfield superoperator, ``inter['rlx_keep'] = 'labframe'``).
    Spins of one isotope are taken as degenerate in the spectral density
    arguments (chemical shift differences are neglected next to Larmor
    frequencies).
    """
    sympy = _sympy()
    if secular not in SECULAR:
        raise ValueError(f"secular must be one of {SECULAR}, not {secular!r}.")
    unknown = set(mechanisms) - set(KINDS)
    if unknown:
        raise ValueError(f"unknown mechanism kind(s) {sorted(unknown)}; use {KINDS}.")
    if any(kind != "S" for kind in spin_system.types):
        raise NotImplementedError("relaxation pathways are only available for spins, not bosonic modes.")
    tau_c = _tau_c(spin_system, tau_c)

    nspins = spin_system.nspins
    mults = [int(m) for m in spin_system.mults]
    isotopes = tuple(dict.fromkeys(spin_system.isotopes))
    iso_of = [isotopes.index(name) for name in spin_system.isotopes]
    signs = [int(np.sign(spin_system.basefrqs[spin_system.isotopes.index(name)])) or 1 for name in isotopes]
    groups = _groups(spin_system, groups)
    names = _spin_names(spin_system)
    letters = _group_letters(spin_system, groups, names)

    iso_labels = _isotope_labels(spin_system)
    omega_symbols = {name: sympy.Symbol(f"omega_{iso_labels[name]}", positive=True) for name in isotopes}
    latex_names = {sym: rf"\omega_{{{iso_labels[name]}}}" for name, sym in omega_symbols.items()}
    symbols: dict[Any, float] = {
        omega_symbols[name]: float(abs(spin_system.basefrqs[spin_system.isotopes.index(name)])) for name in isotopes
    }
    mechs = _mechanisms(spin_system, names, omega_symbols, symbols, latex_names, mechanisms)

    # Hilbert space: single-spin operators embedded in the product space, and the coherence of every
    # matrix element per isotope (the change of total projection, in units of 1)
    dim = int(np.prod(mults))
    embed = _embedder(mults)
    levels = np.array(np.unravel_index(np.arange(dim), mults)).T  # (dim, nspins) Zeeman level, 0 = highest m
    two_m = np.array([[mults[n] - 1 - 2 * levels[s, n] for n in range(nspins)] for s in range(dim)])
    m_iso = np.zeros((dim, len(isotopes)), dtype=np.int64)
    for n in range(nspins):
        m_iso[:, iso_of[n]] += two_m[:, n]
    delta = (m_iso[:, None, :] - m_iso[None, :, :]) // 2  # (dim, dim, niso)

    operators = _operators(
        spin_system.isotopes, mults, groups, letters, iso_of, len(isotopes), embed, ranks, max_order, max_per_group
    )
    if coherence is not None:
        wanted = {coherence} if isinstance(coherence, (int, np.integer)) else set(coherence)
        operators = [op for op in operators if op.order in wanted]
    if not operators:
        raise ValueError("no operators left in the basis; check coherence, ranks and max_order.")
    spin_ops = [{axis: embed(n, pauli(mults[n])[axis].toarray()) for axis in "xyz"} for n in range(nspins)]
    components = [_components(_spin_tensor(mech, spin_ops), delta) for mech in mechs]
    kets = [sps.csr_array(op.matrix) for op in operators]

    frequencies: dict[Any, float] = {}

    def frequency(key: tuple[int, ...]):
        coeffs = [-signs[k] * key[k] for k in range(len(isotopes))]  # w = sum_k dm_k * basefrq_k
        first = next((c for c in coeffs if c != 0), 0)
        if first < 0:
            coeffs = [-c for c in coeffs]
        expr = sympy.Add(*(c * omega_symbols[name] for c, name in zip(coeffs, isotopes)))
        frequencies.setdefault(expr, float(sum(c * symbols[omega_symbols[name]] for c, name in zip(coeffs, isotopes))))
        return expr

    def keep(key: tuple[int, ...], other: tuple[int, ...]) -> bool:
        if secular == "isotope":
            return key == other
        if secular == "total":
            return sum(key) == sum(other)
        return True

    # <O_i| [T_mu, [T_nu^+, O_j]] > = < [T_mu^+, O_i] | [T_nu^+, O_j] >: the commutator of every frequency
    # component with every basis operator is formed once, as a row of a sparse matrix, and each block of
    # mechanism pairs with the same q is then one sparse product of those rows
    shapes = np.array([mech.shape for mech in mechs])
    G = np.real(shapes.conj() @ shapes.T) if mechs else np.zeros((0, 0))
    n = len(operators)
    terms: dict[str, dict[tuple[Any, Any], np.ndarray]] = {}
    for q in range(5):
        rows: dict[tuple[int, ...], list[tuple[int, sps.csr_array]]] = {}
        for a in range(len(mechs)):
            for key, T in components[a][q].items():
                rows.setdefault(key, []).append((a, _commutator_rows(T, kets)))
        for key_mu, left in rows.items():
            stacked = sps.vstack([r for _, r in left], format="csr").conj()
            for key_nu, right in rows.items():
                if not keep(key_mu, key_nu):
                    continue
                gram = (stacked @ sps.vstack([r for _, r in right], format="csr").T).toarray()  # GPU: block products
                for x, (a, _) in enumerate(left):
                    for y, (b, _) in enumerate(right):
                        g = G[a, b]
                        if abs(g) < _ZERO:
                            continue
                        M = gram[x * n : (x + 1) * n, y * n : (y + 1) * n]
                        if np.abs(M).max() < _ZERO:
                            continue
                        slot = terms.setdefault(_class_name(mechs[a], mechs[b]), {})
                        index = (mechs[a].strength * mechs[b].strength, frequency(key_nu))
                        slot[index] = slot.get(index, 0) + (g / 5) * M
    for cls in terms:
        for coeff in terms[cls].values():
            if np.abs(coeff.imag).max() > 1e-8 * max(np.abs(coeff).max(), 1.0):
                raise RuntimeError("complex relaxation matrix element: the operator basis is not real.")
        terms[cls] = {k: np.real(v) for k, v in terms[cls].items() if np.abs(v).max() > _ZERO}
    terms = {cls: t for cls, t in terms.items() if t}
    return Pathways(
        operators=operators,
        mechanisms=mechs,
        isotopes=isotopes,
        groups=groups,
        symbols=symbols,
        latex_names=latex_names,
        frequencies=frequencies,
        tau_c=tau_c,
        secular=secular,
        terms=terms,
    )


def _tau_c(spin_system: SpinSystem, tau_c: float | None) -> float:
    if tau_c is not None:
        return float(tau_c)
    taus = spin_system.rlx.tau_c
    if len(taus) != 1 or np.size(taus[0]) != 1:
        raise ValueError("give tau_c, or a single isotropic correlation time in inter['tau_c'].")
    return float(np.ravel(taus[0])[0])


def _element(isotope: str) -> str:
    match = re.fullmatch(r"\d*([A-Za-z]+)", isotope)
    return match[1] if match else isotope


def _groups(spin_system: SpinSystem, groups: Sequence[Sequence[int]] | None) -> tuple[tuple[int, ...], ...]:
    nspins = spin_system.nspins
    if groups is None:
        # Spins of one isotope with the same isotropic chemical shift are taken as equivalent
        keys: dict[tuple[str, float], list[int]] = {}
        for n in range(nspins):
            shift = np.trace(spin_system.zeeman_matrix[n]) / 3 / spin_system.basefrqs[n]
            keys.setdefault((spin_system.isotopes[n], round(float(shift), 9)), []).append(n)
        return tuple(tuple(g) for g in keys.values())
    out = tuple(tuple(int(n) for n in g) for g in groups)
    flat = sorted(n for g in out for n in g)
    if flat != list(range(nspins)):
        raise ValueError(f"groups must hold every spin index 0..{nspins - 1} exactly once.")
    for g in out:
        if len({spin_system.isotopes[n] for n in g}) > 1:
            raise ValueError("spins of one group must be of the same isotope.")
    return out


def _isotope_labels(spin_system: SpinSystem) -> dict[str, str]:
    """The element letter of each isotope, or the isotope name when two isotopes of one element are present."""
    isotopes = list(dict.fromkeys(spin_system.isotopes))
    elements = [_element(name) for name in isotopes]
    return {name: name if elements.count(element) > 1 else element for name, element in zip(isotopes, elements)}


def _spin_names(spin_system: SpinSystem) -> list[str]:
    """Element letter of every spin, numbered in order when an element has several spins (``B``, ``F1`` ... ``F4``)."""
    labels = _isotope_labels(spin_system)
    elements = [labels[name] for name in spin_system.isotopes]
    names = []
    for n, element in enumerate(elements):
        count = elements.count(element)
        number = elements[: n + 1].count(element)
        separator = "_" if element[-1].isdigit() else ""
        names.append(element if count == 1 else f"{element}{separator}{number}")
    return names


def _group_letters(spin_system: SpinSystem, groups: tuple[tuple[int, ...], ...], names: list[str]) -> list[str]:
    """Operator labels of the groups: the element for all its spins, else the names of the spins (``F1``, ``F234``)."""
    labels = _isotope_labels(spin_system)
    elements = [labels[name] for name in spin_system.isotopes]
    letters = []
    for g in groups:
        element = elements[g[0]]
        if len(g) == elements.count(element):
            letters.append(element)
        else:
            letters.append(element + "".join(names[n][len(element) :].lstrip("_") for n in g))
    return letters


def _rank2(M) -> list:
    """Rank-2 spherical components ``+2 ... -2`` of a 3x3 array of numbers or of matrices.

    With these components a symmetric traceless ``A`` contracts with an operator
    tensor as ``sum_ab A_ab Op_ab = sum_q (-1)^q A_{-q} T_q`` and ``A_{-q} = (-1)^q A_q^*``.
    """
    (xx, xy, xz), (yx, yy, yz), (zx, zy, zz) = M
    return [
        0.5 * (xx - yy + 1j * (xy + yx)),
        -0.5 * (xz + zx + 1j * (yz + zy)),
        (2 * zz - xx - yy) / np.sqrt(6),
        0.5 * (xz + zx - 1j * (yz + zy)),
        0.5 * (xx - yy - 1j * (xy + yx)),
    ]


def _traceless(A: np.ndarray) -> np.ndarray:
    S = (A + A.T) / 2
    return S - np.eye(3) * np.trace(S) / 3


def _principal(A2: np.ndarray) -> tuple[float, float]:
    """Haeberlen anisotropy ``delta`` (eigenvalue largest in size) and asymmetry of a traceless tensor."""
    eigs = np.linalg.eigvalsh(A2)
    order = np.argsort(np.abs(eigs))
    yy, xx, zz = eigs[order[1]], eigs[order[0]], eigs[order[2]]
    return float(zz), float((yy - xx) / zz) if zz else 0.0


def _mechanisms(spin_system, names, omega_symbols, symbols, latex_names, kinds) -> list[Mechanism]:
    """The relaxation mechanisms of the spin system, with shared symbols for equal strengths."""
    sympy = _sympy()
    tols = spin_system.tols
    nspins = spin_system.nspins
    # Symbols are named after the elements, so that equal interactions share a symbol whatever the
    # grouping of the basis; a pair is named in the order of first appearance of its isotopes
    labels = _isotope_labels(spin_system)
    letter_of = {n: labels[spin_system.isotopes[n]] for n in range(nspins)}
    rank_of = {name: k for k, name in enumerate(dict.fromkeys(spin_system.isotopes))}
    found: list[tuple[str, tuple[int, ...], str, float, np.ndarray]] = []  # kind, spins, stem, value, rank-2 tensor

    def pair_stem(n: int, k: int) -> str:
        first, second = sorted((n, k), key=lambda m: rank_of[spin_system.isotopes[m]])
        return letter_of[first] + letter_of[second]

    for n in range(nspins):
        A2 = _traceless(spin_system.zeeman_matrix[n])
        if "CSA" in kinds and np.linalg.norm(A2) > 1e-9 * abs(spin_system.basefrqs[n]) + 1e-12:
            delta, _ = _principal(A2 / spin_system.basefrqs[n])  # shielding anisotropy, dimensionless
            found.append(("CSA", (n,), letter_of[n], 1.5 * delta, A2))
        coupling = spin_system.coupling_matrix[n][n]
        if (
            "Q" in kinds
            and coupling is not None
            and np.linalg.norm(_traceless(coupling)) > 2 * np.pi * tols.inter_cutoff
        ):
            A2 = _traceless(coupling)
            spin = (spin_system.mults[n] - 1) / 2
            delta, _ = _principal(A2)
            found.append(("Q", (n,), letter_of[n], delta * 4 * spin * (2 * spin - 1) / (4 * np.pi), A2))
    for n in range(nspins):
        for k in range(n + 1, nspins):
            dipolar = np.zeros((3, 3))
            xyz_n, xyz_k = spin_system.coordinates[n], spin_system.coordinates[k]
            if xyz_n is not None and xyz_k is not None and spin_system.proxmatrix[n, k]:
                dv = (xyz_k - xyz_n) * 1e-10
                r = float(np.linalg.norm(dv))
                b = spin_system.gammas[n] * spin_system.gammas[k] * tols.hbar * tols.mu0 / (4 * np.pi * r**3)
                ort = dv / r
                dipolar = b * (np.eye(3) - 3 * np.outer(ort, ort))
                if "DD" in kinds:
                    found.append(("DD", (n, k), pair_stem(n, k), b, dipolar))
            # The pair enters the Hamiltonian as I_n.(C_nk + C_kn^T).I_k
            forward, backward = spin_system.coupling_matrix[n][k], spin_system.coupling_matrix[k][n]
            total = (np.zeros((3, 3)) if forward is None else forward) + (0 if backward is None else backward.T)
            JA = _traceless(total) - dipolar
            threshold = max(2 * np.pi * tols.inter_cutoff, 1e-9 * np.linalg.norm(total))  # above rounding
            if "JA" in kinds and np.linalg.norm(JA) > threshold:
                delta, _ = _principal(JA)
                found.append(("JA", (n, k), pair_stem(n, k), 1.5 * delta / (2 * np.pi), JA))

    # One symbol per kind, stem and value; numbered when one stem carries several values
    stems: dict[tuple[str, str], list[float]] = {}
    for kind, _, stem, value, _ in found:
        values = stems.setdefault((kind, stem), [])
        if not any(abs(value - v) <= 1e-6 * max(abs(v), 1e-300) for v in values):
            values.append(value)
    prefix = {"DD": "b", "CSA": "dsigma", "Q": "CQ", "JA": "dJ"}
    tex = {"DD": "b", "CSA": r"\Delta\sigma", "Q": "C_{Q}", "JA": r"\Delta J"}
    out = []
    for kind, spins, stem, value, A2 in found:
        values = stems[(kind, stem)]
        k = next(i for i, v in enumerate(values) if abs(value - v) <= 1e-6 * max(abs(v), 1e-300))
        suffix = "" if len(values) == 1 else str(k + 1)
        sym = sympy.Symbol(f"{prefix[kind]}_{stem}{suffix}", real=True)
        symbols[sym] = float(values[k])
        sub = f"{stem}{suffix}"
        latex_names[sym] = f"{tex[kind]}^{{{sub}}}" if kind == "Q" else f"{tex[kind]}_{{{sub}}}"
        if kind == "CSA":
            strength = omega_symbols[spin_system.isotopes[spins[0]]] * sym
        elif kind in ("Q", "JA"):
            strength = 2 * sympy.pi * sym
        else:
            strength = sym
        number = float(strength.subs(symbols))
        label = f"{kind}({','.join(names[n] for n in spins)})"
        out.append(Mechanism(kind, spins, label, strength, number, np.array(_rank2(A2)) / number))
    return out


def _embedder(mults: list[int]):
    def embed(n: int, op: np.ndarray) -> np.ndarray:
        out = np.ones((1, 1), dtype=complex)
        for k, mult in enumerate(mults):
            out = np.kron(out, op if k == n else np.eye(mult))
        return out

    return embed


def _commutator_rows(T: np.ndarray, kets: list[sps.csr_array]) -> sps.csr_array:
    """``[T^+, O_j]`` for every basis operator ``O_j``, each flattened into one row of a sparse matrix."""
    dim = T.shape[0]
    dagger = sps.csr_array(T.conj().T)
    rows, cols, vals = [], [], []
    for j, O in enumerate(kets):
        K = sps.coo_array(dagger @ O - O @ dagger)
        rows.append(np.full(K.nnz, j))
        cols.append(K.row * dim + K.col)
        vals.append(K.data)
    shape = (len(kets), dim * dim)
    return sps.csr_array((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=shape)


def _spin_tensor(mech: Mechanism, spin_ops: list[dict[str, np.ndarray]]) -> list[np.ndarray]:
    """Spin tensor operators ``T_q`` (``q = +2 ... -2``) of a mechanism, from its Cartesian spin operators."""
    axes = "xyz"
    if mech.kind == "CSA":
        (n,) = mech.spins
        zero = np.zeros_like(spin_ops[n]["z"])
        cart = [[spin_ops[n][a] if b == "z" else zero for b in axes] for a in axes]
    else:
        n, k = (mech.spins[0], mech.spins[0]) if mech.kind == "Q" else mech.spins
        cart = [[spin_ops[n][a] @ spin_ops[k][b] for b in axes] for a in axes]
    return _rank2(cart)


def _components(T: list[np.ndarray], delta: np.ndarray) -> list[dict[tuple[int, ...], np.ndarray]]:
    """Split each ``T_q`` into parts of definite coherence per isotope (definite Zeeman frequency)."""
    out = []
    for Tq in T:
        parts: dict[tuple[int, ...], np.ndarray] = {}
        rows, cols = np.nonzero(np.abs(Tq) > 1e-14)
        for key in {tuple(int(x) for x in delta[r, c]) for r, c in zip(rows, cols)}:
            mask = np.all(delta == np.array(key), axis=2)
            parts[key] = np.where(mask, Tq, 0)
        out.append(parts)
    return out


def _single_spin(mult: int, ranks: Sequence[int]) -> list[tuple[str, str, str, int, np.ndarray]]:
    """Single-spin operators ``(label, html, latex, m, matrix)``: ``z``, ``+``, ``-`` (rank 1) and ``(l,m)``."""
    ops = pauli(mult)
    out = []
    for rank in ranks:
        if rank >= mult:
            continue
        if rank == 1:
            out += [
                ("z", "z", "z", 0, ops["z"].toarray()),
                ("+", "+", "+", 1, ops["p"].toarray()),
                ("-", "−", "-", -1, ops["m"].toarray()),
            ]
            continue
        for k, T in enumerate(irr_sph_ten(mult, rank)):
            m = rank - k
            signed = f"{m:+d}" if m else "0"
            out.append((f"({rank},{signed})", f"{rank},{signed}".replace("-", "−"), f"{rank},{m}", m, T.toarray()))
    return out


def _operators(
    isotopes, mults, groups, letters, iso_of, niso, embed, ranks, max_order, max_per_group
) -> list[ProductOperator]:
    """Collective product operators over the groups, ordered by total coherence order (0, +1, ..., -1)."""
    ranks = ranks or {}
    per_group = []
    for g, letter in zip(groups, letters):
        wanted = ranks.get(letter, ranks.get(isotopes[g[0]], (1,)))
        singles = _single_spin(mults[g[0]], wanted)
        choices = [((), None)]
        for size in range(1, min(max_per_group, len(g)) + 1):
            for combo in itertools.combinations_with_replacement(range(len(singles)), size):
                choices.append((combo, singles))
        per_group.append((g, letter, choices))

    out = []
    for choice in itertools.product(*(range(len(choices)) for _, _, choices in per_group)):
        rank_key = (sum(len(per_group[k][2][c][0]) for k, c in enumerate(choice)), tuple(c or 10**6 for c in choice))
        choice = [per_group[k][2][c] for k, c in enumerate(choice)]
        active = rank_key[0]
        if not 1 <= active <= max_order:
            continue
        matrix = np.eye(int(np.prod(mults)), dtype=complex)
        label = html = tex = ""
        coherence = [0] * niso
        for (g, letter, _), (combo, singles) in zip(per_group, choice):
            if not combo:
                continue
            matrix = matrix @ _symmetrised(g, combo, singles, embed)
            for index in combo:
                name, name_html, name_tex, m, _ = singles[index]
                label += f"{letter}{name}"
                html += f"{letter}<sub>{name_html}</sub>"
                tex += f"{letter}_{{{name_tex}}}"
                coherence[iso_of[g[0]]] += m
        norm = np.linalg.norm(matrix)
        if norm < 1e-12:
            continue
        out.append((rank_key, ProductOperator(label, html, tex, tuple(coherence), matrix / norm)))

    # Blocks of total coherence order 0, +1, +2, ..., -2, -1 (so that +p and -p mirror around the circle
    # of plot_pathways), sub-blocks of the coherence of each isotope in the same pattern (the secular
    # approximation never connects two sub-blocks), then fewer active spins first and the order of the groups
    def pattern(p: int) -> tuple[bool, int]:
        return (p < 0, p)

    return [
        op
        for _, op in sorted(
            out, key=lambda item: (pattern(item[1].order), tuple(pattern(c) for c in item[1].coherence), item[0])
        )
    ]


def _symmetrised(group: tuple[int, ...], combo: tuple[int, ...], singles, embed) -> np.ndarray:
    """Sum over the distinct placements of the single-spin operators ``combo`` on distinct spins of the group."""
    placements = {frozenset(zip(spins, combo)) for spins in itertools.permutations(group, len(combo))}
    total = 0
    for placement in placements:
        factors = [embed(n, singles[index][4]) for n, index in sorted(placement)]
        total = total + reduce(np.matmul, factors)
    return total


def _class_name(mu: Mechanism, nu: Mechanism) -> str:
    if mu is nu:
        return mu.kind
    first, second = sorted((mu.kind, nu.kind), key=KINDS.index)
    return f"{first}/{second}"


def _class_order(cls: str) -> tuple[int, int, int]:
    parts = cls.split("/")
    return (len(parts), *(KINDS.index(p) for p in parts), *([0] * (2 - len(parts))))


def _exact(x: float):
    """``x`` as a SymPy number: a rational, or the signed square root of one, when it is one; a float otherwise.

    The coefficients are accurate to about 1e-12 (a shielding anisotropy is the small difference of
    two Zeeman tensors), so a match within 1e-9 to a fraction of small denominator counts as exact; a
    float that only happens to lie that close to one does so with a probability of about 1e-3.
    """
    sympy = _sympy()
    if abs(x) < _ZERO:
        return sympy.Integer(0)
    for value, limit, root in ((abs(x), 1000, False), (x * x, 2000, True)):
        fraction = Fraction(value).limit_denominator(limit)
        if fraction and abs(float(fraction) - value) <= 1e-9 * value:
            number = sympy.Rational(fraction.numerator, fraction.denominator)
            number = sympy.sqrt(number) if root else number
            return number if x > 0 else -number
    return sympy.Float(x, 6)
