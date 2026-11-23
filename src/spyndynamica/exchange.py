"""Extended Liouville space: one copy of operator space per site, for systems in chemical exchange.

A :class:`SiteDependent` object holds one operator, or one superoperator, per site. An
:class:`ExtendedOperatorBasis` repeats an operator basis once per site: the vector of a site-dependent operator
is the concatenation of the vectors of its sites, the matrix of a site-dependent superoperator has one diagonal
block per site, and a :class:`KineticExchange` between the sites is the Kronecker product of its rate matrix
with the unit matrix of operator space. Sums of these are represented together, so
``basis.numeric(-1j * commutation(hamiltonians) + exchange)`` is the generator of the exchanging system.

SpinDynamica has this as ``QM`ExtendedLiouville` ``, a module that its loader leaves out.

Author: Alexej Jerschow.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import sympy as sp

from spyndynamica.liouville import OperatorBasis, Superoperator
from spyndynamica.operators import Operator


class _Term:
    """Something with a place in a sum of extended superoperators."""

    def __add__(self, other: Any):
        if isinstance(other, _Sum):
            return _Sum([self, *other.terms])
        if isinstance(other, (_Term, Superoperator)):
            return _Sum([self, other])
        return NotImplemented

    def __radd__(self, other: Any):
        if isinstance(other, Superoperator):
            return _Sum([other, self])
        return NotImplemented

    def __sub__(self, other: Any):
        return self + (-1) * other

    def __neg__(self):
        return self * -1


class _Sum(_Term):
    """A sum of site-dependent superoperators, kinetic exchange and plain superoperators."""

    def __init__(self, terms: Sequence[Any]):
        self.terms = list(terms)

    def __add__(self, other: Any):
        if isinstance(other, _Sum):
            return _Sum(self.terms + other.terms)
        if isinstance(other, (_Term, Superoperator)):
            return _Sum([*self.terms, other])
        return NotImplemented

    def __mul__(self, scalar: Any):
        return _Sum([term * scalar for term in self.terms])

    __rmul__ = __mul__

    def __repr__(self) -> str:
        return " + ".join(repr(term) for term in self.terms)


class SiteDependent(_Term):
    """One operator, or one superoperator, per site (SpinDynamica's ``SiteDependentOperator`` and
    ``SiteDependentSuperoperator``).

    ``SiteDependent({"A": H_A, "B": H_B})`` gives the sites and their objects. Sums, differences, products with
    ``@`` and multiples act site by site; the other operand has the same sites, or is a single operator or
    superoperator, which then stands at every site. A site that is left out carries zero.
    """

    __hash__ = None  # type: ignore[assignment]

    def __init__(self, items: Mapping[Any, Any]):
        self.items = dict(items)
        if not self.items:
            raise ValueError("a site-dependent object needs at least one site.")
        kinds = {isinstance(value, Superoperator) for value in self.items.values()}
        if len(kinds) != 1 or not all(isinstance(v, (Operator, Superoperator)) for v in self.items.values()):
            raise TypeError("the sites hold operators, or they hold superoperators.")
        self.is_superoperator = kinds.pop()

    @property
    def sites(self) -> tuple[Any, ...]:
        return tuple(self.items)

    def __repr__(self) -> str:
        return "SiteDependent({" + ", ".join(f"{site!r}: {value!r}" for site, value in self.items.items()) + "})"

    def __getitem__(self, site: Any):
        return self.items[site]

    def map(self, function: Callable[[Any], Any]) -> SiteDependent:
        """The object with ``function`` applied at every site (``commutation``, for example)."""
        return SiteDependent({site: function(value) for site, value in self.items.items()})

    def _with(self, other: Any, function: Callable[[Any, Any], Any]):
        if isinstance(other, SiteDependent):
            if other.sites != self.sites:
                raise ValueError("two site-dependent objects are combined only when they have the same sites.")
            return SiteDependent({site: function(value, other.items[site]) for site, value in self.items.items()})
        if isinstance(other, (Operator, Superoperator)):
            return SiteDependent({site: function(value, other) for site, value in self.items.items()})
        return NotImplemented

    def __add__(self, other: Any):
        if isinstance(other, (KineticExchange, _Sum)):
            return _Term.__add__(self, other)
        if isinstance(other, Superoperator) and not self.is_superoperator:
            return NotImplemented
        return self._with(other, lambda a, b: a + b)

    def __radd__(self, other: Any):
        return self._with(other, lambda a, b: b + a)

    def __sub__(self, other: Any):
        if isinstance(other, (KineticExchange, _Sum)):
            return _Term.__sub__(self, other)
        return self._with(other, lambda a, b: a - b)

    def __rsub__(self, other: Any):
        return self._with(other, lambda a, b: b - a)

    def __matmul__(self, other: Any):
        return self._with(other, lambda a, b: a @ b)

    def __rmatmul__(self, other: Any):
        return self._with(other, lambda a, b: b @ a)

    def __mul__(self, scalar: Any):
        return SiteDependent({site: value * scalar for site, value in self.items.items()})

    __rmul__ = __mul__

    def __truediv__(self, scalar: Any):
        return SiteDependent({site: value / scalar for site, value in self.items.items()})

    def __call__(self, operator: Any) -> SiteDependent:
        """Site-dependent superoperators applied to an operator, or site by site to site-dependent operators."""
        if not self.is_superoperator:
            raise TypeError("only site-dependent superoperators act on operators.")
        return self._with(operator, lambda a, b: a(b))


def kinetic_exchange_matrix(sites: Sequence[Any], rates: Any, *, balance: bool = True) -> sp.Matrix:
    """The matrix ``K`` of first-order exchange between sites, ``dc/dt = K c`` (SpinDynamica's
    ``KineticExchangeMatrix``).

    ``rates`` is a square matrix with the rate constant of the conversion of site ``j`` into site ``i`` in row
    ``i`` and column ``j``, or a mapping ``{(i, j): k}`` from site ``i`` to site ``j``; a pair given in one
    direction only has the same rate constant in the other. With ``balance`` (the default) the diagonal is
    filled in so that every column adds up to zero, which conserves the total amount; a matrix whose columns
    already do so is left as it is, and one with a diagonal that conflicts is refused. Rate constants may be
    symbols.
    """
    sites = list(sites)
    n = len(sites)
    if len(set(sites)) != n:
        raise ValueError("the sites must be distinct.")
    if isinstance(rates, Mapping):
        matrix = sp.zeros(n, n)
        given = {}
        for (source, target), rate in rates.items():
            if source not in sites or target not in sites or source == target:
                raise ValueError(f"({source!r}, {target!r}) is not a pair of the sites {sites}.")
            given[(source, target)] = sp.sympify(rate)
        for (source, target), rate in list(given.items()):
            given.setdefault((target, source), rate)
        for (source, target), rate in given.items():
            matrix[sites.index(target), sites.index(source)] = rate
    else:
        matrix = sp.Matrix(rates)
        if matrix.shape != (n, n):
            raise ValueError(f"the rate matrix of {n} sites has {n} rows and {n} columns.")
    if balance:
        for column in range(n):
            total = sp.simplify(sum(matrix[:, column]))
            if total == 0:
                continue
            if matrix[column, column] != 0:
                raise ValueError(
                    f"column {column + 1} of the rate matrix neither adds up to zero nor has a zero diagonal."
                )
            matrix[column, column] = -total
    return matrix


class KineticExchange(_Term):
    """Exchange between sites as a superoperator of extended Liouville space (SpinDynamica's
    ``KineticExchangeSuperoperator``): every operator component moves between the sites with the rates of
    :func:`kinetic_exchange_matrix`, whose arguments this takes."""

    def __init__(self, sites: Sequence[Any], rates: Any, *, balance: bool = True):
        self.sites = tuple(sites)
        self.matrix = kinetic_exchange_matrix(self.sites, rates, balance=balance)
        self.balanced = all(sp.simplify(sum(self.matrix[:, c])) == 0 for c in range(len(self.sites)))

    def __repr__(self) -> str:
        return f"KineticExchange({list(self.sites)}, {self.matrix.tolist()})"

    def __mul__(self, scalar: Any):
        return KineticExchange(self.sites, self.matrix * scalar, balance=False)

    __rmul__ = __mul__


class ExtendedOperatorBasis:
    """An operator basis repeated once per site (SpinDynamica's ``ExtendedOperatorBasis[sites, opbasis]``)."""

    def __init__(self, sites: Sequence[Any], basis: OperatorBasis):
        self.sites = tuple(sites)
        if not self.sites or len(set(self.sites)) != len(self.sites):
            raise ValueError("the sites must be distinct, and there must be at least one.")
        self.basis = basis
        self.system = basis.system

    def __repr__(self) -> str:
        return f"<{self.basis.name} operator basis at the sites {list(self.sites)} of {self.system!r}>"

    def __len__(self) -> int:
        return len(self.sites) * len(self.basis)

    def _check_sites(self, sites: Sequence[Any]) -> None:
        missing = [site for site in sites if site not in self.sites]
        if missing:
            raise ValueError(f"the sites {missing} are not among the sites {list(self.sites)} of the basis.")

    # ------------------------------------------------------------------ operators
    def _operators(self, operator: Any) -> list[Operator]:
        if isinstance(operator, SiteDependent):
            if operator.is_superoperator:
                raise TypeError("expected operators, not superoperators.")
            self._check_sites(operator.sites)
            return [operator.items.get(site, self.system.zero()) for site in self.sites]
        if not isinstance(operator, Operator):
            operator = sp.sympify(operator) * self.system.unity()
        return [operator] * len(self.sites)

    def vector(self, operator: Any) -> sp.Matrix:
        """The components of a site-dependent operator, site after site (``ExtendedOperatorVectorRepresentation``).

        An operator stands at every site, and so does a number, as that multiple of the unity operator.
        """
        return sp.Matrix.vstack(*(self.basis.vector(op) for op in self._operators(operator)))

    def split(self, vector: Any) -> SiteDependent:
        """The site-dependent operator with these components: the inverse of :meth:`vector`."""
        flat = list(sp.Matrix(vector)) if not isinstance(vector, np.ndarray) else [sp.nsimplify(v) for v in vector]
        size = len(self.basis)
        if len(flat) != len(self):
            raise ValueError(f"expected {len(self)} components.")
        return SiteDependent(
            {site: self.basis.express(flat[n * size : (n + 1) * size]) for n, site in enumerate(self.sites)}
        )

    # ------------------------------------------------------------------ superoperators
    def _blocks(self, term: Any, represent: Callable[[Superoperator], Any], zero: Any, kron: Callable, unit: Any):
        """The matrix of one term as a list of rows of blocks."""
        n = len(self.sites)
        if isinstance(term, KineticExchange):
            self._check_sites(term.sites)
            rows = [[zero for _ in range(n)] for _ in range(n)]
            for a, target in enumerate(term.sites):
                for b, source in enumerate(term.sites):
                    rows[self.sites.index(target)][self.sites.index(source)] = kron(term.matrix[a, b], unit)
            return rows
        if isinstance(term, SiteDependent):
            if not term.is_superoperator:
                raise TypeError("expected superoperators, not operators.")
            self._check_sites(term.sites)
            diagonal = [represent(term.items[site]) if site in term.items else zero for site in self.sites]
        elif isinstance(term, Superoperator):
            diagonal = [represent(term)] * n
        else:
            diagonal = [kron(sp.sympify(term), unit)] * n
        return [[diagonal[a] if a == b else zero for b in range(n)] for a in range(n)]

    @staticmethod
    def _terms(superoperator: Any) -> list[Any]:
        return superoperator.terms if isinstance(superoperator, _Sum) else [superoperator]

    def matrix(self, superoperator: Any) -> sp.Matrix:
        """The exact matrix in extended Liouville space (``ExtendedSuperoperatorMatrixRepresentation``).

        ``superoperator`` is a :class:`SiteDependent` set of superoperators (one diagonal block per site), a
        superoperator or a number (the same block at every site), a :class:`KineticExchange`, or a sum of these.
        """
        size = len(self.basis)
        total = sp.zeros(len(self), len(self))
        for term in self._terms(superoperator):
            rows = self._blocks(
                term, lambda s: s.matrix(self.basis), sp.zeros(size, size), lambda c, u: c * u, sp.eye(size)
            )
            total += sp.Matrix(sp.BlockMatrix(rows))
        return total

    def numeric(self, superoperator: Any, values: Mapping[Any, Any] | None = None) -> np.ndarray:
        """The matrix in extended Liouville space as a NumPy array, with ``values`` put in for symbols."""
        size = len(self.basis)

        def number(c, unit):
            c = sp.sympify(c)
            return complex(c.subs(values) if values else c) * unit

        total = np.zeros((len(self), len(self)), dtype=complex)
        for term in self._terms(superoperator):
            rows = self._blocks(
                term, lambda s: s.numeric(values, basis=self.basis), np.zeros((size, size)), number, np.eye(size)
            )
            total += np.block(rows)
        return total


__all__ = ["ExtendedOperatorBasis", "KineticExchange", "SiteDependent", "kinetic_exchange_matrix"]
