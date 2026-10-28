"""Relaxation pathways (spynach.pathways): rates against Spynach's Redfield superoperator and Solomon
theory, exact expressions, the symmetry of a tetrahedral AX4 system, and the chord diagram."""

import importlib.util

import numpy as np
import pytest

import spynach
from spynach.conventions import eeqq2nqi
from spynach.pathways import _exact, relaxation_pathways

sympy = pytest.importorskip("sympy")
J = sympy.Function("J")
LIOUV = {"formalism": "sphten-liouv", "approximation": "none"}


def spin_pair(keep="labframe", tau_c=20e-12):
    """11B-19F with every mechanism, at general orientations: CSA on both spins, the 11B quadrupole, DD."""
    rng = np.random.default_rng(7)
    euler = [list(rng.uniform(0, np.pi, 3)) for _ in range(3)]
    bond = np.array([0.3, -0.5, 0.81])
    inter = {
        "zeeman": {"eigs": [[-20, -10, 30], [-150, 60, 90]], "euler": euler[:2]},
        "coupling": {"matrix": [[eeqq2nqi(1.5e6, 0.3, 1.5, euler[2]).tolist(), None], [None, None]]},
        "coordinates": [[0, 0, 0], list(1.40 * bond / np.linalg.norm(bond))],
        "relaxation": ["redfield"],
        "equilibrium": "zero",
        "rlx_keep": keep,
        "tau_c": [[tau_c]],
    }
    return spynach.create({"isotopes": ["11B", "19F"], "magnet": 9.4}, inter)


def tetrahedral_ax4(keep="labframe"):
    """11BF4-: 19F shielding axial along each bond, 11B shielding and quadrupole along the first bond."""
    corners = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]]) / np.sqrt(3)
    z = corners[0]
    x = np.cross(z, [0, 0, 1.0])
    x /= np.linalg.norm(x)
    bonds = corners @ np.array([x, np.cross(z, x), z]).T

    def axial(n, anisotropy):
        return (anisotropy * (np.outer(n, n) - np.eye(3) / 3)).tolist()

    quadrupole = eeqq2nqi(0.1e6, 0.0, 1.5, [0, 0, 0]).tolist()
    inter = {
        "zeeman": {"matrix": [axial([0, 0, 1], 2.0)] + [axial(n, 100.0) for n in bonds]},
        "coupling": {"matrix": [[quadrupole] + [None] * 4] + [[None] * 5 for _ in range(4)]},
        "coordinates": [[0, 0, 0]] + (1.40 * bonds).tolist(),
        "relaxation": ["redfield"],
        "equilibrium": "zero",
        "rlx_keep": keep,
        "tau_c": [[10e-12]],
    }
    return spynach.create({"isotopes": ["11B"] + ["19F"] * 4, "magnet": 9.4}, inter)


def spynach_gamma(spin_system, pathways):
    """-R of spynach.relaxation between the pathway operators, through Pathways.states."""
    spin_system = spynach.basis(spin_system, LIOUV)
    R = spynach.relaxation(spin_system)
    V = pathways.states(spin_system)
    assert np.allclose(np.linalg.norm(V, axis=1), 1)
    return -(V.conj() @ (R @ V.T)).real


def same_coherence(pathways, per_isotope=True):
    coherence = np.array([op.coherence for op in pathways.operators])
    if not per_isotope:
        coherence = coherence.sum(axis=1, keepdims=True)
    return (coherence[:, None, :] == coherence[None, :, :]).all(axis=2)


@pytest.mark.parametrize("secular", ["none", "isotope", "total"])
def test_rates_match_spynach_redfield_with_every_mechanism(secular):
    pathways = relaxation_pathways(spin_pair(), secular=secular, ranks={"B": (1, 2, 3)})
    assert pathways.classes == ["DD", "CSA", "Q", "DD/CSA", "DD/Q", "CSA/CSA", "CSA/Q"]
    reference = spynach_gamma(spin_pair(), pathways)
    if secular != "none":  # the secular forms are masks of the lab-frame superoperator
        reference = np.where(same_coherence(pathways, per_isotope=secular == "isotope"), reference, 0)
    gamma = pathways.rate_matrix()
    assert np.abs(gamma - reference).max() < 1e-5 * np.abs(gamma).max()


def test_rates_match_spynach_redfield_for_a_collective_basis():
    pathways = relaxation_pathways(tetrahedral_ax4(), ranks={"11B": (1, 2, 3)}, max_per_group=2, max_order=3)
    assert len(pathways.operators) == 159
    reference = np.where(same_coherence(pathways), spynach_gamma(tetrahedral_ax4(), pathways), 0)
    gamma = pathways.rate_matrix()
    assert np.abs(gamma - reference).max() < 2e-4 * np.abs(gamma).max()


def test_dipolar_pair_gives_the_textbook_expressions():
    spin_system = spynach.create(
        {"isotopes": ["1H", "13C"], "magnet": 14.1},
        {
            "coordinates": [[0, 0, 0], [0, 0, 1.03]],
            "relaxation": ["redfield"],
            "rlx_keep": "secular",
            "equilibrium": "zero",
            "tau_c": [[100e-12]],
        },
    )
    pathways = relaxation_pathways(spin_system)
    b = sympy.Symbol("b_HC", real=True)
    wH, wC = sympy.symbols("omega_H omega_C", positive=True)
    expected = {
        ("Hz", "Hz"): b**2 / 10 * (J(wH - wC) + 3 * J(wH) + 6 * J(wH + wC)),
        ("Hz", "Cz"): b**2 / 10 * (6 * J(wH + wC) - J(wH - wC)),
        ("H+", "H+"): b**2 / 20 * (4 * J(0) + J(wH - wC) + 3 * J(wH) + 6 * J(wC) + 6 * J(wH + wC)),
        ("HzCz", "HzCz"): 3 * b**2 / 10 * (J(wH) + J(wC)),
        ("H+C-", "H+C-"): b**2 / 20 * (2 * J(wH - wC) + 3 * J(wH) + 3 * J(wC)),
        ("H+C+", "H+C+"): b**2 / 20 * (12 * J(wH + wC) + 3 * J(wH) + 3 * J(wC)),
    }
    for (source, target), expression in expected.items():
        assert sympy.expand(pathways.expression(source, target) - expression) == 0
    i, j = pathways.index("Hz"), pathways.index("Cz")
    value = pathways.expression(i, j).replace(J, lambda w: pathways.spectral_density(w)).subs(pathways.symbols)
    assert float(value) == pytest.approx(pathways.rate_matrix()[i, j], rel=1e-12)


def test_negative_magnetogyric_ratio_swaps_zero_and_double_quantum_frequencies():
    # 15N has gamma < 0, so the H+N- flip-flop oscillates at w_H + w_N (magnitudes): sigma_HN carries
    # 6 J(w_H - w_N) - J(w_H + w_N). The 15N is the third spin but the second isotope.
    inter = {
        "coordinates": [[0, 0, 0], [0, 1.8, 0.6], [0, 0, 1.02]],
        "relaxation": ["redfield"],
        "rlx_keep": "labframe",
        "equilibrium": "zero",
        "tau_c": [[5e-9]],
    }
    spin_system = spynach.create({"isotopes": ["1H", "1H", "15N"], "magnet": 14.1}, inter)
    pathways = relaxation_pathways(spin_system, groups=[[0], [1], [2]], secular="none")
    gamma = pathways.rate_matrix()
    assert np.abs(gamma - spynach_gamma(spin_system, pathways)).max() < 1e-5 * np.abs(gamma).max()
    b = sympy.Symbol("b_HN1", real=True)
    wH, wN = sympy.symbols("omega_H omega_N", positive=True)
    expected = b**2 / 10 * (6 * J(wH - wN) - J(wH + wN))
    assert sympy.expand(pathways.expression("H1z", "Nz", "DD") - expected) == 0


def test_tetrahedral_symmetry_cancels_the_boron_cross_correlations():
    # For any axis n, sum_k P2(n . r_k) = 0 over the four bonds of a tetrahedron, so in the collective basis
    # the 11B quadrupole and shielding do not cross-correlate with the fluorine mechanisms
    pathways = relaxation_pathways(tetrahedral_ax4(), ranks={"B": (1, 2, 3)}, coherence=0)
    assert "DD/Q" not in pathways.classes and "CSA/Q" in pathways.classes
    assert [op.order for op in pathways.operators] == [0] * len(pathways.operators)
    # the 11B quadrupole alone couples the rank-1 and rank-3 longitudinal orders
    wB = sympy.Symbol("omega_B", positive=True)
    cq = sympy.Symbol("CQ_B", real=True)
    expected = 4 * sympy.pi**2 * cq**2 * (J(wB) - J(2 * wB)) / 25
    assert sympy.simplify(pathways.expression("Bz", "B(3,0)", "Q") - expected) == 0
    # one B-F pair alone has the DD/Q cross-correlation
    assert "DD/Q" in relaxation_pathways(spin_pair()).classes


def test_symbols_are_shared_by_equal_interactions():
    pathways = relaxation_pathways(tetrahedral_ax4())
    names = sorted(str(s) for s in pathways.symbols)
    assert names == ["CQ_B", "b_BF", "b_FF", "dsigma_B", "dsigma_F", "omega_B", "omega_F"]
    assert pathways.symbols[sympy.Symbol("CQ_B", real=True)] == pytest.approx(0.1e6)
    assert pathways.symbols[sympy.Symbol("dsigma_F", real=True)] == pytest.approx(100e-6)
    assert [m.name for m in pathways.mechanisms][:3] == ["CSA(B)", "Q(B)", "CSA(F1)"]
    assert pathways.labels[:3] == ["Bz", "Fz", "BzFz"]


def test_exact_numbers():
    assert _exact(np.sqrt(5) / 15) == sympy.sqrt(5) / 15
    assert _exact(-0.3) == sympy.Rational(-3, 10)
    assert _exact(0.0) == 0
    assert isinstance(_exact(np.pi), sympy.Float)


def test_table_and_links():
    pathways = relaxation_pathways(tetrahedral_ax4())
    links = pathways.links()
    assert all(abs(a.rate) >= abs(b.rate) for a, b in zip(links, links[1:]))
    top = links[0]
    assert {pathways.labels[top.source], pathways.labels[top.target]} in ({"F+", "BzF+"}, {"F-", "BzF-"})
    assert top.signature == ("DD/CSA",)
    text = pathways.table(latex=False)
    assert "F+ <-> BzF+ | DD/CSA" in text
    assert r"\Delta\sigma_{F}" in pathways.table()


def test_arguments_are_checked():
    with pytest.raises(ValueError, match="secular"):
        relaxation_pathways(spin_pair(), secular="rotating")
    with pytest.raises(ValueError, match="mechanism"):
        relaxation_pathways(spin_pair(), mechanisms=["DD", "SR"])
    with pytest.raises(ValueError, match="no operators"):
        relaxation_pathways(spin_pair(), coherence=5)
    with pytest.raises(ValueError, match="tau_c"):
        relaxation_pathways(spynach.create({"isotopes": ["1H"], "magnet": 14.1}, {}))
    only_dipolar = relaxation_pathways(spin_pair(), mechanisms=["DD"])
    assert only_dipolar.classes == ["DD"]


@pytest.mark.skipif(importlib.util.find_spec("plotly") is None, reason="plotly not installed")
def test_chord_diagram():
    from spynach.plotting import plot_pathways

    pathways = relaxation_pathways(tetrahedral_ax4(), secular="total")
    fig = plot_pathways(pathways)
    ribbons = [trace for trace in fig.data if trace.showlegend is not False]
    assert sorted(trace.name for trace in ribbons) == ["DD", "DD + DD/DD", "DD/CSA"]
    assert len(fig.layout.annotations) == len(pathways.operators) + 1
    split = plot_pathways(pathways, split=True, merge={"DD/DD": "DD"})
    assert {trace.name for trace in split.data if trace.showlegend is not False} == {"DD", "DD/CSA"}
    assert len(plot_pathways(pathways, min_rate=0.15).data) < len(fig.data)
