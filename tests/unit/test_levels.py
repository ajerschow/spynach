"""Energy levels at zero and ultralow field (spynach.pathways.energy_levels): the Hamiltonian, the K, F, M labels,
the adiabatic transfer against the contributed state of user_ml/, and the relaxation of level populations and the
widths of the lines against Spynach's Redfield superoperator."""

import importlib.util
from functools import reduce
from pathlib import Path

import numpy as np
import pytest

import spynach
from spynach.conventions import eeqq2nqi
from spynach.operators import irr_sph_ten
from spynach.pathways import energy_levels, relaxation_pathways

pytest.importorskip("sympy")
LIOUV = {"formalism": "sphten-liouv", "approximation": "none"}
ROOT = Path(__file__).resolve().parents[2]


def pf6(field, tau_c=65e-12, J=710.0):
    """PF6- as in user_ml/ZULF_PF6_1D.m: 1J(PF) = 710 Hz, an octahedron of 1.6 Angstrom."""
    directions = [[0, 0, 0], [-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, -1], [0, 0, 1]]
    inter = {
        "coupling": {"scalar": [[0] + [J] * 6] + [[0] * 7 for _ in range(6)]},
        "coordinates": (1.6 * np.array(directions, dtype=float)).tolist(),
        "relaxation": ["redfield"],
        "rlx_keep": "labframe",
        "equilibrium": "zero",
        "tau_c": [[tau_c]],
    }
    return spynach.create({"isotopes": ["31P"] + ["19F"] * 6, "magnet": field}, inter)


def small(kind, field):
    """Systems small enough for the Redfield superoperator: 31P-19F2 with CSA, and 11B-19F with a quadrupole."""
    if kind == "PF2":
        inter = {
            "zeeman": {"matrix": [None, np.diag([-40.0, -40, 80]).tolist(), np.diag([30.0, 30, -60]).tolist()]},
            "coupling": {"scalar": [[0, 710.0, 710.0], [0, 0, 0], [0, 0, 0]]},
            "coordinates": [[0, 0, 0], [1.6, 0, 0], [-0.4, 1.55, 0]],
        }
        isotopes = ["31P", "19F", "19F"]
    else:
        quadrupole = eeqq2nqi(150e3, 0.2, 1.5, [0.3, 0.7, 0.1]).tolist()
        inter = {
            "coupling": {"matrix": [[quadrupole, None], [None, None]], "scalar": [[0, 20.0], [0, 0]]},
            "coordinates": [[0, 0, 0], [0.5, 0.3, 1.25]],
        }
        isotopes = ["11B", "19F"]
    inter.update(relaxation=["redfield"], rlx_keep="labframe", equilibrium="zero", tau_c=[[20e-12]])
    return spynach.create({"isotopes": isotopes, "magnet": field}, inter)


def test_hamiltonian_is_spynachs_lab_frame_hamiltonian():
    spin_system = small("PF2", 2.7e-6)
    levels = energy_levels(spin_system)
    hilbert = spynach.basis(spin_system, {"formalism": "zeeman-hilb", "approximation": "none"})
    H = spynach.hamiltonian(spynach.assume(hilbert, "labframe"))
    H = H.toarray() if hasattr(H, "toarray") else np.asarray(H)
    np.testing.assert_allclose(levels.hamiltonian, H.real, atol=1e-9 * np.abs(H).max())
    # the eigenvectors diagonalise it, level by level
    D = levels.vectors.T @ levels.hamiltonian @ levels.vectors
    energies = np.concatenate([[level.energy] * level.degeneracy for level in levels])
    np.testing.assert_allclose(D, np.diag(energies), atol=1e-9 * np.abs(energies).max())


def test_zero_field_levels_and_lines_of_pf6():
    levels = energy_levels(pf6(0.0))
    assert len(levels) == 32 and sorted({level.degeneracy for level in levels}) == [1, 5, 9]
    assert levels.labels[:3] == ["3,7/2,+7/2", "3,7/2,+5/2", "3,7/2,+3/2"]
    # E(F) = (J/2) (F(F+1) - K(K+1) - 3/4) for the phosphorus spin coupled to the fluorine spin K
    for level in levels:
        (K,) = level.K
        expected = 2 * np.pi * 710.0 / 2 * (level.F * (level.F + 1) - K * (K + 1) - 0.75)
        assert level.energy == pytest.approx(expected, abs=1e-6)
    lines = levels.lines("z")
    np.testing.assert_allclose(lines.frequencies, [1.5 * 710, 2.5 * 710, 3.5 * 710])
    # at 2.7 uT the K = 2 line splits into four, by M; F stays the label of the zero-field level
    ulf = energy_levels(pf6(2.7e-6))
    assert [level.F for level in ulf] == [level.F for level in levels]
    assert len([line for line in ulf.lines("z") if 1700 < line.frequency < 1850]) == 4


def test_adiabatic_transfer_reproduces_the_contributed_state():
    # user_ml/rho_adiabatic_drop.csv: the PF6- state after an adiabatic drop to 2.7 uT, in the sphten-liouv basis
    csv = (ROOT / "user_ml" / "rho_adiabatic_drop.csv").read_text().split()
    target = np.array([complex(x.replace("i", "j")) for x in csv])
    levels = energy_levels(pf6(2.7e-6))
    populations = levels.adiabatic(from_field=1.0)
    rho = sum(p * levels.vectors[:, lv.states] @ levels.vectors[:, lv.states].T for p, lv in zip(populations, levels))
    spin_system = spynach.basis(pf6(2.7e-6), LIOUV)
    model = np.zeros(len(target), dtype=complex)
    for k, row in enumerate(spin_system.basis.basis):
        ranks = [int(np.sqrt(index)) for index in row]
        projections = [rank * rank + rank - int(index) for rank, index in zip(ranks, row)]
        if sum(projections) == 0:
            T = reduce(np.kron, [irr_sph_ten(2, r)[r - p].toarray() for r, p in zip(ranks, projections)])
            model[k] = np.vdot(T, rho) / np.linalg.norm(T)
    deviation = target.copy()
    deviation[0] = 0
    scale = np.vdot(model, deviation) / np.vdot(model, model)
    assert np.linalg.norm(deviation - scale * model) < 0.01 * np.linalg.norm(deviation)
    # the populations are the high-temperature deviations: their mean is zero
    d = np.array([level.degeneracy for level in levels])
    assert abs(np.sum(populations * d)) < 1e-12 * np.abs(populations).max()


@pytest.mark.parametrize(("kind", "field"), [("PF2", 2.7e-6), ("PF2", 0.0), ("BF", 3e-6), ("BF", 50e-6)])
def test_populations_and_line_widths_match_spynach_redfield(kind, field):
    spin_system = small(kind, field)
    levels = energy_levels(spin_system)
    liouville = spynach.basis(spin_system, LIOUV)
    R = spynach.relaxation(liouville)
    R = R.toarray() if hasattr(R, "toarray") else np.asarray(R)
    for basis in (levels, levels.lines("z"), levels.lines("x")):
        pathways = relaxation_pathways(spin_system, basis=basis)
        V = pathways.states(liouville)
        reference = -(V.conj() @ R @ V.T).real
        gamma = pathways.rate_matrix()
        if basis is not levels:  # only the diagonal (the widths) is secular between lines
            gamma, reference = np.diag(gamma), np.diag(reference)
        assert np.abs(gamma - reference).max() < 1e-4 * np.abs(gamma).max()
    # populations are conserved: Gamma annihilates the identity, sum_k sqrt(d_k) P_k
    gamma = relaxation_pathways(spin_system, basis=levels).rate_matrix()
    d = np.array([level.degeneracy for level in levels])
    assert np.abs(gamma @ np.sqrt(d)).max() < 1e-9 * np.abs(gamma).max()


def test_level_pathways_have_expressions_in_J0():
    levels = energy_levels(pf6(2.7e-6))
    pathways = relaxation_pathways(pf6(2.7e-6), basis=levels, mechanisms=("DD",), resolve="interaction")
    assert pathways.classes == ["DD(FF)", "DD(PF)", "DD(FF)/DD(FF')", "DD(FF)/DD(PF)", "DD(PF)/DD(PF')"]
    link = pathways.links()[0]
    text = str(link.expressions["DD(PF)"])
    assert "J(0)" in text and "b_PF" in text and "omega" not in text
    widths = relaxation_pathways(pf6(2.7e-6), basis=levels.lines("z"), mechanisms=("DD",))
    assert np.all(np.diag(widths.rate_matrix()) > 0) and not widths.links()


def test_eigenbasis_arguments_are_checked():
    levels = energy_levels(pf6(2.7e-6))
    with pytest.raises(ValueError, match="another spin system or another field"):
        relaxation_pathways(pf6(5e-6), basis=levels)
    with pytest.raises(TypeError, match="basis must be"):
        relaxation_pathways(pf6(2.7e-6), basis=[1, 2, 3])
    with pytest.raises(ValueError, match="not far below 1/tau_c"):  # 19F at 9.4 T with a nanosecond tumbling
        high = small("PF2", 9.4)
        relaxation_pathways(high, basis=energy_levels(high), tau_c=1e-9)
    with pytest.raises(ValueError, match="detection"):
        levels.lines("y")


@pytest.mark.skipif(importlib.util.find_spec("plotly") is None, reason="plotly not installed")
def test_level_chord_diagram_shows_energies_on_hover():
    from spynach.plotting import chord_layout

    levels = energy_levels(small("BF", 3e-6))
    layout = chord_layout(relaxation_pathways(small("BF", 3e-6), basis=levels))
    assert len(layout["arcs"]) == len(levels) and layout["secular"] == "levels"
    assert "E/2π" in layout["arcs"][0]["hover"] and "coherence order" not in layout["arcs"][0]["hover"]
