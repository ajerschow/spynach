"""Fixture-free tests of relaxation: Redfield rates against Solomon theory, T1/T2 and damping models,
retention policies, thermal equilibrium and thermalisation, and NOE trajectories."""

import numpy as np
import pytest
import scipy.linalg as sla
import scipy.sparse as sps

import spynach
from spynach.oracle import assert_matches

LIOUV = {"formalism": "sphten-liouv", "approximation": "none"}
HILB = {"formalism": "zeeman-hilb", "approximation": "none"}


def dense(x):
    return x.toarray() if sps.issparse(x) else np.asarray(x)


def redfield_pair(isotopes, r, tau_c, keep="kite", equilibrium="dibari", **extra):
    inter = {
        "zeeman": {"scalar": [0.0, 0.0]},
        "coordinates": [[0, 0, 0], [0, 0, r]],
        "relaxation": ["redfield"],
        "equilibrium": equilibrium,
        "rlx_keep": keep,
        "temperature": 298,
        "tau_c": [[tau_c]],
        **extra,
    }
    spin_system = spynach.create({"isotopes": isotopes, "magnet": 14.1}, inter)
    return spynach.basis(spin_system, LIOUV)


def rate(R, bra, ket):
    """-<bra|R|ket> / (|bra||ket|): the relaxation rate between normalised states."""
    return -np.vdot(bra, dense(R) @ ket).real / np.sqrt(np.vdot(bra, bra).real * np.vdot(ket, ket).real)


def solomon(spin_system, r, tau_c):
    b = (
        spin_system.gammas[0]
        * spin_system.gammas[1]
        * spin_system.tols.hbar
        * spin_system.tols.mu0
        / (4 * np.pi * r**3)
    )

    def J(w):
        return tau_c / (1 + (w * tau_c) ** 2)

    wI, wS = abs(spin_system.basefrqs[0]), abs(spin_system.basefrqs[1])
    return {
        "rho_I": (b**2 / 10) * (J(wI - wS) + 3 * J(wI) + 6 * J(wI + wS)),
        "rho_S": (b**2 / 10) * (J(wI - wS) + 3 * J(wS) + 6 * J(wI + wS)),
        "sigma": (b**2 / 10) * (6 * J(wI + wS) - J(wI - wS)),
    }


@pytest.mark.parametrize("isotopes, r, tau_c", [(["1H", "13C"], 1.03e-10, 100e-12), (["1H", "1H"], 2.0e-10, 1e-9)])
def test_redfield_dipolar_rates_match_solomon(isotopes, r, tau_c):
    """The Redfield integral is truncated at rlx_integration = 1e-4, hence the tolerance."""
    spin_system_liouv = redfield_pair(isotopes, r * 1e10, tau_c, equilibrium="zero")
    R = spynach.relaxation(spin_system_liouv)
    Lz1, Lz2 = spynach.state(spin_system_liouv, ["Lz"], [0]), spynach.state(spin_system_liouv, ["Lz"], [1])
    ref = solomon(spin_system_liouv, r, tau_c)
    assert np.isclose(rate(R, Lz1, Lz1), ref["rho_I"], rtol=1e-4)
    assert np.isclose(rate(R, Lz2, Lz2), ref["rho_S"], rtol=1e-4)
    assert np.isclose(rate(R, Lz1, Lz2), ref["sigma"], rtol=1e-4)
    # 'kite' keeps longitudinal cross-relaxation and the diagonal only; R is real (dfs ignored)
    Rd = dense(R)
    assert np.allclose(Rd.imag, 0) and Rd[0, 0] == 0
    _, projs = spynach.operators.lin2lm(spin_system_liouv.basis.basis)
    transverse = np.abs(projs).sum(axis=1) != 0
    off = Rd.copy()
    np.fill_diagonal(off, 0)
    assert np.allclose(off[transverse], 0) and np.allclose(off[:, transverse], 0)


def test_retention_policies_and_dfs():
    spin_system_labframe = redfield_pair(
        ["1H", "13C"], 1.03, 100e-12, keep="labframe", equilibrium="zero", rlx_dfs="keep"
    )
    R_lab = dense(spynach.relaxation(spin_system_labframe))
    assert np.abs(R_lab.imag).max() > 0  # dynamic frequency shifts kept
    R_sec = dense(spynach.relaxation(redfield_pair(["1H", "13C"], 1.03, 100e-12, keep="secular", equilibrium="zero")))
    _, projs = spynach.operators.lin2lm(spin_system_labframe.basis.basis)
    frequencies = projs @ spin_system_labframe.basefrqs
    rows, cols = np.nonzero(R_sec)
    assert np.all(frequencies[rows] == frequencies[cols])
    R_diag = dense(spynach.relaxation(redfield_pair(["1H", "13C"], 1.03, 100e-12, keep="diagonal", equilibrium="zero")))
    assert np.allclose(R_diag, np.diag(np.diag(R_diag))) and R_diag[0, 0] == 0
    np.testing.assert_allclose(np.diag(R_diag), np.diag(R_lab.real), atol=1e-9)


def test_secular_policy_groups_frequencies_equal_to_within_rounding():
    # 13CF3 at a field where the frequency of a state such as C+F1+F2- (w_C + w_F - w_F) differs from
    # that of C+ in the last bits: exact equality, as in Spinach 2.13, drops secular elements
    def cf3(keep):
        corners = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1]]) * 0.77
        inter = {
            "zeeman": {"matrix": [(np.diag([-10.0, -10, 20]) * k).tolist() for k in (1, 2, 3, 4)]},
            "coordinates": [[0, 0, 0], *corners.tolist()],
            "relaxation": ["redfield"],
            "rlx_keep": keep,
            "equilibrium": "zero",
            "tau_c": [[1e-9]],
        }
        spin_system = spynach.create({"isotopes": ["13C", "19F", "19F", "19F"], "magnet": 5 + 7 * 23.2 / 12}, inter)
        return spynach.basis(spin_system, LIOUV)

    spin_system = cf3("labframe")
    R_lab = dense(spynach.relaxation(spin_system))
    R_sec = dense(spynach.relaxation(cf3("secular")))
    _, projs = spynach.operators.lin2lm(spin_system.basis.basis)
    coherence = np.stack([projs[:, 0], projs[:, 1:].sum(axis=1)], axis=1)
    same = (coherence[:, None, :] == coherence[None, :, :]).all(axis=2)
    frequencies = projs @ spin_system.basefrqs
    assert ((frequencies[:, None] != frequencies[None, :]) & same & (R_lab != 0)).any()  # rounding splits groups
    np.testing.assert_array_equal(R_sec, np.where(same, R_lab, 0))


def test_t1_t2_model_and_damping():
    spin_system = spynach.create(
        {"isotopes": ["1H", "13C"], "magnet": 14.1},
        {
            "relaxation": ["t1_t2"],
            "equilibrium": "zero",
            "rlx_keep": "diagonal",
            "r1_rates": [1.0, 2.0],
            "r2_rates": [5.0, 7.0],
        },
    )
    spin_system_liouv = spynach.basis(spin_system, LIOUV)
    R = dense(spynach.relaxation(spin_system_liouv))
    assert np.allclose(R, np.diag(np.diag(R)))

    def row(spec):
        return np.flatnonzero((spin_system_liouv.basis.basis == spec).all(axis=1))[0]

    assert R[row([2, 0]), row([2, 0])] == -1.0 and R[row([0, 2]), row([0, 2])] == -2.0
    assert R[row([1, 0]), row([1, 0])] == -5.0 and R[row([1, 3]), row([1, 3])] == -12.0
    assert R[row([2, 2]), row([2, 2])] == -3.0 and R[0, 0] == 0
    # Anisotropic rates need an orientation; along z the zz element applies
    spin_system_aniso = spynach.create(
        {"isotopes": ["1H"], "magnet": 14.1},
        {
            "relaxation": ["t1_t2"],
            "equilibrium": "zero",
            "rlx_keep": "diagonal",
            "r1_rates": [np.diag([1.0, 2.0, 3.0]).tolist()],
            "r2_rates": [4.0],
        },
    )
    spin_system_aniso_liouv = spynach.basis(spin_system_aniso, LIOUV)
    with pytest.raises(ValueError, match="Euler angles"):
        spynach.relaxation(spin_system_aniso_liouv)
    assert dense(spynach.relaxation(spin_system_aniso_liouv, [0, 0, 0]))[2, 2] == -3.0
    assert np.isclose(dense(spynach.relaxation(spin_system_aniso_liouv, [0, np.pi / 2, 0]))[2, 2], -1.0)
    # Damping: every state but the unit state in Liouville space; everything in Hilbert space
    spin_system_damp = spynach.create(
        {"isotopes": ["1H", "1H"], "magnet": 14.1},
        {"relaxation": ["damp"], "equilibrium": "zero", "rlx_keep": "diagonal", "damp_rate": 5.0},
    )
    R = dense(spynach.relaxation(spynach.basis(spin_system_damp, LIOUV)))
    assert R[0, 0] == 0 and np.allclose(np.diag(R)[1:], -5.0)
    R = dense(spynach.relaxation(spynach.basis(spin_system_damp, HILB)))
    assert np.allclose(R, -2.5 * np.eye(4))


def test_equilibrium_and_thermalisation():
    spin_system = spynach.create(
        {"isotopes": ["1H", "13C"], "magnet": 14.1, "tols": {"small_matrix": 2}},
        {"relaxation": ["damp"], "equilibrium": "zero", "rlx_keep": "diagonal", "damp_rate": 5.0, "temperature": 40},
    )
    spin_system_liouv, spin_system_hilb = spynach.basis(spin_system, LIOUV), spynach.basis(spin_system, HILB)
    rho_eq = spynach.equilibrium(spin_system_liouv)
    beta = spin_system.tols.hbar / (spin_system.tols.kbol * 40)
    # High-temperature limit: polarisation of each spin is -tanh(beta*omega/2)*... = beta*omega/2 to first order
    Lz1 = spynach.state(spin_system_liouv, ["Lz"], [0])
    polarisation = np.vdot(Lz1, rho_eq).real / np.vdot(Lz1, Lz1).real / 2  # in units of the 'exact' Lz state
    assert np.isclose(polarisation, -beta * spin_system.basefrqs[0] / 2, rtol=1e-3)
    assert rho_eq[0] == 1
    rho_h = dense(spynach.equilibrium(spin_system_hilb))
    assert np.isclose(np.trace(rho_h), 1)
    H_lab = dense(spynach.hamiltonian(spynach.assume(spin_system_hilb, "labframe")))
    np.testing.assert_allclose(rho_h, sla.expm(-beta * H_lab) / np.trace(sla.expm(-beta * H_lab)), rtol=0, atol=1e-10)
    # Both thermalisation methods make rho_eq stationary (fundamentals/state_tests/thermal_equilibrium_4.m)
    R = spynach.relaxation(spin_system_liouv)
    H_left = spynach.hamiltonian(spynach.assume(spin_system_liouv, "labframe"), "left")
    for method, kwargs in (("dibari", {"H_left": H_left, "temperature": 40.0}), ("IME", {"rho_eq": rho_eq})):
        Rt = spynach.thermalize(spin_system_liouv, R, method, **kwargs)
        assert np.linalg.norm(dense(Rt) @ rho_eq) < 1e-9, method
    with pytest.raises(ValueError, match="thermalized already"):
        spynach.thermalize(
            spin_system_liouv, spynach.thermalize(spin_system_liouv, R, "IME", rho_eq=rho_eq), "IME", rho_eq=rho_eq
        )


def test_transient_noe_follows_the_solomon_equations():
    """examples/nmr_liquids/noe_two_spin_het.m: invert the proton and watch both spins recover."""
    r, tau_c = 1.03e-10, 100e-12
    spin_system_liouv = redfield_pair(["1H", "13C"], 1.03, tau_c)
    R = spynach.relaxation(spin_system_liouv)
    rho_eq = spynach.equilibrium(
        spin_system_liouv, spynach.hamiltonian(spynach.assume(spin_system_liouv, "labframe"), "left")
    )
    Lz1, Lz2 = spynach.state(spin_system_liouv, ["Lz"], [0]), spynach.state(spin_system_liouv, ["Lz"], [1])
    rho = rho_eq - 2 * Lz1 * np.vdot(Lz1, rho_eq) / np.vdot(Lz1, Lz1).real
    coil = np.stack([Lz1, Lz2], axis=1)
    dt, nsteps = 1e-2, 400
    answer = spynach.evolution(spin_system_liouv, 1j * R, rho, dt, nsteps, "multichannel", coil=coil)
    assert answer.shape == (2, nsteps + 1)
    # Solomon equations for the deviations from equilibrium
    ref = solomon(spin_system_liouv, r, tau_c)
    M = -np.array([[ref["rho_I"], ref["sigma"]], [ref["sigma"], ref["rho_S"]]])
    eq = np.array([np.vdot(Lz1, rho_eq), np.vdot(Lz2, rho_eq)]).real
    dev0 = np.array([-2 * eq[0], 0.0])
    t = dt * np.arange(nsteps + 1)
    expected = np.array([eq + sla.expm(M * tn) @ dev0 for tn in t]).T
    np.testing.assert_allclose(answer.real, expected, rtol=0, atol=2e-4 * abs(eq[0]))
    assert np.allclose(answer[:, -1].real, eq, rtol=0.1)  # relaxed back after 4 s


def test_relaxation_inputs_are_validated():
    base = {"isotopes": ["1H", "13C"], "magnet": 14.1}
    with pytest.raises(ValueError, match="rlx_keep"):
        spynach.create(base, {"relaxation": ["damp"], "equilibrium": "zero", "damp_rate": 1.0})
    with pytest.raises(ValueError, match="equilibrium"):
        spynach.create(base, {"relaxation": ["damp"], "rlx_keep": "diagonal", "damp_rate": 1.0})
    with pytest.raises(ValueError, match="tau_c"):
        spynach.create(base, {"relaxation": ["redfield"], "rlx_keep": "kite", "equilibrium": "zero"})
    with pytest.raises(ValueError, match="requires a relaxation theory"):
        spynach.create(base, {"tau_c": [[1e-9]]})
    with pytest.raises(NotImplementedError, match="lindblad"):
        spynach.create(base, {"relaxation": ["lindblad"], "rlx_keep": "kite", "equilibrium": "zero"})
    with pytest.raises(ValueError, match="validity condition"):
        spynach.relaxation(redfield_pair(["1H", "1H"], 1.0, 1e-5, equilibrium="zero"))  # R2 tau_c >> 1
    spin_system_hilb = spynach.basis(
        spynach.create(
            base,
            {
                "coordinates": [[0, 0, 0], [0, 0, 1.03]],
                "relaxation": ["redfield"],
                "rlx_keep": "kite",
                "equilibrium": "zero",
                "tau_c": [[1e-10]],
            },
        ),
        HILB,
    )
    with pytest.raises(NotImplementedError, match="sphten-liouv"):
        spynach.relaxation(spin_system_hilb)
    assert spynach.relaxation(spynach.basis(spynach.create(base, {}), LIOUV)).count_nonzero() == 0


def quadrupolar_pair(theories, **extra):
    """1H-14N pair with a 14N quadrupolar tensor and a 50 Hz scalar coupling; 14N relaxes fast."""
    inter = {
        "zeeman": {"scalar": [0.0, 0.0]},
        "coordinates": [[0, 0, 0], [0, 0, 1.0]],
        "coupling": {
            "scalar": [[0, 50.0], [0, 0]],
            "eigs": [[None, None], [None, [1e6, 1e6, -2e6]]],
            "euler": [[None, None], [None, [0, 0, 0]]],
        },
        "relaxation": theories,
        "rlx_keep": "secular",
        "equilibrium": "zero",
        "tau_c": [[1e-9]],
        **extra,
    }
    return spynach.basis(spynach.create({"isotopes": ["1H", "14N"], "magnet": 11.75}, inter), LIOUV)


def test_srsk_adds_abragam_rates_to_the_coupled_spin():
    spin_system_redfield = quadrupolar_pair(["redfield"])
    spin_system_srsk = quadrupolar_pair(["redfield", "SRSK"], srsk_sources=[1])
    R_base, R = spynach.relaxation(spin_system_redfield), spynach.relaxation(spin_system_srsk)
    Lz, Lp = spynach.state(spin_system_redfield, ["Lz"], [0]), spynach.state(spin_system_redfield, ["L+"], [0])
    Nz, Np = spynach.state(spin_system_redfield, ["Lz"], [1]), spynach.state(spin_system_redfield, ["L+"], [1])
    T1k, T2k = 1 / rate(R_base, Nz, Nz), 1 / rate(R_base, Np, Np)
    A, Sk, dw = 2 * np.pi * 50.0, 1.0, spin_system_redfield.basefrqs[0] - spin_system_redfield.basefrqs[1]
    R1add = (2 / 3) * A**2 * Sk * (Sk + 1) * T2k / (1 + dw**2 * T2k**2)
    R2add = (1 / 3) * A**2 * Sk * (Sk + 1) * (T1k + T2k / (1 + dw**2 * T2k**2))
    assert np.isclose(rate(R, Lz, Lz), rate(R_base, Lz, Lz) + R1add, rtol=1e-9)
    assert np.isclose(rate(R, Lp, Lp), rate(R_base, Lp, Lp) + R2add, rtol=1e-9)
    assert np.isclose(rate(R, Nz, Nz), rate(R_base, Nz, Nz))  # the source spin itself is untouched
    # too slow a source: the quadrupolar coupling a hundred times smaller
    spin_system_slow = quadrupolar_pair(["redfield", "SRSK"], srsk_sources=[1])
    spin_system_slow = spynach.create(
        {"isotopes": ["1H", "14N"], "magnet": 11.75},
        {
            "zeeman": {"scalar": [0.0, 0.0]},
            "coordinates": [[0, 0, 0], [0, 0, 1.0]],
            "coupling": {
                "scalar": [[0, 50.0], [0, 0]],
                "eigs": [[None, None], [None, [1e4, 1e4, -2e4]]],
                "euler": [[None, None], [None, [0, 0, 0]]],
            },
            "relaxation": ["redfield", "SRSK"],
            "srsk_sources": [1],
            "rlx_keep": "secular",
            "equilibrium": "zero",
            "tau_c": [[1e-9]],
        },
    )
    with pytest.raises(ValueError, match="source spin relaxation is too slow"):
        spynach.relaxation(spynach.basis(spin_system_slow, LIOUV))


def test_srfk_is_the_scalar_redfield_integral():
    from spynach.relaxation import expmint

    spin_system = spynach.create(
        {"isotopes": ["1H", "1H"], "magnet": 14.1},
        {
            "zeeman": {"scalar": [0.0, 2.0]},
            "relaxation": ["SRFK"],
            "rlx_keep": "labframe",
            "equilibrium": "zero",
            "srfk_tau_c": [[1.0, 1e-3]],
            "srfk_mdepth": [[None, 15.0], [None, None]],
        },
    )
    spin_system = spynach.basis(spin_system, LIOUV)
    R = dense(spynach.relaxation(spin_system))
    assert R.shape == (16, 16) and np.abs(R.imag).max() == 0 and R[0, 0] == 0
    assert np.all(np.diag(R) <= 0) and np.diag(R).min() < -1
    # the explicit formula of rlx_scalar.m
    H0 = spynach.hamiltonian(spynach.assume(spin_system, "labframe"), "comm")
    spin_system_coupled = spynach.create(
        {"isotopes": ["1H", "1H"], "magnet": 14.1}, {"coupling": {"scalar": [[0, 15.0], [0, 0]]}}
    )
    H1 = spynach.hamiltonian(spynach.assume(spynach.basis(spin_system_coupled, LIOUV), "labframe", "couplings"), "comm")
    H0, H1 = dense((H0 + H0.conj().T) / 2), dense((H1 + H1.conj().T) / 2)
    from spynach.utilities import clean_up

    tau, upper = 1e-3, 2 * 1e-3 * np.log(1 / spin_system.tols.rlx_integration)
    H0c = dense(clean_up(spin_system.tols, H0, 1e-2 / upper))  # the coarse grid of rlx_scalar.m
    expected = -H1 @ dense(expmint(spin_system, H0c, H1.conj().T, H0c + (1j / tau) * np.eye(16), upper))
    assert_matches(R, expected.real, rtol=1e-9, atol=1e-12, name="SRFK")
    # validity: a modulation slower than the relaxation it causes is refused
    spin_system_too_slow = spynach.create(
        {"isotopes": ["1H", "1H"], "magnet": 14.1},
        {
            "zeeman": {"scalar": [0.0, 2.0]},
            "relaxation": ["SRFK"],
            "rlx_keep": "labframe",
            "equilibrium": "zero",
            "srfk_tau_c": [[1.0, 1.0]],
            "srfk_mdepth": [[None, 150.0], [None, None]],
        },
    )
    with pytest.raises(ValueError, match="srfk_tau_c validity"):
        spynach.relaxation(spynach.basis(spin_system_too_slow, LIOUV))


def test_scalar_relaxation_inputs_are_validated():
    sys_ = {"isotopes": ["1H", "14N"], "magnet": 11.75}
    base = {"zeeman": {"scalar": [0.0, 0.0]}, "rlx_keep": "secular", "equilibrium": "zero"}
    with pytest.raises(ValueError, match="srfk_tau_c"):
        spynach.create(sys_, {**base, "relaxation": ["SRFK"], "srfk_mdepth": [[None, 1.0], [None, None]]})
    with pytest.raises(ValueError, match="srfk_mdepth"):
        spynach.create(sys_, {**base, "relaxation": ["SRFK"], "srfk_tau_c": [[1.0, 1e-3]]})
    with pytest.raises(ValueError, match="2-element"):
        spynach.create(
            sys_, {**base, "relaxation": ["SRFK"], "srfk_tau_c": [[1.0]], "srfk_mdepth": [[None, 1.0], [None, None]]}
        )
    with pytest.raises(ValueError, match="non-negative"):
        spynach.create(
            sys_,
            {**base, "relaxation": ["SRFK"], "srfk_tau_c": [[1.0, -1e-3]], "srfk_mdepth": [[None, 1.0], [None, None]]},
        )
    with pytest.raises(ValueError, match="requires SRFK"):
        spynach.create(sys_, {**base, "relaxation": ["damp"], "damp_rate": 1.0, "srfk_tau_c": [[1.0, 1e-3]]})
    with pytest.raises(ValueError, match="srsk_sources"):
        spynach.create(sys_, {**base, "relaxation": ["redfield", "SRSK"], "tau_c": [[1e-9]]})
    with pytest.raises(ValueError, match="between 0 and 1"):
        spynach.create(sys_, {**base, "relaxation": ["redfield", "SRSK"], "tau_c": [[1e-9]], "srsk_sources": [2]})
    with pytest.raises(ValueError, match="requires SRSK"):
        spynach.create(sys_, {**base, "relaxation": ["redfield"], "tau_c": [[1e-9]], "srsk_sources": [1]})
    spin_system = spynach.create(
        sys_, {**base, "relaxation": ["redfield", "SRSK"], "tau_c": [[1e-9]], "srsk_sources": [1]}
    )
    with pytest.raises(NotImplementedError, match="sphten-liouv"):
        spynach.relaxation(spynach.basis(spin_system, HILB))
