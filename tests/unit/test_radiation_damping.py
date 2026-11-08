"""Radiation damping (spynach.probe): the Bloembergen-Pound closed form, the classical Bloch-Maxwell limit of an
isochromat ensemble, the invariants of the unitary feedback, the linear regime (damping rate and frequency pulling
of a detuned probe), basis restriction, Hilbert against Liouville space, and the probe conversions."""

import numpy as np
import pytest
from scipy.integrate import solve_ivp

import spynach
from spynach.oracle import assert_matches
from spynach.probe import line_weights, radiation_damping, rd_phase, rd_rate
from spynach.propagation import iserstep

HILB = {"formalism": "zeeman-hilb", "approximation": "none"}
LIOUV = {"formalism": "sphten-liouv", "approximation": "none"}


def one_spin(bas, **inter):
    return spynach.basis(spynach.create({"isotopes": ["1H"], "magnet": 14.1}, inter), bas)


def ax_system(bas):
    spin_system = spynach.create(
        {"isotopes": ["1H", "13C"], "magnet": 14.1},
        {"zeeman": {"scalar": [4.7, 40.0]}, "coupling": {"scalar": [[0, 140.0], [0, 0]]}},
    )
    return spynach.assume(spynach.basis(spin_system, bas), "nmr")


def tilted(spin_system, theta):
    """The Lz state of the proton rotated by theta about y (a pulse of flip angle theta)."""
    return spynach.step(
        spin_system, spynach.operator(spin_system, "Ly", "1H"), spynach.state(spin_system, "Lz", "1H"), theta
    )


def zero_generator(spin_system):
    n = spin_system.basis.nstates if spin_system.basis.formalism == "sphten-liouv" else spin_system.basis.nstates
    return np.zeros((n, n), dtype=complex)


def run(spin_system, L, rho, dt, nsteps, method="LG4"):
    out = []
    for n in range(nsteps):
        rho = iserstep(spin_system, L, n * dt, method, rho, dt)
        out.append(L.magnetisation(rho))
    return np.array(out)


@pytest.mark.parametrize("bas,method", [(LIOUV, "LG4"), (HILB, "RKMK4")], ids=["liouv", "hilb"])
def test_bloembergen_pound_closed_form(bas, method):
    """One spin, no relaxation, tuned probe: m_z = tanh(rate (t - t0)), m_perp = sech(rate (t - t0)), and the
    tilt angle follows tan(theta/2) = tan(theta0/2) exp(-rate t)."""
    spin_system = one_spin(bas)
    rate, theta0, dt, nsteps = 50.0, np.pi * 170 / 180, 5e-5, 1600
    rd = radiation_damping(spin_system, zero_generator(spin_system), "1H", rate)
    assert_matches(
        np.array(rd.magnetisation(tilted(spin_system, theta0))),
        [np.sin(theta0), 0, np.cos(theta0)],
        rtol=1e-12,
        atol=1e-12,
    )
    m = run(spin_system, rd, tilted(spin_system, theta0), dt, nsteps, method)
    t = np.arange(1, nsteps + 1) * dt
    t0 = -np.arctanh(np.cos(theta0)) / rate
    assert np.abs(m[:, 2] - np.tanh(rate * (t - t0))).max() < 1e-6
    assert np.abs(np.hypot(m[:, 0], m[:, 1]) - 1 / np.cosh(rate * (t - t0))).max() < 1e-6
    theta = np.arctan2(np.hypot(m[:, 0], m[:, 1]), m[:, 2])
    assert np.abs(np.tan(theta / 2) - np.tan(theta0 / 2) * np.exp(-rate * t)).max() < 1e-5
    assert np.abs(m[:, 1]).max() < 1e-9  # the feedback field rotates about the axis of the magnetisation


def test_bloch_maxwell_limit_of_an_isochromat_ensemble():
    """Uncoupled isochromats with T1/T2 relaxation, offsets and a detuned probe, sharing one coil: the spin system
    equals the classical Bloch-Maxwell equations integrated adaptively."""
    r1, r2 = 2.0, 6.0
    spin_system = one_spin(
        LIOUV, relaxation=["t1_t2"], r1_rates=[r1], r2_rates=[r2], rlx_keep="diagonal", equilibrium="zero"
    )
    offsets = np.array([-40.0, 0.0, 25.0])
    weights = line_weights(offsets, 60.0)
    rate, phase, theta0 = 80.0, 0.4, np.pi * 160 / 180
    L0 = 1j * spynach.relaxation(spin_system)
    rd = radiation_damping(spin_system, L0, "1H", rate, phase=phase, offsets=offsets, weights=weights)
    rho = rd.stack(tilted(spin_system, theta0))
    dt, nsteps = 2e-4, 300
    quantum = run(spin_system, rd, rho, dt, nsteps, "RKMK-DP8")

    def classical(t, m):
        m = m.reshape(3, -1)
        mplus = weights @ (m[0] + 1j * m[1])
        w = -1j * rate * np.cos(phase) * np.exp(1j * phase) * mplus
        wx, wy = w.real, w.imag
        dm = np.empty_like(m)
        for k, offset in enumerate(offsets):
            wz = 2 * np.pi * offset
            dm[0, k] = wy * m[2, k] - wz * m[1, k] - r2 * m[0, k]
            dm[1, k] = wz * m[0, k] - wx * m[2, k] - r2 * m[1, k]
            dm[2, k] = wx * m[1, k] - wy * m[0, k] - r1 * m[2, k]
        return dm.reshape(-1)

    m0 = np.tile([np.sin(theta0), 0.0, np.cos(theta0)], (len(offsets), 1)).T.reshape(-1)
    times = np.arange(1, nsteps + 1) * dt
    sol = solve_ivp(classical, (0, times[-1]), m0, t_eval=times, method="DOP853", rtol=1e-11, atol=1e-13)
    y = sol.y.reshape(3, len(offsets), len(times))
    expected = np.array([weights @ y[i] for i in range(3)]).T
    assert np.abs(quantum - expected).max() < 1e-8
    assert rd.nisochromats == 3 and rd.columns(rho).shape == (spin_system.basis.nstates, 3)


def test_feedback_is_unitary():
    """Without relaxation the feedback keeps the purity of every isochromat and, in Hilbert space, the trace and
    Hermiticity of the density matrix."""
    spin_system = ax_system(LIOUV)
    H = spynach.hamiltonian(spin_system)
    rd = radiation_damping(spin_system, H, "1H", 100.0, phase=0.3, offsets=[-10.0, 15.0])
    rho = rd.stack(tilted(spin_system, 2.8))
    purity0 = [np.vdot(c, c).real for c in rd.columns(rho).T]
    for n in range(50):
        rho = iserstep(spin_system, rd, n * 1e-4, "LG4", rho, 1e-4)
    purity = [np.vdot(c, c).real for c in rd.columns(rho).T]
    assert np.abs(np.array(purity) - purity0).max() < 1e-10
    hilb = ax_system(HILB)
    rd_h = radiation_damping(hilb, spynach.hamiltonian(hilb), "1H", 100.0, phase=0.3)
    rho_h = tilted(hilb, 2.8)
    rho_h = rho_h.toarray() if hasattr(rho_h, "toarray") else rho_h
    trace0 = np.trace(rho_h)
    for n in range(50):
        rho_h = iserstep(hilb, rd_h, n * 1e-4, "RKMK4", rho_h, 1e-4)
    assert abs(np.trace(rho_h) - trace0) < 1e-12
    assert np.abs(rho_h - rho_h.conj().T).max() < 1e-12


def test_linear_regime_damping_rate_and_frequency_pulling():
    """A small tilt decays at rate cos^2(phase) and is pulled by rate cos(phase) sin(phase) / 2 pi, on top of T2."""
    r2 = 3.0
    spin_system = one_spin(
        LIOUV, relaxation=["t1_t2"], r1_rates=[0.0], r2_rates=[r2], rlx_keep="diagonal", equilibrium="zero"
    )
    rate, phase = 40.0, 0.5
    rd = radiation_damping(spin_system, 1j * spynach.relaxation(spin_system), "1H", rate, phase=phase)
    dt, nsteps = 1e-4, 400
    m = run(spin_system, rd, tilted(spin_system, 0.01), dt, nsteps)
    t = np.arange(1, nsteps + 1) * dt
    mplus = m[:, 0] + 1j * m[:, 1]
    decay = -np.polyfit(t, np.log(np.abs(mplus)), 1)[0]
    pulling = np.polyfit(t, np.unwrap(np.angle(mplus)), 1)[0]
    assert abs(decay - (r2 + rate * np.cos(phase) ** 2)) < 0.05
    assert abs(pulling - (-rate * np.cos(phase) * np.sin(phase))) < 0.05


def test_basis_restriction_does_not_change_the_feedback():
    """A three-proton chain in IK-1 (two-spin correlations) and in the complete basis agree while the damping
    acts: the feedback operators are single-spin."""
    sys_ = {"isotopes": ["1H", "1H", "1H"], "magnet": 14.1}
    inter = {"zeeman": {"scalar": [1.0, 1.2, 1.4]}, "coupling": {"scalar": [[0, 8.0, 0], [0, 0, 8.0], [0, 0, 0]]}}
    traces = []
    for bas in (
        LIOUV,
        {
            "formalism": "sphten-liouv",
            "approximation": "IK-1",
            "inter_level": 2,
            "prox_level": 1,
            "connectivity": "scalar_couplings",
        },
    ):
        spin_system = spynach.assume(spynach.basis(spynach.create(sys_, inter), bas), "nmr")
        H = spynach.frqoffset(spin_system, spynach.hamiltonian(spin_system), {"spins": ["1H"], "offset": [720.0]})
        rd = radiation_damping(spin_system, H, "1H", 60.0)
        traces.append(run(spin_system, rd, tilted(spin_system, 2.5), 1e-4, 100))
    assert traces[1].shape == traces[0].shape
    assert np.abs(traces[1] - traces[0]).max() < 1e-4  # IK-1 drops the three-spin orders, a small effect here


def test_hilbert_and_liouville_space_agree():
    """The same AX system under the same feedback in both formalisms, with the same (eighth-order) integrator."""
    out = []
    for bas in (LIOUV, HILB):
        spin_system = ax_system(bas)
        rd = radiation_damping(spin_system, spynach.hamiltonian(spin_system), "1H", 120.0, phase=0.2)
        out.append(run(spin_system, rd, tilted(spin_system, 2.9), 5e-5, 200, "RKMK-DP8"))
    assert np.abs(out[0] - out[1]).max() < 1e-9


def test_probe_conversions_and_input_checks():
    spin_system = one_spin(LIOUV)
    rate = rd_rate(spin_system, "1H", eta=0.5, Q=200, concentration=110.0)
    assert 300 < rate < 3000  # water at 14.1 T: T_rd of a millisecond or so
    assert rd_rate(spin_system, "1H", eta=0.5, Q=200, concentration=110.0, temperature=149.0) == pytest.approx(2 * rate)
    phase = rd_phase(spin_system, "1H", Q=200, detuning=1e5)
    nu0 = spynach.isotopes.spin("1H")[0] * 14.1 / (2 * np.pi)
    assert phase == pytest.approx(np.arctan(2 * 200 * 1e5 / nu0))
    assert rd_phase(spin_system, "1H", Q=200, detuning=0.0) == 0.0
    weights = line_weights([-1.0, 0.0, 1.0], 2.0)
    assert weights.sum() == pytest.approx(1.0) and weights[0] == pytest.approx(weights[2]) and weights[1] > weights[0]
    assert line_weights([0.0, 1.0], 2.0, "lorentzian")[1] == pytest.approx(0.5 / 1.5)
    with pytest.raises(ValueError, match="not in the spin system"):
        radiation_damping(spin_system, zero_generator(spin_system), "13C", 1.0)
    with pytest.raises(ValueError, match="weights"):
        radiation_damping(spin_system, zero_generator(spin_system), "1H", 1.0, offsets=[0.0, 1.0], weights=[1.0])
    with pytest.raises(NotImplementedError, match="sphten-liouv"):
        hilb = one_spin(HILB)
        radiation_damping(hilb, zero_generator(hilb), "1H", 1.0, offsets=[0.0, 1.0])
    with pytest.raises(ValueError, match="positive"):
        rd_rate(spin_system, "1H", eta=0.0, Q=200, concentration=1.0)
    with pytest.raises(ValueError, match="shape"):
        line_weights([0.0], 1.0, "square")
