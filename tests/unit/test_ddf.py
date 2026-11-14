"""The distant dipolar field (spynach.ddf): the uniform demagnetising shift and its sign, the CRAZED build-up against
the Bessel closed form and its convergence in the number of modes, the Fourier-mode lattice against a 1-D grid for
a coupled pair with relaxation, the bookkeeping (Hermitian symmetry, thermalisation in the uniform mode, number-density
weights), the conversion from the concentration, and the input checks.

Author: Alexej Jerschow.
"""

import numpy as np
import pytest
import scipy.sparse as sps
from scipy.special import jv

import spynach
from spynach.ddf import DipolarFieldSpec, ddf_time, dipolar_field
from spynach.filters import isotope_orders
from spynach.program import Acquire, Delay, Evolve, Gradient, Program, Pulse, Sample
from spynach.propagation import iserstep

LIOUV = {"formalism": "sphten-liouv", "approximation": "none"}
LONG = Sample(z_max=1.0, x_max=1.0)  # so long that only the uniform mode is detected
TAU_D = 0.1


def protons(shifts, couplings=None, **inter):
    """``(spin_system, H, R)`` of protons at 9.4 T with the given shifts (ppm) and relaxation options."""
    n = len(shifts)
    inter = {"zeeman": {"scalar": list(shifts)}, **inter}
    if couplings is not None:
        inter["coupling"] = {"scalar": couplings}
    spin_system = spynach.create({"isotopes": ["1H"] * n, "magnet": 9.4}, inter)
    spin_system = spynach.assume(spynach.basis(spin_system, LIOUV), "nmr")
    H = spynach.hamiltonian(spin_system)
    R = spynach.relaxation(spin_system) if "relaxation" in inter else 0 * H
    return spin_system, H, R


def crazed(m, G, delta, dt, n, theta=0.0, **kwargs):
    """90x - G - 90x - mG - acquire, the gradients at angle ``theta`` to B0; one gradient unit is ``G delta``."""
    c, s = np.cos(theta), np.sin(theta)
    return Program(
        [
            Pulse("1H", np.pi / 2),
            Gradient(delta, z=G * c, x=G * s),
            Pulse("1H", np.pi / 2),
            Gradient(delta, z=m * G * c, x=m * G * s),
            Acquire("1H", dt, n),
        ],
        mode="explicit",
        sample=LONG,
        **kwargs,
    )


def uniform(program, spin_system, H, R):
    """The signal of the uniform mode alone (no sinc leak of the helix through the finite sample)."""
    modes = program.run(spin_system, H, R, 0 * R, output="modes")
    return modes.signals[modes.modes == 0][0]


def single_pulse(spin_system, H, R, dt=1e-3, **kwargs):
    program = Program([Pulse("1H", np.pi / 2), Acquire("1H", dt, 1)], mode="explicit", **kwargs)
    return uniform(program, spin_system, H, R)[0]


def bessel(m, a):
    """Mode ``m`` of ``cos(ks) exp(i a sin ks)``: ``(m/a) J_m(a)``, in units of the single-pulse signal."""
    a = np.asarray(a, dtype=float)
    safe = np.where(a != 0, a, 1.0)
    return np.where(a != 0, m / safe * jv(m, safe), 0.5 if m == 1 else 0.0)


def precession_rate(spin_system, L, rho, dt=1e-3, n=50):
    """The phase slope of the uniform transverse magnetisation under ``L`` (rad/s) and its final ``m_z``."""
    vec = L.stack(rho)
    out = []
    for _ in range(n):
        vec = iserstep(spin_system, L, 0.0, "LG4", vec, dt)
        mx, my, mz = L.magnetisation(vec)
        out.append(mx[0] + 1j * my[0])
    phase = np.unwrap(np.angle(out))
    return (phase[-1] - phase[0]) / ((n - 1) * dt), float(np.real(mz[0]))


def test_uniform_mode_demagnetising_shift():
    """Without gradients the field is the demagnetising shift ``(3 N_z - 1)/2 cos(beta) / tau_d``: none for a sphere,
    the sign of a field adding to B0 (that of a positive chemical shift) for a cylinder along B0, m_z untouched; the
    program route equals the direct iserstep loop."""
    spin_system, H, R = protons([0.0])
    beta = np.pi / 3
    rho = spynach.step(
        spin_system, spynach.operator(spin_system, "Ly", "1H"), spynach.state(spin_system, "Lz", "1H"), beta
    )
    rates = {}
    for name, Nz in (("sphere", 1 / 3), ("cylinder", 0.0), ("disc", 1.0)):
        L = dipolar_field(spin_system, H + 0j * H, "1H", TAU_D, demagnetising=Nz)
        rates[name], mz = precession_rate(spin_system, L, rho)
        assert abs(mz - np.cos(beta)) < 1e-9
        assert abs(abs(rates[name]) - abs(3 * Nz - 1) / 2 * np.cos(beta) / TAU_D) < 1e-4
    shifted, H_shift, _ = protons([1e-3])  # +0.4 Hz at 9.4 T: a field adding to B0
    reference, _ = precession_rate(shifted, dipolar_field(shifted, H_shift + 0j * H_shift, "1H", 1e9), rho)
    assert np.sign(rates["cylinder"]) == np.sign(reference) and np.sign(rates["disc"]) == -np.sign(reference)
    assert abs(rates["sphere"]) < 1e-9
    # the program with a field and no gradient is the same single mode
    dt, n = 1e-3, 40
    rho0 = spynach.state(spin_system, "Lz", "1H", "cheap")
    coil = spynach.state(spin_system, "L+", "1H", "cheap")
    spec = DipolarFieldSpec("1H", TAU_D, demagnetising=0.0)
    program = Program([Pulse("1H", beta, phase=[90]), Acquire("1H", dt, n)], mode="explicit", field=spec, rho0=rho0)
    fid = program.run(spin_system, H, R, 0 * R)
    L = dipolar_field(spin_system, H + 0j * H, "1H", TAU_D, demagnetising=0.0, rho_eq=rho0)
    vec = L.stack(spynach.step(spin_system, spynach.operator(spin_system, "Ly", "1H"), rho0, beta))
    manual = []
    for _ in range(n):
        manual.append(L.observe(coil, vec))
        vec = iserstep(spin_system, L, 0.0, "LG4", vec, dt)
    assert np.abs(fid - manual).max() < 1e-10 * np.abs(manual).max()


@pytest.mark.parametrize("theta_deg", [0.0, np.degrees(np.arccos(1 / np.sqrt(3))), 90.0], ids=["z", "magic", "x"])
def test_crazed_build_up_against_the_closed_form(theta_deg):
    """One spin, no relaxation: the mode ``m`` of ``cos(ks) exp(i a sin ks)`` with ``a = Delta t / tau_d`` is the
    CRAZED signal of a second gradient of ``m`` units, read during the acquisition; the field vanishes at the magic
    angle (the ``m = 1`` echo stays at half the single-pulse signal), and ``Delta = -1/2`` across B0. The closed form
    holds when the uniform mode carries the same factor as the modulation, ``N_z = (2 Delta + 1)/3``."""
    spin_system, H, R = protons([0.0])
    theta = np.radians(theta_deg)
    Delta = (3 * np.cos(theta) ** 2 - 1) / 2
    delta, G, dt, n = 20e-6, 2.0, 1e-3, 120
    reference = single_pulse(spin_system, H, R)
    t = np.arange(n) * dt + delta  # the field acts during the second gradient as well
    for m in (1, 2, 3):
        spec = DipolarFieldSpec("1H", TAU_D, nmodes=12, demagnetising=(2 * Delta + 1) / 3)
        fid = uniform(crazed(m, G, delta, dt, n, theta, field=spec), spin_system, H, R) / reference
        expected = bessel(m, abs(Delta) * t / TAU_D)
        assert np.abs(np.abs(fid) - np.abs(expected)).max() < 1e-5
        if abs(Delta) < 1e-9:
            assert np.abs(np.abs(fid) - (0.5 if m == 1 else 0.0)).max() < 1e-8
    if theta_deg == 0.0:  # convergence in the number of modes kept beyond the gradients' reach
        errors = []
        for nmodes in (0, 1, 4):
            spec = DipolarFieldSpec("1H", TAU_D, nmodes=nmodes, demagnetising=1.0)
            fid = uniform(crazed(2, G, delta, dt, n, field=spec), spin_system, H, R) / reference
            errors.append(np.abs(np.abs(fid) - np.abs(bessel(2, t / TAU_D))).max())
        assert errors[0] > 10 * errors[1] > 100 * errors[2] and errors[2] < 1e-6


class GridField:
    """The dipolar field on a grid of ``npoints`` positions over one pitch of the helix, for a block-diagonal
    iserstep: the local field at each point from its own magnetisation less the uniform part, which carries the
    factor of the sample shape, as the lattice does."""

    def __init__(self, spin_system, L0, isotope, tau_d, nz, npoints, rho_eq):
        self.model = dipolar_field(spin_system, L0, isotope, tau_d, rho_eq=rho_eq)
        self.L0 = sps.csr_array(L0)
        self.delta_s, self.delta_shape = 1.0, (3 * nz - 1) / 2
        self.tau_d, self.npoints = tau_d, npoints
        self.nstates = self.model.nstates

    def columns(self, vec):
        return vec.reshape(self.npoints, self.nstates).T

    def __call__(self, t, vec):
        cols = self.columns(vec)
        m = self.model
        blocks = []
        parts = []
        for c in m.coils:
            values = np.conj(c) @ cols / m.mz_eq
            parts.append(self.delta_s * (values - values.mean()) + self.delta_shape * values.mean())
        mx, my, mz = parts
        for j in range(self.npoints):
            H_d = m.sign / self.tau_d * (2 / 3 * mz[j] * m.Cz - (mx[j] * m.Cx + my[j] * m.Cy) / 3)
            blocks.append(self.L0 + H_d)
        return sps.block_diag(blocks, format="csr")


def test_lattice_against_a_grid_for_a_coupled_pair():
    """A J-coupled pair with different shifts and T1/T2 relaxation in a spherical sample: the Fourier-mode lattice
    equals the local field on a grid of 32 positions along the helix, pulses, gradients and acquisition included."""
    spin_system, H, R = protons(
        [0.2, -0.2],
        [[0, 15.0], [0, 0]],
        relaxation=["t1_t2"],
        r1_rates=[1.0, 1.5],
        r2_rates=[3.0, 4.0],
        rlx_keep="diagonal",
        equilibrium="zero",
    )
    L0 = H + 1j * R
    gamma = spin_system.gammas[0]
    delta, G, dt, n, m = 0.5e-3, 0.05, 0.5e-3, 40, 2
    k0 = gamma * G * delta
    spec = DipolarFieldSpec("1H", TAU_D, nmodes=12, timestep=dt)
    fid = uniform(crazed(m, G, delta, dt, n, field=spec), spin_system, H, R)
    npoints = 32
    s = 2 * np.pi / k0 * np.arange(npoints) / npoints
    ratio = -isotope_orders(spin_system)["1H"].astype(float)  # the program's orders are minus Spinach's
    rho0 = spynach.state(spin_system, "Lz", "1H", "cheap")
    coil = spynach.state(spin_system, "L+", "1H", "cheap")
    Lx = spynach.operator(spin_system, "Lx", "1H")
    grid = GridField(spin_system, L0, "1H", TAU_D, 1 / 3, npoints, rho0)
    stack = np.tile(np.asarray(rho0, dtype=complex).reshape(-1, 1), (1, npoints))

    def evolve(stack, duration):
        return grid.columns(iserstep(spin_system, grid, 0.0, "LG4", stack.T.reshape(-1), duration))

    def gradient(stack, units):
        return stack * np.exp(1j * units * ratio[:, None] * k0 * s[None, :])

    stack = gradient(evolve(spynach.step(spin_system, Lx, stack, np.pi / 2), delta), 1)
    stack = gradient(evolve(spynach.step(spin_system, Lx, stack, np.pi / 2), delta), m)
    reference = np.zeros(n, dtype=complex)
    for i in range(n):
        reference[i] = (np.conj(coil) @ stack).mean()
        if i < n - 1:
            stack = evolve(stack, dt)
    assert np.abs(fid - reference).max() < 1e-9 * np.abs(reference).max()
    assert np.abs(reference).max() > 0.01 * abs(single_pulse(spin_system, H, R))  # a build-up did happen


def test_symmetry_thermalisation_and_weights():
    """The magnetisation coefficients of a Hermitian state are conjugate-symmetric in the mode index; T1 recovery
    appears in the uniform mode only; number-density weights set the share of each spin in the field."""
    spin_system, H, R = protons(
        [0.0],
        relaxation=["t1_t2"],
        r1_rates=[5.0],
        r2_rates=[8.0],
        rlx_keep="diagonal",
        equilibrium="IME",
        temperature=300.0,
    )
    rho_eq = spynach.equilibrium(spin_system)
    L = dipolar_field(spin_system, H + 1j * R, "1H", TAU_D, nmodes=3, rho_eq=rho_eq)
    orders = -isotope_orders(spin_system)["1H"]
    transverse = spynach.step(spin_system, spynach.operator(spin_system, "Lx", "1H"), rho_eq, np.pi / 2)
    columns = np.zeros((L.nstates, L.size), dtype=complex)
    columns[:, L.half] = transverse * (orders == 0)  # the unit state and what is longitudinal stay uniform
    columns[:, L.half + 1] = transverse * (orders == 1)  # the helix: the order blocks at their wavevectors
    columns[:, L.half - 1] = transverse * (orders == -1)
    vec = L.stack(columns)
    dt, nsteps = 2e-3, 100
    for _ in range(nsteps):
        vec = iserstep(spin_system, L, 0.0, "LG4", vec, dt)
    mx, my, mz = L.magnetisation(vec)
    for coefficient in (mx, my, mz):
        assert np.abs(coefficient - np.conj(coefficient[::-1])).max() < 1e-12
    assert abs(mz[L.half] - (1 - np.exp(-5.0 * nsteps * dt))) < 1e-6  # recovery toward the uniform equilibrium
    assert np.abs(np.delete(mz, L.half)).max() < 1e-12  # none in the modulated modes
    assert abs(abs(mx[L.half + 1] + 1j * my[L.half + 1]) - np.exp(-8.0 * nsteps * dt)) < 1e-6  # T2 on the helix
    # Weights: two protons on resonance, the helix of the second reversed by a selective pi pulse before the
    # second 90x, so the longitudinal helices are antiparallel and the transverse ones stay parallel. The field
    # torques each spin at (2/3) of the weighted total longitudinal helix plus (1/3) of its own (the transverse
    # part of the field acts through the spin's own m_z): equal weights give no signal at all, and for weights
    # (w1, w2) the two spins precess at a_i = [(2/3) sigma +/- 1/3] t / tau_d with sigma = (w1 - w2)/(w1 + w2),
    # a closed form while the two transverse magnetisations stay parallel (the first few percent of tau_d)
    pair, H2, R2 = protons([0.0, 0.0])
    delta, G, dt, n = 20e-6, 2.0, 1e-3, 12
    t = np.arange(n) * dt + delta
    reference = single_pulse(pair, H2, R2)

    def reversed_pair(weights):
        spec = DipolarFieldSpec("1H", TAU_D, nmodes=10, demagnetising=1.0, weights=weights)
        program = Program(
            [
                Pulse("1H", np.pi / 2),
                Gradient(delta, z=G),
                Pulse([1], np.pi),
                Pulse("1H", np.pi / 2),
                Gradient(delta, z=2 * G),
                Acquire("1H", dt, n),
            ],
            mode="explicit",
            sample=LONG,
            field=spec,
        )
        return uniform(program, pair, H2, R2) / reference

    assert np.abs(reversed_pair([1.0, 1.0])).max() < 1e-8
    for weights in ([1.0, 3.0], [3.0, 1.0], [1.0, 0.0]):
        sigma = (weights[0] - weights[1]) / sum(weights)
        a1, a2 = (2 / 3 * sigma + 1 / 3) * t / TAU_D, (2 / 3 * sigma - 1 / 3) * t / TAU_D
        expected = (bessel(2, a1) + bessel(2, a2)) / 2
        assert np.abs(np.abs(reversed_pair(weights)) - np.abs(expected)).max() < 1e-5


def test_conversions_and_input_checks():
    spin_system, H, R = protons([0.0])
    assert abs(ddf_time(spin_system, "1H", concentration=110.0, temperature=300.0) - 0.0994) < 1e-3
    assert (
        abs(
            ddf_time(spin_system, "1H", concentration=110.0, temperature=300.0) * 1.5
            - ddf_time(spin_system, "1H", concentration=110.0, temperature=450.0)
        )
        < 1e-12
    )
    with pytest.raises(ValueError, match="concentration"):
        ddf_time(spin_system, "1H", concentration=0.0)
    with pytest.raises(ValueError, match="not in the spin system"):
        dipolar_field(spin_system, H, "13C", TAU_D)
    with pytest.raises(ValueError, match="tau_d"):
        dipolar_field(spin_system, H, "1H", -1.0)
    with pytest.raises(ValueError, match="nmodes"):
        dipolar_field(spin_system, H, "1H", TAU_D, nmodes=-1)
    with pytest.raises(ValueError, match="weights"):
        dipolar_field(spin_system, H, "1H", TAU_D, weights=[1.0, 1.0])
    with pytest.raises(ValueError, match="longitudinal"):
        dipolar_field(spin_system, H, "1H", TAU_D, rho_eq=spynach.state(spin_system, "Lx", "1H"))
    hilbert = spynach.basis(
        spynach.create({"isotopes": ["1H"], "magnet": 9.4}, {}), {"formalism": "zeeman-hilb", "approximation": "none"}
    )
    with pytest.raises(NotImplementedError, match="sphten-liouv"):
        dipolar_field(hilbert, np.zeros((2, 2)), "1H", TAU_D)
    L = dipolar_field(spin_system, H + 0j * H, "1H", TAU_D, nmodes=2)
    assert L.size == 5 and L.modes.tolist() == [-2, -1, 0, 1, 2] and np.allclose(L.delta, [1, 1, 0, 1, 1])
    with pytest.raises(ValueError, match="elements"):
        L.stack(np.zeros(3))
    with pytest.raises(ValueError, match="mode stack"):
        L.columns(np.zeros(3))
    vec = L.stack(spynach.state(spin_system, "L+", "1H"))
    coil = spynach.state(spin_system, "L+", "1H")
    assert L.observe(coil, vec) == L.observe(coil, vec, np.array([0, 0, 1, 0, 0]))
    assert L.observe(coil, vec, np.array([1, 1, 0, 1, 1])) == 0
    with pytest.raises(ValueError, match="mode='explicit'"):
        Program([Pulse("1H", np.pi / 2), Acquire("1H", 1e-3, 4)], field=DipolarFieldSpec("1H", TAU_D))
    with pytest.raises(NotImplementedError, match="Evolve"):
        Program(
            [Pulse("1H", np.pi / 2), Evolve(1e-3, 4), Pulse("1H", np.pi / 2), Acquire("1H", 1e-3, 4)],
            mode="explicit",
            field=DipolarFieldSpec("1H", TAU_D),
        ).run(spin_system, H, R, 0 * R)
    with pytest.raises(ValueError, match="timestep"):
        Program(
            [Pulse("1H", np.pi / 2), Delay(1e-3), Acquire("1H", 1e-3, 4)],
            mode="explicit",
            field=DipolarFieldSpec("1H", TAU_D, timestep=-1.0),
        ).run(spin_system, H, R, 0 * R)
