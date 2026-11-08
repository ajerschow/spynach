"""The state-dependent integrator iserstep: Spinach's own checks (constant generators reduce to step, the nonlinear
one-proton case of test_dynamic_iserstep_highorder.m against a refined DP8 reference, invariants), the convergence
orders on the Bloch-Maxwell radiation-damping model of examples/fundamentals/product_quadratures_2.m, and parity
with the MATLAB values dumped by oracle/matlab/dumps/dump_iserstep.m (skipped without the fixture)."""

import numpy as np
import pytest
from scipy.integrate import solve_ivp

import spynach
from spynach.oracle import assert_matches, find_case, load_fixture
from spynach.propagation import ISERSTEP_METHODS, iserstep

HILB = {"formalism": "zeeman-hilb", "approximation": "none"}
LIOUV = {"formalism": "sphten-liouv", "approximation": "none"}

# The one-proton Hilbert-space case of Spinach's test_dynamic_iserstep_highorder.m: a Hermitian density matrix with
# coherences and a mildly nonlinear, non-commuting generator; the tolerances are Spinach's.
SX = np.array([[0, 0.5], [0.5, 0]])
SY = np.array([[0, -0.5j], [0.5j, 0]])
SZ = np.diag([0.5, -0.5])
RHO = np.array([[0.7, 0.2 + 0.1j], [0.2 - 0.1j, 0.3]])
H0, H1, H2 = 0.31 * SX + 0.17 * SZ, -0.23 * SY + 0.05 * SX, 0.11 * SX - 0.07 * SZ
DT = 5e-3
HIGH_ORDER = {"LG4A": 5e-8, "RKMK4": 5e-8, "RKMK-DP5": 5e-9, "RKMK-DP8": 5e-10, "RKMK-RKF45": 5e-9}


def nonlinear_generator(t, rho):
    return H0 + np.sin(4 * t) * H1 + 0.2 * np.real(np.trace(SZ @ rho)) * H2


def one_proton(bas):
    return spynach.basis(spynach.create({"isotopes": ["1H"], "magnet": 0.0}, {"zeeman": {"scalar": [0.0]}}), bas)


def ab_system(bas):
    spin_system = spynach.create(
        {"isotopes": ["1H", "1H"], "magnet": 14.1},
        {"zeeman": {"scalar": [1.0, 1.2]}, "coupling": {"scalar": [[0, 12.0], [0, 0]]}},
    )
    return spynach.assume(spynach.basis(spin_system, bas), "nmr")


@pytest.mark.parametrize("bas", [LIOUV, HILB], ids=["liouv", "hilb"])
def test_constant_generator_reduces_to_step(bas):
    """A state-independent generator has no time-ordering correction beyond the exact step (Spinach's check)."""
    spin_system = ab_system(bas)
    L = spynach.hamiltonian(spin_system)
    rho = spynach.state(spin_system, "L+", "1H")
    exact = spynach.step(spin_system, L, rho, 1e-3)
    for method in ISERSTEP_METHODS:
        if method == "LG4" and bas is HILB:
            with pytest.raises(NotImplementedError, match="Lie-group quadratures"):
                iserstep(spin_system, lambda t, r: L, 0.0, method, rho, 1e-3)
            continue
        out = iserstep(spin_system, lambda t, r: L, 0.0, method, rho, 1e-3)
        assert_matches(out, exact, rtol=1e-12, atol=1e-15, name=f"constant {method}")


def test_zero_step_returns_the_input():
    spin_system = one_proton(HILB)
    for method in ISERSTEP_METHODS:
        assert iserstep(spin_system, nonlinear_generator, 0.0, method, RHO, 0.0) is RHO


def test_nonlinear_high_order_methods_against_a_refined_dp8_reference():
    """Spinach's test_dynamic_iserstep_highorder.m: one step of each high-order method against two DP8 half steps,
    with the density-matrix trace and Hermiticity preserved by the unitary propagation."""
    spin_system = one_proton(HILB)
    ref = iserstep(spin_system, nonlinear_generator, 0.0, "RKMK-DP8", RHO, DT / 2)
    ref = iserstep(spin_system, nonlinear_generator, DT / 2, "RKMK-DP8", ref, DT / 2)
    for method, tol in HIGH_ORDER.items():
        out = iserstep(spin_system, nonlinear_generator, 0.0, method, RHO, DT)
        assert np.abs(out - ref).max() < tol, method
        assert abs(np.trace(out) - np.trace(RHO)) < 1e-12, method
        assert np.abs(out - out.conj().T).max() < 1e-12, method


def test_lie_methods_run_the_nonlinear_case_in_liouville_space():
    """The same generator as a commutation superoperator on the vectorised density matrix: every method, including
    LG4, agrees with a DP8 reference, and the two formalisms give the same density matrix."""
    hilb, liouv = one_proton(HILB), one_proton(LIOUV)
    unit = np.eye(2)

    def vec(rho):  # column-stacked density matrix, the zeeman-liouv convention
        return rho.reshape(-1, order="F")

    def generator(t, rho_vec):
        rho = rho_vec.reshape(2, 2, order="F")
        H = nonlinear_generator(t, rho)
        return np.kron(unit, H) - np.kron(H.T, unit)

    ref = iserstep(hilb, nonlinear_generator, 0.0, "RKMK-DP8", RHO, DT / 2)
    ref = iserstep(hilb, nonlinear_generator, DT / 2, "RKMK-DP8", ref, DT / 2)
    for method in ISERSTEP_METHODS:
        out = iserstep(liouv, generator, 0.0, method, vec(RHO), DT).reshape(2, 2, order="F")
        tol = HIGH_ORDER.get(method, 2e-4)
        assert np.abs(out - ref).max() < tol, method
        if method != "LG4":  # the three-generator step is Liouville-space only
            assert_matches(out, iserstep(hilb, nonlinear_generator, 0.0, method, RHO, DT), rtol=1e-10, atol=1e-13)


# ---------------------------------------------------------------------------- the Bloch-Maxwell model
CHIRP, R1, R2, RRD = 2 * np.pi * 400, 10.0, 10.0, 40.0
TILT = np.pi * 178 / 180
MU0 = np.array([np.sin(TILT), 0.0, np.cos(TILT)], dtype=complex)  # euler2dcm(0, 178 deg, 0) * [0 0 1]'
T_END = 0.5


def bloch_maxwell(t, mu):
    """Generator of product_quadratures_2.m: chirped offset, relaxation and the Bloembergen-Pound damping term."""
    linear = np.array([[R2, CHIRP * t, 0], [-CHIRP * t, R2, 0], [0, 0, R1]])
    damping = RRD * np.array([[mu[2], 0, 0], [0, mu[2], 0], [-mu[0], -mu[1], 0]])
    return -1j * (linear + damping)


def integrate(spin_system, method, npoints):
    dt = T_END / (npoints - 1)
    mu = MU0
    for n in range(npoints - 1):
        mu = iserstep(spin_system, bloch_maxwell, n * dt, method, mu, dt)
    return mu


def test_convergence_orders_on_the_radiation_damping_model():
    """Error against an adaptive reference as the grid is refined: slopes of 1, 2, 4, 4, 4, 5 and 8 (a Liouville
    formalism carries the tolerances; step() exponentiates any generator on any vector there)."""
    spin_system = spynach.basis(spynach.create({"isotopes": ["1H"], "magnet": 0.0}, {}), LIOUV)
    reference = solve_ivp(
        lambda t, mu: -1j * bloch_maxwell(t, mu) @ mu, (0, T_END), MU0, method="DOP853", rtol=1e-13, atol=1e-15
    ).y[:, -1]
    grids = np.ceil(2.0 ** np.linspace(7, 9.5, 5)).astype(int)
    nominal = {"PWCL": 1, "LG2": 2, "LG4": 4, "LG4A": 4, "RKMK4": 4, "RKMK-DP5": 5, "RKMK-DP8": 8}
    for method, order in nominal.items():
        errors = np.array([np.linalg.norm(integrate(spin_system, method, n) - reference) for n in grids])
        errors /= np.linalg.norm(reference)
        slope = -np.polyfit(np.log(grids[2:]), np.log(errors[2:]), 1)[0]
        assert abs(slope - order) < 0.5 if order < 5 else slope > order - 0.5, (method, slope, errors)
        assert errors[-1] < errors[0], method
    assert np.linalg.norm(integrate(spin_system, "RKMK-DP8", grids[-1]) - reference) < 1e-9


def test_input_checks():
    spin_system = one_proton(HILB)
    with pytest.raises(ValueError, match="method"):
        iserstep(spin_system, nonlinear_generator, 0.0, "RK4", RHO, DT)
    with pytest.raises(ValueError, match="function"):
        iserstep(spin_system, H0, 0.0, "LG2", RHO, DT)
    with pytest.raises(ValueError, match="timestep"):
        iserstep(spin_system, nonlinear_generator, 0.0, "LG2", RHO, np.inf)


# ---------------------------------------------------------------------------- parity with Spinach
@pytest.fixture
def fx():
    path = find_case("iserstep_hilb_1spin")
    if path is None:
        pytest.skip("fixture iserstep_hilb_1spin not available")
    return load_fixture(path)


def test_parity_with_spinach(fx):
    """The MATLAB test values of every method, one step of DT from RHO, and the two-half-step DP8 reference."""
    spin_system = one_proton(HILB)
    assert_matches(fx["/inputs/rho"], RHO, rtol=0, atol=0, name="rho")
    for method in ISERSTEP_METHODS:
        key = method.replace("-", "_")
        if method == "LG4":
            continue  # Hilbert-space Lie quadratures are not ported
        out = iserstep(spin_system, nonlinear_generator, 0.0, method, RHO, DT)
        assert_matches(out, fx[f"/steps/{key}"], rtol=1e-12, atol=1e-14, name=method)
    ref = iserstep(spin_system, nonlinear_generator, 0.0, "RKMK-DP8", RHO, DT / 2)
    ref = iserstep(spin_system, nonlinear_generator, DT / 2, "RKMK-DP8", ref, DT / 2)
    assert_matches(ref, fx["/steps/reference"], rtol=1e-12, atol=1e-14, name="reference")
