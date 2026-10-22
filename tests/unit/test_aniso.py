"""Fixture-free tests of anisotropic interactions: conventions, tensor input, Q, orientation, powder averages."""

import numpy as np
import pytest
import scipy.sparse as sps

import spynach
from spynach.conventions import eeqq2nqi, euler2dcm, mat2sphten, wigner
from spynach.oracle import assert_matches

HILB = {"formalism": "zeeman-hilb", "approximation": "none"}
LIOUV = {"formalism": "sphten-liouv", "approximation": "none"}


def dense(x):
    return x.toarray() if sps.issparse(x) else np.asarray(x)


def system(isotopes, bas=HILB, **inter):
    return spynach.assume(spynach.basis(spynach.create({"isotopes": isotopes, "magnet": 14.1}, inter), bas), "nmr")


def hamiltonian_at(spin_system, angles):
    I, Q = spynach.hamiltonian(spin_system, aniso=True)
    H = dense(I) + dense(spynach.orientation(Q, angles))
    return (H + H.conj().T) / 2


def test_conventions():
    angles = (0.3, 0.9, 1.2)
    R = euler2dcm(angles)
    assert np.allclose(R @ R.T, np.eye(3)) and np.isclose(np.linalg.det(R), 1)
    assert np.allclose(
        euler2dcm((0.3, 0, 0)), [[np.cos(0.3), -np.sin(0.3), 0], [np.sin(0.3), np.cos(0.3), 0], [0, 0, 1]]
    )
    for rank in (1, 2, 3):
        D = wigner(rank, *angles)
        assert D.shape == (2 * rank + 1,) * 2 and np.allclose(D @ D.conj().T, np.eye(2 * rank + 1))
    assert np.allclose(wigner(2, 0, 0, 0), np.eye(5))
    # A symmetric traceless tensor has no rank-1 part; its rank-2 components rotate with D^2
    M = np.array([[1.0, 0.2, 0.3], [0.2, -3.0, 0.4], [0.3, 0.4, 2.0]])
    iso, lam, phi = mat2sphten(M)
    assert iso == 0 and np.allclose(lam, 0) and np.isclose(phi[2], np.sqrt(1 / 6) * (2 * 2.0 - 1.0 + 3.0))
    assert np.isclose(np.linalg.norm(phi) ** 2, np.sum(M * M))  # Frobenius norm is preserved
    _, _, phi_rot = mat2sphten(R @ M @ R.T)
    assert np.allclose(phi_rot, wigner(2, *angles) @ phi) or np.allclose(phi_rot, wigner(2, *angles).conj() @ phi)
    nqi = eeqq2nqi(1.24e6, 0.22, 1, [0.1, 0.2, 0.3])
    assert np.isclose(np.trace(nqi), 0) and np.allclose(nqi, nqi.T)
    assert np.allclose(
        np.sort(np.linalg.eigvalsh(eeqq2nqi(1.24e6, 0.22, 1, [0, 0, 0]))),
        np.sort([-1.24e6 * 0.78 / 4, -1.24e6 * 1.22 / 4, 1.24e6 / 2]),
    )


def test_tensor_inputs():
    spin_system = spynach.create(
        {"isotopes": ["1H", "13C"], "magnet": 14.1},
        {
            "zeeman": {"eigs": [[-2, -2, 4], None], "euler": [[0.3, 0.9, 1.2], None], "scalar": [1.0, 50.0]},
            "coupling": {
                "eigs": [[None, [-1.0, -1.0, 2.0]], [None, None]],
                "euler": [[None, [0.1, 0.2, 0.3]], [None, None]],
                "scalar": [[0, 140.0], [0, 0]],
            },
        },
    )
    R = euler2dcm([0.3, 0.9, 1.2])
    shift = (spin_system.zeeman_matrix[0] / spin_system.basefrqs[0] - np.eye(3)) * 1e6
    assert np.allclose(shift, R @ np.diag([-2, -2, 4]) @ R.T + np.eye(3))
    R = euler2dcm([0.1, 0.2, 0.3])
    assert np.allclose(
        spin_system.coupling_matrix[0][1] / (2 * np.pi), R @ np.diag([-1.0, -1.0, 2.0]) @ R.T + 140.0 * np.eye(3)
    )
    assert spin_system.coupling_matrix[1][0] is None
    with pytest.raises(ValueError, match="euler"):
        spynach.create({"isotopes": ["1H"], "magnet": 14.1}, {"zeeman": {"eigs": [[1, 2, 3]]}})
    with pytest.raises(ValueError, match="identical non-empty"):
        spynach.create({"isotopes": ["1H"], "magnet": 14.1}, {"zeeman": {"eigs": [[1, 2, 3]], "euler": [None]}})


def test_dipolar_couplings_from_coordinates():
    """dipolar.m stores half the dipolar tensor in both [n][k] and [k][n]; the pair counts twice in H."""
    spin_system = spynach.create(
        {"isotopes": ["1H", "13C"], "magnet": 14.1}, {"coordinates": [[0, 0, 0], [0, 0, 1.09]]}
    )
    r = 1.09e-10
    b = (
        spin_system.gammas[0]
        * spin_system.gammas[1]
        * spin_system.tols.hbar
        * spin_system.tols.mu0
        / (4 * np.pi * r**3)
    )
    D = spin_system.coupling_matrix[0][1]
    assert np.allclose(D, 0.5 * b * np.diag([1, 1, -2])) and np.allclose(spin_system.coupling_matrix[1][0], D)
    assert spin_system.proxmatrix.tolist() == [[False, True], [True, False]]
    assert np.isclose(abs(b) / (2 * np.pi), 23.3e3, rtol=0.01)  # the usual 1H-13C one-bond dipolar coupling
    spin_system_far = spynach.create(
        {"isotopes": ["1H", "13C"], "magnet": 14.1, "tols": {"prox_cutoff": 1.0}},
        {"coordinates": [[0, 0, 0], [0, 0, 1.09]]},
    )
    assert spin_system_far.coupling_matrix[0][1] is None and not spin_system_far.proxmatrix.any()
    with pytest.raises(ValueError, match="collision"):
        spynach.create({"isotopes": ["1H", "1H"], "magnet": 14.1}, {"coordinates": [[0, 0, 0], [0, 0, 0.3]]})
    # Coordinates only couple spins within one chemical subsystem
    spin_system_two = spynach.create(
        {"isotopes": ["1H", "1H"], "magnet": 14.1},
        {"coordinates": [[0, 0, 0], [0, 0, 2.0]], "chem": {"parts": [[0], [1]], "concs": [1, 1]}},
    )
    assert spin_system_two.coupling_matrix[0][1] is None


def test_proximity_graph_drives_ik_bases():
    coords = [[0, 0, 0], [0, 0, 1.5], [0, 0, 3.0], [0, 0, 20.0]]
    spin_system = spynach.create(
        {"isotopes": ["1H"] * 4, "magnet": 14.1, "tols": {"prox_cutoff": 2.0}}, {"coordinates": coords}
    )
    assert spin_system.proxmatrix.sum() == 4  # 0-1 and 1-2, both ways
    spin_system_ik1 = spynach.basis(
        spin_system,
        {**LIOUV, "approximation": "IK-1", "connectivity": "scalar_couplings", "inter_level": 1, "prox_level": 2},
    ).basis
    # pairs (0,1) and (1,2) correlate; spin 3 and the pair (0,2) do not
    assert spin_system_ik1.nstates == 1 + 4 * 3 + 2 * 9


@pytest.mark.parametrize("bas", [HILB, LIOUV])
def test_csa_hamiltonian_follows_the_rotated_tensor(bas):
    spin_system = system(["1H"], bas, zeeman={"eigs": [[-2, -2, 4]], "euler": [[0, 0, 0]], "scalar": [3.0]})
    Lz = dense(spynach.operator(spin_system, "Lz", "1H"))
    for angles in ([0, 0, 0], [0.3, 0.9, 1.2], [1.0, 2.0, 3.0]):
        R = euler2dcm(angles)
        Z = R @ spin_system.zeeman_matrix[0] @ R.T
        expected = (Z[2, 2] - spin_system.basefrqs[0]) * Lz  # secular: the zz element of the rotated tensor
        np.testing.assert_allclose(hamiltonian_at(spin_system, angles), expected, rtol=0, atol=1e-6)


def test_dipolar_pair_angular_dependence():
    spin_system = system(["1H", "1H"], HILB, coordinates=[[0, 0, 0], [0, 0, 3.0]])
    LzSz = dense(spynach.operator(spin_system, ["Lz", "Lz"], [0, 1]))
    flip = dense(spynach.operator(spin_system, ["L+", "L-"], [0, 1])) + dense(
        spynach.operator(spin_system, ["L-", "L+"], [0, 1])
    )
    for beta in (0.0, 0.7, np.pi / 2, np.arccos(1 / np.sqrt(3))):
        D = euler2dcm([0, beta, 0]) @ spin_system.coupling_matrix[0][1] @ euler2dcm([0, beta, 0]).T
        expected = 2 * D[2, 2] * (LzSz - flip / 4)  # twice: both [0][1] and [1][0] hold half the tensor
        np.testing.assert_allclose(hamiltonian_at(spin_system, [0, beta, 0]), expected, rtol=0, atol=1e-7)
    assert np.abs(hamiltonian_at(spin_system, [0, np.arccos(1 / np.sqrt(3)), 0])).max() < 1e-7  # magic angle


def test_quadrupolar_coupling_enters_through_q():
    nqi = eeqq2nqi(1.24e6, 0.22, 1, [0, 0, 0])
    spin_system = system(["14N"], HILB, coupling={"matrix": [[nqi.tolist()]]})
    I, Q = spynach.hamiltonian(spin_system, aniso=True)
    assert np.abs(dense(I)).max() == 0  # no isotropic part
    H = hamiltonian_at(spin_system, [0, 0, 0])
    T20 = dense(spynach.operator(spin_system, ["T2,0"], [0]))
    _, _, phi = mat2sphten(2 * np.pi * nqi)
    np.testing.assert_allclose(H, phi[2] * T20, rtol=1e-12, atol=1e-6)  # secular: T(2,0) only under 'nmr'
    with pytest.raises(ValueError, match="traceless"):
        spynach.hamiltonian(system(["14N"], HILB, coupling={"matrix": [[np.eye(3).tolist()]]}), aniso=True)


def test_powder_average_and_crystal():
    spin_system = spynach.basis(
        spynach.create(
            {"isotopes": ["1H"], "magnet": 14.1},
            {"zeeman": {"eigs": [[-2, -2, 4]], "euler": [[0, 0, 0]], "scalar": [3.0]}},
        ),
        LIOUV,
    )
    pars = {
        "spins": ["1H"],
        "offset": 0.0,
        "sweep": 20000.0,
        "npoints": 64,
        "rho0": spynach.state(spin_system, "L+", "1H"),
        "coil": spynach.state(spin_system, "L+", "1H"),
    }
    single = spynach.powder(spin_system, spynach.acquire, {**pars, "grid": "single_crystal"}, "nmr")
    assert_matches(
        single,
        spynach.crystal(spin_system, spynach.acquire, {**pars, "orientation": [0, 0, 0]}, "nmr"),
        rtol=1e-12,
        atol=0,
    )
    fid = spynach.powder(spin_system, spynach.acquire, {**pars, "grid": "leb_2ang_rank_23"}, "nmr")
    assert fid.shape == (64,)
    # The powder-averaged FID at t = 0 is the full magnetisation, and its initial phase velocity is the
    # isotropic shift (first moment of the powder pattern); use a dwell time short against the anisotropy
    assert np.isclose(fid[0], 0.5)
    fast = spynach.powder(
        spin_system, spynach.acquire, {**pars, "sweep": 1e6, "npoints": 2, "grid": "leb_2ang_rank_23"}, "nmr"
    )
    iso = spin_system.basefrqs[0] * 3.0e-6
    assert np.isclose(np.angle(fast[1] / fast[0]) * 1e6, -iso, rtol=1e-3)
    per_orientation = spynach.powder(
        spin_system, spynach.acquire, {**pars, "grid": "leb_2ang_rank_11", "sum_up": False}, "nmr"
    )
    grid = spynach.load_grid("leb_2ang_rank_11")
    assert len(per_orientation) == grid["weights"].size and np.isclose(grid["weights"].sum(), 1)
    with pytest.raises(ValueError, match="unknown spherical grid"):
        spynach.load_grid("no_such_grid")
    with pytest.raises(ValueError, match="orientation"):
        spynach.crystal(spin_system, spynach.acquire, pars, "nmr")


def test_liouville_and_hilbert_q_agree():
    inter = {
        "zeeman": {"eigs": [[-2, -2, 4], [1, -3, 2]], "euler": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]},
        "coordinates": [[0, 0, 0], [0, 2.5, 0.5]],
    }
    spin_system_hilb, spin_system_liouv = system(["1H", "1H"], HILB, **inter), system(["1H", "1H"], LIOUV, **inter)
    from .test_superop import basis_operators, reference_superop

    B = basis_operators(spin_system_liouv)
    for angles in ([0, 0, 0], [0.5, 1.1, 2.0]):
        H = hamiltonian_at(spin_system_hilb, angles)
        L = hamiltonian_at(spin_system_liouv, angles)
        np.testing.assert_allclose(L, reference_superop(B, H, "comm"), rtol=0, atol=1e-6)


def test_powder_reports_progress_after_each_orientation():
    spin_system = spynach.basis(
        spynach.create({"isotopes": ["1H"], "magnet": 14.1}, {"zeeman": {"eigs": [[-2, -2, 4]], "euler": [[0, 0, 0]]}}),
        LIOUV,
    )
    pars = {
        "spins": ["1H"],
        "offset": 0.0,
        "sweep": 20000.0,
        "npoints": 16,
        "grid": "leb_2ang_rank_11",
        "rho0": spynach.state(spin_system, "L+", "1H"),
        "coil": spynach.state(spin_system, "L+", "1H"),
    }
    calls, keys = [], []

    def acquire(spin_system, parameters, H, R, K):
        keys.append(set(parameters))
        return spynach.acquire(spin_system, parameters, H, R, K)

    fid = spynach.powder(spin_system, acquire, {**pars, "progress": lambda *args: calls.append(args)}, "nmr")
    n = spynach.load_grid("leb_2ang_rank_11")["weights"].size
    assert calls == [(k, n) for k in range(1, n + 1)]
    assert not any("progress" in seen for seen in keys)  # the sequence never sees the hook
    assert_matches(fid, spynach.powder(spin_system, spynach.acquire, pars, "nmr"), rtol=1e-14, atol=0)
    with pytest.raises(TypeError, match="progress"):
        spynach.powder(spin_system, spynach.acquire, {**pars, "progress": 1}, "nmr")
