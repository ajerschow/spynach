"""Fixture-free tests of the epreasy foundation: constants and units, nuclear data, rotations, spin and
Stevens operators, spin system validation, Hamiltonian components and energy levels against closed forms."""

import numpy as np
import pytest

import epreasy as es
from spynach.conventions import euler2dcm


# ------------------------------------------------------------------------------------- units, nuclear data
def test_units_and_constants():
    assert es.mt2mhz(350) == pytest.approx(350 * 1e-3 * es.bmagn * es.gfree / es.planck / 1e6)
    assert es.mhz2mt(es.mt2mhz(350, 2.1), 2.1) == pytest.approx(350)
    assert es.unitconvert(1, "cm^-1->MHz") == pytest.approx(29979.2458)
    assert es.unitconvert(es.unitconvert(3.0, "eV->K"), "K->eV") == pytest.approx(3.0)
    with pytest.raises(ValueError, match="Did you mean"):
        es.unitconvert(1, "mt->mhz")
    with pytest.raises(ValueError, match="Unknown unit"):
        es.unitconvert(1, "furlong->MHz")


def test_nucdata():
    data = es.nucdata("1H,14N, 63Cu,")
    assert data.nucs == ["1H", "14N", "63Cu"]
    np.testing.assert_array_equal(data.spin, [0.5, 1, 1.5])
    assert data.gn[0] == pytest.approx(5.5856946893)
    assert es.nucspin("13C") == 0.5 and es.nucabund("13C") == pytest.approx(0.0107, rel=1e-2)
    assert es.nucqmom("1H") == 0 and es.nucqmom("14N") == pytest.approx(0.02044)
    assert np.isnan(es.nucqmom("125Sb"))  # "not measured" entries of the table
    assert len(es.isotopes()) == 351 and es.isotopes()["14N"].name == "nitrogen"
    with pytest.raises(ValueError, match="Please specify one of"):
        es.nucdata("N")
    with pytest.raises(ValueError, match="Unknown element"):
        es.nucdata("1Xx")
    with pytest.raises(ValueError, match="Use , and not ;"):
        es.nucstring2list("1H;14N")
    assert es.nucstring2list("") == [] and es.nucstring2list(["1H", "2H"]) == ["1H", "2H"]


# --------------------------------------------------------------------------------------------- rotations
@pytest.mark.parametrize("angles", [[0.4, 1.1, 2.3], [0, 0, 0], [np.pi / 2, np.pi / 2, 0], [-0.3, 2.9, 5.1]])
def test_erot_is_a_passive_zyz_rotation(angles):
    R = es.erot(angles)
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-15)
    assert np.linalg.det(R) == pytest.approx(1)
    # EasySpin's passive rotation is the transpose of Spinach's active one for the same angles
    np.testing.assert_allclose(R, np.asarray(euler2dcm(angles)).T, atol=1e-15)
    np.testing.assert_allclose(es.erot(*angles), R)
    # the explicit product of the three elementary rotations
    a, b, g = angles
    Ra = np.array([[np.cos(a), np.sin(a), 0], [-np.sin(a), np.cos(a), 0], [0, 0, 1]])
    Rb = np.array([[np.cos(b), 0, -np.sin(b)], [0, 1, 0], [np.sin(b), 0, np.cos(b)]])
    Rg = np.array([[np.cos(g), np.sin(g), 0], [-np.sin(g), np.cos(g), 0], [0, 0, 1]])
    np.testing.assert_allclose(R, Rg @ Rb @ Ra, atol=1e-15)


def test_eulang_inverts_erot():
    for angles in ([0.4, 1.1, 2.3], [1.0, 3.0, 0.5], [5.9, 0.2, 2.0]):
        back = es.eulang(es.erot(angles))
        np.testing.assert_allclose(es.erot(back), es.erot(angles), atol=1e-14)
        assert back[1] >= 0
    np.testing.assert_allclose(es.eulang(np.eye(3)), [0, 0, 0])
    with pytest.raises(ValueError, match="not orthogonal"):
        es.eulang(2 * np.eye(3))
    with pytest.raises(ValueError, match="negative determinant"):
        es.eulang(np.diag([1, 1, -1]))


def test_angles_and_vectors():
    phi, theta = [0, 0.3, 1.7, 4.0], [0, 0.5, 1.2, 2.9]
    v = es.ang2vec(phi, theta)
    assert v.shape == (3, 4)
    np.testing.assert_allclose(np.linalg.norm(v, axis=0), 1)
    p, t = es.vec2ang(v)
    np.testing.assert_allclose(t, theta)
    np.testing.assert_allclose(np.mod(p[1:], 2 * np.pi), np.mod(phi[1:], 2 * np.pi))
    np.testing.assert_allclose(es.letter2vec("-xy"), [-1, -1, 0] / np.sqrt(2))
    np.testing.assert_allclose(es.ang2vec(0.5, [0.1, 0.2]).shape, (3, 2))
    assert es.vec2ang("z")[1] == 0
    with pytest.raises(ValueError, match="Unknown value"):
        es.letter2vec("w")


# --------------------------------------------------------------------------------------------- operators
@pytest.mark.parametrize("S", [0.5, 1, 1.5, 2.5])
def test_sop_angular_momentum_algebra(S):
    Sx, Sy, Sz, Sp, Sm, E = es.sop(S, "x", "y", "z", "+", "-", "e")
    np.testing.assert_allclose(Sx @ Sy - Sy @ Sx, 1j * Sz, atol=1e-15)
    np.testing.assert_allclose(Sp, Sx + 1j * Sy, atol=1e-15)
    np.testing.assert_allclose(Sm, Sx - 1j * Sy, atol=1e-15)
    np.testing.assert_allclose(Sx @ Sx + Sy @ Sy + Sz @ Sz, S * (S + 1) * E, atol=1e-14)
    assert np.allclose(np.diag(Sz), np.arange(S, -S - 1, -1))  # +S first, as in EasySpin


def test_sop_syntaxes_agree():
    spins = [0.5, 1, 1.5]
    a = es.sop(spins, "xye")
    b = es.sop(spins, "x1y2")
    c = es.sop(spins, "x1,y2")
    d = es.sop(spins, np.array([[1, 1], [2, 2]]))
    e = np.kron(np.kron(es.sop(0.5, "x"), es.sop(1, "y")), np.eye(4))
    for op in (b, c, d, e):
        np.testing.assert_allclose(op, a, atol=1e-15)
    assert es.sop(spins, "z3", sparse=True).shape == (24, 24)
    x13 = es.sop(1.5, "x(1|3)")
    assert x13[0, 2] == 0.5 and x13[2, 0] == 0.5 and np.count_nonzero(x13) == 2
    e2 = es.sop(1.5, "e(2)")
    assert e2[1, 1] == 1 and np.count_nonzero(e2) == 1
    assert es.sop(0.5, "a")[0, 0] == 1 and es.sop(0.5, "b")[1, 1] == 1
    for bad in ("xyzz", "x4", "x1x1", "x(1|1)", "e(1|2)", "x(3|1)", "x(2)", "x1,y", "q"):
        with pytest.raises(ValueError):
            es.sop(spins, bad)
    with pytest.raises(ValueError, match="spin-1/2"):
        es.sop(1, "a")


def test_stevens_operators_closed_forms():
    for S in (1, 2.5):
        Sx, Sy, Sz, Sp, Sm = es.sop(S, "x", "y", "z", "+", "-")
        n = int(2 * S + 1)
        np.testing.assert_allclose(es.stev(S, 2, 0), 3 * Sz @ Sz - S * (S + 1) * np.eye(n), atol=1e-13)
        np.testing.assert_allclose(es.stev(S, 2, 2), (Sp @ Sp + Sm @ Sm) / 2, atol=1e-13)
        np.testing.assert_allclose(es.stev(S, 2, -2), (Sp @ Sp - Sm @ Sm) / 2j, atol=1e-13)
        np.testing.assert_allclose(es.stev(S, 1, 0), Sz, atol=1e-13)
        np.testing.assert_allclose(es.stev(S, 1, 1), Sx, atol=1e-13)
        np.testing.assert_allclose(es.stev(S, 1, -1), Sy, atol=1e-13)
    O40 = es.stev(2.5, 4, 0)
    Sz = es.sop(2.5, "z")
    S = 2.5
    X = S * (S + 1)
    expected = 35 * Sz @ Sz @ Sz @ Sz - (30 * X - 25) * Sz @ Sz + (3 * X**2 - 6 * X) * np.eye(6)
    np.testing.assert_allclose(O40, expected, atol=1e-11)
    assert es.stev([0.5, 2.5], 4, 1, ispin=2).shape == (12, 12)
    with pytest.raises(ValueError, match="larger than 2\\*S"):
        es.stev(1, 4, 0)
    with pytest.raises(ValueError, match="between -k and k"):
        es.stev(2.5, 4, 5)


# ------------------------------------------------------------------------------------- spin systems
def test_validatespinsys_processing():
    full = es.validatespinsys({"g": 2.1, "Nucs": "14N", "A": [16, 86], "Q": [-0.5, 0.2], "lwpp": 0.3})
    np.testing.assert_allclose(full["g"], [[2.1, 2.1, 2.1]])
    np.testing.assert_allclose(full["A"], [[16, 16, 86]])
    np.testing.assert_allclose(full["Q"], np.array([[-1 + 0.2, -1 - 0.2, 2]]) * (-0.5 / 4))
    np.testing.assert_allclose(full["lw"], [0.3 * np.sqrt(2 * np.log(2)), 0])
    assert full["nStates"] == 6 and full["Spins"].tolist() == [0.5, 1] and full["processed"]
    de = es.validatespinsys({"S": 1, "D": [1000, 100]})
    np.testing.assert_allclose(de["D"], [[-1000 / 3 + 100, -1000 / 3 - 100, 2000 / 3]])
    pair = es.validatespinsys({"S": [0.5, 0.5], "J": 5, "dip": [-10, 2], "dvec": [0.3, -0.2, 0.1]})
    assert pair["fullee"] and pair["ee"].shape == (3, 3)
    np.testing.assert_allclose(np.diag(pair["ee"]), [5 - 10 + 2, 5 - 10 - 2, 5 + 20])
    removed = es.validatespinsys({"Nucs": "12C,13C,16O", "A": [[0, 0, 0], [20, 20, 30], [0, 0, 0]]})
    assert removed["Nucs"] == ["13C"] and removed["nStates"] == 4 and removed["A"].shape == (1, 3)
    assert es.hsdim([0.5, 1, 1.5]) == 24 and es.hsdim({"S": 2.5, "Nucs": "55Mn", "A": 1}) == 36
    assert es.spinvec({"S": [1, 0.5], "ee": 1}).tolist() == [1, 0.5]


def test_validatespinsys_errors():
    with pytest.raises(ValueError, match="Fix capitalization"):
        es.validatespinsys({"nucs": "1H", "A": 1})
    with pytest.raises(ValueError, match="No hyperfine tensors"):
        es.validatespinsys({"Nucs": "1H"})
    with pytest.raises(ValueError, match="coupling terms are missing"):
        es.validatespinsys({"S": [0.5, 0.5]})
    with pytest.raises(ValueError, match="only one or the other"):
        es.validatespinsys({"S": [0.5, 0.5], "ee": 1, "J": 1})
    with pytest.raises(ValueError, match="Sys.g has wrong size"):
        es.validatespinsys({"g": [2, 2, 2, 2]})
    with pytest.raises(ValueError, match="Sys.D and Sys.B2"):
        es.validatespinsys({"S": 1, "D": 100, "B2": 1})
    with pytest.raises(ValueError, match="no nucleus"):
        es.validatespinsys({"A": 5})
    with pytest.raises(NotImplementedError, match="Orbital"):
        es.validatespinsys({"L": 1, "soc": 100})
    with pytest.raises(NotImplementedError, match="Higher-order Zeeman"):
        es.validatespinsys({"Ham112": [1, 2, 3]})


# ---------------------------------------------------------------------------------------- Hamiltonians
def test_free_electron_and_proton_zeeman():
    E = es.levels({"S": 0.5}, "z", 350)
    np.testing.assert_allclose(E, np.array([-1, 1]) * es.mt2mhz(350) / 2)
    # a proton: gn * nmagn * B / h, the nuclear moment is parallel to the field
    E = es.levels({"S": 0.5, "Nucs": "1H", "A": 0}, "z", 350)
    nu_p = es.nucgval("1H")[0] * es.nmagn * 0.35 / es.planck / 1e6
    assert E[1] - E[0] == pytest.approx(nu_p, rel=1e-12)


def test_isotropic_hyperfine_first_order():
    sys = {"S": 0.5, "Nucs": "14N", "A": 50}
    E = es.levels(sys, "z", 350)
    # high field, first order: E = nu_e mS + A mS mI - nu_n mI, so the mI levels of the mS = +1/2
    # manifold (the upper three) are spaced by A/2 - nu_n and those of mS = -1/2 by A/2 + nu_n
    nu_n = es.nucgval("14N")[0] * es.nmagn * 0.35 / es.planck / 1e6
    upper, lower = np.sort(E[3:]), np.sort(E[:3])
    np.testing.assert_allclose(np.diff(upper), 25 - nu_n, atol=0.2)
    np.testing.assert_allclose(np.diff(lower), 25 + nu_n, atol=0.2)


def test_zero_field_splitting_levels():
    D, E = 1000.0, 120.0
    levels = es.levels({"S": 1, "D": [D, E]}, "z", 0)
    # S.D.S with principal values (-D/3+E, -D/3-E, 2D/3): the m = 0 level at -2D/3, m = +-1 mixed by E
    np.testing.assert_allclose(np.sort(levels), np.sort([-2 * D / 3, D / 3 - E, D / 3 + E]), atol=1e-9)
    # the same system described with Stevens operators: D (Sz^2 - S(S+1)/3) = (D/3) O_2^0 and
    # E (Sx^2 - Sy^2) = E O_2^2; the B2 columns run from q = +2 to q = -2
    stevens = es.levels({"S": 1, "B2": [[E, 0, D / 3, 0, 0]]}, "z", 0)
    np.testing.assert_allclose(np.sort(stevens), np.sort(levels), atol=1e-9)


def test_ham_components_and_field_split():
    sys = {
        "S": [0.5, 1], "g": [[2.0023, 2.0023, 2.0023], [2.0, 2.01, 2.02]], "gFrame": [[0, 0, 0], [0.3, 0.6, 0.9]],
        "J": 5, "dip": [-10, 2], "dvec": [0.3, -0.2, 0.1], "eeFrame": [0.1, 0.4, 0.7], "ee2": 2,
        "D": [[0, 0], [500, 50]], "DFrame": [[0, 0, 0], [0.2, 0.5, 0.9]],
        "Nucs": "14N,1H", "A": [[10, 10, 12, 1, 1, 1], [5, 5, 5, 0, 0, 0]],
        "AFrame": [[0.3, 0.7, 1.1, 0, 0, 0], [0] * 6], "Q": [[-0.5, 0.2], [0, 0]], "nn": [0.05, 0.05, -0.1],
    }  # fmt: skip
    H0, mux, muy, muz = es.ham(sys)
    parts = [es.ham_zf(sys), es.ham_ee(sys), es.ham_hf(sys), es.ham_nq(sys), es.ham_nn(sys)]
    np.testing.assert_allclose(H0, sum(parts), atol=1e-12)
    for H in parts + [mux, muy, muz]:
        np.testing.assert_allclose(H, H.conj().T, atol=1e-12)
    ex, ez = es.ham_ez(sys), es.ham_nz(sys)
    for k in range(3):
        np.testing.assert_allclose((mux, muy, muz)[k], ex[k] + ez[k], atol=1e-12)
    B = np.array([120.0, -45.0, 300.0])
    np.testing.assert_allclose(es.ham(sys, B), H0 - B[0] * mux - B[1] * muy - B[2] * muz, atol=1e-10)
    H0s, muzL = es.ham(sys, B, split=True)
    np.testing.assert_allclose(H0s - np.linalg.norm(B) * muzL, es.ham(sys, B), atol=1e-10)
    np.testing.assert_allclose(es.ham_ez(sys, B), -(B[0] * ex[0] + B[1] * ex[1] + B[2] * ex[2]), atol=1e-12)
    assert es.ham(sys, B, sparse=True).nnz < H0.size
    # tensor frames: a tilted A tensor equals the explicit rotation of its principal values
    tilted = es.ham_hf({"Nucs": "14N", "A": [16, 16, 86], "AFrame": [0.3, 0.7, 1.1]})
    R = es.erot([0.3, 0.7, 1.1]).T
    full = R @ np.diag([16, 16, 86]) @ R.T
    explicit = es.ham_hf({"Nucs": "14N", "A": full})
    np.testing.assert_allclose(tilted, explicit, atol=1e-12)


def test_levels_shapes_and_orientations():
    sys = {"g": [2.0088, 2.0061, 2.0027], "Nucs": "14N", "A": [16, 16, 86]}
    E = es.levels(sys, [[0, 0], [np.pi / 3, np.pi / 4], [1.2, 2.0]], [0, 100, 350, 1200])
    assert E.shape == (3, 4, 6)
    assert np.all(np.diff(E, axis=-1) >= 0)
    E2 = es.levels(sys, phi=[0, 0.5], theta=[0, 1, 2], B=[300, 400])  # a 101-point range
    assert E2.shape == (2, 3, 101, 6)
    np.testing.assert_allclose(es.levels(sys, "z", 350), es.levels(sys, [[0, 0, 0.7]], 350))
    Ev, V = es.levels(sys, "x", 350, vectors=True)
    assert V.shape == (6, 6)
    H = es.ham(sys, [350, 0, 0])
    np.testing.assert_allclose(V.conj().T @ H @ V, np.diag(Ev), atol=1e-9)
    with pytest.raises(TypeError):
        es.levels(sys, "z")
