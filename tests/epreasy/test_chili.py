"""Fixture-free tests of the slow-motion simulator and its angular-momentum helpers: 3j symbols against
Clebsch-Gordan coefficients, Wigner matrices against their defining properties, the tensor conversions,
and EasySpin's own consistency checks of chili (solvers, field-sweep methods, double integrals, temperature,
the fast-motion limit)."""

import numpy as np
import pytest

import epreasy as es
from epreasy import slowmotion as sm

NITROXIDE = {"g": [2.008, 2.0061, 2.0027], "Nucs": "14N", "A": [16, 16, 86], "lw": 0.1, "tcorr": 1e-9}


def _coupled_3j(j1, j2, j3, m1, m2):
    """A 3j symbol from a Clebsch-Gordan coefficient computed by diagonalising the coupled J^2."""
    m3 = -(m1 + m2)

    def ops(j):
        m = np.arange(j, -j - 1, -1)
        jp = np.diag(np.sqrt(j * (j + 1) - m[1:] * (m[1:] + 1)), 1)
        return (jp + jp.T) / 2, (jp - jp.T) / 2j, np.diag(m)

    a, b = ops(j1), ops(j2)
    n1, n2 = int(2 * j1 + 1), int(2 * j2 + 1)
    J = [np.kron(a[k], np.eye(n2)) + np.kron(np.eye(n1), b[k]) for k in range(3)]
    J2 = sum(Jk @ Jk for Jk in J)
    vals, vecs = np.linalg.eigh(J2 + 1e-3 * J[2])
    jtot = (np.sqrt(4 * vals + 1) - 1) / 2
    mz = np.real(np.einsum("ij,jk,ki->i", vecs.conj().T, J[2], vecs))
    k = [i for i in range(vals.size) if abs(jtot[i] - j3) < 1e-2 and abs(mz[i] + m3) < 1e-6][0]
    v = vecs[:, k]
    top = sorted(
        (int(j1 - a_), int(j2 - b_))
        for a_ in np.arange(j1, -j1 - 1, -1)
        for b_ in np.arange(j2, -j2 - 1, -1)
        if abs(a_ + b_ + m3) < 1e-6
    )
    sign = np.sign(np.real(v[top[0][0] * n2 + top[0][1]])) or 1  # Condon-Shortley: the largest m1 is positive
    cg = np.real(v[int(j1 - m1) * n2 + int(j2 - m2)]) * sign
    return (-1) ** (j1 - j2 - m3) / np.sqrt(2 * j3 + 1) * cg


def test_wigner3j_and_6j():
    for args in (
        (2, 2, 2, 1, -1),
        (3, 2, 1, 2, -2),
        (1, 1, 0, 0, 0),
        (3, 3, 2, 1, 0),
        (1.5, 0.5, 1, 0.5, -0.5),
        (4, 2, 4, 3, -1),
        (6, 4, 8, 2, 1),
        (12, 10, 6, 3, -2),
    ):
        j1, j2, j3, m1, m2 = args
        assert es.wigner3j(j1, j2, j3, m1, m2, -(m1 + m2)) == pytest.approx(_coupled_3j(*args), abs=1e-13)
    assert (
        es.wigner3j([2, 1], [2, -1], [2, 0])
        == es.wigner3j([2, 2, 2], [1, -1, 0])
        == es.wigner3j([[2, 2, 2], [1, -1, 0]])
    )
    assert (
        es.wigner3j(2, 2, 2, 1, 1, 0) == 0 and es.wigner3j(1, 1, 3, 0, 0, 0) == 0 and es.wigner3j(1, 1, 1, 0, 0, 0) == 0
    )
    assert es.wigner3j(0, 0, 0, 0, 0, 0) == 1
    # orthogonality of the 3j symbols
    total = sum((2 * j3 + 1) * es.wigner3j(2, 3, j3, 1, -2, 1) ** 2 for j3 in range(1, 6))
    assert total == pytest.approx(1.0, abs=1e-14)
    assert es.wigner6j(1, 1, 1, 1, 1, 1) == pytest.approx(1 / 6)
    assert es.wigner6j(2, 2, 2, 2, 2, 2) == pytest.approx(-3 / 70)
    assert es.wigner6j(2, 3, 4, 3, 2, 0) == pytest.approx(
        -1 / np.sqrt(35)
    )  # {a b c; b a 0} = (-1)^(a+b+c)/sqrt((2a+1)(2b+1))
    assert es.wigner6j(1, 1, 1, 1, 1, 3) == 0  # triangle condition
    with pytest.raises(ValueError, match="integers or half-integers"):
        es.wigner3j(0.3, 1, 1, 0, 0, 0)


def test_wignerd_matrices_and_elements():
    for J in (0.5, 1, 2, 3.5):
        D = es.wignerd(J, 0.3, 0.7, 1.1)
        n = int(2 * J + 1)
        np.testing.assert_allclose(D @ D.conj().T, np.eye(n), atol=1e-12)
        d = es.wignerd(J, 0.7)
        assert np.isrealobj(d) and np.allclose(d @ d.T, np.eye(n), atol=1e-12)
        for m1 in np.arange(J, -J - 1, -1):
            for m2 in np.arange(J, -J - 1, -1):
                assert es.wignerd([J, m1, m2], 0.3, 0.7, 1.1) == pytest.approx(D[int(J - m1), int(J - m2)], abs=1e-12)
    np.testing.assert_allclose(es.wignerd(2, 0.0), np.eye(5))
    beta = np.linspace(0, np.pi, 9)
    np.testing.assert_allclose(es.wignerd([2, 0, 0], beta), (3 * np.cos(beta) ** 2 - 1) / 2, atol=1e-12)
    np.testing.assert_allclose(es.wignerd([1, 1, 0], beta), -np.sin(beta) / np.sqrt(2), atol=1e-12)
    plus = es.wignerd(1, 0.3, 0.7, 1.1, phase="+")
    np.testing.assert_allclose(plus, es.wignerd(1, -0.3, -0.7, -1.1), atol=1e-12)
    with pytest.raises(ValueError, match="J or"):
        es.wignerd([1, 2], 0.3)


def test_tensor_conversions_and_istotensor():
    Tc = np.array([[2.0, 0.3, 0.1], [0.2, 2.1, 0.4], [0.0, 0.5, 2.3]])
    T0, T1, T2 = es.tensor_cart2sph(Tc)
    np.testing.assert_allclose(es.tensor_sph2cart(T0, T1, T2), Tc, atol=1e-14)
    assert T0 == pytest.approx(-np.trace(Tc) / np.sqrt(3))
    T0v, T1v, T2v = es.tensor_cart2sph([1, 2, 3])
    assert np.all(T1v == 0) and T2v[0] == T2v[4] == -0.5 and T2v[2] == pytest.approx(np.sqrt(2 / 3) * 1.5)
    assert es.tensor_cart2sph(2.5)[0] == pytest.approx(-np.sqrt(3) * 2.5)
    with pytest.raises(ValueError, match="wrong size"):
        es.tensor_cart2sph([1, 2])
    # the spherical tensor operators of two spin vectors contract to the scalar product and the dipolar form
    S = [es.sop([0.5, 0.5], f"{c}1") for c in "xyz"]
    I = [es.sop([0.5, 0.5], f"{c}2") for c in "xyz"]
    T0, T1, T2 = sm._istotensor(S, I)
    np.testing.assert_allclose(T0, -(S[0] @ I[0] + S[1] @ I[1] + S[2] @ I[2]) / np.sqrt(3), atol=1e-14)
    np.testing.assert_allclose(T2[2], np.sqrt(2 / 3) * (S[2] @ I[2] - 0.5 * (S[0] @ I[0] + S[1] @ I[1])), atol=1e-14)
    np.testing.assert_allclose(
        T2[4], T2[0].conj().T, atol=1e-14
    )  # T_{2,-2} = T_{2,2}^dagger for Hermitian spin vectors


def test_basis_and_pq_order():
    basis = sm._generateoribasis(2, 0, 2, 2, 1, True)
    assert basis.tolist()[0] == [0, 1, 0, 0] and basis.shape == (11, 4)
    full = sm._generateoribasis(4, 3, 1, 2, -1, False)
    assert np.all(full[:, 0] <= 4) and np.all(np.abs(full[:, 3]) <= 1) and np.all(full[:, 2] <= 2)
    assert not np.any((full[:, 2] == 0) & (full[:, 1] != (-1) ** full[:, 0]))  # K = 0 needs jK = (-1)^L
    pq = sm._pqorder([0.5, 1])
    assert pq.shape == (36, 4) and set(pq[:, 0]) == {-1, 0, 1} and set(pq[:, 2]) == {-2, -1, 0, 1, 2}


def test_chili_lines_solvers_and_sweeps():
    exp = {"mwFreq": 9.8, "Harmonic": 0}
    r = es.chili(NITROXIDE, exp)
    sp = r.spec
    peaks = [k for k in range(1, sp.size - 1) if sp[k] > sp[k - 1] and sp[k] > sp[k + 1] and sp[k] > 0.1 * sp.max()]
    assert len(peaks) == 3 and r.info["nBasis"] > 1000  # three hyperfine lines, broadened by the motion
    np.testing.assert_allclose(np.diff(r.x[peaks]), es.mhz2mt(39.33, 2.0056), atol=0.1)
    # in the fast limit chili approaches garlic's fast-motion spectrum
    # (with a Lorentzian width: at 10 ps the intrinsic lines are far narrower than the field step)
    fast = es.chili(
        {**NITROXIDE, "tcorr": 1e-11, "lw": [0.1, 0.05]},
        {"mwFreq": 9.8, "Range": [345, 353], "nPoints": 600, "Harmonic": 0},
    )
    iso = es.garlic(
        {**NITROXIDE, "tcorr": 1e-11, "lw": [0.1, 0.05]},
        {"mwFreq": 9.8, "Range": [345, 353], "nPoints": 600, "Harmonic": 0},
    )
    assert np.corrcoef(fast.spec, iso.spec)[0, 1] > 0.95
    fast_peaks = fast.x[
        [
            k
            for k in range(1, 599)
            if fast.spec[k] > fast.spec[k - 1]
            and fast.spec[k] > fast.spec[k + 1]
            and fast.spec[k] > 0.3 * fast.spec.max()
        ]
    ]
    iso_peaks = iso.x[
        [
            k
            for k in range(1, 599)
            if iso.spec[k] > iso.spec[k - 1] and iso.spec[k] > iso.spec[k + 1] and iso.spec[k] > 0.3 * iso.spec.max()
        ]
    ]
    np.testing.assert_allclose(fast_peaks, iso_peaks, atol=0.05)
    # all solvers agree (EasySpin's chili_solvers test, tolerance 5e-3)
    gax, fs = {"g": [2, 2, 2.1], "tcorr": 1e-9}, {"Field": 350, "mwRange": [9, 11], "nPoints": 100}
    spc = np.array([es.chili(gax, fs, {"LLMK": [8, 0, 2, 0], "Solver": s}).spec for s in ("L", "\\", "E", "C", "B")])
    assert (spc.max(axis=0) - spc.min(axis=0)).max() / spc.max() < 5e-3
    rtl = es.chili(gax, fs, {"LLMK": [8, 0, 2, 0], "Lentz": False}).spec
    assert np.abs(rtl - spc[0]).max() / spc.max() < 5e-3
    # approximate and explicit field sweeps (chili_general_fieldsweeps, 1 %)
    dbl, fe = (
        {"g": [2.01, 2.003], "lw": 0.1, "tcorr": 1e-9},
        {"mwFreq": 9.5, "Harmonic": 0, "nPoints": 200, "Range": [337, 339]},
    )
    y1 = es.chili(dbl, fe, {"FieldSweepMethod": "approxlin"}).spec
    y2 = es.chili(dbl, fe, {"FieldSweepMethod": "explicit"}).spec
    y3 = es.chili(dbl, fe, {"FieldSweepMethod": "approxinv"}).spec
    assert np.abs(y1 / y1.max() - y2 / y2.max()).max() < 0.01 and np.abs(y1 / y1.max() - y3 / y3.max()).max() < 0.01
    # the double integral does not depend on the nuclear spin (chili_nucspins, 1 %)
    base, fexp, opt = (
        {"g": [2, 2.001], "tcorr": 1e-6, "lw": 5},
        {"Field": 350, "mwRange": [9.4, 10.2], "nPoints": 3000},
        {"LLMK": [30, 0, 0, 0]},
    )
    ints = []
    for nuc, A in (("15N", [200, 400]), ("14N", [400 / 3, 800 / 3]), ("63Cu", [100, 200])):
        s = es.chili({**base, "Nucs": nuc, "A": A}, fexp, opt)
        ints.append(s.spec.sum() * (s.x[1] - s.x[0]))
    assert (max(ints) - min(ints)) / max(ints) < 0.01
    # temperature: intensity * T is constant (chili_temperature)
    I = [
        es.chili(
            {"g": [2, 2.02], "lwpp": 0.1, "tcorr": 20e-9}, {"mwFreq": 9.5, "Range": [335, 341], "Temperature": T}
        ).spec.max()
        * T
        for T in (100, 300, 1000)
    ]
    assert max(I) / min(I) < 1 + 1e-4
    f = es.chili(
        {"g": [2.01, 2.02, 2.03], "Nucs": "14N", "A": [10, 20, 30], "tcorr": 1e-8, "lw": 0.1},
        {"Field": 340, "mwRange": [9.4, 9.8]},
    )
    assert f.x[0] == pytest.approx(9.4) and f.x[-1] == pytest.approx(9.8) and f.spec.shape == (1024,)


def test_chili_potentials_momd_and_options():
    pot = {"g": [2.05, 2.00], "logDiff": 7}
    pe = {"Field": 340, "mwRange": [9.4, 9.9], "nPoints": 200, "SampleFrame": [0, 0, 0]}
    po = {"LLMK": [4, 0, 2, 2], "highField": True}
    for lmk in ([2, 0, 0, 2], [2, 1, 0, 2], [2, 0, 1, 2]):
        r = es.chili({**pot, "Potential": [lmk]}, pe, po)
        assert np.isfinite(r.spec).all() and r.spec.max() > 0 and 0 < r.info["normPeq"] <= 1 + 1e-9
    nopot = es.chili(pot, pe, po)
    weak = es.chili({**pot, "Potential": [[2, 0, 0, 1e-6]]}, pe, po)
    assert np.abs(weak.spec - nopot.spec).max() < 1e-3 * np.abs(nopot.spec).max()
    momd = {
        "g": [2.008, 2.006, 2.003],
        "Nucs": "14N",
        "A": [20, 20, 85],
        "lw": 0.3,
        "tcorr": 1e-8,
        "Potential": [[2, 0, 0, 1]],
    }
    me = {"mwFreq": 9.5, "CenterSweep": [338, 20], "nPoints": 300}
    grid = es.chili(momd, me, {"GridSize": 5, "LLMK": [8, 0, 2, 2]})
    single = es.chili(momd, me, {"GridSize": 1, "LLMK": [8, 0, 2, 2]})
    director = es.chili(momd, {**me, "SampleFrame": [0, 0, 0]}, {"LLMK": [8, 0, 2, 2]})
    np.testing.assert_allclose(single.spec, director.spec, rtol=1e-10)
    assert not np.allclose(grid.spec, single.spec)
    ordered = es.chili(momd, {**me, "Ordering": 1.5}, {"GridSize": 5, "LLMK": [8, 0, 2, 2]})
    assert not np.allclose(ordered.spec, grid.spec)
    # options
    two = es.chili([NITROXIDE, {**NITROXIDE, "tcorr": 1e-8, "weight": 0.5}], {"mwFreq": 9.8, "Range": [344, 354]})
    one = es.chili(NITROXIDE, {"mwFreq": 9.8, "Range": [344, 354]})
    other = es.chili({**NITROXIDE, "tcorr": 1e-8}, {"mwFreq": 9.8, "Range": [344, 354]})
    np.testing.assert_allclose(two.spec, one.spec + 0.5 * other.spec, rtol=1e-12)
    hf = es.chili(NITROXIDE, {"mwFreq": 9.8, "Range": [344, 354]}, {"highField": True})
    assert np.corrcoef(hf.spec, one.spec)[0, 1] > 0.99  # the high-field approximation drops the pS = -1 coherences
    mod = es.chili(NITROXIDE, {"mwFreq": 9.8, "Range": [344, 354], "ModAmp": 0.1})
    assert np.corrcoef(mod.spec, one.spec)[0, 1] > 0.99  # pseudo-modulation broadens the narrow lines slightly
    absorption = es.chili(NITROXIDE, {"mwFreq": 9.8, "Range": [344, 354], "Harmonic": 0})
    disp = es.chili(NITROXIDE, {"mwFreq": 9.8, "Range": [344, 354], "mwPhase": np.pi / 2, "Harmonic": 0})
    flipped = es.chili(NITROXIDE, {"mwFreq": 9.8, "Range": [344, 354], "mwPhase": np.pi, "Harmonic": 0})
    np.testing.assert_allclose(flipped.spec, -absorption.spec, rtol=1e-10)
    assert abs(np.corrcoef(disp.spec, absorption.spec)[0, 1]) < 0.5  # the dispersion is in quadrature
    pc = es.chili(
        {"g": [2.08, 2.006, 2.002], "Nucs": "14N,1H", "A": [[20, 20, 100], [5, 5, 8]], "tcorr": 0.02e-9},
        {"mwFreq": 9.5, "Range": [330, 338], "nPoints": 400},
        {"LLMK": [6, 0, 2, 2], "PostConvNucs": [2]},
    )
    assert np.isfinite(pc.spec).all() and pc.info["nBasis"] < 2000
    with pytest.raises(ValueError, match="Only one of"):
        es.chili({**NITROXIDE, "Diff": 1e8}, {"mwFreq": 9.8})
    with pytest.raises(ValueError, match="is required"):
        es.chili({"g": [2.008, 2.0061, 2.0027]}, {"mwFreq": 9.8})
    with pytest.raises(ValueError, match="isotropic spin system"):
        es.chili({"g": 2.0, "tcorr": 1e-9}, {"mwFreq": 9.8, "Range": [340, 345]})
    with pytest.raises(ValueError, match="Please provide basis set"):
        es.chili({**pot, "Potential": [[2, 0, 0, 1]]}, pe)
    with pytest.raises(ValueError, match="Give either"):
        es.chili(NITROXIDE, {"mwFreq": 9.8, "Field": 350})
    with pytest.raises(NotImplementedError, match="fast"):
        es.chili(NITROXIDE, {"mwFreq": 9.8}, {"LiouvMethod": "fast"})
    with pytest.raises(ValueError, match="does not support strains"):
        es.chili({**NITROXIDE, "HStrain": [10, 10, 10]}, {"mwFreq": 9.8})
    with pytest.raises(ValueError, match="LLMK\\(1\\) must be an even"):
        es.chili(NITROXIDE, {"mwFreq": 9.8}, {"LLMK": [5, 3, 2, 2]})
