"""Fixture-free tests of the ENDOR functions and curry: first-order ENDOR frequencies, the perturbation orders
against matrix diagonalisation, salt's methods and options against each other, orientation selection, and
curry against the Curie law, the Brillouin function and its unit conversions."""

import numpy as np
import pytest

import epreasy as es

PROTON = {"Nucs": "1H", "A": 20}


def test_endorfrq_first_order_frequencies():
    """A weakly coupled proton gives nu_I +- A/2; the intensities are symmetric."""
    r = es.endorfrq(PROTON, {"Field": 350})
    nu = float(np.asarray(es.nucgval("1H")).ravel()[0]) * es.nmagn * 350e-3 / es.planck / 1e6
    np.testing.assert_allclose(np.sort(r.pos.ravel()), [nu - 10, nu + 10], atol=0.02)  # second-order shifts
    assert r.trans.shape == (2, 2) and r.int.shape == (2, 1)
    np.testing.assert_allclose(r.int[0], r.int[1], rtol=1e-6)
    assert r.info["Selectivity"] == 0
    only = es.endorfrq(PROTON, {"Field": 350, "Range": [nu + 5, nu + 15]})
    assert only.pos.shape == (1, 1)
    with pytest.raises(ValueError, match="doesn't contain any nuclei"):
        es.endorfrq({"g": 2}, {"Field": 350})
    with pytest.raises(ValueError, match="Exp.Field is missing"):
        es.endorfrq(PROTON, {"mwFreq": 9.5})
    with pytest.raises(ValueError, match="mwFreq is missing"):
        es.endorfrq(PROTON, {"Field": 350, "ExciteWidth": 100})


def test_endorfrq_options_change_what_they_should():
    nitro = {"g": [2.0088, 2.0061, 2.0027], "Nucs": "14N", "A": [16, 16, 86], "Q": [-0.5, 0.2], "HStrain": [10, 10, 30]}
    exp = {"Field": 350, "mwFreq": 9.8, "SampleFrame": [[0, 0, 0], [0.3, 1.1, 0.7]]}
    base = es.endorfrq(nitro, exp)
    enhanced = es.endorfrq(nitro, exp, {"Enhancement": "on"})
    np.testing.assert_allclose(enhanced.pos, base.pos)
    assert not np.allclose(enhanced.int, base.int)
    excited = es.endorfrq(nitro, {**exp, "ExciteWidth": 100})
    assert excited.info["Selectivity"] > 0 and excited.int.shape == base.int.shape
    cold = es.endorfrq(nitro, {**exp, "Temperature": 4})
    assert np.all(cold.int < base.int * 1.01) and not np.allclose(cold.int, base.int)
    chosen = es.endorfrq(nitro, exp, {"Transitions": [[1, 2], [4, 5]]})
    assert chosen.trans.tolist() == [[1, 2], [4, 5]]
    no_int = es.endorfrq(nitro, exp, {"Intensity": "off"})
    assert no_int.int is None
    pulse = es.endorfrq(nitro, exp, {"Intensity": "on"})
    flipped = es.endorfrq(nitro, {**exp, "rf": 0.3})
    assert np.all(flipped.int <= pulse.int + 1e-12)
    skipped = es.endorfrq(nitro, exp, {"OriWeights": [1, 0]})
    assert np.all(np.isnan(skipped.pos[:, 1])) and not np.any(np.isnan(skipped.pos[:, 0]))
    with pytest.raises(ValueError, match="Opt.Nuclei is out of range"):
        es.endorfrq(nitro, exp, {"Nuclei": [2]})


def test_perturbation_theory_approaches_matrix_diagonalisation():
    for A in (120, -120):
        for nuc in ("1H", "15N"):
            sysd = {"Nucs": nuc, "A": A}
            exact = np.sort(es.endorfrq(sysd, {"Field": 350}).pos.ravel())
            first = np.sort(es.endorfrq_perturb(sysd, {"Field": 350}, {"PerturbOrder": 1}).pos.ravel())
            second = np.sort(es.endorfrq_perturb(sysd, {"Field": 350}, {"PerturbOrder": 2}).pos.ravel())
            assert np.all(np.abs(second - exact) < np.abs(first - exact))
            assert np.abs(second - exact).max() < 1e-3
    triplet = {"S": 1.5, "D": 200, "Nucs": "1H", "A": [3, 5]}
    exp = {"Field": 350, "SampleFrame": [0.3, 1.1, 0.7]}
    pt = es.endorfrq_perturb(triplet, exp)
    assert pt.pos.shape == (4, 1) and pt.int.shape == (4, 1) and pt.trans.shape == (0, 2)
    with pytest.raises(ValueError, match="1 electron"):
        es.endorfrq_perturb({"S": [0.5, 0.5], "ee": 10, "Nucs": "1H", "A": [10, 0]}, {"Field": 350})
    with pytest.raises(ValueError, match="Opt.PerturbOrder"):
        es.endorfrq_perturb(PROTON, {"Field": 350}, {"PerturbOrder": 3})
    with pytest.raises(ValueError, match="temperature effects"):
        es.endorfrq_perturb(PROTON, {"Field": 350, "Temperature": 4})


def test_orisel_and_anisosubsys():
    sysd = {"g": [2.25, 2.25, 2], "HStrain": [100, 100, 100]}
    exp = {"Field": 326.5, "mwFreq": 9.7, "ExciteWidth": 50}
    w = es.orisel(sysd, exp, {"GridSize": 10})
    assert w.shape == (10,) and np.all(w >= 0) and w.max() > 0
    weights, trans = es.orisel(sysd, exp, {"GridSize": 10}, per_transition=True)
    assert weights.shape == (10, 1) and trans.tolist() == [[1, 2]]
    np.testing.assert_allclose(weights.sum(axis=1), w)
    # at the g_perpendicular field (theta = 90 deg) the weight peaks; the resonance of g = 2.25 is at 308 mT
    ori = np.array([[0.0, 0.0], [0.0, np.pi / 2]])
    at_308 = es.orisel(sysd, {**exp, "Field": 308.0, "Orientations": ori.T})
    assert at_308[1] > 100 * at_308[0]
    infinite = es.orisel(sysd, {**exp, "ExciteWidth": np.inf, "Orientations": ori.T})
    np.testing.assert_allclose(infinite, [1, 1], atol=1e-12)
    with pytest.raises(ValueError, match="ExciteWidth is missing"):
        es.orisel(sysd, {"Field": 326.5, "mwFreq": 9.7})
    sub = es.anisosubsys({**sysd, "Nucs": "14N,1H", "A": [[44, 49, 54], [1, 1, 2]]})
    assert sub["nNuclei"] == 1 and list(np.atleast_1d(sub["Nucs"])) == ["14N"]
    np.testing.assert_allclose(sub["HStrain"], np.sqrt(100**2 + 2**2), rtol=1e-12)


def test_salt_methods_and_options():
    s32 = {"S": 1.5, "D": 200, "Nucs": "1H", "A": [3, 5], "lwEndor": 0.2}
    exp = {"Field": 350, "mwFreq": 9.5, "Range": [0, 30]}
    matrix = es.salt(s32, exp)
    perturb = es.salt(s32, exp, {"Method": "perturb2"})
    assert matrix.x.shape == (1024,) and np.abs(matrix.spec - perturb.spec).max() < 0.01 * np.abs(matrix.spec).max()
    assert matrix.info["Transitions"].shape[1] == 2 and perturb.info["Transitions"].shape == (0, 2)
    separate = es.salt(s32, exp, {"separate": "transitions"})
    np.testing.assert_allclose(separate.spec.sum(axis=0), matrix.spec, atol=1e-9 * np.abs(matrix.spec).max())
    no_interp = es.salt(s32, exp, {"GridSize": [31, 1]})
    assert np.corrcoef(no_interp.spec, matrix.spec)[0, 1] > 0.99
    derivative = es.salt(s32, {**exp, "Harmonic": 1})
    assert abs(derivative.spec.sum()) < 1e-6 * np.abs(derivative.spec).max()
    lorentz = es.salt({**s32, "lwEndor": [0, 0.3]}, exp)
    assert lorentz.spec.sum() == pytest.approx(matrix.spec.sum(), rel=0.05)
    auto = es.salt(s32, {"Field": 350, "mwFreq": 9.5})
    assert auto.x[0] >= 0 and auto.x[-1] > 20
    crystal = es.salt(
        s32, {**exp, "SampleFrame": [[0, 0, 0], [0.3, 1.1, 0.7]], "MolFrame": [0, 0, 0]}, {"separate": "orientations"}
    )
    summed = es.salt(s32, {**exp, "SampleFrame": [[0, 0, 0], [0.3, 1.1, 0.7]], "MolFrame": [0, 0, 0]})
    assert crystal.spec.shape == (2, 1024)
    np.testing.assert_allclose(crystal.spec.sum(axis=0), summed.spec, atol=1e-12 * np.abs(summed.spec).max())
    two = es.salt(
        [{"Nucs": "1H", "A": [3, 5], "lwEndor": 0.2}, {"Nucs": "1H", "A": [8, 10], "lwEndor": 0.3, "weight": 0.5}], exp
    )
    one = es.salt({"Nucs": "1H", "A": [3, 5], "lwEndor": 0.2}, exp)
    other = es.salt({"Nucs": "1H", "A": [8, 10], "lwEndor": 0.3}, exp)
    np.testing.assert_allclose(two.spec, one.spec + 0.5 * other.spec, rtol=1e-12)
    empty = es.salt({"g": 2}, exp)
    assert np.all(empty.spec == 0)
    with pytest.raises(ValueError, match="Experiment.Field is missing"):
        es.salt(s32, {"mwFreq": 9.5, "Range": [0, 30]})
    with pytest.raises(ValueError, match="equivalent nuclei"):
        es.salt({"Nucs": "1H", "A": 5, "n": [2], "lwEndor": 0.2}, exp)
    with pytest.raises(ValueError, match="Cannot return separate transitions for crystal"):
        es.salt(s32, {**exp, "SampleFrame": [0, 0, 0], "MolFrame": [0, 0, 0]}, {"separate": "transitions"})


def test_salt_intensity_enhancement_excitation_and_temperature():
    n14 = {
        "Nucs": "14N",
        "g": [2.25, 2.25, 2],
        "A": [44, 49, 54],
        "Q": [0.84, 0.84, -1.68],
        "HStrain": [100, 100, 100],
        "lwEndor": 0,
    }
    exp = {"Range": [16, 65], "Field": 326.5, "nPoints": 2001, "mwFreq": 9.7}
    on = es.salt(n14, exp, {"Threshold": 1e-4, "GridSize": [31, 12], "Enhancement": "on"})
    off = es.salt(n14, exp, {"Threshold": 1e-4, "GridSize": [31, 12], "Enhancement": "off"})
    none = es.salt(n14, exp, {"Threshold": 1e-4, "GridSize": [31, 12], "Intensity": "off"})
    assert on.spec.sum() > 10 * off.spec.sum() > 0 and none.spec.sum() > 0  # hyperfine enhancement
    w14 = {
        "S": 0.5,
        "Nucs": "14N",
        "g": [2.2, 2.2, 2],
        "A": [4, 4, 5],
        "Q": [0.84, 0.84, -1.68],
        "HStrain": [1, 1, 1],
        "lwEndor": 0.1,
    }
    wexp = {"Range": [0, 35], "Field": 3394, "nPoints": 4096, "mwFreq": 95}
    full = es.salt(w14, wexp, {"Threshold": 1e-4, "GridSize": 50})
    gz = es.salt(w14, {**wexp, "ExciteWidth": 3e3}, {"Threshold": 1e-4, "GridSize": 50})
    gx = es.salt(w14, {**wexp, "ExciteWidth": 3e3, "mwFreq": 104.51}, {"Threshold": 1e-4, "GridSize": 50})
    assert full.info["Selectivity"] == 0 and gz.info["Selectivity"] > 1
    # orientation selection at g_z and g_x picks different parts of the powder pattern
    assert np.corrcoef(gz.spec, gx.spec)[0, 1] < 0.9
    peaks = [
        es.salt(
            {"Nucs": "13C", "A": 5, "lwEndor": 0.5}, {"Field": 1150, "Range": [10, 20], "Temperature": T}
        ).spec.max()
        for T in (5, 10, 20, 40, 80)
    ]
    assert np.all(np.diff(peaks) < 0)
    pre = es.salt(n14, {**exp, "ExciteWidth": 50}, {"Threshold": 1e-4, "GridSize": [31, 12], "OriPreSelect": 1})
    without = es.salt(n14, {**exp, "ExciteWidth": 50}, {"Threshold": 1e-4, "GridSize": [31, 12]})
    assert np.corrcoef(pre.spec, without.spec)[0, 1] > 0.99


def test_curry_against_closed_forms():
    S, g = 0.5, 2.0
    T = np.arange(1, 101.0)
    curie = S * (S + 1) * g**2 * es.mu0 * es.bmagn**2 * es.avogadro / 3 / es.boltzm / T
    chimol = es.curry({"S": S, "g": g}, {"Temperature": T, "Field": 0}, {"Output": "chimol"})
    assert chimol.shape == (1, 100)
    np.testing.assert_allclose(chimol.ravel(), curie, atol=1e-10)
    # the Brillouin function for S = 1/2 is tanh
    B = np.linspace(0, 10, 11) * 1e3
    mu = es.curry({"S": 0.5, "g": 2.0}, {"Temperature": 2.0, "Field": B}, {"Output": "muBM"})
    np.testing.assert_allclose(mu.ravel(), np.tanh(es.bmagn * B * 1e-3 / (es.boltzm * 2.0)), rtol=1e-6)
    si = es.curry({"g": 2, "S": 1}, {"Field": 10, "Temperature": 100}, {"Output": "mueff", "Units": "SI"})
    cgs = es.curry({"g": 2, "S": 1}, {"Field": 10, "Temperature": 100}, {"Output": "mueff", "Units": "CGS"})
    assert abs(si.item() - cgs.item()) < 1e-9 and abs(si.item() - 2 * np.sqrt(2)) < 1e-6
    rhombic = {"S": 1.5, "g": [2.1, 2.0, 1.9], "D": [20e3, 3e3]}
    exp = {"Temperature": [2, 20, 200], "Field": [0, 500, 5000]}
    mu_o, chi_o = es.curry(rhombic, exp)
    mu_p, chi_p = es.curry(rhombic, exp, {"Method": "partitionfunction", "Output": "muBM chimol"})
    assert (
        mu_o.shape == (3, 3)
        and np.abs(mu_o - mu_p).max() < 1e-5
        and np.abs(chi_o - chi_p).max() < 1e-3 * np.abs(chi_o).max()
    )
    outs = es.curry(rhombic, exp, {"Output": "mu mumol muBM chimol chimolT 1/chimol mueff"})
    mu, mumol, muBM, chimol, chimolT, inv, mueff = outs
    np.testing.assert_allclose(mumol, mu * es.avogadro)
    np.testing.assert_allclose(muBM, mu / es.bmagn)
    np.testing.assert_allclose(chimolT, chimol * np.array(exp["Temperature"])[None, :])
    np.testing.assert_allclose(inv, 1 / chimol)
    np.testing.assert_allclose(mueff, np.sqrt(chimolT * 3 * es.boltzm / es.avogadro / es.bmagn**2 / es.mu0))
    mu_cgs, chi_cgs = es.curry(rhombic, exp, {"Output": "mu chimol", "Units": "CGS"})
    np.testing.assert_allclose(mu_cgs, mu / 1e-3)
    np.testing.assert_allclose(chi_cgs, chimol / (4 * np.pi * 1e-6))
    vec = es.curry(
        rhombic,
        {"Temperature": 2, "Field": 500, "MolFrame": [0.2, 0.4, 0.6], "SampleFrame": [0.3, 1.1, 0.7]},
        {"Output": "muvec"},
    )
    assert vec.shape == (3, 1, 1)
    dimer = {"S": [3.5, 0.5], "ee": -2 * 5 * 30e3}
    mu1 = es.curry(dimer, {"Temperature": 4, "Field": [0, 1000, 3000]}, {"Output": "mu", "Spins": 1})
    mu2 = es.curry(dimer, {"Temperature": 4, "Field": [0, 1000, 3000]}, {"Output": "mu", "Spins": 2})
    both = es.curry(dimer, {"Temperature": 4, "Field": [0, 1000, 3000]}, {"Output": "mu", "Spins": [1, 2]})
    # as in EasySpin, Opt.Spins also restricts the Zeeman term of the Hamiltonian, so the moments do not add up
    assert np.all(mu1[1:] > 10 * mu2[1:]) and np.allclose(
        both, es.curry(dimer, {"Temperature": 4, "Field": [0, 1000, 3000]}, {"Output": "mu"})
    )
    tip = es.curry({"S": 0.5, "g": 2, "TIP": 1e-9}, {"Temperature": 50, "Field": 100}, {"Output": "chimol"})
    plain = es.curry({"S": 0.5, "g": 2}, {"Temperature": 50, "Field": 100}, {"Output": "chimol"})
    np.testing.assert_allclose(tip - plain, 1e-9, rtol=1e-8)
    with pytest.raises(ValueError, match="Exp.Temperature is missing"):
        es.curry(rhombic, {"Field": 0})
    with pytest.raises(ValueError, match="keyword in Opt.Output is not known"):
        es.curry(rhombic, exp, {"Output": "magnetisation"})
    with pytest.raises(ValueError, match="multiple components"):
        es.curry([rhombic, rhombic], exp)
