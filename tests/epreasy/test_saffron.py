"""Fixture-free tests of saffron and its helpers: the two-pulse ESEEM modulation formula for a weakly coupled
proton, the predefined sequences against the same sequences written out pulse by pulse, frequency-domain
against time-domain evolution, the product rule, relaxation decays, orientation selection, Mims ENDOR, crystals,
components, the helpers apowin, fdaxis, exponfit, larmorfrq and sf_pathways, and the error messages."""

import numpy as np
import pytest

import epreasy as es

N14 = {"Nucs": "14N", "A": [0.3, 0.3, 1.8], "Q": 1}
H1 = {"Nucs": "1H", "A": [1, 1, 5]}
TWO = {"Nucs": "14N,1H", "A": [[0.3, 0.3, 1.8], [1, 1, 4]], "Q": [[1, 0], [0, 0]]}
OPT = {"GridSize": 19}
P90, P180 = {"Flip": np.pi / 2}, {"Flip": np.pi}


def _exp(seq, **kw):
    e = {"Field": 330, "Sequence": seq, "dt": 0.01, "nPoints": 200, "tau": 0.08}
    e.update(kw)
    return e


def test_two_pulse_eseem_weak_proton():
    """A weakly coupled proton in a single crystal orientation modulates with 1 - k/4 (2 - 2cos(wa t) - 2cos(wb t)
    + cos(w- t) + cos(w+ t)) at the time axis tau + t, with ideal pulses, exactly (Mims 1972)."""
    field = 340.0
    nu_i = float(es.larmorfrq("1H", field).ravel()[0])
    A, B = 3.0, 1.5  # secular and pseudo-secular hyperfine (MHz) in the lab frame: A = [B 0 A] rows
    sys = {"Nucs": "1H", "A": [[0, 0, B], [0, 0, 0], [B, 0, A]]}
    exp = {"Field": field, "Sequence": "2pESEEM", "dt": 0.004, "nPoints": 150, "tau": 0.0, "MolFrame": [0, 0, 0]}
    r = es.saffron(sys, exp, {"TimeDomain": True})
    wa = np.hypot(nu_i + A / 2, B / 2)
    wb = np.hypot(nu_i - A / 2, B / 2)
    k = (B * nu_i / (wa * wb)) ** 2
    t = np.asarray(r.x)
    v = 1 - k / 4 * (
        2
        - 2 * np.cos(2 * np.pi * wa * t)
        - 2 * np.cos(2 * np.pi * wb * t)
        + np.cos(2 * np.pi * (wa - wb) * t)
        + np.cos(2 * np.pi * (wa + wb) * t)
    )
    sig = np.real(np.asarray(r.spec)) / (4 * np.pi)
    np.testing.assert_allclose(sig, v, atol=1e-10)
    assert r.info["nSkippedOrientations"] == 0
    assert r.info["f"].size == 2 * t.size and r.info["fd"].size == 2 * t.size


def test_predefined_equal_custom_sequences():
    """The predefined 2p, 3p and HYSCORE experiments equal the explicit sequences of ideal pulses."""
    a = es.saffron(N14, _exp("2pESEEM"), OPT)
    b = es.saffron(
        N14, {"Field": 330, "Sequence": [P90, 0.08, P180, 0.08], "nPoints": 200, "Dim1": [["d1,d2", 0.01]]}, OPT
    )
    np.testing.assert_allclose(b.spec, a.spec, atol=1e-12 * np.abs(a.spec).max())
    np.testing.assert_allclose(b.x, np.asarray(a.x) - 0.08)
    a = es.saffron(N14, _exp("3pESEEM", T=0.1), OPT)
    b = es.saffron(
        N14, {"Field": 330, "Sequence": [P90, 0.08, P90, 0.1, P90, 0.08], "nPoints": 200, "Dim1": [["d2", 0.01]]}, OPT
    )
    np.testing.assert_allclose(b.spec, a.spec, atol=1e-12 * np.abs(a.spec).max())
    a = es.saffron(N14, _exp("HYSCORE", nPoints=24, dt=0.02), OPT)
    seq = [P90, 0.08, P90, 0, P180, 0, P90, 0.08]
    b = es.saffron(
        N14, {"Field": 330, "Sequence": seq, "nPoints": [24, 24], "Dim1": [["d2", 0.02]], "Dim2": [["d3", 0.02]]}, OPT
    )
    np.testing.assert_allclose(b.spec, -a.spec, atol=1e-12 * np.abs(a.spec).max())  # EasySpin's sign convention
    assert len(b.x) == 2 and b.info["f1"].size == 48


@pytest.mark.parametrize("seq, kw", [("2pESEEM", {}), ("3pESEEM", {"T": 0.1}), ("4pESEEM", {"T": 0.1})])
def test_frequency_domain_against_time_domain(seq, kw):
    """The binned frequency-domain route reproduces explicit time-domain evolution up to the binning error."""
    fd = es.saffron(N14, _exp(seq, **kw), {**OPT, "Expand": 8})
    td = es.saffron(N14, _exp(seq, **kw), {**OPT, "TimeDomain": True})
    np.testing.assert_allclose(fd.spec, td.spec, atol=1e-2 * np.abs(td.spec).max())


def test_product_rule_and_nuclei_selection():
    r = es.saffron(TWO, _exp("3pESEEM"), OPT)
    p = es.saffron(TWO, _exp("3pESEEM"), {**OPT, "ProductRule": True})
    np.testing.assert_allclose(p.spec, r.spec, atol=1e-4 * np.abs(r.spec).max())
    # Opt.Nuclei moves the other nuclei into the electron core (numerical manifolds, all nuclear states)
    one = es.saffron(TWO, _exp("3pESEEM"), {**OPT, "Nuclei": [1]})
    alone = es.saffron(N14, _exp("3pESEEM"), OPT)
    assert one.spec.shape == alone.spec.shape and not np.allclose(one.spec, alone.spec)
    with pytest.raises(ValueError, match="out of range"):
        es.saffron(TWO, _exp("3pESEEM"), {**OPT, "Nuclei": [3]})


def test_relaxation_decays():
    sys = {**H1, "T1": 2, "T2": 1}
    undamped = es.saffron(H1, _exp("3pESEEM"), OPT)
    damped = es.saffron(sys, _exp("3pESEEM"), OPT)
    t = np.asarray(damped.x)
    np.testing.assert_allclose(damped.spec, undamped.spec * np.exp(-2 * 0.08 / 1) * np.exp(-t / 2), rtol=1e-12)
    two = es.saffron(sys, _exp("2pESEEM"), OPT)
    np.testing.assert_allclose(two.spec, es.saffron(H1, _exp("2pESEEM"), OPT).spec * np.exp(-2 * t / 1), rtol=1e-12)
    with pytest.raises(ValueError, match="custom sequences"):
        es.saffron(sys, {"Field": 330, "Sequence": [P90, 0.08, P180, 0.08], "nPoints": 50, "Dim1": [["d1,d2", 0.01]]})


def test_orientation_selection():
    sys = {"Nucs": "1H", "A": [1, 6], "g": [2, 2.2]}
    exp = {
        "Sequence": "2pESEEM",
        "mwFreq": 9.5,
        "dt": 0.015,
        "nPoints": 64,
        "ExciteWidth": 200,
        "tau": 0.001,
        "Field": 350,
    }
    r = es.saffron(sys, exp, {"GridSize": 31})
    assert r.info["nSkippedOrientations"] > 0
    wide = es.saffron(sys, {**exp, "ExciteWidth": 1e6}, {"GridSize": 31})
    assert wide.info["nSkippedOrientations"] == 0
    with pytest.raises(ValueError, match="ExciteWidth"):
        es.saffron(sys, {k: v for k, v in exp.items() if k != "ExciteWidth"})
    with pytest.raises(ValueError, match="mwFreq is missing"):
        es.saffron(sys, {k: v for k, v in exp.items() if k != "mwFreq"})
    # S = 1: the numerical manifolds; restricting the transitions changes the modulation
    trip = {"S": 1, "D": [300, 20], "Nucs": "1H", "A": [2, 5]}
    texp = {
        "Sequence": "3pESEEM",
        "mwFreq": 9.7,
        "dt": 0.01,
        "nPoints": 64,
        "ExciteWidth": 150,
        "tau": 0.1,
        "Field": 340,
    }
    a = es.saffron(trip, texp, OPT)
    b = es.saffron(trip, texp, {**OPT, "Transitions": [[1, 2]]})
    assert not np.allclose(a.spec, b.spec)


def test_mims_endor():
    nu = float(es.larmorfrq("1H", 325).ravel()[0])
    exp = {"Sequence": "MimsENDOR", "Field": 325, "tau": 0.1, "Range": [nu - 10, nu + 10], "nPoints": 401}
    r = es.saffron({**H1, "lwEndor": 0.1}, exp, OPT)
    x, y = np.asarray(r.x), np.real(r.spec)
    assert y.shape == (401,) and r.info["td"] is None
    # the doublet is centred at the proton Larmor frequency (the Mims blind spots make it asymmetric)
    assert y[:20].max() < 1e-6 * y.max() and y[-20:].max() < 1e-6 * y.max()
    assert abs(x[np.argmax(y[:200])] + x[200 + np.argmax(y[200:])] - 2 * nu) < 0.1
    m1 = es.saffron({**H1, "lwEndor": 0.1}, exp, {**OPT, "EndorMethod": 1})
    m2 = es.saffron({**H1, "lwEndor": 0.1}, exp, {**OPT, "EndorMethod": 2})
    for m in (m1, m2):
        assert np.argmax(np.real(m.spec)) in range(np.argmax(y) - 5, np.argmax(y) + 6)
    with pytest.raises(ValueError, match="lwEndor"):
        es.saffron(H1, exp, OPT)
    with pytest.raises(ValueError, match="Exp.Range"):
        es.saffron({**H1, "lwEndor": 0.1}, {k: v for k, v in exp.items() if k != "Range"}, OPT)
    with pytest.raises(ValueError, match="incorrect results"):
        es.saffron({**TWO, "lwEndor": 0.1}, exp, {**OPT, "EndorMethod": 0})


def test_crystal_and_components():
    exp = {**_exp("2pESEEM"), "MolFrame": [0.3, 0.5, 0.7], "SampleFrame": [[0, 0, 0], [0, np.pi / 4, 0]]}
    r = es.saffron(H1, exp)
    single = [es.saffron(H1, {**exp, "SampleFrame": sf}) for sf in exp["SampleFrame"]]
    np.testing.assert_allclose(r.spec, single[0].spec + single[1].spec, rtol=1e-12)
    tau0 = es.saffron(H1, {**exp, "tau": 0, "SampleFrame": [0, 0, 0]})
    np.testing.assert_allclose(np.real(tau0.spec[0]), 4 * np.pi, rtol=1e-10)  # no modulation at t = 0
    two = es.saffron([N14, {**H1, "weight": 0.5}], _exp("2pESEEM"), OPT)
    expect = es.saffron(N14, _exp("2pESEEM"), OPT).spec + 0.5 * es.saffron(H1, _exp("2pESEEM"), OPT).spec
    np.testing.assert_allclose(two.spec, expect, rtol=1e-12)


def test_custom_sequence_options():
    base = {"Field": 330, "Sequence": [P90, 0.08, P90, 0.0, P90, 0.08], "nPoints": 100, "Dim1": [["d2", 0.01]]}
    r = es.saffron(N14, base, OPT)
    a = es.saffron(N14, {**base, "Filter": "+a-"}, OPT)
    b = es.saffron(N14, {**base, "Pathways": "+b-"}, OPT)
    np.testing.assert_allclose(a.spec + b.spec, 2 * r.spec, rtol=1e-10)  # td is normalised per pathway count
    with pytest.raises(ValueError, match="too restrictive"):
        es.saffron(N14, {**base, "Filter": "+++"}, OPT)
    with pytest.raises(ValueError, match="no focused echo"):
        es.saffron(N14, {**base, "Sequence": [P90, 0.08, P90, 0.0, P90, 0.03]}, OPT)
    real = es.saffron(
        {"Nucs": "1H", "A": [2, 7]},
        {**base, "Sequence": [{"Flip": np.pi / 2, "tp": 0.016}, 0.08, P90, 0.0, P90, 0.08]},
        {**OPT, "nOffsets": 11},
    )
    assert real.spec.shape == (100,)
    with pytest.raises(NotImplementedError, match="thyme"):
        es.saffron(
            N14, {**base, "Sequence": [{"Flip": np.pi / 2, "Type": "sech/tanh"}, 0.08, P90, 0.0, P90, 0.08]}, OPT
        )
    with pytest.raises(NotImplementedError, match="thyme"):
        es.saffron(N14, _exp("2pESEEM"), {**OPT, "SimulationMode": "thyme"})
    with pytest.raises(ValueError, match="Exp.tau must be larger"):
        es.saffron(N14, _exp("3pESEEM", tau=0), OPT)
    with pytest.raises(NotImplementedError, match="more than one electron"):
        es.saffron({"S": [0.5, 0.5], "ee": 10, "Nucs": "1H", "A": [5, 2]}, _exp("2pESEEM"), OPT)
    with pytest.raises(ValueError, match="Exp.Field is missing"):
        es.saffron(N14, {"Sequence": "2pESEEM", "dt": 0.01})


def test_sf_pathways():
    assert es.sf_pathways([0.08, 0, 0.08], [0, 1, 0]).tolist() == [[3, 1, 4], [3, 2, 4]]
    assert es.sf_pathways([0.08, 0.08], [1, 1]).tolist() == [[3, 4]]
    assert es.sf_pathways(0.1, 0).tolist() == [[4]]
    p = es.sf_pathways([0.08, 0, 0, 0.08], [0, 1, 2, 0])
    assert p.shape[1] == 4 and all(row[-1] == 4 for row in p)
    with pytest.raises(ValueError, match="same number"):
        es.sf_pathways([0.1, 0.2], [0])


def test_helpers():
    w = es.apowin("ham", 65)
    assert w.max() == 1 and np.isclose(w[0], 0.08) and np.allclose(w, w[::-1])
    assert es.apowin("ham+", 33)[0] == 1 and es.apowin("ham-", 33)[-1] == 1
    np.testing.assert_allclose(es.apowin("han", 11), 0.5 + 0.5 * np.cos(np.pi * np.linspace(-1, 1, 11)))
    assert es.apowin("kai", 10, 3).max() == 1
    with pytest.raises(ValueError, match="Unknown apodization"):
        es.apowin("foo", 10)
    with pytest.raises(ValueError, match="number of input"):
        es.apowin("gau", 10)
    f = es.fdaxis(0.01, 256)
    assert f[128] == 0 and np.isclose(f[-1], 50 - 100 / 256) and np.isclose(f[0], -50)
    np.testing.assert_allclose(es.fdaxis(np.arange(100) * 0.02), es.fdaxis(0.02, 100))
    t = np.arange(200) * 0.01
    y = 3 + 2 * np.exp(-t / 0.5)
    k, c, fit = es.exponfit(t, y, 1)
    np.testing.assert_allclose(k, [2.0], rtol=1e-6)
    np.testing.assert_allclose(c, [3, 2], rtol=1e-6)
    np.testing.assert_allclose(fit, y, atol=1e-6)
    k2, c2, _ = es.exponfit(t, 3 + 2 * np.exp(-t / 0.5) + np.exp(-t / 0.1), 2)
    np.testing.assert_allclose(np.sort(k2), [2, 10], rtol=1e-4)
    knc, cnc, _ = es.exponfit(t, 2 * np.exp(-t / 0.5), 1, noconst=True)
    np.testing.assert_allclose(knc, [2.0], rtol=1e-6)
    assert cnc[0] == 0 and np.isclose(cnc[1], 2)
    kc, cc, fc = es.exponfit(t, np.column_stack([y, 1 - 0.5 * np.exp(-t / 0.3)]), 1)
    assert kc.shape == (2, 1) and cc.shape == (2, 2) and fc.shape == (200, 2)
    np.testing.assert_allclose(kc[:, 0], [2, 1 / 0.3], rtol=1e-5)
    nu = es.larmorfrq("1H", 350)
    assert nu.shape == (1, 1) and np.isclose(nu[0, 0], 14.902, atol=1e-3)
    assert es.larmorfrq("1H,14N", [300, 350]).shape == (2, 2)
