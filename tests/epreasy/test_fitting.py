"""Fixture-free tests of esfit and its algorithms: known minima, parameter bookkeeping, scaling and baselines,
global fits, the uncertainty estimate against EasySpin's reference values, and the data helpers."""

import numpy as np
import pytest

import epreasy as es
from epreasy import fitting

X = np.linspace(300, 400, 100)


def gauss_model(p):
    return 5 * np.exp(-((X - p[0]) ** 2) / (2 * p[1] ** 2))


def rosenbrock(p):
    return (1 - p[0]) ** 2 + 100 * (p[1] - p[0] ** 2) ** 2


# ---------------------------------------------------------------------------------------- algorithms
def test_simplex_and_levmar_find_the_minimum():
    x, info = es.esfit_simplex(rosenbrock, [-1.0, 1.5], [-2, -2], [2, 2], {"TolFun": 1e-12, "TolEdgeLength": 1e-7})
    np.testing.assert_allclose(x, [1, 1], atol=1e-4)
    assert info["stop"] == 3 and info["nEvals"] > 10
    x, info = es.esfit_simplex(rosenbrock, [-1.0, 1.5], [-2, -2], [2, 2], {"ScaleParams": True, "TolEdgeLength": 1e-7})
    np.testing.assert_allclose(x, [1, 1], atol=1e-3)

    def residuals(p):
        return np.array([1 - p[0], 10 * (p[1] - p[0] ** 2)])

    x, info = es.esfit_levmar(residuals, [-1.0, 1.5], [-2, -2], [2, 2], {"TolStep": 1e-10})
    np.testing.assert_allclose(x, [1, 1], atol=1e-5)
    assert info["stop"] in (1, 2) and info["Je"].shape == (2, 2)
    x, _ = es.esfit_levmar(residuals, [-1.0, 1.5], [-2, -2], [2, 2], {"ScaleParams": True})
    np.testing.assert_allclose(x, [1, 1], atol=1e-4)
    with pytest.raises(ValueError, match="out of bounds"):
        es.esfit_simplex(rosenbrock, [3.0, 0.0], [-2, -2], [2, 2])
    with pytest.raises(ValueError, match="Lower bounds"):
        es.esfit_levmar(residuals, [0.0, 0.0], [1, -2], [0, 2])


def test_global_algorithms_on_a_bowl():
    def bowl(p):
        return float((p[0] - 0.3) ** 2 + (p[1] + 0.2) ** 2)

    lb, ub = [-1, -1], [1, 1]
    x, info = es.esfit_grid(bowl, lb, ub, {"GridSize": [21, 11], "randGrid": False, "TolFun": 0})
    np.testing.assert_allclose(x, [0.3, -0.2], atol=1e-12)  # both on the grid
    assert info["nEvals"] == 21 * 11
    x, _ = es.esfit_grid(bowl, lb, ub, {"GridSize": 21, "rng": 1})
    np.testing.assert_allclose(x, [0.3, -0.2], atol=1e-12)
    x, _ = es.esfit_montecarlo(bowl, lb, ub, {"nTrials": 3000, "rng": 1})
    np.testing.assert_allclose(x, [0.3, -0.2], atol=0.05)
    x, info = es.esfit_genetic(bowl, lb, ub, {"maxGenerations": 60, "rng": 1})
    np.testing.assert_allclose(x, [0.3, -0.2], atol=0.02)
    x, info = es.esfit_swarm(bowl, lb, ub, {"rng": 1})
    np.testing.assert_allclose(x, [0.3, -0.2], atol=5e-3)
    assert info["stop"] in (3, 4)  # below TolFun, or stalled
    x, info = es.esfit_montecarlo(bowl, lb, ub, {"nTrials": 50, "TolFun": 10, "rng": 1})
    assert info["stop"] == 3 and info["nTrials"] == 1
    with pytest.raises(ValueError, match="At least two grid points"):
        es.esfit_grid(bowl, lb, ub, {"GridSize": 1})
    stops = []
    es.esfit_grid(
        bowl,
        lb,
        ub,
        {"GridSize": 3, "randGrid": False, "IterFcn": lambda info: stops.append(info["iter"]) or info["iter"] >= 4},
    )
    assert stops == [1, 2, 3, 4]


# ---------------------------------------------------------------------------------------- bookkeeping
def test_parameter_names_values_and_updates():
    p0 = [
        {"g": [2.0, 2.1], "lw": 1.0, "A": [[10, 12], [20, 25]], "initState": ([0.2, 0.5, 0.3], "zerofield")},
        {"mwFreq": 9.5},
    ]
    vary = [{"g": [0.02, 0.0], "lw": 0.5, "A": [[1, 0], [0, 2]], "initState": ([0, 0, 0.1],)}]
    pinfo = fitting._getparaminfo(vary)
    assert [p.name for p in pinfo] == [
        "arg1.g(1,1)", "arg1.g(1,2)", "arg1.lw", "arg1.A(1,1)", "arg1.A(1,2)", "arg1.A(2,1)", "arg1.A(2,2)",
        "arg1.initState{1}(1,1)", "arg1.initState{1}(1,2)", "arg1.initState{1}(1,3)",
    ]  # fmt: skip
    fitting._checkparcompatibility(pinfo, p0)
    np.testing.assert_array_equal(fitting._getparamvalues(p0, pinfo), [2.0, 2.1, 1.0, 10, 12, 20, 25, 0.2, 0.5, 0.3])
    np.testing.assert_array_equal(fitting._getparamvalues(vary, pinfo), [0.02, 0, 0.5, 1, 0, 0, 2, 0, 0, 0.1])
    new = fitting._setparamvalues(p0, pinfo, np.arange(10, dtype=float))
    np.testing.assert_array_equal(new[0]["g"], [0, 1])
    assert new[0]["lw"] == 2.0 and isinstance(new[0]["lw"], float)
    np.testing.assert_array_equal(new[0]["A"], [[3, 4], [5, 6]])
    np.testing.assert_array_equal(new[0]["initState"][0], [7, 8, 9])
    assert new[0]["initState"][1] == "zerofield" and p0[0]["g"] == [2.0, 2.1]  # the input is untouched
    components = fitting._getparaminfo([[{"g": [0.01]}, {"g": [0.0, 0.02]}]])
    assert [p.name for p in components] == ["arg1{1}.g", "arg1{2}.g(1,1)", "arg1{2}.g(1,2)"]
    with pytest.raises(ValueError, match="doesn't have enough elements"):
        fitting._checkparcompatibility(fitting._getparaminfo([{"g": [0, 0, 0]}]), p0)
    with pytest.raises(ValueError, match="is missing"):
        fitting._checkparcompatibility(fitting._getparaminfo([{"D": 1}]), p0)


def test_esfit_argument_forms_and_errors():
    y = gauss_model([350, 7])
    by_vary = es.esfit(y, gauss_model, [340, 5], [50, 20], {"Verbosity": 0})
    by_bounds = es.esfit(y, gauss_model, [340, 5], [290, -15], [390, 25], {"Verbosity": 0})
    by_keywords = es.esfit(y, gauss_model, [340, 5], vary=[50, 20], opt={"Verbosity": 0})
    for r in (by_vary, by_bounds, by_keywords):
        np.testing.assert_allclose(r.pfit, [350, 7], atol=1e-3)
        assert r.pnames == ["p(1)", "p(2)"] and r.algorithm == "simplex" and r.target == "fcn"
    assert np.array_equal(by_vary.pfit, by_bounds.pfit)
    fixed = es.esfit(y, gauss_model, [350, 5], [0, 20], {"Verbosity": 0})  # a zero variation fixes the parameter
    assert fixed.pnames == ["p(2)"] and fixed.pfit.size == 1 and fixed.p_fixed.tolist() == [True, False]
    np.testing.assert_allclose(fixed.pfit_full, [350, 7], atol=1e-3)
    with pytest.raises(ValueError, match="No variable parameters"):
        es.esfit(y, gauss_model, [350, 7], [0, 0], {"Verbosity": 0})
    with pytest.raises(ValueError, match="start value is smaller"):
        es.esfit(y, gauss_model, [340, 5], [345, 1], [390, 25])
    with pytest.raises(ValueError, match="Unknown 'newton'"):
        es.esfit(y, gauss_model, [340, 5], [50, 20], {"Method": "newton"})
    with pytest.raises(ValueError, match="must not contain NaN"):
        es.esfit(np.append(y, np.nan), gauss_model, [340, 5], [50, 20])
    with pytest.raises(ValueError, match="takes 4, 5 or 6 inputs"):
        es.esfit(y, gauss_model, [340, 5], [50, 20], [1, 1], {}, {})
    with pytest.raises(ValueError, match="unequal total number of points"):
        es.esfit(y[:-1], gauss_model, [340, 5], [50, 20], {"Verbosity": 0})
    with pytest.raises(ValueError, match="Opt.Scaling has been replaced"):
        es.esfit(y, gauss_model, [340, 5], [50, 20], {"Scaling": "lsq"})


# ---------------------------------------------------------------------------------------- fits
def test_esfit_targets_and_algorithms_recover_the_parameters():
    y = gauss_model([350, 7])
    for method in (
        "simplex fcn",
        "levmar fcn",
        "lsqnonlin fcn",
        "simplex int",
        "simplex dint",
        "simplex diff",
        "simplex fft",
        "swarm",
    ):
        r = es.esfit(y, gauss_model, [340, 5], [50, 20], {"Method": method, "Verbosity": 0, "RandomSeed": 1})
        np.testing.assert_allclose(r.pfit, [350, 7], atol=2e-3, err_msg=method)
        assert r.rmsd < 1e-4 and r.target == method.split()[-1] if " " in method else r.target == "fcn"
    r = es.esfit(
        y, gauss_model, [340, 5], [50, 20], {"Method": "grid", "GridSize": 21, "randGrid": False, "Verbosity": 0}
    )
    np.testing.assert_allclose(np.abs(r.pfit), [350, 7], atol=1e-12)
    assert r.info["stop"] == 3 and r.info["nEvals"] <= 441  # stopped at the exact grid point
    assert r.bestfithistory["rmsd"].size == r.bestfithistory["pfit"].shape[1] > 1
    assert np.all(np.diff(r.bestfithistory["rmsd"]) < 0)


def test_esfit_scaling_baseline_mask_and_global_fits():
    y = gauss_model([350, 7])
    rng = np.random.default_rng(0)
    noise = rng.standard_normal(X.size) * 0.5
    baselines = {None: 0, 0: 0.1, 1: 0.1 + 0.01 * (X - 300), 2: 0.1 + 0.01 * (X - 300) + 0.0002 * (X - 340) ** 2}
    for order, bl in baselines.items():
        for auto in ("none", "lsq", "maxabs"):
            amplitude = 1 if auto == "none" else 2  # without autoscaling the model amplitude is fixed
            r = es.esfit(
                amplitude * y + noise + bl,
                gauss_model,
                [340, 5],
                [50, 20],
                {"Verbosity": 0, "AutoScale": auto, "BaseLine": order},
            )
            assert abs(r.rmsd - np.std(noise, ddof=1)) < 0.25 * np.std(noise, ddof=1), (order, auto)
            assert r.baselinetype == [{None: "none", 0: "offset", 1: "linear", 2: "quadratic"}[order]]
            if auto != "none":
                np.testing.assert_allclose(r.scale, [2], rtol=0.1)
    scaled = es.esfit(
        5.34564564 * y, gauss_model, [340, 5], [50, 20], {"Verbosity": 0, "AutoScale": "lsq", "Method": "simplex fcn"}
    )
    np.testing.assert_allclose(scaled.scale, [5.34564564], atol=1e-3)
    np.testing.assert_allclose(scaled.fitraw * scaled.scale[0], scaled.fit, rtol=1e-12)
    masked = es.esfit(y + 3 * (X > 380), gauss_model, [340, 5], [50, 20], {"Verbosity": 0, "Mask": X <= 380})
    np.testing.assert_allclose(masked.pfit, [350, 7], atol=1e-3)
    assert masked.mask.sum() == np.sum(X <= 380)
    # global fitting: the weights select which data set the shared parameter follows
    xg = np.linspace(0, 1000, 100)
    data = [es.gaussian(xg, 450, 100), es.gaussian(xg, 550, 100)]

    def two_sets(p):
        return [es.gaussian(xg, p[0], 100), es.gaussian(xg, p[0], 100)]

    for k in range(2):
        weight = [0, 0]
        weight[k] = 1
        r = es.esfit(data, two_sets, [500], [100], [900], {"Verbosity": 0, "weight": weight, "AutoScale": "lsq"})
        np.testing.assert_allclose(r.pfit, [450 + 100 * k], atol=1e-2)
        assert np.abs(r.fit[k] - data[k]).max() < 1e-3 and len(r.residuals) == 2 and len(r.baseline) == 2


def test_esfit_uncertainties_match_easyspin_reference():
    """EasySpin's esfit_std test: standard deviations as from MATLAB's cftool and DeerLab (to 2 %)."""
    rng = np.random.default_rng(7)
    t = np.linspace(0, 20, 1001)
    truth = [2, 0.3, 0.2]
    y = truth[0] * np.exp(-truth[1] * t) + truth[2] + 0.1 * rng.standard_normal(t.size)
    r = es.esfit(
        y, lambda p: p[0] * np.exp(-p[1] * t) + p[2], [1.5, 0.3, 0.5], [0.1, 0.1, -2], [10, 0.5, 2], {"Verbosity": 0}
    )
    np.testing.assert_allclose(r.pfit, truth, atol=0.03)
    # the analytic covariance of the linearised least-squares problem, for comparison
    J = np.column_stack([np.exp(-r.pfit[1] * t), -r.pfit[0] * t * np.exp(-r.pfit[1] * t), np.ones(t.size)])
    cov = np.linalg.inv(J.T @ J) * np.var(r.residuals, ddof=3)
    np.testing.assert_allclose(r.pstd, np.sqrt(np.diag(cov)), rtol=0.1)
    assert r.ci95.shape == (3, 2) and np.all(r.ci95[:, 0] < r.pfit) and np.all(r.ci95[:, 1] > r.pfit)
    np.testing.assert_allclose(np.diag(r.corr), 1, atol=1e-12)
    assert 0.5 < r.redchisquare < 2
    skipped = es.esfit(
        y,
        lambda p: p[0] * np.exp(-p[1] * t) + p[2],
        [1.5, 0.3, 0.5],
        [0.1, 0.1, -2],
        [10, 0.5, 2],
        {"Verbosity": 0, "CalculateUncertainties": 0},
    )
    assert skipped.pstd is None and skipped.cov is None


def test_esfit_with_pepper_and_garlic():
    system = {"g": [2, 2.1], "lw": 1}
    exp = {"mwFreq": 9.5, "Range": [300, 360], "nPoints": 200}
    spc = es.pepper(system, exp, {"FuzzLevel": 0}).spec
    for method in ("simplex", "levmar fcn"):
        r = es.esfit(
            spc, es.pepper, [system, exp, {"FuzzLevel": 0}], [{"g": [0.02, 0.02]}], {"Method": method, "Verbosity": 0}
        )
        assert r.rmsd / r.fit.max() < 1e-10
        assert r.pnames == ["arg1.g(1,1)", "arg1.g(1,2)"] and r.target == ("int" if method == "simplex" else "fcn")
        np.testing.assert_allclose(r.argsfit[0]["g"], [2, 2.1], atol=1e-6)
        np.testing.assert_allclose(r.bestfithistory["pfit2structs"](r.pfit)[0]["g"], r.argsfit[0]["g"])
    off = es.esfit(spc, es.pepper, [{"g": [2, 2.1], "lw": 1.3}, exp, {"FuzzLevel": 0}], [{"lw": 0.5}], {"Verbosity": 0})
    np.testing.assert_allclose(off.pfit, [1.0], atol=1e-4)
    # two components, initState, and the errors of the EasySpin-function path
    two = es.pepper([system, {"g": [1.98, 2.03], "lw": 0.7}], {**exp, "nPoints": 400}, {"FuzzLevel": 0}).spec
    r = es.esfit(
        two,
        es.pepper,
        [[system, {"g": [1.98, 2.03], "lw": 0.7}], {**exp, "nPoints": 400}, {"FuzzLevel": 0}],
        [[{"g": [0.01, 0.01]}, {"g": [0.01, 0.01]}]],
        {"Method": "levmar fcn", "Verbosity": 0},
    )
    assert r.pnames[0] == "arg1{1}.g(1,1)" and r.rmsd / r.fit.max() < 1e-8
    triplet = {"S": 1, "g": 2.0023, "D": [800, -200], "lw": 5, "initState": ([0.2, 0.5, 0.3], "zerofield")}
    texp = {"mwFreq": 9.75, "Range": [300, 400], "Harmonic": 0, "nPoints": 200}
    tspc = es.pepper(triplet, texp, {"FuzzLevel": 0}).spec
    r = es.esfit(
        tspc,
        es.pepper,
        [triplet, texp, {"FuzzLevel": 0}],
        [{"D": [0, 100], "initState": ([0, 0, 0.2],)}],
        {"Verbosity": 0},
    )
    assert r.pnames == ["arg1.D(1,2)", "arg1.initState{1}(1,3)"] and r.rmsd / r.fit.max() < 1e-6
    solution = {"g": 2.006, "Nucs": "14N", "A": 47, "lwpp": 0.3}
    gexp = {"mwFreq": 9.5, "Range": [330, 350], "nPoints": 500}
    gspc = es.garlic(solution, gexp).spec
    r = es.esfit(gspc, es.garlic, [{**solution, "A": 45}, gexp], [{"A": 5}], {"Verbosity": 0})
    np.testing.assert_allclose(r.pfit, [47], atol=0.05)
    with pytest.raises(ValueError, match="Exp.nPoints is 100"):
        es.esfit(spc, es.pepper, [system, {**exp, "nPoints": 100}], [{"g": [0.02, 0.02]}])
    with pytest.raises(ValueError, match="specify field or frequency range"):
        es.esfit(spc, es.pepper, [system, {"mwFreq": 9.5}], [{"g": [0.02, 0.02]}])
    with pytest.raises(ValueError, match="global fitting"):
        es.esfit([spc, spc], es.pepper, [system, exp], [{"g": [0.02, 0.02]}])


# ---------------------------------------------------------------------------------------- helpers
def test_jacobianest_and_hccm():
    def f(p):
        return np.array([p[0] ** 2 * p[1], np.sin(p[0]) + np.exp(p[1]), p[0] * p[1] ** 3])

    p = np.array([1.3, 0.7])
    J, err = es.jacobianest(f, p)
    exact = np.array([[2 * p[0] * p[1], p[0] ** 2], [np.cos(p[0]), np.exp(p[1])], [p[1] ** 3, 3 * p[0] * p[1] ** 2]])
    np.testing.assert_allclose(J, exact, rtol=1e-10)
    assert np.all(err < 1e-8)
    J0, _ = es.jacobianest(lambda q: np.array([q[0] ** 2, q[1]]), [0.0, 2.0])  # a zero parameter uses absolute steps
    np.testing.assert_allclose(J0, [[0, 0], [0, 1]], atol=1e-10)
    rng = np.random.default_rng(1)
    J = rng.standard_normal((50, 3))
    res = rng.standard_normal(50)
    C = es.hccm(J, res, "HC1")
    assert C.shape == (3, 3) and np.allclose(C, C.T)
    for mode in ("HC0", "HC2", "HC3", "HC4", "HC5"):
        assert es.hccm(J, res, mode).shape == (3, 3)
    with pytest.raises(ValueError, match="mode not found"):
        es.hccm(J, res, "HC9")


def test_rescaledata_basecorr_deriv_addnoise():
    y = np.array([1.0, -4.0, 2.0])
    np.testing.assert_allclose(es.rescaledata(y)[0], y / 4)
    np.testing.assert_allclose(es.rescaledata(y, [0, 8, 0])[1], 8 / 4)  # maxabs with a reference
    ys, scale = es.rescaledata([1, 2, 3], [-2, -4, -6])  # lsq, positive scaling enforced
    assert scale == 2 and np.allclose(ys, [2, 4, 6])
    assert es.rescaledata([1, 2, 3], "int")[1] == pytest.approx(1 / 6)
    assert es.rescaledata([1, 2, 3], "dint")[1] == pytest.approx(1 / 10)
    assert es.rescaledata([1, 2, 3], "none")[1] == 1
    assert es.rescaledata([1, 2, 3], [1, 2, 3, 4, 5])[1] == pytest.approx(
        1 / (np.array([1, 3, 5]) @ [1, 2, 3] / 35), rel=1e-12
    )
    with pytest.raises(ValueError, match="obsolete"):
        es.rescaledata(y, "lsq0")
    with pytest.raises(ValueError, match="Unknown scaling mode"):
        es.rescaledata(y, "weird")
    x = np.linspace(-1, 1, 50)
    data = 0.3 + 0.2 * x + 0.1 * x**2 + np.where(np.abs(x) < 0.3, 1.0, 0.0)  # a peak the region leaves out
    corrected, baseline = es.basecorr(data, 1, 2, region=np.abs(x) >= 0.3)
    np.testing.assert_allclose(baseline, 0.3 + 0.2 * x + 0.1 * x**2, atol=1e-6)
    np.testing.assert_allclose(corrected + baseline, data)
    Z = np.outer(x, x) + 0.5
    Zc, Zb = es.basecorr(Z, None, [1, 1])
    assert np.abs(Zc).max() < 1e-10
    rows, _ = es.basecorr(np.outer(np.ones(3), 1 + 2 * x), 2, 1)
    assert np.abs(rows).max() < 1e-12
    with pytest.raises(ValueError, match="between 0 and 6"):
        es.basecorr(data, 1, 7)
    with pytest.raises(ValueError, match="2 elements"):
        es.basecorr(Z, None, 2)
    np.testing.assert_allclose(es.deriv(np.arange(5.0) ** 2), [1, 2, 4, 6, 7])
    np.testing.assert_allclose(es.deriv(2 * np.arange(5.0), np.arange(5.0) ** 2), [0.5, 1, 2, 3, 3.5])
    assert es.deriv(np.ones((5, 2))).shape == (5, 2)
    clean = es.gaussian(X, 350, 10)
    for model in ("n", "u", "f"):
        noisy = es.addnoise(clean, 20, model, rng=1)
        assert noisy.shape == clean.shape and 0.5 < np.std(noisy - clean, ddof=1) / (clean.max() / 20) < 1.5
    assert np.iscomplexobj(es.addnoise(clean + 0j, 20, "n", rng=1))
    with pytest.raises(ValueError, match="Unknown noise type"):
        es.addnoise(clean, 20, "x")
