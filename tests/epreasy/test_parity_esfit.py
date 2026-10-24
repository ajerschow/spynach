"""Parity of esfit, rescaledata, basecorr and deriv with EasySpin 6.0.15: the cases of esdump_esfit.m. The data
that EasySpin fitted (noise included) are read from the fixture, the model is rebuilt from its name. Fits of
analytic models follow the same deterministic algorithm steps and agree at 1e-6 (parameters, rmsd, fit) and
1e-3 (uncertainties, from the Romberg Jacobian); fits of pepper spectra, whose model agrees at 1e-7, at 1e-3."""

import numpy as np
import pytest

import epreasy as es
from epreasy.oracle import assert_matches, fixture_root, load_fixture

ROOT = fixture_root()
ALL = (
    sorted(p for p in ROOT.glob("*/ef_*") if p.is_dir() and (p / "data.h5").is_file() and (p / "meta.json").is_file())
    if ROOT.is_dir()
    else []
)
HELPERS = sorted(p for p in ROOT.glob("*/helpers3") if (p / "data.h5").is_file()) if ROOT.is_dir() else []

pytestmark = pytest.mark.skipif(not ALL, reason=f"no EasySpin esfit fixtures under {ROOT}")


def _models(extra: dict) -> dict:
    x = np.asarray(extra.get("x", []), dtype=float).ravel()
    t = np.asarray(extra.get("t", []), dtype=float).ravel()
    return {
        "gaussmodel": lambda p: 5 * np.exp(-((x - p[0]) ** 2) / (2 * p[1] ** 2)),
        "globalmodel": lambda p: [es.gaussian(x, p[0], 100), es.gaussian(x, p[0], 100)],
        "globalmodel2": lambda p: [es.gaussian(x, p[0], p[1]), es.gaussian(x, p[0], p[2])],
        "expmodel": lambda p: p[0] * np.exp(-p[1] * t) + p[2],
        "pepper": es.pepper,
    }


def _tuple_initstate(obj):
    """JSON lists for EasySpin cell arrays: initState {state, basis} and the vary cell {state} become tuples."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            out[k] = (
                tuple(v)
                if k == "initState" and isinstance(v, list) and (len(v) == 1 or isinstance(v[-1], str))
                else _tuple_initstate(v)
            )
        return out
    if isinstance(obj, list):
        return [_tuple_initstate(v) for v in obj]
    return obj


def _args(inputs: dict, key: str):
    val = inputs.get(key)
    if val is None or (isinstance(val, list) and len(val) == 0):
        return None
    if inputs["fcn"] == "pepper":
        return _tuple_initstate(val)
    return np.asarray(val, dtype=float).ravel()


@pytest.mark.parametrize("case", ALL, ids=lambda p: p.name)
def test_esfit(case):
    fx = load_fixture(case)
    inputs = fx.meta["inputs"]
    data = []
    k = 1
    while f"/input/data{k}" in fx:
        data.append(np.asarray(fx[f"/input/data{k}"]).ravel())
        k += 1
    fcn = _models(inputs.get("extra") or {})[inputs["fcn"]]
    opt = dict(inputs["FitOpt"])
    # EasySpin 6.0.15 swaps the two targets: its 'diff' keyword takes the Fourier magnitude of the residuals and
    # 'fft' the derivative; epreasy uses the names as meant, so the keyword is swapped to compare like with like
    method = opt.get("Method", "")
    if "diff" in method.split():
        opt["Method"] = method.replace("diff", "fft")
    elif "fft" in method.split():
        opt["Method"] = method.replace("fft", "diff")
    if "Mask" in opt:
        opt["Mask"] = np.asarray(opt["Mask"]).ravel().astype(bool)
    p0 = _args(inputs, "p0")
    vary, lb, ub = _args(inputs, "vary"), _args(inputs, "lb"), _args(inputs, "ub")
    if vary is not None:
        r = es.esfit(data if len(data) > 1 else data[0], fcn, p0, vary, opt)
    else:
        r = es.esfit(data if len(data) > 1 else data[0], fcn, p0, lb, ub, opt)
    model_tol = 1e-3 if inputs["fcn"] == "pepper" else 1e-6
    pnames = inputs["pnames"] if isinstance(inputs["pnames"], list) else [inputs["pnames"]]
    assert r.pnames == pnames
    assert r.baselinetype == (
        inputs["baselinetype"] if isinstance(inputs["baselinetype"], list) else [inputs["baselinetype"]]
    )
    assert_matches(r.p_start, np.asarray(fx["/result/p_start"]).ravel(), rtol=1e-12, atol=0, name="p_start")
    pfit_ref = np.asarray(fx["/result/pfit"]).ravel()
    if "grid" in opt.get("Method", ""):
        pfit_ref = np.abs(pfit_ref)  # the symmetric Gaussian width: the grid holds +-7 and the first hit may differ
        r.pfit = np.abs(r.pfit)
    assert_matches(r.pfit, pfit_ref, rtol=model_tol, atol=0, name="pfit")
    data_scale = max(np.abs(d).max() for d in data)
    # noise-free fits end at rmsd ~1e-6 of the data on either side, at whichever step the tolerances stop them
    rmsd_ref = float(np.asarray(fx["/result/rmsd"]).ravel()[0])
    assert_matches(r.rmsd, rmsd_ref, rtol=model_tol, atol=10 * model_tol * data_scale, name="rmsd")
    assert_matches(r.scale, np.asarray(fx["/result/scale"]).ravel(), rtol=model_tol, atol=0, name="scale")
    fits = [r.fit] if len(data) == 1 else r.fit
    for k, fit in enumerate(fits, start=1):
        assert_matches(
            fit,
            np.asarray(fx[f"/result/fit{k}"]).ravel(),
            rtol=10 * model_tol,
            atol=1e-12 * np.abs(fit).max(),
            name=f"fit{k}",
        )
    if "/result/pstd" in fx:
        unc_tol = 1e-2 if inputs["fcn"] == "pepper" else 1e-3  # the Romberg Jacobian amplifies the model differences
        unc_atol = 1e-6 * np.abs(r.pfit).max()  # noise-free fits: both sides report numerical-noise uncertainties
        assert_matches(r.pstd, np.asarray(fx["/result/pstd"]).ravel(), rtol=unc_tol, atol=unc_atol, name="pstd")
        assert_matches(
            r.ci95, np.asarray(fx["/result/ci95"]).reshape(r.ci95.shape), rtol=unc_tol, atol=unc_atol, name="ci95"
        )
        corr_ref = np.asarray(fx["/result/corr"]).reshape(r.corr.shape)
        if np.all(np.isfinite(corr_ref)):  # an exact fit has zero variances and no correlations
            assert_matches(r.corr, corr_ref, rtol=1e-2, atol=1e-3, name="corr")


@pytest.mark.skipif(not HELPERS, reason="no helpers3 case")
def test_helpers():
    fx = load_fixture(HELPERS[0])
    inputs = fx.meta["inputs"]
    y, yref = np.asarray(fx["/rescale/y"]).ravel(), np.asarray(fx["/rescale/yref"]).ravel()
    region = np.asarray(inputs["region"]).ravel().astype(bool)
    checks = {
        "maxabs": es.rescaledata(y, "maxabs"),
        "maxabs_ref": es.rescaledata(y, yref, "maxabs"),
        "lsq": es.rescaledata(y, yref, "lsq"),
        "int": es.rescaledata(y, "int"),
        "dint": es.rescaledata(y, "dint"),
        "lsq_region": es.rescaledata(y, yref, "lsq", region),
    }
    for name, (ys, scale) in checks.items():
        assert_matches(ys, np.asarray(fx[f"/rescale/{name}"]).ravel(), rtol=1e-12, atol=0, name=name)
        assert_matches(
            scale, float(np.asarray(fx[f"/rescale/{name}_scale"]).ravel()[0]), rtol=1e-12, atol=0, name=name + " scale"
        )
    yb = np.asarray(fx["/basecorr/y"]).ravel()
    for n in range(4):
        corr, base = es.basecorr(yb, 1, n)
        assert_matches(corr, np.asarray(fx[f"/basecorr/corr{n}"]).ravel(), rtol=1e-10, atol=1e-12, name=f"basecorr {n}")
        assert_matches(base, np.asarray(fx[f"/basecorr/base{n}"]).ravel(), rtol=1e-10, atol=1e-12, name=f"baseline {n}")
    corr, base = es.basecorr(yb, 1, 2, region)
    assert_matches(
        corr, np.asarray(fx["/basecorr/corr2_region"]).ravel(), rtol=1e-10, atol=1e-12, name="basecorr region"
    )
    Z = np.asarray(fx["/basecorr/Z"])
    for name, args in (("Z2d", (None, [1, 2])), ("Zrows", (2, 1)), ("Zcols", (1, 3))):
        corr, base = es.basecorr(Z, *args)
        assert_matches(corr, fx[f"/basecorr/{name}"], rtol=1e-9, atol=1e-10, name=name)
        assert_matches(base, fx[f"/basecorr/{name}_base"], rtol=1e-9, atol=1e-10, name=name + " base")
    yd = np.asarray(fx["/deriv/y"]).ravel()
    x = np.asarray(inputs["x"], dtype=float).ravel()
    assert_matches(es.deriv(yd), np.asarray(fx["/deriv/d"]).ravel(), rtol=1e-12, atol=0, name="deriv")
    assert_matches(es.deriv(x, yd), np.asarray(fx["/deriv/dx"]).ravel(), rtol=1e-12, atol=0, name="deriv x")
