"""Parity of saffron and its helpers with EasySpin 6.0.15: the cases of esdump_saffron.m. Time-domain signals,
their processed spectra and Mims ENDOR spectra are compared normwise at 1e-6 (the frequency-domain binning is
deterministic, so the only noise is the eigendecomposition of the nuclear sub-Hamiltonians); helpers at 1e-12,
exponfit at 1e-6 (two Levenberg-Marquardt implementations converge to the same minimum, not the same path)."""

import numpy as np
import pytest

import epreasy as es
from epreasy.oracle import assert_matches, fixture_root, load_fixture

ROOT = fixture_root()
ALL = (
    sorted(p for p in ROOT.glob("*/pu_*") if p.is_dir() and (p / "data.h5").is_file() and (p / "meta.json").is_file())
    if ROOT.is_dir()
    else []
)
HELPERS = sorted(p for p in ROOT.glob("*/helpers5") if (p / "data.h5").is_file()) if ROOT.is_dir() else []

pytestmark = pytest.mark.skipif(not ALL, reason=f"no EasySpin saffron fixtures under {ROOT}")


def _sequence(seq):
    """A custom sequence from meta.json: pulses are dicts, delays numbers (a 1-element list is a number)."""
    if isinstance(seq, str):
        return seq
    return [s if isinstance(s, dict) else float(np.asarray(s).ravel()[0]) for s in seq]


@pytest.mark.parametrize("case", ALL, ids=lambda p: p.name)
def test_saffron(case):
    fx = load_fixture(case)
    inputs = fx.meta["inputs"]
    exp = dict(inputs["Exp"])
    exp["Sequence"] = _sequence(exp["Sequence"])
    opt = dict(inputs["Opt"] or {})
    r = es.saffron(inputs["Sys"], exp, opt)
    y = np.asarray(fx["/spec/y"])
    tol = 1e-6
    if "/spec/x" in fx:
        assert_matches(np.asarray(r.x), np.asarray(fx["/spec/x"]).ravel(), rtol=1e-12, atol=1e-14, name="x")
        y = y.ravel()
    else:
        assert_matches(np.asarray(r.x[0]), np.asarray(fx["/spec/x1"]).ravel(), rtol=1e-12, atol=1e-14, name="x1")
        assert_matches(np.asarray(r.x[1]), np.asarray(fx["/spec/x2"]).ravel(), rtol=1e-12, atol=1e-14, name="x2")
    assert_matches(np.asarray(r.spec), y, rtol=tol, atol=tol * np.abs(y).max(), name="y")
    if "/info/fd" in fx:
        fd = np.asarray(fx["/info/fd"])
        assert_matches(r.info["fd"], fd.reshape(r.info["fd"].shape), rtol=tol, atol=tol * np.abs(fd).max(), name="fd")
    for axis in ("f", "f1", "f2"):
        if f"/info/{axis}" in fx:
            assert_matches(r.info[axis], np.asarray(fx[f"/info/{axis}"]).ravel(), rtol=1e-12, atol=1e-12, name=axis)


@pytest.mark.skipif(not HELPERS, reason="no helpers5 case")
def test_helpers():
    fx = load_fixture(HELPERS[0])
    assert_matches(es.larmorfrq("1H", 350), fx["/larmorfrq/one"], rtol=1e-12, atol=0, name="larmorfrq")
    assert_matches(
        es.larmorfrq("1H,14N,63Cu,2H", [300, 350, 1200]), fx["/larmorfrq/many"], rtol=1e-12, atol=0, name="larmorfrq"
    )
    for w in ("ham", "han", "bla", "bar", "con", "cos", "wel"):
        assert_matches(es.apowin(w, 64), np.asarray(fx[f"/apowin/{w}"]).ravel(), rtol=1e-12, atol=1e-14, name=w)
        assert_matches(
            es.apowin(w + "+", 65), np.asarray(fx[f"/apowin/{w}_plus"]).ravel(), rtol=1e-12, atol=1e-14, name=w
        )
        assert_matches(
            es.apowin(w + "-", 63), np.asarray(fx[f"/apowin/{w}_minus"]).ravel(), rtol=1e-12, atol=1e-14, name=w
        )
    assert_matches(es.apowin("kai", 64, 3), np.asarray(fx["/apowin/kai"]).ravel(), rtol=1e-12, atol=1e-14, name="kai")
    assert_matches(es.apowin("gau", 64, 0.4), np.asarray(fx["/apowin/gau"]).ravel(), rtol=1e-12, atol=1e-14, name="gau")
    assert_matches(es.apowin("exp", 65, 2), np.asarray(fx["/apowin/exp"]).ravel(), rtol=1e-12, atol=1e-14, name="exp")
    assert_matches(
        es.apowin("exp+", 64, 2), np.asarray(fx["/apowin/exp_plus"]).ravel(), rtol=1e-12, atol=1e-14, name="exp+"
    )
    assert_matches(es.fdaxis(0.01, 256), np.asarray(fx["/fdaxis/even"]).ravel(), rtol=1e-12, atol=1e-12, name="fdaxis")
    assert_matches(es.fdaxis(0.01, 255), np.asarray(fx["/fdaxis/odd"]).ravel(), rtol=1e-12, atol=1e-12, name="fdaxis")
    assert_matches(
        es.fdaxis(np.arange(100) * 0.02),
        np.asarray(fx["/fdaxis/fromaxis"]).ravel(),
        rtol=1e-12,
        atol=1e-12,
        name="fdaxis",
    )
    t = np.asarray(fx["/exponfit/t"]).ravel()
    for tag, y, n, kw in (("1", "y", 1, {}), ("n", "yn", 1, {}), ("nc", "y", 1, {"noconst": True}), ("2", "yd", 2, {})):
        k, c, fit = es.exponfit(t, np.asarray(fx[f"/exponfit/{y}"]).ravel(), n, **kw)
        assert_matches(
            np.sort(k), np.sort(np.asarray(fx[f"/exponfit/k{tag}"]).ravel()), rtol=1e-6, atol=1e-9, name=f"k{tag}"
        )
        assert_matches(fit, np.asarray(fx[f"/exponfit/fit{tag}"]).ravel(), rtol=1e-6, atol=1e-8, name=f"fit{tag}")
        if n == 1:
            assert_matches(c, np.asarray(fx[f"/exponfit/c{tag}"]).ravel(), rtol=1e-6, atol=1e-8, name=f"c{tag}")
    k, c, fit = es.exponfit(t, np.asarray(fx["/exponfit/y2"]), 1)
    assert_matches(k, fx["/exponfit/kcols"], rtol=1e-6, atol=1e-9, name="kcols")
    assert_matches(c, fx["/exponfit/ccols"], rtol=1e-6, atol=1e-8, name="ccols")
    assert_matches(fit, fx["/exponfit/fitcols"], rtol=1e-6, atol=1e-8, name="fitcols")
    for tag, t_, inc in (
        ("p3", [0.08, 0, 0.08], [0, 1, 0]),
        ("p2", [0.08, 0.08], [1, 1]),
        ("p4", [0.08, 0, 0, 0.08], [0, 1, 2, 0]),
        ("p4b", [0.08, 0, 0, 0.08], [0, 1, 1, 0]),
        ("p5", [0.1, 0.2, 0.1, 0.2, 0.1], [0, 1, 0, -1, 0]),
        ("p1", 0.1, 0),
    ):
        ours = es.sf_pathways(t_, inc)
        assert ours.tolist() == np.asarray(fx[f"/sf_pathways/{tag}"]).astype(int).reshape(ours.shape).tolist()
