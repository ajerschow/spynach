"""Parity of chili and the angular-momentum helpers with EasySpin 6.0.15: the cases of esdump_chili.m. Spectra of
the general Liouvillian method are compared normwise at 1e-5 (the Lanczos convergence threshold and the numerical
integrals of the equilibrium vector carry 1e-6 of noise); the cases run with EasySpin's fast method (the C kernel
that is not ported) at 1e-3, the agreement EasySpin's own test demands between its two methods; 3j and 6j
symbols, Wigner matrices and tensor conversions at 1e-12. EasySpin's right-to-left continued fraction returns the
201-point test axis of its convergence check instead of the requested points; that case is interpolated."""

import numpy as np
import pytest

import epreasy as es
from epreasy.oracle import assert_matches, fixture_root, load_fixture

ROOT = fixture_root()
ALL = (
    sorted(p for p in ROOT.glob("*/ch_*") if p.is_dir() and (p / "data.h5").is_file() and (p / "meta.json").is_file())
    if ROOT.is_dir()
    else []
)
HELPERS = sorted(p for p in ROOT.glob("*/helpers4") if (p / "data.h5").is_file()) if ROOT.is_dir() else []

pytestmark = pytest.mark.skipif(not ALL, reason=f"no EasySpin chili fixtures under {ROOT}")


def _complex(obj):
    """meta.json carries complex potential coefficients as {"real":, "imag":} pairs."""
    if isinstance(obj, dict) and set(obj) == {"real", "imag"}:
        return complex(obj["real"], obj["imag"])
    if isinstance(obj, dict):
        return {k: _complex(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_complex(v) for v in obj]
    return obj


@pytest.mark.parametrize("case", ALL, ids=lambda p: p.name)
def test_chili(case):
    fx = load_fixture(case)
    inputs = _complex(fx.meta["inputs"])
    opt = dict(inputs["Opt"] or {})
    fast = opt.get("LiouvMethod", "fast" if "LiouvMethod" not in opt else opt["LiouvMethod"]) != "general"
    opt.pop("LiouvMethod", None)
    if "Potential" in inputs["Sys"] if isinstance(inputs["Sys"], dict) else False:
        inputs["Sys"]["Potential"] = np.atleast_2d(np.asarray(inputs["Sys"]["Potential"], dtype=complex))
    r = es.chili(inputs["Sys"], inputs["Exp"], opt)
    x, spec = np.asarray(fx["/spec/x"]).ravel(), np.asarray(fx["/spec/spec"]).ravel()
    assert_matches(r.x, x, rtol=1e-12, atol=0, name="x")
    tol = 1e-3 if fast else 1e-5
    if opt.get("PostConvNucs") is not None:
        tol = 1e-4  # the isotropic post-convolution spectrum comes from garlic's binning
    if spec.size != x.size:  # EasySpin's Lentz=false path returns its 201-point test axis
        spec = np.interp(x, np.linspace(x[0], x[-1], spec.size), spec)
        tol = 1e-2
    assert_matches(r.spec, spec, rtol=tol, atol=tol * 1e-3 * np.abs(spec).max(), name="spec")


@pytest.mark.skipif(not HELPERS, reason="no helpers4 case")
def test_angular_momentum_helpers():
    fx = load_fixture(HELPERS[0])
    jm = np.asarray(fx["/wigner3j/jm"]).reshape(-1, 6)
    vals = np.asarray(fx["/wigner3j/val"]).ravel()
    ours = np.array([es.wigner3j(*row) for row in jm])
    assert_matches(ours, vals, rtol=1e-12, atol=1e-14, name="wigner3j")
    jj = np.asarray(fx["/wigner6j/jj"]).reshape(-1, 6)
    v6 = np.asarray(fx["/wigner6j/val"]).ravel()
    assert_matches(np.array([es.wigner6j(*row) for row in jj]), v6, rtol=1e-12, atol=1e-14, name="wigner6j")
    ang = np.asarray(fx.meta["inputs"]["ang"], dtype=float).ravel()
    for J in (0.5, 1, 1.5, 2, 3, 4, 6):
        tag = f"{J:g}".replace(".", "p")
        assert_matches(es.wignerd(J, *ang), fx[f"/wignerd/D{tag}"], rtol=1e-12, atol=1e-13, name=f"D{tag}")
        assert_matches(es.wignerd(J, ang[1]), fx[f"/wignerd/d{tag}"], rtol=1e-12, atol=1e-13, name=f"d{tag}")
        assert_matches(
            es.wignerd(J, *ang, phase="+"), fx[f"/wignerd/Dplus{tag}"], rtol=1e-12, atol=1e-13, name=f"Dplus{tag}"
        )
    beta = np.asarray(fx["/wignerd/beta"]).ravel()
    el = np.asarray(fx["/wignerd/el"]).reshape(-1, 3)
    for k, (L, m1, m2) in enumerate(el, start=1):
        assert_matches(
            es.wignerd([L, m1, m2], beta),
            np.asarray(fx[f"/wignerd/el{k}"]).ravel(),
            rtol=1e-12,
            atol=1e-13,
            name=f"el{k}",
        )
        assert_matches(
            es.wignerd([L, m1, m2], 0.3, beta, 1.1),
            np.asarray(fx[f"/wignerd/elfull{k}"]).ravel(),
            rtol=1e-12,
            atol=1e-13,
            name=f"elfull{k}",
        )
    Tc = np.asarray(fx["/tensor/Tc"])
    T0, T1, T2 = es.tensor_cart2sph(Tc)
    assert_matches(T0, float(np.asarray(fx["/tensor/T0"]).ravel()[0]), rtol=1e-12, atol=0, name="T0")
    assert_matches(T1, np.asarray(fx["/tensor/T1"]).ravel(), rtol=1e-12, atol=1e-14, name="T1")
    assert_matches(T2, np.asarray(fx["/tensor/T2"]).ravel(), rtol=1e-12, atol=1e-14, name="T2")
    assert_matches(es.tensor_sph2cart(T0, T1, T2), fx["/tensor/back"], rtol=1e-12, atol=1e-14, name="sph2cart")
    T0v, _, T2v = es.tensor_cart2sph([1, 2, 3])
    assert_matches(T0v, float(np.asarray(fx["/tensor/vecT0"]).ravel()[0]), rtol=1e-12, atol=0, name="vecT0")
    assert_matches(T2v, np.asarray(fx["/tensor/vecT2"]).ravel(), rtol=1e-12, atol=1e-14, name="vecT2")
    assert_matches(
        es.tensor_cart2sph(2.5)[0],
        float(np.asarray(fx["/tensor/scalT0"]).ravel()[0]),
        rtol=1e-12,
        atol=0,
        name="scalT0",
    )
