"""Parity of pepper, garlic, fastmotion, hamsymm, fieldmod and equivcouple with EasySpin 6.0.15: the cases of
esdump_pepper.m. pepper spectra are compared normwise at 1e-5 relative: the resonance positions of both sides
carry the cancellation error of the closed-form cubic roots (about 1e-6 mT; easyspyn polishes its roots, EasySpin's
MEX does not), which the sharp turning points of a powder spectrum amplify to a few 1e-6 of the norm. garlic
spectra agree at 1e-8; fast-motion widths, symmetry frames, modulation and coupling at 1e-12."""

import numpy as np
import pytest

import easyspyn as es
from easyspyn.oracle import assert_matches, fixture_root, load_fixture

ROOT = fixture_root()
ALL = (
    sorted(p for p in ROOT.glob("*/*") if p.is_dir() and (p / "data.h5").is_file() and (p / "meta.json").is_file())
    if ROOT.is_dir()
    else []
)
PEPPER = [p for p in ALL if p.name.startswith("pp_")]
GARLIC = [p for p in ALL if p.name.startswith("gl_")]
ONE = {p.name: p for p in ALL}
GROUPS = ["C1", "Ci", "C2h", "D2h", "C4h", "D4h", "S6", "D3d", "C6h", "D6h", "Th", "Oh", "Dinfh", "O3"]

pytestmark = pytest.mark.skipif(not ALL, reason=f"no EasySpin fixtures under {ROOT}")


def _opt(opt: dict) -> dict:
    out = dict(opt or {})
    if "HybridCoreNuclei" in out:
        out["HybridCoreNuclei"] = np.atleast_1d(out["HybridCoreNuclei"]).tolist()
    if "GridSize" in out:
        out["GridSize"] = np.atleast_1d(out["GridSize"]).tolist()
    return out


def _sys(sysd: dict) -> dict:
    out = dict(sysd)
    if "n" in out:
        out["n"] = np.atleast_1d(out["n"]).tolist()
    return out


@pytest.mark.parametrize("case", PEPPER, ids=lambda p: p.name)
def test_pepper(case):
    fx = load_fixture(case)
    inputs = fx.meta["inputs"]
    r = es.pepper(_sys(inputs["Sys"]), inputs["Exp"], _opt(inputs["Opt"]))
    x, spec = np.asarray(fx["/spec/x"]).ravel(), np.asarray(fx["/spec/spec"])
    assert_matches(r.x, x, rtol=1e-12, atol=0, name="x")
    ours = np.atleast_2d(r.spec) if spec.ndim == 2 and spec.shape[0] > 1 else np.asarray(r.spec).ravel()
    ref = spec if spec.ndim == 2 and spec.shape[0] > 1 else spec.ravel()
    assert_matches(ours, ref, rtol=1e-5, atol=1e-12 * np.abs(ref).max(), name="spec")
    if "/spec/transitions" in fx:
        trans = np.asarray(fx["/spec/transitions"])
        if trans.size:
            assert r.info["Transitions"].tolist() == trans.astype(int).tolist()


@pytest.mark.parametrize("case", GARLIC, ids=lambda p: p.name)
def test_garlic(case):
    fx = load_fixture(case)
    inputs = fx.meta["inputs"]
    r = es.garlic(_sys(inputs["Sys"]), inputs["Exp"], _opt(inputs["Opt"]))
    x, spec = np.asarray(fx["/spec/x"]).ravel(), np.asarray(fx["/spec/spec"]).ravel()
    assert_matches(r.x, x, rtol=1e-12, atol=0, name="x")
    assert_matches(r.spec, spec, rtol=1e-8, atol=1e-12 * np.abs(spec).max(), name="spec")
    assert_matches(
        np.sort(r.info["resfields"]),
        np.sort(np.asarray(fx["/spec/resfields"]).ravel()),
        rtol=1e-10,
        atol=0,
        name="resfields",
    )


@pytest.mark.skipif("helpers2" not in ONE, reason="no helpers2 case")
def test_helpers():
    fx = load_fixture(ONE["helpers2"])
    inputs = fx.meta["inputs"]
    lw, mI = es.fastmotion(inputs["fm1"], 350, 1e-10)
    assert_matches(lw, np.asarray(fx["/fastmotion/lw1"]).ravel(), rtol=1e-12, atol=0, name="lw1")
    np.testing.assert_array_equal(mI, np.asarray(fx["/fastmotion/mI1"]).reshape(mI.shape))
    lw, mI = es.fastmotion(inputs["fm2"], 340, 3e-11)
    assert_matches(lw, np.asarray(fx["/fastmotion/lw2"]).ravel(), rtol=1e-12, atol=0, name="lw2")
    np.testing.assert_array_equal(mI, np.asarray(fx["/fastmotion/mI2"]).reshape(mI.shape))
    lw, _ = es.fastmotion(inputs["fm2"], 340, 3e-11, "freq")
    assert_matches(lw, np.asarray(fx["/fastmotion/lw2freq"]).ravel(), rtol=1e-12, atol=0, name="lw2freq")
    for k, sysd in enumerate(inputs["hamsymm"], start=1):
        group, R = es.hamsymm(sysd)
        assert group == GROUPS[int(np.asarray(fx[f"/hamsymm/group_{k}"]).item()) - 1], f"hamsymm case {k}"
        assert_matches(R, fx[f"/hamsymm/R_{k}"], rtol=1e-12, atol=1e-14, name=f"R_{k}")
    B, spc = np.asarray(fx["/fieldmod/B"]).ravel(), np.asarray(fx["/fieldmod/spc"]).ravel()
    assert_matches(
        es.fieldmod(B, spc, 0.5, 1), np.asarray(fx["/fieldmod/h1"]).ravel(), rtol=1e-12, atol=1e-15, name="fieldmod h1"
    )
    assert_matches(
        es.fieldmod(B, spc, 3, 2), np.asarray(fx["/fieldmod/h2"]).ravel(), rtol=1e-12, atol=1e-15, name="fieldmod h2"
    )
    for k, (I, n) in enumerate(inputs["equivcouple"], start=1):
        F, N = es.equivcouple(I, int(n))
        np.testing.assert_array_equal(F, np.asarray(fx[f"/equivcouple/F_{k}"]).ravel())
        np.testing.assert_array_equal(N, np.asarray(fx[f"/equivcouple/N_{k}"]).ravel())
