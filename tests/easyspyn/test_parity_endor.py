"""Parity of endorfrq, endorfrq_perturb, orisel, salt and curry with EasySpin 6.0.15: the cases of esdump_endor.m.
ENDOR frequencies and intensities on crystal orientations and curry outputs are compared at 1e-9; salt spectra
normwise at 1e-6 (the projection and interpolation of pepper, whose cubic roots are not involved here)."""

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
ENDORFRQ = [p for p in ALL if p.name.startswith("en_")]
SALT = [p for p in ALL if p.name.startswith("sa_")]
CURRY = [p for p in ALL if p.name.startswith("cu_") and p.name != "cu_2n"]  # cu_2n is a ham case
ONE = {p.name: p for p in ALL}

pytestmark = pytest.mark.skipif(not ENDORFRQ, reason=f"no EasySpin ENDOR fixtures under {ROOT}")


def _opt(opt: dict, lists=("Nuclei", "Spins", "GridSize")) -> dict:
    out = dict(opt or {})
    for key in lists:
        if key in out:
            out[key] = np.atleast_1d(out[key]).tolist()
    return out


@pytest.mark.parametrize("case", ENDORFRQ, ids=lambda p: p.name)
def test_endorfrq(case):
    fx = load_fixture(case)
    inputs = fx.meta["inputs"]
    fcn = es.endorfrq if inputs["fn"] == "endorfrq" else es.endorfrq_perturb
    r = fcn(inputs["Sys"], inputs["Exp"], _opt(inputs["Opt"]))
    pos, intensity = np.asarray(fx["/res/pos"]), np.asarray(fx["/res/int"])
    assert_matches(r.pos, pos.reshape(r.pos.shape), rtol=1e-9, atol=1e-9, name="pos")
    assert_matches(r.int, intensity.reshape(r.int.shape), rtol=1e-9, atol=1e-12 * np.abs(intensity).max(), name="int")
    assert_matches(
        r.info["Selectivity"],
        float(np.asarray(fx["/res/selectivity"]).ravel()[0]),
        rtol=1e-9,
        atol=1e-12,
        name="selectivity",
    )
    if "/res/trans" in fx:
        assert r.trans.tolist() == np.asarray(fx["/res/trans"]).astype(int).reshape(r.trans.shape).tolist()


@pytest.mark.skipif("orisel" not in ONE, reason="no orisel case")
def test_orisel():
    fx = load_fixture(ONE["orisel"])
    inputs = fx.meta["inputs"]
    sysd, exp, sys2 = inputs["Sys"], inputs["Exp"], inputs["Sys2"]
    ori = np.asarray(inputs["ori"], dtype=float)
    assert_matches(
        es.orisel(sysd, exp, {"GridSize": 10}),
        np.asarray(fx["/orisel/grid"]).ravel(),
        rtol=1e-9,
        atol=1e-12,
        name="grid",
    )
    weights, trans = es.orisel(sysd, exp, {"GridSize": 10}, per_transition=True)
    assert_matches(
        weights,
        np.asarray(fx["/orisel/grid_weights"]).reshape(weights.shape),
        rtol=1e-9,
        atol=1e-12,
        name="grid weights",
    )
    assert trans.tolist() == np.asarray(fx["/orisel/grid_trans"]).astype(int).reshape(trans.shape).tolist()
    assert_matches(
        es.orisel(sysd, {**exp, "Orientations": ori}),
        np.asarray(fx["/orisel/user"]).ravel(),
        rtol=1e-9,
        atol=1e-12,
        name="user",
    )
    assert_matches(
        es.orisel(sys2, {**exp, "Orientations": ori}),
        np.asarray(fx["/orisel/triplet"]).ravel(),
        rtol=1e-9,
        atol=1e-12,
        name="triplet",
    )
    assert_matches(
        es.orisel(sys2, {**exp, "Orientations": ori}, {"AveragedIntensity": False}),
        np.asarray(fx["/orisel/triplet_nonavg"]).ravel(), rtol=1e-9, atol=1e-12, name="triplet nonavg",
    )  # fmt: skip


@pytest.mark.parametrize("case", SALT, ids=lambda p: p.name)
def test_salt(case):
    fx = load_fixture(case)
    inputs = fx.meta["inputs"]
    sysd = inputs["Sys"]
    r = es.salt(sysd, inputs["Exp"], _opt(inputs["Opt"]))
    x, spec = np.asarray(fx["/spec/x"]).ravel(), np.asarray(fx["/spec/spec"])
    assert_matches(r.x, x, rtol=1e-12, atol=0, name="x")
    ours = np.atleast_2d(r.spec) if spec.ndim == 2 and spec.shape[0] > 1 else np.asarray(r.spec).ravel()
    ref = spec if spec.ndim == 2 and spec.shape[0] > 1 else spec.ravel()
    assert_matches(ours, ref, rtol=1e-6, atol=1e-12 * np.abs(ref).max(), name="spec")
    if "/spec/transitions" in fx:
        trans = np.asarray(fx["/spec/transitions"])
        assert r.info["Transitions"].tolist() == trans.astype(int).reshape(r.info["Transitions"].shape).tolist()


@pytest.mark.parametrize("case", CURRY, ids=lambda p: p.name)
def test_curry(case):
    fx = load_fixture(case)
    inputs = fx.meta["inputs"]
    out = es.curry(inputs["Sys"], inputs["Exp"], _opt(inputs["Opt"], lists=("Spins",)))
    outs = list(out) if isinstance(out, tuple) else [out]
    # the partition-function method differences ln Z twice over 1 mT, which amplifies roundoff
    rtol = 1e-5 if inputs["Opt"].get("Method") == "partitionfunction" else 1e-9
    for k, ours in enumerate(outs, start=1):
        ref = np.asarray(fx[f"/curry/out{k}"])
        assert_matches(ours, ref.reshape(ours.shape), rtol=rtol, atol=1e-12 * np.abs(ref).max(), name=f"out{k}")
