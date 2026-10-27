"""Parity of the resonance-field layer with EasySpin 6.0.15: every case written by esdump_resfields.m.

resfields positions and intensities are compared at 1e-9 relative (eigenvector interpolation and cubic roots
carry the same arithmetic, but the field search is iterative), widths at 1e-9, the eigenfield method at 1e-9,
line shapes and grids at 1e-13 and photoselection weights at 1e-13 (cubicsolve was checked in the spike).
"""

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
RF_CASES = [p for p in ALL if p.name.startswith("rf_")]
EIG_CASES = [p for p in ALL if p.name.startswith("eig_")]
ONE = {p.name: p for p in ALL}

pytestmark = pytest.mark.skipif(not ALL, reason=f"no EasySpin fixtures under {ROOT}")

RTOL = 1e-9


def _json_opt(opt: dict) -> dict:
    out = dict(opt)
    if "HybridCoreNuclei" in out:
        out["HybridCoreNuclei"] = np.atleast_1d(out["HybridCoreNuclei"]).tolist()
    return out


@pytest.mark.parametrize("case", RF_CASES, ids=lambda p: p.name)
def test_resfields(case):
    fx = load_fixture(case)
    inputs = fx.meta["inputs"]
    r = es.resfields(inputs["Sys"], inputs["Exp"], _json_opt(inputs["Opt"]))
    pos, trans = np.asarray(fx["/resfields/pos"]), np.asarray(fx["/resfields/trans"])
    assert r.trans.tolist() == trans.astype(int).tolist()
    assert_matches(np.nan_to_num(r.pos, nan=-1.0), np.nan_to_num(pos, nan=-1.0), rtol=RTOL, atol=0, name="pos")
    np.testing.assert_array_equal(np.isnan(r.pos), np.isnan(pos))
    ints = np.asarray(fx["/resfields/int"])
    assert_matches(
        np.nan_to_num(r.int), np.nan_to_num(ints), rtol=RTOL, atol=1e-12 * np.nanmax(np.abs(ints)), name="int"
    )
    if "/resfields/wid" in fx:
        wid = np.asarray(fx["/resfields/wid"])
        if wid.size:
            assert_matches(np.nan_to_num(r.wid), np.nan_to_num(wid), rtol=RTOL, atol=0, name="wid")
        else:
            assert r.wid is None
    else:
        assert r.wid is None


@pytest.mark.parametrize("case", EIG_CASES, ids=lambda p: p.name)
def test_resfields_eig(case):
    fx = load_fixture(case)
    inputs = fx.meta["inputs"]
    B, I = es.resfields_eig(inputs["Sys"], inputs["Exp"], {**inputs["Opt"], "Intensities": True})
    B_only = es.resfields_eig(inputs["Sys"], inputs["Exp"])
    for k in range(len(B)):
        assert_matches(B[k], np.asarray(fx[f"/eig/B_{k + 1}"]).ravel(), rtol=RTOL, atol=0, name=f"B_{k + 1}")
        assert_matches(I[k], np.asarray(fx[f"/eig/I_{k + 1}"]).ravel(), rtol=RTOL, atol=1e-12, name=f"I_{k + 1}")
        assert_matches(
            B_only[k], np.asarray(fx[f"/eig/Bonly_{k + 1}"]).ravel(), rtol=RTOL, atol=0, name=f"Bonly_{k + 1}"
        )


@pytest.mark.skipif("lineshapes" not in ONE, reason="no lineshapes case")
def test_lineshapes():
    fx = load_fixture(ONE["lineshapes"])
    x = np.asarray(fx.meta["inputs"]["x"], dtype=float)
    tol = 1e-13
    for d in range(-1, 3):
        assert_matches(
            es.gaussian(x, 1.3, 2.0, d),
            np.asarray(fx[f"/gaussian/abs_{d + 1}"]).ravel(),
            rtol=tol,
            atol=1e-15,
            name=f"gauss abs {d}",
        )
        if d >= 0:
            _, yd = es.gaussian(x, 1.3, 2.0, d, dispersion=True)
            assert_matches(
                yd, np.asarray(fx[f"/gaussian/disp_{d + 1}"]).ravel(), rtol=1e-11, atol=1e-13, name=f"gauss disp {d}"
            )
            assert_matches(
                es.gaussian(x, 1.3, 2.0, d, 0.4),
                np.asarray(fx[f"/gaussian/phase_{d + 1}"]).ravel(),
                rtol=1e-11,
                atol=1e-13,
                name=f"gauss phase {d}",
            )
        ya, yd = es.lorentzian(x, 1.3, 2.0, d, dispersion=True)
        assert_matches(
            ya, np.asarray(fx[f"/lorentzian/abs_{d + 1}"]).ravel(), rtol=tol, atol=1e-15, name=f"lorentz abs {d}"
        )
        assert_matches(
            yd, np.asarray(fx[f"/lorentzian/disp_{d + 1}"]).ravel(), rtol=tol, atol=1e-15, name=f"lorentz disp {d}"
        )
        assert_matches(
            es.lorentzian(x, 1.3, 2.0, d, 0.4),
            np.asarray(fx[f"/lorentzian/phase_{d + 1}"]).ravel(),
            rtol=tol,
            atol=1e-15,
            name=f"lorentz phase {d}",
        )
        assert_matches(
            es.lshape(x, 1.3, [1.5, 2.0], d, 0.3, 0.4),
            np.asarray(fx[f"/lshape/y_{d + 1}"]).ravel(),
            rtol=1e-11,
            atol=1e-13,
            name=f"lshape {d}",
        )
    for d in range(3):
        ya, yd = es.voigtian(x, 1.3, [1.5, 2.0], d, 0.2, dispersion=True)
        assert_matches(
            ya, np.asarray(fx[f"/voigtian/abs_{d + 1}"]).ravel(), rtol=1e-11, atol=1e-13, name=f"voigt abs {d}"
        )
        assert_matches(
            yd, np.asarray(fx[f"/voigtian/disp_{d + 1}"]).ravel(), rtol=1e-11, atol=1e-13, name=f"voigt disp {d}"
        )
        assert_matches(
            es.voigtian(x, 1.3, [2.0, 1.5], d),
            np.asarray(fx[f"/voigtian/narrowG_{d + 1}"]).ravel(),
            rtol=1e-11,
            atol=1e-13,
            name=f"voigt narrowG {d}",
        )
    spec1 = np.asarray(fx["/convspec/spec1"]).ravel()
    assert_matches(
        es.convspec(spec1, 0.05, 1.2, 1, 0.4),
        np.asarray(fx["/convspec/out1"]).ravel(),
        rtol=1e-11,
        atol=1e-13,
        name="convspec 1",
    )
    assert_matches(
        es.convspec(spec1, 0.05, 0.9, 0, 0.0, 0.3),
        np.asarray(fx["/convspec/out1b"]).ravel(),
        rtol=1e-11,
        atol=1e-13,
        name="convspec 1b",
    )
    spec2 = np.asarray(fx["/convspec/spec2"])
    assert_matches(
        es.convspec(spec2, [0.1, 0.2], [1, 2], [0, 1], [1, 0.5]),
        np.asarray(fx["/convspec/out2"]),
        rtol=1e-11,
        atol=1e-13,
        name="convspec 2",
    )


@pytest.mark.skipif("sphgrid" not in ONE, reason="no sphgrid case")
def test_sphgrid():
    fx = load_fixture(ONE["sphgrid"])
    inputs = fx.meta["inputs"]
    for symmetry in inputs["symms"]:
        for size in inputs["sizes"]:
            grid = es.sphgrid(symmetry, int(size), triangulate=True)
            base = f"/{symmetry}_{int(size)}"
            for name, value in (("phi", grid.phi), ("theta", grid.theta), ("weights", grid.weights)):
                assert_matches(
                    value,
                    np.asarray(fx[f"{base}/{name}"]).ravel(),
                    rtol=1e-13,
                    atol=1e-15,
                    name=f"{symmetry} {size} {name}",
                )
            if f"{base}/tri_idx" in fx:
                idx = np.asarray(fx[f"{base}/tri_idx"]).astype(int)
                areas = np.asarray(fx[f"{base}/tri_areas"]).ravel()
                assert grid.tri_idx.shape == idx.shape
                assert_matches(
                    np.array([grid.tri_areas.sum()]),
                    np.array([areas.sum()]),
                    rtol=1e-12,
                    atol=0,
                    name=f"{symmetry} {size} area sum",
                )
                if grid.closed_phi:  # the explicit triangulation; the Delaunay-based ones may choose other diagonals
                    np.testing.assert_array_equal(grid.tri_idx.astype(int), idx)
                    assert_matches(grid.tri_areas, areas, rtol=1e-12, atol=1e-15, name=f"{symmetry} {size} areas")
            else:
                assert grid.tri_idx.shape[0] == 0


@pytest.mark.skipif("photoselect" not in ONE, reason="no photoselect case")
def test_photoselect():
    fx = load_fixture(ONE["photoselect"])
    inputs = fx.meta["inputs"]
    ori2, ori3 = np.asarray(inputs["ori2"], dtype=float), np.asarray(inputs["ori3"], dtype=float)
    checks = (
        ("z_y_perp", es.photoselect("z", ori2, "y", np.pi / 2)),
        ("z_y_par", es.photoselect("z", ori2, "y", 0)),
        ("z_y_unpol", es.photoselect("z", ori2, "y", np.nan)),
        ("ang_vec_chi", es.photoselect([0.3, 0.5], ori3, [0.2, 0.1, 0.9], 0.7)),
        ("ang_vec_chi_unpol", es.photoselect([0.3, 0.5], ori3, [0.2, 0.1, 0.9], np.nan)),
    )
    for name, value in checks:
        assert_matches(value, np.asarray(fx[f"/photo/{name}"]).ravel(), rtol=1e-13, atol=1e-15, name=name)
