"""Fixture-free tests of the resonance-field layer: resfields against the eigenfield method and closed forms,
the hybrid method, strains, line shapes, spherical grids and convspec."""

import numpy as np
import pytest

import epreasy as es

NITROXIDE = {"g": [2.0088, 2.0061, 2.0027], "Nucs": "14N", "A": [16, 16, 86]}
XBAND = {"mwFreq": 9.5, "Range": [330, 350]}
NOFUZZ = {"FuzzLevel": 0}


def test_free_electron_resonance():
    for g in (2.0, 2.0023, 2.3):
        r = es.resfields({"S": 0.5, "g": g}, {"mwFreq": 9.5, "Range": [250, 400]}, NOFUZZ)
        B = es.mhz2mt(9500, g)
        np.testing.assert_allclose(r.pos, [[B]], rtol=1e-7)  # EasySpin's stated precision; the cubic roots cancel
        # intensity = transition rate (g muB / 2h)^2 times dB/dE = h / (g muB), in MHz/mT, for one orientation
        np.testing.assert_allclose(r.int, [[g * es.bmagn / es.planck / 1e9 / 4]], rtol=1e-7)
        assert r.trans.tolist() == [[1, 2]] and r.wid is None
        B_eig, I_eig = es.resfields_eig({"S": 0.5, "g": g}, {"mwFreq": 9.5, "Range": [250, 400]}, {"Intensities": True})
        np.testing.assert_allclose(B_eig, [B], rtol=1e-10)
        np.testing.assert_allclose(I_eig, r.int.ravel(), rtol=1e-10)


def test_spline_method_matches_eigenfields():
    exp = {**XBAND, "SampleFrame": [[0, 0, 0], [0.3, 1.1, 0.7], [2.0, 2.5, 4.0]]}
    r = es.resfields(NITROXIDE, exp, {**NOFUZZ, "Transitions": "all"})
    B_eig = es.resfields_eig(NITROXIDE, exp)
    assert len(B_eig) == 3 and r.pos.shape[1] == 3
    for o in range(3):
        found = r.pos[:, o][~np.isnan(r.pos[:, o])]
        assert all(np.abs(B_eig[o] - p).min() < 1e-6 for p in found)
    # the allowed lines carry the intensity and the hyperfine triplet is spaced by about A_zz along z
    strong = np.sort(r.pos[r.int[:, 0] > 0.1 * r.int[:, 0].max(), 0])
    assert strong.size == 3
    np.testing.assert_allclose(np.diff(strong), es.mhz2mt(86, 2.0027), rtol=0.02)


def test_hybrid_method_keeps_the_total_intensity():
    cu = {
        "S": 0.5, "g": [2.05, 2.2], "Nucs": "63Cu,14N,14N",
        "A": [[50, 50, 500], [40, 40, 44], [38, 38, 42]], "Q": [[0, 0], [-1.2, 0.3], [1.0, 0.5]],
    }  # fmt: skip
    exp = {"mwFreq": 9.5, "Range": [260, 360], "SampleFrame": [0.3, 1.1, 0.7]}
    matrix = es.resfields(cu, exp, NOFUZZ)
    hybrid = es.resfields(cu, exp, {**NOFUZZ, "Method": "hybrid", "HybridCoreNuclei": [1]})
    assert hybrid.pos.shape[0] < matrix.pos.shape[0]
    np.testing.assert_allclose(np.nansum(hybrid.int), np.nansum(matrix.int), rtol=1e-2)  # weak nuclear lines dropped
    # every hybrid line lies within the first-order error of a matrix line
    for p in hybrid.pos.ravel():
        assert np.abs(matrix.pos.ravel() - p).min() < 0.2


def test_strain_widths():
    # an isotropic g with H strain: the width is HStrain projected on the field direction, in field units
    for ori, hs in (([0, 0, 0], 30.0), ([0, np.pi / 2, 0], 10.0)):
        r = es.resfields(
            {"g": 2.0, "HStrain": [10, 10, 30]}, {"mwFreq": 9.5, "Range": [300, 380], "SampleFrame": ori}, NOFUZZ
        )
        np.testing.assert_allclose(r.wid, [[es.mhz2mt(hs, 2.0)]], rtol=1e-9)
    # g strain: dB = B * dg / g for the free-electron line
    r = es.resfields({"g": 2.0, "gStrain": 0.01}, {"mwFreq": 9.5, "Range": [300, 380]}, NOFUZZ)
    np.testing.assert_allclose(r.wid, r.pos * 0.01 / 2.0, rtol=1e-7)
    with pytest.raises(ValueError, match="tilted g"):
        es.resfields(
            {"g": [2.0, 2.0, 2.1], "gFrame": [0.1, 0.2, 0.3], "gStrain": 0.01},
            {"mwFreq": 9.5, "Range": [300, 380]},
            NOFUZZ,
        )


def test_boltzmann_and_nonequilibrium_polarisation():
    exp = {"mwFreq": 9.5, "Range": [300, 380]}
    r_inf = es.resfields({"g": 2.0}, exp, NOFUZZ)
    r_cold = es.resfields({"g": 2.0}, {**exp, "Temperature": 0.5}, NOFUZZ)
    pol = np.tanh(es.planck * 9.5e9 / (2 * es.boltzm * 0.5))
    np.testing.assert_allclose(r_cold.int / r_inf.int, pol, rtol=1e-8)
    inverted = es.resfields({"g": 2.0, "initState": ([0.2, 0.8], "eigen")}, exp, NOFUZZ)
    np.testing.assert_allclose(inverted.int / r_inf.int, 0.2 - 0.8, rtol=1e-8)


def test_excitation_modes_and_options():
    exp = {**XBAND, "SampleFrame": [0.3, 1.1, 0.7]}
    perp = es.resfields(NITROXIDE, exp, NOFUZZ)
    par = es.resfields(NITROXIDE, {**exp, "mwMode": "parallel"}, NOFUZZ)
    assert np.nansum(par.int) < 0.05 * np.nansum(perp.int)  # parallel mode is nearly forbidden here
    unpol = es.resfields(NITROXIDE, {**exp, "mwMode": ["y", "unpolarized"]}, NOFUZZ)
    assert unpol.pos.shape == perp.pos.shape
    user = es.resfields(NITROXIDE, exp, {**NOFUZZ, "Transitions": [[1, 6], [2, 5]]})
    assert user.trans.tolist() == [[1, 6], [2, 5]]
    centre = es.resfields(NITROXIDE, {"mwFreq": 9.5, "CenterSweep": [340, 20], "SampleFrame": [0.3, 1.1, 0.7]}, NOFUZZ)
    np.testing.assert_allclose(centre.pos, perp.pos, equal_nan=True)
    with pytest.raises(ValueError, match="mwFreq is missing"):
        es.resfields(NITROXIDE, {"Range": [300, 400]})
    with pytest.raises(ValueError, match="similar to"):
        es.resfields(NITROXIDE, {"mwfreq": 9.5, "Range": [300, 400]})
    with pytest.raises(NotImplementedError, match="symmetry"):
        es.resfields(NITROXIDE, {**XBAND, "CrystalSymmetry": "P212121"})
    with pytest.warns(UserWarning, match="No resonance fields"):
        es.resfields(NITROXIDE, {"mwFreq": 9.5, "Range": [100, 120]}, NOFUZZ)


def test_photoselection_weights():
    ori2 = np.array([[0, 0], [0.3, 1.1], [2.0, 2.5]])
    # a transition dipole along z and light polarised along the lab x axis: weight = sin^2(theta)/2 after chi averaging
    w = es.photoselect("z", ori2, "y", np.pi / 2)
    np.testing.assert_allclose(w, np.sin(ori2[:, 1]) ** 2 / 2, atol=1e-12)
    # unpolarised light along y: (1 - weight along k)/2
    wu = es.photoselect("z", ori2, "y", np.nan)
    assert np.all((wu >= 0) & (wu <= 0.5))
    r = es.resfields(
        {**NITROXIDE, "tdm": "z"}, {**XBAND, "SampleFrame": [0.3, 1.1, 0.7], "lightBeam": "perpendicular"}, NOFUZZ
    )
    r0 = es.resfields(NITROXIDE, {**XBAND, "SampleFrame": [0.3, 1.1, 0.7]}, NOFUZZ)
    lab = es.eulang(es.erot([0.3, 1.1, 0.7]).T)  # the lab-frame orientation that resfields derives from SampleFrame
    np.testing.assert_allclose(r.int, r0.int * es.photoselect("z", [lab], "y", -np.pi / 2)[0], rtol=1e-10)


def test_nucspinrmv():
    sys = {"S": 0.5, "g": [2, 2, 3], "Nucs": "14N,1H", "A": [[4, 6, 10], [1, 1, 2]], "Q": [[-1, 0.2], [0, 0]]}
    out = es.nucspinrmv(sys, 1)
    assert out["Nucs"] == "1H" and np.asarray(out["A"]).tolist() == [[1, 1, 2]]
    assert "Nucs" not in es.nucspinrmv(sys, [1, 2])
    with pytest.raises(ValueError, match="Index out of range"):
        es.nucspinrmv(sys, 3)


def test_cubicsolve_roots():
    rng = np.random.default_rng(3)
    C = rng.standard_normal((500, 4))
    for loop in (True, False):
        roots, count = es.cubicsolve(C, loop)
        for k in range(C.shape[0]):
            t = roots[k, : count[k]]
            assert np.all((t >= 0) & (t <= 1)) and np.all(np.diff(t) >= 0)
            np.testing.assert_allclose(np.polyval(C[k], t), 0, atol=1e-8)
            # every real root in [0, 1] of the polynomial is found
            true = np.roots(C[k])
            true = true[(np.abs(true.imag) < 1e-9) & (true.real >= 1e-9) & (true.real <= 1 - 1e-9)].real
            assert all(np.abs(t - r).min() < 1e-7 for r in true)


# ---------------------------------------------------------------------------------------- line shapes
def test_lineshapes_normalisation_and_derivatives():
    x = np.linspace(-40, 40, 16001)
    dx = x[1] - x[0]
    for shape in (
        lambda: es.gaussian(x, 1.3, 2.0),
        lambda: es.lorentzian(x, 1.3, 2.0),
        lambda: es.lshape(x, 1.3, [1.5, 2.0], 0, 0.3),
        lambda: es.voigtian(x, 1.3, [1.5, 2.0]),
    ):
        y = shape()
        assert x[np.argmax(y)] == pytest.approx(1.3, abs=dx)
        assert y.sum() * dx == pytest.approx(1.0, abs=0.02)
    assert es.gaussian(x, 1.3, 2.0).sum() * dx == pytest.approx(1.0, abs=1e-9)
    g = es.gaussian(x, 1.3, 2.0)
    half = x[g >= g.max() / 2]
    assert half[-1] - half[0] == pytest.approx(2.0, abs=2 * dx)
    for f in (es.gaussian, es.lorentzian):
        np.testing.assert_allclose(f(x, 1.3, 2.0, 1), np.gradient(f(x, 1.3, 2.0), dx), atol=2e-4)
        np.testing.assert_allclose(f(x, 1.3, 2.0, 2), np.gradient(f(x, 1.3, 2.0, 1), dx), atol=2e-3)
        np.testing.assert_allclose(np.gradient(f(x, 1.3, 2.0, -1), dx), f(x, 1.3, 2.0), atol=2e-4)
        ya, yd = f(x, 0.0, 2.0, dispersion=True)
        np.testing.assert_allclose(yd, -yd[::-1], atol=1e-12)  # dispersion is antisymmetric
        np.testing.assert_allclose(f(x, 0.0, 2.0, 0, np.pi / 2), yd, atol=1e-12)
    with pytest.raises(ValueError):
        es.lorentzian(x, 0.0, 2.0, 3)
    # the dispersion integral (EasySpin: a 2F2 from the Symbolic Toolbox) differentiates to the dispersion shape
    _, disp_int = es.gaussian(x, 0.0, 2.0, -1, dispersion=True)
    _, disp = es.gaussian(x, 0.0, 2.0, 0, dispersion=True)
    np.testing.assert_allclose(np.gradient(disp_int, dx), disp, atol=2e-4)


def test_convspec_matches_the_line_shape():
    x = np.arange(801) * 0.05
    spec = np.zeros(801)
    spec[400] = 1.0
    for alpha in (1.0, 0.0, 0.4):
        out = es.convspec(spec, 0.05, 1.2, 0, alpha)
        np.testing.assert_allclose(out, es.lshape(x, x[400], 1.2, 0, alpha) * 0.05, atol=1e-9)
    out1 = es.convspec(spec, 0.05, 1.2, 1)
    np.testing.assert_allclose(out1, es.gaussian(x, x[400], 1.2, 1) * 0.05, atol=1e-9)
    spec2 = np.zeros((40, 60))
    spec2[20, 30] = 1
    out2 = es.convspec(spec2, [0.1, 0.2], [1.0, 2.0])
    np.testing.assert_allclose(out2.sum(), 1.0, atol=1e-5)  # tails lost at the zero-padded border
    with pytest.raises(ValueError, match="smaller"):
        es.convspec(spec, 1.0, 0.5, 1)


# -------------------------------------------------------------------------------------- spherical grids
@pytest.mark.parametrize("symmetry", ["D2h", "Ci", "C1", "Dinfh", "C2h", "C4h", "D4h", "O3", "Th", "S6"])
def test_sphgrid_weights_and_triangles(symmetry):
    grid = es.sphgrid(symmetry, 7, triangulate=True)
    assert grid.weights.sum() == pytest.approx(4 * np.pi)
    np.testing.assert_allclose(np.linalg.norm(grid.vecs, axis=0), 1)
    max_phi, closed, n_oct = es.gridparam(symmetry)
    assert grid.phi.max() <= max_phi + 1e-12 and (grid.phi.max() == pytest.approx(max_phi) or not closed or n_oct <= 0)
    if n_oct > 0:
        assert grid.tri_areas.sum() == pytest.approx(4 * np.pi)
        assert grid.tri_idx.min() >= 1 and grid.tri_idx.max() <= grid.n
        assert np.all(np.diff(grid.tri_idx.astype(int), axis=1) >= 0)
        if grid.closed_phi:
            assert np.all(np.diff(grid.tri_idx.astype(int), axis=1) > 0)
    else:
        assert grid.tri_idx.shape[0] == 0
    with pytest.raises(ValueError, match="Unsupported symmetry"):
        es.gridparam("D5h")


def test_sphgrid_closed_octant_is_explicit():
    grid = es.sphgrid("D2h", 7, triangulate=True)
    assert grid.n == 28 and grid.tri_idx.shape == (36, 3)  # (GridSize-1)^2 triangles
    assert grid.tri_idx[0].tolist() == [1, 2, 3]
    assert es.sphgrid(1, 5).n == es.sphgrid("C4h", 5).n
