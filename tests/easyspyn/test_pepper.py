"""Fixture-free tests of the simulators: pepper and garlic against each other and closed forms, the kernels,
the orientation interpolation, the Hamiltonian symmetry, field modulation and equivalent-nuclei coupling."""

import numpy as np
import pytest

import easyspyn as es
from easyspyn import interp, kernels

NITROXIDE = {"g": [2.0088, 2.0061, 2.0027], "Nucs": "14N", "A": [16, 16, 86], "lwpp": 0.3}
XBAND = {"mwFreq": 9.5, "Range": [330, 350], "nPoints": 1024}
NOFUZZ = {"FuzzLevel": 0}


# ---------------------------------------------------------------------------------------- kernels
def test_lisum1i_reproduces_the_line_shape():
    x = np.linspace(300, 400, 4001)
    x0T, wT = 5e4, 2e4
    template = es.gaussian(np.arange(2 * x0T), x0T, wT, -1)  # integral template, as pepper uses it
    y = kernels.lisum1i(template, x0T, wT, np.array([342.0]), np.array([2.0]), np.array([1.5]), x)
    np.testing.assert_allclose(
        y, 2.0 * es.gaussian(x, 342.0, 1.5), rtol=1e-3, atol=1e-4
    )  # bin averages of the line shape
    assert kernels.lisum1i(template, x0T, wT, np.array([np.nan]), np.array([1.0]), np.array([1.0]), x).sum() == 0
    if kernels.HAVE_NUMBA:
        args = (template, x0T, wT, np.array([342.0, 351.3]), np.array([2.0, 1.0]), np.array([1.5, 0.4]), x)
        # the compiled and the NumPy summation differ by roundoff (1e-14) in the tails of the lines
        np.testing.assert_allclose(kernels.lisum1i(*args), kernels.lisum1i_numpy(*args), rtol=1e-12, atol=1e-13)


def test_projection_kernels_conserve_weights():
    x = np.linspace(300, 400, 1001)
    dx = x[1] - x[0]
    grid = es.sphgrid("D2h", 19, triangulate=True)
    pos = 340 + 20 * np.cos(grid.theta) ** 2
    amp = 1 + 0.2 * np.sin(grid.theta)
    spec = kernels.projecttriangles(grid.tri_idx.T, grid.tri_areas, pos, amp, x)
    expected = np.sum(grid.tri_areas * amp[grid.tri_idx.astype(int) - 1].mean(axis=1))
    assert spec.sum() * dx == pytest.approx(expected, rel=1e-12)
    np.testing.assert_allclose(
        kernels.projecttriangles_numpy(grid.tri_idx.T, grid.tri_areas, pos, amp, x), spec, rtol=1e-12
    )
    theta = np.linspace(0, np.pi / 2, 40)
    pos = 340 + 20 * np.cos(theta) ** 2
    w = -np.diff(np.cos(theta))
    zones = kernels.projectzones(pos, np.ones(1), w, x)
    assert zones.sum() * dx == pytest.approx(w.sum(), rel=1e-12)
    np.testing.assert_allclose(kernels.projectzones_numpy(pos, np.ones(1), w, x), zones, rtol=1e-12)
    with pytest.raises(ValueError, match="segment weights"):
        kernels.projectzones(pos, np.ones(1), w[:-1], x)


# ---------------------------------------------------------------------------------- interpolation
def test_splines_and_gridinterp():
    y = np.sin(np.linspace(0, 3, 9))
    xs = np.linspace(1, 9, 33)
    from scipy.interpolate import CubicSpline

    for endcon, bc in ((0, "not-a-knot"), (1, "clamped")):
        np.testing.assert_allclose(
            interp.esspline1d(y, endcon, xs), CubicSpline(np.arange(1, 10), y, bc_type=bc)(xs), atol=1e-13
        )
    yp = np.append(y[:-1], y[0])
    np.testing.assert_allclose(
        interp.esspline1d(yp, 2, xs), CubicSpline(np.arange(1, 10), yp, bc_type="periodic")(xs), atol=1e-13
    )
    # linear data are interpolated exactly on every path
    grid = es.sphgrid("D2h", 19)
    fine = es.sphgrid("D2h", 73)
    params = [grid.n, 19, grid.closed_phi, grid.n_octants, grid.max_phi]
    f = 3 * np.cos(grid.theta) + 2.0
    for mode in ("G3", "L1"):
        out = interp.gridinterp(f, params, fine.phi, fine.theta, mode)
        assert out.shape == fine.phi.shape
        np.testing.assert_allclose(out, 3 * np.cos(fine.theta) + 2.0, atol=0.03 if mode == "L1" else 5e-4)
    axial = es.sphgrid("Dinfh", 19)
    axial_fine = es.sphgrid("Dinfh", 73)
    f = 340 + 10 * np.cos(axial.theta) ** 2
    for mode in ("G3", "L3", "L1"):
        out = interp.gridinterp(f, [axial.n, 19, axial.closed_phi, 0, 0.0], axial_fine.phi, axial_fine.theta, mode)
        np.testing.assert_allclose(out, 340 + 10 * np.cos(axial_fine.theta) ** 2, atol=0.02)
    with pytest.raises(ValueError, match="Wrong number of points"):
        interp.gridinterp(f[:-1], [axial.n, 19, axial.closed_phi, 0, 0.0], axial_fine.phi, axial_fine.theta)


# ---------------------------------------------------------------------------------------- symmetry
@pytest.mark.parametrize(
    "system, group",
    [
        ({"g": 2.0}, "O3"),
        ({"g": [2.0, 2.0, 2.2]}, "Dinfh"),
        ({"g": [2.0, 2.1, 2.2]}, "D2h"),
        ({"g": [2.0, 2.1, 2.2], "Nucs": "14N", "A": [10, 10, 50], "AFrame": [0.3, 0.7, 1.1]}, "Ci"),
        ({"g": [2.0, 2.0, 2.2], "Nucs": "14N", "A": [50, 10, 10]}, "D2h"),
        ({"g": [2.0, 2.0, 2.2], "Nucs": "14N", "A": [10, 10, 50], "AFrame": [0, 0.4, 0]}, "C2h"),
        ({"g": 2.0, "tdm": "z"}, "C1"),
        ({"S": 1, "D": [500, 0]}, "Dinfh"),
    ],
)
def test_hamsymm_geometric(system, group):
    found, R = es.hamsymm(system)
    assert found == group
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)


def test_hamsymm_eigenvalue_analysis():
    assert es.hamsymm({"g": [[2.01, 0.002, 0], [0.002, 2.02, 0], [0, 0, 2.05]]})[0] == "C2h"
    assert es.hamsymm({"S": 2.5, "B4": [[0, 0, 0, 0, 10, 0, 0, 0, 0]]})[0] == "Dinfh"
    assert es.hamsymm({"S": 2.5, "B4": [[0, 0, 0, 0, 10, 0, 0, 0, 3]]})[0] in ("D4h", "Oh")


# ---------------------------------------------------------------------------------------- pepper
def test_pepper_methods_agree_on_the_nitroxide():
    matrix = es.pepper(NITROXIDE, XBAND, NOFUZZ)
    assert matrix.x.shape == (1024,) and matrix.spec.shape == (1024,)
    for method in ("perturb2", "hybrid"):
        other = es.pepper(NITROXIDE, XBAND, {**NOFUZZ, "Method": method})
        assert np.abs(other.spec - matrix.spec).max() < 0.05 * np.abs(matrix.spec).max()
    # the absorption areas of the projection, the eigenfield and the brute-force routes agree
    dx = matrix.x[1] - matrix.x[0]
    areas = [
        es.pepper(NITROXIDE, {**XBAND, "Harmonic": 0}, {**NOFUZZ, **o}).spec.sum() * dx
        for o in ({}, {"Method": "eig"}, {"BruteForce": 1})
    ]
    np.testing.assert_allclose(areas, areas[0], rtol=1e-4)
    separate = es.pepper(NITROXIDE, XBAND, {**NOFUZZ, "separate": "transitions"})
    assert separate.spec.ndim == 2
    np.testing.assert_allclose(separate.spec.sum(axis=0), matrix.spec, atol=1e-9 * np.abs(matrix.spec).max())


def test_pepper_isotropic_powder_matches_garlic():
    sys = {"g": 2.006, "Nucs": "14N", "A": 47, "lwpp": 0.3}
    exp = {"mwFreq": 9.5, "Range": [330, 350], "nPoints": 2048}
    powder = es.pepper(sys, exp, NOFUZZ)
    solution = es.garlic(sys, exp)
    # an isotropic system has no orientation dependence: pepper and garlic give the same three lines
    # (garlic bins the positions to the field grid, hence the loose tolerance)
    pe, ga = powder.spec / np.abs(powder.spec).max(), solution.spec / np.abs(solution.spec).max()
    assert np.abs(pe - ga).max() < 0.05
    assert powder.info["nOrientations"] == 1


def test_pepper_strains_crystals_and_options():
    strained = es.pepper({**NITROXIDE, "lwpp": 0, "HStrain": [20, 20, 60]}, XBAND, NOFUZZ)
    assert np.isfinite(strained.spec).all() and strained.spec.min() < 0 < strained.spec.max()
    crystal_exp = {**XBAND, "SampleFrame": [[0, 0, 0], [0.3, 1.1, 0.7]], "MolFrame": [0, 0, 0]}
    crystal = es.pepper(NITROXIDE, crystal_exp, {**NOFUZZ, "separate": "orientations"})
    assert crystal.spec.shape == (2, 1024)
    summed = es.pepper(NITROXIDE, crystal_exp, NOFUZZ)  # summed over the orientations by default
    assert summed.spec.shape == (1024,)
    np.testing.assert_allclose(summed.spec, crystal.spec.sum(axis=0), rtol=1e-12)
    axial = es.pepper({"g": [2.0, 2.0, 2.2], "lwpp": 1}, {"mwFreq": 9.5, "Range": [280, 360]}, NOFUZZ)
    assert axial.info["nOrientations"] == 19  # Dinfh: a quarter meridian
    mod = es.pepper(NITROXIDE, {**XBAND, "ModAmp": 0.1}, NOFUZZ)
    base = es.pepper(NITROXIDE, XBAND, NOFUZZ)
    assert np.corrcoef(mod.spec, base.spec)[0, 1] > 0.999
    auto = es.pepper(NITROXIDE, {"mwFreq": 9.5}, NOFUZZ)
    assert auto.x[0] < 335 < 342 < auto.x[-1]  # the resonances lie between 335 and 342 mT
    with pytest.raises(ValueError, match="equivalent nuclei"):
        es.pepper({**NITROXIDE, "n": [2]}, XBAND)
    with pytest.raises(ValueError, match="63Cu, 65Cu"):  # isotope mixtures are not ported
        es.pepper({"g": 2.1, "Nucs": "Cu", "A": 200, "lwpp": 1}, {"mwFreq": 9.5, "Range": [300, 350]})
    with pytest.raises(NotImplementedError, match="Frequency sweeps"):
        es.pepper(NITROXIDE, {"Field": 340, "mwRange": [9, 10]})
    with pytest.raises(ValueError, match="No broadening"):
        es.pepper({**NITROXIDE, "lwpp": 0}, {**XBAND, "Harmonic": 1}, NOFUZZ)


# ---------------------------------------------------------------------------------------- garlic
def test_garlic_lines_and_equivalent_nuclei():
    r = es.garlic({"g": 2.006, "Nucs": "14N", "A": 47, "lwpp": 0.1}, {"mwFreq": 9.5, "Range": [330, 350]})
    lines = np.sort(r.info["resfields"])
    assert lines.size == 3
    np.testing.assert_allclose(np.diff(lines), es.mhz2mt(47, 2.006), rtol=0.01)
    assert np.abs(r.spec.max() + r.spec.min()) < 1e-3 * r.spec.max()  # a symmetric first-derivative spectrum
    F, N = es.equivcouple(0.5, 5)
    np.testing.assert_array_equal(F, [2.5, 1.5, 0.5])
    np.testing.assert_array_equal(N, [1, 4, 5])
    three = es.garlic(
        {"g": 2.0, "Nucs": "1H", "A": 30, "n": [3], "lw": [0.2, 0]}, {"mwFreq": 9.5, "Range": [330, 348], "Harmonic": 0}
    )
    pos, amp = three.info["resfields"], three.info["intensities"]
    order = np.argsort(pos)
    merged = np.round(pos[order], 1)
    sums = [amp[order][merged == v].sum() for v in np.unique(merged)]
    np.testing.assert_allclose(np.array(sums) / min(sums), [1, 3, 3, 1], rtol=1e-6)


def test_garlic_methods_and_options():
    sys = {"g": 2.006, "Nucs": "14N", "A": 47, "lwpp": [0, 0.1]}
    exp = {"mwFreq": 9.5, "Range": [330, 350]}
    exact = es.garlic(sys, exp)
    perturb = es.garlic(sys, exp, {"Method": "perturb2"})
    np.testing.assert_allclose(np.sort(perturb.info["resfields"]), np.sort(exact.info["resfields"]), rtol=1e-6)
    explicit = es.garlic(sys, exp, {"AccumMethod": "explicit"})
    template = es.garlic(sys, exp, {"AccumMethod": "template"})
    scale = np.abs(explicit.spec).max()
    assert np.abs(template.spec - explicit.spec).max() < 0.02 * scale
    assert np.abs(exact.spec - explicit.spec).max() < 0.25 * scale  # binning rounds the positions to the grid
    cold = es.garlic(sys, {**exp, "Temperature": 1.5})
    pol = np.tanh(es.planck * 9.5e9 / (2 * es.boltzm * 1.5))
    np.testing.assert_allclose(cold.spec, exact.spec * pol, rtol=1e-9)
    freq = es.garlic({"g": 2.006, "Nucs": "14N", "A": 47, "lw": 0.1}, {"Field": 340, "mwRange": [9.4, 9.6]})
    assert freq.x[0] == 9.4 and freq.x[-1] == 9.6 and np.sort(freq.info["resfields"]).size == 3
    with pytest.raises(ValueError, match="S=1/2"):
        es.garlic({"S": 1, "g": 2.0}, exp)
    with pytest.raises(ValueError, match="Unknown method"):
        es.garlic(sys, exp, {"Method": "perturb9"})


def test_fastmotion_widths():
    A = es.unitconvert(np.array([7.59, 5.95, 31.76]) / 10, "mT->MHz")
    lw, mI = es.fastmotion({"g": [2.0088, 2.0064, 2.0027], "Nucs": "14N", "A": A}, 350, 1e-10)
    assert lw.shape == (3,) and np.all(lw > 0)
    np.testing.assert_array_equal(mI.ravel(), [-1, 0, 1])
    assert lw[0] > lw[2] > 0  # the mI = -1 line is the broadest for a nitroxide
    # the widths are linear in the correlation time in the fast limit (omega*tau << 1: j0 ~ j1 ~ tau)
    lw1, _ = es.fastmotion({"g": [2.0088, 2.0064, 2.0027], "Nucs": "14N", "A": A}, 350, 1e-12)
    lw2, _ = es.fastmotion({"g": [2.0088, 2.0064, 2.0027], "Nucs": "14N", "A": A}, 350, 2e-13)
    np.testing.assert_allclose(lw1 / lw2, 5, rtol=0.01)
    spec = es.garlic(
        {"g": [2.0088, 2.0064, 2.0027], "Nucs": "14N", "A": A, "tcorr": 1e-10}, {"mwFreq": 9.5, "Range": [335, 345]}
    )
    assert np.isfinite(spec.spec).all()
    with pytest.raises(ValueError, match="anisotropic"):
        es.fastmotion({"g": 2.0, "Nucs": "14N", "A": 47}, 350, 1e-10)


def test_fieldmod_small_amplitude_is_the_derivative():
    B = np.linspace(300, 400, 2001)
    spc = es.lorentzian(B, 342, 4)
    mod = es.fieldmod(B, spc, 0.2)
    assert np.corrcoef(mod, np.gradient(spc, B))[0, 1] > 0.9999
    second = es.fieldmod(B, spc, 0.5, 2)
    assert np.corrcoef(second, np.gradient(np.gradient(spc, B), B))[0, 1] > 0.999
    with pytest.raises(ValueError, match="positive"):
        es.fieldmod(B, spc, 0)
