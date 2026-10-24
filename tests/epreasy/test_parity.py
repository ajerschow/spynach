"""Parity of the epreasy foundation with EasySpin: every case written by oracle/matlab/easyspin/dumps/esdump_ham.m.

Each case is rebuilt from the inputs recorded in its meta.json and compared stage by stage. Tolerances:
operators, rotations and units at 1e-14 (the same arithmetic), Hamiltonians at 1e-12, levels at 1e-12.
Skips when no fixtures are present under $EPREASY_FIXTURES (default /easyspin-fixtures).
"""

import numpy as np
import pytest

import epreasy as es
from epreasy.oracle import assert_matches, fixture_root, load_fixture

ROOT = fixture_root()
CASES = (
    sorted(p for p in ROOT.glob("*/*") if p.is_dir() and (p / "data.h5").is_file() and (p / "meta.json").is_file())
    if ROOT.is_dir()
    else []
)
SYSTEM_CASES = [p for p in CASES if "/ham/H0" in load_fixture(p)]  # the esdump_ham spin-system cases
HELPERS = [p for p in CASES if p.name == "helpers"]

pytestmark = pytest.mark.skipif(not CASES, reason=f"no EasySpin fixtures under {ROOT}")

OPERATOR_TOL = 1e-14
HAMILTONIAN_TOL = 1e-12


def _sys(fx):
    return fx.meta["inputs"]["Sys"]


@pytest.fixture(params=SYSTEM_CASES, ids=lambda p: p.name)
def case(request):
    return load_fixture(request.param)


def test_spin_system(case):
    sys = _sys(case)
    np.testing.assert_array_equal(es.spinvec(sys), np.asarray(case["/sys/Spins"]).ravel())
    assert es.hsdim(sys) == int(np.asarray(case["/sys/nStates"]).item())


def test_hamiltonian_components(case):
    sys = _sys(case)
    B0 = np.asarray(case.meta["inputs"]["B0"], dtype=float)
    for name, H in (("zf", es.ham_zf(sys)), ("ee", es.ham_ee(sys)), ("hf", es.ham_hf(sys)), ("nq", es.ham_nq(sys)),
                    ("nn", es.ham_nn(sys)), ("ez_B", es.ham_ez(sys, B0))):  # fmt: skip
        assert_matches(H, case[f"/ham/{name}"], rtol=HAMILTONIAN_TOL, atol=1e-12 * max(1.0, np.abs(H).max()), name=name)
    if "/ham/nz_B" in case:
        assert_matches(es.ham_nz(sys, B0), case["/ham/nz_B"], rtol=HAMILTONIAN_TOL, atol=0, name="nz_B")


def test_ham_and_field_split(case):
    sys = _sys(case)
    B0 = np.asarray(case.meta["inputs"]["B0"], dtype=float)
    H0, mux, muy, muz = es.ham(sys)
    for name, H in (("H0", H0), ("mux", mux), ("muy", muy), ("muz", muz)):
        assert_matches(H, case[f"/ham/{name}"], rtol=HAMILTONIAN_TOL, atol=1e-12 * max(1.0, np.abs(H).max()), name=name)
    assert_matches(es.ham(sys, B0), case["/ham/H_B"], rtol=HAMILTONIAN_TOL, atol=0, name="H_B")
    _, muzL = es.ham(sys, B0, split=True)
    assert_matches(muzL, case["/ham/muzL"], rtol=HAMILTONIAN_TOL, atol=0, name="muzL")


def test_levels(case):
    sys = _sys(case)
    ori = np.asarray(case.meta["inputs"]["Ori"], dtype=float)
    fields = np.asarray(case.meta["inputs"]["Bfields"], dtype=float)
    assert_matches(es.levels(sys, ori, fields), case["/levels/E"], rtol=HAMILTONIAN_TOL, atol=0, name="E")
    assert_matches(
        es.levels(sys, "z", 350).ravel(),
        np.asarray(case["/levels/E_z"]).ravel(),
        rtol=HAMILTONIAN_TOL,
        atol=0,
        name="E_z",
    )


def test_spin_operators(case):
    sys = _sys(case)
    spins = es.spinvec(sys)
    for i in range(1, spins.size + 1):
        for c in range(1, 6):
            op = es.sop(spins, np.array([[i, c]]))
            assert_matches(op, case[f"/sop/s{i}_c{c}"], rtol=OPERATOR_TOL, atol=0, name=f"s{i}_c{c}")
    if spins.size >= 2:
        assert_matches(
            es.sop(spins, np.array([[1, 1], [2, 3]])), case["/sop/s1x_s2z"], rtol=OPERATOR_TOL, atol=0, name="s1x_s2z"
        )
    spec = "x(1|3)1" if 2 * spins[0] + 1 >= 3 else "z(1|2)1"
    assert_matches(es.sop(spins, spec), case["/sop/tsel"], rtol=OPERATOR_TOL, atol=0, name="tsel")


def test_stevens_operators(case):
    sys = _sys(case)
    spins = es.spinvec(sys)
    S1 = spins[0]
    for k in range(1, int(min(2 * S1, 6)) + 1):
        for q in range(-k, k + 1):
            op = es.stev(spins, k, q, 1)
            assert_matches(op, case[f"/stev/k{k}_q{q}"], rtol=1e-13, atol=0, name=f"k{k}_q{q}")


@pytest.mark.skipif(not HELPERS, reason="no helpers case")
def test_helpers():
    fx = load_fixture(HELPERS[0])
    inputs = fx.meta["inputs"]
    angles = np.asarray(inputs["angles"], dtype=float)
    for n, a in enumerate(angles, start=1):
        R = es.erot(a)
        assert_matches(R, fx[f"/rot/erot_{n}"], rtol=OPERATOR_TOL, atol=1e-15, name=f"erot_{n}")
        assert_matches(
            es.eulang(R), np.asarray(fx[f"/rot/eulang_{n}"]).ravel(), rtol=1e-13, atol=1e-13, name=f"eulang_{n}"
        )
    phi, theta = np.asarray(inputs["phi"], dtype=float), np.asarray(inputs["theta"], dtype=float)
    assert_matches(es.ang2vec(phi, theta), fx["/rot/ang2vec"], rtol=OPERATOR_TOL, atol=1e-15, name="ang2vec")
    p, t = es.vec2ang(es.ang2vec(phi, theta))
    assert_matches(np.vstack([p, t]), fx["/rot/vec2ang"], rtol=1e-13, atol=1e-13, name="vec2ang")
    fields = np.asarray(inputs["fields"], dtype=float)
    assert_matches(es.mt2mhz(fields), np.asarray(fx["/units/mt2mhz"]).ravel(), rtol=OPERATOR_TOL, atol=0, name="mt2mhz")
    assert_matches(
        es.mt2mhz(fields, 2.1), np.asarray(fx["/units/mt2mhz_g"]).ravel(), rtol=OPERATOR_TOL, atol=0, name="mt2mhz_g"
    )
    assert_matches(
        es.mhz2mt([100, 9500, 35000], 2.0023),
        np.asarray(fx["/units/mhz2mt"]).ravel(),
        rtol=OPERATOR_TOL,
        atol=0,
        name="mhz2mt",
    )
    assert_matches(
        es.unitconvert([1, 10, 100], "cm^-1->MHz"),
        np.asarray(fx["/units/wn2mhz"]).ravel(),
        rtol=OPERATOR_TOL,
        atol=0,
        name="wn2mhz",
    )
    assert_matches(
        es.unitconvert([1, 10, 100], "cm^-1->mT", 2.05),
        np.asarray(fx["/units/wn2mt"]).ravel(),
        rtol=OPERATOR_TOL,
        atol=0,
        name="wn2mt",
    )
    assert_matches(
        es.unitconvert([0.1, 1, 10], "K->MHz"),
        np.asarray(fx["/units/k2mhz"]).ravel(),
        rtol=OPERATOR_TOL,
        atol=0,
        name="k2mhz",
    )
    assert_matches(
        es.unitconvert([0.001, 0.01], "eV->cm^-1"),
        np.asarray(fx["/units/ev2wn"]).ravel(),
        rtol=OPERATOR_TOL,
        atol=0,
        name="ev2wn",
    )
    data = es.nucdata(inputs["nucs"])
    for name, value in (("I", data.spin), ("gn", data.gn), ("abund", data.abund)):
        assert_matches(value, np.asarray(fx[f"/nuc/{name}"]).ravel(), rtol=0, atol=0, name=name)
    qm = np.asarray(fx["/nuc/qm"]).ravel()
    np.testing.assert_array_equal(np.isnan(data.qm), np.isnan(qm))
    np.testing.assert_array_equal(data.qm[~np.isnan(qm)], qm[~np.isnan(qm)])
