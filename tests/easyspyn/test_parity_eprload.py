"""Parity of eprload with EasySpin 6.0.15 on the files of esdump_eprload.m: Bruker BES3T pairs written by eprsave
(1-D, complex, nonlinear axes, 2-D, the scaling letters), hand-written ESP and WinEPR files, and any instrument
files supplied. The Python loader reads the same bytes from the case directory; the abscissa, data, scaled data
and parameter values agree to rounding (1e-12), scalings that error in EasySpin raise ValueError here."""

import numpy as np
import pytest

import easyspyn as es
from easyspyn.oracle import assert_matches, fixture_root, load_fixture

ROOT = fixture_root()
CASES = (
    sorted(p for p in ROOT.glob("*/el_*") if p.is_dir() and (p / "data.h5").is_file() and (p / "meta.json").is_file())
    if ROOT.is_dir()
    else []
)
pytestmark = pytest.mark.skipif(not CASES, reason=f"no EasySpin eprload fixtures under {ROOT}")

TOL = 1e-12


@pytest.fixture(params=CASES, ids=lambda p: p.name)
def case(request):
    return request.param, load_fixture(request.param)


def _arrays(value):
    """A loaded value as a list of arrays (one per data value or axis)."""
    return [np.asarray(v) for v in value] if isinstance(value, list) else [np.asarray(value)]


def _check_data(actual, fx, group: str, count_key: str, name: str) -> None:
    arrays = _arrays(actual)
    count = int(np.ravel(fx[f"{group}/{count_key}"])[0]) if count_key else 1
    assert len(arrays) == count, f"{name}: {len(arrays)} arrays, EasySpin has {count}"
    for k, array in enumerate(arrays):
        expected = np.asarray(fx[f"{group}/{name}{k + 1}" if count > 1 else f"{group}/{name}"])
        if array.ndim <= 1 or 1 in expected.shape:
            array, expected = np.ravel(array), np.ravel(expected)
        else:
            assert array.shape == expected.shape, f"{name}: shape {array.shape}, EasySpin {expected.shape}"
        assert_matches(array, expected, rtol=TOL, atol=0, name=f"{name}{k + 1 if count > 1 else ''}")


def test_eprload_matches_easyspin(case):
    path, fx = case
    inputs = fx.meta["inputs"]
    file = path / inputs["file"]
    x, y, params = es.eprload(file)
    _check_data(x, fx, "/load", "naxes", "x")
    _check_data(y, fx, "/load", "nvalues", "y")
    for letters in inputs.get("scalings", []):
        if f"/scaled/{letters}/error" in fx:
            with pytest.raises(ValueError):
                es.eprload(file, letters)
            continue
        _, scaled, _ = es.eprload(file, letters)
        if isinstance(scaled, list):
            for j, array in enumerate(scaled):
                assert_matches(array, np.asarray(fx[f"/scaled/{letters}/y{j + 1}"]), rtol=TOL, atol=0, name=letters)
        else:
            expected = np.asarray(fx[f"/scaled/{letters}/y"])
            if scaled.ndim <= 1 or 1 in expected.shape:
                scaled, expected = np.ravel(scaled), np.ravel(expected)
            assert_matches(scaled, expected, rtol=TOL, atol=0, name=f"scaled {letters}")
    # the parameter structure, key by key; continuation lines are joined differently (see the module docstring)
    reference = inputs["params"]
    skipped = {key for key, value in reference.items() if isinstance(value, str) and ("\\" in value or "\n" in value)}
    for key, value in reference.items():
        if key in skipped:
            continue
        assert key in params, f"parameter {key} missing"
        got = params[key]
        if isinstance(value, bool):
            assert got is value, key
        elif isinstance(value, str):
            assert got == value, key
        elif value is None:
            assert isinstance(got, float) and np.isnan(got), key
        else:
            np.testing.assert_allclose(np.atleast_1d(got), np.atleast_1d(value), rtol=TOL, err_msg=key)
    extra = {key for key in params if key not in reference and key.isidentifier()}
    assert extra <= skipped, f"keys not in EasySpin's structure: {sorted(extra - skipped)}"
