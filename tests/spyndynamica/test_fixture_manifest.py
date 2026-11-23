"""The committed fixture data belongs to the committed case files."""

import json

import numpy as np
import pytest

from tests.spyndynamica import reference as fixtures


def manifest():
    return json.loads((fixtures.DATA / "manifest.json").read_text(encoding="utf-8"))


def test_every_case_file_has_current_data():
    listed = manifest()["files"]
    case_files = sorted(p.name for p in fixtures.CASES.glob("*.wl"))
    assert case_files == sorted(listed), "run fixtures/generate.wls: the case files and the manifest differ"
    for name in case_files:
        digest = fixtures.sha256(fixtures.CASES / name)
        assert digest == listed[name]["sha256"], f"{name} changed after its data was generated"
        data = fixtures.load(name[:-3])
        assert data["source_sha256"] == digest
        assert len(data["cases"]) == listed[name]["cases"]


def test_one_spindynamica_and_mathematica_version():
    reference = manifest()
    for name in fixtures.names():
        data = fixtures.load(name)
        assert data["spindynamica"] == reference["spindynamica"]
        assert data["mathematica"] == reference["mathematica"]


@pytest.mark.parametrize("name", fixtures.names())
def test_every_case_decodes_and_is_resolved(name):
    for key, case in fixtures.cases(name).items():
        assert "$Failed" not in json.dumps(case), key
        value = fixtures.decode(case["value"])
        if isinstance(value, np.ndarray):
            assert np.all(np.isfinite(value)), key
        if isinstance(value, fixtures.Symbolic):
            assert len(value.points) == 3, key
            for _, at_point in value.points:
                assert isinstance(at_point, np.ndarray) and np.all(np.isfinite(at_point)), key


@pytest.mark.parametrize("name", fixtures.names())
def test_exact_text_agrees_with_the_numbers(name):
    """The exact Mathematica text, parsed by SymPy, gives the stored floating-point values."""
    checked = 0
    for key, case in fixtures.cases(name).items():
        value = case["value"]
        if value["type"] != "array" or "exact" not in value or len(value["exact"]) > 4000:
            continue
        assert np.allclose(fixtures.exact_array(value), fixtures.array(value), rtol=1e-14, atol=1e-13), key
        checked += 1
    if not checked:
        pytest.skip("no exact results in this file")
