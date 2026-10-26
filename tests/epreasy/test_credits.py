"""The EasySpin contributors named in the docstrings of epreasy and in docs/easyspin-credits.md agree with
porting/easyspin_authors.json, which porting/tools/gen_easyspin_authors.py reads from EasySpin's Git history at the
release tag; no network is needed here."""

import json
import sys
from pathlib import Path

import epreasy as es

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "porting" / "tools"))
import gen_easyspin_authors as tool  # noqa: E402

DATA = json.loads((ROOT / "porting" / "easyspin_authors.json").read_text(encoding="utf-8"))


def test_docstrings_and_page_are_current():
    assert tool.stale(DATA) == [], "run: python porting/tools/gen_easyspin_authors.py --write"


def test_every_mapped_function_has_recorded_authors():
    for qual, entry in DATA["functions"].items():
        for item in entry["files"]:
            info = DATA["files"][item["file"]]
            assert info["contributors"] or info["external"], f"{qual}: nobody recorded for {item['file']}"
    ported = [q for q, e in DATA["functions"].items() if e["relation"] == "port"]
    assert len(ported) >= 60


def test_known_attributions():
    assert DATA["files"]["pepper.m"]["contributors"][0]["name"] == "Stefan Stoll"
    assert DATA["files"]["jacobianest.m"]["external"] == ["John D'Errico"]
    assert DATA["functions"]["epreasy.pepper.pepper"]["relation"] == "port"
    assert DATA["functions"]["epreasy.kernels.lisum1i_numpy"]["relation"] == "derived"
    assert DATA["repository"] == "https://github.com/StollLab/EasySpin" and DATA["tag"] == "v6.0.15"


def test_help_shows_the_authors():
    assert "EasySpin authors (pepper.m): Stefan Stoll" in es.pepper.__doc__
    assert "John D'Errico" in es.jacobianest.__doc__


def test_names_only_in_the_outputs():
    """Contributors are named; no email addresses are recorded."""
    for path in [ROOT / "porting" / "easyspin_authors.json", tool.PAGE]:
        text = path.read_text(encoding="utf-8")
        assert "@" not in text, path.name
