"""The Spinach authors named in the docstrings and in docs/spinach-credits.md agree with porting/spinach_authors.json,
which porting/tools/gen_spinach_authors.py reads from the headers of a Spinach checkout; no checkout is needed here."""

import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "porting" / "tools"))
import gen_spinach_authors as tool  # noqa: E402

DATA = json.loads((ROOT / "porting" / "spinach_authors.json").read_text(encoding="utf-8"))


def test_docstrings_and_page_are_current():
    assert tool.stale(DATA) == [], "run: python porting/tools/gen_spinach_authors.py --write"


def test_every_translated_function_names_its_authors():
    for qual, entry in DATA["functions"].items():
        for item in entry["files"]:
            info = DATA["files"][item["file"]]
            assert info["authors"] or info["unnamed"], f"{qual}: no author for {item['file']}"
    ported = [q for q, e in DATA["functions"].items() if e["relation"] == "port"]
    assert len(ported) >= 60


def test_known_attributions():
    assert DATA["files"]["propagator.m"]["authors"] == ["Ilya Kuprov", "Luke Edwards"]
    assert DATA["files"]["fourdif.m"]["authors"] == ["J.A.C. Weideman", "S.C. Reddy"]
    assert "Alex Nevzorov" in DATA["files"]["clean_up.m"]["authors"]
    assert DATA["functions"]["spynach.propagation.propagator"]["relation"] == "port"
    assert DATA["functions"]["spynach.propagation._reordered_taylor"]["relation"] == "derived"
    assert DATA["functions"]["spynach.kernels.matlab_round"]["files"] == []


def test_help_shows_the_authors():
    import spynach

    assert "Spinach authors (propagator.m): Ilya Kuprov, Luke Edwards." in spynach.propagator.__doc__
    assert "Spinach authors, by original file" in spynach.propagation.__doc__


def test_no_email_addresses_in_the_outputs():
    """The headers give addresses; the repository outputs carry names only."""
    for path in [ROOT / "porting" / "spinach_authors.json", tool.PAGE, *tool.python_files()]:
        text = path.read_text(encoding="utf-8")
        for address in tool.NAMES:
            assert address not in text, f"{address} in {path.name}"


def test_docstring_edits_leave_valid_python():
    for py in tool.python_files():
        ast.parse(py.read_text(encoding="utf-8"))
