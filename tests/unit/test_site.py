"""The generated spans of the project site (navigation, timing tables, guide tables, the playground's source
list) agree with the data they are rendered from; scripts/build_site.py rewrites them."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import build_site  # noqa: E402


def test_site_spans_are_current():
    assert build_site.stale() == [], "run: python scripts/build_site.py"


def test_every_guide_is_filed_in_a_group():
    rows = build_site.guide_rows()
    assert set(rows) == {stem for stems in build_site.GROUPS.values() for stem in stems}


def test_source_list_covers_the_three_packages():
    files = build_site.source_files()
    assert "src/spynach/__init__.py" in files
    assert "src/ctps/__init__.py" in files
    assert "src/easyspyn/__init__.py" in files
    assert "src/easyspyn/data/isotopedata.txt" in files
    assert not any("__pycache__" in f for f in files)
