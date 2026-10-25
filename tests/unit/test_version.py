"""The version is stated in one place per file format and the copies agree."""

import re
import tomllib
from pathlib import Path

import spynach

ROOT = Path(__file__).resolve().parents[2]


def test_version_copies_agree():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]
    assert spynach.__version__ == pyproject
    citation = re.search(r"^version: (\S+)$", (ROOT / "CITATION.cff").read_text(encoding="utf-8"), flags=re.M)
    assert citation and citation.group(1) == pyproject
    assert f"## {pyproject} - " in (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")


def test_ctps_carries_the_distribution_version():
    """The three packages ship in one distribution, so ctps keeps its version equal to it."""
    import ctps

    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]
    assert ctps.__version__ == pyproject
