"""epreasy ships in the distribution of spynach and carries its version; the check lives here because only
tests/epreasy may import epreasy (tests/epreasy/test_dependency_direction.py)."""

import tomllib
from pathlib import Path

import epreasy

ROOT = Path(__file__).resolve().parents[2]


def test_epreasy_carries_the_distribution_version():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]
    assert epreasy.__version__ == pyproject
