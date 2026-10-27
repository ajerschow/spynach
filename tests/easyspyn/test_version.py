"""EasySpyn ships in the distribution of spynach and carries its version; the check lives here because only
tests/easyspyn may import easyspyn (tests/easyspyn/test_dependency_direction.py)."""

import tomllib
from pathlib import Path

import easyspyn

ROOT = Path(__file__).resolve().parents[2]


def test_easyspyn_carries_the_distribution_version():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]
    assert easyspyn.__version__ == pyproject
