"""The PyPI distribution of SpynDynamica (packaging/spyndynamica, scripts/build_spyndynamica.py): its version is
the package's, its dependencies are what the package imports, and the data files it lists are the ones the
package reads."""

import ast
import sys
import tomllib
from pathlib import Path

import spyndynamica as sd

ROOT = Path(__file__).resolve().parents[2]
PROJECT = tomllib.loads((ROOT / "packaging" / "spyndynamica" / "pyproject.toml").read_text(encoding="utf-8"))
SOURCE = ROOT / "src" / "spyndynamica"


def test_distribution_carries_the_package_version():
    assert PROJECT["project"]["name"] == "spyndynamica"
    assert PROJECT["project"]["version"] == sd.__version__
    assert PROJECT["tool"]["setuptools"]["packages"] == ["spyndynamica"]


def test_dependencies_are_what_the_package_imports():
    imported = set()
    for path in SOURCE.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                imported |= {alias.name.split(".")[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                imported.add(node.module.split(".")[0])
    third_party = imported - set(sys.stdlib_module_names) - {"spyndynamica"}

    def names(requirements):
        return {r.split(">")[0].split("=")[0].split("[")[0].strip() for r in requirements}

    required = names(PROJECT["project"]["dependencies"])
    optional = names(PROJECT["project"]["optional-dependencies"]["plot"])
    assert required == {"numpy", "scipy", "sympy"}
    assert third_party - optional == required, third_party
    assert third_party & optional == {"plotly"}


def test_data_files_are_listed():
    patterns = PROJECT["tool"]["setuptools"]["package-data"]["spyndynamica"]
    listed = {path.name for pattern in patterns for path in SOURCE.glob(pattern)}
    present = {path.name for path in (SOURCE / "data").iterdir() if path.is_file()}
    assert listed == present and present == {"isotopes.json", "orientations.npz"}


def test_readme_of_the_distribution_runs():
    text = (ROOT / "packaging" / "spyndynamica" / "README.md").read_text(encoding="utf-8")
    code = text.split("```python\n")[1].split("```")[0]
    exec(compile(code, "<readme>", "exec"), {"__name__": "__main__"})
    assert "doi:10.1002/mrc.4642" in text and "terms of use" in text
