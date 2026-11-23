"""SpynDynamica is a separate import: it uses nothing of spynach, easyspyn or ctps, and nothing else imports it."""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OWN = {ROOT / "src" / "spyndynamica", ROOT / "tests" / "spyndynamica"}
OTHERS = {"spynach", "easyspyn", "ctps"}


def imported(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.split(".")[0])
    return names


def test_spyndynamica_imports_none_of_the_other_packages():
    offenders = [p.name for p in (ROOT / "src" / "spyndynamica").rglob("*.py") if imported(p) & OTHERS]
    assert not offenders, f"spyndynamica must not import spynach, easyspyn or ctps: {offenders}"


def test_nothing_else_imports_spyndynamica():
    offenders = []
    for base in (ROOT / "src", ROOT / "tests", ROOT / "oracle", ROOT / "comparison"):
        for path in base.rglob("*.py"):
            if any(own in path.parents for own in OWN) or "__pycache__" in path.parts:
                continue
            if "spyndynamica" in imported(path):
                offenders.append(path.relative_to(ROOT).as_posix())
    assert not offenders, f"only its own tests import spyndynamica: {offenders}"
