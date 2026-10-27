"""EasySpyn depends on spynach, never the other way round: nothing outside src/easyspyn and tests/easyspyn imports
it."""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OWN = {ROOT / "src" / "easyspyn", ROOT / "tests" / "easyspyn"}


def imports_easyspyn(path: Path) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(alias.name.split(".")[0] == "easyspyn" for alias in node.names):
                return True
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            if node.module.split(".")[0] == "easyspyn":
                return True
    return False


def test_nothing_outside_easyspyn_imports_it():
    offenders = []
    for base in (ROOT / "src", ROOT / "tests", ROOT / "oracle", ROOT / "comparison", ROOT / "docs"):
        for path in base.rglob("*.py"):
            if any(own in path.parents for own in OWN) or "__pycache__" in path.parts:
                continue
            if imports_easyspyn(path):
                offenders.append(path.relative_to(ROOT).as_posix())
    assert not offenders, f"the Spinach-based code must not import easyspyn: {offenders}"
