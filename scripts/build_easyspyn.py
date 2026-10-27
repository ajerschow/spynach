"""Builds the PyPI distribution "easyspyn" (EasySpyn, the package ``easyspyn``) into dist/easyspyn/.

    python scripts/build_easyspyn.py

The package is built from a staged copy so that the source tree stays untouched and the distribution holds
exactly ``src/easyspyn`` (with its isotope table), the repository ``LICENSE`` (which carries the Spinach and
EasySpin notices), and the README and project file of ``packaging/easyspyn``. Needs the ``build`` package.
Uploading is a separate step, with the owner's PyPI token:

    TWINE_USERNAME=__token__ TWINE_PASSWORD=<token> python -m twine upload dist/easyspyn/*
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGING = ROOT / "packaging" / "easyspyn"
OUT = ROOT / "dist" / "easyspyn"


def stage(target: Path) -> None:
    shutil.copytree(
        ROOT / "src" / "easyspyn",
        target / "easyspyn",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "README.md"),
    )
    for name in ("pyproject.toml", "README.md"):
        shutil.copy2(PACKAGING / name, target / name)
    shutil.copy2(ROOT / "LICENSE", target / "LICENSE")


def main() -> int:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    with tempfile.TemporaryDirectory() as tmp:
        staged = Path(tmp) / "easyspyn"
        staged.mkdir()
        stage(staged)
        subprocess.run([sys.executable, "-m", "build", "--outdir", str(OUT), str(staged)], check=True)
    for path in sorted(OUT.iterdir()):
        print(path.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
