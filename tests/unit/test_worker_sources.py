"""The browser playground (try.html) runs Spynach, epreasy and ctps in Pyodide; worker.js fetches the sources
from a list that scripts/build_site.py generates, which has to name every module and data file of the three
packages or the import fails in the browser only."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PACKAGES = ("spynach", "ctps", "epreasy")
SUFFIXES = {".py", ".txt"}


def test_worker_fetches_every_module_and_data_file():
    worker = (ROOT / "worker.js").read_text()
    listed = set(re.findall(r'"(src/[\w/.-]+)"', re.search(r"const SOURCE = \[(.*?)\];", worker, re.S).group(1)))
    on_disk = {
        path.relative_to(ROOT).as_posix()
        for package in PACKAGES
        for path in (ROOT / "src" / package).rglob("*")
        if path.suffix in SUFFIXES and "__pycache__" not in path.parts
    }
    assert listed == on_disk, f"worker.js SOURCE: missing {sorted(on_disk - listed)}, stale {sorted(listed - on_disk)}"
    assert "src/epreasy/data/isotopedata.txt" in listed
