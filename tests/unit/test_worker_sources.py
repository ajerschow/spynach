"""The browser playground (try.html) runs Spynach, easyspyn, ctps and spyndynamica in Pyodide; worker.js fetches
the sources from a list that scripts/build_site.py generates, which has to name every module and data file of
the packages or the import fails in the browser only."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PACKAGES = ("spynach", "ctps", "easyspyn", "spyndynamica")
EXTRA = {"spyndynamica": {".json", ".npz"}}  # as scripts/build_site.py SOURCE_EXTRA
SUFFIXES = {".py", ".txt", ".DSC", ".DTA"}  # as scripts/build_site.py SOURCE_SUFFIXES


def test_worker_fetches_every_module_and_data_file():
    worker = (ROOT / "worker.js").read_text()
    listed = set(re.findall(r'"(src/[\w/.-]+)"', re.search(r"const SOURCE = \[(.*?)\];", worker, re.S).group(1)))
    on_disk = {
        path.relative_to(ROOT).as_posix()
        for package in PACKAGES
        for path in (ROOT / "src" / package).rglob("*")
        if path.suffix in SUFFIXES | EXTRA.get(package, set()) and "__pycache__" not in path.parts
    }
    assert listed == on_disk, f"worker.js SOURCE: missing {sorted(on_disk - listed)}, stale {sorted(listed - on_disk)}"
    assert "src/easyspyn/data/isotopedata.txt" in listed
    assert "src/spyndynamica/data/orientations.npz" in listed and "src/spynach/grids/single_crystal.npz" not in listed
    # the binary data file is written as bytes, and SymPy is loaded when a run imports spyndynamica
    assert "DTA|dta|npz" in worker and '"spyndynamica" not in code' in worker
