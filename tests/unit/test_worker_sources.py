"""The browser playground (try.html) runs Spynach in Pyodide; worker.js fetches the Python sources from a
hard-coded list, which has to name every module of spynach and ctps or the import fails in the browser only."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_worker_fetches_every_module():
    worker = (ROOT / "worker.js").read_text()
    listed = set(re.findall(r'"(src/[\w/]+\.py)"', re.search(r"const SOURCE = \[(.*?)\];", worker, re.S).group(1)))
    on_disk = {
        path.relative_to(ROOT).as_posix()
        for package in ("spynach", "ctps")
        for path in (ROOT / "src" / package).rglob("*.py")
        if "__pycache__" not in path.parts
    }
    assert listed == on_disk, f"worker.js SOURCE: missing {sorted(on_disk - listed)}, stale {sorted(listed - on_disk)}"
