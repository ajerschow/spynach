"""The Python runner embedded in worker.js (the RUNNER string that Pyodide executes around the editor's code)
runs natively against every playground example with stand-ins for the JavaScript callbacks: the plotting
helpers accept easyspyn spectra and leave EPR axes uninverted, the esfit wrapper reports iterations, the
grid prefetch tolerates a missing server, and no example raises. Examples marked data-slow are skipped."""

import ast
import asyncio
import json
import re
import sys
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_playground_examples import ALL  # noqa: E402


def runner_source(tmp_path: Path) -> str:
    worker = (ROOT / "worker.js").read_text(encoding="utf-8")
    source = re.search(r"const RUNNER = `\n(.*?)`;", worker, re.S).group(1)
    return source.replace("/repo/src/spynach/grids", str(tmp_path / "grids"))


class Calls:
    def __init__(self) -> None:
        self.lines: list[dict] = []
        self.images: list[dict] = []
        self.chords: list[dict] = []
        self.progress: list[tuple] = []

    def clear(self) -> None:
        self.lines, self.images, self.chords, self.progress = [], [], [], []


CALLS = Calls()


def install_stubs() -> None:
    """Fake ``js`` and ``pyodide.http`` modules, as the worker provides them, installed once: the runner wraps
    easyspyn.esfit and spynach.powder with hooks that keep referring to the first run's callbacks, as they do in
    Pyodide, where every run shares one global namespace."""
    if "js" in sys.modules:
        return
    js = types.ModuleType("js")
    js.beginLine = lambda title, xlabel, ylabel, invert, x: CALLS.lines.append(
        {"title": title, "xlabel": xlabel, "invert": invert, "n": len(x), "series": []}
    )
    js.addSeries = lambda name, y: CALLS.lines[-1]["series"].append((name, len(y)))
    js.beginImage = lambda title, xlabel, ylabel, invert, x, y, rows, cols: CALLS.images.append(
        {"title": title, "invert": invert, "rows": rows, "cols": cols, "nx": len(x), "ny": len(y)}
    )
    js.setImageZ = lambda z: CALLS.images[-1].update(nz=np.asarray(z).size)
    js.beginChord = lambda text: CALLS.chords.append(json.loads(text))
    js.reportProgress = lambda run_id, kind, done, total: CALLS.progress.append((kind, done, total))
    pyodide = types.ModuleType("pyodide")
    http = types.ModuleType("pyodide.http")

    async def pyfetch(url):
        return types.SimpleNamespace(status=404)

    http.pyfetch = pyfetch
    pyodide.http = http
    sys.modules["js"] = js
    sys.modules["pyodide"] = pyodide
    sys.modules["pyodide.http"] = http


@pytest.fixture
def js_stubs():
    install_stubs()
    CALLS.clear()
    return CALLS


def run_example(code: str, tmp_path: Path) -> tuple[dict, str]:
    ns = {"USER_CODE": code, "RUN_ID": 1, "RUN_ERROR": ""}
    compiled = compile(runner_source(tmp_path), "<runner>", "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
    result = eval(compiled, ns)
    if asyncio.iscoroutine(result):
        asyncio.run(result)
    return ns, ns["RUN_ERROR"]


@pytest.mark.parametrize("example", [e for e in ALL if not e[2]], ids=lambda e: e[0])
def test_runner_executes_example(example, js_stubs, tmp_path):
    example_id, group, _, code = example
    _, error = run_example(code, tmp_path)
    assert error == "", error
    if example_id == "ex-ctps":
        assert not js_stubs.lines and not js_stubs.images
        return
    if example_id == "ex-pathways":
        (chord,) = js_stubs.chords
        assert chord["ribbons"] and chord["arcs"] and chord["title"].startswith("11BF4-")
        assert {"outline", "hover", "colour"} <= set(chord["ribbons"][0])
        return
    assert js_stubs.lines or js_stubs.images, f"{example_id} drew nothing"
    for line in js_stubs.lines:
        assert line["series"] and all(n == line["n"] for _, n in line["series"])
    for image in js_stubs.images:
        assert image["nz"] == image["rows"] * image["cols"] == image["nx"] * image["ny"]
    if group.startswith("EPR"):
        assert all(not p["invert"] for p in js_stubs.lines + js_stubs.images), f"{example_id} inverted an EPR axis"
    if example_id == "ex-epr-esfit":
        assert any(kind == "fit iteration" and total == 0 for kind, _, total in js_stubs.progress)


def test_runner_plots_leftover_spectrum_and_spectrum_objects(js_stubs, tmp_path):
    code = "import numpy as np\nspectrum = np.ones(8)\naxis = np.arange(8.0)\nxlabel = 'ppm'"
    _, error = run_example(code, tmp_path)
    assert error == "" and js_stubs.lines[-1]["invert"] is True  # the NMR fallback keeps its convention
    code = (
        "import easyspyn as es\n"
        "spec = es.pepper({'g': 2, 'lw': 1}, {'mwFreq': 9.5, 'Range': [335, 345], 'nPoints': 64})\n"
        "show_spectrum(spec, xlabel='B / mT')"
    )
    _, error = run_example(code, tmp_path)
    assert error == "" and js_stubs.lines[-1]["invert"] is False and js_stubs.lines[-1]["n"] == 64
