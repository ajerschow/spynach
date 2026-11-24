"""The cell view of the playground: assets/playground-cells.js splits a source at its "# %%" lines and joins it
again (under Node), every cell of every example in cells is Python on its own, and each such example runs cell by
cell through the worker's runner with a value under most cells."""

import ast
import json
import shutil
import subprocess

import pytest

from tests.unit.test_playground_examples import ALL, ROOT
from tests.unit.test_playground_runner import js_stubs, run_cell  # noqa: F401

CELL_EXAMPLES = [e for e in ALL if "# %%" in e[3]]

SCRIPT = """
import { readFileSync } from "node:fs";
import { hasCells, joinCells, splitCells } from %r;
const examples = JSON.parse(readFileSync(0, "utf8"));
const out = {};
for (const { id, source } of examples) {
  const cells = splitCells(source);
  out[id] = { has: hasCells(source), cells, joined: joinCells(cells), again: splitCells(joinCells(cells)) };
}
const src = "import x\\n\\n\\n# %%%% First\\na = 1\\n\\na\\n\\n# %%%%\\n\\n# %%%%   Third title  \\nb = 2\\n";
const crlf = splitCells(src.replace(/\\n/g, "\\r\\n"));
out.synthetic = { cells: splitCells(src), joined: joinCells(splitCells(src)), crlf };
out.plain = { has: hasCells("a = 1  # %%%% not a marker\\n"), cells: splitCells("a = 1\\n") };
console.log(JSON.stringify(out));
"""


@pytest.fixture(scope="module")
def node():
    if shutil.which("node") is None:
        pytest.skip("needs Node")
    payload = json.dumps([{"id": e[0], "source": e[3]} for e in ALL])
    module = (ROOT / "assets" / "playground-cells.js").as_uri()
    done = subprocess.run(
        ["node", "--input-type=module", "-e", SCRIPT % module], input=payload, capture_output=True, text=True
    )
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


def test_there_are_examples_in_cells():
    assert len(CELL_EXAMPLES) >= 6 and all(e[1] == "Symbolic NMR (SpynDynamica)" for e in CELL_EXAMPLES)


def test_cells_are_split_and_joined(node):
    synthetic = node["synthetic"]
    assert synthetic["cells"] == [
        {"title": "", "code": "import x"},
        {"title": "First", "code": "a = 1\n\na"},
        {"title": "", "code": ""},
        {"title": "Third title", "code": "b = 2"},
    ]
    assert synthetic["joined"] == "# %%\nimport x\n\n# %% First\na = 1\n\na\n\n# %%\n\n# %% Third title\nb = 2"
    assert synthetic["crlf"] == synthetic["cells"]
    assert node["plain"] == {"has": False, "cells": [{"title": "", "code": "a = 1"}]}
    for example_id, _, _, code in ALL:
        entry = node[example_id]
        assert entry["has"] == ("# %%" in code), example_id
        if not entry["has"]:
            continue
        # the examples are stored in the form that joining gives, so an edit in a cell changes nothing else
        assert entry["joined"] == code.strip(), example_id
        assert entry["again"] == entry["cells"]
        assert all(cell["title"] for cell in entry["cells"]), f"{example_id}: every cell has a title"


@pytest.mark.parametrize("example", CELL_EXAMPLES, ids=lambda e: e[0])
def test_example_runs_cell_by_cell(example, node, js_stubs, tmp_path):  # noqa: F811
    example_id, _, _, code = example
    ns: dict = {}
    shown = 0
    for index, cell in enumerate(node[example_id]["cells"]):
        ast.parse(cell["code"])
        value, error = run_cell(cell["code"], ns, tmp_path, reset=index == 0, context=code)
        assert error == "", f"{example_id}, cell {index + 1}: {error}"
        if value:
            shown += 1
            entry = json.loads(value)
            assert entry["text"] and "$" not in entry.get("latex", "")
    # a cell shows a value, prints or draws; most show a value
    assert shown >= len(node[example_id]["cells"]) - 2
