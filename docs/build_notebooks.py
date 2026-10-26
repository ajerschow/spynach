"""Generate a Jupyter notebook from each Markdown guide in docs/.

The Markdown guides are the source of truth: they render on any Git host and
their Python blocks are executed by tests/docs/. Each guide docs/<name>.md
becomes docs/notebooks/<name>.ipynb, with every ```python block as a code
cell and the prose in between as Markdown cells split at section headings.
Each notebook is then run in a fresh kernel and saved with its outputs, so
the plots and results are there as soon as it is opened. Every Plotly
figure is saved with a PNG copy in the same output (GitHub's notebook
viewer shows the PNG, JupyterLab, VS Code and nbviewer the interactive
figure) and written to docs/figures/ under the name that the image line
following the plotting block in the guide gives it, so that the guide
shows the figure on GitHub; the image line itself is left out of the
notebook, where the cell above shows the figure.

    python docs/build_notebooks.py               # rebuild and run every notebook
    python docs/build_notebooks.py --no-execute  # rebuild without outputs (quick)
    python docs/build_notebooks.py --check       # exit 1 if any is stale or not run

Running needs the "docs" extra (Plotly, nbclient, ipykernel, and Kaleido or
Matplotlib for the PNG copies); building and checking use only the
standard library.
"""

from __future__ import annotations

import argparse
import base64
import functools
import json
import os
import re
import sys
from pathlib import Path

DOCS = Path(__file__).resolve().parent
NOTEBOOKS = DOCS / "notebooks"
FIGURES = DOCS / "figures"
PLOTLY = "application/vnd.plotly.v1+json"

# Guides that are not tutorials and get no notebook
SKIP = {"README.md", "parity.md", "spinach-credits.md"}  # the index and the generated parity and credits pages

FENCE = re.compile(r"^```python\n(.*?)^```$", flags=re.S | re.M)
HEADING = re.compile(r"^(?=#{1,3} )", flags=re.M)
LINK = re.compile(r"\]\((?!https?://|mailto:)([^)]+)\)")
# An image line right after a plotting block names the figure file the block's output is written to
LEADING_IMAGE = re.compile(r"\n*!\[[^\]]*\]\(figures/([\w.-]+\.png)\)[ \t]*(?:\n|$)")

INTRO = (
    "*Generated from [`docs/{name}.md`](../{name}.md) by `docs/build_notebooks.py`;"
    " edit the Markdown guide, not this notebook.*\n\n"
    "Run the cells in order: later cells use names defined by earlier ones."
    " The saved outputs come from running the notebook top to bottom;"
    ' plotting cells need Plotly (`pip install -e ".[docs]"`).'
)

# Run before the first cell when executing, and not saved: store Plotly figures
# as JSON that JupyterLab and VS Code render interactively, without embedding
# plotly.js in every notebook
SETUP = "import plotly.io as _pio\n_pio.renderers.default = 'plotly_mimetype'"

METADATA = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}


def guides() -> list[Path]:
    return sorted(p for p in DOCS.glob("*.md") if p.name not in SKIP)


def notebook_path(guide: Path) -> Path:
    return NOTEBOOKS / f"{guide.stem}.ipynb"


def build(guide: Path) -> str:
    """Notebook JSON text for one guide, without outputs."""
    return dumps(build_notebook(guide))


def build_notebook(guide: Path) -> dict:
    """Notebook for one guide, as a dictionary, without outputs."""
    text = guide.read_text(encoding="utf-8")
    cells: list[tuple[str, str]] = []
    position = 0
    for match in FENCE.finditer(text):
        cells.extend(
            ("markdown", part) for part in _split_prose(_without_image(text[position : match.start()], position))
        )
        cells.append(("code", match.group(1)))
        position = match.end()
    cells.extend(("markdown", part) for part in _split_prose(_without_image(text[position:], position)))

    # The title cell comes first, then the provenance note
    intro = ("markdown", INTRO.format(name=guide.stem))
    cells.insert(1 if cells and cells[0][1].startswith("# ") else 0, intro)

    return {
        "cells": [_cell(kind, source, n) for n, (kind, source) in enumerate(cells)],
        "metadata": METADATA,
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def _without_image(prose: str, position: int) -> str:
    """The prose after a code block without the image line that names the block's figure file."""
    return LEADING_IMAGE.sub("", prose, count=1) if position else prose


def figure_files(guide: Path) -> list[str | None]:
    """For each Python block of the guide, the figure file named by the image line right after it, or None."""
    text = guide.read_text(encoding="utf-8")
    names = []
    for match in FENCE.finditer(text):
        image = LEADING_IMAGE.match(text, match.end())
        names.append(image.group(1) if image else None)
    return names


def dumps(notebook: dict) -> str:
    return json.dumps(notebook, indent=1, ensure_ascii=False) + "\n"


def strip_outputs(text: str) -> str:
    """Notebook JSON text with outputs and execution counts removed."""
    notebook = json.loads(text)
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
    return dumps(notebook)


def is_executed(text: str) -> bool:
    """True if every code cell of the notebook has been run."""
    return all(c["execution_count"] is not None for c in json.loads(text)["cells"] if c["cell_type"] == "code")


def execute(notebook: dict, figures: list[str | None] | None = None) -> dict:
    """Run a notebook top to bottom in a fresh kernel and return it with outputs.

    The kernel imports spynach from this checkout's src/, whatever is
    installed, and runs in docs/notebooks/ as it would when opened there.
    Every Plotly figure output gets a PNG copy in the same output. With
    ``figures`` (one entry per code cell, from :func:`figure_files`) the PNG
    of each plotting cell is also written to docs/figures/ under that name;
    a plotting cell without a name, or a name without a figure, is an error.
    """
    import nbclient
    import nbformat

    env_path = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = os.pathsep.join(p for p in (str(DOCS.parent / "src"), env_path) if p)
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

    setup = {
        "cell_type": "code",
        "id": "setup",
        "metadata": {},
        "source": SETUP,
        "execution_count": None,
        "outputs": [],
    }
    node = nbformat.reads(json.dumps({**notebook, "cells": [setup, *notebook["cells"]]}), as_version=4)
    nbclient.NotebookClient(
        node,
        timeout=600,
        kernel_name="python3",
        record_timing=False,
        resources={"metadata": {"path": str(NOTEBOOKS)}},
    ).execute()

    # Keep the cells as built and add the outputs, without the setup cell;
    # execution counts then start at 1
    cells = []
    block = 0
    for cell, run in zip(notebook["cells"], node["cells"][1:], strict=True):
        if cell["cell_type"] == "code":
            outputs = _merge_streams(json.loads(json.dumps(run["outputs"])))
            pngs = [_add_png(output) for output in outputs if PLOTLY in output.get("data", {})]
            if figures is not None:
                _write_figure(figures[block], pngs, block)
            block += 1
            cell = {**cell, "execution_count": run["execution_count"] - 1, "outputs": outputs}
        cells.append(cell)
    return {**notebook, "cells": cells}


def _merge_streams(outputs: list[dict]) -> list[dict]:
    """Join consecutive stream outputs of the same name: the kernel splits them at arbitrary points."""
    merged: list[dict] = []
    for output in outputs:
        last = merged[-1] if merged else None
        if (
            output["output_type"] == "stream"
            and last
            and last["output_type"] == "stream"
            and last["name"] == output["name"]
        ):
            last["text"] = "".join(last["text"]) + "".join(output["text"])
        else:
            merged.append(output)
    return merged


def _add_png(output: dict) -> bytes:
    """Add a PNG copy of the Plotly figure to the output's data; return a larger PNG for docs/figures/.

    The embedded copy is rendered at screen resolution to keep the notebook small (GitHub's
    content API, which nbviewer reads through, serves files up to 1 MB); the figure file for the
    guides is rendered at twice that.
    """
    sys.path.insert(0, str(DOCS.parent / "src"))
    import plotly.io as pio

    from spynach.plotting import figure_png

    figure = pio.from_json(json.dumps(output["data"][PLOTLY]))
    embedded = figure_png(figure, scale=1)
    # The PNG goes first: viewers that rank types (JupyterLab, VS Code, nbviewer) still pick the
    # Plotly figure, and a viewer that takes the first type it knows gets the image
    output["data"] = {"image/png": base64.b64encode(embedded).decode("ascii"), **output["data"]}
    return figure_png(figure, scale=2)


def _write_figure(name: str | None, pngs: list[bytes], block: int) -> None:
    if name is None and not pngs:
        return
    if name is None:
        raise ValueError(f"Python block {block} plots but no image line after it names its figure file")
    if len(pngs) != 1:
        raise ValueError(f"Python block {block} names a figure file but shows {len(pngs)} figures instead of one")
    FIGURES.mkdir(exist_ok=True)
    (FIGURES / name).write_bytes(pngs[0])


def _split_prose(prose: str) -> list[str]:
    """Markdown cells for the prose between code blocks, one per section heading."""
    parts = [_relink(part.strip("\n")) for part in HEADING.split(prose)]
    return [part for part in parts if part.strip()]


def _relink(markdown: str) -> str:
    """Point links at their targets as seen from docs/notebooks/.

    Links to other guides go to the sibling notebook, links to anything
    else one directory up, and section anchors are translated from
    GitHub's slugs to Jupyter's.
    """

    def target(match: re.Match[str]) -> str:
        path, _, anchor = match.group(1).partition("#")
        if path.startswith("notebooks/"):
            path = path.removeprefix("notebooks/")
        elif "/" not in path and path.endswith(".md") and path not in SKIP:
            path = Path(path).with_suffix(".ipynb").name
        elif path:
            path = f"../{path}"
        if anchor:
            anchor = "#" + _jupyter_anchors().get(anchor, anchor)
        return f"]({path}{anchor})"

    return LINK.sub(target, markdown)


@functools.cache
def _jupyter_anchors() -> dict[str, str]:
    """Map GitHub heading slugs to Jupyter heading ids, over all guides.

    GitHub lowercases the heading text and drops punctuation; Jupyter keeps
    the rendered text and replaces spaces with hyphens.
    """
    anchors = {}
    for guide in guides():
        prose = re.sub(r"^```.*?^```$", "", guide.read_text(encoding="utf-8"), flags=re.S | re.M)
        for heading in re.findall(r"^#{1,6} (.+)$", prose, flags=re.M):
            text = heading.replace("`", "").strip()
            github = re.sub(r"[^\w\- ]", "", text.lower()).replace(" ", "-")
            anchors.setdefault(github, text.replace(" ", "-"))
    return anchors


def _cell(kind: str, source: str, n: int) -> dict:
    cell = {"cell_type": kind, "id": f"cell-{n:03d}", "metadata": {}, "source": source.splitlines(keepends=True)}
    if kind == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    # A trailing newline in the last line is noise in Jupyter
    if cell["source"]:
        cell["source"][-1] = cell["source"][-1].rstrip("\n")
    return cell


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="report stale notebooks instead of writing them")
    parser.add_argument("--no-execute", action="store_true", help="write notebooks without running them")
    args = parser.parse_args()
    NOTEBOOKS.mkdir(exist_ok=True)
    status = 0
    for guide in guides():
        path = notebook_path(guide)
        name = path.relative_to(DOCS.parent)
        current = path.read_text(encoding="utf-8") if path.exists() else None
        if args.check:
            if current is None or strip_outputs(current) != build(guide):
                print(f"stale: {name}")
                status = 1
            elif not is_executed(current):
                print(f"not run: {name}")
                status = 1
            for figure in filter(None, figure_files(guide)):
                if not (FIGURES / figure).exists():
                    print(f"missing: docs/figures/{figure}")
                    status = 1
            continue
        notebook = build_notebook(guide)
        text = dumps(notebook if args.no_execute else execute(notebook, figure_files(guide)))
        if text != current:
            path.write_text(text, encoding="utf-8")
            print(f"wrote: {name}")
    return status


if __name__ == "__main__":
    sys.exit(main())
