"""Acknowledge the authors of the Spinach functions that Spynach translates.

Spinach lists the authors of each function as email addresses at the end of the header comment of its ``.m``
file. This tool turns them into names (the table ``NAMES`` below, checked against the Spinach developer team page),
records them per function in ``porting/spinach_authors.json`` (names only), and writes them into the docstrings of
``src/spynach`` and into ``docs/spinach-credits.md``. Function names are never changed: Spynach keeps Spinach's
names so that scripts map line by line.

    python porting/tools/gen_spinach_authors.py --spinach /path/to/spinach   # refresh the JSON from a checkout
    python porting/tools/gen_spinach_authors.py --write                      # docstrings and page from the JSON
    python porting/tools/gen_spinach_authors.py --check                      # nothing stale? (no checkout needed)

Relations: ``port`` (a translation of the Spinach function of that name), ``derived`` (translated from a part of
a Spinach file, or a different implementation of what it computes), ``new`` (no Spinach counterpart; any Spinach
authors listed are those of the algorithms it implements). A header address that cannot be matched to a name from
the developer team page maps to ``None`` and is counted, not named, until a name is supplied.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from credits_common import apply_edits, wrap_block  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src" / "spynach"
JSON_PATH = ROOT / "porting" / "spinach_authors.json"
PAGE = ROOT / "docs" / "spinach-credits.md"
TEAM_URL = "https://spindynamics.org/wiki/index.php?title=Spinach_developer_team"

# Header address -> name as on the Spinach developer team page (three supplied by the project owner); None where
# a name is not known, in which case the address is counted but not named.
NAMES: dict[str, str | None] = {
    "ilya.kuprov@weizmann.ac.il": "Ilya Kuprov",
    "ledwards@cbs.mpg.de": "Luke Edwards",
    "luke.edwards@chem.ox.ac.uk": "Luke Edwards",
    "luke.edwards@ucl.ac.uk": "Luke Edwards",
    "hannah.hogben@chem.ox.ac.uk": "Hannah Hogben",
    "matthew.krzystyniak@oerc.ox.ac.uk": "Matthew Krzystyniak",
    "kpervushin@ntu.edu.sg": "Konstantin Pervushin",
    "d.savostyanov@soton.ac.uk": "Dmitry Savostyanov",
    "alex_nevzorov@ncsu.edu": "Alex Nevzorov",
    "gareth.charnock@oerc.ox.ac.uk": "Gareth Charnock",
    "bud.macaulay@soton.ac.uk": "Bud Macaulay",
    "ohad.levinkron@weizmann.ac.il": "Ohad Levinkron",
    "david.goodwin@inano.au.dk": "David Goodwin",
    "alexander.karabanov@nottingham.ac.uk": "Alexander Karabanov",
    "walter.kockenberger@nottingham.ac.uk": "Walter Kockenberger",
    "a.biternas@soton.ac.uk": "Andreas Biternas",
    "a.acharya@soton.ac.uk": "Anupama Acharya",
    # not on the developer team page; names supplied by the project owner
    "stst@uw.edu": "Stefan Stoll",
    "c.musselwhite@soton.ac.uk": "Callum Musselwhite",
    "fije@inano.au.dk": "Filip Vestergaard Jensen",
}
# Files whose header names its authors in text instead of by address.
FILE_AUTHORS = {"fourdif.m": ["J.A.C. Weideman", "S.C. Reddy"]}

EXCLUDE_SAME_NAME = {"nnz"}  # shares a name with MATLAB overloads, not a Spinach function that is translated
EXCLUDE_FUNCTIONS = {"_grumble"}  # an input check that every translated function has, not a port of one file

# Mappings the docstrings cannot give: qualified name -> [(Spinach file, relation, note)]
EXTRA: dict[str, list[tuple[str, str, str]]] = {
    "spynach.propagation._reordered_taylor": [("step.m", "derived", "subfunction reordered_taylor")],
    "spynach.propagation._commutator_series": [("step.m", "derived", "subfunction comm_series")],
    "spynach.plotting.axis_2d": [("plot_2d.m", "derived", "the axis calculation of plot_2d.m")],
    "spynach.utilities.chop": [("clean_up.m", "derived", "the rounding of clean_up.m without the storage decision")],
    "spynach.utilities.matmul_clean": [
        ("clean_up.m", "new", "fused with the product of the propagator's squarings"),
        ("propagator.m", "new", "the squarings"),
    ],
    "spynach.kernels.round_array": [("clean_up.m", "derived", "the rounding, as a compiled kernel")],
    "spynach.kernels.spgemm_round": [
        ("clean_up.m", "new", "product and rounding in one pass"),
        ("propagator.m", "new", "the squarings"),
    ],
    "spynach.kernels.taylor_term": [("step.m", "derived", "one term of subfunction reordered_taylor, fused")],
}
# Module-level credits for modules that derive from several Spinach files without a function-level mapping.
MODULE_EXTRA = {
    "spynach.backend": ["propagator.m", "step.m", "krylov.m", "evolution.m", "clean_up.m", "cheap_norm.m"],
    "spynach.kernels": ["propagator.m", "step.m", "clean_up.m"],
}

# The directories of the Spinach checkout that hold the library (examples, tests and tutorials reuse file names).
LIBRARY_DIRS = {"kernel", "experiments", "etc"}

CITE = re.compile(r"``((?:[\w]+/)*[\w]+\.m)``")


# ---------------------------------------------------------------------------- headers of the Spinach files
def spinach_index(spinach: Path) -> dict[str, str]:
    """Spinach file name -> path relative to the checkout (overloads under ``@`` directories left out)."""
    index: dict[str, str] = {}
    for path in sorted(spinach.rglob("*.m")):
        rel = path.relative_to(spinach).as_posix()
        if "@" in rel or rel.split("/")[0] not in LIBRARY_DIRS:
            continue
        if path.name in index:
            raise SystemExit(f"{path.name} exists twice in the Spinach checkout: {index[path.name]}, {rel}")
        index[path.name] = rel
    return index


def header_authors(path: Path) -> tuple[list[str], int]:
    """Names (in order of appearance, once each) and the count of addresses without a name in a header."""
    names: list[str] = []
    unnamed = 0
    seen: set[str] = set()
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith("%"):
            break
        match = re.match(r"^%\s*([\w.\-+]+@[\w.\-]+)\s*$", line)
        if not match:
            continue
        address = match.group(1).lower()
        if address not in NAMES:
            raise SystemExit(f"{path.name}: address {address} is not in NAMES; add it (a name, or None)")
        name = NAMES[address]
        key = name or address
        if key in seen:
            continue
        seen.add(key)
        if name is None:
            unnamed += 1
        else:
            names.append(name)
    return names, unnamed


# ---------------------------------------------------------------------------- what in Spynach maps to what
def module_name(path: Path) -> str:
    return f"spynach.{path.stem}"


def python_files() -> list[Path]:
    return [p for p in sorted(SRC.glob("*.py")) if p.name not in ("__init__.py", "_isotope_data.py")]


def scan(index: dict[str, str]) -> tuple[dict[str, dict], dict[str, set[str]]]:
    """Function entries (qualified name -> files, relation, note) and module -> Spinach files."""
    functions: dict[str, dict] = {}
    modules: dict[str, set[str]] = {}
    for py in python_files():
        tree = ast.parse(py.read_text(encoding="utf-8"))
        mod = module_name(py)
        modules[mod] = {Path(c).name for c in CITE.findall(ast.get_docstring(tree) or "") if Path(c).name in index}
        for node in tree.body:
            if not isinstance(node, ast.FunctionDef) or node.name in EXCLUDE_FUNCTIONS:
                continue
            qual = f"{mod}.{node.name}"
            doc = ast.get_docstring(node) or ""
            own = [Path(c).name for c in CITE.findall(doc) if Path(c).name in index]
            stem = node.name.lstrip("_")
            entry: dict = {"files": [], "relation": "new", "note": ""}
            if qual in EXTRA:
                for file, relation, note in EXTRA[qual]:
                    entry["files"].append({"file": file, "note": note})
                    entry["relation"] = relation
            elif own:
                entry["files"] = [{"file": f, "note": ""} for f in dict.fromkeys(own)]
                entry["relation"] = "port" if all(f[:-2] == stem for f in own) else "derived"
                if entry["relation"] == "derived":
                    for item in entry["files"]:
                        item["note"] = "a branch of the original" if len(own) > 1 else ""
            elif f"{stem}.m" in index and node.name not in EXCLUDE_SAME_NAME and not node.name.startswith("_"):
                entry["files"] = [{"file": f"{stem}.m", "note": ""}]
                entry["relation"] = "port"
            if node.name.startswith("_") and not entry["files"]:
                continue  # private helpers without a Spinach source need no row
            functions[qual] = entry
            modules[mod] |= {item["file"] for item in entry["files"] if item["file"] in index}
    for mod, files in MODULE_EXTRA.items():
        modules.setdefault(mod, set()).update(f for f in files if f in index)
    return functions, modules


def build_json(spinach: Path) -> dict:
    index = spinach_index(spinach)
    functions, modules = scan(index)
    needed = sorted(
        {item["file"] for entry in functions.values() for item in entry["files"]} | set().union(*modules.values())
    )
    files = {}
    for name in needed:
        if name in FILE_AUTHORS:
            authors, unnamed = list(FILE_AUTHORS[name]), 0
        else:
            authors, unnamed = header_authors(spinach / index[name])
        files[name] = {"path": index[name], "authors": authors, "unnamed": unnamed}
    commit = subprocess.run(
        ["git", "-C", str(spinach), "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    ).stdout.strip()
    return {
        "spinach_version": "2.13",
        "spinach_commit": commit,
        "team_page": TEAM_URL,
        "files": files,
        "functions": dict(sorted(functions.items())),
        "modules": {m: sorted(f) for m, f in sorted(modules.items()) if f},
    }


# ---------------------------------------------------------------------------- text
def author_text(info: dict) -> str:
    names = list(info["authors"])
    unnamed = info["unnamed"]
    if unnamed:
        extra = (
            "one contributor listed only by email address"
            if unnamed == 1
            else f"{unnamed} contributors listed only by email address"
        )
        names.append(extra)
    return ", ".join(names)


def function_block(data: dict, entry: dict) -> list[str]:
    lines = []
    for item in entry["files"]:
        label = item["file"] + (f", {item['note']}" if item["note"] else "")
        lines.append(f"Spinach authors ({label}): {author_text(data['files'][item['file']])}.")
    return lines


def module_block(data: dict, files: list[str]) -> list[str]:
    lines = [
        f"Spinach authors, by original file (headers of Spinach {data['spinach_version']}; "
        "see docs/spinach-credits.md):"
    ]
    lines += [f"  {name}: {author_text(data['files'][name])}" for name in files]
    return lines


def docstring_targets(data: dict) -> dict[Path, list[tuple[ast.Expr, list[str]]]]:
    targets: dict[Path, list[tuple[ast.Expr, list[str]]]] = {}
    for py in python_files():
        tree = ast.parse(py.read_text(encoding="utf-8"))
        mod = module_name(py)
        found: list[tuple[ast.Expr, list[str]]] = []
        if mod in data["modules"] and tree.body and isinstance(tree.body[0], ast.Expr):
            found.append((tree.body[0], wrap_block(module_block(data, data["modules"][mod]), 0)))
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                entry = data["functions"].get(f"{mod}.{node.name}")
                if entry and entry["files"] and node.body and isinstance(node.body[0], ast.Expr):
                    if isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
                        block = wrap_block(function_block(data, entry), node.body[0].col_offset)
                        found.append((node.body[0], block))
        if found:
            targets[py] = found
    return targets


def apply_docstrings(data: dict) -> int:
    return apply_edits(docstring_targets(data))


def docstring_problems(data: dict) -> list[str]:
    """Docstrings that lack the lines the JSON asks for (works from the JSON alone)."""
    problems = []
    for py, found in docstring_targets(data).items():
        for expr, block in found:
            doc = expr.value.value
            for line in block:
                if line not in doc:
                    problems.append(f"{py.relative_to(ROOT)}:{expr.lineno}: missing '{line[:70]}'")
    return problems


def uncited_files(data: dict) -> list[str]:
    """Spinach files cited in a docstring of src/spynach that the JSON does not know."""
    missing = []
    for py in python_files():
        text = py.read_text(encoding="utf-8")
        for name in {Path(c).name for c in CITE.findall(text)}:
            if name not in data["files"] and name != "basis.m":
                missing.append(f"{py.name}: {name}")
    return missing


# ---------------------------------------------------------------------------- the credits page
def render_page(data: dict) -> str:
    out = [
        "# Spinach function authors",
        "",
        "*Generated by `porting/tools/gen_spinach_authors.py` from `porting/spinach_authors.json`; do not edit.*",
        "",
        f"Spynach is a translation of [Spinach](https://spindynamics.org) {data['spinach_version']}. "
        "Each Spinach function lists its authors at the end of its header, by email address. "
        "This page gives them by name, taken from "
        f"the [Spinach developer team]({data['team_page']}) page, for every Spynach function that translates or "
        "derives from a Spinach function; the same lines are in the docstrings (`help(spynach.propagator)`). "
        "Function names are Spinach's, so that scripts map line by line, and are not changed to carry authors. "
        "A header address that cannot be matched to a name on that page is counted but not named.",
        "",
        "How a Spynach function relates to its original:",
        "",
        "- **port**: a translation of the Spinach function of that name;",
        "- **derived**: translated from part of a Spinach file, or a different implementation of what it computes;",
        "- **new**: no Spinach counterpart; any authors listed wrote the algorithm it implements.",
        "",
        "Functions of Spynach with no Spinach counterpart at all are listed with a dash. The translation was done by "
        "reading the Spinach sources; the licence notice is in [`LICENSE`](../LICENSE).",
        "",
        "## By Spynach function",
        "",
        "| Spynach function | Spinach source | Relation | Spinach authors |",
        "|---|---|---|---|",
    ]
    for qual, entry in data["functions"].items():
        if not entry["files"]:
            out.append(f"| `{qual}` | — | new | — |")
            continue
        for item in entry["files"]:
            info = data["files"][item["file"]]
            note = f" ({item['note']})" if item["note"] else ""
            out.append(f"| `{qual}` | `{info['path']}`{note} | {entry['relation']} | {author_text(info)} |")
    out += ["", "## By Spinach file", "", "| Spinach file | Authors |", "|---|---|"]
    for info in sorted(data["files"].values(), key=lambda value: value["path"]):
        out.append(f"| `{info['path']}` | {author_text(info)} |")
    out.append("")
    return "\n".join(out)


# ---------------------------------------------------------------------------------------------------------- main
def stale(data: dict) -> list[str]:
    out = list(docstring_problems(data))
    out += [f"cited but not in the JSON: {m}" for m in uncited_files(data)]
    if not PAGE.exists() or PAGE.read_text(encoding="utf-8") != render_page(data):
        out.append(str(PAGE.relative_to(ROOT)))
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--spinach", type=Path, help="Spinach checkout: refresh the JSON from the headers")
    parser.add_argument("--write", action="store_true", help="write docstrings and the credits page from the JSON")
    parser.add_argument("--check", action="store_true", help="exit 1 and list what is stale")
    args = parser.parse_args(argv)
    if args.spinach:
        data = build_json(args.spinach)
        JSON_PATH.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        unnamed = {n: i["unnamed"] for n, i in data["files"].items() if i["unnamed"]}
        print(
            f"wrote {JSON_PATH.relative_to(ROOT)}: {len(data['files'])} Spinach files, "
            f"{len(data['functions'])} functions"
        )
        if unnamed:
            print("files with authors named only by address:", unnamed)
    data = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    if args.write:
        print(f"docstrings updated in {apply_docstrings(data)} files")
        PAGE.write_text(render_page(data), encoding="utf-8")
        print(f"wrote {PAGE.relative_to(ROOT)}")
    if args.check:
        problems = stale(data)
        for problem in problems:
            print("stale:", problem)
        return 1 if problems else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
