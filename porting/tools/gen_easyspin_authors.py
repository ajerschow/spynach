"""Acknowledge the EasySpin contributors behind the functions that EPReasy translates.

EasySpin's function headers do not list authors (the project's notice is "Copyright (c) 2006-2024 Stefan Stoll and
other contributors", MIT), apart from code taken from elsewhere, such as ``jacobianest.m`` ("Author: John D'Errico").
The record of who wrote a file is its Git history, so this tool lists, for each EasySpin file that an EPReasy function
translates, the people who authored commits to that file up to the release tag (by number of commits), plus any
author named in the source itself. The result goes to ``porting/easyspin_authors.json`` (names only), into the
docstrings of ``src/epreasy`` and into ``docs/easyspin-credits.md``. Function names are never changed.

    python porting/tools/gen_easyspin_authors.py --fetch     # refresh the JSON from GitHub (needs the gh CLI)
    python porting/tools/gen_easyspin_authors.py --write     # docstrings and page from the JSON
    python porting/tools/gen_easyspin_authors.py --check     # nothing stale? (no network needed)

The lists therefore name everyone who changed a file, including small edits, and are not a statement of who wrote
which algorithm; the page says so. Relations are as in gen_spinach_authors.py.
"""

from __future__ import annotations

import argparse
import ast
import collections
import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from credits_common import apply_edits, wrap_block  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src" / "epreasy"
JSON_PATH = ROOT / "porting" / "easyspin_authors.json"
PAGE = ROOT / "docs" / "easyspin-credits.md"
REPO = "StollLab/EasySpin"
TAG = "v6.0.15"
REPO_URL = f"https://github.com/{REPO}"

# Commit-author names that are spellings or handles of one person, or GitHub handles without a profile name.
ALIASES = {
    "claudiatait": "Claudia Tait",
    "cetait": "Claudia Tait",
    "AlaCodon": "AlaCodon (GitHub user)",
    "sikuenstner": "sikuenstner (GitHub user)",
    "AlbanBloorRiley": "AlbanBloorRiley (GitHub user)",
}
# Names that are not people (bots, placeholders) are dropped.
NOT_PEOPLE = {"GitHub", "github-actions[bot]", "dependabot[bot]"}

# Mappings the function names cannot give: qualified name -> [(EasySpin file name, relation, note)]
EXTRA = {
    "epreasy.kernels.lisum1i_numpy": [("lisum1i.c", "derived", "the C kernel, in NumPy")],
    "epreasy.kernels.projecttriangles_numpy": [("projecttriangles.c", "derived", "the C kernel, in NumPy")],
    "epreasy.kernels.projectzones_numpy": [("projectzones.c", "derived", "the C kernel, in NumPy")],
    "epreasy.interp.esspline_coeffs": [("esspline1d.m", "derived", "the coefficient step")],
    "epreasy.interp.esspline_eval": [("esspline1d.m", "derived", "the evaluation step")],
    "epreasy.resfields.resfields_oriented": [("resfields.m", "derived", "the per-orientation part")],
    "epreasy.perturb.resfields_perturb_oriented": [("resfields_perturb.m", "derived", "the per-orientation part")],
    "epreasy.pulsed._sf_peaks": [("sf_peaks.c", "derived", "the peak-binning kernel, in NumPy")],
    "epreasy.pulsed._sf_evolve": [("saffron.m", "derived", "the time-domain evolution of the fast engine")],
    "epreasy.pulsed._pathwayparser": [("saffron.m", "derived", "the pathway parser of the fast engine")],
}
EXCLUDE_MODULES = {"__init__", "oracle"}
SKIP_STEMS = {"main"}


# ---------------------------------------------------------------------------- reading EasySpin through gh
def gh(*args: str) -> str:
    result = subprocess.run(["gh", "api", *args], capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(f"gh api {' '.join(args)} failed: {result.stderr.strip()[:300]}")
    return result.stdout


def easyspin_index() -> dict[str, str]:
    """File stem -> path under easyspin/ at the release tag (.m and .c files; overload directories left out)."""
    paths = gh(f"repos/{REPO}/git/trees/{TAG}?recursive=1", "--jq", '.tree[] | select(.type=="blob") | .path').split()
    index: dict[str, str] = {}
    for path in paths:
        if path.startswith("easyspin/") and re.search(r"\.(m|c)$", path) and "/@" not in path:
            stem = Path(path).name
            if stem in index:
                raise SystemExit(f"{stem} exists twice in EasySpin: {index[stem]}, {path}")
            index[stem] = path
    return index


def commit_authors(path: str) -> list[tuple[str, int]]:
    """People who authored commits to ``path`` up to the tag, by number of commits."""
    text = gh(
        "--paginate",
        f"repos/{REPO}/commits?path={path}&sha={TAG}&per_page=100",
        "--jq",
        ".[] | .commit.author.name",
    )
    counts: collections.Counter[str] = collections.Counter()
    for name in text.splitlines():
        name = ALIASES.get(name.strip(), name.strip())
        if name and name not in NOT_PEOPLE:
            counts[name] += 1
    return counts.most_common()


def source_credits(path: str) -> list[str]:
    """Authors named in the source itself (``% Author: ...``)."""
    text = gh(f"repos/{REPO}/contents/{path}?ref={TAG}", "-H", "Accept: application/vnd.github.raw")
    return [
        m.group(1).strip()
        for line in text.splitlines()
        if (m := re.match(r"^\s*(?:%|//|\*|/\*)\s*Authors?:\s*(.+?)\s*$", line))
    ]


# ---------------------------------------------------------------------------- what in EPReasy maps to what
def python_files() -> list[Path]:
    return [p for p in sorted(SRC.glob("*.py")) if p.stem not in EXCLUDE_MODULES]


def scan(index: dict[str, str]) -> tuple[dict[str, dict], dict[str, set[str]]]:
    """Function entries (qualified name -> files, relation, note) and module -> EasySpin files."""
    by_stem = {Path(name).stem: name for name in index}
    functions: dict[str, dict] = {}
    modules: dict[str, set[str]] = {}
    for py in python_files():
        tree = ast.parse(py.read_text(encoding="utf-8"))
        mod = f"epreasy.{py.stem}"
        modules[mod] = set()
        for node in tree.body:
            if not isinstance(node, ast.FunctionDef):
                continue
            qual = f"{mod}.{node.name}"
            stem = node.name.lstrip("_")
            entry: dict = {"files": [], "relation": "new", "note": ""}
            if qual in EXTRA:
                entry["files"] = [{"file": f, "note": n} for f, _, n in EXTRA[qual]]
                entry["relation"] = EXTRA[qual][0][1]
            else:
                hit = next((c for c in (stem, "chili_" + stem, "esfit_" + stem) if c in by_stem), None)
                if hit:
                    entry["files"] = [{"file": by_stem[hit], "note": ""}]
                    entry["relation"] = "port" if hit == stem and not node.name.startswith("_") else "derived"
                    if entry["relation"] == "derived":
                        entry["files"][0]["note"] = "translated as a private function"
            if node.name.startswith("_") and not entry["files"]:
                continue
            functions[qual] = entry
            modules[mod] |= {item["file"] for item in entry["files"]}
    return functions, modules


def build_json() -> dict:
    index = easyspin_index()
    functions, modules = scan(index)
    needed = sorted({item["file"] for e in functions.values() for item in e["files"]})
    files = {}
    for name in needed:
        path = index[name]
        contributors = commit_authors(path)
        external = source_credits(path)
        files[name] = {
            "path": path,
            "external": external,
            "contributors": [{"name": n, "commits": c} for n, c in contributors],
        }
        print(
            f"  {name:28s} {', '.join(n for n, _ in contributors) or '-'}"
            + (f"  [{', '.join(external)}]" if external else "")
        )
    return {
        "easyspin_version": TAG.lstrip("v"),
        "tag": TAG,
        "repository": REPO_URL,
        "notice": "Copyright (c) 2006-2024 Stefan Stoll and other contributors",
        "files": files,
        "functions": dict(sorted(functions.items())),
        "modules": {m: sorted(f) for m, f in sorted(modules.items()) if f},
    }


# ---------------------------------------------------------------------------- text
def author_text(info: dict) -> str:
    names = list(info["external"])
    for contributor in info["contributors"]:
        if contributor["name"] not in names:
            names.append(contributor["name"])
    return ", ".join(names) if names else "no author recorded"


def label(item: dict, info: dict) -> str:
    parts = [item["file"]]
    if item["note"]:
        parts.append(item["note"])
    if info["external"]:
        parts.append(f"code of {', '.join(info['external'])} included in EasySpin")
    return ", ".join(parts)


def function_block(data: dict, entry: dict) -> list[str]:
    return [
        f"EasySpin authors ({label(item, data['files'][item['file']])}): {author_text(data['files'][item['file']])}."
        for item in entry["files"]
    ]


def module_block(data: dict, files: list[str]) -> list[str]:
    head = (
        f"EasySpin authors, by original file (contributors to the file up to EasySpin {data['easyspin_version']}, "
        "see docs/easyspin-credits.md):"
    )
    return [head, *[f"  {name}: {author_text(data['files'][name])}" for name in files]]


def docstring_targets(data: dict) -> dict[Path, list[tuple[ast.Expr, list[str]]]]:
    targets: dict[Path, list[tuple[ast.Expr, list[str]]]] = {}
    for py in python_files():
        tree = ast.parse(py.read_text(encoding="utf-8"))
        mod = f"epreasy.{py.stem}"
        found: list[tuple[ast.Expr, list[str]]] = []
        if mod in data["modules"] and tree.body and isinstance(tree.body[0], ast.Expr):
            found.append((tree.body[0], wrap_block(module_block(data, data["modules"][mod]), 0)))
        for node in tree.body:
            if not isinstance(node, ast.FunctionDef):
                continue
            entry = data["functions"].get(f"{mod}.{node.name}")
            if not (entry and entry["files"] and node.body and isinstance(node.body[0], ast.Expr)):
                continue
            if isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
                found.append((node.body[0], wrap_block(function_block(data, entry), node.body[0].col_offset)))
        if found:
            targets[py] = found
    return targets


def without_docstring(data: dict) -> list[str]:
    """Mapped functions that have no docstring to put the line in (they are still in the credits page)."""
    missing = []
    for py in python_files():
        tree = ast.parse(py.read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                qual = f"epreasy.{py.stem}.{node.name}"
                if data["functions"].get(qual, {}).get("files") and not ast.get_docstring(node):
                    missing.append(qual)
    return missing


def docstring_problems(data: dict) -> list[str]:
    problems = []
    for py, found in docstring_targets(data).items():
        for expr, block in found:
            doc = expr.value.value
            problems += [
                f"{py.relative_to(ROOT)}:{expr.lineno}: missing '{line[:70]}'" for line in block if line not in doc
            ]
    return problems


# ---------------------------------------------------------------------------- the credits page
def render_page(data: dict) -> str:
    out = [
        "# EasySpin function authors",
        "",
        "*Generated by `porting/tools/gen_easyspin_authors.py` from `porting/easyspin_authors.json`; do not edit.*",
        "",
        f"EPReasy is a translation of [EasySpin](https://easyspin.org) {data['easyspin_version']}, "
        f"{data['notice']}, MIT licence (the notice is in [`LICENSE`](../LICENSE)). EasySpin's function headers do "
        "not list authors, so the record used here is the Git history of each original file at the release tag "
        f"[{data['tag']}]({data['repository']}/tree/{data['tag']}): the people who authored commits to it, in order of "
        "the number of commits, together with any author the source itself names (code taken from elsewhere). The "
        "lists therefore include everyone who changed a file, small edits as well as large ones, and do not say who "
        "wrote which algorithm. The same lines are in the docstrings (`help(epreasy.pepper)`). Function names are "
        "EasySpin's and are not changed to carry authors.",
        "",
        "How an EPReasy function relates to its original:",
        "",
        "- **port**: a translation of the EasySpin function of that name;",
        "- **derived**: translated from part of an EasySpin file, a private helper, or a different implementation "
        "of what it computes (the compiled kernels are rewritten in NumPy);",
        "- **new**: no EasySpin counterpart.",
        "",
        "## By EPReasy function",
        "",
        "| EPReasy function | EasySpin source | Relation | EasySpin authors |",
        "|---|---|---|---|",
    ]
    for qual, entry in data["functions"].items():
        if not entry["files"]:
            out.append(f"| `{qual}` | — | new | — |")
            continue
        for item in entry["files"]:
            info = data["files"][item["file"]]
            notes = [item["note"]] if item["note"] else []
            if info["external"]:
                notes.append(f"code of {', '.join(info['external'])} included in EasySpin")
            note = f" ({', '.join(notes)})" if notes else ""
            out.append(f"| `{qual}` | `{info['path']}`{note} | {entry['relation']} | {author_text(info)} |")
    out += ["", "## By EasySpin file", "", "| EasySpin file | Authors (commits) |", "|---|---|"]
    for info in sorted(data["files"].values(), key=lambda value: value["path"]):
        people = [*info["external"], *[f"{c['name']} ({c['commits']})" for c in info["contributors"]]]
        out.append(f"| `{info['path']}` | {', '.join(people) or '—'} |")
    out.append("")
    return "\n".join(out)


# ---------------------------------------------------------------------------------------------------------- main
def stale(data: dict) -> list[str]:
    out = list(docstring_problems(data))
    if not PAGE.exists() or PAGE.read_text(encoding="utf-8") != render_page(data):
        out.append(str(PAGE.relative_to(ROOT)))
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fetch", action="store_true", help="refresh the JSON from GitHub (gh CLI)")
    parser.add_argument("--write", action="store_true", help="write docstrings and the credits page from the JSON")
    parser.add_argument("--check", action="store_true", help="exit 1 and list what is stale")
    args = parser.parse_args(argv)
    if args.fetch:
        data = build_json()
        JSON_PATH.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(
            f"wrote {JSON_PATH.relative_to(ROOT)}: {len(data['files'])} EasySpin files, "
            f"{len(data['functions'])} functions"
        )
    data = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    if args.write:
        print(f"docstrings updated in {apply_edits(docstring_targets(data))} files")
        PAGE.write_text(render_page(data), encoding="utf-8")
        print(f"wrote {PAGE.relative_to(ROOT)}")
        missing = without_docstring(data)
        if missing:
            print(f"{len(missing)} mapped functions have no docstring (credited on the page only): {missing}")
    if args.check:
        problems = stale(data)
        for problem in problems:
            print("stale:", problem)
        return 1 if problems else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
