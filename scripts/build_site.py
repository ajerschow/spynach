"""Rewrites the generated spans of the project site from the data they come from.

The site is hand-written HTML at the repository root (index, spinach, easyspin,
extras, docs, try), served by GitHub Pages as is. The parts that are data rather
than prose sit between HTML-comment markers and are rewritten by this script:

    <!-- site:nav --> ... <!-- /site:nav -->                  the top navigation of every page
    <!-- site:timing-spinach --> ...                          spinach.html, from comparison/runs/*/summary.json
    <!-- site:timing-easyspin --> ...                         easyspin.html, from comparison/easyspin/timings.json
    <!-- site:docs-<group> --> ...                            docs.html, from the guide table of docs/README.md
    const SOURCE = [ ... ];                                   worker.js, the files the playground fetches

    python scripts/build_site.py            # rewrite the spans
    python scripts/build_site.py --check    # exit 1 and name the files whose spans are stale

tests/unit/test_site.py runs the check. The parity paragraph of spinach.html is
written by oracle/parity_summary.py between its own markers, not by this script.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GITHUB = "https://github.com/ajerschow/spynach"
NBVIEWER = "https://nbviewer.org/github/ajerschow/spynach/blob/main/docs/notebooks"

SITE = "https://ajerschow.github.io/spynach/"
PAGES = ("index.html", "spinach.html", "easyspin.html", "extras.html", "docs.html", "try.html")
NAV = (
    ("spinach.html", "Spinach port"),
    ("easyspin.html", "EasySpin port"),
    ("extras.html", "Extras"),
    ("docs.html", "Documentation"),
    ("try.html", "Try it"),
    (GITHUB, "GitHub"),
)
# The guides of docs/README.md by site section; a guide in the README but in no group is an error.
GROUPS = {
    "spinach": [
        "hilbert-space",
        "liouville-space",
        "solid-state",
        "relaxation",
        "magic-angle-spinning",
        "pulse-sequences",
        "coming-from-spinach",
    ],
    "easyspin": ["easyspyn", "easyspyn-simulators", "coming-from-easyspin"],
    "extras": ["ctps", "ctps-examples", "relaxation-pathways", "beyond-the-port"],
}
# One line per Spinach comparison case, the only prose in the timing table.
SPINACH_CASES = {
    "noesy_strychnine": "22 protons, IK-2 basis (12 316 states), Redfield relaxation, 500 ms NOESY, 512 x 512 points",
    "zulf_pf6_1d": "31P and six 19F at 2.7 µT, complete basis (16 384 states), lab frame, selective pulse, "
    "16 384-point acquisition",
    "li_exchange_equilibrium": "two 7Li sites with quadrupolar and shift tensors in chemical exchange: thermal "
    "state equilibrated under exchange, thermalised Redfield relaxation, a 2 s trajectory analysed by tensor "
    "projections and state classes (31 states)",
    "maz_noesy_2": "methylaziridine, 7 protons + 14N with DFT shielding tensors and the 14N quadrupole: Redfield "
    "theory with scalar relaxation of both kinds, IK-2 basis (19 314 states), 2 s NOESY",
}
SOURCE_PACKAGES = ("spynach", "ctps", "easyspyn")
SOURCE_SUFFIXES = {".py", ".txt"}


def esc(text: object) -> str:
    return html.escape(str(text), quote=True)


def code(text: str) -> str:
    """HTML-escape, then turn `backticks` into <code>."""
    return re.sub(r"`([^`]+)`", r"<code>\1</code>", esc(text))


def span(text: str, name: str, body: str, path: str) -> str:
    start, end = f"<!-- site:{name} -->", f"<!-- /site:{name} -->"
    pattern = re.compile(re.escape(start) + r".*?" + re.escape(end), flags=re.S)
    if not pattern.search(text):
        raise SystemExit(f"{path} has no {start} ... {end} span")
    return pattern.sub(lambda _: f"{start}\n{body}\n{end}", text, count=1)


# ---------------------------------------------------------------- navigation
def render_nav(page: str) -> str:
    current = ' aria-current="page"' if page == "index.html" else ""
    items = [f'  <a class="brand" href="index.html"{current}>Spynach</a>']
    for href, label in NAV:
        current = ' aria-current="page"' if href == page else ""
        items.append(f'  <a href="{href}"{current}>{label}</a>')
    return '<nav class="top" aria-label="Site">\n' + "\n".join(items) + "\n</nav>"


# ---------------------------------------------------------------- head: icons and link previews
def render_head(page: str, text: str) -> str:
    """Icons and the Open Graph / Twitter tags that make a pasted link show a card; title and description come
    from the page's own <title> and description meta, and the image URLs are absolute, as link crawlers need."""
    title = re.search(r"<title>(.*?)</title>", text, flags=re.S).group(1)
    desc = re.search(r'<meta name="description" content="(.*?)">', text, flags=re.S).group(1)
    image = SITE + "assets/social-card.png"
    alt = "Spynach: spin dynamics in Python, with a halftone spiral of coloured dots"
    return "\n".join(
        [
            '<link rel="icon" href="assets/favicon.svg" type="image/svg+xml">',
            '<link rel="icon" href="assets/favicon-32.png" type="image/png" sizes="32x32">',
            '<link rel="apple-touch-icon" href="assets/apple-touch-icon.png">',
            '<meta property="og:type" content="website">',
            '<meta property="og:site_name" content="Spynach">',
            f'<meta property="og:title" content="{title}">',
            f'<meta property="og:description" content="{desc}">',
            f'<meta property="og:url" content="{SITE}{"" if page == "index.html" else page}">',
            f'<meta property="og:image" content="{image}">',
            '<meta property="og:image:width" content="1200">',
            '<meta property="og:image:height" content="630">',
            f'<meta property="og:image:alt" content="{alt}">',
            '<meta name="twitter:card" content="summary_large_image">',
            f'<meta name="twitter:title" content="{title}">',
            f'<meta name="twitter:description" content="{desc}">',
            f'<meta name="twitter:image" content="{image}">',
        ]
    )


# ---------------------------------------------------------------- timing tables
def cell_s(x) -> str:
    """A seconds cell; None means the side was not timed."""
    if x is None:
        return '<td class="num na">not timed</td>'
    return f'<td class="num">{x:.2f}</td>' if x >= 0.1 else f'<td class="num">{x:.3f}</td>'


def cell_ratio(a, b) -> str:
    return '<td class="num na"></td>' if a is None or b is None or a == 0 else f'<td class="num">{b / a:.2f}x</td>'


def cell_rel(x) -> str:
    return '<td class="num na"></td>' if x is None else f'<td class="num">{x:.1e}</td>'


def spinach_summaries() -> list[dict]:
    runs = ROOT / "comparison" / "runs"
    out = []
    for path in sorted(runs.glob("*/summary.json")):
        out.append({**json.loads(path.read_text(encoding="utf-8")), "run": path.parent.name})
    return out


def _total(summary: dict) -> dict:
    return next((r for r in summary["stages"] if r["label"] == "total (wall clock)"), {})


def _agreement(summary: dict) -> float | None:
    rels = [a["rel"] for a in summary.get("arrays", []) if a.get("rel") is not None]
    return max(rels) if rels else None


def render_timing_spinach() -> str:
    """One row per case: Spinach and the fastest Spynach run (``<case>`` or ``<case>-numba``)."""
    by_run = {s["run"]: s for s in spinach_summaries()}
    rows = []
    for case in sorted({s["case"] for s in by_run.values() if not s["run"].endswith(("-numba", "-nonumba"))}):
        candidates = [by_run[r] for r in (case, f"{case}-numba") if r in by_run]
        if not candidates:
            continue
        best = min(candidates, key=lambda s: _total(s).get("spynach") or float("inf"))
        spinach_s = _total(best).get("spinach")
        spynach_s = _total(best).get("spynach")
        rows.append(
            "<tr>"
            f'<td><code>{esc(case)}</code><br><span class="note">{esc(SPINACH_CASES.get(case, ""))}</span></td>'
            + cell_s(spinach_s)
            + cell_s(spynach_s)
            + cell_ratio(spinach_s, spynach_s)
            + cell_rel(_agreement(best))
            + f'<td class="links"><a href="comparison/runs/{esc(best["run"])}/report.html">report</a></td>'
            "</tr>"
        )
    if not rows:
        return '<p class="note">No comparison runs with a summary.json.</p>'
    return (
        '<div class="scroll"><table class="timing">\n<thead><tr><th>Case</th>'
        '<th class="num">Spinach / s</th><th class="num">Spynach / s</th><th class="num">Spynach / Spinach</th>'
        '<th class="num">Agreement</th><th>Report</th></tr></thead>\n'
        "<tbody>\n" + "\n".join(rows) + '\n</tbody></table></div>\n<p class="note">Wall clock of the whole case on '
        "each side, Spynach in its fastest configuration; agreement is the largest relative difference over the "
        "arrays both sides saved (FIDs, spectra, trajectories).</p>"
    )


def easyspin_timings() -> dict | None:
    path = ROOT / "comparison" / "easyspin" / "timings.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def render_timing_easyspin() -> str:
    data = easyspin_timings()
    if not data:
        return '<p class="note">No EasySpin timings yet: run scripts/easyspin-timings.py with the fixtures.</p>'
    rows = []
    for c in data["cases"]:
        times = [t for t in (c.get("easyspyn_numpy_s"), c.get("easyspyn_numba_s")) if t is not None]
        best = min(times) if times else None
        browser = (
            f'<td class="links"><a href="try.html#{esc(c["example"])}">run</a></td>'
            if c.get("example")
            else "<td></td>"
        )
        rows.append(
            "<tr>"
            f'<td><code>{esc(c["function"])}</code><br><span class="note">{esc(c["description"])}</span></td>'
            + cell_s(c.get("easyspin_s"))
            + cell_s(best)
            + cell_ratio(c.get("easyspin_s"), best)
            + cell_rel(c.get("rel_err"))
            + browser
            + "</tr>"
        )
    es_ = data["easyspin"]
    ep = data["easyspyn"]
    numba = f", Numba {ep['numba']}" if ep.get("numba") else ""
    machine = (
        f"EasySpin {esc(es_['version'])} in MATLAB {esc(es_['matlab_version'])}; EasySpyn {esc(ep['version'])} with "
        f"Python {esc(ep['python'])}, NumPy {esc(ep['numpy'])}, SciPy {esc(ep['scipy'])} on "
        f"{esc(ep['blas'])}{esc(numba)}; {esc(ep['platform'])}, {ep['cores']} cores. "
        f"EasySpin times are one call inside the fixture run after a warm-up call (a fit is timed once); "
        f"EasySpyn times are the median of {data['repeats']} calls after a warm-up, the faster of the NumPy and "
        f"the Numba kernels where a kernel is on the path. Generated {esc(data['generated'])}."
    )
    return (
        '<div class="scroll"><table class="timing">\n<thead><tr><th>Case</th><th class="num">EasySpin / s</th>'
        '<th class="num">EasySpyn / s</th><th class="num">EasySpyn / EasySpin</th><th class="num">Agreement</th>'
        "<th>Browser</th></tr></thead>\n"
        "<tbody>\n" + "\n".join(rows) + f'\n</tbody></table></div>\n<p class="note">{machine}</p>'
    )


# ---------------------------------------------------------------- documentation tables
ROW = re.compile(
    r"^\| \[(?P<title>[^\]]+)\]\((?P<stem>[\w-]+)\.md\) \| \[(?P=stem)\.ipynb\]\([^)]*\) · "
    r"\[nbviewer\]\((?P<nb>[^)]+)\) \| (?P<covers>.*) \|$"
)


def guide_rows() -> dict[str, dict]:
    rows = {}
    for line in (ROOT / "docs" / "README.md").read_text(encoding="utf-8").splitlines():
        match = ROW.match(line)
        if match:
            rows[match["stem"]] = match.groupdict()
    if not rows:
        raise SystemExit("no guide rows found in docs/README.md")
    grouped = {stem for stems in GROUPS.values() for stem in stems}
    unfiled = sorted(set(rows) - grouped)
    if unfiled:
        raise SystemExit(f"guides in docs/README.md that are in no group of scripts/build_site.py: {unfiled}")
    missing = sorted(grouped - set(rows))
    if missing:
        raise SystemExit(f"guides in scripts/build_site.py that are not in docs/README.md: {missing}")
    return rows


def render_docs(group: str) -> str:
    rows = guide_rows()
    body = []
    for stem in GROUPS[group]:
        r = rows[stem]
        body.append(
            f"<tr><td>{code(r['title'])}</td><td>{code(r['covers'])}</td>"
            f'<td class="links"><a href="{GITHUB}/blob/main/docs/{stem}.md">guide</a> · '
            f'<a href="{esc(r["nb"])}">notebook</a></td></tr>'
        )
    return (
        '<div class="scroll"><table>\n<thead><tr><th>Guide</th><th>Covers</th><th>Open</th></tr></thead>\n<tbody>\n'
        + "\n".join(body)
        + "\n</tbody></table></div>"
    )


# ---------------------------------------------------------------- the playground's source list
def source_files() -> list[str]:
    out = []
    for package in SOURCE_PACKAGES:
        for path in sorted((ROOT / "src" / package).rglob("*")):
            if path.suffix in SOURCE_SUFFIXES and "__pycache__" not in path.parts:
                out.append(path.relative_to(ROOT).as_posix())
    return out


def rendered_worker(text: str) -> str:
    pattern = re.compile(r"const SOURCE = \[.*?\];", flags=re.S)
    if not pattern.search(text):
        raise SystemExit("worker.js has no const SOURCE = [ ... ]; block")
    body = "const SOURCE = [\n" + "".join(f'  "{f}",\n' for f in source_files()) + "];"
    return pattern.sub(lambda _: body, text, count=1)


# ---------------------------------------------------------------- pages
def rendered_page(page: str, text: str) -> str:
    text = span(text, "nav", render_nav(page), page)
    text = span(text, "head", render_head(page, text), page)
    if page == "spinach.html":
        text = span(text, "timing-spinach", render_timing_spinach(), page)
    if page == "easyspin.html":
        text = span(text, "timing-easyspin", render_timing_easyspin(), page)
    if page == "docs.html":
        for group in GROUPS:
            text = span(text, f"docs-{group}", render_docs(group), page)
    return text


def outputs() -> dict[str, tuple[str | None, str | None]]:
    """``{file: (current text, rendered text)}``; a missing file renders to None."""
    out = {}
    for page in PAGES:
        path = ROOT / page
        current = path.read_text(encoding="utf-8") if path.exists() else None
        out[page] = (current, rendered_page(page, current) if current is not None else None)
    worker = (ROOT / "worker.js").read_text(encoding="utf-8")
    out["worker.js"] = (worker, rendered_worker(worker))
    return out


def stale() -> list[str]:
    return [name for name, (current, new) in outputs().items() if current is None or current != new]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="report stale files instead of writing them")
    args = parser.parse_args(argv)
    status = 0
    for name, (current, new) in outputs().items():
        if current is None:
            print(f"missing: {name}")
            status = 1
            continue
        if current == new:
            continue
        if args.check:
            print(f"stale: {name}")
            status = 1
        else:
            (ROOT / name).write_text(new, encoding="utf-8")
            print(f"wrote: {name}")
    return status


if __name__ == "__main__":
    sys.exit(main())
