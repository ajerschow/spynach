"""Builds the HTML report of a comparison run from its spinach/ and spynach/ directories.

    python comparison/report.py <run> [<run> ...]        # <run> is a name under comparison/runs or a path

Writes comparison/runs/<run>/report.html: provenance of both sides, stage timings
side by side, normwise differences of every array both sides saved, the figures,
the profiler tops, and links to the console logs. A missing side is reported as
such, so the report can be built after either half of the protocol. Beside it,
summary.json holds the same timings, differences and provenance as data, from
which scripts/build_site.py renders the timing table of the project site. The
report inlines the site's stylesheet (assets/site.css) and carries its navigation.
"""

from __future__ import annotations

import base64
import html
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "comparison" / "runs"
SIDES = ("spinach", "spynach")
TOP_SHOWN = 25


def load_side(run_dir: Path, side: str) -> dict | None:
    side_dir = run_dir / side
    if not side_dir.is_dir():
        return None
    out = {"dir": side_dir}
    for name in ("meta", "timings", "profile"):
        path = side_dir / f"{name}.json"
        out[name] = json.loads(path.read_text()) if path.exists() else None
    out["data"] = load_data(side_dir)
    out["figures"] = {p.stem[len("figure_") :]: p for p in sorted(side_dir.glob("figure_*.png"))}
    out["figure_html"] = {p.stem[len("figure_") :]: p for p in sorted(side_dir.glob("figure_*.html"))}
    out["console"] = side_dir / "console.log" if (side_dir / "console.log").exists() else None
    return out


def load_data(side_dir: Path) -> dict[str, np.ndarray] | None:
    if (side_dir / "data.mat").exists():
        from scipy.io import loadmat

        raw = loadmat(str(side_dir / "data.mat"))
        return {k: np.asarray(v) for k, v in raw.items() if not k.startswith("__")}
    if (side_dir / "data.npz").exists():
        with np.load(side_dir / "data.npz") as raw:
            return {k: raw[k] for k in raw.files}
    return None


def compare_arrays(a: dict | None, b: dict | None) -> list[dict]:
    """Rows for every variable: shapes, normwise relative error ||spynach - spinach|| / ||spinach||, max |difference|."""
    a, b = a or {}, b or {}
    rows = []
    for name in sorted(set(a) | set(b)):
        row = {"name": name, "spinach_shape": None, "spynach_shape": None, "rel": None, "max_abs": None, "note": ""}
        if name in a:
            row["spinach_shape"] = a[name].shape
        if name in b:
            row["spynach_shape"] = b[name].shape
        if name in a and name in b:
            x, y = np.asarray(b[name]), np.asarray(a[name])
            if x.shape != y.shape and np.squeeze(x).shape == np.squeeze(y).shape:
                x, y = np.squeeze(x), np.squeeze(y)  # MATLAB stores a vector as 1 x n
                row["note"] = "compared after squeezing singleton dimensions"
            if x.shape != y.shape:
                row["note"] = "shape mismatch"
            else:
                diff = np.linalg.norm((x - y).ravel())
                ref = np.linalg.norm(y.ravel())
                row["rel"] = float(diff / ref) if ref > 0 else (0.0 if diff == 0 else float("inf"))
                row["max_abs"] = float(np.max(np.abs(x - y))) if x.size else 0.0
        else:
            row["note"] = "only in " + ("spinach" if name in a else "spynach")
        rows.append(row)
    return rows


def timing_rows(sides: dict) -> list[dict]:
    labels: list[str] = []
    per_side = {}
    for side in SIDES:
        t = (sides.get(side) or {}).get("timings")
        per_side[side] = {r["label"]: r["seconds"] for r in (t or {}).get("stages", [])}
        for label in per_side[side]:
            if label not in labels:
                labels.append(label)
    rows = [{"label": l, "spinach": per_side["spinach"].get(l), "spynach": per_side["spynach"].get(l)} for l in labels]
    totals = {side: ((sides.get(side) or {}).get("timings") or {}).get("total_s") for side in SIDES}
    rows.append({"label": "total (wall clock)", "spinach": totals["spinach"], "spynach": totals["spynach"]})
    return rows


# ---------------------------------------------------------------- HTML
SITE_CSS = ROOT / "assets" / "site.css"
# The site's stylesheet is inlined so that a copied report still renders; these rules are the report's own.
CSS = """
:root { --warn: #b3261e; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { --warn: #ff8a80; } }
.wrap.report { max-width: 76rem; }
main { padding-top: 1rem; }
h1 { font-size: 1.7rem; margin: 0.6rem 0 0.2rem; }
h2 { margin: 2.2rem 0 0.6rem; border-bottom: 1px solid var(--line); padding-bottom: 0.3rem; }
h3 { font-size: 0.85rem; margin: 0; color: var(--muted); text-transform: uppercase; letter-spacing: 0.04em; }
p { max-width: none; }
p.sub { color: var(--muted); margin: 0 0 1.2rem; }
table { margin: 0.4rem 0 0.9rem; }
th, td { padding: 0.35rem 0.6rem 0.35rem 0; }
td.mono, span.mono { font-family: var(--mono); font-size: 0.85em; word-break: break-all; }
.cols { display: grid; grid-template-columns: 1fr 1fr; gap: 1.2rem; margin-bottom: 1.2rem; }
@media (max-width: 800px) { .cols { grid-template-columns: 1fr; } }
.card img { width: 100%; height: auto; border-radius: 4px; background: #fff; }
.missing { color: var(--warn); } .ok { color: var(--accent-ink); }
.small { font-size: 0.85rem; color: var(--muted); }
"""


def esc(x) -> str:
    return html.escape(str(x))


def fmt_s(x) -> str:
    return "" if x is None else f"{x:.2f}"


def fmt_ratio(a, b) -> str:
    return "" if a is None or b is None or a == 0 else f"{b / a:.2f}x"


def fmt_rel(x) -> str:
    return "" if x is None else f"{x:.3e}"


def img_tag(path: Path | None) -> str:
    if path is None:
        return '<p class="missing">no figure</p>'
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f'<img src="data:image/png;base64,{data}" alt="{esc(path.name)}">'


def meta_table(sides: dict) -> str:
    keys = {
        "spinach": [
            ("created_utc", "run date (UTC)"),
            ("spinach_describe", "Spinach version"),
            ("spinach_commit", "commit"),
            ("spinach_dirty", "local modifications"),
            ("matlab_version", "MATLAB"),
            ("hostname", "host"),
            ("platform", "platform"),
            ("cores", "cores"),
            ("blas", "BLAS"),
            ("comp_threads", "computational threads"),
            ("pool_workers", "parallel pool workers"),
            ("gpu", "GPU requested"),
            ("gpu_device", "GPU device"),
            ("options", "case options"),
        ],
        "spynach": [
            ("created_utc", "run date (UTC)"),
            ("spynach_describe", "spynach version"),
            ("spynach_commit", "commit"),
            ("spynach_dirty", "local modifications"),
            ("python_version", "Python"),
            ("numpy_version", "NumPy"),
            ("scipy_version", "SciPy"),
            ("numba_version", "Numba (compiled kernels)"),
            ("hostname", "host"),
            ("platform", "platform"),
            ("cores", "cores"),
            ("blas", "BLAS"),
            ("thread_env", "thread settings"),
            ("gpu", "GPU requested"),
            ("gpu_device", "GPU device"),
            ("options", "case options"),
            ("error", "not run"),
        ],
    }
    parts = []
    for side in SIDES:
        info = sides.get(side)
        rows = ""
        if info is None:
            rows = f'<tr><td colspan="2" class="missing">no {side}/ directory in this run</td></tr>'
        elif not info["meta"]:
            rows = '<tr><td colspan="2" class="missing">meta.json missing</td></tr>'
        else:
            for key, label in keys[side]:
                value = info["meta"].get(key)
                if key == "thread_env" and value == {}:
                    value = "library defaults"
                value = json.dumps(value) if isinstance(value, (dict, list)) else value
                rows += f'<tr><th>{esc(label)}</th><td class="mono">{esc("" if value is None else value)}</td></tr>'
        parts.append(f'<div class="card"><h3>{side}</h3><table>{rows}</table></div>')
    return f'<div class="cols">{"".join(parts)}</div>'


def timings_table(rows: list[dict]) -> str:
    body = "".join(
        f'<tr><td>{esc(r["label"])}</td><td class="num">{fmt_s(r["spinach"])}</td><td class="num">{fmt_s(r["spynach"])}</td>'
        f'<td class="num">{fmt_ratio(r["spinach"], r["spynach"])}</td></tr>'
        for r in rows
    )
    return (
        '<table><tr><th>stage</th><th class="num">Spinach / s</th><th class="num">spynach / s</th><th class="num">spynach / Spinach</th></tr>'
        f"{body}</table>"
    )


def data_table(rows: list[dict]) -> str:
    if not rows:
        return '<p class="missing">no data arrays on either side</p>'
    body = ""
    for r in rows:
        body += (
            f'<tr><td class="mono">{esc(r["name"])}</td><td class="mono">{esc(r["spinach_shape"] or "")}</td>'
            f'<td class="mono">{esc(r["spynach_shape"] or "")}</td><td class="num">{fmt_rel(r["rel"])}</td>'
            f'<td class="num">{fmt_rel(r["max_abs"])}</td><td class="missing">{esc(r["note"])}</td></tr>'
        )
    return (
        '<table><tr><th>variable</th><th>Spinach shape</th><th>spynach shape</th><th class="num">relative error</th>'
        f'<th class="num">max |difference|</th><th></th></tr>{body}</table>'
        '<p class="small">relative error = ||spynach - Spinach||<sub>F</sub> / ||Spinach||<sub>F</sub>, the metric of the oracle tests.</p>'
    )


def figures_section(sides: dict) -> str:
    names: list[str] = []
    for side in SIDES:
        for name in (sides.get(side) or {}).get("figures") or {}:
            if name not in names:
                names.append(name)
    if not names:
        return '<p class="missing">no figures</p>'
    out = ""
    for name in names:
        cols = ""
        for side in SIDES:
            info = sides.get(side) or {}
            png = (info.get("figures") or {}).get(name)
            link = (info.get("figure_html") or {}).get(name)
            extra = (
                f'<p class="small"><a href="{esc(link.relative_to(link.parents[1]))}">interactive figure</a></p>'
                if link
                else ""
            )
            cols += f'<div class="card"><h3>{side}: {esc(name)}</h3>{img_tag(png)}{extra}</div>'
        out += f'<div class="cols">{cols}</div>'
    return out


def profile_tables(sides: dict) -> str:
    cols = ""
    for side in SIDES:
        prof = (sides.get(side) or {}).get("profile")
        if not prof:
            cols += f'<div class="card"><h3>{side}</h3><p class="missing">no profile</p></div>'
            continue
        block = f'<div class="card"><h3>{side}</h3>'
        for key, title in (("by_self", "by self time"), ("by_total", "by total time (with callees)")):
            rows = "".join(
                f'<tr><td class="mono">{esc(r["function"])}</td><td class="num">{esc(r["calls"])}</td>'
                f'<td class="num">{fmt_s(r["self_s"])}</td><td class="num">{fmt_s(r["total_s"])}</td></tr>'
                for r in prof.get(key, [])[:TOP_SHOWN]
            )
            block += (
                f'<p class="small">{title}</p><table><tr><th>function</th><th class="num">calls</th>'
                f'<th class="num">self / s</th><th class="num">total / s</th></tr>{rows}</table>'
            )
        cols += block + "</div>"
    return f'<div class="cols">{cols}</div>'


def logs_section(sides: dict, run_dir: Path) -> str:
    items = ""
    for side in SIDES:
        info = sides.get(side)
        log = info["console"] if info else None
        items += (
            f"<li>{side}: "
            + (
                f'<a href="{esc(log.relative_to(run_dir))}">console.log</a>'
                if log
                else '<span class="missing">none</span>'
            )
            + "</li>"
        )
    return f"<ul>{items}</ul>"


def build_report(run_dir: Path) -> Path:
    run_dir = Path(run_dir)
    sides = {side: load_side(run_dir, side) for side in SIDES}
    case = next((s["meta"]["case"] for s in sides.values() if s and s["meta"]), run_dir.name)
    present = [side for side in SIDES if sides[side]]
    gpu = (
        any(s and s["meta"] and s["meta"].get("gpu") for s in sides.values())
        or run_dir.name.endswith("-gpu")
        or "-gpu-" in run_dir.name
    )
    accel = "GPU" if gpu else "CPU"
    status = "both sides present" if len(present) == 2 else f"only {present[0]} present" if present else "no results"
    failed = [side for side in present if sides[side]["meta"] and sides[side]["meta"].get("error")]
    if failed:
        status += "; " + ", ".join(f"{side} did not run ({sides[side]['meta']['error']})" for side in failed)
    timings = timing_rows(sides)
    arrays = compare_arrays((sides["spinach"] or {}).get("data"), (sides["spynach"] or {}).get("data"))
    nav = (
        '<nav class="top" aria-label="Site"><a class="brand" href="../../../index.html">Spynach</a>'
        '<a href="../../../spinach.html#timing">Parity and timing</a>'
        '<a href="https://github.com/ajerschow/spynach/blob/main/comparison/README.md">Protocol</a></nav>'
    )
    body = f"""<div class="wrap report">{nav}<main>
<h1>Spinach vs spynach: {esc(case)} ({accel})</h1>
<p class="sub">run <span class="mono">{esc(run_dir.name)}</span>, {esc(status)}. Built by comparison/report.py; protocol in comparison/README.md.</p>
<h2>Provenance</h2>{meta_table(sides)}
<h2>Timings</h2>{timings_table(timings)}
<h2>Data</h2>{data_table(arrays)}
<h2>Figures</h2>{figures_section(sides)}
<h2>Profiles</h2>{profile_tables(sides)}
<h2>Console logs</h2>{logs_section(sides, run_dir)}
</main></div>"""
    site_css = SITE_CSS.read_text(encoding="utf-8") if SITE_CSS.exists() else ""
    page = (
        f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>Spinach vs spynach: {esc(case)} ({accel})</title><style>{site_css}{CSS}</style></head><body>{body}</body></html>"
    )
    write_summary(run_dir, case, gpu, sides, timings, arrays)
    out = run_dir / "report.html"
    out.write_text(page, encoding="utf-8")
    return out


PROVENANCE_KEYS = {
    "spinach": (
        "created_utc",
        "spinach_describe",
        "matlab_version",
        "hostname",
        "platform",
        "cores",
        "blas",
        "pool_workers",
    ),
    "spynach": (
        "created_utc",
        "spynach_describe",
        "python_version",
        "numpy_version",
        "scipy_version",
        "numba_version",
        "hostname",
        "platform",
        "cores",
        "blas",
    ),
}


def write_summary(run_dir: Path, case: str, gpu: bool, sides: dict, timings: list[dict], arrays: list[dict]) -> Path:
    """comparison/runs/<run>/summary.json: the numbers of the report, for the timing table of the project site."""
    provenance = {}
    for side in SIDES:
        meta = ((sides.get(side) or {}).get("meta")) or {}
        provenance[side] = {k: meta.get(k) for k in PROVENANCE_KEYS[side]}
    summary = {
        "case": case,
        "run": run_dir.name,
        "gpu": gpu,
        "built_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "stages": timings,
        "arrays": [{k: r[k] for k in ("name", "rel", "max_abs", "note")} for r in arrays],
        "provenance": provenance,
    }
    out = run_dir / "summary.json"
    out.write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    return out


def main(argv=None) -> None:
    args = argv if argv is not None else sys.argv[1:]
    if not args:
        raise SystemExit(__doc__)
    for arg in args:
        run_dir = Path(arg) if Path(arg).is_dir() else RUNS / arg
        if not run_dir.is_dir():
            raise SystemExit(f"no run directory {run_dir}")
        print(build_report(run_dir))


if __name__ == "__main__":
    main()
