"""Write assets/relaxation-chord.svg: the chord diagram of the 11BF4- zero-quantum space for the landing page.

The geometry is spynach.plotting.chord_layout of the analysis in docs/relaxation-pathways.md (11B ranks 1 to 3,
coherence order 0, one ribbon per mechanism class); only the colours differ, chosen for the dark slide of the hero
gallery (assets/hero-gallery.js), which inlines the file and animates it. Every ribbon carries data-source,
data-target and data-rate, every operator arc data-index, so the page can follow the network.

    PYTHONPATH=src python scripts/make_hero_chord.py
"""

from __future__ import annotations

import re
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np

import spynach
from spynach.conventions import eeqq2nqi
from spynach.plotting import chord_layout

OUT = Path(__file__).resolve().parent.parent / "assets" / "relaxation-chord.svg"
# Brighter versions of the colours of plot_pathways, readable on the dark slide
COLOURS = {"DD": "#ff9a78", "DD/CSA": "#78aaff", "CSA/Q": "#82e6cd", "Q": "#ffc95a"}
SPARE = ["#c3a6ff", "#8fe08f", "#ff8f8f", "#e0e0e0"]
ARC = "#dcdad2"
INK = "#0b0b0b"


def tetrafluoroborate(magnet=9.4, r_bf=1.40, dsigma_f=100.0, dsigma_b=2.0, cq_b=0.1e6, tau_c=10e-12):
    """11BF4-, as in docs/relaxation-pathways.md: boron at the origin, the first B-F bond along z."""
    corners = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]]) / np.sqrt(3)
    z = corners[0]
    x = np.cross(z, [0, 0, 1.0])
    x /= np.linalg.norm(x)
    bonds = corners @ np.array([x, np.cross(z, x), z]).T

    def axial(n, anisotropy):
        return (anisotropy * (np.outer(n, n) - np.eye(3) / 3)).tolist()

    quadrupole = eeqq2nqi(cq_b, 0.0, 1.5, [0, 0, 0]).tolist()
    return spynach.create(
        {"isotopes": ["11B", "19F", "19F", "19F", "19F"], "magnet": magnet},
        {
            "zeeman": {"matrix": [axial([0, 0, 1], dsigma_b)] + [axial(n, dsigma_f) for n in bonds]},
            "coupling": {"matrix": [[quadrupole] + [None] * 4] + [[None] * 5 for _ in range(4)]},
            "coordinates": [[0, 0, 0]] + (r_bf * bonds).tolist(),
            "relaxation": ["redfield"],
            "rlx_keep": "secular",
            "equilibrium": "zero",
            "tau_c": [[tau_c]],
        },
    )


def plain(hover: str) -> str:
    """The hover text of chord_layout (HTML) as plain text for a tooltip."""
    text = re.sub(r"<br\s*/?>", "\n", hover)
    return re.sub(r"<[^>]+>", "", text)


def label(html: str) -> str:
    """SVG tspans for an operator label with <sub> parts (B<sub>z</sub>F<sub>+</sub>), as svgLabel in try.html."""
    parts, lowered = [], False
    for part in filter(None, re.split(r"(<sub>.*?</sub>)", html)):
        sub = part.startswith("<sub>")
        text = escape(part[5:-6] if sub else part)
        attrs = ""
        if sub != lowered:
            attrs += ' dy="0.3em"' if sub else ' dy="-0.3em"'
        if sub:
            attrs += ' font-size="75%"'
        lowered = sub
        parts.append(f"<tspan{attrs}>{text}</tspan>")
    return "".join(parts)


def path(points) -> str:
    return "M" + "L".join(f"{x:g},{-y:g}" for x, y in points) + "Z"


def main() -> None:
    zq = spynach.relaxation_pathways(tetrafluoroborate(), ranks={"B": (1, 2, 3)}, coherence=0)
    layout = chord_layout(zq, split=True)
    colour = {}
    spare = iter(SPARE)
    for name in layout["names"]:
        colour[name] = COLOURS.get(name) or next(spare)
    size = max(0.035, min(0.075, layout["span"] * 1.07 / 5.5))
    out = [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="-1.3 -1.3 2.6 2.6" role="img"'
        ' aria-label="Cross-relaxation pathways of 11BF4-, zero-quantum space: a chord diagram">',
        "<title>Cross-relaxation pathways of 11BF4-, zero-quantum space</title>",
        '<g class="ribbons" fill-opacity="0.8">',
    ]
    for ribbon in layout["ribbons"]:
        name = layout["names"][ribbon["colour"]]
        out.append(
            f'<path class="ribbon" d="{path(ribbon["outline"])}" fill="{colour[name]}"'
            f' data-class="{escape(name)}" data-source="{ribbon["source"]}" data-target="{ribbon["target"]}"'
            f' data-rate="{ribbon["rate"]:.6g}"><title>{escape(plain(ribbon["hover"]))}</title></path>'
        )
    out.append("</g>")
    out.append(
        f'<g class="arcs" font-family="system-ui, -apple-system, Segoe UI, Roboto, sans-serif" font-size="{size:g}">'
    )
    for k, arc in enumerate(layout["arcs"]):
        x, y, angle = arc["text"]
        out.append(
            f'<g class="arc" data-index="{k}"><path d="{path(arc["outline"])}" fill="{ARC}">'
            f"<title>{escape(plain(arc['hover']))}</title></path>"
            f'<text transform="translate({x:g} {-y:g}) rotate({angle:g})" text-anchor="middle"'
            f' dominant-baseline="central" fill="{INK}" pointer-events="none">{label(arc["html"])}</text></g>'
        )
    out.append("</g>")
    out.append("</svg>")
    OUT.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"{OUT.name}: {len(layout['arcs'])} operators, {len(layout['ribbons'])} ribbons, classes {layout['names']}")


if __name__ == "__main__":
    main()
