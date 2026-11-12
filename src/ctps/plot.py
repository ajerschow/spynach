"""Interactive figures: the pathway explorer, the cogwheel selection diagram, phase vectors and Pareto fronts.

Plotly figures that render interactively in JupyterLab, VS Code and nbviewer (no widgets
needed); the colours and chrome are those of ``spynach.plotting``, copied here so that ``ctps``
stays importable on its own. ``plot_pathways`` draws the pulse sequence above the coherence
orders as the 1998 Fig. 4 does, with the line width of every pathway scaled by its weight; a
slider steps through the ranked pathways, highlighting one and showing its factored weights.

Author: Alexej Jerschow.
"""

from __future__ import annotations

import math
from collections.abc import Sequence as SequenceABC

import numpy as np

from ctps.model import PathwayResult, PhaseCycle, Sequence
from ctps.optimize import CycleSolution

SERIES = {
    "light": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"],
    "dark": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"],
}
CHROME = {
    "light": {
        "surface": "#fcfcfb",
        "ink": "#0b0b0b",
        "ink2": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
    },
    "dark": {
        "surface": "#1a1a19",
        "ink": "#ffffff",
        "ink2": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
    },
}
FONT = 'system-ui, -apple-system, "Segoe UI", Roboto, sans-serif'
GRADIENT_COLOURS = {"x": "#e34948", "y": "#1baf7a", "z": "#2a78d6"}


def _plotly():
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as error:
        raise ImportError('ctps.plot needs Plotly: pip install -e ".[plot]"') from error
    return go, make_subplots


def _time_axis(sequence: Sequence, to_scale: bool = False) -> np.ndarray:
    """Period boundaries: one unit per period (schematic, as pathway diagrams are drawn), or real durations."""
    durations = sequence.durations()
    if not to_scale or durations.sum() <= 0:
        return np.arange(sequence.n_periods + 1, dtype=float)
    # periods of zero duration still get a small slot so that pulses do not overlap
    slot = max(durations[durations > 0].min() * 0.3, 1e-9)
    return np.concatenate([[0.0], np.cumsum(np.where(durations > 0, durations, slot))])


def _axis_style(c: dict) -> dict:
    return {
        "showgrid": True,
        "gridcolor": c["grid"],
        "zeroline": False,
        "showline": True,
        "linecolor": c["axis"],
        "ticks": "outside",
        "tickcolor": c["axis"],
        "tickfont": {"color": c["ink2"], "size": 12},
        "title": {"font": {"color": c["ink2"], "size": 13}},
    }


def plot_pathways(
    result: PathwayResult,
    *,
    top: int = 12,
    threshold: float = 0.0,
    width: str = "log",
    to_scale: bool = False,
    theme: str = "light",
    height: int | None = None,
    title: str | None = None,
):
    """The pathway explorer: pulse sequence lane, coherence orders per nucleus and in total, ranked pathways.

    ``top`` pathways are drawn (the strongest), those below ``threshold`` times the strongest
    amplitude are left out; line widths follow ``|total|`` on a ``"log"`` (three decades) or
    ``"linear"`` scale; wanted pathways are solid, the others dashed. Periods are drawn with
    equal width unless ``to_scale`` asks for the real delays. The slider under the figure steps
    through the pathways, dimming the rest and listing the selected one's factors; hovering a
    segment shows them too; the legend toggles single pathways.
    """
    go, make_subplots = _plotly()
    if theme not in CHROME:
        raise ValueError("theme must be 'light' or 'dark'.")
    c = CHROME[theme]
    sequence = result.sequence
    shown = result.top(top)
    if threshold > 0 and len(shown):
        keep = shown.amplitude >= threshold * result.amplitude.max()
        shown = shown.select(np.flatnonzero(keep))
    k = len(shown)
    edges = _time_axis(sequence, to_scale)
    mids = (edges[:-1] + edges[1:]) / 2
    nuclei = [n.name for n in sequence.nuclei]
    panels = (["total"] if len(nuclei) > 1 else []) + nuclei
    rows = 1 + len(panels)
    heights = [0.3] + [0.7 / len(panels)] * len(panels)
    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=heights,
        subplot_titles=[""] + [f"coherence order, {p}" if p != "total" else "total coherence order" for p in panels],
    )
    for note in fig.layout.annotations:  # subplot titles: quieter than Plotly's default
        note.font.size = 12
        note.font.color = c["ink2"]
    # Sequence lane: pulses as bars on the baseline, labels in one row above them; gradients as boxes
    # standing on (or hanging from) the baseline, labelled below the lane
    baseline, pulse_height = 0.0, 1.0
    span = edges[-1] - edges[0]
    for j, pulse in enumerate(sequence.pulses):
        x = edges[j + 1]
        wide = pulse.kind in ("perfect_180", "imperfect_180", "cccp_1998")
        w = span * (0.010 if wide else 0.004)
        fig.add_shape(
            type="rect",
            x0=x - w,
            x1=x + w,
            y0=baseline,
            y1=pulse_height,
            fillcolor=c["ink"],
            line={"width": 0},
            row=1,
            col=1,
        )
        fig.add_annotation(
            x=x,
            y=pulse_height + 0.22,
            text=pulse.label or f"{pulse.nucleus} {j + 1}",
            showarrow=False,
            font={"size": 10, "color": c["ink2"]},
            xanchor="center",
            yanchor="bottom",
            row=1,
            col=1,
        )
    fig.add_shape(
        type="line",
        x0=edges[0],
        x1=edges[-1],
        y0=baseline,
        y1=baseline,
        line={"color": c["axis"], "width": 1},
        row=1,
        col=1,
    )
    areas = sequence.gradient_areas()
    scale = np.abs(areas).max() or 1.0
    for i, period in enumerate(sequence.periods):
        active = [(a, axis) for a, axis in enumerate("xyz") if period.gradient[a]]
        for n, (a, axis) in enumerate(active):
            h = 0.75 * period.gradient[a] / scale
            width_px = edges[i + 1] - edges[i]
            x0 = edges[i] + (0.12 + 0.3 * n) * width_px
            x1 = x0 + 0.26 * width_px
            fig.add_shape(
                type="rect",
                x0=x0,
                x1=x1,
                y0=baseline,
                y1=baseline + h,
                fillcolor=GRADIENT_COLOURS[axis],
                opacity=0.65,
                line={"width": 0},
                row=1,
                col=1,
            )
            fig.add_annotation(
                x=(x0 + x1) / 2,
                y=min(baseline, baseline + h) - 0.1,
                text=f"G{axis}",
                showarrow=False,
                font={"size": 10, "color": GRADIENT_COLOURS[axis]},
                xanchor="center",
                yanchor="top",
                row=1,
                col=1,
            )
    fig.update_yaxes(visible=False, range=[-1.35, 1.8], row=1, col=1)

    # Pathways
    amplitude = shown.amplitude
    reference = amplitude.max() if k else 1.0
    if width == "log":
        widths = 1 + 5 * np.clip(1 + np.log10(np.maximum(amplitude / reference, 1e-12)) / 3, 0, 1)
    else:
        widths = 1 + 5 * amplitude / reference
    palette = SERIES[theme]
    # Each period is a horizontal segment; the transition at a pulse is a short slanted segment across it
    slant = np.array([0.15 * min(edges[j + 1] - edges[j], edges[j + 2] - edges[j + 1]) for j in range(len(edges) - 2)])
    x_steps = np.empty(2 * (len(edges) - 1))
    x_steps[0], x_steps[-1] = edges[0], edges[-1]
    x_steps[1:-1:2] = edges[1:-1] - slant  # end of each period, short of the pulse
    x_steps[2:-1:2] = edges[1:-1] + slant  # start of the next period, past the pulse
    for r, panel in enumerate(panels, start=2):
        orders = shown.orders if panel == "total" else shown.per_nucleus[panel]
        for i in range(k):
            y = np.repeat(orders[i], 2)
            f = shown[i].factors
            hover = (
                f"pathway {i + 1}: {','.join(str(p) for p in shown.orders[i])}<br>"
                f"total {amplitude[i]:.4g}, ψ {shown.phase[i]:.0f}°<br>"
                f"pc {f['pc']:.4g} · grad {f['grad']:.4g} · diff {f['diff']:.4g} · pulse {f['pulse']:.4g}"
            )
            fig.add_scatter(
                x=x_steps,
                y=y,
                mode="lines",
                line={
                    "color": palette[i % len(palette)],
                    "width": float(widths[i]),
                    "dash": "solid" if shown.wanted[i] else "dot",
                },
                opacity=1.0 if shown.wanted[i] else 0.7,
                name=f"{i + 1}: {','.join(str(p) for p in shown.orders[i])} ({amplitude[i]:.3g})",
                legendgroup=str(i),
                showlegend=r == 2,
                hovertemplate=hover + "<extra></extra>",
                row=r,
                col=1,
            )
        p_max = int(np.abs(orders).max()) if k else 1
        fig.update_yaxes(
            range=[-p_max - 0.5, p_max + 0.5],
            dtick=1,
            **_axis_style(c),
            row=r,
            col=1,
        )
    # Slider: step through the pathways
    n_traces = k * len(panels)
    steps = []
    all_opacity = [1.0 if shown.wanted[i % k] else 0.7 for i in range(n_traces)]
    steps.append(
        {
            "label": "all",
            "method": "update",
            "args": [{"opacity": all_opacity}, {"annotations": fig.layout.annotations}],
        }
    )
    base_annotations = list(fig.layout.annotations)
    for i in range(k):
        opacity = [1.0 if (t % k) == i else 0.12 for t in range(n_traces)]
        f = shown[i].factors
        text = (
            f"<b>pathway {i + 1}</b> {','.join(str(p) for p in shown.orders[i])}"
            f"{' (wanted)' if shown.wanted[i] else ''}<br>total {amplitude[i]:.4g}, ψ {shown.phase[i]:.0f}°<br>"
            f"phase cycle {f['pc']:.4g}<br>gradient {f['grad']:.4g}<br>"
            f"diffusion {f['diff']:.4g}<br>pulses {f['pulse']:.4g}"
        )
        panel = {
            "xref": "paper",
            "yref": "paper",
            "x": 1.0,
            "y": 0.0,
            "xanchor": "right",
            "yanchor": "bottom",
            "align": "left",
            "showarrow": False,
            "bgcolor": c["surface"],
            "bordercolor": c["axis"],
            "borderwidth": 1,
            "font": {"size": 11, "color": c["ink"]},
            "text": text,
        }
        steps.append(
            {
                "label": str(i + 1),
                "method": "update",
                "args": [{"opacity": opacity}, {"annotations": [*base_annotations, panel]}],
            }
        )
    fig.update_layout(
        sliders=[
            {
                "active": 0,
                "currentvalue": {"prefix": "pathway: ", "font": {"color": c["ink2"]}},
                "pad": {"t": 30},
                "steps": steps,
                "font": {"color": c["ink2"]},
            }
        ]
        if k
        else [],
        height=height or 280 + 190 * len(panels),
        font={"family": FONT, "color": c["ink"], "size": 13},
        paper_bgcolor=c["surface"],
        plot_bgcolor=c["surface"],
        margin={"l": 64, "r": 24, "t": 56 if title else 32, "b": 48},
        title={"text": title, "x": 0, "xanchor": "left", "font": {"size": 15, "color": c["ink"]}} if title else None,
        legend={
            "orientation": "v",
            "x": 1.02,
            "y": 1,
            "font": {"size": 10, "color": c["ink2"]},
            "bgcolor": "rgba(0,0,0,0)",
        },
        hovermode="closest",
        modebar={
            "bgcolor": "rgba(0,0,0,0)",
            "color": c["muted"],
            "activecolor": c["ink2"],
            "remove": ["lasso2d", "select2d"],
        },
    )
    fig.update_xaxes(showgrid=False, showticklabels=False, row=1, col=1)
    fig.update_xaxes(
        title_text="time / s" if to_scale and sequence.durations().sum() > 0 else "period",
        row=rows,
        col=1,
        **_axis_style(c),
    )
    if not to_scale:
        fig.update_xaxes(tickvals=mids, ticktext=[str(i) for i in range(len(mids))], row=rows, col=1)
    return fig


def plot_selection_diagram(
    solution: CycleSolution | PhaseCycle,
    sequence: Sequence,
    wanted: SequenceABC[SequenceABC[int]] = (),
    *,
    top: int | None = 400,
    theme: str = "light",
    height: int = 420,
):
    """Levitt's cogwheel selection diagram: the running phase index ``Σ ν_j Δp_j`` of every pathway, pulse by pulse.

    Pathways end on the right against a barrier whose holes sit at ``-ν_rec + kN``; those that
    meet a hole survive. Wanted pathways are drawn bold.
    """
    go, _ = _plotly()
    from ctps.optimize import pathway_jumps

    if isinstance(solution, PhaseCycle):
        if solution.winding is None:
            raise ValueError("the selection diagram needs a cogwheel cycle.")
        N, *nu, nu_rec = solution.winding
    else:
        N, nu, nu_rec = solution.N, list(solution.nu), solution.nu_rec
    c = CHROME[theme]
    orders, dp = pathway_jumps(sequence)
    running = np.cumsum(dp * np.array(nu)[None, :], axis=1)
    running = np.hstack([np.zeros((len(dp), 1), dtype=running.dtype), running])
    final = (running[:, -1] + nu_rec) % N == 0
    targets = {tuple(int(v) for v in w) for w in wanted}
    is_wanted = np.array([tuple(int(v) for v in row) in targets for row in orders])
    fig = go.Figure()
    x = np.arange(dp.shape[1] + 1)
    shown = np.arange(len(dp)) if top is None else np.argsort(~is_wanted, kind="stable")[:top]
    for i in shown:
        fig.add_scatter(
            x=x,
            y=running[i],
            mode="lines",
            line={
                "color": SERIES[theme][0] if is_wanted[i] else (SERIES[theme][1] if final[i] else c["muted"]),
                "width": 3 if is_wanted[i] else 1,
            },
            opacity=1.0 if is_wanted[i] or final[i] else 0.25,
            hovertemplate=f"{','.join(str(p) for p in orders[i])}: phase index {running[i, -1]}<extra></extra>",
            showlegend=False,
        )
    lo, hi = running[:, -1].min() - N, running[:, -1].max() + N
    for kk in range(int(math.floor((lo + nu_rec) / N)), int(math.ceil((hi + nu_rec) / N)) + 1):
        y = kk * N - nu_rec
        fig.add_shape(
            type="line", x0=x[-1] + 0.05, x1=x[-1] + 0.35, y0=y, y1=y, line={"color": SERIES[theme][2], "width": 3}
        )
    fig.update_layout(
        height=height,
        font={"family": FONT, "color": c["ink"], "size": 13},
        paper_bgcolor=c["surface"],
        plot_bgcolor=c["surface"],
        margin={"l": 64, "r": 24, "t": 48, "b": 56},
        title={
            "text": f"COG{N}({','.join(str(v) for v in nu)};{nu_rec}): {int(final.sum())} of {len(dp)} pathways pass",
            "x": 0,
        },
        xaxis={**_axis_style(c), "title": {"text": "pulse"}, "dtick": 1, "range": [-0.2, x[-1] + 0.5]},
        yaxis={**_axis_style(c), "title": {"text": "Σ ν Δp (holes every N)"}},
        showlegend=False,
    )
    return fig


def plot_phase_vectors(result: PathwayResult, index: int, *, theme: str = "light", height: int = 380):
    """The ``N`` signal phasors ``exp(-iψ_m)`` of one pathway and their sum: the phase-cycle factor seen as vectors."""
    go, _ = _plotly()
    from ctps.engine.paths import delta_p

    c = CHROME[theme]
    if not 0 <= index < len(result):
        raise IndexError(f"pathway index {index} out of range: the result holds {len(result)} pathways.")
    dp = delta_p(result.sequence, {name: o[index : index + 1] for name, o in result.per_nucleus.items()})[0]
    psi = dp @ result.cycle.phase_matrix() + result.cycle.receiver_phases()
    vectors = np.exp(-1j * psi)
    total = vectors.sum() / len(vectors)
    fig = go.Figure()
    for m, v in enumerate(vectors):
        fig.add_scatter(
            x=[0, v.real],
            y=[0, v.imag],
            mode="lines+markers",
            line={"color": SERIES[theme][m % len(SERIES[theme])], "width": 1.5},
            marker={"size": [0, 7]},
            name=f"step {m}: ψ = {np.degrees(psi[m]) % 360:.0f}°",
            showlegend=len(vectors) <= 8,
            hovertemplate=f"step {m}: ψ = {np.degrees(psi[m]) % 360:.1f}°<extra></extra>",
        )
    fig.add_scatter(
        x=[0, total.real],
        y=[0, total.imag],
        mode="lines+markers",
        line={"color": c["ink"], "width": 4},
        marker={"size": [0, 10], "color": c["ink"]},
        name=f"mean: |s|/N = {abs(total):.3g}",
    )
    fig.update_layout(
        height=height,
        width=height + 160,
        font={"family": FONT, "color": c["ink"], "size": 13},
        paper_bgcolor=c["surface"],
        plot_bgcolor=c["surface"],
        margin={"l": 48, "r": 24, "t": 48, "b": 48},
        title={"text": f"pathway {index + 1}: {','.join(str(p) for p in result.orders[index])}", "x": 0},
        xaxis={**_axis_style(c), "range": [-1.15, 1.15], "scaleanchor": "y", "title": {"text": "real"}},
        yaxis={**_axis_style(c), "range": [-1.15, 1.15], "title": {"text": "imaginary"}},
        legend={"font": {"size": 10, "color": c["ink2"]}, "bgcolor": "rgba(0,0,0,0)"},
    )
    return fig


def plot_pareto(
    front: SequenceABC[dict], *, x: str = "steps", y: str = "residual", theme: str = "light", height: int = 380
):
    """Scatter of a Pareto front: dictionaries with at least the ``x`` and ``y`` keys (and a ``label``)."""
    go, _ = _plotly()
    c = CHROME[theme]
    fig = go.Figure(
        go.Scatter(
            x=[f[x] for f in front],
            y=[f[y] for f in front],
            mode="markers+lines",
            text=[f.get("label", "") for f in front],
            marker={"color": SERIES[theme][0], "size": 9},
            line={"color": c["axis"], "width": 1},
            hovertemplate="%{text}<br>" + x + " %{x}, " + y + " %{y:.3g}<extra></extra>",
        )
    )
    fig.update_layout(
        height=height,
        font={"family": FONT, "color": c["ink"], "size": 13},
        paper_bgcolor=c["surface"],
        plot_bgcolor=c["surface"],
        margin={"l": 64, "r": 24, "t": 32, "b": 56},
        xaxis={**_axis_style(c), "title": {"text": x}},
        yaxis={**_axis_style(c), "title": {"text": y}},
    )
    return fig
