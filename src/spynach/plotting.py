"""Frequency axes and interactive plots of simulation results.

``ft_axis`` and ``axis_1d`` are ports of Spinach's ``kernel/utilities/ft_axis.m``
and ``kernel/utilities/axis_1d.m``. ``plot_1d`` follows Spinach's
``plot_1d.m`` (same parameters, defaults, and axis conventions) but draws an
interactive Plotly figure and returns it instead of drawing into the current
MATLAB axes. ``plot_2d`` follows ``plot_2d.m`` (same
parameters, contour spacing by ``contspacing.m``, and axis conventions) as
an interactive contour figure. ``plot_fid`` has no Spinach counterpart, nor
has ``plot_pathways``, the chord diagram of :func:`spynach.relaxation_pathways`,
whose geometry ``chord_layout`` returns as plain lists (the browser playground
draws it from those). ``figure_png`` renders any of the figures to PNG bytes
(Kaleido, or a Matplotlib redraw when Kaleido cannot run).

The plotting functions need Plotly (``pip install -e ".[plot]"``); the axis
functions and ``chord_layout`` need only NumPy.

Spinach authors, by original file (headers of Spinach 2.13; see docs/spinach-credits.md):
  axis_1d.m: Ilya Kuprov
  contspacing.m: Ilya Kuprov
  ft_axis.m: Ilya Kuprov
  plot_1d.m: Ilya Kuprov, Luke Edwards
  plot_2d.m: Ilya Kuprov
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

import numpy as np

from spynach.isotopes import spin
from spynach.system import FREE_ELECTRON_G, SpinSystem

# Categorical series colours, in fixed order, stepped separately for light and
# dark surfaces. Adjacent pairs stay distinguishable under the common forms of
# colour blindness; the first three are distinguishable all-pairs.
SERIES = {
    "light": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"],
    "dark": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"],
}

# Surface, ink, and rule colours
CHROME = {
    "light": {
        "surface": "#fcfcfb",
        "ink": "#0b0b0b",
        "ink2": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "border": "rgba(11,11,11,0.10)",
    },
    "dark": {
        "surface": "#1a1a19",
        "ink": "#ffffff",
        "ink2": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "border": "rgba(255,255,255,0.10)",
    },
}

FONT = 'system-ui, -apple-system, "Segoe UI", Roboto, sans-serif'

# Hover precision per axis unit
_HOVER_FORMAT = {
    "ppm": ".4f",
    "Hz": ".2f",
    "kHz": ".4f",
    "MHz": ".6f",
    "MHz-labframe": ".6f",
    "GHz": ".6f",
    "GHz-labframe": ".6f",
    "Gauss": ".3f",
    "mT": ".4f",
    "gtensor": ".6f",
    "points": "d",
}


def ft_axis(offset: float, sweep: float, npoints: int) -> np.ndarray:
    """Frequency axis (Hz) of a Fourier-transformed signal, as Spinach's ``ft_axis``.

    ``npoints`` points spaced ``sweep / npoints`` apart, centred on
    ``offset``; for even ``npoints`` the axis starts at ``offset - sweep/2``.
    This is the axis of ``np.fft.fftshift(np.fft.fft(fid, npoints))``.

    Spinach authors (ft_axis.m): Ilya Kuprov.
    """
    if not np.isscalar(offset) or np.iscomplexobj(offset):
        raise ValueError("offset must be a real scalar.")
    if not np.isscalar(sweep) or np.iscomplexobj(sweep) or sweep <= 0:
        raise ValueError("sweep must be a positive real scalar.")
    if int(npoints) != npoints or npoints <= 2:
        raise ValueError("npoints must be a real integer greater than 2")
    npoints = int(npoints)
    ax = np.linspace(-sweep / 2, sweep / 2, npoints + 1)
    if npoints % 2 == 1:
        return ax[1:] - (ax[1] - ax[0]) / 2 + offset
    return ax[:-1] + offset


def axis_1d(spin_system: SpinSystem, parameters: dict[str, Any]) -> tuple[np.ndarray, str]:
    """Axis and axis label of a 1D spectrum, as Spinach's ``axis_1d``.

    Uses ``parameters['sweep']`` (a scalar sweep width with
    ``parameters['offset']``, or a two-element ``[low, high]`` range in Hz),
    ``parameters['zerofill']`` (number of points), ``parameters['spins']``
    (one isotope), and ``parameters['axis_units']``: ``'ppm'``, ``'Hz'``,
    ``'kHz'``, ``'MHz'``, ``'GHz'``, ``'MHz-labframe'``, ``'GHz-labframe'``,
    ``'Gauss'``, ``'mT'``, ``'gtensor'``, or ``'points'``.

    Spinach authors (axis_1d.m): Ilya Kuprov.
    """
    sweep = np.atleast_1d(np.asarray(parameters.get("sweep"), dtype=float)) if "sweep" in parameters else None
    if sweep is None:
        raise ValueError("sweep width should be specified in parameters['sweep'].")
    if sweep.size not in (1, 2):
        raise ValueError("parameters['sweep'] should have one or two elements.")
    if sweep.size == 2:
        if sweep[0] >= sweep[1]:
            raise ValueError("the first element of parameters['sweep'] must be smaller than the second.")
        if "offset" in parameters:
            raise ValueError("offset should not be specified when parameters['sweep'] has two elements.")
        ax = np.linspace(sweep[0], sweep[1], int(parameters["zerofill"]))
    else:
        if "offset" not in parameters:
            raise ValueError("offset should be specified in parameters['offset'].")
        ax = ft_axis(float(parameters["offset"]), float(sweep[0]), parameters["zerofill"])

    spins = parameters.get("spins")
    if not isinstance(spins, (list, tuple)) or len(spins) != 1:
        raise ValueError("parameters['spins'] must be a list with exactly one isotope.")
    isotope = spins[0]
    basefrq = -spin(isotope)[0] * spin_system.magnet / (2 * np.pi)

    units = parameters["axis_units"]
    if units == "ppm":
        return -1e6 * ax / basefrq, f"{isotope} chemical shift / ppm"
    if units == "Gauss":
        return 1e4 * (spin_system.magnet - 2 * np.pi * ax / spin("E")[0]), "Magnetic induction / Gauss"
    if units == "mT":
        return 1e3 * (spin_system.magnet - 2 * np.pi * ax / spin("E")[0]), "Magnetic induction / mT"
    if units in ("Hz", "kHz", "MHz", "GHz"):
        scale = {"Hz": 1, "kHz": 1e-3, "MHz": 1e-6, "GHz": 1e-9}[units]
        return scale * ax, f"{isotope} offset frequency / {units}"
    if units in ("MHz-labframe", "GHz-labframe"):
        scale = 1e-6 if units.startswith("MHz") else 1e-9
        return -scale * (ax - basefrq), f"{isotope} frequency / {units.split('-')[0]}"
    if units == "gtensor":
        return -FREE_ELECTRON_G * (ax - basefrq) / basefrq, f"{isotope} g-tensor / Bohr magneton"
    if units == "points":
        return np.arange(1, int(parameters["zerofill"]) + 1, dtype=float), "digitisation points"
    raise ValueError("unknown axis units.")


def plot_1d(
    spin_system: SpinSystem,
    spectrum: np.ndarray | Sequence[np.ndarray],
    parameters: dict[str, Any],
    *,
    names: Sequence[str] | None = None,
    title: str | None = None,
    theme: str = "light",
    height: int = 400,
):
    """Interactive plot of one or more 1D spectra, after Spinach's ``plot_1d``.

    ``spectrum`` is a 1-D array, a 2-D array with one spectrum per column
    (as in MATLAB's ``plot``), or a list of 1-D arrays. Complex spectra are
    drawn as their real and imaginary parts, as in Spinach.

    ``parameters`` as for :func:`axis_1d`, with Spinach's defaults:
    ``axis_units='ppm'``, ``invert_axis=1`` (NMR tradition: frequency
    decreasing to the right), ``offset=0``, and ``zerofill`` taken from the
    spectrum length. ``parameters['derivative']`` is not ported.

    ``names`` labels the spectra in the legend and hover box; ``theme`` is
    ``'light'`` or ``'dark'``. Drag to zoom, double-click to zoom out; the
    hover box reads off every spectrum at the cursor. Returns a ``plotly.graph_objects.Figure``: display
    it by making it the last expression of a notebook cell, or call
    ``fig.show()`` or ``fig.write_html(path)``.

    Spinach authors (plot_1d.m): Ilya Kuprov, Luke Edwards.
    """
    traces = _as_traces(spectrum, names)
    npts = traces[0][1].size
    if any(y.size != npts for _, y in traces):
        raise ValueError("all spectra must have the same number of points.")

    parameters = dict(parameters)
    if parameters.get("derivative"):
        raise NotImplementedError("parameters['derivative'] is not ported yet.")
    if "offset" not in parameters and np.size(parameters.get("sweep")) == 1:
        parameters["offset"] = 0
    parameters.setdefault("axis_units", "ppm")
    parameters.setdefault("invert_axis", 1)
    parameters.setdefault("zerofill", npts)
    if parameters["zerofill"] != npts:
        raise ValueError(f"parameters['zerofill'] is {parameters['zerofill']} but the spectrum has {npts} points.")
    if not isinstance(parameters["axis_units"], str):
        raise ValueError("parameters['axis_units'] must be a character string.")

    ax, label = axis_1d(spin_system, parameters)
    fig = _figure(theme, title, height, traces)
    fig.update_traces(x=ax)
    lo, hi = float(ax.min()), float(ax.max())
    fig.update_xaxes(
        title_text=_rich(label, parameters["spins"][0]),
        range=[hi, lo] if parameters["invert_axis"] else [lo, hi],
        hoverformat=_HOVER_FORMAT[parameters["axis_units"]],
    )
    fig.update_yaxes(title_text="intensity / a.u.")
    return fig


def contspacing(
    smax: float, smin: float, delta: Sequence[float], k: int, signs: str, ncont: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Contour levels for a 2D spectrum (port of ``contspacing.m``).

    Positive levels run from ``delta[0] * smax`` to ``delta[1] * smax`` and
    negative ones from ``delta[2] * smin`` to ``delta[3] * smin``, spaced
    as the ``k``-th power of a linear grid of ``ncont`` points; ``signs``
    is ``'positive'``, ``'negative'`` or ``'both'``. Returns all levels in
    ascending order, then the positive and the negative sets.

    Spinach authors (contspacing.m): Ilya Kuprov.
    """
    delta = np.asarray(delta, dtype=float)
    if delta.size != 4 or np.any(delta < 0) or np.any(delta > 1) or delta[0] > delta[1] or delta[2] > delta[3]:
        raise ValueError(
            "delta must give ascending minimum and maximum contour fractions between 0 and 1 for each sign."
        )
    if signs not in ("positive", "negative", "both"):
        raise ValueError("signs must be 'positive', 'negative', or 'both'.")
    if int(k) != k or k < 1 or int(ncont) != ncont or ncont < 1:
        raise ValueError("k and ncont must be positive integers.")
    grid = np.linspace(0, 1, int(ncont)) ** k
    pos = (
        (delta[1] - delta[0]) * smax * grid + smax * delta[0]
        if smax > 0 and signs in ("positive", "both")
        else np.array([])
    )
    neg = (
        (delta[3] - delta[2]) * smin * grid + smin * delta[2]
        if smin < 0 and signs in ("negative", "both")
        else np.array([])
    )
    return np.concatenate([neg[::-1], pos]), pos, neg


def axis_2d(
    spin_system: SpinSystem, spectrum: np.ndarray, parameters: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, str, str]:
    """F1 and F2 axes and labels of a 2D spectrum, as ``plot_2d.m`` derives them.

    ``spectrum`` is ``(nF2, nF1)``: rows are the direct dimension F2
    (``parameters['sweep'][1]``, ``['offset'][1]``, ``['spins'][1]``),
    columns the indirect dimension F1. Scalar ``spins``, ``offset`` and
    ``sweep`` apply to both dimensions.

    Spinach authors (plot_2d.m, the axis calculation of plot_2d.m): Ilya Kuprov.
    """
    spins = list(parameters["spins"]) * 2 if len(parameters["spins"]) == 1 else list(parameters["spins"])
    offset = np.atleast_1d(np.asarray(parameters.get("offset", 0.0), dtype=float))
    sweep = np.atleast_1d(np.asarray(parameters["sweep"], dtype=float))
    offset = np.repeat(offset, 2) if offset.size == 1 else offset
    sweep = np.repeat(sweep, 2) if sweep.size == 1 else sweep
    units = parameters.get("axis_units", "ppm")
    axis_f1 = ft_axis(offset[0], sweep[0], spectrum.shape[1])
    axis_f2 = ft_axis(offset[1], sweep[1], spectrum.shape[0])
    if units == "ppm":
        axis_f1 = 1e6 * 2 * np.pi * axis_f1 / (spin(spins[0])[0] * spin_system.magnet)
        axis_f2 = 1e6 * 2 * np.pi * axis_f2 / (spin(spins[1])[0] * spin_system.magnet)
        labels = (f"F1: {spins[0]} chemical shift / ppm", f"F2: {spins[1]} chemical shift / ppm")
    elif units in ("Hz", "kHz", "MHz"):
        scale = {"Hz": 1.0, "kHz": 1e-3, "MHz": 1e-6}[units]
        axis_f1, axis_f2 = scale * axis_f1, scale * axis_f2
        labels = (f"F1: {spins[0]} linear frequency / {units}", f"F2: {spins[1]} linear frequency / {units}")
    elif units == "Gauss":
        axis_f1 = 1e4 * (spin_system.magnet - 2 * np.pi * axis_f1 / spin("E")[0])
        axis_f2 = 1e4 * (spin_system.magnet - 2 * np.pi * axis_f2 / spin("E")[0])
        labels = ("F1: magnetic induction / Gauss", "F2: magnetic induction / Gauss")
    elif units == "points":
        axis_f1, axis_f2 = np.arange(1, spectrum.shape[1] + 1), np.arange(1, spectrum.shape[0] + 1)
        labels = (f"F1: {spins[0]} linear frequency / points", f"F2: {spins[1]} linear frequency / points")
    else:
        raise ValueError("unknown axis units.")
    return axis_f1, axis_f2, labels[0], labels[1]


def plot_2d(
    spin_system: SpinSystem,
    spectrum: np.ndarray,
    parameters: dict[str, Any],
    *,
    ncont: int = 20,
    delta: Sequence[float] = (0.02, 0.2, 0.02, 0.2),
    k: int = 2,
    ncol: int = 256,
    m: int = 6,
    signs: str = "both",
    title: str | None = None,
    theme: str = "light",
    height: int = 560,
    max_points: int | None = 512,
):
    """Interactive contour plot of a real 2D spectrum, after Spinach's ``plot_2d``.

    ``spectrum`` is the ``(nF2, nF1)`` array that ``fftshift(fft2(fid))``
    of a sequence's FID gives (rows along the direct dimension); its real
    part is drawn, positive contours in red and negative ones in blue.
    ``ncont``, ``delta``, ``k`` and ``signs`` set the contour levels as in
    :func:`contspacing`; ``ncol`` and ``m`` (Spinach's colour map
    parameters) are accepted for compatibility and ``m`` steepens the
    colour gradient. Spinach passes these six positionally; here they are
    keyword-only, with Spinach's names. Both axes decrease to the right/top as in Spinach.
    Returns a ``plotly.graph_objects.Figure``.

    The figure embeds the matrix it draws, so a dimension longer than
    ``max_points`` is reduced first: consecutive points are binned and
    each bin keeps its value of largest magnitude (peaks and signs are
    preserved, the contour levels are those of the full spectrum). A
    2048 x 2048 spectrum reduces to 512 x 512 and a few MB of HTML;
    ``max_points=None`` draws every point.

    Spinach authors (plot_2d.m): Ilya Kuprov.
    """
    try:
        import plotly.graph_objects as go
    except ImportError as error:
        raise ImportError('plotting needs Plotly: pip install -e ".[plot]"') from error
    if theme not in CHROME:
        raise ValueError(f"theme must be 'light' or 'dark', not {theme!r}.")
    spectrum = np.asarray(spectrum)
    if np.iscomplexobj(spectrum):
        spectrum = spectrum.real
    if spectrum.ndim != 2:
        raise ValueError("spectrum must be a 2D array.")
    c = CHROME[theme]
    axis_f1, axis_f2, label_f1, label_f2 = axis_2d(spin_system, spectrum, parameters)
    smax, smin = float(spectrum.max()), float(spectrum.min())
    _, pos, neg = contspacing(smax, smin, delta, k, signs, ncont)
    if max_points is not None:
        spectrum, axis_f2 = _bin_extremes(spectrum, axis_f2, int(max_points), axis=0)
        spectrum, axis_f1 = _bin_extremes(spectrum, axis_f1, int(max_points), axis=1)
    fig = go.Figure()
    reds, blues = ("#b3261e", "#f2b8b5"), ("#1d4ed8", "#bfdbfe")

    def shade(start: str, end: str, fraction: float) -> str:
        a, b = (int(start[i : i + 2], 16) for i in (1, 3, 5)), (int(end[i : i + 2], 16) for i in (1, 3, 5))
        return "#" + "".join(f"{int(round(x + (y - x) * fraction)):02x}" for x, y in zip(a, b))

    # One trace per sign: the spectrum is mapped through a monotone piecewise-linear function that sends
    # the n-th contour level to n, so Plotly's uniform levels 0, 1, ..., reproduce Spinach's non-uniform
    # ones exactly, and the colour scale over the level index gives each contour its own shade.
    z = spectrum.T  # Plotly draws z[y, x]; x is F2 (rows of the spectrum), y is F1
    for levels, (strong, weak) in ((pos, reds), (neg, blues)):
        if not len(levels):
            continue
        nlev = len(levels)
        colours = [shade(weak, strong, (1 - n / max(nlev - 1, 1)) ** (1 / m)) for n in range(nlev)]
        colorscale = [(n / max(nlev - 1, 1), colour) for n, colour in enumerate(colours)]
        if nlev == 1:
            colorscale = [(0.0, colours[0]), (1.0, colours[0])]
        fig.add_contour(
            x=axis_f2,
            y=axis_f1,
            z=_level_index(z, levels),
            showscale=False,
            hoverinfo="skip",
            zmin=0,
            zmax=nlev - 1,
            autocontour=False,
            contours={
                "type": "levels",
                "coloring": "lines",
                "start": 0,
                "end": nlev - 1,
                "size": 1,
                "showlabels": False,
            },
            colorscale=colorscale,
            line={"width": 1.2},
            showlegend=False,
        )
    if not len(pos) and not len(neg):
        fig.add_annotation(text="all-zero spectrum", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
    axis = {
        "showgrid": True,
        "gridcolor": c["grid"],
        "zeroline": False,
        "showline": True,
        "linecolor": c["axis"],
        "ticks": "outside",
        "tickcolor": c["axis"],
        "tickfont": {"color": c["ink2"], "size": 12},
        "title": {"font": {"color": c["ink2"], "size": 13}},
        "mirror": True,
    }
    isotopes = list(parameters["spins"]) * 2 if len(parameters["spins"]) == 1 else list(parameters["spins"])
    fig.update_layout(
        height=height,
        width=height,
        font={"family": FONT, "color": c["ink"], "size": 13},
        paper_bgcolor=c["surface"],
        plot_bgcolor=c["surface"],
        margin={"l": 72, "r": 24, "t": 56 if title else 24, "b": 64},
        title={"text": title, "x": 0, "xanchor": "left", "font": {"size": 15, "color": c["ink"]}} if title else None,
        xaxis={
            **axis,
            "title": {**axis["title"], "text": _rich(label_f2, isotopes[1])},
            "range": [float(axis_f2.max()), float(axis_f2.min())],
        },
        yaxis={
            **axis,
            "title": {**axis["title"], "text": _rich(label_f1, isotopes[0])},
            "range": [float(axis_f1.max()), float(axis_f1.min())],
        },
        modebar={
            "bgcolor": "rgba(0,0,0,0)",
            "color": c["muted"],
            "activecolor": c["ink2"],
            "remove": ["lasso2d", "select2d"],
        },
        dragmode="zoom",
        showlegend=False,
    )
    return fig


def _level_index(z: np.ndarray, levels: np.ndarray) -> np.ndarray:
    """Monotone map sending ``levels[n]`` to ``n`` (piecewise linear between levels, extended linearly outside).

    Values more than one level spacing beyond the end levels are clamped to
    -1 and ``len(levels)``, outside the range of contours that are drawn.
    """
    order = np.argsort(levels)
    xs, ys = np.asarray(levels, dtype=float)[order], np.arange(len(levels), dtype=float)[order]
    span = xs[-1] - xs[0] if len(xs) > 1 else max(abs(xs[0]), 1.0)
    step = (xs[1] - xs[0]) if len(xs) > 1 else span
    last = (xs[-1] - xs[-2]) if len(xs) > 1 else span
    xs = np.concatenate([[xs[0] - step], xs, [xs[-1] + last]])
    ys = np.concatenate(
        [[ys[0] + (ys[0] - ys[1] if len(ys) > 1 else -1)], ys, [ys[-1] + (ys[-1] - ys[-2] if len(ys) > 1 else 1)]]
    )
    # float32 halves the size of the figure in a saved notebook; the level index needs no more precision
    return np.interp(z, xs, ys).astype(np.float32)


def _bin_extremes(
    spectrum: np.ndarray, axis_values: np.ndarray, max_points: int, axis: int
) -> tuple[np.ndarray, np.ndarray]:
    """Reduce ``spectrum`` along ``axis`` to at most ``max_points`` bins, keeping each bin's largest-magnitude value.

    The axis coordinate of a bin is the mean of its points. Returns the
    spectrum unchanged when it is short enough.
    """
    n = spectrum.shape[axis]
    if max_points < 1:
        raise ValueError("max_points must be a positive integer or None.")
    if n <= max_points:
        return spectrum, axis_values
    step = -(-n // max_points)
    nbins = -(-n // step)
    pad = nbins * step - n
    moved = np.moveaxis(spectrum, axis, 0)
    padded = np.concatenate([moved, np.zeros((pad,) + moved.shape[1:], dtype=moved.dtype)]) if pad else moved
    binned = padded.reshape(nbins, step, *moved.shape[1:])
    pick = np.argmax(np.abs(binned), axis=1)
    reduced = np.take_along_axis(binned, pick[:, None, ...], axis=1)[:, 0, ...]
    coords = np.concatenate([np.asarray(axis_values, dtype=float), np.full(pad, np.nan)]).reshape(nbins, step)
    return np.moveaxis(reduced, 0, axis), np.nanmean(coords, axis=1)


def plot_fid(
    fid: np.ndarray | Sequence[np.ndarray],
    parameters: dict[str, Any],
    *,
    names: Sequence[str] | None = None,
    title: str | None = None,
    theme: str = "light",
    height: int = 360,
):
    """Interactive plot of one or more free induction decays.

    The time axis starts at zero with dwell time ``1 / parameters['sweep']``
    and is drawn in ms when the acquisition is shorter than a second.
    Complex FIDs are drawn as their real and imaginary parts. Other
    arguments as for :func:`plot_1d`.
    """
    traces = _as_traces(fid, names)
    npts = traces[0][1].size
    if any(y.size != npts for _, y in traces):
        raise ValueError("all FIDs must have the same number of points.")
    if np.size(parameters.get("sweep")) != 1:
        raise ValueError("parameters['sweep'] must be a scalar sweep width in Hz.")
    t = np.arange(npts) / float(parameters["sweep"])
    scale, unit = (1e3, "ms") if t[-1] < 1 else (1.0, "s")

    fig = _figure(theme, title, height, traces)
    fig.update_traces(x=scale * t)
    fig.update_xaxes(title_text=f"time / {unit}", range=[0, scale * t[-1]], hoverformat=".4g")
    fig.update_yaxes(title_text="signal / a.u.")
    return fig


def _as_traces(data: np.ndarray | Sequence[np.ndarray], names: Sequence[str] | None) -> list[tuple[str, np.ndarray]]:
    """(name, real 1-D array) pairs, splitting complex data into real and imaginary parts."""
    if isinstance(data, np.ndarray) and data.ndim <= 2:
        columns = [data] if data.ndim == 1 else list(data.T)
    else:
        columns = [np.asarray(d) for d in data]
    if any(c.ndim != 1 for c in columns):
        raise ValueError("expected a 1-D array, a 2-D array of columns, or a list of 1-D arrays.")
    if names is None:
        names = [f"spectrum {n + 1}" for n in range(len(columns))] if len(columns) > 1 else [""]
    if len(names) != len(columns):
        raise ValueError(f"{len(names)} names given for {len(columns)} traces.")

    traces = []
    for name, column in zip(names, columns):
        if np.iscomplexobj(column):
            traces.append((f"{name} real".strip(), column.real))
            traces.append((f"{name} imag".strip(), column.imag))
        else:
            traces.append((name, column))
    return traces


def _rich(label: str, isotope: str) -> str:
    """Write the mass number of the isotope in an axis label as a superscript."""
    match = re.fullmatch(r"(\d+)([A-Za-z]+)", isotope)
    if not match:
        return label
    return label.replace(isotope, f"<sup>{match[1]}</sup>{match[2]}", 1)


def _figure(theme: str, title: str | None, height: int, traces: list[tuple[str, np.ndarray]]):
    """Line plot of the traces (x set by the caller) with the spynach look.

    Thin rules and quiet axes; a vertical crosshair and one hover box listing
    every trace at the cursor; a legend only when there are several traces.
    """
    try:
        import plotly.graph_objects as go
    except ImportError as error:
        raise ImportError('plotting needs Plotly: pip install -e ".[plot]"') from error
    if theme not in CHROME:
        raise ValueError(f"theme must be 'light' or 'dark', not {theme!r}.")
    c = CHROME[theme]

    axis = {
        "showgrid": True,
        "gridcolor": c["grid"],
        "gridwidth": 1,
        "zeroline": False,
        "showline": True,
        "linecolor": c["axis"],
        "linewidth": 1,
        "mirror": False,
        "ticks": "outside",
        "tickcolor": c["axis"],
        "ticklen": 4,
        "tickfont": {"color": c["ink2"], "size": 12},
        "title": {"font": {"color": c["ink2"], "size": 13}},
        "showspikes": True,
        "spikemode": "across",
        "spikesnap": "cursor",
        "spikethickness": 1,
        "spikedash": "solid",
        "spikecolor": c["muted"],
    }
    fig = go.Figure()
    fig.update_layout(
        height=height,
        font={"family": FONT, "color": c["ink"], "size": 13},
        paper_bgcolor=c["surface"],
        plot_bgcolor=c["surface"],
        margin={"l": 64, "r": 24, "t": 56 if title else 24, "b": 56},
        title={"text": title, "x": 0, "xanchor": "left", "xref": "paper", "font": {"size": 15, "color": c["ink"]}}
        if title
        else None,
        xaxis=axis,
        yaxis={**axis, "showspikes": False, "exponentformat": "power"},
        legend={
            "orientation": "h",
            "x": 0,
            "xanchor": "left",
            "y": 1.02,
            "yanchor": "bottom",
            "bgcolor": "rgba(0,0,0,0)",
            "font": {"color": c["ink2"]},
        },
        hoverlabel={"bgcolor": c["surface"], "bordercolor": c["border"], "font": {"color": c["ink"], "family": FONT}},
        modebar={
            "bgcolor": "rgba(0,0,0,0)",
            "color": c["muted"],
            "activecolor": c["ink2"],
            "remove": ["lasso2d", "select2d"],
        },
        dragmode="zoom",
        hovermode="x unified",
        showlegend=len(traces) > 1,
    )
    if len(traces) > len(SERIES[theme]):
        raise ValueError(f"at most {len(SERIES[theme])} traces can be told apart by colour; plot fewer.")
    for n, (name, y) in enumerate(traces):
        fig.add_scatter(
            y=y,
            name=name,
            mode="lines",
            line={"color": SERIES[theme][n], "width": 2},
            hovertemplate=f"{name}: %{{y:.4g}}<extra></extra>" if name else "%{y:.4g}<extra></extra>",
        )
    return fig


def plot_time_series(
    time_axis: np.ndarray,
    series: np.ndarray | Sequence[np.ndarray],
    *,
    names: Sequence[str] | None = None,
    xlabel: str = "time / s",
    ylabel: str = "amplitude",
    title: str | None = None,
    theme: str = "light",
    height: int = 360,
):
    """Interactive line plot of one or more real series against a common axis (MATLAB's ``plot(t, y)``).

    ``series`` is a 1-D array, a 2-D array with one series per row, or a
    list of 1-D arrays, each as long as ``time_axis``. More than eight
    series get an evenly spaced hue cycle, as Spinach's ``trajan`` draws.
    """
    import colorsys

    rows = (
        [np.asarray(series)]
        if isinstance(series, np.ndarray) and series.ndim == 1
        else [np.asarray(r).real for r in series]
    )
    rows = [r.real for r in rows]
    time_axis = np.asarray(time_axis, dtype=float)
    if any(r.ndim != 1 or r.size != time_axis.size for r in rows):
        raise ValueError("every series must be a 1-D array as long as time_axis.")
    if names is None:
        names = [f"series {n + 1}" for n in range(len(rows))] if len(rows) > 1 else [""]
    if len(names) != len(rows):
        raise ValueError(f"{len(names)} names given for {len(rows)} series.")
    fig = _figure(theme, title, height, [])
    if theme not in CHROME:
        raise ValueError(f"theme must be 'light' or 'dark', not {theme!r}.")
    palette = (
        SERIES[theme]
        if len(rows) <= len(SERIES[theme])
        else [
            "#" + "".join(f"{int(255 * v):02x}" for v in colorsys.hsv_to_rgb((n + 1) / len(rows), 0.75, 0.75))
            for n in range(len(rows))
        ]
    )
    for n, (name, y) in enumerate(zip(names, rows)):
        fig.add_scatter(
            x=time_axis,
            y=y,
            name=name,
            mode="lines",
            line={"color": palette[n], "width": 2},
            hovertemplate=f"{name}: %{{y:.4g}}<extra></extra>" if name else "%{y:.4g}<extra></extra>",
        )
    fig.update_layout(showlegend=len(rows) > 1)
    fig.update_xaxes(title_text=xlabel, range=[float(time_axis.min()), float(time_axis.max())], hoverformat=".4g")
    fig.update_yaxes(title_text=ylabel)
    return fig


def plot_trajan(
    spin_system: SpinSystem,
    traj: np.ndarray,
    property: str,
    time_axis: np.ndarray | None = None,
    *,
    title: str | None = None,
    theme: str = "light",
    height: int = 360,
):
    """Trajectory analysis plot, after Spinach's ``trajan``: the amplitudes from :func:`spynach.trajan` against time.

    Without ``time_axis`` the x axis is the trajectory point number. The
    y range is padded by five percent of the largest amplitude, as in Spinach.
    """
    from spynach.states import trajan

    result, labels = trajan(spin_system, traj, property)
    x = np.arange(result.shape[1]) if time_axis is None else np.asarray(time_axis, dtype=float)
    ylabel = {
        "correlation_order": "correlation order amplitude",
        "coherence_order": "coherence order amplitude",
        "total_each_spin": "density touching each spin",
        "local_each_spin": "density local to each spin",
    }[property]
    heading = {
        "correlation_order": "correlation orders",
        "coherence_order": "coherence orders",
        "total_each_spin": "spin populations",
        "local_each_spin": "spin populations",
    }[property]
    fig = plot_time_series(
        x,
        result,
        names=labels,
        xlabel="time / s" if time_axis is not None else "trajectory point",
        ylabel=ylabel,
        title=title or heading,
        theme=theme,
        height=height,
    )
    top = float(result.max()) if result.size else 1.0
    fig.update_yaxes(range=[-0.05 * top, 1.05 * top])
    return fig


def chord_layout(
    pathways,
    *,
    classes: Sequence[str] | None = None,
    merge: dict[str, str] | None = None,
    split: bool = False,
    min_rate: float = 0.0,
) -> dict[str, Any]:
    """Geometry of the chord diagram of :func:`plot_pathways`, as plain lists (JSON-ready), with no Plotly.

    Returns ``{"arcs", "ribbons", "names", "widest", "span", "secular"}``: one arc per basis operator
    (``label``, ``html``, ``order``, ``auto`` rate, ``outline`` polygon, ``text`` position and angle in
    degrees clockwise, ``hover``), one ribbon per drawn pathway or class (``source``, ``target``, ``name``,
    ``colour`` index into ``names``, ``rate``, ``outline``, ``hover``), the ribbon names heaviest first, the
    largest drawn rate and the angular span of an arc. The circle has unit radius, ``y`` points up.
    :func:`plot_pathways` draws it with Plotly, and the browser playground (``try.html``) as SVG.
    """
    merge = merge or {}
    ops = pathways.operators
    n = len(ops)
    gamma = pathways.rate_matrix()

    # Ribbons (source, target, name, rate, hover text), and the colour of each name
    ribbons = []
    for link in pathways.links():
        present = [cls for cls in link.signature if classes is None or cls in classes]
        if not present:
            continue
        names: dict[str, list[str]] = {}
        for cls in present:
            names.setdefault(merge.get(cls, cls), []).append(cls)
        hover = _pathway_hover(pathways, link, present)
        parts = (
            [(name, sum(link.rates[cls] for cls in members)) for name, members in names.items()]
            if split
            else [(" + ".join(names), sum(link.rates[cls] for cls in present))]
        )
        for name, rate in parts:
            if abs(rate) > min_rate and rate != 0:
                ribbons.append((link.source, link.target, name, rate, hover))
    weight: dict[str, float] = {}
    for _, _, name, rate, _ in ribbons:
        weight[name] = weight.get(name, 0.0) + abs(rate)
    order = sorted(weight, key=lambda m: -weight[m])

    # Arcs: equal spans, clockwise, the first block of total coherence order (or of the coherence of
    # every isotope, when all operators share one total order) centred on the left
    gap = min(0.05, 0.6 / n)
    span = (2 * np.pi - n * gap) / n
    same = (lambda op: op.order == ops[0].order) if len({op.order for op in ops}) > 1 else None
    same = same or (lambda op: op.coherence == ops[0].coherence)
    first_block = next((k for k, op in enumerate(ops) if not same(op)), n)
    begin = np.pi + first_block * (span + gap) / 2 - gap / 2 - np.arange(n) * (span + gap)
    centre = begin - span / 2

    # Ribbon ends: as wide as the rate, packed in the middle of each arc, ordered so that ribbons do not cross
    load = np.zeros(n)
    for i, j, _, rate, _ in ribbons:
        load[i] += abs(rate)
        load[j] += abs(rate)
    scale = 0.9 * span / load.max() if ribbons else 0.0
    width = [min(max(abs(rate) * scale, 0.004), span) for _, _, _, rate, _ in ribbons]
    ends: dict[int, list[tuple[int, int, int]]] = {k: [] for k in range(n)}
    for r, (i, j, _, _, _) in enumerate(ribbons):
        ends[i].append((r, j, 1))
        ends[j].append((r, i, -1))
    slots: dict[tuple[int, int], tuple[float, float]] = {}
    for k, items in ends.items():
        items.sort(key=lambda item: ((item[1] - k) % n, item[2] * item[0]), reverse=True)
        a = centre[k] + min(sum(width[r] for r, _, _ in items), span) / 2
        for r, _, _ in items:
            slots[(r, k)] = (a, a - width[r])
            a -= width[r]

    rim = 1.0

    def arc(a0: float, a1: float, radius: float) -> np.ndarray:
        t = np.linspace(a0, a1, max(int(abs(a1 - a0) / 0.02), 2))
        return radius * np.stack([np.cos(t), np.sin(t)], axis=1)

    def point(angle: float) -> np.ndarray:
        return rim * np.array([np.cos(angle), np.sin(angle)])

    def curve(p0: np.ndarray, p1: np.ndarray, control: np.ndarray) -> np.ndarray:
        t = np.linspace(0, 1, 48)[:, None]
        return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * control + t**2 * p1

    def points(outline: np.ndarray) -> list[list[float]]:
        return np.round(outline, 4).tolist()

    drawn = []
    for r, (i, j, name, rate, hover) in enumerate(ribbons):
        (a0, a1), (b0, b1) = slots[(r, i)], slots[(r, j)]
        separation = abs(np.angle(np.exp(1j * (centre[i] - centre[j]))))
        bisector = np.angle(np.exp(1j * centre[i]) + np.exp(1j * centre[j]))
        control = 0.85 * rim * (1 - separation / np.pi) * np.array([np.cos(bisector), np.sin(bisector)])
        outline = np.concatenate(
            [
                arc(a0, a1, rim),
                curve(point(a1), point(b0), control),
                arc(b0, b1, rim),
                curve(point(b1), point(a0), control),
            ]
        )
        drawn.append(
            {
                "source": int(i),
                "target": int(j),
                "name": name,
                "colour": order.index(name),
                "rate": float(rate),
                "outline": points(outline),
                "hover": hover,
            }
        )
    arcs = []
    for k, op in enumerate(ops):
        outline = np.concatenate(
            [arc(begin[k], begin[k] - span, rim + 0.015), arc(begin[k] - span, begin[k], rim + 0.13)]
        )
        tangent = (np.degrees(centre[k]) - 90 + 90) % 180 - 90  # along the arc, kept upright
        arcs.append(
            {
                "label": op.label,
                "html": op.html,
                "order": op.order,
                "auto": float(gamma[k, k]),
                "outline": points(outline),
                "text": [
                    round(float((rim + 0.072) * np.cos(centre[k])), 4),
                    round(float((rim + 0.072) * np.sin(centre[k])), 4),
                    round(float(-tangent), 2),
                ],
                "hover": f"<b>{op.html}</b><br>{op.info or f'coherence order {op.order:+d}'}"
                f"<br>auto-relaxation {gamma[k, k]:.4g} s⁻¹",
            }
        )
    widest = max((abs(rate) for _, _, _, rate, _ in ribbons), default=0.0)
    return {
        "arcs": arcs,
        "ribbons": drawn,
        "names": order,
        "widest": float(widest),
        "span": float(span),
        "secular": pathways.secular,
    }


def plot_pathways(
    pathways,
    *,
    classes: Sequence[str] | None = None,
    merge: dict[str, str] | None = None,
    split: bool = False,
    min_rate: float = 0.0,
    title: str | None = None,
    theme: str = "light",
    height: int = 720,
):
    """Chord diagram of cross-relaxation pathways (:func:`spynach.pathways.relaxation_pathways`).

    Every basis operator is an arc on the circle, in blocks of total
    coherence order (0 on the left, then +1, +2, ... clockwise, -1 at the
    bottom); every nonzero off-diagonal element ``Gamma[i, j]`` is a ribbon
    between two arcs, as wide as ``|Gamma[i, j]|``. The colour names the
    mechanisms: with ``split=False`` there is one ribbon per pathway,
    coloured by the combination of mechanism classes that contribute to it
    (``DD + DD/CSA``, say) and as wide as the total rate; with ``split=True``
    one ribbon per class, side by side, each as wide as that class's rate.
    ``classes`` keeps only the classes listed, ``merge`` renames classes
    before they are combined (``{"DD/DD": "DD"}`` counts the dipolar
    cross-correlations as dipolar), and ``min_rate`` (s^-1) hides the weaker
    pathways. Hovering over a ribbon shows each class's rate and expression,
    hovering over an arc the operator's auto-relaxation rate. The geometry
    comes from :func:`chord_layout`.
    """
    try:
        import plotly.graph_objects as go
    except ImportError as error:
        raise ImportError('plotting needs Plotly: pip install -e ".[plot]"') from error
    if theme not in CHROME:
        raise ValueError(f"theme must be 'light' or 'dark', not {theme!r}.")
    c = CHROME[theme]
    layout = chord_layout(pathways, classes=classes, merge=merge, split=split, min_rate=min_rate)
    palette = SERIES[theme]

    fig = go.Figure()
    shown: set[str] = set()
    for ribbon in layout["ribbons"]:
        name, outline = ribbon["name"], np.asarray(ribbon["outline"])
        colour = palette[ribbon["colour"] % len(palette)]
        fig.add_scatter(
            x=outline[:, 0],
            y=outline[:, 1],
            mode="lines",
            fill="toself",
            fillcolor=colour,
            opacity=0.75,
            line={"color": colour, "width": 0.5},
            name=name,
            legendgroup=name,
            showlegend=name not in shown,
            hoveron="fills",
            hoverinfo="text",
            text=ribbon["hover"],
        )
        shown.add(name)

    # The operator arcs, with their labels written along them
    size = float(np.clip(layout["span"] * 1.07 * height / 2.7 / 5, 7, 14))
    annotations = []
    for arc in layout["arcs"]:
        outline = np.asarray(arc["outline"])
        fig.add_scatter(
            x=outline[:, 0],
            y=outline[:, 1],
            mode="lines",
            fill="toself",
            fillcolor=c["ink"],
            line={"color": c["ink"], "width": 0.5},
            showlegend=False,
            hoveron="fills",
            hoverinfo="text",
            text=arc["hover"],
        )
        x, y, angle = arc["text"]
        annotations.append(
            {
                "x": x,
                "y": y,
                "xref": "x",
                "yref": "y",
                "text": arc["html"],
                "showarrow": False,
                "textangle": angle,
                "font": {"color": c["surface"], "size": size, "family": FONT},
            }
        )
    annotations.append(
        {
            "x": 0,
            "y": 0,
            "xref": "paper",
            "yref": "paper",
            "xanchor": "left",
            "yanchor": "bottom",
            "text": f"ribbon width ∝ |Γ|, widest {layout['widest']:.3g} s⁻¹ · secular: {layout['secular']}",
            "showarrow": False,
            "font": {"color": c["muted"], "size": 11},
        }
    )
    hidden = {"visible": False, "range": [-1.3, 1.3], "fixedrange": False}
    fig.update_layout(
        height=height,
        width=height,
        font={"family": FONT, "color": c["ink"], "size": 13},
        paper_bgcolor=c["surface"],
        plot_bgcolor=c["surface"],
        margin={"l": 16, "r": 16, "t": 56 if title else 40, "b": 24},
        title={"text": title, "x": 0, "xanchor": "left", "xref": "paper", "font": {"size": 15, "color": c["ink"]}}
        if title
        else None,
        xaxis=hidden,
        yaxis={**hidden, "scaleanchor": "x", "scaleratio": 1},
        legend={
            "orientation": "h",
            "x": 0,
            "xanchor": "left",
            "y": 1.0,
            "yanchor": "bottom",
            "bgcolor": "rgba(0,0,0,0)",
            "font": {"color": c["ink2"]},
        },
        hoverlabel={"bgcolor": c["surface"], "bordercolor": c["border"], "font": {"color": c["ink"], "family": FONT}},
        modebar={"bgcolor": "rgba(0,0,0,0)", "color": c["muted"], "activecolor": c["ink2"]},
        hovermode="closest",
        showlegend=bool(layout["ribbons"]),
        annotations=annotations,
    )
    return fig


def _pathway_hover(pathways, link, present: Sequence[str]) -> str:
    """Hover text of a pathway: total rate, then every class's rate and expression."""
    a, b = pathways.operators[link.source], pathways.operators[link.target]
    total = sum(link.rates[cls] for cls in present)
    lines = [f"<b>{a.html} ↔ {b.html}</b>", f"Γ = {total:.4g} s⁻¹"]
    for cls in present:
        lines.append(f"<b>{cls}</b> {link.rates[cls]:.4g} s⁻¹:")
        lines += _wrap(_html_expression(str(link.expressions[cls])), 110)
    return "<br>".join(lines)


# SymPy's text form of the pathway expressions, made readable in a hover box
_EXPRESSION_HTML = (
    (re.compile(r"sqrt\((\d+)\)"), r"√\1"),
    (re.compile(r"\*\*(\d+)"), r"<sup>\1</sup>"),
    (re.compile(r"\bomega_(\w+?)\b"), r"ω<sub>\1</sub>"),
    (re.compile(r"\bdsigma_(\w+?)\b"), r"Δσ<sub>\1</sub>"),
    (re.compile(r"\bCQ_(\w+?)\b"), r"C<sub>Q</sub><sup>\1</sup>"),
    (re.compile(r"\bdJ_(\w+?)\b"), r"ΔJ<sub>\1</sub>"),
    (re.compile(r"\bb_(\w+?)\b"), r"b<sub>\1</sub>"),
    (re.compile(r"\bJ_(\d)\("), r"J<sub>\1</sub>("),
    (re.compile(r"\bpi\b"), "π"),
    (re.compile(r"\*"), "·"),
)


def _html_expression(text: str) -> str:
    for pattern, replacement in _EXPRESSION_HTML:
        text = pattern.sub(replacement, text)
    return text


def _wrap(text: str, width: int) -> list[str]:
    lines, line = [], ""
    for word in text.split(" "):
        if line and len(line) + len(word) + 1 > width:
            lines.append(line)
            line = "    " + word
        else:
            line = f"{line} {word}" if line else word
    return [*lines, line]


def figure_png(fig, *, scale: float = 2) -> bytes:
    """PNG bytes of a Plotly figure, for GitHub, reports and papers (MATLAB's ``print -dpng``).

    Kaleido renders the figure exactly when it can run (it needs a Chrome
    build, ``plotly_get_chrome``); otherwise the traces are redrawn with
    Matplotlib in the figure's colours, with its titles, ranges and legend.
    The Matplotlib rendering covers what ``spynach.plotting`` draws (lines,
    bars and the contour figures of ``plot_2d``) and is close to, not
    identical with, the Plotly figure. ``scale`` multiplies the pixel size.
    """
    global _KALEIDO
    if _KALEIDO is not False:
        try:
            import plotly.io as pio

            png = pio.to_image(fig, format="png", scale=scale)
            _KALEIDO = True
            return png
        except Exception:  # Kaleido not installed, or no Chrome for it: the same answer for every figure
            _KALEIDO = False
    try:
        import matplotlib
    except ImportError as error:
        raise ImportError(
            'figure_png needs Kaleido with a Chrome build, or Matplotlib: pip install -e ".[docs]"'
        ) from error
    matplotlib.use("Agg")
    return _png_with_matplotlib(fig, scale)


_KALEIDO: bool | None = None  # whether Kaleido rendered a figure in this process; None until tried


def _png_with_matplotlib(fig, scale: float) -> bytes:
    """Static rendering of a Plotly figure with Matplotlib: traces, subplots, shapes and text.

    Subplots are placed from their axis domains inside the figure's margins; rectangle and line
    shapes and text annotations are drawn on the axes (or the figure) they refer to; sliders and
    buttons are ignored, so the static copy shows the figure's initial state.
    """
    import base64
    import io
    import re as _re

    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    def plain(text) -> str:
        return _re.sub(r"<br\s*/?>", "\n", _re.sub(r"<(?!br)[^>]+>", "", text or ""))

    def values(data) -> np.ndarray:
        # Figures rebuilt from notebook JSON carry arrays in Plotly's binary form
        if isinstance(data, dict) and "bdata" in data:
            array = np.frombuffer(base64.b64decode(data["bdata"]), dtype=np.dtype(data["dtype"]))
            return array.reshape([int(n) for n in str(data["shape"]).split(",")]) if "shape" in data else array
        return np.asarray(data)

    layout = fig.layout
    surface = layout.paper_bgcolor or "#ffffff"
    ink = (layout.font.color if layout.font else None) or "#1c2024"
    width_px = layout.width or 700
    height_px = layout.height or 450
    margin = layout.margin
    left = (margin.l if margin.l is not None else 80) / width_px
    right = (margin.r if margin.r is not None else 80) / width_px
    top = (margin.t if margin.t is not None else 100) / height_px
    bottom = (margin.b if margin.b is not None else 80) / height_px
    plot_w, plot_h = 1 - left - right, 1 - top - bottom
    mpl_fig = plt.figure(figsize=(width_px / 96, height_px / 96))
    mpl_fig.patch.set_facecolor(surface)

    def axis_object(name: str):  # "x", "x2", "y3" -> layout.xaxis, layout.xaxis2, layout.yaxis3
        kind, number = name[0], name[1:]
        return layout[f"{kind}axis{number}"]

    axes: dict[tuple[str, str], object] = {}

    def axes_for(x_name: str, y_name: str):
        key = (x_name, y_name)
        if key not in axes:
            xd = axis_object(x_name).domain or (0.0, 1.0)
            yd = axis_object(y_name).domain or (0.0, 1.0)
            ax = mpl_fig.add_axes(
                [left + xd[0] * plot_w, bottom + yd[0] * plot_h, (xd[1] - xd[0]) * plot_w, (yd[1] - yd[0]) * plot_h]
            )
            ax.set_facecolor(layout.plot_bgcolor or surface)
            axes[key] = ax
        return axes[key]

    dashes = {"dot": ":", "dash": "--", "dashdot": "-.", "longdash": "--", "longdashdot": "-."}
    labelled: dict[object, int] = {}
    for trace in fig.data:
        ax = axes_for(trace.xaxis or "x", trace.yaxis or "y")
        label = plain(trace.name) if trace.name and trace.showlegend is not False else None
        if label is not None:
            labelled[ax] = labelled.get(ax, 0) + 1
        alpha = trace.opacity if trace.opacity is not None else 1.0
        if trace.type == "contour":
            start, end, size = trace.contours.start, trace.contours.end, trace.contours.size
            levels = np.arange(start, end + size / 2, size)
            colorscale = trace.colorscale
            colours = [
                colorscale[min(int(round(k * (len(colorscale) - 1) / max(len(levels) - 1, 1))), len(colorscale) - 1)][1]
                for k in range(len(levels))
            ]
            ax.contour(values(trace.x), values(trace.y), values(trace.z), levels=levels, colors=colours, linewidths=0.6)
        elif trace.type in ("scatter", "scattergl"):
            x, y = values(trace.x).astype(float), values(trace.y).astype(float)
            mode = trace.mode or "lines"
            if trace.fill == "toself":  # filled outlines: the ribbons and arcs of plot_pathways
                ax.fill(x, y, color=trace.fillcolor or trace.line.color, alpha=alpha, lw=0, label=label)
            elif "lines" in mode:
                ax.plot(
                    x,
                    y,
                    lw=(trace.line.width or 2) * 0.6,
                    color=trace.line.color,
                    ls=dashes.get(trace.line.dash or "solid", "-"),
                    alpha=alpha,
                    label=label,
                )
            if "markers" in mode:
                size = trace.marker.size
                ms = float(np.max(size)) / 2 if size is not None and np.ndim(size) else (size or 6) / 2
                ax.plot(
                    x,
                    y,
                    "o",
                    ms=ms,
                    color=trace.marker.color or trace.line.color,
                    alpha=alpha,
                    label=None if "lines" in mode else label,
                )
        elif trace.type == "bar":
            ax.bar(values(trace.x), values(trace.y).astype(float), color=trace.marker.color, alpha=alpha, label=label)

    def ref_axes(xref, yref):
        if xref in (None, "paper") or yref in (None, "paper"):
            return None
        return axes_for(xref, yref)

    for shape in layout.shapes or ():
        ax = ref_axes(shape.xref, shape.yref)
        colour = shape.fillcolor or (shape.line.color if shape.line else None) or ink
        if ax is None:
            continue
        if shape.type == "rect":
            ax.add_patch(
                Rectangle(
                    (shape.x0, shape.y0),
                    shape.x1 - shape.x0,
                    shape.y1 - shape.y0,
                    facecolor=shape.fillcolor or "none",
                    edgecolor=(shape.line.color if shape.line and shape.line.width else "none"),
                    alpha=shape.opacity if shape.opacity is not None else 1.0,
                )
            )
        elif shape.type == "line":
            ax.plot([shape.x0, shape.x1], [shape.y0, shape.y1], color=colour, lw=(shape.line.width or 1) * 0.6)
    for note in layout.annotations or ():
        if not note.text:
            continue
        size = (note.font.size if note.font and note.font.size else 11) * 0.85
        colour = (note.font.color if note.font and note.font.color else None) or ink
        if note.xref == "paper" and note.yref == "paper":
            x = left + note.x * plot_w if note.x is not None else 0.5
            y = bottom + note.y * plot_h if note.y is not None else 0.95
            ha = note.xanchor if note.xanchor in ("left", "center", "right") else "center"
            va = {"top": "top", "bottom": "bottom", "middle": "center"}.get(note.yanchor or "", "bottom")
            mpl_fig.text(x, y, plain(note.text), fontsize=size, color=colour, ha=ha, va=va)
        else:
            ax = ref_axes(note.xref or "x", note.yref or "y")
            if ax is not None:
                ax.text(
                    note.x,
                    note.y,
                    plain(note.text),
                    fontsize=size,
                    color=colour,
                    ha="center",
                    va="center",
                    rotation=-(note.textangle or 0),
                    rotation_mode="anchor",
                )

    for (x_name, y_name), ax in axes.items():
        for axis, mpl_axis, setter in (
            (axis_object(x_name), ax.xaxis, ax.set_xlim),
            (axis_object(y_name), ax.yaxis, ax.set_ylim),
        ):
            if axis.type == "log":
                (ax.set_xscale if mpl_axis is ax.xaxis else ax.set_yscale)("log")
                if axis.range:  # Plotly stores log-axis ranges as log10 values
                    setter(*(10.0 ** np.asarray(axis.range, dtype=float)))
            elif axis.range:
                setter(*axis.range)
            if axis.visible is False:
                mpl_axis.set_visible(False)
                for side in ax.spines.values():
                    side.set_visible(False)
                continue
            mpl_axis.label.set_color((axis.title.font.color if axis.title and axis.title.font else None) or ink)
            ax.tick_params(
                axis=mpl_axis.axis_name, colors=(axis.tickfont.color if axis.tickfont else None) or ink, labelsize=9
            )
            if axis.showgrid is not False:
                ax.grid(True, axis=mpl_axis.axis_name, color=axis.gridcolor or "#e1e0d9", linewidth=0.6)
            if axis.showticklabels is False:
                mpl_axis.set_ticklabels([])
            if axis.dtick and axis.range:
                lo, hi = sorted(axis.range)
                ticks = np.arange(np.ceil(lo / axis.dtick) * axis.dtick, hi + 1e-9, axis.dtick)
                (ax.set_xticks if mpl_axis is ax.xaxis else ax.set_yticks)(ticks)
            if axis.title and axis.title.text:
                (ax.set_xlabel if mpl_axis is ax.xaxis else ax.set_ylabel)(plain(axis.title.text), fontsize=10)
        if axis_object(y_name).scaleanchor == x_name:
            ax.set_aspect("equal", adjustable="datalim")
        for spine in ax.spines.values():
            spine.set_color(axis_object(x_name).linecolor or "#c3c2b7")
        if labelled.get(ax, 0) > 1 and layout.showlegend is not False:
            ax.legend(fontsize=8, labelcolor=ink, loc="best", framealpha=0.7, facecolor=surface, edgecolor="none")
    # Shared axes (Plotly's ``matches``): copy the limits of the axis they follow
    for (x_name, y_name), ax in axes.items():
        for name, setter in ((x_name, ax.set_xlim), (y_name, ax.set_ylim)):
            target = axis_object(name).matches
            if target:
                for (ox, oy), other in axes.items():
                    if target in (ox, oy):
                        setter(*(other.get_xlim() if target[0] == "x" else other.get_ylim()))
                        break
    if layout.title and layout.title.text:
        mpl_fig.text(left, 1 - top * 0.45, plain(layout.title.text), fontsize=11, color=ink, ha="left", va="center")
    buffer = io.BytesIO()
    mpl_fig.savefig(buffer, format="png", dpi=96 * scale, facecolor=surface)
    plt.close(mpl_fig)
    return buffer.getvalue()
