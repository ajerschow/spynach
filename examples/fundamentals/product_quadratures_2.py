"""Lie-group product quadratures on a radiation-damped chirped oscillator: translation of Spinach's
examples/fundamentals/quadratures/product_quadratures_2.m.

    OPENBLAS_NUM_THREADS=4 PYTHONPATH=src python examples/fundamentals/product_quadratures_2.py \
        [--np-ref N] [--grids N] [--out DIR] [--compare FILE.mat] [--no-plot]

A classical magnetisation vector under a chirped offset (400 Hz/s), relaxation (R1 = R2 = 10 Hz) and the
Bloembergen-Pound radiation damping term (40 Hz) is a Lie equation whose generator depends on time and on the
state. Starting 178 degrees from equilibrium, the reference trajectory over 0.5 s is integrated with RKMK-DP8 on
4096 points; then every method of iserstep (PWCL, LG2, LG4, LG4A, RKMK4, RKMK-DP5, RKMK-DP8) is run on ten grids
from 256 to 2048 points, the relative error of the final vector is plotted against the grid size on log-log axes,
and the empirical convergence order of each method is printed (the slope over the finest third of the grids).

Outputs in DIR: product_quadratures_2_traj.npy (3 x np_ref reference trajectory), product_quadratures_2_bench.npy
(grids x methods relative errors), product_quadratures_2_grids.npy, and product_quadratures_2.html (the two
figures). --compare FILE.mat reports the normwise relative error against mu_traj and bench saved by the MATLAB
script (add `save('product_quadratures_2.mat','mu_traj','bench','-v7')` before its plotting section).

Differences from the MATLAB script: the spin system that carries the tolerances is one proton in sphten-liouv
(Spinach's bootstrap() makes a ghost spin, which Spynach does not have); the generator is a Python function
instead of an anonymous function; --np-ref and --grids reduce the sizes for quick checks (default: Spinach's).
Stage labels (reference, benchmark, plot) are those of comparison/cases/product_quadratures_2; keep them in step.
"""

from __future__ import annotations

import argparse
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

import spynach
from spynach.conventions import euler2dcm
from spynach.propagation import iserstep

METHODS = ("PWCL", "LG2", "LG4", "LG4A", "RKMK4", "RKMK-DP5", "RKMK-DP8")
LABELS = ("LP", "LG-2", "LG-4", "LG-4A", "RKMK4", "RKMK-DP5", "RKMK-DP8")
CHIRP, R1, R2, RRD = 2 * np.pi * 400, 10.0, 10.0, 40.0  # 400 Hz/s chirp rate, 10 Hz relaxation, 40 Hz damping
DURATION = 0.5  # seconds


class Timer:
    """Stage timer: ``with timer("label"):`` records and prints the wall-clock time of the block."""

    def __init__(self, verbose: bool = True):
        self.records: list[tuple[str, float]] = []
        self.verbose = verbose

    @contextmanager
    def __call__(self, label: str):
        t0 = time.perf_counter()
        yield
        seconds = time.perf_counter() - t0
        self.records.append((label, seconds))
        if self.verbose:
            print(f"{label:32s} {seconds:8.1f} s", flush=True)


def generator(t: float, mu: np.ndarray) -> np.ndarray:
    """Bloch-Maxwell generator: chirped offset and relaxation, plus the Bloembergen-Pound radiation damping term."""
    linear = np.array([[R2, CHIRP * t, 0], [-CHIRP * t, R2, 0], [0, 0, R1]])
    damping = RRD * np.array([[mu[2], 0, 0], [0, mu[2], 0], [-mu[0], -mu[1], 0]])
    return -1j * (linear + damping)


def initial_magnetisation() -> np.ndarray:
    """178 degrees from +z, as euler2dcm(0, pi*178/180, 0) * [0 0 1]'."""
    return (euler2dcm([0.0, np.pi * 178 / 180, 0.0]) @ np.array([0.0, 0.0, 1.0])).astype(complex)


def bootstrap() -> spynach.SpinSystem:
    """A one-proton Liouville-space system: iserstep and step take their tolerances from it, nothing else."""
    spin_system = spynach.create({"isotopes": ["1H"], "magnet": 0.0}, {})
    return spynach.basis(spin_system, {"formalism": "sphten-liouv", "approximation": "none"})


def trajectory(spin_system, method: str, npoints: int, keep: bool = False):
    """``npoints`` points over DURATION with ``method``; the final vector, or the whole trajectory with ``keep``."""
    dt = DURATION / (npoints - 1)
    mu = initial_magnetisation()
    traj = np.zeros((3, npoints), dtype=complex)
    traj[:, 0] = mu
    for n in range(1, npoints):
        mu = iserstep(spin_system, generator, (n - 1) * dt, method, mu, dt)
        traj[:, n] = mu
    return traj if keep else mu


def simulate(np_ref: int = 2**12, grids: int = 10, timer: Timer | None = None):
    """Returns (mu_traj, grid sizes, bench): the reference trajectory and the relative errors (grids x methods)."""
    timer = Timer() if timer is None else timer
    spin_system = bootstrap()
    with timer("reference"):
        mu_traj = trajectory(spin_system, "RKMK-DP8", np_ref, keep=True)
    mu_ref = mu_traj[:, -1]
    sizes = np.ceil(2.0 ** np.linspace(8, 11, grids)).astype(int)
    bench = np.zeros((len(sizes), len(METHODS)))
    with timer("benchmark"):
        for k, npoints in enumerate(sizes):
            for m, method in enumerate(METHODS):
                mu = trajectory(spin_system, method, int(npoints))
                bench[k, m] = np.linalg.norm(mu - mu_ref) / np.linalg.norm(mu_ref)
    return mu_traj, sizes, bench


def convergence_orders(sizes: np.ndarray, bench: np.ndarray) -> dict[str, float]:
    """Slope of the log-log error curve over the finest third of the grids, per method (NaN when a grid
    reproduces the reference exactly, as in a quick run whose finest grid is the reference grid)."""
    idx = slice(int(np.floor(2 * len(sizes) / 3)) - 1, None)  # MATLAB: floor(2*numel(np)/3):numel(np)
    orders = {}
    with np.errstate(divide="ignore"):
        for m, label in enumerate(LABELS):
            p = np.polyfit(np.log(sizes[idx]), np.log(bench[idx, m]), 1)
            orders[label] = -p[0]
    return orders


def plot(mu_traj: np.ndarray, sizes: np.ndarray, bench: np.ndarray, timer: Timer | None = None) -> dict:
    """The trajectory and the convergence plot as Plotly figures, keyed as the comparison case names them."""
    import plotly.graph_objects as go

    from spynach.plotting import plot_time_series

    timer = Timer() if timer is None else timer
    with timer("plot"):
        time_axis = np.linspace(0, DURATION, mu_traj.shape[1])
        traj = plot_time_series(
            time_axis,
            mu_traj.real,
            names=["x-magnetisation", "y-magnetisation", "z-magnetisation"],
            xlabel="time, seconds",
            ylabel="magnetisation, a.u.",
            title="Radiation-damped chirped oscillator, RKMK-DP8 reference",
        )
        conv = go.Figure()
        for m, label in enumerate(LABELS):
            conv.add_trace(go.Scatter(x=sizes, y=bench[:, m], mode="lines+markers", name=label))
        conv.update_layout(
            title="Convergence of the product quadratures",
            xaxis={"title": "number of points in the time grid", "type": "log"},
            yaxis={"title": "||difference|| / ||exact||", "type": "log", "range": [-7, 0]},
            template="plotly_white",
            height=420,
        )
    return {"trajectory": traj, "convergence": conv}


def compare_with_spinach(path: Path, mu_traj: np.ndarray, bench: np.ndarray) -> None:
    from scipy.io import loadmat

    from spynach.oracle import compare

    mat = loadmat(path)
    for name, ours in (("mu_traj", mu_traj), ("bench", bench)):
        theirs = np.asarray(mat[name])
        if theirs.shape != ours.shape:
            raise SystemExit(f"{name}: shape {theirs.shape} in {path}, {ours.shape} here; run both at the same sizes")
        print(f"{name}: {compare(ours, theirs, rtol=1e-6, atol=0, name=name)}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--np-ref", type=int, default=2**12, help="points of the reference trajectory (Spinach: 4096)")
    ap.add_argument("--grids", type=int, default=10, help="number of grids between 2^8 and 2^11 points (Spinach: 10)")
    ap.add_argument("--out", type=Path, default=Path(__file__).with_name("out"))
    ap.add_argument("--compare", type=Path, default=None)
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args(argv)

    timer = Timer()
    mu_traj, sizes, bench = simulate(args.np_ref, args.grids, timer)

    print("\nEmpirical convergence orders (slope of log-log error curve):")
    print("-" * 60)
    for label, order in convergence_orders(sizes, bench).items():
        print(f"{label:10s} : {order:6.3f}")
    print("-" * 60)

    args.out.mkdir(parents=True, exist_ok=True)
    np.save(args.out / "product_quadratures_2_traj.npy", mu_traj)
    np.save(args.out / "product_quadratures_2_bench.npy", bench)
    np.save(args.out / "product_quadratures_2_grids.npy", sizes)
    print(f"trajectory and benchmark written to {args.out}/")

    if args.compare is not None:
        compare_with_spinach(args.compare, mu_traj, bench)

    if not args.no_plot:
        figs = plot(mu_traj, sizes, bench, timer)
        with open(args.out / "product_quadratures_2.html", "w", encoding="utf-8") as fh:
            fh.write(figs["trajectory"].to_html(full_html=True, include_plotlyjs="cdn"))
            fh.write(figs["convergence"].to_html(full_html=False, include_plotlyjs=False))
        print(f"figure written to {args.out / 'product_quadratures_2.html'}")


if __name__ == "__main__":
    main(sys.argv[1:])
