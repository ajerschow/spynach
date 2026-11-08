"""Comparison case: Lie-group product quadratures on the radiation-damped chirped oscillator,
examples/fundamentals/product_quadratures_2.py with timed stages.

Stage labels must match product_quadratures_2_spinach.m: reference, benchmark, plot.
"""

from __future__ import annotations

from comparison.run_spynach import load_example

example = load_example("fundamentals/product_quadratures_2")


def run(timer, np_ref: int = 2**12, grids: int = 10, gpu: bool = False) -> tuple[dict, dict]:
    """Returns (data, figs): the reference trajectory, grid sizes and error table; the two figures."""
    if gpu:
        raise NotImplementedError(
            "sys['enable'] 'gpu' is not available yet: the GPU backend is planned, see porting/gpu-notes.md"
        )
    mu_traj, sizes, bench = example.simulate(np_ref, grids, timer)
    data = {"mu_traj": mu_traj, "np": sizes.astype(float), "bench": bench}
    return data, example.plot(mu_traj, sizes, bench, timer)
