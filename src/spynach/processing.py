"""Signal processing (port of ``apodisation.m``).

Window functions are Spinach's, applied along every non-singleton
dimension of the FID with one specification per dimension.

Spinach authors, by original file (headers of Spinach 2.13; see docs/spinach-credits.md):
  apodisation.m: Ilya Kuprov
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

WINDOWS = ("none", "crisp", "exp", "gauss", "cos", "sin", "sqcos", "sqsin", "kaiser")


def apodisation(fid: np.ndarray, window_functions: Sequence[Sequence[Any] | None], fp_half: bool = True) -> np.ndarray:
    """Apply window functions to an FID, one per non-singleton dimension, halving the first points.

    ``window_functions`` is a list such as ``[("exp", 6)]`` for 1-D or
    ``[("sqcos",), ("sqcos",)]`` for 2-D (``None`` skips a dimension).
    Available: ``none``, ``crisp`` (cos^8), ``exp`` and ``gauss`` (with a
    decay parameter), ``cos``, ``sin``, ``sqcos``, ``sqsin``, ``kaiser``
    (with beta). With ``fp_half`` (default) the first point of every
    processed dimension is divided by two, as Spinach does before Fourier
    transforms.

    Spinach authors (apodisation.m): Ilya Kuprov.
    """
    fid = np.array(fid, dtype=complex)
    relevant = [axis for axis, n in enumerate(fid.shape) if n > 1]
    if len(window_functions) != len(relevant):
        raise ValueError("window_functions must have one element per non-singleton dimension of fid.")
    active = [(axis, spec) for axis, spec in zip(relevant, window_functions) if spec]
    for axis, _ in active:
        index = [slice(None)] * fid.ndim
        index[axis] = 0
        if fp_half:
            fid[tuple(index)] /= 2
    for axis, spec in active:
        name, params = spec[0], list(spec[1:])
        npts = fid.shape[axis]
        if name not in WINDOWS:
            raise ValueError(f"window function type {name!r} is not implemented.")
        needs_parameter = name in ("exp", "gauss", "kaiser")
        if needs_parameter != (len(params) == 1):
            raise ValueError(
                f"window function {name!r} takes {'one parameter' if needs_parameter else 'no parameters'}."
            )
        if name == "none":
            wf = np.ones(npts)
        elif name == "crisp":
            wf = np.cos(np.linspace(0, np.pi / 2, npts)) ** 8
        elif name == "exp":
            wf = np.exp(-params[0] * np.linspace(0, 1, npts))
        elif name == "gauss":
            wf = np.exp(-params[0] * np.linspace(0, 1, npts) ** 2)
        elif name == "cos":
            wf = np.cos(np.linspace(0, np.pi / 2, npts))
        elif name == "sin":
            wf = np.sin(np.linspace(0, np.pi, npts))
        elif name == "sqcos":
            wf = np.cos(np.linspace(0, np.pi / 2, npts)) ** 2
        elif name == "sqsin":
            wf = np.sin(np.linspace(0, np.pi, npts)) ** 2
        else:
            from scipy.signal.windows import kaiser

            wf = kaiser(npts, params[0], sym=True)
        shape = [1] * fid.ndim
        shape[axis] = npts
        fid = fid * wf.reshape(shape)
    return fid
