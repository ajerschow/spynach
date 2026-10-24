"""Spectral post-processing shared by the simulators: field modulation (EasySpin's fieldmod)."""

from __future__ import annotations

import numpy as np
from scipy.special import jv


def fieldmod(B, spc, mod_amp: float, harmonic: int = 1) -> np.ndarray:
    """Pseudo-modulated spectrum: the effect of field modulation of peak-to-peak amplitude ``mod_amp`` (mT) on an
    absorption spectrum ``spc`` over the field axis ``B`` (mT), detected at ``harmonic`` (1, 2, ...)."""
    B = np.asarray(B, dtype=float).ravel()
    spc = np.asarray(spc, dtype=float).ravel()
    if harmonic != int(harmonic) or harmonic < 0:
        raise ValueError("Harmonic must be a positive integer (1, 2, 3, etc)!")
    if mod_amp <= 0:
        raise ValueError("Modulation amplitude (3rd argument) must be positive.")
    n = B.size
    if spc.size != n:
        raise ValueError("x and y must have the same length!")
    dx = B[1] - B[0]
    ampl = mod_amp / 2 / dx
    NN = 2 * n + 1
    ffty = np.fft.fft(spc, NN)
    ffty[int(np.ceil(NN / 2)) :] = 0
    S = np.arange(NN) / NN
    y_mod = np.fft.ifft(ffty * jv(harmonic, 2 * np.pi * ampl * S))[:n]
    y_mod = (1j) ** harmonic * y_mod
    return np.real(y_mod)
