"""Gradient, diffusion and flow attenuation (1998 Sections 3 and 4 and the Appendix).

``γ*_i = Σ_nuclei p_i γ`` per period (Eq. 22), ``K_q = Σ_{i≤q} γ*_i h_i`` the k-vector after the
gradient of period ``q`` (Eq. 28/37), with the gradient a delta function at the start of the
period. Attenuation at detection from the final k-vector: the program uses the envelopes
of Eq. 31 and 34, ``|sinc u|`` for ``|u| < π/2`` else ``1/|u|`` along z (``u = k_z z_max``, slab of
half-length ``z_max``) and ``|2J₁(u)/u|`` for ``|u| < 2.44`` else ``2√(2/π)|u|^(-3/2)`` transverse
(``u = |k_xy| x_max``, cylinder of radius ``x_max``); ``envelope=False`` gives the exact forms
(Eq. 30, 32). Deviation from CCCP.m: its small-argument transverse branch is unreachable
(a dead condition in the code); this port implements Eq. 34 as printed. Diffusion:
``F = exp[-D Σ_q |K_q|² Δ_q]`` (Eq. 41/42) with ``Δ_q`` the duration of period ``q``. Flow along z
with a uniform velocity distribution over ``±v_max``: ``F = sinc(v_max Σ_i γ*_i h_iz t_i)``
(Eq. 46/48), ``t_i`` the start time of period ``i``.

Author: Alexej Jerschow.
"""

from __future__ import annotations

import numpy as np
from scipy.special import j1

from ctps.model import Sample, Sequence


def effective_gamma(sequence: Sequence, per_nucleus: dict[str, np.ndarray]) -> np.ndarray:
    """``(k, n_periods)`` effective gyromagnetic ratio ``γ*`` (rad/s/T)."""
    gamma_star = None
    for nucleus in sequence.nuclei:
        term = per_nucleus[nucleus.name].astype(float) * nucleus.gamma
        gamma_star = term if gamma_star is None else gamma_star + term
    return gamma_star


def k_vectors(gamma_star: np.ndarray, areas: np.ndarray) -> np.ndarray:
    """``(k, n_periods, 3)`` cumulative k-vectors ``K_q`` in rad/m."""
    # GPU: broadcast multiply and cumulative sum along the period axis
    return np.cumsum(gamma_star[:, :, None] * areas[None, :, :], axis=1)


def _sinc_envelope(u: np.ndarray, envelope: bool) -> np.ndarray:
    u = np.abs(u)
    exact = np.abs(np.sinc(u / np.pi))  # numpy's sinc is sin(pi x) / (pi x)
    if not envelope:
        return exact
    return np.where(u < np.pi / 2, exact, 1 / np.maximum(u, 1e-300))


def _bessel_envelope(u: np.ndarray, envelope: bool) -> np.ndarray:
    u = np.abs(u)
    safe = np.where(u > 0, u, 1.0)
    exact = np.where(u > 0, np.abs(2 * j1(safe) / safe), 1.0)
    if not envelope:
        return exact
    return np.where(u < 2.44, exact, 2 * np.sqrt(2 / np.pi) * safe ** (-1.5))


def gradient_attenuation(k_final: np.ndarray, sample: Sample, envelope: bool = True) -> np.ndarray:
    """Attenuation ``f_z · f_xy`` from the k-vector at detection, ``(k, 3)`` -> ``(k,)``."""
    f_z = _sinc_envelope(k_final[:, 2] * sample.z_max, envelope)
    f_xy = _bessel_envelope(np.hypot(k_final[:, 0], k_final[:, 1]) * sample.x_max, envelope)
    return f_z * f_xy


def diffusion_attenuation(K: np.ndarray, durations: np.ndarray, diffusion: float) -> np.ndarray:
    """``exp[-D Σ_q |K_q|² Δ_q]`` over periods, ``K`` as from :func:`k_vectors`."""
    if diffusion == 0:
        return np.ones(len(K))
    moment = np.einsum("kqa,kqa,q->k", K, K, durations)  # GPU: fused square-sum
    return np.exp(-diffusion * moment)


def flow_attenuation(gamma_star: np.ndarray, areas: np.ndarray, start_times: np.ndarray, v_max: float) -> np.ndarray:
    """``sinc(ψ(v_max))`` with ``ψ(v) = v Σ_i γ*_i h_iz t_i`` (1998 Eq. 46/48), flow along z."""
    if v_max == 0:
        return np.ones(len(gamma_star))
    psi = v_max * (gamma_star * (areas[None, :, 2] * start_times[None, :])).sum(axis=1)
    return np.abs(np.sinc(psi / np.pi))


def kspace_radius(k_final: np.ndarray, gamma: float) -> np.ndarray:
    """DOTCOPS dephasing measure ``D = |k| / γ`` (MRM 81, 2209 (2019) Eq. 4), geometry-free, in T·s/m."""
    return np.linalg.norm(k_final, axis=1) / gamma
