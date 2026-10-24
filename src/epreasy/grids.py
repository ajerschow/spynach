"""Spherical grids for powder integration: EasySpin's ``sphgrid`` (the SOPHE grid) and ``gridparam``.

Orientations are polar angles ``(phi, theta)`` on the unit sphere, with the
region covered chosen by the point group of the Hamiltonian (one octant for
``D2h``, a quarter meridian for ``Dinfh``, the whole sphere for ``C1``, ...)
and ``GridSize`` knots along a quarter meridian. The triangulation of the grid
is what the powder spectrum is projected from.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .rotations import ang2vec

_GROUPS = ["C1", "Ci", "C2h", "S6", "C4h", "C6h", "D2h", "Th", "D3d", "D4h", "Oh", "D6h", "Dinfh", "O3"]
_MAX_PHI = np.array([2, 2, 1, 2 / 3, 1 / 2, 1 / 3, 1 / 2, 1 / 2, 1 / 3, 1 / 4, 1 / 4, 1 / 6, 0, 0]) * np.pi
_N_OCTANTS = [8, 4, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 0, -1]
_CLOSED_PHI = [False] * 6 + [True] * 8


def gridparam(point_group: str) -> tuple[float, bool, int]:
    """``(maxPhi, closedPhi, nOctants)`` of the grid region for a point group (Schoenflies symbol)."""
    if point_group not in _GROUPS:
        raise ValueError(f"Unsupported symmetry group {point_group}!")
    k = _GROUPS.index(point_group)
    return float(_MAX_PHI[k]), _CLOSED_PHI[k], _N_OCTANTS[k]


@dataclass(frozen=True)
class SphGrid:
    symmetry: str | int
    grid_size: int
    closed_phi: bool
    n_octants: int
    max_phi: float
    phi: np.ndarray
    theta: np.ndarray
    vecs: np.ndarray  # (3, n)
    weights: np.ndarray  # sum 4 pi
    tri_idx: np.ndarray | None = None  # (nTri, 3), 1-based knot indices as in EasySpin
    tri_areas: np.ndarray | None = None  # sum 4 pi

    @property
    def n(self) -> int:
        return self.weights.size


def _sphgrid_(n_octants: int, max_phi: float, grid_size: int, closed_phi: bool):
    dtheta = (np.pi / 2) / (grid_size - 1)
    if n_octants > 0:
        n_oct = 4 if n_octants == 8 else n_octants
        n_ori = grid_size + n_oct * grid_size * (grid_size - 1) // 2
        phi = np.zeros(n_ori)
        theta = np.zeros(n_ori)
        weights = np.zeros(n_ori)
        sindth2 = np.sin(dtheta / 2)
        w0 = 0.5 if closed_phi else 1.0
        weights[0] = max_phi * (1 - np.cos(dtheta / 2))
        start = 1
        for i_slice in range(2, grid_size):
            n_phi = n_oct * (i_slice - 1) + 1
            d_phi = max_phi / (n_phi - 1)
            idx = start + np.arange(n_phi)
            weights[idx] = (
                2 * np.sin((i_slice - 1) * dtheta) * sindth2 * d_phi * np.concatenate([[w0], np.ones(n_phi - 2), [0.5]])
            )
            phi[idx] = np.linspace(0, max_phi, n_phi)
            theta[idx] = (i_slice - 1) * dtheta
            start += n_phi
        n_phi = n_oct * (grid_size - 1) + 1
        d_phi = max_phi / (n_phi - 1)
        idx = start + np.arange(n_phi)
        phi[idx] = np.linspace(0, max_phi, n_phi)
        theta[idx] = np.pi / 2
        weights[idx] = sindth2 * d_phi * np.concatenate([[w0], np.ones(n_phi - 2), [0.5]])
        if not closed_phi:
            rmv = np.cumsum(n_oct * np.arange(1, grid_size) + 1)  # 0-based: MATLAB's cumsum(...)+1 minus 1
            keep = np.ones(n_ori, bool)
            keep[rmv] = False
            phi, theta, weights = phi[keep], theta[keep], weights[keep]
        if n_octants == 8:
            idx = np.arange(theta.size - n_phi, -1, -1)
            weights[idx] = weights[idx] / 2
            phi = np.concatenate([phi, phi[idx]])
            theta = np.concatenate([theta, np.pi - theta[idx]])
            weights = np.concatenate([weights, weights[idx]])
        weights = 2 * (2 * np.pi / max_phi) * weights
    elif n_octants == 0:
        phi = np.zeros(grid_size)
        theta = np.linspace(0, np.pi / 2, grid_size)
        edges = np.concatenate([[0], np.arange(dtheta / 2, np.pi / 2 + 1e-12, dtheta), [np.pi / 2]])
        weights = -2 * (2 * np.pi) * np.diff(np.cos(edges))
    elif n_octants == -1:
        phi, theta, weights = np.zeros(1), np.zeros(1), np.array([4 * np.pi])
    else:
        raise ValueError(f"Unsupported value {n_octants} for nOctants.")
    return phi, theta, weights


def _triangle_areas(tri: np.ndarray, vecs: np.ndarray) -> np.ndarray:
    """Solid angles of spherical triangles (d'Huilier's formula); ``tri`` is 0-based here."""
    x1, x2, x3 = vecs[:, tri[:, 0]], vecs[:, tri[:, 1]], vecs[:, tri[:, 2]]
    a1 = np.arccos(np.clip(np.sum(x2 * x3, axis=0), -1, 1))
    a2 = np.arccos(np.clip(np.sum(x3 * x1, axis=0), -1, 1))
    a3 = np.arccos(np.clip(np.sum(x1 * x2, axis=0), -1, 1))
    s = (a1 + a2 + a3) / 2
    x = np.sqrt(np.abs(np.tan(s / 2) * np.tan((s - a1) / 2) * np.tan((s - a2) / 2) * np.tan((s - a3) / 2)))
    return 4 * np.arctan(x)


def _delaunay(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    from scipy.spatial import Delaunay

    return Delaunay(np.column_stack([x, y])).simplices


def sphgrid(symmetry, grid_size: int, *, closed_phi: bool = False, triangulate: bool = False) -> SphGrid:
    """The SOPHE grid for a point group (``'D2h'``, ``'Ci'``, ...) or a number of octants.

    ``grid_size`` is the number of knots per quarter meridian.

    ``triangulate=True`` also returns the triangulation (``tri_idx``, 1-based
    knot indices as EasySpin lists them, and ``tri_areas``). For the closed
    one-octant grid the triangulation is EasySpin's explicit one; the open
    grids are triangulated with ``scipy.spatial.Delaunay``, which may pick a
    different diagonal than MATLAB's ``delaunay`` where four knots are
    co-circular.
    """
    if isinstance(symmetry, str):
        max_phi, closed, n_octants = gridparam(symmetry)
    else:
        n_octants = int(symmetry)
        max_phi = {-1: 0.0, 0: 0.0, 1: np.pi / 2, 2: np.pi, 4: 2 * np.pi, 8: 2 * np.pi}.get(n_octants)
        if max_phi is None:
            raise ValueError("Wrong number of octants, must be -1, 0, 1, 2, 4, or 8.")
        closed = False
    if closed_phi:
        closed = True
    if closed and n_octants == 2:
        raise ValueError(f"Cannot provide closed-phi grid for this symmetry ({n_octants} octants).")
    if grid_size != int(grid_size) or grid_size < 2:
        raise ValueError("GridSize (2nd input argument) must be an integer >= 2.")
    grid_size = int(grid_size)
    phi, theta, weights = _sphgrid_(n_octants, max_phi, grid_size, closed)
    vecs = ang2vec(phi, theta)
    tri_idx = tri_areas = None
    if triangulate:
        if n_octants in (0, -1):
            tri = np.zeros((0, 3), dtype=np.intp)
            areas = np.zeros(0)
        elif n_octants == 1:
            if closed:
                a = np.arange(1, grid_size * (grid_size + 1) // 2 + 1)
                a = np.delete(a, (np.arange(2, grid_size + 2) * np.arange(1, grid_size + 1) // 2) - 1)
                k = np.arange(1, (grid_size - 1) * (grid_size - 2) // 2 + 1)
                b = a[k - 1]
                first = np.column_stack([np.arange(1, grid_size * (grid_size - 1) // 2 + 1), a, a + 1])
                second = np.column_stack([b, 2 * (b + 1) - k, b + 1])
                tri = np.vstack([first, second])
                areas = _triangle_areas(tri - 1, vecs)
            else:
                phx = np.concatenate([phi, max_phi * np.ones(grid_size - 1)])
                thx = np.concatenate([theta, np.pi / 2 * np.arange(1, grid_size) / (grid_size - 1)])
                tri = _delaunay(thx * np.cos(phx), thx * np.sin(phx)) + 1
                areas = _triangle_areas(tri - 1, ang2vec(phx, thx))
                rmv = np.arange(phi.size + 1, phx.size + 1)
                rpl = np.arange(0, grid_size - 1) * np.arange(1, grid_size) // 2 + 2
                for r, p in zip(rmv, rpl):
                    tri[tri == r] = p
        elif n_octants == 2:
            phx = np.concatenate([phi, max_phi * np.ones(grid_size - 1)])
            thx = np.concatenate([theta, np.pi / 2 * np.arange(1, grid_size) / (grid_size - 1)])
            tri = _delaunay(thx * np.cos(phx), thx * np.sin(phx)) + 1
            areas = _triangle_areas(tri - 1, ang2vec(phx, thx))
            rmv = np.arange(phi.size + 1, phx.size + 1)
            rpl = np.arange(0, grid_size - 1) ** 2 + np.arange(0, grid_size - 1) + 2
            for r, p in zip(rmv, rpl):
                tri[tri == r] = p
        elif n_octants == 4:
            tri = _delaunay(theta * np.cos(phi), theta * np.sin(phi)) + 1
            areas = _triangle_areas(tri - 1, vecs)
        else:
            phx, thx, _ = _sphgrid_(4, 2 * np.pi, grid_size, False)
            tri_upper = _delaunay(thx * np.cos(phx), thx * np.sin(phx)) + 1
            n_total = 4 * grid_size**2 - 8 * grid_size + 6
            tri_lower = n_total + 1 - tri_upper
            n_equator = 4 * (grid_size - 1)
            on_equator = tri_upper > thx.size - n_equator
            tri_lower[on_equator] = tri_upper[on_equator]
            tri = np.vstack([tri_upper, tri_lower])
            areas = _triangle_areas(tri - 1, vecs)
        if n_octants >= 1 and not closed and tri.shape[0]:
            keep = areas >= np.mean(areas) * 1e-3
            tri, areas = tri[keep], areas[keep]
        if tri.shape[0]:
            tri = np.sort(tri, axis=1)
            order = np.lexsort(tri.T[::-1])
            tri, areas = tri[order], areas[order]
            areas = areas / areas.sum() * 4 * np.pi
        tri_idx, tri_areas = tri.astype(np.uint32), areas
    return SphGrid(symmetry, grid_size, closed, n_octants, max_phi, phi, theta, vecs, weights, tri_idx, tri_areas)
