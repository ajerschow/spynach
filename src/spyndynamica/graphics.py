"""Three-dimensional pictures: ovaloids of rank-2 tensors, axis systems, and the shapes used to show rotations.

Every routine returns a :class:`Scene`, plain geometry (surfaces, lines, arrows, labels) held in NumPy arrays.
Scenes add up, and they can be moved and rotated like vectors. ``scene.figure()`` turns one into a Plotly
figure; Plotly is needed for that step only and is not a dependency of the package.

Author: Alexej Jerschow.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from spyndynamica.rotations import _as_system, _euler, _vector, axis_system, rotate_euler, rotate_polar
from spyndynamica.tensors import spherical_to_cartesian


@dataclass(frozen=True)
class Surface:
    """A surface as polygons: ``faces`` holds, for every polygon, the numbers of its ``vertices``. ``values``
    (one per vertex) pick between the two ``colors``: the first for positive values, the second for negative."""

    vertices: np.ndarray
    faces: tuple[tuple[int, ...], ...]
    colors: tuple[str, str] = ("lightgray", "lightgray")
    opacity: float = 1.0
    values: np.ndarray | None = None

    def polygons(self) -> list[np.ndarray]:
        """The corner points of every polygon."""
        return [self.vertices[list(face)] for face in self.faces]


@dataclass(frozen=True)
class Line:
    points: np.ndarray
    color: str = "black"
    width: float = 2.0


@dataclass(frozen=True)
class Arrow:
    begin: np.ndarray
    end: np.ndarray
    label: str | None = None
    color: str = "black"
    width: float = 3.0


@dataclass(frozen=True)
class Label:
    text: str
    position: np.ndarray


@dataclass(frozen=True)
class Scene:
    """Geometry to draw: surfaces, lines, arrows and labels. ``a + b`` shows two scenes together."""

    surfaces: tuple[Surface, ...] = field(default_factory=tuple)
    lines: tuple[Line, ...] = field(default_factory=tuple)
    arrows: tuple[Arrow, ...] = field(default_factory=tuple)
    labels: tuple[Label, ...] = field(default_factory=tuple)

    def __add__(self, other: Scene) -> Scene:
        if not isinstance(other, Scene):
            return NotImplemented
        merged = (getattr(self, name) + getattr(other, name) for name in ("surfaces", "lines", "arrows", "labels"))
        return Scene(*merged)

    def _map(self, function) -> Scene:
        return Scene(
            tuple(replace(s, vertices=function(s.vertices)) for s in self.surfaces),
            tuple(replace(line, points=function(line.points)) for line in self.lines),
            tuple(replace(a, begin=function(a.begin), end=function(a.end)) for a in self.arrows),
            tuple(replace(label, position=function(label.position)) for label in self.labels),
        )

    def translated(self, shift: Sequence[Any]) -> Scene:
        """The scene moved by a vector."""
        shift = _vector(shift)
        return self._map(lambda points: points + shift)

    def rotated_euler(self, angles: Sequence[Any], axes: Any = None) -> Scene:
        """The scene rotated as ``rotate_euler`` rotates vectors."""
        return self._map(lambda points: rotate_euler(points, angles, axes))

    def rotated_polar(self, xi: Any, direction: Sequence[Any]) -> Scene:
        """The scene rotated through ``ξ`` about an axis, as ``rotate_polar`` rotates vectors."""
        return self._map(lambda points: rotate_polar(points, xi, direction))

    def figure(self, **layout: Any):
        """A Plotly figure of the scene (needs Plotly). Keyword arguments update the layout."""
        try:
            import plotly.graph_objects as go
        except ImportError as error:  # pragma: no cover - depends on the installation
            raise ImportError("drawing a scene needs Plotly: pip install plotly") from error

        def path(points: Any, color: str, width: float):
            x, y, z = np.asarray(points).T
            return go.Scatter3d(x=x, y=y, z=z, mode="lines", line=dict(color=color, width=width), hoverinfo="skip")

        figure = go.Figure()
        for surface in self.surfaces:
            triangles = [(face[0], face[k], face[k + 1]) for face in surface.faces for k in range(1, len(face) - 1)]
            i, j, k = (list(column) for column in zip(*triangles))
            x, y, z = surface.vertices.T
            style: dict[str, Any] = dict(color=surface.colors[0])
            if surface.values is not None:
                scale = [[0, surface.colors[1]], [1, surface.colors[0]]]
                style = dict(intensity=np.sign(surface.values), colorscale=scale, cmin=-1, cmax=1, showscale=False)
            mesh = go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, opacity=surface.opacity, hoverinfo="skip", **style)
            figure.add_trace(mesh)
        for line in self.lines:
            figure.add_trace(path(line.points, line.color, line.width))
        for arrow in self.arrows:
            figure.add_trace(path([arrow.begin, arrow.end], arrow.color, arrow.width))
            direction = arrow.end - arrow.begin
            head = 0.12 * direction / (float(np.linalg.norm(direction)) or 1.0)
            tip = {name: [value] for name, value in zip("xyzuvw", [*arrow.end, *head])}
            colors = [[0, arrow.color], [1, arrow.color]]
            cone = go.Cone(anchor="tip", sizemode="absolute", sizeref=0.12, showscale=False, colorscale=colors, **tip)
            figure.add_trace(cone.update(hoverinfo="skip"))
        texts = list(self.labels) + [Label(a.label, a.begin + 1.1 * (a.end - a.begin)) for a in self.arrows if a.label]
        if texts:
            x, y, z = np.array([label.position for label in texts]).T
            words = [label.text for label in texts]
            figure.add_trace(go.Scatter3d(x=x, y=y, z=z, mode="text", text=words, hoverinfo="skip"))
        hidden = dict(visible=False)
        scene = dict(aspectmode="data", xaxis=hidden, yaxis=hidden, zaxis=hidden)
        figure.update_layout(showlegend=False, scene=scene, margin=dict(l=0, r=0, t=0, b=0))
        figure.update_layout(**layout)
        return figure


# ====================================================================================== ovaloid
def _rank_2(tensor: Any) -> tuple[np.ndarray, float]:
    """The symmetric traceless part of a tensor as a 3 × 3 matrix, and the third of its trace."""
    array = np.asarray(tensor, dtype=complex)
    if array.shape == (5,):
        return spherical_to_cartesian(array).real, 0.0
    if array.shape != (3, 3) or np.abs(array.imag).max() > 0:
        raise ValueError("the tensor is a real 3 x 3 matrix or the five components of a rank-2 spherical tensor.")
    matrix = array.real
    iso = float(np.trace(matrix)) / 3
    return (matrix + matrix.T) / 2 - iso * np.eye(3), iso


def ovaloid_radius(tensor: Any, theta: Any, phi: Any, *, isotropic: Any = False) -> np.ndarray:
    """The radius of the ovaloid of a rank-2 tensor in the direction with the polar angles ``(θ, φ)``:
    ``n·A·n`` for the symmetric traceless part ``A`` of the tensor and the unit vector ``n``, plus the isotropic
    part when one is asked for (see :func:`ovaloid`). The radius has a sign."""
    matrix, iso = _rank_2(tensor)
    offset = iso if isotropic is True or isotropic == "auto" else (0.0 if isotropic is False else float(isotropic))
    theta, phi = np.broadcast_arrays(np.asarray(theta, dtype=float), np.asarray(phi, dtype=float))
    n = np.stack([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)], axis=-1)
    return np.einsum("...a,ab,...b->...", n, matrix, n) + offset


def ovaloid(
    tensor: Any,
    center: Sequence[Any] = (0.0, 0.0, 0.0),
    *,
    scale: float = 1.0,
    isotropic: Any = False,
    colors: Sequence[str] | str = ("orange", "blue"),
    opacity: float = 0.5,
    points: tuple[int, int] = (41, 81),
) -> Scene:
    """The ovaloid of a rank-2 tensor (SpinDynamica's ``Ovaloid``): the surface whose distance from ``center``
    in every direction is the value of the tensor along that direction, times ``scale``.

    ``tensor`` is a real 3 × 3 matrix, of which the symmetric traceless part is taken, or the five components
    of a rank-2 spherical tensor. Positive lobes have the first of ``colors`` and negative lobes the second.
    ``isotropic`` adds an isotropic part: a number, or ``True`` for the third of the trace of the matrix.
    ``points`` are the numbers of polar and azimuthal angles of the mesh. The representation is described by
    L. Mueller, doi:10.1002/mrc.4793.
    """
    n_theta, n_phi = points
    theta, phi = np.meshgrid(np.linspace(0, np.pi, n_theta), np.linspace(0, 2 * np.pi, n_phi), indexing="ij")
    radius = scale * ovaloid_radius(tensor, theta, phi, isotropic=isotropic)
    n = np.stack([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)], axis=-1)
    vertices = (radius[..., None] * n).reshape(-1, 3) + _vector(center)
    faces = []
    for a in range(n_theta - 1):
        for b in range(n_phi - 1):
            corner = a * n_phi + b
            faces.append((corner, corner + n_phi, corner + n_phi + 1, corner + 1))
    pair = (colors, colors) if isinstance(colors, str) else tuple(colors)
    surface = Surface(vertices, tuple(faces), pair, opacity, radius.reshape(-1))  # type: ignore[arg-type]
    return Scene(surfaces=(surface,))


# ====================================================================================== arrows and axes
def arrow_3d(
    begin: Sequence[Any], end: Sequence[Any] | None = None, *, label: str | None = None, **style: Any
) -> Scene:
    """An arrow from ``begin`` to ``end``, or from the origin to the only point given (``Arrow3D``)."""
    if end is None:
        begin, end = (0.0, 0.0, 0.0), begin
    return Scene(arrows=(Arrow(_vector(begin), _vector(end), label, **style),))


def axes_3d(
    axes: Any = None,
    origin: Sequence[Any] = (0.0, 0.0, 0.0),
    *,
    euler: Sequence[Any] | None = None,
    length: float = 1.0,
    labels: Sequence[str | None] = ("x", "y", "z"),
    **style: Any,
) -> Scene:
    """Three arrows along the axes of an axis system, starting at ``origin`` (SpinDynamica's ``Axes3D``).

    ``axes`` is given as for ``axis_system`` (default: the laboratory axes). With ``euler = (α, β, γ)`` every
    axis is rotated by ``Rz(α) Ry(β) Rz(γ)`` about the laboratory axes first.
    """
    system = _as_system(axes)
    if euler is not None:
        system = rotate_euler(system, euler)
    start = _vector(origin)
    return Scene(
        arrows=tuple(Arrow(start, start + length * axis, label, **style) for axis, label in zip(system, labels))
    )


def visualize_euler(angles: Sequence[Any]) -> Scene:
    """The laboratory axes (thin) and the axis system whose Euler angles relative to them are ``angles``
    (thick): the system ``P`` with ``axes_to_euler(P) == angles`` (SpinDynamica's ``VisualizeEuler``)."""
    rotated = _euler(*(float(a) for a in angles))  # the rows are the axes of the system
    return axes_3d(width=2.0) + axes_3d(rotated, width=6.0)


# ====================================================================================== shapes
def l_shape(dimensions: Sequence[float] = (0.5, 0.8, 0.2, 0.2, 0.2), color: str = "lightgray") -> Scene:
    """A block in the shape of the letter L, which shows the sense of a rotation (SpinDynamica's ``LShape``).

    ``dimensions = (a, b, c, d, e)``: the foot is ``a`` long (along x) and ``e`` high, the stem ``b`` high (along
    z) and ``c`` wide, and the block is ``d`` thick (along y).
    """
    a, b, c, d, e = (float(v) for v in dimensions)
    front = np.array([[a, d, 0], [0, d, 0], [0, d, b], [c, d, b], [c, d, e], [a, d, e]])
    back = front - [0, d, 0]
    inner = np.array([[c, d, 0], [0, d, e]])  # where the foot and the stem meet, on the front face
    vertices = np.vstack([front, back, inner, inner - [0, d, 0]])
    f, k, m = 0, 6, 12  # offsets of the front outline, of the back outline, and of the inner points
    faces = [
        (f + 0, m + 0, f + 4, f + 5), (m + 0, f + 1, m + 1, f + 4), (m + 1, f + 2, f + 3, f + 4),
        (k + 0, m + 2, k + 4, k + 5), (m + 2, k + 1, m + 3, k + 4), (m + 3, k + 2, k + 3, k + 4),
        (f + 0, f + 1, k + 1, k + 0), (f + 1, f + 2, k + 2, k + 1), (f + 3, f + 2, k + 2, k + 3),
        (k + 4, f + 4, f + 3, k + 3), (f + 5, f + 4, k + 4, k + 5), (f + 0, f + 5, k + 5, k + 0),
    ]  # fmt: skip
    outline = lambda face: Line(np.vstack([face, face[:1]]), width=2.0)  # noqa: E731
    return Scene(surfaces=(Surface(vertices, tuple(faces), (color, color)),), lines=(outline(front), outline(back)))


def rotor(radius: float = 0.3, length: float = 1.0, points: int = 100) -> Scene:
    """A sample rotor: a cylinder along z, half dark and half light so that its rotation shows
    (SpinDynamica's ``Rotor``). ``points`` is the number of corners of its circles."""
    half = points // 2
    surfaces = []
    for start, color in ((np.pi / points, "dimgray"), (np.pi + np.pi / points, "gainsboro")):
        angles = start + np.arange(half + 1) * (2 * np.pi / points)
        circle = np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=1)
        top = np.column_stack([circle, np.full(half + 1, length / 2)])
        bottom = np.column_stack([circle, np.full(half + 1, -length / 2)])
        count = half + 1
        faces = [tuple(range(count)), tuple(range(count, 2 * count))]
        faces += [(n, n + 1, count + n + 1, count + n) for n in range(half)]
        surfaces.append(Surface(np.vstack([top, bottom]), tuple(faces), (color, color)))
    return Scene(surfaces=tuple(surfaces))


def sphere_and_axes(points: int = 100) -> Scene:
    """A transparent unit sphere with its three great circles and labelled axes, as a stage for trajectories
    (SpinDynamica's ``SphereAndAxes``)."""
    angles = np.arange(points + 1) * (2 * np.pi / points)
    cos, sin, zero = np.cos(angles), np.sin(angles), np.zeros(points + 1)
    circles = [np.column_stack(c) for c in ((zero, cos, sin), (cos, sin, zero), (cos, zero, sin))]
    disks = tuple(Surface(circle, (tuple(range(points + 1)),), ("gray", "gray"), 0.1) for circle in circles)
    theta, phi = np.meshgrid(np.linspace(0, np.pi, 25), np.linspace(0, 2 * np.pi, 49), indexing="ij")
    vertices = np.stack([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)], axis=-1).reshape(
        -1, 3
    )
    faces = tuple(
        (a * 49 + b, (a + 1) * 49 + b, (a + 1) * 49 + b + 1, a * 49 + b + 1) for a in range(24) for b in range(48)
    )
    sphere = Surface(vertices, faces, ("gray", "gray"), 0.1)
    arrows = tuple(Arrow(np.zeros(3), axis) for axis in axis_system())
    labels = tuple(Label(text, 1.1 * axis) for text, axis in zip("xyz", axis_system()))
    return Scene(surfaces=(*disks, sphere), arrows=arrows, labels=labels)


__all__ = [
    "Arrow",
    "Label",
    "Line",
    "Scene",
    "Surface",
    "arrow_3d",
    "axes_3d",
    "l_shape",
    "ovaloid",
    "ovaloid_radius",
    "rotor",
    "sphere_and_axes",
    "visualize_euler",
]
