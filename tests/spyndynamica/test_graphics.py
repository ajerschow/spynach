"""Three-dimensional pictures: the geometry of the scenes against SpinDynamica's graphics objects."""

import numpy as np
import pytest

import spyndynamica as sd
from tests.spyndynamica import reference

TOL = dict(rtol=0, atol=1e-9)
MATRIX = np.array([[1, 0.2, 0], [0.2, -2, 0.5], [0, 0.5, 3]])
ANGLES = (0.3, 0.7, -0.4)


def stored(key):
    return reference.get("14_graphics", key)


def polygon_set(polygons):
    return sorted(tuple(sorted(map(tuple, np.round(np.asarray(p), 9)))) for p in polygons)


def arrows(scene):
    return np.array([[arrow.begin, arrow.end] for arrow in scene.arrows])


OVALOIDS = {
    "points": (MATRIX, 1, False),
    "points_scaled_and_moved": (MATRIX, 0.5, False),
    "points_isotropic_automatic": (MATRIX, 1, True),
    "points_isotropic_number": (MATRIX, 1, 1.5),
    "points_spherical_tensor": (sd.cartesian_to_spherical(2, MATRIX), 1, False),
    "points_spherical_tensor_isotropic": (sd.cartesian_to_spherical(2, MATRIX), 1, 1.5),
}


@pytest.mark.parametrize("name", list(OVALOIDS))
def test_ovaloid_radius(name):
    tensor, scale, isotropic = OVALOIDS[name]
    # every point of SpinDynamica's surface lies at the port's radius for its direction
    points = stored(f"ovaloid/{name}")
    distance = np.linalg.norm(points, axis=1)
    n = points / distance[:, None]
    radius = scale * sd.ovaloid_radius(tensor, np.arccos(n[:, 2]), np.arctan2(n[:, 1], n[:, 0]), isotropic=isotropic)
    assert np.allclose(np.abs(radius), distance, **TOL)


def test_ovaloid_scene():
    traceless = (MATRIX + MATRIX.T) / 2 - np.trace(MATRIX) / 3 * np.eye(3)
    # independent: along a principal axis the radius is the principal value
    values, axes = np.linalg.eigh(traceless)
    for value, axis in zip(values, axes.T):
        theta, phi = np.arccos(axis[2]), np.arctan2(axis[1], axis[0])
        assert np.isclose(sd.ovaloid_radius(MATRIX, theta, phi), value)
        assert np.isclose(sd.ovaloid_radius(MATRIX, theta, phi, isotropic=True), value + np.trace(MATRIX) / 3)
    assert np.allclose(stored("ovaloid/translation"), [[1, 2, 3]])
    scene = sd.ovaloid(MATRIX, [1, 2, 3], scale=0.5, points=(9, 17))
    (surface,) = scene.surfaces
    assert surface.vertices.shape == (9 * 17, 3) and len(surface.faces) == 8 * 16
    offsets = surface.vertices - [1, 2, 3]
    assert np.allclose(np.linalg.norm(offsets, axis=1), np.abs(surface.values), **TOL)
    assert np.isclose(surface.values[0], 0.5 * traceless[2, 2]) and surface.colors == ("orange", "blue")
    assert (surface.values > 0).any() and (surface.values < 0).any()
    assert sd.ovaloid(MATRIX, colors="red").surfaces[0].colors == ("red", "red")
    with pytest.raises(ValueError, match="3 x 3"):
        sd.ovaloid_radius(np.eye(2), 0, 0)


def test_axes_arrows_and_euler_picture():
    g = "axes_and_shapes"
    cases = {
        "axes_default": sd.axes_3d(),
        "axes_from_vectors": sd.axes_3d([[1, 0, 0], [1, 1, 1]], [1, 2, 3], length=2),
        "axes_from_euler_angles": sd.axes_3d(euler=ANGLES),
        "axes_from_euler_angles_and_system": sd.axes_3d([[1, 1, 1]], euler=ANGLES),
        "visualize_euler": sd.visualize_euler(ANGLES),
        "arrow": sd.arrow_3d([1, 0, 0], [1, 2, 3]),
        "sphere_and_axes_arrows": sd.sphere_and_axes(),
    }
    for name, scene in cases.items():
        assert np.allclose(arrows(scene), stored(f"{g}/{name}"), **TOL), name
    # independent: the thick axes are the system whose Euler angles relative to the laboratory are the angles
    thick = arrows(sd.visualize_euler(ANGLES))[3:, 1]
    assert np.allclose(sd.axes_to_euler(thick), ANGLES, **TOL)
    assert [arrow.label for arrow in sd.axes_3d().arrows] == ["x", "y", "z"]
    assert np.allclose(arrows(sd.arrow_3d([1, 2, 3])), [[[0, 0, 0], [1, 2, 3]]])


def test_shapes():
    g = "axes_and_shapes"
    shape = sd.l_shape()
    assert np.allclose([line.points for line in shape.lines], stored(f"{g}/l_shape_edges"), **TOL)
    assert polygon_set(shape.surfaces[0].polygons()) == polygon_set(stored(f"{g}/l_shape_polygons"))
    other = sd.l_shape((1, 2, 0.5, 0.25, 1 / 3))
    assert polygon_set(other.surfaces[0].polygons()) == polygon_set(stored(f"{g}/l_shape_polygons_other_dimensions"))
    for name, scene in [
        ("rotor_polygons", sd.rotor(points=8)),
        ("rotor_polygons_other_dimensions", sd.rotor(0.5, 3, 6)),
    ]:
        mine = [polygon for surface in scene.surfaces for polygon in surface.polygons()]
        assert polygon_set(mine) == polygon_set(stored(f"{g}/{name}")), name
    stage = sd.sphere_and_axes()
    assert np.allclose([label.position for label in stage.labels], stored(f"{g}/sphere_and_axes_labels"), **TOL)
    assert np.allclose([s.vertices for s in stage.surfaces[:3]], stored(f"{g}/sphere_and_axes_circles"), **TOL)
    assert np.allclose(np.linalg.norm(stage.surfaces[3].vertices, axis=1), 1)


def test_scenes_combine_move_and_rotate():
    scene = sd.l_shape() + sd.axes_3d() + sd.sphere_and_axes()
    assert (len(scene.surfaces), len(scene.lines), len(scene.arrows), len(scene.labels)) == (5, 2, 6, 3)
    moved = sd.l_shape().translated([1, 2, 3])
    assert np.allclose(moved.surfaces[0].vertices, sd.l_shape().surfaces[0].vertices + [1, 2, 3])
    assert np.allclose(moved.lines[0].points, sd.l_shape().lines[0].points + [1, 2, 3])
    turned = sd.axes_3d().rotated_euler(ANGLES)
    assert np.allclose(arrows(turned), arrows(sd.axes_3d(euler=ANGLES)), **TOL)
    about = sd.axes_3d().rotated_polar(0.3, [1, 1, 1])
    assert np.allclose(arrows(about)[:, 1], sd.rotate_polar(np.eye(3), 0.3, [1, 1, 1]), **TOL)
    frame = sd.axis_system([1, 1, 1])
    in_frame = sd.arrow_3d([1, 2, 3]).rotated_euler(ANGLES, frame)
    assert np.allclose(arrows(in_frame)[0, 1], sd.rotate_euler([1, 2, 3], ANGLES, frame), **TOL)


def test_figure():
    pytest.importorskip("plotly")
    scene = sd.ovaloid(MATRIX, points=(9, 17)) + sd.axes_3d(length=4) + sd.l_shape().translated([3, 0, 0]) + sd.rotor()
    figure = scene.figure(width=400)
    kinds = [trace.type for trace in figure.data]
    assert kinds.count("mesh3d") == 4 and kinds.count("cone") == 3 and "scatter3d" in kinds
    assert figure.layout.width == 400
    assert figure.data[0].intensity is not None and len(figure.data[0].i) == 2 * 8 * 16
