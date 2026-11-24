"""Axis systems and Euler angles of axes that contain symbols."""

import numpy as np
import pytest
import sympy as sp

import spyndynamica as sd
from tests.spyndynamica import reference

TOL = dict(rtol=0, atol=1e-11)
TH, PH, TH2, PH2, A, B, G = sp.symbols("theta phi theta2 phi2 alpha beta gamma")
# the two points of the case file, in the first quadrant
POINTS = [
    {TH: 0.3, PH: 1.1, TH2: 0.7, PH2: 0.4, A: 0.2, B: 0.9, G: 1.3},
    {TH: 1.4, PH: 0.25, TH2: 0.5, PH2: 1.2, A: 1.1, B: 0.4, G: 0.3},
]


def stored(key):
    return reference.get("17_symbolic_axes", f"symbolic_axes/{key}")


def direction(theta, phi):
    return [sp.sin(theta) * sp.cos(phi), sp.sin(theta) * sp.sin(phi), sp.cos(theta)]


def rz(x):
    return sp.Matrix([[sp.cos(x), -sp.sin(x), 0], [sp.sin(x), sp.cos(x), 0], [0, 0, 1]])


def ry(x):
    return sp.Matrix([[sp.cos(x), 0, sp.sin(x)], [0, 1, 0], [-sp.sin(x), 0, sp.cos(x)]])


FRAME = rz(A) * ry(B) * rz(G)
CASES = {
    "z_axis": ([direction(TH, PH)],),
    "x_and_z_axes": ([[sp.cos(A), sp.sin(A), 0], [0, 0, 1]],),
    "tilted_about_y": ([[sp.cos(TH), 0, -sp.sin(TH)], [sp.sin(TH), 0, sp.cos(TH)]],),
    "two_systems": ([direction(TH, PH)], [direction(TH2, PH2)]),
    "laboratory_to_z_axis": ([[0, 0, 1]], [direction(TH, PH)]),
    "symbolic_and_numeric": ([[sp.sin(TH), 0, sp.cos(TH)]], [[0, 1, 0], [0, 0, 1]]),
    "three_axes": (FRAME.tolist(),),
    "three_axes_transposed": (FRAME.T.tolist(),),
    "not_normalised": ([[3 * sp.cos(A), 3 * sp.sin(A), 0], [0, 0, 2 + TH]],),
}


def numbers(expressions, point):
    return np.array([complex(sp.sympify(e).subs(point)) for e in expressions]).real


def rotation(angles):
    return np.array(sd.euler_matrix(*(float(a) for a in angles)).evalf(), dtype=float)


def numeric(system, point):
    return [[float(sp.sympify(c).subs(point)) for c in axis] for axis in system]


@pytest.mark.parametrize("case", CASES)
def test_symbolic_euler_angles(case):
    systems = CASES[case]
    angles = sd.axes_to_euler(*systems)
    theirs = stored(case)
    for point, reference_angles in zip(POINTS, theirs):
        ours = numbers(angles, point)
        # the angles are those of the numeric routine at that point (angles are compared as rotations)
        direct = sd.axes_to_euler(*(numeric(system, point) for system in systems))
        for mine, other in ((ours, direct), (ours, reference_angles.real)):
            assert np.allclose(rotation(mine), rotation(other), **TOL), case


def test_symbolic_angles_come_out_in_simple_forms():
    assert sd.axes_to_euler(FRAME.tolist()) == (A, B, G)
    assert stored("three_axes_text") == "{a, b, g}"
    assert sd.axes_to_euler(FRAME.T.tolist()) == (sp.pi - G, B, sp.pi - A)
    assert stored("three_axes_transposed_text") == "{-g + Pi, b, -a + Pi}"
    assert sd.axes_to_euler([[sp.cos(A), sp.sin(A), 0], [0, 0, 1]]) == (-A, 0, 0)
    assert stored("x_and_z_axes_text") == "{-a, 0, 0}"
    assert sd.axes_to_euler(*CASES["tilted_about_y"]) == (sp.pi, TH, sp.pi)
    assert stored("tilted_about_y_text") == "{Pi, th, Pi}"
    alpha, beta, gamma = sd.axes_to_euler(direction(TH, PH))
    assert (beta, gamma) == (TH, sp.pi - PH)
    # SpinDynamica writes the remaining angle as -Pi + ArcTan[Sec[th] Tan[ph]]
    assert stored("z_axis_text") == "{-Pi + ArcTan[Sec[th]*Tan[ph]], th, -ph + Pi}"
    assert alpha == sp.atan2(-sp.sin(PH), -sp.cos(PH) * sp.cos(TH))
    for point in POINTS:
        assert np.isclose(float(alpha.subs(point)), -np.pi + np.arctan(np.tan(point[PH]) / np.cos(point[TH])))
    # a vector stands for its z axis, as with numbers
    assert sd.axes_to_euler(direction(TH, PH)) == sd.axes_to_euler([direction(TH, PH)])


def test_symbolic_axis_systems():
    for case, vectors in (
        ("axis_system_of_z", (direction(TH, PH),)),
        ("axis_system_of_x_and_z", ([1, TH, 0], direction(TH, PH))),
    ):
        system = sd.axis_system(*vectors)
        assert isinstance(system, sp.Matrix)
        # orthonormal and right-handed, identically
        assert (system * system.T - sp.eye(3)).applyfunc(sp.simplify) == sp.zeros(3)
        assert sp.simplify(system.det() - 1) == 0
        for point, theirs in zip(POINTS, stored(case)):
            ours = np.array(system.subs(point).evalf(), dtype=float)
            assert np.allclose(ours, theirs.real, **TOL), case
            assert np.allclose(ours, sd.axis_system(*numeric(vectors, point)), **TOL)
    three = sd.axis_system(*(2 * FRAME).T.tolist())
    assert (three - FRAME.T).applyfunc(sp.simplify) == sp.zeros(3)
    with pytest.raises(ValueError, match="one, two or three"):
        sd.axis_system([TH, 0, 1], [0, 1, 0], [1, 0, 0], [0, 0, 1])
