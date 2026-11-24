"""Bounds on amplitudes, correlated eigensystems, adiabatic events, trajectory functions, embedded coordinates
and the shift-and-z basis of spins above 1/2."""

import math

import numpy as np
import pytest
import scipy.stats
import sympy as sp

import spyndynamica as sd
from spyndynamica import Adiabatic, SpinSystem, TimeDependent
from tests.spyndynamica import reference

TOL = dict(rtol=0, atol=1e-11)
T = sp.Symbol("t", real=True)
B = sp.Symbol("b", real=True)


def analysis(key):
    return reference.get("18_analysis_adiabatic", key)


def trajectories(key):
    return reference.get("19_trajectory_functions", f"two_half/{key}")


def ramp(s, duration):
    """Two coupled spins whose shift difference is swept from +100 Hz to -100 Hz."""
    i = s.I
    sweep = 2 * sp.pi * 100 * (1 - 2 * T / duration)
    return TimeDependent(sweep * (i(1, "z") - i(2, "z")) / 2 + 2 * sp.pi * 10 * s.dot(1, 2), T)


def test_amplitude_bounds():
    s = SpinSystem(2)
    i = s.I
    assert np.allclose(sd.amplitude_bounds(i(1, "z"), i(2, "z")), analysis("bounds/z1_to_z2"), **TOL)
    targets = [i(1, "z"), 2 * i(1, "z") @ i(2, "z"), i(1, "x"), s.dot(1, 2)]
    several = sd.amplitude_bounds(i(1, "z") + i(2, "z"), targets)
    assert several.shape == (4, 2) and np.allclose(several, analysis("bounds/total_z_to_several"), **TOL)
    assert np.allclose(sd.amplitude_bounds(i(1, "z"), s.dot(1, 2)), analysis("bounds/z1_to_singlet_order"), **TOL)
    polarized = s.unity() / 4 + i(1, "z") / 3 + i(2, "z") / 5
    bounds = sd.amplitude_bounds(polarized, s.dot(1, 2))
    assert np.allclose(bounds, analysis("bounds/polarized_to_singlet_order"), **TOL)
    # SpinDynamica's NormalizationFactor has no effect here; the port divides by it
    assert np.allclose(analysis("bounds/normalization_factor"), [-1, 1])
    assert np.allclose(sd.amplitude_bounds(i(1, "z"), i(2, "z"), normalization=2), [-0.5, 0.5])
    assert analysis("bounds/not_hermitian_is_rejected") == "True"
    with pytest.raises(ValueError, match="Hermitian"):
        sd.amplitude_bounds(i(1, "z"), i(2, "-"))
    with pytest.raises(ValueError, match="zero"):
        sd.amplitude_bounds(i(1, "z"), i(2, "z"), normalization=0)
    three = SpinSystem(3)
    j = three.I
    assert np.allclose(sd.amplitude_bounds(j(1, "z"), j(3, "z")), analysis("bounds_three/z1_to_z3"), **TOL)
    order = 4 * j(1, "z") @ j(2, "z") @ j(3, "z")
    assert np.allclose(
        sd.amplitude_bounds(j(None, "z"), order), analysis("bounds_three/total_z_to_three_spin_order"), **TOL
    )


def test_amplitude_bounds_hold_and_are_reached():
    # independent of SpinDynamica: no unitary exceeds the bounds, and the unitary that maps the eigenvectors
    # of the state onto those of the target in the same (opposite) order reaches the upper (lower) one
    s = SpinSystem(2)
    i = s.I
    rho = s.unity() / 4 + i(1, "z") / 3 + i(2, "z") / 5 + i(1, "x") @ i(2, "x") / 7
    target = s.dot(1, 2) + i(1, "z") / 2
    low, high = sd.amplitude_bounds(rho, target)
    r, q = rho.numeric(), target.numeric()

    def amplitude(u):
        return (np.vdot(q, u @ r @ u.conj().T) / np.vdot(q, q)).real

    for seed in range(40):
        u = scipy.stats.unitary_group.rvs(4, random_state=seed)
        assert low - 1e-12 <= amplitude(u) <= high + 1e-12
    _, vr = np.linalg.eigh(r)
    _, vq = np.linalg.eigh(q)
    assert np.isclose(amplitude(vq @ vr.conj().T), high) and np.isclose(amplitude(vq[:, ::-1] @ vr.conj().T), low)


def test_correlated_eigensystems():
    s = SpinSystem(2)
    i = s.I
    h = B * i(1, "z") + 2 * B * i(2, "z") + 2 * sp.pi * 5 * s.dot(1, 2)
    fields = [0, 10, 20, 30, 40]
    found = sd.correlated_eigensystems(h, B, fields)
    assert found.eigenvalues.shape == (5, 4) and found.eigenvectors.shape == (5, 4, 4)
    assert np.array_equal(found.values, fields)
    # every state is an eigenstate at its field, and the set of eigenvalues is right
    for value, energies, vectors in zip(fields, found.eigenvalues, found.eigenvectors):
        matrix = h.numeric({B: value})
        for energy, vector in zip(energies, vectors):
            assert np.allclose(matrix @ vector, energy * vector, **TOL)
    # the states keep their character: the overlap of neighbours is the largest in its row
    for before, after in zip(found.eigenvectors[1:-1], found.eigenvectors[2:]):
        overlap = np.abs(before.conj() @ after.T)
        assert (overlap.argmax(axis=1) == np.arange(4)).all()
    # SpinDynamica: the same tracks (the degenerate triplet at zero field may be assigned in another order)
    for case in ("list_eigenvalues", "internal_eigenvalues"):
        theirs = analysis(f"correlated/{case}").real
        assert np.allclose(np.sort(found.eigenvalues[1:].T, axis=0), np.sort(theirs[1:].T, axis=0), **TOL), case
        assert np.allclose(np.sort(found.eigenvalues[0]), np.sort(theirs[0]), **TOL)
    tracks = analysis("correlated/internal_eigenvalue_tracks").real
    assert found.tracks().shape == tracks.shape == (4, 5, 2)
    assert np.allclose(found.tracks()[:, :, 0], tracks[:, :, 0])
    # the public forms with a parameter are refused by SpinDynamica (divergence D15)
    assert analysis("correlated/with_parameter_is_rejected") == "True"
    assert analysis("correlated/matrix_with_parameter_is_rejected") == "True"
    # a list of matrices, a matrix of expressions, a background
    matrices = analysis("correlated/matrices")
    from_list = sd.correlated_eigensystems(matrices)
    assert np.allclose(from_list.eigenvalues, found.eigenvalues, **TOL) and np.array_equal(from_list.values, range(5))
    x = sp.Symbol("x")
    crossing = sd.correlated_eigensystems(sp.Matrix([[x, 0], [0, -x]]), x, [-2, -1, 0.5, 2])
    assert np.allclose(crossing.eigenvalues, [[-2, 2], [-1, 1], [0.5, -0.5], [2, -2]])  # the lines cross
    avoided = sd.correlated_eigensystems(sp.Matrix([[x, 1], [1, -x]]), x, [-2, -1, 1, 2])
    assert (avoided.eigenvalues[:, 0] < 0).all() and (avoided.eigenvalues[:, 1] > 0).all()
    split = sd.correlated_eigensystems(
        B * i(1, "z") + 2 * B * i(2, "z"), B, fields, background=2 * sp.pi * 5 * s.dot(1, 2)
    )
    assert np.allclose(split.eigenvalues, found.eigenvalues, **TOL)
    with pytest.raises(ValueError, match="parameter and its values"):
        sd.correlated_eigensystems(h)
    with pytest.raises(ValueError, match="background and basis"):
        sd.correlated_eigensystems(matrices, background=h)


def test_correlated_eigensystems_of_a_liouvillian():
    s = SpinSystem(1)
    z = s.I(1, "z")
    liouvillian = -sp.I * sd.commutation(B * z) + sd.phenomenological_relaxation(s, 1.0, 0.5)
    found = sd.correlated_eigensystems(liouvillian, B, [0, 1, 2, 3])
    assert found.eigenvalues.shape == (4, 4)
    # two coherences at -1/T2 ∓ iB, two longitudinal modes at 0 and -1/T1, each followed along the field
    final = found.eigenvalues[-1]
    assert np.allclose(sorted(final, key=lambda v: (v.real, v.imag)), [-2 - 3j, -2 + 3j, -1, 0])
    for track in found.eigenvalues.T:
        slopes = np.diff(track.imag)
        assert np.allclose(slopes, slopes[0])  # frequency linear in the field: no state changes its line


def test_adiabatic_events():
    s = SpinSystem(2)
    i = s.I
    fast = ramp(s, 0.5)
    populations = sd.propagate(i(1, "z"), [Adiabatic(fast, 0.5, sampling=50)])
    assert np.allclose(populations, analysis("adiabatic/populations"), **TOL)
    # the populations follow the eigenstates: Iz of spin 1 has become Iz of spin 2, up to the mixing by the coupling
    assert np.isclose(np.vdot(i(2, "z").numeric(), populations).real, 0.995, atol=1e-3)
    mixed = sd.propagate(i(1, "x") + i(1, "z"), [Adiabatic(fast, 0.5, sampling=50)])
    assert np.allclose(mixed, analysis("adiabatic/coherences_are_deleted"), **TOL)
    assert np.allclose(mixed, populations, **TOL)  # the coherences are gone
    grid = sd.propagate(i(1, "z"), [Adiabatic(fast, 0.5, sampling=list(np.arange(26) * 0.02))])
    assert np.allclose(grid, analysis("adiabatic/time_grid"), **TOL)
    events = [sd.rotation_superoperator(s, sp.pi / 3, "y"), Adiabatic(fast, 0.5, sampling=50)]
    assert np.allclose(sd.propagate(i(None, "z"), events), analysis("adiabatic/after_a_pulse"), **TOL)
    # with coherences: the phases are integrals over the eigenvalues, which SpinDynamica interpolates differently
    for case, duration, sampling in (("with_coherences", 0.5, 50), ("slow_with_coherences", 5, 400)):
        event = Adiabatic(ramp(s, duration), duration, sampling=sampling, delete_coherences=False)
        assert np.allclose(sd.propagate(i(1, "x"), [event]), analysis(f"adiabatic/{case}"), rtol=0, atol=2e-5), case
    with pytest.raises(TypeError, match="TimeDependent"):
        sd.propagate(i(1, "z"), [Adiabatic(i(1, "z"), 0.5)])
    with pytest.raises(ValueError, match="one step or more"):
        sd.propagate(i(1, "z"), [Adiabatic(fast, 0.5, sampling=0)])
    with pytest.raises(ValueError, match="ascend"):
        sd.propagate(i(1, "z"), [Adiabatic(fast, 0.5, sampling=[0.3, 0.1])])
    with pytest.raises(NotImplementedError, match="relaxation"):
        sd.propagate(i(1, "z"), [Adiabatic(fast, 0.5)], background=sd.phenomenological_relaxation(s, 1.0))


def test_adiabatic_events_approach_the_exact_propagation():
    # independent of SpinDynamica: the slower the sweep, the closer the integrated equation of motion
    s = SpinSystem(2)
    i = s.I
    errors = []
    for duration in (0.5, 2.0, 8.0):
        sweep = ramp(s, duration)
        exact = sd.propagate(i(1, "x") + i(1, "z"), [(sweep, duration)])
        event = Adiabatic(sweep, duration, sampling=int(400 * duration), delete_coherences=False)
        errors.append(np.abs(sd.propagate(i(1, "x") + i(1, "z"), [event]) - exact).max())
    assert errors[0] > errors[1] > errors[2] and errors[2] < 0.01
    # an adiabatic event is unitary when the coherences are kept, and its parts compose
    sweep = ramp(s, 2.0)
    whole = sd.propagator([Adiabatic(sweep, 2.0, sampling=400, delete_coherences=False)])
    assert np.allclose(whole @ whole.conj().T, np.eye(4), **TOL)
    function = sd.operator_trajectory_function(
        i(1, "x"), [Adiabatic(sweep, 2.0, sampling=400, delete_coherences=False)]
    )
    assert np.allclose(function(2.0), whole @ i(1, "x").numeric() @ whole.conj().T, **TOL)
    halfway = function(1.0)
    assert np.isclose(np.trace(halfway @ halfway).real, 1.0)  # still a rotated Ix: the norm is kept
    # a Hamiltonian that only changes its strength: the eigenstates stay, the phase is the integral
    one = SpinSystem(1)
    z, x = one.I(1, "z"), one.I(1, "x")
    out = sd.propagate(
        x, [Adiabatic(TimeDependent(2 * sp.pi * 10 * T * z, T), 0.3, sampling=30, delete_coherences=False)]
    )
    angle = 2 * math.pi * 10 * 0.3**2 / 2
    assert np.allclose(
        out, math.cos(angle) * x.numeric() + math.sin(angle) * one.I(1, "y").numeric(), rtol=0, atol=1e-9
    )


def test_trajectory_functions():
    s = SpinSystem(2)
    i = s.I
    ham = 2 * sp.pi * 10 * i(1, "z") - 2 * sp.pi * 4 * i(2, "z") + 2 * sp.pi * 7 * s.dot(1, 2)
    events = [
        sd.rotation_superoperator(s, sp.pi / 2, "y"),
        (ham, 0.1),
        sd.rotation_superoperator(s, sp.pi, "x"),
        (ham, 0.15),
    ]
    times = trajectories("times").real
    one = sd.trajectory_function(i(1, "z"), i(1, "x"), events)
    assert one.range == (0.0, 0.25) and np.allclose(sorted(trajectories("time_range").reshape(-1)), [0, 0.25])
    assert np.allclose(one(times), trajectories("one_observable"), **TOL)
    assert np.isclose(one(0.013), trajectories("one_observable")[1]) and np.ndim(one(0.013)) == 0
    several = sd.trajectory_function(i(None, "z"), [i(1, "x"), i(2, "y"), i(1, "z")], events)
    assert several(times).shape == (3, 7) and several(0.05).shape == (3,)
    assert np.allclose(several(times).T, trajectories("several_observables"), **TOL)
    relaxing = sd.trajectory_function(
        i(1, "z"), i(1, "x"), events, background=sd.phenomenological_relaxation(s, 0.5, 0.1)
    )
    assert np.allclose(relaxing(times), trajectories("with_relaxation"), **TOL)
    states = sd.operator_trajectory_function(i(1, "z"), events)
    assert np.allclose(states(times), trajectories("operator"), **TOL)
    assert np.allclose(sorted(trajectories("operator_time_range").reshape(-1)), states.range)
    # independent: the events up to the time, propagated in one go; a pulse at the time has acted
    assert np.allclose(states(0.17), sd.propagate(i(1, "z"), [*events[:3], (ham, 0.07)]), **TOL)
    assert np.allclose(states(0.1), sd.propagate(i(1, "z"), events[:3]), **TOL)
    assert np.allclose(states(0.0), sd.propagate(i(1, "z"), events[:1]), **TOL)
    assert np.allclose(states(0.25), sd.propagate(i(1, "z"), events), **TOL)
    # the sampled trajectory lies on the function
    sample_times, samples = sd.trajectory(i(1, "z"), i(1, "x"), events, points=8)
    assert np.allclose(one(sample_times[1:9]), samples[1:9], **TOL)
    assert np.allclose(sd.trajectory_function(i(1, "z"), i(1, "x"), events, normalization=2)(0.05), one(0.05) / 2)
    with pytest.raises(ValueError, match="outside the events"):
        one(0.3)
    with pytest.raises(ValueError, match="outside the events"):
        states(-0.01)


def test_trajectory_function_under_a_time_dependent_hamiltonian():
    s = SpinSystem(1)
    x, y, z = s.I(1)
    w = 2 * sp.pi * 30 * T
    rf = TimeDependent(2 * sp.pi * 20 * (x * sp.cos(w) + y * sp.sin(w)) + 2 * sp.pi * 25 * z, T)
    function = sd.trajectory_function(z, z, [(rf, 0.05)])
    theirs = trajectories("time_dependent").real
    assert np.allclose(function([0, 0.01, 0.025, 0.05]), theirs, rtol=0, atol=5e-7)
    # the analytic nutation in the rotating frame: offset -5 Hz, field 20 Hz
    effective = 2 * math.pi * math.hypot(20, 5)
    expected = (5 / math.hypot(20, 5)) ** 2 + (20 / math.hypot(20, 5)) ** 2 * math.cos(effective * 0.025)
    assert np.isclose(function(0.025), expected, rtol=0, atol=1e-9)


def test_embedded_coordinates():
    grid = np.array([[np.exp(-(j + 2 * k) / 5) * np.exp(1j * (j - k)) for k in range(6)] for j in range(4)])
    t1, t2 = 0.1 * np.arange(4), 0.05 * np.arange(6)
    rows = sd.embed_coordinates(grid, t1, t2)
    assert rows.shape == (24, 3) and np.allclose(rows[7], [0.1, 0.05, grid[1, 1]])
    back = sd.split_coordinates(rows)
    assert np.allclose(back[0], t1) and np.allclose(back[1], t2) and np.allclose(back[2], grid)
    stored = lambda case: reference.get("16_two_dimensional", f"transform/{case}")  # noqa: E731
    for case, fill in (("triples", False), ("sample_intervals_zero_filled", True)):
        ours, theirs = sd.ft_coordinates(rows, zero_fill=fill), stored(case)
        assert np.allclose(ours[:, 2], theirs[:, 2], **TOL), case
        # the frequencies are in Hz; SpinDynamica's are divided by both sampling rates (divergence D14)
        assert np.allclose(ours[:, :2].real, theirs[:, :2].real * 200, **TOL), case
    f1, f2, spectrum = sd.split_coordinates(sd.ft_coordinates(rows, zero_fill=(4, False)))
    assert np.allclose(spectrum, sd.ft2(grid, zero_fill=(4, False)), **TOL)
    assert np.allclose(f1, sd.frequency_axis(4, 0.1, False)) and np.allclose(f2, sd.frequency_axis(6, 0.05, 4))
    # one dimension: SpinDynamica's pairs
    pairs = sd.embed_coordinates([1.0, 0.5j, -0.25, -0.125j], [0, 0.1, 0.2, 0.3])
    for case, fill in (("pairs", True), ("pairs_no_zero_fill", False)):
        assert np.allclose(
            sd.ft_coordinates(pairs, zero_fill=fill), reference.get("06_signals", f"fourier/{case}"), **TOL
        )
    with pytest.raises(ValueError, match="one axis per dimension"):
        sd.embed_coordinates(grid, t1)
    with pytest.raises(ValueError, match="full grid"):
        sd.split_coordinates(rows[:-1])
    with pytest.raises(ValueError, match="evenly spaced"):
        sd.ft_coordinates(sd.embed_coordinates([1, 2, 3], [0, 0.1, 0.3]))
    with pytest.raises(ValueError, match="rows"):
        sd.split_coordinates(grid)


@pytest.mark.parametrize(
    "group, spins", [("spin2", [2]), ("one_one", [1, 1]), ("three_halves_half", [sp.Rational(3, 2), sp.Rational(1, 2)])]
)
def test_shift_and_z_basis_of_higher_spins_is_the_same_set_in_every_coherence_order(group, spins):
    system = SpinSystem([(n + 1, spin) for n, spin in enumerate(spins)])
    theirs = reference.get("20_shift_and_z_order", f"{group}/shift_and_z")
    dim = system.dimension
    ours = np.array([op.numeric() for op in sd.shift_and_z_basis(system).operators])
    assert theirs.shape == ours.shape == (dim * dim, dim, dim)
    overlap = np.abs(np.einsum("iab,jab->ij", theirs.conj(), ours))
    # every operator of SpinDynamica is one operator of the port, and the other way round
    assert np.allclose(np.sort(overlap, axis=1)[:, -1], 1, **TOL) and np.allclose(overlap.sum(axis=1), 1, **TOL)
    position = overlap.argmax(axis=1)
    assert sorted(position) == list(range(dim * dim))

    def coherence_order(matrix):
        rows, columns = np.nonzero(np.abs(matrix) > 1e-9)
        weights = np.array([float(sum(m)) for m in system.quantum_numbers])
        found = set(np.round(weights[rows] - weights[columns], 9))
        assert len(found) == 1
        return found.pop()

    their_orders = [coherence_order(m) for m in theirs]
    assert their_orders == sorted(their_orders) == [coherence_order(m) for m in ours]
    # the sequence inside one coherence order differs: SpinDynamica's comes from Mathematica's sorting
    if group == "spin2":
        assert not np.array_equal(position, np.arange(dim * dim))
