"""Pulse programs: the branch run against the explicit phase cycle, the ported sequences, spatial averaging and ctps.

Author: Alexej Jerschow.
"""

import sys
from dataclasses import replace

import numpy as np
import pytest
import scipy.linalg as sla
import scipy.sparse as sps

import spynach
from spynach.filters import coherence_mask, isotope_orders, leakage, orders_present
from spynach.program import (
    Acquire,
    Block,
    Delay,
    Evolve,
    Gradient,
    Homospoil,
    Program,
    Pulse,
    Refocus,
    Sample,
    sample_envelope,
)

LIOUV = {"formalism": "sphten-liouv", "approximation": "none"}
MHZ_PER_T = 42.577478  # 1H


def system(isotopes, magnet=14.1, **inter):
    return spynach.basis(spynach.create({"isotopes": isotopes, "magnet": magnet}, inter), LIOUV)


def generators(spin_system, parameters):
    """H (with offsets) and zero R, K, as the liquid context hands them to a sequence."""
    spin_system = spynach.assume(spin_system, "nmr")
    H = spynach.hamiltonian(spin_system)
    H = spynach.frqoffset(spin_system, H, parameters)
    return H, 0 * H, 0 * H


def pair():
    """Two coupled protons, carrier between them."""
    spin_system = system(["1H", "1H"], zeeman={"scalar": [1.0, 1.5]}, coupling={"scalar": [[0, 12.0], [0, 0]]})
    return spin_system, {"spins": ["1H"], "offset": 1.25 * MHZ_PER_T * 14.1}


def dense(M):
    return np.asarray(M.toarray() if sps.issparse(M) else M, dtype=complex)


def manual_cycle(spin_system, L, blocks, phases, receiver, rho0, coil):
    """The explicit cycle by hand: ``blocks`` is a list of ("pulse", spins, angle, j) / ("delay", t) /
    ("evolve", dt, n) / ("refocus", dt) / ("homospoil",) / ("acquire", dt, n); ``phases[j]`` the phase list in degrees
    of pulse ``j``, lists repeating cyclically over the ``N`` steps."""
    N = int(np.lcm.reduce([len(p) for p in phases] + [len(receiver)]))
    total = 0
    for m in range(N):
        rho = np.asarray(rho0, dtype=complex).reshape(-1, 1)
        shape = []
        for block in blocks:
            if block[0] == "pulse":
                _, spins, angle, j = block
                phi = np.radians(phases[j][m % len(phases[j])])
                op = np.cos(phi) * spynach.operator(spin_system, "Lx", spins) + np.sin(phi) * spynach.operator(
                    spin_system, "Ly", spins
                )
                rho = spynach.step(spin_system, op, rho, angle)
            elif block[0] == "delay":
                rho = spynach.evolution(spin_system, L, rho, block[1], 1, "final").reshape(rho.shape[0], -1)
            elif block[0] == "evolve":
                rho = np.hstack(
                    [
                        spynach.evolution(spin_system, L, rho[:, c], block[1], block[2] - 1, "trajectory")
                        for c in range(rho.shape[1])
                    ]
                )
                shape.append(block[2])
            elif block[0] == "refocus":
                rho = spynach.evolution(spin_system, L, rho, block[1], 0, "refocus")
            elif block[0] == "homospoil":
                rho = spynach.homospoil(spin_system, rho.T, "destroy").T
            elif block[0] == "acquire":
                fid = spynach.evolution(spin_system, L, rho, block[1], block[2] - 1, "observable", coil=coil)
                # the receiver acts as exp(-i phi) on the textbook signal, i.e. exp(+i phi) on Spinach's conjugate
                fid = fid.reshape(block[2], *shape) * np.exp(1j * np.radians(receiver[m % len(receiver)]))
                total = total + fid
    return total / N


def relative(a, b):
    return np.abs(a - b).max() / np.abs(b).max()


# ---------------------------------------------------------------------------------------------- filters


def test_coherence_mask_and_isotope_orders():
    spin_system = system(["1H", "1H", "13C"], zeeman={"scalar": [1.0, 2.0, 30.0]})
    rho = np.arange(spin_system.basis.nstates, dtype=complex) + 1
    spec = [("1H", [1, -1]), ("13C", 0)]
    expected = spynach.coherence(spin_system, rho, spec)
    mask = coherence_mask(spin_system, spec)
    assert np.array_equal(expected != 0, mask) and np.allclose(expected[mask], rho[mask])
    orders = isotope_orders(spin_system)
    assert set(orders) == {"1H", "13C"}
    assert np.array_equal(orders["1H"] + orders["13C"], spin_system.basis.tot_proj)
    assert list(orders_present(spin_system, spynach.state(spin_system, "L+", "1H"), "1H")) == [1]
    assert list(orders_present(spin_system, spynach.state(spin_system, "L+", "1H"), "13C")) == [0]
    stack = np.stack([spynach.state(spin_system, "L+", "1H"), spynach.state(spin_system, "L-", "13C")])
    assert list(orders_present(spin_system, stack, "1H")) == [0, 1]


def test_leakage_measures_order_conservation():
    spin_system, pars = pair()
    H, _, _ = generators(spin_system, pars)
    assert leakage(spin_system, H, {"1H": 1}) == 0
    rf = H + 2 * np.pi * 1000 * spynach.operator(spin_system, "Lx", "1H")  # a transverse field moves order
    assert leakage(spin_system, rf, {"1H": 1}) > 0.1
    with pytest.raises(ValueError, match="no basis state"):
        leakage(spin_system, H, {"1H": 7})


# ------------------------------------------------------------------------ layer 1: manual cycle = branch run


def test_dqf_cosy_cycle_equals_branch_run_and_explicit_mode():
    spin_system, pars = pair()
    H, R, K = generators(spin_system, pars)
    L = H
    dt, n1, n2 = 1 / 400, 6, 24
    four = [0, 90, 180, 270]
    program = Program(
        [
            Pulse("1H", np.pi / 2, phase=four),
            Evolve(dt, n1),
            Pulse("1H", np.pi / 2, phase=four),
            Delay(3e-6),
            Pulse("1H", np.pi / 2, phase=[90]),
            Acquire("1H", dt, n2, receiver=[0, 180, 0, 180]),
        ]
    )
    assert program.steps(spin_system) == 4
    rho0 = spynach.state(spin_system, "Lz", "1H", "cheap")
    coil = spynach.state(spin_system, "L+", "1H", "cheap")
    blocks = [("pulse", "1H", np.pi / 2, 0), ("evolve", dt, n1), ("pulse", "1H", np.pi / 2, 1), ("delay", 3e-6)]
    blocks += [("pulse", "1H", np.pi / 2, 2), ("acquire", dt, n2)]
    reference = manual_cycle(spin_system, L, blocks, [four, four, [90]], [0, 180, 0, 180], rho0, coil)
    fid = program.run(spin_system, H, R, K)
    assert fid.shape == (n2, n1) and relative(fid, reference) < 1e-10
    assert relative(program.explicit().run(spin_system, H, R, K), reference) < 1e-12
    # Through the liquid context, as a sequence; and the pathways pass through double-quantum coherence only
    assert relative(spynach.liquid(spin_system, program, pars, "nmr"), reference) < 1e-10
    selection = program.select(spin_system)
    assert set(np.unique(selection.orders[:, 0, 2])) == {-2, 2} and selection.kept.all()
    assert set(selection.orders[:, 0, -1]) == {-1}  # textbook convention: the detected order is -1
    assert (
        len(selection.table()) == len(selection)
        and "pathways" in repr(selection)
        and "<table>" in selection._repr_html_()
    )
    # The same selection, handed in, gives the same run; per-pathway output sums to the FID
    paths = program.run(spin_system, H, R, K, output="pathways", selection=selection)
    # Lz under one 90° pulse is single-quantum only, so the enumerated pathways through 0 and ±2 in t1 are empty
    assert paths.fids.shape == (4, n2, n1) and relative(paths.sum(), fid) < 1e-12
    assert set(paths.orders[:, 0, 1]) == {-1, 1} and paths.pathway(0) == selection.pathway(int(paths.index[0]))


def test_partial_weights_and_fixed_phases():
    """A three-step list on a four-phase grid gives |pc| = 1/3 weights; a y pulse equals a hand-written Ly step."""
    spin_system, pars = pair()
    H, R, K = generators(spin_system, pars)
    dt, n = 1 / 400, 16
    rho0 = spynach.state(spin_system, "Lz", "1H", "cheap")
    coil = spynach.state(spin_system, "L+", "1H", "cheap")
    program = Program(
        [Pulse("1H", np.pi / 2, phase=[0, 90, 180]), Delay(1e-3), Pulse("1H", np.pi / 3), Acquire("1H", dt, n)]
    )
    selection = program.select(spin_system)
    assert np.allclose(np.sort(np.abs(selection.pc)), np.sort(np.abs(selection.pc))) and np.any(
        np.isclose(np.abs(selection.pc), 1 / 3)
    )
    blocks = [("pulse", "1H", np.pi / 2, 0), ("delay", 1e-3), ("pulse", "1H", np.pi / 3, 1), ("acquire", dt, n)]
    reference = manual_cycle(spin_system, H, blocks, [[0, 90, 180], [0]], [0], rho0, coil)
    assert relative(program.run(spin_system, H, R, K), reference) < 1e-10
    y = Program([Pulse("1H", np.pi / 2, phase=[90]), Acquire("1H", dt, n)])
    rho = spynach.step(spin_system, spynach.operator(spin_system, "Ly", "1H"), rho0, np.pi / 2)
    assert (
        relative(
            y.run(spin_system, H, R, K), spynach.evolution(spin_system, H, rho, dt, n - 1, "observable", coil=coil)
        )
        < 1e-12
    )


def test_aliased_order_jumps_are_kept():
    """A four-step cycle on five spins also passes Δp = +3 and -5 beside -1; the run keeps them, as the cycle does."""
    spin_system = system(
        ["1H"] * 5,
        zeeman={"scalar": [1.0, 1.2, 1.4, 1.6, 1.8]},
        coupling={"scalar": (np.triu(np.full((5, 5), 7.0), 1)).tolist()},
    )
    pars = {"spins": ["1H"], "offset": 1.4 * MHZ_PER_T * 14.1}
    H, R, K = generators(spin_system, pars)
    dt, n = 1 / 2000, 8
    four = [0, 90, 180, 270]
    program = Program(
        [Pulse("1H", np.pi / 2, phase=four), Delay(2e-3), Pulse("1H", np.pi / 2), Acquire("1H", dt, n, receiver=four)]
    )
    selection = program.select(spin_system)
    assert set(selection.orders[:, 0, 1]) == {-1, 3, -5}
    rho0 = spynach.state(spin_system, "Lz", "1H", "cheap")
    coil = spynach.state(spin_system, "L+", "1H", "cheap")
    blocks = [("pulse", "1H", np.pi / 2, 0), ("delay", 2e-3), ("pulse", "1H", np.pi / 2, 1), ("acquire", dt, n)]
    reference = manual_cycle(spin_system, H, blocks, [four, [0]], four, rho0, coil)
    assert np.abs(reference).max() > 0.1 and relative(program.run(spin_system, H, R, K), reference) < 1e-8


def test_nested_cogwheel_and_block_phases():
    spin_system, pars = pair()
    H, R, K = generators(spin_system, pars)
    dt, n = 1 / 400, 12
    rho0 = spynach.state(spin_system, "Lz", "1H", "cheap")
    coil = spynach.state(spin_system, "L+", "1H", "cheap")
    # Nested: the second list steps slower than the first (8 steps)
    p1, p2, rec = [0, 90, 180, 270], [0, 0, 0, 0, 180, 180, 180, 180], [0, 270, 180, 90, 0, 270, 180, 90]
    nested = Program(
        [
            Pulse("1H", np.pi / 2, phase=p1),
            Delay(1e-3),
            Pulse("1H", np.pi / 2, phase=p2),
            Acquire("1H", dt, n, receiver=rec),
        ]
    )
    blocks = [("pulse", "1H", np.pi / 2, 0), ("delay", 1e-3), ("pulse", "1H", np.pi / 2, 1), ("acquire", dt, n)]
    assert nested.steps(spin_system) == 8
    reference = manual_cycle(spin_system, H, blocks, [p1, p2], rec, rho0, coil)
    assert np.abs(reference).max() > 0.1 and relative(nested.run(spin_system, H, R, K), reference) < 1e-10
    # Cogwheel COG5(1, 2; 2) selects the 0, -1, +1 pathway: phases 360 nu m / N
    cog = Program(
        [
            Pulse("1H", np.pi / 2, phase=("cog", 1)),
            Delay(1e-3),
            Pulse("1H", np.pi / 2, phase=("cog", 2)),
            Acquire("1H", dt, n, receiver=("cog", 3)),
        ],
        cycle_length=5,
    )
    lists = [[360 * nu * m / 5 for m in range(5)] for nu in (1, 2, 3)]
    reference = manual_cycle(spin_system, H, blocks, lists[:2], lists[2], rho0, coil)
    assert np.abs(reference).max() > 0.1 and relative(cog.run(spin_system, H, R, K), reference) < 1e-10
    assert cog.select(spin_system).orders[:, 0, 1].tolist() == [1]  # COG5(1,2;3) keeps 0, +1, -1
    # Block phases add to every pulse inside (the receiver follows the block: Δp over the block is +1)
    rec = [0, 90, 180, 270]  # the receiver follows the block: Δp over the block is -1
    block = Program(
        [
            Block([Pulse("1H", np.pi / 2), Delay(1e-3), Pulse("1H", np.pi / 2, phase=[0, 180])], phase=p1),
            Acquire("1H", dt, n, receiver=rec),
        ]
    )
    second = [(a + b) % 360 for a, b in zip(np.resize(p1, 8), np.resize([0, 180], 8))]
    reference = manual_cycle(spin_system, H, blocks, [np.resize(p1, 8).tolist(), second], rec, rho0, coil)
    assert np.abs(reference).max() > 0.1 and relative(block.run(spin_system, H, R, K), reference) < 1e-10
    with pytest.raises(ValueError, match="cycle_length"):
        Program([Pulse("1H", np.pi / 2, phase=("cog", 1)), Acquire("1H", dt, n)]).steps(spin_system)
    with pytest.raises(ValueError, match="1° grid"):
        Program([Pulse("1H", np.pi / 2, phase=[0, 0.3]), Acquire("1H", dt, n)]).steps(spin_system)


def test_heteronuclear_cycle_with_simultaneous_pulses():
    """Eight steps with phases on both channels, a pulse on both isotopes at once, and a selective pulse."""
    spin_system = system(["1H", "13C"], zeeman={"scalar": [3.0, 40.0]}, coupling={"scalar": [[0, 140.0], [0, 0]]})
    pars = {"spins": ["1H", "13C"], "offset": [3.0 * MHZ_PER_T * 14.1, 40.0 * 10.7084 * 14.1]}
    H, R, K = generators(spin_system, pars)
    dt, n = 1 / 2000, 10
    ph_h, ph_c, rec = [0, 90, 180, 270], [0, 0, 0, 0, 180, 180, 180, 180], [0, 270, 180, 90, 180, 90, 0, 270]
    program = Program(
        [
            Pulse("1H", np.pi / 2, phase=ph_h),
            Delay(1 / 560),
            Pulse(["1H", "13C"], np.pi),
            Delay(1 / 560),
            Pulse([0], np.pi / 2, phase=[90]),
            Pulse("13C", np.pi / 2, phase=ph_c),
            Delay(1e-3),
            Pulse(["13C", "1H"], np.pi / 2),
            Acquire("1H", dt, n, receiver=rec),
        ]
    )
    assert program.steps(spin_system) == 8
    rho0 = spynach.state(spin_system, "Lz", "1H", "cheap")
    coil = spynach.state(spin_system, "L+", "1H", "cheap")
    blocks = [
        ("pulse", "1H", np.pi / 2, 0),
        ("delay", 1 / 560),
        ("pulse", "1H", np.pi, 1),
        ("pulse", "13C", np.pi, 1),
        ("delay", 1 / 560),
    ]
    blocks += [
        ("pulse", [0], np.pi / 2, 2),
        ("pulse", "13C", np.pi / 2, 3),
        ("delay", 1e-3),
        ("pulse", "13C", np.pi / 2, 4),
        ("pulse", "1H", np.pi / 2, 4),
    ]
    blocks += [("acquire", dt, n)]
    reference = manual_cycle(spin_system, H, blocks, [ph_h, [0], [90], ph_c, [0]], rec, rho0, coil)
    assert np.abs(reference).max() > 0.1 and relative(program.run(spin_system, H, R, K), reference) < 1e-10
    selection = program.select(spin_system)
    assert selection.isotopes == ("1H", "13C") and selection.orders.shape[1:] == (2, 8)


# ---------------------------------------------------------------------------- layer 2: the ported sequences


def test_cosy_program_equals_the_port():
    spin_system, pars = pair()
    pars = {**pars, "angle": np.pi / 2, "sweep": 400.0, "npoints": [8, 16]}
    program = Program(
        [
            Pulse("1H", np.pi / 2),
            Evolve(1 / 400, 8),
            Delay(0.0, orders={"1H": [-1]}),  # cosy keeps Spinach's +1 in t1, the textbook -1
            Pulse("1H", np.pi / 2),
            Acquire("1H", 1 / 400, 16),
        ]
    )
    assert (
        relative(
            spynach.liquid(spin_system, program, pars, "nmr"), spynach.liquid(spin_system, spynach.cosy, pars, "nmr")
        )
        < 1e-12
    )


def test_noesy_program_equals_the_port():
    spin_system = system(
        ["1H", "1H"],
        zeeman={"scalar": [1.0, 3.0]},
        coordinates=[[0, 0, 0], [0, 0, 2.5]],
        relaxation=["redfield"],
        equilibrium="IME",
        rlx_keep="kite",
        temperature=298,
        tau_c=[[2e-9]],
    )
    pars = {
        "tmix": 0.1,
        "offset": 1200.0,
        "sweep": [2000.0, 2000.0],
        "npoints": [6, 12],
        "spins": ["1H"],
        "needs": ["rho_eq"],
    }
    port = spynach.liquid(spin_system, spynach.noesy, pars, "nmr")
    rho0 = spynach.liquid(spin_system, lambda ss, p, H, R, K: p["rho0"], pars, "nmr")
    dt = 1 / 2000

    def noesy(second):
        return Program(
            [
                Pulse("1H", np.pi / 2),
                Evolve(dt, 6),
                Pulse("1H", np.pi / 2, phase=second),
                Homospoil(),
                Delay(0.1, generator="relaxation"),
            ]
            + [Pulse("1H", np.pi / 2, phase=[90]), Acquire("1H", dt, 12, receiver=[0, 180])],
            rho0=rho0,
            check_leakage=True,
        )

    cos = noesy([0, 180])
    assert relative(2 * spynach.liquid(spin_system, cos, pars, "nmr"), port["cos"]) < 1e-10
    assert relative(2 * spynach.liquid(spin_system, noesy([90, 270]), pars, "nmr"), port["sin"]) < 1e-10
    selection = cos.select(spin_system)
    spynach.liquid(spin_system, cos, {**pars}, "nmr")
    run_selection = cos.select(spin_system)
    cos.run(spin_system, *generators(spin_system, pars), selection=run_selection)
    assert (
        run_selection.leakage and max(run_selection.leakage.values()) < 1e-12 and len(selection) == len(run_selection)
    )


def test_hsqc_program_equals_the_port():
    spin_system = system(
        ["1H", "13C", "1H"],
        magnet=5.9,
        zeeman={"scalar": [3.0, 40.0, 1.0]},
        coupling={"scalar": [[0, 140.0, 7.0], [0, 0, 0], [0, 0, 0]]},
    )
    pars = {
        "J": 140.0,
        "sweep": [3000.0, 2000.0],
        "offset": [2500.0, 500.0],
        "npoints": [6, 10],
        "spins": ["13C", "1H"],
        "decouple_f1": ["1H"],
        "decouple_f2": [],
    }
    port = spynach.liquid(spin_system, spynach.hsqc, pars, "nmr")
    delta, dt1, dt2 = 1 / 280, 1 / 3000, 1 / 2000

    def hsqc(order):
        return Program(
            [
                Pulse("1H", np.pi / 2),
                Delay(delta / 2),
                Pulse(["1H", "13C"], np.pi),
                Delay(delta / 2),
                Pulse("1H", np.pi / 2, phase=[90]),
                Pulse("13C", np.pi / 2, phase=[0, 180]),
                Evolve(dt1 / 2, 6),
                Pulse("1H", np.pi),
                Refocus(dt1 / 2),
                Delay(0.0, orders={"1H": [0], "13C": [order]}),
                Pulse(["13C", "1H"], np.pi / 2),
                Delay(delta / 2),
                Pulse(["13C", "1H"], np.pi),
                Delay(delta / 2),
                Acquire("1H", dt2, 10, receiver=[0, 180]),
            ]
        )

    for key, order in (("pos", -1), ("neg", 1)):  # the port's pos is Spinach's +1, the textbook -1
        assert relative(2 * spynach.liquid(spin_system, hsqc(order), pars, "nmr"), port[key]) < 1e-10
    with pytest.raises(ValueError, match="Evolve block before"):
        Program([Pulse("1H", np.pi / 2), Refocus(dt1), Acquire("1H", dt2, 4)]).steps(spin_system)


# --------------------------------------------------------------------------------------- layer 3: gradients


def brute_force_z(spin_system, L, coil, rho0, Lx, sample, stages, n_z=2001):
    """Average over midpoint positions of an explicit run with the gradient as the field term -γ G z Σ F_z.

    ``stages``: ("pulse", angle) / ("grad", G, δ) / ("delay", t) / ("acquire", G_read, dt, n).
    """
    gamma = spin_system.gammas[0]
    Gz = -gamma * dense(spynach.operator(spin_system, "Lz", "1H"))
    edges = np.linspace(-sample.z_max, sample.z_max, n_z + 1)
    zs = (edges[:-1] + edges[1:]) / 2
    U = lambda M, t: sla.expm(-1j * M * t)  # noqa: E731
    total = None
    for z in zs:
        rho = rho0.copy()
        for stage in stages:
            if stage[0] == "pulse":
                rho = U(Lx, stage[1]) @ rho
            elif stage[0] == "grad":
                rho = U(L + z * stage[1] * Gz, stage[2]) @ rho
            elif stage[0] == "delay":
                rho = U(L, stage[1]) @ rho
            else:
                P = U(L + z * stage[1] * Gz, stage[2])
                fid = np.zeros(stage[3], dtype=complex)
                for i in range(stage[3]):
                    fid[i] = coil.conj() @ rho
                    rho = P @ rho
                total = fid if total is None else total + fid
    return total / n_z


def test_gradients_against_spatial_averaging():
    """A gradient pair around a 180, then a read gradient: every pathway's sinc weight and the echo time."""
    spin_system, pars = pair()
    H, R, K = generators(spin_system, pars)
    L, Lx = dense(H), dense(spynach.operator(spin_system, "Lx", "1H"))
    rho0 = spynach.state(spin_system, "Lz", "1H", "cheap")
    coil = spynach.state(spin_system, "L+", "1H", "cheap")
    sample = Sample(z_max=5e-3, x_max=2.5e-3)
    G, delta, tau, dt, n = 0.01, 1e-3, 2e-3, 1 / 8000, 64
    gamma = spin_system.gammas[0]
    program = Program(
        [
            Pulse("1H", np.pi / 2),
            Gradient(delta, z=G),
            Delay(tau),
            Pulse("1H", np.pi / 2),
            Gradient(delta, z=G),
            Acquire("1H", dt, n, gradient=Gradient(0, z=-G / 2)),
        ],
        sample=sample,
    )
    selection = program.select(spin_system)
    echo = selection.orders[:, 0, 1] == 1  # 0, +1, -1 refocuses by the pair
    assert np.allclose(selection.k0[echo], 0) and np.isclose(selection.echo_time[echo][0], 0)
    plus = (
        selection.orders[:, 0, 1] == -1
    )  # 0, -1, -1 is dephased by both gradients; the read gradient refocuses it at 4 delta
    assert np.isclose(np.linalg.norm(selection.k0[plus][0]), 2 * gamma * G * delta) and np.isclose(
        selection.echo_time[plus][0], 4 * delta
    )
    assert np.isnan(
        selection.echo_time[selection.orders[:, 0, 1] == 2][0]
    )  # its echo would come before the acquisition
    assert np.isclose(selection.envelope[echo][0], 1) and np.isclose(
        selection.envelope[plus][0], 1
    )  # echo in the window
    assert selection.envelope[selection.orders[:, 0, 1] == 2][0] < 0.1
    paths = program.run(spin_system, H, R, K, output="pathways")
    fid = paths.sum()
    stages = [
        ("pulse", np.pi / 2),
        ("grad", G, delta),
        ("delay", tau),
        ("pulse", np.pi / 2),
        ("grad", G, delta),
        ("acquire", -G / 2, dt, n),
    ]
    reference = brute_force_z(spin_system, L, coil, rho0, Lx, sample, stages)
    assert relative(fid, reference) < 2e-4
    # The +1, +1 pathway's own FID peaks at its echo
    late = np.flatnonzero(paths.echo_time > 1e-3)
    i = int(late[np.argmax([np.abs(paths.fids[j]).max() for j in late])])
    assert (
        paths.orders[i, 0].tolist() == [0, -1, -1] and abs(np.argmax(np.abs(paths.fids[i])) * dt - 4 * delta) <= 3 * dt
    )
    # A single crusher leaves the sinc residual of the kept pathways and nothing else
    crusher = Program([Pulse("1H", np.pi / 2), Gradient(delta, z=G), Acquire("1H", dt, n)], sample=sample)
    reference = brute_force_z(
        spin_system, L, coil, rho0, Lx, sample, [("pulse", np.pi / 2), ("grad", G, delta), ("acquire", 0.0, dt, n)]
    )
    assert relative(crusher.run(spin_system, H, R, K), reference) < 2e-4
    assert np.isclose(crusher.select(spin_system).envelope[0], abs(np.sinc(gamma * G * delta * sample.z_max / np.pi)))


def test_gradient_selection_by_manual_filters():
    """Echo/antiecho pair and a three-gradient chain against coherence filters with the envelope applied by hand."""
    spin_system, pars = pair()
    H, R, K = generators(spin_system, pars)
    rho0 = spynach.state(spin_system, "Lz", "1H", "cheap")
    coil = spynach.state(spin_system, "L+", "1H", "cheap")
    Lx = spynach.operator(spin_system, "Lx", "1H")
    sample = Sample(z_max=5e-3, x_max=2.5e-3)
    gamma = spin_system.gammas[0]
    G, delta, tau, dt, n = 0.3, 1e-3, 2e-3, 1 / 400, 16
    for sign, order in ((-1, 1), (1, -1)):  # G, -G keeps +1 before the second pulse (antiecho); G, G keeps -1 (echo)
        program = Program(
            [
                Pulse("1H", np.pi / 2),
                Gradient(delta, z=G),
                Delay(tau),
                Pulse("1H", np.pi / 2),
                Gradient(delta, z=sign * G),
                Acquire("1H", dt, n),
            ],
            sample=sample,
            crush_tol=1e-9,
        )
        rho = spynach.step(spin_system, Lx, rho0, np.pi / 2)
        rho = spynach.evolution(spin_system, H, rho, delta + tau, 1, "final")
        rho = spynach.coherence(spin_system, rho, [("1H", order)])
        rho = spynach.step(spin_system, Lx, rho, np.pi / 2)
        rho = spynach.evolution(spin_system, H, rho, delta, 1, "final")
        manual = spynach.evolution(spin_system, H, rho, dt, n - 1, "observable", coil=coil)
        # the other pathways carry at least one uncompensated gradient: sinc(γGδ z_max) ~ 1e-3 of the echo
        assert relative(program.run(spin_system, H, R, K), manual) < 5e-3
        selection = program.select(spin_system)
        chosen = (selection.orders[:, 0, 1] == -order) & np.isclose(
            selection.envelope, 1
        )  # the filter uses Spinach's sign
        assert chosen.sum() == 1
    # Gradients 1:1:-2 refocus the pathways with p1 + p2 = 2; only 0, +1, +1, +1 has amplitude after a 90° pulse on Lz
    chain = Program(
        [
            Pulse("1H", np.pi / 2),
            Gradient(delta, z=G),
            Pulse("1H", np.pi / 6),
            Gradient(delta, z=G),
            Pulse("1H", np.pi / 6),
            Gradient(delta, z=-2 * G),
            Acquire("1H", dt, n),
        ],
        sample=sample,
        crush_tol=0.5,
    )
    selection = chain.select(spin_system)
    kept = selection.orders[selection.kept][:, 0, :].tolist()
    assert sorted(kept) == [[0, -2, 0, -1], [0, -1, -1, -1], [0, 0, -2, -1]]
    paths = chain.run(spin_system, H, R, K, output="pathways")
    assert paths.orders[:, 0, :].tolist() == [[0, -1, -1, -1]]
    rho = spynach.step(spin_system, Lx, rho0, np.pi / 2)
    rho = spynach.coherence(spin_system, rho, [("1H", 1)])
    rho = spynach.evolution(spin_system, H, rho, delta, 1, "final")
    rho = spynach.step(spin_system, Lx, rho, np.pi / 6)
    rho = spynach.coherence(spin_system, rho, [("1H", 1)])
    rho = spynach.evolution(spin_system, H, rho, delta, 1, "final")
    rho = spynach.step(spin_system, Lx, rho, np.pi / 6)
    rho = spynach.coherence(spin_system, rho, [("1H", 1)])
    rho = spynach.evolution(spin_system, H, rho, delta, 1, "final")
    manual = spynach.evolution(spin_system, H, rho, dt, n - 1, "observable", coil=coil)
    assert relative(chain.run(spin_system, H, R, K), manual) < 1e-10
    row = selection.orders[:, 0, :].tolist().index([0, -1, -1, -1])
    assert np.allclose(np.linalg.norm(selection.k[row], axis=1), [0, gamma * G * delta, 2 * gamma * G * delta, 0])


def test_pgse_diffusion_against_stejskal_tanner_and_ctps():
    spin_system = system(["1H"], zeeman={"scalar": [1.0]})
    pars = {"spins": ["1H"], "offset": 1.0 * MHZ_PER_T * 14.1}
    H, R, K = generators(spin_system, pars)
    gamma = spin_system.gammas[0]
    D, G, delta, Delta, dt, n = 2e-9, 0.3, 1e-3, 20e-3, 1 / 400, 8

    def pgse(diffusion, width):
        area = G * delta / width * width  # same area whatever the width
        return Program(
            [
                Pulse("1H", np.pi / 2),
                Gradient(width, z=area / width),
                Delay(Delta - width),
                Pulse("1H", np.pi),
                Gradient(width, z=area / width),
                Acquire("1H", dt, n),
            ],
            diffusion=diffusion,
        )

    ratio = abs(pgse(D, delta).run(spin_system, H, R, K)[0] / pgse(0.0, delta).run(spin_system, H, R, K)[0])
    assert np.isclose(ratio, np.exp(-D * gamma**2 * G**2 * delta**2 * (Delta - delta / 3)), rtol=1e-10)
    # ctps's narrow-gradient model agrees in the short-gradient limit, and its k-vectors always
    ctps = pytest.importorskip("ctps")
    from ctps.engine.gradient import diffusion_attenuation, effective_gamma, k_vectors

    for width, tol in ((delta, 2e-2), (1e-6, 1e-4)):
        program = pgse(D, width)
        selection = program.select(spin_system)
        sequence, cycle = program.to_ctps(spin_system)
        result = ctps.simulate(sequence, cycle, cutoff=0.0)
        echo = [i for i in range(len(result)) if result.per_nucleus["1H"][i].tolist() == [0, 1, -1]][0]
        ours = selection.orders[:, 0, :].tolist().index([0, 1, -1])
        K_ctps = k_vectors(effective_gamma(sequence, result.per_nucleus), sequence.gradient_areas())
        assert np.allclose(K_ctps[echo], selection.k[ours])
        theirs = diffusion_attenuation(K_ctps, sequence.durations(), D)[echo]
        assert abs(np.exp(-D * selection.b[ours, -1]) - theirs) / theirs < tol
    # A gradient inside an indirect dimension would need k during t1: diffusion is per column there
    program = Program(
        [
            Pulse("1H", np.pi / 2),
            Gradient(delta, z=G / 100),
            Evolve(dt, 4),
            Pulse("1H", np.pi),
            Gradient(delta, z=G / 100),
            Acquire("1H", dt, n),
        ],
        diffusion=D,
    )
    fid = program.run(spin_system, H, R, K)
    k = gamma * G / 100 * delta
    expected = np.exp(-D * k**2 * np.arange(4) * dt)
    assert np.allclose(np.abs(fid[0]) / np.abs(fid[0, 0]), expected / expected[0], rtol=1e-8)


def test_explicit_lattice_equals_the_pathway_runner():
    """Explicit mode on the Fourier-mode lattice: every gradient program of this file, with every pathway kept
    (``crush_tol=0``), gives the FID of the pathway runner to rounding; inclined axis, read gradient, diffusion,
    an indirect dimension and a phase cycle included; the per-mode output sums to the FID and puts the echo in
    mode 0."""
    spin_system, pars = pair()
    H, R, K = generators(spin_system, pars)
    sample = Sample(z_max=5e-3, x_max=2.5e-3)
    G, delta, tau, dt, n = 0.01, 1e-3, 2e-3, 1 / 8000, 32
    theta = np.radians(30)
    inclined = Gradient(delta, z=G * np.cos(theta), x=G * np.sin(theta))
    programs = [
        Program(
            [
                Pulse("1H", np.pi / 2),
                Gradient(delta, z=G),
                Delay(tau),
                Pulse("1H", np.pi / 2),
                Gradient(delta, z=G),
                Acquire("1H", dt, n, gradient=Gradient(0, z=-G / 2)),
            ],
            sample=sample,
        ),
        Program(
            [Pulse("1H", np.pi / 2), inclined, Delay(tau), Pulse("1H", np.pi / 2), inclined, Acquire("1H", dt, n)],
            sample=sample,
        ),
        Program(
            [
                Pulse("1H", np.pi / 2, phase=[0, 90, 180, 270]),
                Gradient(delta, z=G),
                Delay(tau),
                Pulse("1H", np.pi / 2),
                Gradient(delta, z=G),
                Acquire("1H", dt, n, receiver=[0, 90, 180, 270]),
            ],
            sample=sample,
        ),
        Program(
            [
                Pulse("1H", np.pi / 2),
                Gradient(delta, z=G),
                Evolve(dt, 4),
                Pulse("1H", np.pi),
                Gradient(delta, z=G),
                Acquire("1H", dt, n, gradient=Gradient(0, z=-G / 4)),
            ],
            sample=sample,
            diffusion=2e-9,
        ),
    ]
    for program in programs:
        program = replace(program, crush_tol=0.0)
        assert relative(program.run(spin_system, H, R, K), program.explicit().run(spin_system, H, R, K)) < 1e-10
    modes = replace(programs[0], crush_tol=0.0).explicit().run(spin_system, H, R, K, output="modes")
    assert relative(modes.sum(), programs[0].explicit().run(spin_system, H, R, K)) < 1e-12
    assert modes.fids.shape == (modes.modes.size, n) and np.allclose(modes.axis, [0, 0, 1])
    # the 0, +1, -1 echo sits in mode 0 at t = 0 and the read gradient moves it out; the envelope of mode 0 is 1
    zero = int(np.flatnonzero(modes.modes == 0)[0])
    assert modes.envelope[zero] == 1 and np.argmax(np.abs(modes.fids[zero])) == 0
    gamma = spin_system.gammas[0]
    assert np.isclose(modes.k0, gamma * G / 2 * dt)  # the smallest area present: the read gradient per dwell
    # the 23Na filter with gradients 1 : 1 : 4 and its equilibrium state
    spin_system, H, R, K, rho_eq = sodium()
    tqf = Program(
        [
            Pulse("23Na", np.pi / 2),
            Gradient(1e-3, z=0.5),
            Delay(2e-3),
            Pulse("23Na", np.pi, phase=(90.0,)),
            Delay(3e-3),
            Pulse("23Na", np.pi / 2, phase=(90.0,)),
            Gradient(1e-3, z=0.5),
            Pulse("23Na", np.pi / 2),
            Gradient(1e-3, z=2.0),
            Acquire("23Na", 50e-6, 32),
        ],
        rho0=rho_eq,
        crush_tol=0.0,
    )
    assert relative(tqf.run(spin_system, H, R, K), tqf.explicit().run(spin_system, H, R, K)) < 1e-10
    # gradient_unit set by hand, and the errors of the lattice
    spin_system, pars = pair()
    H, R, K = generators(spin_system, pars)
    by_hand = replace(programs[1], gradient_unit=G * delta / 2, crush_tol=0.0)
    assert relative(by_hand.run(spin_system, H, R, K), by_hand.explicit().run(spin_system, H, R, K)) < 1e-10
    with pytest.raises(ValueError, match="collinear"):
        Program(
            [Pulse("1H", np.pi / 2), Gradient(delta, z=G), Gradient(delta, x=G), Acquire("1H", dt, n)], mode="explicit"
        ).run(spin_system, H, R, K)
    with pytest.raises(ValueError, match="integer multiples"):
        Program(
            [Pulse("1H", np.pi / 2), Gradient(delta, z=G), Gradient(delta, z=G * np.sqrt(2)), Acquire("1H", dt, n)],
            mode="explicit",
        ).run(spin_system, H, R, K)
    with pytest.raises(ValueError, match="gradient_unit must be positive"):
        Program(
            [Pulse("1H", np.pi / 2), Gradient(delta, z=G), Acquire("1H", dt, n)], mode="explicit", gradient_unit=-1
        ).run(spin_system, H, R, K)
    # a gradient during 13C coherence moves it by gamma_C / gamma_H modes: not on the lattice of the detected 1H
    hetero = system(["1H", "13C"], zeeman={"scalar": [1.0, 10.0]}, coupling={"scalar": [[0, 140.0], [0, 0]]})
    H, R, K = generators(hetero, {"spins": ["1H", "13C"], "offset": [1.0 * MHZ_PER_T * 14.1, 0.0]})
    with pytest.raises(NotImplementedError, match="second isotope"):
        Program(
            [Pulse("13C", np.pi / 2), Gradient(delta, z=G), Pulse("13C", np.pi / 2), Acquire("1H", dt, n)],
            mode="explicit",
            rho0=spynach.state(hetero, "Lz", "13C"),
        ).run(hetero, H, R, K)


# ------------------------------------------------------------------------------- layer 4: limits made visible


def test_leakage_shows_when_the_branch_run_is_not_the_cycle():
    spin_system, pars = pair()
    H, R, K = generators(spin_system, pars)
    rf = H + 2 * np.pi * 50 * spynach.operator(spin_system, "Lx", "1H")  # a weak field during the delays
    rho0 = spynach.state(spin_system, "Lz", "1H", "cheap")
    coil = spynach.state(spin_system, "L+", "1H", "cheap")
    four = [0, 90, 180, 270]
    program = Program(
        [
            Pulse("1H", np.pi / 2, phase=four),
            Delay(2e-3),
            Pulse("1H", np.pi / 2),
            Acquire("1H", 1 / 400, 16, receiver=four),
        ],
        check_leakage=True,
    )
    blocks = [("pulse", "1H", np.pi / 2, 0), ("delay", 2e-3), ("pulse", "1H", np.pi / 2, 1), ("acquire", 1 / 400, 16)]
    assert (
        relative(program.run(spin_system, H, R, K), manual_cycle(spin_system, H, blocks, [four, [0]], four, rho0, coil))
        < 1e-10
    )
    selection = program.select(spin_system)
    fid = program.run(spin_system, rf, R, K, selection=selection)
    reference = manual_cycle(spin_system, rf, blocks, [four, [0]], four, rho0, coil)
    assert max(selection.leakage.values()) > 1e-3 and relative(fid, reference) > 1e-3


# --------------------------------------------------------------------------------------- layer 5: structure


def test_to_ctps_round_trip_and_errors(monkeypatch):
    ctps = pytest.importorskip("ctps")
    spin_system, pars = pair()
    program = Program(
        [
            Pulse("1H", np.pi / 2, phase=[0, 90, 180, 270]),
            Delay(1e-3, orders={"1H": [2, -2]}),
            Pulse("1H", np.pi / 2),
            Acquire("1H", 1 / 400, 8, receiver=[0, 180, 0, 180]),
        ]
    )
    sequence, cycle = program.to_ctps(spin_system)
    assert sequence.n_pulses == 2 and cycle.steps == 4 and sequence.allowed["1H"].orders(1).tolist() == [-2, 2]
    assert sequence.allowed["1H"].orders(0).tolist() == [0] and sequence.allowed["1H"].orders(2).tolist() == [-1]
    result = ctps.simulate(sequence, cycle, cutoff=0.0)
    selection = program.select(spin_system)
    assert sorted(map(tuple, result.per_nucleus["1H"].tolist())) == sorted(
        map(tuple, selection.orders[:, 0, :].tolist())
    )
    assert selection.result is result or isinstance(selection.result, ctps.PathwayResult)
    assert (
        Program([Pulse("1H", np.pi / 2), Acquire("1H", 1 / 400, 8)], max_order=1).to_ctps(spin_system).__class__
        is tuple
    )
    with pytest.raises(ValueError, match="exactly one Acquire"):
        Program([Pulse("1H", np.pi / 2)]).steps(spin_system)
    with pytest.raises(ValueError, match="mode"):
        Program([Acquire("1H", 1 / 400, 8)], mode="loop")
    with pytest.raises(ValueError, match="per-mode output"):
        program.run(spin_system, *generators(spin_system, pars), output="modes")
    with pytest.raises(ValueError, match="per-pathway output"):
        program.explicit().run(spin_system, *generators(spin_system, pars), output="pathways")
    monkeypatch.setitem(sys.modules, "ctps", None)
    with pytest.raises(ImportError, match="ctps"):
        program.select(spin_system)


def test_sample_envelope():
    sample = Sample(z_max=5e-3, x_max=2.5e-3)
    assert sample_envelope(np.zeros(3), sample) == 1
    assert np.isclose(sample_envelope(np.array([0, 0, 1000.0]), sample), np.sin(5) / 5)
    assert sample_envelope(np.array([1000.0, 0, 0]), sample) < 1


# ----------------------------------------------------------------------------------------------------------
# A triple-quantum filter for 23Na: the phase cycle and the gradients select the same pathways


def sodium():
    """One 23Na spin under slow-motion quadrupolar relaxation (C_q 200 kHz, tau_c 1 ns, 14.1 T), as the playground.

    Author: Alexej Jerschow.
    """
    from spynach.conventions import eeqq2nqi

    nqi = eeqq2nqi(200e3, 0.0, 1.5, [0, 0, 0])
    spin_system = spynach.create(
        {"isotopes": ["23Na"], "magnet": 14.1},
        {
            "coupling": {"matrix": [[nqi.tolist()]]},
            "relaxation": ["redfield"],
            "rlx_keep": "secular",
            "equilibrium": "dibari",
            "temperature": 298.0,
            "tau_c": [[1e-9]],
        },
    )
    spin_system = spynach.assume(spynach.basis(spin_system, LIOUV), "nmr")
    H = spynach.hamiltonian(spin_system)
    R = spynach.relaxation(spin_system)
    return spin_system, H, R, 0 * R, spynach.equilibrium(spin_system)


def tqf_preparation(tau, pad=0.0):
    """90x - tau/2 - 180y - tau/2 - 90y: the second 90 is 90 degrees out of phase with the first (see the guide).

    ``pad`` is a delay after the first pulse and before the last one, where the gradient version has a gradient
    and its matching delay.
    """
    return [
        Pulse("23Na", np.pi / 2),
        *([Delay(pad)] if pad else []),
        Delay(tau / 2),
        Pulse("23Na", np.pi, phase=(90.0,)),
        Delay(tau / 2),
        *([Delay(pad)] if pad else []),
        Pulse("23Na", np.pi / 2, phase=(90.0,)),
    ]


def tqf_cycle(rho_eq, offset, tau=4e-3, delta=1e-3, after=0.0, pad=0.0):
    """The six-step cycle: the preparation block stepped by 60 degrees from ``offset``, the receiver alternating.

    ``after`` is a delay before acquisition and ``pad`` one after the first pulse and before the third, where the
    gradient version has its gradients.
    """
    return Program(
        [
            Block(tqf_preparation(tau, pad), phase=[offset + 60 * n for n in range(6)]),
            Delay(delta),
            Pulse("23Na", np.pi / 2),
            *([Delay(after)] if after else []),
            Acquire("23Na", 50e-6, 64, receiver=[0, 180]),
        ],
        rho0=rho_eq,
    )


def route_sums(fids):
    """FIDs summed per route: transverse (order ±1 after the first pulse) and longitudinal (order 0)."""
    orders = fids.selection.orders[fids.index, 0, :]
    transverse = fids.fids[np.abs(orders[:, 1]) == 1].sum(axis=0)
    longitudinal = fids.fids[orders[:, 1] == 0].sum(axis=0)
    return transverse, longitudinal


def test_tqf_sodium_cycle_equals_explicit():
    spin_system, H, R, K, rho_eq = sodium()
    program = tqf_cycle(rho_eq, 30)
    assert program.steps(spin_system) == 6
    fids = program.run(spin_system, H, R, K, output="pathways")
    explicit = program.explicit().run(spin_system, H, R, K)
    assert relative(fids.sum(), explicit) < 1e-10
    # the cycle passes every Δp = ±3 over the block; only the two routes to T(3,±3) carry amplitude
    orders = fids.selection.orders[fids.index, 0, :]
    assert np.all(orders[:, -1] == -1) and np.all(np.abs(orders[:, -2]) == 3)
    amplitude = np.abs(fids.fids).max(axis=(1,))
    strong = orders[amplitude > 1e-3 * amplitude.max()]
    assert {tuple(p) for p in strong} == {
        (0, 1, -1, 3, -1),
        (0, -1, 1, 3, -1),
        (0, 1, -1, -3, -1),
        (0, -1, 1, -3, -1),
        (0, 0, 0, 3, -1),
        (0, 0, 0, -3, -1),
    }


def test_tqf_sodium_offset_selects_route():
    spin_system, H, R, K, rho_eq = sodium()
    transverse, longitudinal = route_sums(tqf_cycle(rho_eq, 30).run(spin_system, H, R, K, output="pathways"))
    scale = np.abs(transverse).max()
    assert scale > 0 and np.abs(longitudinal).max() < 1e-5 * scale
    transverse, longitudinal = route_sums(tqf_cycle(rho_eq, 0).run(spin_system, H, R, K, output="pathways"))
    assert np.abs(longitudinal).max() > 0.01 * scale and np.abs(transverse).max() < 1e-5 * scale
    # with both preparation pulses along x the transverse route makes no triple-quantum coherence at all
    blocks = list(tqf_cycle(rho_eq, 30).blocks)
    prep = list(blocks[0].blocks)
    prep[-1] = Pulse("23Na", np.pi / 2)
    both_x = Program([Block(prep, phase=blocks[0].phase), *blocks[1:]], rho0=rho_eq)
    transverse, _ = route_sums(both_x.run(spin_system, H, R, K, output="pathways"))
    assert np.abs(transverse).max() < 1e-5 * scale


def test_tqf_sodium_gradients_match_cycle():
    """Gradients 1 : 1 : 4 after the first pulse, after the third and before acquisition refocus one pathway."""
    spin_system, H, R, K, rho_eq = sodium()
    tau, delta, g = 4e-3, 1e-3, 0.5
    cycled = tqf_cycle(rho_eq, 30, tau, delta, after=delta, pad=delta).run(spin_system, H, R, K, output="pathways")
    for sign, wanted in ((1, (0, 1, -1, 3, -1)), (-1, (0, -1, 1, -3, -1))):
        program = Program(
            [
                Pulse("23Na", np.pi / 2),
                Gradient(delta, z=sign * g),
                Delay(tau / 2),
                Pulse("23Na", np.pi, phase=(90.0,)),
                Delay(tau / 2),
                Delay(delta),
                Pulse("23Na", np.pi / 2, phase=(90.0,)),
                Gradient(delta, z=sign * g),
                Pulse("23Na", np.pi / 2),
                Gradient(delta, z=4 * g),
                Acquire("23Na", 50e-6, 64),
            ],
            rho0=rho_eq,
        )
        fids = program.run(spin_system, H, R, K, output="pathways")
        selection = fids.selection
        amplitude = np.abs(fids.fids).max(axis=(1,))
        with_amplitude = amplitude > 1e-6 * amplitude.max()
        refocused = selection.envelope[fids.index] > 0.999
        # exactly one pathway with amplitude refocuses: the transverse route of the sign the gradients pick
        assert [tuple(selection.orders[i, 0, :]) for i in fids.index[refocused & with_amplitude]] == [wanted]
        # the longitudinal route, the mirror route and the magnetisation recovering during the gradients are dephased
        orders = selection.orders[fids.index, 0, :]
        k0 = np.abs(selection.k0[fids.index]).max(axis=1)
        dephased = (orders[:, 1] == 0) | (orders[:, -2] == -3 * sign) | (np.abs(orders[:, 1:-1]).sum(axis=1) == 0)
        assert np.all(k0[dephased] > 0)
        # the refocused pathway equals the same pathway of the six-step cycle once both are stripped of their
        # weights (the fixed y phases of two pulses give the uncycled program a phase per pathway as well)
        (n,) = np.flatnonzero(refocused & with_amplitude)
        i = fids.index[n]
        j = [m for m, c in enumerate(cycled.index) if tuple(cycled.selection.orders[c, 0, :]) == wanted]
        assert len(j) == 1
        reference = cycled.fids[j[0]] / np.conj(cycled.selection.pc[cycled.index[j[0]]])
        assert relative(fids.fids[n] / np.conj(selection.pc[i]), reference) < 1e-10
