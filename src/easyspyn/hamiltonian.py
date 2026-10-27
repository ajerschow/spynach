"""Spin Hamiltonian builders: ham, ham_ez, ham_nz, ham_hf, ham_zf, ham_nq, ham_ee, ham_nn, levels.

Units are EasySpin's: Hamiltonians in MHz, magnetic fields in mT, magnetic
moment operators in MHz/mT. ``ham(sys)`` returns the field-independent part
``H0`` and the moment components ``mux, muy, muz`` (molecular frame) such
that ``H = H0 - B[0]*mux - B[1]*muy - B[2]*muz`` for a field vector ``B``;
``ham(sys, B)`` returns that ``H``. Spin numbers are 1-based.

EasySpin authors, by original file (contributors to the file up to EasySpin 6.0.15, see
  docs/easyspin-credits.md):
  ham.m: Stefan Stoll
  ham_ee.m: Stefan Stoll
  ham_ez.m: Stefan Stoll
  ham_hf.m: Stefan Stoll
  ham_nn.m: Stefan Stoll
  ham_nq.m: Stefan Stoll
  ham_nz.m: Stefan Stoll
  ham_zf.m: Stefan Stoll
  levels.m: Stefan Stoll
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import scipy.sparse as sps

from .constants import bmagn, nmagn, planck
from .rotations import ang2vec, letter2vec, vec2ang
from .spinops import sop, stev
from .spinsystem import validatespinsys


def _out(H, sparse: bool):
    H = sps.csr_array(H)
    return H if sparse else H.toarray()


def _hermitise(H):
    return (H + H.conj().T) / 2


def _rotate_to_molecular(tensor: np.ndarray, angles: np.ndarray) -> np.ndarray:
    """Tensor given in its own frame (Euler angles ``angles`` of that frame) expressed in the molecular frame."""
    from .rotations import erot

    if np.any(angles):
        R_g2M = erot(angles).T
        return R_g2M @ tensor @ R_g2M.T
    return tensor


def _indices(spec, n: int, what: str):
    if spec is None or np.size(spec) == 0:
        return list(range(1, n + 1))
    idx = [int(i) for i in np.atleast_1d(spec)]
    if any(i < 1 or i > n for i in idx):
        raise ValueError(f"{what} indices invalid!")
    return idx


def ham_ez(system: dict, B=None, espins=None, *, sparse: bool = False):
    """Electron Zeeman term: ``(mux, muy, muz)`` in MHz/mT, or the Hamiltonian ``H`` for field vector ``B`` (mT).

    EasySpin authors (ham_ez.m): Stefan Stoll.
    """
    sys = validatespinsys(system)
    spins = sys["Spins"]
    n = sys["nStates"]
    idx = _indices(espins, sys["nElectrons"], "Electron spin")
    if B is not None and np.size(B) != 3:
        raise ValueError("Magnetic field vector must be a 3-element array.")
    mux = sps.csr_array((n, n))
    muy = sps.csr_array((n, n))
    muz = sps.csr_array((n, n))
    pre = -bmagn / planck * sys["g"] / 1e9  # MHz/mT
    for i in idx:
        g = pre[3 * (i - 1) : 3 * i] if sys["fullg"] else np.diag(pre[i - 1])
        g = _rotate_to_molecular(g, sys["gFrame"][i - 1])
        for k in range(3):
            Sk = sop(spins, np.array([[i, k + 1]]), sparse=True)
            mux = mux + g[0, k] * Sk
            muy = muy + g[1, k] * Sk
            muz = muz + g[2, k] * Sk
    if B is None:
        return _out(mux, sparse), _out(muy, sparse), _out(muz, sparse)
    B = np.asarray(B, dtype=float).ravel()
    return _out(-(mux * B[0] + muy * B[1] + muz * B[2]), sparse)


def ham_nz(system: dict, B=None, nspins=None, *, sparse: bool = False):
    """Nuclear Zeeman term: ``(mux, muy, muz)`` in MHz/mT, or the Hamiltonian ``H`` for field vector ``B`` (mT).

    EasySpin authors (ham_nz.m): Stefan Stoll.
    """
    sys = validatespinsys(system)
    spins = sys["Spins"]
    n = sys["nStates"]
    n_el = sys["nElectrons"]
    idx = _indices(nspins, sys["nNuclei"], "Nuclear spin")
    if B is not None and np.size(B) != 3:
        raise ValueError("Magnetic field vector must be a 3-element array.")
    mux = sps.csr_array((n, n))
    muy = sps.csr_array((n, n))
    muz = sps.csr_array((n, n))
    pre = nmagn / planck * sys["gn"] * sys["gnscale"][: sys["nNuclei"]] / 1e9  # MHz/mT
    for i in idx:
        sigma = sys["sigma"][3 * (i - 1) : 3 * i] if sys["fullsigma"] else np.diag(sys["sigma"][i - 1])
        sigma = _rotate_to_molecular(sigma, sys["sigmaFrame"][i - 1])
        for k in range(3):
            Ik = sop(spins, np.array([[n_el + i, k + 1]]), sparse=True)
            mux = mux + pre[i - 1] * sigma[0, k] * Ik
            muy = muy + pre[i - 1] * sigma[1, k] * Ik
            muz = muz + pre[i - 1] * sigma[2, k] * Ik
    if B is None:
        return _out(mux, sparse), _out(muy, sparse), _out(muz, sparse)
    B = np.asarray(B, dtype=float).ravel()
    return _out(-(mux * B[0] + muy * B[1] + muz * B[2]), sparse)


def ham_hf(system: dict, espins=None, nspins=None, *, sparse: bool = False):
    """Hyperfine Hamiltonian (MHz) between electron spins ``espins`` and nuclei ``nspins`` (default: all).

    EasySpin authors (ham_hf.m): Stefan Stoll.
    """
    sys = validatespinsys(system)
    spins = sys["Spins"]
    n = sys["nStates"]
    n_el, n_nuc = sys["nElectrons"], sys["nNuclei"]
    H = sps.csr_array((n, n))
    if n_nuc == 0:
        return _out(H, sparse)
    el = _indices(espins, n_el, "Electron spin")
    nu = _indices(nspins, n_nuc, "Nuclear spin")
    for e in el:
        eidx = slice(3 * (e - 1), 3 * e)
        for k in nu:
            if sys["I"][k - 1] == 0:
                continue
            A = sys["A"][3 * (k - 1) : 3 * k, eidx] if sys["fullA"] else np.diag(sys["A"][k - 1, eidx])
            if not np.any(A):
                continue
            A = _rotate_to_molecular(A, sys["AFrame"][k - 1, eidx])
            for c1 in range(3):
                for c2 in range(3):
                    H = H + A[c1, c2] * sop(spins, np.array([[e, c1 + 1], [n_el + k, c2 + 1]]), sparse=True)
    return _out(_hermitise(H), sparse)


def ham_zf(system: dict, espins=None, *, sparse: bool = False):
    """Zero-field splitting Hamiltonian (MHz): the ``S.D.S`` terms and the Stevens-operator terms ``B2``..``B12``.

    EasySpin authors (ham_zf.m): Stefan Stoll.
    """
    sys = validatespinsys(system)
    spins = sys["Spins"]
    n = sys["nStates"]
    idx = _indices(espins, sys["nElectrons"], "Electron spin")
    H = sps.csr_array((n, n))
    D = np.zeros((3, 3))
    for i in idx:
        D = sys["D"][3 * (i - 1) : 3 * i] if sys["fullD"] else np.diag(sys["D"][i - 1])
        if not np.any(D):
            continue
        D = _rotate_to_molecular(D, sys["DFrame"][i - 1])
        Sxyz = [sop(spins, np.array([[i, c + 1]]), sparse=True) for c in range(3)]
        for c1 in range(3):
            for c2 in range(3):
                H = H + D[c1, c2] * (Sxyz[c1] @ Sxyz[c2])
    for k, Bk in sys["B"].items():
        for i in idx:
            D_present = np.any(D)
            if D_present and k == 2:
                continue
            if not np.any(Bk[i - 1]):
                continue
            if np.any(sys["BFrame"][k][i - 1]):
                raise NotImplementedError("Tilted Stevens-operator frames (Sys.B?Frame) are not ported yet.")
            qs = range(k, -k - 1, -1)
            for iq, q in enumerate(qs):
                if Bk[i - 1, iq] != 0:
                    H = H + Bk[i - 1, iq] * stev(spins, k, q, i, sparse=True)
    return _out(_hermitise(H), sparse)


def ham_nq(system: dict, nspins=None, *, sparse: bool = False):
    """Nuclear quadrupole Hamiltonian (MHz) for nuclei ``nspins`` (default: all).

    EasySpin authors (ham_nq.m): Stefan Stoll.
    """
    sys = validatespinsys(system)
    spins = sys["Spins"]
    n = sys["nStates"]
    n_el = sys["nElectrons"]
    idx = _indices(nspins, sys["nNuclei"], "Nuclear spin")
    H = sps.csr_array((n, n))
    for i in idx:
        Q = sys["Q"][3 * (i - 1) : 3 * i] if sys["fullQ"] else np.diag(sys["Q"][i - 1])
        if not np.any(Q):
            continue
        Q = _rotate_to_molecular(Q, sys["QFrame"][i - 1])
        I = [sop(spins, np.array([[n_el + i, c + 1]]), sparse=True) for c in range(3)]
        for k in range(3):
            for q in range(3):
                H = H + I[k] @ (Q[k, q] * I[q])
    return _out(_hermitise(H), sparse)


def ham_ee(system: dict, espins=None, *, sparse: bool = False):
    """Electron-electron coupling Hamiltonian (MHz): bilinear ``S1.ee.S2`` and biquadratic ``ee2 (S1.S2)^2`` terms.

    EasySpin authors (ham_ee.m): Stefan Stoll.
    """
    sys = validatespinsys(system)
    spins = sys["Spins"]
    n = sys["nStates"]
    n_el = sys["nElectrons"]
    H = sps.csr_array((n, n))
    if n_el == 1 or (not np.any(sys["ee"]) and not np.any(sys["ee2"])):
        return _out(H, sparse)
    sel = sorted(_indices(espins, n_el, "Electron spin"))
    if len(sel) < 2:
        raise ValueError("Spins (2nd argument) must contain at least 2 values!")
    if len(set(sel)) != len(sel):
        raise ValueError("Spins (2nd argument) contains double entries!")
    all_pairs = list(combinations(range(1, n_el + 1), 2))
    for a, b in combinations(sel, 2):
        p = all_pairs.index((a, b))
        J = sys["ee"][3 * p : 3 * p + 3] if sys["fullee"] else np.diag(sys["ee"][p])
        J = _rotate_to_molecular(J, sys["eeFrame"][p])
        for c1 in range(3):
            so1 = sop(spins, np.array([[a, c1 + 1]]), sparse=True)
            for c2 in range(3):
                so2 = sop(spins, np.array([[b, c2 + 1]]), sparse=True)
                H = H + so1 @ (J[c1, c2] * so2)
        if sys["ee2"][p, 0] != 0:
            dot = sps.csr_array((n, n))
            for c in range(3):
                dot = dot + sop(spins, np.array([[a, c + 1], [b, c + 1]]), sparse=True)
            H = H + sys["ee2"][p, 0] * (dot @ dot)
    return _out(_hermitise(H), sparse)


def ham_nn(system: dict, nspins=None, *, sparse: bool = False):
    """Nuclear-nuclear coupling Hamiltonian (MHz): bilinear ``I1.nn.I2`` terms.

    EasySpin authors (ham_nn.m): Stefan Stoll.
    """
    sys = validatespinsys(system)
    spins = sys["Spins"]
    n = sys["nStates"]
    n_el, n_nuc = sys["nElectrons"], sys["nNuclei"]
    H = sps.csr_array((n, n))
    if n_nuc < 2 or not np.any(sys["nn"]):
        return _out(H, sparse)
    sel = sorted(_indices(nspins, n_nuc, "Nuclear spin"))
    if len(sel) < 2:
        raise ValueError("Spins (2nd argument) must contain at least 2 values!")
    if len(set(sel)) != len(sel):
        raise ValueError("Spins (2nd argument) contains double entries!")
    all_pairs = list(combinations(range(1, n_nuc + 1), 2))
    for a, b in combinations(sel, 2):
        p = all_pairs.index((a, b))
        J = sys["nn"][3 * p : 3 * p + 3] if sys["fullnn"] else np.diag(sys["nn"][p])
        if not np.any(J):
            continue
        J = _rotate_to_molecular(J, sys["nnFrame"][p])
        for c1 in range(3):
            so1 = sop(spins, np.array([[n_el + a, c1 + 1]]), sparse=True)
            for c2 in range(3):
                so2 = sop(spins, np.array([[n_el + b, c2 + 1]]), sparse=True)
                H = H + so1 @ (J[c1, c2] * so2)
    return _out(_hermitise(H), sparse)


def ham(system: dict, B=None, *, sparse: bool = False, split: bool = False):
    """Spin Hamiltonian of a spin system.

    ``ham(sys)`` returns ``(H0, mux, muy, muz)``: the field-independent part
    (MHz) and the magnetic moment operators (MHz/mT) along the molecular axes.
    ``ham(sys, B)`` with a field vector ``B`` (mT, molecular frame) returns
    ``H = H0 - B[0]*mux - B[1]*muy - B[2]*muz``; with ``split=True`` it returns
    ``(H0, muzL)`` with ``muzL`` the moment along ``B``, so ``H = H0 - |B| muzL``.

    EasySpin authors (ham.m): Stefan Stoll.
    """
    sys = validatespinsys(system)
    H0 = (
        ham_zf(sys, sparse=True)
        + ham_ee(sys, sparse=True)
        + ham_hf(sys, sparse=True)
        + ham_nq(sys, sparse=True)
        + ham_nn(sys, sparse=True)
    )
    mux, muy, muz = ham_ez(sys, sparse=True)
    if sys["nNuclei"] > 0:
        nx, ny, nz = ham_nz(sys, sparse=True)
        mux, muy, muz = mux + nx, muy + ny, muz + nz
    if B is None:
        if split:
            raise ValueError("split=True needs a field vector B.")
        return _out(H0, sparse), _out(mux, sparse), _out(muy, sparse), _out(muz, sparse)
    B = np.asarray(B, dtype=float).ravel()
    if B.size != 3:
        raise ValueError("Magnetic field vector must be 3-element array!")
    norm = np.linalg.norm(B)
    nB = B / norm if norm > 0 else np.zeros(3)
    muzL = nB[0] * mux + nB[1] * muy + nB[2] * muz
    if split:
        return _out(H0, sparse), _out(muzL, sparse)
    return _out(H0 - norm * muzL, sparse)


def levels(system: dict, ori=None, B=None, *, phi=None, theta=None, vectors: bool = False):
    """Energy levels (MHz, ascending) of a spin system for field orientations and magnitudes.

    ``ori`` is ``'x'``, ``'y'``, ``'z'``, ``'xy'``, ... for a field along that
    molecular direction, or an (n, 2) array of polar angles ``[phi, theta]``
    (an (n, 3) array's third column is ignored). Alternatively give ``phi``
    and ``theta`` as vectors, which are combined on a grid. ``B`` (mT) is one
    value, two values (a 101-point range) or a vector. The result has the
    axes (orientations or phi, theta), field, level, with singleton axes
    removed as in EasySpin; ``vectors=True`` also returns the eigenvectors.

    EasySpin authors (levels.m): Stefan Stoll.
    """
    sys = validatespinsys(system)
    if B is None:
        raise TypeError("The field range B (mT) is required.")
    field = np.asarray(B, dtype=float).ravel()
    if field.size == 2:
        field = np.linspace(field[0], field[1], 101)
    H0, mux, muy, muz = ham(sys)
    if phi is not None or theta is not None:
        if ori is not None:
            raise TypeError("Give either ori or phi and theta.")
        phi = np.asarray(phi, dtype=float).ravel()
        theta = np.asarray(theta, dtype=float).ravel()
        P, T = np.meshgrid(phi, theta, indexing="ij")
        zL = ang2vec(P.ravel(), T.ravel())
        lead_shape = (phi.size, theta.size)
    else:
        if ori is None:
            raise TypeError("The orientation ori (or phi and theta) is required.")
        if isinstance(ori, str):
            p, t = vec2ang(letter2vec(ori))
            ori = np.column_stack([p, t])
        ori = np.atleast_2d(np.asarray(ori, dtype=float))
        if ori.shape[1] not in (2, 3):
            raise ValueError(
                "ori must be a direction letter, an (n, 2) array [phi theta] or an (n, 3) array [phi theta chi]."
            )
        zL = ang2vec(ori[:, 0], ori[:, 1])
        lead_shape = (ori.shape[0],)
    muzL = zL[0][:, None, None] * mux + zL[1][:, None, None] * muy + zL[2][:, None, None] * muz  # (nOri, n, n)
    H = H0[None, None] - field[None, :, None, None] * muzL[:, None]  # (nOri, nB, n, n)
    if vectors:
        E, V = np.linalg.eigh(H)
        E = E.reshape(*lead_shape, field.size, -1)
        V = V.reshape(*lead_shape, field.size, V.shape[-2], V.shape[-1])
        return np.squeeze(E), np.squeeze(V)
    E = np.linalg.eigvalsh(H).reshape(*lead_shape, field.size, -1)
    return np.squeeze(E)
