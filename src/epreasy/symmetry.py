"""Point group of a spin Hamiltonian: EasySpin's ``hamsymm``.

The group decides the region of the sphere a powder integration has to cover
(``sphgrid``). With principal values and frames only, the symmetry is found
by combining the symmetries of the tensors geometrically; with full tensors
or Stevens-operator terms, by comparing the eigenvalues of the Hamiltonian
for symmetry-related field directions.

EasySpin authors, by original file (contributors to the file up to EasySpin 6.0.15, see
  docs/easyspin-credits.md):
  hamsymm.m: Stefan Stoll
"""

from __future__ import annotations

import numpy as np

from .rotations import ang2vec, erot
from .spinsystem import validatespinsys

GROUP_NAMES = ["Ci", "C2h", "D2h", "C4h", "D4h", "S6", "D3d", "C6h", "D6h", "Th", "Oh", "Dinfh", "O3"]
_O3, _DINFH, _D2H, _C2H, _CI = 0, 1, 2, 3, 4
_GEOM_GROUPS = ["O3", "Dinfh", "D2h", "C2h", "Ci"]


def hamsymm(system: dict) -> tuple[str, np.ndarray]:
    """``(point group, symmetry frame)``: the Schoenflies symbol and the matrix whose columns are the frame axes.

    EasySpin authors (hamsymm.m): Stefan Stoll.
    """
    sys = validatespinsys(system)
    high_order = bool(sys["B"])
    full_tensors = any(sys[k] for k in ("fullg", "fullA", "fullD", "fullee", "fullQ", "fullsigma", "fullnn"))
    if not full_tensors and not high_order and _isisotropic(sys):
        group, R = "O3", np.eye(3)
    elif high_order or full_tensors:
        group, R = _hamsymm_eigs(sys)
    else:
        group, R = _hamsymm_geom(sys)
    if sys.get("tdm") is not None:
        group = "C1"
    return group, R


def _isisotropic(sys: dict) -> bool:
    # EasySpin's isisotropic(): true only when every listed tensor has differing principal values (it tests the
    # opposite of its name); ported as is, so the result matches EasySpin
    def isiso(name):
        return name in sys and np.size(sys[name]) > 0 and bool(np.any(np.diff(np.atleast_2d(sys[name]), axis=1)))

    iso = isiso("g") and isiso("D") and isiso("ee") and isiso("Q") and isiso("sigma") and isiso("nn")
    if iso and "A" in sys and np.size(sys["A"]) > 0:
        for e in range(sys["nElectrons"]):
            iso = iso and bool(np.any(np.diff(sys["A"][:, 3 * e : 3 * e + 3], axis=1)))
    return iso


# ----------------------------------------------------------------------------- geometric analysis
def _tensorsymmetry(pv: np.ndarray, angles: np.ndarray) -> tuple[int, np.ndarray]:
    pv = np.asarray(pv, dtype=float).ravel()
    n_equal = int(np.sum(np.diff(np.sort(pv)) == 0))
    if n_equal == 0:
        group, z_axis = _D2H, 3
    elif n_equal == 2:
        group, z_axis = _O3, 3
    else:
        group = _DINFH
        if pv[1] == pv[2]:
            z_axis = 1
        elif pv[0] == pv[2]:
            z_axis = 2
        else:
            z_axis = 3
    idx = [2, 3, 1, 2, 3]
    frame = erot(angles).T
    cols = [idx[z_axis - 1 + k] - 1 for k in range(3)]
    return group, frame[:, cols]


def _combinesymms(sym1: int, rot1: np.ndarray, sym2: int, rot2: np.ndarray) -> tuple[int, np.ndarray]:
    delta = 0.2
    parallel = delta * np.pi / 180
    perp = (90 - delta) * np.pi / 180
    if sym1 < sym2:
        sym1, sym2 = sym2, sym1
        rot1, rot2 = rot2, rot1
    if sym1 == _CI or sym2 == _O3:
        return sym1, rot1
    if sym1 == _O3:
        return sym2, rot2
    idx = [2, 3, 1, 2, 3]
    if sym1 == _DINFH and sym2 == _DINFH:
        z1, z2 = rot1[:, 2], rot2[:, 2]
        angle = np.real(np.arccos(np.clip(abs(z1 @ z2), -1, 1)))
        if angle < parallel:
            return _DINFH, rot1
        newz = np.cross(z1, z2)
        newz = newz / np.linalg.norm(newz)
        rot = np.column_stack([np.cross(z1, newz), z1, newz])
        return (_D2H if angle > perp else _C2H), rot
    if sym1 == _D2H and sym2 == _DINFH:
        z2 = rot2[:, 2]
        angles = np.real(np.arccos(np.clip(np.abs(z2 @ rot1), -1, 1)))
        if np.any(angles < parallel):
            return _D2H, rot1
        if np.any(angles > perp):
            k = int(np.argmax(angles))
            return _C2H, rot1[:, [idx[k + j] - 1 for j in range(3)]]
        return _CI, rot1
    if sym1 == _C2H and sym2 == _DINFH:
        z1, z2 = rot1[:, 2], rot2[:, 2]
        angle = np.real(np.arccos(np.clip(abs(z1 @ z2), -1, 1)))
        return (_C2H if angle < parallel or angle > perp else _CI), rot1
    if sym1 == _D2H and sym2 == _D2H:
        all_angles = np.real(np.arccos(np.clip(np.abs(rot2.T @ rot1), -1, 1)))
        collinear = all_angles.min(axis=0) < parallel
        if np.all(collinear):
            return _D2H, rot1
        if np.any(collinear):
            k = int(np.argmin(all_angles.min(axis=0)))
            return _C2H, rot1[:, [idx[k + j] - 1 for j in range(3)]]
        return _CI, rot1
    if sym1 == _C2H and sym2 == _D2H:
        z1 = rot1[:, 2]
        angles = np.real(np.arccos(np.clip(np.abs(z1 @ rot2), -1, 1)))
        return (_C2H if np.any(angles < parallel) else _CI), rot1
    raise ValueError(f"Cannot compute symmetry combination: {sym1} {sym2}")


def _hamsymm_geom(sys: dict) -> tuple[str, np.ndarray]:
    n_el, n_nuc = sys["nElectrons"], sys["nNuclei"]
    tensors: list[tuple[int, np.ndarray]] = []
    for e in range(n_el):
        tensors.append(_tensorsymmetry(sys["g"][e], sys["gFrame"][e]))
    for e in range(n_el):
        if sys["S"][e] > 0.5:
            tensors.append(_tensorsymmetry(sys["D"][e], sys["DFrame"][e]))
    for c in range(n_el * (n_el - 1) // 2):
        tensors.append(_tensorsymmetry(sys["ee"][c], sys["eeFrame"][c]))
    for k in range(n_nuc):
        tensors.append(_tensorsymmetry(sys["sigma"][k], sys["sigmaFrame"][k]))
    for k in range(n_nuc):
        for e in range(n_el):
            tensors.append(_tensorsymmetry(sys["A"][k, 3 * e : 3 * e + 3], sys["AFrame"][k, 3 * e : 3 * e + 3]))
    for k in range(n_nuc):
        if sys["I"][k] > 0.5:
            tensors.append(_tensorsymmetry(sys["Q"][k], sys["QFrame"][k]))
    for c in range(n_nuc * (n_nuc - 1) // 2):
        tensors.append(_tensorsymmetry(sys["nn"][c], sys["nnFrame"][c]))
    group, frame = _O3, np.eye(3)
    for sym, ax in tensors:
        if sym == _O3:
            continue
        group, frame = _combinesymms(group, frame, sym, ax)
    return _GEOM_GROUPS[group], frame


# ----------------------------------------------------------------------------- quantum-mechanical analysis
def _eqeig(ea: np.ndarray, eb: np.ndarray) -> bool:
    ea, eb = np.sort(ea), np.sort(eb)
    return bool(np.linalg.norm(ea - eb) < 1e-12 * np.linalg.norm(ea))


def _hamsymm_eigs(sys: dict) -> tuple[str, np.ndarray]:
    from .hamiltonian import ham

    pa = [np.zeros(3)]
    for name in ("gFrame", "eeFrame", "DFrame"):
        if name in sys and np.size(sys[name]):
            pa.extend(np.atleast_2d(sys[name]))
    if "AFrame" in sys and np.size(sys["AFrame"]):
        pa.extend(np.asarray(sys["AFrame"]).reshape(-1, 3))
    for name in ("QFrame", "nnFrame", "sigmaFrame"):
        if name in sys and np.size(sys[name]):
            pa.extend(np.atleast_2d(sys[name]))
    pa = np.array(pa)
    _, first = np.unique(pa, axis=0, return_index=True)
    pa = pa[np.sort(first)]
    Rz = np.eye(3)
    Rx = Rz[[1, 2, 0], :]
    Ry = Rz[[2, 0, 1], :]
    rots = []
    for row in pa:
        R = erot(row).T
        rots.extend([R @ Rz, R @ Rx, R @ Ry])
    flat = np.array([r.ravel() for r in rots])
    _, first = np.unique(flat, axis=0, return_index=True)
    rots = [rots[k] for k in np.sort(first)]
    H0, mux, muy, muz = ham(sys)

    def eigs(B):
        return np.linalg.eigvalsh(H0 - B[0] * mux - B[1] * muy - B[2] * muz)

    q, qq = 7.3234, 6.3456
    the = np.pi / 180 * np.array([q, q, q, q / 2, q / 2, q, 180 - q, 180 - q, 180 - q, 90 - q, q, q])
    phi = np.pi / 180 * (np.array([0, 120, 90, 90, 0, -90, 0, -90, 180, 90, -2 * qq, 180]) + qq)
    field_vecs = 350 * ang2vec(phi, the)
    highest = 0
    R_best = Rz
    for R in rots:
        B = R @ field_vecs
        eA, eB, eC = eigs(B[:, 0]), eigs(B[:, 1]), eigs(B[:, 2])
        C3 = _eqeig(eB, eA)
        C4 = _eqeig(eC, eA)
        if not C3 and not C4:
            if not _eqeig(eA, eigs(B[:, 11])):
                pg = 1
            else:
                pg = 3 if _eqeig(eA, eigs(B[:, 10])) else 2
        elif C3 and not C4:
            if _eqeig(eA, eigs(B[:, 6])):
                if _eqeig(eC, eigs(B[:, 5])):
                    pg = 9 if _eqeig(eC, eigs(B[:, 7])) else 8
                else:
                    pg = 10
            else:
                C2x = _eqeig(eC, eigs(B[:, 7]))
                C2y = False if C2x else _eqeig(eA, eigs(B[:, 8]))
                pg = 7 if (C2x or C2y) else 6
        elif not C3 and C4:
            if _eqeig(eC, eigs(B[:, 7])):
                Bs = B[:, 9]
                C3d = _eqeig(eigs(np.array([Bs[1], Bs[2], Bs[0]])), eigs(Bs))
                pg = 11 if C3d else 5
            else:
                pg = 4
        else:
            pg = 13 if _eqeig(eC, eigs(B[:, 3])) else 12
        if pg > highest:
            highest, R_best = pg, R
    if highest == 0:
        raise ValueError("No point group found! Save input spin system and report bug!")
    return GROUP_NAMES[highest - 1], R_best
