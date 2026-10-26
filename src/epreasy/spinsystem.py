"""Spin system validation: EasySpin's ``validatespinsys``, ``spinvec`` and ``hsdim``.

A spin system is a dict with EasySpin's field names (``S``, ``g``, ``gFrame``,
``Nucs``, ``A``, ``AFrame``, ``D``, ``Q``, ``ee``/``J``/``dip``/``dvec``, ``nn``,
``B2``..``B12``, line widths and strains). ``validatespinsys`` returns a new
dict with every field supplemented and brought to EasySpin's processed
layout: tensor fields are 2-D arrays with one row per spin (or one 3x3
block per spin for full tensors, flagged by ``fullg``, ``fullA``, ...),
nuclei of spin zero are removed, and ``Spins``, ``nStates``, ``I`` and
``gn`` are added. Spin numbers are 1-based, as in EasySpin.

Numbers follow MATLAB's shapes: a scalar or flat list is one row, so
``g=[2.0088, 2.0061, 2.0027]`` is the three principal values of one
electron and ``A=[[16, 16, 86], [5, 5, 5]]`` is one row per nucleus.

EasySpin authors, by original file (contributors to the file up to EasySpin 6.0.15, see
  docs/easyspin-credits.md):
  hsdim.m: Stefan Stoll
  spinvec.m: Stefan Stoll
  validatespinsys.m: Stefan Stoll, Claudia Tait, Stephan Pribitzer, Joscha Nehrkorn, Matthew Krzyaniak
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .constants import gfree
from .nucdata import nucdata, nucstring2list
from .rotations import ang2vec, letter2vec

_CORRECT_FIELDS = [
    "S", "Nucs", "Abund", "n",
    "g", "g_", "gFrame", "gStrain",
    "D", "D_", "DFrame", "DStrain",
    "ee", "J", "dip", "dvec", "ee2", "eeFrame",
    "A", "A_", "AFrame", "AStrain",
    "Q", "QFrame",
    "HStrain",
    "L", "soc", "gL",
    "initState",
    "tdm",
    "lw", "lwpp", "lwEndor",
    "tcorr", "logtcorr", "Diff", "logDiff",
] + [f"B{k}" for k in range(1, 13)] + [f"CF{k}" for k in range(1, 13)]  # fmt: skip


def _m(x) -> np.ndarray:
    """A numeric field as MATLAB sees it: scalars and flat lists are one row."""
    a = np.asarray(x, dtype=float)
    if a.ndim == 0:
        return a.reshape(1, 1)
    if a.ndim == 1:
        return a.reshape(1, -1)
    return a


def _issize(a: np.ndarray, shape: tuple[int, int]) -> bool:
    return a.shape == tuple(shape)


def _sizecheck(sys: dict, field: str, shape: tuple[int, int]) -> None:
    if not _issize(sys[field], shape):
        raise ValueError(f"Spin system field {field} has wrong size for the given spins.")


def _frame(sys: dict, field: str, shape: tuple[int, int]) -> None:
    if field not in sys or np.size(sys[field]) == 0:
        sys[field] = np.zeros(shape)
    else:
        sys[field] = _m(sys[field])
    _sizecheck(sys, field, shape)


def hsdim(spins) -> int:
    """Dimension of the Hilbert space of a vector of spin quantum numbers or of a spin system.

    EasySpin authors (hsdim.m): Stefan Stoll.
    """
    if isinstance(spins, dict):
        full = validatespinsys(spins)
        n_equiv = full["n"]
        return int(
            np.prod(2 * full["S"] + 1)
            * np.prod((2 * full["I"] + 1) ** n_equiv)
            * np.prod(2 * np.asarray(full["L"]) + 1)
        )
    return int(np.prod(2 * np.asarray(spins, dtype=float) + 1))


def spinvec(system: dict) -> np.ndarray:
    """Vector of all spin quantum numbers ``[S..., I...]`` of a spin system.

    EasySpin authors (spinvec.m): Stefan Stoll.
    """
    return validatespinsys(system)["Spins"]


def validatespinsys(sys_in: dict) -> dict[str, Any]:  # noqa: C901
    """Check a spin system and return it in EasySpin's processed layout (see the module docstring).

    EasySpin authors (validatespinsys.m): Stefan Stoll, Claudia Tait, Stephan Pribitzer, Joscha Nehrkorn,
      Matthew Krzyaniak.
    """
    if not isinstance(sys_in, dict):
        raise TypeError("Spin system must be a dict.")
    if sys_in.get("processed"):
        return sys_in
    sys: dict[str, Any] = dict(sys_in)

    for given in list(sys):
        if given == "ZeemanFreq":
            raise ValueError("Field Sys.ZeemanFreq can only be used in conjunction with the function spidyan.")
        for correct in _CORRECT_FIELDS:
            if given.lower() == correct.lower() and given != correct:
                raise ValueError(f"Fix capitalization: Sys.{given} should be Sys.{correct}")
    if any(f.lower().startswith("ham") for f in sys):
        raise NotImplementedError("Higher-order Zeeman terms (Sys.Ham...) are not ported (EasySpin's ham_ezho).")

    # Electron spins
    S = np.atleast_1d(np.asarray(sys.get("S", 0.5), dtype=float)).ravel()
    if S.size == 0 or np.any(np.mod(S, 0.5)) or np.any(S < 0):
        raise ValueError("Electron spins in S must be positive integer multiples of 1/2.")
    if np.any(S == 0):
        raise ValueError("Systems with zero spin are not allowed.")
    sys["S"] = S
    n_el = S.size
    sys["nElectrons"] = n_el

    # g tensors
    if "g" in sys:
        if "g_" in sys:
            raise ValueError("Sys.g and Sys.g_ are given. Remove one of them.")
        g = _m(sys["g"])
        sys["fullg"] = _issize(g, (3 * n_el, 3))
        if sys["fullg"]:
            pass
        elif g.size == n_el:
            g = g.reshape(-1, 1) * np.ones((1, 3))
        elif _issize(g, (n_el, 2)):
            g = g[:, [0, 0, 1]]
        elif _issize(g, (n_el, 3)):
            pass
        else:
            raise ValueError("Sys.g has wrong size.")
        sys["g"] = g
    elif "g_" in sys:
        sys["fullg"] = False
        g_ = _m(sys["g_"])
        if g_.size == n_el:
            g_ = np.hstack([g_.reshape(-1, 1), np.zeros((n_el, 2))])
        elif _issize(g_, (n_el, 2)):
            g_ = np.hstack([g_, np.zeros((n_el, 1))])
        elif not _issize(g_, (n_el, 3)):
            raise ValueError("Sys.g_ has wrong size.")
        sys["g_"] = g_
        sys["g"] = g_[:, [0]] + g_[:, [1]] * np.array([[-1, -1, 2]]) + g_[:, [2]] * np.array([[-1, 1, 0]])
    else:
        sys["fullg"] = False
        sys["g"] = gfree * np.ones((n_el, 3))
    if "gpa" in sys:
        raise ValueError("Obsolete field Sys.gpa. Use Sys.gFrame instead.")
    _frame(sys, "gFrame", (n_el, 3))

    # Zero-field splittings
    if "D" in sys and "D_" in sys:
        raise ValueError("Zero-field splitting given in both Sys.D and Sys.D_. Please remove one of them.")
    if "D" in sys:
        D = _m(sys["D"])
        sys["fullD"] = _issize(D, (3 * n_el, 3))
        if sys["fullD"]:
            pass
        elif D.size == n_el:
            D = D.reshape(-1, 1) * np.array([[-1 / 3, -1 / 3, 2 / 3]])
        elif _issize(D, (n_el, 2)):
            D = D[:, [0]] * np.array([[-1 / 3, -1 / 3, 2 / 3]]) + D[:, [1]] * np.array([[1, -1, 0]])
        elif not _issize(D, (n_el, 3)):
            raise ValueError("Sys.D has wrong size.")
        sys["D"] = D
    elif "D_" in sys:
        sys["fullD"] = False
        D_ = _m(sys["D_"])
        if not _issize(D_, (n_el, 2)):
            raise ValueError("Sys.D_ has wrong size.")
        Dv = D_[:, [0]]
        Ev = Dv * D_[:, [1]]
        sys["D"] = Dv * np.array([[-1 / 3, -1 / 3, 2 / 3]]) + Ev * np.array([[1, -1, 0]])
    else:
        sys["D"] = np.zeros((n_el, 3))
        sys["fullD"] = False
    if "Dpa" in sys:
        raise ValueError("Obsolete field Sys.Dpa. Use Sys.DFrame instead.")
    _frame(sys, "DFrame", (n_el, 3))

    # High-order zero-field terms B1..B12
    if "aF" in sys or "aFFrame" in sys:
        raise ValueError("Sys.aF and Sys.aFFrame are no longer supported. Use Sys.B4 and Sys.B4Frame instead.")
    if "BFrame" in sys:
        raise ValueError("Sys.BFrame is not supported. Use Sys.B1Frame, Sys.B2Fame, etc instead.")
    B: dict[int, np.ndarray] = {}
    BFrame: dict[int, np.ndarray] = {}
    D_present = bool(np.any(sys["D"]))
    for k in range(1, 13):
        name = f"B{k}"
        if name not in sys:
            continue
        Bk = _m(sys[name])
        if k == 2 and np.any(Bk) and D_present:
            raise ValueError("Cannot use Sys.D and Sys.B2 simultaneously. Remove one of them.")
        if Bk.shape[0] != n_el:
            raise ValueError(f"Field Sys.{name} has to have {n_el} rows, since there are {n_el} electron spins.")
        if Bk.shape[1] == 1:
            Bk = np.hstack([np.zeros((n_el, k)), Bk, np.zeros((n_el, k))])
        elif Bk.shape[1] != 2 * k + 1:
            raise ValueError(f"Field Sys.{name} has {Bk.shape[1]} columns, but {2 * k + 1} are required.")
        if np.iscomplexobj(sys[name]):
            raise ValueError(f"Field Sys.{name} contains complex numbers. Only real ones are.")
        sys[name] = Bk
        B[k] = Bk
        frame = f"{name}Frame"
        if frame in sys and np.any(np.asarray(sys[frame])):
            BFrame[k] = _m(sys[frame])
            if not _issize(BFrame[k], (n_el, 3)):
                raise ValueError(f"Spin system field {frame} has wrong size for the given spins.")
        else:
            BFrame[k] = np.zeros((n_el, 3))
    sys["B"] = B
    sys["BFrame"] = BFrame

    # Electron-electron couplings
    sys.setdefault("fullee", False)
    if n_el > 1:
        n_pairs = n_el * (n_el - 1) // 2
        ee_matrix = "ee" in sys
        jdd = any(np.size(sys.get(f, [])) > 0 for f in ("J", "dvec", "dip"))
        if not ee_matrix and not jdd:
            raise ValueError(
                "Spin system contains 2 or more electron spins, but coupling terms are missing (ee; or J, dip, dvec)!"
            )
        if ee_matrix and jdd:
            raise ValueError("Both Sys.ee and (Sys.J,Sys.dip,Sys.dvec) are given - use only one or the other!")
        if ee_matrix:
            ee = _m(sys["ee"])
            if ee.size == n_pairs:
                ee = ee.reshape(-1, 1) * np.ones((1, 3))
            sys["fullee"] = _issize(ee, (3 * n_pairs, 3))
            sys["ee"] = ee
            if not sys["fullee"]:
                _sizecheck(sys, "ee", (n_pairs, 3))
        else:
            J = np.asarray(sys.get("J", np.zeros(n_pairs)), dtype=float).reshape(-1, 1)
            sys["J"] = J
            _sizecheck(sys, "J", (n_pairs, 1))
            dvec = _m(sys["dvec"]) if "dvec" in sys else np.zeros((n_pairs, 3))
            sys["dvec"] = dvec
            _sizecheck(sys, "dvec", (n_pairs, 3))
            dip = _m(sys["dip"]) if "dip" in sys else np.zeros((n_pairs, 3))
            if dip.size == n_pairs:
                dip = dip.reshape(-1, 1)
            if dip.shape[0] != n_pairs:
                raise ValueError(
                    f"Sys.dip must contain {n_pairs} rows, since there are {n_pairs} unique pairs of electron spins."
                )
            if "eeD" in sys:
                raise ValueError("Sys.eeD is obsolete. Use Sys.dip instead.")
            if dip.shape[1] == 1:
                dip = dip * np.array([[1, 1, -2]])
            elif dip.shape[1] == 2:
                dip = dip[:, [0]] * np.array([[1, 1, -2]]) + dip[:, [1]] * np.array([[1, -1, 0]])
            elif dip.shape[1] == 3:
                dip = dip - dip.mean(axis=1, keepdims=True)
            else:
                raise ValueError("Sys.dip must contain 1, 2, or 3 columns.")
            sys["dip"] = dip
            sys["fullee"] = bool(np.any(dvec != 0))
            if sys["fullee"]:
                blocks = []
                for p in range(n_pairs):
                    d = dvec[p]
                    anti = np.array([[0, d[2], -d[1]], [-d[2], 0, d[0]], [d[1], -d[0], 0]])
                    blocks.append(J[p, 0] * np.eye(3) + anti + np.diag(dip[p]))
                sys["ee"] = np.vstack(blocks)
            else:
                sys["ee"] = J + dip
        if "eepa" in sys:
            raise ValueError("Obsolete field Sys.eepa. Use Sys.eeFrame instead.")
        _frame(sys, "eeFrame", (n_pairs, 3))
        ee2 = np.asarray(sys.get("ee2", np.zeros(n_pairs)), dtype=float).reshape(-1, 1)
        sys["ee2"] = ee2
        _sizecheck(sys, "ee2", (n_pairs, 1))

    # Nuclear spins
    nucs = nucstring2list(sys.get("Nucs", ""))
    if not nucs and any(np.size(sys.get(f, [])) > 0 for f in ("A", "AFrame", "Q", "QFrame")):
        raise ValueError("The system contains A and/or Q fields, but no nucleus is specified!")
    sys["Nucs"] = nucs
    data = nucdata(nucs)
    I, gn = data.spin, data.gn
    sys["I"], sys["gn"] = I, gn
    n_nuc = I.size
    sys["nNuclei"] = n_nuc
    if "n" in sys and np.size(sys["n"]) > 0:
        sys["n"] = _m(sys["n"])
        _sizecheck(sys, "n", (1, n_nuc))
        sys["n"] = sys["n"].ravel()
    else:
        sys["n"] = np.ones(n_nuc)
    if "gnscale" in sys:
        sys["gnscale"] = np.asarray(sys["gnscale"], dtype=float).ravel()
        if sys["gnscale"].size < n_nuc:
            raise ValueError("Incorrect number of elements in Sys.gnscale.")
    else:
        sys["gnscale"] = np.ones(n_nuc)

    # Chemical shielding
    if "sigma" in sys:
        sigma = _m(sys["sigma"])
        sys["fullsigma"] = _issize(sigma, (3 * n_nuc, 3))
        if sys["fullsigma"]:
            pass
        elif sigma.size == n_nuc:
            sigma = sigma.reshape(-1, 1) * np.ones((1, 3))
        elif _issize(sigma, (n_nuc, 2)):
            sigma = sigma[:, [0, 0, 1]]
        elif not _issize(sigma, (n_nuc, 3)):
            raise ValueError("Sys.sigma has wrong size.")
        sys["sigma"] = sigma
    else:
        sys["fullsigma"] = False
        sys["sigma"] = np.ones((n_nuc, 3))
    _frame(sys, "sigmaFrame", (n_nuc, 3))

    # Hyperfine couplings
    sys["fullA"] = False
    if n_nuc > 0:
        if "A" not in sys and "A_" not in sys:
            raise ValueError(
                f"No hyperfine tensors A given for the {n_el} electron spins and {n_nuc} nuclei in the system!"
            )
        if "A" in sys and "A_" in sys:
            raise ValueError("Hyperfine data given in both Sys.A and Sys.A_. Please remove one of them.")
        if "A_" in sys:
            A_ = _m(sys["A_"])
            if _issize(A_, (1, n_nuc)):
                A_ = np.hstack([A_.T, np.zeros((n_nuc, 2))])
            elif _issize(A_, (n_nuc, n_el)):
                A_ = np.hstack([np.column_stack([A_[:, e], np.zeros(n_nuc), np.zeros(n_nuc)]) for e in range(n_el)])
            elif _issize(A_, (n_nuc, 2 * n_el)):
                A_ = np.hstack(
                    [np.column_stack([A_[:, 2 * e], A_[:, 2 * e + 1], np.zeros(n_nuc)]) for e in range(n_el)]
                )
            elif not _issize(A_, (n_nuc, 3 * n_el)):
                raise ValueError("Wrong size of the A_ hyperfine array in the spin system.")
            sys["A_"] = A_
            A = np.zeros((n_nuc, 3 * n_el))
            for e in range(n_el):
                sph = A_[:, 3 * e : 3 * e + 3]
                A[:, 3 * e : 3 * e + 3] = (
                    sph[:, [0]] + np.array([[-1, -1, 2]]) * sph[:, [1]] + np.array([[-1, 1, 0]]) * sph[:, [2]]
                )
            sys["A"] = A
        else:
            A = _m(sys["A"])
            if _issize(A, (3 * n_nuc, 3 * n_el)):
                sys["fullA"] = True
            elif _issize(A, (1, n_nuc)):
                if n_el != 1:
                    raise ValueError("Size of Sys.A matrix is inconsistent with number of electrons and nuclei.")
                A = A.reshape(-1, 1) * np.ones((1, 3))
            elif _issize(A, (n_nuc, n_el)):
                A = np.kron(A, np.ones((1, 3)))
            elif _issize(A, (n_nuc, 2 * n_el)):
                idx = [0, 0, 1]
                for k in range(2, n_el + 1):
                    idx += [2 * k - 2, 2 * k - 2, 2 * k - 1]
                A = A[:, idx]
            elif not _issize(A, (n_nuc, 3 * n_el)):
                raise ValueError(
                    f"Size of Sys.A ({A.shape[0]}x{A.shape[1]}) is inconsistent with number of nuclei ({n_nuc}) "
                    f"and electrons ({n_el})."
                )
            sys["A"] = A
        if "Apa" in sys:
            raise ValueError("Obsolete field Sys.Apa. Use Sys.AFrame instead.")
        _frame(sys, "AFrame", (n_nuc, 3 * n_el))

    # Nuclear quadrupole couplings
    sys["fullQ"] = False
    if n_nuc > 0:
        if "Q" not in sys:
            sys["Q"] = np.zeros((n_nuc, 3))
        else:
            Q = _m(sys["Q"])
            sys["fullQ"] = _issize(Q, (3 * n_nuc, 3))
            if not sys["fullQ"]:
                if Q.size == n_nuc:
                    Q = np.hstack([Q.reshape(-1, 1), np.zeros((n_nuc, 1))])
                if _issize(Q, (n_nuc, 2)):
                    Q3 = np.zeros((n_nuc, 3))
                    for k in range(n_nuc):
                        if I[k] >= 1:
                            eeqQh, eta = Q[k]
                            Q3[k] = eeqQh / (4 * I[k] * (2 * I[k] - 1)) * np.array([-1 + eta, -1 - eta, 2])
                    Q = Q3
                sys["Q"] = Q
                _sizecheck(sys, "Q", (n_nuc, 3))
            else:
                sys["Q"] = Q
                for k in range(n_nuc):
                    Qk = Q[3 * k : 3 * k + 3]
                    if np.any(Qk) and np.linalg.norm(Qk - Qk.T) / np.linalg.norm(Qk) > 1e-5:
                        raise ValueError(
                            "Sys.Q contains asymmetric full Q matrix. Only symmetric Q matrices are allowed."
                        )
        if "Qpa" in sys:
            raise ValueError("Obsolete field Sys.Qpa. Use Sys.QFrame instead.")
        _frame(sys, "QFrame", (n_nuc, 3))

    # Nuclear-nuclear couplings
    sys["fullnn"] = False
    if n_nuc < 2:
        if "nn" in sys and np.size(sys["nn"]) > 0 and np.any(np.asarray(sys["nn"])):
            raise ValueError("Nuclear-nuclear couplings specified in Sys.nn, but fewer than two nuclei given.")
    else:
        n_npairs = n_nuc * (n_nuc - 1) // 2
        if "nn" in sys and np.size(sys["nn"]) > 0 and np.any(np.asarray(sys["nn"])):
            nn = _m(sys["nn"])
            if nn.size == n_npairs:
                nn = nn.reshape(-1, 1) * np.ones((1, 3))
            sys["fullnn"] = _issize(nn, (3 * n_npairs, 3))
            sys["nn"] = nn
            if not sys["fullnn"]:
                _sizecheck(sys, "nn", (n_npairs, 3))
        else:
            sys["nn"] = np.zeros((n_npairs, 3))
            sys["nnFrame"] = np.zeros((n_npairs, 3))
        _frame(sys, "nnFrame", (n_npairs, 3))

    # Remove spin-zero nuclei
    rmv = I == 0
    if np.any(rmv):
        keep = ~rmv
        sys["Nucs"] = [n for n, k in zip(nucs, keep) if k]
        sys["I"], sys["gn"] = I[keep], gn[keep]
        sys["nNuclei"] = int(keep.sum())
        sys["gnscale"] = sys["gnscale"][keep]
        sys["n"] = sys["n"][keep]
        if sys["fullA"]:
            sys["A"] = sys["A"][np.repeat(keep, 3)]
        else:
            sys["A"] = sys["A"][keep]
            sys["AFrame"] = sys["AFrame"][keep]
        if sys["fullQ"]:
            sys["Q"] = sys["Q"][np.repeat(keep, 3)]
        else:
            sys["Q"] = sys["Q"][keep]
            sys["QFrame"] = sys["QFrame"][keep]

    # Line widths
    for field in ("lw", "lwEndor", "lwpp"):
        value = np.asarray(sys.get(field, [0, 0]), dtype=float).ravel()
        if value.size == 1:
            value = np.array([value[0], 0.0])
        if value.size != 2:
            raise ValueError(f"System.{field} has wrong size.")
        if np.any(value < 0):
            raise ValueError(
                "Linewidths cannot be negative." if field == "lwpp" else f"System.{field} cannot be negative."
            )
        sys[field] = value
    if np.any(sys["lwpp"]):
        if np.any(sys["lw"]):
            raise ValueError("System.lw and Sys.lwpp cannot be used simultaneously.")
        sys["lw"] = sys["lwpp"] * np.array([np.sqrt(2 * np.log(2)), np.sqrt(3)])
        sys["lwpp"] = np.zeros(2)

    # Strains
    gs = _m(sys["gStrain"]) if np.size(sys.get("gStrain", [])) > 0 else np.zeros((n_el, 3))
    if gs.shape[0] != n_el:
        raise ValueError(f"Sys.gStrain must have {n_el} rows, one per electron spin!")
    if gs.shape[1] == 1:
        gs = gs[:, [0, 0, 0]]
    elif gs.shape[1] == 2:
        gs = gs[:, [0, 0, 1]]
    elif gs.shape[1] != 3:
        raise ValueError("Sys.gStrain must have 1, 2, or 3 columns!")
    if np.any(gs < 0):
        raise ValueError("Sys.gStrain must contain nonnegative values!")
    sys["gStrain"] = gs
    ds = _m(sys["DStrain"]) if np.size(sys.get("DStrain", [])) > 0 else np.zeros((n_el, 3))
    sys["DStrainCorr"] = np.asarray(sys.get("DStrainCorr", np.zeros(n_el)), dtype=float).ravel()
    if ds.shape[0] != n_el:
        raise ValueError(f"Sys.DStrain must have {n_el} rows, one per electron spin!")
    if ds.shape[1] in (1, 2):
        ds = np.hstack([ds, np.zeros((n_el, 3 - ds.shape[1]))])
    elif ds.shape[1] != 3:
        raise ValueError("Sys.DStrain must have 1, 2, or 3 columns!")
    if sys["DStrainCorr"].size != n_el:
        raise ValueError(f"Sys.DStrainCorr must contain {n_el} elements, since you have {n_el} electron spins")
    if np.any(ds < 0):
        raise ValueError("Sys.DStrain must contain nonnegative values!")
    if np.any(sys["DStrainCorr"] < -1) or np.any(sys["DStrainCorr"] > 1):
        raise ValueError("D-E strain correlation coefficient in Sys.DStrainCorr must be between -1 and +1.")
    sys["DStrain"] = ds
    for field, n_elem in (("HStrain", 3), ("AStrain", 3)):
        if np.size(sys.get(field, [])) == 0:
            sys[field] = np.zeros(n_elem)
            continue
        p = np.asarray(sys[field], dtype=float).ravel()
        if p.size == 1:
            p = p[[0, 0, 0]]
        if p.size == 2:
            p = p[[0, 0, 1]]
        if p.size != n_elem:
            raise ValueError(f"Sys.{field} must have {n_elem} elements!")
        if np.any(p < 0):
            raise ValueError(f"Sys.{field} must contain nonnegative values!")
        sys[field] = p

    # Non-equilibrium state and transition dipole moment: kept, used by later simulators
    if "Pop" in sys:
        raise ValueError(
            "Sys.Pop is obsolete. Use Sys.initState to specify a non-equilibrium state for the spin system."
        )
    sys.setdefault("initState", None)
    tdm = sys.get("tdm")
    if tdm is not None and np.size(tdm) > 0:
        if isinstance(tdm, str):
            sys["tdm"] = letter2vec(tdm)
        elif np.size(tdm) == 3:
            v = np.asarray(tdm, dtype=float).ravel()
            sys["tdm"] = v / np.linalg.norm(v)
        elif np.size(tdm) == 2:
            sys["tdm"] = ang2vec(tdm[0], tdm[1]).ravel()
        else:
            raise ValueError("tdm must be a letter designating a direction, a 3-vector, or an array with two angles.")
    else:
        sys["tdm"] = None

    # Rotational diffusion
    for field in ("Diff", "logDiff", "tcorr", "logtcorr"):
        if field in sys and np.size(sys[field]) not in (0, 1, 2, 3):
            raise ValueError(f"Sys.{field} must have 1, 2, or 3 elements.")
    if "DiffFrame" in sys:
        sys["DiffFrame"] = _m(sys["DiffFrame"])
        _sizecheck(sys, "DiffFrame", (1, 3))

    # Orbital angular momentum: not ported
    if np.size(sys.get("L", [])) > 0:
        raise NotImplementedError("Orbital angular momentum (Sys.L, soc, gL, CF*) is not ported.")
    for field in ("gL", "soc"):
        if np.size(sys.get(field, [])) > 0:
            raise ValueError(f"Sys.{field} is given, but Sys.L is missing. Specify Sys.L.")
    for k in range(1, 13):
        if np.size(sys.get(f"CF{k}", [])) > 0:
            raise ValueError(f"Sys.CF{k} is given, but Sys.L is missing. Specify Sys.L.")
    sys["L"] = np.zeros(0)
    sys["gL"] = np.zeros(0)
    sys["nL"] = 0

    sys["Spins"] = np.concatenate([sys["S"], sys["I"], sys["L"]])
    sys["nStates"] = int(np.prod(2 * sys["Spins"] + 1))
    sys["processed"] = True
    return sys
