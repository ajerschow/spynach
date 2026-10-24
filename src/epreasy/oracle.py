"""EasySpin reference data for the parity tests.

Fixtures are written by ``oracle/matlab/easyspin/dumps/esdump_*.m`` running
EasySpin in MATLAB, in the HDF5 layout of ``oracle/matlab/lib/oracle_write.m``,
under ``<root>/<easyspin version>/<case>/``. The root is ``$EPREASY_FIXTURES``
(default ``/easyspin-fixtures``), separate from the Spinach fixtures of
``spynach.oracle``; the reader and the comparison helpers are shared.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

import numpy as np

from spynach.oracle.compare import assert_matches, compare
from spynach.oracle.fixture import Fixture, load_fixture

__all__ = [
    "DEFAULT_FIXTURE_ROOT",
    "Fixture",
    "assert_matches",
    "case_call",
    "compare",
    "find_case",
    "fixture_root",
    "load_fixture",
]

DEFAULT_FIXTURE_ROOT = Path("/easyspin-fixtures")


def fixture_root() -> Path:
    """Root of the EasySpin fixture tree: ``$EPREASY_FIXTURES`` or ``/easyspin-fixtures``."""
    return Path(os.environ.get("EPREASY_FIXTURES", DEFAULT_FIXTURE_ROOT))


def find_case(case: str, version: str | None = None) -> Path | None:
    """Locate ``<root>/<version>/<case>``; with no version, the case must exist under one version only."""
    root = fixture_root()
    if version is not None:
        path = root / version / case
        return path if path.is_dir() else None
    hits = sorted(p for p in root.glob(f"*/{case}") if p.is_dir())
    if len(hits) > 1:
        raise LookupError(f"case {case!r} exists under several EasySpin versions, pass version=: {hits}")
    return hits[0] if hits else None


def _lists(d: dict, keys: tuple[str, ...]) -> dict:
    """meta.json stores a one-element MATLAB vector as a scalar; these fields are lists in epreasy."""
    out = dict(d or {})
    for key in keys:
        if key in out:
            out[key] = np.atleast_1d(out[key]).tolist()
    return out


def _complex(obj):
    """meta.json carries complex numbers as {"real":, "imag":} pairs."""
    if isinstance(obj, dict) and set(obj) == {"real", "imag"}:
        return complex(obj["real"], obj["imag"])
    if isinstance(obj, dict):
        return {k: _complex(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_complex(v) for v in obj]
    return obj


def _tuples(obj):
    """EasySpin cell arrays ``initState {state, basis}`` and the vary cell ``{state}`` become tuples."""
    if isinstance(obj, dict):
        return {
            k: (
                tuple(v)
                if k == "initState" and isinstance(v, list) and (len(v) == 1 or isinstance(v[-1], str))
                else _tuples(v)
            )
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [_tuples(v) for v in obj]
    return obj


def case_call(fixture: Fixture) -> tuple[Callable[[], np.ndarray], np.ndarray]:
    """The epreasy call of a fixture case rebuilt from its recorded inputs: ``(run, reference)``.

    ``run()`` executes the simulator on the inputs of ``meta.json`` and returns the array the parity tests
    compare (the spectrum, the first curry output, the fitted parameters); ``reference`` is EasySpin's.
    Serves the timing script of the project page; the parity tests keep their own, finer comparisons."""
    import epreasy as es

    inputs = fixture.meta["inputs"]
    fn = inputs.get("fn") or ("esfit" if "fcn" in inputs else None)  # esfit cases record the model as "fcn"
    sysd, exp, opt = inputs.get("Sys"), inputs.get("Exp"), inputs.get("Opt") or {}
    if fn in ("pepper", "garlic"):
        system = _lists(sysd, ("n",)) if isinstance(sysd, dict) else sysd
        opt = _lists(opt, ("HybridCoreNuclei", "GridSize"))
        simulator = es.pepper if fn == "pepper" else es.garlic
        return lambda: np.asarray(simulator(system, exp, opt).spec), np.asarray(fixture["/spec/spec"])
    if fn == "salt":
        opt = _lists(opt, ("Nuclei", "Spins", "GridSize"))
        return lambda: np.asarray(es.salt(sysd, exp, opt).spec), np.asarray(fixture["/spec/spec"])
    if fn == "curry":
        opt = _lists(opt, ("Spins",))

        def run_curry():
            out = es.curry(sysd, exp, opt)
            return np.asarray(out[0] if isinstance(out, tuple) else out)

        return run_curry, np.asarray(fixture["/curry/out1"])
    if fn == "chili":
        opt = dict(opt)
        opt.pop("LiouvMethod", None)  # the general method is the ported one
        system = _complex(sysd)
        if isinstance(system, dict) and "Potential" in system:
            system["Potential"] = np.atleast_2d(np.asarray(system["Potential"], dtype=complex))
        return lambda: np.asarray(es.chili(system, exp, opt).spec), np.asarray(fixture["/spec/spec"])
    if fn == "saffron":
        exp = dict(exp)
        if not isinstance(exp["Sequence"], str):
            exp["Sequence"] = [s if isinstance(s, dict) else float(np.asarray(s).ravel()[0]) for s in exp["Sequence"]]
        return lambda: np.asarray(es.saffron(sysd, exp, opt).spec), np.asarray(fixture["/spec/y"])
    if fn in ("endorfrq", "endorfrq_perturb"):
        opt = _lists(opt, ("Nuclei", "GridSize"))
        simulator = es.endorfrq if fn == "endorfrq" else es.endorfrq_perturb
        return lambda: np.asarray(simulator(sysd, exp, opt).pos), np.asarray(fixture["/res/pos"])
    if fn == "esfit":
        if inputs["fcn"] != "pepper":
            raise ValueError(
                "case_call rebuilds esfit cases of pepper models only; the analytic models live in the tests"
            )
        data = []
        k = 1
        while f"/input/data{k}" in fixture:
            data.append(np.asarray(fixture[f"/input/data{k}"]).ravel())
            k += 1
        fit_opt = dict(inputs["FitOpt"])
        method = fit_opt.get("Method", "")
        # EasySpin 6.0.15 swaps its 'diff' and 'fft' targets; epreasy uses the names as meant
        if "diff" in method.split():
            fit_opt["Method"] = method.replace("diff", "fft")
        elif "fft" in method.split():
            fit_opt["Method"] = method.replace("fft", "diff")
        if "Mask" in fit_opt:
            fit_opt["Mask"] = np.asarray(fit_opt["Mask"]).ravel().astype(bool)

        def arg(key):
            val = inputs.get(key)
            return None if val is None or (isinstance(val, list) and len(val) == 0) else _tuples(val)

        p0, vary, lb, ub = arg("p0"), arg("vary"), arg("lb"), arg("ub")
        one = data if len(data) > 1 else data[0]

        def run_fit():
            if vary is not None:
                return np.asarray(es.esfit(one, es.pepper, p0, vary, fit_opt).pfit)
            return np.asarray(es.esfit(one, es.pepper, p0, lb, ub, fit_opt).pfit)

        return run_fit, np.asarray(fixture["/result/pfit"]).ravel()
    raise ValueError(f"no case_call for fixture function {fn!r}")
