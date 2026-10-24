"""Times epreasy against EasySpin on representative fixture cases, for the project site.

    EPREASY_FIXTURES=... python scripts/easyspin-timings.py            # writes comparison/easyspin/timings.json
    EPREASY_FIXTURES=... python scripts/easyspin-timings.py --list     # the cases and their inputs, no timing

Each case is an EasySpin fixture written by oracle/matlab/easyspin/dumps/esdump_*.m,
which record the wall clock of the EasySpin call (after a warm-up call; a fit
is timed once). The epreasy call is rebuilt from the recorded inputs by
epreasy.oracle.case_call, run once to warm up, then timed as the median of
REPEATS calls with the NumPy kernels, and again with the Numba kernels when
Numba is installed; the relative difference of the two results is recorded
with the timings. scripts/build_site.py renders the JSON into easyspin.html.
Lives in scripts/ because nothing under comparison/ or oracle/ may import epreasy.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

OUT = ROOT / "comparison" / "easyspin" / "timings.json"
REPEATS = 5
KERNEL_FUNCTIONS = {"pepper", "garlic", "salt", "esfit"}  # the ones that go through the MEX-kernel ports
SECONDS_PATHS = ("/spec/seconds", "/curry/seconds", "/result/seconds")

# (fixture case, function shown, description, playground example id or None)
CASES = [
    (
        "pp_nitroxide_matrix",
        "pepper",
        "nitroxide powder at X band, matrix diagonalisation, 1024 points",
        "ex-epr-pepper",
    ),
    ("pp_nitroxide_perturb2", "pepper", "the same spectrum by second-order perturbation theory", "ex-epr-pepper"),
    ("pp_cu4n_hybrid", "pepper", "Cu(II) with four 14N (648 states), hybrid method, 1024 points", "ex-epr-cu4n"),
    ("pp_cu4n_perturb2", "pepper", "Cu(II) with four 14N, second-order perturbation theory", "ex-epr-cu4n"),
    ("gl_nitroxide", "garlic", "isotropic nitroxide in solution, three lines", "ex-epr-garlic"),
    ("gl_cu_n", "garlic", "Cu(II) with 14N in solution, Breit-Rabi line positions", None),
    (
        "ef_pepper_simplex",
        "esfit",
        "two g values of a noisy powder spectrum fitted by Nelder-Mead simplex",
        "ex-epr-esfit",
    ),
    ("sa_n14_enh_on", "salt", "14N ENDOR powder spectrum with hyperfine enhancement, 2001 points", "ex-epr-salt"),
    ("cu_dimer_chi", "curry", "molar susceptibility of an exchange-coupled dimer at 100 temperatures", "ex-epr-curry"),
    (
        "ch_nitroxide_1e-09",
        "chili",
        "slow-motion nitroxide, tcorr 1 ns, general Liouvillian, 512 points",
        "ex-epr-chili",
    ),
    ("ch_nitroxide_1e-07", "chili", "slow-motion nitroxide, tcorr 100 ns", "ex-epr-chili"),
    ("ch_momd", "chili", "nitroxide with an ordering potential, MOMD over a director grid", None),
    ("pu_3p_14n", "saffron", "three-pulse ESEEM of 14N, 200 points", "ex-epr-saffron"),
    ("pu_hyscore_14n", "saffron", "HYSCORE of 14N, 48 x 48 points", "ex-epr-hyscore"),
]


def blas_info() -> str:
    names = []
    for name in ("numpy", "scipy"):
        try:
            config = __import__(name).show_config(mode="dicts")
            names.append(config["Build Dependencies"]["blas"]["name"])
        except Exception:
            names.append("unknown")
    return names[0] if len(set(n.lower() for n in names)) == 1 else "/".join(names)


def easyspin_seconds(fixture) -> float | None:
    for path in SECONDS_PATHS:
        if path in fixture:
            return float(np.asarray(fixture[path]).ravel()[0])
    return None


def timed(run, repeats: int) -> float:
    run()  # warm-up: imports, Numba compilation, caches
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        run()
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples)


def relative_difference(ours: np.ndarray, reference: np.ndarray) -> float:
    ours, reference = np.asarray(ours), np.asarray(reference)
    if ours.shape != reference.shape:
        ours, reference = ours.ravel(), reference.ravel()
    norm = np.linalg.norm(reference)
    return float(np.linalg.norm(ours - reference) / norm) if norm else float(np.linalg.norm(ours))


def time_case(fixture, case_call, kernels, repeats: int, *, with_numba: bool) -> dict:
    """EasySpin's recorded seconds, epreasy's median seconds with the NumPy kernels (and the Numba kernels when
    asked), and the relative difference of the two results."""
    run, reference = case_call(fixture)
    kernels.HAVE_NUMBA = False
    numpy_s = timed(run, repeats)
    rel = relative_difference(run(), reference)
    numba_s = None
    if with_numba:
        kernels.HAVE_NUMBA = True
        numba_s = timed(run, repeats)
        kernels.HAVE_NUMBA = False
    return {
        "easyspin_s": easyspin_seconds(fixture),
        "epreasy_numpy_s": numpy_s,
        "epreasy_numba_s": numba_s,
        "rel_err": rel,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--list", action="store_true", help="print the cases and their inputs without timing")
    parser.add_argument("--repeats", type=int, default=REPEATS)
    args = parser.parse_args(argv)

    import epreasy
    from epreasy import kernels
    from epreasy.oracle import case_call, find_case, load_fixture

    try:
        import numba

        numba_version = numba.__version__
    except ImportError:
        numba_version = None
    if args.list:
        for case, fn, description, example in CASES:
            path = find_case(case)
            inputs = load_fixture(path).meta["inputs"] if path else None
            print(f"{case} ({fn}): {description} [{example}]\n  {json.dumps(inputs)[:400]}")
        return 0

    rows = []
    easyspin_meta = None
    for case, fn, description, example in CASES:
        path = find_case(case)
        if path is None:
            raise SystemExit(f"fixture case {case} not found; generate it with oracle/run_easyspin_fixtures.sh")
        fixture = load_fixture(path)
        easyspin_meta = easyspin_meta or fixture.meta
        row = time_case(
            fixture, case_call, kernels, args.repeats, with_numba=bool(numba_version) and fn in KERNEL_FUNCTIONS
        )
        row.update(
            case=case, function=fn, description=description, example=example, uses_kernels=fn in KERNEL_FUNCTIONS
        )
        rows.append(row)
        es_s = "not timed" if row["easyspin_s"] is None else f"{row['easyspin_s']:.3f} s"
        nb_s = "" if row["epreasy_numba_s"] is None else f", numba {row['epreasy_numba_s']:.3f} s"
        numpy_s = row["epreasy_numpy_s"]
        print(f"{case:24s} EasySpin {es_s:>10s}  epreasy numpy {numpy_s:.3f} s{nb_s}  rel {row['rel_err']:.1e}")

    import scipy

    out = {
        "generated": time.strftime("%Y-%m-%d"),
        "repeats": args.repeats,
        "easyspin": {
            "version": easyspin_meta["easyspin_version"],
            "matlab_version": easyspin_meta["matlab_version"],
        },
        "epreasy": {
            "version": epreasy.__version__,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "numba": numba_version,
            "blas": blas_info(),
            "platform": platform.platform(),
            "cores": os.cpu_count(),
        },
        "cases": rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
