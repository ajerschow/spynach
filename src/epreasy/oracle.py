"""EasySpin reference data for the parity tests.

Fixtures are written by ``oracle/matlab/easyspin/dumps/esdump_*.m`` running
EasySpin in MATLAB, in the HDF5 layout of ``oracle/matlab/lib/oracle_write.m``,
under ``<root>/<easyspin version>/<case>/``. The root is ``$EPREASY_FIXTURES``
(default ``/easyspin-fixtures``), separate from the Spinach fixtures of
``spynach.oracle``; the reader and the comparison helpers are shared.
"""

from __future__ import annotations

import os
from pathlib import Path

from spynach.oracle.compare import assert_matches, compare
from spynach.oracle.fixture import Fixture, load_fixture

__all__ = ["DEFAULT_FIXTURE_ROOT", "Fixture", "assert_matches", "compare", "find_case", "fixture_root", "load_fixture"]

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
