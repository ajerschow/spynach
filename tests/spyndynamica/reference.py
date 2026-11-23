"""Loader for the SpinDynamica reference fixtures in ``tests/spyndynamica/fixtures/data``.

The files are written by ``fixtures/generate.wls`` (see ``fixtures/README.md``). A fixture records what
SpinDynamica returns, which is not always what is right: ``porting/spyndynamica/DIVERGENCES.md`` lists the
known differences.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np

FIXTURES = Path(__file__).resolve().parent / "fixtures"
CASES = FIXTURES / "cases"
DATA = FIXTURES / "data"


@dataclass(frozen=True)
class Symbolic:
    """A result that depends on symbols: its Mathematica text and its values at fixed parameter points."""

    variables: tuple[str, ...]
    text: str
    points: tuple[tuple[dict[str, float], Any], ...]


@cache
def load(name: str) -> dict[str, Any]:
    """The content of ``fixtures/data/<name>.json``."""
    return json.loads((DATA / f"{name}.json").read_text(encoding="utf-8"))


def names() -> list[str]:
    """The fixture files, without the manifest and the timings."""
    return sorted(p.stem for p in DATA.glob("[0-9]*.json"))


def cases(name: str) -> dict[str, Any]:
    return load(name)["cases"]


def array(value: dict[str, Any]) -> np.ndarray:
    """A stored array: real when the imaginary part was zero everywhere, complex otherwise."""
    if value["type"] != "array":
        raise TypeError(f"not an array but {value['type']!r}")
    shape = tuple(value["shape"])
    re = np.asarray(value["re"], dtype=float)
    out = re + 1j * np.asarray(value["im"], dtype=float) if "im" in value else re
    return out.reshape(shape)


def decode(value: dict[str, Any]) -> Any:
    """A stored value as a Python object."""
    kind = value["type"]
    if kind == "array":
        return array(value)
    if kind in ("string", "strings"):
        return value["value"]
    if kind == "list":
        return [decode(item) for item in value["items"]]
    if kind == "expr":
        return value["text"]
    if kind == "symbolic":
        points = tuple(
            (dict(zip(value["variables"], point["parameters"], strict=True)), decode(point["value"]))
            for point in value["points"]
        )
        return Symbolic(tuple(value["variables"]), value["text"], points)
    raise ValueError(f"unknown fixture type {kind!r}")


def get(name: str, key: str) -> Any:
    """The decoded value of case ``key`` (``'group/case'``) of fixture file ``name``."""
    return decode(cases(name)[key]["value"])


def exact(value: dict[str, Any]):
    """The exact form of a stored array as a SymPy object (``None`` when the result was not exact)."""
    from sympy.parsing.mathematica import parse_mathematica

    if "exact" not in value:
        return None
    return parse_mathematica(value["exact"])


def exact_array(value: dict[str, Any]) -> np.ndarray:
    """The exact form of a stored array evaluated by SymPy to double precision, in the stored shape."""
    import sympy

    parsed = exact(value)
    flat = sympy.flatten(parsed) if isinstance(parsed, (list, tuple, sympy.Tuple)) else [parsed]
    return np.array([complex(sympy.N(x, 17)) for x in flat]).reshape(tuple(value["shape"]))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
