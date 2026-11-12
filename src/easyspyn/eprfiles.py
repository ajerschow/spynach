"""EPR data files: Bruker BES3T and ESP/WinEPR readers, text columns, a BES3T writer (EasySpin's eprload, eprsave).

``eprload(file)`` returns ``(x, y, params)`` for Bruker BES3T (``.DSC``/``.DTA``, 1-D to 3-D, real or complex,
with ``.XGF``/``.YGF``/``.ZGF`` axis files and the scaling letters ``n P G T c``), Bruker ESP/WinEPR
(``.par``/``.spc``) and text columns; the other formats EasySpin reads raise ``NotImplementedError``. ``x`` is in
the units of the file (Gauss for Bruker cw spectra), as in EasySpin; ``exp_from_params`` turns the parameters
into the ``exp`` dict of the simulators and ``esfit`` (field in mT). ``eprsave`` writes BES3T files.

Deviations from EasySpin: continuation lines of a descriptor are joined including their last line (EasySpin
parses that line on its own); companion files are found case-insensitively; a directory is an error (no file
dialog); nothing is plotted.

EasySpin authors, by original file (contributors to the file up to EasySpin 6.0.15, see
  docs/easyspin-credits.md):
  eprload.m: Stefan Stoll, Stephan Pribitzer
  eprload_BrukerBES3T.m: Stefan Stoll, Claudia Tait
  eprload_BrukerESP.m: Stefan Stoll
  eprsave.m: Stefan Stoll
  getmatrix.m: Stefan Stoll
  parsefieldparams.m: Stefan Stoll
"""

from __future__ import annotations

import os
import re
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["eprload", "eprsave", "exp_from_params"]

_SCALING_LETTERS = set("nPGTc")
_DATA_FORMATS = {"C": "i1", "S": "i2", "I": "i4", "F": "f4", "D": "f8"}
_AXIS_FORMATS = {"D": "f8", "F": "f4", "I": "i4", "S": "i2"}
_TEXT_SUFFIXES = {".TXT", ".CSV", ".TSV", ".ASC"}
_UNPORTED = {
    ".D01": "SpecMan",
    ".SPE": "Magnettech binary",
    ".XML": "Magnettech XML",
    ".ESR": "Active Spectrum",
    ".DAT": "Adani",
    ".JSON": "Adani JSON",
    ".ECO": "qese/tryscore",
    ".PLT": "MAGRES",
    ".SPK": "Varian",
    ".REF": "Varian",
    ".D00": "Weizmann/ETH",
}
_JEOL = re.compile(rb"^spin|^cAcqu|^endor|^pAcqu|^cidep|^sod|^iso|^ani")
_NUMBER = re.compile(r"^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$|^[+-]?(?:inf|nan)$", re.IGNORECASE)
_LEADING_NUMBER = re.compile(r"^\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?|[+-]?(?:inf|nan))", re.IGNORECASE)
_FIELD_UNITS = {"G": 0.1, "GAUSS": 0.1, "MT": 1.0, "T": 1000.0}


# ------------------------------------------------------------------------------------------------ eprload
def eprload(file: str | os.PathLike, scaling: str = "", *, format: str | None = None) -> tuple[Any, Any, dict]:
    """Read an EPR data file: ``(x, y, params)``.

    ``file`` names a Bruker BES3T pair (``.DSC``/``.DTA``, either member or the name without extension), a Bruker
    ESP/WinEPR pair (``.par``/``.spc``) or a text file of columns (``.txt``, ``.csv``, ``.tsv``, ``.asc``);
    ``format`` (``'BES3T'``, ``'ESP'`` or ``'text'``) overrides the extension. ``x`` is the axis in the units of
    the file (Gauss for Bruker cw spectra; a list of axes for 2-D and 3-D data), ``y`` the data (float or complex,
    ``(nx,)``, ``(nx, ny)`` or ``(nx, ny, nz)``; a list of arrays when the file holds several values per point),
    ``params`` the parameter file as a dict (numbers converted, ``true``/``false`` to booleans). ``scaling``
    (Bruker files only) divides by the number of scans (``'n'``), the square root of the microwave power in mW
    (``'P'``), the receiver gain (``'G'``) or the conversion time in ms (``'c'``), and multiplies by the
    temperature in K (``'T'``); several letters combine.

    EasySpin authors (eprload.m): Stefan Stoll, Stephan Pribitzer.
    """
    path = Path(file)
    if path.is_dir():
        raise ValueError(f"{path} is a directory; give the name of a data file.")
    base, ext = _resolve(path)
    if format is not None:
        kind = {"bes3t": "BES3T", "esp": "ESP", "text": "text"}.get(format.lower())
        if kind is None:
            raise ValueError("format must be 'BES3T', 'ESP' or 'text'.")
    else:
        kind = _format_from_extension(base, ext)
    if scaling:
        if kind in ("BES3T", "ESP"):
            if set(scaling) - _SCALING_LETTERS:
                raise ValueError("Scaling can only contain 'n', 'P', 'G', 'T', and 'c'.")
        else:
            raise ValueError("Scaling does not work for this file type.")
    if kind == "BES3T":
        return _load_bes3t(base, ext, scaling)
    if kind == "ESP":
        return _load_esp(base, ext, scaling)
    return _load_text(base.with_name(base.name + ext))


def _resolve(path: Path) -> tuple[Path, str]:
    """The file's base path (no extension) and its extension, with EasySpin's search for a missing extension."""
    suffix = path.suffix
    if len(suffix) < 2:
        base = path.with_suffix("") if suffix else path
        for ext in (".dta", ".dsc", ".spc", ".par"):
            hit = _sibling(base, ext)
            if hit is not None:
                return hit.with_suffix(""), hit.suffix
        raise FileNotFoundError(f"The file or directory {path} does not exist!")
    if not path.is_file():
        hit = _sibling(path.with_suffix(""), suffix)
        if hit is None:
            raise FileNotFoundError(f"The file or directory {path} does not exist!")
        path = hit
    return path.with_suffix(""), path.suffix


def _sibling(base: Path, ext: str) -> Path | None:
    """``base + ext`` if it exists, else the file of that name in any case; ``None`` when there is none."""
    exact = base.with_name(base.name + ext)
    if exact.is_file():
        return exact
    parent = base.parent if base.parent != Path("") else Path(".")
    try:
        entries = list(parent.iterdir())
    except OSError:
        return None
    wanted = (base.name + ext).lower()
    for entry in entries:
        if entry.name.lower() == wanted and entry.is_file():
            return entry
    return None


def _format_from_extension(base: Path, ext: str) -> str:
    upper = ext.strip().upper()
    if upper in (".DTA", ".DSC"):
        return "BES3T"
    if upper in (".PAR", ".SPC"):
        return "ESP"
    if upper in _TEXT_SUFFIXES:
        return "text"
    if upper in _UNPORTED:
        raise NotImplementedError(
            f"{_UNPORTED[upper]} files ({ext}) are not ported; EasySpin's eprload reads them"
            + (" (pass format='text' for a text file)." if upper in (".DAT", ".JSON") else ".")
        )
    with open(base.with_name(base.name + ext), "rb") as handle:
        head = handle.read(16)
    if _JEOL.match(head):
        raise NotImplementedError("JEOL files are not ported; EasySpin's eprload reads them.")
    raise ValueError(f"Unsupported file extension {ext}")


# --------------------------------------------------------------------------------- parameter-file parsing
def _read_dsc(path: Path) -> dict[str, str]:
    """The key-value lines of a BES3T descriptor (EasySpin's readDSCfile); values are strings.

    Lines ending in a backslash continue on the next line (the literal ``\\n`` inside becomes a newline), keys
    not starting with a letter are skipped, ``#MHL`` (the manipulation history layer) ends the parse, and a
    value enclosed in single quotes loses them.

    EasySpin authors (eprload_BrukerBES3T.m, the descriptor reader readDSCfile): Stefan Stoll, Claudia Tait.
    """
    lines = path.read_text(encoding="latin-1").splitlines()
    params: dict[str, str] = {}
    k = 0
    while k < len(lines):
        line = lines[k]
        k += 1
        if not line:
            continue
        if line.endswith("\\"):
            parts = [line[:-1]]
            while k < len(lines) and lines[k].endswith("\\"):
                parts.append(lines[k][:-1])
                k += 1
            if k < len(lines):
                parts.append(lines[k])
                k += 1
            line = "".join(parts).replace("\\n", "\n")
        key, value = _split_key(line)
        if not key:
            continue
        if not key[0].isalpha():
            if key.upper() == "#MHL":
                break
            continue
        params[key] = value
    return params


def _read_par(path: Path) -> dict[str, str]:
    """The key-value lines of an ESP/WinEPR parameter file (EasySpin's eprload_readPARfile); values are strings.

    EasySpin authors (eprload_BrukerESP.m, the parameter-file reader eprload_readPARfile): Stefan Stoll.
    """
    params: dict[str, str] = {}
    for line in path.read_text(encoding="latin-1").splitlines():
        key, value = _split_key(line)
        if not key or not key[0].isalpha():
            continue
        params[key] = value
    return params


def _split_key(line: str) -> tuple[str, str]:
    """``(key, value)`` of a parameter line: the first token and the rest, trimmed and unquoted."""
    parts = line.split(None, 1)
    if not parts:
        return "", ""
    value = parts[1].strip() if len(parts) > 1 else ""
    if value and value[0] == "'" and value[-1] == "'":
        value = value[1:-1]
    return parts[0], value


def _parsefieldparams(raw: dict[str, str]) -> dict[str, Any]:
    """Numbers, arrays and booleans for the values that are such (EasySpin's parsefieldparams).

    ``true``/``false`` become booleans, a value starting with a letter stays a string, and a value whose
    whitespace-separated tokens are all numbers becomes a float or a float array; anything else (``'3345 G'``,
    dates, ``'1,2'``) stays a string.

    EasySpin authors (parsefieldparams.m, the value parser): Stefan Stoll.
    """
    out: dict[str, Any] = {}
    for key, value in raw.items():
        if value == "":
            out[key] = value
        elif value.lower() == "true":
            out[key] = True
        elif value.lower() == "false":
            out[key] = False
        elif value[0].isalpha():
            out[key] = value
        else:
            tokens = value.split()
            if all(_NUMBER.match(token) for token in tokens):
                numbers = np.array([float(token) for token in tokens])
                out[key] = float(numbers[0]) if numbers.size == 1 else numbers
            else:
                out[key] = value
    return out


def _num(raw: dict[str, str], key: str) -> float:
    """The number a value starts with, as ``sscanf(value, '%f')`` reads it (``'3345 G'`` gives 3345)."""
    match = _LEADING_NUMBER.match(raw[key])
    if match is None:
        raise ValueError(f"Cannot read a number from {key} = {raw[key]!r} in the parameter file.")
    return float(match.group(1))


# --------------------------------------------------------------------------------------- binary data files
def _getmatrix(path: Path, dims: tuple[int, ...], code: str, byteorder: str, is_complex: list[bool]):
    """The data of a binary file as arrays of shape ``dims`` with trailing singleton axes dropped (getmatrix).

    Each point holds one real for every real value and two (real, imaginary) for every complex value, point after
    point; the points run with the first axis fastest. Integer data become floats.

    EasySpin authors (getmatrix.m, the binary reader, with NumPy): Stefan Stoll.
    """
    n_reals = sum(2 if c else 1 for c in is_complex)
    n_points = int(np.prod(dims))
    try:
        values = np.frombuffer(path.read_bytes(), dtype=np.dtype(byteorder + code), count=n_reals * n_points)
    except ValueError as error:
        raise ValueError("Unable to read all expected data.") from error
    table = values.reshape(n_points, n_reals)
    arrays, column = [], 0
    for complex_value in is_complex:
        if complex_value:
            data = table[:, column].astype(np.float64) + 1j * table[:, column + 1].astype(np.float64)
            column += 2
        else:
            data = table[:, column].astype(np.float64)
            column += 1
        data = data.reshape(dims, order="F")
        while data.ndim > 1 and data.shape[-1] == 1:
            data = data[..., 0]
        arrays.append(data)
    return arrays[0] if len(arrays) == 1 else arrays


def _companion(base: Path, given: str, wanted: str) -> Path | None:
    """The companion file ``base + wanted`` in the case of the extension the user gave, or in any case."""
    ext = wanted.upper() if given.isupper() else wanted.lower()
    return _sibling(base, ext)


# ------------------------------------------------------------------------------------------- Bruker BES3T
def _load_bes3t(base: Path, ext: str, scaling: str):
    """Bruker BES3T 1.2 (Xepr, Elexsys, EMX): descriptor ``.DSC``, data ``.DTA`` (eprload_BrukerBES3T).

    EasySpin authors (eprload_BrukerBES3T.m, the BES3T reader, as a private function): Stefan Stoll, Claudia
      Tait.
    """
    descriptor = _companion(base, ext, ".DSC")
    if descriptor is None:
        raise FileNotFoundError(f"Cannot find the file {base}{'.DSC' if ext.isupper() else '.dsc'}.")
    raw = _read_dsc(descriptor)

    # IKKF: one CPLX or REAL per data value stored at every point
    if "IKKF" in raw:
        is_complex = []
        for part in raw["IKKF"].split(","):
            if part == "CPLX":
                is_complex.append(True)
            elif part == "REAL":
                is_complex.append(False)
            else:
                raise ValueError("Unknown value for keyword IKKF in .DSC file!")
    else:
        warnings.warn("Keyword IKKF not found in .DSC file! Assuming IKKF=REAL.", stacklevel=3)
        is_complex = [False]

    # XPTS, YPTS, ZPTS: points per axis
    if "XPTS" not in raw:
        raise ValueError("No XPTS in DSC file.")
    dims = tuple(int(_num(raw, key)) if key in raw else 1 for key in ("XPTS", "YPTS", "ZPTS"))

    # BSEQ: byte order
    if "BSEQ" in raw:
        if raw["BSEQ"] == "BIG":
            byteorder = ">"
        elif raw["BSEQ"] == "LIT":
            byteorder = "<"
        else:
            raise ValueError("Unknown value for keyword BSEQ in .DSC file!")
    else:
        warnings.warn("Keyword BSEQ not found in .DSC file! Assuming BSEQ=BIG.", stacklevel=3)
        byteorder = ">"

    # IRFMT, IIFMT: number format of the real and imaginary parts (one per data value; the last one is used)
    if "IRFMT" not in raw:
        raise ValueError("Keyword IRFMT not found in .DSC file!")
    parts = raw["IRFMT"].split(",")
    if len(parts) != len(is_complex):
        raise ValueError("Problem in BES3T DSC file: inconsistent IKKF and IRFMT fields.")
    code = ""
    for part in parts:
        letter = part.upper()
        if letter in _DATA_FORMATS:
            code = _DATA_FORMATS[letter]
        elif letter == "A":
            raise ValueError("Cannot read BES3T data in ASCII format!")
        elif letter in ("0", "N"):
            raise ValueError("No BES3T data!")
        else:
            raise ValueError("Unknown value for keyword IRFMT in .DSC file!")
    if "IIFMT" in raw and raw["IIFMT"].upper() != raw["IRFMT"].upper():
        raise ValueError("IRFMT and IIFMT in DSC file must be identical.")

    # Axes: IDX (linear from *MIN and *WID), IGD (companion .XGF/.YGF/.ZGF file), NTUP (not readable)
    axes = []
    for a, name in enumerate("XYZ"):
        n = dims[a]
        if n <= 1:
            continue
        if name + "TYP" not in raw:
            raise ValueError(f"No {name}TYP in DSC file.")
        axis_type = raw[name + "TYP"]
        axis = None
        if axis_type == "IGD":
            companion = _companion(base, ext, f".{name}GF")
            companion_name = f"{base}.{name}GF"
            fmt = raw.get(name + "FMT", "")
            if fmt not in _AXIS_FORMATS:
                raise ValueError(f"Cannot read data format {fmt} for companion file {companion_name}")
            if companion is not None:
                dtype = np.dtype(byteorder + _AXIS_FORMATS[fmt])
                buffer = companion.read_bytes()
                if len(buffer) < n * dtype.itemsize:
                    raise ValueError(f"Could not read the expected {n} values from companion file {companion_name}.")
                axis = np.frombuffer(buffer, dtype=dtype, count=n).astype(np.float64)
            else:
                warnings.warn(
                    f"Could not read companion file {companion_name} for nonlinear axis. Assuming linear axis.",
                    stacklevel=3,
                )
                axis_type = "IDX"
        if axis_type == "IDX":
            minimum, width = _num(raw, name + "MIN"), _num(raw, name + "WID")
            if width == 0:
                warnings.warn(f"{name} range has zero width.", stacklevel=3)
                minimum, width = 1.0, n - 1.0
            axis = minimum + np.linspace(0.0, width, n)
        elif axis_type == "NTUP":
            raise ValueError("Cannot read data with NTUP axes.")
        if axis is None:
            raise ValueError(f"Unknown axis type {axis_type} for the {name} axis in the DSC file.")
        axes.append(axis)
    x = axes[0] if len(axes) == 1 else axes

    # Data
    data_file = _companion(base, ext, ".DTA")
    if data_file is None:
        raise FileNotFoundError(f"Unable to open data file {base}{'.DTA' if ext.isupper() else '.dta'}")
    data = _getmatrix(data_file, dims, code, byteorder, is_complex)

    if scaling:
        data = _scale_bes3t(data, raw, scaling)
    return x, data, _parsefieldparams(raw)


def _scale_bes3t(data, raw: dict[str, str], scaling: str):
    """EasySpin's scaling of BES3T data by scans, receiver gain, conversion time, microwave power, temperature.

    EasySpin authors (eprload_BrukerBES3T.m, the scaling step): Stefan Stoll, Claudia Tait.
    """
    if "EXPT" not in raw:
        raise ValueError("Missing EXPT field in the DSC file; cannot tell whether these are cw data.")
    cw = raw["EXPT"] == "CW"
    prescaled = raw.get("SctNorm", "").lower() == "true"
    factor = 1.0
    if "n" in scaling:
        if "AVGS" not in raw:
            raise ValueError("Missing AVGS field in the DSC file.")
        averages = int(_num(raw, "AVGS"))
        if prescaled:
            raise ValueError(
                "Scaling by number of scans not possible, since data in DSC/DTA are already averaged over "
                f"{averages} scans."
            )
        factor /= averages
    if cw and "G" in scaling:
        if "RCAG" not in raw:
            raise ValueError("Cannot scale by receiver gain, since RCAG in the DSC file is missing.")
        factor /= 10 ** (_num(raw, "RCAG") / 20)  # RCAG in dB
    if cw and "c" in scaling:
        if "SPTP" not in raw:
            raise ValueError("Cannot scale by sampling time, since SPTP in the DSC file is missing.")
        factor /= _num(raw, "SPTP") * 1000  # s -> ms
    if "P" in scaling:
        if not cw:
            raise ValueError("Cannot scale by microwave power, since these are not CW-EPR data.")
        if "MWPW" not in raw:
            raise ValueError("Cannot scale by power, since MWPW is absent in parameter file.")
        factor /= np.sqrt(_num(raw, "MWPW") * 1000)  # W -> mW
    if "T" in scaling:
        if "STMP" not in raw:
            raise ValueError("Cannot scale by temperature, since STMP in the DSC file is missing.")
        factor *= _num(raw, "STMP")
    return [d * factor for d in data] if isinstance(data, list) else data * factor


# -------------------------------------------------------------------------------------- Bruker ESP/WinEPR
def _load_esp(base: Path, ext: str, scaling: str):
    """Bruker ESP, ECS, WinEPR and Simfonia files: parameters ``.par``, data ``.spc`` (eprload_BrukerESP).

    EasySpin authors (eprload_BrukerESP.m, the ESP/WinEPR reader, as a private function): Stefan Stoll.
    """
    parameters = _companion(base, ext, ".PAR")
    if parameters is None:
        raise FileNotFoundError(f"Cannot find the file {base}{'.PAR' if ext.isupper() else '.par'}.")
    raw = _read_par(parameters)

    # WinEPR (DOS) files are little-endian floats, ESP files big-endian 32-bit integers
    winepr = "DOS" in raw
    byteorder = "<" if winepr else ">"
    flags = int(_num(raw, "JSS")) if "JSS" in raw else 0
    is_complex = bool(flags >> 4 & 1)
    two_d = bool(flags >> 12 & 1)
    nx, ny = 1024, 1
    if two_d and "SSX" in raw:
        nx = _num(raw, "SSX")
        if is_complex:
            nx /= 2
    if two_d and "SSY" in raw:
        ny = _num(raw, "SSY")
    if "ANZ" in raw:
        total = _num(raw, "ANZ")
        if not two_d:
            nx = total / 2 if is_complex else total
        elif nx * ny != total:
            raise ValueError("Two-dimensional data: SSX, SSY and ANZ in .par file are inconsistent.")
    if "RES" in raw:
        nx = _num(raw, "RES")
    if "REY" in raw:
        ny = _num(raw, "REY")
    if "XPLS" in raw:
        nx = _num(raw, "XPLS")
    nx, ny = int(nx), int(ny)
    code = "f4" if winepr else "i4"

    x: Any = np.zeros(0)
    power_axis = None
    if nx > 1:
        raw.setdefault("JEX", "field-sweep")
        raw.setdefault("JEY", "")
        endor = raw["JEX"] == "ENDOR"
        time_sweep = raw["JEX"] == "Time-Sweep"
        ranges = {key: _num(raw, key) if key in raw else None for key in ("GST", "GSI", "HCF", "HSW")}
        emx = {key: _num(raw, key) if key in raw else None for key in ("XXLB", "XXWI", "XYLB", "XYWI")}
        choice = 0  # 1: GST/GSI, 2: HCF/HSW, 3: XXLB/XXWI (and XYLB/XYWI)
        if endor:
            choice = 1
        elif all(v is not None for v in emx.values()):
            choice = 3
        elif all(v is not None for v in ranges.values()):
            choice = 1
        elif ranges["HCF"] is not None and ranges["HSW"] is not None:
            choice = 2
        elif ranges["GST"] is not None and ranges["GSI"] is not None:
            choice = 1
        elif ranges["GSI"] is None and ranges["HSW"] is None:
            ranges["HSW"] = 50.0
            choice = 2
        elif ranges["HCF"] is None:
            choice = 3
        if time_sweep:
            conversion = _num(raw, "RCT") if "RCT" in raw else 1.0
            x = np.arange(nx) * conversion / 1e3
        elif choice == 1:
            x = ranges["GST"] + ranges["GSI"] * np.linspace(0.0, 1.0, nx)
        elif choice == 2:
            if ranges["HCF"] is None:
                raise ValueError("Could not determine abscissa range from parameter file!")
            x = ranges["HCF"] + ranges["HSW"] / 2 * np.linspace(-1.0, 1.0, nx)
        elif choice == 3:
            if emx["XXLB"] is not None and emx["XXWI"] is not None:
                x = emx["XXLB"] + np.linspace(0.0, emx["XXWI"], nx)
                if emx["XYLB"] is not None and emx["XYWI"] is not None:
                    power_axis = emx["XYLB"] + np.linspace(0.0, emx["XYWI"], ny)
                    x = [x, power_axis]
        else:
            raise ValueError("Could not determine abscissa range from parameter file!")

    # A slice of 2-D data saved by WinEPR: RES/REY give the original size, the JSS flag is not set
    if not two_d and ny > 1:
        ny = 1
    data_file = _companion(base, ext, ".SPC")
    if data_file is None:
        raise FileNotFoundError(f"Unable to open data file {base}{'.SPC' if ext.isupper() else '.spc'}")
    data = _getmatrix(data_file, (nx, ny, 1), code, byteorder, [is_complex])

    if scaling:
        if "n" in scaling:
            if "JSD" not in raw:
                raise ValueError("Cannot scale by number of scans, since JSD is absent in parameter file.")
            data = data / _num(raw, "JSD")
        if "G" in scaling:
            if "RRG" not in raw:
                raise ValueError("Cannot scale by gain, since RRG is absent in parameter file.")
            data = data / _num(raw, "RRG")
        if "P" in scaling:
            if "MP" not in raw:
                raise ValueError("Cannot scale by power, since MP is absent in parameter file.")
            power = _num(raw, "MP")  # mW
            if raw.get("JEY") == "mw-power-sweep":
                if "XYLB" not in raw or "XYWI" not in raw:
                    raise ValueError("A power sweep needs XYLB and XYWI in the parameter file.")
                n_powers = data.shape[1] if data.ndim > 1 else 1
                attenuation = _num(raw, "XYLB") + np.linspace(0.0, _num(raw, "XYWI"), n_powers)
                data = data / np.sqrt(power * 10 ** (-attenuation / 10))
            else:
                data = data / np.sqrt(power)
        if "T" in scaling:
            if "TE" not in raw:
                raise ValueError("Cannot scale by temperature, since TE is absent in parameter file.")
            temperature = _num(raw, "TE")
            if temperature == 0:
                raise ValueError("Cannot scale by temperature, since TE is zero in parameter file.")
            data = data * temperature
        if "c" in scaling:
            if "RCT" not in raw:
                raise ValueError("Cannot scale by sampling time, since RCT in the .par file is missing.")
            data = data / _num(raw, "RCT")
    return x, data, _parsefieldparams(raw)


# ---------------------------------------------------------------------------------------------- text files
def _load_text(path: Path):
    """Columns of numbers: one column is ``y`` (``x = 1..n``), two are ``x, y``, more give ``y`` of shape ``(n, k-1)``.

    Separators are whitespace, commas or semicolons; lines without numbers (comments, headers) are kept in
    ``params["Comments"]``. New here, no EasySpin counterpart (its eprload refers to textscan).
    """
    comments, rows = [], []
    for line in path.read_text(encoding="latin-1").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(("#", "%", "//")):
            comments.append(stripped)
            continue
        tokens = [t for t in re.split(r"[,;\s]+", stripped) if t]
        try:
            rows.append([float(t) for t in tokens])
        except ValueError:
            comments.append(stripped)
    if not rows:
        raise ValueError(f"No numbers found in {path}.")
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError(f"The rows of {path} do not all have {width} columns.")
    table = np.array(rows, dtype=float)
    params = {"Format": "text", "Comments": comments, "nColumns": width}
    if width == 1:
        return np.arange(1, len(rows) + 1, dtype=float), table[:, 0], params
    if width == 2:
        return table[:, 0], table[:, 1], params
    return table[:, 0], table[:, 1:], params


# ------------------------------------------------------------------------------------------------ eprsave
def eprsave(file: str | os.PathLike, x, y, title: str = "", mw_freq: float | None = None) -> None:
    """Write ``y`` with its axis ``x`` as a Bruker BES3T pair ``<file>.DSC``/``<file>.DTA``.

    ``y`` is real or complex, 1-D or 2-D (``(nx, ny)`` with ``x = [x, y_axis]``); a nonlinear axis goes to a
    companion ``.XGF``/``.YGF`` file; ``title`` becomes ``TITL`` and ``mw_freq`` (GHz) ``MWFQ`` (Hz). The files are
    what EasySpin's eprsave writes (big-endian float64), apart from the exported-from comment and the date.

    EasySpin authors (eprsave.m): Stefan Stoll.
    """
    y = np.asarray(y)
    if y.ndim > 2:
        raise ValueError("Cannot save data with more than 2 dimensions.")
    two_d = y.ndim == 2 and min(y.shape) > 1
    if two_d:
        if not isinstance(x, (list, tuple)) or len(x) != 2:
            raise ValueError("For two-dimensional data, x must be a list containing the two axes.")
        x_axis, y_axis = (np.asarray(a, dtype=float).ravel() for a in x)
    else:
        x_axis = np.asarray(x[0] if isinstance(x, (list, tuple)) else x, dtype=float).ravel()
        y_axis = None
        y = y.reshape(-1)
    name = os.fspath(file)
    if name[-4:].upper() in (".DSC", ".DTA"):
        name = name[:-4]
    base = Path(name)
    is_complex = np.iscomplexobj(y)

    values = y.reshape(-1, order="F")
    if is_complex:
        interleaved = np.empty(2 * values.size)
        interleaved[0::2], interleaved[1::2] = values.real, values.imag
        values = interleaved
    base.with_name(base.name + ".DTA").write_bytes(np.asarray(values, dtype=">f8").tobytes())

    now = datetime.now()
    lines = [
        f"* Exported from Python using EasySpyn, {now.strftime('%d-%b-%Y %H:%M:%S')}",
        "#DESC\t1.2 * DESCRIPTOR INFORMATION ***********************",
        "DSRC\tMAN",
        "BSEQ\tBIG",
        "IKKF\tCPLX" if is_complex else "IKKF\tREAL",
        "IRFMT\tD",
    ]
    if is_complex:
        lines.append("IIFMT\tD")
    x_linear = _is_linear(x_axis)
    if not x_linear:
        base.with_name(base.name + ".XGF").write_bytes(np.asarray(x_axis, dtype=">f8").tobytes())
        lines.append("XFMT\tD")
    y_linear = True
    if two_d:
        y_linear = _is_linear(y_axis)
        if not y_linear:
            base.with_name(base.name + ".YGF").write_bytes(np.asarray(y_axis, dtype=">f8").tobytes())
            lines.append("YFMT\tD")
    lines += [
        "XTYP\t" + ("IDX" if x_linear else "IGD"),
        "YTYP\t" + (("IDX" if y_linear else "IGD") if two_d else "NODATA"),
        "ZTYP\tNODATA",
        f"XPTS\t{x_axis.size:d}",
        f"XMIN\t{x_axis[0]:g}",
        f"XWID\t{x_axis[-1] - x_axis[0]:g}",
    ]
    if two_d:
        lines += [f"YPTS\t{y_axis.size:d}", f"YMIN\t{y_axis[0]:g}", f"YWID\t{y_axis[-1] - y_axis[0]:g}"]
    if title:
        lines.append(f"TITL\t'{title}'")
    lines += [
        "*",
        "************************************************************",
        "*",
        "#SPL\t1.2 * STANDARD PARAMETER LAYER",
        "OPER\t",
        f"DATE\t{now.strftime('%d/%m/%y')}",
        f"TIME\t{now.strftime('%H:%M:%S')}",
        "CMNT\t",
        "SAMP\t",
        "SFOR\t",
    ]
    if mw_freq is not None:
        lines.append(f"MWFQ\t{mw_freq * 1e9:0.9g}")
    base.with_name(base.name + ".DSC").write_text("".join(line + "\n" for line in lines), encoding="latin-1")


def _is_linear(axis: np.ndarray) -> bool:
    """EasySpin's test: the largest and smallest steps agree to 1e-5 (a single point counts as linear)."""
    if axis.size < 2:
        return True
    steps = np.diff(axis)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = steps.max() / steps.min()
    return bool(abs(ratio - 1) < 1e-5)


# ------------------------------------------------------------------------------------------ the fit link
def exp_from_params(params: dict, x, *, field_unit: str | None = None, mw_freq: float | None = None) -> dict:
    """The ``exp`` dict for ``pepper``, ``garlic``, ``chili`` and ``esfit`` from a loaded file's parameters.

    ``mwFreq`` (GHz) comes from ``MWFQ`` (BES3T, Hz) or ``MF`` (ESP, GHz) or ``mw_freq``; ``Range`` (mT) and
    ``nPoints`` from the axis ``x`` (a list of axes uses the first), with the field unit from ``XUNI``, ``JUN`` or
    ``field_unit`` (``'G'``, ``'mT'``, ``'T'``; an argument overrides the file); ``Harmonic`` from ``RCHM`` (or the
    signal channel's ``Harmonic``) or ``RHA``; ``ModAmp`` (mT) from ``B0MA`` (T) or the signal channel's ``ModAmp``
    string or ``RMA`` (G); ``Temperature`` (K) from ``STMP`` or ``TE`` when positive (it only matters for high-spin
    systems; drop the key to leave it out). The axis must be an ascending uniform field sweep: other sweeps raise
    ``ValueError`` with the advice to resample and give ``Range`` and ``nPoints`` by hand. New here, no EasySpin
    counterpart.
    """
    axis = np.asarray(x[0] if isinstance(x, (list, tuple)) else x, dtype=float).ravel()
    if axis.size < 2:
        raise ValueError("The axis has fewer than two points.")
    if "IKKF" in params or "XPTS" in params:
        kind = "BES3T"
    elif any(key in params for key in ("JSS", "HCF", "GST", "DOS", "ANZ", "RES", "MF")):
        kind = "ESP"
    elif params.get("Format") == "text":
        kind = "text"
    else:
        raise ValueError("Cannot tell the file format from the parameters.")

    harmonic = mod_amp = temperature = None
    unit = None
    frequency = mw_freq
    if kind == "BES3T":
        experiment = params.get("EXPT")
        if isinstance(experiment, str) and experiment and experiment != "CW":
            raise ValueError(f"These are not cw data (EXPT = {experiment}); give exp by hand.")
        axis_name = params.get("XNAM")
        if isinstance(axis_name, str) and axis_name and "field" not in axis_name.lower():
            raise ValueError(f"The x axis is {axis_name!r}, not a field sweep; give exp by hand.")
        unit = params.get("XUNI") if isinstance(params.get("XUNI"), str) else None
        if frequency is None and _is_number(params.get("MWFQ")):
            frequency = float(params["MWFQ"]) / 1e9
        if _is_number(params.get("RCHM")):
            harmonic = int(params["RCHM"])
        elif _is_number(params.get("Harmonic")):
            harmonic = int(params["Harmonic"])
        if _is_number(params.get("B0MA")) and params["B0MA"] > 0:
            mod_amp = float(params["B0MA"]) * 1000  # T -> mT
        elif isinstance(params.get("ModAmp"), str):
            mod_amp = _field_string_mt(params["ModAmp"])
        if _is_number(params.get("STMP")):
            temperature = float(params["STMP"])
    elif kind == "ESP":
        sweep = params.get("JEX", "field-sweep")
        if isinstance(sweep, str) and sweep and sweep != "field-sweep":
            raise ValueError(f"These are not field-sweep data (JEX = {sweep}); give exp by hand.")
        unit = params.get("JUN") if isinstance(params.get("JUN"), str) else "G"
        if frequency is None and _is_number(params.get("MF")):
            frequency = float(params["MF"])
        if _is_number(params.get("RHA")):
            harmonic = int(params["RHA"])
        if _is_number(params.get("RMA")) and params["RMA"] > 0:
            mod_amp = float(params["RMA"]) / 10  # G -> mT
        if _is_number(params.get("TE")):
            temperature = float(params["TE"])
    if field_unit is not None:
        unit = field_unit
    if unit is None:
        raise ValueError("The field unit is not in the parameters; pass field_unit='G' or 'mT'.")
    if unit.upper() not in _FIELD_UNITS:
        raise ValueError(f"The x axis unit {unit!r} is not a field unit; give exp by hand.")
    factor = _FIELD_UNITS[unit.upper()]
    if frequency is None:
        raise ValueError("No microwave frequency in the parameters; pass mw_freq in GHz.")
    steps = np.diff(axis)
    if steps.mean() == 0 or np.ptp(steps) > 1e-5 * abs(steps.mean()):
        raise ValueError(
            "The x axis is not a uniform sweep; resample y onto a uniform axis (np.interp) and give Range and "
            "nPoints yourself."
        )
    if steps.mean() < 0:
        raise ValueError("The x axis is descending; flip x and y (x[::-1], y[::-1]) first.")
    exp = {
        "mwFreq": float(frequency),
        "Range": [float(axis[0] * factor), float(axis[-1] * factor)],
        "nPoints": int(axis.size),
    }
    if harmonic is not None:
        exp["Harmonic"] = harmonic
    if mod_amp is not None and mod_amp > 0:
        exp["ModAmp"] = float(mod_amp)
    if temperature is not None and np.isfinite(temperature) and temperature > 0:
        exp["Temperature"] = temperature
    return exp


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool)


def _field_string_mt(text: str) -> float | None:
    """``'1.000 G'`` or ``'0.1 mT'`` in mT; ``None`` when the string is not a field."""
    match = re.match(r"^\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s*([A-Za-z]+)\s*$", text)
    if match is None or match.group(2).upper() not in _FIELD_UNITS:
        return None
    return float(match.group(1)) * _FIELD_UNITS[match.group(2).upper()]
