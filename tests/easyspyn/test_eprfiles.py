"""Fixture-free tests of eprload, eprsave and exp_from_params: the parameter parsers, Bruker BES3T round trips
and hand-written descriptors (formats, byte orders, axes, companion files, scaling, errors), Bruker ESP/WinEPR
files written here from the byte layout of EasySpin's loader, text columns, name resolution, and a fit of a saved
pepper spectrum through the exp helper."""

import struct
import warnings

import numpy as np
import pytest

import easyspyn as es
from easyspyn import eprfiles

X = np.linspace(3300, 3400, 32)
Y = np.exp(-(((X - 3350) / 12) ** 2))


# ----------------------------------------------------------------------------------- writers for the tests
def write_dsc(tmp_path, name, lines, data=None, dtype=">f8", ext=".DSC"):
    """A descriptor from ``lines`` (``key value``) and, if given, ``data`` written with ``dtype``; the base path."""
    (tmp_path / f"{name}{ext}").write_text("".join(line + "\n" for line in lines), encoding="latin-1")
    if data is not None:
        (tmp_path / f"{name}.DTA").write_bytes(np.asarray(data).astype(dtype).tobytes())
    return tmp_path / name


def write_esp(tmp_path, name, params, data, winepr=False):
    """An ESP (big-endian int32) or WinEPR (little-endian float32, ``DOS Format``) pair; the base path."""
    lines = (["DOS  Format"] if winepr else []) + [f"{key}  {value}" for key, value in params.items()]
    (tmp_path / f"{name}.par").write_text("".join(line + "\n" for line in lines))
    values = np.asarray(data)
    if np.iscomplexobj(values):
        interleaved = np.empty(2 * values.size)
        interleaved[0::2], interleaved[1::2] = values.real.reshape(-1, order="F"), values.imag.reshape(-1, order="F")
        values = interleaved
    else:
        values = values.reshape(-1, order="F")
    (tmp_path / f"{name}.spc").write_bytes(values.astype("<f4" if winepr else ">i4").tobytes())
    return tmp_path / name


BES3T_LINES = [
    "#DESC 1.2 * DESCRIPTOR INFORMATION",
    "DSRC MAN",
    "BSEQ BIG",
    "IKKF REAL",
    "IRFMT D",
    "XTYP IDX",
    "YTYP NODATA",
]


# ------------------------------------------------------------------------------------------- the parsers
@pytest.mark.parametrize(
    "value, expected",
    [
        ("3350.000000", 3350.0),
        ("1.0e-04", 1e-4),
        ("-Inf", -np.inf),
        ("+5", 5.0),
        (".5", 0.5),
        ("5.", 5.0),
        ("3345 G", "3345 G"),
        ("12/03/21", "12/03/21"),
        ("10:23:45", "10:23:45"),
        ("1,2", "1,2"),
        ("True", True),
        ("false", False),
        ("CPLX", "CPLX"),
        ("Inf", "Inf"),
        ("", ""),
        ("0 0 0", np.zeros(3)),
        ("1 2.5 -3e1", np.array([1.0, 2.5, -30.0])),
    ],
)
def test_parsefieldparams_table(value, expected):
    out = eprfiles._parsefieldparams({"K": value})["K"]
    if isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(out, expected)
    elif isinstance(expected, float):
        assert isinstance(out, float) and out == expected
    else:
        assert out == expected and type(out) is type(expected)


def test_read_dsc_rules(tmp_path):
    text = (
        "* a comment line\r\n"
        "#DESC\t1.2 * DESCRIPTOR INFORMATION\r\n"
        "DSRC\tEXP\r\n"
        "TITL\t'my title'\r\n"
        "ONEQ\t'half quoted\r\n"
        "EMPT\t\r\n"
        ".DVC\tsignalChannel, 1.0\r\n"
        "ModAmp\t1.000 G\r\n"
        "SctNorm\tTrue\r\n"
        "DATE\t12/03/21\r\n"
        "VEC\t0 0 0\r\n"
        "PlsSPELPrgTxt\t'\\n\\\r\n"
        ";first\\n\\\r\n"
        "second\\n\\\r\n"
        "'\r\n"
        "DUPL\t1\r\n"
        "DUPL\t2\r\n"
        "LATN\t\xe9\r\n"
        "#MHL\t1.2 * MANIPULATION HISTORY LAYER\r\n"
        "AFTER\t5\r\n"
    )
    (tmp_path / "r.DSC").write_bytes(text.encode("latin-1"))
    raw = eprfiles._read_dsc(tmp_path / "r.DSC")
    assert raw["DSRC"] == "EXP" and raw["TITL"] == "my title" and raw["ONEQ"] == "'half quoted"
    assert raw["EMPT"] == "" and ".DVC" not in raw and "#DESC" not in raw and "*" not in raw
    assert raw["ModAmp"] == "1.000 G" and raw["DUPL"] == "2" and raw["LATN"] == "\xe9"
    assert raw["PlsSPELPrgTxt"] == "\n;first\nsecond\n"  # the continuation, including its last line
    assert "AFTER" not in raw  # #MHL ends the parse
    params = eprfiles._parsefieldparams(raw)
    assert params["SctNorm"] is True and params["DATE"] == "12/03/21" and params["ModAmp"] == "1.000 G"
    np.testing.assert_array_equal(params["VEC"], [0, 0, 0])


def test_read_par_rules(tmp_path):
    (tmp_path / "r.par").write_text("DOS  Format\nJEX  field-sweep\n  HCF  3350.000\n#X  1\nTITL  'name'\nEMP\n")
    raw = eprfiles._read_par(tmp_path / "r.par")
    assert raw == {"DOS": "Format", "JEX": "field-sweep", "HCF": "3350.000", "TITL": "name", "EMP": ""}


def test_num_reads_the_leading_number():
    assert eprfiles._num({"K": "3345 G"}, "K") == 3345 and eprfiles._num({"K": "-1e-4"}, "K") == -1e-4
    with pytest.raises(ValueError, match="Cannot read a number"):
        eprfiles._num({"K": "abc"}, "K")


# ----------------------------------------------------------------------------- BES3T written by eprsave
def test_eprsave_bytes(tmp_path):
    x = np.array([3300.0, 3350.0, 3400.0])
    es.eprsave(tmp_path / "b", x, [1.0, -2.0, 3.0], "a title", 9.5)
    assert (tmp_path / "b.DTA").read_bytes() == struct.pack(">3d", 1.0, -2.0, 3.0)
    lines = (tmp_path / "b.DSC").read_text().split("\n")
    assert lines[0].startswith("* Exported from Python using EasySpyn, ")
    assert lines[1:14] == [
        "#DESC\t1.2 * DESCRIPTOR INFORMATION ***********************",
        "DSRC\tMAN",
        "BSEQ\tBIG",
        "IKKF\tREAL",
        "IRFMT\tD",
        "XTYP\tIDX",
        "YTYP\tNODATA",
        "ZTYP\tNODATA",
        "XPTS\t3",
        "XMIN\t3300",
        "XWID\t100",
        "TITL\t'a title'",
        "*",
    ]
    assert lines[14:17] == [
        "************************************************************",
        "*",
        "#SPL\t1.2 * STANDARD PARAMETER LAYER",
    ]
    assert lines[17] == "OPER\t" and lines[18].startswith("DATE\t") and lines[19].startswith("TIME\t")
    assert lines[20:25] == ["CMNT\t", "SAMP\t", "SFOR\t", "MWFQ\t9.5e+09", ""]
    es.eprsave(tmp_path / "c.DSC", x, np.array([1 + 2j, 3 + 4j, 5 + 6j]))
    assert (tmp_path / "c.DTA").read_bytes() == struct.pack(">6d", 1, 2, 3, 4, 5, 6)
    text = (tmp_path / "c.DSC").read_text()
    assert "IKKF\tCPLX\nIRFMT\tD\nIIFMT\tD\n" in text and "TITL" not in text and "MWFQ" not in text


def test_bes3t_round_trips(tmp_path):
    es.eprsave(tmp_path / "real", X, Y, "spectrum", 9.4)
    x, y, params = es.eprload(tmp_path / "real")
    np.testing.assert_allclose(x, X, rtol=1e-6)  # XMIN and XWID are written with six significant digits
    np.testing.assert_array_equal(y, Y)
    assert y.shape == (32,) and y.dtype == np.float64 and x.dtype == np.float64
    assert params["MWFQ"] == 9.4e9 and params["XPTS"] == 32 and params["TITL"] == "spectrum"
    assert (
        params["DSRC"] == "MAN" and params["BSEQ"] == "BIG" and params["IKKF"] == "REAL" and params["ZTYP"] == "NODATA"
    )

    es.eprsave(tmp_path / "cplx", X, Y * (1 - 0.5j))
    x, y, params = es.eprload(tmp_path / "cplx.DTA")
    np.testing.assert_array_equal(y, Y * (1 - 0.5j))
    assert y.dtype == np.complex128 and params["IKKF"] == "CPLX" and params["IIFMT"] == "D"

    t = np.linspace(0, 6, 7)
    Z = np.outer(Y, np.exp(-t))
    es.eprsave(tmp_path / "two", [X, t], Z)
    x, y, params = es.eprload(tmp_path / "two.DSC")
    assert isinstance(x, list) and len(x) == 2 and y.shape == (32, 7)
    np.testing.assert_allclose(x[0], X, rtol=1e-6)
    np.testing.assert_array_equal(x[1], t)
    np.testing.assert_array_equal(y, Z)
    assert params["YPTS"] == 7 and params["YTYP"] == "IDX"

    es.eprsave(tmp_path / "twoc", [X, t], Z * 1j)
    _, y, _ = es.eprload(tmp_path / "twoc")
    np.testing.assert_array_equal(y, Z * 1j)

    xn = 3300 + 100 * np.linspace(0, 1, 32) ** 2
    es.eprsave(tmp_path / "nonlin", xn, Y)
    x, y, params = es.eprload(tmp_path / "nonlin")
    np.testing.assert_array_equal(x, xn)
    assert params["XTYP"] == "IGD" and params["XFMT"] == "D" and (tmp_path / "nonlin.XGF").stat().st_size == 32 * 8

    tn = np.logspace(0, 1, 7)
    es.eprsave(tmp_path / "nonliny", [X, tn], Z)
    x, y, params = es.eprload(tmp_path / "nonliny")
    np.testing.assert_array_equal(x[1], tn)
    assert params["YTYP"] == "IGD" and params["YFMT"] == "D" and (tmp_path / "nonliny.YGF").exists()


def test_eprsave_errors(tmp_path):
    with pytest.raises(ValueError, match="more than 2 dimensions"):
        es.eprsave(tmp_path / "e", X, np.zeros((2, 2, 2)))
    with pytest.raises(ValueError, match="two axes"):
        es.eprsave(tmp_path / "e", X, np.zeros((32, 3)))


# ------------------------------------------------------------------------------ hand-written descriptors
def test_bes3t_formats_and_byte_orders(tmp_path):
    data = np.array([1, -2, 3, 4])
    for letter, dtype in (("I", "<i4"), ("S", "<i2"), ("C", "<i1"), ("F", "<f4"), ("D", "<f8")):
        lines = ["BSEQ LIT", "IKKF REAL", f"IRFMT {letter}", "XTYP IDX", "XPTS 4", "XMIN 10", "XWID 3"]
        base = write_dsc(tmp_path, "f" + letter, lines, data, dtype)
        x, y, params = es.eprload(base)
        np.testing.assert_array_equal(y, data)
        np.testing.assert_array_equal(x, [10, 11, 12, 13])
        assert y.dtype == np.float64 and params["IRFMT"] == letter
    # two data values per point: a complex one and a real one
    raw = np.array([1, 2, 10, 3, 4, 20, 5, 6, 30], dtype=float)
    base = write_dsc(
        tmp_path, "twov", ["BSEQ BIG", "IKKF CPLX,REAL", "IRFMT D,D", "XTYP IDX", "XPTS 3", "XMIN 0", "XWID 2"], raw
    )
    x, y, _ = es.eprload(base)
    assert isinstance(y, list) and len(y) == 2
    np.testing.assert_array_equal(y[0], [1 + 2j, 3 + 4j, 5 + 6j])
    np.testing.assert_array_equal(y[1], [10, 20, 30])
    # three dimensions, x fastest
    cube = np.arange(24, dtype=float).reshape(2, 3, 4, order="F")
    base = write_dsc(
        tmp_path,
        "cube",
        ["BSEQ BIG", "IKKF REAL", "IRFMT D", "XTYP IDX", "YTYP IDX", "ZTYP IDX", "XPTS 2", "YPTS 3", "ZPTS 4"]
        + ["XMIN 0", "XWID 1", "YMIN 0", "YWID 2", "ZMIN 0", "ZWID 3"],
        cube.reshape(-1, order="F"),
    )
    x, y, _ = es.eprload(base)
    assert y.shape == (2, 3, 4) and len(x) == 3
    np.testing.assert_array_equal(y, cube)
    np.testing.assert_array_equal(x[2], [0, 1, 2, 3])


def test_bes3t_descriptor_warnings_and_errors(tmp_path):
    data = np.arange(4, dtype=float)
    with pytest.warns(UserWarning, match="IKKF not found"):
        _, y, _ = es.eprload(
            write_dsc(tmp_path, "noikkf", ["BSEQ BIG", "IRFMT D", "XTYP IDX", "XPTS 4", "XMIN 0", "XWID 3"], data)
        )
    np.testing.assert_array_equal(y, data)
    with pytest.warns(UserWarning, match="BSEQ not found"):
        _, y, _ = es.eprload(
            write_dsc(tmp_path, "nobseq", ["IKKF REAL", "IRFMT D", "XTYP IDX", "XPTS 4", "XMIN 0", "XWID 3"], data)
        )
    np.testing.assert_array_equal(y, data)
    with pytest.warns(UserWarning, match="zero width"):
        x, _, _ = es.eprload(write_dsc(tmp_path, "zero", BES3T_LINES + ["XPTS 4", "XMIN 7", "XWID 0"], data))
    np.testing.assert_array_equal(x, [1, 2, 3, 4])
    with pytest.warns(UserWarning, match="Could not read companion file"):
        x, _, _ = es.eprload(
            write_dsc(
                tmp_path,
                "noxgf",
                ["BSEQ BIG", "IKKF REAL", "IRFMT D", "XTYP IGD", "XFMT D", "XPTS 4", "XMIN 0", "XWID 3"],
                data,
            )
        )
    np.testing.assert_array_equal(x, [0, 1, 2, 3])
    cases = {
        "IRFMT A": ("Cannot read BES3T data in ASCII format", ["IKKF REAL", "IRFMT A"]),
        "IRFMT 0": ("No BES3T data", ["IKKF REAL", "IRFMT 0"]),
        "IRFMT Q": ("Unknown value for keyword IRFMT", ["IKKF REAL", "IRFMT Q"]),
        "no IRFMT": ("Keyword IRFMT not found", ["IKKF REAL"]),
        "IIFMT": ("must be identical", ["IKKF CPLX", "IRFMT D", "IIFMT F"]),
        "count": ("inconsistent IKKF and IRFMT", ["IKKF CPLX,REAL", "IRFMT D"]),
        "IKKF": ("Unknown value for keyword IKKF", ["IKKF cplx", "IRFMT D"]),
        "BSEQ": ("Unknown value for keyword BSEQ", ["BSEQ MID", "IKKF REAL", "IRFMT D"]),
        "NTUP": ("NTUP axes", ["IKKF REAL", "IRFMT D", "XTYP NTUP"]),
        "no XTYP": ("No XTYP", ["IKKF REAL", "IRFMT D"]),
        "axis type": ("Unknown axis type", ["IKKF REAL", "IRFMT D", "XTYP ABC"]),
        "XFMT": ("Cannot read data format Q", ["IKKF REAL", "IRFMT D", "XTYP IGD", "XFMT Q"]),
    }
    for name, (message, extra) in cases.items():
        lines = ["BSEQ BIG", "XPTS 4", "XMIN 0", "XWID 3"] + extra
        with pytest.raises(ValueError, match=message):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                es.eprload(write_dsc(tmp_path, name.replace(" ", "_"), lines, data))
    with pytest.raises(ValueError, match="No XPTS"):
        es.eprload(write_dsc(tmp_path, "noxpts", BES3T_LINES, data))
    with pytest.raises(ValueError, match="Unable to read all expected data"):
        es.eprload(write_dsc(tmp_path, "short", BES3T_LINES + ["XPTS 8", "XMIN 0", "XWID 7"], data))
    base = write_dsc(
        tmp_path,
        "shortxgf",
        ["BSEQ BIG", "IKKF REAL", "IRFMT D", "XTYP IGD", "XFMT D", "XPTS 4", "XMIN 0", "XWID 3"],
        data,
    )
    (tmp_path / "shortxgf.XGF").write_bytes(struct.pack(">2d", 1, 2))
    with pytest.raises(ValueError, match="Could not read the expected 4 values"):
        es.eprload(base)
    write_dsc(tmp_path, "nodta", BES3T_LINES + ["XPTS 4", "XMIN 0", "XWID 3"])
    with pytest.raises(FileNotFoundError, match="Unable to open data file"):
        es.eprload(tmp_path / "nodta.DSC")
    (tmp_path / "nodsc.DTA").write_bytes(b"")
    with pytest.raises(FileNotFoundError, match="Cannot find the file"):
        es.eprload(tmp_path / "nodsc.DTA")


def test_bes3t_companion_formats_and_mixed_case(tmp_path):
    data = np.arange(4, dtype=float)
    for fmt, dtype in (("F", "<f4"), ("I", "<i4"), ("S", "<i2")):
        base = write_dsc(
            tmp_path, "ax" + fmt, ["BSEQ LIT", "IKKF REAL", "IRFMT D", "XTYP IGD", f"XFMT {fmt}", "XPTS 4"], data, "<f8"
        )
        (tmp_path / f"ax{fmt}.XGF").write_bytes(np.array([5, 6, 8, 11]).astype(dtype).tobytes())
        x, _, _ = es.eprload(base)
        np.testing.assert_array_equal(x, [5, 6, 8, 11])
    # a lower-case descriptor next to an upper-case data file, and the other way round
    (tmp_path / "mixed.dsc").write_text("".join(line + "\n" for line in BES3T_LINES + ["XPTS 4", "XMIN 0", "XWID 3"]))
    (tmp_path / "mixed.DTA").write_bytes(data.astype(">f8").tobytes())
    for name in ("mixed", "mixed.dsc", "mixed.DTA", "MIXED.dta"):
        _, y, _ = es.eprload(tmp_path / name)
        np.testing.assert_array_equal(y, data)


def test_bes3t_scaling(tmp_path):
    data = np.array([2.0, 4.0, 8.0, 16.0])
    spl = ["EXPT CW", "AVGS 4", "RCAG 30", "SPTP 0.04096", "MWPW 0.002", "STMP 300"]
    base = write_dsc(tmp_path, "sc", BES3T_LINES + ["XPTS 4", "XMIN 0", "XWID 3"] + spl, data)
    expected = {
        "n": data / 4,
        "G": data / 10 ** (30 / 20),
        "c": data / (0.04096 * 1000),
        "P": data / np.sqrt(0.002 * 1000),
        "T": data * 300,
    }
    for letter, values in expected.items():
        np.testing.assert_allclose(es.eprload(base, letter)[1], values, rtol=1e-14)
    _, y, _ = es.eprload(base, "nPGTc")
    np.testing.assert_allclose(y, data / 4 / 10**1.5 / 40.96 / np.sqrt(2) * 300, rtol=1e-13)
    np.testing.assert_array_equal(es.eprload(base)[1], data)
    with pytest.raises(ValueError, match="can only contain"):
        es.eprload(base, "x")
    pre = write_dsc(tmp_path, "pre", BES3T_LINES + ["XPTS 4", "XMIN 0", "XWID 3"] + spl + ["SctNorm True"], data)
    with pytest.raises(ValueError, match="already averaged"):
        es.eprload(pre, "n")
    np.testing.assert_allclose(es.eprload(pre, "P")[1], expected["P"])
    pulse = write_dsc(tmp_path, "pulse", BES3T_LINES + ["XPTS 4", "XMIN 0", "XWID 3", "EXPT PLS", "AVGS 2"], data)
    with pytest.raises(ValueError, match="not CW-EPR data"):
        es.eprload(pulse, "P")
    np.testing.assert_array_equal(es.eprload(pulse, "Gc")[1], data)  # ignored outside cw
    np.testing.assert_array_equal(es.eprload(pulse, "n")[1], data / 2)
    for letter, key in (("n", "AVGS"), ("G", "RCAG"), ("c", "SPTP"), ("P", "MWPW"), ("T", "STMP")):
        lines = BES3T_LINES + ["XPTS 4", "XMIN 0", "XWID 3"] + [line for line in spl if not line.startswith(key)]
        with pytest.raises(ValueError, match=key):
            es.eprload(write_dsc(tmp_path, "miss" + key, lines, data), letter)
    with pytest.raises(ValueError, match="Missing EXPT"):
        es.eprload(write_dsc(tmp_path, "noexpt", BES3T_LINES + ["XPTS 4", "XMIN 0", "XWID 3", "AVGS 2"], data), "n")
    # scaling applies to every data value of a file with several
    base = write_dsc(
        tmp_path,
        "scv",
        ["BSEQ BIG", "IKKF REAL,REAL", "IRFMT D,D", "XTYP IDX", "XPTS 2", "XMIN 0", "XWID 1", "EXPT CW", "AVGS 2"],
        [1.0, 10.0, 2.0, 20.0],
    )
    _, y, _ = es.eprload(base, "n")
    np.testing.assert_array_equal(y[0], [0.5, 1]) and np.testing.assert_array_equal(y[1], [5, 10])


# ----------------------------------------------------------------------------------------------- ESP
def test_esp_axes_and_sizes(tmp_path):
    data = np.arange(1, 9) * 100
    x, y, params = es.eprload(write_esp(tmp_path, "hcf", {"ANZ": 8, "HCF": 3350, "HSW": 100}, data))
    np.testing.assert_allclose(x, 3350 + 50 * np.linspace(-1, 1, 8))
    np.testing.assert_array_equal(y, data)
    assert y.shape == (8,) and y.dtype == np.float64
    assert params["JEX"] == "field-sweep" and params["JEY"] == "" and params["ANZ"] == 8
    x, _, _ = es.eprload(write_esp(tmp_path, "gst", {"ANZ": 8, "GST": 3300, "GSI": 100}, data))
    np.testing.assert_allclose(x, 3300 + 100 * np.linspace(0, 1, 8))
    x, _, _ = es.eprload(write_esp(tmp_path, "both", {"ANZ": 8, "HCF": 1, "HSW": 1, "GST": 3300, "GSI": 100}, data))
    np.testing.assert_allclose(x, 3300 + 100 * np.linspace(0, 1, 8))  # GST/GSI win
    x, _, _ = es.eprload(
        write_esp(tmp_path, "endor", {"ANZ": 8, "JEX": "ENDOR", "HCF": 1, "HSW": 1, "GST": 1, "GSI": 20}, data)
    )
    np.testing.assert_allclose(x, 1 + 20 * np.linspace(0, 1, 8))
    x, _, _ = es.eprload(write_esp(tmp_path, "time", {"ANZ": 8, "JEX": "Time-Sweep", "RCT": 40.96}, data))
    np.testing.assert_allclose(x, np.arange(8) * 40.96 / 1e3)
    x, _, _ = es.eprload(write_esp(tmp_path, "norange", {"ANZ": 8, "HCF": 3350}, data))
    np.testing.assert_allclose(x, 3350 + 25 * np.linspace(-1, 1, 8))  # HSW defaults to 50
    x, _, _ = es.eprload(write_esp(tmp_path, "emx1", {"ANZ": 8, "HSW": 100, "XXLB": 3300, "XXWI": 70}, data))
    np.testing.assert_allclose(x, 3300 + np.linspace(0, 70, 8))
    x, _, _ = es.eprload(write_esp(tmp_path, "nothing", {"ANZ": 8, "GSI": 5}, data))
    assert x.size == 0  # EasySpin returns an empty axis here as well
    with pytest.raises(ValueError, match="Could not determine abscissa range"):
        es.eprload(write_esp(tmp_path, "nocentre", {"ANZ": 8, "XXLB": 3300, "XXWI": 70}, data))
    # sizes: RES and XPLS override ANZ, a complex 1-D set halves ANZ, the default is 1024 points
    x, y, _ = es.eprload(write_esp(tmp_path, "res", {"ANZ": 16, "RES": 8, "HCF": 3350, "HSW": 100}, data))
    assert y.shape == (8,)
    _, y, _ = es.eprload(write_esp(tmp_path, "xpls", {"ANZ": 16, "RES": 4, "XPLS": 8, "HCF": 3350, "HSW": 100}, data))
    assert y.shape == (8,)
    cplx = np.arange(1, 9) + 1j * np.arange(11, 19)
    _, y, _ = es.eprload(write_esp(tmp_path, "cplx", {"ANZ": 16, "JSS": 16, "HCF": 3350, "HSW": 100}, cplx))
    np.testing.assert_array_equal(y, cplx)
    _, y, _ = es.eprload(write_esp(tmp_path, "default", {"HCF": 3350, "HSW": 100}, np.arange(1024)))
    assert y.shape == (1024,)
    # 2-D: SSX/SSY with ANZ, EMX axes as a list, a slice saved by WinEPR
    Z = np.arange(24).reshape(8, 3, order="F")
    x, y, _ = es.eprload(
        write_esp(tmp_path, "twod", {"JSS": 4096, "SSX": 8, "SSY": 3, "ANZ": 24, "HCF": 3350, "HSW": 100}, Z)
    )
    np.testing.assert_array_equal(y, Z) and np.testing.assert_allclose(x, 3350 + 50 * np.linspace(-1, 1, 8))
    with pytest.raises(ValueError, match="inconsistent"):
        es.eprload(
            write_esp(tmp_path, "twodbad", {"JSS": 4096, "SSX": 8, "SSY": 3, "ANZ": 23, "HCF": 3350, "HSW": 100}, Z)
        )
    x, y, _ = es.eprload(
        write_esp(
            tmp_path, "emx2", {"JSS": 4096, "SSX": 8, "SSY": 3, "XXLB": 3300, "XXWI": 70, "XYLB": 0, "XYWI": 20}, Z
        )
    )
    assert isinstance(x, list) and y.shape == (8, 3)
    np.testing.assert_allclose(x[1], [0, 10, 20])
    _, y, _ = es.eprload(write_esp(tmp_path, "slice", {"RES": 8, "REY": 3, "HCF": 3350, "HSW": 100}, data))
    assert y.shape == (8,)
    # WinEPR: little-endian floats
    _, y, params = es.eprload(write_esp(tmp_path, "win", {"RES": 8, "HCF": 3350, "HSW": 100}, data / 7, winepr=True))
    np.testing.assert_allclose(y, data / 7, rtol=1e-6)
    assert params["DOS"] == "Format"


def test_esp_scaling(tmp_path):
    data = np.arange(1, 9) * 100
    keys = {"ANZ": 8, "HCF": 3350, "HSW": 100, "JSD": 4, "RRG": 2e4, "MP": 20, "TE": 298, "RCT": 40.96}
    base = write_esp(tmp_path, "sc", keys, data)
    expected = {"n": data / 4, "G": data / 2e4, "P": data / np.sqrt(20), "T": data * 298, "c": data / 40.96}
    for letter, values in expected.items():
        np.testing.assert_allclose(es.eprload(base, letter)[1], values, rtol=1e-14)
    np.testing.assert_allclose(es.eprload(base, "nGPTc")[1], data / 4 / 2e4 / np.sqrt(20) * 298 / 40.96, rtol=1e-13)
    for letter, key in (("n", "JSD"), ("G", "RRG"), ("P", "MP"), ("T", "TE"), ("c", "RCT")):
        with pytest.raises(ValueError, match=key):
            es.eprload(write_esp(tmp_path, "miss" + key, {k: v for k, v in keys.items() if k != key}, data), letter)
    with pytest.raises(ValueError, match="TE is zero"):
        es.eprload(write_esp(tmp_path, "te0", {**keys, "TE": 0}, data), "T")
    Z = np.ones((8, 3))
    sweep = {
        "JSS": 4096,
        "SSX": 8,
        "SSY": 3,
        "ANZ": 24,
        "JEY": "mw-power-sweep",
        "MP": 200,
        "XXLB": 3300,
        "XXWI": 70,
        "XYLB": 0,
        "XYWI": 20,
    }
    _, y, _ = es.eprload(write_esp(tmp_path, "power", sweep, Z), "P")
    np.testing.assert_allclose(y, Z / np.sqrt(200 * 10 ** (-np.array([0, 10, 20]) / 10)))


# ------------------------------------------------------------------------- resolution, dispatch, text
def test_resolution_and_dispatch(tmp_path):
    es.eprsave(tmp_path / "s", X, Y)
    (tmp_path / "sub").mkdir()
    for name in ("s", "s.DSC", "s.DTA", "s.dta", "S.dsc"):
        _, y, _ = es.eprload(str(tmp_path / name))
        np.testing.assert_array_equal(y, Y)
    base = write_esp(tmp_path, "e", {"ANZ": 8, "HCF": 3350, "HSW": 100}, np.arange(8))
    assert es.eprload(base)[1].shape == (8,) and es.eprload(tmp_path / "e.PAR")[1].shape == (8,)
    with pytest.raises(ValueError, match="is a directory"):
        es.eprload(tmp_path / "sub")
    with pytest.raises(FileNotFoundError, match="does not exist"):
        es.eprload(tmp_path / "missing")
    with pytest.raises(FileNotFoundError, match="does not exist"):
        es.eprload(tmp_path / "missing.DTA")
    for ext, name in (
        ("d01", "SpecMan"),
        ("spe", "Magnettech"),
        ("xml", "Magnettech"),
        ("ESR", "Active"),
        ("dat", "Adani"),
        ("json", "Adani"),
        ("eco", "qese"),
        ("plt", "MAGRES"),
        ("spk", "Varian"),
        ("ref", "Varian"),
        ("d00", "ETH"),
    ):
        (tmp_path / f"u.{ext}").write_text("1 2\n")
        with pytest.raises(NotImplementedError, match=name):
            es.eprload(tmp_path / f"u.{ext}")
    (tmp_path / "jeol.xyz").write_bytes(b"spin-something" + bytes(32))
    with pytest.raises(NotImplementedError, match="JEOL"):
        es.eprload(tmp_path / "jeol.xyz")
    (tmp_path / "odd.xyz").write_bytes(bytes(40))
    with pytest.raises(ValueError, match="Unsupported file extension .xyz"):
        es.eprload(tmp_path / "odd.xyz")
    (tmp_path / "cols.dat").write_text("1 5\n2 6\n")
    x, y, _ = es.eprload(tmp_path / "cols.dat", format="text")
    np.testing.assert_array_equal(y, [5, 6])
    with pytest.raises(ValueError, match="format must be"):
        es.eprload(tmp_path / "cols.dat", format="xyz")
    with pytest.raises(ValueError, match="Scaling does not work for this file type"):
        es.eprload(tmp_path / "cols.dat", "n", format="text")


def test_text_columns(tmp_path):
    (tmp_path / "ws.txt").write_text("# a comment\nfield intensity\n3300 1.5\n3310 2.5\n\n3320 -1\n")
    x, y, params = es.eprload(tmp_path / "ws.txt")
    np.testing.assert_array_equal(x, [3300, 3310, 3320]) and np.testing.assert_array_equal(y, [1.5, 2.5, -1])
    assert (
        params["Format"] == "text"
        and params["Comments"] == ["# a comment", "field intensity"]
        and params["nColumns"] == 2
    )
    (tmp_path / "c.csv").write_text("% header\n3300,1.5,7\n3310,2.5,8\n")
    x, y, _ = es.eprload(tmp_path / "c.csv")
    assert y.shape == (2, 2) and np.array_equal(y[:, 1], [7, 8])
    (tmp_path / "s.asc").write_text("1;2\n3;4\n")
    x, y, _ = es.eprload(tmp_path / "s.asc")
    np.testing.assert_array_equal(x, [1, 3])
    (tmp_path / "one.txt").write_text("5\n6\n7\n")
    x, y, _ = es.eprload(tmp_path / "one.txt")
    np.testing.assert_array_equal(x, [1, 2, 3]) and np.testing.assert_array_equal(y, [5, 6, 7])
    (tmp_path / "ragged.txt").write_text("1 2\n3\n")
    with pytest.raises(ValueError, match="columns"):
        es.eprload(tmp_path / "ragged.txt")
    (tmp_path / "empty.txt").write_text("# nothing\n")
    with pytest.raises(ValueError, match="No numbers"):
        es.eprload(tmp_path / "empty.txt")


# ------------------------------------------------------------------------------------- the fit link
def test_exp_from_params():
    x = np.linspace(3300, 3400, 101)
    bes3t = {
        "IKKF": "REAL",
        "XPTS": 101.0,
        "XUNI": "G",
        "XNAM": "Field",
        "EXPT": "CW",
        "MWFQ": 9.4e9,
        "RCHM": 1.0,
        "B0MA": 1e-4,
        "STMP": 295.0,
    }
    assert es.exp_from_params(bes3t, x) == {
        "mwFreq": 9.4,
        "Range": [330.0, 340.0],
        "nPoints": 101,
        "Harmonic": 1,
        "ModAmp": 0.1,
        "Temperature": 295.0,
    }
    dsl = {
        "IKKF": "REAL",
        "XPTS": 101.0,
        "XUNI": "mT",
        "MWFQ": 9.4e9,
        "Harmonic": 2.0,
        "ModAmp": "1.000 G",
        "STMP": 0.0,
    }
    assert es.exp_from_params(dsl, x) == {
        "mwFreq": 9.4,
        "Range": [3300.0, 3400.0],
        "nPoints": 101,
        "Harmonic": 2,
        "ModAmp": 0.1,
    }
    assert es.exp_from_params({"IKKF": "REAL", "MWFQ": 9.4e9}, [x, np.arange(3)], field_unit="G")["Range"] == [
        330.0,
        340.0,
    ]
    assert es.exp_from_params({"IKKF": "REAL", "XUNI": "T", "MWFQ": 9.4e9}, x / 1e4)["Range"] == pytest.approx(
        [330.0, 340.0]
    )
    esp = {"JSS": 0.0, "HCF": 3350.0, "MF": 9.78, "RHA": 1.0, "RMA": 2.0, "TE": 298.0, "JEX": "field-sweep", "JUN": "G"}
    assert es.exp_from_params(esp, x) == {
        "mwFreq": 9.78,
        "Range": [330.0, 340.0],
        "nPoints": 101,
        "Harmonic": 1,
        "ModAmp": 0.2,
        "Temperature": 298.0,
    }
    assert es.exp_from_params({"Format": "text"}, x / 10, field_unit="mT", mw_freq=9.5) == {
        "mwFreq": 9.5,
        "Range": [330.0, 340.0],
        "nPoints": 101,
    }
    assert es.exp_from_params(bes3t, x, mw_freq=34.0)["mwFreq"] == 34.0
    with pytest.raises(ValueError, match="field unit"):
        es.exp_from_params({"IKKF": "REAL", "MWFQ": 9.4e9}, x)
    with pytest.raises(ValueError, match="not a field unit"):
        es.exp_from_params({"IKKF": "REAL", "XUNI": "s", "MWFQ": 9.4e9}, x)
    with pytest.raises(ValueError, match="microwave frequency"):
        es.exp_from_params({"Format": "text"}, x, field_unit="G")
    with pytest.raises(ValueError, match="not a uniform sweep"):
        es.exp_from_params(bes3t, 3300 + 100 * np.linspace(0, 1, 101) ** 2)
    with pytest.raises(ValueError, match="descending"):
        es.exp_from_params(bes3t, x[::-1])
    with pytest.raises(ValueError, match="not cw data"):
        es.exp_from_params({**bes3t, "EXPT": "PLS"}, x)
    with pytest.raises(ValueError, match="not a field sweep"):
        es.exp_from_params({**bes3t, "XNAM": "Time"}, x)
    with pytest.raises(ValueError, match="not field-sweep"):
        es.exp_from_params({**esp, "JEX": "Time-Sweep"}, x)
    with pytest.raises(ValueError, match="fewer than two"):
        es.exp_from_params(bes3t, x[:1])
    with pytest.raises(ValueError, match="Cannot tell the file format"):
        es.exp_from_params({}, x)


def test_fit_of_a_saved_spectrum(tmp_path):
    truth = {"g": [2.0, 2.1], "lw": 3.0}
    exp = {"mwFreq": 9.5, "Range": [300, 360], "nPoints": 200}
    spectrum = es.pepper(truth, exp, {"FuzzLevel": 0})
    es.eprsave(tmp_path / "powder", spectrum.x * 10, spectrum.spec, "axial", 9.5)  # the file in Gauss
    (tmp_path / "powder.DSC").write_text(
        (tmp_path / "powder.DSC").read_text() + "EXPT\tCW\nXUNI\tG\nXNAM\tField\nRCHM\t1\nB0MA\t0\n"
    )
    x, y, params = es.eprload(tmp_path / "powder")
    exp_file = es.exp_from_params(params, x)
    assert exp_file == {"mwFreq": 9.5, "Range": [300.0, 360.0], "nPoints": 200, "Harmonic": 1}
    start = [{"g": [2.01, 2.09], "lw": 2.5}, exp_file, {"FuzzLevel": 0}]
    result = es.esfit(y, es.pepper, start, [{"g": [0.03, 0.03], "lw": 1.5}], {"Method": "simplex", "Verbosity": 0})
    np.testing.assert_allclose(result.argsfit[0]["g"], truth["g"], atol=1e-3)
    assert abs(result.argsfit[0]["lw"] - truth["lw"]) < 0.2
