"""Every example of the browser playground (try.html) runs natively, with stand-ins for the page's plotting
helpers that record what they were given: each example draws something (the ctps pathway example prints
instead, the relaxation pathway example draws a chord diagram), and the EPR examples never leave the NMR
convention of an inverted axis in place. Examples marked data-slow
(the 12800-orientation powder) are skipped."""

import re
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
BLOCK = re.compile(r'<script type="text/plain" class="example" id="([\w-]+)"([^>]*)>\n(.*?)</script>', re.S)


def examples() -> list[tuple[str, str, str]]:
    page = (ROOT / "try.html").read_text(encoding="utf-8")
    out = []
    for match in BLOCK.finditer(page):
        attrs = dict(re.findall(r'data-(\w+)="([^"]*)"', match.group(2)))
        out.append((match.group(1), attrs.get("group", ""), attrs.get("slow", ""), match.group(3)))
    return out


ALL = examples()


def _is_spectrum(obj) -> bool:
    return hasattr(obj, "spec") and hasattr(obj, "x") and hasattr(obj, "info")


class Recorder:
    """The playground's show_spectrum, show_fid and show_image with the same signatures and invert rules."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def show_spectrum(self, y, x=None, *, title="", xlabel="", ylabel="intensity", invert=None, names=None):
        spectrum = _is_spectrum(y) or (isinstance(y, (list, tuple)) and y and all(_is_spectrum(v) for v in y))
        if spectrum:
            invert = False if invert is None else invert
            y = y.spec if _is_spectrum(y) else [v.spec for v in y]
        if invert is None:
            invert = True
        series = [np.asarray(v).ravel() for v in y] if isinstance(y, (list, tuple)) else [np.asarray(y).ravel()]
        assert all(s.size == series[0].size and s.size > 0 for s in series)
        if x is not None and not isinstance(x, tuple):
            assert np.asarray(x).ravel().size == series[0].size
        self.calls.append({"kind": "line", "invert": bool(invert), "title": title, "xlabel": xlabel})

    def show_fid(self, fid, sweep, *, title="Free induction decay"):
        assert np.asarray(fid).ndim == 1
        self.calls.append({"kind": "fid", "invert": False, "title": title, "xlabel": "time / s"})

    def show_image(self, z, x=None, y=None, *, title="", xlabel="", ylabel="", invert=None):
        z = np.asarray(z)
        assert z.ndim == 2
        if x is not None:
            assert np.asarray(x).size == z.shape[1]
        if y is not None:
            assert np.asarray(y).size == z.shape[0]
        self.calls.append({"kind": "image", "invert": True if invert is None else bool(invert), "title": title})

    def show_pathways(self, pathways, *, title="", classes=None, merge=None, split=False, min_rate=0.0):
        from spynach.plotting import chord_layout

        layout = chord_layout(pathways, classes=classes, merge=merge, split=split, min_rate=min_rate)
        assert len(layout["arcs"]) == len(pathways.operators)
        assert all(0 <= ribbon["colour"] < len(layout["names"]) for ribbon in layout["ribbons"])
        self.calls.append({"kind": "chord", "invert": False, "title": title, "ribbons": len(layout["ribbons"])})


@pytest.mark.parametrize("example", [e for e in ALL if not e[2]], ids=lambda e: e[0])
def test_example_runs_and_draws(example, capsys):
    example_id, group, _, code = example
    recorder = Recorder()
    ns = {
        "__name__": "__main__",
        "show_spectrum": recorder.show_spectrum,
        "show_fid": recorder.show_fid,
        "show_image": recorder.show_image,
        "show_pathways": recorder.show_pathways,
    }
    exec(compile(code, f"<{example_id}>", "exec"), ns, ns)
    printed = capsys.readouterr().out
    if example_id == "ex-ctps":
        assert not recorder.calls and "pathways enumerated" in printed
        return
    assert recorder.calls, f"{example_id} drew nothing"
    if example_id == "ex-pathways":
        assert [call["kind"] for call in recorder.calls] == ["chord"] and recorder.calls[0]["ribbons"] > 0
        assert "DD/CSA" in printed
        return
    if example_id.startswith("ex-ulf-"):
        # zero and ultralow field: the population chord diagram, then line frequencies and times left to right
        assert recorder.calls[0]["kind"] == "chord" and recorder.calls[0]["ribbons"] > 0
        assert all(not call["invert"] for call in recorder.calls[1:]) and "levels |K, F, M>" in printed
    elif group.startswith("EPR"):
        assert all(not call["invert"] for call in recorder.calls), f"{example_id} left an EPR axis inverted"
    else:
        assert any(call["invert"] or call["kind"] == "fid" for call in recorder.calls)


def test_examples_are_grouped_and_unique():
    ids = [e[0] for e in ALL]
    assert len(ids) == len(set(ids))
    groups = {e[1] for e in ALL}
    assert groups == {"NMR (spynach)", "EPR (EasySpyn)", "Pathways (ctps)", "Relaxation (spynach)"}
    assert sum(e[1].startswith("EPR") for e in ALL) == 11
