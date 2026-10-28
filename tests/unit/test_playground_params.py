"""The Parameters view of the browser playground: assets/playground-params.js reads the "GUI:" comments of an
example, and writes a changed value back into its source. The module is plain JavaScript, so the tests run it
under Node (skipped without it): every parameter is documented, the default value writes back unchanged, values
outside the range are refused, and every example still runs with each parameter at the ends of its range, with
every other choice of a list, and with every checkbox flipped."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from test_playground_examples import ALL, ROOT, Recorder  # noqa: E402

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="needs Node")

SCRIPT = """
import { readFileSync } from "node:fs";
import { parseParams, setParam, validate } from %r;
const examples = JSON.parse(readFileSync(0, "utf8"));
const out = {};
for (const { id, source } of examples) {
  const params = parseParams(source);
  const entry = { params, identity: true, refused: [], variants: [] };
  entry.unused = params.filter((p) => !p.used).map((p) => p.name);
  entry.duplicates = params.filter((p) => p.duplicate).map((p) => p.name);
  for (const p of params) {
    if (!p.editable) { entry.identity = false; continue; }
    if (setParam(source, p.name, p.value, p) !== source) entry.identity = false;
    if (p.min !== null) {
      const below = p.kind === "list" ? p.value.map(() => p.min - 1) : p.min - 1;
      try { setParam(source, p.name, below); } catch (e) { entry.refused.push(p.name); }
      for (const edge of [p.min, p.max]) {
        const value = p.kind === "list" ? p.value.map(() => edge) : edge;
        if (validate(p, value)) throw new Error(`${id} ${p.name}: ${edge} is not valid`);
        entry.variants.push({ name: p.name, edge, source: setParam(source, p.name, value) });
      }
    }
    const others = p.kind === "choice" ? p.choices.filter((c) => c !== p.value) : p.kind === "bool" ? [!p.value] : [];
    for (const value of others) {
      entry.variants.push({ name: p.name, edge: value, source: setParam(source, p.name, value) });
    }
  }
  out[id] = entry;
}
// What the form does when the code is rewritten under it
const base = "J = 12.0  # GUI: J | Hz | Coupling | 0..30\\nK = 3.0  # GUI: K | Hz | Other | 0..30\\nprint(J, K)\\n";
const flags = (src) =>
  parseParams(src).map((p) => ({ name: p.name, used: p.used, duplicate: p.duplicate, editable: p.editable }));
out.synthetic = {
  intact: flags(base),
  oneUseRemoved: flags(base.replace("print(J, K)", "print(K)")),
  onlyInComment: flags(base.replace("print(J, K)", "print(K)  # J is gone")),
  repeated: flags(base + "K = 5.0  # GUI: K | Hz | Again | 0..30\\n"),
  noLines: flags("print(1)\\n"),
};
console.log(JSON.stringify(out));
"""


def _run_node() -> dict:
    module = (ROOT / "assets" / "playground-params.js").as_uri()
    payload = json.dumps([{"id": e[0], "source": e[3]} for e in ALL])
    result = subprocess.run(
        ["node", "--input-type=module", "-e", SCRIPT % module], input=payload, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


@pytest.fixture(scope="module")
def parsed() -> dict:
    return _run_node()


def test_parameters_are_documented(parsed):
    converted = {k: v for k, v in parsed.items() if k != "synthetic" and v["params"]}
    assert {"ex-epr-pepper", "ex-cosy", "ex-epr-saffron"} <= set(converted)
    for example_id, entry in converted.items():
        names = [p["name"] for p in entry["params"]]
        assert len(names) == len(set(names)), f"{example_id}: a parameter is listed twice"
        for p in entry["params"]:
            where = f"{example_id}.{p['name']}"
            assert p["editable"], f"{where}: the literal on its line is not a number, list, bool or choice"
            assert p["description"], f"{where}: no description"
            assert p["label"] and p["label"] != p["name"], f"{where}: no label"
            if p["kind"] in ("number", "list"):
                assert p["unit"], f"{where}: no unit (write '-' for a dimensionless value)"
                assert p["min"] is not None and p["max"] is not None, f"{where}: no range"
                values = p["value"] if p["kind"] == "list" else [p["value"]]
                assert all(p["min"] <= v <= p["max"] for v in values), f"{where}: the default is outside the range"
            if p["kind"] == "choice":
                assert p["value"] in p["choices"], f"{where}: the default is not among the choices"


def test_every_shipped_parameter_is_used_and_listed_once(parsed):
    for example_id, entry in parsed.items():
        if example_id == "synthetic":
            continue
        assert not entry["unused"], f"{example_id}: {entry['unused']} are not used by the code"
        assert not entry["duplicates"], f"{example_id}: {entry['duplicates']} are assigned twice"


def test_a_rewritten_source_is_flagged_not_rejected(parsed):
    synthetic = parsed["synthetic"]
    assert all(p["used"] and not p["duplicate"] and p["editable"] for p in synthetic["intact"])
    by_name = {p["name"]: p for p in synthetic["oneUseRemoved"]}
    assert by_name["J"]["used"] is False and by_name["K"]["used"] is True
    assert {p["name"]: p["used"] for p in synthetic["onlyInComment"]}["J"] is False  # a comment is not a use
    repeated = synthetic["repeated"]
    assert [p["duplicate"] for p in repeated] == [False, False, True] and not repeated[2]["editable"]
    assert synthetic["noLines"] == []


def test_defaults_write_back_unchanged_and_out_of_range_is_refused(parsed):
    for example_id, entry in parsed.items():
        if example_id != "synthetic" and entry["params"]:
            assert entry["identity"], f"{example_id}: setting the default changed the source"
            ranged = [p["name"] for p in entry["params"] if p["min"] is not None]
            assert sorted(entry["refused"]) == sorted(ranged), f"{example_id}: a value below the range was accepted"


def _variants() -> list[tuple[str, str, str]]:
    if shutil.which("node") is None:
        return []
    out = []
    for example_id, entry in _run_node().items():
        if example_id == "synthetic":
            continue
        slow = any(e[0] == example_id and e[2] for e in ALL)
        if slow:
            continue
        out += [(example_id, f"{v['name']}={v['edge']}", v["source"]) for v in entry["variants"]]
    return out


@pytest.mark.parametrize("variant", _variants(), ids=lambda v: f"{v[0]}-{v[1]}")
def test_example_runs_with_each_parameter_at_the_ends_of_its_range(variant):
    example_id, _, source = variant
    recorder = Recorder()
    ns = {
        "__name__": "__main__",
        "show_spectrum": recorder.show_spectrum,
        "show_fid": recorder.show_fid,
        "show_image": recorder.show_image,
        "show_pathways": recorder.show_pathways,
    }
    exec(compile(source, f"<{example_id}>", "exec"), ns, ns)
    if example_id != "ex-ctps":  # the pathway example prints a table instead of drawing
        assert recorder.calls
