/* Spynach in Pyodide. Every simulation runs in this worker, off the page thread.
   Pyodide 314 refuses classic workers, so this file is loaded as a module.
   The page must be served over HTTP from the repository root so src/ can be fetched. */
import { loadPyodide } from "https://cdn.jsdelivr.net/pyodide/v314.0.7/full/pyodide.mjs";

const INDEX_URL = "https://cdn.jsdelivr.net/pyodide/v314.0.7/full/";

const SOURCE = [
  "src/spynach/__init__.py",
  "src/spynach/_isotope_data.py",
  "src/spynach/assume.py",
  "src/spynach/backend.py",
  "src/spynach/basis.py",
  "src/spynach/contexts.py",
  "src/spynach/conventions.py",
  "src/spynach/filters.py",
  "src/spynach/hamiltonian.py",
  "src/spynach/isotopes.py",
  "src/spynach/kinetics.py",
  "src/spynach/molecules.py",
  "src/spynach/operators.py",
  "src/spynach/oracle/__init__.py",
  "src/spynach/oracle/compare.py",
  "src/spynach/oracle/fixture.py",
  "src/spynach/plotting.py",
  "src/spynach/processing.py",
  "src/spynach/propagation.py",
  "src/spynach/reduction.py",
  "src/spynach/relaxation.py",
  "src/spynach/sequences.py",
  "src/spynach/states.py",
  "src/spynach/system.py",
  "src/spynach/tolerances.py",
  "src/spynach/utilities.py",
  "src/ctps/__init__.py",
  "src/ctps/engine/__init__.py",
  "src/ctps/engine/gradient.py",
  "src/ctps/engine/paths.py",
  "src/ctps/engine/phase.py",
  "src/ctps/engine/pulse.py",
  "src/ctps/engine/rank.py",
  "src/ctps/io.py",
  "src/ctps/model.py",
  "src/ctps/optimize.py",
  "src/ctps/plot.py",
  "src/ctps/simulate.py",
];

const RUNNER = `
import os
import re
import time
import traceback

import numpy as np
import spynach
from js import addSeries, beginImage, beginLine, reportProgress, setImageZ
from pyodide.http import pyfetch

state = {"called": False}
RUN_ERROR = ""

async def prefetch(code):
    names = set(re.findall(r"""["']((?:leb_|rep_|icos_)[A-Za-z0-9_]+|single_crystal)["']""", code))
    if not names:
        return
    os.makedirs("/repo/src/spynach/grids", exist_ok=True)
    for name in sorted(names):
        dest = f"/repo/src/spynach/grids/{name}.npz"
        if os.path.exists(dest):
            continue
        try:
            resp = await pyfetch(f"src/spynach/grids/{name}.npz")
        except Exception as exc:
            print(f"could not fetch grid {name}: {exc}")
            continue
        if getattr(resp, "status", 0) != 200:
            print(f"grid {name!r} is not on the server")
            continue
        data = await resp.bytes()
        with open(dest, "wb") as handle:
            handle.write(data)
        print(f"loaded grid {name} ({len(data)} bytes)")

def f64(values):
    return np.ascontiguousarray(values, dtype=np.float64)

def show_spectrum(y, x=None, *, title="", xlabel="", ylabel="intensity", invert=True, names=None):
    state["called"] = True
    series = []
    if isinstance(y, (list, tuple)):
        for i, yi in enumerate(y):
            label = names[i] if names and i < len(names) else f"series {i + 1}"
            series.append((label, np.real(np.asarray(yi)).ravel()))
    else:
        arr = np.asarray(y)
        if arr.ndim == 2 and min(arr.shape) > 1:
            raise ValueError("show_spectrum draws 1-D data; use show_image for a 2-D spectrum.")
        arr = arr.ravel()
        complex_data = np.iscomplexobj(arr)
        series.append(("real" if complex_data else "", np.real(arr)))
        if complex_data and np.any(arr.imag):
            series.append(("imag", np.imag(arr)))
    if not series or series[0][1].size == 0:
        raise ValueError("show_spectrum received no points.")
    n = int(series[0][1].shape[0])
    if x is None:
        x = np.arange(n, dtype=float)
        if not xlabel:
            xlabel = "point"
        invert = False
    elif isinstance(x, tuple) and len(x) == 2 and isinstance(x[1], str):
        if not xlabel:
            xlabel = x[1]
        x = np.asarray(x[0], dtype=float).ravel()
    else:
        x = np.asarray(x, dtype=float).ravel()
    if x.shape[0] != n:
        m = int(min(x.shape[0], n))
        print(f"show_spectrum: axis length {x.shape[0]} and data length {n} differ; plotting {m} points")
        x = x[:m]
        series = [(label, yi[:m]) for label, yi in series]
    beginLine(str(title), str(xlabel), str(ylabel), bool(invert), f64(x))
    for label, yi in series:
        addSeries(str(label), f64(yi))

def show_fid(fid, sweep, *, title="Free induction decay"):
    fid = np.asarray(fid)
    if fid.ndim != 1:
        raise ValueError("show_fid expects a 1-D FID. Fourier-transform a 2-D FID and call show_image.")
    dwell = float(np.asarray(sweep).reshape(-1)[0])
    show_spectrum(fid, np.arange(fid.shape[0]) / dwell, title=title, xlabel="time / s", ylabel="signal", invert=False)

def downsample(z, x, y, limit=256):
    ny, nx = z.shape
    fy = int(np.ceil(ny / limit))
    fx = int(np.ceil(nx / limit))
    if fy == 1 and fx == 1:
        return z, x, y
    ny2, nx2 = ny // fy, nx // fx
    block = z[: ny2 * fy, : nx2 * fx].reshape(ny2, fy, nx2, fx)
    pos = block.max(axis=(1, 3))
    neg = block.min(axis=(1, 3))
    out = np.where(np.abs(pos) >= np.abs(neg), pos, neg)

    def take(axis, n_full, n_out, factor):
        if axis is None:
            return None
        idx = (np.arange(n_out) * factor + (factor - 1) / 2).astype(int)
        idx = np.clip(idx, 0, min(len(axis), n_full) - 1)
        return np.asarray(axis)[idx]

    print(f"show_image: binned {ny} x {nx} to {out.shape[0]} x {out.shape[1]}")
    return out, take(x, nx, nx2, fx), take(y, ny, ny2, fy)

def show_image(z, x=None, y=None, *, title="", xlabel="", ylabel="", invert=True):
    state["called"] = True
    z = np.real(np.asarray(z))
    if z.ndim != 2:
        raise ValueError("show_image expects a 2-D array.")
    if isinstance(x, tuple) and len(x) == 2 and isinstance(x[1], str):
        if not xlabel:
            xlabel = x[1]
        x = x[0]
    if isinstance(y, tuple) and len(y) == 2 and isinstance(y[1], str):
        if not ylabel:
            ylabel = y[1]
        y = y[0]
    x_arr = None if x is None else np.asarray(x, dtype=float).ravel()
    y_arr = None if y is None else np.asarray(y, dtype=float).ravel()
    z, x_arr, y_arr = downsample(z, x_arr, y_arr)
    ny, nx = z.shape
    if x_arr is None:
        x_arr = np.arange(nx, dtype=float)
    if y_arr is None:
        y_arr = np.arange(ny, dtype=float)
    beginImage(str(title), str(xlabel), str(ylabel), bool(invert), f64(x_arr[:nx]), f64(y_arr[:ny]), int(ny), int(nx))
    setImageZ(f64(np.ascontiguousarray(z)))

def _orientation_hook(kind):
    last = [0.0]
    def report(done, total):
        now = time.monotonic()
        if done < total and done > 1 and now - last[0] < 0.3:
            return
        last[0] = now
        reportProgress(int(RUN_ID), kind, int(done), int(total))
    return report

def _with_progress(fn, kind):
    def wrapped(spin_system, pulse_sequence, parameters, assumptions):
        if not callable(parameters.get("progress")):
            parameters = dict(parameters)
            parameters["progress"] = _orientation_hook(kind)
        return fn(spin_system, pulse_sequence, parameters, assumptions)
    return wrapped

if not getattr(spynach.powder, "_counts_orientations", False):
    spynach.powder = _with_progress(spynach.powder, "powder")
    spynach.singlerot = _with_progress(spynach.singlerot, "MAS")
    spynach.powder._counts_orientations = True
    spynach.singlerot._counts_orientations = True

try:
    await prefetch(USER_CODE)
    ns = {
        "__name__": "__main__",
        "show_spectrum": show_spectrum,
        "show_fid": show_fid,
        "show_image": show_image,
    }
    exec(compile(USER_CODE, "<editor>", "exec"), ns, ns)
    if not state["called"]:
        if "spectrum" in ns:
            show_spectrum(
                ns["spectrum"],
                ns.get("axis"),
                title=str(ns.get("title") or ""),
                xlabel=str(ns.get("xlabel") or ""),
                invert=bool(ns.get("invert", True)),
            )
        if "fid" in ns and "sweep" in ns and np.size(ns["sweep"]) == 1 and np.asarray(ns["fid"]).ndim == 1:
            show_fid(ns["fid"], ns["sweep"])
except Exception:
    RUN_ERROR = traceback.format_exc()
`;

function post(message, transfer) {
  self.postMessage(message, transfer || []);
}

let pyodide = null;
let booted = false;
let bucket = [];
const waiting = [];

function rawBytes(view) {
  const bytes = new Uint8Array(view.buffer, view.byteOffset, view.byteLength);
  return new Uint8Array(bytes);
}

function asF64(value) {
  if (value == null) return null;
  try {
    if (value instanceof Float64Array) return new Float64Array(value);
    if (value instanceof Float32Array) return Float64Array.from(value);
    if (typeof value.getBuffer === "function") {
      const buf = value.getBuffer();
      try {
        const copy = rawBytes(buf.data);
        return new Float64Array(copy.buffer, copy.byteOffset, copy.byteLength / 8);
      } finally {
        buf.release();
        if (typeof value.destroy === "function") value.destroy();
      }
    }
    if (typeof value.toJs === "function") {
      const js = value.toJs();
      if (typeof value.destroy === "function") value.destroy();
      if (js instanceof Float64Array) return new Float64Array(js);
      if (ArrayBuffer.isView(js)) return Float64Array.from(js);
      if (Array.isArray(js)) return Float64Array.from(js);
    }
  } catch (err) {
    post({ type: "stderr", text: "Could not read an array for plotting: " + err + "\n" });
  }
  return null;
}

self.reportProgress = (id, kind, done, total) => {
  post({ type: "progress", id: Number(id), kind: String(kind), done: Number(done), total: Number(total) });
};

self.beginLine = (title, xlabel, ylabel, invert, x) => {
  bucket.push({
    kind: "line",
    title: String(title || ""),
    xlabel: String(xlabel || ""),
    ylabel: String(ylabel || ""),
    invert: Boolean(invert),
    x: asF64(x),
    series: [],
  });
};

self.addSeries = (name, y) => {
  const plot = bucket[bucket.length - 1];
  if (!plot || plot.kind !== "line") return;
  plot.series.push({ name: String(name || ""), y: asF64(y) });
};

self.beginImage = (title, xlabel, ylabel, invert, x, y, rows, cols) => {
  bucket.push({
    kind: "image",
    title: String(title || ""),
    xlabel: String(xlabel || ""),
    ylabel: String(ylabel || ""),
    invert: Boolean(invert),
    x: asF64(x),
    y: asF64(y),
    z: null,
    rows: Number(rows),
    cols: Number(cols),
  });
};

self.setImageZ = (z) => {
  const plot = bucket[bucket.length - 1];
  if (plot && plot.kind === "image") plot.z = asF64(z);
};

function mkdirp(dir) {
  const parts = dir.split("/").filter(Boolean);
  let cur = "";
  for (const part of parts) {
    cur += "/" + part;
    try {
      pyodide.FS.mkdir(cur);
    } catch (err) {
      /* The directory already exists, or another fetch created it just now. */
    }
  }
}

async function mountSources() {
  let cursor = 0;
  let done = 0;
  async function next() {
    while (cursor < SOURCE.length) {
      const rel = SOURCE[cursor++];
      const response = await fetch(new URL(rel, self.location.href));
      if (!response.ok) throw new Error("Could not fetch " + rel + " (" + response.status + "). Serve the repository root over HTTP.");
      const text = await response.text();
      const dest = "/repo/" + rel;
      mkdirp(dest.slice(0, dest.lastIndexOf("/")));
      pyodide.FS.writeFile(dest, text);
      done += 1;
      if (done === SOURCE.length || done % 10 === 0) {
        post({ type: "status", text: "Loading Spynach source (" + done + "/" + SOURCE.length + ")" });
      }
    }
  }
  await Promise.all(Array.from({ length: 6 }, next));
}

function line(text) {
  if (!text) return;
  return text.endsWith("\n") ? text : text + "\n";
}

async function boot() {
  post({ type: "status", text: "Loading Pyodide…" });
  pyodide = await loadPyodide({
    indexURL: INDEX_URL,
    stdout: (text) => post({ type: "stdout", text: line(String(text)) }),
    stderr: (text) => post({ type: "stderr", text: line(String(text)) }),
  });
  post({ type: "status", text: "Loading NumPy, SciPy, and h5py…" });
  await pyodide.loadPackage(["numpy", "scipy", "h5py"]);
  post({ type: "status", text: "Loading Spynach source…" });
  await mountSources();
  const raw = await pyodide.runPythonAsync(
    'import sys\nsys.path.insert(0, "/repo/src")\nimport spynach, ctps\nspynach.__version__ + "|" + ctps.__version__ + "|" + sys.version.split()[0]\n',
    { dedent: false },
  );
  const text = String(raw);
  if (raw && typeof raw.destroy === "function") raw.destroy();
  const parts = text.split("|");
  booted = true;
  post({ type: "ready", spynach: parts[0], ctps: parts[1], python: parts[2] });
  for (const message of waiting.splice(0)) handle(message);
}

async function handle(message) {
  if (!message || message.type !== "run") return;
  const id = message.id;
  bucket = [];
  try {
    pyodide.globals.set("USER_CODE", message.code);
    pyodide.globals.set("RUN_ID", id);
    pyodide.globals.set("RUN_ERROR", "");
    await pyodide.runPythonAsync(RUNNER, { dedent: false });
    const errProxy = pyodide.globals.get("RUN_ERROR");
    const error = errProxy ? String(errProxy) : "";
    if (errProxy && typeof errProxy.destroy === "function") errProxy.destroy();
    const transfer = [];
    for (const plot of bucket) {
      for (const key of ["x", "y", "z"]) if (plot[key]) transfer.push(plot[key].buffer);
      if (plot.series) for (const series of plot.series) if (series.y) transfer.push(series.y.buffer);
    }
    post({ type: "result", id, plots: bucket, error }, transfer);
  } catch (err) {
    post({ type: "error", id, message: String((err && err.message) || err) });
  }
}

self.onmessage = (event) => {
  if (!booted) waiting.push(event.data);
  else handle(event.data);
};

boot().catch((err) => post({ type: "fatal", message: String((err && err.message) || err) }));
