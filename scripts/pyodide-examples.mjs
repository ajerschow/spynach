// Runs the playground examples of try.html in Pyodide under Node, the same 32-bit WebAssembly build the browser
// uses: NumPy's index type is int32 there, so an int64 array passed as repeat counts or bincount indices fails
// in the browser only (use np.intp). Mounts the repository's src/ into the Pyodide file system, loads NumPy and
// SciPy from the Pyodide CDN (cached in node_modules), and executes each example with stub show_* helpers.
//
//     cd /tmp/pyodide-check && npm init -y && npm install pyodide        # once, outside the repository
//     node /path/to/spynach/scripts/pyodide-examples.mjs /path/to/spynach [example id ...]
//
// Without ids every example runs except those marked data-slow. Not part of the test suite (needs npm and the
// network); tests/unit/test_playground_examples.py runs the examples natively.
import { loadPyodide } from "pyodide";
import { readFileSync } from "node:fs";

const root = process.argv[2];
const only = process.argv.slice(3);
const page = readFileSync(root + "/try.html", "utf8");
const blocks = [...page.matchAll(/<script type="text\/plain" class="example" id="([\w-]+)"([^>]*)>\n([\s\S]*?)<\/script>/g)];

const pyodide = await loadPyodide();
await pyodide.loadPackage(["numpy", "scipy"]);
pyodide.FS.mkdir("/repo");
pyodide.FS.mount(pyodide.FS.filesystems.NODEFS, { root: root }, "/repo");
await pyodide.runPythonAsync(`
import sys
sys.path.insert(0, "/repo/src")
import numpy as np, spynach, ctps, epreasy
print("python", sys.version.split()[0], "numpy", np.__version__, "intp", np.dtype(np.intp))
calls = []
def show_spectrum(y, x=None, **kw): calls.append(("line", np.asarray(getattr(y, "spec", y)).shape))
def show_fid(fid, sweep, **kw): calls.append(("fid", np.asarray(fid).shape))
def show_image(z, x=None, y=None, **kw): calls.append(("image", np.asarray(z).shape))
`);

let failed = 0;
for (const [, id, attrs, code] of blocks) {
  if (only.length && !only.includes(id)) continue;
  if (/data-slow=/.test(attrs) && !only.includes(id)) continue;
  const t0 = performance.now();
  pyodide.globals.set("USER_CODE", code);
  try {
    await pyodide.runPythonAsync(`
calls.clear()
ns = {"__name__": "__main__", "show_spectrum": show_spectrum, "show_fid": show_fid, "show_image": show_image}
import io, contextlib
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    exec(compile(USER_CODE, "<editor>", "exec"), ns, ns)
summary = f"{len(calls)} plots {[c[0] for c in calls]}; printed {len(buf.getvalue().splitlines())} lines"
`);
    const summary = pyodide.globals.get("summary");
    console.log(`ok   ${id.padEnd(18)} ${((performance.now() - t0) / 1000).toFixed(1)} s  ${summary}`);
  } catch (err) {
    failed += 1;
    const lines = String(err.message || err).trim().split("\n");
    console.log(`FAIL ${id.padEnd(18)} ${lines.slice(-3).join(" | ")}`);
  }
}
console.log(failed ? `${failed} example(s) failed` : "all examples ran");
process.exit(failed ? 1 : 0);
