// Parameters of a playground example, read from and written back to its Python source.
//
// A parameter is an ordinary assignment with a trailing "GUI:" comment that carries what the form needs:
//
//     mw_freq = 9.5   # GUI: Microwave frequency | GHz | The spectrometer frequency; X band is about 9.5 | 1..100
//     g = [2.0088, 2.0061, 2.0027]   # GUI: g values | - | Principal values of the g tensor | 1.9..2.2
//     method = "matrix"   # GUI: Method | - | How the resonance fields are found | matrix, perturb2
//
// After "GUI:" come label | unit | description | range or choices, separated by "|"; the last field is optional.
// A dimensionless parameter has "-" as its unit (the form shows no unit for it).
// A range is min..max or min..max..step, choices are a comma-separated list. A line "# --- Sample ---" starts a
// section of the form. The code stays the only state: the form is parsed from it and an edit rewrites just the
// literal of one line, so the comment, the spacing and every other line are untouched.
// No dependencies, so the same file runs in the page and under Node (tests/unit/test_playground_params.py).

const LINE = /^(\s*[A-Za-z_]\w*\s*=\s*)(.*?)(\s*#\s*GUI:\s*)(.*)$/;
const SECTION = /^\s*#\s*---\s*(.*?)\s*-*\s*$/;
const NUMBER = /^[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?$/;

/** All parameters of a source, in order: {name, line, kind, value, label, unit, description, min, max, step,
 *  choices, section, editable}. kind is "number", "list", "choice" or "bool"; editable is false when the literal on
 *  the line is not one of those (the form then shows it greyed out, "edited in code"). */
export function parseParams(source) {
  const params = [];
  let section = "";
  source.split("\n").forEach((text, line) => {
    const heading = SECTION.exec(text);
    if (heading && !LINE.test(text)) {
      section = heading[1];
      return;
    }
    const m = LINE.exec(text);
    if (!m) return;
    const name = /([A-Za-z_]\w*)\s*=\s*$/.exec(m[1])[1];
    const [label = name, unit = "", description = "", spec = ""] = m[4].split("|").map((s) => s.trim());
    const param = { name, line, label: label || name, unit, description, section, min: null, max: null, step: null, choices: null };
    Object.assign(param, readLiteral(m[2].trim()), readSpec(spec));
    if (param.kind === "number" || param.kind === "list") param.step ??= defaultStep(param);
    if (param.kind === "string") {
      param.kind = param.choices ? "choice" : "string";
      param.editable = Boolean(param.choices);
    }
    params.push(param);
  });
  return params;
}

/** The source with parameter `name` set to `value` (a number, an array of numbers, a string or a boolean).
 *  Throws RangeError for a value the parameter cannot take; the source is never half-written. `original` (the
 *  parameter as first parsed) lets a value equal to the original come back with its original text: 1.10, not 1.1. */
export function setParam(source, name, value, original = null) {
  const lines = source.split("\n");
  const param = parseParams(source).find((p) => p.name === name);
  if (!param) throw new RangeError(`no parameter ${name}`);
  if (!param.editable) throw new RangeError(`${name} is edited in the code`);
  const m = LINE.exec(lines[param.line]);
  const like = original && original.kind === param.kind ? { ...param, value: original.value, texts: original.texts } : param;
  lines[param.line] = m[1] + writeLiteral(like, value) + m[3] + m[4];
  return lines.join("\n");
}

/** Check a value against a parameter; returns "" when it is acceptable, otherwise a short message. */
export function validate(param, value) {
  const numbers = param.kind === "list" ? value : [value];
  if (param.kind === "list" && (!Array.isArray(value) || value.length !== param.value.length)) {
    return `${param.value.length} values needed`;
  }
  if (param.kind === "number" || param.kind === "list") {
    for (const v of numbers) {
      if (typeof v !== "number" || !Number.isFinite(v)) return "not a number";
      if (param.min !== null && v < param.min) return `at least ${param.min}`;
      if (param.max !== null && v > param.max) return `at most ${param.max}`;
      if (param.integer && !Number.isInteger(v)) return "a whole number";
    }
  }
  if (param.kind === "choice" && !param.choices.includes(value)) return `one of ${param.choices.join(", ")}`;
  if (param.kind === "bool" && typeof value !== "boolean") return "true or false";
  return "";
}

function readLiteral(text) {
  if (NUMBER.test(text)) return { kind: "number", value: Number(text), texts: [text], integer: /^[-+]?\d+$/.test(text), editable: true };
  if (text === "True" || text === "False") return { kind: "bool", value: text === "True", editable: true };
  const str = /^(["'])([^"'\\]*)\1$/.exec(text);
  if (str) return { kind: "string", value: str[2], editable: false };
  const list = /^\[(.*)\]$/.exec(text);
  if (list) {
    const items = list[1].split(",").map((s) => s.trim());
    if (items.length && items.every((s) => NUMBER.test(s))) {
      return { kind: "list", value: items.map(Number), texts: items, integer: items.every((s) => /^[-+]?\d+$/.test(s)), editable: true };
    }
  }
  return { kind: "number", value: null, editable: false, raw: text };
}

function readSpec(spec) {
  if (!spec) return {};
  const range = spec.split("..").map((s) => s.trim());
  if (range.length >= 2 && range.length <= 3 && range.every((s) => NUMBER.test(s))) {
    const [min, max, step] = range.map(Number);
    return { min, max, ...(step === undefined ? {} : { step }) };
  }
  return { choices: spec.split(",").map((s) => s.trim()).filter(Boolean) };
}

function defaultStep(param) {
  if (param.integer) return 1;
  const reference = param.kind === "list" ? Math.max(...param.value.map(Math.abs)) : Math.abs(param.value);
  if (param.min !== null && param.max !== null) return niceStep((param.max - param.min) / 200);
  return reference ? niceStep(reference / 100) : 0.01;
}

// 1, 2 or 5 times a power of ten, the largest not above x: slider and arrow steps that read well
function niceStep(x) {
  const magnitude = 10 ** Math.floor(Math.log10(x));
  const m = x / magnitude;
  return (m >= 5 ? 5 : m >= 2 ? 2 : 1) * magnitude;
}

function writeLiteral(param, value) {
  const message = validate(param, value);
  if (message) throw new RangeError(`${param.name}: ${message}`);
  // A value that did not change keeps the text it had (1.00 stays 1.00)
  const number = (v, i) => (v === (param.kind === "list" ? param.value[i] : param.value) ? param.texts[i] : param.integer ? String(Math.round(v)) : formatFloat(v));
  if (param.kind === "number") return number(value, 0);
  if (param.kind === "list") return "[" + value.map(number).join(", ") + "]";
  if (param.kind === "bool") return value ? "True" : "False";
  return JSON.stringify(value);
}

// The shortest text that reads back as the same double and is valid Python: 5 -> "5.0", 1e-7 stays "1e-07"
// in JS as "1e-7", which Python also accepts.
function formatFloat(v) {
  const text = String(v);
  return /^[-+]?\d+$/.test(text) ? text + ".0" : text;
}
