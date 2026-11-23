# Reference fixtures

`tests/spyndynamica/fixtures/generate.wls` loads SpinDynamica, runs every case file in `cases/` and writes one JSON file per case file to
`data/`, with `manifest.json` (versions and the SHA-256 of every case file) and `timings.json` (seconds per
case, a baseline for the port to beat). The Python tests read `data/` and need no Mathematica.

## Running it

```
SPINDYNAMICA_HOME=<folder that holds SpinDynamica.m> wolframscript -file tests/spyndynamica/fixtures/generate.wls
```

It runs without the notebook front end (checked with Mathematica 14.0 and SpinDynamica 3.10.2; about a minute and a half).
From a notebook: `SetEnvironment["SPINDYNAMICA_HOME" -> "..."]; Get["<repo>/tests/spyndynamica/fixtures/generate.wls"]`.
The SpinDynamica folder is only read. Always regenerate everything in one pass, so that all data comes from
one SpinDynamica and one Mathematica version, and commit `data/` together with the case files.
`tests/spyndynamica/test_fixture_manifest.py` fails when a case file changed after its data was generated.

Nothing is written when the canary (a few results known by hand) fails, when a case emits a message that is
not listed as informational, when a result contains `$Failed`, or when two cases share a name.

## Writing a case file

A case file evaluates to a list of groups:

```mathematica
<|"group" -> "two_half", "setup" :> SetSpinSystem[2], "allow" -> {"Sym::tag"},
  "cases" -> {"I1x" :> MatrixRepresentation[opI[1, "x"]], ...}|>
```

- Every group starts from a clean state: `$Chronology = "LeftToRight"`, user level 2, the Zeeman basis and the
  default operator basis of its spin system. The setup may then change them.
- Build operators inside the case (`:>`), never in the arguments of a helper that evaluates them: the spin
  system of the group is not set while the file is read. Helpers that take an operator are `HoldAll`.
- Pin values through `MatrixRepresentation`, `OperatorAmplitude` and the like. SpinDynamica often returns
  unevaluated `OperatorProduct` forms, and printed forms are never compared.
- A result that depends on symbols is wrapped as `Symbolic[expr, {a3, b3}]`: its text and its values at three
  fixed parameter points are stored (the exact rational points are in the file).
- `"allow"` lists messages that are informational for the group. Any other message stops the generation.
- A case that records a behaviour which differs from the documentation is named `..._is_ignored` or similar
  and has an entry in `porting/spyndynamica/DIVERGENCES.md`.
- A result of `$Failed` cannot be stored. A call that SpinDynamica refuses is recorded through a helper of the
  case file, `rejected[call]`, which stores `True`; the messages of the call go into the group's `"allow"`.

## Data format

Each case has `input` (the expression as text), `messages` (the tags it emitted) and `value`:

| `type` | Content |
|---|---|
| `array` | `shape`, `re`, `im` (left out when zero), flattened in row-major order; `exact` holds the Mathematica text when the result was exact |
| `symbolic` | `variables`, `text`, and three `points` with `parameters`, `parameters_exact` and the `value` there |
| `string`, `strings` | text |
| `list` | `items`, each encoded in turn |
| `expr` | the Mathematica text of anything else |
