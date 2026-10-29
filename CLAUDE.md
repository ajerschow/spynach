# Spynach: notes for agents

Read `HANDOFF.md` for goals and design decisions, `README.md` for status,
`porting/ROADMAP.md` for the current step, and `porting/design-notes.md`
for the Liouville-space kernel design (literature and Spinach source review).
GPU acceleration is deferred: `porting/gpu-notes.md` lists the entry points;
keep the hot loops going through `spynach.backend` and mark new device-relevant
lines with `# GPU:`. `comparison/README.md` is the protocol for timed and
profiled Spinach-vs-spynach runs (both sides timed on the host, Spynach also
runnable here for checks, HTML report).
`LICENSE` is MIT and carries the Spinach notice: material translated from
Spinach keeps that attribution, and contributed scripts in `user_ml/` keep
their author's name.

## Contributions

When applying or reviewing a contribution, follow `CONTRIBUTING.md`: examples that run on the existing kernel
come first (parity with Spinach or EasySpin, then browser examples with a notebook), Spinach and EasySpin names
are kept, and a kernel change is its own pull request with a regression test.

## User documentation is part of every change

- Every user-facing change (new port, option, behaviour change) includes
  user documentation in the same commit. Follow the conventions in
  `docs/README.md`.
- Guides are `docs/*.md`; their `python` blocks are executed by
  `tests/docs/`. Run `python docs/build_notebooks.py` after editing a guide:
  notebooks in `docs/notebooks/` are generated and saved with their outputs
  (plots included), never hand-edited. Plot with `spynach.plotting`.
- For each ported feature, update `docs/coming-from-spinach.md` with what is
  the same as in Spinach, what differs, and what is still not ported.
- Split a guide by theme when it passes ~500 lines or a new formalism,
  context, or experiment family arrives; add it to the table in
  `docs/README.md`.

## Testing

```
OPENBLAS_NUM_THREADS=4 PYTHONPATH=src python -m pytest -q
```

The `OPENBLAS_NUM_THREADS` setting matters in this container (4-CPU quota,
12 visible cores).

## Code style

Run `ruff format && ruff check` (configured in `pyproject.toml`, `dev`
extra) before every commit. Prefer Python idioms (keyword arguments,
small functions) over one-to-one Spinach call signatures, and record each
such deviation in `docs/coming-from-spinach.md`. Spinach's own short
matrix names may stay in code that mirrors a Spinach function.
