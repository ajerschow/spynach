# Contributing to Spynach

The most useful contributions are **examples that run on the existing kernel**. They extend what people can do
without a change to the code that every result depends on. In order of priority:

1. **Parity with Spinach or EasySpin.** A script that reproduces a Spinach or EasySpin example, or a
   published result, with the same inputs and numbers.
2. **A browser example plus a notebook.** Add the example to the playground (`try.html`, run by `worker.js`
   through Pyodide). Every example has a matching guide in `docs/<name>.md`, and
   `python docs/build_notebooks.py` generates the notebook from it. A browser example without a notebook, or the
   reverse, is incomplete. See "Browser examples" below.
3. **Other new examples that use only existing functions.**
4. **Changes to the kernel** (`src/spynach/` and `src/easyspyn/`, for example `relaxation.py`, `basis.py`,
   `propagation.py`): please open an issue first. Send a kernel change as its own pull request, with a
   regression test that fails on the old code. Do not bundle it with examples, so that it can be reviewed
   and reverted on its own.

## Names follow the originals

Use Spinach's and EasySpin's names wherever possible: `spin_system`, `inter`, `bas`, `sys`, `Exp`, `Opt`,
`parameters`, `pepper`, `garlic`, and so on, and the original option and field names. Where Python
needs a different spelling (keyword arguments, return types), record it in
`docs/coming-from-spinach.md` or `docs/coming-from-easyspin.md`.

## Browser examples (optional, highly desired)

A browser example is not required for a contribution, but it is highly desired wherever the cost allows. It lets
people try your work with nothing installed. Add it to `try.html` (run by `worker.js` in Pyodide) and write the
guide in `docs/<name>.md`, from which `python docs/build_notebooks.py` generates the matching notebook. An
example that goes in the playground always has the notebook as well.

Check that it fits first:
- The browser runs one worker, **without Numba** (NumPy and SciPy only), so expect it to be several times slower
  than the same code on your machine. Aim for a few seconds per run.
- Every example not marked `data-slow` is run by `tests/unit/test_playground_examples.py`, so keep it quick.
- An example that takes about a minute can go in with `data-slow` (it turns off live updating and is skipped in the
  tests). If it needs more than that, or a lot of memory, keep it in the notebook only and say so.
- Reduce the size (spins, orientations, points) for the browser and say how to restore the full size, as in the
  other guides.

## Check parity before you open the pull request

Start from whichever side you have:

- **From MATLAB:** put your Spinach script in `user_ml/`, translate it line by line into
  `examples/user_ml/<name>.py`, and add a case in `comparison/cases/`.
- **From Python:** write the example in Python, then write the MATLAB equivalent (Spinach or EasySpin) next to it.
- Run both with the protocol in `comparison/README.md` (`comparison/run_spinach.sh` and
  `comparison/run_spynach.py`) and compare the outputs. State in the pull request what you compared and the
  largest difference. If you have no MATLAB licence, say so, and the maintainers will run the MATLAB side.

## Checklist

- [ ] No file in `src/` changed (or the kernel change is its own pull request with a regression test)
- [ ] Browser example in `try.html` and guide + generated notebook (`python docs/build_notebooks.py`), where the
      cost allows
- [ ] Spinach/EasySpin names
- [ ] Parity check reported (case in `comparison/cases/`, numbers in the description)
- [ ] `docs/coming-from-*.md` updated; `ruff format && ruff check`; tests pass

The other conventions (user documentation with every change, notebooks and figures generated, tests and code
style) are in [`docs/README.md`](docs/README.md#conventions-for-contributors) and [`CLAUDE.md`](CLAUDE.md).
