# spyndynamica

A Python port in progress of [SpinDynamica](https://www.spindynamica.soton.ac.uk) 3.10.2, Malcolm Levitt's
Mathematica package for spin dynamics and NMR. It is a separate import inside the Spynach repository: it uses
NumPy and SymPy only, imports nothing of `spynach`, `easyspyn` or `ctps`, and nothing else imports it
(`tests/spyndynamica/test_dependency_direction.py`). It is not part of the `spynach` distribution.

**Status:** the symbolic core (spin systems, operators, rotations, bases, kets, exact matrices). The guide is
`docs/spyndynamica.md`; the phases and the conventions are in `porting/spyndynamica/PLAN.md`.

**Licence:** not settled. The SpinDynamica archive carries no licence file, so the port is not released as a
package until its author has agreed. No SpinDynamica source is in this repository.

| Path | Content |
|---|---|
| `src/spyndynamica/_local.py` | the operator basis of one spin: labels, exact matrices, products, rotations |
| `src/spyndynamica/operators.py` | `Operator`: sums of product and tensor operators with symbolic coefficients |
| `src/spyndynamica/system.py` | `SpinSystem`, `Basis`, `Ket`, `Bra` |
| `src/spyndynamica/rotations.py` | Wigner matrices, Clebsch-Gordan coefficients, Euler matrices |
| `tests/spyndynamica/fixtures/` | the generator, the case files and the reference data from SpinDynamica |
| `tests/spyndynamica/test_reference_invariants.py` | checks of the reference that do not depend on SpinDynamica |
| `tests/spyndynamica/test_core.py` | the port against the reference |
| `porting/spyndynamica/DIVERGENCES.md` | where SpinDynamica differs from its documentation |

To remove the package, delete `src/spyndynamica`, `tests/spyndynamica`, `porting/spyndynamica`,
`docs/spyndynamica.md` with its notebook, and the rows that name it in `docs/README.md` and
`scripts/build_site.py`.
