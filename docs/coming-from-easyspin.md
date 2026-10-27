# Coming from EasySpin

What carries over from [EasySpin](https://easyspin.org) 6.0.15 to `easyspyn`,
what differs, and what is not ported. The guides are
[EPR spin systems with EasySpyn](easyspyn.md) and
[EPR simulations with EasySpyn](easyspyn-simulators.md); this page is the reference,
updated with every ported function. The companion notebook is
[`notebooks/coming-from-easyspin.ipynb`](notebooks/coming-from-easyspin.ipynb).

## Contents

1. [The rules](#the-rules)
2. [Spin systems](#spin-systems)
3. [Spin operators and rotations](#spin-operators-and-rotations)
4. [Hamiltonians and energy levels](#hamiltonians-and-energy-levels)
5. [Units, constants, nuclear data](#units-constants-nuclear-data)
6. [Resonance fields](#resonance-fields)
7. [Line shapes and grids](#line-shapes-and-grids)
8. [Powder and crystal spectra: pepper](#powder-and-crystal-spectra-pepper)
9. [Solution spectra: garlic](#solution-spectra-garlic)
10. [Fitting: esfit](#fitting-esfit)
11. [ENDOR: endorfrq, salt, orisel](#endor-endorfrq-salt-orisel)
12. [Magnetometry: curry](#magnetometry-curry)
13. [Slow motion: chili](#slow-motion-chili)
14. [Pulse EPR: saffron](#pulse-epr-saffron)
15. [Not ported](#not-ported)
16. [Verification](#verification)

## The rules

- **EasySpin's names, fields and units.** Functions keep their names,
  spin systems are dicts with EasySpin's field names, fields are in mT,
  couplings and energies in MHz, magnetic moments in MHz/mT, angles in
  radians. Spin and level numbers are 1-based, as in EasySpin (this differs
  from `spynach`, which uses Spinach's 0-based indices).
- **MATLAB shapes.** A scalar or a flat list is one row (`g=[gx, gy, gz]` for
  one electron, `A=[16, 86]` for one axial tensor), a list of lists is one
  row per spin, and `3n x 3` arrays are full tensors. This keeps EasySpin's
  size rules unchanged.
- **Python idioms where MATLAB's calling conventions do not carry over.**
  Options are keyword arguments (`sparse=True` instead of a trailing
  `'sparse'`), functions return tuples instead of depending on `nargout`,
  and constants are values, not functions. Each such deviation is listed
  below.
- **Errors are EasySpin's messages**, raised as `ValueError`; features that
  are not ported raise `NotImplementedError`.
- **Authors are credited by name, not in function names.** EasySpin's headers name no authors, so each EasySpyn
  function's docstring (`help(easyspyn.pepper)`) and the page [EasySpin function authors](easyspin-credits.md) name
  the people who authored commits to the original file up to release 6.0.15, and any author its source names;
  function names stay EasySpin's.

```python
import numpy as np

import easyspyn as es

spin_system = {"S": 1 / 2, "g": [2.0088, 2.0061, 2.0027], "Nucs": "14N", "A": [16, 16, 86]}
H0, mux, muy, muz = es.ham(spin_system)  # EasySpin: [H0,mux,muy,muz] = ham(Sys)
H = es.ham(spin_system, [0, 0, 350])  # EasySpin: H = ham(Sys,[0 0 350])
print(H.shape, np.allclose(H, H0 - 350 * muz))
```

## Spin systems

| EasySpin | EasySpyn |
|---|---|
| `Sys` structure with fields `S`, `g`, `gFrame`, `Nucs`, `A`, `AFrame`, `D`, `DFrame`, `Q`, `QFrame`, `ee`/`J`/`dip`/`dvec`, `eeFrame`, `ee2`, `nn`, `nnFrame`, `B1`..`B12`, `sigma`, `gnscale`, `n`, `lw`, `lwpp`, `lwEndor`, `gStrain`, `DStrain`, `HStrain`, `AStrain` | a dict with the same keys and the same size rules and shorthands (`g_`, `D_`, `A_` spherical forms, `D=[D, E]`, `Q=[eeqQ, eta]`) |
| `validatespinsys(Sys)` (private) returns `[FullSys, err]` | `validatespinsys(sys)` is public and returns the processed dict; errors raise |
| `spinvec(Sys)`, `hsdim(Sys)` | the same |
| spin-zero nuclei are removed | the same; `ham_hf` then works (in EasySpin 6.0.15 `ham_hf` errors for a system listing spin-zero nuclei with one `A` row per nucleus, since it tests the unprocessed `A` for the full-matrix form) |
| `Sys.initState`, `Sys.tdm`, diffusion fields | kept on the dict for the simulators; `tdm` is normalised as in EasySpin; `initState` is stored as given |

## Spin operators and rotations

| EasySpin | EasySpyn |
|---|---|
| `sop(Spins,'xye')`, `sop(Spins,'x1,y2')`, `sop(Spins,'x(1|3)')`, `sop(Spins,[1 1; 2 3])` | the same strings and the same `(N, 2)` numeric array; `[Sx,Sy,Sz] = sop(S,'x','y','z')` is `Sx, Sy, Sz = sop(S, "x", "y", "z")` |
| `sop(...,'sparse')` | `sop(..., sparse=True)` returns `scipy.sparse` CSR |
| `stev(S,[k,q])`, `stev(Spins,[k,q,iSpin])` | `stev(S, k, q)`, `stev(spins, k, q, ispin=i)` |
| `erot(Angles)`, `erot(a,b,c)`, `[x,y,z] = erot(...,'rows'/'cols')` | `erot(angles)`, `erot(a, b, c)`; rows and columns are `R[k]` and `R[:, k]` |
| `eulang(R)` | the same, with the same degenerate-case handling |
| `ang2vec(phi,theta)` returns `[x;y;z]` | the same `(3, n)` array |
| `vec2ang(v)` returns `[phi;theta]` or two outputs | always the tuple `(phi, theta)` |
| `letter2vec` (private) | public |

## Hamiltonians and energy levels

| EasySpin | EasySpyn |
|---|---|
| `[H0,mux,muy,muz] = ham(Sys)`, `H = ham(Sys,B0)`, `[H0,muzL] = ham(Sys,B0)` | `ham(sys)` returns the 4-tuple; `ham(sys, B0)` returns `H`; `ham(sys, B0, split=True)` returns `(H0, muzL)` |
| `ham(...,'sparse')` and the same option on every component | `sparse=True` keyword |
| `ham_ez(Sys,B)`, `[mux,muy,muz] = ham_ez(Sys)`, `ham_ez(Sys,B,eSpins)` | `ham_ez(sys, B)`, `ham_ez(sys)`, `ham_ez(sys, B, espins)`; the same for `ham_nz` with `nspins` |
| `ham_hf(Sys,eSpins,nSpins)`, `ham_zf(Sys,eSpins)`, `ham_nq(Sys,nSpins)`, `ham_ee(Sys,eSpins)`, `ham_nn(Sys,nSpins)` | the same, spin numbers 1-based |
| `ham_so`, `ham_cf`, `ham_oz`, `ham_ezho` | not ported (orbital angular momentum, higher-order Zeeman) |
| `levels(Sys,Ori,B)`, `levels(Sys,phi,theta,B)`, `[E,V] = levels(...)` | `levels(sys, ori, B)`, `levels(sys, phi=..., theta=..., B=...)`, `levels(..., vectors=True)`; singleton axes are squeezed as in EasySpin; levels are computed for all orientations and fields in one batched `eigh` |

## Units, constants, nuclear data

| EasySpin | EasySpyn |
|---|---|
| `bmagn`, `nmagn`, `planck`, `hbar`, `gfree`, `clight`, `mu0`, `boltzm`, `evolt`, `echarge`, `amu`, `avogadro` as functions | module attributes with the same values (`es.bmagn`) |
| `unitconvert(value,'cm^-1->MHz')`, `unitconvert(value,'mT->MHz',g)`, `mt2mhz`, `mhz2mt` | the same |
| `nucdata('14N,1H')` returns up to five outputs; `nucspin`, `nucgval`, `nucqmom`, `nucabund` | `nucdata` returns a named tuple `(spin, gn, qm, abund, nucs)`; the wrappers are the same |
| the isotope table `private/isotopedata.txt` | shipped verbatim as `easyspyn/data/isotopedata.txt`; `isotopes()` returns it. It differs from `spynach`'s table (from Spinach) in some values, so each package keeps its own |

## Resonance fields

| EasySpin | EasySpyn |
|---|---|
| `[Pos,Int,Wid,Trans] = resfields(Sys,Exp,Opt)` with outputs chosen by `nargout` | `resfields(sys, exp, opt)` returns `Resonances(pos, int, wid, trans)`; `wid` is `None` without strains. The gradient output (`nargout>4`) is not returned (EasySpin leaves it unsorted and `pepper` does not use it) |
| `Exp.mwFreq`, `Range`, `CenterSweep`, `Temperature`, `mwMode` (`'perpendicular'`, `'parallel'`, `{k alpha}`, `{k 'unpolarized'}`, `{k 'circular+'}`), `SampleFrame`, `MolFrame`, `lightBeam`, `lightScatter` | the same keys; `mwMode` cells become lists `[k, alpha]` |
| `Exp.CrystalSymmetry`, `Opt.Sites`, `Exp.SampleRotation` | only symmetry P1 (`1`, `'P1'`); others raise `NotImplementedError` |
| `Opt.Transitions`, `Threshold`, `Method`/`Hybrid`, `HybridCoreNuclei`, `FuzzLevel`, `Freq2Field`, `maxSegments`, `ModellingAccuracy`, `RediagLimit`, `Intensity` | the same keys; `Opt.Sparse` raises `NotImplementedError` (dense only); `Opt.Verbosity` is accepted and ignored |
| `Opt.FuzzLevel = 1e-10` adds unseeded random noise to break degeneracies | the same, with NumPy's generator; use `FuzzLevel = 0` to reproduce numbers |
| `Sys.initState` as a density matrix or `{populations, basis}` | a matrix, or a tuple `(state, basis)` with basis `'uncoupled'`, `'eigen'` or `'zerofield'`; `'coupled'`, `'xyz'` and the string shorthands are not ported |
| `resfields_eig(Sys,Exp,Opt)`, `[B,Int] = resfields_eig(...)` | `resfields_eig(sys, exp, opt)` returns the fields; `opt["Intensities"] = True` returns `(fields, intensities)`; one orientation gives arrays, several give lists |
| `cubicsolve` (MEX) | `cubicsolve(C, loop_fields)` on an (N, 4) array of cubics at once, NaN-padded roots |
| `photoselect`, `nucspinrmv` | the same |
| the pepper path (grid orientations with averaging over chi) | the private `resfields_oriented` for `pepper` |

## Line shapes and grids

| EasySpin | EasySpyn |
|---|---|
| `gaussian(x,x0,fwhm,diff,phase)`, `[ya,yd] = gaussian(...)`; the same for `lorentzian`, `voigtian` | the same arguments; `dispersion=True` returns `(absorption, dispersion)`; the Gaussian dispersion integral (`diff=-1`) raises `NotImplementedError` (EasySpin needs the Symbolic Toolbox) |
| `lshape(x,x0,fwhm,diff,alpha,phase)` | the same |
| `convspec(spec,df,fwhm,deriv,alpha,phase)` | the same, any dimension, one value per axis |
| `[grid,tri] = sphgrid(Symmetry,GridSize)` | `sphgrid(symmetry, grid_size, triangulate=True)` returns a `SphGrid` with `phi`, `theta`, `vecs`, `weights`, `tri_idx` (1-based, as EasySpin), `tri_areas`; open-phi grids are triangulated by `scipy.spatial.Delaunay`, which may pick other diagonals than MATLAB's `delaunay` where knots are co-circular (same areas and weights) |
| `gridparam` (private) | public |

## Powder and crystal spectra: pepper

| EasySpin | EasySpyn |
|---|---|
| `[B,spc,info] = pepper(Sys,Exp,Opt)`, outputs by `nargout`, a plot without outputs | `pepper(sys, exp, opt)` returns `Spectrum(x, spec, info)`; no plotting |
| `Exp.mwFreq`, `Range`/`CenterSweep`, `nPoints`, `Harmonic`, `ModAmp`, `mwPhase`, `Temperature`, `mwMode`, `SampleFrame`, `MolFrame`, `lightBeam`, `lightScatter` | the same keys |
| `Exp.Field` with `mwRange` (frequency sweeps), `Exp.Ordering`, `Exp.SampleRotation`, `Exp.CrystalSymmetry` other than P1 | `NotImplementedError` |
| `Opt.Method` `'matrix'`, `'hybrid'`, `'perturb'`, `'perturb1'`, `'perturb2'`, `'eig'`; `GridSize`, `GridSymmetry`, `GridFrame`, `Threshold`, `Transitions`, `Intensity`, `Freq2Field`, `BruteForce`, `separate` (`''`, `'transitions'`, `'orientations'`, `'sites'`), `Hybrid*`, `FuzzLevel`, `Verbosity` | the same keys; `Opt.ImmediateBinning` raises `NotImplementedError` |
| `Opt.separate` defaults to `''`: crystal orientations are summed | the same |
| the matrix and hybrid methods search `Exp.Range` only (6.0.15's `resfields` ignores `Exp.SearchRange`), the perturbation methods search 20 % beyond it | the same, so that lines just outside the range enter the spectrum only in the perturbation methods |
| `Sys.Nucs = 'Cu'` and other isotope mixtures (`isotopologues`) | `ValueError` naming the isotopes to choose from; mixtures are not ported |
| `Opt.FuzzLevel = 1e-10` by default | the same; use 0 for reproducible numbers |
| the MEX kernels `lisum1i`, `projecttriangles`, `projectzones` | `easyspyn.kernels`, bit-exact with the MEX and as fast when Numba is installed (`pip install spynach-spin[fast]`), 5 to 12 times slower in the pure NumPy fallback |
| `hamsymm(Sys)` returns `[Group,R]` | `hamsymm(sys)` returns `(group, R)`, with EasySpin's `isisotropic` test as it is in 6.0.15 (inverted logic, so a full `g` with equal eigenvalues is not reduced to `O3`) |
| `gridinterp`, `esspline1d`, `esspline2d` (private) | public in `easyspyn.interp`; the spline end conditions are EasySpin's (not-a-knot, clamped, periodic) |
| `fieldmod(B,spc,ModAmp,Harmonic)` | the same, Bessel-function pseudo-modulation in the Fourier domain |
| run time | the nitroxide at X band takes 0.19 s against EasySpin's 0.06 s (the per-orientation loop is Python); Cu(II) with four 14N takes 0.05 s by second-order perturbation theory (EasySpin 0.04 s) and 0.3 s by the hybrid method (EasySpin 0.08 s), where full diagonalisation takes about a minute on either side |

```python
nitroxide = {"g": [2.0088, 2.0061, 2.0027], "Nucs": "14N", "A": [16, 16, 86], "lwpp": 0.3}
exp = {"mwFreq": 9.5, "Range": [330, 350], "nPoints": 1024}
spectrum = es.pepper(nitroxide, exp, {"FuzzLevel": 0})  # EasySpin: [B,spc,info] = pepper(Sys,Exp,Opt)
print(spectrum.x.shape, spectrum.spec.shape, spectrum.info["Transitions"].tolist())
```

## Solution spectra: garlic

| EasySpin | EasySpyn |
|---|---|
| `[B,spc,info] = garlic(Sys,Exp,Opt)` | `garlic(sys, exp, opt)` returns `Spectrum(x, spec, info)`; `info["resfields"]` are the line positions, `info["intensities"]` their amplitudes |
| `Sys.g` (isotropic, or principal values whose mean is used), `Nucs`, `A`, `n`, `lw`, `lwpp`, `tcorr`/`logtcorr`, `Q` | the same |
| `Exp.mwFreq` with `Range`/`CenterSweep` (automatic if absent), or `Exp.Field` with `mwRange`/`mwCenterSweep`; `nPoints`, `Harmonic`, `ModAmp`, `mwPhase`, `Temperature` | the same keys |
| `Opt.Method` `'exact'` (Breit-Rabi) or `'perturb1'` to `'perturb5'`; `Opt.AccumMethod` `'binning'`, `'template'`, `'explicit'`; `Opt.IsoCutoff` | the same |
| `'template'` accumulation in 6.0.15 omits the amplitude factor `(template width / line width)^harmonic` | the factor is applied (EasySpin's development branch does the same), so template spectra match the other accumulation methods |
| `fastmotion(Sys,B,tcorr)`, `[lw,mI] = fastmotion(...)`, `'freq'` domain | `fastmotion(sys, field, tcorr, domain)` returns `(lw, mI)`; `Sys.Q` is used as given, like EasySpin, so a two-element `[e2qQ/h, eta]` row is not converted to principal values (`garlic` hands it the validated system, where it is) |
| `equivcouple(I,n)` | the same tuple `(F, N)` |
| isotope mixtures (`Nucs = 'Cu'`) | not ported |

```python
solution = {"g": 2.006, "Nucs": "14N", "A": 47, "lwpp": 0.1}
lines = es.garlic(solution, {"mwFreq": 9.5, "Range": [330, 350]})  # EasySpin: [B,spc] = garlic(Sys,Exp)
print("three lines at", np.sort(lines.info["resfields"]).round(3), "mT")
```

## Fitting: esfit

| EasySpin | EasySpyn |
|---|---|
| `esfit(data,fcn,p0,vary)`, `esfit(data,fcn,p0,lb,ub)`, `esfit(___,FitOpt)`; `result = esfit(...)`, or the GUI without an output | the same positional forms (`vary`, `lb`, `ub`, `opt` also as keywords) returning a `FitResult` dataclass with EasySpin's fields; no GUI |
| `pepper({Sys1,Sys2},Exp)` with `Sys.weight`, `Opt.separate = 'components'` | `pepper([sys1, sys2], exp)` and the same for `garlic`; isotopologues are not expanded |
| EasySpin-style parameters `{Sys0,Exp0}`, `{vSys}`, `{vSys,vExp}`; names `arg1.g(1,2)`, `arg1{2}.g(1,1)`, `arg1.initState{1}(1,3)` | lists of dicts, the same names; `initState` is the tuple `(state, basis)` and its vary entry `([0, 0, 0.2],)` |
| custom model functions receive the vector of *active* parameters (fixed ones are dropped) | the full vector, fixed parameters in place |
| `FitOpt.Method` keywords `simplex`, `levmar`, `montecarlo`, `genetic`, `grid`, `swarm`, `lsqnonlin` (Optimization Toolbox) and `fcn`, `int`, `dint`, `diff`, `fft` | the same; `lsqnonlin` is `scipy.optimize.least_squares` (trust-region reflective) |
| the `diff` keyword computes the Fourier-transform residual and `fft` the derivative (6.0.15 swaps the two cases) | `diff` is the derivative, `fft` the Fourier magnitude |
| `AutoScale`, `BaseLine`, `Mask`, `weight`, `OutArg`, `RandomSeed`, `RandomStart`, `CalculateUncertainties`, `maxTime`, `TolFun`, `TolEdgeLength`, `TolStep`, `delta`, `lambda`, `Gradient`, `SimplexPars`, `ScaleParams`, `nTrials`, `PopulationSize`, `EliteCount`, `maxGenerations`, `GridSize`, `randGrid`, `maxGridPoints`, `nParticles`, `SwarmParams`, `TolStallIter`, `IterFcn`, `Verbosity` | the same keys; `RandomSeed` seeds a NumPy generator (random draws differ from MATLAB's), `Verbosity = 1` prints the summary and `2` the iterations (EasySpin prints both at 1) |
| `esfit_simplex(fcn,x0,lb,ub,Opt)`, `esfit_levmar`, `esfit_montecarlo(fcn,lb,ub,Opt)`, `esfit_genetic`, `esfit_grid`, `esfit_swarm`, outputs by `nargout` | the same arguments, always `(x, info)` |
| `rescaledata(y,mode)`, `rescaledata(y,yref,mode,region)` | `rescaledata(y, yref=None, mode=None, region=None)` returning `(yscaled, scale)` |
| `basecorr(data,dim,n,region)`, `deriv(y)`, `deriv(x,y)`, `addnoise(y,SNR,model)` | the same (`dim=None` for the 2-D fit, `addnoise(..., rng=seed)`); `basecorr` and `deriv` return the tuple `(corrected, baseline)` and the derivative |
| `jacobianest`, `hccm` (private) | public |

```python
system = {"g": [2.0, 2.1], "lw": 1.2}
exp = {"mwFreq": 9.5, "Range": [300, 360], "nPoints": 300}
measured = es.pepper(system, exp, {"FuzzLevel": 0}).spec
# EasySpin: result = esfit(spc,@pepper,{Sys0,Exp,Opt},{Vary},FitOpt)
result = es.esfit(measured, es.pepper, [{"g": [2.01, 2.09], "lw": 1.2}, exp, {"FuzzLevel": 0}], [{"g": [0.03, 0.03]}], {"Verbosity": 0})
print(result.pnames, result.pfit.round(5), f"rmsd {result.rmsd:.1e}")
```

## ENDOR: endorfrq, salt, orisel

| EasySpin | EasySpyn |
|---|---|
| `[Pos,Int,Tra,Info] = endorfrq(Sys,Exp,Opt)`, intensities only with two or more outputs | `endorfrq(sys, exp, opt)` returns `EndorResonances(pos, int, trans, info)`; `int` is `None` with `Opt.Intensity = 'off'` |
| `Exp.Field`, `mwFreq`, `ExciteWidth`, `Temperature`, `Range`, `rf`, `SampleFrame`, `MolFrame`; `Opt.Transitions`, `Threshold`, `Nuclei`, `Intensity`, `Enhancement`, `OriWeights`, `OriThreshold` | the same keys |
| `endorfrq_perturb(Sys,Exp,Opt)` with `Opt.PerturbOrder` | the same; `trans` is empty, as in EasySpin |
| `orisel(Sys,Exp)` returns the summed weights, `[Weights,Trans] = orisel(...)` one column per transition | `orisel(sys, exp, opt)` returns the summed weights; `per_transition=True` the pair `(weights, transitions)`; `Exp.Orientations` takes rows `[phi, theta]` (or the `2 x n` array) |
| `anisosubsys` (private) | public |
| `[rf,spec,info] = salt(Sys,Exp,Opt)`, a plot without outputs | `salt(sys, exp, opt)` returns `Spectrum(x, spec, info)`; `info` carries `Transitions`, `Selectivity`, `nOrientations` and, for strong orientation selection, a `warning` string instead of the printed warning |
| `Exp.Ordering`, `SampleRotation`, crystal symmetries other than P1 | `NotImplementedError` |
| `Sys.lwEndor`, `Opt.Method`, `GridSize`, `GridSymmetry`, `GridFrame`, `separate`, `OriPreSelect`, `OriWeights`, `Threshold`, `Nuclei`, `Intensity`, `Enhancement` | the same keys |
| `salt({Sys1,Sys2},Exp)` | `salt([sys1, sys2], exp)` with `weight`; isotopologues are not expanded |

```python
proton = {"S": 1.5, "D": 200, "Nucs": "1H", "A": [3, 5], "lwEndor": 0.2}
spectrum = es.salt(proton, {"Field": 350, "mwFreq": 9.5, "Range": [5, 25]})  # EasySpin: [rf,spc] = salt(Sys,Exp)
print(spectrum.x.shape, spectrum.info["Transitions"].tolist())
```

## Magnetometry: curry

| EasySpin | EasySpyn |
|---|---|
| `curry(Sys,Exp,Opt)` with outputs chosen by `Opt.Output` and `nargout` (`'muBM chimol'` for two outputs) | `curry(sys, exp, opt)`; `Opt.Output` defaults to `'muBM chimol'`; one keyword returns the array, several the tuple in that order; shapes are `(nFields, nTemperatures)` as in EasySpin |
| `Exp.Field`, `Temperature`, `SampleFrame`, `MolFrame`, `CrystalSymmetry` (P1); `Sys.TIP`; `Opt.Output`, `Units`, `Method`, `GridSize`, `GridSymmetry`, `GridFrame`, `Spins`, `deltaB` | the same keys; `Exp.Ordering` and `SampleRotation` raise `NotImplementedError` |
| `curry({Sys1,Sys2},...)` is an error | the same |
| the plot without outputs | no plotting |

```python
chi = es.curry({"S": 0.5, "g": 2}, {"Temperature": [10, 100], "Field": 0}, {"Output": "chimol"})  # EasySpin: chi = curry(Sys,Exp,Opt)
print(chi.shape, chi.round(12))
```

## Slow motion: chili

| EasySpin | EasySpyn |
|---|---|
| `[B,spc,info] = chili(Sys,Exp,Opt)`, a plot without outputs | `chili(sys, exp, opt)` returns `Spectrum(x, spec, info)`; `info` carries the basis size, the norm of the equilibrium vector and a `warning` string where EasySpin prints one |
| `Opt.LiouvMethod`: `'fast'` (C kernel, default for S = 1/2 with up to two nuclei) or `'general'` | the general method only; `'fast'` raises `NotImplementedError`. EasySpin's own test requires the two to agree at 1e-3 |
| `Sys.tcorr`, `logtcorr`, `Diff`, `logDiff` (1, 2 or 3 values), `DiffFrame`, `lw`, `lwpp`, `Potential`, `Exchange` | the same, except `DiffFrame` (ignored by EasySpin's general method too) and `Exchange` (fast method only; `NotImplementedError`) |
| `Exp.mwFreq` with `Range`/`CenterSweep` or `Field` with `mwRange`/`mwCenterSweep`, `nPoints`, `Harmonic`, `ModAmp`, `mwPhase`, `Temperature`, `SampleFrame`, `Ordering` (number or function) | the same keys |
| `Opt.LLMK`, `evenK`, `jKmin`, `highField`, `pImax`, `pImaxall`, `MpSymm`, `GridSize`, `GridSymmetry`, `GridFrame`, `PostConvNucs`, `Solver` (`'L'`, `'\'`, `'E'`, `'C'`, `'B'`), `FieldSweepMethod`, `Lentz`, `Threshold`, `Rescale`, `IncludeNZI`, `PeqTol`, `separate` | the same keys; `Opt.useLMKbasis`, `StartVec`, `Diagnostics`, `BasisAnalysis` are not ported |
| `wigner3j`, `wigner6j` (`'f'`/`'r'` methods) | exact integer arithmetic for every rank; the same call forms |
| `wignerd(J,...)`, `wignerd([J m1 m2],...)`, phase `'+'`/`'-'` | the same, `phase=` keyword; single elements are vectorised over the angles |
| `tensor_cart2sph`, `tensor_sph2cart` | the same |
| `istotensor`, `hil2liouv`, `generateoribasis`, `jjjsymbol`, `liouvhamiltonian`, `diffsuperop`, `chili_xlmk`, `chili_eqpopvec`, `chili_lanczos`, `chili_conjgrad`, `chili_contfracspec`, `pqorder`, `magint`, `rbos` (private) | private functions of `easyspyn.slowmotion` |

```python
slow = {"g": [2.008, 2.0061, 2.0027], "Nucs": "14N", "A": [16, 16, 86], "lw": 0.1, "tcorr": 1e-9}
spectrum = es.chili(slow, {"mwFreq": 9.8, "Range": [344, 354]})  # EasySpin: [B,spc] = chili(Sys,Exp)
print(spectrum.spec.shape, spectrum.info["nBasis"])
```

## Pulse EPR: saffron

| EasySpin | EasySpyn |
|---|---|
| `[t,td,info] = saffron(Sys,Exp,Opt)`, `{t1,t2}` for 2-D experiments, a plot without outputs | `saffron(sys, exp, opt)` returns `Spectrum(x, spec, info)`; `x` is the time axis or a tuple of two, `spec` the complex modulation signal (the ENDOR spectrum for Mims ENDOR), `info` holds `td`, `fd`, `f`/`f1`/`f2` and `nSkippedOrientations` |
| `Opt.SimulationMode = 'fast'` (ideal pulses, frequency-domain histogram) or `'thyme'` (explicit propagation with shaped pulses) | the fast engine only; `'thyme'`, shaped pulses, `Exp.DetWindow`, resonators and the pulse-parameter dimensions raise `NotImplementedError` |
| `Exp.Sequence = '2pESEEM'`, `'3pESEEM'`, `'4pESEEM'`, `'HYSCORE'`, `'MimsENDOR'` with `Field`, `dt`, `nPoints`, `tau`, `T`, `t1`, `t2`, `Range`, `tprf` | the same keys |
| `Exp.Sequence = {p1 tau p2 T p3 tau}` with `Exp.Dim1 = {'d2' dt}`, pulses `struct('Flip',pi/2,'Phase',0,'tp',0.016)`, `Exp.Filter`, `Exp.Pathways` | a list alternating pulse dicts and delays, `Exp["Dim1"] = [["d2", dt]]` (several delays as `"d1,d2"`, negative `dt` decrements); ideal pulses and rectangular real pulses (`Opt.nOffsets`, `lwOffset`); the thyme-only variants (`'p1.Flip'`, non-linear increments, more than two dimensions) raise |
| `Exp.mwFreq` + `Exp.ExciteWidth` (orientation selection), `Opt.OriThreshold`, `Opt.Transitions`, `Sys.HStrain` | the same |
| `Sys.T1`, `Sys.T2` (decays of the predefined experiments) | the same, as single values |
| `Opt.ProductRule`, `Opt.Nuclei`, `Opt.TimeDomain`, `Opt.Expand`, `Opt.EndorMethod`, `Opt.Window`, `Opt.ZeroFillFactor`, `Opt.GridSize`, `Opt.GridSymmetry`, `Opt.GridFrame` | the same keys |
| `Exp.SampleFrame`, `Exp.MolFrame` (crystals), `Exp.CrystalSymmetry` | P1 only; `Exp.Ordering` and `SampleRotation` are not ported |
| `Sys = {Sys1, Sys2}` with `Sys.weight` | a list of dicts; `info["fd"]` is the weighted sum |
| `sf_peaks` (C), `sf_evolve`, `sf_pathways`, `pathwayparser` (private) | NumPy; `sf_pathways` is public, the rest private in `easyspyn.pulsed` |
| `larmorfrq`, `apowin`, `fdaxis`, `exponfit`, `basecorr` | the same; `exponfit(x, y, n, noconst=True)` instead of the `'noconst'` flag |
| isotope mixtures in `Sys.Nucs` (`'N'`, `'Cu'`) | not ported; give isotopes |

```python
seq = {"Field": 350, "Sequence": "3pESEEM", "dt": 0.01, "nPoints": 512, "tau": 0.1}
eseem = es.saffron({"Nucs": "14N", "A": [-1, -1, 2], "Q": [-1, -1, 2]}, seq)  # EasySpin: [t,td] = saffron(Sys,Exp)
print(eseem.spec.shape, eseem.info["fd"].shape)
```

## Not ported

The simulator `spidyan` and `saffron`'s thyme engine (shaped pulses,
explicit propagation, resonators, transient detection). `esfit`'s window
and `eprconvert`. In `salt` and `curry`: partial ordering, sample
rotations. In `chili`: the fast C kernel, Heisenberg exchange, the LMK basis
option, user-supplied starting vectors. In `pepper` and `garlic`: frequency sweeps of
powders, isotope mixtures, partial ordering, sample rotations, immediate
binning. Not planned: orbital
angular momentum (`Sys.L`, `soc`, `gL`, `CF*`), higher-order Zeeman terms
(`Sys.Ham...`), tilted Stevens-operator frames (`Sys.B?Frame`, needs
`wignerd`), crystal symmetries beyond P1 and sample rotations in the
resonance-field functions, the GUIs, MD-trajectory methods (`cardamom`,
`mdhmm`).

## Verification

Reference data are written by EasySpin itself: the dump scripts in
`oracle/matlab/easyspin/dumps/` (`esdump_ham.m`, `esdump_resfields.m`,
`esdump_pepper.m`, `esdump_esfit.m`, `esdump_endor.m`, `esdump_chili.m`,
`esdump_saffron.m`) run in MATLAB with the installed EasySpin on the path
(read-only) and write one case per spin system or spectrum to
`$EASYSPYN_FIXTURES/<version>/<case>/`, with the inputs recorded in
`meta.json` so that `tests/easyspyn/test_parity*.py` rebuild each case from
the same input. The cases and the measured agreement are listed in
[`porting/EASYSPIN_FIXTURES.md`](../porting/EASYSPIN_FIXTURES.md).
