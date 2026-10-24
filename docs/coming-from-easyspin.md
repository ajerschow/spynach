# Coming from EasySpin

What carries over from [EasySpin](https://easyspin.org) 6.0.15 to `epreasy`,
what differs, and what is not ported. The foundation is covered in the guide
[EPR spin systems with `epreasy`](epreasy.md); this page is the reference,
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
11. [Not ported](#not-ported)
12. [Verification](#verification)

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

```python
import numpy as np

import epreasy as es

spin_system = {"S": 1 / 2, "g": [2.0088, 2.0061, 2.0027], "Nucs": "14N", "A": [16, 16, 86]}
H0, mux, muy, muz = es.ham(spin_system)  # EasySpin: [H0,mux,muy,muz] = ham(Sys)
H = es.ham(spin_system, [0, 0, 350])  # EasySpin: H = ham(Sys,[0 0 350])
print(H.shape, np.allclose(H, H0 - 350 * muz))
```

## Spin systems

| EasySpin | epreasy |
|---|---|
| `Sys` structure with fields `S`, `g`, `gFrame`, `Nucs`, `A`, `AFrame`, `D`, `DFrame`, `Q`, `QFrame`, `ee`/`J`/`dip`/`dvec`, `eeFrame`, `ee2`, `nn`, `nnFrame`, `B1`..`B12`, `sigma`, `gnscale`, `n`, `lw`, `lwpp`, `lwEndor`, `gStrain`, `DStrain`, `HStrain`, `AStrain` | a dict with the same keys and the same size rules and shorthands (`g_`, `D_`, `A_` spherical forms, `D=[D, E]`, `Q=[eeqQ, eta]`) |
| `validatespinsys(Sys)` (private) returns `[FullSys, err]` | `validatespinsys(sys)` is public and returns the processed dict; errors raise |
| `spinvec(Sys)`, `hsdim(Sys)` | the same |
| spin-zero nuclei are removed | the same; `ham_hf` then works (in EasySpin 6.0.15 `ham_hf` errors for a system listing spin-zero nuclei with one `A` row per nucleus, since it tests the unprocessed `A` for the full-matrix form) |
| `Sys.initState`, `Sys.tdm`, diffusion fields | kept on the dict for the simulators; `tdm` is normalised as in EasySpin; `initState` is stored as given |

## Spin operators and rotations

| EasySpin | epreasy |
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

| EasySpin | epreasy |
|---|---|
| `[H0,mux,muy,muz] = ham(Sys)`, `H = ham(Sys,B0)`, `[H0,muzL] = ham(Sys,B0)` | `ham(sys)` returns the 4-tuple; `ham(sys, B0)` returns `H`; `ham(sys, B0, split=True)` returns `(H0, muzL)` |
| `ham(...,'sparse')` and the same option on every component | `sparse=True` keyword |
| `ham_ez(Sys,B)`, `[mux,muy,muz] = ham_ez(Sys)`, `ham_ez(Sys,B,eSpins)` | `ham_ez(sys, B)`, `ham_ez(sys)`, `ham_ez(sys, B, espins)`; the same for `ham_nz` with `nspins` |
| `ham_hf(Sys,eSpins,nSpins)`, `ham_zf(Sys,eSpins)`, `ham_nq(Sys,nSpins)`, `ham_ee(Sys,eSpins)`, `ham_nn(Sys,nSpins)` | the same, spin numbers 1-based |
| `ham_so`, `ham_cf`, `ham_oz`, `ham_ezho` | not ported (orbital angular momentum, higher-order Zeeman) |
| `levels(Sys,Ori,B)`, `levels(Sys,phi,theta,B)`, `[E,V] = levels(...)` | `levels(sys, ori, B)`, `levels(sys, phi=..., theta=..., B=...)`, `levels(..., vectors=True)`; singleton axes are squeezed as in EasySpin; levels are computed for all orientations and fields in one batched `eigh` |

## Units, constants, nuclear data

| EasySpin | epreasy |
|---|---|
| `bmagn`, `nmagn`, `planck`, `hbar`, `gfree`, `clight`, `mu0`, `boltzm`, `evolt`, `echarge`, `amu`, `avogadro` as functions | module attributes with the same values (`es.bmagn`) |
| `unitconvert(value,'cm^-1->MHz')`, `unitconvert(value,'mT->MHz',g)`, `mt2mhz`, `mhz2mt` | the same |
| `nucdata('14N,1H')` returns up to five outputs; `nucspin`, `nucgval`, `nucqmom`, `nucabund` | `nucdata` returns a named tuple `(spin, gn, qm, abund, nucs)`; the wrappers are the same |
| the isotope table `private/isotopedata.txt` | shipped verbatim as `epreasy/data/isotopedata.txt`; `isotopes()` returns it. It differs from `spynach`'s table (from Spinach) in some values, so each package keeps its own |

## Resonance fields

| EasySpin | epreasy |
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

| EasySpin | epreasy |
|---|---|
| `gaussian(x,x0,fwhm,diff,phase)`, `[ya,yd] = gaussian(...)`; the same for `lorentzian`, `voigtian` | the same arguments; `dispersion=True` returns `(absorption, dispersion)`; the Gaussian dispersion integral (`diff=-1`) raises `NotImplementedError` (EasySpin needs the Symbolic Toolbox) |
| `lshape(x,x0,fwhm,diff,alpha,phase)` | the same |
| `convspec(spec,df,fwhm,deriv,alpha,phase)` | the same, any dimension, one value per axis |
| `[grid,tri] = sphgrid(Symmetry,GridSize)` | `sphgrid(symmetry, grid_size, triangulate=True)` returns a `SphGrid` with `phi`, `theta`, `vecs`, `weights`, `tri_idx` (1-based, as EasySpin), `tri_areas`; open-phi grids are triangulated by `scipy.spatial.Delaunay`, which may pick other diagonals than MATLAB's `delaunay` where knots are co-circular (same areas and weights) |
| `gridparam` (private) | public |

## Powder and crystal spectra: pepper

| EasySpin | epreasy |
|---|---|
| `[B,spc,info] = pepper(Sys,Exp,Opt)`, outputs by `nargout`, a plot without outputs | `pepper(sys, exp, opt)` returns `Spectrum(x, spec, info)`; no plotting |
| `Exp.mwFreq`, `Range`/`CenterSweep`, `nPoints`, `Harmonic`, `ModAmp`, `mwPhase`, `Temperature`, `mwMode`, `SampleFrame`, `MolFrame`, `lightBeam`, `lightScatter` | the same keys |
| `Exp.Field` with `mwRange` (frequency sweeps), `Exp.Ordering`, `Exp.SampleRotation`, `Exp.CrystalSymmetry` other than P1 | `NotImplementedError` |
| `Opt.Method` `'matrix'`, `'hybrid'`, `'perturb'`, `'perturb1'`, `'perturb2'`, `'eig'`; `GridSize`, `GridSymmetry`, `GridFrame`, `Threshold`, `Transitions`, `Intensity`, `Freq2Field`, `BruteForce`, `separate` (`''`, `'transitions'`, `'orientations'`, `'sites'`), `Hybrid*`, `FuzzLevel`, `Verbosity` | the same keys; `Opt.ImmediateBinning` raises `NotImplementedError` |
| `Opt.separate` defaults to `''`: crystal orientations are summed | the same |
| the matrix and hybrid methods search `Exp.Range` only (6.0.15's `resfields` ignores `Exp.SearchRange`), the perturbation methods search 20 % beyond it | the same, so that lines just outside the range enter the spectrum only in the perturbation methods |
| `Sys.Nucs = 'Cu'` and other isotope mixtures (`isotopologues`) | `ValueError` naming the isotopes to choose from; mixtures are not ported |
| `Opt.FuzzLevel = 1e-10` by default | the same; use 0 for reproducible numbers |
| the MEX kernels `lisum1i`, `projecttriangles`, `projectzones` | `epreasy.kernels`, bit-exact with the MEX and as fast when Numba is installed (`pip install spynach-spin[fast]`), 5 to 12 times slower in the pure NumPy fallback |
| `hamsymm(Sys)` returns `[Group,R]` | `hamsymm(sys)` returns `(group, R)`, with EasySpin's `isisotropic` test as it is in 6.0.15 (inverted logic, so a full `g` with equal eigenvalues is not reduced to `O3`) |
| `gridinterp`, `esspline1d`, `esspline2d` (private) | public in `epreasy.interp`; the spline end conditions are EasySpin's (not-a-knot, clamped, periodic) |
| `fieldmod(B,spc,ModAmp,Harmonic)` | the same, Bessel-function pseudo-modulation in the Fourier domain |
| run time | the nitroxide at X band takes 0.19 s against EasySpin's 0.06 s (the per-orientation loop is Python); Cu(II) with four 14N takes 0.05 s by second-order perturbation theory (EasySpin 0.04 s) and 0.3 s by the hybrid method (EasySpin 0.08 s), where full diagonalisation takes about a minute on either side |

```python
nitroxide = {"g": [2.0088, 2.0061, 2.0027], "Nucs": "14N", "A": [16, 16, 86], "lwpp": 0.3}
exp = {"mwFreq": 9.5, "Range": [330, 350], "nPoints": 1024}
spectrum = es.pepper(nitroxide, exp, {"FuzzLevel": 0})  # EasySpin: [B,spc,info] = pepper(Sys,Exp,Opt)
print(spectrum.x.shape, spectrum.spec.shape, spectrum.info["Transitions"].tolist())
```

## Solution spectra: garlic

| EasySpin | epreasy |
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

| EasySpin | epreasy |
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

## Not ported

The simulators `salt`, `curry`, `chili`, `saffron` and `spidyan`: the next
phases. `esfit`'s window and `eprconvert`. In `pepper` and `garlic`: frequency sweeps of
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
`esdump_pepper.m`, `esdump_esfit.m`) run in MATLAB with the installed EasySpin on the path
(read-only) and write one case per spin system or spectrum to
`$EPREASY_FIXTURES/<version>/<case>/`, with the inputs recorded in
`meta.json` so that `tests/epreasy/test_parity*.py` rebuild each case from
the same input. The cases and the measured agreement are listed in
[`porting/EASYSPIN_FIXTURES.md`](../porting/EASYSPIN_FIXTURES.md).
