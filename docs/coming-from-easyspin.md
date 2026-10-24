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
8. [Not ported](#not-ported)
9. [Verification](#verification)

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

## Not ported

The simulators `pepper`, `garlic`, `salt`, `curry`, `chili`, `saffron`,
`spidyan` and fitting (`esfit`): the next phases. Not planned: orbital
angular momentum (`Sys.L`, `soc`, `gL`, `CF*`), higher-order Zeeman terms
(`Sys.Ham...`), tilted Stevens-operator frames (`Sys.B?Frame`, needs
`wignerd`), crystal symmetries beyond P1 and sample rotations in the
resonance-field functions, the GUIs, MD-trajectory methods (`cardamom`,
`mdhmm`).

## Verification

Reference data are written by EasySpin itself: `oracle/matlab/easyspin/dumps/esdump_ham.m`
runs in MATLAB with the installed EasySpin on the path (read-only) and
writes one case per spin system to `$EPREASY_FIXTURES/<version>/<case>/`,
with the spin system recorded in `meta.json` so that
`tests/epreasy/test_parity.py` rebuilds the case from the same input. The
five systems and the measured agreement are listed in
[`porting/EASYSPIN_FIXTURES.md`](../porting/EASYSPIN_FIXTURES.md).
