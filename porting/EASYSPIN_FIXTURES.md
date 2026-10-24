# EasySpin fixtures

Reference data for the `epreasy` parity tests, written by EasySpin 6.0.15 in
MATLAB R2026b on the Mac through `oracle/run_easyspin_fixtures.sh`
(`oracle/matlab/easyspin/run_easyspin_fixtures.m`, settings block at the top).
Fixtures live at `/Users/alexej/Documents/PROG/easyspin-fixtures/6.0.15/<case>/`
(`$EPREASY_FIXTURES` for the tests), separate from the Spinach fixtures. The
installed EasySpin release is only read.

| dump script | cases | contents | parity test | status |
|---|---|---|---|---|
| `esdump_resfields.m` | `rf_nitroxide`, `rf_nitroxide_parallel`, `rf_nitroxide_unpol`, `rf_nitroxide_circ`, `rf_nitroxide_alltrans`, `rf_nitroxide_molframe`, `rf_nitroxide_strain`, `rf_nitroxide_photo`, `rf_cu2n_matrix`, `rf_cu2n_hybrid`, `rf_mn_boltzmann`, `rf_mn_dstrain`, `rf_radical_pair`, `eig_nitroxide`, `eig_mn`, `lineshapes`, `sphgrid`, `cubic_photo` | `resfields` positions, intensities, widths and transitions on three crystal orientations (`Opt.FuzzLevel = 0`) for the excitation modes, `Transitions = 'all'`, `CenterSweep` with `MolFrame`, H/g/A strain, photoselection, the matrix and hybrid methods, Boltzmann populations, D strain; `resfields_eig` fields and intensities; `gaussian`, `lorentzian`, `lshape`, `voigtian` for every derivative and phase, `convspec` in 1-D and 2-D; `sphgrid` for ten point groups at two sizes with triangulations; the `cubicsolve` MEX on 2000 cubics; `photoselect` | `tests/epreasy/test_parity_resfields.py` | generated 2026-10-24; `resfields` positions, intensities, widths and transitions agree at 1e-9 in every case (matrix and hybrid, strains, Boltzmann, excitation modes, photoselection), `resfields_eig` fields and intensities at 1e-9, line shapes at 1e-11 to 1e-13, grid angles and weights at 1e-13 (the explicit closed-octant triangulation exactly; Delaunay-based ones by area sum, since `scipy.spatial.Delaunay` and MATLAB's `delaunay` may differ on co-circular knots), `photoselect` at 1e-13 |
| `esdump_ham.m` | `nitroxide_frames`, `cu_2n`, `mn_s52_stevens`, `radical_pair`, `triplet_full_tensors`, `helpers` | `spinvec`, `hsdim`; `ham` (`H0`, `mux`, `muy`, `muz`, `H(B0)`, `muzL`); `ham_zf`, `ham_ee`, `ham_hf`, `ham_nq`, `ham_nn`, `ham_ez(B0)`, `ham_nz(B0)`; `levels` on three orientations x four fields and along z; `sop` for every component of every spin, a product and a transition-selective operator; `stev` for all ranks up to min(2S, 6); `erot`, `eulang`, `ang2vec`, `vec2ang`, `mt2mhz`, `mhz2mt`, `unitconvert`, `nucdata` | `tests/epreasy/test_parity.py` | generated 2026-10-24; all stages agree (operators and rotations at 1e-14, Hamiltonians and levels at 1e-12) |

A system listing spin-zero nuclei with one `A` row per nucleus
(`'12C,13C,16O'`) is not a case: EasySpin 6.0.15's `ham_hf` errors on it
(it tests the unprocessed `A` for the full-matrix form after the nuclei were
removed). `epreasy` handles it; `tests/epreasy/test_foundation.py` covers it.
