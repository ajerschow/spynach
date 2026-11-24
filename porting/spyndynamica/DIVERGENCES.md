# Divergences

A fixture records what SpinDynamica 3.10.2 returns. That is the reference for conventions, but it is not
always what is documented or what is right. Each entry says what SpinDynamica does, what the port does, and
how the difference was established. The cases that record the SpinDynamica behaviour stay in the fixtures.

## D1. `OperatorVectorRepresentation` ignores `OperatorBasis -> opbasis`

- **Documented:** `OperatorVectorRepresentation[op, OperatorBasis -> basis]` gives the vector in that basis.
- **SpinDynamica does:** the option is ignored and the vector comes back in the current operator basis. Only
  the positional form `OperatorVectorRepresentation[op, opbasis]` uses the given basis.
- **Established by:** `(B_k | op)` computed from the stored basis operators
  (`test_operator_vectors_brackets_and_amplitudes`); case `04_liouville: two_half_vectors/option_form_is_ignored`.
- **Port:** the basis is an explicit argument and is always honoured.
- **Report to the author:** yes (documentation or code).

## D2. The option `Chronology` is ignored

- **Documented:** `Chronology` is "an option for many routines", with `$Chronology` as its default.
- **SpinDynamica does:** `NPropagate`, `NPropagationOperator` and `TransformationAmplitude` return the same
  result with `Chronology -> "RightToLeft"` as without it, and with `Chronology -> "LeftToRight"` while
  `$Chronology = "RightToLeft"`. Only the global `$Chronology` changes the order of the events.
- **Established by:** propagation with NumPy matrix exponentials in both orders
  (`test_chronology_follows_the_global_setting_only`); cases `05_propagation: two_half/chronology_option_is_ignored*`
  and `two_half_right_to_left/*`.
- **Port:** the order is an argument of the call (default left to right) and there is no global setting.
- **Report to the author:** yes.

## D3. `Signal1D` returns an odd number of points for an even request

- **Documented:** `ForceEvenNumberOfPoints -> True` (the default) drops the last point "if the requested
  number of acquired points is odd", because the fast Fourier transform wants an even number.
- **SpinDynamica does:** the opposite. A request for 16 points (`{{ωSW, 16}}`, or a time range with 16 samples)
  gives 15, with the message `Signal1D::droplast`; a request for 17 gives 17.
- **Established by:** the sampling stored with the signal and the length of its expansion; cases
  `06_signals: two_half/timing_*` and `points_*`.
- **Port:** `signal_1d` returns the number of points asked for.
- **Report to the author:** yes.

## D4. `FT` of an odd number of points without zero filling is shifted by one point

- **SpinDynamica does:** for an odd number of points and `ZeroFill -> False` the frequency axis is symmetric
  about zero, but every value sits one point too low on it (the value of the highest frequency is at the far
  end). With zero filling (the default) the number of points is even and the axis is right.
- **Established by:** the discrete Fourier sum evaluated at the stated frequencies
  (`test_fourier_transform`); cases `06_signals: fourier/list_odd_length`, `two_half/ab_spectrum_no_zero_fill`.
- **Port:** `ft` returns the transform at the stated frequencies for every length.
- **Report to the author:** yes. Together with D3 it affects the default 16-point request when zero filling is
  switched off.

## D5. `PolarizedDensityOperator[{p, axis}]` ignores the level and the axis

- **Documented:** `PolarizedDensityOperator[{p, "z"}]` "generates a density operator with polarization level
  p along the z-axis for all spins in the current SpinSystem[]".
- **SpinDynamica does:** for two spins-1/2 it returns the pure state `|αα><αα|` for `{3/10, "z"}`, for
  `{3/10, "x"}` and for `{1, "z"}` alike. The forms with a list of spins, `[{1}, {p, "z"}]`, and with one
  entry per spin work: the state is the product of `1/2 + p I_axis`.
- **Established by:** the stored matrices; cases `09_states: two_half/polarized_all_z`, `polarized_all_x`,
  `polarized_full`, against `polarized_spin_1_z` and `polarized_each`.
- **Port:** `polarized(s, p, axis)` is the product of `1/2 + p I_axis` over all spins.
- **Report to the author:** yes.

## D6. `OperatorProjection` and `ProjectedMatrixRepresentation` are documented but not implemented

- **Documented:** both have usage messages in `QM/SpinOperators.m`.
- **SpinDynamica does:** nothing. The package holds no definition of either, and the calls come back
  unevaluated.
- **Established by:** a search of the source for the two names (only the usage messages) and the unevaluated
  results of the generator.
- **Port:** `A.projection(B)` is `B (B|A)/(B|B)`. The projected matrix representation is what
  `secularize(A, H)` gives: the coherences between non-degenerate eigenstates of `H` removed.
- **Report to the author:** yes.

## D7. `RotateEuler` about an axis system applies the inverse frame change

- **Documented:** `RotateEuler[obj, {α, β, γ}, {vx, vy, vz}]` "rotates obj by the angle γ about the axis vz,
  followed by a rotation by the angle β about vy, followed by a rotation through the angle α about vz", and
  with `{α, β, γ} = AxesToEuler[sys1, sys2]` the call `RotateEuler[sys1, {α, β, γ}, sys1]` superimposes `sys1`
  on `sys2`.
- **SpinDynamica does:** with `A` the matrix whose columns are the axes and `R` the Euler rotation matrix, the
  documented rotation is `A R Aᵀ`; the routine applies `Aᵀ R A`. The documented example does not give `sys2`.
  Without the third argument (laboratory axes) there is no difference.
- **Established by:** the product of the three rotations about `vz`, `vy`, `vz` built from Rodrigues' formula,
  and the superposition of two axis systems (`test_axes_to_euler_and_rotate_euler`); cases
  `12_bases_filters_geometry: geometry/rotate_euler_about_axes`, `rotate_euler_about_z_only`,
  `rotate_euler_superimpose_fails` against `system_B`.
- **Port:** `rotate_euler(obj, angles, axes)` is the documented rotation.
- **Report to the author:** yes. `AxesToEuler` itself is right.

## D8. `SphericalTensorOperatorBasis` rejects the documented order by rank

- **Documented:** "The option Sorted->True or Sorted->{SphericalRank,CoherenceOrder} may be used to sort the
  basis, first according to spherical rank and then with respect to coherence order."
- **SpinDynamica does:** `Sorted -> {SphericalRank, CoherenceOrder}` is refused (`OperatorBasis::invalidsortopt`,
  the value is missing from the list of allowed ones although the code for it exists), and `True`, `False`,
  `SphericalRank` and `CoherenceOrder` all give the order by coherence order and then rank.
- **Established by:** cases `tensor_basis_sorting/rank_then_order_is_rejected`,
  `spherical_rank_same_as_default`, `unsorted_same_as_default`.
- **Port:** `spherical_tensor_basis(s, sort="coherence")` is SpinDynamica's order; `sort="rank"` is the
  documented one.
- **Report to the author:** yes.

## D9. `SphericalComponentFiltrationSuperoperator` pairs components of different tensors

- **Documented:** the superoperator "for filtering out the spherical component of rank λ with quantum order
  being transformed from μ1 to μ2".
- **SpinDynamica does:** it lists the normalised operators of rank λ and order μ2, and those of order μ1, each
  list put through `Union` (which sorts it), and pairs them by position. The two lists are sorted
  independently, so when μ1 and μ2 differ and more than one tensor has the rank (three spins-1/2 have three
  tensors of rank 1 and two of rank 2), component μ1 of one tensor can be sent to component μ2 of another.
  For two spins-1/2, and whenever μ1 = μ2 in the cases tried, the result is right.
- **Established by:** the projection `(2λ + 1)/(8π²) ∫ D*(μ2, μ1; Ω) R(Ω) dΩ` over the rotation group,
  evaluated by exact quadrature (`test_spherical_component_filter_of_three_spins`); cases
  `three_half_filter/applied_1_1_0`, `applied_2_m2_2` and `applied_2_0_1` differ from it, the other thirteen
  agree.
- **Port:** `spherical_component_filter` pairs the two components within each tensor, and is defined for any
  spins (SpinDynamica: spins-1/2 only).
- **Report to the author:** yes.

## D10. `DefineOperatorBasis` with an ordering stores the inverse ordering in the transformation matrix

- **Documented:** `DefineOperatorBasis[new, opbasis, superop, ordering]` "defines an operator basis by applying
  the superoperator to each of the basis operators of opbasis, ordered according to the list ordering".
- **SpinDynamica does:** `BasisOperators[new]` are as documented, but the stored
  `OperatorBasisTransformationMatrix[new, opbasis]` is built with the transposed permutation matrix. Vectors
  and superoperator matrices in the new basis are computed from that matrix, so they belong to the inverse
  ordering and not to the basis operators. An ordering that is its own inverse (a set of swaps) hides it.
- **Established by:** the brackets `(B_i | A)` and `(B_i | S B_j)` computed from SpinDynamica's own stored basis
  operators (`test_defined_operator_bases`); cases `defined_operator_bases/transformation_matrix`,
  `vector_in_transformed_reordered`, `commutation_in_transformed_reordered` against `transformed_reordered`.
- **Port:** `basis.reordered(order)` and `basis.transformed(superoperator, order)` return a basis whose vectors
  and matrices are brackets with its operators.
- **Report to the author:** yes.

## D11. `SingletPolarizationLevelOperator` refuses separate pairs and accepts pairs that share a spin

- **Documented:** `SingletPolarizationLevelOperator[{{j1,k1},{j2,k2}..}]` "is the operator for the mean level of
  singlet polarization for the specified spin pairs", and the message `sharedspins` says that "spins may not
  be shared between different pairs".
- **SpinDynamica does:** the test is the wrong way round. `[{{1, 2}, {3, 4}}]` gives the message and `$Failed`;
  `[{{1, 2}, {2, 3}}]` returns the sum of the two pair operators.
- **Established by:** cases `13_polarization_exchange_algebras: four_half_polarization/separate_pairs_are_rejected`
  and `pairs_that_share_a_spin`; for separate pairs the amplitude of the summed operator in a density operator
  with singlet polarization `p` of both pairs is `p` (`test_polarization_levels_of_other_systems`).
- **Port:** `singlet_polarization_level(s, pairs)` adds the operators of pairs that share no spin and refuses
  the others.
- **Report to the author:** yes.

## D12. `SymmetryAdaptedBasis` depends on random numbers and can return kets that are no basis

- **Documented:** `SymmetryAdaptedBasis` "constructs a symmetry-adapted basis of spin states".
- **SpinDynamica does:** it diagonalises a combination of commuting operators with random integer
  coefficients, and separates the representations with random real numbers. The kets therefore change with
  the draw. Out of eleven seeds tried, three gave a wrong result in one of seven cases: two null kets among
  the eight of three spins-1/2 under the symmetric group (seeds 1 and 4), and a failure with messages for
  four spins-1/2 under the cyclic group (seed 9); without a seed the symmetric group of four spins failed
  once. The form with explicit group elements fails for the dihedral group of four spins-1/2 for every seed
  tried.
- **Established by:** the norms of the stored kets; case
  `three_half_groups_other_draw/symmetry_adapted_basis_is_defective` (seed 1) against
  `three_half_groups/symmetry_adapted_basis` (seed 2, used for all the reference bases).
- **Port:** `symmetry_adapted_basis` works with highest-weight kets and a fixed seed: the same basis in every
  session, for any permutation group of equal spins. The tests check orthonormality, the block structure of
  every group operator, irreducibility, equal matrices for equivalent representations, and the `(J, M)` labels.
- **Report to the author:** yes.

## D13. `Signal1D` by the COMPUTE method is normalised differently from the other methods

- **Documented:** `Signal1D` returns the amplitude of the observable in the density operator,
  `(Q|ρ(t)) / (Q|Q)`, whatever the calculation method. With the default settings the signal starts at 1.
- **SpinDynamica does:** in Hilbert space the COMPUTE method divides by `Norm[Q]^2 Log[2, dimension]`, where
  `Norm` of a matrix is its largest singular value and not the norm of the operator. That product equals
  `(Q|Q)` for the default observable of one, two or three spins-1/2 and for one spin observed out of one or
  two, and differs otherwise: four spins-1/2 start at 4/3, one spin observed out of four at twice the value
  of the "Direct" method, a single spin-1 at 1.26. In Liouville space (with relaxation, or with
  `UseHilbertSpaceWhenPossible -> False`) it divides by `(Q|Q) Log[2, dimension]`: the signal of `N`
  spins-1/2 is smaller by the factor `N`, with and without `CarouselAverage`.
- **Established by:** the "Direct" method of SpinDynamica itself and the first point of the signal; cases
  `15_carousel_relaxation: normalization/*`, and `relaxation/*` and `carousel/fid_in_liouville_space`, which
  are half of the port's values (`test_compute_signals_are_normalised_like_the_other_methods`,
  `test_periodic_hamiltonian_with_relaxation`).
- **Port:** every method divides by `(Q|Q)`; COMPUTE, the step-by-step method and the explicit average over
  the start of the acquisition agree to rounding.
- **Report to the author:** yes.

## D14. `FT` embeds frequency coordinates of two-dimensional data that are divided by both sampling rates

- **Documented:** with `EmbedCoordinates -> True` the output of `FT` has "frequency coordinates instead of
  time coordinates"; the sampling frequency is in Hz.
- **SpinDynamica does:** for two-dimensional data every point comes back as `{f1/(sw1 sw2), f2/(sw1 sw2),
  value}`, with `sw1` and `sw2` the two sampling rates. For sampling intervals of 0.1 s and 0.05 s the first
  coordinates are -0.0125 and -0.0333 where the frequencies are -2.5 Hz and -6.67 Hz. One-dimensional data
  get their frequencies in Hz.
- **Established by:** the frequency axis of the one-dimensional transform and the sum that defines the
  transform; cases `16_two_dimensional: transform/sample_intervals`, `sample_intervals_zero_filled` and
  `triples` (`test_frequency_axes`).
- **Port:** the spectrum is a matrix and `frequency_axis` gives the axis of each dimension in Hz.
- **Report to the author:** yes.

## D15. `CorrelatedEigensystems` with a parameter fails for every input

- **Documented:** `CorrelatedEigensystems[G(x), {x, xa, xb, δx}]`, with `{x, xa, xb, {nx}}` or `{x, {x1, ...}}`,
  for an operator, a superoperator or a matrix that depends on `x`; the usage text gives
  `CorrelatedEigensystems[B opI["z"] + 2π×5 opI[1].opI[2], {B, 0, 1, 0.2}]` as an example.
- **SpinDynamica does:** all these forms return `$Failed` with the message `CorrelatedEigensystems::parAssigned`
  ("The parameter should be an unassigned symbol"), whatever the symbol. The test counts the result of
  `UnassignedSymbols[x]`, and that routine ends with a semicolon after its last expression, so it returns
  `Null` for every argument. Only the form with a list of numeric matrices works, and the adiabatic events,
  which call the routine behind that test.
- **Established by:** cases `18_analysis_adiabatic: correlated/with_parameter_is_rejected` and
  `matrix_with_parameter_is_rejected`; `correlated/internal_eigenvalues` enters behind the test with the tags
  SpinDynamica passes internally and returns the eigenvalues that the port finds
  (`test_correlated_eigensystems`, which also checks every eigenpair against its matrix).
- **Port:** `correlated_eigensystems(generator, parameter, values)` for operators, superoperators and matrices,
  and for a list of matrices.
- **Report to the author:** yes (one character in `General/Miscellaneous.m`).

## Limits found while generating the reference data

- `QM/ExtendedLiouville.m` (site-dependent operators, kinetic exchange) is in the SpinDynamica folder but
  `SpinDynamica.m` does not load it; `Needs["QM`ExtendedLiouville`"]` does. Several operations are unfinished
  there: a power of a `SiteDependentOperator` and its product with an operator on the left give vectors of
  twice the length, the sum of a superoperator and a `SiteDependentSuperoperator` fails
  (`CommutationSuperoperator::badplus`), the sum with a `KineticExchangeSuperoperator` and a multiple of the
  latter are not represented, and `KineticExchangeMatrix` cannot fill in the diagonal for symbolic rate
  constants (`baddims`). The fixtures pin the vectors, the block matrices and the exchange matrices, which work.
- `AbelianSubalgebra` can return operators that are not Hermitian (for `{opI["z"]}` of two spins-1/2 one of
  the three is `|βα><αβ|`), and with a second argument that is not closed under commutation its result need
  not be abelian.
- `SpinPermutationGroup` and `SymmetryAdaptedBasis` take spins that are numbered from 1; other labels abort.
- `SolidAngle[{v1, v2, v3, v4}]`, documented as the sum of two triangles, is defined for Mathematica versions
  before 12 only and stays unevaluated in later ones. The port has it.

- `opT[{1, 2, 3}, {1, 0}]` returns three operators with the message `opT::multiple`, which says to pick one
  with `opT[interactions, {Lam, M}, index]`. For a list of spins that form fails (`opT::fail`) for any index
  above 1; it works with the explicit form `opT[{{1, 1}, {2, 1}, {3, 1}}, {1, 0}, 2]`.
- `DefineOperatorBasis` with an existing basis emits `Set::write` (it appends to an option value that is not a
  symbol). The basis is defined all the same.
- A `Ket` is written `Ket[{{1, 1/2}, {2, -1/2}}]`; `Ket[1/2, -1/2]` and `Ket[{1/2, -1/2}]` run into the
  recursion limit.

- `AngularMomentumBasis` orders its kets in a way that follows no stated rule, and for three spins-1/2 it
  makes a choice within the two `J = 1/2` doublets. The port couples the spins one after the other and
  documents its order; the tests compare the `(J, M)` subspaces.

- `ShapedPulse` takes its frequency argument as an offset of the radiofrequency: the Hamiltonian in the frame
  of the pulse gains `−frequency Iz`. The usage text does not state the sign; the fixture
  `10_sequences: one_half/shaped_pulse_phase_and_frequency` fixes it.
- `PulseSequence` with a total duration needs the spins as its first argument (`PulseSequence[{1}, τ, {...}]`);
  without them it reports `PulseSequence::invalspins`.

- `ThermalEquilibriumDensityOperator` and `ThermalizeSuperoperator` return expressions in the symbols `ℏ` and
  `kB`; the fixtures put in `PhysicalConstantValues`.

- `Signal1D` with `EnsembleAverage` returns a signal of zeros when SpinDynamica runs without the notebook
  front end (messages `FrontEndObject::notavail`, `NDiagonalize::notnummat`, `Trajectory::noprediag`).
  `TransformationAmplitude` with `EnsembleAverage` works. No ensemble-averaged signal is in the fixtures; the
  port's powder average is checked against the analytic sum instead.
- A time-dependent Hamiltonian (`Function[t, ...]` or `PeriodicFunction`) together with a relaxation
  superoperator as `BackgroundGenerator` fails, in `NPropagate` and in `Signal1D` (`NDSolveValue::dsvar`,
  `Superoperator::badsop`). What works is the Liouvillian as the time-dependent generator,
  `PeriodicFunction[t, T, Liouvillian[H, R]]`, with the COMPUTE method; the "Direct" method fails for it
  (`Piecewise::pairs`, `Transpose::nmtx`). The port takes the relaxation as `background` with every method;
  it is checked against a fine piecewise-constant propagation and, for signals, against the events of the
  engine sample by sample.
- `CombineData` with `EmbedCoordinates -> True` refuses data of more than two points
  (`CombineData::embcoordbaddim`: it tests the first dimension of the data where the last is meant), and
  `CombineData` of signals with peaks builds a new signal with default options, so the peak frequencies move
  to the frequency grid even when the signals that went in had `DigitalFrequencyResolution -> False`.
  `CombineComplementarySpectra` is documented with a list of two spectra and defined for two arguments; the
  list form stays unevaluated. `ReflectSpectrum` stays unevaluated for an odd number of points, and for a
  matrix with three columns or fewer, which it takes for a list of coordinates.
- `TransformationAmplitudeBounds` reads `NormalizationFactor` and returns the same bounds whatever its value
  (`18_analysis_adiabatic: bounds/normalization_factor`, the bounds for a factor of 2 are those without it).
- An adiabatic event needs `EventSampling -> {N}`; the documented `EventSampling -> N` aborts the evaluation
  (`NPropagate::wrongevsampling`). Its phases with `DeleteCoherences -> False` come from interpolating
  functions of the eigenvalues, so they agree with the port's spline integral to about 5e-6.
- `DefineBasis` with fewer kets than dimensions warns (`CheckBasis::incomplete`) and defines the basis;
  `MatrixRepresentation` in it then runs into `$RecursionLimit`. No such basis is in the fixtures.
- The sequence of `ShiftAndZOperatorBasis` within one coherence order is that of `SortBy`, which falls back
  on Mathematica's canonical order of the operator expressions. For spins above 1/2 the operators are sums
  with irrational coefficients, and the sequence follows no rule of the spin system
  (`20_shift_and_z_order`).
- SpinDynamica integrates with the default tolerances of `NDSolve`, so its time-dependent results agree with
  the port to about 5e-8, not to rounding.

## Approximations that are defaults in SpinDynamica (not followed)

- `DigitalFrequencyResolution -> Automatic` moves every peak of a signal to the nearest point of the frequency
  grid before the signal is expanded, for speed. For a 17-point signal of an AB system the peaks move by up
  to 3 Hz. The port keeps exact frequencies; the fixtures are generated with
  `DigitalFrequencyResolution -> False`, and the cases `ab_peaks_snapped_to_grid` and
  `fourier/signal_peaks_expanded` record the default.

## Conventions that are uneven inside SpinDynamica (followed by the port)

- `Lorentzian[w, {a, w0, λ}]` is `a/(λ + i(w − w0))`, whose imaginary part is minus `Dispersion`. The usage
  text of `Dispersion` calls `λ` the full width; it is the half width, as for the others.

- `opT` of rank 1 and 2 is defined by formulas (`T(1, 0) = Iz`, `T(2, 0) = (3Iz² − I(I+1))/√6`), whose norm
  depends on the spin. Ranks above 2 come from the Wigner-Eckart theorem with a reduced matrix element chosen
  so that `T(k, 0)` has unit norm. The ranks are therefore normalised differently. The Racah commutators hold
  for every rank, so rotations are unaffected. The port follows SpinDynamica for all ranks.

- The order of the operator bases that are sorted by coherence order comes from `SortBy`, which breaks ties by
  Mathematica's canonical order of expressions. For spins-1/2 that reduces to simple rules, which the port
  follows and which are checked for one, two and three spins. For the shift-and-z basis of spins above 1/2 the
  basis operators are sums, and the port keeps its own sequence within one coherence order.
- `PhenomenologicalRelaxationSuperoperator` counts the factors `Iz` and `I±` in the shift-and-z basis
  operators. For spins above 1/2 those are orthogonalised polynomials, so the count has no clear physical
  meaning; the port offers the model for spins-1/2 only.

## Global state to keep in mind (not errors)

- `SetSpinSystem` resets the Hilbert-space basis to the Zeeman basis but leaves the operator basis alone. A
  basis chosen earlier is in force again whenever the spin system is the same again, even after other spin
  systems were used in between. An operator basis of another spin system is replaced by the default one at the
  next use, with the message `OperatorBasis::inapprop`.
- `opT` of rank above 2 emits `opT::high\[Lambda]` once per session and then switches the message off.
- Loading SpinDynamica adds definitions to `Dot`, `Plus`, `Times`, `Power`, `Exp`, `Expand`, `Simplify` and
  `WignerD`, and sets options of `Plot`, `ListPlot` and `Graphics`.

The port has an explicit spin-system object and no global state, so none of this carries over.
