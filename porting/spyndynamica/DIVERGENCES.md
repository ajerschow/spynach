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

## Limits found while generating the reference data

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
- A time-dependent Hamiltonian (`Function[t, ...]`) together with a relaxation superoperator as
  `BackgroundGenerator` fails (`NDSolveValue::dsvar`, `Superoperator::badsop`). The port integrates that case
  in Liouville space; it is checked against a fine piecewise-constant propagation.
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
