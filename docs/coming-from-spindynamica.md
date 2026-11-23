# Coming from SpinDynamica

SpynDynamica (`import spyndynamica`) is a Python port in progress of
[SpinDynamica](https://www.spindynamica.soton.ac.uk) 3.10.2, Malcolm Levitt's Mathematica package. This page
lists the SpinDynamica names with their counterparts, what is the same, what differs, and what is not ported.
The guides are [Symbolic spin operators](spyndynamica.md), [Simulations](spyndynamica-simulations.md) and
[Algebras, symmetry and pictures](spyndynamica-symmetry.md).

The examples on this page use two spins-1/2:

```python
import sympy as sp

from spyndynamica import SpinSystem, commutator

s = SpinSystem(2)
I = s.I
print(commutator(I(1, "x"), I(1, "y")), "|", I(1, "x").rotate(sp.pi / 2, "y", spins=[1]))
```

## Names

| SpinDynamica | SpynDynamica |
|---|---|
| `SetSpinSystem[2]` | `s = SpinSystem(2)`; `I = s.I` |
| `SetSpinSystem[{{1, 1/2}, {2, 1}}]` | `SpinSystem([1/2, 1])` or `SpinSystem({1: 1/2, 2: 1})` |
| `opI[1, "x"]` | `I(1, "x")` |
| `opI[1, "x"].opI[2, "z"]` | `I(1, "x") @ I(2, "z")` |
| `opI[1].opI[2]` | `s.dot(1, 2)` |
| `opT[1, {2, 0}]`, `opT[{1, 2}, {2, 0}]` | `s.T(1, 2, 0)`, `s.T([1, 2], 2, 0)` |
| `opT[{{1, 1}, {2, 1}, {3, 1}}, {1, 1, 1}, {1, 0}]` | `s.T([1, 2, 3], 1, 0, history=(1, 1))` |
| `opT[{{1, 1}, {2, 2}}, {3, 0}]` | `s.T([1, 2], 3, 0, ranks=(1, 2))` |
| `Commutator[A, B]` | `commutator(A, B)` |
| `RotationSuperoperator[{1}, {β, "x"}][ρ]` | `ρ.rotate(β, "x", spins=[1])` |
| `RotationSuperoperator[{α, β, γ}][ρ]` | `ρ.rotate_euler(α, β, γ)` |
| `opR[{β, φ}]` | `s.rotation(β, φ)` |
| `OperatorAmplitude[ρ -> A]` | `ρ.amplitude(A)` |
| `LiouvilleBracket[A, B]` | `A.bracket(B)` |
| `MatrixRepresentation[A]`, `[A, basis]` | `A.matrix()`, `A.matrix(basis)` |
| `N[MatrixRepresentation[A /. rules]]` | `A.numeric(values)` |
| `OperatorTrace`, `OperatorNorm`, `CoherenceOrder` | `A.trace()`, `A.norm()`, `A.coherence_orders()` |
| `ZeemanBasis[]`, `SingletTripletBasis[]` | `s.zeeman`, `s.singlet_triplet()` |
| `BasisTransformationMatrix[b2, b1]` | `b2.transformation(b1)` |
| `AngularMomentumBasis[]`, `Eigenbasis[H, TargetBasis -> b]` | `s.angular_momentum_basis()`, `s.eigenbasis(H, target=b)` |
| `TransformedBasis[op, basis]` | `basis.transformed(op)` |
| `DefineBasis[name, kets, BasisLabels -> labels]`, `CheckBasis[kets]` | `define_basis(kets, labels)`, `check_basis(kets)` |
| `ProductBasis[b1, b2, SpinSystem -> system]` | `product_basis(b1, b2, system=system)` |
| `MagneticQuantumNumber[ket, spins]` | `ket.magnetic_quantum_numbers(spins)` |
| `NormalizeOperator[A, n]`, `OperatorProjection[A, B]` | `A.normalized(n)`, `A.projection(B)` |
| `ShiftSuperoperator[B, A]` | `shift_superoperator(B, A)` |
| `PopulationOperator`, `CoherenceOperator`, `SingleTransitionOperator` | `s.population`, `s.coherence`, `s.single_transition` |
| `SpinPermutationOperator[{1, 2, 3}]` | `s.permutation([1, 2, 3])` |
| `Wignerd`, `WignerD`, `CG`, `EulerRotationMatrix` | `wigner_d`, `wigner_D`, `clebsch_gordan`, `euler_matrix` |
| `AxisSystem[x, z]`, `AxesToEuler[system2, system1]` | `axis_system(x, z)`, `axes_to_euler(system2, system1)` |
| `RotateEuler[v, {α, β, γ}, axes]` | `rotate_euler(v, (α, β, γ), axes)` |
| `PolarRotationMatrix[{ξ, θ, φ}]`, `RotatePolar[v, {ξ, θ, φ}]` | `polar_rotation_matrix(ξ, θ, φ)`, `rotate_polar(v, ξ, (θ, φ))` |
| `PolarToQuaternion`, `QuaternionToPolar`, `QuaternionToEuler` | `polar_to_quaternion`, `quaternion_to_polar`, `quaternion_to_euler` |
| `QuaternionProduct[q2, q1]` | `quaternion_product(q2, q1)` |
| `VectorToPolar[v]`, `PolarToVector[r, {θ, φ}]` | `vector_to_polar(v)`, `polar_to_vector(r, (θ, φ))` |
| `Distance`, `BondAngle`, `SubtendedAngle`, `TorsionAngle`, `SolidAngle` | `distance`, `bond_angle`, `subtended_angle`, `torsion_angle`, `solid_angle` |
| `CommutationSuperoperator[A]`, `Lindbladian[A, B]` | `commutation(A)`, `lindbladian(A, B)` |
| `DoubleCommutationSuperoperator[A, B]` | `double_commutation(A, B)` |
| `CoherenceOrderFiltrationSuperoperator[{0}]` | `coherence_filter(s, [0])` |
| `PhenomenologicalRelaxationSuperoperator[T1, T2]` | `phenomenological_relaxation(s, T1, T2)` |
| `Liouvillian[H, R]` | `liouvillian(H, R)` |
| `SuperoperatorMatrixRepresentation[S, opbasis]` | `S.numeric(basis=opbasis)`, `S.matrix(opbasis)` |
| `OperatorVectorRepresentation[A, opbasis]` | `opbasis.vector(A)` |
| `CartesianProductOperatorBasis[]` and the others | `cartesian_product_basis(s)` and the others |
| `SphericalTensorOperatorBasis[]`, `SphericalTensorIndices[basis]` | `spherical_tensor_basis(s)`, `basis.indices` |
| `SphericalComponentFiltrationSuperoperator[{λ, μ2, μ1}]` | `spherical_component_filter(s, λ, μ2, μ1)` |
| `DefineOperatorBasis[name, operators]`, `CheckOperatorBasis[operators]` | `define_operator_basis(operators)`, `check_operator_basis(operators)` |
| `DefineOperatorBasis[name, opbasis, superoperator, ordering]` | `opbasis.transformed(superoperator, order)`, `opbasis.reordered(order)` |
| `NPropagate[events][ρ]` | `propagate(ρ, events)` |
| `NPropagationOperator[events]` | `propagator(events)` |
| `TransformationAmplitude[ρ -> A, events, BackgroundGenerator -> H]` | `amplitudes(ρ, A, events, background=H)` |
| `Trajectory[ρ -> A, events]` | `trajectory(ρ, A, events)` |
| `{Pulse[{π/2, "x"}], Delay[τ]}`, `Repeat[events, n]` | `[Pulse(pi/2, "x"), Delay(τ)]`, `Repeat(events, n)` |
| `PulseSequence[spins, τ, {.., Delay[Automatic]}, PhaseShift -> φ]` | `PulseSequence([.., Delay("auto")], spins=.., duration=τ, phase_shift=φ)` |
| `Delay[{Automatic, w}]` | `Delay(("auto", w))` |
| `ShapedPulse[spins, τp, {τ, {amplitude, phase, frequency}}]` | `ShapedPulse(τp, amplitude, phase, frequency, tau=τ, spins=spins)` |
| `CoherenceOrderFiltration[{-2, 2}]` | `coherence_filter(s, [-2, 2])` as an event |
| `Precalculate[events, BackgroundGenerator -> H]` | `precalculate(events, background=H)` |
| `PhaseShift`, `AmplitudeFactor`, `NormalizationFactor`, `InitialTimePoint` | `phase_shift`, `amplitude_factor`, `normalization`, `start_time` |
| `$Chronology = "RightToLeft"` | `order="right_to_left"` in the call |
| `Signal1D[{{ωSW, n}}, H, LineBroadening -> w]` | `signal_1d(H, points=n, bandwidth=ωSW, line_broadening=w)` |
| `InitialDensityOperator`, `Preparation`, `Observable` | `initial`, `preparation`, `observable` |
| `SignalCalculationMethod -> "Direct"` | `method="direct"` |
| `Expand[signal]` | `signal.times`, `signal.values` |
| `FT[signal]` | `ft(signal)` |
| `ApplyLineBroadening[signal, LineBroadening -> w]` | `signal.broadened(w)` |
| `Lorentzian`, `Absorption`, `Dispersion`, `Gaussian` | `lorentzian`, `absorption`, `dispersion`, `gaussian` |
| `MagnetogyricRatio`, `NuclearSpinQuantumNumber`, `NaturalAbundance` | `magnetogyric_ratio`, `spin_quantum_number`, `natural_abundance` |
| `LarmorFrequency`, `DirectDipolarCoupling` | `larmor_frequency`, `dipolar_coupling` |
| `{Function[t, H[t]], τ}` | `(TimeDependent(H, t), τ)` |
| `PeriodicFunction[t, T, H]`, `SignalCalculationMethod -> "COMPUTE"` | `Periodic(H, t, T)`, `method="compute"` |
| `EnsembleAverage -> {var, {values..}}` | `ensemble_average(function, values, weights)` |
| `OrientationsAndWeights["ZCW50"]` | `orientations("ZCW50")` |
| `OrientationsAndWeights["Step\[Beta]", n]` | `step_beta(n)` |
| `CartesianToSpherical[2, A]`, `SphericalToCartesian[T]` | `cartesian_to_spherical(2, A)`, `spherical_to_cartesian(T)` |
| `TransformSphericalTensor[T, {α, β, γ}]` | `transform_spherical_tensor(T, (α, β, γ))` |
| `ThermalEquilibriumDensityOperator[H, T]` | `thermal_equilibrium(H, T)` |
| `PolarizedDensityOperator[{1}, {p, "z"}]` | `polarized(s, p, "z", spins=[1])` |
| `SingletPolarizedDensityOperator[p]` | `singlet_polarized(s, p)` |
| `Secularize[A, H0]`, `Secularize[A, {{1}, {2}}]` | `secularize(A, H0)`, `secularize(A, [[1], [2]])` |
| `ThermalizeSuperoperator[R, H, T]` | `thermalize(R, H, T)` |
| `TransformationAmplitudeTable[ρ -> A, events, {p, a, b, {n}}]` | `amplitude_table(ρ, A, events, p, np.linspace(a, b, n))` |
| `MaximizeTransformationAmplitude[ρ -> A, events, {{p, p0}}]` | `maximize_amplitude(ρ, A, events, {p: p0})` |
| `PolarizationLevelOperator[{1, 2}, "z"]` | `polarization_level(s, "z", spins=[1, 2])` |
| `SingletPolarizationLevelOperator[{1, 2}]` | `singlet_polarization_level(s, (1, 2))` |
| `SiteDependentOperator[{{1, A}, {2, B}}]`, `SiteDependentSuperoperator[..]` | `SiteDependent({1: A, 2: B})` |
| `ExtendedOperatorBasis[sites, opbasis]` | `ExtendedOperatorBasis(sites, opbasis)` |
| `ExtendedOperatorVectorRepresentation[x]` | `xbasis.vector(x)` |
| `ExtendedSuperoperatorMatrixRepresentation[x]` | `xbasis.numeric(x)`, `xbasis.matrix(x)` |
| `KineticExchangeMatrix[sites, {{1 -> 2, k}}]` | `kinetic_exchange_matrix(sites, {(1, 2): k})` |
| `KineticExchangeSuperoperator[sites, {{1 -> 2, k}}]` | `KineticExchange(sites, {(1, 2): k})` |
| `Commutant[{Q1, Q2}]`, `Commutant[{{Q1}, {B1, B2, ..}}]` | `commutant([Q1, Q2])`, `commutant([Q1], [B1, B2, ..])` |
| `Bicommutant[{Q1, Q2}]` | `bicommutant([Q1, Q2])` |
| `LieAlgebra[{Q1, Q2}, LieAlgebraField -> "C"]` | `lie_algebra([Q1, Q2], field="C")` |
| `AbelianSubalgebra[{A1}]`, `MaximalAbelianQ[{A1, A2}]` | `abelian_subalgebra([A1])`, `is_maximal_abelian([A1, A2])` |
| `InaccessibleOperatorQ[Qi -> {Qf1, Qf2}, {H1, H2}]` | `inaccessible(Qi, [Qf1, Qf2], [H1, H2])` |
| `SpinPermutationGroup[{"SymmetricGroup", 3}]` | `spin_permutation_group(s, ("SymmetricGroup", 3))` |
| `SymmetryAdaptedBasis[{"CyclicGroup", 3}]` | `symmetry_adapted_basis(s, ("CyclicGroup", 3))` |
| `DualPairing[3, 1/2]` | `dual_pairing(3, 1/2)` |
| `Ovaloid[r, tensor, OvaloidScale -> c]` | `ovaloid(tensor, r, scale=c)` |
| `Axes3D[axes, origin]`, `Arrow3D[begin, end]` | `axes_3d(axes, origin)`, `arrow_3d(begin, end)` |
| `VisualizeEuler[{α, β, γ}]` | `visualize_euler((α, β, γ))` |
| `LShape[]`, `Rotor[]`, `SphereAndAxes[]` | `l_shape()`, `rotor()`, `sphere_and_axes()` |
| `Show[{g1, g2}]`, `RotateEuler[g, {α, β, γ}]` | `g1 + g2`, `g.rotated_euler((α, β, γ))` |

## The same and different

**The same:** the order of the Zeeman and singlet-triplet bases, the phases and normalisation of the tensor
operators, the sense of every rotation, the zyz Euler convention and the index order of the Wigner matrices.

**Different:**

- There is no global state. The spin system is an object, and a basis is an argument where one is needed.
- An operator is always in expanded form. SpinDynamica often returns an unevaluated product, which
  `ExpressOperator` or `MatrixRepresentation` then resolves.
- SpinDynamica normalises tensor operators of rank 3 and above differently from ranks 1 and 2; the port
  follows it.
- Closed-form evolution exists for commuting Hamiltonian terms of spins-1/2 only; everything else goes through
  the numeric engine.
- The operator bases hold the same operators in the same order, with one exception: for spins above 1/2 the
  shift-and-z basis is sorted by coherence order as in SpinDynamica, but the sequence within one order is the
  port's own (SpinDynamica's comes from Mathematica's sorting of expressions).
- A basis is an object, not a registered name: `define_basis` and `define_operator_basis` return it. A state
  basis always has as many kets as the Hilbert space has dimensions.
- `s.T` asks for `history` when several sequences of intermediate ranks lead to the rank; SpinDynamica returns
  the list of all the operators.
- `spherical_tensor_basis(s, sort="rank")` sorts by rank first. SpinDynamica documents that order
  (`Sorted -> {SphericalRank, CoherenceOrder}`) but rejects the value, and every value it accepts sorts by
  coherence order first.
- `spherical_component_filter` exists for any spins (SpinDynamica: spins-1/2), and it pairs the two components
  within each tensor. From three spins on SpinDynamica pairs components of different tensors when the two
  components differ, so its result is not the filter it documents.
- Orderings of an operator basis count from 0. A basis that is transformed and reordered is consistent with its
  operators; in SpinDynamica the vectors and matrices in a basis defined with an ordering belong to the inverse
  ordering.
- `rotate_euler(v, angles, axes)` rotates about the axes of `axes` as documented. SpinDynamica applies the
  inverse arrangement of the frame change, and the example in its own usage text (turning one axis system onto
  another) does not hold there.
- The geometry routines take numbers. SpinDynamica's `AxesToEuler` also takes symbolic angles, which it assumes
  to lie in the first quadrant.
- `singlet_polarization_level` adds up pairs that do not share spins. SpinDynamica refuses exactly those and
  accepts pairs that share a spin.
- The extended Liouville space (`SiteDependent`, `ExtendedOperatorBasis`, `KineticExchange`) is a module that
  SpinDynamica ships but does not load. There one object type holds operators and another superoperators, the
  sites are a global setting, and sums of site-dependent superoperators with plain superoperators or with the
  exchange are not represented; here the sites belong to the objects and such sums have a matrix. Rate
  constants may be symbols when the diagonal of the exchange matrix is filled in.
- The algebra routines return operators with exact coefficients and turn floating-point coefficients into
  rational numbers. A commutant, a Lie algebra or an abelian subalgebra is one basis of a space among many, and
  the port's basis is not SpinDynamica's. `abelian_subalgebra` returns Hermitian operators, as SpinDynamica
  documents; SpinDynamica's own result can contain operators that are not Hermitian.
- `symmetry_adapted_basis` is numeric and deterministic, and takes any permutation group of equal spins.
  SpinDynamica draws random numbers: its basis differs from session to session, and for some draws it fails or
  returns kets that are no basis. The kets are ordered by `J` descending; SpinDynamica orders them by `J`
  ascending for the symmetric group and descending for the cyclic group.
- `dual_pairing` returns a dictionary, and pictures are `Scene` objects of plain arrays that Plotly draws;
  SpinDynamica returns graphics. Colours, fonts and the placing of labels are the port's own.
- The order of events is an argument of each call. In SpinDynamica the option `Chronology` has no effect and
  only the global `$Chronology` counts.
- `trajectory` returns sampled values, not interpolating functions.
- Large systems are handled by block diagonalisation of sparse Hamiltonians, chosen automatically.
- The phenomenological relaxation model is for spins-1/2.
- `signal_1d` returns the number of points asked for. SpinDynamica drops the last point of an even request.
- Peak frequencies are exact. SpinDynamica by default moves every peak to the nearest point of its frequency
  grid (`DigitalFrequencyResolution`), which changes a short signal visibly.
- An ensemble average is a function call around any calculation, not an option of each routine. A
  time-dependent Hamiltonian is integrated with relaxation in the background as well, which fails in
  SpinDynamica.
- `polarized(s, p, "z")` polarizes all spins to the level `p`, as SpinDynamica documents. SpinDynamica's
  short form `PolarizedDensityOperator[{p, "z"}]` returns the fully polarized state whatever `p` and the axis.
- The coupled basis is ordered by `J`, the intermediate angular momenta and `M`, with Condon-Shortley
  phases. SpinDynamica's `AngularMomentumBasis` spans the same `(J, M)` subspaces in another order.
  `A.projection(B)` exists here; SpinDynamica documents `OperatorProjection` but does not implement it.
- `ft` of an odd number of points without zero filling returns the transform at the stated frequencies;
  SpinDynamica's values sit one point lower on its axis in that case.

## Not ported yet

**Not ported yet:** signals under time-dependent Hamiltonians with relaxation, the carousel average
(`CarouselAverage`, γ-COMPUTE), two-dimensional signals, relaxation theory beyond the phenomenological model,
and symbolic axes in `axes_to_euler`.
`porting/spyndynamica/PLAN.md` has the phases, and `porting/spyndynamica/DIVERGENCES.md` lists every case found
in which SpinDynamica does something else than it documents.
