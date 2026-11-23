# Symbolic spin operators with SpynDynamica

SpynDynamica is a Python port in progress of [SpinDynamica](https://www.spindynamica.soton.ac.uk), Malcolm
Levitt's Mathematica package. It is a separate import (`import spyndynamica`) that uses NumPy and SymPy and
nothing of Spynach. This guide covers the symbolic core and Liouville space: operators are sums of product
operators and spherical tensor operators with symbolic coefficients, and products, commutators, rotations and
weak-coupling evolution are exact algebra; operator bases and superoperators come on top, and axis systems,
Euler angles and quaternions describe frames. The numeric engine (pulse sequences with strong coupling and
relaxation, signals and spectra, time-dependent Hamiltonians, powder averages) is in
[Simulations with SpynDynamica](spyndynamica-simulations.md). The table of SpinDynamica names for both guides
is [at the end of this one](#coming-from-spindynamica).

SpinDynamica is the work of Malcolm H. Levitt and co-authors. Its [terms of use](https://www.spindynamica.soton.ac.uk/about/termsofuse/)
state that it is open-source and freely available for use, and ask that published work cite
C. Bengs, M. H. Levitt, *SpinDynamica: Symbolic and numerical magnetic resonance in a Mathematica environment*, Magn. Reson. Chem. (2017), doi:10.1002/mrc.4642. Work that uses SpynDynamica should cite that publication.

Every convention below is the one of SpinDynamica 3.10.2 and is checked against it
(`tests/spyndynamica/`, 752 reference cases generated from the Mathematica package).

## Contents

1. [Spin systems and operators](#spin-systems-and-operators)
2. [Rotations and pulses](#rotations-and-pulses)
3. [A product-operator calculation](#a-product-operator-calculation)
4. [Matrices and numbers](#matrices-and-numbers)
5. [Bases, kets and bras](#bases-kets-and-bras)
6. [Spins above 1/2 and tensor operators](#spins-above-12-and-tensor-operators)
7. [Superoperators and operator bases](#superoperators-and-operator-bases)
8. [Axis systems, Euler angles and quaternions](#axis-systems-euler-angles-and-quaternions)
9. [Coming from SpinDynamica](#coming-from-spindynamica)

## Spin systems and operators

A spin system is an object, and several can exist side by side. `SpinSystem(2)` is two spins-1/2 with the
labels 1 and 2. Operators come from the system, `@` is the operator product and `*` multiplies by a scalar.

```python
import sympy as sp

from spyndynamica import SpinSystem, commutator

s = SpinSystem(2)
I = s.I

print(commutator(I(1, "x"), I(1, "y")))
print(I(1, "x") @ I(2, "z") + 3 * I(1, "y"))
print(I(1, "+"), "|", I(None, "z"), "|", s.dot(1, 2))
```

`I(spin, component)` takes `'x'`, `'y'`, `'z'`, `'+'`, `'-'`, `'alpha'` and `'beta'`, an angle (the component in
the xy plane at that phase) or polar angles `(θ, φ)`. A list of spins, or `None` for all of them, gives the sum.
`s.dot(1, 2)` is the scalar product `I1·I2`. In a notebook an operator is typeset.

An operator knows its trace, norm, coherence orders, and how much of another operator it contains:

```python
rho = 2 * I(1, "x") @ I(2, "z") + 3 * I(1, "y")
print(rho.amplitude(I(1, "x") @ I(2, "z")), rho.amplitude(I(1, "y")))
print((I(1, "+") @ I(2, "+")).coherence_orders(), I(1, "x").coherence_orders())
print(I(1, "x").norm(), s.unity().trace())
```

`a.amplitude(b)` is `(b|a)/(b|b)` with the Liouville bracket `(a|b) = Tr(a† b)`, SpinDynamica's
`OperatorAmplitude[a -> b]`.

## Rotations and pulses

`rotate(angle, phase)` returns `R A R†` for `R = exp(−i angle n·I)`. The axis lies in the xy plane at `phase`
(`'x'`, `'y'`, `'-x'`, `'-y'` or an angle), is `'z'`, or has the polar angles `axis=(θ, φ)`. `spins` restricts
the rotation; the default is all spins.

```python
beta, phi = sp.symbols("beta phi", real=True)

print(I(1, "z").rotate(sp.pi / 2, "x"))
print(I(1, "z").rotate(beta, phi, spins=[1]))
print((I(1, "z") @ I(2, "z")).rotate(sp.pi / 2, "y", spins=[1]))
print(I(1, "x").rotate_euler(sp.pi / 2, sp.pi / 2, 0))
```

`rotate_euler(α, β, γ)` is the zyz rotation `Rz(α) Ry(β) Rz(γ)`. The rotation operator itself is
`s.rotation(angle, phase)`:

```python
print(s.rotation(beta, "x", spins=[1]))
```

## A product-operator calculation

`evolve(H, t)` is `exp(−iHt) ρ exp(+iHt)` in closed form when the terms of `H` commute, which covers chemical
shifts and weak couplings of spins-1/2. A 90° pulse on spin 1, then evolution:

```python
omega, J, t, tau = sp.symbols("Omega1 J t tau", real=True)
H = omega * I(1, "z") + 2 * sp.pi * J * (I(1, "z") @ I(2, "z"))

rho = I(1, "z").rotate(sp.pi / 2, "x", spins=[1]).evolve(H, t)
print(rho)
print(rho.amplitude(2 * I(1, "x") @ I(2, "z")))
```

A spin echo refocuses the shift and keeps the coupling, and INEPT turns the antiphase term into magnetisation
of the second spin:

```python
start = I(1, "z").rotate(sp.pi / 2, "x", spins=[1])
echo = start.evolve(H, tau).rotate(sp.pi, "y").evolve(H, tau).simplify()
print(echo)

inept = echo.rotate(sp.pi / 2, "y", spins=[1]).rotate(sp.pi / 2, "x", spins=[2]).simplify()
print(inept.amplitude(2 * I(1, "z") @ I(2, "y")))
```

A Hamiltonian whose terms do not commute (strong coupling) raises `NotImplementedError`: that case belongs to
the numeric engine ([Simulations with SpynDynamica](spyndynamica-simulations.md)).

## Matrices and numbers

`matrix()` is the exact SymPy matrix and `numeric(values)` a NumPy array with numbers put in for the symbols.
In the Zeeman basis every spin runs from `m = +I` down and the first spin changes fastest, so two spins-1/2
are ordered αα, βα, αβ, ββ.

```python
print(I(1, "x").matrix())
print(H.numeric({omega: 2 * sp.pi * 100, J: 7}).real.diagonal().round(2))
print(s.from_matrix(I(1, "x").matrix() * I(2, "z").matrix()))
```

`s.from_matrix` goes back from a matrix to the operator.

## Bases, kets and bras

`s.zeeman` and `s.singlet_triplet()` are bases; the second is for two spins-1/2, in SpinDynamica's order
`S0, T+1, T0, T-1`. A matrix can be taken in either, and states are numbered from 1 or named.

```python
st = s.singlet_triplet()
print(st.ket("S0"))
print(s.dot(1, 2).matrix(st))
print(st.transformation(s.zeeman))

up, down = sp.Rational(1, 2), -sp.Rational(1, 2)
a = s.ket(up, up)
print(I(1, "-") @ a, "|", a.dagger() @ a, "|", a @ s.ket(down, up).dagger())
```

Three more bases come from the system or from another basis: `s.angular_momentum_basis()` couples the spins to
eigenkets of the total angular momentum, `s.eigenbasis(H)` holds the eigenkets of an operator, and
`basis.transformed(op)` applies an operator to the kets of a basis.

```python
coupled = s.angular_momentum_basis()
print(coupled.quantum_numbers)
print(s.dot(1, 2).matrix(coupled).diagonal())
print(st.ket("T0").magnetic_quantum_numbers(), I(1, "x").normalized().norm())
```

A basis of your own comes from `define_basis(kets, labels)`, which requires the kets to be orthonormal
(`check_basis(kets)` says what a set of kets is). `product_basis` combines bases of separate sets of spins: the
kets of the first basis change fastest, and `system` is the spin system of the result.

```python
from spyndynamica import define_basis, product_basis

k = s.ket
triplets_first = [k(up, up), (k(up, down) + k(down, up)) / sp.sqrt(2), k(down, down), (k(up, down) - k(down, up)) / sp.sqrt(2)]
mine = define_basis(triplets_first, labels=["T+", "T0", "T-", "S"])
print(s.dot(1, 2).matrix(mine).diagonal(), "|", mine.ket("S"))

three = SpinSystem(3)
first_two, third = SpinSystem([(1, up), (2, up)]), SpinSystem([(3, up)])
st_and_zeeman = product_basis(first_two.singlet_triplet(), third.zeeman, system=three)
print(three.dot(1, 2).matrix(st_and_zeeman).diagonal())
```

`ket @ bra` is an operator, `bra @ ket` a number and `operator @ ket` a ket. The operators of basis states are
`s.population(r)`, `s.coherence(r, s)` and `s.single_transition(r, s, 'x')`; `s.permutation([1, 2])` exchanges
spins.

## Spins above 1/2 and tensor operators

Any spin quantum numbers can be mixed. A spin above 1/2 is written in spherical tensor operators `T(k, q)`,
which is also how its operators print (`T1(2,0)` is rank 2, order 0, of spin 1).

```python
d = SpinSystem([1])
print(d.I(1, "z") @ d.I(1, "z"))
print(d.T(1, 2, 0).rotate(beta, "y"))

pair = SpinSystem({"H": 0.5, "D": 1})
print(pair.dimension, pair.I("H", "z") @ pair.I("D", "+"))
print(s.T([1, 2], 2, 0), "|", s.T([1, 2], 0, 0))
```

`s.T(spin, k, q)` is SpinDynamica's `opT`: `T(1, ±1) = ∓I±/√2`, `T(2, 0) = (3Iz² − I(I+1))/√6`. With a pair of
spins it couples their rank-1 operators, so `T(0, 0) = −I1·I2/√3`.

More spins, or single-spin operators of other ranks (`ranks`), are coupled one after the other from left to
right with Clebsch-Gordan coefficients. When several sequences of intermediate ranks lead to the same total
rank, `history` names the rank after each coupling step; `rank_histories` lists the sequences.

```python
from spyndynamica import rank_histories

print(rank_histories([1, 1, 1]))
print(three.T([1, 2, 3], 3, 3), "|", three.T([1, 2, 3], 0, 0, history=(1, 0)))
print(pair.T(["H", "D"], 3, 3, ranks=(1, 2)))
```

The Wigner matrices and Clebsch-Gordan coefficients are available in the same conventions:

```python
from spyndynamica import clebsch_gordan, wigner_d, wigner_D

print(wigner_d(1, beta))
print(clebsch_gordan(1, 1, 1, -1, 2, 0), wigner_D(1, 0, sp.pi / 2, 0)[0, 2])
```

## Superoperators and operator bases

A superoperator maps operators to operators. It is built from operators, acts exactly, and composes with `@`.

```python
from spyndynamica import commutation, default_operator_basis, lindbladian, phenomenological_relaxation

print(commutation(I(1, "z"))(I(1, "x")))
print(lindbladian(I(1, "-"))(s.unity()))

T1, T2 = sp.symbols("T1 T2", positive=True)
relaxation = phenomenological_relaxation(s, T1, T2)
print(relaxation(I(1, "z")), "|", relaxation(I(1, "+") @ I(2, "z")))
```

Besides `commutation` there are `left_multiplication`, `right_multiplication`, `double_commutation`,
`projection`, `coherence_filter`, `rotation_superoperator` and `permutation_superoperator`, and
`liouvillian(H, relaxation)` is `−i[H, ·]` plus the relaxation. `phenomenological_relaxation` is the
uncorrelated model for spins-1/2: every `Iz` in a product adds `1/T1` and every `I±` adds `1/T2`.

A superoperator has a matrix in an operator basis, `S_ij = (B_i | S B_j)`. The default basis is SpinDynamica's
`ShiftAndZOperatorBasis` sorted by coherence order.

```python
basis = default_operator_basis(s)
print([str(b) for b in basis.operators[:4]])
print(commutation(I(1, "z")).numeric(basis=basis).real.diagonal())
print(basis.vector(I(1, "x")).T)
```

The other bases are `cartesian_product_basis`, `shift_and_polarization_basis`, `zeeman_ket_bra_basis`,
`singlet_triplet_operator_basis` and `spherical_tensor_basis`.

`spherical_tensor_basis` holds the normalised irreducible tensor operators of any spin system, coupled as
`s.T` couples them; `basis.indices` lists their `(rank, order)`. `spherical_component_filter(s, rank, to,
source)` is built on them. It takes the component `source` of every tensor of that rank to its component `to`
and removes everything else; with one component it is the projector onto that rank and order.

```python
from spyndynamica import spherical_component_filter, spherical_tensor_basis

print(spherical_tensor_basis(s).indices)
mixture = I(1, "z") + 2 * I(1, "z") @ I(2, "z") + I(1, "x") @ I(2, "y")
print(spherical_component_filter(s, 1, 0)(mixture))
print(spherical_component_filter(s, 2, 0)(mixture), "|", spherical_component_filter(s, 1, 1, 0)(I(1, "z")))
```

An operator basis of your own is `define_operator_basis(operators)`, which requires the operators to be
orthonormal (`check_operator_basis` says what a set is). An existing basis is reordered with
`basis.reordered(order)`, where positions count from 0, and transformed with `basis.transformed(superoperator)`,
for which the superoperator must be unitary:

```python
from spyndynamica import cartesian_product_basis, rotation_superoperator

cartesian = cartesian_product_basis(s)
turned = cartesian.transformed(rotation_superoperator(s, sp.pi / 2, "y"))
print([str(b) for b in turned.operators[1:4]], turned.check())
print(turned.vector(I(1, "x")).T)
```

## Axis systems, Euler angles and quaternions

These routines work with numbers. `axis_system` builds a right-handed orthonormal set of axes, as the rows of
an array, from a z axis, from an x and a z axis (the x axis of the result lies in their plane) or from all
three. `axes_to_euler(system2, system1)` gives the Euler angles of the rotation `Rz(α) Ry(β) Rz(γ)` that
brings every axis of the first argument onto the corresponding axis of the second (default: the laboratory
axes). With the principal axis system of an interaction and a reference frame these are the angles that go into
the spherical tensor form of a Hamiltonian.

```python
import numpy as np

from spyndynamica import axes_to_euler, axis_system, rotate_euler

principal = axis_system([1, 2, 0], [1, 1, 1])
molecular = axis_system([0, 1, 1], [-1, 2, 0.5])
omega = axes_to_euler(principal, molecular)
print(np.round(omega, 4))
print(np.allclose(rotate_euler(principal, omega, principal), molecular))
```

`rotate_euler(vectors, angles)` rotates about the laboratory axes; with a third argument it rotates about the
axes of that system, by `γ` about its z axis, then by `β` about its y axis, then by `α` about its z axis.
`rotate_polar(vectors, ξ, direction)` rotates through `ξ` about an axis given by its polar angles `(θ, φ)` or
as a vector, and `polar_rotation_matrix(ξ, θ, φ)` is the matrix, which may be symbolic. A rotation is also a
unit quaternion `(cos ξ/2, n sin ξ/2)`:

```python
from spyndynamica import polar_rotation_matrix, polar_to_quaternion, quaternion_product, quaternion_to_euler
from spyndynamica import quaternion_to_polar, rotate_polar

print(rotate_polar([1, 0, 0], np.pi / 2, [0, 0, 1]).round(3))
print(polar_rotation_matrix(sp.Symbol("xi"), 0, 0))
quarter_z = polar_to_quaternion(np.pi / 2, 0, 0)
quarter_x = polar_to_quaternion(np.pi / 2, np.pi / 2, 0)
both = quaternion_product(quarter_x, quarter_z)
print(both.round(4), np.round(quaternion_to_polar(both), 4), np.round(quaternion_to_euler(both), 4))
```

`quaternion_product(q2, q1)` belongs to the rotation `q1` followed by `q2`. `vector_to_polar` and
`polar_to_vector` convert between a vector and its length and polar angles.

## Coming from SpinDynamica

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

**Not ported yet:** signals under time-dependent Hamiltonians with relaxation, the carousel average
(`CarouselAverage`, γ-COMPUTE), two-dimensional signals, relaxation theory beyond the phenomenological model,
the group theory modules and graphics.
`porting/spyndynamica/PLAN.md` has the phases, and `porting/spyndynamica/DIVERGENCES.md` lists every case found
in which SpinDynamica does something else than it documents.
