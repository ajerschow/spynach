# Symbolic spin operators with SpynDynamica

SpynDynamica is a Python port in progress of [SpinDynamica](https://www.spindynamica.soton.ac.uk), Malcolm
Levitt's Mathematica package. It is a separate import (`import spyndynamica`) that uses NumPy and SymPy and
nothing of Spynach. This guide covers what exists so far. In the symbolic core, operators are sums of product
operators and spherical tensor operators with symbolic coefficients, and products, commutators, rotations and
weak-coupling evolution are exact algebra. Liouville space adds operator bases and superoperators, and a
numeric engine propagates sequences of pulses and delays with strong coupling and relaxation. Signals,
time-dependent Hamiltonians and ensemble averages are not ported yet.

SpinDynamica is the work of Malcolm H. Levitt and co-authors. Its [terms of use](https://www.spindynamica.soton.ac.uk/about/termsofuse/)
state that it is open-source and freely available for use, and ask that published work cite
C. Bengs, M. H. Levitt, *SpinDynamica: Symbolic and numerical magnetic resonance in a Mathematica environment*, Magn. Reson. Chem. (2017), doi:10.1002/mrc.4642. Work that uses SpynDynamica should cite that publication.

Every convention below is the one of SpinDynamica 3.10.2 and is checked against it
(`tests/spyndynamica/`, 466 reference cases generated from the Mathematica package).

## Contents

1. [Spin systems and operators](#spin-systems-and-operators)
2. [Rotations and pulses](#rotations-and-pulses)
3. [A product-operator calculation](#a-product-operator-calculation)
4. [Matrices and numbers](#matrices-and-numbers)
5. [Bases, kets and bras](#bases-kets-and-bras)
6. [Spins above 1/2 and tensor operators](#spins-above-12-and-tensor-operators)
7. [Superoperators and operator bases](#superoperators-and-operator-bases)
8. [Pulse sequences, strong coupling and relaxation](#pulse-sequences-strong-coupling-and-relaxation)
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
the numeric engine, which is not ported yet.

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
spins it couples their rank-1 operators, so `T(0, 0) = −I1·I2/√3`. The Wigner matrices and Clebsch-Gordan
coefficients are available in the same conventions:

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

## Pulse sequences, strong coupling and relaxation

The numeric engine takes a list of events. `(H, t)` is evolution under a Hamiltonian for a time,
`Pulse(flip, phase)` an ideal pulse, `Delay(t)` an interval under the background, `Repeat(events, n)` a loop,
and a superoperator acts instantaneously. `background` acts during every event with a duration. Strong coupling
needs no special handling here.

```python
import numpy as np

from spyndynamica import Delay, Pulse, amplitudes, propagate, trajectory

strong = 2 * sp.pi * 100 * I(1, "z") - 2 * sp.pi * 40 * I(2, "z") + 2 * sp.pi * 7 * s.dot(1, 2)
echo = [Pulse(sp.pi / 2, "x"), Delay(0.004), Pulse(sp.pi, "y"), Delay(0.004)]

print(amplitudes(I(1, "z"), [I(1, "y"), 2 * I(1, "x") @ I(2, "z")], echo, background=strong).round(4))
print(propagate(I(1, "z"), echo, background=strong).round(3)[0])
```

`amplitudes` is SpinDynamica's `TransformationAmplitude`, `propagate` its `NPropagate` (the result is the
density matrix as a NumPy array), and `propagator` gives the unitary. Symbols in the events get numbers
through `values`, and a relaxation superoperator in the background switches the engine to Liouville space:

```python
values = {T1: 2.0, T2: 0.2}
times, decay = trajectory(I(1, "x"), I(1, "x"), [Delay(0.3)], points=3, background=[strong, relaxation], values=values)
print(times.round(2), np.abs(decay).round(4))
```

`trajectory` samples the amplitudes during the sequence. A pulse with `nutation_frequency` or `duration` has a
finite length, and the background acts during it. Events run from left to right; `order="right_to_left"`
reverses them for one call.

## Coming from SpinDynamica

| SpinDynamica | SpynDynamica |
|---|---|
| `SetSpinSystem[2]` | `s = SpinSystem(2)`; `I = s.I` |
| `SetSpinSystem[{{1, 1/2}, {2, 1}}]` | `SpinSystem([1/2, 1])` or `SpinSystem({1: 1/2, 2: 1})` |
| `opI[1, "x"]` | `I(1, "x")` |
| `opI[1, "x"].opI[2, "z"]` | `I(1, "x") @ I(2, "z")` |
| `opI[1].opI[2]` | `s.dot(1, 2)` |
| `opT[1, {2, 0}]`, `opT[{1, 2}, {2, 0}]` | `s.T(1, 2, 0)`, `s.T([1, 2], 2, 0)` |
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
| `PopulationOperator`, `CoherenceOperator`, `SingleTransitionOperator` | `s.population`, `s.coherence`, `s.single_transition` |
| `SpinPermutationOperator[{1, 2, 3}]` | `s.permutation([1, 2, 3])` |
| `Wignerd`, `WignerD`, `CG`, `EulerRotationMatrix` | `wigner_d`, `wigner_D`, `clebsch_gordan`, `euler_matrix` |
| `CommutationSuperoperator[A]`, `Lindbladian[A, B]` | `commutation(A)`, `lindbladian(A, B)` |
| `DoubleCommutationSuperoperator[A, B]` | `double_commutation(A, B)` |
| `CoherenceOrderFiltrationSuperoperator[{0}]` | `coherence_filter(s, [0])` |
| `PhenomenologicalRelaxationSuperoperator[T1, T2]` | `phenomenological_relaxation(s, T1, T2)` |
| `Liouvillian[H, R]` | `liouvillian(H, R)` |
| `SuperoperatorMatrixRepresentation[S, opbasis]` | `S.numeric(basis=opbasis)`, `S.matrix(opbasis)` |
| `OperatorVectorRepresentation[A, opbasis]` | `opbasis.vector(A)` |
| `CartesianProductOperatorBasis[]` and the others | `cartesian_product_basis(s)` and the others |
| `NPropagate[events][ρ]` | `propagate(ρ, events)` |
| `NPropagationOperator[events]` | `propagator(events)` |
| `TransformationAmplitude[ρ -> A, events, BackgroundGenerator -> H]` | `amplitudes(ρ, A, events, background=H)` |
| `Trajectory[ρ -> A, events]` | `trajectory(ρ, A, events)` |
| `{Pulse[{π/2, "x"}], Delay[τ]}`, `Repeat[events, n]` | `[Pulse(pi/2, "x"), Delay(τ)]`, `Repeat(events, n)` |
| `$Chronology = "RightToLeft"` | `order="right_to_left"` in the call |

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
  port's own (SpinDynamica's comes from Mathematica's sorting of expressions). The spherical tensor basis is
  available for one or two spins.
- The order of events is an argument of each call. In SpinDynamica the option `Chronology` has no effect and
  only the global `$Chronology` counts.
- `trajectory` returns sampled values, not interpolating functions.
- The phenomenological relaxation model is for spins-1/2.

**Not ported yet:** time-dependent Hamiltonians and shaped pulses, `Signal1D` and Fourier transforms,
relaxation theory beyond the phenomenological model, ensemble averages, the group theory modules and graphics.
`porting/spyndynamica/PLAN.md` has the phases, and `porting/spyndynamica/DIVERGENCES.md` lists two documented
options that SpinDynamica itself ignores.
