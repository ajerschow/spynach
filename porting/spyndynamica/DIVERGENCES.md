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

## Conventions that are uneven inside SpinDynamica (followed by the port)

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
