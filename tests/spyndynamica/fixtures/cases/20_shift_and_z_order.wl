(* The shift-and-z operator basis, sorted by coherence order, for spins above 1/2. Within one coherence order
   SpinDynamica's sequence comes from Mathematica's canonical order of the operator expressions (for a spin 2
   the operators of order zero come as the cubic, the linear, the quadratic one, unity, the quartic one). The
   port has the same operators in every coherence order, in its own sequence. *)
m[op_] := MatrixRepresentation[op];
basisMatrices[opbasis_] := Normal[m[#]] & /@ BasisOperators[opbasis];
sorted := "cases" -> {"shift_and_z" :> basisMatrices[ShiftAndZOperatorBasis[]]};
{
 <|"group" -> "spin2", "setup" :> SetSpinSystem[{{1, 2}}], sorted|>,
 <|"group" -> "one_one", "setup" :> SetSpinSystem[{{1, 1}, {2, 1}}], sorted|>,
 <|"group" -> "three_halves_half", "setup" :> SetSpinSystem[{{1, 3/2}, {2, 1/2}}], sorted|>
}
