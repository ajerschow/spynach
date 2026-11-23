(* The remaining Hilbert-space bases and operator utilities. OperatorProjection and
   ProjectedMatrixRepresentation have usage messages but no definitions in SpinDynamica 3.10.2, so they have
   no cases (DIVERGENCES.md, D6). *)
m[op_] := MatrixRepresentation[op];
m[op_, basis_] := MatrixRepresentation[op, basis];
kets[basis_] := Normal[VectorRepresentation[#, ZeemanBasis[]]] & /@ BasisKets[basis];
ham := 2 Pi 30 opI[1, "z"] - 2 Pi 10 opI[2, "z"] + 2 Pi 12 opI[1] . opI[2];
mixed := opI[1, "x"] . opI[2, "x"] + 2 opI[1, "z"] . opI[2, "z"] + 3 opI[1, "x"] + opI[1, "+"] . opI[2, "-"] + 5 opI[2, "z"];
coupled[name_, system_] := <|"group" -> name, "setup" :> SetSpinSystem[system], "cases" -> {
     "angular_momentum_kets" :> kets[AngularMomentumBasis[]],
     "total_spin_squared" :> m[opI[] . opI[], AngularMomentumBasis[]],
     "total_Iz" :> m[opI["z"], AngularMomentumBasis[]]
     }|>;
{
 coupled["two_half", 2],
 coupled["three_half", 3],
 coupled["half_one", {{1, 1/2}, {2, 1}}],
 coupled["one_one", {{1, 1}, {2, 1}}],
 <|"group" -> "two_half_bases", "setup" :> SetSpinSystem[2], "cases" -> {
    "eigenbasis_kets" :> kets[Eigenbasis[ham]],
    "hamiltonian_in_eigenbasis" :> m[ham, Eigenbasis[ham]],
    "eigenbasis_target_singlet_triplet" :> kets[Eigenbasis[opI[1] . opI[2], TargetBasis -> SingletTripletBasis[]]],
    "transformed_basis_kets" :> kets[TransformedBasis[opR[{1}, {Pi/2, "y"}], ZeemanBasis[]]],
    "I1z_in_transformed_basis" :> m[opI[1, "z"], TransformedBasis[opR[{1}, {Pi/2, "y"}], ZeemanBasis[]]],
    "magnetic_quantum_numbers" :> (MagneticQuantumNumber /@ BasisKets[ZeemanBasis[]]),
    "magnetic_quantum_numbers_spin_1" :> (MagneticQuantumNumber[#, {1}] & /@ BasisKets[ZeemanBasis[]]),
    "normalized" :> m[NormalizeOperator[mixed]],
    "normalized_to_3" :> m[NormalizeOperator[mixed, 3]],
    "normalized_to_operator" :> m[NormalizeOperator[mixed, opI[1, "x"]]],
    "shift_superoperator" :> SuperoperatorMatrixRepresentation[ShiftSuperoperator[opI[1, "x"], opI[2, "z"]]],
    "shift_superoperator_applied" :> m[ShiftSuperoperator[opI[1, "x"], opI[2, "z"]][mixed]]
    }|>
 }
