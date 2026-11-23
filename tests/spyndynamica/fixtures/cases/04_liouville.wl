(* Liouville space: operator bases (order and normalisation of the basis operators), brackets, amplitudes
   and the matrix representations of superoperators. *)
m[op_] := MatrixRepresentation[op];
basisMatrices[opbasis_] := Normal[m[#]] & /@ BasisOperators[opbasis];
(* the operator basis is the second argument; the documented form OperatorBasis -> opbasis is ignored
   (DIVERGENCES.md, D1), which the case option_form_is_ignored records *)
vector[op_, opbasis_] := OperatorVectorRepresentation[op, opbasis];
s[sop_] := SuperoperatorMatrixRepresentation[sop];
sIn[sop_, opbasis_] := SuperoperatorMatrixRepresentation[sop, opbasis];
ham := 2 Pi 100 opI[1, "z"] - 2 Pi 40 opI[2, "z"] + 2 Pi 7 opI[1] . opI[2];
superoperators = {
   "unity" :> s[UnitySuperoperator[]],
   "commutation_I1z" :> s[CommutationSuperoperator[opI[1, "z"]]],
   "commutation_I1x" :> s[CommutationSuperoperator[opI[1, "x"]]],
   "commutation_hamiltonian" :> s[CommutationSuperoperator[ham]],
   "left_I1x" :> s[LeftMultiplicationSuperoperator[opI[1, "x"]]],
   "right_I1x" :> s[RightMultiplicationSuperoperator[opI[1, "x"]]],
   "left_I1plus" :> s[LeftMultiplicationSuperoperator[opI[1, "+"]]],
   "right_I1plus" :> s[RightMultiplicationSuperoperator[opI[1, "+"]]],
   "double_commutation_I1z_I1z" :> s[DoubleCommutationSuperoperator[opI[1, "z"], opI[1, "z"]]],
   "double_commutation_I1x_I2z" :> s[DoubleCommutationSuperoperator[opI[1, "x"], opI[2, "z"]]],
   "lindbladian_I1minus" :> s[Lindbladian[opI[1, "-"]]],
   "lindbladian_I1plus_I2minus" :> s[Lindbladian[opI[1, "+"], opI[2, "-"]]],
   "projection_I1z" :> s[ProjectionSuperoperator[opI[1, "z"]]],
   "filter_order_0" :> s[CoherenceOrderFiltrationSuperoperator[{0}]],
   "filter_orders_plus_minus_1" :> s[CoherenceOrderFiltrationSuperoperator[{-1, 1}]],
   "filter_spin1_order_plus_1" :> s[CoherenceOrderFiltrationSuperoperator[{1}, {1}]],
   "rotation_spin1_x_90" :> s[RotationSuperoperator[{1}, {Pi/2, "x"}]],
   "rotation_both_euler" :> s[RotationSuperoperator[{1, 2}, {Pi/5, Pi/3, 2 Pi/7}]],
   "permutation_12" :> s[SpinPermutationSuperoperator[{1, 2}]]};
{
 <|"group" -> "two_half_bases", "setup" :> SetSpinSystem[2], "cases" -> {
    "default_basis" :> ToString[OperatorBasis[], InputForm],
    "cartesian" :> basisMatrices[CartesianProductOperatorBasis[]],
    "cartesian_sorted" :> basisMatrices[CartesianProductOperatorBasis[Sorted -> True]],
    "shift_and_polarization" :> basisMatrices[ShiftAndPolarizationOperatorBasis[]],
    "shift_and_polarization_sorted" :> basisMatrices[ShiftAndPolarizationOperatorBasis[Sorted -> True]],
    "shift_and_z" :> basisMatrices[ShiftAndZOperatorBasis[]],
    "shift_and_z_default" :> basisMatrices[OperatorBasis[]],
    "singlet_triplet" :> basisMatrices[SingletTripletOperatorBasis[]],
    "singlet_triplet_unsorted" :> basisMatrices[SingletTripletOperatorBasis[Sorted -> False]],
    "zeeman_ket_bra" :> basisMatrices[ZeemanKetBraOperatorBasis[]],
    "zeeman_ket_bra_sorted" :> basisMatrices[ZeemanKetBraOperatorBasis[Sorted -> True]],
    "zeeman_ket_bra_canonical" :> basisMatrices[ZeemanKetBraOperatorBasis[Sorted -> "Canonical"]],
    "spherical_tensor" :> basisMatrices[SphericalTensorOperatorBasis[]],
    "spherical_tensor_sorted" :> basisMatrices[SphericalTensorOperatorBasis[Sorted -> True]],
    "spherical_tensor_indices" :> SphericalTensorIndices[SphericalTensorOperatorBasis[]],
    "transformation_cartesian_from_default" :> OperatorBasisTransformationMatrix[CartesianProductOperatorBasis[], OperatorBasis[]],
    "transformation_spherical_from_cartesian" :> OperatorBasisTransformationMatrix[SphericalTensorOperatorBasis[], CartesianProductOperatorBasis[]]
    }|>,
 <|"group" -> "two_half_vectors", "setup" :> SetSpinSystem[2], "cases" -> {
    "I1x_in_cartesian" :> vector[opI[1, "x"], CartesianProductOperatorBasis[]],
    "I1xI2z_in_cartesian" :> vector[opI[1, "x"] . opI[2, "z"], CartesianProductOperatorBasis[]],
    "option_form_is_ignored" :> OperatorVectorRepresentation[opI[1, "x"], OperatorBasis -> CartesianProductOperatorBasis[]],
    "I1x_in_default" :> vector[opI[1, "x"], OperatorBasis[]],
    "I1_dot_I2_in_spherical" :> vector[opI[1] . opI[2], SphericalTensorOperatorBasis[]],
    "I1plus_in_zeeman_ket_bra" :> vector[opI[1, "+"], ZeemanKetBraOperatorBasis[]],
    "bracket_I1x_I1x" :> LiouvilleBracket[opI[1, "x"], opI[1, "x"]],
    "bracket_I1plus_I1x" :> LiouvilleBracket[opI[1, "+"], opI[1, "x"]],
    "bracket_I1x_I1plus" :> LiouvilleBracket[opI[1, "x"], opI[1, "+"]],
    "bracket_normalized" :> LiouvilleBracket[opI[1, "x"], opI[1, "x"] + opI[1, "y"], Normalize -> True],
    "bracket_with_superoperator" :> LiouvilleBracket[opI[1, "y"], CommutationSuperoperator[opI[1, "z"]], opI[1, "x"]],
    "amplitude_same" :> OperatorAmplitude[opI[1, "x"] . opI[2, "z"] -> opI[1, "x"] . opI[2, "z"]],
    "amplitude_mixture" :> OperatorAmplitude[2 opI[1, "x"] . opI[2, "z"] + 3 opI[1, "y"] -> opI[1, "x"] . opI[2, "z"]],
    "amplitude_scaled_target" :> OperatorAmplitude[opI[1, "x"] . opI[2, "z"] -> 2 opI[1, "x"] . opI[2, "z"]],
    "amplitude_I1x_to_I1plus" :> OperatorAmplitude[opI[1, "x"] -> opI[1, "+"]],
    "amplitude_I1x_to_I1minus" :> OperatorAmplitude[opI[1, "x"] -> opI[1, "-"]],
    "amplitude_orthogonal" :> OperatorAmplitude[opI[1, "x"] -> opI[2, "x"]],
    "amplitude_symbolic" :> Symbolic[OperatorAmplitude[RotationSuperoperator[{1}, {b3, "x"}][opI[1, "z"]] -> opI[1, "y"]], {b3}]
    }|>,
 <|"group" -> "two_half_superoperators", "setup" :> SetSpinSystem[2], "cases" -> superoperators|>,
 <|"group" -> "two_half_superoperators_cartesian",
  "setup" :> (SetSpinSystem[2]; SetOperatorBasis[CartesianProductOperatorBasis[]]), "cases" -> superoperators|>,
 <|"group" -> "two_half_superoperators_explicit_basis", "setup" :> SetSpinSystem[2], "cases" -> {
    "commutation_I1z_in_zeeman_ket_bra" :> sIn[CommutationSuperoperator[opI[1, "z"]], ZeemanKetBraOperatorBasis[]],
    "lindbladian_I1minus_in_spherical" :> sIn[Lindbladian[opI[1, "-"]], SphericalTensorOperatorBasis[]],
    "commutation_hamiltonian_direct" :> SuperoperatorMatrixRepresentation[CommutationSuperoperator[ham], Method -> "Direct"]
    }|>,
 <|"group" -> "one_half", "setup" :> SetSpinSystem[1], "cases" -> {
    "default_basis" :> ToString[OperatorBasis[], InputForm],
    "default_basis_operators" :> basisMatrices[OperatorBasis[]],
    "cartesian" :> basisMatrices[CartesianProductOperatorBasis[]],
    "spherical_tensor" :> basisMatrices[SphericalTensorOperatorBasis[]],
    "commutation_I1z" :> s[CommutationSuperoperator[opI[1, "z"]]],
    "lindbladian_I1minus" :> s[Lindbladian[opI[1, "-"]]],
    "rotation_x_90" :> s[RotationSuperoperator[{1}, {Pi/2, "x"}]]
    }|>,
 <|"group" -> "spin1", "setup" :> SetSpinSystem[{{1, 1}}], "cases" -> {
    "default_basis" :> ToString[OperatorBasis[], InputForm],
    "default_basis_operators" :> basisMatrices[OperatorBasis[]],
    "zeeman_ket_bra" :> basisMatrices[ZeemanKetBraOperatorBasis[]],
    "spherical_tensor" :> basisMatrices[SphericalTensorOperatorBasis[]],
    "spherical_tensor_indices" :> SphericalTensorIndices[SphericalTensorOperatorBasis[]],
    "commutation_I1z" :> s[CommutationSuperoperator[opI[1, "z"]]],
    "double_commutation_T20" :> s[DoubleCommutationSuperoperator[opT[1, {2, 0}], opT[1, {2, 0}]]],
    "filter_order_0" :> s[CoherenceOrderFiltrationSuperoperator[{0}]]
    }|>,
 <|"group" -> "half_one", "setup" :> SetSpinSystem[{{1, 1/2}, {2, 1}}], "cases" -> {
    "default_basis" :> ToString[OperatorBasis[], InputForm],
    "default_basis_operators" :> basisMatrices[OperatorBasis[]],
    "spherical_tensor" :> basisMatrices[SphericalTensorOperatorBasis[]],
    "commutation_I1zI2z" :> s[CommutationSuperoperator[opI[1, "z"] . opI[2, "z"]]]
    }|>
 }
