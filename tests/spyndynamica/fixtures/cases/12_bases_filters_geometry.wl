(* The rest of the quantum-mechanics layer: coupled tensor operators and the spherical tensor operator basis
   of more than two spins, the spherical component filter, user-defined state and operator bases, product
   bases, and the geometry routines (axis systems, Euler angles of axis systems, polar rotations,
   quaternions). Cases named ..._is_rejected or ..._fails record behaviour that differs from the
   documentation (DIVERGENCES.md, D7 and D8). *)
m[op_] := Normal[MatrixRepresentation[op]];
(* whether a call was refused; the result itself cannot be stored *)
rejected[result_] := result === $Failed;
kets[basis_] := Normal[VectorRepresentation[#, ZeemanBasis[]]] & /@ BasisKets[basis];
basisMatrices[opbasis_] := m /@ BasisOperators[opbasis];
sparseBasis[opbasis_] := Flatten[MapIndexed[
    Function[{mat, k}, (Join[{First[k]}, #[[1]], {#[[2]]}] &) /@ Most[ArrayRules[SparseArray[mat]]]],
    basisMatrices[opbasis]], 1];
s[sop_, opbasis_] := Normal[SuperoperatorMatrixRepresentation[sop, opbasis]];
filter[spec_] := SphericalComponentFiltrationSuperoperator[spec];
k2[a_, b_] := Ket[{{1, a}, {2, b}}];
(* operators with components of every rank and order *)
generic2 := 3 opI[1, "z"] + 2 opI[2, "x"] - opI[2, "y"] + opI[1, "x"] . opI[2, "z"] + 5 opI[1, "y"] . opI[2, "y"] +
   4 opI[1, "+"] . opI[2, "+"] + 7 opI[1, "z"] . opI[2, "-"] + 2 UnityOperator[] + opI[1, "x"] . opI[2, "y"];
generic3 := opI[1, "z"] + 2 opI[2, "x"] - 3 opI[3, "y"] + opI[1, "x"] . opI[2, "z"] + 5 opI[1, "y"] . opI[3, "y"] +
   4 opI[2, "+"] . opI[3, "+"] + 7 opI[1, "z"] . opI[2, "-"] . opI[3, "x"] + 2 opI[1, "+"] . opI[2, "+"] . opI[3, "-"] +
   6 opI[1, "x"] . opI[2, "y"] . opI[3, "z"] + opI[1, "+"] . opI[2, "+"] . opI[3, "+"] + 3 opI[2, "z"] . opI[3, "x"];
(* HoldFirst: the operator is built inside the case, under the spin system of its group *)
SetAttributes[applied, HoldFirst];
specName[spec_] := "applied_" <> StringRiffle[StringReplace[ToString[#], "-" -> "m"] & /@ spec, "_"];
applied[op_, specs_List] := Function[spec, specName[spec] :> m[filter[spec][op]]] /@ specs;
cartesianOrder = Join[{16}, Range[15]];
sysA = AxisSystem[{1, 2, 0}, {1, 1, 1}]; sysB = AxisSystem[{0, 1, 1}, {-1, 2, 1/2}];
angles = {3/10, 7/10, -2/5};
qa = PolarToQuaternion[{0.3, 0.7, -0.4}]; qb = PolarToQuaternion[{1.3, 0.2, 2.4}]; qc = PolarToQuaternion[{0.5, 1.7, 0.4}];
{
 <|"group" -> "three_half_tensors", "setup" :> SetSpinSystem[3], "allow" -> {"opT::multiple"}, "cases" -> {
    "spherical_tensor" :> sparseBasis[SphericalTensorOperatorBasis[]],
    "spherical_tensor_indices" :> SphericalTensorIndices[SphericalTensorOperatorBasis[]],
    "T_history_1_1_1_component_0" :> m[opT[{{1, 1}, {2, 1}, {3, 1}}, {1, 1, 1}, {1, 0}]],
    "T_history_1_0_1_component_1" :> m[opT[{{1, 1}, {2, 1}, {3, 1}}, {1, 0, 1}, {1, 1}]],
    "T_history_1_2_1_component_m1" :> m[opT[{{1, 1}, {2, 1}, {3, 1}}, {1, 2, 1}, {1, -1}]],
    "T_history_1_2_2_component_1" :> m[opT[{{1, 1}, {2, 1}, {3, 1}}, {1, 2, 2}, {2, 1}]],
    "T_history_1_1_0_component_0" :> m[opT[{{1, 1}, {2, 1}, {3, 1}}, {1, 1, 0}, {0, 0}]],
    "T_rank_3_component_2" :> m[opT[{1, 2, 3}, {3, 2}]],
    "T_rank_1_component_0_all" :> (m /@ opT[{1, 2, 3}, {1, 0}]),
    "T_rank_2_component_1_all" :> (m /@ opT[{1, 2, 3}, {2, 1}]),
    "T_rank_1_component_0_index_2" :> m[opT[{{1, 1}, {2, 1}, {3, 1}}, {1, 0}, 2]],
    "T_spins_1_3_rank_2_component_0" :> m[opT[{1, 3}, {2, 0}]]
    }|>,
 <|"group" -> "half_one_half_tensors", "setup" :> SetSpinSystem[{{1, 1/2}, {2, 1}, {3, 1/2}}], "cases" -> {
    "spherical_tensor" :> sparseBasis[SphericalTensorOperatorBasis[]],
    "spherical_tensor_indices" :> SphericalTensorIndices[SphericalTensorOperatorBasis[]],
    "T_ranks_1_2_1_history_1_2_3" :> m[opT[{{1, 1}, {2, 2}, {3, 1}}, {1, 2, 3}, {3, -2}]],
    "T_ranks_1_2_1_history_1_3_2" :> m[opT[{{1, 1}, {2, 2}, {3, 1}}, {1, 3, 2}, {2, 1}]]
    }|>,
 <|"group" -> "tensor_basis_sorting", "setup" :> SetSpinSystem[2], "allow" -> {"OperatorBasis::invalidsortopt"}, "cases" -> {
    "rank_then_order_is_rejected" :> rejected[SphericalTensorOperatorBasis[Sorted -> {SphericalRank, CoherenceOrder}]],
    "spherical_rank_same_as_default" :> (basisMatrices[SphericalTensorOperatorBasis[Sorted -> SphericalRank]] === basisMatrices[SphericalTensorOperatorBasis[]]),
    "unsorted_same_as_default" :> (basisMatrices[SphericalTensorOperatorBasis[Sorted -> False]] === basisMatrices[SphericalTensorOperatorBasis[]])
    }|>,
 <|"group" -> "two_half_filter", "setup" :> SetSpinSystem[2], "cases" -> Join[{
     "matrix_1_0_0" :> s[filter[{1, 0, 0}], CartesianProductOperatorBasis[]],
     "matrix_2_1_m1" :> s[filter[{2, 1, -1}], CartesianProductOperatorBasis[]],
     "matrix_0_0_0" :> s[filter[{0, 0, 0}], CartesianProductOperatorBasis[]],
     "rank_3_is_null" :> ToString[filter[{3, 0, 0}], InputForm]},
    applied[generic2, {{0, 0, 0}, {1, 1, 1}, {1, 0, 0}, {1, -1, -1}, {1, 1, 0}, {1, 0, -1}, {1, -1, 1}, {2, 2, 2},
      {2, 0, 0}, {2, 1, -1}, {2, -2, 2}, {2, 0, 1}}]]|>,
 <|"group" -> "three_half_filter", "setup" :> SetSpinSystem[3], "cases" ->
   applied[generic3, {{0, 0, 0}, {1, 1, 1}, {1, 0, 0}, {1, -1, -1}, {1, 1, 0}, {1, 0, -1}, {1, -1, 1}, {2, 2, 2},
     {2, 0, 0}, {2, 1, -1}, {2, -2, 2}, {2, 0, 1}, {3, 3, 3}, {3, 0, 0}, {3, 1, -2}, {3, -3, 3}}]|>,
 <|"group" -> "defined_bases", "setup" :> SetSpinSystem[2],
  "allow" -> {"DefineBasis::defined", "CheckBasis::orthonormal", "CheckBasis::orthogonalsamenorm",
    "CheckBasis::orthogonaldiffnorm", "CheckBasis::notorthogonal", "CheckBasis::incomplete",
    "DefineBasis::notdefined", "UndefineBasis::undefined"}, "cases" -> {
    "define" :> ToString[DefineBasis[fixtureBasis12, {k2[1/2, 1/2], (k2[1/2, -1/2] + k2[-1/2, 1/2])/Sqrt[2],
        k2[-1/2, -1/2], (k2[1/2, -1/2] - k2[-1/2, 1/2])/Sqrt[2]}, BasisLabels -> {"a", "b", "c", "d"}]],
    "kets" :> kets[fixtureBasis12],
    "labels" :> BasisLabels[fixtureBasis12],
    "I1z" :> m[opI[1, "z"]] . {{1}, {0}, {0}, {0}},
    "I1z_in_defined_basis" :> Normal[MatrixRepresentation[opI[1, "z"], fixtureBasis12]],
    "I1x_in_defined_basis" :> Normal[MatrixRepresentation[opI[1, "x"], fixtureBasis12]],
    "check_orthonormal" :> CheckBasis[{k2[1/2, 1/2], k2[-1/2, 1/2], k2[1/2, -1/2], k2[-1/2, -1/2]}],
    "check_same_norm" :> CheckBasis[{2 k2[1/2, 1/2], 2 k2[-1/2, 1/2], 2 k2[1/2, -1/2], 2 k2[-1/2, -1/2]}],
    "check_different_norms" :> CheckBasis[{k2[1/2, 1/2], 2 k2[-1/2, 1/2], k2[1/2, -1/2], k2[-1/2, -1/2]}],
    "check_not_orthogonal" :> CheckBasis[{k2[1/2, 1/2], k2[1/2, -1/2] + k2[-1/2, 1/2], k2[1/2, -1/2], k2[-1/2, -1/2]}],
    "check_incomplete_orthonormal" :> CheckBasis[{k2[1/2, 1/2], k2[-1/2, -1/2]}],
    "defective_is_not_defined" :> rejected[DefineBasis[fixtureBasis12bad, {k2[1/2, 1/2], 2 k2[-1/2, 1/2], k2[1/2, -1/2], k2[-1/2, -1/2]}]]
    }|>,
 <|"group" -> "product_bases", "setup" :> SetSpinSystem[3], "allow" -> {"ProductBasis::usecurrspinsys"}, "cases" -> {
    "singlet_triplet_12_zeeman_3" :> kets[ProductBasis[SingletTripletBasis[{{1, 1/2}, {2, 1/2}}], ZeemanBasis[{{3, 1/2}}]]],
    "zeeman_3_singlet_triplet_12" :> kets[ProductBasis[ZeemanBasis[{{3, 1/2}}], SingletTripletBasis[{{1, 1/2}, {2, 1/2}}]]],
    "singlet_triplet_13_zeeman_2" :> kets[ProductBasis[SingletTripletBasis[{{1, 1/2}, {3, 1/2}}], ZeemanBasis[{{2, 1/2}}]]],
    "three_zeeman_bases" :> kets[ProductBasis[ZeemanBasis[{{2, 1/2}}], ZeemanBasis[{{3, 1/2}}], ZeemanBasis[{{1, 1/2}}]]],
    "I3z_in_product_basis" :> Normal[MatrixRepresentation[opI[3, "z"],
       ProductBasis[SingletTripletBasis[{{1, 1/2}, {2, 1/2}}], ZeemanBasis[{{3, 1/2}}]]]],
    "I1I2_in_product_basis" :> Normal[MatrixRepresentation[opI[1] . opI[2],
       ProductBasis[SingletTripletBasis[{{1, 1/2}, {2, 1/2}}], ZeemanBasis[{{3, 1/2}}]]]]
    }|>,
 (* Set::write comes from DefineOperatorBasis itself (it appends to an option value that is not a symbol) *)
 <|"group" -> "defined_operator_bases", "setup" :> SetSpinSystem[2],
  "allow" -> {"DefineOperatorBasis::maybeslow", "DefineOperatorBasis::defined", "DefineOperatorBasis::notUsop",
    "DefineOperatorBasis::notdef", "CheckOperatorBasis::orthonormal", "CheckOperatorBasis::incomplete",
    "CheckOperatorBasis::orthogonalsamenorm", "CheckOperatorBasis::orthogonaldiffnorm",
    "CheckOperatorBasis::notorthogonal", "Set::write"}, "cases" -> {
    "define_from_operators" :> ToString[DefineOperatorBasis[fixtureOpBasis12a, Reverse[BasisOperators[CartesianProductOperatorBasis[]]]]],
    "from_operators" :> basisMatrices[fixtureOpBasis12a],
    "commutation_in_basis_from_operators" :> s[CommutationSuperoperator[opI[1, "x"]], fixtureOpBasis12a],
    "define_reordered" :> ToString[DefineOperatorBasis[fixtureOpBasis12b, CartesianProductOperatorBasis[], cartesianOrder]],
    "reordered" :> basisMatrices[fixtureOpBasis12b],
    "define_transformed" :> ToString[DefineOperatorBasis[fixtureOpBasis12c, CartesianProductOperatorBasis[], RotationSuperoperator[{Pi/2, "x"}]]],
    "transformed" :> basisMatrices[fixtureOpBasis12c],
    "define_transformed_reordered" :> ToString[DefineOperatorBasis[fixtureOpBasis12d, CartesianProductOperatorBasis[],
       RotationSuperoperator[{1}, {Pi/3, "y"}], cartesianOrder]],
    "transformed_reordered" :> basisMatrices[fixtureOpBasis12d],
    "transformation_matrix" :> Normal[OperatorBasisTransformationMatrix[fixtureOpBasis12d, CartesianProductOperatorBasis[]]],
    "vector_in_transformed_reordered" :> Normal[OperatorVectorRepresentation[opI[1, "z"] + 2 opI[1, "x"] . opI[2, "y"], fixtureOpBasis12d]],
    "commutation_in_transformed_reordered" :> s[CommutationSuperoperator[opI[1, "z"]], fixtureOpBasis12d],
    "non_unitary_is_rejected" :> rejected[DefineOperatorBasis[fixtureOpBasis12e, CartesianProductOperatorBasis[], CommutationSuperoperator[opI[1, "z"]]]],
    "check_orthonormal" :> CheckOperatorBasis[BasisOperators[CartesianProductOperatorBasis[]]],
    "check_same_norm" :> CheckOperatorBasis[2 BasisOperators[CartesianProductOperatorBasis[]]],
    "check_different_norms" :> CheckOperatorBasis[{opI[1, "z"], 2 opI[2, "z"]}],
    "check_not_orthogonal" :> CheckOperatorBasis[{opI[1, "z"], opI[1, "z"] + opI[2, "z"]}],
    "check_incomplete_orthonormal" :> CheckOperatorBasis[{opI[1, "z"], opI[2, "z"]}]
    }|>,
 <|"group" -> "geometry", "setup" :> SetSpinSystem[1], "cases" -> {
    "axis_system_z" :> AxisSystem[{1, 1, 1}],
    "axis_system_x_z" :> AxisSystem[{1, 2, 0}, {1, 1, 1}],
    "axis_system_x_y_z" :> AxisSystem[{0, 2, 0}, {-3, 0, 0}, {0, 0, 1}],
    "axis_system_z_along_x" :> AxisSystem[{2, 0, 0}],
    "axis_system_z_against_x" :> AxisSystem[{-2, 0, 0}],
    "axis_system_x_parallel_to_z" :> AxisSystem[{1, 1, 1}, {2, 2, 2}],
    "axis_system_default" :> AxisSystem[],
    "axes_to_euler_z" :> AxesToEuler[{{1, 1, 1}}],
    "axes_to_euler_system" :> AxesToEuler[sysA],
    "axes_to_euler_x_z" :> AxesToEuler[{{1, 2, 0}, {1, 1, 1}}],
    "axes_to_euler_two_systems" :> AxesToEuler[sysA, sysB],
    "axes_to_euler_two_systems_reversed" :> AxesToEuler[sysB, sysA],
    "axes_to_euler_z_parallel" :> AxesToEuler[{{0, 1, 0}, {0, 0, 1}}],
    "axes_to_euler_z_antiparallel" :> AxesToEuler[{{0, 1, 0}, {0, 0, -1}}],
    "rotate_euler_list" :> RotateEuler[{{1, 2, 3}, {0, 0, 1}}, angles],
    "rotate_euler_about_axes" :> RotateEuler[{1, 2, 3}, angles, sysA],
    "rotate_euler_about_z_only" :> RotateEuler[{1, 2, 3}, angles, {{1, 1, 1}}],
    "rotate_euler_superimpose_fails" :> RotateEuler[N[sysA], AxesToEuler[sysA, sysB], sysA],
    "system_B" :> N[sysB],
    "polar_rotation_matrix" :> PolarRotationMatrix[angles],
    "polar_rotation_matrix_symbolic" :> Symbolic[PolarRotationMatrix[{a3, b3, c3}], {a3, b3, c3}],
    "rotate_polar" :> RotatePolar[{1, 2, 3}, angles],
    "rotate_polar_about_vector" :> RotatePolar[{1, 2, 3}, {3/10, {1, 1, 1}}],
    "rotate_polar_list" :> RotatePolar[{{1, 2, 3}, {0, 0, 1}}, angles],
    "polar_to_quaternion" :> PolarToQuaternion[angles],
    "quaternion_to_polar" :> QuaternionToPolar[qb],
    "quaternion_to_polar_identity" :> QuaternionToPolar[{1, 0, 0, 0}],
    "quaternion_to_polar_about_z" :> QuaternionToPolar[{Cos[0.2], 0, 0, Sin[0.2]}],
    "quaternion_to_euler" :> QuaternionToEuler[qa],
    "quaternion_to_euler_b" :> QuaternionToEuler[qb],
    "quaternion_to_euler_identity" :> QuaternionToEuler[{1, 0, 0, 0}],
    "quaternion_product" :> QuaternionProduct[qa, qb],
    "quaternion_product_of_three" :> QuaternionProduct[qc, qa, qb],
    "quaternion_product_of_one" :> QuaternionProduct[qa],
    "quaternions" :> {qa, qb, qc},
    "vector_to_polar" :> VectorToPolar[{1, 2, 3}],
    "vector_to_polar_along_z" :> VectorToPolar[{0, 0, -2}],
    "polar_to_vector" :> PolarToVector[2, {3/10, 2/5}],
    "distance" :> Distance[{1, 2, 3}, {0, 1, 5}],
    "subtended_angle_of_two_vectors" :> SubtendedAngle[{1, 0, 0}, {1, 1, 0}],
    "subtended_angle_at_a_point" :> SubtendedAngle[{1, 1/5, 0}, {0, 1/2, 0}, {-3/10, 1, 1/10}],
    "subtended_angle_of_four_points" :> SubtendedAngle[{1, 1/5, 0}, {0, 0, 0}, {0, 3/2, 0}, {-1/2, 17/10, 4/5}],
    "bond_angle" :> BondAngle[{1, 1/5, 0}, {0, 0, 0}, {-3/10, 1, 1/10}],
    "torsion_angle" :> TorsionAngle[{1, 1/5, 0}, {0, 0, 0}, {0, 3/2, 0}, {-1/2, 17/10, 4/5}],
    "torsion_angle_other_side" :> TorsionAngle[{1, 1/5, 0}, {0, 0, 0}, {0, 3/2, 0}, {-1/2, 17/10, -4/5}],
    "torsion_angle_trans" :> TorsionAngle[{1, 0, 0}, {0, 0, 0}, {0, 1, 0}, {-1, 1, 0}],
    "solid_angle_octant" :> SolidAngle[{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}],
    "solid_angle" :> N[SolidAngle[{{1, 1/5, 0}, {0, 2, 3/10}, {1/10, 0, 3}}]],
    "solid_angle_of_four_points_is_not_evaluated" :> (Head[SolidAngle[{{1, 0, 0}, {0, 1, 0}, {-1, 0, 0}, {0, 0, 1}}]] === SolidAngle)
    }|>
 }
