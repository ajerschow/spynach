(* Spin operators: exact matrices in the Zeeman basis, and the scalar properties of operators. *)
m[op_] := MatrixRepresentation[op];
components[spin_] := {
   "x" :> m[opI[spin, "x"]], "y" :> m[opI[spin, "y"]], "z" :> m[opI[spin, "z"]],
   "plus" :> m[opI[spin, "+"]], "minus" :> m[opI[spin, "-"]]};
prefixed[prefix_String, cases_List] := cases /. RuleDelayed[n_String, e_] :> RuleDelayed @@ Hold[prefix <> n, e];
tensors[spin_, ranks_List] :=
  Flatten[Table[With[{k = k, q = q}, ("T" <> ToString[k] <> "_" <> ToString[q]) :> m[opT[spin, {k, q}]]],
    {k, ranks}, {q, k, -k, -1}]];
{
 <|"group" -> "one_half", "setup" :> SetSpinSystem[1], "cases" -> Join[
    prefixed["I1", components[1]],
    {"I1alpha" :> m[opI[1, "\[Alpha]"]], "I1beta" :> m[opI[1, "\[Beta]"]],
     "I1_phase_pi3" :> m[opI[1, Pi/3]],
     "I1_polar_pi3_pi4" :> m[opI[1, {Pi/3, Pi/4}]],
     "I1_phase_symbolic" :> Symbolic[m[opI[1, phi]], {phi}],
     "I1_polar_symbolic" :> Symbolic[m[opI[1, {theta, phi}]], {theta, phi}],
     "unity" :> m[UnityOperator[]],
     "null" :> m[NullOperator[]]},
    prefixed["spin1_", tensors[1, {1}]]]|>,
 <|"group" -> "two_half", "setup" :> SetSpinSystem[2], "cases" -> Join[
    prefixed["I1", components[1]], prefixed["I2", components[2]],
    {"I1alpha" :> m[opI[1, "\[Alpha]"]], "I2beta" :> m[opI[2, "\[Beta]"]],
     "Ix_total" :> m[opI["x"]], "Iz_total" :> m[opI["z"]], "Iz_list" :> m[opI[{1, 2}, "z"]],
     "I1xI2z" :> m[opI[1, "x"] . opI[2, "z"]], "I1yI2x" :> m[opI[1, "y"] . opI[2, "x"]],
     "I1zI2z" :> m[opI[1, "z"] . opI[2, "z"]], "I1plusI2minus" :> m[opI[1, "+"] . opI[2, "-"]],
     "I1_dot_I2" :> m[opI[1] . opI[2]],
     "I1x_I1y_product" :> m[opI[1, "x"] . opI[1, "y"]],
     "commutator_I1x_I1y" :> m[Commutator[opI[1, "x"], opI[1, "y"]]],
     "commutator_I1xI2z_I1y" :> m[Commutator[opI[1, "x"] . opI[2, "z"], opI[1, "y"]]],
     "unity" :> m[UnityOperator[]], "unity_spin1" :> m[UnityOperator[1]],
     "st_x_12" :> m[SingleTransitionOperator[{1, 2}, "x"]], "st_y_12" :> m[SingleTransitionOperator[{1, 2}, "y"]],
     "st_z_12" :> m[SingleTransitionOperator[{1, 2}, "z"]], "st_plus_23" :> m[SingleTransitionOperator[{2, 3}, "+"]],
     "st_minus_14" :> m[SingleTransitionOperator[{1, 4}, "-"]], "st_unity_23" :> m[SingleTransitionOperator[{2, 3}, "u"]],
     "population_2" :> m[PopulationOperator[2]],
     "coherence_1_2" :> m[CoherenceOperator[{1, 2}]], "coherence_3_1" :> m[CoherenceOperator[{3, 1}]],
     "shift_ket1_bra2" :> m[ShiftOperator[Ket[1], Bra[2]]],
     "projection_ket3" :> m[ProjectionOperator[Ket[3]]],
     "projection_spin1_alpha" :> m[ProjectionOperator[{{1, 1/2}}]],
     "permutation_12" :> m[SpinPermutationOperator[{1, 2}]],
     "trace_I1zI1z" :> OperatorTrace[opI[1, "z"] . opI[1, "z"]],
     "trace_unity" :> OperatorTrace[UnityOperator[]],
     "norm_I1x" :> OperatorNorm[opI[1, "x"]], "norm_I1xI2z" :> OperatorNorm[opI[1, "x"] . opI[2, "z"]],
     "norm_I1plus" :> OperatorNorm[opI[1, "+"]],
     "coherence_order_I1plusI2plus" :> CoherenceOrder[opI[1, "+"] . opI[2, "+"]],
     "coherence_order_I1plusI2minus" :> CoherenceOrder[opI[1, "+"] . opI[2, "-"]],
     "coherence_order_I1x" :> CoherenceOrder[opI[1, "x"]],
     "coherence_order_per_spin" :> CoherenceOrder[{{1}, {2}}, opI[1, "+"] . opI[2, "-"]],
     "spin_product_rank_I1xI2y" :> SpinProductRank[opI[1, "x"] . opI[2, "y"]],
     "spin_product_rank_I1x" :> SpinProductRank[opI[1, "x"]],
     "commute_I1z_I2x" :> OperatorsCommuteQ[opI[1, "z"], opI[2, "x"]],
     "commute_I1z_I1x" :> OperatorsCommuteQ[opI[1, "z"], opI[1, "x"]]},
    prefixed["spin1_", tensors[1, {1}]], prefixed["spin2_", tensors[2, {1}]],
    prefixed["pair_", tensors[{1, 2}, {0, 1, 2}]]]|>,
 <|"group" -> "three_half", "setup" :> SetSpinSystem[3], "cases" -> {
    "I3y" :> m[opI[3, "y"]], "I2plus" :> m[opI[2, "+"]],
    "I1xI2yI3z" :> m[opI[1, "x"] . opI[2, "y"] . opI[3, "z"]],
    "I1_dot_I3" :> m[opI[1] . opI[3]],
    "permutation_12" :> m[SpinPermutationOperator[{1, 2}]],
    "permutation_cycle_123" :> m[SpinPermutationOperator[{1, 2, 3}]],
    "permutation_cycle_132" :> m[SpinPermutationOperator[{1, 3, 2}]],
    "permutation_12_after_23" :> m[SpinPermutationOperator[{{1, 2}, {2, 3}}]],
    "permuted_I1z_by_cycle_123" :> m[SpinPermutationOperator[{1, 2, 3}] . opI[1, "z"] . Adjoint[SpinPermutationOperator[{1, 2, 3}]]]
    }|>,
 <|"group" -> "spin1", "setup" :> SetSpinSystem[{{1, 1}}], "cases" -> Join[
    prefixed["I1", components[1]],
    {"I1z_squared" :> m[opI[1, "z"] . opI[1, "z"]], "I1_dot_I1" :> m[opI[1] . opI[1]],
     "norm_I1z" :> OperatorNorm[opI[1, "z"]]},
    prefixed["", tensors[1, {1, 2}]]]|>,
 <|"group" -> "half_one", "setup" :> SetSpinSystem[{{1, 1/2}, {2, 1}}], "cases" -> Join[
    prefixed["I1", components[1]], prefixed["I2", components[2]],
    {"I1zI2z" :> m[opI[1, "z"] . opI[2, "z"]], "I1_dot_I2" :> m[opI[1] . opI[2]],
     "I1plusI2minus" :> m[opI[1, "+"] . opI[2, "-"]]},
    prefixed["spin2_", tensors[2, {1, 2}]]]|>,
 (* opT::high\[Lambda] only says that ranks above 2 come from the Wigner-Eckart theorem *)
 <|"group" -> "spin32", "setup" :> SetSpinSystem[{{1, 3/2}}], "allow" -> {"opT::high\[Lambda]"}, "cases" -> Join[
    prefixed["I1", components[1]],
    {"I1_dot_I1" :> m[opI[1] . opI[1]]},
    prefixed["", tensors[1, {1, 2, 3}]]]|>
 }
