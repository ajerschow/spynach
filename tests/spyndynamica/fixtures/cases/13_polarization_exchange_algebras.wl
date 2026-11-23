(* Polarization level operators, extended Liouville space (chemical exchange), operator algebras, spin
   permutation groups and symmetry-adapted bases.

   The extended Liouville space is the module QM`ExtendedLiouville`, which is in the SpinDynamica folder but is
   not loaded by SpinDynamica.m; the groups that use it load it. Commutants, Lie algebras and symmetry-adapted
   bases are bases of spaces and not unique: the tests compare the spaces. Cases named ..._is_rejected or
   ..._fail record behaviour that differs from the documentation (DIVERGENCES.md). *)
m[op_] := Normal[MatrixRepresentation[op]];
(* whether a call was refused; the result itself cannot be stored *)
rejected[result_] := result === $Failed;
kets[basis_] := Normal[VectorRepresentation[#, ZeemanBasis[]]] & /@ BasisKets[basis];
(* SymmetryAdaptedBasis draws random numbers. For some draws it fails or returns kets that are no basis
   (DIVERGENCES.md, D12): seed 2 gives valid bases in all the cases below, seed 1 does not. *)
SetAttributes[seeded, HoldAll];
seeded[expr_] := (SeedRandom[2]; expr);
xv[op_] := Normal[ExtendedOperatorVectorRepresentation[op]];
xm[sop_] := Normal[ExtendedSuperoperatorMatrixRepresentation[sop]];
sdo := SiteDependentOperator[{{1, opI["z"]}, {2, 3 opI["x"]}}];
sds := CommutationSuperoperator[SiteDependentOperator[{{1, 10 opI["z"]}, {2, -20 opI["z"]}}]];
lieB := {opI[1, "x"], opI[1, "y"], opI[1, "z"], opI[2, "x"], opI[2, "y"], opI[2, "z"]};
(* the content of the graphic that DualPairing returns: for every Young diagram its row lengths and the
   labels and ranks of the rotation representations beside it *)
dual[n_, j_] := Cases[DualPairing[n, j], Grid[{{young_Grid, _Text, rest___}}, ___] :>
    {Length /@ First[young], Cases[{rest}, Superscript[label_, rank_] :> {label, rank}, Infinity]}, Infinity];
exchangeAllow = {"SetSites::setsites", "SetSites::alreadyset", "SetExtendedOperatorBasis::set",
   "SetExtendedOperatorBasis::alreadyset", "ExtendedOperatorBasis::inappspinsys", "OperatorBasis::inapprop"};
{
 <|"group" -> "two_half_polarization", "setup" :> SetSpinSystem[2], "cases" -> {
    "level_default" :> m[PolarizationLevelOperator[]],
    "level_z" :> m[PolarizationLevelOperator["z"]],
    "level_spin_1_x" :> m[PolarizationLevelOperator[{1}, "x"]],
    "level_spin_2_y" :> m[PolarizationLevelOperator[2, "y"]],
    "level_of_polarized_spin" :> OperatorAmplitude[PolarizedDensityOperator[{1}, {3/10, "z"}] -> PolarizationLevelOperator[{1}, "z"]],
    "mean_level_of_two_spins" :> OperatorAmplitude[PolarizedDensityOperator[{{1, {3/10, "z"}}, {2, {1/10, "z"}}}] -> PolarizationLevelOperator["z"]],
    "singlet_level" :> m[SingletPolarizationLevelOperator[]],
    "singlet_level_of_singlet_polarized" :> OperatorAmplitude[SingletPolarizedDensityOperator[3/10] -> SingletPolarizationLevelOperator[]]
    }|>,
 <|"group" -> "one_half_polarization", "setup" :> SetSpinSystem[{{1, 1}, {2, 1/2}}], "cases" -> {
    "level_spin_1" :> m[PolarizationLevelOperator[{1}, "z"]],
    "level_all" :> m[PolarizationLevelOperator["z"]],
    "level_of_polarized_spin_1" :> OperatorAmplitude[(UnityOperator[]/6 + opI[1, "z"]/8) -> PolarizationLevelOperator[{1}, "z"]]
    }|>,
 <|"group" -> "four_half_polarization", "setup" :> SetSpinSystem[4],
  "allow" -> {"SingletPolarizationLevelOperator::sharedspins"}, "cases" -> {
    "singlet_level_pair_1_2" :> m[SingletPolarizationLevelOperator[{1, 2}]],
    "singlet_level_pair_3_4" :> m[SingletPolarizationLevelOperator[{3, 4}]],
    "separate_pairs_are_rejected" :> rejected[SingletPolarizationLevelOperator[{{1, 2}, {3, 4}}]],
    "pairs_that_share_a_spin" :> m[SingletPolarizationLevelOperator[{{1, 2}, {2, 3}}]]
    }|>,
 <|"group" -> "exchange_two_sites", "setup" :> (Needs["QM`ExtendedLiouville`"]; SetSpinSystem[1]; SetSites[2]),
  "allow" -> Join[exchangeAllow, {"KineticExchangeMatrix::baddims", "KineticExchangeMatrix::detbalerr",
     "Transpose::nmtx", "SparseArray::list", "CommutationSuperoperator::badplus",
     "KineticExchangeSuperoperator::notdetbal"}], "cases" -> {
    "sites" :> Sites[],
    "basis_operators" :> (m /@ BasisOperators[ExtendedOperatorBasis[]]),
    "vector_site_dependent" :> xv[sdo],
    "vector_operator" :> xv[opI["z"]],
    "vector_one_site" :> xv[SiteDependentOperator[{2, opI["z"]}]],
    "vector_sum_with_operator" :> xv[sdo + opI["x"]],
    "vector_product" :> xv[sdo . sdo],
    "vector_multiple" :> xv[2 sdo],
    "vector_number" :> xv[3],
    "vector_exponential" :> xv[Exp[SiteDependentOperator[{{1, opI["z"]}, {2, 2 opI["z"]}}]]],
    "vector_in_cartesian_basis" :> Normal[ExtendedOperatorVectorRepresentation[SiteDependentOperator[{{1, opI["z"]}, {2, 2 opI["x"]}}],
       ExtendedOperatorBasis[{1, 2}, CartesianProductOperatorBasis[]]]],
    "vector_at_three_sites" :> Normal[ExtendedOperatorVectorRepresentation[SiteDependentOperator[{{1, opI["z"]}, {2, 2 opI["x"]}}],
       ExtendedOperatorBasis[{2, 5, 1}, CartesianProductOperatorBasis[]]]],
    "matrix_site_dependent_commutation" :> xm[sds],
    "matrix_commutation" :> xm[CommutationSuperoperator[opI["z"]]],
    "matrix_multiple" :> xm[-I sds],
    "matrix_sum" :> xm[sds + 2 sds],
    "matrix_product" :> xm[sds . sds],
    "matrix_product_with_superoperator" :> xm[sds . CommutationSuperoperator[opI["x"]]],
    "matrix_number" :> xm[3],
    "matrix_one_site" :> xm[SiteDependentSuperoperator[{2, CommutationSuperoperator[opI["z"]]}]],
    "matrix_two_superoperators" :> xm[SiteDependentSuperoperator[{{1, CommutationSuperoperator[opI["z"]]}, {2, Lindbladian[opI["-"]]}}]],
    "matrix_double_commutation" :> xm[DoubleCommutationSuperoperator[SiteDependentOperator[{{1, opI["z"]}, {2, 2 opI["z"]}}], opI["x"]]],
    "matrix_double_commutation_both" :> xm[DoubleCommutationSuperoperator[SiteDependentOperator[{{1, opI["z"]}, {2, 2 opI["z"]}}],
       SiteDependentOperator[{{1, opI["x"]}, {2, opI["y"]}}]]],
    "kinetic_matrix_from_rule" :> Normal[Expand[KineticExchangeMatrix[{{1 -> 2, 7}}]]],
    "kinetic_matrix_two_rules" :> Normal[Expand[KineticExchangeMatrix[{{1 -> 2, 7}, {2 -> 1, 3}}]]],
    "kinetic_matrix_from_matrix" :> Normal[Expand[KineticExchangeMatrix[{{0, 3}, {5, 0}}]]],
    "kinetic_matrix_already_balanced" :> Normal[Expand[KineticExchangeMatrix[{1, 2}, {{-2, 3}, {2, -3}}]]],
    "kinetic_matrix_not_balanced" :> Normal[Expand[KineticExchangeMatrix[{{0, 3}, {5, 0}}, ImposeDetailedBalance -> False]]],
    "kinetic_matrix_symbolic_not_balanced" :> Symbolic[Normal[Expand[KineticExchangeMatrix[{{1 -> 2, a3}, {2 -> 1, b3}},
         ImposeDetailedBalance -> False]]], {a3, b3}],
    "kinetic_matrix_symbolic_is_rejected" :> rejected[KineticExchangeMatrix[{{1 -> 2, a3}}]],
    "kinetic_matrix_conflict_is_rejected" :> rejected[KineticExchangeMatrix[{1, 2}, {{1, 3}, {2, -3}}]],
    "kinetic_superoperator" :> xm[KineticExchangeSuperoperator[{{1 -> 2, 7}}]],
    "kinetic_superoperator_two_rules" :> xm[KineticExchangeSuperoperator[{{1 -> 2, 7}, {2 -> 1, 3}}]],
    "kinetic_matrix_of_superoperator" :> Normal[KineticExchangeMatrix[KineticExchangeSuperoperator[{{1 -> 2, 7}}]]],
    "sum_with_superoperator_is_rejected" :> rejected[ExtendedSuperoperatorMatrixRepresentation[CommutationSuperoperator[opI["z"]] + sds]],
    "sum_with_exchange_stays_unevaluated" :> (Head[-I sds + KineticExchangeSuperoperator[{{1 -> 2, 7}}]] === Plus)
    }|>,
 <|"group" -> "exchange_three_sites", "setup" :> (Needs["QM`ExtendedLiouville`"]; SetSpinSystem[2]; SetSites[3]),
  "allow" -> exchangeAllow, "cases" -> {
    "kinetic_matrix" :> Normal[Expand[KineticExchangeMatrix[{{1 -> 2, 1}, {2 -> 3, 2}, {3 -> 2, 5}}]]],
    "kinetic_superoperator_dimensions" :> Dimensions[xm[KineticExchangeSuperoperator[{{1 -> 2, 1}, {2 -> 3, 2}, {3 -> 2, 5}}]]],
    "kinetic_superoperator_corner" :> xm[KineticExchangeSuperoperator[{{1 -> 2, 1}, {2 -> 3, 2}, {3 -> 2, 5}}]][[{1, 17, 33}, {1, 17, 33}]],
    "vector_two_of_three_sites" :> xv[SiteDependentOperator[{{1, opI[1, "z"]}, {3, opI[1] . opI[2]}}]],
    "matrix_site_dependent_commutation_diagonal" :> Diagonal[xm[CommutationSuperoperator[
        SiteDependentOperator[{{1, opI[1, "z"]}, {2, 2 opI[2, "z"]}, {3, opI["z"]}}]]]]
    }|>,
 <|"group" -> "two_half_algebras", "setup" :> SetSpinSystem[2], "cases" -> {
    "commutant_Iz" :> (m /@ Commutant[{opI["z"]}]),
    "commutant_scalar_coupling_and_Iz" :> (m /@ Commutant[{opI[1] . opI[2], opI["z"]}]),
    "commutant_of_one_operator" :> (m /@ Commutant[opI[1] . opI[2]]),
    "commutant_within" :> (m /@ Commutant[{{opI[1, "z"]}, lieB}]),
    "commutant_in_cartesian_basis" :> (m /@ Commutant[{opI["z"]}, OperatorBasis -> CartesianProductOperatorBasis[]]),
    "bicommutant_Iz" :> (m /@ Bicommutant[{opI["z"]}]),
    "bicommutant_scalar_coupling_and_Iz" :> (m /@ Bicommutant[{opI[1] . opI[2], opI["z"]}]),
    "lie_algebra_total_x_y" :> (m /@ LieAlgebra[{opI["x"], opI["y"]}]),
    "lie_algebra_spin_1_x_y" :> (m /@ LieAlgebra[{opI[1, "x"], opI[1, "y"]}]),
    "lie_algebra_weak_coupling" :> (m /@ LieAlgebra[{opI[1, "x"], opI[2, "x"], opI[1, "z"] . opI[2, "z"], opI[1, "y"], opI[2, "y"]}]),
    "lie_algebra_scalar_coupling" :> (m /@ LieAlgebra[{opI["x"], opI["y"], opI[1] . opI[2]}]),
    "lie_algebra_real_field" :> (m /@ LieAlgebra[{opI[1, "+"], opI[1, "z"]}]),
    "lie_algebra_complex_field" :> (m /@ LieAlgebra[{opI[1, "+"], opI[1, "z"]}, LieAlgebraField -> "C"]),
    "abelian_subalgebra_Iz" :> (m /@ AbelianSubalgebra[{opI["z"]}]),
    "abelian_subalgebra_two" :> (m /@ AbelianSubalgebra[{opI[1, "z"], opI[2, "z"]}]),
    "abelian_subalgebra_within" :> (m /@ AbelianSubalgebra[{{opI[1, "z"]}, lieB}]),
    "maximal_abelian_three" :> MaximalAbelianQ[{opI[1, "z"], opI[2, "z"], opI[1, "z"] . opI[2, "z"]}],
    "maximal_abelian_two" :> MaximalAbelianQ[{opI[1, "z"], opI[2, "z"]}],
    "maximal_abelian_within" :> MaximalAbelianQ[{{opI[1, "z"], opI[2, "x"]}, lieB}],
    "maximal_abelian_within_one" :> MaximalAbelianQ[{{opI[1, "z"]}, lieB}],
    "inaccessible_other_spin" :> InaccessibleOperatorQ[opI[1, "z"] -> opI[2, "z"], {opI[1, "x"], opI[1, "y"]}],
    "inaccessible_list" :> InaccessibleOperatorQ[opI[1, "z"] -> {opI[1, "x"], opI[1, "y"], opI[2, "z"], opI[1, "z"] . opI[2, "z"],
        opI[1, "x"] . opI[2, "z"], opI[1, "y"] . opI[2, "z"]}, {opI[1, "x"], opI[1, "z"] . opI[2, "z"]}],
    "inaccessible_singlet_order" :> InaccessibleOperatorQ[opI["z"] -> {opI[1] . opI[2], opI[1, "z"] - opI[2, "z"], opI["x"]},
      {opI["x"], opI["y"], opI[1] . opI[2]}],
    "accessible_singlet_order" :> InaccessibleOperatorQ[opI[1, "z"] - opI[2, "z"] -> {opI[1] . opI[2], opI["z"]},
      {opI[1, "x"] - opI[2, "x"], opI[1] . opI[2], opI[1, "z"] - opI[2, "z"]}]
    }|>,
 <|"group" -> "three_half_algebras", "setup" :> SetSpinSystem[3], "cases" -> {
    "commutant_count" :> Length[Commutant[{opI["z"], opI[1] . opI[2] + opI[2] . opI[3] + opI[1] . opI[3]}]],
    "commutant" :> (m /@ Commutant[{opI["z"], opI[1] . opI[2] + opI[2] . opI[3] + opI[1] . opI[3]}]),
    "lie_algebra_count" :> Length[LieAlgebra[{opI[1, "x"], opI[1, "y"], opI[1] . opI[2], opI[2] . opI[3]}]]
    }|>,
 <|"group" -> "three_half_groups", "setup" :> SetSpinSystem[3], "cases" -> {
    "symmetric_group" :> (m /@ SpinPermutationGroupElements[SpinPermutationGroup[{"SymmetricGroup", 3}]]),
    "cyclic_group" :> (m /@ SpinPermutationGroupElements[SpinPermutationGroup[{"CyclicGroup", 3}]]),
    "alternating_group" :> (m /@ SpinPermutationGroupElements[SpinPermutationGroup[{"AlternatingGroup", 3}]]),
    "symmetric_group_of_spins_2_3" :> (m /@ SpinPermutationGroupElements[SpinPermutationGroup[{2, 3}, {"SymmetricGroup", 2}]]),
    "element_2" :> m[SpinPermutationGroup[{"SymmetricGroup", 3}][2]],
    "symmetry_adapted_basis" :> seeded[kets[SymmetryAdaptedBasis[]]],
    "symmetry_adapted_basis_symmetric" :> seeded[kets[SymmetryAdaptedBasis[{"SymmetricGroup", 3}]]],
    "symmetry_adapted_basis_cyclic" :> seeded[kets[SymmetryAdaptedBasis[{"CyclicGroup", 3}]]],
    "symmetry_adapted_basis_from_elements" :> seeded[kets[SymmetryAdaptedBasis[{{"AlternatingGroup", 3},
        SpinPermutationGroupElements[SpinPermutationGroup[{"AlternatingGroup", 3}]]}]]]
    }|>,
 (* the option makes this another object than the basis that the group above has stored *)
 <|"group" -> "three_half_groups_other_draw", "setup" :> SetSpinSystem[3], "cases" -> {
    "symmetry_adapted_basis_is_defective" :> (SeedRandom[1]; kets[SymmetryAdaptedBasis[BasisLabels -> Automatic]])
    }|>,
 <|"group" -> "four_half_groups", "setup" :> SetSpinSystem[4], "cases" -> {
    "symmetric_group_count" :> Length[SpinPermutationGroupElements[SpinPermutationGroup[{"SymmetricGroup", 4}]]],
    "dihedral_group" :> (m /@ SpinPermutationGroupElements[SpinPermutationGroup[{"DihedralGroup", 4}]]),
    "symmetry_adapted_basis" :> seeded[kets[SymmetryAdaptedBasis[]]],
    "symmetry_adapted_basis_cyclic" :> seeded[kets[SymmetryAdaptedBasis[{"CyclicGroup", 4}]]]
    }|>,
 <|"group" -> "two_spin1_groups", "setup" :> SetSpinSystem[{{1, 1}, {2, 1}}], "cases" -> {
    "symmetry_adapted_basis" :> seeded[kets[SymmetryAdaptedBasis[]]]
    }|>,
 <|"group" -> "dual_pairing", "setup" :> SetSpinSystem[1], "cases" -> {
    "three_half" :> dual[3, 1/2],
    "four_half" :> dual[4, 1/2],
    "three_one" :> dual[3, 1],
    "two_three_half" :> dual[2, 3/2],
    "four_one" :> dual[4, 1]
    }|>
 }
