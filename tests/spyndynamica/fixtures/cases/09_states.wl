(* Density operators, secularisation, thermalised relaxation, amplitude tables and optimisation. *)
(* the thermal results keep the symbols of the constants; put their SI values in *)
m[op_] := N[Normal[MatrixRepresentation[op]] /. PhysicalConstantValues];
s[sop_] := N[Normal[SuperoperatorMatrixRepresentation[sop]] /. PhysicalConstantValues];
best[result_] := Join[{First[result]}, Last /@ Last[result]];
hlab := -2 Pi 400*^6 opI[1, "z"] - 2 Pi 100*^6 opI[2, "z"] + 2 Pi 7 opI[1] . opI[2];
mixed := opI[1, "x"] . opI[2, "x"] + 2 opI[1, "z"] . opI[2, "z"] + 3 opI[1, "x"] + opI[1, "+"] . opI[2, "-"] + 5 opI[2, "z"];
relax := PhenomenologicalRelaxationSuperoperator[2., 0.2];
xyz := {opI[1, "x"], opI[1, "y"], opI[1, "z"]};
{
(* informational: the high-temperature approximation is in use; a higher user level is recommended; the
   table needs the front end for its progress display only; which FindMinimum is deployed *)
 <|"group" -> "two_half", "setup" :> SetSpinSystem[2],
  "allow" -> {"ThermalEquilibriumDensityOperator::highTapprox", "ThermalEquilibriumDensityOperator::symblowul"}, "cases" -> {
    "thermal_300K" :> m[ThermalEquilibriumDensityOperator[hlab, 300]],
    "thermal_10mK" :> m[ThermalEquilibriumDensityOperator[hlab, 0.01]],
    "thermal_10mK_exact" :> m[ThermalEquilibriumDensityOperator[hlab, 0.01, HighTemperatureApproximation -> False]],
    "thermal_list_syntax" :> m[ThermalEquilibriumDensityOperator[{hlab, 300}]],
    "polarized_all_z" :> m[PolarizedDensityOperator[{3/10, "z"}]],
    "polarized_spin_1_z" :> m[PolarizedDensityOperator[{1}, {3/10, "z"}]],
    "polarized_all_x" :> m[PolarizedDensityOperator[{3/10, "x"}]],
    "polarized_each" :> m[PolarizedDensityOperator[{{1, {3/10, "x"}}, {2, {1/10, "z"}}}]],
    "polarized_full" :> m[PolarizedDensityOperator[{1, "z"}]],
    "singlet_polarized" :> m[SingletPolarizedDensityOperator[4/10]],
    "singlet_polarized_full" :> m[SingletPolarizedDensityOperator[1]],
    "singlet_polarized_negative" :> m[SingletPolarizedDensityOperator[-1/3]],
    "secularize_hamiltonian" :> m[Secularize[mixed, 2 Pi 100 opI[1, "z"] + 2 Pi 30 opI[2, "z"]]],
    "secularize_homonuclear" :> m[Secularize[mixed, {{1, 2}}]],
    "secularize_heteronuclear" :> m[Secularize[mixed, {{1}, {2}}]],
    "secularize_threshold" :> m[Secularize[mixed, 2 Pi 100 opI[1, "z"] + 2 Pi 30 opI[2, "z"], 500]],
    "thermalized_relaxation_300K" :> s[ThermalizeSuperoperator[relax, hlab, 300]],
    "thermalized_relaxation_10mK" :> s[ThermalizeSuperoperator[relax, hlab, 0.01]],
    "thermalized_relaxation_10mK_exact" :>
     s[ThermalizeSuperoperator[relax, hlab, 0.01, HighTemperatureApproximation -> False]]
    }|>,
 <|"group" -> "one_half", "setup" :> SetSpinSystem[1],
  "allow" -> {"FrontEndObject::notavail", "MaximizeTransformationAmplitude::meth",
    "MaximizeTransformationAmplitude::unconstr", "MaximizeTransformationAmplitude::constr"}, "cases" -> {
    "table_flip_angle" :> TransformationAmplitudeTable[opI[1, "z"] -> xyz,
      {RotationSuperoperator[{bb, "x"}]}, {bb, 0, Pi, {5}}],
    "table_step" :> TransformationAmplitudeTable[opI[1, "z"] -> opI[1, "y"],
      {RotationSuperoperator[{bb, "x"}]}, {bb, 0, 1, 1/4}],
    "table_delay" :> TransformationAmplitudeTable[opI[1, "x"] -> xyz,
      {{2 Pi 100 opI[1, "z"], tt}}, {tt, 0, 1/100, {5}}],
    "maximize_flip_angle" :> best@MaximizeTransformationAmplitude[opI[1, "z"] -> -opI[1, "y"],
      {RotationSuperoperator[{bb, "x"}]}, {{bb, 1}}],
    "maximize_two_parameters" :> best@MaximizeTransformationAmplitude[opI[1, "z"] -> opI[1, "x"],
      {RotationSuperoperator[{bb, "x"}], {2 Pi 100 opI[1, "z"], tt}}, {{bb, 1}, {tt, 1/1000}},
      ConstrainPositive -> True]
    }|>
 }
