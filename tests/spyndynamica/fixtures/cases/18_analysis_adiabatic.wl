(* Bounds on transformation amplitudes, correlated eigensystems and adiabatic events.
   CorrelatedEigensystems with a parameter is refused by SpinDynamica 3.10.2 for every input (DIVERGENCES.md,
   D15); the cases "internal_..." enter behind the test that fails, with the tags that SpinDynamica itself
   passes when an adiabatic event calls the routine. Cases named ..._is_rejected hold True when SpinDynamica
   refuses the call. Times in seconds. *)
m[op_] := MatrixRepresentation[op];
rejected[result_] := result === $Failed;
hfield := bfield opI[1, "z"] + 2 bfield opI[2, "z"] + 2 Pi 5 opI[1] . opI[2];
fields = {0, 10, 20, 30, 40};
matrices := N[m[hfield /. bfield -> #]] & /@ fields;
ramp := Function[t, 2 Pi 100 (1 - 2 t/0.5) (opI[1, "z"] - opI[2, "z"])/2 + 2 Pi 10 opI[1] . opI[2]];
slow := Function[t, 2 Pi 100 (1 - 2 t/5) (opI[1, "z"] - opI[2, "z"])/2 + 2 Pi 10 opI[1] . opI[2]];
adiabatic[gen_, tau_, rho_, opts___] :=
  Normal[m[NPropagate[{{gen, tau, PropagationMethod -> Adiabatic, opts}}][rho]]];
{
 <|"group" -> "bounds", "setup" :> SetSpinSystem[2],
  "allow" -> {"TransformationAmplitudeBounds::nonherm"}, "cases" -> {
    "z1_to_z2" :> TransformationAmplitudeBounds[opI[1, "z"] -> opI[2, "z"]],
    "total_z_to_several" :> TransformationAmplitudeBounds[
      opI[1, "z"] + opI[2, "z"] -> {opI[1, "z"], 2 opI[1, "z"] . opI[2, "z"], opI[1, "x"], opI[1] . opI[2]}],
    "z1_to_singlet_order" :> TransformationAmplitudeBounds[opI[1, "z"] -> opI[1] . opI[2]],
    "polarized_to_singlet_order" :> TransformationAmplitudeBounds[
      UnityOperator[]/4 + opI[1, "z"]/3 + opI[2, "z"]/5 -> opI[1] . opI[2]],
    "normalization_factor" :> TransformationAmplitudeBounds[opI[1, "z"] -> opI[2, "z"], NormalizationFactor -> 2],
    "not_hermitian_is_rejected" :> rejected[TransformationAmplitudeBounds[opI[1, "z"] -> opI[2, "-"]]]
    }|>,
 <|"group" -> "bounds_three", "setup" :> SetSpinSystem[3], "cases" -> {
    "z1_to_z3" :> TransformationAmplitudeBounds[opI[1, "z"] -> opI[3, "z"]],
    "total_z_to_three_spin_order" :> TransformationAmplitudeBounds[
      opI[1, "z"] + opI[2, "z"] + opI[3, "z"] -> 4 opI[1, "z"] . opI[2, "z"] . opI[3, "z"]]
    }|>,
 <|"group" -> "correlated", "setup" :> SetSpinSystem[2],
  "allow" -> {"CorrelatedEigensystems::parAssigned"}, "cases" -> {
    "matrices" :> matrices,
    "list_eigenvalues" :> Normal /@ CorrelatedEigensystems[matrices][[1]],
    "list_eigenvectors" :> Normal /@ CorrelatedEigensystems[matrices][[2]],
    "internal_eigenvalues" :> Normal /@ CorrelatedEigensystems[hfield, {bfield, fields},
        ReturnDetailedEigensystem -> False, "Operator", {"1", "4"}][[1]],
    "internal_eigenvalue_tracks" :> N[CorrelatedEigensystems[hfield, {bfield, fields}, "Operator", {"1", "4"}][[1]]],
    "with_parameter_is_rejected" :> rejected[CorrelatedEigensystems[hfield, {bfield, 0, 40, {5}}]],
    "matrix_with_parameter_is_rejected" :>
     rejected[CorrelatedEigensystems[{{xpar, 1}, {1, -xpar}}, {xpar, {-2, -1, 0, 1, 2}}]]
    }|>,
 <|"group" -> "adiabatic", "setup" :> SetSpinSystem[2], "cases" -> {
    "populations" :> adiabatic[ramp, 0.5, opI[1, "z"], EventSampling -> {50}],
    "populations_exact" :> m[NPropagate[{{ramp, 0.5}}][opI[1, "z"]]],
    "coherences_are_deleted" :> adiabatic[ramp, 0.5, opI[1, "x"] + opI[1, "z"], EventSampling -> {50}],
    "with_coherences" :> adiabatic[ramp, 0.5, opI[1, "x"], EventSampling -> {50}, DeleteCoherences -> False],
    "with_coherences_exact" :> m[NPropagate[{{ramp, 0.5}}][opI[1, "x"]]],
    "slow_with_coherences" :> adiabatic[slow, 5, opI[1, "x"], EventSampling -> {400}, DeleteCoherences -> False],
    "slow_with_coherences_exact" :> m[NPropagate[{{slow, 5}}][opI[1, "x"]]],
    "time_grid" :> adiabatic[ramp, 0.5, opI[1, "z"], EventSampling -> Range[0, 0.5, 0.02]],
    "after_a_pulse" :> Normal[m[NPropagate[{RotationSuperoperator[{Pi/3, "y"}],
         {ramp, 0.5, PropagationMethod -> Adiabatic, EventSampling -> {50}}}][opI[1, "z"] + opI[2, "z"]]]]
    }|>
}
