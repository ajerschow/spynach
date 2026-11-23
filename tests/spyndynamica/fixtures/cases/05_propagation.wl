(* Propagation of time-independent events, used from phase 3 on: free evolution, instantaneous rotations,
   pulses, relaxation, repetition, both chronologies and transformation amplitudes. Times in seconds. *)
m[op_] := MatrixRepresentation[op];
ham := 2 Pi 100 opI[1, "z"] - 2 Pi 40 opI[2, "z"] + 2 Pi 7 opI[1] . opI[2];
weak := 2 Pi 100 opI[1, "z"] - 2 Pi 40 opI[2, "z"] + 2 Pi 7 opI[1, "z"] . opI[2, "z"];
relax := PhenomenologicalRelaxationSuperoperator[2., 0.2];
pi2x := RotationSuperoperator[{1, 2}, {Pi/2, "x"}];
piy := RotationSuperoperator[{1, 2}, {Pi, "y"}];
targets := {opI[1, "x"], opI[1, "y"], opI[1, "z"], opI[1, "x"] . opI[2, "z"], opI[1, "y"] . opI[2, "z"]};
{
 <|"group" -> "two_half", "setup" :> SetSpinSystem[2], "cases" -> {
    "free_strong_I1x_10ms" :> m[NPropagate[{{ham, 0.01}}][opI[1, "x"]]],
    "free_weak_I1x_10ms" :> m[NPropagate[{{weak, 0.01}}][opI[1, "x"]]],
    "propagator_strong_10ms" :> m[NPropagationOperator[{{ham, 0.01}}]],
    "rotation_then_free" :> m[NPropagate[{pi2x, {ham, 0.004}}][opI[1, "z"]]],
    "spin_echo" :> m[NPropagate[{pi2x, {weak, 0.004}, piy, {weak, 0.004}}][opI[1, "z"]]],
    "two_different_delays" :> m[NPropagate[{pi2x, {ham, 0.002}, {weak, 0.005}}][opI[1, "z"]]],
    (* DIVERGENCES.md, D2: the option Chronology is ignored; only $Chronology changes the order *)
    "chronology_option_is_ignored" :>
     m[NPropagate[{pi2x, {ham, 0.002}, {weak, 0.005}}, Chronology -> "RightToLeft"][opI[1, "z"]]],
    "chronology_option_is_ignored_propagator" :>
     m[NPropagationOperator[{{ham, 0.002}, {weak, 0.005}}, Chronology -> "RightToLeft"]],
    "chronology_option_is_ignored_amplitude" :>
     TransformationAmplitude[opI[1, "z"] -> opI[1, "y"], {pi2x, {ham, 0.002}, {weak, 0.005}}, Chronology -> "RightToLeft"],
    "propagator_two_delays" :> m[NPropagationOperator[{{ham, 0.002}, {weak, 0.005}}]],
    "amplitudes_after_rotation_and_delay" :>
     TransformationAmplitude[opI[1, "z"] -> targets, {pi2x, {weak, 0.004}}],
    "amplitudes_with_background" :>
     TransformationAmplitude[opI[1, "z"] -> targets, {pi2x, {None, 0.004}}, BackgroundGenerator -> weak],
    "amplitude_spin_echo" :>
     TransformationAmplitude[opI[1, "z"] -> opI[1, "y"], {pi2x, {weak, 0.004}, piy, {weak, 0.004}}]
    }|>,
 <|"group" -> "two_half_right_to_left", "setup" :> (SetSpinSystem[2]; $Chronology = "RightToLeft"), "cases" -> {
    "two_different_delays" :> m[NPropagate[{pi2x, {ham, 0.002}, {weak, 0.005}}][opI[1, "z"]]],
    "spin_echo" :> m[NPropagate[{{weak, 0.004}, piy, {weak, 0.004}, pi2x}][opI[1, "z"]]],
    "propagator_two_delays" :> m[NPropagationOperator[{{ham, 0.002}, {weak, 0.005}}]],
    "amplitude" :> TransformationAmplitude[opI[1, "z"] -> opI[1, "y"], {pi2x, {ham, 0.002}, {weak, 0.005}}],
    "option_left_to_right_is_ignored" :>
     m[NPropagate[{pi2x, {ham, 0.002}, {weak, 0.005}}, Chronology -> "LeftToRight"][opI[1, "z"]]]
    }|>,
 <|"group" -> "two_half_relaxation", "setup" :> SetSpinSystem[2], "cases" -> {
    "relaxation_superoperator" :> SuperoperatorMatrixRepresentation[relax],
    "relaxation_superoperator_per_spin" :>
     SuperoperatorMatrixRepresentation[PhenomenologicalRelaxationSuperoperator[{{1, 2., 0.2}, {2, 1., 0.5}}]],
    "relaxation_superoperator_single_time" :>
     SuperoperatorMatrixRepresentation[PhenomenologicalRelaxationSuperoperator[0.5]],
    "liouvillian" :> SuperoperatorMatrixRepresentation[Liouvillian[ham, relax]],
    "decay_I1x_50ms" :> m[NPropagate[{{ham, 0.05}}, BackgroundGenerator -> relax][opI[1, "x"]]],
    "decay_I1z_500ms" :> m[NPropagate[{{relax, 0.5}}][opI[1, "z"]]],
    "amplitudes_with_relaxation" :>
     TransformationAmplitude[opI[1, "z"] -> targets, {pi2x, {weak, 0.02}}, BackgroundGenerator -> relax]
    }|>,
 <|"group" -> "one_half_pulses", "setup" :> SetSpinSystem[1], "cases" -> {
    "pulse_90x_on_Iz" :> TransformationAmplitude[opI[1, "z"] -> {opI[1, "x"], opI[1, "y"], opI[1, "z"]}, {Pulse[{Pi/2, "x"}]}],
    "pulse_90y_on_Iz" :> TransformationAmplitude[opI[1, "z"] -> {opI[1, "x"], opI[1, "y"], opI[1, "z"]}, {Pulse[{Pi/2, "y"}]}],
    "pulse_90_phase_pi4_on_Iz" :> TransformationAmplitude[opI[1, "z"] -> {opI[1, "x"], opI[1, "y"], opI[1, "z"]}, {Pulse[{Pi/2, Pi/4}]}],
    "pulse_finite_off_resonance" :> TransformationAmplitude[opI[1, "z"] -> {opI[1, "x"], opI[1, "y"], opI[1, "z"]},
      {Pulse[{Pi/2, "x"}, NutationFrequency -> 2 Pi 1000]}, BackgroundGenerator -> 2 Pi 300 opI[1, "z"]],
    "pulse_delay_sequence" :> TransformationAmplitude[opI[1, "z"] -> {opI[1, "x"], opI[1, "y"], opI[1, "z"]},
      {Pulse[{Pi/2, "x"}], Delay[0.001], Pulse[{Pi, "y"}], Delay[0.001]}, BackgroundGenerator -> 2 Pi 300 opI[1, "z"]],
    "repeated_pulse" :> TransformationAmplitude[opI[1, "z"] -> {opI[1, "x"], opI[1, "y"], opI[1, "z"]},
      {Repeat[{Pulse[{Pi/6, "x"}]}, 3]}],
    "symbolic_flip_angle" :> Symbolic[TransformationAmplitude[opI[1, "z"] -> {opI[1, "x"], opI[1, "y"], opI[1, "z"]},
       {RotationSuperoperator[{1}, {b3, a3}]}], {a3, b3}]
    }|>
 }
