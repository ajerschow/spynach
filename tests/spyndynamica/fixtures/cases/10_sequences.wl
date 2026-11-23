(* Pulse sequence objects, shaped pulses, stored propagators, normalisation, time origin, two-dimensional FT. *)
xyz := {opI[1, "x"], opI[1, "y"], opI[1, "z"]};
offset := 2 Pi 300 opI[1, "z"];
echo := {Pulse[{Pi/2, "x"}], Delay[0.001], Pulse[{Pi, "y"}], Delay[0.001]};
rf := Function[t, 2 Pi 50 (opI[1, "x"] Cos[2 Pi 300 t] + opI[1, "y"] Sin[2 Pi 300 t]) + 2 Pi 280 opI[1, "z"]];
amp[events_, opts___] := TransformationAmplitude[opI[1, "z"] -> xyz, events, opts];
grid2 = Table[N[Exp[-(j + 2 k)/5] Exp[I (j - k)]], {j, 0, 3}, {k, 0, 3}];
{
 <|"group" -> "one_half", "setup" :> SetSpinSystem[1], "allow" -> {"FrontEndObject::notavail"}, "cases" -> {
    "sequence" :> amp[PulseSequence[echo], BackgroundGenerator -> offset],
    "sequence_in_a_list" :> amp[{PulseSequence[echo], Pulse[{Pi/3, "x"}]}, BackgroundGenerator -> offset],
    "sequence_phase_shift" :> amp[PulseSequence[echo, PhaseShift -> Pi/2], BackgroundGenerator -> offset],
    "sequence_total_duration" :> amp[PulseSequence[{1}, 4/1000, {Pulse[{Pi/2, "x"}], Delay[{Automatic, 1}],
        Pulse[{Pi/3, "y"}], Delay[{Automatic, 2}]}], BackgroundGenerator -> offset],
    "sequence_one_automatic_delay" :> amp[PulseSequence[{1}, 4/1000, {Pulse[{Pi/2, "x"}], Delay[0.001],
        Pulse[{Pi/3, "y"}], Delay[Automatic]}], BackgroundGenerator -> offset],
    "sequence_repeated" :> amp[Repeat[PulseSequence[{Pulse[{Pi/5, "x"}], Delay[0.0005]}], 3], BackgroundGenerator -> offset],
    "pulse_phase_shift" :> amp[{Pulse[{Pi/3, "x"}, PhaseShift -> Pi/2]}],
    "pulse_amplitude_factor" :> amp[{Pulse[{Pi/2, "x"}, AmplitudeFactor -> 0.9]}],
    "pulse_amplitude_factor_finite" :> amp[{Pulse[{Pi/2, "x"}, AmplitudeFactor -> 0.9, NutationFrequency -> 2 Pi 1000]},
      BackgroundGenerator -> offset],
    "shaped_pulse_gaussian" :> amp[{ShapedPulse[1, 0.01, {tau, {2 Pi 80 Exp[-(tau/0.002)^2], 0, 0}}]}],
    "shaped_pulse_phase_and_frequency" :> amp[{ShapedPulse[1, 0.01, {tau, {2 Pi 60, Pi/4, 2 Pi 40}}]},
      BackgroundGenerator -> 2 Pi 25 opI[1, "z"]],
    "precalculated" :> amp[{Precalculate[echo, BackgroundGenerator -> offset], Pulse[{Pi/3, "x"}]},
      BackgroundGenerator -> offset],
    "normalization_factor" :> amp[{Pulse[{Pi/2, "x"}]}, NormalizationFactor -> 2],
    "initial_time_point" :> amp[{{rf, 0.003}}, InitialTimePoint -> 0.001],
    "ft_two_dimensional" :> FT[grid2, EmbedCoordinates -> False],
    "ft_two_dimensional_no_zero_fill" :> FT[grid2, EmbedCoordinates -> False, ZeroFill -> False]
    }|>,
 <|"group" -> "two_half", "setup" :> SetSpinSystem[2],
  "allow" -> {"FrontEndObject::notavail", "Signal1D::droplast", "Signal1D::reportscm"}, "cases" -> {
    "filtration_event" :> TransformationAmplitude[opI["z"] -> {opI[1, "x"] . opI[2, "x"], opI[1, "z"] . opI[2, "z"], opI[1, "y"] . opI[2, "y"]},
      {Pulse[{Pi/2, "x"}], {2 Pi 10 opI[1, "z"] . opI[2, "z"], 0.025}, Pulse[{Pi/2, "x"}],
       CoherenceOrderFiltration[{-2, 2}], Pulse[{Pi/3, "y"}]}],
    "filtration_one_spin" :> TransformationAmplitude[opI["x"] -> {opI[1, "x"], opI[2, "x"], opI[1, "-"], opI[2, "+"]},
      {CoherenceOrderFiltration[{1}, {1}]}],
    "sequence_on_one_spin" :> TransformationAmplitude[opI["z"] -> {opI[1, "y"], opI[2, "y"], opI[1, "z"], opI[2, "z"]},
      PulseSequence[{2}, {Pulse[{Pi/2, "x"}], Delay[0.001]}], BackgroundGenerator -> 2 Pi 100 opI[2, "z"]],
    "signal_normalization_factor" :> N[Expand[Signal1D[{{2 Pi 200, 17}}, 2 Pi 30 opI[1, "z"], LineBroadening -> None,
        DigitalFrequencyResolution -> False, NormalizationFactor -> 2]]]
    }|>
 }
