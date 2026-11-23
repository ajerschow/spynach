(* Time-dependent generators (integrated numerically), ensemble averages and spherical tensors. *)
m[op_] := MatrixRepresentation[op];
xyz := {opI[1, "x"], opI[1, "y"], opI[1, "z"]};
rf := Function[t, 2 Pi 50 (opI[1, "x"] Cos[2 Pi 300 t] + opI[1, "y"] Sin[2 Pi 300 t]) + 2 Pi 280 opI[1, "z"]];
chirp := Function[t, 2 Pi 40 opI[1, "x"] + 2 Pi (-200 + 40000 t) opI[1, "z"]];
(* two spins under a rotation of 1 kHz: modulated shifts and a modulated homonuclear dipolar coupling *)
spinning := PeriodicFunction[t, 1/1000,
   2 Pi (300 + 400 Cos[2 Pi 1000 t] + 150 Cos[4 Pi 1000 t + 1/2]) opI[1, "z"] +
    2 Pi (-200 + 250 Cos[2 Pi 1000 t + 1]) opI[2, "z"] +
    2 Pi (120 Cos[2 Pi 1000 t + 3/10] + 60 Cos[4 Pi 1000 t]) (3 opI[1, "z"] . opI[2, "z"] - opI[1] . opI[2])];
elements[t_] := List @@ t /. {SphericalTensor[_, e_List, ___] :> e};
flat[set_] := Join[First[#], {Last[#]}] & /@ set;
cart = {{1, 2, 3}, {4, 5, 6}, {7, 8, 10}};
angles = {Pi/5, Pi/3, 2 Pi/7};
tabulated := Select[Ensemble`OrientationalSampling`Private`DefinedOrientationalSamplingSchemes,
   StringStartsQ[#, "Leboct" | "POLYTOPE" | "REPULSION" | "ZCW"] &];
(* FrontEndObject::notavail: the progress display of ensemble averages needs the notebook front end. Signal1D
   with EnsembleAverage returns zeros without it, so no ensemble-averaged signal is stored here. *)
info = {"WignerD::mod", "FrontEndObject::notavail"};
symmetric := CartesianToSpherical[2, {{1, 2, 3}, {2, 5, 6}, {3, 6, 10}}];
{
 <|"group" -> "one_half", "setup" :> SetSpinSystem[1], "allow" -> info, "cases" -> {
    "rotating_field" :> m[NPropagate[{{rf, 0.004}}][opI[1, "z"]]],
    "rotating_field_amplitudes" :> TransformationAmplitude[opI[1, "z"] -> xyz, {{rf, 0.004}}],
    "rotating_field_after_a_delay" :>
     TransformationAmplitude[opI[1, "z"] -> xyz, {{2 Pi 100 opI[1, "z"], 0.001}, {rf, 0.003}}],
    "rotating_field_on_background" :>
     TransformationAmplitude[opI[1, "z"] -> xyz,
      {{Function[t, 2 Pi 50 (opI[1, "x"] Cos[2 Pi 300 t] + opI[1, "y"] Sin[2 Pi 300 t])], 0.004}},
      BackgroundGenerator -> 2 Pi 280 opI[1, "z"]],
    "chirp" :> TransformationAmplitude[opI[1, "z"] -> xyz, {{chirp, 0.01}}],
    "average_over_offsets" :> TransformationAmplitude[opI[1, "z"] -> xyz,
      {RotationSuperoperator[{Pi/2, "x"}], {offset opI[1, "z"], 0.002}},
      EnsembleAverage -> {offset, {2 Pi 50, 2 Pi 100, 2 Pi 200}}],
    "weighted_average_over_offsets" :> TransformationAmplitude[opI[1, "z"] -> xyz,
      {RotationSuperoperator[{Pi/2, "x"}], {offset opI[1, "z"], 0.002}},
      EnsembleAverage -> {offset, {{{2 Pi 50}, 1/2}, {{2 Pi 100}, 1/3}, {{2 Pi 200}, 1/6}}}]
    }|>,
 <|"group" -> "periodic", "setup" :> SetSpinSystem[2],
  "allow" -> {"Signal1D::droplast", "Signal1D::reportscm", "Signal1D::autolb", "Signal1D::matchsampling"}, "cases" -> {
    "compute_fid" :> N[Expand[Signal1D[{{2 Pi 8000, 65}}, spinning, LineBroadening -> 2 Pi 5,
        DigitalFrequencyResolution -> False, SignalCalculationMethod -> "COMPUTE"]]],
    "compute_peaks" :> Signal1D[{{2 Pi 8000, 65}}, spinning, LineBroadening -> 2 Pi 5,
        DigitalFrequencyResolution -> False, SignalCalculationMethod -> "COMPUTE"][[2, 2]],
    "direct_fid" :> N[Expand[Signal1D[{{2 Pi 8000, 65}}, spinning, LineBroadening -> 2 Pi 5,
        SignalCalculationMethod -> "Direct"]]],
    "automatic_method_fid" :> N[Expand[Signal1D[{{2 Pi 8000, 65}}, spinning, LineBroadening -> 2 Pi 5,
        DigitalFrequencyResolution -> False]]]
    }|>,
 <|"group" -> "tensors", "setup" :> SetSpinSystem[1], "allow" -> info, "cases" -> {
    "rank_2_rotated" :> elements[TransformSphericalTensor[symmetric, angles]],
    "rank_1_rotated" :> elements[TransformSphericalTensor[CartesianToSpherical[1, {1, 2, 3}], angles]],
    "rank_2_symmetric" :> elements[symmetric],
    "rank_2_rotated_twice" :>
     elements[TransformSphericalTensor[symmetric, {angles, {Pi/7, Pi/4, Pi/9}}]],
    "cartesian_to_rank_0" :> elements[CartesianToSpherical[0, cart]],
    "cartesian_to_rank_1" :> elements[CartesianToSpherical[1, cart]],
    "cartesian_to_rank_2" :> elements[CartesianToSpherical[2, cart]],
    "vector_to_rank_1" :> elements[CartesianToSpherical[1, {1, 2, 3}]],
    "rank_2_to_cartesian" :> SphericalToCartesian[CartesianToSpherical[2, cart]],
    "rank_1_to_cartesian" :> SphericalToCartesian[CartesianToSpherical[1, cart]],
    "rank_0_to_cartesian" :> SphericalToCartesian[CartesianToSpherical[0, cart]]
    }|>,
 <|"group" -> "orientations", "setup" :> SetSpinSystem[1], "cases" ->
   (With[{name = #}, name :> flat[OrientationsAndWeights[name]]] & /@
     {"Leboct10", "Leboct16", "Leboct19", "Leboct22", "Leboct31", "Leboct37", "Leboct46", "Leboct85", "POLYTOPE12",
      "POLYTOPE60", "REPULSION100", "REPULSION150", "REPULSION168", "REPULSION232", "REPULSION376", "REPULSION700",
      "ZCW50", "ZCW144", "ZCW200", "ZCW300", "ZCW538", "ZCW1154", "ZCW6044"})|>
 }
