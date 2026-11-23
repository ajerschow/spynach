(* Signals: Signal1D by diagonalisation and by direct propagation, the Fourier transform, line broadening and
   the lineshape functions. Frequencies in rad/s, times in s. DigitalFrequencyResolution -> False keeps the
   exact peak frequencies; the default moves every peak to the nearest point of the frequency grid. *)
ham := 2 Pi 30 opI[1, "z"] - 2 Pi 10 opI[2, "z"] + 2 Pi 12 opI[1] . opI[2];
abx := 2 Pi 30 opI[1, "z"] - 2 Pi 10 opI[2, "z"] + 2 Pi 70 opI[3, "z"] + 2 Pi 12 opI[1] . opI[2] +
   2 Pi 5 opI[1, "z"] . opI[3, "z"] + 2 Pi 3 opI[2, "z"] . opI[3, "z"];
sw = 2 Pi 200;
lb = 2 Pi 2;
exact[timing_, gen_, opts___] := Signal1D[timing, gen, opts, DigitalFrequencyResolution -> False];
timingOf[sig_] := N[First[sig]];
peaks[sig_] := sig[[2, 2]];
fid[sig_] := N[Expand[sig]];
ws = {-3, -1/2, 0, 3/10, 2};
info = {"Signal1D::droplast", "Signal1D::reportscm", "Signal1D::autolb"};
{
 <|"group" -> "two_half", "setup" :> SetSpinSystem[2], "allow" -> info, "cases" -> {
    "timing_sw_16" :> timingOf[exact[{{sw, 16}}, ham, LineBroadening -> lb]],
    "timing_sw_17" :> timingOf[exact[{{sw, 17}}, ham, LineBroadening -> lb]],
    "timing_range_step_16_points" :> timingOf[exact[{0, 75/1000, 1/200}, ham, LineBroadening -> lb]],
    "timing_range_step_17_points" :> timingOf[exact[{0, 80/1000, 1/200}, ham, LineBroadening -> lb]],
    "timing_range_count_16" :> timingOf[exact[{0, 80/1000, {16}}, ham, LineBroadening -> lb]],
    "points_sw_16" :> Length[fid[exact[{{sw, 16}}, ham, LineBroadening -> lb]]],
    "points_sw_17" :> Length[fid[exact[{{sw, 17}}, ham, LineBroadening -> lb]]],
    "points_sw_16_not_forced" :> Length[fid[exact[{{sw, 16}}, ham, LineBroadening -> lb, ForceEvenNumberOfPoints -> False]]],
    "ab_peaks" :> peaks[exact[{{sw, 17}}, ham, LineBroadening -> lb]],
    "ab_fid" :> fid[exact[{{sw, 17}}, ham, LineBroadening -> lb]],
    "ab_fid_direct" :> fid[Signal1D[{{sw, 17}}, ham, LineBroadening -> lb, SignalCalculationMethod -> "Direct"]],
    "ab_spectrum" :> N[FT[exact[{{sw, 17}}, ham, LineBroadening -> lb]]],
    "ab_spectrum_no_zero_fill" :> N[FT[exact[{{sw, 17}}, ham, LineBroadening -> lb], ZeroFill -> False]],
    "ab_peaks_snapped_to_grid" :> peaks[Signal1D[{{sw, 17}}, ham, LineBroadening -> lb]],
    "ab_peaks_automatic_broadening" :> peaks[exact[{{sw, 17}}, ham]],
    "ab_peaks_no_broadening" :> peaks[exact[{{sw, 17}}, ham, LineBroadening -> None]],
    "ab_peaks_observable_spin_1" :> peaks[exact[{{sw, 17}}, ham, LineBroadening -> lb, Observable -> {1}]],
    "ab_peaks_observable_I1x" :> peaks[exact[{{sw, 17}}, ham, LineBroadening -> lb, Observable -> opI[1, "x"]]],
    "ab_peaks_from_I1x_without_preparation" :>
     peaks[exact[{{sw, 17}}, ham, LineBroadening -> lb, InitialDensityOperator -> opI[1, "x"], Preparation -> None]],
    "ab_fid_background_generator" :> fid[exact[{{sw, 17}}, None, LineBroadening -> lb, BackgroundGenerator -> ham]],
    "ab_fid_with_relaxation" :> fid[exact[{{sw, 17}}, ham, LineBroadening -> None,
       BackgroundGenerator -> PhenomenologicalRelaxationSuperoperator[1., 0.05]]],
    "ab_fid_with_relaxation_direct" :> fid[Signal1D[{{sw, 17}}, ham, LineBroadening -> None,
       BackgroundGenerator -> PhenomenologicalRelaxationSuperoperator[1., 0.05], SignalCalculationMethod -> "Direct"]]
    }|>,
 <|"group" -> "one_half", "setup" :> SetSpinSystem[1], "allow" -> info, "cases" -> {
    "peaks" :> peaks[exact[{{sw, 17}}, 2 Pi 30 opI[1, "z"], LineBroadening -> lb]],
    "fid" :> fid[exact[{{sw, 17}}, 2 Pi 30 opI[1, "z"], LineBroadening -> lb]],
    "fid_start_time" :> fid[exact[{1/100, {sw, 17}}, 2 Pi 30 opI[1, "z"], LineBroadening -> lb]]
    }|>,
 <|"group" -> "three_half", "setup" :> SetSpinSystem[3], "allow" -> info, "cases" -> {
    "abx_fid" :> fid[exact[{{2 Pi 400, 33}}, abx, LineBroadening -> lb]],
    "abx_fid_direct" :> fid[Signal1D[{{2 Pi 400, 33}}, abx, LineBroadening -> lb, SignalCalculationMethod -> "Direct"]],
    "abx_number_of_peaks" :> Length[peaks[exact[{{2 Pi 400, 33}}, abx, LineBroadening -> lb]]]
    }|>,
 <|"group" -> "fourier", "setup" :> SetSpinSystem[1], "allow" -> {"ApplyLineBroadening::autolb"}, "cases" -> {
    "list" :> FT[{1., 0.5, 0.25, 0.125}, EmbedCoordinates -> False],
    "list_no_zero_fill" :> FT[{1., 0.5, 0.25, 0.125}, EmbedCoordinates -> False, ZeroFill -> False],
    "list_first_point_1" :>
     FT[{1., 0.5, 0.25, 0.125}, EmbedCoordinates -> False, ZeroFill -> False, FirstPointMultiplicationFactor -> 1],
    "list_zero_fill_4" :> FT[{1., 0.5, 0.25, 0.125}, EmbedCoordinates -> False, ZeroFill -> 4],
    "list_odd_length" :> FT[{1., 0.5 I, -0.25, -0.125 I, 0.0625}, EmbedCoordinates -> False, ZeroFill -> False],
    "pairs" :> FT[{{0, 1.}, {0.1, 0.5 I}, {0.2, -0.25}, {0.3, -0.125 I}}],
    "pairs_no_zero_fill" :> FT[{{0, 1.}, {0.1, 0.5 I}, {0.2, -0.25}, {0.3, -0.125 I}}, ZeroFill -> False],
    "signal_data" :> N[FT[Signal[{0, 3/10, 1/10}, {1., 0.5 I, -0.25, -0.125 I}, LineBroadening -> None]]],
    "signal_data_broadened" :> N[Expand[Signal[{0, 3/10, 1/10}, {1., 0.5 I, -0.25, -0.125 I}, LineBroadening -> 2 Pi 1]]],
    "signal_data_automatic_broadening" :> N[Expand[Signal[{0, 3/10, 1/10}, {1., 1., 1., 1.}]]],
    "apply_line_broadening" :>
     N[Expand[ApplyLineBroadening[Signal[{0, 3/10, 1/10}, {1., 1., 1., 1.}, LineBroadening -> None], LineBroadening -> 2 Pi 1]]],
    "signal_peaks_expanded" :>
     N[Expand[Signal[{0, 3/10, 1/10}, {"Lorentzian", {{1 + I, 2 Pi 2, 3}, {1/2, -2 Pi, 1}}}, LineBroadening -> None]]]
    }|>,
 <|"group" -> "lineshapes", "setup" :> SetSpinSystem[1], "cases" -> {
    "lorentzian" :> (Lorentzian[#, {2, 1/2, 3/10}] & /@ ws),
    "lorentzian_complex_amplitude" :> (Lorentzian[#, {2 + I, 1/2, 3/10}] & /@ ws),
    "lorentzian_unit" :> (Lorentzian[#, {1/2, 3/10}] & /@ ws),
    "lorentzian_two_lines" :> (Lorentzian[#, {{2, 1/2, 3/10}, {1, -1, 1/5}}] & /@ ws),
    "absorption" :> (Absorption[#, {2, 1/2, 3/10}] & /@ ws),
    "dispersion" :> (Dispersion[#, {2, 1/2, 3/10}] & /@ ws),
    "gaussian" :> (Gaussian[#, {2, 1/2, 3/10}] & /@ ws),
    "gaussian_two_lines" :> (Gaussian[#, {{2, 1/2, 3/10}, {1, -1, 1/5}}] & /@ ws)
    }|>
 }
