(* The carousel average of the COMPUTE method (CarouselAverage, "gamma-COMPUTE") and signals under periodic
   Hamiltonians with relaxation, where COMPUTE works in Liouville space. Cases named ..._is_rejected hold True
   when SpinDynamica refuses the call. The group "normalization" records the first point of signals that
   start at 1 by the definition of Signal1D: SpinDynamica's COMPUTE method normalises differently from its
   other methods (see DIVERGENCES.md, D13). *)
(* two spins under a rotation of 1 kHz: modulated shifts and a modulated homonuclear dipolar coupling *)
spinning := PeriodicFunction[t, 1/1000,
   2 Pi (300 + 400 Cos[2 Pi 1000 t] + 150 Cos[4 Pi 1000 t + 1/2]) opI[1, "z"] +
    2 Pi (-200 + 250 Cos[2 Pi 1000 t + 1]) opI[2, "z"] +
    2 Pi (120 Cos[2 Pi 1000 t + 3/10] + 60 Cos[4 Pi 1000 t]) (3 opI[1, "z"] . opI[2, "z"] - opI[1] . opI[2])];
relax := PhenomenologicalRelaxationSuperoperator[{{1, 0.05, 0.004}, {2, 0.03, 0.006}}];
(* with relaxation the periodic generator is the Liouvillian: SpinDynamica fails when the relaxation
   superoperator is the BackgroundGenerator of a time-dependent Hamiltonian, and its "Direct" method fails
   for a periodic Liouvillian (both with messages of the kernel, so neither is stored) *)
relaxing := PeriodicFunction[t, 1/1000, Liouvillian[spinning[[3]], relax]];
rejected[result_] := result === $Failed;
SetAttributes[{fid, peaks}, HoldRest];
first[opts___] := N[Expand[Signal1D[{{2 Pi 8000, 16}},
      PeriodicFunction[t, 1/1000, 2 Pi (300 + 400 Cos[2 Pi 1000 t]) opI["z"]], opts,
      LineBroadening -> None, DigitalFrequencyResolution -> False]]][[1, 2]];
fid[gen_, opts___] := N[Expand[Signal1D[{{2 Pi 8000, 65}}, gen, opts, DigitalFrequencyResolution -> False]]];
peaks[gen_, opts___] := Signal1D[{{2 Pi 8000, 65}}, gen, opts, DigitalFrequencyResolution -> False][[2, 2]];
{
 <|"group" -> "carousel", "setup" :> SetSpinSystem[2],
  "allow" -> {"Signal1D::droplast", "Signal1D::reportscm", "Signal1D::autolb", "Signal1D::matchsampling",
    "Signal1D::nocarav"}, "cases" -> {
    "fid" :> fid[spinning, LineBroadening -> 2 Pi 5,
       SignalCalculationMethod -> "COMPUTE", CarouselAverage -> True],
    "peaks" :> peaks[spinning, LineBroadening -> 2 Pi 5,
       SignalCalculationMethod -> "COMPUTE", CarouselAverage -> True],
    "fid_from_I1x" :> fid[spinning, LineBroadening -> 2 Pi 5,
       SignalCalculationMethod -> "COMPUTE", CarouselAverage -> True,
       InitialDensityOperator -> opI[1, "x"], Preparation -> None],
    "fid_observing_spin_2" :> fid[spinning, LineBroadening -> 2 Pi 5,
       SignalCalculationMethod -> "COMPUTE", CarouselAverage -> True, Observable -> opI[2, "-"]],
    "fid_in_liouville_space" :> fid[spinning, LineBroadening -> 2 Pi 5,
       SignalCalculationMethod -> "COMPUTE", CarouselAverage -> True, UseHilbertSpaceWhenPossible -> False],
    "direct_method_is_rejected" :> rejected[Signal1D[{{2 Pi 8000, 65}}, spinning,
       SignalCalculationMethod -> "Direct", CarouselAverage -> True]]
    }|>,
 <|"group" -> "relaxation", "setup" :> SetSpinSystem[2],
  "allow" -> {"Signal1D::droplast", "Signal1D::reportscm", "Signal1D::autolb", "Signal1D::matchsampling"}, "cases" -> {
    "compute_fid" :> fid[relaxing, LineBroadening -> None, SignalCalculationMethod -> "COMPUTE"],
    "compute_peaks" :> peaks[relaxing, LineBroadening -> None, SignalCalculationMethod -> "COMPUTE"],
    "automatic_method_fid" :> fid[relaxing, LineBroadening -> None],
    "carousel_fid" :> fid[relaxing, LineBroadening -> None,
       SignalCalculationMethod -> "COMPUTE", CarouselAverage -> True],
    "carousel_peaks" :> peaks[relaxing, LineBroadening -> None,
       SignalCalculationMethod -> "COMPUTE", CarouselAverage -> True]
    }|>,
 <|"group" -> "normalization", "setup" :> SetSpinSystem[4],
  "allow" -> {"Signal1D::droplast", "Signal1D::reportscm", "Signal1D::autolb", "Signal1D::matchsampling"}, "cases" -> {
    "four_spins_direct" :> first[SignalCalculationMethod -> "Direct"],
    "four_spins_compute" :> first[SignalCalculationMethod -> "COMPUTE"],
    "four_spins_compute_in_liouville_space" :>
     first[SignalCalculationMethod -> "COMPUTE", UseHilbertSpaceWhenPossible -> False],
    "four_spins_compute_observing_spin_1" :> first[SignalCalculationMethod -> "COMPUTE", Observable -> opI[1, "-"]],
    "four_spins_direct_observing_spin_1" :> first[SignalCalculationMethod -> "Direct", Observable -> opI[1, "-"]]
    }|>
}
