(* Trajectories as functions of time: Trajectory returns a TrajectoryFunction, NPropagate with
   Continuous -> True an OperatorTrajectoryFunction. Each case holds the values at the times below. *)
m[op_] := MatrixRepresentation[op];
ham := 2 Pi 10 opI[1, "z"] - 2 Pi 4 opI[2, "z"] + 2 Pi 7 opI[1] . opI[2];
relax := PhenomenologicalRelaxationSuperoperator[0.5, 0.1];
events := {RotationSuperoperator[{Pi/2, "y"}], {ham, 0.1}, RotationSuperoperator[{Pi, "x"}], {ham, 0.15}};
times = {0, 0.013, 0.05, 0.1, 0.1001, 0.17, 0.25};
rf := Function[t, 2 Pi 20 (opI[1, "x"] Cos[2 Pi 30 t] + opI[1, "y"] Sin[2 Pi 30 t]) + 2 Pi 25 opI[1, "z"]];
{
 <|"group" -> "two_half", "setup" :> SetSpinSystem[2], "cases" -> {
    "times" :> times,
    "one_observable" :> N[Trajectory[opI[1, "z"] -> opI[1, "x"], events] /@ times],
    "several_observables" :>
     N[Through[Trajectory[opI[1, "z"] + opI[2, "z"] -> {opI[1, "x"], opI[2, "y"], opI[1, "z"]}, events][#]] & /@ times],
    "with_relaxation" :> N[Trajectory[opI[1, "z"] -> opI[1, "x"], events, BackgroundGenerator -> relax] /@ times],
    "time_range" :> N[First[Trajectory[opI[1, "z"] -> opI[1, "x"], events]]],
    "time_dependent" :> N[Trajectory[opI[1, "z"] -> opI[1, "z"], {{rf, 0.05}}] /@ {0, 0.01, 0.025, 0.05}],
    "operator" :> (Normal[N[m[NPropagate[events, Continuous -> True][opI[1, "z"]][#]]]] & /@ times),
    "operator_time_range" :> N[First[NPropagate[events, Continuous -> True][opI[1, "z"]]]]
    }|>
}
