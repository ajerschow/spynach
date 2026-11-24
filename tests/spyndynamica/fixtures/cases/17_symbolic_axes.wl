(* AxesToEuler with axes that contain symbols. SpinDynamica simplifies under the assumption that every symbol
   lies between 0 and Pi/2, so the angles are compared at points in that range: each case holds the angles at
   the two points below. The cases named ..._text hold the form of SpinDynamica's result. *)
zdir[t_, p_] := {Sin[t] Cos[p], Sin[t] Sin[p], Cos[t]};
frame := RotationMatrix[a, {0, 0, 1}] . RotationMatrix[b, {0, 1, 0}] . RotationMatrix[g, {0, 0, 1}];
points = {{th -> 3/10, ph -> 11/10, th2 -> 7/10, ph2 -> 2/5, a -> 1/5, b -> 9/10, g -> 13/10},
   {th -> 7/5, ph -> 1/4, th2 -> 1/2, ph2 -> 6/5, a -> 11/10, b -> 2/5, g -> 3/10}};
SetAttributes[at, HoldAll];
at[expr_] := N[expr /. points];
form[expr_] := ToString[expr, InputForm];
{
 <|"group" -> "symbolic_axes", "setup" :> SetSpinSystem[1], "cases" -> {
    "z_axis" :> at[AxesToEuler[{zdir[th, ph]}]],
    "z_axis_text" :> form[AxesToEuler[{zdir[th, ph]}]],
    "x_and_z_axes" :> at[AxesToEuler[{{Cos[a], Sin[a], 0}, {0, 0, 1}}]],
    "x_and_z_axes_text" :> form[AxesToEuler[{{Cos[a], Sin[a], 0}, {0, 0, 1}}]],
    "tilted_about_y" :> at[AxesToEuler[{{Cos[th], 0, -Sin[th]}, {Sin[th], 0, Cos[th]}}]],
    "tilted_about_y_text" :> form[AxesToEuler[{{Cos[th], 0, -Sin[th]}, {Sin[th], 0, Cos[th]}}]],
    "two_systems" :> at[AxesToEuler[{zdir[th, ph]}, {zdir[th2, ph2]}]],
    "laboratory_to_z_axis" :> at[AxesToEuler[{{0, 0, 1}}, {zdir[th, ph]}]],
    "laboratory_to_z_axis_text" :> form[AxesToEuler[{{0, 0, 1}}, {zdir[th, ph]}]],
    "symbolic_and_numeric" :> at[AxesToEuler[{{Sin[th], 0, Cos[th]}}, {{0, 1, 0}, {0, 0, 1}}]],
    "three_axes" :> at[AxesToEuler[frame]],
    "three_axes_text" :> form[AxesToEuler[frame]],
    "three_axes_transposed" :> at[AxesToEuler[Transpose[frame]]],
    "three_axes_transposed_text" :> form[AxesToEuler[Transpose[frame]]],
    "not_normalised" :> at[AxesToEuler[{{3 Cos[a], 3 Sin[a], 0}, {0, 0, 2 + th}}]],
    "axis_system_of_z" :> at[AxisSystem[zdir[th, ph]]],
    "axis_system_of_x_and_z" :> at[AxisSystem[{1, th, 0}, zdir[th, ph]]]
    }|>
}
