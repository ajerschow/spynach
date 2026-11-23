(* Rotations: Wigner matrices, Clebsch-Gordan coefficients, Euler matrices, rotation operators and
   rotated operators. The angles of the symbolic cases are a3, b3, c3. *)
m[op_] := MatrixRepresentation[op];
angles = {Pi/5, Pi/3, 2 Pi/7};
cgTable[j1_, j2_] := Flatten[Table[{j1, m1, j2, m2, j3, m1 + m2, CG[{j3, m1 + m2}, {j1, m1}, {j2, m2}]},
    {j3, Abs[j1 - j2], j1 + j2}, {m1, j1, -j1, -1}, {m2, j2, -j2, -1}], 2];
rotationOperators[spins_] := {
   "R_x_90" :> m[opR[spins, {Pi/2, "x"}]], "R_y_90" :> m[opR[spins, {Pi/2, "y"}]],
   "R_minus_x_90" :> m[opR[spins, {Pi/2, "-x"}]], "R_z_60" :> m[opR[spins, {Pi/3, "z"}]],
   "R_phase_pi4_120" :> m[opR[spins, {2 Pi/3, Pi/4}]],
   "R_polar" :> m[opR[spins, {Pi/3, {Pi/4, Pi/5}}]],
   "R_euler" :> m[opR[spins, angles]],
   "R_flip_phase_symbolic" :> Symbolic[m[opR[spins, {b3, a3}]], {a3, b3}],
   "R_euler_symbolic" :> Symbolic[m[opR[spins, {a3, b3, c3}]], {a3, b3, c3}]};
(* HoldAll: the operator must be built inside the case, under the spin system of its group *)
SetAttributes[rotated, HoldAll];
rotated[spins_, op_, name_String] := {
   (name <> "_by_x_90") :> m[RotationSuperoperator[spins, {Pi/2, "x"}][op]],
   (name <> "_by_y_90") :> m[RotationSuperoperator[spins, {Pi/2, "y"}][op]],
   (name <> "_by_z_60") :> m[RotationSuperoperator[spins, {Pi/3, "z"}][op]],
   (name <> "_by_euler") :> m[RotationSuperoperator[spins, angles][op]],
   (name <> "_by_flip_symbolic") :> Symbolic[m[RotationSuperoperator[spins, {b3, "x"}][op]], {b3}]};
{
 (* WignerD::mod only says that SpinDynamica extends the built-in WignerD *)
 <|"group" -> "wigner", "setup" :> SetSpinSystem[1], "allow" -> {"WignerD::mod"}, "cases" -> {
    "d_half_symbolic" :> Symbolic[Wignerd[1/2][b3], {b3}],
    "d_1_symbolic" :> Symbolic[Wignerd[1][b3], {b3}],
    "d_3half_symbolic" :> Symbolic[Wignerd[3/2][b3], {b3}],
    "d_2_symbolic" :> Symbolic[Wignerd[2][b3], {b3}],
    "d_2_element_1_0_symbolic" :> Symbolic[Wignerd[2, {1, 0}][b3], {b3}],
    "d_2_row_1" :> Wignerd[2, {1}][Pi/3],
    "d_2_column_1" :> Wignerd[2, {{1}}][Pi/3],
    "D_half" :> WignerD[1/2][angles],
    "D_1" :> WignerD[1][angles],
    "D_3half" :> WignerD[3/2][angles],
    "D_2" :> WignerD[2][angles],
    "D_2_element_1_m2" :> WignerD[2, {1, -2}][angles],
    "D_1_product_of_two" :> WignerD[1][{angles, {Pi/7, Pi/4, Pi/9}}],
    "D_1_numeric" :> WignerD[1][{0.3, 1.1, -0.7}]
    }|>,
 <|"group" -> "clebsch_gordan", "setup" :> SetSpinSystem[1], "cases" -> {
    "half_half" :> cgTable[1/2, 1/2], "one_half" :> cgTable[1, 1/2], "one_one" :> cgTable[1, 1],
    "three_half_one" :> cgTable[3/2, 1], "two_one" :> cgTable[2, 1], "two_two" :> cgTable[2, 2],
    "short_form" :> CG[{2}, {1, 1}, {1, -1}],
    "wrong_projection" :> CG[{1, 1}, {1/2, 1/2}, {1/2, -1/2}]
    }|>,
 <|"group" -> "euler", "setup" :> SetSpinSystem[1], "cases" -> {
    "matrix_symbolic" :> Symbolic[EulerRotationMatrix[{a3, b3, c3}], {a3, b3, c3}],
    "matrix" :> EulerRotationMatrix[angles],
    "rotate_vector" :> RotateEuler[{1, 2, 3}, angles]
    }|>,
 <|"group" -> "one_half", "setup" :> SetSpinSystem[1], "cases" -> Join[rotationOperators[1],
    rotated[{1}, opI[1, "z"], "I1z"], rotated[{1}, opI[1, "x"], "I1x"], rotated[{1}, opI[1, "+"], "I1plus"]]|>,
 <|"group" -> "two_half", "setup" :> SetSpinSystem[2], "cases" -> Join[rotationOperators[{1, 2}],
    {"R_spin1_only_x_90" :> m[opR[1, {Pi/2, "x"}]], "R_all_x_90" :> m[opR[{Pi/2, "x"}]]},
    rotated[{1}, opI[1, "z"] . opI[2, "z"], "I1zI2z_spin1"],
    rotated[{1, 2}, opI[1, "x"] . opI[2, "z"], "I1xI2z_both"],
    rotated[{1, 2}, opT[{1, 2}, {2, 1}], "T21_both"]]|>,
 <|"group" -> "spin1", "setup" :> SetSpinSystem[{{1, 1}}], "cases" -> Join[rotationOperators[1],
    rotated[{1}, opI[1, "z"], "I1z"], rotated[{1}, opT[1, {2, 0}], "T20"], rotated[{1}, opT[1, {2, 2}], "T22"]]|>,
 <|"group" -> "spin32", "setup" :> SetSpinSystem[{{1, 3/2}}], "cases" -> Join[rotationOperators[1],
    rotated[{1}, opI[1, "z"], "I1z"]]|>
 }
