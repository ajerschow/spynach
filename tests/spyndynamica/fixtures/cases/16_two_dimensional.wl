(* Two-dimensional data: FT with separate options for the two dimensions and with embedded coordinates,
   ReflectSpectrum, CombineComplementarySpectra and CombineData. SpinDynamica has no routine for a
   two-dimensional signal; the rows of a matrix belong to the indirect dimension (t1), the columns to the
   direct one (t2). Cases named ..._is_unevaluated hold True when SpinDynamica returns the call as it is,
   cases named ..._is_rejected when it refuses the call. The frequency coordinates that FT embeds in
   two-dimensional data are the frequencies divided by the product of the two sampling rates (DIVERGENCES.md,
   D14); the cases "sample_intervals..." and "triples" record that. *)
grid = Table[N[Exp[-(j + 2 k)/5] Exp[I (j - k)]], {j, 0, 3}, {k, 0, 5}];
other = Table[N[Exp[-(2 j + k)/7] Exp[I (j + 2 k)]], {j, 0, 3}, {k, 0, 5}];
triples = Flatten[Table[{j/10, k/20, grid[[j + 1, k + 1]]}, {j, 0, 3}, {k, 0, 5}], 1];
list1 = {1., 0.5 I, -0.25, -0.125 I, 0.3, 0.2};
pairs1 = Transpose[{Range[-2, 3]/6., list1}];
unevaluated[result_, head_] := Head[result] === head;
rejected[result_] := result === $Failed;
exact = Sequence[LineBroadening -> None, DigitalFrequencyResolution -> False];
peaksA := Signal[{0, 7/10, 1/10}, {"Lorentzian", {{1., 3., 0.5}, {0.5 I, -2., 0.25}}}, exact];
peaksB := Signal[{0, 7/10, 1/10}, {"Lorentzian", {{0.25, 1., 0.125}}}, exact];
dataA := Signal[{0, 3/10, 1/10}, {1., 0.5 I, -0.25, -0.125 I}, LineBroadening -> None];
dataB := Signal[{0, 3/10, 1/10}, {0.5, 0.25, 0.125, 0.0625}, LineBroadening -> None];
{
 <|"group" -> "transform", "setup" :> SetSpinSystem[1], "allow" -> {"CombineData::ambig", "CombineData::embcoordbaddim"}, "cases" -> {
    "grid" :> FT[grid, EmbedCoordinates -> False],
    "zero_fill_per_dimension" :> FT[grid, EmbedCoordinates -> False, ZeroFill -> {4, False}],
    "first_point_per_dimension" :>
     FT[grid, EmbedCoordinates -> False, ZeroFill -> False, FirstPointMultiplicationFactor -> {0.5, 1.}],
    "sample_intervals" :> FT[grid, SampleInterval -> {1/10, 1/20}, ZeroFill -> False],
    "sample_intervals_zero_filled" :> FT[grid, SampleInterval -> {1/10, 1/20}],
    "triples" :> FT[triples, ZeroFill -> False]
    }|>,
 <|"group" -> "reflection", "setup" :> SetSpinSystem[1], "allow" -> {"CombineData::ambig", "CombineData::embcoordbaddim"}, "cases" -> {
    "list" :> ReflectSpectrum[list1],
    "pairs" :> ReflectSpectrum[pairs1],
    "matrix" :> ReflectSpectrum[grid],
    "triples" :> ReflectSpectrum[triples],
    "odd_list_is_unevaluated" :> unevaluated[ReflectSpectrum[Most[list1]], ReflectSpectrum],
    "complementary" :> CombineComplementarySpectra[grid, other],
    "complementary_list_form_is_unevaluated" :>
     unevaluated[CombineComplementarySpectra[{grid, other}], CombineComplementarySpectra]
    }|>,
 <|"group" -> "combination", "setup" :> SetSpinSystem[1], "allow" -> {"CombineData::ambig", "CombineData::embcoordbaddim"}, "cases" -> {
    "matrices_weighted" :> CombineData[{grid, other}, {1/4, 3/4}],
    "matrices_mean" :> CombineData[{grid, other}],
    "lists" :> CombineData[{list1, Reverse[list1]}, {2, -1}],
    "pairs_with_coordinates_is_rejected" :>
     rejected[CombineData[{pairs1, ReflectSpectrum[pairs1]}, {1/2, 1/2}, EmbedCoordinates -> True]],
    "two_pairs_with_coordinates" :>
     CombineData[{pairs1[[;; 2]], pairs1[[;; 2]]}, {1/4, 1/4}, EmbedCoordinates -> True],
    "signals_with_peaks" :> CombineData[{peaksA, peaksB}, {1/4, 3/4}][[2, 2]],
    "signals_with_peaks_fid" :> N[Expand[CombineData[{peaksA, peaksB}, {1/4, 3/4}]]],
    "signals_with_data" :> N[Expand[CombineData[{dataA, dataB}, {1/4, 3/4}]]]
    }|>
}
