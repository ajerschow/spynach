(* Data tables: isotopes, physical constants and the orientational sampling sets. *)
isotopes := Interactions`Interactions`Private`IsotopicData;
schemes := Ensemble`OrientationalSampling`Private`DefinedOrientationalSamplingSchemes;
tabulated := Select[schemes, StringStartsQ[#, "Leboct" | "POLYTOPE" | "REPULSION" | "ZCW"] &];
flat[set_] := Join[First[#], {Last[#]}] & /@ set;
summary[name_] := With[{set = flat[OrientationsAndWeights[name]]},
   Join[{Length[set]}, Total[set], Total[set^2], First[set], Last[set]]];
{
 <|"group" -> "isotopes", "setup" :> SetSpinSystem[1], "cases" -> {
    "names" :> isotopes[[All, 2]],
    "mass_gamma_spin_abundance" :> isotopes[[All, {1, 3, 4, 5}]],
    "spin_13C" :> NuclearSpinQuantumNumber["13C"],
    "spin_by_mass_number_2" :> NuclearSpinQuantumNumber[2],
    "gamma_1H" :> MagnetogyricRatio["1H"],
    "gamma_15N" :> GyromagneticRatio["15N"],
    "abundance_13C" :> NaturalAbundance["13C"],
    "isotope_13" :> Isotope[13],
    "larmor_1H_9p4T" :> LarmorFrequency["1H", 9.4],
    "larmor_15N_9p4T" :> LarmorFrequency["15N", 9.4],
    "dipolar_1H_1H_2A" :> DirectDipolarCoupling["1H", 2.*^-10],
    "dipolar_1H_13C_1p1A" :> DirectDipolarCoupling[{"1H", "13C"}, 1.1*^-10],
    "dipolar_1H_15N_1A" :> DirectDipolarCoupling[{"1H", "15N"}, 1.*^-10]
    }|>,
 <|"group" -> "constants", "setup" :> SetSpinSystem[1], "cases" -> {
    "names" :> (ToString /@ PhysicalConstantValues[[All, 1]]),
    "values" :> PhysicalConstantValues[[All, 2]]
    }|>,
 <|"group" -> "orientations", "setup" :> SetSpinSystem[1], "cases" -> Join[{
     "schemes" :> schemes,
     "tabulated_names" :> tabulated,
     "tabulated_summaries" :> (summary /@ tabulated),
     "step_beta_4" :> flat[OrientationsAndWeights["Step\[Beta]", 4, {0, Pi}]],
     "leboct10" :> flat[OrientationsAndWeights["Leboct10"]],
     "zcw50" :> flat[OrientationsAndWeights["ZCW50"]],
     "repulsion100" :> flat[OrientationsAndWeights["REPULSION100"]],
     "polytope12" :> flat[OrientationsAndWeights["POLYTOPE12"]],
     "step_alpha_beta_3_4" :> flat[OrientationsAndWeights["Step\[Alpha]\[Beta]", 3, 4]]}]|>
 }
