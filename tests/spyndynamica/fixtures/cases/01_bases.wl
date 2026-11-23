(* Hilbert-space bases: order, labels and phases of the kets. *)
kets[basis_] := Normal[VectorRepresentation[#, ZeemanBasis[]]] & /@ BasisKets[basis];
quantumNumbers[basis_] := BasisKets[basis] /. Ket[spec_, ___] :> spec;
zeemanGroup[name_, system_] := <|"group" -> name, "setup" :> SetSpinSystem[system], "cases" -> {
     "spin_system" :> SpinSystem[],
     "dimension" :> BasisDimension[ZeemanBasis[]],
     "labels" :> BasisLabels[ZeemanBasis[]],
     "zeeman_quantum_numbers" :> quantumNumbers[ZeemanBasis[]],
     "zeeman_kets" :> kets[ZeemanBasis[]],
     "Iz_total_diagonal" :> Diagonal[Normal[MatrixRepresentation[opI["z"]]]]
     }|>;
{
 zeemanGroup["one_half", 1],
 zeemanGroup["two_half", 2],
 zeemanGroup["three_half", 3],
 zeemanGroup["spin1", {{1, 1}}],
 zeemanGroup["half_one", {{1, 1/2}, {2, 1}}],
 zeemanGroup["spin32", {{1, 3/2}}],
 <|"group" -> "singlet_triplet", "setup" :> SetSpinSystem[2], "cases" -> {
    "labels" :> BasisLabels[SingletTripletBasis[]],
    "kets_in_zeeman" :> kets[SingletTripletBasis[]],
    "transformation_st_from_zeeman" :> BasisTransformationMatrix[SingletTripletBasis[], ZeemanBasis[]],
    "transformation_zeeman_from_st" :> BasisTransformationMatrix[ZeemanBasis[], SingletTripletBasis[]],
    "I1z_in_st" :> MatrixRepresentation[opI[1, "z"], SingletTripletBasis[]],
    "I1I2_in_st" :> MatrixRepresentation[opI[1] . opI[2], SingletTripletBasis[]],
    "Ix_total_in_st" :> MatrixRepresentation[opI["x"], SingletTripletBasis[]]
    }|>
 }
