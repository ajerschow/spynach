#!/usr/bin/env wolframscript
(* SpynDynamica fixture generator.

   Runs every case file in fixtures/cases against SpinDynamica and writes fixtures/data/<case>.json,
   manifest.json and timings.json. Start it with

       SPINDYNAMICA_HOME=<folder that holds SpinDynamica.m> wolframscript -file fixtures/generate.wls

   or, from a notebook, with SetEnvironment["SPINDYNAMICA_HOME" -> "..."]; Get["<repo>/fixtures/generate.wls"].
   The SpinDynamica folder is only read. Output goes to fixtures/data in this repository and nowhere else.

   A case file evaluates to a list of groups

       <|"group" -> "name", "setup" :> (SetSpinSystem[2]), "allow" -> {"Sym::tag", ...},
         "cases" -> {"name" :> expr, ...}|>

   Wrap a result that depends on symbols as Symbolic[expr, {a, b}]: its text and its values at three fixed
   parameter points are stored. Nothing is written unless the canary and every case pass. *)

fixturesDir = DirectoryName[$InputFileName];
casesDir = FileNameJoin[{fixturesDir, "cases"}];
dataDir = FileNameJoin[{fixturesDir, "data"}];
quit[code_] := If[TrueQ[$Notebooks], Abort[], Exit[code]];

sdHome = Environment["SPINDYNAMICA_HOME"];
If[! StringQ[sdHome] || ! FileExistsQ[FileNameJoin[{sdHome, "SpinDynamica.m"}]],
  Print["Set SPINDYNAMICA_HOME to the folder that holds SpinDynamica.m."]; quit[1]];

(* SpinDynamica reads VersionInfo.txt by relative path, so it is loaded from inside its folder *)
PrependTo[$Path, sdHome];
SetDirectory[sdHome];
Block[{Print = (Null &), $Messages = {}}, Get["SpinDynamica.m"]];
sdVersion = $SpinDynamicaVersion;
ResetDirectory[];
Print["Loaded ", sdVersion, " in Mathematica ", $Version];
Off[General::stop];  (* the whole script is one evaluation: never let a repeated message go silent *)

(* ---------------------------------------------------------------- encoding *)
text[e_] := ToString[e, InputForm];
exactQ[v_] := Precision[v] === Infinity;
numericQ[v_] := NumericQ[v] || (ArrayQ[v, _, NumericQ] && Length[v] > 0);
parameterPoints = {37/100, 123/100, -81/100, 59/100, -17/10, 211/100, 9/10, -43/100, 151/100, -29/100};
pointRules[vars_List, k_Integer] :=
  MapIndexed[#1 -> parameterPoints[[Mod[First[#2] + 3 k - 4, Length[parameterPoints]] + 1]] &, vars];

encode[v_SparseArray] := encode[Normal[v]];
(* arrays are stored flattened in row-major order; "im" is left out when it is zero everywhere *)
encode[v_?numericQ] := With[{re = Flatten[{N[Re[v]]}], im = Flatten[{N[Im[v]]}]},
   DeleteCases[<|"type" -> "array", "shape" -> If[ArrayQ[v], Dimensions[v], {}], "re" -> re,
     "im" -> If[AllTrue[im, # == 0 &], Missing[], im], "exact" -> If[exactQ[v], text[v], Missing[]]|>, _Missing]];
encode[v_String] := <|"type" -> "string", "value" -> v|>;
encode[v : {__String}] := <|"type" -> "strings", "value" -> v|>;
encode[Symbolic[v_, vars_List]] := With[{w = Normal[v]},
   <|"type" -> "symbolic", "variables" -> (text /@ vars), "text" -> text[w],
    "points" -> Table[
      With[{rules = pointRules[vars, k]},
       <|"parameters" -> N[vars /. rules], "parameters_exact" -> (text /@ (vars /. rules)),
        "value" -> With[{x = Quiet[N[N[w /. rules, 30]], N::meprec]},
          If[numericQ[x], encode[x], <|"type" -> "expr", "text" -> text[x]|>]]|>],
      {k, 3}]|>];
encode[v_List] := <|"type" -> "list", "items" -> (encode /@ v)|>;
encode[v_] := <|"type" -> "expr", "text" -> text[v]|>;

(* a result that is still a SpinDynamica call (an unevaluated routine) is unresolved as well *)
unresolvedQ[enc_] := ! FreeQ[enc, s_String /; StringContainsQ[s, "$Failed" | "$Aborted" | "opI[" | "OperatorProduct["]] ||
   (MatchQ[enc, KeyValuePattern["type" -> "symbolic"]] && ! FreeQ[enc["points"], "expr"]);

(* ---------------------------------------------------------------- running *)
informational = {"SetSpinSystem::set", "SetBasis::set", "SetOperatorBasis::set", "SetUserLevel::nochge",
   "SetUserLevel::ul"};
tag[m_] := Replace[m, Hold[Message[mn_, ___], _] :> ToString[Unevaluated[mn], InputForm]];
SetAttributes[capture, HoldAll];
capture[expr_] := Module[{msgs = {}, res, t},
   {t, res} = AbsoluteTiming[
     Block[{$Messages = {}, Print = (Null &)},
      Internal`HandlerBlock[{"Message", (If[TrueQ[Last[#]], AppendTo[msgs, tag[#]]] &)}, expr]]];
   {res, msgs, t}];

(* SetSpinSystem does not touch the operator basis: a basis chosen by one group stays the current one whenever
   the spin system is the same again. Asking for the operator basis under a spin system that no group uses
   replaces it by the default (OperatorBasis::inapprop), and the same happens again after the group's setup. *)
reset[] := ($Chronology = "LeftToRight"; SetUserLevel[2]; SetSpinSystem[{{1, 9/2}}]; OperatorBasis[];);

runCase[group_, allow_, RuleDelayed[name_String, expr_]] := Module[{res, msgs, t, enc, bad},
   {res, msgs, t} = capture[expr];
   enc = encode[res];
   bad = Complement[msgs, informational, allow];
   <|"name" -> group <> "/" <> name, "input" -> StringTake[text[HoldForm[expr]], {10, -2}], "value" -> enc,
    "messages" -> Union[msgs], "seconds" -> t,
    "problem" -> Which[bad =!= {}, "unexpected messages: " <> text[bad], unresolvedQ[enc], "unresolved result",
      True, Null]|>];

runGroup[g_Association] := Module[{setupMsgs, allow = Lookup[g, "allow", {}], name = g["group"], bad},
   capture[reset[]];
   setupMsgs = capture[g["setup"]][[2]];
   (* SetSpinSystem leaves the operator basis of the previous spin system in place; the first OperatorBasis[]
      call replaces it by the default one (message OperatorBasis::inapprop). Do that here, once per group. *)
   capture[OperatorBasis[]];
   bad = Complement[setupMsgs, informational, allow];
   If[bad =!= {},
    {<|"name" -> name <> "/<setup>", "problem" -> "unexpected messages: " <> text[bad], "seconds" -> 0|>},
    runCase[name, allow, #] & /@ g["cases"]]];

setupText[g_Association] := StringTake[text[Extract[g, {Key["setup"]}, HoldForm]], {10, -2}];

runFile[file_String] := Module[{groups = Get[file], results},
   results = Flatten[runGroup /@ groups];
   <|"file" -> FileNameTake[file], "sha256" -> Hash[ReadByteArray[file], "SHA256", "HexString"],
    "setups" -> Association[#["group"] -> setupText[#] & /@ groups], "results" -> results|>];

(* ---------------------------------------------------------------- canary *)
canary[] := Module[{m},
   reset[]; SetSpinSystem[2];
   m[op_] := Normal[MatrixRepresentation[op]];
   And[
    m[opI[1, "x"]] === {{0, 1/2, 0, 0}, {1/2, 0, 0, 0}, {0, 0, 0, 1/2}, {0, 0, 1/2, 0}},
    m[Commutator[opI[1, "x"], opI[1, "y"]]] === m[I opI[1, "z"]],
    m[RotationSuperoperator[{1}, {Pi/2, "x"}][opI[1, "z"]]] === m[-opI[1, "y"]],
    m[opI[1, "z"] . opI[2, "z"]] === DiagonalMatrix[{1/4, -1/4, -1/4, 1/4}]]];
If[! TrueQ[capture[canary[]][[1]]],
  Print["Canary failed: SpinDynamica did not load correctly. Nothing written."]; quit[2]];

(* ---------------------------------------------------------------- all case files *)
files = Sort[FileNames["*.wl", casesDir]];
runs = Map[(Print["  ", FileNameTake[#]]; runFile[#]) &, files];
all = Flatten[Lookup[runs, "results"]];
problems = Select[all, #["problem"] =!= Null &];
If[problems =!= {},
  Print["Problems, nothing written:"]; (Print["  ", #["name"], ": ", #["problem"]] &) /@ problems; quit[3]];
duplicates = Select[Tally[Lookup[all, "name"]], Last[#] > 1 &];
If[duplicates =!= {}, Print["Duplicate case names, nothing written: ", duplicates]; quit[4]];

write[name_String, data_] := Export[FileNameJoin[{dataDir, name}], data, "JSON"];
Do[
  write[FileBaseName[run["file"]] <> ".json",
   <|"spindynamica" -> sdVersion, "mathematica" -> $Version, "source" -> run["file"],
    "source_sha256" -> run["sha256"], "setups" -> run["setups"],
    "cases" -> Association[(#["name"] -> KeyTake[#, {"input", "value", "messages"}] &) /@ run["results"]]|>],
  {run, runs}];
write["manifest.json",
  <|"spindynamica" -> sdVersion, "mathematica" -> $Version, "system" -> $System,
   "generated" -> DateString["ISODateTime"],
   "files" -> Association[(#["file"] -> <|"sha256" -> #["sha256"], "cases" -> Length[#["results"]]|> &) /@ runs]|>];
write["timings.json", Association[(#["name"] -> Round[#["seconds"], 0.0001] &) /@ all]];
Print["Wrote ", Length[all], " cases from ", Length[runs], " files to ", dataDir];
