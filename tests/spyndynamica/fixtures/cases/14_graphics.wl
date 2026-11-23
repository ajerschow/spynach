(* Graphics: the geometry inside the Graphics3D objects (points of the ovaloid surface, ends of the axis
   arrows, polygons of the L shape and of the rotor). Colours, fonts and the placing of labels are not pinned. *)
points[g_] := First[Cases[g, GraphicsComplex[q_, ___] :> q, Infinity]];
arrows[g_] := N[Cases[g, Arrow[Tube[{b_, e_}, ___], ___] | Arrow[{b_, e_}, ___] :> {b, e}, Infinity]];
polygons[g_] := N[Cases[g, Polygon[q_] :> q, Infinity]];
mat = {{1, 1/5, 0}, {1/5, -2, 1/2}, {0, 1/2, 3}};
angles = {3/10, 7/10, -2/5};
{
 <|"group" -> "ovaloid", "setup" :> SetSpinSystem[1], "allow" -> {"WignerD::mod"}, "cases" -> {
    "points" :> points[Ovaloid[mat, PlotPoints -> 8]],
    "points_scaled_and_moved" :> points[Ovaloid[{1, 2, 3}, mat, OvaloidScale -> 1/2, PlotPoints -> 8]],
    "translation" :> Cases[Ovaloid[{1, 2, 3}, mat, OvaloidScale -> 1/2, PlotPoints -> 8], Translate[_, v_] :> v, Infinity],
    "points_isotropic_automatic" :> points[Ovaloid[mat, OvaloidIsotropicComponent -> Automatic, PlotPoints -> 8]],
    "points_isotropic_number" :> points[Ovaloid[mat, OvaloidIsotropicComponent -> 3/2, PlotPoints -> 8]],
    "points_spherical_tensor" :> points[Ovaloid[CartesianToSpherical[2, mat], PlotPoints -> 8]],
    "points_spherical_tensor_isotropic" :> points[Ovaloid[CartesianToSpherical[2, mat], OvaloidIsotropicComponent -> 3/2, PlotPoints -> 8]]
    }|>,
 <|"group" -> "axes_and_shapes", "setup" :> SetSpinSystem[1], "cases" -> {
    "axes_default" :> arrows[Axes3D[]],
    "axes_from_vectors" :> arrows[Axes3D[{{1, 0, 0}, {1, 1, 1}}, {1, 2, 3}, Axes3DLength -> 2]],
    "axes_from_euler_angles" :> arrows[Axes3D[angles]],
    "axes_from_euler_angles_and_system" :> arrows[Axes3D[angles, AxisSystem[{1, 1, 1}]]],
    "visualize_euler" :> arrows[VisualizeEuler[angles]],
    "arrow" :> arrows[Arrow3D[{1, 0, 0}, {1, 2, 3}]],
    "l_shape_edges" :> N[Cases[LShape[], Line[q_] :> q, Infinity]],
    "l_shape_polygons" :> polygons[LShape[]],
    "l_shape_polygons_other_dimensions" :> polygons[LShape[LShapeDimensions -> {1, 2, 1/2, 1/4, 1/3}]],
    "rotor_polygons" :> polygons[Rotor[PlotPoints -> 8]],
    "rotor_polygons_other_dimensions" :> polygons[Rotor[RotorDimensions -> {1/2, 3}, PlotPoints -> 6]],
    "sphere_and_axes_arrows" :> arrows[SphereAndAxes[]],
    "sphere_and_axes_labels" :> N[Cases[SphereAndAxes[], Text[_, position_, ___] :> position, Infinity]],
    "sphere_and_axes_circles" :> polygons[SphereAndAxes[]]
    }|>
 }
