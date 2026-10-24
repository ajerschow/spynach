"""EasySpin's loop over component spin systems (``compisoloop`` without the isotopologue expansion).

``pepper({Sys1, Sys2}, Exp)`` sums the spectra of the components, each
weighted by ``Sys.weight`` (default 1); ``Opt.separate = 'components'``
stacks them. Isotope mixtures (``isotopologues``) are not ported, so each
component is one spin system.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

_RANGE_FIELDS = ("Range", "CenterSweep", "mwRange", "mwCenterSweep")


def compisoloop(simfcn: Callable, systems: list | tuple, exp: dict, opt: dict | None):
    """Sum (or stack) the spectra of the component systems ``systems`` simulated by ``simfcn``."""
    from .garlic import Spectrum

    opt = dict(opt or {})
    separate = opt.get("separate", "") == "components"
    if separate:
        opt["separate"] = ""
    if len(systems) > 1 and not any(exp.get(k) is not None for k in _RANGE_FIELDS):
        raise ValueError(
            "For multiple components, EasySpin cannot automatically determine a sweep range. "
            "Please specify sweep range manually using Exp.Range or Exp.CenterSweep "
            "(Exp.mwRange or Exp.mwCenterSweep for frequency sweeps)."
        )
    spectra = []
    info: dict = {"Component": [], "Transitions": [], "nTransitions": []}
    x = None
    for k, system in enumerate(systems, start=1):
        sys_k = dict(system)
        weight = float(np.asarray(sys_k.pop("weight", 1.0)).ravel()[0])
        res = simfcn(sys_k, exp, opt)
        x = res.x
        spectra.append(np.asarray(res.spec, dtype=float) * weight)
        info["Component"].append(k)
        for key in ("Transitions", "nOrientations", "nSites", "resfields"):
            if key in res.info:
                info.setdefault(key, []).append(res.info[key])
        if "Transitions" in res.info:
            info["nTransitions"].append(np.asarray(res.info["Transitions"]).shape[0])
    if not info["Transitions"]:
        info.pop("Transitions")
        info.pop("nTransitions")
    elif len(info["Transitions"]) == 1:
        info["Transitions"] = info["Transitions"][0]
    if "resfields" in info and len(info["resfields"]) == 1:
        info["resfields"] = info["resfields"][0]
    if separate:
        spec = np.stack([np.atleast_2d(s) for s in spectra]).reshape(-1, spectra[0].shape[-1])
    else:
        spec = sum(spectra)
    return Spectrum(x, spec, info)
