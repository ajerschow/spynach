"""Least-squares fitting: EasySpin's ``esfit`` without its GUI, with the fitting algorithms and data helpers.

``esfit(data, fcn, p0, vary)`` or ``esfit(data, fcn, p0, lb, ub)``, with an
optional options dict last, fits the model ``fcn`` to ``data``. For the
EasySpin simulators (``pepper``, ``garlic``) the parameters are the spin
system, experiment and option dicts and ``vary``/``lb``/``ub`` are dicts of
the same shape naming the fields to vary; for any other callable they are
parameter vectors. The algorithms are EasySpin's: Nelder-Mead simplex,
Levenberg-Marquardt, Monte Carlo, genetic, grid search and particle swarm,
plus ``lsqnonlin`` through ``scipy.optimize.least_squares``. The parameter
uncertainties follow EasySpin (Romberg-extrapolated Jacobian, HC1
heteroscedasticity-consistent covariance). ``rescaledata``, ``basecorr``,
``deriv`` and ``addnoise`` are the data helpers that go with it.
"""

from __future__ import annotations

import copy
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.special import erfcinv

__all__ = [
    "FitResult",
    "addnoise",
    "basecorr",
    "deriv",
    "esfit",
    "esfit_genetic",
    "esfit_grid",
    "esfit_levmar",
    "esfit_montecarlo",
    "esfit_simplex",
    "esfit_swarm",
    "hccm",
    "jacobianest",
    "rescaledata",
]

_ALGORITHMS = {"simplex": 1, "levmar": 2, "montecarlo": 3, "genetic": 4, "grid": 5, "swarm": 6, "lsqnonlin": 7}
_ALGORITHM_NAMES = [
    "Nelder-Mead simplex",
    "Levenberg-Marquardt",
    "Monte Carlo",
    "genetic algorithm",
    "grid search",
    "particle swarm",
    "lsqnonlin",
]
_TARGETS = {"fcn": 1, "int": 2, "dint": 3, "diff": 4, "fft": 5}
_TARGET_NAMES = ["data as is", "integral", "double integral", "derivative", "Fourier transform"]
_AUTOSCALE = {"none": 0, "lsq": 1, "maxabs": 2}
_BASELINE_NAMES = {-1: "none", 0: "offset", 1: "linear", 2: "quadratic", 3: "cubic"}
_NONNEGATIVE_FIELDS = {"lw", "lwpp", "HStrain", "gStrain", "DStrain", "AStrain", "weight", "initState"}
_EASYSPIN_FUNCTIONS = {"pepper", "garlic", "chili", "salt"}


# ---------------------------------------------------------------------------------- algorithm options
def _algdefaults(algorithm: str) -> dict:
    """EasySpin's ``esfit_algdefaults``: the options of each algorithm."""
    opt: dict[str, Any] = {"TolFun": 1e-5, "maxTime": np.inf, "IterFcn": None, "Verbosity": 0, "RandomStart": 0}
    if algorithm == "Nelder-Mead simplex":
        opt.update(delta=0.1, TolEdgeLength=1e-4, SimplexPars=None, ScaleParams=False)
    elif algorithm == "Levenberg-Marquardt":
        opt.update(lambda_=1e-3, delta=1e-3, TolStep=1e-6, Gradient=1e-5, ScaleParams=False)
    elif algorithm == "Monte Carlo":
        opt.update(nTrials=20000)
    elif algorithm == "genetic algorithm":
        opt.update(PopulationSize=20, EliteCount=None, maxGenerations=10000)
    elif algorithm == "grid search":
        opt.update(GridSize=7, randGrid=True, maxGridPoints=100000)
    elif algorithm == "particle swarm":
        opt.update(nParticles=None, SwarmParams=[0.2, 0.5, 2, 1], TolStallIter=6)
    return opt


def _options(fitopt: dict | None, algorithm: str) -> dict:
    opt = _algdefaults(algorithm)
    for k, v in (fitopt or {}).items():
        opt["lambda_" if k == "lambda" else k] = v
    if opt.get("Gradient") is None:
        opt["Gradient"] = opt["TolFun"]
    return opt


def _check_bounds(x0, lb, ub, *, with_x0=True):
    lb = np.asarray(lb, dtype=float).ravel()
    ub = np.asarray(ub, dtype=float).ravel()
    if lb.size != ub.size:
        raise ValueError("Arrays for lower and upper bound must have the same number of elements.")
    if np.any(lb > ub):
        raise ValueError("Lower bounds must not be greater than upper bounds.")
    if not with_x0:
        return None, lb, ub
    x0 = np.asarray(x0, dtype=float).ravel()
    if lb.size != x0.size:
        raise ValueError("Lower-bounds array must have the same number of elements as x0.")
    if ub.size != x0.size:
        raise ValueError("Upper-bounds array must have the same number of elements as x0.")
    if np.any(~np.isfinite(x0)):
        raise ValueError("x0 must be real and finite.")
    if np.any(x0 < lb) or np.any(x0 > ub):
        raise ValueError("Some elements in x0 are out of bounds.")
    return x0, lb, ub


def _rng(opt: dict) -> np.random.Generator:
    rng = opt.get("rng")
    return rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)


def _call_iter(opt: dict, info: dict) -> bool:
    fcn = opt.get("IterFcn")
    return bool(fcn(info)) if fcn is not None else False


def _log(opt: dict, text: str, level: int = 1) -> None:
    """Print ``text`` at ``Verbosity >= level``: 1 for summaries, 2 for the per-iteration lines."""
    if opt.get("Verbosity", 0) >= level:
        print(text)


# ---------------------------------------------------------------------------------- Nelder-Mead simplex
def esfit_simplex(fcn: Callable, x0, lb, ub, fitopt: dict | None = None) -> tuple[np.ndarray, dict]:
    """Nelder-Mead simplex minimisation of the scalar ``fcn(x)`` within ``[lb, ub]``: ``(xmin, info)``.

    Options: ``delta`` (edge length of the initial simplex as a fraction of the bounds), ``SimplexPars``
    (``[rho, chi, psi, sigma]``; adaptive above two parameters), ``TolFun``, ``TolEdgeLength``, ``maxTime``
    (minutes), ``ScaleParams``, ``RandomStart``, ``IterFcn``, ``Verbosity``."""
    opt = _options(fitopt, "Nelder-Mead simplex")
    x0, lb, ub = _check_bounds(x0, lb, ub)
    n = x0.size
    pars = opt["SimplexPars"]
    if pars is None or np.size(pars) == 0:
        pars = [1, 2, 0.5, 0.5] if n <= 2 else [1, 1 + 2 / n, 0.75 - 1 / (2 * n), 1 - 1 / n]  # Gao & Han 2010
    rho, chi, psi, sigma = (float(p) for p in pars)
    delta = opt["delta"]
    if opt["RandomStart"]:
        x0 = lb + _rng(opt).random(n) * (ub - lb)
    f = fcn
    if opt["ScaleParams"]:
        lb0, ub0 = lb, ub

        def untransform(x):
            return lb0 + (ub0 - lb0) * (x / 2 + 0.5)

        def f(x, _fcn=fcn):
            return _fcn(untransform(x))

        x0 = 2 * (x0 - lb0) / (ub0 - lb0) - 1
        lb, ub = -np.ones(n), np.ones(n)

    def constrain(x):
        return np.maximum(np.minimum(x, ub), lb)

    start = time.time()
    n_evals = 0
    stop_code = 0
    v = np.repeat(x0[:, None], n + 1, axis=1)
    for i in range(n):
        v[i, i + 1] += delta * (ub[i] - lb[i])
    info: dict[str, Any] = {"simplex_initial": v.copy()}
    fv = np.zeros(n + 1)
    for k in range(n, -1, -1):  # EasySpin evaluates the vertices in reverse order
        fv[k] = f(constrain(v[:, k]))
        n_evals += 1
    order = np.argsort(fv, kind="stable")
    fv, v = fv[order], v[:, order]
    iteration = 1
    procedure = "initial simplex"
    _log(opt, f" iteration {iteration:3d}: value {fv[0]:0.5e}   edge {delta:0.5e}   {procedure}", 2)
    while True:
        fv_old0 = fv[0]
        elapsed = (time.time() - start) / 60
        if elapsed > opt["maxTime"]:
            stop_code = 1
            break
        largest_diff = np.max(np.abs(fv[0] - fv[1:]))
        longest_edge = np.max(np.abs(v[:, 1:] - v[:, [0]]))
        if largest_diff <= opt["TolFun"] and longest_edge <= opt["TolEdgeLength"]:
            stop_code = 3
            break
        xbar = v[:, :-1].mean(axis=1)
        xr = constrain(xbar + rho * (xbar - v[:, -1]))
        fr = f(xr)
        n_evals += 1
        reduce = False
        if fr < fv[0]:
            xe = constrain(xbar + rho * chi * (xbar - v[:, -1]))
            fe = f(xe)
            n_evals += 1
            if fe < fr:
                procedure, v[:, -1], fv[-1] = "expansion", xe, fe
            else:
                procedure, v[:, -1], fv[-1] = "reflection", xr, fr
        elif fr < fv[n - 1]:
            procedure, v[:, -1], fv[-1] = "reflection", xr, fr
        elif fr < fv[-1]:
            xco = constrain(xbar + psi * rho * (xbar - v[:, -1]))
            fco = f(xco)
            n_evals += 1
            if fco <= fr:
                procedure, v[:, -1], fv[-1] = "contraction outside", xco, fco
            else:
                reduce = True
        else:
            xci = constrain(xbar - psi * (xbar - v[:, -1]))
            fci = f(xci)
            n_evals += 1
            if fci < fv[-1]:
                procedure, v[:, -1], fv[-1] = "contraction inside", xci, fci
            else:
                reduce = True
        if reduce:
            procedure = "reduction"
            for k in range(1, n + 1):
                xshr = constrain(v[:, 0] + sigma * (v[:, k] - v[:, 0]))
                fv[k] = f(xshr)
                n_evals += 1
                v[:, k] = xshr
        order = np.argsort(fv, kind="stable")
        fv, v = fv[order], v[:, order]
        iteration += 1
        info.update(bestx=v[:, 0].copy(), minF=fv[0], nEvals=n_evals, iter=iteration, newbest=fv[0] < fv_old0)
        if _call_iter(opt, info):
            stop_code = 2
        if opt.get("Verbosity", 0) >= 2:
            step = np.max(np.abs(v[:, 1:] - v[:, [0]]))
            _log(opt, f" iteration {iteration:3d}: value {fv[0]:0.5e}   edge {step:0.5e}   {procedure}", 2)
        if stop_code:
            break
    x = untransform(v[:, 0]) if opt["ScaleParams"] else v[:, 0].copy()
    info.update(simplex_final=v, F=fv[0], nIterations=iteration, elapsedTime=(time.time() - start) / 60)
    messages = {
        1: f"Time limit of {opt['maxTime']:f} minutes reached.",
        2: "Stopped by user.",
        3: f"Converged (edge length < {opt['TolEdgeLength']:g} and all errors < {opt['TolFun']:g}).",
    }
    _log(opt, f"Terminated: {messages.get(stop_code, '')}\n")
    info["stop"] = stop_code
    info["msg"] = messages.get(stop_code, "")
    return x, info


# ---------------------------------------------------------------------------------- Levenberg-Marquardt
def esfit_levmar(fcn: Callable, x0, lb, ub, fitopt: dict | None = None) -> tuple[np.ndarray, dict]:
    """Levenberg-Marquardt minimisation of ``sum(fcn(x)**2)`` within ``[lb, ub]``: ``(xfit, info)``.

    EasySpin's variant: Gauss-Newton with Marquardt damping and a Jacobian that is updated by secant steps
    between finite-difference refreshes. Options: ``lambda`` (initial damping), ``delta`` (difference step),
    ``TolStep``, ``Gradient``, ``maxTime``, ``ScaleParams``, ``RandomStart``, ``IterFcn``, ``Verbosity``."""
    opt = _options(fitopt, "Levenberg-Marquardt")
    x0, lb, ub = _check_bounds(x0, lb, ub)
    n = x0.size
    delta = opt["delta"]
    if opt["RandomStart"]:
        x0 = lb + _rng(opt).random(n) * (ub - lb)
    f_res = fcn
    if opt["ScaleParams"]:
        lb0, ub0 = lb, ub

        def untransform(x):
            return lb0 + (ub0 - lb0) * (x / 2 + 0.5)

        def f_res(x, _fcn=fcn):
            return _fcn(untransform(x))

        x0 = 2 * (x0 - lb0) / (ub0 - lb0) - 1
        lb, ub = -np.ones(n), np.ones(n)

    def funeval(x):
        r = np.asarray(f_res(x), dtype=float).ravel()
        if np.any(np.isnan(r)):
            raise ValueError("f contains at least one NaN value.")
        if np.any(np.isinf(r)):
            raise ValueError("f contains at least one Inf value.")
        F = float(r @ r)
        return (-5 if np.isinf(F) else 0), F, r

    start = time.time()
    x = x0.copy()
    stop_code, F, f = funeval(x)
    n_evals = 1
    info: dict[str, Any] = {}
    Je = np.zeros((f.size, n))
    norm_g = np.nan
    if not stop_code:
        for k in range(n):  # finite-difference Jacobian
            x1 = x.copy()
            x1[k] += delta
            Je[:, k] = (np.asarray(f_res(x1), dtype=float).ravel() - f) / delta
        n_evals += n
        if np.any(~np.isfinite(Je)):
            stop_code = -6
        else:
            g = Je.T @ f
            norm_g = np.linalg.norm(g, np.inf)
            A = Je.T @ Je
            if np.isinf(norm_g) or np.isinf(np.linalg.norm(A.ravel(), np.inf)):
                stop_code = -5
    if stop_code:
        info.update(F=F, norm_g=norm_g, Je=Je, stop=stop_code, nEvals=n_evals)
        return (untransform(x) if opt["ScaleParams"] else x), info

    mu = opt["lambda_"] * np.max(np.diag(A))
    nu = 2.0
    norm_h = 0.0
    j = 0
    iteration = 0
    while not stop_code:
        iteration += 1
        if norm_g <= opt["Gradient"]:
            stop_code = 1
            break
        h, mu = _lm_step(A, g, mu)
        norm_h = np.linalg.norm(h)
        _log(
            opt,
            f" iteration {iteration:4d}:  value {np.sqrt(F * 2):0.5e}    gradient {norm_g:0.5e}    step {norm_h:0.5e}",
            2,
        )
        if norm_h <= opt["TolStep"] * (opt["TolStep"] + np.linalg.norm(x)):
            stop_code = 2
            break
        xnew = np.minimum(np.maximum(x + h, lb), ub)
        stop_code, Fnew, fnew = funeval(xnew)
        n_evals += 1
        if stop_code:
            break
        info.update(bestx=xnew.copy(), minF=Fnew, nEvals=n_evals, iter=iteration, newbest=True)
        if _call_iter(opt, info):
            stop_code = 4
            break
        j = j % n + 1  # cyclic finite-difference refresh of one Jacobian column
        if abs(h[j - 1]) < 0.8 * norm_h:
            xu = x.copy()
            xu[j - 1] = x[j - 1] + delta
            stop_u, _, fu = funeval(xu)
            n_evals += 1
            if not stop_u:
                hu = xu - x
                Je = Je + np.outer((fu - f - Je @ hu) / (hu @ hu), hu)
        Je = Je + np.outer((fnew - f - Je @ h) / (h @ h), h)
        rho = (F - Fnew) / (0.5 * (h @ (mu * h - g)))
        if rho > 0:
            x, F, f = xnew, Fnew, fnew
            mu = mu * max(1 / 3, 1 - (2 * rho - 1) ** 3)
            nu = 2.0
        else:
            mu = mu * nu
            nu = 2 * nu
        g = Je.T @ f
        norm_g = np.linalg.norm(g, np.inf)
        A = Je.T @ Je
        if np.isinf(norm_g) or np.isinf(np.linalg.norm(A.ravel(), np.inf)):
            stop_code = -5
            break
        if (time.time() - start) / 60 > opt["maxTime"]:
            stop_code = 3
            break
    if stop_code and iteration == 1:
        info.update(bestx=x.copy(), minF=F, nEvals=n_evals, iter=iteration, newbest=True)
        _call_iter(opt, info)
    messages = {
        1: f"Gradient below threshold of {opt['Gradient']:g}",
        2: f"Parameter step below threshold of {opt['TolStep']:g}",
        3: f"Time limit of {opt['maxTime']:f} minutes reached",
        4: "Stopped by user",
    }
    _log(opt, f"Terminated: {messages.get(stop_code, '')}\n")
    xfit = untransform(x) if opt["ScaleParams"] else x
    info.update(
        F=F,
        norm_g=norm_g,
        norm_h=norm_h,
        Je=Je,
        lambda_=np.nan if stop_code < 0 else mu / np.max(np.diag(A)),
        nIterations=iteration - 1,
        stop=stop_code,
        nEvals=n_evals,
        elapsedTime=(time.time() - start) / 60,
        msg=messages.get(stop_code, ""),
    )
    return xfit, info


def _lm_step(A, g, mu):
    """Solve (A + mu I) h = -g by Cholesky, raising mu until the matrix is positive definite."""
    n = A.shape[0]
    while True:
        try:
            R = np.linalg.cholesky(A + mu * np.eye(n)).T
        except np.linalg.LinAlgError:
            mu = 10 * mu
            continue
        if 1 / np.linalg.cond(R, 1) < 1e-15:
            mu = 10 * mu
            continue
        break
    h = np.linalg.solve(R, np.linalg.solve(R.T, -g))
    return h, mu


# ---------------------------------------------------------------------------------- Monte Carlo
def esfit_montecarlo(fcn: Callable, lb, ub, fitopt: dict | None = None) -> tuple[np.ndarray, dict]:
    """Random search of ``fcn`` over ``[lb, ub]`` with ``nTrials`` trials: ``(bestx, info)``."""
    opt = _options(fitopt, "Monte Carlo")
    _, lb, ub = _check_bounds(None, lb, ub, with_x0=False)
    rng = _rng(opt)
    n = lb.size
    Fmin, bestx = np.inf, np.zeros(n)
    start = time.time()
    stop_code = 0
    info: dict[str, Any] = {}
    trial = 0
    for trial in range(1, int(opt["nTrials"]) + 1):
        X = lb + (ub - lb) * rng.random(n)
        F = fcn(X)
        newbest = F < Fmin
        if newbest:
            Fmin, bestx = F, X
            _log(opt, f"{trial}:   error {F:0.5e}  best so far", 2)
        info.update(bestx=bestx.copy(), minF=Fmin, nEvals=trial, iter=trial, newbest=newbest)
        if _call_iter(opt, info):
            stop_code = 2
        if (time.time() - start) / 60 > opt["maxTime"]:
            stop_code = 1
        if F < opt["TolFun"]:
            stop_code = 3
        if stop_code:
            break
    messages = {
        0: "Terminated: maximum number of trials reached.",
        1: f"Terminated: Time limit of {opt['maxTime']:f} minutes reached.",
        2: "Terminated: Stopped by user.",
        3: f"Terminated: Found a parameter set with error less than {opt['TolFun']:g}.",
    }
    _log(opt, messages[stop_code])
    info.update(nTrials=trial, elapsedTime=(time.time() - start) / 60, stop=stop_code, msg=messages[stop_code])
    return bestx, info


# ---------------------------------------------------------------------------------- genetic algorithm
def esfit_genetic(fcn: Callable, lb, ub, fitopt: dict | None = None) -> tuple[np.ndarray, dict]:
    """Genetic algorithm (roulette selection, blend recombination, sectioned mutation, elitism): ``(bestx, info)``.

    Options: ``PopulationSize``, ``EliteCount``, ``maxGenerations``, ``TolFun``, ``maxTime``."""
    opt = _options(fitopt, "genetic algorithm")
    _, lb, ub = _check_bounds(None, lb, ub, with_x0=False)
    rng = _rng(opt)
    n = lb.size
    pop_size = int(opt["PopulationSize"])
    elite = opt["EliteCount"]
    elite = max(2, int(np.ceil(0.1 * pop_size))) if elite is None else int(elite)
    _log(opt, f"  {n} parameters\n  population {pop_size}, elite {elite}\n  {opt['maxGenerations']} generations")
    start = time.time()
    n_evals = 0
    stop_code = 0
    population = lb + (ub - lb) * rng.random((pop_size, n))
    best_score, bestx = np.inf, population[0].copy()
    info: dict[str, Any] = {}
    _log(opt, "initial population")
    scores = np.full(pop_size, np.inf)
    for k in range(pop_size):
        scores[k] = fcn(population[k])
        n_evals += 1
        newbest = scores[k] < best_score
        if newbest:
            bestx, best_score = population[k].copy(), scores[k]
        info.update(bestx=bestx.copy(), minF=best_score, nEvals=n_evals, iter=0, newbest=newbest)
        if _call_iter(opt, info):
            stop_code = 3
            break
    order = np.argsort(scores, kind="stable")
    scores, population = scores[order], population[order]
    fitness = scores[-1] - scores
    gen = 1
    while not stop_code:
        best_score = min(best_score, scores.min())
        _log(opt, f"gen {gen:5d}:  min {scores.min():g}   mean {scores.mean():g}", 2)
        if gen >= opt["maxGenerations"]:
            break
        if (time.time() - start) / 60 > opt["maxTime"]:
            stop_code = 1
            break
        if best_score < opt["TolFun"]:
            stop_code = 2
            break
        wheel = np.concatenate([[0], np.cumsum(fitness) / fitness.sum()]) if fitness.sum() > 0 else None
        balls = (0.5 + np.arange(pop_size)) / pop_size
        parents = np.array([np.sum(b > wheel) for b in balls]) if wheel is not None else np.arange(1, pop_size + 1)
        parents = np.clip(parents, 1, pop_size) - 1
        r = parents[np.floor(pop_size * rng.random((2, pop_size))).astype(int)]
        a = rng.random(pop_size) * 1.2 - 0.1
        offspring = a[:, None] * population[r[0]] + (1 - a[:, None]) * population[r[1]]
        nk = 10
        direction = 2 * rng.integers(0, 2, pop_size) - 1
        section = rng.integers(1, nk, pop_size)
        for k in range(pop_size):
            lo, hi = (lb, offspring[k]) if direction[k] == -1 else (offspring[k], ub)
            offspring[k] = offspring[k] + direction[k] * rng.random(n) * (hi - lo) / nk * section[k]
        offspring = np.minimum(np.maximum(offspring, lb), ub)
        off_scores = np.zeros(pop_size)
        for k in range(pop_size):
            off_scores[k] = fcn(offspring[k])
            n_evals += 1
            info.update(nEvals=n_evals, iter=gen)
            if _call_iter(opt, info):
                stop_code = 3
                break
        if stop_code == 3:
            break
        order = np.argsort(off_scores, kind="stable")
        off_scores, offspring = off_scores[order], offspring[order]
        keep = pop_size - elite
        population[elite:] = offspring[:keep]
        scores[elite:] = off_scores[:keep]
        order = np.argsort(scores, kind="stable")
        scores, population = scores[order], population[order]
        newbest = scores[0] < best_score
        if newbest:
            best_score, bestx = scores[0], population[0].copy()
        fitness = scores[-1] - scores
        info.update(
            bestx=bestx.copy(),
            minF=best_score,
            nEvals=n_evals,
            elapsedTime=(time.time() - start) / 60,
            iter=gen,
            newbest=newbest,
        )
        if _call_iter(opt, info):
            stop_code = 3
            break
        gen += 1
    messages = {
        0: f"Maximum number of generations ({opt['maxGenerations']}) reached.",
        1: f"Time limit of {opt['maxTime']:f} minutes reached.",
        2: f"Error below threshold {opt['TolFun']:g}.",
        3: "Stopped by user.",
    }
    _log(opt, f"Terminated: {messages[stop_code]}\n")
    info.update(stop=stop_code, msg=messages[stop_code], nGenerations=gen)
    return bestx, info


# ---------------------------------------------------------------------------------- grid search
def esfit_grid(fcn: Callable, lb, ub, fitopt: dict | None = None) -> tuple[np.ndarray, dict]:
    """Evaluate ``fcn`` on a grid over ``[lb, ub]`` (``GridSize`` points per parameter): ``(bestx, info)``."""
    opt = _options(fitopt, "grid search")
    _, lb, ub = _check_bounds(None, lb, ub, with_x0=False)
    n = lb.size
    grid_size = np.atleast_1d(np.asarray(opt["GridSize"], dtype=int))
    if grid_size.size == 1:
        grid_size = np.repeat(grid_size, n)
    if grid_size.size != n:
        raise ValueError("FitOpt.GridSize must have as many elements as there are fitting parameters.")
    if np.any(grid_size < 2):
        raise ValueError("At least two grid points per parameter are needed.")
    n_points = int(np.prod(grid_size))
    if n_points > opt["maxGridPoints"]:
        raise ValueError(
            f"Cannot do grid search with more than {opt['maxGridPoints']} points. Reduce number of parameters."
        )
    _log(opt, f"{n} parameters, {n_points} grid points total\n")
    axes = [np.linspace(lb[p], ub[p], grid_size[p]) for p in range(n)]
    grids = np.meshgrid(*axes, indexing="ij")
    X = np.stack([g.ravel(order="F") for g in grids])  # MATLAB's ndgrid order: first parameter fastest
    if opt["randGrid"]:
        X = X[:, _rng(opt).permutation(n_points)]
    minF, bestx = np.inf, np.full(n, np.nan)
    start = time.time()
    stop_code = 0
    n_evals = 0
    info: dict[str, Any] = {}
    for k in range(n_points):
        F = fcn(X[:, k])
        n_evals += 1
        newbest = F < minF
        if newbest:
            minF, bestx = F, X[:, k].copy()
            _log(opt, f"  Point {k + 1:4d}/{n_points}:   error {F:0.5e}  best so far", 2)
        info.update(
            currx=X[:, k].copy(), currF=F, bestx=bestx.copy(), minF=minF, nEvals=n_evals, iter=k + 1, newbest=newbest
        )
        stop = _call_iter(opt, info)
        if (time.time() - start) / 60 > opt["maxTime"]:
            stop_code = 1
        if stop:
            stop_code = 2
        if minF < opt["TolFun"]:
            stop_code = 3
        if stop_code:
            break
    messages = {
        0: "Terminated: all grid points searched.",
        1: f"Terminated: Time limit of {opt['maxTime']:f} minutes reached.",
        2: "Terminated: Stopped by user.",
        3: f"Terminated: Found a parameter set with error less than {opt['TolFun']:g}.",
    }
    _log(opt, messages[stop_code])
    info.update(stop=stop_code, elapsedTime=(time.time() - start) / 60, msg=messages[stop_code])
    return bestx, info


# ---------------------------------------------------------------------------------- particle swarm
def esfit_swarm(fcn: Callable, lb, ub, fitopt: dict | None = None) -> tuple[np.ndarray, dict]:
    """Particle swarm minimisation of ``fcn`` over ``[lb, ub]``: ``(bestx, info)``.

    Options: ``nParticles`` (default ``20 + 10 n``), ``SwarmParams`` ``[k, w, c1, c2]``, ``TolStallIter``,
    ``TolFun``, ``maxTime``."""
    opt = _options(fitopt, "particle swarm")
    _, lb, ub = _check_bounds(None, lb, ub, with_x0=False)
    rng = _rng(opt)
    n = lb.size
    n_particles = opt["nParticles"]
    n_particles = 20 + n * 10 if n_particles is None else int(n_particles)
    k, w, c1, c2 = (float(p) for p in opt["SwarmParams"])
    _log(
        opt,
        f"Particle swarm optimization parameters:\n   n = {n_particles}\n   k = {k:g}\n   w = {w:g}\n"
        f"   c1 = {c1:g}\n   c2 = {c2:g}",
    )
    X = lb[:, None] + (ub - lb)[:, None] * rng.random((n, n_particles))
    v = (ub - lb)[:, None] * rng.random((n, n_particles)) * k
    bestX = X.copy()
    gX = bestX[:, 0].copy()
    bestF = np.full(n_particles, np.inf)
    F = np.full(n_particles, np.inf)
    global_best = np.inf
    start = time.time()
    stalled = 0
    iteration = 0
    stop_code = 0
    info: dict[str, Any] = {}
    newbest = False
    while stop_code == 0:
        iteration += 1
        for p in range(n_particles):
            F[p] = fcn(X[:, p])
        better = F < bestF
        bestF[better] = F[better]
        bestX[:, better] = X[:, better]
        idx = int(np.argmin(F))
        newbest = F[idx] < global_best
        if newbest:
            stalled = 0
            global_best = F[idx]
            gX = X[:, idx].copy()
        else:
            stalled += 1
        for p in range(n_particles):
            v[:, p] = w * v[:, p] + c1 * rng.random() * (bestX[:, p] - X[:, p]) + c2 * rng.random() * (gX - X[:, p])
            X[:, p] = np.minimum(np.maximum(X[:, p] + v[:, p], lb), ub)
        _log(opt, f"  Iteration {iteration:4d}:   value {global_best:0.5e}  best so far ({stalled})", 2)
        if (time.time() - start) / 60 > opt["maxTime"]:
            stop_code = 1
        if global_best < opt["TolFun"]:
            stop_code = 3
        if stalled == opt["TolStallIter"]:
            stop_code = 4
        info.update(bestx=gX.copy(), minF=global_best, iter=iteration, newbest=newbest)
        if _call_iter(opt, info):
            stop_code = 2
    messages = {
        1: f"Terminated: Time limit of {opt['maxTime']:f} minutes reached.",
        2: "Terminated: Stopped by user.",
        3: f"Terminated: Found a parameter set with function value less than {opt['TolFun']:g}.",
        4: f"Terminated: Function value didn't change for {opt['TolStallIter']} iterations.",
    }
    _log(opt, messages[stop_code])
    info.update(nIterations=iteration, elapsedTime=(time.time() - start) / 60, stop=stop_code, msg=messages[stop_code])
    return gX, info


# ---------------------------------------------------------------------------------- uncertainties
def jacobianest(fun: Callable, x0, max_rel_step: float | np.ndarray = 0.01) -> tuple[np.ndarray, np.ndarray]:
    """Jacobian of the vector function ``fun`` at ``x0`` by Romberg-extrapolated central differences.

    Returns ``(jac, err)``: the derivative estimates and their error estimates (EasySpin's ``jacobianest``,
    after John D'Errico's DERIVESTsuite)."""
    x0 = np.asarray(x0, dtype=float).ravel()
    nx = x0.size
    step_ratio = 2 + 1e-7
    n_steps = 10
    rel = np.broadcast_to(np.asarray(max_rel_step, dtype=float), (nx,))
    f0 = np.asarray(fun(x0), dtype=float).ravel()
    n = f0.size
    if n == 0:
        return np.zeros((0, nx)), np.zeros((0, nx))
    steps = step_ratio ** -np.arange(n_steps + 1)
    jac = np.zeros((n, nx))
    err = np.zeros((n, nx))
    for i in range(nx):
        rel_delta = rel[i] * steps
        delta = x0[i] * rel_delta if x0[i] != 0 else rel_delta
        fdel = np.zeros((n, steps.size))
        for j in range(steps.size):
            xp, xm = x0.copy(), x0.copy()
            xp[i] = x0[i] + delta[j]
            xm[i] = x0[i] - delta[j]
            fdel[:, j] = np.asarray(fun(xp), dtype=float).ravel() - np.asarray(fun(xm), dtype=float).ravel()
        derest = fdel * (0.5 / delta)[None, :]
        for j in range(n):
            der_romb, errest = _romberg(step_ratio, derest[j], (2, 4))
            nest = der_romb.size
            trim = [0, 1, 2, nest - 3, nest - 2, nest - 1]
            tags = np.argsort(der_romb, kind="stable")
            der_romb = der_romb[tags]
            keep = np.ones(nest, bool)
            keep[trim] = False
            der_romb, tags = der_romb[keep], tags[keep]
            errest = errest[tags]
            ind = int(np.argmin(errest))
            err[j, i] = errest[ind]
            jac[j, i] = der_romb[ind]
    return jac, err


def _romberg(step_ratio: float, der_init: np.ndarray, rombexpon: tuple[int, int]):
    srinv = 1 / step_ratio
    expon = np.asarray(rombexpon, dtype=float)
    rmat = np.ones((4, 3))
    rmat[1, 1:3] = srinv**expon
    rmat[2, 1:3] = srinv ** (2 * expon)
    rmat[3, 1:3] = srinv ** (3 * expon)
    q, r = np.linalg.qr(rmat)
    ne = der_init.size
    m = ne - 4
    rhs = np.array([[der_init[i + j] for j in range(m)] for i in range(4)])
    rombcoefs = np.linalg.solve(r, q.T @ rhs)
    der_romb = rombcoefs[0]
    s = np.sqrt(np.sum((rhs - rmat @ rombcoefs) ** 2, axis=0))
    rinv = np.linalg.solve(r, np.eye(3))
    cov1 = np.sum(rinv**2, axis=1)
    errest = s * 12.7062047361747 * np.sqrt(cov1[0])
    return der_romb, errest


def hccm(J: np.ndarray, res: np.ndarray, mode: str = "HC1") -> np.ndarray:
    """Heteroscedasticity-consistent covariance matrix of the parameters from the Jacobian ``J`` and the
    residuals ``res`` (modes ``HC0`` to ``HC5``; EasySpin uses ``HC1``)."""
    J = np.asarray(J, dtype=float)
    res = np.asarray(res, dtype=float).ravel()
    N, M = J.shape
    pinvJTJ = np.linalg.pinv(J.T @ J)
    h = np.sum((J @ pinvJTJ) * J, axis=1)
    mode = mode.upper()
    if mode == "HC0":
        diagV = res**2
    elif mode == "HC1":
        diagV = N / (N - M) * res**2
    elif mode == "HC2":
        diagV = res**2 / (1 - h)
    elif mode == "HC3":
        diagV = (res / (1 - h)) ** 2
    elif mode == "HC4":
        diagV = res**2 / (1 - h) ** np.minimum(4, N * h / M)
    elif mode == "HC5":
        alpha = np.minimum(np.maximum(4, 0.7 * h.max() / h.mean()), h / h.mean())
        diagV = res**2 / np.sqrt((1 - h) ** alpha)
    else:
        raise ValueError("HCCM estimation mode not found.")
    return pinvJTJ @ (J.T * diagV[None, :]) @ J @ pinvJTJ


# ---------------------------------------------------------------------------------- data helpers
def deriv(*args) -> np.ndarray:
    """Numerical derivative along the first axis: ``deriv(y)`` or ``deriv(x, y)`` (EasySpin's ``deriv``)."""
    if len(args) == 1:
        y, x = np.asarray(args[0], dtype=float), None
    elif len(args) == 2:
        x, y = (np.asarray(a, dtype=float) for a in args)
    else:
        raise ValueError("Wrong number of input arguments!")
    row = y.ndim == 1
    y2 = y[:, None] if row else y
    xx = np.arange(1, y2.shape[0] + 1, dtype=float) if x is None else x.ravel()
    d = np.diff(y2, axis=0) / np.diff(xx)[:, None]
    d = (d[[0, *range(d.shape[0])]] + d[[*range(d.shape[0]), d.shape[0] - 1]]) / 2
    return d[:, 0] if row else d


def rescaledata(y, yref=None, mode: str | None = None, region=None) -> tuple[np.ndarray, float]:
    """Rescale ``y``: ``'maxabs'`` (to a maximum magnitude of 1, or of ``yref``), ``'lsq'`` (least-squares fit
    to ``yref``), ``'int'``, ``'dint'`` (unit integral or double integral) or ``'none'``; ``region`` is a mask
    of the points to use. Returns ``(yscaled, scale)``; the scale is never negative."""
    if isinstance(yref, str):
        mode, yref = yref, None
    if mode is None:
        mode = "maxabs" if yref is None else "lsq"
    y = np.asarray(y, dtype=float)
    if y.ndim != 1:
        raise ValueError("y must be a row or column vector. rescaledata() does not work on 2D arrays.")
    if mode in ("lsq0", "lsq1", "lsq2", "minmax"):
        raise ValueError(f"Mode '{mode}' is obsolete. Use basecorr() instead.")
    if mode not in ("maxabs", "lsq", "int", "dint", "none"):
        raise ValueError(f"Unknown scaling mode '{mode}'.")
    if mode == "lsq":
        if yref is None:
            raise ValueError("For 'lsq' mode, a reference is needed.")
        yref = np.asarray(yref, dtype=float).ravel()
        if yref.size != y.size:
            from scipy.interpolate import Akima1DInterpolator

            yref = Akima1DInterpolator(np.linspace(0, 1, yref.size), yref, method="makima")(np.linspace(0, 1, y.size))
    reg = np.ones(y.size, bool) if region is None else np.asarray(region, dtype=bool).ravel()
    if mode == "maxabs":
        idx = ~np.isnan(y) & reg
        scale = 1 / np.max(np.abs(y[idx]))
        if yref is not None:
            yref = np.asarray(yref, dtype=float).ravel()
            idx = ~np.isnan(yref) & reg
            scale = np.max(np.abs(yref[idx])) * scale
    elif mode == "lsq":
        idx = ~np.isnan(y) & ~np.isnan(yref) & reg
        scale = 1 / (yref[idx] @ y[idx] / (yref[idx] @ yref[idx]))
    elif mode == "int":
        idx = ~np.isnan(y) & reg
        scale = 1 / np.sum(y[idx])
    elif mode == "dint":
        idx = ~np.isnan(y) & reg
        scale = 1 / np.sum(np.cumsum(y[idx]))
    else:
        scale = 1.0
    if np.real(scale) < 0:
        scale = abs(scale)
    return y * scale, float(scale)


def basecorr(data, dim: int | None, n, region=None) -> tuple[np.ndarray, np.ndarray]:
    """Polynomial baseline correction of 1-D or 2-D ``data`` along ``dim`` (1 columns, 2 rows, ``None`` for a
    2-D surface with orders ``n = [n1, n2]``), optionally fitted on the points where ``region`` is true.
    Returns ``(corrected, baseline)``."""
    data = np.asarray(data, dtype=float)
    if data.size == 0 or data.ndim > 2:
        raise ValueError("Only 1D or 2D data can be fitted.")
    two_d = dim is None
    if not two_d and dim not in (1, 2):
        raise ValueError("dim (2nd input) must either 1 (fit columns), 2 (fit rows), or None (2D fit).")
    n_arr = np.atleast_1d(np.asarray(n))
    if np.any(n_arr < 0) or np.any(n_arr % 1) or np.any(n_arr > 6):
        raise ValueError("Order must contain integers between 0 and 6!")
    vector = data.ndim == 1
    if vector:
        data2 = data[:, None]
        dim = 1
    else:
        data2 = data
    if two_d:
        if n_arr.size != 2:
            raise ValueError(
                "For 2D fit, the polynomial order n (3rd input) must have 2 elements, one for each dimension."
            )
        r, c = data2.shape
        for k, size in enumerate((r, c)):
            if n_arr[k] >= size:
                raise ValueError(
                    f"The provided polynomial order n={n_arr[k]} must be smaller than the number of points, {size}."
                )
        xs = np.repeat(np.linspace(-1, 1, r), c)
        ys = np.tile(np.linspace(-1, 1, c), r)
        D = np.column_stack([xs**i * ys**j for j in range(int(n_arr[1]) + 1) for i in range(int(n_arr[0]) + 1)])
        p = np.linalg.lstsq(D, data2.ravel(), rcond=None)[0]
        baseline = (D @ p).reshape(r, c)
    else:
        if n_arr.size != 1:
            raise ValueError("Polynomial order (3rd input) must be a single number.")
        order = int(n_arr[0])
        if order >= data2.shape[dim - 1]:
            raise ValueError(
                f"The provided polynomial order n={order} must be smaller than the number of points, "
                f"{data2.shape[dim - 1]}."
            )
        work = data2.T if dim == 2 else data2
        x = np.linspace(-1, 1, work.shape[0])
        D = x[:, None] ** np.arange(order + 1)
        if region is None:
            p = np.linalg.lstsq(D, work, rcond=None)[0]
        else:
            reg = np.asarray(region, dtype=bool).ravel()
            p = np.linalg.lstsq(D[reg], work[reg], rcond=None)[0]
        baseline = D @ p
        if dim == 2:
            baseline = baseline.T
    corrected = data2 - baseline
    if vector:
        return corrected[:, 0], baseline[:, 0]
    return corrected, baseline


def addnoise(y, snr: float, noisemodel: str, rng: np.random.Generator | int | None = None) -> np.ndarray:
    """Add noise of the given signal-to-noise ratio to ``y``: ``'n'`` normal, ``'u'`` uniform, ``'f'`` 1/f."""
    if np.size(snr) != 1 or snr <= 0:
        raise ValueError("The second input, the signal-to-noise ratio (SNR), must be a positive number.")
    rng = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    y = np.asarray(y)
    shape = y.shape
    if noisemodel == "u":

        def noisefun():
            return rng.random(shape) - 0.5

    elif noisemodel == "n":

        def noisefun():
            return rng.standard_normal(shape)

    elif noisemodel == "f":

        def noisefun():
            return _one_over_f_noise(shape, 1.0, rng)

    else:
        raise ValueError(f"Unknown noise type '{noisemodel}'. Use 'n', 'u', or 'f'.")
    if np.iscomplexobj(y):
        noise = noisefun() + 1j * noisefun()
    else:
        noise = noisefun()
    noise = noise / np.std(noise.ravel(), ddof=1)
    level = np.max(y) - np.min(y)
    level = max(np.real(level), np.imag(level))
    noiselevel = level / snr if level > 0 else 1.0
    return y + noise * noiselevel


def _one_over_f_noise(shape, alpha, rng):
    n = int(np.prod(shape))
    freq = np.arange(n // 2 + 1, dtype=float)
    amp = np.ones(freq.size)
    amp[1:] = freq[1:] ** (-alpha / 2)
    fd = amp * np.exp(2j * np.pi * rng.random(amp.size))
    d = np.concatenate([fd, np.conj(fd[-1:0:-1])]) if n % 2 else np.concatenate([fd, np.conj(fd[-2:0:-1])])
    return np.real(np.fft.ifft(d)).reshape(shape)


# ---------------------------------------------------------------------------------- parameter bookkeeping
@dataclass
class _ParamInfo:
    arg: int  # 1-based argument index
    cell: int  # 1-based component index, 0 for a plain dict
    parcell: int  # 1-based element of a tuple-valued field, 0 otherwise
    fieldname: str
    subscripts: tuple[int, int]  # 1-based (row, column) in the 2-D view of the field
    name: str


def _is_numeric(value) -> bool:
    if isinstance(value, (str, bytes, bool, dict)) or value is None:
        return False
    try:
        arr = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return False
    return arr.size > 0


def _view2d(value) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    return np.atleast_2d(arr) if arr.ndim < 2 else arr


def _getparaminfo(args: list) -> list[_ParamInfo]:
    """The fit parameters named by the numeric fields of the dicts in ``args`` (EasySpin's ``getparaminfo``)."""
    if not isinstance(args, (list, tuple)):
        raise ValueError("Model function arguments are expected as a list.")
    pinfo: list[_ParamInfo] = []
    for a, par in enumerate(args, start=1):
        if par is None or (isinstance(par, (list, tuple)) and len(par) == 0) or (isinstance(par, dict) and not par):
            continue
        cells = [par] if isinstance(par, dict) else list(par)
        for c, cell in enumerate(cells, start=1):
            if not isinstance(cell, dict):
                raise ValueError(f"Arg {a}, cell {c}: dict expected.")
            for fieldname, value in cell.items():
                elements = list(value) if isinstance(value, tuple) else [value]
                for pc, element in enumerate(elements, start=1):
                    if not _is_numeric(element):
                        continue
                    v2 = _view2d(element)
                    scalar = v2.size == 1  # MATLAB's isscalar: one element, whatever the shape
                    for r in range(v2.shape[0]):
                        for col in range(v2.shape[1]):
                            idx_name = "" if scalar else f"({r + 1},{col + 1})"
                            cell_name = "" if isinstance(par, dict) else f"{{{c}}}"
                            pc_name = f"{{{pc}}}" if isinstance(value, tuple) else ""
                            name = f"arg{a}{cell_name}.{fieldname}{pc_name}{idx_name}"
                            pinfo.append(
                                _ParamInfo(
                                    a,
                                    0 if isinstance(par, dict) else c,
                                    pc if isinstance(value, tuple) else 0,
                                    fieldname,
                                    (r + 1, col + 1),
                                    name,
                                )
                            )
    return pinfo


def _container(args: list, p: _ParamInfo) -> dict:
    arg = args[p.arg - 1]
    if p.cell > 0:
        if not isinstance(arg, (list, tuple)) or p.cell > len(arg):
            raise ValueError(f"Not enough cell entries in argument {p.arg}.")
        arg = arg[p.cell - 1]
    if not isinstance(arg, dict):
        raise ValueError("Structure expected.")
    return arg


def _field_element(container: dict, p: _ParamInfo):
    if p.fieldname not in container:
        raise ValueError(f"Field .{p.fieldname} is missing in input {p.arg}.")
    value = container[p.fieldname]
    if p.parcell > 0:
        if not isinstance(value, (tuple, list)) or p.parcell > len(value):
            raise ValueError(f"Field .{p.fieldname} of input {p.arg} has no element {p.parcell}.")
        value = value[p.parcell - 1]
    return value


def _checkparcompatibility(pinfo: list[_ParamInfo], args: list) -> None:
    for k, p in enumerate(pinfo, start=1):
        if p.arg > len(args):
            raise ValueError(f"Parameter {k}: Only {len(args)} args given, cannot access arg no {p.arg}.")
        element = _field_element(_container(args, p), p)
        v2 = _view2d(element)
        if p.subscripts[0] > v2.shape[0] or p.subscripts[1] > v2.shape[1]:
            raise ValueError(f"Parameter {k}: Field .{p.fieldname} doesn't have enough elements.")


def _getparamvalues(args: list, pinfo: list[_ParamInfo]) -> np.ndarray:
    vals = np.zeros(len(pinfo))
    for k, p in enumerate(pinfo):
        element = _view2d(_field_element(_container(args, p), p))
        vals[k] = element[p.subscripts[0] - 1, p.subscripts[1] - 1]
    return vals


def _setparamvalues(args: list, pinfo: list[_ParamInfo], newvals) -> list:
    """A deep copy of ``args`` with the parameters set (fields become float arrays of their original shape)."""
    newvals = np.asarray(newvals, dtype=float).ravel()
    if newvals.size != len(pinfo):
        raise ValueError("Number of elements in parameter vector doesn't match number of free parameters.")
    out = copy.deepcopy(args)
    for k, p in enumerate(pinfo):
        container = _container(out, p)
        element = _field_element(container, p)
        orig = np.asarray(element, dtype=float)
        arr = np.array(orig, dtype=float, copy=True)
        v2 = np.atleast_2d(arr) if arr.ndim < 2 else arr
        v2[p.subscripts[0] - 1, p.subscripts[1] - 1] = newvals[k]
        new_value = float(v2.ravel()[0]) if orig.ndim == 0 else v2.reshape(orig.shape)
        if p.parcell > 0:
            parts = list(container[p.fieldname])
            parts[p.parcell - 1] = new_value
            container[p.fieldname] = tuple(parts) if isinstance(container[p.fieldname], tuple) else parts
        else:
            container[p.fieldname] = new_value
    return out


# ---------------------------------------------------------------------------------- the fit
@dataclass
class FitResult:
    """The result of ``esfit``: the fitted parameters with their uncertainties and the fitted data."""

    algorithm: str
    target: str
    pfit: np.ndarray
    pnames: list[str]
    pfit_full: np.ndarray
    argsfit: list | None
    pstd: np.ndarray | None
    ci95: np.ndarray | None
    cov: np.ndarray | None
    corr: np.ndarray | None
    p_start: np.ndarray
    p_fixed: np.ndarray
    fit: Any
    fitraw: Any
    scale: np.ndarray
    baseline: Any
    baselinetype: list[str]
    mask: Any
    residuals: Any
    ssr: float
    rmsd: float
    redchisquare: float
    bestfithistory: dict = field(default_factory=dict)
    info: dict = field(default_factory=dict)


def _calculate_residuals(A, B, mode, idx, mask=None):
    A = np.array(A, dtype=float)
    B = np.array(B, dtype=float)
    if mask is not None:
        A[~mask] = 0
        B[~mask] = 0
    nan = np.isnan(A) | np.isnan(B)
    A[nan] = 0
    B[nan] = 0
    res0_parts, res_parts = [], []
    for sel in idx:
        r0 = A[sel] - B[sel]
        res0_parts.append(r0)
        if mode == 1:
            res_parts.append(r0)
        elif mode == 2:
            res_parts.append(np.cumsum(r0))
        elif mode == 3:
            res_parts.append(np.cumsum(np.cumsum(r0)))
        elif mode == 4:
            res_parts.append(deriv(r0))
        else:
            res_parts.append(np.abs(np.fft.fft(r0)))
    return np.concatenate(res_parts), np.concatenate(res0_parts)


class _Fitter:
    """The model evaluation, scaling, baseline and residual machinery shared by the algorithms."""

    def __init__(self, data, fcn, args, pinfo, p_start, lb, ub, opts, easyspin, out_arg, p_full=None, keep=None):
        self.data = data
        self.p_full = p_full
        self.keep = keep
        self.fcn = fcn
        self.args = args
        self.pinfo = pinfo
        self.p_start = p_start
        self.lb, self.ub = lb, ub
        self.opts = opts
        self.easyspin = easyspin
        self.out_arg = out_arg
        self.n_sets = len(opts["idx"])
        self.idx = opts["idx"]
        self.sizes = opts["datasize"]
        self.mask = opts["Mask"]
        self.weight = opts["weight"]
        self.track = True
        self.best: dict[str, Any] = {"rmsd": np.inf, "rmsdtarget": np.inf}
        self.history_rmsd: list[float] = []
        self.history_par: list[np.ndarray] = []
        self.rmsd_history: list[float] = []
        self.n_evals = 0

    def p2args(self, pars):
        return _setparamvalues(self.args, self.pinfo, pars)

    def evaluate(self, par) -> np.ndarray:
        self.n_evals += 1
        if self.args is not None:
            out = self.fcn(*self.p2args(par))
        else:
            full = self.p_full.copy()  # the fixed parameters stay in place (EasySpin passes the reduced vector)
            full[self.keep] = par
            out = self.fcn(full)
        if self.easyspin:
            sim = out.spec
        elif self.out_arg is not None:
            sim = out[self.out_arg - 1]
        else:
            sim = out
        if isinstance(sim, np.ndarray) and sim.ndim == 2 and self.n_sets == sim.shape[0] and self.n_sets > 1:
            sim = list(sim)
        sims = sim if isinstance(sim, (list, tuple)) else [sim]
        if len(sims) != self.n_sets:
            raise ValueError(
                "Experimental and model data have unequal number of datasets: "
                f"experimental: {self.n_sets}, model: {len(sims)}"
            )
        vec = np.concatenate([np.asarray(s, dtype=float).ravel() for s in sims])
        if vec.size != self.data.size:
            raise ValueError(
                "Experimental data and model have unequal total number of points: "
                f"experimental: {self.data.size}, model: {vec.size}"
            )
        return vec

    def residuals(self, x):
        """Residuals of the target function and the rmsd for the active parameters ``x``."""
        opts = self.opts
        par = self.p_start.copy()
        par[opts["active"]] = x
        sim = self.evaluate(par)
        expdata = self.data
        mask = self.mask
        baseline = np.zeros(sim.size)
        scale = np.ones(self.n_sets)
        types = []
        for k in range(self.n_sets):
            order = opts["BaseLine"][k]
            types.append(_BASELINE_NAMES[order])
            idx = self.idx[k]
            m = mask[idx]
            e = expdata[idx]
            s = sim[idx]
            auto = opts["AutoScaleID"]
            if order != -1:
                N = self.sizes[k]
                xk = np.arange(1, N + 1) / N
                D = xk[:, None] ** np.arange(order + 1)
                if auto == 0:
                    coeffs = np.linalg.lstsq(D[m], e[m] - s[m], rcond=None)[0]
                    base = D @ coeffs
                    sim[idx] = s + base
                elif auto == 1:
                    D2 = np.column_stack([s, D])
                    coeffs = np.linalg.lstsq(D2[m], e[m], rcond=None)[0]
                    coeffs[0] = abs(coeffs[0])
                    base = D2[:, 1:] @ coeffs[1:]
                    sim[idx] = D2 @ coeffs
                    scale[k] = coeffs[0]
                else:
                    D2 = np.column_stack([s, D])
                    coeffs = np.linalg.lstsq(D2[m], e[m], rcond=None)[0]
                    base = D2[:, 1:] @ coeffs[1:]
                    coeff = np.max(np.abs(e[m] - base[m])) / np.max(np.abs(s[m]))
                    sim[idx] = coeff * s + base
                    scale[k] = coeff
                baseline[idx] = base
            elif auto == 1:
                coeff = abs(float(s[m] @ e[m] / (s[m] @ s[m])))
                sim[idx] = s * coeff
                scale[k] = coeff
            elif auto == 2:
                coeff = np.max(np.abs(e[m])) / np.max(np.abs(s[m]))
                sim[idx] = s * coeff
                scale[k] = coeff
        residuals, residuals0 = _calculate_residuals(sim, expdata, opts["TargetID"], self.idx, mask)
        rmsd = float(np.sqrt(np.mean(np.abs(residuals) ** 2 * self.weight)))
        rmsd0 = float(np.sqrt(np.mean(np.abs(residuals0) ** 2)))
        self.curr = {"rmsd": rmsd0, "sim": sim, "par": par, "scale": scale, "baseline": baseline, "baselinetype": types}
        if self.track:
            self.rmsd_history.append(rmsd0)
            if rmsd < self.best["rmsdtarget"]:
                self.best.update(
                    residuals=residuals0,
                    rmsdtarget=rmsd,
                    rmsd=rmsd0,
                    fit=sim,
                    scale=scale,
                    par=par,
                    baseline=baseline,
                    baselinetype=types,
                )
                self.history_rmsd.append(rmsd0)
                self.history_par.append(par.copy())
        return residuals, rmsd

    def rmsd(self, x) -> float:
        return self.residuals(x)[1]


def _norm_icdf(p):
    return -np.sqrt(2) * erfcinv(2 * p)


def esfit(data, fcn: Callable, p0, *args, vary=None, lb=None, ub=None, opt: dict | None = None) -> FitResult:  # noqa: C901
    """Least-squares fit of the model ``fcn`` to ``data``: EasySpin's ``esfit`` without the GUI.

    ``esfit(data, fcn, p0, vary)``, ``esfit(data, fcn, p0, vary, opt)``, ``esfit(data, fcn, p0, lb, ub)`` or
    ``esfit(data, fcn, p0, lb, ub, opt)``, as in EasySpin; ``vary``, ``lb``, ``ub`` and ``opt`` may also be
    keywords. For ``pepper`` and ``garlic``, ``p0 = [sys, exp]`` or ``[sys, exp, opt]`` and ``vary``/``lb``/
    ``ub`` are lists of dicts whose numeric fields name the parameters (``[{"g": [0.02, 0.02]}]``); for any
    other callable ``fcn(p)`` they are parameter vectors. ``data`` is one array or a list of arrays (global
    fitting). Options: ``Method`` (algorithm ``simplex``, ``levmar``, ``montecarlo``, ``genetic``, ``grid``,
    ``swarm``, ``lsqnonlin`` and target ``fcn``, ``int``, ``dint``, ``diff``, ``fft``), ``AutoScale`` (``lsq``,
    ``maxabs``, ``none``), ``BaseLine`` (polynomial order per dataset), ``Mask``, ``weight``, ``OutArg``,
    ``RandomSeed``, ``CalculateUncertainties``, ``Verbosity``, and the algorithm options."""
    # Argument list: 4, 5 or 6 positional inputs as in EasySpin
    if len(args) == 1:
        vary = args[0]
    elif len(args) == 2:
        if isinstance(args[1], dict) and not isinstance(args[0], dict):
            vary, opt = args
        else:
            lb, ub = args
    elif len(args) == 3:
        lb, ub, opt = args
    elif len(args) > 3:
        raise ValueError(
            "esfit takes 4, 5 or 6 inputs: esfit(data, fcn, p0, vary), esfit(data, fcn, p0, vary, opt), "
            "esfit(data, fcn, p0, lb, ub) or esfit(data, fcn, p0, lb, ub, opt)."
        )
    vary_provided = vary is not None
    if not vary_provided and (lb is None or ub is None):
        raise ValueError("Provide either vary, or lb and ub.")
    opt = dict(opt or {})
    if not callable(fcn):
        raise ValueError("The simulation/model function (2nd input) must be callable.")
    rng = np.random.default_rng(opt.get("RandomSeed"))
    opt["rng"] = rng

    # Experimental data
    datasets = list(data) if isinstance(data, (list, tuple)) else [data]
    vecs = []
    for d in datasets:
        d = np.asarray(d, dtype=float).ravel()
        if d.size == 0:
            raise ValueError(
                "First input must be numeric experimental data in the form of a vector or a list of vectors."
            )
        if np.any(np.isnan(d)):
            raise ValueError("The experimental data must not contain NaN.")
        vecs.append(d)
    data_vec = np.concatenate(vecs)
    sizes = [v.size for v in vecs]
    idx = []
    offsets = np.concatenate([[0], np.cumsum(sizes)])
    for k in range(len(vecs)):
        sel = np.zeros(data_vec.size, bool)
        sel[offsets[k] : offsets[k + 1]] = True
        idx.append(sel)
    n_sets = len(vecs)

    fcn_name = getattr(fcn, "__name__", "model")
    easyspin = fcn_name in _EASYSPIN_FUNCTIONS
    cell_inputs = isinstance(p0, (list, tuple)) and len(p0) > 0 and isinstance(p0[0], (dict, list, tuple))
    if isinstance(p0, dict):
        raise ValueError(
            "The third input must be a list such as [sys, exp] (when using an EasySpin function) or an array of "
            "initial parameter values (when using a custom function)."
        )

    # Parameters, bounds and names
    if cell_inputs:
        p0 = copy.deepcopy(list(p0))
        if vary_provided:
            pinfo = _getparaminfo(list(vary))
            _checkparcompatibility(pinfo, p0)
            pvec_0 = _getparamvalues(p0, pinfo)
            pvec_vary = _getparamvalues(list(vary), pinfo)
            pvec_lb = pvec_0 - pvec_vary
            pvec_ub = pvec_0 + pvec_vary
            nonneg = np.array([p.fieldname in _NONNEGATIVE_FIELDS for p in pinfo]) & (pvec_lb < 0)
            pvec_lb[nonneg] = 0
        else:
            pinfo = _getparaminfo(list(lb))
            _checkparcompatibility(pinfo, p0)
            _checkparcompatibility(pinfo, list(ub))
            pvec_0 = _getparamvalues(p0, pinfo)
            pvec_lb = _getparamvalues(list(lb), pinfo)
            pvec_ub = _getparamvalues(list(ub), pinfo)
        names = [p.name for p in pinfo]
    else:
        pvec_0 = np.asarray(p0, dtype=float).ravel()
        if vary_provided:
            pvec_vary = np.asarray(vary, dtype=float).ravel()
            pvec_lb, pvec_ub = pvec_0 - pvec_vary, pvec_0 + pvec_vary
        else:
            pvec_lb, pvec_ub = np.asarray(lb, dtype=float).ravel(), np.asarray(ub, dtype=float).ravel()
        pinfo = []
        names = [f"p({k + 1})" for k in range(pvec_0.size)]
    n_params = pvec_0.size
    if pvec_lb.size != n_params:
        raise ValueError(f"Vector of lower bounds has {pvec_lb.size} elements, but {n_params} are expected.")
    if pvec_ub.size != n_params:
        raise ValueError(f"Vector of upper bounds has {pvec_ub.size} elements, but {n_params} are expected.")
    bad = np.where(pvec_lb > pvec_ub)[0]
    if bad.size:
        raise ValueError(f"Parameter #{bad[0] + 1}: upper bound cannot be smaller than lower bound.")
    bad = np.where(pvec_0 < pvec_lb)[0]
    if bad.size:
        raise ValueError(f"Parameter #{bad[0] + 1}: start value is smaller than lower bound.")
    bad = np.where(pvec_0 > pvec_ub)[0]
    if bad.size:
        raise ValueError(f"Parameter #{bad[0] + 1}: start value is larger than upper bound.")
    keep = pvec_lb != pvec_ub  # fixed parameters are dropped from the fit
    p_full0 = pvec_0.copy()
    p_full0 = pvec_0.copy()
    pinfo = [p for p, k in zip(pinfo, keep, strict=True) if k] if pinfo else pinfo
    names = [n for n, k in zip(names, keep, strict=True) if k]
    pvec_0, pvec_lb, pvec_ub = pvec_0[keep], pvec_lb[keep], pvec_ub[keep]
    n_parameters = pvec_0.size
    if n_parameters == 0:
        raise ValueError("No variable parameters to fit.")
    if n_parameters > opt.get("maxParameters", 30):
        raise ValueError(f"Cannot fit more than {opt.get('maxParameters', 30)} parameters simultaneously.")

    # Experimental parameters of EasySpin functions
    if easyspin:
        if n_sets > 1:
            raise ValueError(
                "Cannot use EasySpin functions for global fitting directly. Write a custom model function."
            )
        if not cell_inputs or len(p0) < 2:
            raise ValueError("The third input must contain the initial parameters, e.g. [sys, exp] or [sys, exp, opt].")
        exp0 = dict(p0[1])
        if exp0.get("nPoints") is not None:
            if int(exp0["nPoints"]) != sizes[0]:
                raise ValueError(f"Exp.nPoints is {exp0['nPoints']}, but the data vector has {sizes[0]} elements.")
        else:
            exp0["nPoints"] = sizes[0]
        if not any(exp0.get(k) is not None for k in ("Range", "CenterSweep", "mwRange", "mwCenterSweep")):
            raise ValueError(
                "Please specify field or frequency range, in Exp.Range/Exp.mwRange or in "
                "Exp.CenterSweep/Exp.mwCenterSweep."
            )
        p0[1] = exp0

    # Options
    verbosity = opt.get("Verbosity", 1)
    if "Scaling" in opt:
        raise ValueError("Fitting option Opt.Scaling has been replaced by Opt.AutoScale.")
    out_arg = None
    if "OutArg" in opt:
        n_out, i_out = (int(v) for v in np.asarray(opt["OutArg"]).ravel())
        if i_out > n_out:
            raise ValueError("Opt.OutArg: second number cannot be larger than first one.")
        out_arg = i_out
    method = opt.get("Method", "simplex fcn")
    target_id = None
    if easyspin:
        harmonic = exp0.get("Harmonic")
        if harmonic is not None and harmonic > 0:
            target_id = 2
        elif fcn_name in ("pepper", "garlic"):
            target_id = 2
    algorithm_id = 1
    for word in str(method).split():
        if word in _ALGORITHMS:
            algorithm_id = _ALGORITHMS[word]
        elif word in _TARGETS:
            target_id = _TARGETS[word]
        else:
            raise ValueError(f"Unknown '{word}' in Opt.Method.")
    if target_id is None:
        target_id = 1
    if "Mask" in opt:
        mask_in = opt["Mask"]
        mask = (
            np.concatenate([np.asarray(m).ravel() for m in mask_in])
            if isinstance(mask_in, (list, tuple))
            else np.asarray(mask_in).ravel()
        )
        mask = mask.astype(bool)
        if mask.size != data_vec.size:
            raise ValueError(f"Opt.Mask has {mask.size} elements, but the data has {data_vec.size} elements.")
    else:
        mask = np.ones(data_vec.size, bool)
    autoscale = opt.get("AutoScale", "lsq" if easyspin or n_sets > 1 else "none")
    if autoscale not in _AUTOSCALE:
        raise ValueError("Unknown setting for Opt.AutoScale - possible values are 'lsq', 'maxabs' and 'none'.")
    baseline_opt = opt.get("BaseLine")
    if baseline_opt is None or np.size(baseline_opt) == 0:
        baseline = [-1] * n_sets
    else:
        baseline = [int(b) for b in np.atleast_1d(np.asarray(baseline_opt)).ravel()]
        if len(baseline) == 1 and n_sets != 1:
            baseline = baseline * n_sets
    if len(baseline) != n_sets:
        raise ValueError(f"The number of entries in Opt.BaseLine must be equal to the number of datasets ({n_sets}).")
    weights = np.atleast_1d(np.asarray(opt.get("weight", np.ones(n_sets)), dtype=float)).ravel()
    if weights.size != n_sets:
        raise ValueError("The number of elements in Opt.weight must be equal to the number of datasets.")
    weight = np.zeros(data_vec.size)
    for k in range(n_sets):
        weight[idx[k]] = weights[k]
    opts = {
        **opt,
        "idx": idx,
        "datasize": sizes,
        "Mask": mask,
        "weight": weight,
        "AutoScaleID": _AUTOSCALE[autoscale],
        "BaseLine": baseline,
        "TargetID": target_id,
        "AlgorithmID": algorithm_id,
        "active": np.ones(n_parameters, bool),
        "Verbosity": verbosity,
    }
    if verbosity >= 1:
        print("-- esfit ------------------------------------------------")
        print(f"Number of datasets:       {n_sets}")
        for k in range(n_sets):
            print(f"Data set {k + 1}:           {sizes[k]} points")
        print(f"Model function name:      {fcn_name}")
        print(f"Number of fit parameters: {n_parameters}")
        print(f"Minimization algorithm:   {_ALGORITHM_NAMES[algorithm_id - 1]}")
        print(f"Residuals computed from:  {_TARGET_NAMES[target_id - 1]}")
        print(f"Autoscaling:              {autoscale}")
        print("---------------------------------------------------------")

    fitter = _Fitter(
        data_vec,
        fcn,
        p0 if cell_inputs else None,
        pinfo,
        pvec_0.copy(),
        pvec_lb,
        pvec_ub,
        opts,
        easyspin,
        out_arg,
        p_full0,
        keep,
    )
    alg_opt = {k: v for k, v in opt.items() if k not in ("Method", "Mask", "BaseLine", "weight", "AutoScale", "OutArg")}
    alg_opt["Verbosity"] = verbosity
    if verbosity >= 1:
        print(f"Running optimization algorithm with {n_parameters} active parameters...")
    residualfun = lambda x: fitter.residuals(x)[0]  # noqa: E731
    rmsdfun = fitter.rmsd
    alg_info: dict = {}
    if algorithm_id == 1:
        pfit_active, alg_info = esfit_simplex(rmsdfun, pvec_0, pvec_lb, pvec_ub, alg_opt)
    elif algorithm_id == 2:
        pfit_active, alg_info = esfit_levmar(residualfun, pvec_0, pvec_lb, pvec_ub, alg_opt)
    elif algorithm_id == 3:
        pfit_active, alg_info = esfit_montecarlo(rmsdfun, pvec_lb, pvec_ub, alg_opt)
    elif algorithm_id == 4:
        pfit_active, alg_info = esfit_genetic(rmsdfun, pvec_lb, pvec_ub, alg_opt)
    elif algorithm_id == 5:
        pfit_active, alg_info = esfit_grid(rmsdfun, pvec_lb, pvec_ub, alg_opt)
    elif algorithm_id == 6:
        pfit_active, alg_info = esfit_swarm(rmsdfun, pvec_lb, pvec_ub, alg_opt)
    else:
        from scipy.optimize import least_squares

        sol = least_squares(residualfun, pvec_0, bounds=(pvec_lb, pvec_ub), method="trf")
        pfit_active = sol.x
        alg_info = {"msg": sol.message, "nEvals": sol.nfev, "stop": sol.status}
    pfit_active = np.asarray(pfit_active, dtype=float).ravel()
    pfit = p_full0.copy()  # all parameters, the fixed ones in place
    pfit[keep] = pfit_active
    argsfit = fitter.p2args(pfit_active) if cell_inputs else None

    # Best-fit data and goodness of fit
    best = fitter.best
    fit = best["fit"]
    scale = best["scale"]
    fitraw = fit.copy()
    for k in range(n_sets):
        fitraw[idx[k]] = fit[idx[k]] / scale[k]
    residuals0 = best["residuals"]
    rmsd0 = best["rmsd"]
    ssr0 = float(np.sum(np.abs(residuals0) ** 2))
    noise_std = float(np.std(residuals0, ddof=1)) if residuals0.size > 1 else 0.0
    N = data_vec.size
    noise_der = (
        1.482602 / np.sqrt(6) * np.median(np.abs(2.0 * data_vec[2 : N - 2] - data_vec[0 : N - 4] - data_vec[4:N]))
        if N >= 5
        else np.nan
    )
    if noise_std > noise_der:
        red_chisquare = np.sum(residuals0**2) / N / noise_der**2
        chisq_info = "(using noise der estimate)"
    else:
        red_chisquare = np.sum(residuals0**2) / N / noise_std**2 if noise_std > 0 else np.nan
        chisq_info = "(using noise std estimate; upper limit)"

    # Parameter uncertainties
    pstd = ci95 = covmatrix = corrmatrix = None
    if opt.get("CalculateUncertainties", 1):
        if verbosity >= 1:
            print("Calculating parameter uncertainties...\n  Estimating Jacobian...")
        fitter.track = False
        J, _ = jacobianest(residualfun, pfit_active)
        if J.size and not np.any(np.isnan(J)):
            if verbosity >= 1:
                print("  Calculating parameter covariance matrix...")
            residuals, _ = _calculate_residuals(fit, data_vec, target_id, idx)
            covmatrix = hccm(J, residuals, "HC1")
            pstd = np.sqrt(np.diag(covmatrix))
            ci = _norm_icdf(0.5 + 0.95 / 2) * pstd
            ci95 = np.column_stack([pfit_active - ci, pfit_active + ci])
            with np.errstate(divide="ignore", invalid="ignore"):
                Q = np.diag(np.diag(covmatrix) ** -0.5)
                corrmatrix = Q @ covmatrix @ Q
            if verbosity >= 1:
                lines = [""]
                if opts["AutoScaleID"] != 0:
                    lines.append(f"Fitted scale:      {np.array2string(scale, precision=6)}")
                lines.append("Parameters:")
                lines.append(_printparlist(pfit_active, names, pstd, ci95))
                if n_parameters > 1:
                    lines.append("Correlation matrix:")
                    lines.append(np.array2string(corrmatrix, precision=6, suppress_small=True))
                    triu = np.triu(np.abs(corrmatrix), 1)
                    order = np.argsort(triu.ravel(), kind="stable")[::-1]
                    lines.append("Strongest correlations:")
                    for flat in order[: min(5, n_parameters * (n_parameters - 1) // 2)]:
                        i1, i2 = np.unravel_index(flat, corrmatrix.shape)
                        lines.append(f"    p({i1 + 1})-p({i2 + 1}):    {corrmatrix[i1, i2]:g}")
                    if np.any(triu > 0.8):
                        lines.append("    WARNING! Strong correlations between parameters.")
                lines += [
                    "",
                    "Goodness of fit:",
                    f"   ssr             {ssr0:g}",
                    f"   rmsd            {rmsd0:g}",
                    f"   noise std       {noise_std:g} (estimated from residuals; assumes excellent fit)",
                    f"   noise der       {noise_der:g} (estimated using der_snr algorithm)",
                    f"   red chi-square  {red_chisquare:g} {chisq_info}",
                ]
                print("-" * 110)
                print("\n".join(lines))
                print("-" * 110)
        elif verbosity >= 1:
            print("  NaN elements in Jacobian, cannot calculate parameter uncertainties.")
    elif verbosity >= 1:
        print("Skipping uncertainty quantification as requested.")

    def split(vec):
        return [vec[sel] for sel in idx] if n_sets > 1 else vec

    history = {
        "rmsd": np.asarray(fitter.history_rmsd),
        "pfit": np.array(fitter.history_par).T if fitter.history_par else np.zeros((n_parameters, 0)),
    }
    if cell_inputs:
        history["pfit2structs"] = fitter.p2args
    return FitResult(
        algorithm=[k for k, v in _ALGORITHMS.items() if v == algorithm_id][0],
        target=[k for k, v in _TARGETS.items() if v == target_id][0],
        pfit=pfit_active,
        pnames=names,
        pfit_full=pfit,
        argsfit=argsfit,
        pstd=pstd,
        ci95=ci95,
        cov=covmatrix,
        corr=corrmatrix,
        p_start=pvec_0,
        p_fixed=~keep,
        fit=split(fit),
        fitraw=split(fitraw),
        scale=scale,
        baseline=split(best["baseline"]),
        baselinetype=best["baselinetype"],
        mask=split(mask),
        residuals=split(residuals0),
        ssr=ssr0,
        rmsd=rmsd0,
        redchisquare=float(red_chisquare),
        bestfithistory=history,
        info={**alg_info, "nEvals": fitter.n_evals, "rmsdhistory": np.asarray(fitter.rmsd_history)},
    )


def _printparlist(par, names, pstd, ci95) -> str:
    width = max(len(n) for n in names)
    indent = "   "
    lines = [
        indent
        + "    name"
        + " " * (max(width - 4, 0) + 2)
        + "value        standard deviation        95% confidence interval"
    ]
    for p, (name, value) in enumerate(zip(names, par, strict=True), start=1):
        rel = pstd[p - 1] / value * 100 if value != 0 else np.inf
        lines.append(
            f"{indent}{p:2d}  {name.ljust(width)}  {value:<#12.7g} {pstd[p - 1]:<#12.7g} ({rel:6.3f} %)   "
            f"{ci95[p - 1, 0]:<#12.7g} - {ci95[p - 1, 1]:<#12.7g}"
        )
    return "\n".join(lines)
