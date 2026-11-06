"""Measurement uncertainty settings of the GUI (shared by all solvers).

Translates the settings in :class:`~PyTracerLab.gui.state.AppState` into the
``sigma`` / ``sigma_rel`` arguments of the solvers (see
:class:`~PyTracerLab.model.solver.Solver`). The module does not depend on Qt.

Settings
--------
- ``use_file_uncertainty``: use the uncertainties from the ``_sd`` columns of
  the observation file where given.
- ``uncertainty_mode`` for all other observations: ``"none"`` (DE and least
  squares unweighted, MCMC and DREAM estimate the standard deviation of each
  tracer), ``"absolute"`` (``sigma_abs``, one standard deviation per tracer)
  or ``"relative"`` (``sigma_rel_percent`` of the observed value).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from ..model.solver import _observation_sigma

MODES = ("none", "absolute", "relative")


def observations(state) -> Optional[np.ndarray]:
    """Observed values as an array ``(N, K)``, or ``None``."""
    target = getattr(state, "target_series", None)
    if target is None or target[1] is None:
        return None
    y = np.asarray(target[1], dtype=float)
    return y.reshape(len(y), -1)


def file_uncertainty(state) -> Optional[np.ndarray]:
    """Uncertainties from the observation file as an array ``(N, K)``.

    ``None`` if no observation file is loaded or the file has no ``_sd``
    values.
    """
    y = observations(state)
    data = getattr(state, "target_data", None)
    sd = getattr(data, "uncertainty", None)
    if y is None or sd is None:
        return None
    sd = np.asarray(sd, dtype=float).reshape(y.shape[0], -1)
    if sd.shape != y.shape or not np.any(np.isfinite(sd) & ~np.isnan(y)):
        return None
    return sd


def _n_tracers(state) -> int:
    y = observations(state)
    if y is not None:
        return y.shape[1]
    t2 = getattr(state, "tracer2", None)
    return 2 if t2 and str(t2).lower() not in {"", "none"} else 1


def solver_sigma_kwargs(state) -> Dict[str, Any]:
    """``sigma`` and ``sigma_rel`` for the solvers from the GUI settings."""
    sd = file_uncertainty(state) if getattr(state, "use_file_uncertainty", True) else None
    mode = getattr(state, "uncertainty_mode", "none")
    sigma: Any = sd
    sigma_rel = None
    if mode == "absolute":
        k = _n_tracers(state)
        given = list(getattr(state, "sigma_abs", None) or [])
        vals = np.array(
            [np.nan if i >= len(given) or given[i] is None else float(given[i]) for i in range(k)]
        )
        sigma = vals if sd is None else np.where(np.isfinite(sd), sd, vals[np.newaxis, :])
    elif mode == "relative":
        pct = getattr(state, "sigma_rel_percent", None)
        if pct is not None:
            sigma_rel = float(pct) / 100.0
    return {"sigma": sigma, "sigma_rel": sigma_rel}


def effective_sigma(state) -> Optional[np.ndarray]:
    """Standard deviation of each observation as used by the solvers (for plots).

    Returns ``None`` if no uncertainties are specified. If the settings
    leave some observations without uncertainty (the solver would report an
    error), only the values from the observation file are returned.
    """
    y = observations(state)
    if y is None:
        return None
    kw = solver_sigma_kwargs(state)
    try:
        return _observation_sigma(y, kw["sigma"], kw["sigma_rel"])
    except ValueError:
        sd = file_uncertainty(state)
        return None if sd is None else np.where(np.isnan(y), np.nan, sd)


def _fmt(values) -> str:
    return " / ".join("–" if v is None else f"{float(v):g}" for v in values)


def summary(state) -> str:
    """One-line description of the current settings for the GUI."""
    y = observations(state)
    sd = file_uncertainty(state)
    use_file = bool(getattr(state, "use_file_uncertainty", True)) and sd is not None
    mode = getattr(state, "uncertainty_mode", "none")
    k = _n_tracers(state)

    if mode == "absolute":
        other = f"absolute σ = {_fmt((list(state.sigma_abs) + [None, None])[:k])}"
    elif mode == "relative" and getattr(state, "sigma_rel_percent", None) is not None:
        other = f"{float(state.sigma_rel_percent):g} % of the observed value"
    else:
        other = None

    if use_file:
        observed = ~np.isnan(y)
        n_file = int(np.sum(observed & np.isfinite(sd)))
        n_missing = int(np.sum(observed & ~np.isfinite(sd)))
        text = f"from the observation file ({n_file} values)"
        if n_missing and other:
            text += f"; other observations: {other}"
        elif n_missing:
            text += f"; {n_missing} observation(s) without uncertainty – choose a fallback"
    elif other:
        text = other
    else:
        text = "not specified (DE, least squares: unweighted; MCMC, DREAM: estimated per tracer)"
    return f"Measurement uncertainty: {text}"
