"""Shared GUI state container used across tabs and controller."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

# Store the current state of the app in a centralized fashion

ArrayLike = np.ndarray


@dataclass
class AppState:
    """Central application state used by GUI components.

    Attributes
    ----------
    is_monthly : bool
        Whether input/target series are monthly (else yearly).
    tracer : str
        LEGACY: single-tracer selection kept for backward compatibility.
    tracer1 : str
        Primary tracer (e.g., ``"Tritium"``, ``"Carbon-14"``).
    tracer2 : str | None
        Optional second tracer; if ``None`` or ``"None"``, runs single-tracer mode.
    solver_key : str
        Selected solver registry key (``"de"``, ``"lsq"``, or ``"mcmc"``).
    solver_params : dict
        Per-solver configuration dictionary.
    input_series : tuple(ndarray, ndarray) | None
        Timestamps and input values.
    target_series : tuple(ndarray, ndarray) | None
        Timestamps and observation values (optional).
    selected_units : list of str
        Unique unit types (legacy support for parameters tab).
    design_units : list of (str, float)
        Flat list of chosen units and fractions (can repeat unit types).
    unit_fractions : dict[str, float]
        Per-instance fraction mapping keyed by unique instance prefix.
    params : dict
        Nested parameter values/bounds/fixed flags per instance prefix.
    steady_state_input : float | list of float
        Value(s) for optional steady-state warmup input. Provide a scalar for
        single-tracer runs or one value per tracer for multi-tracer runs.
    n_warmup_half_lives : int
        Warmup length in half-lives of the tracer.
    last_simulation : object | None
        Last simulation result or solver payload for plotting.
    last_times : ndarray | None
        Cached timestamps for plotting (if needed).
    """

    is_monthly: bool = True
    is_halfyearly: bool = False
    tracer1: str = "Tritium"
    tracer2: Optional[str] = None
    # Selected solver (registry key); default to Differential Evolution
    solver_key: str = "de"
    # Per-solver parameters
    solver_params: Dict[str, dict] = field(
        default_factory=lambda: {
            "de": {
                "maxiter": 10000,
                "popsize": 15,
                "mutation": (0.5, 1.0),
                "recombination": 0.7,
                "tol": 0.01,
                "sigma": None,
            },
            "lsq": {
                "ftol": 1e-8,
                "xtol": 1e-8,
                "gtol": 1e-8,
                "max_nfev": 10000,
                "sigma": None,
            },
            "mcmc": {
                "n_samples": 10000,
                "burn_in": 2000,
                "thin": 2,
                "rw_scale": 0.05,
                "sigma": None,
            },
        }
    )
    input_series: Optional[Tuple[ArrayLike, ArrayLike]] = None
    target_series: Optional[Tuple[ArrayLike, ArrayLike]] = None
    manual_observations: Dict[datetime, List[float]] = field(default_factory=dict)
    selected_units: List[str] = field(default_factory=list)  # unique registry keys for params tab
    # detailed design: list of (unit_name, fraction) allowing duplicates
    design_units: List[Tuple[str, float]] = field(default_factory=list)
    # aggregated by prefix for controller/model: {"epm": 0.5, ...}
    unit_fractions: Dict[str, float] = field(default_factory=dict)
    params: Dict[str, Dict[str, Dict[str, float]]] = field(default_factory=dict)
    # params[prefix][key] = {"val":..., "lb":..., "ub":..., "fixed":0/1}
    steady_state_input: Union[float, List[float]] = 0.0
    n_warmup_half_lives: int = 10
    tt_results: Optional[ArrayLike] = None
    tt_mtt_values: Optional[np.ndarray] = None
    tt_obs_indices: Optional[np.ndarray] = None
    tt_param_key: Optional[str] = None
    # Can be either a plain ndarray (legacy) or a payload dict from solver.run_solver
    last_simulation: Optional[object] = None
    last_times: Optional[ArrayLike] = None

    def clear_tracer_tracer(self) -> None:
        """Reset cached tracer-tracer sweep results."""
        self.tt_results = None
        self.tt_mtt_values = None
        self.tt_obs_indices = None
        self.tt_param_key = None
