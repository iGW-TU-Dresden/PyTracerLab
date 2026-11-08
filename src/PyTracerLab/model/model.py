"""Model container with a parameter registry."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import scipy.signal

from .units import Unit

# Each registry record stores numeric state + metadata used by the solver.
ParamRecord = Dict[str, object]

# Upper limit for the number of warm-up time steps.
_MAX_WARMUP_STEPS = 5000

# Local name of the mixing-fraction parameter in the registry.
FRACTION = "fraction"


class PyTracerLabWarning(UserWarning):
    """Warning about the model or calibration setup (e.g. a capped warm-up)."""


@dataclass
class Model:
    """Forward model container with a parameter registry.

    The model aggregates units, keeps their mixing fractions, and performs the
    convolution-based simulation. It also manages an explicit parameter
    registry that stores **current values**, **initial values**, **optimizer
    bounds**, and **fixed flags** per parameter.

    Parameters
    ----------
    dt : float
        Time step of the simulation (same units as ``mtt`` used by units).
    lambda_ : float or ndarray
        Decay constant(s) in 1/time units. Provide a scalar for single-tracer
        runs or an array-like of length ``n_tracers`` for multi-tracer runs.
    input_series : ndarray
        Forcing time series of shape ``(N,)`` for single tracer or ``(N, K)``
        for ``K`` tracers.
    production : bool or sequence of bool, optional
        If True, simulate production from decay. If a bool, this is a global
        setting. Use a sequence of bools for per-tracer specification. The
        order of the sequence must match the order of the columns in the
        ``input_series``. The input should contain the parent tracer from
        which production is simulated.
    target_series : ndarray, optional
        Observed output series of shape ``(N,)`` or ``(N, K)``; used only for
        calibration/loss and reporting.
    steady_state_input : float or sequence of float, optional
        If provided, a warmup of constant input is prepended. Supply a scalar
        for single-tracer runs or one value per tracer for multi-tracer runs.
    n_warmup_half_lives : int, optional
        Heuristic warmup scaling in half-lives (kept for compatibility).
    n_warmup_steps : int, optional
        Number of warmup time steps prepended to the input series. If given,
        it overrides the warmup steps calculated from ``n_warmup_half_lives``.

    Notes
    -----
    - Units are added via :meth:`add_unit`. The method also registers unit
      parameters into the model's registry.
    - Bounds are **optimization bounds** only and can be provided at add time
      or later via :meth:`set_bounds`.
    - **Mixing fraction.** For a binary mixture (exactly two units), the
      fraction of the first unit is a registry parameter
      ``"<prefix>.fraction"`` (bounds ``(0, 1)``); the second unit
      contributes ``1 - fraction``. It is fixed by default and can be
      calibrated like any other parameter::

          m.add_unit(lab.EPMUnit(mtt=10.0, eta=1.5), fraction=0.6, prefix="young",
                     bounds=[(1.0, 100.0), (1.0, 3.0)])
          m.add_unit(lab.PMUnit(mtt=500.0), fraction=0.4, prefix="old",
                     bounds=[(100.0, 1000.0)])
          m.set_fixed("old.mtt")                 # often necessary, see below
          m.set_fixed("young.fraction", False)   # calibrate the mixing fraction
          m.set_bounds("young.fraction", (0.2, 0.8))

      Whether the fraction and the mean travel times of both units can be
      determined together depends on the number of independent observations
      and on the tracers (e.g. tracers with contrasting input histories). If
      the information is limited, fix one mean travel time. With three or
      more units, the fractions are fixed model properties.

    """

    dt: float
    lambda_: Union[float, np.ndarray]
    input_series: np.ndarray
    production: Optional[Union[bool, Sequence[bool]]] = False
    target_series: Optional[np.ndarray] = None
    steady_state_input: Optional[Union[float, Sequence[float]]] = None
    n_warmup_half_lives: int = 2
    n_warmup_steps: int = None
    time_steps: Optional[Union[Sequence, np.ndarray]] = None

    units: List[Unit] = field(default_factory=list)
    unit_fractions: List[float] = field(default_factory=list)
    # parameter prefix of each unit (same order as ``units``)
    unit_prefixes: List[str] = field(default_factory=list, init=False, repr=False)

    # Parameter registry: key -> record
    params: Dict[str, ParamRecord] = field(default_factory=dict, init=False)

    # Parameter uncertainty utility for GUI
    # We want to have a structure that allows us to transfer uncertainty
    # estimates of model parameters from the GUI-solver utilities to the
    # model. Only then can we easily pass those values to the report.
    # Otherwise we would need to transfer everything via the GUI itself.
    # For each parameter (keyed in the dict), we contain the 1%-50%-99%
    # quantiles of parameters.
    param_uncert: Dict[str, List[float, float, float]] = None
    # We create a similar dict for the parameter maximum a posteriori (MAP)
    # values.
    param_map: Dict[str, float] = None

    # Internal warmup state
    _is_warm: bool = field(default=False, init=False, repr=False)
    _n_warmup: int = field(default=0, init=False, repr=False)

    def add_unit(
        self,
        unit: Unit,
        fraction: float,
        prefix: Optional[str] = None,
        bounds: Optional[List[Tuple[float, float]]] = None,
    ) -> None:
        """Add a unit, register its parameters, and set its mixture fraction.

        Adding the second unit also registers the mixing fraction of the
        first unit as parameter ``"<prefix>.fraction"`` (fixed by default,
        bounds ``(0, 1)``); adding a third unit removes it again, because
        mixing fractions can only be calibrated for binary mixtures (see
        :class:`Model`).

        Unit parameters are registered as free, except those the unit marks
        as fixed by default (``"fixed": True`` in its ``PARAMS``, e.g. the
        screen geometry of :class:`~PyTracerLab.model.units.PEMUnit`); free
        them with :meth:`set_fixed`.

        Parameters
        ----------
        unit : :class:`~PyTracerLab.model.units.Unit`
            The unit instance to add.
        fraction : float
            Mixture fraction of this unit in the overall response, between 0
            and 1. Fractions must sum to ~1 across all units.
        prefix : str, optional
            Namespace prefix for the unit's parameters (e.g., ``"epm"``). If
            omitted, ``"u{index}"`` is used in insertion order.
        bounds : list of (float, float), optional
            Optimizer bounds for the unit's parameters in the same order as
            returned by ``unit.param_values()``. If omitted, bounds are left
            as ``None`` and can be supplied later via :meth:`set_bounds`.

        Raises
        ------
        ValueError
            If ``bounds`` is provided and its length does not match the number
            of unit parameters, if ``fraction`` is not between 0 and 1, if the
            prefix is already used, or if a third unit is added while the
            mixing fraction of a binary mixture is free.
        """
        idx = len(self.units)
        prefix = prefix or f"u{idx}"
        frac = float(fraction)
        local_params = list(unit.param_values().items())

        # validate everything before the model is changed
        if bounds is not None and len(bounds) != len(local_params):
            raise ValueError("Length of bounds list must match number of unit parameters")
        if not 0.0 <= frac <= 1.0:
            raise ValueError(
                f"The fraction of unit '{prefix}' must be between 0 and 1 (got {frac})."
            )
        if prefix in self.unit_prefixes:
            raise ValueError(f"The prefix '{prefix}' is already used by another unit.")
        if any(name == FRACTION for name, _ in local_params):
            raise ValueError(f"Unit parameters must not be named '{FRACTION}' (reserved).")
        frac_key = self.fraction_key()
        if frac_key is not None and not self.params[frac_key]["fixed"]:
            raise ValueError(
                f"The mixing fraction can only be calibrated for two units, but '{frac_key}' "
                f"is free. Fix it with set_fixed('{frac_key}') before adding a third unit."
            )

        if frac_key is not None:
            del self.params[frac_key]  # three or more units: fractions are fixed
        self.units.append(unit)
        self.unit_fractions.append(frac)
        self.unit_prefixes.append(prefix)

        # parameters a unit marks as fixed by default (e.g. the PEM geometry)
        spec = {p.get("key"): p for p in getattr(type(unit), "PARAMS", [])}
        for i, (local_name, val) in enumerate(local_params):
            key = f"{prefix}.{local_name}"
            b = bounds[i] if bounds is not None else None
            self.params[key] = {
                "value": float(val),
                "initial": float(val),
                "bounds": b,
                "fixed": bool(spec.get(local_name, {}).get("fixed", False)),
                "unit_index": idx,
                "local_name": local_name,
            }

        if idx == 1:
            # binary mixture: fraction of the first unit as (fixed) parameter
            f0 = float(self.unit_fractions[0])
            self.params[f"{self.unit_prefixes[0]}.{FRACTION}"] = {
                "value": f0,
                "initial": f0,
                "bounds": (0.0, 1.0),
                "fixed": True,
                "unit_index": 0,
                "local_name": FRACTION,
            }

    def fraction_key(self) -> Optional[str]:
        """Key of the mixing-fraction parameter, or ``None``.

        The parameter exists only for binary mixtures (exactly two units);
        it holds the fraction of the first unit.
        """
        for key, rec in self.params.items():
            if rec["local_name"] == FRACTION:
                return key
        return None

    def _record(self, key: str) -> ParamRecord:
        """Registry record of ``key`` with a helpful error for fraction keys."""
        try:
            return self.params[key]
        except KeyError:
            if key.endswith("." + FRACTION):
                raise KeyError(
                    f"No parameter '{key}': the mixing fraction is a parameter only for "
                    "exactly two units, named after the prefix of the first unit "
                    f"(available: {self.fraction_key()})."
                ) from None
            raise

    def _write_through(self, key: str, value: float) -> None:
        """Push a parameter value to its unit (or to the mixing fractions)."""
        rec = self.params[key]
        idx = int(rec["unit_index"])  # type: ignore[arg-type]
        local = str(rec["local_name"])
        if local == FRACTION:
            # binary mixture: the second unit contributes the remainder
            self.unit_fractions[0] = value
            self.unit_fractions[1] = 1.0 - value
        else:
            self.units[idx].set_param_values({local: value})

    def param_keys(self, free_only: bool = False) -> List[str]:
        """Return parameter keys in a stable order.

        Parameters
        ----------
        free_only : bool, optional
            If ``True``, return only parameters with ``fixed == False``.

        Returns
        -------
        list of str
            Fully-qualified parameter keys (e.g., ``"epm.mtt"``).
        """
        items = sorted(
            self.params.items(), key=lambda kv: (kv[1]["unit_index"], kv[1]["local_name"])  # type: ignore
        )
        return [k for k, rec in items if not (free_only and rec.get("fixed"))]

    def get_vector(self, which: str = "value", free_only: bool = False) -> List[float]:
        """Export parameter values as a flat vector in registry order.

        Parameters
        ----------
        which : {"value", "initial"}
            Whether to export current values or initial guesses.
        free_only : bool, optional
            If ``True``, export only free parameters.

        Returns
        -------
        list of float
            Parameter vector following :meth:`param_keys` order.
        """
        assert which in {"value", "initial"}
        keys = self.param_keys(free_only=free_only)
        return [float(self.params[k][which]) for k in keys]

    def set_vector(
        self, vec: Sequence[float], which: str = "value", free_only: bool = False
    ) -> None:
        """Write a vector into the registry (and units) in registry order.

        Parameters
        ----------
        vec : sequence of float
            Values to assign (length must match the number of addressed params).
        which : {"value", "initial"}
            Destination field to write (``"value"`` also writes through to units).
        free_only : bool, optional
            If ``True``, write into free parameters only.
        """
        assert which in {"value", "initial"}
        keys = self.param_keys(free_only=free_only)
        it = iter(map(float, vec))
        for k in keys:
            v = next(it)
            self.params[k][which] = v
            if which == "value":
                # push through to owning unit immediately
                self._write_through(k, v)

    def set_param(self, key: str, value: float) -> None:
        """Set a single parameter's **current** value and update the unit.

        This is a convenience wrapper around :meth:`set_vector` for one value.
        Setting the mixing fraction also sets the fraction of the second unit
        to ``1 - value``.
        """
        rec = self._record(key)
        rec["value"] = float(value)
        self._write_through(key, float(value))

    def set_initial(self, key: str, value: float) -> None:
        """Set a single parameter's **initial** value used for optimization seeding."""
        self._record(key)["initial"] = float(value)

    def set_bounds(self, key: str, bounds: Tuple[float, float]) -> None:
        """Set optimizer bounds for a single parameter.

        Parameters
        ----------
        key : str
            Fully-qualified parameter key (e.g., ``"epm.mtt"``).
        bounds : (float, float)
            Lower and upper search bounds for the optimizer. Bounds of the
            mixing fraction must lie within ``[0, 1]``.
        """
        rec = self._record(key)
        lo, hi = float(bounds[0]), float(bounds[1])
        if rec["local_name"] == FRACTION and not 0.0 <= lo <= hi <= 1.0:
            raise ValueError(
                f"Bounds of the mixing fraction '{key}' must satisfy 0 <= lower <= upper <= 1 "
                f"(got ({lo}, {hi}))."
            )
        rec["bounds"] = (lo, hi)

    def set_fixed(self, key: str, fixed: bool = True) -> None:
        """Mark a parameter as fixed (not optimized) or, with ``fixed=False``, as free."""
        self._record(key)["fixed"] = bool(fixed)

    def get_bounds(self, free_only: bool = False) -> List[Tuple[float, float]]:
        """Return bounds for parameters in registry order.

        Raises a ``ValueError`` if any addressed parameter has no bounds set.
        """
        keys = self.param_keys(free_only=free_only)
        out: List[Tuple[float, float]] = []
        for k in keys:
            b = self.params[k]["bounds"]
            if b is None:
                raise ValueError(f"Missing optimizer bounds for parameter: {k}")
            out.append(b)  # type: ignore[arg-type]
        return out

    @property
    def n_warmup(self) -> int:
        """Number of warmup steps prepended to the series."""
        return self._n_warmup

    def _steady_state_vector(self, n_tracers: int) -> np.ndarray:
        """Return steady-state input as a 1D vector matching ``n_tracers``.

        Parameters
        ----------
        n_tracers : int
            Number of tracer channels in the model input.

        Returns
        -------
        ndarray
            A vector of length ``n_tracers`` with steady-state input values.

        Raises
        ------
        ValueError
            If provided values cannot be broadcast to ``n_tracers``.
        """

        if self.steady_state_input is None:
            raise ValueError("steady_state_input is None")
        arr = np.asarray(self.steady_state_input, dtype=float)
        if arr.ndim == 0:
            return np.full(n_tracers, float(arr))
        if arr.shape == (n_tracers,):
            return arr.astype(float, copy=False)
        if arr.size == 1:
            return np.full(n_tracers, float(arr.reshape(-1)[0]))
        raise ValueError("steady_state_input must be scalar or length equal to number of tracers")

    def _warmup(self) -> None:
        """Prepend a steady-state warmup to input/target and set bookkeeping.

        Uses ``n_warmup_half_lives`` and the decay constant to determine the
        warmup length. If ``steady_state_input`` is not provided or length is
        non-positive, no warmup is applied. Stable tracers (decay constant 0)
        are ignored when the length is derived from half-lives; if all tracers
        are stable, ``n_warmup_steps`` must be given.
        """
        if self.steady_state_input is None:
            self._n_warmup = 0
        elif self.n_warmup_half_lives is not None and self.n_warmup_steps is None:
            lam = np.atleast_1d(np.asarray(self.lambda_, dtype=float))
            lam = lam[lam > 0.0]
            if lam.size == 0:
                raise ValueError(
                    "The warm-up length cannot be derived from half-lives because all "
                    "tracers are stable (decay constant 0); give n_warmup_steps instead."
                )
            t12 = 0.693 / lam
            self._n_warmup = int(np.max(t12)) * self.n_warmup_half_lives
        else:
            self._n_warmup = int(self.n_warmup_steps)

        if self.steady_state_input is None or self._n_warmup <= 0:
            # no warmup requested → ensure we don't slice anything off
            self._n_warmup = 0
            self._is_warm = True
            return

        # Ensure that warmup is not too long
        if self._n_warmup > _MAX_WARMUP_STEPS:
            warnings.warn(
                f"The warm-up period was limited to {_MAX_WARMUP_STEPS} time steps "
                f"(requested: {self._n_warmup}).",
                PyTracerLabWarning,
                stacklevel=3,
            )
            self._n_warmup = _MAX_WARMUP_STEPS
        # prepend steady-state warmup to input; support 1D or 2D inputs
        if self.input_series.ndim == 1:
            val = self._steady_state_vector(1)[0]
            warm = np.full(self._n_warmup, float(val))
        else:
            n_tr = int(self.input_series.shape[1])
            vals = self._steady_state_vector(n_tr)
            warm = np.repeat(vals[np.newaxis, :], self._n_warmup, axis=0)
        self.input_series = np.concatenate((warm, self.input_series))
        if self.target_series is not None:
            if self.target_series.ndim == 1:
                warm_nan = np.full(self._n_warmup, np.nan)
            else:
                n_tr_tg = int(self.target_series.shape[1])
                warm_nan = np.full((self._n_warmup, n_tr_tg), np.nan)
            self.target_series = np.concatenate((warm_nan, self.target_series))
        self._is_warm = True

    def _validate_series(self) -> None:
        """Check that ``target_series`` is given on the time grid of ``input_series``.

        Raises
        ------
        ValueError
            If the number of time steps (or, for 2D arrays, of tracers) of
            ``target_series`` and ``input_series`` differ.
        """
        if self.input_series is None or self.target_series is None:
            return
        shape_in = np.shape(self.input_series)
        shape_obs = np.shape(self.target_series)
        if shape_in[0] != shape_obs[0]:
            raise ValueError(
                f"target_series has {shape_obs[0]} time steps but input_series has "
                f"{shape_in[0]}. Observations must be given on the time grid of the "
                "input series, with NaN where nothing was observed. "
                "PyTracerLab.read_series(path, align_to=input_series) does this for "
                "observation files that list only the sampling dates."
            )
        if len(shape_in) == 2 and len(shape_obs) == 2 and shape_in[1] != shape_obs[1]:
            raise ValueError(
                f"target_series has {shape_obs[1]} tracer columns but input_series has "
                f"{shape_in[1]}."
            )

    def _check(self) -> None:
        """Ensure warmup is applied and mixture fractions are properly normalized.

        Raises
        ------
        ValueError
            If a unit fraction is not between 0 and 1 or the sum of unit
            fractions deviates too much from 1.0.
        """
        if not self._is_warm:
            self._warmup()
        if any(not -1e-12 <= f <= 1.0 + 1e-12 for f in self.unit_fractions):
            raise ValueError(
                "Unit fractions must be between 0 and 1 (got "
                + ", ".join(f"{f:.6g}" for f in self.unit_fractions)
                + ")."
            )
        s = sum(self.unit_fractions) if self.unit_fractions else 0.0
        if not (0.99 <= s <= 1.01):
            raise ValueError("Sum of unit fractions must be ~1.0.")

    def get_ttds(self, n_steps: Optional[int] = None) -> List[np.ndarray]:
        """Return travel time distributions for all units using current registry
        values.

        Returns
        -------
        Dict
            Dict of unit fractions and travel time distributions for all units.
            Each travel time distribution is a numpy array with one element per
            time step. Time steps and time step units correspond to general
            model settings. Fractions can be used to determine a global
            model-impulse response from the unit-specific responses.
        """
        # initialize travel time distribution dict for all units
        tt_dists = {"fractions": self.unit_fractions, "distributions": []}

        if n_steps is None:
            # Determine number of time steps from input dimensionality
            x = np.asarray(self.input_series, dtype=float)
            if x.ndim == 1:
                x = x.reshape(-1, 1)
            n, k = x.shape
            t = np.arange(0.0, n * self.dt, self.dt)
        else:
            t = np.arange(0.0, n_steps * self.dt, self.dt)

        for unit in self.units:
            # up to tracer-specific properties (decay), the impulse response
            # is equal for all tracers in a model and just depends on the
            # unit
            h = unit.get_impulse_response(t, self.dt, 0.0, False)
            tt_dists["distributions"].append(h)

        return tt_dists

    def simulate(self) -> np.ndarray:
        """Run the forward model using current registry values.

        Returns
        -------
        ndarray
            Simulated output aligned with ``target_series`` (warmup removed).
        """
        # Check before simulating
        self._check()

        # Determine number of tracers from input dimensionality
        x = np.asarray(self.input_series, dtype=float)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        n, k = x.shape
        t = np.arange(0.0, n * self.dt, self.dt)

        # Support scalar or vector lambda_
        if isinstance(self.lambda_, (list, tuple, np.ndarray)):
            lam_vec = np.asarray(self.lambda_, dtype=float)
            if lam_vec.ndim == 0:
                lam_vec = np.full(k, float(lam_vec))
            elif lam_vec.shape != (k,):
                # broadcast a single value or truncate/extend conservatively
                if lam_vec.size == 1:
                    lam_vec = np.full(k, float(lam_vec.ravel()[0]))
                else:
                    raise ValueError("lambda_ must be scalar or length equal to number of tracers")
        else:
            lam_vec = np.full(k, float(self.lambda_))

        # Handle production bool; make it a per-tracer-vector if a single bool
        if isinstance(self.production, bool):
            prod_vec = [self.production] * k
        else:
            prod_vec = self.production

        sim = np.zeros((n, k), dtype=float)
        for frac, unit in zip(self.unit_fractions, self.units):
            # per-tracer impulse responses and contributions
            for j in range(k):
                h = unit.get_impulse_response(t, self.dt, float(lam_vec[j]), prod_vec[j])

                # Normalization of the impulse response happens within the
                # model units. If we normalize here, we remove the effect
                # of radioactive decay.

                contrib = scipy.signal.fftconvolve(x[:, j], h)[:n]  # * self.dt
                sim[:, j] += float(frac) * contrib

        # Remove warmup
        sim = sim[self._n_warmup :, :]

        # Return 1D for single-tracer to preserve backward compatibility
        if sim.shape[1] == 1:
            return sim.ravel()
        else:
            return sim

    def write_report(
        self,
        filename: str,
        frequency: str,
        tracer: Optional[str] = None,
        sim: Optional[Union[str, Sequence[str]]] = None,
        title: str = "Model Report",
        include_initials: bool = True,
        include_bounds: bool = True,
        convert_mtt_to_years: bool = False,
    ) -> str:
        """
        Create a simple text report of the current model configuration and fit.

        Parameters
        ----------
        filename : str
            Path of the text file to write.
        frequency : str
            Simulation frequency (e.g., ``"1h"``). This is not checked
            internally and directly written to the report.
        tracer : str or sequence of str, optional
            Name(s) of the tracer(s) in the report. If not given, decay
            constants are shown for all tracers instead of tracer names.
        sim : ndarray, optional
            Simulated series corresponding to the *current* parameters. If not
            provided and `target_series` is present, the method will call
            :meth:`simulate` to compute one.
        title : str, optional
            Title shown at the top of the report.
        include_initials : bool, optional
            Whether to include initial values in the parameter table.
        include_bounds : bool, optional
            Whether to include optimizer bounds in the parameter table.
        convert_mtt_to_years : bool, optional
            Whether to convert mean travel times to years from months.

        Returns
        -------
        str
            The full report text that was written to `filename`.

        Notes
        -----
        - Parameters are grouped by their namespace prefix (e.g., ``"epm"`` in
          keys like ``"epm.mtt"``).
        - If `target_series` is available, the report includes the mean squared
          error (MSE) between the simulation and observations using overlapping,
          non-NaN entries.

        """
        lines: list[str] = []

        # Header
        lines.append(f"{title}")
        lines.append("=" * max(len(title), 20))
        lines.append("")

        # Model settings
        lines.append("Model settings")
        lines.append("--------------")
        lines.append(f"Time step (dt): {frequency}")
        if tracer is None:
            # If there is no tracer name given, we use the decay constants
            lines.append(
                "Decay constant (lambda): "
                + (
                    ", ".join(f"{float(v):.6g}" for v in np.atleast_1d(self.lambda_))
                    if isinstance(self.lambda_, (list, tuple, np.ndarray))
                    else f"{float(self.lambda_):.6g}"
                )
            )
        else:
            if isinstance(tracer, str):
                lines.append(f"Tracer: {tracer}")
            elif isinstance(tracer, (list, tuple, np.ndarray)):
                lines.append(f"Tracers: {', '.join(tracer)}")
            else:
                raise ValueError("Tracer must be a string or sequence of strings.")
        lines.append(f"Warmup steps: {self._n_warmup} (auto)")
        if self.steady_state_input is None:
            steady = "n/a"
        else:
            arr = np.asarray(self.steady_state_input, dtype=float)
            if arr.ndim == 0 or arr.size == 1:
                steady = f"{float(arr.reshape(-1)[0]):.6g}"
            else:
                steady = ", ".join(f"{float(v):.6g}" for v in arr.ravel())
        lines.append(f"Steady-state input: {steady}")
        lines.append(f"Units count: {len(self.units)}")
        lines.append("")

        # MSE and data if possible
        mse_text = "n/a"
        if self.target_series is not None:
            if sim is None:
                sim = self.simulate()
            y = self.target_series[self._n_warmup :]
            # coerce to 2D for uniform handling
            y2 = np.asarray(y, dtype=float)
            s2 = np.asarray(sim, dtype=float)
            if y2.ndim == 1:
                y2 = y2.reshape(-1, 1)
            if s2.ndim == 1:
                s2 = s2.reshape(-1, 1)
            if y2.shape[0] == s2.shape[0] and y2.shape[1] == s2.shape[1]:
                per_tr_mse: list[str] = []
                for j in range(y2.shape[1]):
                    mask = ~np.isnan(y2[:, j]) & ~np.isnan(s2[:, j])
                    if np.any(mask):
                        mse_j = float(np.mean((s2[mask, j] - y2[mask, j]) ** 2))
                        per_tr_mse.append(f"T{j+1}={mse_j:.6g}")
                if per_tr_mse:
                    mse_text = ", ".join(per_tr_mse)

        lines.append("Global fit")
        lines.append("----------")
        lines.append(f"MSE: {mse_text}")
        lines.append("")

        lines.append("Observed and Simulated Data")
        lines.append("---------------------------")
        lines.append("")

        # We make a separate table for each tracer. This is mainly because
        # for multiple tracers, the number of observations may be different

        # Make list of tracer names if not given
        if tracer is None:
            tracer_ = [str(i) for i in range(1, y2.shape[1] + 1)]
        else:
            tracer_ = tracer

        # Make list of time steps if not given
        if self.time_steps is None:
            timesteps = [i for i in range(y2.shape[0])]
        else:
            timesteps = self.time_steps

        for i, tracer in enumerate(list(tracer_)):
            # append column headers
            lines.append(f"Tracer {tracer}")
            lines.append("\t".join(["Time", "Obs.", "Sim."]))

            # Get mask for dates where observations are available
            mask = ~np.isnan(y2[:, i]) & ~np.isnan(s2[:, i])
            for t in range(len(timesteps)):
                # Only print if observation is available
                if mask[t]:
                    # Try to format timesteps as "YYYY-MM"
                    try:
                        timestamp = timesteps[t].strftime("%Y-%m")
                    except AttributeError:
                        timestamp = timesteps[t]
                    lines.append("\t".join([f"{timestamp}", f"{y2[t, i]:.3e}", f"{s2[t, i]:.3e}"]))
            lines.append("")

        # Parameter table grouped by unit prefix
        lines.append("Parameters by unit")
        lines.append("------------------")
        grouped: dict[str, list[str]] = {}
        for key in self.param_keys(free_only=False):
            prefix = key.split(".", 1)[0] if "." in key else "(root)"
            grouped.setdefault(prefix, []).append(key)

        # Determine stable group order based on the units' insertion order via recorded unit_index
        prefix_order: list[tuple[str, int]] = []
        for prefix, keys in grouped.items():
            if not keys:
                continue
            one_key = keys[0]
            try:
                uidx = int(self.params[one_key]["unit_index"])  # type: ignore[index]
            except Exception:
                uidx = 10**9
            prefix_order.append((prefix, uidx))
        prefix_order.sort(key=lambda t: t[1])

        # print warning regarding model units
        lines.append("-- --\nATTENTION: Travel Time Parameters are Always Given in [Years]!\n-- --")

        # pretty print per group with correct fraction association
        frac_key = self.fraction_key()
        for prefix, uidx in prefix_order:
            frac = self.unit_fractions[uidx] if 0 <= uidx < len(self.unit_fractions) else None
            frac_str = f"fraction={frac:.3f}" if frac is not None else ""
            if frac_key is not None and uidx == 1:
                # binary mixture: the second unit contributes the remainder
                frac_str += f" (= 1 - {frac_key})"
            lines.append(f"[{prefix}] {frac_str}")
            if (
                frac_key is not None
                and uidx == 1
                and self.param_uncert is not None
                and frac_key in self.param_uncert
            ):
                lo_f, med_f, hi_f = self.param_uncert[frac_key]
                lines.append(
                    f"  {'fraction':15s} 1%-Quantile={1.0 - hi_f:.6g}, "
                    f"50%-Quantile (Median)={1.0 - med_f:.6g}, 99%-Quantile={1.0 - lo_f:.6g}"
                )
            keys = sorted(grouped[prefix], key=lambda k: self.params[k]["local_name"])  # type: ignore[index]
            for k in keys:
                rec = self.params[k]
                val = float(rec["value"])
                # convert to yearly units
                # we always work in months as base units, so we always have to convert
                if "mtt" in k and convert_mtt_to_years:
                    val /= 12.0
                fixed = bool(rec.get("fixed", False))
                row = f"  {k:15s} value={val:.6g}"
                if include_initials:
                    ini = float(rec["initial"])
                    # convert to yearly units
                    if "mtt" in k and convert_mtt_to_years:
                        ini /= 12.0
                    row += f", initial={ini:.6g}"
                if include_bounds and rec.get("bounds") is not None:
                    lo, hi = rec["bounds"]  # type: ignore
                    # convert to yearly units
                    if "mtt" in k and convert_mtt_to_years:
                        lo /= 12.0
                        hi /= 12.0
                    row += f", bounds=({float(lo):.6g}, {float(hi):.6g})"
                row += f", fixed={fixed}"
                lines.append(row)
                # check if we have uncertainty estimates
                if self.param_uncert is not None and self.param_map is not None:
                    if k in self.param_map:
                        map = self.param_map[k]
                        # convert to yearly units
                        if "mtt" in k and convert_mtt_to_years:
                            map /= 12.0
                        row = f"  {k:15s} MAP (maximum a posteriori estimate)={map:.6g}"
                        lines.append(row)
                    if k in self.param_uncert:
                        unc = self.param_uncert[k]
                        # convert to yearly units
                        if "mtt" in k and convert_mtt_to_years:
                            lower = unc[0] / 12.0
                            median = unc[1] / 12.0
                            upper = unc[2] / 12.0
                        else:
                            lower = unc[0]
                            median = unc[1]
                            upper = unc[2]
                        row = f"  {k:15s} 1%-Quantile={lower:.6g}, "
                        row += f"50%-Quantile (Median)={median:.6g}, 99%-Quantile={upper:.6g}"
                        lines.append(row)
                        lines.append("")
            lines.append("")

        report_text = "\n".join(lines)
        with open(filename, "w", encoding="utf-8") as f:
            f.write(report_text)
        return report_text
