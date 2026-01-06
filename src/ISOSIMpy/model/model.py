"""Model container with a parameter registry."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import scipy.signal

from .units import Unit

# Each registry record stores numeric state + metadata used by the solver.
ParamRecord = Dict[str, object]


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
    target_series : ndarray, optional
        Observed output series of shape ``(N,)`` or ``(N, K)``; used only for
        calibration/loss and reporting.
    steady_state_input : float or sequence of float, optional
        If provided, a warmup of constant input is prepended. Supply a scalar
        for single-tracer runs or one value per tracer for multi-tracer runs.
    n_warmup_half_lives : int, optional
        Heuristic warmup scaling in half-lives (kept for compatibility).

    Notes
    -----
    - Units are added via :meth:`add_unit`. The method also registers unit
      parameters into the model's registry.
    - Bounds are **optimization bounds** only and can be provided at add time
      or later via :meth:`set_bounds`.

    """

    dt: float
    lambda_: Union[float, np.ndarray]
    input_series: np.ndarray
    target_series: Optional[np.ndarray] = None
    steady_state_input: Optional[Union[float, Sequence[float]]] = None
    n_warmup_half_lives: int = 2
    time_steps: Optional[Union[Sequence, np.ndarray]] = None

    units: List[Unit] = field(default_factory=list)
    unit_fractions: List[float] = field(default_factory=list)

    # Parameter registry: key -> record
    params: Dict[str, ParamRecord] = field(default_factory=dict, init=False)

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

        Parameters
        ----------
        unit : :class:`~ISOSIMpy.model.units.Unit`
            The unit instance to add.
        fraction : float
            Mixture fraction of this unit in the overall response. Fractions
            should sum to ~1 across all units.
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
            of unit parameters.
        """
        idx = len(self.units)
        self.units.append(unit)
        self.unit_fractions.append(float(fraction))

        prefix = prefix or f"u{idx}"
        local_params = list(unit.param_values().items())
        if bounds is not None and len(bounds) != len(local_params):
            raise ValueError("Length of bounds list must match number of unit parameters")

        for i, (local_name, val) in enumerate(local_params):
            key = f"{prefix}.{local_name}"
            b = bounds[i] if bounds is not None else None
            self.params[key] = {
                "value": float(val),
                "initial": float(val),
                "bounds": b,
                "fixed": False,
                "unit_index": idx,
                "local_name": local_name,
            }

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
                idx = int(self.params[k]["unit_index"])  # type: ignore
                local = str(self.params[k]["local_name"])  # type: ignore
                self.units[idx].set_param_values({local: v})

    def set_param(self, key: str, value: float) -> None:
        """Set a single parameter's **current** value and update the unit.

        This is a convenience wrapper around :meth:`set_vector` for one value.
        """
        self.params[key]["value"] = float(value)
        idx = int(self.params[key]["unit_index"])  # type: ignore
        local = str(self.params[key]["local_name"])  # type: ignore
        self.units[idx].set_param_values({local: float(value)})

    def set_initial(self, key: str, value: float) -> None:
        """Set a single parameter's **initial** value used for optimization seeding."""
        self.params[key]["initial"] = float(value)

    def set_bounds(self, key: str, bounds: Tuple[float, float]) -> None:
        """Set optimizer bounds for a single parameter.

        Parameters
        ----------
        key : str
            Fully-qualified parameter key (e.g., ``"epm.mtt"``).
        bounds : (float, float)
            Lower and upper search bounds for the optimizer.
        """
        lo, hi = bounds
        self.params[key]["bounds"] = (float(lo), float(hi))

    def set_fixed(self, key: str, fixed: bool = True) -> None:
        """Mark a parameter as fixed (not optimized)."""
        self.params[key]["fixed"] = bool(fixed)

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
        non-positive, no warmup is applied.
        """
        t12 = 0.693 / np.asarray(self.lambda_)
        t12 = np.asarray(t12, dtype=float)
        self._n_warmup = int(np.max(t12)) * self.n_warmup_half_lives
        if self.steady_state_input is None or self._n_warmup <= 0:
            # no warmup requested → ensure we don't slice anything off
            self._n_warmup = 0
            self._is_warm = True
            return
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

    def _check(self) -> None:
        """Ensure warmup is applied and mixture fractions are properly normalized.

        Raises
        ------
        ValueError
            If the sum of unit fractions deviates too much from 1.0.
        """
        if not self._is_warm:
            self._warmup()
        s = sum(self.unit_fractions) if self.unit_fractions else 0.0
        if not (0.99 <= s <= 1.01):
            raise ValueError("Sum of unit fractions must be ~1.0.")

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

        sim = np.zeros((n, k), dtype=float)
        for frac, unit in zip(self.unit_fractions, self.units):
            # per-tracer impulse responses and contributions
            for j in range(k):
                h = unit.get_impulse_response(t, self.dt, float(lam_vec[j]))

                # Normalization of the impulse response happens within the
                # model units. If we normalize here, we remove the effect
                # of radioactive decay.

                contrib = scipy.signal.fftconvolve(x[:, j], h)[:n] * self.dt
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

        # pretty print per group with correct fraction association
        for prefix, uidx in prefix_order:
            frac = self.unit_fractions[uidx] if 0 <= uidx < len(self.unit_fractions) else None
            frac_str = f"fraction={frac:.3f}" if frac is not None else ""
            lines.append(f"[{prefix}] {frac_str}")
            keys = sorted(grouped[prefix], key=lambda k: self.params[k]["local_name"])  # type: ignore[index]
            for k in keys:
                rec = self.params[k]
                val = float(rec["value"])
                fixed = bool(rec.get("fixed", False))
                row = f"  {k:15s} value={val:.6g}"
                if include_initials:
                    row += f", initial={float(rec['initial']):.6g}"
                if include_bounds and rec.get("bounds") is not None:
                    lo, hi = rec["bounds"]  # type: ignore
                    row += f", bounds=({float(lo):.6g}, {float(hi):.6g})"
                row += f", fixed={fixed}"
                lines.append(row)
            lines.append("")

        report_text = "\n".join(lines)
        with open(filename, "w", encoding="utf-8") as f:
            f.write(report_text)
        return report_text
