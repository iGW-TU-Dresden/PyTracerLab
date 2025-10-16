"""Qt controller that builds the model and runs simulations/solvers."""

import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal

from ..model import model as mm
from ..model import solver as ms
from ..model.registry import UNIT_REGISTRY
from .database import Tracers


class Controller(QObject):
    """Bridge between GUI state and the computational model.

    Orchestrates model construction from GUI state, runs simulations and
    calibrations, and emits signals with results or errors.

    Signals
    -------
    simulated : object
        Emitted after a forward simulation; carries a NumPy array.
    calibrated : object
        Emitted after calibration; carries a payload ``dict`` with simulation
        and optional envelopes/metadata.
    status : str
        Short user-facing status messages.
    error : str
        Error messages suitable for display.
    """

    # Use generic object payloads to support arrays or dicts from different solvers
    simulated = pyqtSignal(object)
    calibrated = pyqtSignal(object)
    tracer_tracer_ready = pyqtSignal(object)
    status = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, state):
        """Initialize the controller with a shared :class:`AppState`."""
        super().__init__()
        self.state = state
        self.ml = None

    def build_model(self):
        """Construct a :class:`~ISOSIMpy.model.model.Model` from current state.

        Creates per-instance units (including bounds and fixed flags) based on
        the detailed design in ``state.design_instances`` and the per-instance
        parameters in ``state.params``.
        """
        try:
            # We usually work in months (dt = 1.0); for yearly calculations
            # we therefore have to use dt = 12.0
            dt = 1.0 if self.state.is_monthly else 12.0
            # Determine tracer set from state (dual-tracer aware)
            tracer_names = []
            t1 = getattr(self.state, "tracer1", None)
            t2 = getattr(self.state, "tracer2", None)
            if t1:
                tracer_names.append(t1)
            elif hasattr(self.state, "tracer") and getattr(self.state, "tracer"):
                # single-tracer fallback
                tracer_names.append(getattr(self.state, "tracer"))
            if t2 and str(t2).lower() not in {"none", ""}:
                tracer_names.append(t2)

            # Determine the decay constants for the selected tracers
            lam = []
            for name, half_life in Tracers.tracer_data.items():
                if name in tracer_names:
                    lam.append(0.693 / (half_life * 12.0))  # convert to monthly
            if len(lam) == 1:
                lam = lam[0]

            # Determine steady state input for the cases of one or two tracers
            n_tracers = len(tracer_names) if tracer_names else 1
            steady_state_input = getattr(self.state, "steady_state_input", None)
            if steady_state_input is None:
                ss_val = None
            elif n_tracers == 1:
                if isinstance(steady_state_input, (list, tuple)):
                    ss_val = float(steady_state_input[0]) if steady_state_input else 0.0
                else:
                    ss_val = float(steady_state_input)
            else:
                if isinstance(steady_state_input, (list, tuple)):
                    ss_seq = [float(v) for v in steady_state_input]
                else:
                    ss_seq = [float(steady_state_input)] * n_tracers
                if len(ss_seq) != n_tracers:
                    if len(ss_seq) == 1:
                        ss_seq = ss_seq * n_tracers
                    else:
                        raise ValueError("steady_state_input must provide one value per tracer")
                ss_val = ss_seq

            steady_enabled = bool(getattr(self.state, "steady_state_enabled", False))
            if not steady_enabled:
                ss_val = None

            # Set up the model
            x = self.state.input_series
            y = self.state.target_series

            # We have to treat the concentration values according to their time units
            # Because we generally work in montly resolution, we need to convert
            # yearly values to monthly values
            # However, we only have to transform the input series, as those values
            # actually represent fluxes (mass per unit time); the observed
            # concentrations are actually concentrations and not fluxes.
            if not self.state.is_monthly:
                x_ = x[1] / 12.0
            else:
                x_ = x[1]
            x = (x[0], x_)

            self.ml = mm.Model(
                dt,
                lam,
                input_series=x[1] if x else None,
                target_series=y[1] if y else None,
                steady_state_input=ss_val,
                n_warmup_half_lives=self.state.n_warmup_half_lives,
            )

            # Build per-instance units based on the detailed design
            instances = getattr(self.state, "design_instances", [])
            for inst in instances:
                unit_name: str = inst["name"]
                prefix: str = inst["prefix"]
                frac: float = float(inst.get("fraction", 0.0))

                cls = UNIT_REGISTRY[unit_name]
                spec = getattr(cls, "PARAMS", [])

                # Parameter values with safe defaults
                kwargs = {}
                for p in spec:
                    key = p["key"]
                    default_val = p.get("default")
                    rec = self.state.params.get(prefix, {}).get(key)
                    kwargs[key] = rec["val"] if rec is not None else default_val

                unit = cls(**kwargs)

                # Bounds with safe defaults
                bounds = []
                for p in spec:
                    key = p["key"]
                    default_lb, default_ub = p.get("bounds", (None, None))
                    rec = self.state.params.get(prefix, {}).get(key)
                    lb = rec["lb"] if rec is not None else default_lb
                    ub = rec["ub"] if rec is not None else default_ub
                    bounds.append((lb, ub))

                self.ml.add_unit(
                    unit=unit,
                    fraction=frac,
                    prefix=prefix,
                    bounds=bounds,
                )

                # Fixed flags
                for p in spec:
                    key = p["key"]
                    rec = self.state.params.get(prefix, {}).get(key)
                    if rec is not None and rec.get("fixed", False):
                        self.ml.set_fixed(f"{prefix}.{key}", True)

        except Exception as e:
            self.error.emit(str(e))
            self.ml = None

    def simulate(self):
        """Run a forward simulation and emit the result via ``simulated``."""
        try:
            self.build_model()
            if self.ml is None:
                return
            sim = self.ml.simulate()
            self.state.last_simulation = sim
            self.simulated.emit(sim)
            self.status.emit("Simulation finished.")
        except Exception as e:
            self.error.emit(str(e))

    def calibrate(self):
        """Run the selected solver and emit a standardized payload.

        The solver is chosen from ``state.solver_key`` and configured using
        ``state.solver_params``.
        """
        try:
            self.build_model()
            if self.ml is None:
                return
            # Run selected solver via registry and emit standardized payload
            key = getattr(self.state, "solver_key", "de")
            params = getattr(self.state, "solver_params", {}).get(key, {})
            payload = ms.run_solver(self.ml, key, params)
            self.state.last_simulation = payload
            self.calibrated.emit(payload)
            self.status.emit("Calibration finished.")
        except Exception as e:
            self.error.emit(str(e))

    def run_tracer_tracer(self, start: float, stop: float, count: int, param_key: str) -> None:
        """Sweep mean travel time values and emit tracer-tracer results."""
        try:
            if count <= 1:
                raise ValueError("Please choose at least two mean travel time points.")
            if not np.isfinite(start) or not np.isfinite(stop):
                raise ValueError("Start and stop values must be finite numbers.")
            if float(stop) <= float(start):
                raise ValueError("Stop value must be greater than start value.")

            if not getattr(self.state, "tracer2", None):
                raise ValueError("Tracer-tracer analysis requires two tracers.")

            target = self.state.target_series
            if target is None or target[1] is None:
                raise ValueError("No observation series available.")
            obs_times = np.asarray(target[0])
            obs_vals = np.asarray(target[1], dtype=float)
            if obs_vals.ndim == 1:
                obs_vals = obs_vals.reshape(-1, 1)
            if obs_vals.shape[1] < 2:
                raise ValueError("Observation series must contain two tracer columns.")

            self.build_model()
            if self.ml is None:
                raise RuntimeError(
                    "Model is not available. \
                        Configure the model before running a tracer-tracer sweep."
                )
            if param_key not in self.ml.params:
                raise ValueError(f"Unknown parameter: {param_key}")

            base_value = float(self.ml.params[param_key]["value"])
            grid = np.linspace(float(start), float(stop), int(count))
            results = None

            try:
                for idx, val in enumerate(grid):
                    self.ml.set_param(param_key, float(val))
                    sim = self.ml.simulate()
                    sim_arr = np.asarray(sim, dtype=float)
                    if sim_arr.ndim == 1:
                        sim_arr = sim_arr.reshape(-1, 1)
                    if results is None:
                        n_steps, n_tr = sim_arr.shape
                        results = np.zeros((n_tr, grid.size, n_steps), dtype=float)
                    elif sim_arr.shape[1] != results.shape[0]:
                        raise ValueError("Simulation returned an unexpected number of tracers.")
                    elif sim_arr.shape[0] != results.shape[2]:
                        raise ValueError(
                            "Simulation length changed across runs; cannot stack results."
                        )
                    results[:, idx, :] = sim_arr.T
            finally:
                self.ml.set_param(param_key, base_value)

            if results is None:
                raise ValueError("Failed to compute tracer-tracer results.")

            if obs_vals.shape[0] != results.shape[2]:
                raise ValueError("Observation length does not match simulation output.")

            obs_mask = ~np.isnan(obs_vals[:, 0]) & ~np.isnan(obs_vals[:, 1])
            obs_indices = np.nonzero(obs_mask)[0].astype(int)
            if obs_indices.size == 0:
                raise ValueError("No observation dates with values for both tracers.")

            payload = {
                "results": results,
                "mtt_values": grid,
                "param_key": param_key,
                "obs_indices": obs_indices,
                "timestamps": obs_times,
                "observations": obs_vals,
            }

            self.state.tt_results = results
            self.state.tt_mtt_values = grid
            self.state.tt_obs_indices = obs_indices
            self.state.tt_param_key = param_key

            self.tracer_tracer_ready.emit(payload)
            self.status.emit("Tracer-tracer sweep finished.")
        except Exception as exc:
            self.error.emit(str(exc))

    def write_report(self, filename):
        """Write a plain-text model report to ``filename`` using current state."""
        if self.ml is None:
            return
        if self.state.is_monthly:
            frequency = "1 month"
        else:
            frequency = "1 year"
        self.ml.write_report(filename=filename, frequency=frequency)
