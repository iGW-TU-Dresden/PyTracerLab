"""Qt controller that builds the model and runs simulations/solvers."""

from PyQt5.QtCore import QObject, pyqtSignal

from ..model import model as mm
from ..model import solver as ms
from ..model.registry import UNIT_REGISTRY


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
    status = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, state):
        """Initialize the controller with a shared :class:`AppState`."""
        super().__init__()
        self.state = state
        self.ml = None

    def _lambda(self) -> float:
        if self.state.tracer == "Tritium":
            return 0.693 / (12.33 * (12.0 if self.state.is_monthly else 1.0))
        return 0.693 / (5700.0 * (12.0 if self.state.is_monthly else 1.0))

    def build_model(self):
        """Construct a :class:`~ISOSIMpy.model.model.Model` from current state.

        Creates per-instance units (including bounds and fixed flags) based on
        the detailed design in ``state.design_instances`` and the per-instance
        parameters in ``state.params``.
        """
        try:
            # We usually work in months (dt = 1.0); for yearly calculations
            # We therefore have to use dt = 12.0
            dt = 1.0 if self.state.is_monthly else 12.0
            lam = self._lambda()

            x = self.state.input_series
            y = self.state.target_series
            self.ml = mm.Model(
                dt,
                lam,
                input_series=x[1] if x else None,
                target_series=y[1] if y else None,
                steady_state_input=self.state.steady_state_input,
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

    def write_report(self, filename):
        """Write a plain-text model report to ``filename`` using current state."""
        if self.ml is None:
            return
        if self.state.is_monthly:
            frequency = "1 month"
        else:
            frequency = "1 year"
        self.ml.write_report(filename=filename, frequency=frequency)
