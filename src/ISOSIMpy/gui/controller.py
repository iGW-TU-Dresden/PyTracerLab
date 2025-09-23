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

    def _lambda_for_tracer(self, tracer_name: str) -> float:
        name = (tracer_name or "").strip()
        # Accept a few common spellings
        is_month = 12.0 if self.state.is_monthly else 1.0
        if name.lower() in {"tritium", "h-3", "h3", "3h"}:
            # Tritium
            half_life_years = 12.32
        elif name.lower() in {"carbon-14", "14-c", "c-14", "14c"}:
            # Carbon-14
            half_life_years = 5700.0
        elif name.lower() in {"krypton-85", "kr-85", "kr85", "85kr"}:
            # Krypton-85
            half_life_years = 10.73
        else:
            # Raise an error
            raise ValueError(f"Unknown tracer name: {name}")
        return 0.693 / (half_life_years * is_month)

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
            # Determine tracer set from state (dual-tracer aware)
            tracer_names = []
            t1 = getattr(self.state, "tracer1", None)
            t2 = getattr(self.state, "tracer2", None)
            if t1:
                tracer_names.append(t1)
            elif hasattr(self.state, "tracer") and getattr(self.state, "tracer"):
                # legacy single-tracer fallback
                tracer_names.append(getattr(self.state, "tracer"))
            if t2 and str(t2).lower() not in {"none", ""}:
                tracer_names.append(t2)

            lam = (
                [self._lambda_for_tracer(n) for n in tracer_names]
                if len(tracer_names) >= 1
                else [self._lambda_for_tracer("Tritium")]
            )
            if len(lam) == 1:
                lam = lam[0]

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

            x = self.state.input_series
            y = self.state.target_series
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

    def write_report(self, filename):
        """Write a plain-text model report to ``filename`` using current state."""
        if self.ml is None:
            return
        if self.state.is_monthly:
            frequency = "1 month"
        else:
            frequency = "1 year"
        self.ml.write_report(filename=filename, frequency=frequency)
