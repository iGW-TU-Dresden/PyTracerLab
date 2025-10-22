"""Unit class registry used by the GUI."""

from typing import Dict

from .units import DMUnit, EMUnit, ExEPMUnit, PMUnit

# # for the GUI we use the ExEPMUnit instead of the EPMUnit to be more in
# line with how practitioners want to define the model parameters (i.e.,
# the ratio of total volume to exponential-flow volume)
UNIT_REGISTRY = {
    "EM": EMUnit,
    "PM": PMUnit,
    "DM": DMUnit,
    "EPM": ExEPMUnit,
}

# Use placeholders for solver run functions
# Otherwise, we would need to import from ISOSIMpy.model.solver, which
# would create a circular dependency
SOLVER_REGISTRY: Dict[str, Dict[str, object]] = {
    "de": {"name": "Differential Evolution", "run": None},
    "lsq": {"name": "Least Squares", "run": None},
    "mcmc": {"name": "MCMC", "run": None},
}
