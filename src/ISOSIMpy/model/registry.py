"""Unit class registry used by the GUI."""

from typing import Dict

from .units import DMUnit, EMUnit, EPMUnit, PMUnit

UNIT_REGISTRY = {
    "EPM": EPMUnit,
    "EM": EMUnit,
    "PM": PMUnit,
    "DM": DMUnit,
}

# Use placeholders for solver run functions
# Otherwise, we would need to import from ISOSIMpy.model.solver, which
# would create a circular dependency
SOLVER_REGISTRY: Dict[str, Dict[str, object]] = {
    "de": {"name": "Differential Evolution", "run": None},
    "lsq": {"name": "Least Squares", "run": None},
    "mcmc": {"name": "MCMC", "run": None},
}
