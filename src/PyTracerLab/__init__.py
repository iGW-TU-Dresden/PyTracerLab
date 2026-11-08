"""PyTracerLab: lumped parameter modelling of environmental tracers.

The main objects are available directly at package level::

    import PyTracerLab as lab

    inp = lab.read_series("tritium_input.csv")
    obs = lab.read_series("tritium_observations.csv", align_to=inp)
    m = lab.Model(dt=1.0, lambda_=lab.decay_constant("Tritium"),
                  input_series=inp.values, target_series=obs.values)

Imports are lazy: ``import PyTracerLab`` does not import the GUI (PyQt5) or
any heavy dependency until an object is first used.
"""

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("PyTracerLab")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"

# public name -> module that defines it
_EXPORTS = {
    "Model": "PyTracerLab.model.model",
    "Solver": "PyTracerLab.model.solver",
    "Unit": "PyTracerLab.model.units",
    "EPMUnit": "PyTracerLab.model.units",
    "ExEPMUnit": "PyTracerLab.model.units",
    "EMUnit": "PyTracerLab.model.units",
    "PMUnit": "PyTracerLab.model.units",
    "DMUnit": "PyTracerLab.model.units",
    "GMUnit": "PyTracerLab.model.units",
    "LMUnit": "PyTracerLab.model.units",
    "PEMUnit": "PyTracerLab.model.units",
    "read_series": "PyTracerLab.model.io",
    "TracerSeries": "PyTracerLab.model.io",
    "DataFileError": "PyTracerLab.model.io",
    "decay_constant": "PyTracerLab.model.tracers",
    "TRACERS": "PyTracerLab.model.tracers",
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    """Import public objects on first access (keeps ``import PyTracerLab`` light)."""
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module 'PyTracerLab' has no attribute '{name}'")
    obj = getattr(import_module(module), name)
    globals()[name] = obj  # cache for subsequent access
    return obj


def __dir__():
    return sorted(set(globals()) | set(_EXPORTS))
