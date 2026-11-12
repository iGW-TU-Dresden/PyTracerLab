"""Public API for the :mod:`PyTracerLab.model` package.

All names are also available at package level (``import PyTracerLab as lab``),
which is the recommended way to use PyTracerLab.
"""

from importlib import import_module

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
    "AgeBinUnit": "PyTracerLab.model.units",
    "read_series": "PyTracerLab.model.io",
    "TracerSeries": "PyTracerLab.model.io",
    "DataFileError": "PyTracerLab.model.io",
    "decay_constant": "PyTracerLab.model.tracers",
    "TRACERS": "PyTracerLab.model.tracers",
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    """Lazy attribute-based re-exports for convenient imports.

    This keeps import time light while still allowing patterns like
    ``from PyTracerLab.model import Model``.
    """
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module 'PyTracerLab.model' has no attribute '{name}'")
    return getattr(import_module(module), name)


def __dir__():
    return sorted(set(globals()) | set(_EXPORTS))
