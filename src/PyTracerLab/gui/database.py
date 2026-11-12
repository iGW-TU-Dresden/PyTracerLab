"""Holds data of tracers supported in the PyTracerLab GUI."""

from abc import ABC
from dataclasses import dataclass

from ..model.tracers import TRACERS


@dataclass
class Tracers(ABC):
    """Database of all tracers supported by the PyTracerLab GUI.

    The values are taken from :data:`PyTracerLab.model.tracers.TRACERS`, so the
    GUI and the Python package use the same tracer properties.
    """

    # columns:
    #   name : str
    #   half-life : float (half life in years)
    #   production : bool (if tracer is produced from decay [True] or decays [False])
    tracer_data = {
        name: [float(props["half_life"]), bool(props["production"])]
        for name, props in TRACERS.items()
    }
