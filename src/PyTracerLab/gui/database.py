"""Holds data of tracers supported in the PyTracerLab GUI."""

from abc import ABC
from dataclasses import dataclass


@dataclass
class Tracers(ABC):
    """Database of all tracers supported by the PyTracerLab GUI."""

    # columns:
    #   name : str
    #   half-life : float (half life in years)
    #   production : bool (if tracer is produced from decay [True] or decays [False])
    tracer_data = {
        "Carbon-14": [5700.0, False],
        "Tritium": [12.32, False],
        "Krypton-85": [10.73, False],
        "Stable tracer (no decay)": [False],  # artificially large decay time
        "Helium (tritiogenic)": [12.32, True],
    }
