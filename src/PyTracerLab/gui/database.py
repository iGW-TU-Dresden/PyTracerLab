"""Holds data of tracers supported in the PyTracerLab GUI."""

from abc import ABC
from dataclasses import dataclass


@dataclass
class Tracers(ABC):
    """Database of all tracers supported by the PyTracerLab GUI."""

    tracer_data = {
        "Carbon-14": 5700.0,  # years
        "Tritium": 12.32,  # years
        "Krypton-85": 10.73,  # years
        "Stable tracer (no decay)": 1e10,  # artificially large decay time
    }
