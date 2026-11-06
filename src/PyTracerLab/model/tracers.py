"""Built-in tracer properties.

PyTracerLab ships the *properties* of common environmental tracers (half-life
and whether the tracer is produced by decay). It does **not** ship tracer input
records (e.g. tritium in precipitation); these have to be provided by the user.

The same table is used by the GUI.
"""

from __future__ import annotations

import math
from typing import Dict

__all__ = ["TRACERS", "decay_constant"]

#: Tracer properties keyed by the name used in the GUI.
#:
#: ``half_life`` is given in years. ``production`` is ``True`` for tracers that
#: are produced by the decay of a parent tracer (e.g. tritiogenic helium); for
#: these, ``half_life`` is the half-life of the parent and the model input is the
#: parent's input series.
TRACERS: Dict[str, Dict[str, object]] = {
    "Carbon-14": {"half_life": 5700.0, "production": False},
    "Tritium": {"half_life": 12.32, "production": False},
    "Krypton-85": {"half_life": 10.73, "production": False},
    # artificially long half-life instead of zero decay (keeps the warm-up
    # computation of the model finite)
    "Stable tracer (no decay)": {"half_life": 1e12, "production": False},
    "Helium (tritiogenic)": {"half_life": 12.32, "production": True},
}

# Alternative spellings accepted by :func:`decay_constant` (case-insensitive).
_ALIASES = {
    "14c": "Carbon-14",
    "c-14": "Carbon-14",
    "c14": "Carbon-14",
    "radiocarbon": "Carbon-14",
    "3h": "Tritium",
    "h-3": "Tritium",
    "h3": "Tritium",
    "85kr": "Krypton-85",
    "kr-85": "Krypton-85",
    "kr85": "Krypton-85",
    "stable": "Stable tracer (no decay)",
    "3he": "Helium (tritiogenic)",
    "3he(trit)": "Helium (tritiogenic)",
    "3he_trit": "Helium (tritiogenic)",
    "tritiogenic helium": "Helium (tritiogenic)",
}

_TIME_UNITS_PER_YEAR = {"year": 1.0, "month": 12.0, "day": 365.25}


def _resolve(tracer: str) -> str:
    """Return the canonical tracer name for ``tracer`` (case-insensitive)."""
    key = str(tracer).strip().lower()
    for name in TRACERS:
        if name.lower() == key:
            return name
    if key in _ALIASES:
        return _ALIASES[key]
    known = ", ".join(TRACERS)
    raise KeyError(f"Unknown tracer '{tracer}'. Known tracers: {known}.")


def decay_constant(tracer: str, time_unit: str = "year") -> float:
    """Return the decay constant of a built-in tracer.

    Parameters
    ----------
    tracer : str
        Tracer name as listed in :data:`TRACERS` (e.g. ``"Tritium"``) or a
        common abbreviation (e.g. ``"3H"``, ``"14C"``, ``"85Kr"``);
        case-insensitive. For tritiogenic helium the decay constant of the
        parent (tritium) is returned; use it together with
        ``production=True`` in :class:`~PyTracerLab.model.model.Model`.
    time_unit : {"year", "month", "day"}, optional
        Time unit of the returned constant. It must match the time unit of the
        model time step and of the mean travel time parameters.

    Returns
    -------
    float
        Decay constant ``ln(2) / half-life`` in ``1 / time_unit``.
    """
    if time_unit not in _TIME_UNITS_PER_YEAR:
        raise ValueError(
            f"Unknown time_unit '{time_unit}'. Use one of: {', '.join(_TIME_UNITS_PER_YEAR)}."
        )
    half_life_years = float(TRACERS[_resolve(tracer)]["half_life"])
    half_life = half_life_years * _TIME_UNITS_PER_YEAR[time_unit]
    return math.log(2.0) / half_life
