"""Unit class registry used by the GUI."""

from .units import DMUnit, EMUnit, EPMUnit, PMUnit

UNIT_REGISTRY = {
    "EPM": EPMUnit,
    "EM": EMUnit,
    "PM": PMUnit,
    "DM": DMUnit,
}
