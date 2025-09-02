"""Unit class registry used by the GUI."""

from .units import DMUnit, EMUnit, EPMUnit, PMUnit

UNIT_REGISTRY = {
    "PM": PMUnit,
    "EM": EMUnit,
    "EPM": EPMUnit,
    "DM": DMUnit,
}
