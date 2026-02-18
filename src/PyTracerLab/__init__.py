"""PyTracerLab package initializer."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("PyTracerLab")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"
