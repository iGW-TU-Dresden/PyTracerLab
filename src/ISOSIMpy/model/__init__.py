"""Public API for the :mod:`ISOSIMpy.model` package."""

# we intentionally omit the Solver class from __all__ to avoid trouble
# with autodoc (through Solver we otherwise hava a duplication of Model)
__all__ = ["Model", "Unit", "EPMUnit", "EMUnit", "PMUnit", "DMUnit"]


def __getattr__(name):
    """Lazy attribute-based re-exports for convenient imports.

    This keeps import time light while still allowing patterns like
    ``from ISOSIMpy.model import Model``.
    """
    # Lazy re-exports so runtime imports still work:
    if name == "Solver":
        from .solver import Solver

        return Solver
    if name == "Model":
        from .model import Model

        return Model
    if name == "Unit":
        from .units import Unit

        return Unit
    if name == "EPMUnit":
        from .units import EPMUnit

        return EPMUnit
    if name == "EMUnit":
        from .units import EMUnit

        return EMUnit
    if name == "PMUnit":
        from .units import PMUnit

        return PMUnit
    if name == "DMUnit":
        from .units import DMUnit

        return DMUnit
    raise AttributeError(name)


def __dir__():
    """Limit ``dir(ISOSIMpy.model)`` to the public API listed in ``__all__``.

    This avoids showing ``Solver`` at the package level in autodoc listings.
    """
    # Keep Solver out of dir() so autodoc doesn't list it at the package level
    return sorted(__all__)
