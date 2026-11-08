"""Base classes for model units."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
from scipy import integrate, special


class Unit(ABC):
    """Abstract base class for a model unit.

    Concrete units represent hydrological transport schemata and must expose
    and accept their **local** parameter values via a mapping. Units are
    intentionally unaware of optimization bounds; those live in the Model's
    parameter registry.

    Notes
    -----
    - Implementations must keep local parameter names *stable* over time so that
      the Model's registry stays consistent.
    - Names should be short (e.g., ``"mtt"``, ``"eta"``).
    """

    @abstractmethod
    def param_values(self) -> Dict[str, float]:
        """Return current local parameter values.

        Returns
        -------
        Dict[str, float]
            Mapping from local parameter name to value.
        """
        raise NotImplementedError

    @abstractmethod
    def set_param_values(self, values: Dict[str, float]) -> None:
        """Set one or more local parameter values.

        Parameters
        ----------
        values : Dict[str, float]
            Mapping from local parameter name to new value. Keys not present
            are ignored.
        """
        raise NotImplementedError

    def intrinsic_bounds(self) -> Dict[str, Tuple[float, float]]:
        """Ranges that follow from the definition of the unit's parameters.

        The :class:`~PyTracerLab.model.model.Model` registers them as
        optimizer bounds if a unit is added without bounds, and rejects
        bounds outside these ranges. Most units have none (empty mapping);
        :class:`AgeBinUnit` uses it for its coordinates in ``[0, 1]``.
        """
        return {}

    def check_window(self, t_max: float) -> None:
        """Check that the travel time distribution fits into ``[0, t_max]``.

        ``t_max`` is the longest travel time the model resolves (input
        series plus warm-up). Units with unbounded travel time distributions
        are normalised within this window and need no check; units with a
        fixed, finite support (:class:`AgeBinUnit`) raise a ``ValueError``.
        """

    def report_lines(self, time_divisor: float = 1.0) -> List[str]:
        """Extra lines for :meth:`~PyTracerLab.model.model.Model.write_report`.

        Parameters
        ----------
        time_divisor : float, optional
            Time quantities are divided by this value (e.g. ``12`` to report
            years for a model in months).
        """
        return []

    def get_block(
        self, h: np.ndarray, tau: np.ndarray, dt: float, lambda_: float, prod: bool = False
    ) -> np.ndarray:
        """Get 1-dt block response."""
        # area = float(h.sum() * dt)
        area = integrate.trapezoid(h, dx=dt)
        if not np.isfinite(area) or area <= 0:
            raise ValueError(f"Impulse response has non-positive/invalid area: {area}")
        h /= area

        # decay / production
        if prod:
            h *= 1 - np.exp(-lambda_ * tau)
        else:
            h *= np.exp(-lambda_ * tau)

        step = np.cumsum(h) * dt
        step = integrate.cumulative_trapezoid(h, dx=dt, initial=0)
        block = np.append(step[0], np.subtract(step[1:], step[:-1]))

        return block

    def normalize_response(self, h: np.ndarray, dt: float) -> np.ndarray:
        """Normalize impulse response to unit area for conservation of mass."""
        area = float(h.sum() * dt)
        if not np.isfinite(area) or area <= 0:
            raise ValueError(f"Impulse response has non-positive/invalid area: {area}")
        h /= area

        return h

    @abstractmethod
    def get_impulse_response(self, tau: np.ndarray, dt: float, lambda_: float) -> np.ndarray:
        """Evaluate the unit's impulse response on a time grid.

        Parameters
        ----------
        tau : ndarray
            Non-negative time axis (same spacing as simulation time grid).
        dt : float
            Time step size of the discretization.
        lambda_ : float
            Decay constant (1 / time units of ``tau``).

        Returns
        -------
        ndarray
            Impulse response sampled at ``tau``.
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Units with exact time-step integration
#
# The EPM and the GM, LM, PEM and age-bin units do not sample g(τ) on the grid.
# Each time step k receives the exact integral of g(τ)·exp(-λτ) over
# ((k-1)·dt, k·dt], computed from a closed-form "decayed cumulative distribution"
#
#     G_λ(τ) = ∫_0^τ g(s) exp(-λ s) ds.
#
# This is the same discretisation as TracerLPM (Jurgens et al., 2012, Eqs. 9
# and 11). It stays accurate for TTDs with jumps (EPM, LM, PEM) or an integrable
# singularity at τ = 0 (GM with shape < 1), and the response changes smoothly
# with the parameters, which the gradient-based solvers need.
# ---------------------------------------------------------------------------


def _block_from_cdf(cdf, tau: np.ndarray, lambda_: float, prod: bool) -> np.ndarray:
    """Exact 1-dt block response from a decayed cumulative distribution.

    Parameters
    ----------
    cdf : callable
        ``cdf(t, lam)`` returns ``G_lam(t)``, the integral of ``g(s) exp(-lam s)``
        from 0 to ``t`` (``t`` array-like, ``lam >= 0``).
    tau : ndarray
        Regular time grid starting at 0 (``0, dt, 2 dt, ...``).
    lambda_ : float
        Decay constant (1 / time units of ``tau``).
    prod : bool
        If True, return the production response (integral of
        ``g(s) (1 - exp(-lam s))``), used for 3He from 3H decay.

    Returns
    -------
    ndarray
        ``h[0] = 0`` and ``h[k]`` = integral over ``(tau[k-1], tau[k]]``,
        normalised so that the TTD (without decay) has unit area within the
        time window, as for all other units.
    """
    tau = np.asarray(tau, dtype=float)
    if tau.size == 0:
        return np.zeros(0)
    area = float(np.asarray(cdf(tau[-1:], 0.0))[0])
    if not np.isfinite(area) or area <= 0:
        raise ValueError(f"Impulse response has non-positive/invalid area: {area}")
    cum = np.asarray(cdf(tau, lambda_), dtype=float)
    if prod:
        cum = np.asarray(cdf(tau, 0.0), dtype=float) - cum
    return np.diff(cum, prepend=0.0) / area


def _uniform_cdf(t, a: float, b: float, lam: float) -> np.ndarray:
    """Decayed cumulative distribution of the uniform TTD on ``[a, b]``.

    Returns the integral of ``exp(-lam s) / (b - a)`` over ``[a, min(t, b)]``
    (0 for ``t <= a``). Shared by the linear model and the age-bin model.
    """
    x = np.clip(np.asarray(t, dtype=float), a, b)
    if lam == 0.0:
        return (x - a) / (b - a)
    return np.exp(-lam * a) * -np.expm1(-lam * (x - a)) / (lam * (b - a))


def _shifted_exp_cdf(t, t0: float, k: float, lam: float) -> np.ndarray:
    """Decayed cumulative distribution of an exponential TTD shifted by ``t0``.

    TTD ``k exp(-k (τ - t0))`` for ``τ >= t0`` (rate ``k``); returns the
    integral of the TTD times ``exp(-lam τ)`` from 0 to ``t``. With
    ``t0 = mtt (1 - 1/η)`` and ``k = η / mtt`` this is the EPM.
    """
    x = np.maximum(np.asarray(t, dtype=float) - t0, 0.0)
    return k * np.exp(-lam * t0) / (k + lam) * -np.expm1(-(k + lam) * x)


def _epm_block(mtt: float, eta: float, tau: np.ndarray, lambda_: float, prod: bool) -> np.ndarray:
    """Exact block response of the EPM (``eta = 1``: exponential model)."""
    if not (np.isfinite(eta) and eta >= 1.0):
        raise ValueError(f"EPM: eta must be >= 1 (got {eta}).")
    t0, k = mtt * (1.0 - 1.0 / eta), eta / mtt
    return _block_from_cdf(lambda t, lam: _shifted_exp_cdf(t, t0, k, lam), tau, lambda_, prod)


def _check_positive(unit: str, **values: float) -> None:
    """Raise a ValueError unless all given parameter values are finite and > 0."""
    for name, val in values.items():
        if not (np.isfinite(val) and val > 0.0):
            raise ValueError(f"{unit}: parameter '{name}' must be positive (got {val}).")


@dataclass
class EPMUnit(Unit):
    """Exponential Piston-Flow Model (EPM) unit.

    Parameters
    ----------
    mtt : float
        Mean travel time.
    eta : float
        Ratio of total volume to the exponential reservoir (>= 1). ``eta=1``
        reduces to a pure exponential model; ``eta>1`` adds a piston component.
    PREFIX : str
        Prefix for local parameter names. Helper for GUI.
    PARAMS : List[Dict[str, Any]]
        List of (default) parameter definitions. Helper for GUI.

    Note: The parameter key for the mean travel time (``mtt``) is used in the
    GUI explicitly. The GUI assumes the parameter is given in years and
    internally converts it.
    """

    mtt: float
    eta: float
    PREFIX = "epm"
    # The parameter keys are used explicitly in the GUI! Changing them will lead
    # to the GUI not displaying parameter units properly.
    PARAMS = [
        {"key": "mtt", "label": "Mean Travel Time", "default": 10.0, "bounds": (0.0, 10000.0)},
        {"key": "eta", "label": "Eta", "default": 1.1, "bounds": (1.0, 2.0)},
    ]

    def param_values(self) -> Dict[str, float]:
        """Get parameter values.

        Returns
        -------
        Dict[str, float]
            Mapping from local parameter name to value.
        """
        return {"mtt": float(self.mtt), "eta": float(self.eta)}

    def set_param_values(self, values: Dict[str, float]) -> None:
        """Set one or more local parameter values.

        Parameters
        ----------
        values : Dict[str, float]
            Mapping from local parameter name to new value. Keys not present
            are ignored.
        """
        if "mtt" in values:
            self.mtt = float(values["mtt"])
        if "eta" in values:
            self.eta = float(values["eta"])

    def get_impulse_response(
        self, tau: np.ndarray, dt: float, lambda_: float, prod: bool = False
    ) -> np.ndarray:
        """EPM response of each time step (exact integral) with decay.

        The continuous-time EPM response (without decay) is
        ``h(τ) = (η/mtt) * exp(-η τ / mtt + η - 1)`` for
        ``τ >= mtt*(1 - 1/η)`` and ``0`` otherwise, i.e. an exponential
        distribution with rate ``η/mtt`` shifted by the piston-flow time
        ``mtt*(1 - 1/η)``. Each time step receives the exact integral of
        ``h(τ) exp(-λ τ)`` over the step.

        Parameters
        ----------
        tau : ndarray
            Non-negative time axis (same spacing as simulation time grid).
        dt : float
            Time step size of the discretization.
        lambda_ : float
            Decay constant (1 / time units of ``tau``).
        prod : bool, optional
            If True, calculate production response (used to simulate 3He
            production from 3H decay).

        Returns
        -------
        ndarray
            Response of each time step on the grid ``tau``.

        Raises
        ------
        ValueError
            If ``mtt`` is not positive or ``eta`` is smaller than 1.
        """
        _check_positive("EPM", mtt=self.mtt)
        return _epm_block(float(self.mtt), float(self.eta), tau, lambda_, prod)


@dataclass
class ExEPMUnit(Unit):
    """Explicit xponential Piston-Flow Model (EPM) unit.
    This model is essentially the same as the EPMUnit, but the EPM ratio
    (total volume / exponential volume or total area / area receiving
    recharge) is defined via two parameters instead of one aggregated
    parameter. Those two parameters are directly related and can never be
    estimated simultaneously.

    Parameters
    ----------
    mtt : float
        Mean travel time.
    exp_part: float
        Area receiving recharge or exponential volume of the system.
    piston_part: float
        Area not receiving recharge or piston-flow volume of the system.
    PREFIX : str
        Prefix for local parameter names. Helper for GUI.
    PARAMS : List[Dict[str, Any]]
        List of (default) parameter definitions. Helper for GUI.

    Note: The parameter key for the mean travel time (``mtt``) is used in the
    GUI explicitly. The GUI assumes the parameter is given in years and
    internally converts it.
    """

    mtt: float
    exp_part: float
    piston_part: float
    PREFIX = "epm"
    # The parameter keys are used explicitly in the GUI! Changing them will lead
    # to the GUI not displaying parameter units properly.
    PARAMS = [
        {"key": "mtt", "label": "Mean Travel Time", "default": 10.0, "bounds": (0.0, 10000.0)},
        {"key": "exp_part", "label": "Exponential Part", "default": 0.5, "bounds": (0.0, 100.0)},
        {"key": "piston_part", "label": "Piston Part", "default": 1.0, "bounds": (0.0, 100.0)},
    ]

    def param_values(self) -> Dict[str, float]:
        """Get parameter values.

        Returns
        -------
        Dict[str, float]
            Mapping from local parameter name to value.
        """
        return {
            "mtt": float(self.mtt),
            "exp_part": float(self.exp_part),
            "piston_part": float(self.piston_part),
        }

    def set_param_values(self, values: Dict[str, float]) -> None:
        """Set one or more local parameter values.

        Parameters
        ----------
        values : Dict[str, float]
            Mapping from local parameter name to new value. Keys not present
            are ignored.
        """
        if "mtt" in values:
            self.mtt = float(values["mtt"])
        if "exp_part" in values:
            self.exp_part = float(values["exp_part"])
        if "piston_part" in values:
            self.piston_part = float(values["piston_part"])

    def get_impulse_response(
        self, tau: np.ndarray, dt: float, lambda_: float, prod: bool = False
    ) -> np.ndarray:
        """ExEPM response of each time step (exact integral) with decay.

        Same response as :class:`EPMUnit` with
        ``η = piston_part / exp_part + 1``.

        Parameters
        ----------
        tau : ndarray
            Non-negative time axis (same spacing as simulation time grid).
        dt : float
            Time step size of the discretization.
        lambda_ : float
            Decay constant (1 / time units of ``tau``).
        prod : bool, optional
            If True, calculate production response (used to simulate 3He
            production from 3H decay).

        Returns
        -------
        ndarray
            Response of each time step on the grid ``tau``.

        Raises
        ------
        ValueError
            If ``mtt`` or ``exp_part`` is not positive or ``piston_part`` is
            negative.
        """
        _check_positive("EPM", mtt=self.mtt, exp_part=self.exp_part)
        if not (np.isfinite(self.piston_part) and self.piston_part >= 0.0):
            raise ValueError(f"EPM: parameter 'piston_part' must be >= 0 (got {self.piston_part}).")
        eta = float(self.piston_part) / float(self.exp_part) + 1.0
        return _epm_block(float(self.mtt), eta, tau, lambda_, prod)


@dataclass
class DMUnit(Unit):
    """Dispersion Model (DM) unit.

    Parameters
    ----------
    mtt : float
        Mean travel time.
    DP : float
        Dispersion parameter. Represents the inverse of the Peclet number.
        Also represents the ratio of the dispersion coefficient to the
        velocity and outlet / sampling position
    PREFIX : str
        Prefix for local parameter names. Helper for GUI.
    PARAMS : List[Dict[str, Any]]
        List of (default) parameter definitions. Helper for GUI.

    Note: The parameter key for the mean travel time (``mtt``) is used in the
    GUI explicitly. The GUI assumes the parameter is given in years and
    internally converts it.
    """

    mtt: float
    DP: float
    PREFIX = "dm"
    # The parameter keys are used explicitly in the GUI! Changing them will lead
    # to the GUI not displaying parameter units properly.
    PARAMS = [
        {"key": "mtt", "label": "Mean Travel Time", "default": 10.0, "bounds": (1.0, 10000.0)},
        {"key": "DP", "label": "Dispersion Param.", "default": 1.0, "bounds": (0.0001, 10.0)},
    ]

    def param_values(self) -> Dict[str, float]:
        """Get parameter values.

        Returns
        -------
        Dict[str, float]
            Mapping from local parameter name to value.
        """
        return {"mtt": float(self.mtt), "DP": float(self.DP)}

    def set_param_values(self, values: Dict[str, float]) -> None:
        """Set one or more local parameter values.

        Parameters
        ----------
        values : Dict[str, float]
            Mapping from local parameter name to new value. Keys not present
            are ignored.
        """
        if "mtt" in values:
            self.mtt = float(values["mtt"])
        if "DP" in values:
            self.DP = float(values["DP"])

    def get_impulse_response(
        self, tau: np.ndarray, dt: float, lambda_: float, prod: bool = False
    ) -> np.ndarray:
        """DM impulse response with decay.

        The continuous-time DM response (without decay) is
        ``h(τ) = (1/mtt) * 1 / (sqrt(K)) * exp((1 - τ / mtt)^2 / K)`` with
        ``K = 4 pi DP (τ / mtt)``. We also apply an exponential decay
        term ``exp(-λ τ)``.

        Parameters
        ----------
        tau : ndarray
            Non-negative time axis (same spacing as simulation time grid).
        dt : float
            Time step size of the discretization.
        lambda_ : float
            Decay constant (1 / time units of ``tau``).
        prod : bool, optional
            If True, calculate production response (used to simulate 3He
            production from 3H decay).

        Returns
        -------
        ndarray
            Impulse response evaluated at ``tau``.
        """
        # Check for edge cases
        if self.DP <= 0.0 or self.mtt <= 0.0:
            return np.zeros_like(tau)

        # Note: a non-zero response at h[0] corresponds to a response at the
        # same time step as the forcing. This is usually not what we want, so
        # we need to make sure the first time bin is 0.

        # The transfer function breaks down as τ tends towards 0
        # We therefore prepare a result array for h and fill it up, starting
        # with the non-zero values
        h = np.zeros_like(tau)

        # Pre-compute terms
        buffer = 1
        a = tau[buffer:] * np.sqrt(4 * np.pi * self.DP * tau[buffer:] / self.mtt)
        b = -((1 - tau[buffer:] / self.mtt) ** 2.0 / (4 * self.DP * tau[buffer:] / self.mtt))

        h[buffer:] = 1 / a * np.exp(b)

        # get response for constant-input block of length dt
        h = self.get_block(h, tau, dt, lambda_, prod)

        return h


@dataclass
class EMUnit(Unit):
    """Exponential Model (EM) unit.

    Parameters
    ----------
    mtt : float
        Mean travel time.
    PREFIX : str
        Prefix for local parameter names. Helper for GUI.
    PARAMS : List[Dict[str, Any]]
        List of (default) parameter definitions. Helper for GUI.

    Note: The parameter key for the mean travel time (``mtt``) is used in the
    GUI explicitly. The GUI assumes the parameter is given in years and
    internally converts it.
    """

    mtt: float
    PREFIX = "em"
    # The parameter keys are used explicitly in the GUI! Changing them will lead
    # to the GUI not displaying parameter units properly.
    PARAMS = [
        {"key": "mtt", "label": "Mean Travel Time", "default": 10.0, "bounds": (0.0, 10000.0)},
    ]

    def param_values(self) -> Dict[str, float]:
        """Get parameter values.

        Returns
        -------
        Dict[str, float]
            Mapping from local parameter name to value.
        """
        return {"mtt": float(self.mtt)}

    def set_param_values(self, values: Dict[str, float]) -> None:
        """Set one or more local parameter values.

        Parameters
        ----------
        values : Dict[str, float]
            Mapping from local parameter name to new value. Keys not present
            are ignored.
        """
        if "mtt" in values:
            self.mtt = float(values["mtt"])

    def get_impulse_response(
        self, tau: np.ndarray, dt: float, lambda_: float, prod: bool = False
    ) -> np.ndarray:
        """EM impulse response with decay.

        The continuous-time EPM response (without decay) is
        ``h(τ) = (1/mtt) * exp(-τ / mtt)``. We also apply an exponential
        decay term ``exp(-λ τ)``.

        Parameters
        ----------
        tau : ndarray
            Non-negative time axis (same spacing as simulation time grid).
        dt : float
            Time step size of the discretization.
        lambda_ : float
            Decay constant (1 / time units of ``tau``).
        prod : bool, optional
            If True, calculate production response (used to simulate 3He
            production from 3H decay).

        Returns
        -------
        ndarray
            Impulse response evaluated at ``tau``.
        """
        # check for edge cases
        if self.mtt <= 0.0:
            return np.zeros_like(tau)

        # Note: a non-zero response at h[0] corresponds to a response at the
        # same time step as the forcing. This is usually not what we want, so
        # we need to make sure the first time bin is 0.

        # Base EM shape
        h = np.zeros_like(tau)
        h = (1 / self.mtt) * np.exp(-tau / self.mtt)

        # get response for constant-input block of length dt
        h = self.get_block(h, tau, dt, lambda_, prod)

        return h


@dataclass
class PMUnit(Unit):
    """Piston-Flow Model (discrete delta at the mean travel time) with decay.

    Parameters
    ----------
    mtt : float
        Mean travel time where all mass is transported as a plug flow.
    PREFIX : str
        Prefix for local parameter names. Helper for GUI.
    PARAMS : List[Dict[str, Any]]
        List of (default) parameter definitions. Helper for GUI.

    Note: The parameter key for the mean travel time (``mtt``) is used in the
    GUI explicitly. The GUI assumes the parameter is given in years and
    internally converts it.
    """

    mtt: float
    PREFIX = "pm"
    # The parameter keys are used explicitly in the GUI! Changing them will lead
    # to the GUI not displaying parameter units properly.
    PARAMS = [
        {"key": "mtt", "label": "Mean Travel Time", "default": 10.0, "bounds": (0.0, 10000.0)},
    ]

    def param_values(self) -> Dict[str, float]:
        """Get parameter values.

        Returns
        -------
        Dict[str, float]
            Mapping from local parameter name to value.
        """
        return {"mtt": float(self.mtt)}

    def set_param_values(self, values: Dict[str, float]) -> None:
        """Set local parameter value.

        Parameters
        ----------
        values : Dict[str, float]
            Mapping from local parameter name to new value. Keys not present
            are ignored.
        """
        if "mtt" in values:
            self.mtt = float(values["mtt"])

    def get_impulse_response(
        self, tau: np.ndarray, dt: float, lambda_: float, prod: bool = False
    ) -> np.ndarray:
        """Discrete delta response on the grid with exponential decay.

        The delta is represented by setting the bin at ``round(mtt/dt)`` to
        ``1`` (times the decay factor), which preserves unit mass.

        Parameters
        ----------
        tau : ndarray
            Non-negative time axis (same spacing as simulation time grid).
        dt : float
            Time step size of the discretization.
        lambda_ : float
            Decay constant (1 / time units of ``tau``).
        prod : bool, optional
            If True, calculate production response (used to simulate 3He
            production from 3H decay).

        Returns
        -------
        ndarray
            Impulse response evaluated at ``tau``.
        """
        # Check for edge cases
        if self.mtt <= 0.0:
            return np.zeros_like(tau)

        # Note: a non-zero response at h[0] corresponds to a response at the
        # same time step as the forcing. This is usually not what we want, so
        # we need to make sure the first time bin is 0.

        # We don't need to shift the travel time grid here because we (almost surely)
        # avoid having non-zero response at h[0] anyways for the PM.

        h = np.zeros_like(tau)
        idx = max(1, int(round(self.mtt / dt)))

        # All mass in a single time step: like the block responses of the
        # other units, the weights sum to 1 (before decay). (Before v0.3.0,
        # the weight was 1/dt, which scaled the output for dt != 1.)
        if 0 <= idx < len(tau):
            h[idx] = 1.0
            # radioactive/first-order decay/production
            if prod:
                h[idx] *= 1 - np.exp(-lambda_ * self.mtt)
            else:
                h[idx] *= np.exp(-lambda_ * self.mtt)
        return h


@dataclass
class GMUnit(Unit):
    """Gamma Model (GM) unit.

    Travel time distribution (Kirchner et al., 2000; Kirchner, 2016, Eq. 2;
    cf. the general gamma model of Amin and Campana, 1996)::

        g(τ) = τ^(α-1) exp(-α τ / mtt) / ((mtt / α)^α Γ(α)),   τ > 0

    with mean ``mtt`` and standard deviation ``mtt / sqrt(α)``. ``α = 1``
    gives the exponential model; ``α < 1`` gives many short and many long
    travel times (g is infinite at τ = 0, but integrable); ``α > 1`` gives a
    peaked distribution that approaches piston flow for large ``α``.

    Parameters
    ----------
    mtt : float
        Mean travel time (> 0).
    alpha : float
        Shape factor α (> 0, dimensionless).
    PREFIX : str
        Prefix for local parameter names. Helper for GUI.
    PARAMS : List[Dict[str, Any]]
        List of (default) parameter definitions. Helper for GUI.

    Notes
    -----
    The response of each time step is the exact integral of the TTD with
    decay, ``G_λ(τ) = (ρ / (ρ + λ))^α P(α, (ρ + λ) τ)`` with ``ρ = α / mtt``
    and ``P`` the regularised lower incomplete gamma function.

    The parameter key for the mean travel time (``mtt``) is used in the GUI
    explicitly. The GUI assumes the parameter is given in years and
    internally converts it.
    """

    mtt: float
    alpha: float
    PREFIX = "gm"
    # The parameter keys are used explicitly in the GUI! Changing them will lead
    # to the GUI not displaying parameter units properly.
    PARAMS = [
        {"key": "mtt", "label": "Mean Travel Time", "default": 10.0, "bounds": (0.0, 10000.0)},
        {"key": "alpha", "label": "Shape Factor", "default": 1.0, "bounds": (0.1, 10.0)},
    ]

    def param_values(self) -> Dict[str, float]:
        """Get parameter values.

        Returns
        -------
        Dict[str, float]
            Mapping from local parameter name to value.
        """
        return {"mtt": float(self.mtt), "alpha": float(self.alpha)}

    def set_param_values(self, values: Dict[str, float]) -> None:
        """Set one or more local parameter values.

        Parameters
        ----------
        values : Dict[str, float]
            Mapping from local parameter name to new value. Keys not present
            are ignored.
        """
        if "mtt" in values:
            self.mtt = float(values["mtt"])
        if "alpha" in values:
            self.alpha = float(values["alpha"])

    def _cdf(self, t, lam: float) -> np.ndarray:
        """Decayed cumulative distribution ``G_lam(t)`` of the gamma TTD."""
        alpha = float(self.alpha)
        rate = alpha / float(self.mtt)
        scale = np.exp(-alpha * np.log1p(lam / rate))  # (rate / (rate + lam))^alpha
        return scale * special.gammainc(alpha, (rate + lam) * np.asarray(t, dtype=float))

    def get_impulse_response(
        self, tau: np.ndarray, dt: float, lambda_: float, prod: bool = False
    ) -> np.ndarray:
        """GM response of each time step (exact integral) with decay.

        Parameters
        ----------
        tau : ndarray
            Non-negative time axis (same spacing as simulation time grid).
        dt : float
            Time step size of the discretization.
        lambda_ : float
            Decay constant (1 / time units of ``tau``).
        prod : bool, optional
            If True, calculate production response (used to simulate 3He
            production from 3H decay).

        Returns
        -------
        ndarray
            Response of each time step on the grid ``tau``.

        Raises
        ------
        ValueError
            If ``mtt`` or ``alpha`` is not positive.
        """
        _check_positive("GM", mtt=self.mtt, alpha=self.alpha)
        return _block_from_cdf(self._cdf, tau, lambda_, prod)


@dataclass
class LMUnit(Unit):
    """Linear Model (LM) unit.

    Travel time distribution (Maloszewski and Zuber, 1996, Eq. 14; Eriksson,
    1958; Maloszewski and Zuber, 1982)::

        g(τ) = 1 / (2 mtt)   for 0 <= τ <= 2 mtt,   0 otherwise,

    i.e. travel times are uniformly distributed between 0 and twice the mean
    travel time ("all the flow lines have the same velocity but linearly
    increasing flow time"; no mixing between flow lines, the sample is mixed
    in the spring, stream or well).

    Parameters
    ----------
    mtt : float
        Mean travel time (> 0).
    PREFIX : str
        Prefix for local parameter names. Helper for GUI.
    PARAMS : List[Dict[str, Any]]
        List of (default) parameter definitions. Helper for GUI.

    Notes
    -----
    The response of each time step is the exact integral of the TTD with
    decay, so it changes smoothly with ``mtt`` also when ``2 mtt`` falls
    between two grid points.

    The parameter key for the mean travel time (``mtt``) is used in the GUI
    explicitly. The GUI assumes the parameter is given in years and
    internally converts it.
    """

    mtt: float
    PREFIX = "lm"
    # The parameter keys are used explicitly in the GUI! Changing them will lead
    # to the GUI not displaying parameter units properly.
    PARAMS = [
        {"key": "mtt", "label": "Mean Travel Time", "default": 10.0, "bounds": (0.0, 10000.0)},
    ]

    def param_values(self) -> Dict[str, float]:
        """Get parameter values.

        Returns
        -------
        Dict[str, float]
            Mapping from local parameter name to value.
        """
        return {"mtt": float(self.mtt)}

    def set_param_values(self, values: Dict[str, float]) -> None:
        """Set local parameter value.

        Parameters
        ----------
        values : Dict[str, float]
            Mapping from local parameter name to new value. Keys not present
            are ignored.
        """
        if "mtt" in values:
            self.mtt = float(values["mtt"])

    def _cdf(self, t, lam: float) -> np.ndarray:
        """Decayed cumulative distribution ``G_lam(t)`` of the uniform TTD."""
        return _uniform_cdf(t, 0.0, 2.0 * float(self.mtt), lam)

    def get_impulse_response(
        self, tau: np.ndarray, dt: float, lambda_: float, prod: bool = False
    ) -> np.ndarray:
        """LM response of each time step (exact integral) with decay.

        Parameters
        ----------
        tau : ndarray
            Non-negative time axis (same spacing as simulation time grid).
        dt : float
            Time step size of the discretization.
        lambda_ : float
            Decay constant (1 / time units of ``tau``).
        prod : bool, optional
            If True, calculate production response (used to simulate 3He
            production from 3H decay).

        Returns
        -------
        ndarray
            Response of each time step on the grid ``tau``.

        Raises
        ------
        ValueError
            If ``mtt`` is not positive.
        """
        _check_positive("LM", mtt=self.mtt)
        return _block_from_cdf(self._cdf, tau, lambda_, prod)


@dataclass
class PEMUnit(Unit):
    """Partial Exponential Model (PEM) unit.

    A well (or spring) in a homogeneous unconfined aquifer of constant
    thickness with uniform recharge, which is open only over part of the
    saturated thickness (Jurgens et al., 2012, Appendix A, Eq. A6; Jurgens et
    al., 2016). The age of groundwater increases logarithmically with depth,
    as in the exponential model, and the well samples only the ages between
    the top and the bottom of the screen::

        g(τ) = exp(-τ / τ_aq) / (τ_aq (1/n1 - 1/n2))   for t1 <= τ <= t2,

    with ``n1 = H / (H - z1)``, ``n2 = H / (H - z2)``, ``t1 = τ_aq ln n1``,
    ``t2 = τ_aq ln n2``, ``H`` the saturated thickness, ``z1`` and ``z2`` the
    depths of the top and bottom of the screen below the water table, and
    ``τ_aq`` the mean travel time of the whole aquifer.

    The geometry is given as three thicknesses in any (common) length unit;
    only their ratios matter. Like in TracerLPM, ``mtt`` is the mean travel
    time of the **sampled** water; ``τ_aq`` follows from it (Jurgens et al.,
    2012, Eq. A4; see :attr:`mtt_aquifer`).

    Special cases: ``above = below = 0`` gives the exponential model; with
    ``below = 0`` (screen reaching the base of the aquifer) the PEM is the
    model implemented in TracerLPM with "PEM ratio" = ``above / screen`` and
    is identical to the EPM with ``η = 1 + ln(1 + above / screen)``
    (Jurgens et al., 2012, Eqs. 10 and 12).

    Parameters
    ----------
    mtt : float
        Mean travel time of the sampled water (> 0).
    above : float
        Unsampled thickness above the screen, i.e. depth of the top of the
        screen below the water table (>= 0).
    screen : float
        Length of the screen, i.e. sampled thickness (> 0).
    below : float
        Unsampled thickness below the screen, i.e. distance from the bottom
        of the screen to the base of the aquifer (>= 0).
    PREFIX : str
        Prefix for local parameter names. Helper for GUI.
    PARAMS : List[Dict[str, Any]]
        List of (default) parameter definitions. Helper for GUI.

    Notes
    -----
    The geometry parameters are registered as **fixed** when the unit is
    added to a :class:`~PyTracerLab.model.model.Model`, because they are
    normally known from the well construction; free them with
    ``Model.set_fixed(key, False)``.

    The response of each time step is the exact integral of the TTD with
    decay.

    The parameter key for the mean travel time (``mtt``) is used in the GUI
    explicitly. The GUI assumes the parameter is given in years and
    internally converts it.
    """

    mtt: float
    above: float
    screen: float
    below: float
    PREFIX = "pem"
    # The parameter keys are used explicitly in the GUI! Changing them will lead
    # to the GUI not displaying parameter units properly.
    PARAMS = [
        {"key": "mtt", "label": "Mean Travel Time", "default": 10.0, "bounds": (0.0, 10000.0)},
        {
            "key": "above",
            "label": "Unsampled Thickness Above Screen",
            "default": 10.0,
            "bounds": (0.0, 1000.0),
            "fixed": True,
        },
        {
            "key": "screen",
            "label": "Screen Length",
            "default": 20.0,
            "bounds": (0.01, 1000.0),
            "fixed": True,
        },
        {
            "key": "below",
            "label": "Unsampled Thickness Below Screen",
            "default": 20.0,
            "bounds": (0.0, 1000.0),
            "fixed": True,
        },
    ]

    def param_values(self) -> Dict[str, float]:
        """Get parameter values.

        Returns
        -------
        Dict[str, float]
            Mapping from local parameter name to value.
        """
        return {
            "mtt": float(self.mtt),
            "above": float(self.above),
            "screen": float(self.screen),
            "below": float(self.below),
        }

    def set_param_values(self, values: Dict[str, float]) -> None:
        """Set one or more local parameter values.

        Parameters
        ----------
        values : Dict[str, float]
            Mapping from local parameter name to new value. Keys not present
            are ignored.
        """
        for key in ("mtt", "above", "screen", "below"):
            if key in values:
                setattr(self, key, float(values[key]))

    def _geometry(self):
        """Return ``(tau_aq, t1, t2, c)``; ``c = 1 / (1/n1 - 1/n2)``.

        Raises
        ------
        ValueError
            If a parameter is outside its valid range.
        """
        _check_positive("PEM", mtt=self.mtt, screen=self.screen)
        above, screen, below = float(self.above), float(self.screen), float(self.below)
        for name, val in (("above", above), ("below", below)):
            if not (np.isfinite(val) and val >= 0.0):
                raise ValueError(f"PEM: parameter '{name}' must be >= 0 (got {val}).")
        thick = above + screen + below
        q1 = (screen + below) / thick  # 1/n1: share of the thickness below the screen top
        q2 = below / thick  # 1/n2: share of the thickness below the screen bottom
        c = thick / screen  # 1 / (1/n1 - 1/n2)
        # mean travel time of the sampled water in units of tau_aq (Eq. A4)
        m = c * (q1 * (1.0 - np.log(q1)) - q2 + special.xlogy(q2, q2))
        tau_aq = float(self.mtt) / m
        t1 = -tau_aq * np.log(q1)
        t2 = -tau_aq * np.log(q2) if q2 > 0.0 else np.inf
        return tau_aq, t1, t2, c

    @property
    def mtt_aquifer(self) -> float:
        """Mean travel time of the whole aquifer (``τ_aq``) for the current parameters."""
        return float(self._geometry()[0])

    def _cdf(self, t, lam: float) -> np.ndarray:
        """Decayed cumulative distribution ``G_lam(t)`` of the PEM TTD."""
        tau_aq, t1, t2, c = self._geometry()
        r = 1.0 / tau_aq + lam
        x = np.clip(np.asarray(t, dtype=float), t1, t2)
        return c / (tau_aq * r) * np.exp(-r * t1) * -np.expm1(-r * (x - t1))

    def get_impulse_response(
        self, tau: np.ndarray, dt: float, lambda_: float, prod: bool = False
    ) -> np.ndarray:
        """PEM response of each time step (exact integral) with decay.

        Parameters
        ----------
        tau : ndarray
            Non-negative time axis (same spacing as simulation time grid).
        dt : float
            Time step size of the discretization.
        lambda_ : float
            Decay constant (1 / time units of ``tau``).
        prod : bool, optional
            If True, calculate production response (used to simulate 3He
            production from 3H decay).

        Returns
        -------
        ndarray
            Response of each time step on the grid ``tau``.

        Raises
        ------
        ValueError
            If ``mtt`` or ``screen`` is not positive, or ``above`` or
            ``below`` is negative.
        """
        return _block_from_cdf(self._cdf, tau, lambda_, prod)


class AgeBinUnit(Unit):
    """Age-bin model: a discrete, shape-free travel time distribution.

    The travel time distribution is a histogram with ``K`` bins between fixed
    edges ``a_0 < a_1 < ... < a_K`` and a uniform distribution within each
    bin (age histogram or discrete age distribution model; Visser et al.,
    2013, Eq. 3; McCallum et al., 2014)::

        g(τ) = w_k / (a_k - a_(k-1))   for a_(k-1) < τ <= a_k,

    with bin weights ``w_k >= 0`` that sum to 1. No shape is prescribed; the
    weights are calibrated. The mean travel time ``Σ w_k (a_(k-1) + a_k) / 2``
    is derived from the weights (:attr:`mtt`).

    **Parameters.** The weights are not registered directly, because they
    must sum to 1. Instead, the unit has ``K - 1`` coordinates
    ``u1, ..., u(K-1)`` in ``[0, 1]`` that are mapped one-to-one to the
    weights (stick-breaking)::

        b_i = 1 - (1 - u_i)^(1 / (K - i)),
        w_1 = b_1,
        w_k = b_k (1 - b_1) ... (1 - b_(k-1)),
        w_K = (1 - b_1) ... (1 - b_(K-1)).

    Every point of the unit box is a valid histogram, so all solvers work
    with the plain bounds ``(0, 1)``, which are registered automatically.
    A uniform distribution of the coordinates in the box corresponds to the
    uniform (flat Dirichlet) distribution of the weights: with the default
    uniform prior of :meth:`~PyTracerLab.model.solver.Solver.mcmc_sample` and
    :meth:`~PyTracerLab.model.solver.Solver.dream_sample`, every histogram is
    equally probable a priori. With two bins, ``u1`` is the weight of the
    first bin. Use :attr:`weights`, :attr:`cumulative` and :attr:`mtt` for
    the current state and :meth:`weights_from_samples` for posterior samples.

    Parameters
    ----------
    edges : sequence of float
        Bin edges ``a_0, ..., a_K`` (``K >= 1``), strictly increasing,
        ``a_0 >= 0``, in the time unit of the model (same as ``mtt`` of the
        other units). The last edge must lie within the simulation period
        (input series plus warm-up); for old, tracer-free water use a wide
        last bin and a sufficiently long warm-up, or add a second unit.
    weights : sequence of float, optional
        Initial bin weights (``K`` values ``>= 0`` that sum to 1). Default:
        equal weights.

    Notes
    -----
    The response of each time step is the exact integral of the histogram
    with decay (sum of uniform blocks, as in the linear model).

    Tracer data usually constrain only a few properties of the travel time
    distribution, such as the mean and the young fraction (McCallum et al.,
    2014). Use few bins (e.g. 3 to 6) and a sampler
    (:meth:`~PyTracerLab.model.solver.Solver.dream_sample`) to see which
    weights the data determine. No smoothness regularisation is applied.

    Examples
    --------
    ::

        bins = lab.AgeBinUnit(edges=[0, 10, 20, 40, 60, 1000])
        m = lab.Model(dt=1.0, lambda_=lab.decay_constant("Tritium"),
                      input_series=inp.values, target_series=obs.values,
                      steady_state_input=5.0, n_warmup_steps=1000)
        m.add_unit(bins, fraction=1.0, prefix="bins")   # bounds (0, 1) set
        res = lab.Solver(m).dream_sample(n_samples=2000, sigma=obs.uncertainty)
        w = bins.weights_from_samples(res, "bins")      # (n, K) weights
        mtt = w @ bins.centres                          # mean travel times
    """

    PREFIX = "bins"
    # The number of parameters depends on the number of bins, so there is no
    # static parameter list (the GUI does not offer this unit).
    PARAMS: List[Dict[str, Any]] = []

    def __init__(self, edges: Sequence[float], weights: Sequence[float] | None = None) -> None:
        a = np.array(edges, dtype=float)
        if a.ndim != 1 or a.size < 2:
            raise ValueError("Age bins: give at least two bin edges (one bin).")
        if not np.all(np.isfinite(a)):
            raise ValueError("Age bins: all bin edges must be finite.")
        if a[0] < 0.0:
            raise ValueError(f"Age bins: the first edge must be >= 0 (got {a[0]:g}).")
        if np.any(np.diff(a) <= 0.0):
            raise ValueError("Age bins: the bin edges must be strictly increasing.")
        a.setflags(write=False)
        self._edges = a
        k = a.size - 1
        self._u = self._coords(np.full(k, 1.0 / k) if weights is None else weights)

    def __repr__(self) -> str:
        edges = ", ".join(f"{v:g}" for v in self._edges)
        return f"AgeBinUnit(edges=[{edges}], weights={np.round(self.weights, 6).tolist()})"

    # ------------------------------------------------------------- geometry
    @property
    def edges(self) -> np.ndarray:
        """Bin edges ``a_0, ..., a_K`` (read-only array)."""
        return self._edges

    @property
    def n_bins(self) -> int:
        """Number of bins ``K``."""
        return int(self._edges.size - 1)

    @property
    def centres(self) -> np.ndarray:
        """Bin centres; ``weights @ centres`` is the mean travel time."""
        return 0.5 * (self._edges[:-1] + self._edges[1:])

    # ---------------------------------------------------- weights <-> coords
    def _coords(self, weights: Sequence[float]) -> np.ndarray:
        """Coordinates ``u`` (array of length ``K - 1``) of valid bin weights."""
        k = self.n_bins
        w = np.asarray(weights, dtype=float)
        if w.shape != (k,):
            raise ValueError(f"Age bins: give {k} weights, one per bin (got shape {w.shape}).")
        if not np.all(np.isfinite(w)) or np.any(w < 0.0):
            raise ValueError("Age bins: the weights must be finite and >= 0.")
        total = float(w.sum())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Age bins: the weights must sum to 1 (got {total:.6g}).")
        w = w / total
        rest = np.cumsum(w[::-1])[::-1]  # weight of bin i and of all older bins
        with np.errstate(divide="ignore", invalid="ignore"):
            b = np.where(rest[:-1] > 0.0, w[:-1] / rest[:-1], 0.0)
        m = k - np.arange(1, k)  # K - i for i = 1 .. K-1
        return np.clip(1.0 - (1.0 - np.clip(b, 0.0, 1.0)) ** m, 0.0, 1.0)

    def coords_from_weights(self, weights: Sequence[float]) -> Dict[str, float]:
        """Coordinates (local parameter values) for given bin weights.

        Use it to set initial values of a unit that is already part of a
        model, e.g.::

            for name, v in bins.coords_from_weights([0.4, 0.3, 0.2, 0.1]).items():
                m.set_initial(f"bins.{name}", v)
        """
        return {f"u{i + 1}": float(v) for i, v in enumerate(self._coords(weights))}

    def weights_from(self, coords: Sequence[float] | np.ndarray) -> np.ndarray:
        """Bin weights for coordinates ``u`` (vectorised).

        Parameters
        ----------
        coords : array_like, shape ``(..., K - 1)``
            Coordinates in ``[0, 1]``.

        Returns
        -------
        ndarray, shape ``(..., K)``
            Bin weights (``>= 0``, sum 1 along the last axis).
        """
        k = self.n_bins
        u = np.asarray(coords, dtype=float)
        if u.ndim == 0 or u.shape[-1] != k - 1:
            raise ValueError(f"Age bins: expected {k - 1} coordinates (got shape {u.shape}).")
        bad = ~np.isfinite(u) | (u < 0.0) | (u > 1.0)
        if np.any(bad):
            pos = tuple(np.argwhere(bad)[0])
            raise ValueError(
                f"Age bins: coordinate 'u{pos[-1] + 1}' must be between 0 and 1 (got {u[pos]:g})."
            )
        if k == 1:
            return np.ones(u.shape[:-1] + (1,))
        m = k - np.arange(1, k)
        keep = (1.0 - u) ** (1.0 / m)  # 1 - b_i
        rest = np.cumprod(keep, axis=-1)
        w = np.empty(u.shape[:-1] + (k,))
        w[..., 0] = 1.0 - keep[..., 0]
        w[..., 1:-1] = (1.0 - keep[..., 1:]) * rest[..., :-1]
        w[..., -1] = rest[..., -1]
        return w

    def weights_from_samples(self, result: Mapping[str, Any], prefix: str) -> np.ndarray:
        """Bin weights of each sample of an MCMC or DREAM result.

        Parameters
        ----------
        result : dict
            Return value of ``Solver.mcmc_sample`` or ``Solver.dream_sample``
            (uses ``"param_names"`` and ``"samples"``).
        prefix : str
            Prefix under which the unit was added to the model.

        Returns
        -------
        ndarray, shape ``(n_samples, K)``
            Weights; coordinates that were fixed keep their current value.
            ``np.cumsum(w, axis=1)`` gives the fractions younger than each
            upper edge, ``w @ unit.centres`` the mean travel times.
        """
        names = list(result["param_names"])
        samples = np.asarray(result["samples"], dtype=float).reshape(-1, len(names))
        coords = np.tile(self._u, (samples.shape[0], 1))
        found = False
        for i in range(self.n_bins - 1):
            key = f"{prefix}.u{i + 1}"
            if key in names:
                coords[:, i] = samples[:, names.index(key)]
                found = True
        if not found and self.n_bins > 1:
            raise KeyError(f"The result contains no coordinates of the age bins '{prefix}'.")
        return self.weights_from(coords)

    # ------------------------------------------------------- current state
    @property
    def weights(self) -> np.ndarray:
        """Current bin weights (length ``K``)."""
        return self.weights_from(self._u)

    @property
    def cumulative(self) -> np.ndarray:
        """Fraction of water younger than each edge (length ``K + 1``, from 0 to 1)."""
        c = np.concatenate(([0.0], np.cumsum(self.weights)))
        c[-1] = 1.0
        return c

    @property
    def mtt(self) -> float:
        """Mean travel time of the histogram (derived from the weights)."""
        return float(self.weights @ self.centres)

    # ------------------------------------------------------ Unit interface
    def param_values(self) -> Dict[str, float]:
        """Get the coordinates ``u1, ..., u(K-1)``.

        Returns
        -------
        Dict[str, float]
            Mapping from local parameter name to value.
        """
        return {f"u{i + 1}": float(v) for i, v in enumerate(self._u)}

    def set_param_values(self, values: Dict[str, float]) -> None:
        """Set one or more coordinates.

        Parameters
        ----------
        values : Dict[str, float]
            Mapping from local parameter name (``"u1"``, ...) to new value.
            Keys not present are ignored. Values are checked when the
            response is computed.
        """
        for i in range(self.n_bins - 1):
            key = f"u{i + 1}"
            if key in values:
                self._u[i] = float(values[key])

    def intrinsic_bounds(self) -> Dict[str, Tuple[float, float]]:
        """All coordinates lie in ``[0, 1]``."""
        return {f"u{i + 1}": (0.0, 1.0) for i in range(self.n_bins - 1)}

    def check_window(self, t_max: float) -> None:
        """Raise a ``ValueError`` if the last edge lies beyond ``t_max``."""
        last = float(self._edges[-1])
        if last > float(t_max) * (1.0 + 1e-12) + 1e-12:
            raise ValueError(
                f"Age bins: the last bin edge ({last:g}) lies beyond the simulation period "
                f"({float(t_max):g}, input series plus warm-up). Lower the last edge or "
                "extend the warm-up (steady_state_input with n_warmup_steps)."
            )

    def report_lines(self, time_divisor: float = 1.0) -> List[str]:
        """Edges, weights, cumulative fractions and mean travel time."""
        try:
            w, c, mtt = self.weights, self.cumulative, self.mtt
        except ValueError as e:  # invalid coordinates
            return [f"  ({e})"]
        edges = ", ".join(f"{v / time_divisor:.6g}" for v in self._edges)
        return [
            f"  {'bin edges':15s} {edges}",
            f"  {'bin weights':15s} " + ", ".join(f"{v:.4f}" for v in w),
            f"  {'younger than':15s} " + ", ".join(f"{v:.4f}" for v in c) + " (at each edge)",
            f"  {'mean travel time':15s} {mtt / time_divisor:.6g} (derived)",
        ]

    def _cdf(self, t, lam: float) -> np.ndarray:
        """Decayed cumulative distribution ``G_lam(t)`` of the histogram."""
        x = np.asarray(t, dtype=float)
        out = np.zeros(x.shape)
        for wk, a, b in zip(self.weights, self._edges[:-1], self._edges[1:]):
            if wk > 0.0:
                out += wk * _uniform_cdf(x, float(a), float(b), lam)
        return out

    def get_impulse_response(
        self, tau: np.ndarray, dt: float, lambda_: float, prod: bool = False
    ) -> np.ndarray:
        """Age-bin response of each time step (exact integral) with decay.

        Parameters
        ----------
        tau : ndarray
            Non-negative time axis (same spacing as simulation time grid).
        dt : float
            Time step size of the discretization.
        lambda_ : float
            Decay constant (1 / time units of ``tau``).
        prod : bool, optional
            If True, calculate production response (used to simulate 3He
            production from 3H decay).

        Returns
        -------
        ndarray
            Response of each time step on the grid ``tau``.

        Raises
        ------
        ValueError
            If a coordinate is outside ``[0, 1]`` or the last edge lies
            beyond ``tau[-1]``.
        """
        tau = np.asarray(tau, dtype=float)
        if tau.size:
            self.check_window(tau[-1])
        return _block_from_cdf(self._cdf, tau, lambda_, prod)
