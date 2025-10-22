"""Base classes for model units."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict

import numpy as np


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
    """

    mtt: float
    eta: float
    PREFIX = "epm"
    PARAMS = [
        {"key": "mtt", "label": "Mean Transit Time", "default": 120.0, "bounds": (0.0, 10000.0)},
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

    def get_impulse_response(self, tau: np.ndarray, dt: float, lambda_: float) -> np.ndarray:
        """EPM impulse response with decay.

        The continuous-time EPM response (without decay) is
        ``h(τ) = (η/mtt) * exp(-η τ / mtt + η - 1)`` for
        ``τ >= mtt*(1 - 1/η)`` and ``0`` otherwise. We also apply
        an exponential decay term ``exp(-λ τ)``.

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
            Impulse response evaluated at ``tau``.
        """
        # check for edge cases
        if self.eta <= 1.0 or self.mtt <= 0.0:
            return np.zeros_like(tau)

        # base EPM shape
        h_prelim = (self.eta / self.mtt) * np.exp(-self.eta * tau / self.mtt + self.eta - 1.0)
        cutoff = self.mtt * (1.0 - 1.0 / self.eta)
        h = np.where(tau < cutoff, 0.0, h_prelim)
        # radioactive/first-order decay applied to transit time
        h *= np.exp(-lambda_ * tau)
        return h


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
    """

    mtt: float
    exp_part: float
    piston_part: float
    PREFIX = "epm"
    PARAMS = [
        {"key": "mtt", "label": "Mean Transit Time", "default": 120.0, "bounds": (0.0, 10000.0)},
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

    def get_impulse_response(self, tau: np.ndarray, dt: float, lambda_: float) -> np.ndarray:
        """ExEPM impulse response with decay.

        The continuous-time EPM response (without decay) is
        ``h(τ) = (η/mtt) * exp(-η τ / mtt + η - 1)`` for
        ``τ >= mtt*(1 - 1/η)`` and ``0`` otherwise. We also apply
        an exponential decay term ``exp(-λ τ)``.

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
            Impulse response evaluated at ``tau``.
        """
        # calculate eta
        eta = (self.piston_part / self.exp_part) + 1

        # check for edge cases
        if eta <= 1.0 or self.mtt <= 0.0:
            return np.zeros_like(tau)

        # base EPM shape
        h_prelim = (eta / self.mtt) * np.exp(-eta * tau / self.mtt + eta - 1.0)
        cutoff = self.mtt * (1.0 - 1.0 / eta)
        h = np.where(tau < cutoff, 0.0, h_prelim)
        # radioactive/first-order decay applied to transit time
        h *= np.exp(-lambda_ * tau)
        return h


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
    """

    mtt: float
    DP: float
    PREFIX = "dm"
    PARAMS = [
        {"key": "mtt", "label": "Mean Transit Time", "default": 120.0, "bounds": (1.0, 10000.0)},
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

    def get_impulse_response(self, tau: np.ndarray, dt: float, lambda_: float) -> np.ndarray:
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

        Returns
        -------
        ndarray
            Impulse response evaluated at ``tau``.
        """
        # Check for edge cases
        if self.DP <= 0.0 or self.mtt <= 0.0:
            return np.zeros_like(tau)

        # The transfer function breaks down at τ=0
        # We therefore prepare a result array for h and fill it up, starting
        # with the non-zero values
        h = np.zeros_like(tau)
        K = np.zeros_like(tau)

        # Compute K-term
        K[1:] = 4 * np.pi * self.DP * tau[1:] / self.mtt

        # Base DM shape
        h[1:] = (
            (1 / self.mtt)
            * (1 / np.sqrt(K[1:]))
            * np.exp(((1 - (tau[1:] / self.mtt)) ** 2) / K[1:])
        )
        # Radioactive/first-order decay applied to transit time
        h *= np.exp(-lambda_ * tau)
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
    """

    mtt: float
    PREFIX = "em"
    PARAMS = [
        {"key": "mtt", "label": "Mean Transit Time", "default": 120.0, "bounds": (0.0, 10000.0)},
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

    def get_impulse_response(self, tau: np.ndarray, dt: float, lambda_: float) -> np.ndarray:
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

        Returns
        -------
        ndarray
            Impulse response evaluated at ``tau``.
        """
        # check for edge cases
        if self.mtt <= 0.0:
            return np.zeros_like(tau)

        # base EM shape
        h = (1 / self.mtt) * np.exp(-tau / self.mtt)
        # radioactive/first-order decay applied to transit time
        h *= np.exp(-lambda_ * tau)
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
    """

    mtt: float
    PREFIX = "pm"
    PARAMS = [
        {"key": "mtt", "label": "Mean Transit Time", "default": 120.0, "bounds": (0.0, 10000.0)},
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

    def get_impulse_response(self, tau: np.ndarray, dt: float, lambda_: float) -> np.ndarray:
        """Discrete delta response on the grid with exponential decay.

        The delta is represented by setting the bin at ``round(mtt/dt)`` to
        ``1/dt`` to preserve unit mass in the discrete sum.

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
            Impulse response evaluated at ``tau``.
        """
        # check for edge cases
        if self.mtt <= 0.0:
            return np.zeros_like(tau)

        h = np.zeros_like(tau)
        idx = int(round(self.mtt / dt))
        if 0 <= idx < len(tau):
            h[idx] = 1.0 / dt
            h[idx] *= np.exp(-lambda_ * self.mtt)
        return h
