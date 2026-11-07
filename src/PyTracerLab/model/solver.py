"""Optimization wrapper for :class:`~PyTracerLab.model.model.Model`."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit, differential_evolution

from .model import Model, PyTracerLabWarning
from .registry import SOLVER_REGISTRY

# Objective value / residual used for failed simulations (finite, so that the
# optimizers keep working)
_BAD_OBJECTIVE = 1e10


def _broadcast(value: Any, n: int, k: int, name: str, per_obs: bool) -> np.ndarray:
    """Broadcast a scalar, per-tracer or per-observation value to ``(n, k)``."""
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0 or arr.size == 1:
        return np.full((n, k), float(arr.ravel()[0]))
    if arr.ndim == 1 and arr.size == k:
        return np.tile(arr, (n, 1))
    if per_obs and arr.ndim == 1 and k == 1 and arr.size == n:
        return arr.reshape(n, 1).copy()
    if per_obs and arr.ndim == 2 and arr.shape in ((n, k), (1, k)):
        return np.broadcast_to(arr, (n, k)).copy()
    expected = f"a scalar or one value per tracer ({k},)"
    if per_obs:
        expected += f" or one value per observation ({n}, {k})"
    raise ValueError(f"{name} has shape {arr.shape}; expected {expected}.")


def _observation_sigma(
    y: np.ndarray,
    sigma: float | Sequence[float] | np.ndarray | None,
    sigma_rel: float | Sequence[float] | None,
    label: Callable[[int, int], str] | None = None,
) -> np.ndarray | None:
    """Standard deviation of each observation as an array ``(N, K)``.

    Values come from ``sigma`` where it is finite and from
    ``sigma_rel * |observed value|`` otherwise; entries without an
    observation are ``nan`` (see :class:`Solver`).

    Parameters
    ----------
    y : ndarray
        Observations, shape ``(N, K)``, ``nan`` where nothing was observed.
    sigma, sigma_rel
        As in the solver methods.
    label : callable, optional
        ``label(row, column)`` gives the position of an observation in error
        messages (default: row and tracer number).

    Returns
    -------
    ndarray or None
        ``None`` if neither ``sigma`` nor ``sigma_rel`` is given.

    Raises
    ------
    ValueError
        If ``sigma`` or ``sigma_rel`` has an unsupported shape, or if an
        observation is left without a positive standard deviation.
    """
    if sigma is None and sigma_rel is None:
        return None
    y = np.asarray(y, dtype=float)
    n, k = y.shape
    if sigma is None:
        sig = np.full((n, k), np.nan)
    else:
        sig = _broadcast(sigma, n, k, "sigma", per_obs=True)
    if sigma_rel is not None:
        rel = _broadcast(sigma_rel, n, k, "sigma_rel", per_obs=False)
        sig = np.where(np.isfinite(sig), sig, rel * np.abs(y))

    observed = ~np.isnan(y)
    bad = observed & ~(np.isfinite(sig) & (sig > 0.0))
    if np.any(bad):
        if label is None:

            def label(r: int, c: int) -> str:
                return f"row {r}" + (f" (tracer {c + 1})" if k > 1 else "")

        rows, cols = np.nonzero(bad)
        items = []
        for r, c in zip(rows[:5], cols[:5]):
            item = label(int(r), int(c))
            if np.isfinite(sig[r, c]):
                item += f" (sigma = {sig[r, c]:g})"
            items.append(item)
        more = f" and {rows.size - 5} more" if rows.size > 5 else ""
        raise ValueError(
            f"{rows.size} observation(s) have no positive measurement uncertainty: "
            f"{', '.join(items)}{more}. Give sigma (standard deviation, > 0) for "
            "these observations, or sigma_rel (relative uncertainty) as a fallback."
        )
    sig[~observed] = np.nan
    return sig


@dataclass
class Solver:
    """Optimization wrapper for :class:`~PyTracerLab.model.model.Model`.

    The solver interacts **only** with the model's parameter registry. It
    constructs a free-parameter vector and corresponding bounds, runs a chosen
    optimizer, and writes the best solution back to the registry (and thus the
    Units via write-through).

    Notes
    -----
    Parameters with ``fixed=True`` are excluded from optimization but their
    current values are honored in the simulation. The mixing fraction of a
    binary mixture is an ordinary bounded parameter (see
    :class:`~PyTracerLab.model.model.Model`), so all solvers can calibrate
    it. A warning is issued if there are more free parameters than
    observations.

    **Measurement uncertainty.** All solver methods accept the measurement
    uncertainty of the observations as ``sigma`` (absolute standard
    deviation, in the unit of the tracer) and ``sigma_rel`` (relative
    standard deviation as a fraction of the observed value, e.g. ``0.05``
    for 5 %). ``sigma`` can be a scalar, one value per tracer, or one value
    per observation (same shape as the observations, e.g. the
    ``uncertainty`` of a series read with
    :func:`~PyTracerLab.model.io.read_series`, with ``nan`` where no value
    is given). For each observation, the standard deviation
    :math:`\\sigma_i` is taken from ``sigma`` where it is given and is
    otherwise computed from ``sigma_rel``; an error is raised if an
    observation is left without a positive standard deviation. The
    residuals :math:`r_i` (simulated minus observed) are weighted by
    :math:`1/\\sigma_i`: differential evolution minimizes the mean of
    :math:`(r_i/\\sigma_i)^2`, least squares minimizes their sum, and the
    MCMC samplers use the Gaussian log-likelihood

    .. math::

        \\log L = -\\frac{1}{2} \\sum_i \\left(\\frac{r_i}{\\sigma_i}\\right)^2
        - \\sum_i \\log \\sigma_i - \\frac{n}{2} \\log (2 \\pi).

    If neither ``sigma`` nor ``sigma_rel`` is given, differential evolution
    and least squares weight all observations equally, and the MCMC
    samplers treat the standard deviation of each tracer :math:`k` as
    unknown and marginalize it (Jeffreys prior), which gives
    :math:`\\log L = -\\sum_k \\frac{n_k}{2} \\log \\mathrm{SSE}_k` with the
    sum of squared residuals :math:`\\mathrm{SSE}_k` and the number of
    observations :math:`n_k` of tracer :math:`k`.
    """

    model: Model

    def __post_init__(self) -> None:
        # Fail early with a clear message instead of a silently infinite
        # objective if observations are not aligned with the input series.
        self.model._validate_series()

    # ------------------------- internals ---------------------------------
    def _reduced_bounds(self) -> List[Tuple[float, float]]:
        """Bounds for free parameters in registry order."""
        return self.model.get_bounds(free_only=True)

    def _obs_matrix(self) -> np.ndarray:
        """Observations without the warm-up period as a 2D array ``(N, K)``.

        Slicing off ``n_warmup`` rows gives the original observations both
        before and after the warm-up has been prepended to
        ``model.target_series``.
        """
        if self.model.target_series is None:
            raise ValueError("Calibration requires observations (target_series) on the model.")
        y = np.asarray(self.model.target_series[self.model.n_warmup :], dtype=float)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        if not np.any(~np.isnan(y)):
            raise ValueError("The observations contain no values (all entries are NaN).")
        return y

    def _warn_if_underdetermined(self, y: np.ndarray) -> None:
        """Warn if there are more free parameters than observations.

        The parameters then cannot be determined uniquely from the
        observations alone (e.g. both mean travel times and the mixing
        fraction of a binary mixture from two single-sample tracers).
        """
        n_obs = int(np.count_nonzero(~np.isnan(y)))
        n_free = len(self.model.param_keys(free_only=True))
        if n_free > n_obs:
            warnings.warn(
                f"There are more free parameters ({n_free}) than observations ({n_obs}), "
                "so the parameters cannot be determined uniquely. Consider fixing "
                "parameters (e.g. the mean travel time of one unit of a binary mixture) "
                "or adding observations.",
                PyTracerLabWarning,
                stacklevel=3,
            )

    def _obs_label(self, row: int, col: int, n_tracers: int) -> str:
        """Human-readable position of an observation for error messages."""
        label = f"row {row}"
        times = self.model.time_steps
        if times is not None and len(times) == len(self.model.input_series) - self.model.n_warmup:
            t = times[row]
            label = t.strftime("%Y-%m-%d") if hasattr(t, "strftime") else str(t)
        if n_tracers > 1:
            label += f" (tracer {col + 1})"
        return label

    def _sigma_matrix(
        self,
        sigma: float | Sequence[float] | np.ndarray | None,
        sigma_rel: float | Sequence[float] | None,
    ) -> np.ndarray | None:
        """Standard deviation of each observation as an array ``(N, K)``.

        See :func:`_observation_sigma`; error messages name the dates of the
        observations if ``model.time_steps`` is given.
        """
        if sigma is None and sigma_rel is None:
            return None
        y = self._obs_matrix()
        k = y.shape[1]
        return _observation_sigma(y, sigma, sigma_rel, label=lambda r, c: self._obs_label(r, c, k))

    def _simulate_given_free(self, free_params: Sequence[float]) -> np.ndarray:
        """Set free params and return simulation."""
        self.model.set_vector(list(free_params), which="value", free_only=True)
        try:
            sim = self.model.simulate()
        except Exception:
            sim = np.empty((len(self.model.target_series)))
            sim[:] = np.nan
        return sim

    @staticmethod
    def _as_2d(sim: np.ndarray) -> np.ndarray:
        """Simulation as a 2D float array ``(N, K)``."""
        s = np.asarray(sim, dtype=float)
        return s.reshape(-1, 1) if s.ndim == 1 else s

    def _obj(self, free_params: Sequence[float], y: np.ndarray, sig: np.ndarray | None) -> float:
        """Mean of squared (optionally sigma-weighted) residuals over all observations.

        ``y`` and ``sig`` are the arrays from :meth:`_obs_matrix` and
        :meth:`_sigma_matrix`. Failed simulations give a large finite value.
        """
        s2 = self._as_2d(self._simulate_given_free(free_params))
        if s2.shape != y.shape:
            return _BAD_OBJECTIVE
        mask = ~np.isnan(y)
        resid = s2[mask] - y[mask]
        if sig is not None:
            resid = resid / sig[mask]
        val = float(np.mean(resid**2))
        return val if np.isfinite(val) else _BAD_OBJECTIVE

    def differential_evolution(
        self,
        maxiter: int = 10000,
        popsize: int = 100,
        mutation: Tuple[float, float] = (0.5, 1.99),
        recombination: float = 0.5,
        tol: float = 1e-3,
        sigma: Union[float, Sequence[float], np.ndarray, None] = None,
        random_state: int | None = None,
        sigma_rel: Union[float, Sequence[float], None] = None,
    ) -> Tuple[Dict[str, float], np.ndarray]:
        """Run differential evolution and return the best solution.

        Minimizes the mean of the squared residuals, each divided by the
        measurement uncertainty of the observation if ``sigma`` or
        ``sigma_rel`` is given (see :class:`Solver`).

        Parameters
        ----------
        maxiter : int, optional
            Max iterations in SciPy's DE.
        popsize : int, optional
            Population size multiplier.
        mutation : (float, float), optional
            DE mutation constants.
        recombination : float, optional
            DE recombination constant.
        tol : float, optional
            Convergence tolerance.
        sigma : float | array_like | None, optional
            Measurement uncertainty (standard deviation): scalar, one value
            per tracer, or one value per observation (e.g.
            ``obs.uncertainty``, ``nan`` where not given). If neither
            ``sigma`` nor ``sigma_rel`` is given, all observations are
            weighted equally.
        random_state : int | None, optional
            Random seed.
        sigma_rel : float | array_like | None, optional
            Relative measurement uncertainty as a fraction of the observed
            value (e.g. ``0.05``), scalar or one value per tracer. Used for
            all observations for which ``sigma`` gives no value.

        Returns
        -------
        (dict, ndarray)
            Mapping from parameter key to optimized value, and the simulated
            series at that optimum.
        """
        # Validate bounds exist for all free parameters
        bounds = self._reduced_bounds()

        # Observations and their standard deviations (checked before the run)
        y = self._obs_matrix()
        self._warn_if_underdetermined(y)
        sig = self._sigma_matrix(sigma, sigma_rel)

        # Build init vector and repair non-finite initials by midpoint of bounds
        init_free = self.model.get_vector(which="initial", free_only=True)
        keys_free = self.model.param_keys(free_only=True)
        repaired = []
        for k, v, (lo, hi) in zip(keys_free, init_free, bounds):
            if not np.isfinite(v):
                mid = 0.5 * (float(lo) + float(hi))
                self.model.set_initial(k, mid)
                repaired.append((k, v, mid))
        if repaired:
            # (optional) print or log repaired initials
            pass

        # Seed current values from initials for a clean, reproducible start
        init_free = self.model.get_vector(which="initial", free_only=True)
        self.model.set_vector(init_free, which="value", free_only=True)

        result = differential_evolution(
            self._obj,
            bounds=bounds,
            maxiter=maxiter,
            popsize=popsize,
            mutation=mutation,
            recombination=recombination,
            tol=tol,
            rng=random_state,
            args=(y, sig),
        )

        # Write back and simulate once more at the best params
        sim = self._simulate_given_free(result.x)
        # the model parameters are already set at the optimum
        solution = {k: float(self.model.params[k]["value"]) for k in self.model.params}
        return solution, sim

    def least_squares(
        self,
        ftol: float = 1e-8,
        xtol: float = 1e-8,
        gtol: float = 1e-8,
        max_nfev: int = 10000,
        sigma: float | Sequence[float] | np.ndarray | None = None,
        sigma_rel: float | Sequence[float] | None = None,
    ) -> Tuple[Dict[str, float], np.ndarray]:
        """Run gradient-based non-linear least squares and return the best solution.

        Uses :func:`scipy.optimize.curve_fit` with parameter bounds, i.e.,
        the trust-region reflective algorithm, starting from the initial
        parameter values. Minimizes the sum of the squared residuals, each
        divided by the measurement uncertainty of the observation if
        ``sigma`` or ``sigma_rel`` is given (see :class:`Solver`).

        Parameters
        ----------
        ftol : float, optional
            Tolerance for termination by the change of the cost function.
        xtol : float, optional
            Tolerance for termination by the change of the independent
            variables.
        gtol : float, optional
            Tolerance for termination by the norm of the gradient.
        max_nfev : int, optional
            Maximum number of function evaluations.
        sigma : float | array_like | None, optional
            Measurement uncertainty (standard deviation): scalar, one value
            per tracer, or one value per observation (e.g.
            ``obs.uncertainty``, ``nan`` where not given). If neither
            ``sigma`` nor ``sigma_rel`` is given, all observations are
            weighted equally.
        sigma_rel : float | array_like | None, optional
            Relative measurement uncertainty as a fraction of the observed
            value (e.g. ``0.05``), scalar or one value per tracer. Used for
            all observations for which ``sigma`` gives no value.

        Returns
        -------
        (dict, ndarray)
            Mapping from parameter key to optimized value, and the simulated
            series at that optimum.
        """
        # Validate bounds exist for all free parameters
        bounds = self._reduced_bounds()

        # Observations and their standard deviations (checked before the run)
        y = self._obs_matrix()
        self._warn_if_underdetermined(y)
        sig = self._sigma_matrix(sigma, sigma_rel)
        # one mask for observations, simulated equivalents and sigma
        mask = ~np.isnan(y)
        ydata = y[mask]
        sigma_vec = None if sig is None else sig[mask]

        # Build init vector and repair non-finite initials by midpoint of bounds
        init_free = self.model.get_vector(which="initial", free_only=True)
        keys_free = self.model.param_keys(free_only=True)
        repaired = []
        for k, v, (lo, hi) in zip(keys_free, init_free, bounds):
            if not np.isfinite(v):
                mid = 0.5 * (float(lo) + float(hi))
                self.model.set_initial(k, mid)
                repaired.append((k, v, mid))
        if repaired:
            # (optional) print or log repaired initials
            pass

        # Seed current values from initials for a clean, reproducible start
        init_free = self.model.get_vector(which="initial", free_only=True)
        self.model.set_vector(init_free, which="value", free_only=True)

        def simulated_equivalents(_x: np.ndarray, *free_params: float) -> np.ndarray:
            """Simulated values at the observations (same order as ``ydata``)."""
            s2 = self._as_2d(self._simulate_given_free(free_params))
            if s2.shape != y.shape:
                return np.full(ydata.size, _BAD_OBJECTIVE)
            out = s2[mask]
            return np.where(np.isfinite(out), out, _BAD_OBJECTIVE)

        # curve_fit expects bounds as (lower bounds, upper bounds)
        bounds_cf = ([b[0] for b in bounds], [b[1] for b in bounds])

        popt, pcov = curve_fit(
            f=simulated_equivalents,
            xdata=np.arange(ydata.size),  # dummy x data
            ydata=ydata,
            p0=init_free,
            bounds=bounds_cf,
            sigma=sigma_vec,
            absolute_sigma=sigma_vec is not None,
            ftol=ftol,
            xtol=xtol,
            gtol=gtol,
            maxfev=max_nfev,
        )

        # Write back and simulate once more at the best params
        sim = self._simulate_given_free(popt)
        solution = {k: float(self.model.params[k]["value"]) for k in self.model.params}
        return solution, sim

    @staticmethod
    def _log_prior_uniform(theta: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
        """Uniform prior inside [lo, hi], -inf outside."""
        if np.any(theta < lo) or np.any(theta > hi):
            return -np.inf
        return 0.0

    @staticmethod
    def _loglik(y: np.ndarray, sim: np.ndarray, sig: np.ndarray | None) -> float:
        """Gaussian log-likelihood of the observations ``y`` given ``sim``.

        ``y`` and ``sig`` are the arrays from :meth:`_obs_matrix` and
        :meth:`_sigma_matrix`. With ``sig``, the standard deviations are
        treated as known. Without, the standard deviation of each tracer is
        marginalized (Jeffreys prior), giving
        ``sum_k -(n_k / 2) * log(SSE_k)``. Failed simulations (non-finite
        values at observations) give ``-inf``.
        """
        s2 = Solver._as_2d(sim)
        if s2.shape != y.shape:
            return -np.inf
        mask = ~np.isnan(y)
        resid = s2 - y
        if not np.all(np.isfinite(resid[mask])):
            return -np.inf
        if sig is None:
            ll = 0.0
            for j in range(y.shape[1]):
                rj = resid[mask[:, j], j]
                if rj.size == 0:
                    continue
                sse = max(float(np.dot(rj, rj)), 1e-300)
                ll -= 0.5 * rj.size * np.log(sse)
            return float(ll)
        s = sig[mask]
        z = resid[mask] / s
        return float(-0.5 * np.dot(z, z) - np.sum(np.log(s)) - 0.5 * z.size * np.log(2.0 * np.pi))

    def mcmc_sample(
        self,
        n_samples: int,
        burn_in: int = 1000,
        thin: int = 1,
        rw_scale: float = 0.05,
        rw_scale_isotropic: bool = True,
        sigma: Union[float, Sequence[float], np.ndarray, None] = None,
        log_prior: callable | None = None,
        start: Sequence[float] | None = None,
        random_state: int | np.random.Generator | None = None,
        return_sim: bool = False,
        set_model_state: bool = True,
        sigma_rel: Union[float, Sequence[float], None] = None,
    ):
        """Random-Walk Metropolis–Hastings over free parameters.

        Run Metropolis–Hastings MCMC with a RW proposal distribution for
        the free parameters. The method only returns effective samples
        (after burn-in and thinning).

        Parameters
        ----------
        n_samples : int
            Number of accepted samples to draw.
        burn_in : int, optional
            Number of samples to discard as burn-in.
        thin : int, optional
            Thinning factor.
        rw_scale : float | Sequence[float], optional
            Variance of RW proposal distribution. If given as a sequence
            (one scale element for each free parameter in the model), the
            order of the elements has to match the order from
            ``model.param_keys(free_only=True)``.
        rw_scale_isotropic : bool, optional
            If True, use an isotropic RW proposal with rw_scale as variance.
        sigma : float | array_like | None, optional
            Measurement uncertainty (standard deviation of the Gaussian
            likelihood): scalar, one value per tracer, or one value per
            observation (e.g. ``obs.uncertainty``, ``nan`` where not given).
            If neither ``sigma`` nor ``sigma_rel`` is given, the standard
            deviation of each tracer is marginalized (see :class:`Solver`).
        log_prior : callable | None, optional
            Log prior, if not uniform.
        start : Sequence[float] | None, optional
            Starting point for MCMC.
        random_state : int | np.random.Generator | None, optional
            Random seed.
        return_sim : bool, optional
            Return simulated series at each sample.
        set_model_state : bool, optional
            Set model state to the posterior median at the end.
        sigma_rel : float | array_like | None, optional
            Relative measurement uncertainty as a fraction of the observed
            value (e.g. ``0.05``), scalar or one value per tracer. Used for
            all observations for which ``sigma`` gives no value.

        Returns
        -------
        dict
            Dictionary with keys: 'param_names', 'samples', 'logpost',
            'accept_rate', 'posterior_mean', 'posterior_map',
            'map_logpost', ['sims' if requested]
        """
        rng = (
            np.random.default_rng(random_state)
            if not isinstance(random_state, np.random.Generator)
            else random_state
        )

        if self.model.target_series is None:
            raise ValueError("MCMC requires target_series on the model.")

        y_obs = self._obs_matrix()
        self._warn_if_underdetermined(y_obs)
        sig = self._sigma_matrix(sigma, sigma_rel)
        bounds = np.asarray(self._reduced_bounds(), dtype=float)
        lo, hi = bounds[:, 0], bounds[:, 1]
        if np.any(~np.isfinite(lo)) or np.any(~np.isfinite(hi)):
            raise ValueError("All free parameters must have finite bounds for MCMC.")

        keys_free = self.model.param_keys(free_only=True)
        d = len(keys_free)

        # Start vector
        if start is None:
            start_vec = np.asarray(
                self.model.get_vector(which="value", free_only=True), dtype=float
            )
            if np.any(~np.isfinite(start_vec)):
                start_vec = np.asarray(
                    self.model.get_vector(which="initial", free_only=True), dtype=float
                )
            start_vec = np.clip(start_vec, lo, hi)
        else:
            start_vec = np.asarray(start, dtype=float)
            if start_vec.shape != (d,):
                raise ValueError(f"`start` must have length {d}.")
            start_vec = np.clip(start_vec, lo, hi)

        # Proposal steps
        if not rw_scale_isotropic:
            step = rw_scale * (hi - lo)
            step = np.where(step <= 0.0, 1e-12, step)
        else:
            step = np.full(d, rw_scale)

        # Evaluate initial point
        cur = start_vec.copy()
        cur_lp = log_prior(cur) if log_prior is not None else self._log_prior_uniform(cur, lo, hi)
        if not np.isfinite(cur_lp):
            raise ValueError("Initial point has zero prior density; choose a valid `start`.")
        cur_sim = self._simulate_given_free(cur)
        cur_ll = self._loglik(y_obs, cur_sim, sig)
        cur_logpost = cur_lp + cur_ll

        if not np.isfinite(cur_logpost):
            # Try to find a finite start by small jitters
            for _ in range(100):
                trial = np.clip(cur + rng.normal(0.0, step), lo, hi)
                sim = self._simulate_given_free(trial)
                ll = self._loglik(y_obs, sim, sig)
                lp = (
                    log_prior(trial)
                    if log_prior is not None
                    else self._log_prior_uniform(trial, lo, hi)
                )
                lpt = lp + ll
                if np.isfinite(lpt):
                    cur, cur_sim, cur_logpost = trial, sim, lpt
                    break
            if not np.isfinite(cur_logpost):
                raise RuntimeError("Failed to find a finite starting point for MCMC.")

        # Storage
        n_keep = n_samples
        total_needed = burn_in + n_keep * thin
        samples = np.empty((n_keep, d), dtype=float)
        logposts = np.empty(n_keep, dtype=float)
        sims = [] if return_sim else None

        # MH loop
        accepts = 0
        keep_idx = 0

        for it in range(total_needed):
            prop = cur + rng.normal(0.0, step, size=d)

            if np.any(prop < lo) or np.any(prop > hi):
                accept = False  # fast reject
            else:
                prop_sim = self._simulate_given_free(prop)
                prop_ll = self._loglik(y_obs, prop_sim, sig)

                if np.isfinite(prop_ll):
                    prop_lp = (
                        log_prior(prop)
                        if log_prior is not None
                        else self._log_prior_uniform(prop, lo, hi)
                    )
                    prop_logpost = prop_ll + prop_lp
                else:
                    prop_logpost = -np.inf
                log_alpha = prop_logpost - cur_logpost  # symmetric proposal
                # if log_alpha >= 0.0 or np.log(rng.uniform()) < log_alpha:
                p_acc = min(0.0, log_alpha)
                if np.log(rng.uniform()) < p_acc:
                    cur, cur_sim, cur_logpost = prop, prop_sim, prop_logpost
                    accept = True
                else:
                    accept = False

            accepts += int(accept)

            if it >= burn_in and ((it - burn_in) % thin == 0):
                samples[keep_idx, :] = cur
                logposts[keep_idx] = cur_logpost
                if return_sim:
                    sims.append(cur_sim.copy())
                keep_idx += 1
                if keep_idx == n_keep:
                    break

        accept_rate = accepts / float(total_needed)
        post_mean_free = samples.mean(axis=0)
        post_median_free = np.median(samples, axis=0)
        map_idx = int(np.argmax(logposts))
        post_map_free = samples[map_idx].copy()

        posterior_mean = {k: float(v) for k, v in zip(keys_free, post_mean_free)}
        posterior_median = {k: float(v) for k, v in zip(keys_free, post_median_free)}
        posterior_map = {k: float(v) for k, v in zip(keys_free, post_map_free)}

        if set_model_state:
            self.model.set_vector(post_map_free.tolist(), which="value", free_only=True)

        out = {
            "param_names": list(keys_free),
            "samples": samples,
            "logpost": logposts,
            "accept_rate": float(accept_rate),
            "posterior_mean": posterior_mean,
            "posterior_median": posterior_median,
            "posterior_map": posterior_map,
            "map_logpost": float(logposts[map_idx]),
        }
        if return_sim:
            out["sims"] = np.asarray(sims, dtype=float)
        return out

    def dream_sample(
        self,
        n_samples: int,
        n_chains: int | None = None,
        burn_in: int = 1000,
        thin: int = 1,
        n_diff_pairs: int = 3,
        cr: float | Sequence[float] = [0.5, 1.0],
        gamma: float | None = None,
        gamma_jitter: float = 0.1,
        jitter: float = 1e-6,
        sigma: Union[float, Sequence[float], np.ndarray, None] = None,
        init_state_sigma: float = 1e-6,
        log_prior: callable | None = None,
        start: Sequence[float] | np.ndarray | None = None,
        random_state: int | np.random.Generator | None = None,
        return_sim: bool = False,
        set_model_state: bool = True,
        sigma_rel: Union[float, Sequence[float], None] = None,
    ):
        """Basic DREAM sampler.

        Uses multiple chains and differential-evolution proposals with
        crossover/subspace updates. Returns samples after burn-in and thinning.

        Parameters
        ----------
        n_samples : int
            Number of samples to keep per chain.
        n_chains : int, optional
            Number of parallel chains. Default is max(3, 2 * d).
        burn_in : int, optional
            Number of samples to discard as burn-in.
        thin : int, optional
            Thinning factor.
        n_diff_pairs : int, optional
            Number of differential evolution pairs.
        cr : float | Sequence[float], optional
            Crossover probability (or list of probabilities to sample from).
        gamma : float | None, optional
            Differential evolution scale. If None, uses 2.38 / sqrt(2 * m),
            where m is the number of updated dimensions.
        gamma_jitter : float, optional
            Relative jitter applied to gamma (uniform in +/- gamma_jitter).
        jitter : float, optional
            Small Gaussian jitter scale relative to bounds.
        sigma : float | array_like | None, optional
            Measurement uncertainty (standard deviation of the Gaussian
            likelihood): scalar, one value per tracer, or one value per
            observation (e.g. ``obs.uncertainty``, ``nan`` where not given).
            If neither ``sigma`` nor ``sigma_rel`` is given, the standard
            deviation of each tracer is marginalized (see :class:`Solver`).
        init_state_sigma : float, optional
            Standard deviation of the Gaussian noise added to the starting
            point to ensure all chains are initialized at different points.
            Has no effect if no starting point is given or if starting
            points are given for all chains.
        log_prior : callable | None, optional
            Log prior, if not uniform within bounds.
        start : Sequence[float] | ndarray | None, optional
            Starting point(s) for chains. If 1D, it is used as a base
            for all chains with small jitter. If 2D, shape must be
            (n_chains, d).
        random_state : int | np.random.Generator | None, optional
            Random seed.
        return_sim : bool, optional
            Return simulated series at each sample.
        set_model_state : bool, optional
            Set model state to the posterior median at the end.
        sigma_rel : float | array_like | None, optional
            Relative measurement uncertainty as a fraction of the observed
            value (e.g. ``0.05``), scalar or one value per tracer. Used for
            all observations for which ``sigma`` gives no value.

        Returns
        -------
        dict
            Dictionary with keys: 'param_names', 'samples', 'logpost',
            'accept_rate', 'posterior_mean', 'posterior_median',
            'posterior_map', 'map_logpost', plus chain-level arrays and
            ['sims' if requested].
        """
        rng = (
            np.random.default_rng(random_state)
            if not isinstance(random_state, np.random.Generator)
            else random_state
        )

        if self.model.target_series is None:
            raise ValueError("DREAM requires target_series on the model.")

        if n_samples <= 0:
            raise ValueError("n_samples must be positive.")
        if burn_in < 0:
            raise ValueError("burn_in must be non-negative.")
        if thin <= 0:
            raise ValueError("thin must be positive.")
        if n_diff_pairs < 1:
            raise ValueError("n_diff_pairs must be >= 1.")
        if gamma is not None and float(gamma) <= 0.0:
            raise ValueError("gamma must be positive.")
        if gamma_jitter < 0.0:
            raise ValueError("gamma_jitter must be non-negative.")
        if jitter < 0.0:
            raise ValueError("jitter must be non-negative.")

        y_obs = self._obs_matrix()
        self._warn_if_underdetermined(y_obs)
        sig = self._sigma_matrix(sigma, sigma_rel)
        bounds = np.asarray(self._reduced_bounds(), dtype=float)
        lo, hi = bounds[:, 0], bounds[:, 1]
        if np.any(~np.isfinite(lo)) or np.any(~np.isfinite(hi)):
            raise ValueError("All free parameters must have finite bounds for DREAM.")

        keys_free = self.model.param_keys(free_only=True)
        d = len(keys_free)
        if d == 0:
            raise ValueError("No free parameters to sample.")

        start_arr = None
        if start is not None:
            start_arr = np.asarray(start, dtype=float)
            if start_arr.ndim == 2 and n_chains is None:
                n_chains = int(start_arr.shape[0])

        if n_chains is None:
            n_chains = max(3, 2 * d)
        if n_chains < 3:
            raise ValueError("n_chains must be at least 3 for DREAM.")
        if 2 * n_diff_pairs > (n_chains - 1):
            raise ValueError("n_diff_pairs too large for the number of chains.")

        # Crossover probabilities
        if isinstance(cr, (list, tuple, np.ndarray)):
            cr_vals = np.asarray(cr, dtype=float).ravel()
            if cr_vals.size == 0 or np.any(~np.isfinite(cr_vals)):
                raise ValueError("cr must contain finite values.")
        else:
            cr_vals = np.asarray([float(cr)], dtype=float)
        if np.any(cr_vals <= 0.0) or np.any(cr_vals > 1.0):
            raise ValueError("cr values must be in (0, 1].")

        # Build base start vector
        base_vec = np.asarray(self.model.get_vector(which="value", free_only=True), dtype=float)
        if np.any(~np.isfinite(base_vec)):
            base_vec = np.asarray(
                self.model.get_vector(which="initial", free_only=True), dtype=float
            )
        if np.any(~np.isfinite(base_vec)):
            base_vec = 0.5 * (lo + hi)
        base_vec = np.clip(base_vec, lo, hi)

        # Initialize chain states
        if start is None:
            chains = rng.uniform(lo, hi, size=(n_chains, d))
            chains[0, :] = base_vec
        else:
            if start_arr.shape == (d,):
                chains = np.tile(start_arr, (n_chains, 1))
                # small spread to avoid identical chains
                chains += rng.normal(0.0, init_state_sigma, size=(n_chains, d))
                chains = np.clip(chains, lo, hi)
            elif start_arr.shape == (n_chains, d):
                chains = np.clip(start_arr, lo, hi)
            else:
                raise ValueError("start must have shape (d,) or (n_chains, d).")

        # Evaluate initial points; resample if needed
        chain_logpost = np.empty(n_chains, dtype=float)
        chain_sim: list[np.ndarray] = [np.empty(0) for _ in range(n_chains)]
        max_init_tries = 100
        for i in range(n_chains):
            theta = chains[i].copy()
            ok = False
            for _ in range(max_init_tries):
                lp = (
                    log_prior(theta)
                    if log_prior is not None
                    else self._log_prior_uniform(theta, lo, hi)
                )
                if not np.isfinite(lp):
                    theta = rng.uniform(lo, hi, size=d)
                    continue
                sim = self._simulate_given_free(theta)
                ll = self._loglik(y_obs, sim, sig)
                lpst = lp + ll
                if np.isfinite(lpst):
                    chains[i] = theta
                    chain_sim[i] = sim
                    chain_logpost[i] = lpst
                    ok = True
                    break
                theta = rng.uniform(lo, hi, size=d)
            if not ok:
                raise RuntimeError("Failed to find a finite starting point for DREAM.")

        # Storage
        n_keep = n_samples
        total_needed = burn_in + n_keep * thin
        samples_chain = np.empty((n_chains, n_keep, d), dtype=float)
        logposts_chain = np.empty((n_chains, n_keep), dtype=float)
        sims_chain = None
        if return_sim:
            sims_chain = [[None] * n_keep for _ in range(n_chains)]

        accepts = np.zeros(n_chains, dtype=int)
        proposals = np.zeros(n_chains, dtype=int)
        keep_idx = 0

        # Precompute jitter scale
        eps_scale = np.where((hi - lo) > 0.0, (hi - lo), 1.0)

        for it in range(total_needed):
            for i in range(n_chains):
                proposals[i] += 1
                cur = chains[i]
                cur_lpst = chain_logpost[i]

                # Choose differential evolution pairs from other chains
                others = [j for j in range(n_chains) if j != i]
                idx = rng.choice(others, size=2 * n_diff_pairs, replace=False)
                diff = np.zeros(d, dtype=float)
                for k in range(n_diff_pairs):
                    a = idx[2 * k]
                    b = idx[2 * k + 1]
                    diff += chains[a] - chains[b]

                # Crossover mask
                cr_i = float(rng.choice(cr_vals))
                mask = rng.random(d) < cr_i
                if not np.any(mask):
                    mask[rng.integers(0, d)] = True
                m = int(np.sum(mask))

                # Gamma scaling
                if gamma is None:
                    gamma_i = 2.38 / np.sqrt(2.0 * n_diff_pairs * m)
                else:
                    gamma_i = float(gamma)
                if gamma_jitter > 0.0:
                    gamma_i *= 1.0 + rng.uniform(-gamma_jitter, gamma_jitter)

                # Proposal
                prop = cur.copy()
                if jitter > 0.0:
                    eps = rng.normal(0.0, jitter, size=d) * eps_scale
                    prop[mask] = cur[mask] + gamma_i * diff[mask] + eps[mask]
                else:
                    prop[mask] = cur[mask] + gamma_i * diff[mask]

                if np.any(prop < lo) or np.any(prop > hi):
                    accept = False
                else:
                    if log_prior is not None:
                        prop_lp = float(log_prior(prop))
                        if not np.isfinite(prop_lp):
                            accept = False
                        else:
                            prop_sim = self._simulate_given_free(prop)
                            prop_ll = self._loglik(y_obs, prop_sim, sig)
                            prop_logpost = prop_lp + prop_ll
                            log_alpha = prop_logpost - cur_lpst
                            p_acc = min(0.0, log_alpha)
                            if np.log(rng.uniform()) < p_acc:
                                chains[i] = prop
                                chain_sim[i] = prop_sim
                                chain_logpost[i] = prop_logpost
                                accept = True
                            else:
                                accept = False
                    else:
                        prop_sim = self._simulate_given_free(prop)
                        prop_ll = self._loglik(y_obs, prop_sim, sig)
                        prop_logpost = prop_ll  # uniform prior inside bounds
                        log_alpha = prop_logpost - cur_lpst
                        p_acc = min(0.0, log_alpha)
                        if np.log(rng.uniform()) < p_acc:
                            chains[i] = prop
                            chain_sim[i] = prop_sim
                            chain_logpost[i] = prop_logpost
                            accept = True
                        else:
                            accept = False

                accepts[i] += int(accept)

            if it >= burn_in and ((it - burn_in) % thin == 0):
                for i in range(n_chains):
                    samples_chain[i, keep_idx, :] = chains[i]
                    logposts_chain[i, keep_idx] = chain_logpost[i]
                    if return_sim and sims_chain is not None:
                        sims_chain[i][keep_idx] = chain_sim[i].copy()
                keep_idx += 1
                if keep_idx == n_keep:
                    break

        accept_rate_chain = accepts / np.maximum(proposals, 1)
        accept_rate = float(np.mean(accept_rate_chain))

        samples = samples_chain.reshape(-1, d)
        logposts = logposts_chain.reshape(-1)
        post_mean_free = samples.mean(axis=0)
        post_median_free = np.median(samples, axis=0)
        map_idx = int(np.argmax(logposts))
        post_map_free = samples[map_idx].copy()

        posterior_mean = {k: float(v) for k, v in zip(keys_free, post_mean_free)}
        posterior_median = {k: float(v) for k, v in zip(keys_free, post_median_free)}
        posterior_map = {k: float(v) for k, v in zip(keys_free, post_map_free)}

        # compute gelman rubin statistic
        gr = self.gelman_rubin(samples_chain)
        gr = {k: float(v) for k, v in zip(keys_free, gr)}

        if set_model_state:
            self.model.set_vector(post_map_free.tolist(), which="value", free_only=True)

        out = {
            "param_names": list(keys_free),
            "samples": samples,
            "logpost": logposts,
            "samples_chain": samples_chain,
            "logpost_chain": logposts_chain,
            "accept_rate": accept_rate,
            "accept_rate_per_chain": accept_rate_chain.tolist(),
            "posterior_mean": posterior_mean,
            "posterior_median": posterior_median,
            "posterior_map": posterior_map,
            "map_logpost": float(logposts[map_idx]),
            "gelman_rubin": gr,
        }
        if return_sim:
            sims_arr = np.asarray(sims_chain, dtype=float)
            out["sims"] = sims_arr
        return out

    def gelman_rubin(self, samples: np.ndarray) -> np.ndarray:
        """Compute the Gelman-Rubin statistic for a set of MCMC samples.

        Parameters
        ----------
        samples : array-like
            Array of MCMC samples of shape (n_chains, n_samples, n_params).

        Returns
        -------
        np.ndarray
            Gelman-Rubin statistic for each parameter.
        """
        # get shape
        n_chains, n_samples, _ = samples.shape

        # means within chains
        chain_means = np.mean(samples, axis=1)  # (n_chains, n_params)

        # mean of means of all chains
        mean_of_means = np.mean(chain_means, axis=0)  # (n_params,)

        # variance of means of all chains
        B = (
            n_samples / (n_chains - 1) * np.sum((chain_means - mean_of_means) ** 2, axis=0)
        )  # (n_params,)

        # average variances of chains across all chains
        # for the inner sum to work, we swap the axes of the array so that the
        # samples are on the first axis
        samples = np.swapaxes(samples, 0, 1)
        W_inner = (
            1 / (n_samples - 1) * np.sum((samples - chain_means) ** 2, axis=0)
        )  # (n_chains, n_params)
        W = 1 / n_chains * np.sum(W_inner, axis=0)  # (n_params,)

        # compute statistic
        statistic = (((n_samples - 1) / n_samples * W) + ((1 / n_samples) * B)) / W

        return np.atleast_1d(statistic)


###
# Solver registry for GUI
###


def _run_de(model: Model, params: Dict[str, Any] | None = None) -> Dict[str, object]:
    """Run Differential Evolution and return a standardized payload for the GUI."""
    p = params or {}
    maxiter = int(p.get("maxiter", 10000))
    popsize = int(p.get("popsize", 100))
    mutation = p.get("mutation", (0.5, 1.99))
    # Ensure mutation is a 2-tuple
    if isinstance(mutation, (list, tuple)) and len(mutation) == 2:
        mutation = (float(mutation[0]), float(mutation[1]))
    else:
        mutation = (0.5, 1.99)
    recombination = float(p.get("recombination", 0.5))
    tol = float(p.get("tol", 1e-3))

    # measurement uncertainty; checked by the solver (see Solver)
    sigma = p.get("sigma", None)
    sigma_rel = p.get("sigma_rel", None)
    sol = Solver(model=model)
    _, sim = sol.differential_evolution(
        maxiter=maxiter,
        popsize=popsize,
        mutation=mutation,  # type: ignore[arg-type]
        recombination=recombination,
        tol=tol,
        sigma=sigma,
        sigma_rel=sigma_rel,
    )
    return {
        "solver": "de",
        "sim": sim,
        "envelope": None,
        "meta": {"name": "Differential Evolution"},
    }


def _run_lsq(model: Model, params: Dict[str, Any] | None = None) -> Dict[str, object]:
    """Run Least Squares and return a standardized payload for the GUI."""
    p = params or {}
    ftol = float(p.get("ftol", 1e-8))
    xtol = float(p.get("xtol", 1e-8))
    gtol = float(p.get("gtol", 1e-8))
    max_nfev = int(p.get("max_nfev", 10000))

    # measurement uncertainty; checked by the solver (see Solver)
    sigma = p.get("sigma", None)
    sigma_rel = p.get("sigma_rel", None)
    sol = Solver(model=model)
    _, sim = sol.least_squares(
        ftol=ftol,
        xtol=xtol,
        gtol=gtol,
        max_nfev=max_nfev,
        sigma=sigma,
        sigma_rel=sigma_rel,
    )
    return {
        "solver": "lsq",
        "sim": sim,
        "envelope": None,
        "meta": {"name": "Least Squares"},
    }


def _run_mcmc(model: Model, params: Dict[str, Any] | None = None) -> Dict[str, object]:
    """Run MCMC and return standardized payload including percentiles and MAP simulation.

    Notes
    -----
    - Uses moderate defaults; can be exposed in GUI later.
    - Always returns simulations in the payload.
    """
    p = params or {}
    n_samples = int(p.get("n_samples", 1000))
    burn_in = int(p.get("burn_in", 2000))
    thin = int(p.get("thin", 1))
    rw_scale = float(p.get("rw_scale", 0.05))
    # measurement uncertainty; checked by the solver (see Solver)
    sigma = p.get("sigma", None)
    sigma_rel = p.get("sigma_rel", None)
    sol = Solver(model=model)
    res = sol.mcmc_sample(
        n_samples=n_samples,
        burn_in=burn_in,
        thin=thin,
        rw_scale=rw_scale,
        sigma=sigma,
        sigma_rel=sigma_rel,
        return_sim=True,
        set_model_state=True,
    )

    sims = res.get("sims")
    # the ensemble median of all simulations is not equal to the simulation
    # at the posterior median; we therefore do not use the posterior median
    # keys = res["param_names"]
    # theta_median = np.array([res["posterior_median"][k] for k in keys], dtype=float)

    if sims is None or sims.size == 0:
        # Fallback: don't return envelope
        median_sim = None
        env_1_99 = None
        env_20_80 = None
    else:
        median_sim = np.percentile(sims, 50, axis=0)
        p_low, p_high = np.percentile(sims, [1, 99], axis=0)
        env_1_99 = {"low": p_low, "high": p_high}
        p_low, p_high = np.percentile(sims, [20, 80], axis=0)
        env_20_80 = {"low": p_low, "high": p_high}

    # pass uncertainty estimates to the model object
    model.param_uncert = {}
    for num, name in enumerate(res["param_names"]):
        model.param_uncert[name] = [
            np.quantile(res["samples"][:, num], 0.01),
            np.quantile(res["samples"][:, num], 0.5),
            np.quantile(res["samples"][:, num], 0.99),
        ]
    model.param_map = res["posterior_map"]

    # plot chains to inspect convergence
    fig, ax = plt.subplots(
        res["samples"].shape[1],
        2,
        figsize=(6, 1.5 * res["samples"].shape[1]),
        sharey="row",
        gridspec_kw={"width_ratios": [3, 1]},
    )

    # ensure ax is 2D, even if we just have one parameter
    ax = np.atleast_2d(ax)

    for i in range(res["samples"].shape[1]):
        ax[i, 0].plot(res["samples"][:, i], c="k", lw=0.5, alpha=0.4)
        ax[i, 1].hist(
            res["samples"][:, i], orientation="horizontal", bins=30, density=True, color="grey"
        )
        ax[i, 0].set_ylabel(model.param_keys(free_only=True)[i])
        # set x axis limits
        ax[i, 0].set_xlim(0, res["samples"].shape[0])
        if i < res["samples"].shape[1] - 1:
            ax[i, 0].set_xticklabels([])
            ax[i, 1].set_xticklabels([])
        if i == 0:
            ax[i, 0].set_title("Chain")
            ax[i, 1].set_title("Histograms")
    ax[-1, 0].set_xlabel("Steps")
    ax[-1, 1].set_xlabel("Density")

    plt.tight_layout()
    plt.show()

    return {
        "solver": "mcmc",
        "sim": median_sim,
        "envelope_1_99": env_1_99,
        "envelope_20_80": env_20_80,
        "meta": {
            "name": "MCMC",
            "posterior_median": res.get("posterior_median", {}),
            "accept_rate": res.get("accept_rate"),
        },
    }


def _run_dream(model: Model, params: Dict[str, Any] | None = None) -> Dict[str, object]:
    """Run DREAM and return standardized payload including percentiles and MAP simulation."""
    p = params or {}
    n_samples = int(p.get("n_samples", 1000))
    n_chains = p.get("n_chains", None)
    n_chains = int(n_chains) if n_chains is not None else None
    burn_in = int(p.get("burn_in", 2000))
    thin = int(p.get("thin", 1))
    n_diff_pairs = int(p.get("n_diff_pairs", 1))
    n_cr = p.get("n_cr", 4)
    cr = [i / n_cr for i in range(1, n_cr + 1)]
    gamma = p.get("gamma", None)
    gamma = float(gamma) if gamma is not None else None
    gamma_jitter = float(p.get("gamma_jitter", 0.1))
    jitter = float(p.get("jitter", 1e-6))

    # measurement uncertainty; checked by the solver (see Solver)
    sigma = p.get("sigma", None)
    sigma_rel = p.get("sigma_rel", None)

    sol = Solver(model=model)
    res = sol.dream_sample(
        n_samples=n_samples,
        n_chains=n_chains,
        burn_in=burn_in,
        thin=thin,
        n_diff_pairs=n_diff_pairs,
        cr=cr,
        gamma=gamma,
        gamma_jitter=gamma_jitter,
        jitter=jitter,
        sigma=sigma,
        sigma_rel=sigma_rel,
        return_sim=True,
        set_model_state=True,
    )

    sims = res.get("sims")
    if sims is None or np.asarray(sims).size == 0:
        median_sim = None
        env_1_99 = None
        env_20_80 = None
    else:
        sims_arr = np.asarray(sims, dtype=float)
        sims_flat = sims_arr.reshape((-1,) + sims_arr.shape[2:])
        median_sim = np.percentile(sims_flat, 50, axis=0)
        p_low, p_high = np.percentile(sims_flat, [1, 99], axis=0)
        env_1_99 = {"low": p_low, "high": p_high}
        p_low, p_high = np.percentile(sims_flat, [20, 80], axis=0)
        env_20_80 = {"low": p_low, "high": p_high}

    # pass uncertainty estimates to the model object
    model.param_uncert = {}
    for num, name in enumerate(res["param_names"]):
        model.param_uncert[name] = [
            np.quantile(res["samples"][:, num], 0.01),
            np.quantile(res["samples"][:, num], 0.5),
            np.quantile(res["samples"][:, num], 0.99),
        ]
    model.param_map = res["posterior_map"]

    # plot chains to inspect convergence
    fig, ax = plt.subplots(
        res["samples_chain"].shape[2],
        2,
        figsize=(6, 1.5 * res["samples_chain"].shape[2]),
        sharey="row",
        gridspec_kw={"width_ratios": [3, 1]},
    )
    # ensure ax is 2D, even if we just have one parameter
    ax = np.atleast_2d(ax)

    bbox = dict(boxstyle="round", fc="white", ec="k", alpha=0.5)

    for i in range(res["samples_chain"].shape[2]):
        param_name = res["param_names"][i]
        if "mtt" in param_name:
            # conversion factor for travel time parameters to years
            factor = 12.0
        else:
            factor = 1.0
        for j in range(res["samples_chain"].shape[0]):
            ax[i, 0].plot(res["samples_chain"][j, :, i] / factor, c="k", lw=0.5, alpha=0.4)
            ax[i, 1].hist(
                res["samples"][:, i] / factor,
                orientation="horizontal",
                bins=30,
                density=True,
                color="grey",
            )
        ax[i, 0].set_ylabel(model.param_keys(free_only=True)[i])
        ax[i, 0].text(
            0.02,
            0.75,
            f"R={list(res['gelman_rubin'].values())[i]:1.3f}",
            bbox=bbox,
            transform=ax[i, 0].transAxes,
        )
        # set x axis limits
        ax[i, 0].set_xlim(0, res["samples_chain"].shape[1])
        if i < res["samples_chain"].shape[2] - 1:
            ax[i, 0].set_xticklabels([])
            ax[i, 1].set_xticklabels([])
        if i == 0:
            ax[i, 0].set_title("Chains")
            ax[i, 1].set_title("Histograms\n(all Chains)")
    ax[-1, 0].set_xlabel("Steps")
    ax[-1, 1].set_xlabel("Density")

    plt.tight_layout()
    plt.show()

    return {
        "solver": "dream",
        "sim": median_sim,
        "envelope_1_99": env_1_99,
        "envelope_20_80": env_20_80,
        "meta": {
            "name": "DREAM",
            "posterior_median": res.get("posterior_median", {}),
            "accept_rate": res.get("accept_rate"),
            "n_chains": int(res.get("samples_chain", np.empty((0,))).shape[0]),
        },
    }


# Add run-methods to solver registry
SOLVER_REGISTRY["de"]["run"] = _run_de
SOLVER_REGISTRY["lsq"]["run"] = _run_lsq
SOLVER_REGISTRY["mcmc"]["run"] = _run_mcmc
SOLVER_REGISTRY["dream"]["run"] = _run_dream


def available_solvers() -> List[Tuple[str, str]]:
    """List available solvers.

    Returns
    -------
    list of (str, str)
        Pairs of registry key and human-readable display name.
    """
    return [(k, v["name"]) for k, v in SOLVER_REGISTRY.items()]


def run_solver(model: Model, key: str, params: Dict[str, Any] | None = None) -> Dict[str, object]:
    """Run a registered solver and return a standardized payload.

    Parameters
    ----------
    model : PyTracerLab.model.model.Model
        Model instance with inputs, targets, and parameter registry.
    key : str
        Solver registry key (e.g., ``"de"`` or ``"mcmc"``).
    params : dict, optional
        Solver-specific keyword parameters.

    Returns
    -------
    dict
        Standardized payload consumed by the GUI, including at minimum
        the simulated series under the ``"sim"`` key; for MCMC also
        uncertainty envelopes and metadata.

    Raises
    ------
    ValueError
        If the solver key is not registered.
    """
    rec = SOLVER_REGISTRY.get(key)
    if rec is None:
        raise ValueError(f"Unknown solver key: {key}")
    runner: Callable[..., Dict[str, object]] = rec["run"]  # type: ignore[assignment]
    return runner(model, params or {})
