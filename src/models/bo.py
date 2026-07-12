"""Bayesian optimisation over a discrete hyperparameter grid.

GridBO: GP-backed Bayesian optimisation with EI, UCB, PI, and a piBO-style
prior-weighted acquisition. Supports prior-mean warm-start (e.g., from LRTC /
PMF predictions) and an optional LRTC "prior score" used by the piBO variant.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF, Matern, DotProduct, ConstantKernel, WhiteKernel,
)


# ── Kernel factory ──────────────────────────────────────────────────

def make_kernel(name: str, n_dims: int,
                length_scale_bounds: tuple = (1e-2, 1e2)):
    """Return a GP kernel by name.

    Supported: rbf_ard, rbf_iso, matern32, matern52, dotproduct

    length_scale_bounds defaults to (1e-2, 1e2) rather than (1e-3, 1e3):
    with very few observations the wider bounds let the optimiser drive
    length-scales to pathological extremes (≈0 → interpolation spikes,
    ≈∞ → constant), which wrecks the EI signal. The tighter range keeps
    the GP well-behaved in the small-budget regime that this paper targets.
    """
    name = name.lower()
    lsb = length_scale_bounds
    if name == "rbf_ard":
        return ConstantKernel(1.0, (1e-3, 1e3)) * RBF([1.0] * n_dims, lsb)
    if name == "rbf_iso":
        return ConstantKernel(1.0, (1e-3, 1e3)) * RBF(1.0, lsb)
    if name in ("matern32", "matern32_ard"):
        return ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
            [1.0] * n_dims, nu=1.5, length_scale_bounds=lsb)
    if name in ("matern52", "matern52_ard"):
        return ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
            [1.0] * n_dims, nu=2.5, length_scale_bounds=lsb)
    if name == "dotproduct":
        return ConstantKernel(1.0, (1e-3, 1e3)) * DotProduct(
            sigma_0=1.0, sigma_0_bounds=(1e-3, 1e3))
    raise ValueError(f"Unknown kernel: {name}")


# ── Coordinate reconstruction ───────────────────────────────────────

def get_hp_coords(axes: list[np.ndarray], grid_shape: tuple) -> np.ndarray:
    """Reconstruct (n_cells, n_hp) coordinate matrix from per-axis values."""
    grids = np.meshgrid(*axes, indexing="ij")
    return np.column_stack([g.ravel() for g in grids]).astype(np.float64)


# ── GridBO class ────────────────────────────────────────────────────

class GridBO:
    """Bayesian optimisation over a discrete HP grid.

    GP models *residuals* from an optional prior mean, so warm-start
    predictions (e.g., calibrated LRTC output) can be injected.

    prior_score (optional): a per-cell scalar (e.g., the z-scored LRTC
    prediction) used only by the piBO acquisition to bias the search
    toward historically promising cells, with influence that decays as
    observations accumulate.
    """

    def __init__(
        self,
        hp_coords: np.ndarray,
        prior_mean: np.ndarray | None = None,
        prior_score: np.ndarray | None = None,
        kernel_name: str = "rbf_ard",
        alpha: float = 1e-6,
        random_state: int | None = None,
        n_restarts: int = 5,
        length_scale_bounds: tuple = (1e-2, 1e2),
    ):
        self.n_cells, self.n_dims = hp_coords.shape

        hp_min = hp_coords.min(axis=0)
        hp_max = hp_coords.max(axis=0)
        rng_ = hp_max - hp_min
        rng_[rng_ == 0.0] = 1.0
        self._X_norm = (hp_coords - hp_min) / rng_

        self.prior_mean = (
            np.asarray(prior_mean, dtype=np.float64)
            if prior_mean is not None else None)
        # Prior score standardised to ~unit scale so the piBO decay
        # exponent behaves consistently across scenarios.
        if prior_score is not None:
            ps = np.asarray(prior_score, dtype=np.float64)
            sd = ps.std()
            self.prior_score = (ps - ps.mean()) / (sd if sd > 1e-12 else 1.0)
        else:
            self.prior_score = None

        self.kernel_name = kernel_name
        self.alpha = alpha
        self.random_state = random_state
        self.n_restarts = n_restarts
        self.length_scale_bounds = length_scale_bounds

        self._obs_idx: list[int] = []
        self._obs_raw: list[float] = []
        self.y_best: float = -np.inf
        self.gp: GaussianProcessRegressor | None = None

    # ── observation bookkeeping ─────────────────────────────────

    def add_observations(self, flat_indices, raw_values):
        indices = np.atleast_1d(np.asarray(flat_indices, dtype=int))
        values = np.atleast_1d(np.asarray(raw_values, dtype=float))
        self._obs_idx.extend(indices.tolist())
        self._obs_raw.extend(values.tolist())
        self.y_best = max(self.y_best, float(values.max()))

    def set_prior_mean(self, prior_mean: np.ndarray | None):
        """Replace the prior mean (used by the refit-once variant)."""
        self.prior_mean = (
            np.asarray(prior_mean, dtype=np.float64)
            if prior_mean is not None else None)

    @property
    def n_observed(self) -> int:
        return len(self._obs_idx)

    @property
    def observed_mask(self) -> np.ndarray:
        m = np.zeros(self.n_cells, dtype=bool)
        if self._obs_idx:
            m[self._obs_idx] = True
        return m

    # ── GP fitting ──────────────────────────────────────────────

    def fit(self) -> "GridBO":
        if self.n_observed < 2:
            return self
        idx_arr = np.array(self._obs_idx, dtype=int)
        X_obs = self._X_norm[idx_arr]
        y_obs = np.array(self._obs_raw, dtype=np.float64)
        if self.prior_mean is not None:
            y_target = y_obs - self.prior_mean[idx_arr]
        else:
            y_target = y_obs
        kernel = make_kernel(self.kernel_name, self.n_dims,
                             self.length_scale_bounds)
        self.gp = GaussianProcessRegressor(
            kernel=kernel, alpha=self.alpha, normalize_y=False,
            n_restarts_optimizer=self.n_restarts,
            random_state=self.random_state)
        self.gp.fit(X_obs, y_target)
        return self

    # ── prediction ──────────────────────────────────────────────

    def predict(self, flat_indices=None):
        if flat_indices is None:
            flat_indices = np.arange(self.n_cells)
        flat_indices = np.atleast_1d(np.asarray(flat_indices, dtype=int))
        X = self._X_norm[flat_indices]

        if self.gp is None or self.n_observed < 2:
            pm = (self.prior_mean[flat_indices].copy()
                  if self.prior_mean is not None
                  else np.zeros(len(flat_indices)))
            return pm, np.full(len(flat_indices), 0.5)

        mu_res, sigma = self.gp.predict(X, return_std=True)
        if self.prior_mean is not None:
            mu = mu_res + self.prior_mean[flat_indices]
        else:
            mu = mu_res
        return mu, np.maximum(sigma, 1e-12)

    # ── acquisition functions ───────────────────────────────────

    def expected_improvement(self, flat_indices=None, xi: float = 0.0):
        mu, sigma = self.predict(flat_indices)
        imp = mu - self.y_best - xi
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma < 1e-12] = 0.0
        return ei

    def probability_of_improvement(self, flat_indices=None, xi: float = 0.0):
        mu, sigma = self.predict(flat_indices)
        Z = (mu - self.y_best - xi) / sigma
        return norm.cdf(Z)

    def upper_confidence_bound(self, flat_indices=None, kappa: float = 2.0):
        mu, sigma = self.predict(flat_indices)
        return mu + kappa * sigma

    def pibo_score(self, flat_indices=None, xi: float = 0.0,
                   beta: float = 10.0):
        """piBO-style prior-weighted acquisition (Hvarfner et al. 2022).

        log-EI plus a prior log-density term whose weight decays as
        beta / t (t = number of observations). Early on the LRTC prior
        steers the search; as evidence accrues the GP's EI dominates, so
        a wrong prior self-corrects instead of biasing the search forever.
        """
        ei = self.expected_improvement(flat_indices, xi=xi)
        log_ei = np.log(ei + 1e-12)
        if self.prior_score is None:
            return log_ei
        if flat_indices is None:
            ps = self.prior_score
        else:
            ps = self.prior_score[np.atleast_1d(
                np.asarray(flat_indices, dtype=int))]
        t = max(1, self.n_observed)
        return log_ei + (beta / t) * ps

    # ── selection ───────────────────────────────────────────────

    def select_next(self, unobserved_mask: np.ndarray | None = None,
                    acquisition: str = "ei", **acq_kwargs):
        # NB: the array passed in is the *observed* mask; candidates are
        # the cells NOT yet observed (~mask). Name kept for compatibility.
        if unobserved_mask is None:
            unobserved_mask = self.observed_mask
        else:
            unobserved_mask = np.asarray(unobserved_mask, dtype=bool)
        candidates = np.where(~unobserved_mask)[0]
        if len(candidates) == 0:
            raise RuntimeError("No unobserved cells left")

        acq = acquisition.lower()
        if acq == "ucb":
            scores = self.upper_confidence_bound(
                candidates, kappa=acq_kwargs.get("kappa", 2.0))
        elif acq == "pi":
            scores = self.probability_of_improvement(
                candidates, xi=acq_kwargs.get("xi", 0.0))
        elif acq == "pibo":
            scores = self.pibo_score(
                candidates, xi=acq_kwargs.get("xi", 0.0),
                beta=acq_kwargs.get("pibo_beta", 10.0))
        else:
            scores = self.expected_improvement(
                candidates, xi=acq_kwargs.get("xi", 0.0))
        return int(candidates[np.argmax(scores)])
