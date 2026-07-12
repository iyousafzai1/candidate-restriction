"""Additional baseline methods.

- run_tpe_unit : Tree-structured Parzen Estimator (Bergstra et al. 2011),
  a faithful self-contained implementation over the real (normalised) grid
  coordinates — no hyperopt dependency, respects HP ordering.
- run_rgpe_unit: RGPE — Ranking-weighted GP Ensemble transfer BO
  (Feurer, Letham & Bakshy, 2018), the standard meta-learning BO baseline.
"""

from __future__ import annotations

import time
import numpy as np
from scipy.stats import norm
from scipy.linalg import cho_solve
from sklearn.gaussian_process import GaussianProcessRegressor

from prep import zscore_observed


# ── TPE: Tree-structured Parzen Estimator (Bergstra et al. 2011) ─────

def _normalised_coords(axes_list, grid_shape):
    """(n_cells, n_hp) coordinates in [0,1] per HP, ordinal order preserved.

    Uses the rank position of each axis value (not the raw value) so that
    log-spaced axes are evenly spread — TPE's Parzen windows then behave
    consistently across linear and log hyperparameters.
    """
    ranks = []
    for a in axes_list:
        n = len(a)
        ranks.append(np.arange(n) / max(1, n - 1))
    grids = np.meshgrid(*ranks, indexing="ij")
    return np.column_stack([g.ravel() for g in grids]).astype(np.float64)


def _parzen_log_density(centers, bw, Xq):
    """log of a factorised Gaussian Parzen mixture.

    centers: (m, d) kernel centres; bw: (d,) per-dim bandwidth; Xq: (q, d).
    Independent across dims (TPE's factorisation); returns (q,) log-density.
    """
    m, d = centers.shape
    log_p = np.zeros(len(Xq))
    norm_const = np.log(m) + 0.5 * np.log(2 * np.pi)
    for j in range(d):
        # (q, m) squared distances on dim j
        diff = (Xq[:, j][:, None] - centers[:, j][None, :]) / bw[j]
        logk = -0.5 * diff ** 2 - np.log(bw[j]) - 0.5 * np.log(2 * np.pi)
        # log mean over centres
        log_p += np.logaddexp.reduce(logk, axis=1) - np.log(m)
    return log_p


def _tpe_bandwidth(X, d):
    """Silverman per-dim bandwidth with a floor (coords live in [0,1])."""
    m = len(X)
    sd = X.std(axis=0)
    bw = 1.06 * np.maximum(sd, 1e-3) * (m ** (-1.0 / 5.0))
    return np.maximum(bw, 0.10)


def run_tpe_unit(problem, seed: int, tpe_cfg: dict, budgets: list[int],
                 max_budget: int, axes=None) -> dict:
    """One TPE unit: k0 anchors + random startup → TPE acquisition to budget.

    Faithful TPE: split observations into good/bad by a γ-quantile of the
    objective, fit factorised Parzen densities l(x) (good) and g(x) (bad)
    over the normalised grid coordinates, and pick the unobserved cell that
    maximises l(x)/g(x) (the TPE Expected-Improvement surrogate). A prior
    kernel centred mid-domain regularises the densities. Search without
    replacement so the budget counts distinct configurations.
    """
    t0 = time.time()
    k0 = len(problem.anchors)
    rng = np.random.default_rng(seed)
    anchors = problem.anchors
    gamma = float(tpe_cfg.get("gamma", 0.25))
    n_startup = int(tpe_cfg.get("n_startup_jobs", max(k0, 5)))
    n_ei = int(tpe_cfg.get("n_ei_candidates", 2000))

    # Normalised ordinal coordinates for every grid cell.
    if axes is None:
        coords = _normalised_coords(
            [np.arange(s) for s in problem.grid_shape], problem.grid_shape)
    else:
        axes_list = (list(axes.values()) if isinstance(axes, dict)
                     else [np.asarray(a) for a in axes])
        coords = _normalised_coords(axes_list, problem.grid_shape)
    d = coords.shape[1]
    prior_centre = np.full((1, d), 0.5)
    prior_bw = np.ones(d)

    observed_idx = list(anchors[:max_budget])
    observed_mask = np.zeros(problem.n_cells, dtype=bool)
    observed_mask[observed_idx] = True
    if len(observed_idx) < n_startup:
        remaining = np.where(~observed_mask)[0]
        n_extra = min(n_startup - len(observed_idx), len(remaining))
        extra = rng.choice(remaining, size=n_extra, replace=False).tolist()
        observed_idx.extend(extra)
        observed_mask[extra] = True
    observed_vals = [float(v) for v in problem.evaluate(observed_idx)]

    step = len(observed_idx)
    while step < max_budget:
        vals = np.asarray(observed_vals)
        X_obs = coords[np.asarray(observed_idx, dtype=int)]
        # good = top γ by accuracy (TPE minimises loss → here maximise value)
        thr = np.quantile(vals, 1.0 - gamma)
        good = vals >= thr
        if good.sum() < 1 or (~good).sum() < 1:
            order = np.argsort(-vals)
            n_good = max(1, int(np.ceil(gamma * len(vals))))
            good = np.zeros(len(vals), bool)
            good[order[:n_good]] = True

        Xg, Xb = X_obs[good], X_obs[~good]
        # augment with the prior kernel (regularisation, à la hyperopt)
        Cg = np.vstack([Xg, prior_centre])
        Cb = np.vstack([Xb, prior_centre])
        bwg = np.minimum(_tpe_bandwidth(Xg, d), prior_bw) if len(Xg) > 1 else prior_bw
        bwb = np.minimum(_tpe_bandwidth(Xb, d), prior_bw) if len(Xb) > 1 else prior_bw

        cand = np.where(~observed_mask)[0]
        if len(cand) > n_ei:
            cand = rng.choice(cand, size=n_ei, replace=False)
        Xc = coords[cand]
        log_l = _parzen_log_density(Cg, bwg, Xc)
        log_g = _parzen_log_density(Cb, bwb, Xc)
        nxt = int(cand[np.argmax(log_l - log_g)])

        val = float(problem.evaluate([nxt])[0])
        observed_idx.append(nxt)
        observed_vals.append(val)
        observed_mask[nxt] = True
        step += 1

    vals_arr = np.asarray(observed_vals)
    checkpoints = []
    for b in budgets:
        bb = min(b, len(vals_arr))
        best = float(np.max(vals_arr[:bb]))
        checkpoints.append({
            "budget": int(b), "n_evaluated": int(bb),
            "best_found": best,
            "normalized_regret": float(problem.normalized_regret(best)),
        })

    return {
        "method": "tpe",
        "seed": int(seed), "k0": int(k0),
        "checkpoints": checkpoints,
        "eval_order": [int(i) for i in observed_idx],
        "anchors": anchors.tolist(),
        "y_best_true": problem.y_best,
        "y_worst_true": problem.y_worst,
        "runtime_seconds": round(time.time() - t0, 3),
        "tpe_cfg": {"gamma": gamma, "n_startup": n_startup},
    }


# ── RGPE: Ranking-weighted GP Ensemble (Feurer et al. 2018) ──────────

def _ranking_loss(pred: np.ndarray, y: np.ndarray) -> int:
    """Number of misordered pairs between predicted and true values."""
    P = np.sign(pred[:, None] - pred[None, :])
    Y = np.sign(y[:, None] - y[None, :])
    return int(np.triu(P != Y, 1).sum())


def _rgpe_weights(base_pred_obs: np.ndarray, gp, X_obs: np.ndarray,
                  y_obs: np.ndarray, S: int, rng) -> np.ndarray:
    """RGPE Monte-Carlo ranking weights for [base_1..base_m, target].

    Base models are the (deterministic, fully-observed) historical task
    slices; the target model is the GP fit on the new task. The target's
    ranking loss is computed from *leave-one-out* posterior draws (closed
    form via K^{-1}) so the target does not trivially win by interpolation.
    """
    m = base_pred_obs.shape[0]
    n = len(y_obs)
    losses = np.zeros((m + 1, S))
    for i in range(m):
        losses[i, :] = _ranking_loss(base_pred_obs[i], y_obs)

    # Closed-form LOO posterior for the target GP (Rasmussen & Williams 5.12).
    try:
        Kinv = cho_solve((gp.L_, True), np.eye(n))
        alpha = gp.alpha_.ravel()                      # K^{-1} y_centered
        y_mean = gp._y_train_mean if hasattr(gp, "_y_train_mean") else 0.0
        diag = np.clip(np.diag(Kinv), 1e-12, None)
        loo_mean = y_obs - alpha / diag                # already in y units
        loo_var = np.clip(1.0 / diag, 1e-12, None)
        for s in range(S):
            sample = loo_mean + np.sqrt(loo_var) * rng.standard_normal(n)
            losses[m, s] = _ranking_loss(sample, y_obs)
    except Exception:
        losses[m, :] = losses[:m, :].mean() if m else 0.0

    # Weight = fraction of MC draws in which a model has the lowest loss
    # (random tie-break), per the RGPE rule.
    best = np.zeros(m + 1)
    for s in range(S):
        col = losses[:, s]
        winners = np.where(col == col.min())[0]
        best[rng.choice(winners)] += 1.0
    w = best / S
    if w.sum() <= 0:
        w = np.ones(m + 1) / (m + 1)
    return w


def run_rgpe_unit(problem, seed: int, exp_cfg: dict, axes, budgets,
                  max_budget: int) -> dict:
    """RGPE transfer BO over the discrete grid (all values in z-space)."""
    from models.bo import get_hp_coords, make_kernel

    t0 = time.time()
    rng = np.random.default_rng(seed)
    bo_cfg = exp_cfg.get("bo", {})
    rgpe_cfg = exp_cfg.get("rgpe", {})
    S = int(rgpe_cfg.get("n_samples", 50))
    kernel_name = bo_cfg.get("kernel", "rbf_ard")
    alpha = float(bo_cfg.get("alpha", 1e-6))
    n_restarts = int(bo_cfg.get("n_restarts", 5))
    lsb = tuple(bo_cfg.get("length_scale_bounds", (1e-2, 1e2)))
    xi = float(bo_cfg.get("xi", 0.0))

    # Grid coordinates (normalised per dim), and base-task means in z-space.
    if isinstance(axes, dict):
        axis_names = exp_cfg.get("_axis_names", sorted(axes.keys()))
        axes_list = [np.asarray(axes[n], dtype=np.float64)
                     for n in axis_names if n in axes]
    else:
        axes_list = [np.asarray(a, dtype=np.float64) for a in axes]
    hp_coords = get_hp_coords(axes_list, problem.grid_shape)
    hp_min, hp_max = hp_coords.min(0), hp_coords.max(0)
    rng_ = hp_max - hp_min
    rng_[rng_ == 0.0] = 1.0
    X_norm = (hp_coords - hp_min) / rng_

    n_train = problem.train_z.shape[-1]
    base = problem.train_z.reshape(-1, n_train).T          # (n_train, n_cells)

    anchors = problem.anchors
    observed_idx = list(anchors[:max_budget])
    observed_raw = [float(v) for v in problem.evaluate(observed_idx)]
    observed_mask = np.zeros(problem.n_cells, dtype=bool)
    observed_mask[observed_idx] = True

    step = len(observed_idx)
    while step < max_budget:
        obs = np.asarray(observed_idx, dtype=int)
        raw = np.asarray(observed_raw, dtype=float)
        mu_t, sd_t = zscore_observed(raw)
        y_z = (raw - mu_t) / sd_t                           # target in z-space

        gp = GaussianProcessRegressor(
            kernel=make_kernel(kernel_name, X_norm.shape[1], lsb),
            alpha=alpha, normalize_y=False,
            n_restarts_optimizer=n_restarts, random_state=seed)
        gp.fit(X_norm[obs], y_z)

        w = _rgpe_weights(base[:, obs], gp, X_norm[obs], y_z, S, rng)

        gp_mu, gp_sd = gp.predict(X_norm, return_std=True)
        combined_mu = w[-1] * gp_mu + base.T @ w[:-1]       # (n_cells,)
        combined_sd = np.maximum(gp_sd, 1e-12)              # target drives EI

        y_best_z = float(y_z.max())
        imp = combined_mu - y_best_z - xi
        Z = imp / combined_sd
        ei = imp * norm.cdf(Z) + combined_sd * norm.pdf(Z)
        ei[observed_mask] = -np.inf
        nxt = int(np.argmax(ei))

        val = float(problem.evaluate([nxt])[0])
        observed_idx.append(nxt)
        observed_raw.append(val)
        observed_mask[nxt] = True
        step += 1

    vals = np.asarray(observed_raw)
    checkpoints = []
    for b in budgets:
        bb = min(b, len(vals))
        best = float(np.max(vals[:bb]))
        checkpoints.append({"budget": int(b), "n_evaluated": int(bb),
                            "best_found": best,
                            "normalized_regret": float(problem.normalized_regret(best))})

    return {
        "method": "rgpe", "seed": int(seed), "k0": int(len(anchors)),
        "checkpoints": checkpoints,
        "eval_order": [int(i) for i in observed_idx],
        "anchors": anchors.tolist(),
        "y_best_true": problem.y_best, "y_worst_true": problem.y_worst,
        "runtime_seconds": round(time.time() - t0, 3),
        "rgpe_cfg": {"n_samples": S, "n_base": int(n_train)},
    }
