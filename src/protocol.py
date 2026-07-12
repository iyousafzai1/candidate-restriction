"""Paper D protocol: single dispatch for all methods.

Methods:
  rs, lrtc_oneshot           — passive baselines
  bo_cold                     — cold-start GP-EI
  bo_lrtc_prior, bo_pmf_prior — warm-start BO with prior mean
  bo_lrtc_portfolio           — top-K pre-evaluation + cold BO
  bo_lrtc_hybrid              — prior mean + top-K pre-evaluation
  active_pmf                  — Thompson sampling PMF ensemble
  tpe                         — Tree-structured Parzen Estimator
"""

from __future__ import annotations

import time
import numpy as np

from prep import Problem, zscore_observed
from models.pmf import pmf_complete_test_row
from models.lrtc import lrtc_complete_test_slice

BO_METHODS = ("bo_cold", "bo_lrtc_prior", "bo_pmf_prior",
              "bo_lrtc_portfolio", "bo_lrtc_hybrid",
              "bo_lrtc_pibo", "bo_lrtc_feature", "bo_lrtc_refit")

# Which warm-start family supplies the prior scores for each BO method.
_LRTC_BO = ("bo_lrtc_prior", "bo_lrtc_portfolio", "bo_lrtc_hybrid",
            "bo_lrtc_pibo", "bo_lrtc_feature", "bo_lrtc_refit")

ALL_METHODS = (
    "rs", "lrtc_oneshot", "bo_cold", "active_pmf", "tpe", "rgpe",
    "bo_lrtc_prior", "bo_lrtc_portfolio", "bo_lrtc_hybrid",
    "bo_lrtc_pibo", "bo_lrtc_feature", "bo_lrtc_refit",
    "bo_pmf_prior",
)


# ── helpers ─────────────────────────────────────────────────────

def observe_anchors(problem: Problem):
    anchors = problem.anchors
    raw_obs = problem.evaluate(anchors)
    mu, sd = zscore_observed(raw_obs)
    mask = np.zeros(problem.n_cells, dtype=bool)
    mask[anchors] = True
    flat = np.zeros(problem.n_cells)
    flat[anchors] = (raw_obs - mu) / sd
    return raw_obs, flat, mask


def oneshot_scores(problem, model, mcfg, seed, test_flat, test_mask):
    if model == "pmf":
        train_block = problem.train_z.reshape(-1, problem.train_z.shape[-1]).T
        return pmf_complete_test_row(
            train_block, test_flat, test_mask,
            rank=mcfg["rank"], reg=mcfg["reg"],
            iters=mcfg["iters"], seed=seed)
    if model == "lrtc":
        pred = lrtc_complete_test_slice(
            problem.train_z, test_flat.reshape(problem.grid_shape),
            test_mask.reshape(problem.grid_shape),
            rank=mcfg["rank"], beta=mcfg["beta"],
            iters=mcfg["iters"], seed=seed)
        return pred.ravel()
    raise ValueError(f"unknown model '{model}'")


def _model_eval_order(problem, scores, max_budget):
    anchors = problem.anchors
    observed = np.zeros(problem.n_cells, dtype=bool)
    observed[anchors] = True
    cand = np.where(~observed)[0]
    ranked = cand[np.argsort(-scores[cand], kind="stable")]
    n_extra = max(0, max_budget - len(anchors))
    return np.concatenate([anchors, ranked[:n_extra]])


def _checkpoint_results(problem, eval_order, budgets):
    vals = problem.evaluate(eval_order)
    out = []
    for b in budgets:
        bb = min(b, len(eval_order))
        best = float(np.max(vals[:bb]))
        out.append({"budget": int(b), "n_evaluated": int(bb),
                    "best_found": best,
                    "normalized_regret": float(problem.normalized_regret(best))})
    return out


# ── dispatch ─────────────────────────────────────────────────────

def run_unit(method: str, problem: Problem, seed: int,
             exp_cfg: dict, axes: list | dict | None = None) -> dict:
    budgets = sorted(exp_cfg["budgets"])
    max_budget = min(budgets[-1], problem.n_cells)
    t0 = time.time()

    # --- rs ---
    if method == "rs":
        rng = np.random.default_rng(seed)
        order = rng.choice(problem.n_cells, size=max_budget, replace=False)
        extra = {}

    # --- lrtc_oneshot ---
    elif method == "lrtc_oneshot":
        raw_obs, test_flat, test_mask = observe_anchors(problem)
        scores = oneshot_scores(problem, "lrtc", exp_cfg["lrtc"], seed,
                                test_flat, test_mask)
        order = _model_eval_order(problem, scores, max_budget)
        extra = {"anchors": problem.anchors.tolist(),
                 "model_cfg": exp_cfg["lrtc"]}

    # --- active_pmf ---
    elif method == "active_pmf":
        from models.pmf import masked_als
        # Thompson sampling loop (reuses active.py logic inline)
        mcfg = exp_cfg.get("active_pmf", {})
        M = mcfg.get("M", 10)
        ranks = mcfg.get("ranks", [3, 5, 8])
        reg = mcfg.get("reg", 1.0)
        fit_iters = mcfg.get("fit_iters", 50)
        refit_iters = mcfg.get("refit_iters", 15)
        refit_every = mcfg.get("refit_every", 1)

        rng = np.random.default_rng(seed)
        n_train = problem.train_z.shape[-1]
        bf = mcfg.get("bootstrap_frac", 0.9)
        n_keep = max(1, int(round(bf * n_train)))

        # Build ensemble specs
        specs = []
        for m in range(M):
            subset = np.sort(rng.choice(n_train, size=n_keep, replace=False))
            specs.append({"rank": int(rng.choice(ranks)),
                          "subset": subset,
                          "member_seed": int(seed) * 1009 + 101 * m})

        # Active loop
        anchors = problem.anchors
        observed_idx = list(anchors[:max_budget])
        observed_vals = list(problem.evaluate(observed_idx))
        observed_mask = np.zeros(problem.n_cells, dtype=bool)
        observed_mask[observed_idx] = True
        scores_by_member = [None] * M
        idx_seq = np.random.default_rng(int(seed) * 7919 + 13).integers(0, M, size=max_budget)
        step = len(observed_idx)

        while step < max_budget:
            do_refit = (scores_by_member[0] is None
                        or (step - len(anchors)) % refit_every == 0)
            if do_refit:
                mu_z, sd_z = zscore_observed(np.asarray(observed_vals))
                test_flat = np.zeros(problem.n_cells)
                test_flat[observed_idx] = (np.asarray(observed_vals) - mu_z) / sd_z
                for m_idx in range(M):
                    s = specs[m_idx]
                    train_block = problem.train_z[..., s["subset"]].reshape(-1, n_keep).T
                    pred = pmf_complete_test_row(
                        train_block, test_flat, observed_mask,
                        rank=s["rank"], reg=reg,
                        iters=fit_iters if scores_by_member[0] is None else refit_iters,
                        seed=s["member_seed"])
                    scores_by_member[m_idx] = pred
            m_pick = int(idx_seq[step])
            cand = np.where(~observed_mask)[0]
            nxt = int(cand[np.argmax(scores_by_member[m_pick][cand])])
            val = float(problem.evaluate([nxt])[0])
            observed_idx.append(nxt)
            observed_vals.append(val)
            observed_mask[nxt] = True
            step += 1

        order = np.array(observed_idx)
        extra = {"anchors": anchors.tolist(), "ensemble_M": M}

    # --- tpe ---
    elif method == "tpe":
        from models.baselines import run_tpe_unit
        return run_tpe_unit(problem, seed, exp_cfg.get("tpe", {}),
                            budgets, max_budget, axes=axes)

    # --- rgpe (transfer-BO baseline) ---
    elif method == "rgpe":
        from models.baselines import run_rgpe_unit
        return run_rgpe_unit(problem, seed, exp_cfg, axes, budgets, max_budget)

    # --- BO methods ---
    elif method in BO_METHODS:
        return _run_bo_unit(method, problem, seed, exp_cfg, axes,
                            budgets, max_budget)

    else:
        raise ValueError(f"Unknown method '{method}'. Available: {ALL_METHODS}")

    return {
        "method": method, "seed": int(seed),
        "k0": int(len(problem.anchors)),
        "checkpoints": _checkpoint_results(problem, order, budgets),
        "eval_order": np.asarray(order).tolist(),
        "y_best_true": problem.y_best, "y_worst_true": problem.y_worst,
        "runtime_seconds": round(time.time() - t0, 3),
        **extra,
    }


# ── BO unit ───────────────────────────────────────────────────────

def _zspace_from_observed(problem, observed_idx, observed_raw):
    """Build a z-scored sparse test slice from arbitrary observations."""
    obs = np.asarray(observed_idx, dtype=int)
    raw = np.asarray(observed_raw, dtype=float)
    mu, sd = zscore_observed(raw)
    test_flat = np.zeros(problem.n_cells)
    test_mask = np.zeros(problem.n_cells, dtype=bool)
    test_flat[obs] = (raw - mu) / sd
    test_mask[obs] = True
    return test_flat, test_mask


def _prior_scores_z(problem, family, mcfg, seed, observed_idx, observed_raw):
    """z-space LRTC/PMF prediction over the full grid from given observations.

    Defensively sanitised: SNN-BCD/ALS can occasionally diverge to non-finite
    values on ill-conditioned slices; a NaN/Inf prior would silently corrupt
    the GP. Non-finite cells fall back to the observed grid mean (0 in z-space).
    """
    test_flat, test_mask = _zspace_from_observed(problem, observed_idx,
                                                 observed_raw)
    scores = oneshot_scores(problem, family, mcfg, seed, test_flat, test_mask)
    if not np.all(np.isfinite(scores)):
        finite = scores[np.isfinite(scores)]
        fill = float(finite.mean()) if finite.size else 0.0
        scores = np.where(np.isfinite(scores), scores, fill)
    return scores


def _corrupt_scores(scores_z, bo_cfg, seed):
    """Deliberately degrade the prior (misspecified-prior stress test)."""
    mode = bo_cfg.get("prior_corruption", "none")
    if mode in (None, "none"):
        return scores_z
    rng = np.random.default_rng(int(seed) * 104729 + 7)
    if mode == "shuffle":
        out = scores_z.copy()
        rng.shuffle(out)
        return out
    if mode == "noise":
        strength = float(bo_cfg.get("prior_corruption_strength", 1.0))
        sd = scores_z.std()
        sd = sd if sd > 1e-12 else 1.0
        return scores_z + strength * sd * rng.standard_normal(scores_z.shape)
    raise ValueError(f"unknown prior_corruption '{mode}'")


def _calibrate_prior(scores_z, observed_idx, observed_raw):
    """Map z-space scores to raw objective units via least squares a*z + b.

    Fit on the observed (anchor + portfolio) cells. This avoids the bias of
    rescaling by the top-k anchor mean/std (which inflates the level and
    compresses the spread); the linear fit instead calibrates the prior mean
    to the actual raw scale and stays well-defined as observations accrue.
    Degenerate case (near-constant z, e.g. all-top anchors): fall back to a
    shift/scale by the observed raw statistics.
    """
    obs = np.asarray(observed_idx, dtype=int)
    z = np.asarray(scores_z[obs], dtype=np.float64)
    y = np.asarray(observed_raw, dtype=np.float64)
    if len(z) < 2 or np.var(z) < 1e-10:
        sd = float(np.std(y))
        sd = sd if sd > 1e-12 else 1.0
        return scores_z * sd + float(np.mean(y))
    A = np.column_stack([z, np.ones_like(z)])
    # tiny ridge on the slope (not the intercept) for stability with k0=3.
    ATA = A.T @ A + np.array([[1e-6, 0.0], [0.0, 0.0]])
    a, b = np.linalg.solve(ATA, A.T @ y)
    return scores_z * a + b


def _run_bo_unit(method, problem, seed, exp_cfg, axes, budgets, max_budget):
    from models.bo import GridBO, get_hp_coords

    if axes is None:
        raise ValueError("BO methods require axes (grid HP values).")

    if isinstance(axes, dict):
        axis_names = exp_cfg.get("_axis_names", sorted(axes.keys()))
        axes_list = [np.asarray(axes[n], dtype=np.float64)
                     for n in axis_names if n in axes]
    else:
        axes_list = [np.asarray(a, dtype=np.float64) for a in axes]

    bo_cfg = exp_cfg.get("bo", {})
    kernel_name = bo_cfg.get("kernel", "rbf_ard")
    acq = bo_cfg.get("acquisition", "ei")
    alpha = float(bo_cfg.get("alpha", 1e-6))
    n_restarts = int(bo_cfg.get("n_restarts", 5))
    portfolio_size = int(bo_cfg.get("portfolio_size", 5))
    xi = float(bo_cfg.get("xi", 0.0))
    lsb = tuple(bo_cfg.get("length_scale_bounds", (1e-2, 1e2)))
    t0 = time.time()

    # piBO forces its own prior-weighted acquisition.
    if method == "bo_lrtc_pibo":
        acq = "pibo"
    acq_kwargs = {"xi": xi}
    if acq == "ucb":
        acq_kwargs["kappa"] = float(bo_cfg.get("ucb_kappa", 2.0))
    if acq == "pibo":
        acq_kwargs["pibo_beta"] = float(bo_cfg.get("pibo_beta", 10.0))

    hp_coords = get_hp_coords(axes_list, problem.grid_shape)
    raw_obs, _tf, _tm = observe_anchors(problem)
    anchors = problem.anchors

    # ── z-space prior scores (LRTC or PMF) from the anchors ──
    scores_z = None
    if method in _LRTC_BO:
        scores_z = _prior_scores_z(problem, "lrtc", exp_cfg["lrtc"], seed,
                                   anchors, raw_obs)
        scores_z = _corrupt_scores(scores_z, bo_cfg, seed)
    elif method == "bo_pmf_prior":
        scores_z = _prior_scores_z(problem, "pmf", exp_cfg["pmf"], seed,
                                   anchors, raw_obs)

    # ── portfolio pre-evaluation (ranked by z-space scores) ──
    uses_portfolio = method in ("bo_lrtc_portfolio", "bo_lrtc_hybrid")
    portfolio_idx, portfolio_vals = [], []
    if uses_portfolio:
        obs_init = np.zeros(problem.n_cells, dtype=bool)
        obs_init[anchors] = True
        cand = np.where(~obs_init)[0]
        ranked = cand[np.argsort(-scores_z[cand], kind="stable")]
        n_pf = min(portfolio_size, max_budget - len(anchors), len(ranked))
        portfolio_idx = ranked[:n_pf].tolist()
        portfolio_vals = [float(v) for v in problem.evaluate(portfolio_idx)]

    observed_idx = list(anchors) + portfolio_idx
    observed_raw = list(raw_obs) + portfolio_vals

    # ── prior mean: which variants use a calibrated GP mean ──
    uses_prior_mean = method in ("bo_lrtc_prior", "bo_lrtc_hybrid",
                                 "bo_lrtc_refit", "bo_pmf_prior")
    prior_mean = None
    if uses_prior_mean:
        prior_mean = _calibrate_prior(scores_z, observed_idx, observed_raw)

    # ── LRTC-as-feature: append the z-space score as an extra GP input ──
    if method == "bo_lrtc_feature":
        hp_coords = np.column_stack([hp_coords, scores_z])

    # ── piBO: the prior steers the acquisition, not the GP mean ──
    prior_score = scores_z if method == "bo_lrtc_pibo" else None

    bo = GridBO(hp_coords=hp_coords, prior_mean=prior_mean,
                prior_score=prior_score, kernel_name=kernel_name,
                alpha=alpha, random_state=seed, n_restarts=n_restarts,
                length_scale_bounds=lsb)
    bo.add_observations(anchors, raw_obs)
    if portfolio_idx:
        bo.add_observations(portfolio_idx, portfolio_vals)

    observed_mask = bo.observed_mask
    step = len(observed_idx)

    # refit-once schedule (E3): recompute & recalibrate the prior mean once.
    _rb = bo_cfg.get("refit_budget")
    refit_budget = int(_rb) if _rb is not None else max_budget // 2
    refit_done = (method != "bo_lrtc_refit")

    while step < max_budget:
        if (not refit_done) and step >= refit_budget:
            rescores = _prior_scores_z(problem, "lrtc", exp_cfg["lrtc"], seed,
                                       observed_idx, observed_raw)
            bo.set_prior_mean(_calibrate_prior(rescores, observed_idx,
                                               observed_raw))
            refit_done = True
        bo.fit()
        nxt = bo.select_next(observed_mask, acquisition=acq, **acq_kwargs)
        val = float(problem.evaluate([nxt])[0])
        bo.add_observations([nxt], [val])
        observed_idx.append(nxt)
        observed_raw.append(val)
        observed_mask[nxt] = True
        step += 1

    vals = np.asarray(problem.evaluate(observed_idx))
    checkpoints = []
    for b in budgets:
        bb = min(b, len(vals))
        best = float(np.max(vals[:bb]))
        checkpoints.append({"budget": int(b), "n_evaluated": int(bb),
                            "best_found": best,
                            "normalized_regret": float(problem.normalized_regret(best))})

    return {
        "method": method, "seed": int(seed), "k0": int(len(anchors)),
        "checkpoints": checkpoints,
        "eval_order": [int(i) for i in observed_idx],
        "anchors": anchors.tolist(),
        "y_best_true": problem.y_best, "y_worst_true": problem.y_worst,
        "runtime_seconds": round(time.time() - t0, 3),
        "bo_cfg": {"acquisition": acq, "kernel": kernel_name,
                   "has_prior_mean": prior_mean is not None,
                   "has_prior_score": prior_score is not None,
                   "feature_aug": method == "bo_lrtc_feature",
                   "portfolio_size": len(portfolio_idx),
                   "prior_corruption": bo_cfg.get("prior_corruption", "none")},
    }
