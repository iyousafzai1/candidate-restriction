"""Synthetic end-to-end smoke test for all Paper D methods (no YAHPO needed).

Run:  python3 tests/test_methods_synthetic.py
Verifies every method runs, variants differ (portfolio != hybrid), prior
corruption changes the trajectory, and both anchor modes work — without
needing the YAHPO surrogate data downloaded.
"""
import sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from prep import Problem
from protocol import run_unit, ALL_METHODS

rng = np.random.default_rng(0)
# 2-HP grid 12x10, 20 tasks, low-rank-ish structure + noise
n1, n2, N = 12, 10, 20
A = rng.standard_normal((n1, 3)); B = rng.standard_normal((n2, 3)); C = rng.standard_normal((N, 3))
core = np.einsum('ir,jr,kr->ijk', A, B, C)
tensor = 0.7 + 0.05 * (core - core.mean()) / core.std() + 0.01 * rng.standard_normal((n1, n2, N))
tensor = np.clip(tensor, 0.0, 1.0)
task_ids = [f"t{k}" for k in range(N)]
train_ids = task_ids[:15]
test_ids = task_ids[15:]

axes = [np.logspace(-3, 3, n1), np.linspace(0, 1, n2)]
exp_cfg = {
    "budgets": [5, 10, 25], "k0": 3,
    "lrtc": {"rank": 5, "beta": 1.0, "iters": 30},
    "pmf": {"rank": 5, "reg": 1.0, "iters": 50},
    "bo": {"kernel": "rbf_ard", "alpha": 1e-6, "n_restarts": 2,
           "acquisition": "ei", "xi": 0.0, "portfolio_size": 5,
           "length_scale_bounds": [0.01, 100.0], "pibo_beta": 10.0,
           "refit_budget": None, "prior_corruption": "none"},
    "active_pmf": {"M": 5, "ranks": [3, 5], "refit_every": 1, "reg": 1.0,
                   "fit_iters": 20, "refit_iters": 8, "bootstrap_frac": 0.9},
    "rgpe": {"n_samples": 30}, "tpe": {"n_startup_jobs": 3},
}

for anchor_mode in ("informed", "random"):
    print(f"\n=== anchor_mode={anchor_mode} ===")
    prob = Problem(tensor, task_ids, train_ids, test_ids[0], k0=3,
                   anchor_mode=anchor_mode, anchor_seed=1234)
    anchor_best = float(prob.evaluate(prob.anchors).max())
    print(f"  regret_at_anchors = {prob.normalized_regret(anchor_best):.4f}  "
          f"(y_best={prob.y_best:.3f} y_worst={prob.y_worst:.3f})")
    for m in ALL_METHODS:
        try:
            rec = run_unit(m, prob, seed=1, exp_cfg=exp_cfg, axes=axes)
            cps = {c["budget"]: round(c["normalized_regret"], 4) for c in rec["checkpoints"]}
            print(f"  {m:20s} regret@budget {cps}  bo={rec.get('bo_cfg', {}).get('acquisition','-')}")
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  {m:20s} FAILED: {e}")

# Corruption + portfolio!=hybrid check
print("\n=== A1 check: portfolio vs hybrid must differ in eval_order ===")
prob = Problem(tensor, task_ids, train_ids, test_ids[0], k0=3)
pf = run_unit("bo_lrtc_portfolio", prob, 1, exp_cfg, axes)
hy = run_unit("bo_lrtc_hybrid", prob, 1, exp_cfg, axes)
print("  portfolio has_prior_mean:", pf["bo_cfg"]["has_prior_mean"])
print("  hybrid    has_prior_mean:", hy["bo_cfg"]["has_prior_mean"])
print("  eval_orders identical?", pf["eval_order"] == hy["eval_order"])

print("\n=== E2 check: corruption changes trajectory ===")
ec = dict(exp_cfg); ec["bo"] = dict(exp_cfg["bo"], prior_corruption="shuffle")
clean = run_unit("bo_lrtc_prior", prob, 1, exp_cfg, axes)["checkpoints"][-1]["normalized_regret"]
corr = run_unit("bo_lrtc_prior", prob, 1, ec, axes)["checkpoints"][-1]["normalized_regret"]
print(f"  clean regret@25={clean:.4f}  shuffled-prior regret@25={corr:.4f}")
print("\nALL DONE")
