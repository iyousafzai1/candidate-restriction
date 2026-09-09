#!/usr/bin/env python3
"""
E1 -- Wall-clock timing of the acquisition step (answers reviewer Concern 4).

The paper reports a ~20x reduction in *candidates scored*. The reviewer correctly
notes that is a workload count, not a wall-clock speedup: RF *fitting* cost is
unchanged, prediction is vectorised, and fixed overheads may erode the ratio. This
script measures the actual seconds, isolating:
  - RF fit time (unchanged by restriction),
  - scoring time over the FULL pool vs a restricted K-subset,
  - candidate-sampling overhead,
and reports BOTH the acquisition-scoring speedup AND the per-step / end-to-end speedup
(which include the unchanged fit cost). It is deliberately self-contained: timing of
RF.predict over N rows of dimension d depends on N, d, and the 64-tree forest, not on
the data values, so synthetic pools at HPO-B's real (N, d) operating points give a
faithful and reproducible measurement with no data dependency. Uses the identical
scoring code path as adaptive_k_runner.py.

    pip install scikit-learn numpy        # already present
    python3 timing_benchmark.py                 # full sweep, writes timing_results.csv
    python3 timing_benchmark.py --repeats 7
"""
import argparse, time, csv, statistics
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "timing_results.csv"

RF_N_ESTIMATORS, BETA = 64, 1.96
K_RESTRICT = 2000
N_OBS = 25                      # observed points late in a B=25 run (worst case for fit)
ACQ_STEPS = 20                  # B=25 minus 5 warm-up = 20 acquisition steps
# (pool size N, dim d) operating points spanning HPO-B's real range (254..52448 configs)
POOLS = [(300, 3), (1000, 6), (5000, 8), (11000, 9), (25000, 12), (44000, 6), (52000, 16)]


def _score(rf, X):
    """Identical acquisition computation to the runner: per-tree preds -> mean + beta*std."""
    preds = np.array([t.predict(X) for t in rf.estimators_])
    return preds.mean(0) + BETA * preds.std(0)


def time_point(N, d, repeats, rng):
    from sklearn.ensemble import RandomForestRegressor
    X_obs = rng.standard_normal((N_OBS, d)); y_obs = rng.standard_normal(N_OBS)
    X_pool = rng.standard_normal((N, d))
    fit_t, full_t, restr_t, samp_t = [], [], [], []
    for _ in range(repeats):
        t = time.perf_counter()
        rf = RandomForestRegressor(n_estimators=RF_N_ESTIMATORS, max_depth=None,
                                   min_samples_leaf=1, random_state=0, n_jobs=1).fit(X_obs, y_obs)
        fit_t.append(time.perf_counter() - t)
        t = time.perf_counter(); _score(rf, X_pool); full_t.append(time.perf_counter() - t)
        t = time.perf_counter()
        idx = rng.choice(N, min(K_RESTRICT, N), replace=False); samp_t.append(time.perf_counter() - t)
        Xc = X_pool[idx]
        t = time.perf_counter(); _score(rf, Xc); restr_t.append(time.perf_counter() - t)
    med = lambda a: statistics.median(a) * 1e3      # ms
    fit, full, restr, samp = med(fit_t), med(full_t), med(restr_t), med(samp_t)
    per_step_full = fit + full
    per_step_restr = fit + samp + restr
    return dict(N=N, dim=d, fit_ms=round(fit, 2), score_full_ms=round(full, 2),
                score_restr_ms=round(restr, 2), sample_ms=round(samp, 3),
                scoring_speedup=round(full / restr, 1) if restr else 0,
                per_step_full_ms=round(per_step_full, 2), per_step_restr_ms=round(per_step_restr, 2),
                per_step_speedup=round(per_step_full / per_step_restr, 2) if per_step_restr else 0,
                run_full_s=round(per_step_full * ACQ_STEPS / 1e3, 3),
                run_restr_s=round(per_step_restr * ACQ_STEPS / 1e3, 3),
                end_to_end_speedup=round(per_step_full / per_step_restr, 2) if per_step_restr else 0)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--repeats', type=int, default=7); a = ap.parse_args()
    rng = np.random.default_rng(0)
    print(f"Timing acquisition step: {RF_N_ESTIMATORS}-tree RF, K={K_RESTRICT}, {N_OBS} obs, "
          f"{ACQ_STEPS} acq steps/run, median of {a.repeats} repeats.\n")
    rows = []
    hdr = ('N', 'dim', 'fit', 'score_full', 'score_restr', 'scor_x', 'per_step_x', 'run_full_s', 'run_restr_s')
    print("{:>7} {:>4} {:>7} {:>10} {:>11} {:>7} {:>10} {:>10} {:>10}".format(*hdr))
    for (N, d) in POOLS:
        r = time_point(N, d, a.repeats, rng); rows.append(r)
        print("{N:>7} {dim:>4} {fit_ms:>7} {score_full_ms:>10} {score_restr_ms:>11} "
              "{scoring_speedup:>6}x {per_step_speedup:>9}x {run_full_s:>10} {run_restr_s:>10}".format(**r))
    with open(OUT, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"\nwrote {OUT}")
    print("\nHOW TO READ: 'scor_x' = scoring-only speedup (full/restricted); 'per_step_x' = "
          "speedup once the UNCHANGED RF-fit cost is included. On large pools the scoring "
          "speedup tracks N/K and the fit cost is small, so both are large; on small pools "
          "fit dominates and per-step speedup ~1. For HPO-B, each objective eval is a table "
          "lookup (~0 s), so end-to-end BO time is the per-step time summed over steps: "
          "run_full_s vs run_restr_s is the honest end-to-end acquisition wall-clock.")


if __name__ == '__main__':
    main()
