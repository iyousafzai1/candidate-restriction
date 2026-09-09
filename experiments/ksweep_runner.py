#!/usr/bin/env python3
"""
Option 1 --- "How many candidates are enough?"  Coverage-saturation sweep.

For a grid of FIXED candidate-set sizes K, run RF-SMBO restricted to K random
candidates per step and record (a) final normalised regret at B=25 and (b) the
mean number of candidates scored. Averaged over tasks/seeds this gives the
saturation curve: regret vs K (expected to plateau beyond a few thousand) against
cost vs K (linear). The knee K* is the point beyond which extra candidates buy
essentially no regret.

We also log, per task, the good-configuration density p (fraction of the pool
within eta of the optimum) so the a-priori predictor K*_pred ~ ln(1/delta)/p can be
compared against the empirical knee (bonus analysis).

    python3 ksweep_runner.py            # all 14 spaces, 5 seeds
    python3 ksweep_runner.py --seeds 3
    python3 ksweep_runner.py --analyze-only
Frozen surrogate matches the paper: RF (64 trees), UCB beta=1.96, 5-step warmup.
"""
import sys, os, csv, argparse, collections
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
HPOB_REPO = os.environ.get("HPOB_REPO", "/tmp/HPO-B")
HPOB_DATA_ROOT = os.environ.get("HPOB_DATA_ROOT", "/tmp/hpob_data/hpob-data/")
BUDGET, WARMUP = 25, 5
RF_N_ESTIMATORS, RF_UCB_BETA = 64, 1.96
KGRID = [50, 100, 250, 500, 1000, 2000, 4000, 8000, 16000]   # plus 'full' (=N)
ETAS = [0.01, 0.02, 0.05]                                     # good = within eta of optimum
ALL_SPACES = ['6767','5906','5971','5965','6794','7607','7609',
              '5527','5891','5636','5859','5889','4796','6766']
RESULTS = HERE / "ksweep_results.csv"
DENSITY = HERE / "ksweep_density.csv"

def run_fixed_k(ss, task, seed, K, hdlr):
    from sklearn.ensemble import RandomForestRegressor
    X = np.array(hdlr.meta_test_data[ss][task]['X'])
    y = hdlr.normalize(np.array(hdlr.meta_test_data[ss][task]['y'])).ravel()
    rng = np.random.default_rng(seed); n = len(X)
    obs, best, scored = [], 0.0, []
    for step in range(BUDGET):
        if step < WARMUP:
            idx = int(rng.integers(n))
            while idx in obs: idx = int(rng.integers(n))
        else:
            rf = RandomForestRegressor(n_estimators=RF_N_ESTIMATORS, max_depth=None,
                                       min_samples_leaf=1, random_state=seed+step, n_jobs=1)
            rf.fit(X[obs], y[obs])
            uneval = [i for i in range(n) if i not in set(obs)]; U = len(uneval)
            if U == 0:
                idx = int(rng.integers(n))
            else:
                k = min(K, U); cand = rng.choice(uneval, k, replace=False)
                P = np.array([t.predict(X[cand]) for t in rf.estimators_])
                idx = int(cand[int(np.argmax(P.mean(0) + RF_UCB_BETA*P.std(0)))]); scored.append(k)
        obs.append(int(idx)); best = max(best, y[idx])
    return max(0.0, 1.0 - best), (float(np.mean(scored)) if scored else 0.0)

def density(ss, task, hdlr):
    y = hdlr.normalize(np.array(hdlr.meta_test_data[ss][task]['y'])).ravel()
    opt = y.max()
    return {eta: float(np.mean(y >= opt - eta)) for eta in ETAS}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--analyze-only', action='store_true'); a = ap.parse_args()
    if not a.analyze_only:
        sys.path.insert(0, HPOB_REPO)
        from hpob_handler import HPOBHandler
        import warnings; warnings.filterwarnings('ignore')
        hdlr = HPOBHandler(root_dir=HPOB_DATA_ROOT, mode='v3')
        rows, dens = [], []
        for ss in ALL_SPACES:
            for task in hdlr.meta_test_data.get(ss, {}):
                d = density(ss, task, hdlr)
                dens.append(dict(ss=ss, task=task, **{f'p_eta{e}': d[e] for e in ETAS}))
                for seed in range(a.seeds):
                    for K in KGRID + ['full']:
                        Kv = 10**9 if K == 'full' else K
                        reg, mc = run_fixed_k(ss, task, seed, Kv, hdlr)
                        rows.append(dict(ss=ss, task=task, seed=seed, K=str(K), regret=reg, mean_cand=mc))
            print(f"  swept {ss}", flush=True)
        with open(RESULTS, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        with open(DENSITY, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(dens[0].keys())); w.writeheader(); w.writerows(dens)
        print(f"wrote {RESULTS} ({len(rows)} rows), {DENSITY}")
    analyze()

def analyze():
    if not RESULTS.exists(): print("no results; run first."); return
    rows = list(csv.DictReader(open(RESULTS)))
    meta = {}
    tp = HERE.parent / "table2_per_space_b25.csv"
    if tp.exists():
        meta = {r['ss_id']: int(r['n_configs']) for r in csv.DictReader(open(tp))}
    order = [str(k) for k in KGRID] + ['full']

    def curve(spaces=None):
        # per-space mean regret at each K, then normalise by that space's full-regret
        per = collections.defaultdict(lambda: collections.defaultdict(list))   # space -> K -> regrets
        cand = collections.defaultdict(list)
        for r in rows:
            if spaces and r['ss'] not in spaces: continue
            per[r['ss']][r['K']].append(float(r['regret'])); cand[r['K']].append(float(r['mean_cand']))
        rel = collections.defaultdict(list)
        for ss, d in per.items():
            fr = np.mean(d['full']) if d['full'] else np.nan
            if not fr or np.isnan(fr): continue
            for k in order:
                if d[k]: rel[k].append(np.mean(d[k]) / fr)
        return {k: (np.mean(rel[k]) if rel[k] else np.nan) for k in order}, \
               {k: (np.mean(cand[k]) if cand[k] else np.nan) for k in order}

    for label, spaces in [("ALL spaces", None),
                          ("LARGE pools (N>5000)", {s for s, n in meta.items() if n > 5000} or None)]:
        rc, cc = curve(spaces)
        print(f"\n=== saturation curve [{label}]: relative regret (vs full) & cost vs K ===")
        print(f"{'K':>7}{'rel_regret':>12}{'mean_cand':>11}")
        for k in order:
            print(f"{k:>7}{rc[k]:>12.3f}{cc[k]:>11.0f}")
        knee = next((k for k in order if not np.isnan(rc[k]) and rc[k] <= 1.05), 'full')
        print(f"Knee K* (relative regret within 5% of full): K = {knee}")
    print("\nUse ksweep_results.csv + ksweep_density.csv for the figure and the a-priori-K predictor.")

if __name__ == '__main__':
    main()
