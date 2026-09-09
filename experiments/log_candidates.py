#!/usr/bin/env python3
"""
Measure the acquisition-cost ratio N_bar / K_bar on HPO-B (Table tab:eff, tabular row).

For each (space, task, seed) it runs the frozen adaptive-K selection while logging,
at every surrogate step, the number of candidates full-candidate RF-SMBO would score
(N_t = unevaluated pool) and the number adaptive-K actually scores (K_t). It reports
the mean per-step ratio overall and on the large-pool regime (N>F), which is where
restriction is active.

Frozen constants match the study: F=2500, K_MIN=1000, C_GROW=0.75.
A few seeds suffice for a stable mean; default 3.

    python3 log_candidates.py            # all 14 spaces, 3 seeds
    python3 log_candidates.py --seeds 5
"""
import sys, os, csv, argparse, collections
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
HPOB_REPO = os.environ.get("HPOB_REPO", "/tmp/HPO-B")
HPOB_DATA_ROOT = os.environ.get("HPOB_DATA_ROOT", "/tmp/hpob_data/hpob-data/")
BUDGET, WARMUP = 25, 5
RF_N_ESTIMATORS, RF_UCB_BETA = 64, 1.96
F, K_MIN, C_GROW = 2500, 1000, 0.75
ALL_SPACES = ['6767','5906','5971','5965','6794','7607','7609',
              '5527','5891','5636','5859','5889','4796','6766']

def run_logged(ss, task, seed, hdlr, log):
    from sklearn.ensemble import RandomForestRegressor
    X = np.array(hdlr.meta_test_data[ss][task]['X'])
    y = hdlr.normalize(np.array(hdlr.meta_test_data[ss][task]['y']))
    rng = np.random.default_rng(seed); n = len(X)
    obs, s0 = [], None
    for step in range(BUDGET):
        if step < WARMUP:
            idx = int(rng.integers(n))
            while idx in obs: idx = int(rng.integers(n))
        else:
            rf = RandomForestRegressor(n_estimators=RF_N_ESTIMATORS, max_depth=None,
                                       min_samples_leaf=1, random_state=seed+step, n_jobs=1)
            rf.fit(X[obs], y[obs].ravel())
            obs_set = set(obs); uneval = [i for i in range(n) if i not in obs_set]; U = len(uneval)
            if U == 0:
                idx = int(rng.integers(n)); log.append((U, 0)); obs.append(idx); continue
            if U <= F:
                Xp = X[uneval]; preds = np.array([t.predict(Xp) for t in rf.estimators_])
                idx = uneval[int(np.argmax(preds.mean(0)+RF_UCB_BETA*preds.std(0)))]
                scored = U
            else:
                probe = rng.choice(uneval, min(K_MIN, U), replace=False)
                pp = np.array([t.predict(X[probe]) for t in rf.estimators_]); pstd = pp.std(0)
                s_hat = float(pstd.mean())+1e-12
                if s0 is None: s0 = s_hat
                K_t = int(min(U, max(K_MIN, round(K_MIN*(s0/s_hat)**C_GROW))))
                if K_t <= len(probe):
                    cand = probe; score = pp.mean(0)+RF_UCB_BETA*pstd
                else:
                    rem = list(set(uneval)-set(probe.tolist()))
                    extra = rng.choice(rem, min(K_t-len(probe), len(rem)), replace=False)
                    cand = np.concatenate([probe, extra])
                    pc = np.array([t.predict(X[cand]) for t in rf.estimators_])
                    score = pc.mean(0)+RF_UCB_BETA*pc.std(0)
                idx = int(cand[int(np.argmax(score))]); scored = len(cand)
            log.append((U, scored))     # (N_t for full scoring, K_t for adaptive-K)
        obs.append(int(idx))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--seeds', type=int, default=3); a = ap.parse_args()
    sys.path.insert(0, HPOB_REPO)
    from hpob_handler import HPOBHandler
    import warnings; warnings.filterwarnings('ignore')
    hdlr = HPOBHandler(root_dir=HPOB_DATA_ROOT, mode='v3')
    meta = {r['ss_id']: int(r['n_configs']) for r in
            csv.DictReader(open(HERE.parent / "table2_per_space_b25.csv"))}
    rows = []
    allN, allK, bigN, bigK = [], [], [], []
    for ss in ALL_SPACES:
        for task in hdlr.meta_test_data.get(ss, {}):
            for seed in range(a.seeds):
                log = []; run_logged(ss, task, seed, hdlr, log)
                for (N, K) in log:
                    if K == 0: continue
                    allN.append(N); allK.append(K)
                    if meta[ss] > F: bigN.append(N); bigK.append(K)
        print(f"  logged {ss}", flush=True)
    def ratio(N, K): return (np.mean(N)/np.mean(K)) if K else float('nan')
    print(f"\nOVERALL: N̄={np.mean(allN):.0f}  K̄={np.mean(allK):.0f}  ratio N̄/K̄={ratio(allN,allK):.1f}x")
    print(f"LARGE POOLS (N>{F}): N̄={np.mean(bigN):.0f}  K̄={np.mean(bigK):.0f}  ratio={ratio(bigN,bigK):.1f}x")
    with open(HERE / "candidate_cost.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["regime","N_bar","K_bar","ratio"])
        w.writerow(["overall", f"{np.mean(allN):.0f}", f"{np.mean(allK):.0f}", f"{ratio(allN,allK):.1f}"])
        w.writerow(["large_pool", f"{np.mean(bigN):.0f}", f"{np.mean(bigK):.0f}", f"{ratio(bigN,bigK):.1f}"])
    print(f"wrote {HERE/'candidate_cost.csv'}")

if __name__ == '__main__':
    main()
