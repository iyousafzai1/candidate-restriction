#!/usr/bin/env python3
"""
Quantify the optimiser's curse on HPO-B (real surrogate, known true values).

At each surrogate step of an RF-SMBO run we measure the POST-DECISION SURPRISE of
the selected configuration:  over-prediction  =  mu_t(x_t) - f(x_t),
where mu_t is the RF's predicted (normalised) value and f(x_t) is the TRUE value
(known, because HPO-B is precomputed). Averaged, this is the over-selection bias
the optimiser's curse produces. Theory (Lemma 1 / Theorem 1) predicts it grows
like sigma_t * sqrt(2 ln K) in the number K of candidates scored.

We report three things:
  (A) realized curse of the deployed methods: full-candidate RF-SMBO (K=N) vs
      adaptive-K (its scheduled K_t)  -> "full scoring over-predicts by X;
      restriction reduces it by Y%".
  (B) the scaling law: at the surrogate states an RF-SMBO run actually visits, the
      counterfactual over-prediction of an argmax over a random K-subset, for a
      grid of K, fit against sqrt(2 ln K)  -> slope, R^2.
  (C) the same restricted to early steps (few observations, sigma_t large), where
      the curse is strongest and where adaptive-K restricts hardest.

Runs on large-pool spaces (N>5000) so the K-grid is meaningful. A few seeds suffice.

    python3 curse_instrumentation.py            # default 3 seeds
    python3 curse_instrumentation.py --seeds 5
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
KGRID = [50, 100, 250, 500, 1000, 2000, 5000]       # counterfactual K sweep (plus full pool)
ALL_SPACES = ['6767','5906','5971','5965','6794','7607','7609',
              '5527','5891','5636','5859','5889','4796','6766']

def measure(ss, task, seed, hdlr, big, rowsB, rowsA):
    from sklearn.ensemble import RandomForestRegressor
    X = np.array(hdlr.meta_test_data[ss][task]['X'])
    y = hdlr.normalize(np.array(hdlr.meta_test_data[ss][task]['y'])).ravel()   # TRUE values
    rng = np.random.default_rng(seed); n = len(X)
    obs, s0 = [], None
    for step in range(BUDGET):
        if step < WARMUP:
            idx = int(rng.integers(n))
            while idx in obs: idx = int(rng.integers(n))
            obs.append(idx); continue
        rf = RandomForestRegressor(n_estimators=RF_N_ESTIMATORS, max_depth=None,
                                   min_samples_leaf=1, random_state=seed+step, n_jobs=1)
        rf.fit(X[obs], y[obs])
        uneval = np.array([i for i in range(n) if i not in set(obs)]); U = len(uneval)
        # predictions over the whole unevaluated pool (mu, sigma from trees)
        P = np.array([t.predict(X[uneval]) for t in rf.estimators_])
        mu = P.mean(0); sd = P.std(0); acq = mu + RF_UCB_BETA*sd
        sigma_t = float(sd.mean())
        # ---- (A) realized curse: full scoring (K=N) selection ----
        j_full = int(np.argmax(acq))
        over_full = float(mu[j_full] - y[uneval[j_full]])
        # ---- adaptive-K selection (scheduled K_t) ----
        if U <= F:
            j_ak = j_full; K_ak = U
        else:
            if s0 is None: s0 = sigma_t
            K_ak = int(min(U, max(K_MIN, round(K_MIN*(s0/(sigma_t+1e-12))**C_GROW))))
            sub = rng.choice(U, K_ak, replace=False)
            jj = sub[int(np.argmax(acq[sub]))]; j_ak = int(jj); K_ak = len(sub)
        over_ak = float(mu[j_ak] - y[uneval[j_ak]])
        rowsA.append((ss, task, seed, step, U, over_full, K_ak, over_ak, sigma_t))
        # ---- (B) scaling: counterfactual over-prediction for a grid of FIXED K ----
        for K in KGRID:
            if K > U: continue
            sub = rng.choice(U, K, replace=False)
            j = sub[int(np.argmax(acq[sub]))]
            rowsB.append((ss, task, seed, step, K, float(mu[j]-y[uneval[j]]), sigma_t))
        # advance trajectory by the full-scoring choice (a fixed, reproducible base run)
        obs.append(int(uneval[j_full]))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--early', type=int, default=10, help='steps < this count as "early"'); a = ap.parse_args()
    sys.path.insert(0, HPOB_REPO)
    from hpob_handler import HPOBHandler
    import warnings; warnings.filterwarnings('ignore')
    from scipy import stats
    hdlr = HPOBHandler(root_dir=HPOB_DATA_ROOT, mode='v3')
    meta = {r['ss_id']: int(r['n_configs']) for r in
            csv.DictReader(open(HERE.parent / "table2_per_space_b25.csv"))}
    spaces = [s for s in ALL_SPACES if meta[s] > 5000]
    rowsB, rowsA = [], []
    for ss in spaces:
        for task in hdlr.meta_test_data.get(ss, {}):
            for seed in range(a.seeds):
                measure(ss, task, seed, hdlr, meta[ss] > F, rowsB, rowsA)
        print(f"  measured {ss} (N={meta[ss]})", flush=True)
    # save raw
    with open(HERE/"curse_scaling_raw.csv","w",newline="") as f:
        w=csv.writer(f); w.writerow(["ss","task","seed","step","K","over_pred","sigma_t"]); w.writerows(rowsB)
    with open(HERE/"curse_realized_raw.csv","w",newline="") as f:
        w=csv.writer(f); w.writerow(["ss","task","seed","step","N","over_full","K_ak","over_ak","sigma_t"]); w.writerows(rowsA)

    # ---- (A) realized curse: full vs adaptive-K ----
    of = np.array([r[5] for r in rowsA]); oa = np.array([r[7] for r in rowsA])
    print("\n=== (A) Realized optimiser's curse (over-prediction of the selected config) ===")
    print(f"  full-candidate RF-SMBO: mean over-prediction = {of.mean():.4f}")
    print(f"  adaptive-K            : mean over-prediction = {oa.mean():.4f}")
    print(f"  reduction from restriction = {100*(1-oa.mean()/of.mean()):.0f}%   (paired p={stats.wilcoxon(of,oa).pvalue:.2e})")

    # ---- (B) scaling law: bias(K) vs sqrt(2 ln K) ----
    def fit(rows, tag):
        byK=collections.defaultdict(list)
        for (_,_,_,_,K,ov,_) in rows: byK[K].append(ov)
        Ks=sorted(byK); x=np.sqrt(2*np.log(Ks)); yv=np.array([np.mean(byK[k]) for k in Ks])
        sl,ic,r,p,se=stats.linregress(x,yv)
        print(f"\n=== (B{tag}) curse scaling: over-prediction vs sqrt(2 ln K) ===")
        for k in Ks: print(f"    K={k:6d}  mean over-prediction={np.mean(byK[k]):+.4f}")
        print(f"    linear fit: slope={sl:.4f}  R^2={r**2:.4f}  p={p:.2e}  (theory: bias grows ∝ sqrt(2 ln K))")
        return sl, r**2
    fit(rowsB, "-all")
    early=[r for r in rowsB if r[3] < a.early]
    if early: fit(early, "-early")
    print(f"\nwrote curse_scaling_raw.csv, curse_realized_raw.csv")

if __name__ == '__main__':
    main()
