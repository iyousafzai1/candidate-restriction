#!/usr/bin/env python3
"""
Real-training validation slice (de-saturated version).

Purpose: confirm that (a) adaptive-K is NON-INFERIOR to full-candidate RF-SMBO and
(b) the surrogate methods actually beat random search, off precomputed surrogates,
with XGBoost trained live over a large HP grid.

Three anti-saturation measures (all on by default):
  1. BUDGET TRAJECTORY: record best-so-far at B in {5,10,25}. Surrogates separate
     from random at low budget even when everything ties at B=25.
  2. AUC metric (roc_auc / roc_auc_ovr): finer and far less prone to ties than
     accuracy. (Cache is keyed by metric, so this does not reuse the accuracy cache.)
  3. HARDER DATASETS: original 8 + 6 harder (high-dim / genuinely difficult) OpenML
     sets; the analysis flags which are non-saturated so we report the informative ones.

Feasible via a persistent (dataset|metric|config) -> score cache. Rows > 4000 are
subsampled (stratified) to keep XGBoost fits fast.

    pip install xgboost scikit-learn --break-system-packages
    python3 real_training_runner.py --smoke        # 3 datasets, 3 seeds
    python3 real_training_runner.py                 # full slice
    python3 real_training_runner.py --analyze-only
Frozen adaptive-K constants: F=2500, K_MIN=1000, C_GROW=0.75.
"""
import os, sys, json, time, argparse, itertools, collections
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
OUT  = HERE / "real_training"; OUT.mkdir(exist_ok=True)
DATA_DIR = OUT / "datasets"; DATA_DIR.mkdir(exist_ok=True)
METRIC = os.environ.get("METRIC", "auc")               # 'auc' or 'acc'
EVAL_CACHE = OUT / f"eval_cache_{METRIC}.json"
RESULTS = OUT / f"real_training_results_{METRIC}.csv"

BUDGETS = [5, 10, 25]
WARMUP = 5
RF_N_ESTIMATORS, RF_UCB_BETA = 64, 1.96
AK_FULL_FLOOR, AK_K_MIN, AK_C_GROW = 2500, 1000, 0.75
SEEDS = list(range(10))
METHODS = ['random', 'rf_smbo', 'adaptive_k']
MAX_ROWS = 4000

DATASETS = [   # (name, openml_id)   original 8 + 6 harder
    ('diabetes', 37), ('ionosphere', 59), ('wdbc', 1510), ('qsar-biodeg', 1494),
    ('kc1', 1067), ('pc1', 1068), ('spambase', 44), ('phoneme', 1489),
    # harder / higher-dim (less likely to saturate at low budget):
    ('credit-g', 31), ('sonar', 40), ('madelon', 1485), ('hill-valley', 1479),
    ('ozone-8hr', 1487), ('Bioresponse', 4134),
]

GRID_AXES = collections.OrderedDict([
    ('max_depth',        [2,3,4,5,6,7,8,10]),
    ('learning_rate',    [0.005,0.01,0.03,0.05,0.1,0.2,0.3]),
    ('n_estimators',     [50,100,200,400,600]),
    ('subsample',        [0.6,0.8,1.0]),
    ('colsample_bytree', [0.6,0.8,1.0]),
    ('min_child_weight', [1,3,5]),
    ('reg_lambda',       [1,10,100]),
])   # 22,680

def build_grid():
    keys = list(GRID_AXES); axes = [GRID_AXES[k] for k in keys]
    configs = [dict(zip(keys, v)) for v in itertools.product(*axes)]
    def enc(c):
        return [np.log(c['max_depth']), np.log(c['learning_rate']), np.log(c['n_estimators']),
                c['subsample'], c['colsample_bytree'], np.log(c['min_child_weight']),
                np.log(c['reg_lambda'])]
    X = np.array([enc(c) for c in configs], float)
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    return configs, X

_eval = json.load(open(EVAL_CACHE)) if EVAL_CACHE.exists() else {}
_nclass = {}

def load_dataset(name, data_id):
    from sklearn.datasets import fetch_openml
    p = DATA_DIR / f"{name}.npz"
    if p.exists():
        d = np.load(p, allow_pickle=True); X, y = d['X'], d['y']
    else:
        ds = fetch_openml(data_id=data_id, as_frame=True)
        X = ds.data.apply(lambda c: c.cat.codes if str(c.dtype) == 'category' else c).to_numpy(float)
        X = np.nan_to_num(X)
        y = ds.target.astype('category').cat.codes.to_numpy()
        np.savez(p, X=X, y=y)
    if len(X) > MAX_ROWS:                      # stratified subsample for speed
        rng = np.random.default_rng(0); idx = []
        for c in np.unique(y):
            ci = np.where(y == c)[0]
            idx.append(rng.choice(ci, max(1, int(MAX_ROWS*len(ci)/len(y))), replace=False))
        idx = np.concatenate(idx); X, y = X[idx], y[idx]
    return X, y

def eval_config(name, X, y, config):
    key = f"{name}|{METRIC}|" + "_".join(f"{k}={config[k]}" for k in GRID_AXES)
    if key in _eval: return _eval[key]
    from xgboost import XGBClassifier
    from sklearn.model_selection import cross_val_score
    nk = _nclass.setdefault(name, len(np.unique(y)))
    if METRIC == 'auc':
        scoring = 'roc_auc' if nk == 2 else 'roc_auc_ovr'
    else:
        scoring = 'accuracy'
    clf = XGBClassifier(**config, tree_method='hist', eval_metric='logloss',
                        n_jobs=1, verbosity=0, random_state=0)
    s = float(cross_val_score(clf, X, y, cv=3, scoring=scoring).mean())
    _eval[key] = s; return s

def run_method(method, name, X, y, configs, Xg, seed):
    from sklearn.ensemble import RandomForestRegressor
    rng = np.random.default_rng(seed); n = len(configs)
    obs_i, obs_v, best, traj, cc, s0 = [], [], -np.inf, [], [], None
    for step in range(max(BUDGETS)):
        if step < WARMUP or method == 'random':
            idx = int(rng.integers(n))
            while idx in obs_i: idx = int(rng.integers(n))
            scored = 0
        else:
            rf = RandomForestRegressor(n_estimators=RF_N_ESTIMATORS, max_depth=None,
                                       min_samples_leaf=1, random_state=seed+step, n_jobs=1)
            rf.fit(Xg[obs_i], np.array(obs_v))
            uneval = [i for i in range(n) if i not in set(obs_i)]; U = len(uneval)
            if method == 'rf_smbo' or U <= AK_FULL_FLOOR:
                cand = np.array(uneval); scored = U
            else:
                probe = rng.choice(uneval, min(AK_K_MIN, U), replace=False)
                pp = np.array([t.predict(Xg[probe]) for t in rf.estimators_]); s_hat = float(pp.std(0).mean())+1e-12
                if s0 is None: s0 = s_hat
                K_t = int(min(U, max(AK_K_MIN, round(AK_K_MIN*(s0/s_hat)**AK_C_GROW))))
                if K_t <= len(probe): cand = probe
                else:
                    rem = list(set(uneval)-set(probe.tolist()))
                    cand = np.concatenate([probe, rng.choice(rem, min(K_t-len(probe), len(rem)), replace=False)])
                scored = len(cand)
            preds = np.array([t.predict(Xg[cand]) for t in rf.estimators_])
            idx = int(cand[int(np.argmax(preds.mean(0)+RF_UCB_BETA*preds.std(0)))])
        v = eval_config(name, X, y, configs[idx])
        obs_i.append(idx); obs_v.append(v); best = max(best, v); traj.append(best); cc.append(scored)
    active = [c for c in cc if c > 0]
    return {b: traj[min(b, len(traj))-1] for b in BUDGETS}, (float(np.mean(active)) if active else 0.0)

def do_run(datasets, seeds):
    import csv
    configs, Xg = build_grid()
    print(f"grid={len(configs)} configs, metric={METRIC}, datasets={[d[0] for d in datasets]}")
    rows, t0 = [], time.time()
    for (name, did) in datasets:
        try: X, y = load_dataset(name, did)
        except Exception as e: print(f"  SKIP {name}: {e}"); continue
        for seed in seeds:
            for method in METHODS:
                best_by_b, mc = run_method(method, name, X, y, configs, Xg, seed)
                row = {'dataset': name, 'n_rows': len(X), 'seed': seed, 'method': method,
                       'mean_candidates_scored': mc}
                for b in BUDGETS: row[f'best_B{b}'] = best_by_b[b]
                rows.append(row)
            json.dump(_eval, open(EVAL_CACHE, 'w'))
            print(f"  {name} seed{seed} ({len(_eval)} cached, {time.time()-t0:.0f}s)", flush=True)
    with open(RESULTS, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"wrote {RESULTS} ({len(rows)} rows)")

def analyze():
    import csv
    from scipy import stats
    if not RESULTS.exists(): print("no results; run first."); return
    rows = list(csv.DictReader(open(RESULTS)))
    cand = collections.defaultdict(list)
    for r in rows:
        if r['method'] in ('rf_smbo', 'adaptive_k'): cand[r['method']].append(float(r['mean_candidates_scored']))
    print(f"\n=== real-training slice, metric={METRIC} ===")
    for b in BUDGETS:
        by = collections.defaultdict(dict); ds = collections.defaultdict(lambda: collections.defaultdict(list))
        for r in rows:
            by[(r['dataset'], r['seed'])][r['method']] = float(r[f'best_B{b}'])
            ds[r['dataset']][r['method']].append(float(r[f'best_B{b}']))
        keys = [k for k in by if all(m in by[k] for m in METHODS)]
        ak = np.array([by[k]['adaptive_k'] for k in keys]); sm = np.array([by[k]['rf_smbo'] for k in keys]); rd = np.array([by[k]['random'] for k in keys])
        def pw(a, b_):
            d = a-b_; p = stats.wilcoxon(a, b_).pvalue if np.any(np.abs(d) > 1e-12) else float('nan')
            return d.mean(), int((d > 1e-9).sum()), int((d < -1e-9).sum()), p
        nonsat = sum(1 for d in ds if (max(np.mean(v) for v in ds[d].values()) - min(np.mean(v) for v in ds[d].values())) >= 0.005)
        print(f"\n B={b}:  adaptive_k={ak.mean():.4f}  rf_smbo={sm.mean():.4f}  random={rd.mean():.4f}   (non-saturated datasets: {nonsat}/{len(ds)})")
        print("   adaptive_k vs rf_smbo (non-inferiority): mean=%+.4f W/L=%d/%d p=%.3f" % pw(ak, sm))
        print("   rf_smbo   vs random    (surrogate value): mean=%+.4f W/L=%d/%d p=%.3f" % pw(sm, rd))
        print("   adaptive_k vs random   (surrogate value): mean=%+.4f W/L=%d/%d p=%.3f" % pw(ak, rd))
    if cand['rf_smbo'] and cand['adaptive_k']:
        Nb, Kb = np.mean(cand['rf_smbo']), np.mean(cand['adaptive_k'])
        print(f"\n EFFICIENCY: full={Nb:.0f} vs adaptive_k={Kb:.0f} candidates/step -> {Nb/Kb:.1f}x")
    # per-dataset table at B=25 to identify the informative (non-saturated) ones
    print("\n per-dataset spread at B=25 (surrogate value = rf_smbo - random):")
    ds25 = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows: ds25[r['dataset']][r['method']].append(float(r['best_B25']))
    for d in sorted(ds25):
        m = {k: np.mean(v) for k, v in ds25[d].items()}
        print(f"   {d:<14} rand={m['random']:.4f} smbo={m['rf_smbo']:.4f} adapt={m['adaptive_k']:.4f}  smbo-rand={m['rf_smbo']-m['random']:+.4f}")

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true'); ap.add_argument('--analyze-only', action='store_true')
    a = ap.parse_args()
    if not a.analyze_only:
        do_run(DATASETS[:3] if a.smoke else DATASETS, list(range(3)) if a.smoke else SEEDS)
    analyze()
