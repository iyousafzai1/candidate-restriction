#!/usr/bin/env python3
"""
Second learner family (fallback / complement to HPO-B): candidate restriction on a
LIVE-TRAINED learner other than XGBoost. Same protocol, same *frozen* adaptive-K
schedule (F=2500, K_MIN=1000, C_GROW=0.75) as the main study.

Motivation. The main paper evaluates restriction on the HPO-B surrogate suite and on
a live XGBoost grid. A reviewer's remaining worry is "does this hold off that one
benchmark family / one learner?" This script answers it with a DIFFERENT learner
trained live on the same OpenML datasets over a LARGE grid, so the K=2000 restriction
actually bites (pool >> floor). Nothing here depends on HPOBench, so it runs cleanly
on modern Python (3.13, numpy 2.x).

    pip install scikit-learn scipy pandas          # already in the venv
    python3 multi_learner_runner.py --learner rf --smoke     # 3 datasets, 3 seeds
    python3 multi_learner_runner.py --learner rf             # full slice
    python3 multi_learner_runner.py --learner rf --analyze-only

Design mirrors real_training_runner.py (validated) exactly; only eval_config (which
learner is trained) and the grid change. Cache is keyed by learner|dataset|metric|cfg.
"""
import os, sys, json, time, argparse, itertools, collections, csv, signal
from pathlib import Path
import numpy as np

# Timeout protection: skip seeds that hang > 5 minutes
SEED_TIMEOUT_SECS = 300  # 5 minutes per seed
SKIPPED_SEEDS = []

class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError(f"Seed exceeded {SEED_TIMEOUT_SECS}s timeout")

def run_with_timeout(func, *args, **kwargs):
    """Run func with timeout; return (result, skipped) where skipped=True if timeout."""
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(SEED_TIMEOUT_SECS)
    try:
        result = func(*args, **kwargs)
        signal.alarm(0)  # Cancel alarm
        return result, False
    except TimeoutError as e:
        signal.alarm(0)
        print(f"    [TIMEOUT] {e}", flush=True)
        return None, True

HERE = Path(__file__).resolve().parent
OUT  = HERE / "multi_learner"; OUT.mkdir(exist_ok=True)
DATA_DIR = OUT / "datasets"; DATA_DIR.mkdir(exist_ok=True)
LEARNER = os.environ.get("LEARNER", "rf")
METRIC  = "auc"

BUDGETS = [5, 10, 25]
WARMUP  = 5
RF_N_ESTIMATORS, RF_UCB_BETA = 64, 1.96          # surrogate RF (acquisition), NOT the tuned learner
AK_FULL_FLOOR, AK_K_MIN, AK_C_GROW = 2500, 1000, 0.75   # frozen adaptive-K schedule
SEEDS   = list(range(10))
METHODS = ['random', 'rf_smbo', 'adaptive_k']
MAX_ROWS = 4000

DATASETS = [
    ('diabetes', 37), ('ionosphere', 59), ('wdbc', 1510), ('qsar-biodeg', 1494),
    ('kc1', 1067), ('pc1', 1068), ('spambase', 44), ('phoneme', 1489),
    ('credit-g', 31), ('sonar', 40),
    # SKIP: madelon, hill-valley, ozone-8hr, Bioresponse (too slow / unstable)
]

# ---- learner grids (chosen LARGE so restriction K=2000 binds; pool >> floor) ----
GRIDS = {
    'rf': collections.OrderedDict([        # 7*6*6*4*3*2 = 12,096 configs
        ('n_estimators',      [50, 100, 200, 300, 400, 600, 800]),
        ('max_depth',         [3, 5, 8, 12, 20, 0]),          # 0 -> None (unlimited)
        ('max_features',      [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]),
        ('min_samples_leaf',  [1, 2, 4, 8]),
        ('min_samples_split', [2, 5, 10]),
        ('criterion_gini',    [1, 0]),                        # 1 -> gini, 0 -> entropy
    ]),
    'svm': collections.OrderedDict([       # small natural grid: exercises WEAK-DOMINANCE
        ('C',            [10.0**e for e in range(-3, 4)]),    # 7
        ('gamma',        [10.0**e for e in range(-4, 2)]),    # 6
        ('class_weight', [0, 1]),                             # 0 -> None, 1 -> balanced
    ]),                                                       # 84 configs (< floor by design)
}

def build_grid(learner):
    axes = GRIDS[learner]; keys = list(axes)
    configs = [dict(zip(keys, v)) for v in itertools.product(*[axes[k] for k in keys])]
    # numeric encoding for the acquisition surrogate (log-scale where natural), standardised
    def enc(c):
        row = []
        for k in keys:
            v = c[k]
            if learner == 'rf':
                if k in ('n_estimators', 'min_samples_leaf', 'min_samples_split'): row.append(np.log(v))
                elif k == 'max_depth': row.append(np.log(v) if v > 0 else np.log(40))  # None ~ deep
                else: row.append(float(v))
            else:  # svm
                row.append(np.log10(v) if k in ('C', 'gamma') else float(v))
        return row
    X = np.array([enc(c) for c in configs], float)
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    return configs, X

EVAL_CACHE = OUT / f"eval_cache_{LEARNER}_{METRIC}.json"
RESULTS    = OUT / f"multi_learner_results_{LEARNER}_{METRIC}.csv"
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
    if len(X) > MAX_ROWS:
        rng = np.random.default_rng(0); idx = []
        for c in np.unique(y):
            ci = np.where(y == c)[0]
            idx.append(rng.choice(ci, max(1, int(MAX_ROWS*len(ci)/len(y))), replace=False))
        idx = np.concatenate(idx); X, y = X[idx], y[idx]
    return X, y

def _make_estimator(learner, config):
    if learner == 'rf':
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(
            n_estimators=int(config['n_estimators']),
            max_depth=(None if config['max_depth'] == 0 else int(config['max_depth'])),
            max_features=config['max_features'],
            min_samples_leaf=int(config['min_samples_leaf']),
            min_samples_split=int(config['min_samples_split']),
            criterion=('gini' if config['criterion_gini'] == 1 else 'entropy'),
            n_jobs=1, random_state=0)
    else:
        from sklearn.svm import SVC
        return SVC(C=config['C'], gamma=config['gamma'],
                   class_weight=(None if config['class_weight'] == 0 else 'balanced'),
                   probability=False, random_state=0)

def eval_config(learner, name, X, y, config, keys):
    key = f"{learner}|{name}|{METRIC}|" + "_".join(f"{k}={config[k]}" for k in keys)
    if key in _eval: return _eval[key]
    from sklearn.model_selection import cross_val_score
    nk = _nclass.setdefault(name, len(np.unique(y)))
    scoring = 'roc_auc' if nk == 2 else 'roc_auc_ovr'
    clf = _make_estimator(learner, config)
    # SVC has no predict_proba unless probability=True; use decision_function via 'roc_auc' works for binary,
    # for multiclass fall back to accuracy to avoid needing probabilities.
    if learner == 'svm' and nk > 2:
        scoring = 'accuracy'
    s = float(cross_val_score(clf, X, y, cv=3, scoring=scoring).mean())
    _eval[key] = s; return s

def run_method(learner, method, name, X, y, configs, Xg, keys, seed):
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
        v = eval_config(learner, name, X, y, configs[idx], keys)
        obs_i.append(idx); obs_v.append(v); best = max(best, v); traj.append(best); cc.append(scored)
    active = [c for c in cc if c > 0]
    return {b: traj[min(b, len(traj))-1] for b in BUDGETS}, (float(np.mean(active)) if active else 0.0)

def do_run(learner, datasets, seeds):
    configs, Xg = build_grid(learner); keys = list(GRIDS[learner])
    print(f"learner={learner} grid={len(configs)} configs (floor={AK_FULL_FLOOR}; "
          f"restriction {'BINDS' if len(configs) > AK_FULL_FLOOR else 'reduces to full scoring -> tests weak-dominance'})")
    print(f"datasets={[d[0] for d in datasets]}")
    rows, t0 = [], time.time()
    for (name, did) in datasets:
        try: X, y = load_dataset(name, did)
        except Exception as e: print(f"  SKIP {name}: {e}"); continue
        for seed in seeds:
            skipped = False
            for method in METHODS:
                result, timed_out = run_with_timeout(run_method, learner, method, name, X, y, configs, Xg, keys, seed)
                if timed_out:
                    SKIPPED_SEEDS.append((name, seed, method))
                    skipped = True
                    break  # Skip remaining methods for this seed
                best_by_b, mc = result
                row = {'learner': learner, 'dataset': name, 'n_rows': len(X), 'seed': seed,
                       'method': method, 'mean_candidates_scored': mc}
                for b in BUDGETS: row[f'best_B{b}'] = best_by_b[b]
                rows.append(row)
            if not skipped:
                json.dump(_eval, open(EVAL_CACHE, 'w'))
                print(f"  {name} seed{seed} ({len(_eval)} cached, {time.time()-t0:.0f}s)", flush=True)
            else:
                print(f"  {name} seed{seed} [SKIPPED due to timeout]", flush=True)
    with open(RESULTS, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"wrote {RESULTS} ({len(rows)} rows)")
    if SKIPPED_SEEDS:
        print(f"\n[WARNING] {len(SKIPPED_SEEDS)} seeds skipped due to timeout: {SKIPPED_SEEDS}")

def analyze(learner):
    from scipy import stats
    if not RESULTS.exists(): print("no results; run first."); return
    rows = list(csv.DictReader(open(RESULTS)))
    cand = collections.defaultdict(list)
    for r in rows:
        if r['method'] in ('rf_smbo', 'adaptive_k'): cand[r['method']].append(float(r['mean_candidates_scored']))
    print(f"\n=== second-learner slice: learner={learner}, metric={METRIC} ===")
    for b in BUDGETS:
        by = collections.defaultdict(dict); ds = collections.defaultdict(lambda: collections.defaultdict(list))
        for r in rows:
            by[(r['dataset'], r['seed'])][r['method']] = float(r[f'best_B{b}'])
            ds[r['dataset']][r['method']].append(float(r[f'best_B{b}']))
        keys = [k for k in by if all(m in by[k] for m in METHODS)]
        if not keys: print(f" B={b}: no paired rows yet"); continue
        ak = np.array([by[k]['adaptive_k'] for k in keys]); sm = np.array([by[k]['rf_smbo'] for k in keys]); rd = np.array([by[k]['random'] for k in keys])
        def pw(a, b_):
            d = a-b_; p = stats.wilcoxon(a, b_).pvalue if np.any(np.abs(d) > 1e-12) else float('nan')
            return d.mean(), int((d > 1e-9).sum()), int((d < -1e-9).sum()), p
        nonsat = sum(1 for d in ds if (max(np.mean(v) for v in ds[d].values()) - min(np.mean(v) for v in ds[d].values())) >= 0.005)
        print(f"\n B={b}:  adaptive_k={ak.mean():.4f}  rf_smbo={sm.mean():.4f}  random={rd.mean():.4f}   (non-saturated datasets: {nonsat}/{len(ds)})")
        print("   adaptive_k vs rf_smbo (non-inferiority): mean=%+.4f W/L=%d/%d p=%.3f" % pw(ak, sm))
        print("   rf_smbo    vs random   (surrogate value): mean=%+.4f W/L=%d/%d p=%.3f" % pw(sm, rd))
        print("   adaptive_k vs random   (surrogate value): mean=%+.4f W/L=%d/%d p=%.3f" % pw(ak, rd))
    if cand['rf_smbo'] and cand['adaptive_k']:
        Nb, Kb = np.mean(cand['rf_smbo']), np.mean(cand['adaptive_k'])
        print(f"\n EFFICIENCY: full={Nb:.0f} vs adaptive_k={Kb:.0f} candidates/step -> {Nb/Kb:.1f}x")

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--learner', default=os.environ.get('LEARNER', 'rf'), choices=['rf', 'svm'])
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--analyze-only', action='store_true')
    a = ap.parse_args()
    LEARNER = a.learner
    EVAL_CACHE = OUT / f"eval_cache_{LEARNER}_{METRIC}.json"
    RESULTS    = OUT / f"multi_learner_results_{LEARNER}_{METRIC}.csv"
    _eval = json.load(open(EVAL_CACHE)) if EVAL_CACHE.exists() else {}
    if not a.analyze_only:
        do_run(LEARNER, DATASETS[:3] if a.smoke else DATASETS, list(range(3)) if a.smoke else SEEDS)
    analyze(LEARNER)
