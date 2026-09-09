#!/usr/bin/env python3
"""
Second benchmark: candidate restriction on HPOBench (tabular ML benchmarks).

Same protocol as the HPO-B study: for each problem (model x OpenML task) we build
the full discrete configuration pool with its objective values, then run
{random, tpe, rf_smbo (full), adaptive_k (K=2000 restricted)} for B=25 and compare
restricted vs full (non-inferiority) plus the cost ratio.

HPOBench's API differs from HPO-B and cannot be tested offline here, so the data
layer (build_pool) is PROBE-FIRST:

    python3 hpobench_runner.py --probe        # loads ONE benchmark, prints its
                                              # structure so we confirm/fix build_pool
    python3 hpobench_runner.py --workers 4    # full run (server ~85% loaded -> 4 workers)
    python3 hpobench_runner.py --analyze-only

Install (local, no container):
    pip install hpobench            # or: pip install "hpobench[ml_tabular_benchmarks]"
The ml-tabular data auto-downloads on first use.
"""
import sys, os, csv, argparse, collections, warnings, time, itertools
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
warnings.filterwarnings('ignore')

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "hpobench_results.csv"
BUDGET, WARMUP, BETA, K_RESTRICT = 25, 5, 1.96, 2000

# HPOBench ml-tabular: (model, task_id). Adjust after --probe confirms availability.
MODELS = ['xgb', 'rf', 'svm']
TASK_IDS = [10101, 53, 146818, 146821, 31, 3917]      # OpenML task ids used by HPOBench ml
PROBLEMS = [(m, t) for m in MODELS for t in TASK_IDS]

# ---------------------------------------------------------------------------
# DATA LAYER (probe-first). Returns (X_encoded [n,d] float, y [n] in [0,1], opt=1).
# ---------------------------------------------------------------------------
def _load_benchmark(model, task_id):
    # local (non-container) import path; adjust if --probe shows a different one
    from hpobench.benchmarks.ml.tabular_benchmark import TabularBenchmark
    return TabularBenchmark(model=model, task_id=task_id)

def _hplist(cs):
    """Return hyperparameter objects, tolerant of old and new ConfigSpace APIs."""
    try:
        return list(cs.get_hyperparameters())      # ConfigSpace <= 0.6
    except Exception:
        return list(cs.values())                   # ConfigSpace >= 0.7 / 1.x

def _choices_of(hp):
    """Finite ordered value set for a discrete hyperparameter, else None."""
    if hasattr(hp, 'choices'):        return list(hp.choices)      # Categorical
    if hasattr(hp, 'sequence'):       return list(hp.sequence)     # Ordinal
    # Integer hp with a small range is enumerable
    lo, up = getattr(hp, 'lower', None), getattr(hp, 'upper', None)
    if lo is not None and up is not None and float(lo).is_integer() and float(up).is_integer() \
       and (int(up) - int(lo)) <= 512:
        return list(range(int(lo), int(up) + 1))
    return None

def build_pool(model, task_id):
    """Enumerate the discrete config grid and its objective values -> (X, y)."""
    b = _load_benchmark(model, task_id)
    cs = b.get_configuration_space(seed=0)
    names, choices = [], []
    for hp in _hplist(cs):
        names.append(hp.name); choices.append(_choices_of(hp))
    if any(c is None for c in choices):
        bad = [n for n, c in zip(names, choices) if c is None]
        raise RuntimeError(f"non-discrete hyperparameter(s) {bad}; inspect with --probe")
    configs, vals = [], []
    for combo in itertools.product(*choices):
        cfg = dict(zip(names, combo))
        res = b.objective_function(configuration=cfg)   # tabular: returns dict
        fv = res['function_value'] if isinstance(res, dict) else res
        configs.append(combo); vals.append(float(fv))
    # objective is typically a LOSS (lower better); convert to value in [0,1], opt=1
    v = np.array(vals); v = -v
    y = (v - v.min()) / (v.max() - v.min() + 1e-12)
    # encode configs numerically (ordinal index per hyperparameter, standardised)
    idx = {nm: {c: i for i, c in enumerate(ch)} for nm, ch in zip(names, choices)}
    X = np.array([[idx[nm][combo[j]] for j, nm in enumerate(names)] for combo in configs], float)
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    return X, y

# ---------------------------------------------------------------------------
# BO methods (identical to the validated HPO-B runners)
# ---------------------------------------------------------------------------
def _run(method, X, y, seed):
    from sklearn.ensemble import RandomForestRegressor
    from scipy.stats import gaussian_kde
    rng = np.random.default_rng(seed); n = len(X)
    obs, best, scored = [], 0.0, []
    for step in range(BUDGET):
        if step < WARMUP or method == 'random':
            idx = int(rng.integers(n))
            while idx in obs: idx = int(rng.integers(n))
            scored.append(0) if method == 'random' and step >= WARMUP else None
        else:
            uneval = [i for i in range(n) if i not in set(obs)]; U = len(uneval)
            if U == 0:
                idx = int(rng.integers(n))
            elif method == 'tpe':
                arr = np.array([y[i] for i in obs]); sp = np.median(arr)
                good, bad = arr >= sp, arr < sp
                if good.sum() >= 2 and bad.sum() >= 2:
                    try:
                        kg = gaussian_kde(X[[obs[i] for i in np.where(good)[0]]].T)
                        kb = gaussian_kde(X[[obs[i] for i in np.where(bad)[0]]].T)
                        cand = np.array(uneval)
                        acq = kg.logpdf(X[cand].T) - kb.logpdf(X[cand].T)
                        idx = int(cand[int(np.argmax(acq))])
                    except Exception:
                        idx = int(rng.integers(n))
                else:
                    idx = int(rng.integers(n))
            else:   # rf_smbo (full) or adaptive_k (restricted K)
                cand = np.array(uneval) if method == 'rf_smbo' else \
                       rng.choice(uneval, min(K_RESTRICT, U), replace=False)
                rf = RandomForestRegressor(n_estimators=64, max_depth=None, min_samples_leaf=1,
                                           random_state=seed+step, n_jobs=1).fit(
                                           X[obs], np.array([y[i] for i in obs]))
                P = np.array([t.predict(X[cand]) for t in rf.estimators_])
                idx = int(cand[int(np.argmax(P.mean(0) + BETA*P.std(0)))]); scored.append(len(cand))
        obs.append(int(idx)); best = max(best, y[idx])
    return max(0.0, 1.0 - best), (float(np.mean([s for s in scored if s])) if any(scored) else 0.0)

def run_unit(args):
    model, task_id, seed, method = args
    try:
        X, y = _POOLS[(model, task_id)]
        reg, mc = _run(method, X, y, seed)
        return dict(model=model, task=task_id, seed=seed, method=method, ok=1, regret=reg, mean_cand=mc)
    except Exception as e:
        return dict(model=model, task=task_id, seed=seed, method=method, ok=0, regret='', mean_cand='', err=str(e)[:60])

_POOLS = {}
def _init(pools):
    global _POOLS; _POOLS = pools

def probe():
    m, t = PROBLEMS[0]
    print(f"Probing HPOBench: model={m}, task_id={t}")
    b = _load_benchmark(m, t)
    print("benchmark type:", type(b))
    cs = b.get_configuration_space(seed=0)
    print("config space hyperparameters:")
    grid_size = 1
    for hp in _hplist(cs):
        ch = _choices_of(hp)
        kind = 'choices' if hasattr(hp, 'choices') else ('sequence' if hasattr(hp, 'sequence') else 'numeric')
        grid_size *= (len(ch) if ch is not None else 0) or 1
        print(f"   {hp.name:20s} [{kind}] enumerable={ch is not None} n={len(ch) if ch else '?'}  {ch if ch is not None else hp}")
    print(f"-> implied full-grid size (product of discrete axes): {grid_size}")
    cfg = cs.sample_configuration().get_dictionary()
    print("sample config:", cfg)
    res = b.objective_function(configuration=cfg)
    print("objective_function returns:", type(res), res if not isinstance(res, dict) else {k: res[k] for k in list(res)[:6]})
    # look for a BULK table (avoids one objective call per config)
    print("\npublic attributes (looking for a bulk grid/table):")
    print("  ", [a for a in dir(b) if not a.startswith('_')])
    for attr in ['table', 'data', 'df', 'dataset', '_data', 'metrics', 'results']:
        obj = getattr(b, attr, None)
        if obj is not None:
            shp = getattr(obj, 'shape', None) or (len(obj) if hasattr(obj, '__len__') else '?')
            print(f"  candidate bulk table: b.{attr}  type={type(obj).__name__}  shape/len={shp}")
            cols = getattr(obj, 'columns', None)
            if cols is not None: print(f"      columns: {list(cols)[:20]}")
    try:
        X, y = build_pool(m, t)
        print(f"BUILT POOL: {len(X)} configs, dim {X.shape[1]}, y in [{y.min():.3f},{y.max():.3f}] opt=1")
        print("If this looks right, run the full sweep. If not, paste this output and I'll fix build_pool.")
    except Exception as e:
        print("build_pool FAILED:", e, "\n-> paste the hyperparameter/return info above and I'll adapt it.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--probe', action='store_true')
    ap.add_argument('--seeds', type=int, default=15)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--analyze-only', action='store_true')
    a = ap.parse_args()
    if a.probe: return probe()
    if a.analyze_only: return analyze()
    print("Building pools (one per model x task)...")
    pools = {}
    for (m, t) in PROBLEMS:
        try:
            pools[(m, t)] = build_pool(m, t); print(f"  {m}/{t}: {len(pools[(m,t)][0])} configs")
        except Exception as e:
            print(f"  SKIP {m}/{t}: {e}")
    units = [(m, t, s, meth) for (m, t) in pools for s in range(a.seeds)
             for meth in ['random', 'tpe', 'rf_smbo', 'adaptive_k']]
    done = set()
    if RESULTS.exists():
        for r in csv.DictReader(open(RESULTS)):
            if r.get('ok') == '1': done.add((r['model'], r['task'], r['seed'], r['method']))
    todo = [u for u in units if (u[0], str(u[1]), str(u[2]), u[3]) not in done]
    print(f"pools {len(pools)}, units {len(units)}, todo {len(todo)}, workers {a.workers}")
    fields = ['model', 'task', 'seed', 'method', 'ok', 'regret', 'mean_cand', 'err']
    f = open(RESULTS, 'a', newline=''); w = csv.DictWriter(f, fieldnames=fields)
    if not done: w.writeheader()
    t0 = time.time(); c = 0
    with ProcessPoolExecutor(max_workers=a.workers, initializer=_init, initargs=(pools,)) as ex:
        for res in as_completed([ex.submit(run_unit, u) for u in todo]):
            row = res.result(); row.setdefault('err', ''); w.writerow(row); c += 1
            if c % 100 == 0:
                f.flush(); el = time.time()-t0
                print(f"  {c}/{len(todo)} {el:.0f}s ETA {(len(todo)-c)/(c/el):.0f}s", flush=True)
    f.close(); print(f"done {c} in {time.time()-t0:.0f}s"); analyze()

def analyze():
    from scipy import stats
    if not RESULTS.exists(): print("no results."); return
    rows = [r for r in csv.DictReader(open(RESULTS)) if r.get('ok') == '1']
    cell = collections.defaultdict(dict); cand = collections.defaultdict(list); base = collections.defaultdict(dict)
    for r in rows:
        cell[(r['model'], r['task'], r['seed'])][r['method']] = float(r['regret'])
        if r['method'] in ('rf_smbo', 'adaptive_k'): cand[r['method']].append(float(r['mean_cand']))
    keys = [k for k in cell if 'rf_smbo' in cell[k] and 'adaptive_k' in cell[k]]
    if not keys: print("no paired results yet."); return
    full = np.array([cell[k]['rf_smbo'] for k in keys]); res = np.array([cell[k]['adaptive_k'] for k in keys])
    d = full - res
    p = stats.wilcoxon(res, full).pvalue if np.any(np.abs(d) > 1e-12) else float('nan')
    print(f"\n=== HPOBench: restricted (K={K_RESTRICT}) vs full RF-SMBO, n={len(keys)} pairs ===")
    print(f"  full={full.mean():.4f} restricted={res.mean():.4f} median Δ={np.median(d):+.4f} "
          f"W/L={int((d>1e-9).sum())}/{int((d<-1e-9).sum())} Wilcoxon p={p:.3f} (non-inferiority: expect ns)")
    if cand['rf_smbo'] and cand['adaptive_k']:
        Nb, Kb = np.mean(cand['rf_smbo']), np.mean(cand['adaptive_k'])
        print(f"  cost: full N̄={Nb:.0f} vs restricted K̄={Kb:.0f} -> {Nb/Kb:.1f}x fewer")
    # vs random (sanity)
    rk = [k for k in cell if 'adaptive_k' in cell[k] and 'random' in cell[k]]
    if rk:
        a2 = np.array([cell[k]['adaptive_k'] for k in rk]); r2 = np.array([cell[k]['random'] for k in rk])
        print(f"  adaptive_k vs random: {a2.mean():.4f} vs {r2.mean():.4f} (surrogate value)")

if __name__ == '__main__':
    main()
