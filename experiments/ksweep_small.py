#!/usr/bin/env python3
"""
Completion run for the saturation curve: the RISING part below K=50, at multiple
budgets, to pin the knee and defend against the "B=25 artifact" objection.

For K in {1,2,3,5,8,13,20,35,50} (small; K=1 is essentially random selection),
run RF-SMBO restricted to K random candidates per step and record the best-found
value at budgets B in {5,10,25}. Parallel across (space,task,seed,K); resumable.

    python3 ksweep_small.py                # all cores
    python3 ksweep_small.py --seeds 5 --workers 8
    python3 ksweep_small.py --analyze-only
Merge with ksweep_results.csv (the K>=50 part you already have) for the full curve.
"""
import sys, os, csv, argparse, collections, warnings, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
warnings.filterwarnings('ignore')

HERE = Path(__file__).resolve().parent
HPOB_REPO = os.environ.get("HPOB_REPO", "/tmp/HPO-B")
HPOB_DATA_ROOT = os.environ.get("HPOB_DATA_ROOT", "/tmp/hpob_data/hpob-data/")
BUDGETS, WARMUP, BETA = [5, 10, 25], 5, 1.96
KGRID_SMALL = [1, 2, 3, 5, 8, 13, 20, 35, 50]
ALL_SPACES = ['6767','5906','5971','5965','6794','7607','7609',
              '5527','5891','5636','5859','5889','4796','6766']
RESULTS = HERE / "ksweep_small.csv"
_H = {}
def _init():
    sys.path.insert(0, HPOB_REPO)
    from hpob_handler import HPOBHandler
    _H['h'] = HPOBHandler(root_dir=HPOB_DATA_ROOT, mode='v3')

def run_unit(args):
    ss, task, seed, K = args
    try:
        from sklearn.ensemble import RandomForestRegressor
        h = _H['h']
        X = np.array(h.meta_test_data[ss][task]['X'])
        y = h.normalize(np.array(h.meta_test_data[ss][task]['y'])).ravel()
        rng = np.random.default_rng(seed); n = len(X)
        obs, traj, scored = [], [], []
        best = 0.0
        for step in range(max(BUDGETS)):
            if step < WARMUP:
                idx = int(rng.integers(n))
                while idx in obs: idx = int(rng.integers(n))
            else:
                uneval = [i for i in range(n) if i not in set(obs)]; U = len(uneval)
                if U == 0:
                    idx = int(rng.integers(n))
                else:
                    k = min(K, U); cand = rng.choice(uneval, k, replace=False)
                    P = np.array([t.predict(X[cand]) for t in RandomForestRegressor(
                        n_estimators=64, max_depth=None, min_samples_leaf=1,
                        random_state=seed+step, n_jobs=1).fit(X[obs], np.array([y[i] for i in obs])).estimators_])
                    idx = int(cand[int(np.argmax(P.mean(0) + BETA*P.std(0)))]); scored.append(k)
            obs.append(int(idx)); best = max(best, y[idx]); traj.append(best)
        row = dict(ss=ss, task=task, seed=seed, K=K, ok=1,
                   mean_cand=(float(np.mean(scored)) if scored else 0.0))
        for b in BUDGETS: row[f'regret_B{b}'] = max(0.0, 1.0 - traj[min(b, len(traj))-1])
        return row
    except Exception as e:
        return dict(ss=ss, task=task, seed=seed, K=K, ok=0, mean_cand='',
                    **{f'regret_B{b}': '' for b in BUDGETS}, err=str(e)[:50])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2)))
    ap.add_argument('--analyze-only', action='store_true')
    a = ap.parse_args()
    if a.analyze_only: return analyze()
    sys.path.insert(0, HPOB_REPO); from hpob_handler import HPOBHandler
    h = HPOBHandler(root_dir=HPOB_DATA_ROOT, mode='v3')
    units = [(ss, task, seed, K) for ss in ALL_SPACES
             for task in h.meta_test_data.get(ss, {}) for seed in range(a.seeds) for K in KGRID_SMALL]
    fields = ['ss','task','seed','K','ok','mean_cand'] + [f'regret_B{b}' for b in BUDGETS] + ['err']
    done = set()
    if RESULTS.exists():
        for r in csv.DictReader(open(RESULTS)):
            if r.get('ok') == '1': done.add((r['ss'], r['task'], r['seed'], r['K']))
    todo = [u for u in units if (u[0],u[1],str(u[2]),str(u[3])) not in done]
    print(f"units {len(units)}, done {len(done)}, todo {len(todo)}, workers {a.workers}")
    f = open(RESULTS, 'a', newline=''); w = csv.DictWriter(f, fieldnames=fields)
    if not done: w.writeheader()
    t0 = time.time(); c = 0
    with ProcessPoolExecutor(max_workers=a.workers, initializer=_init) as ex:
        for res in as_completed([ex.submit(run_unit, u) for u in todo]):
            row = res.result(); row.setdefault('err', ''); w.writerow(row); c += 1
            if c % 200 == 0:
                f.flush(); el = time.time()-t0
                print(f"  {c}/{len(todo)} {el:.0f}s ETA {(len(todo)-c)/(c/el):.0f}s", flush=True)
    f.close(); print(f"done {c} units in {time.time()-t0:.0f}s"); analyze()

def analyze():
    if not RESULTS.exists(): print("no small-K results yet."); return
    rows = [r for r in csv.DictReader(open(RESULTS)) if r.get('ok') == '1']
    print("\n=== rising part of the saturation curve (small K), by budget ===")
    print(f"{'K':>5}" + "".join(f"{'reg_B'+str(b):>10}" for b in BUDGETS) + f"{'mean_cand':>11}")
    byK = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        for b in BUDGETS: byK[int(r['K'])][b].append(float(r[f'regret_B{b}']))
        byK[int(r['K'])]['cand'].append(float(r['mean_cand']))
    for K in sorted(byK):
        print(f"{K:>5}" + "".join(f"{np.mean(byK[K][b]):>10.4f}" for b in BUDGETS) + f"{np.mean(byK[K]['cand']):>11.1f}")
    print("\nMerge with ksweep_results.csv (K>=50, B=25) for the full curve figure.")

if __name__ == '__main__':
    main()
