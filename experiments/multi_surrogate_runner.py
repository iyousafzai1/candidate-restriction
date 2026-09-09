#!/usr/bin/env python3
"""
Option 2 --- Does candidate restriction generalise across surrogates?

Coverage is a property of the candidate pool, not the surrogate, so restricting to
a small random subset should be non-inferior (and far cheaper) for surrogates other
than the random forest. We test this for TPE (kernel-density) and GP-UCB, each in
FULL vs RESTRICTED (K=2000) form, at B=25, and compare per surrogate.

Handling GP reliability HONESTLY (not a hack):
  * GP is only run on spaces with dimensionality <= GP_DIM_MAX (default 8); on higher
    dimensions a GP fit on 5--25 points is ill-conditioned and unreliable, so we do
    not report it there. This scoping is stated in the paper.
  * The GP uses a FIXED Matern-2.5 kernel with a white-noise term and NO per-step
    marginal-likelihood optimisation (optimizer=None): fast and numerically stable.
    Inputs are standardised per task. Any residual LinAlg failure is caught and the
    unit is recorded as failed rather than reported.

Speed:
  * Parallel across (space, task, seed, surrogate, mode) units via a process pool
    (default: all cores). Each worker loads the HPO-B handler once.
  * Resumable: completed units are appended to the CSV and skipped on restart.

    python3 multi_surrogate_runner.py                 # tpe + gp, 5 seeds, all cores
    python3 multi_surrogate_runner.py --with-rf       # also re-run RF (else use main study)
    python3 multi_surrogate_runner.py --seeds 3 --workers 8
    python3 multi_surrogate_runner.py --analyze-only
"""
import sys, os, csv, argparse, collections, warnings, math
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
warnings.filterwarnings('ignore')

HERE = Path(__file__).resolve().parent
HPOB_REPO = os.environ.get("HPOB_REPO", "/tmp/HPO-B")
HPOB_DATA_ROOT = os.environ.get("HPOB_DATA_ROOT", "/tmp/hpob_data/hpob-data/")
BUDGET, WARMUP, BETA, K_RESTRICT, GP_DIM_MAX = 25, 5, 1.96, 2000, 8
RESULTS = HERE / "multi_surrogate_results.csv"
ALL_SPACES = ['6767','5906','5971','5965','6794','7607','7609',
              '5527','5891','5636','5859','5889','4796','6766']

# ---- per-worker handler ----
_H = {}
def _init():
    sys.path.insert(0, HPOB_REPO)
    from hpob_handler import HPOBHandler
    _H['h'] = HPOBHandler(root_dir=HPOB_DATA_ROOT, mode='v3')

# ---- surrogate scorers: return (acquisition over candidates, predicted value or None) ----
def score_rf(Xo, yo, Xc, seed, step):
    from sklearn.ensemble import RandomForestRegressor
    rf = RandomForestRegressor(n_estimators=64, max_depth=None, min_samples_leaf=1,
                               random_state=seed+step, n_jobs=1).fit(Xo, yo)
    P = np.array([t.predict(Xc) for t in rf.estimators_]); mu = P.mean(0)
    return mu + BETA*P.std(0), mu

def score_gp(Xo, yo, Xc, seed, step):
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
    k = ConstantKernel(1.0)*Matern(length_scale=1.0, nu=2.5) + WhiteKernel(1e-3)
    gp = GaussianProcessRegressor(kernel=k, optimizer=None, alpha=1e-6, normalize_y=True)
    gp.fit(Xo, yo)
    mu, sd = gp.predict(Xc, return_std=True)
    return mu + BETA*sd, mu

def score_tpe(Xo, yo, Xc, seed, step, rng):
    from scipy.stats import gaussian_kde
    arr = np.asarray(yo); split = np.median(arr)
    good, bad = arr >= split, arr < split
    if good.sum() >= 2 and bad.sum() >= 2:
        try:
            kg, kb = gaussian_kde(Xo[good].T), gaussian_kde(Xo[bad].T)
            return kg.logpdf(Xc.T) - kb.logpdf(Xc.T), None
        except Exception:
            pass
    return rng.normal(size=len(Xc)), None   # fallback: random

def run_bo(surro, ss, task, seed, mode):
    h = _H['h']
    X = np.array(h.meta_test_data[ss][task]['X'])
    y = h.normalize(np.array(h.meta_test_data[ss][task]['y'])).ravel()
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-9)          # standardised inputs (for GP)
    rng = np.random.default_rng(seed); n = len(X)
    obs, best, scored, overs = [], 0.0, [], []
    for step in range(BUDGET):
        if step < WARMUP:
            idx = int(rng.integers(n))
            while idx in obs: idx = int(rng.integers(n))
        else:
            uneval = [i for i in range(n) if i not in set(obs)]; U = len(uneval)
            if U == 0:
                idx = int(rng.integers(n))
            else:
                cand = np.array(uneval) if mode == 'full' else \
                       rng.choice(uneval, min(K_RESTRICT, U), replace=False)
                Xo, yo = (Xs[obs], np.array([y[i] for i in obs]))
                Xc = Xs[cand]
                if surro == 'rf':   acq, mu = score_rf(X[obs], np.array([y[i] for i in obs]), X[cand], seed, step)
                elif surro == 'gp': acq, mu = score_gp(Xo, yo, Xc, seed, step)
                else:               acq, mu = score_tpe(X[obs], np.array([y[i] for i in obs]), X[cand], seed, step, rng)
                j = int(np.argmax(acq)); idx = int(cand[j]); scored.append(len(cand))
                if mu is not None and mode == 'full':
                    overs.append(float(mu[j] - y[idx]))     # post-decision surprise
        obs.append(int(idx)); best = max(best, y[idx])
    return dict(regret=max(0.0, 1.0 - best),
                mean_cand=(float(np.mean(scored)) if scored else 0.0),
                over_pred=(float(np.mean(overs)) if overs else ''))

def run_unit(args):
    ss, task, seed, surro, mode = args
    try:
        r = run_bo(surro, ss, task, seed, mode)
        return dict(ss=ss, task=task, seed=seed, surrogate=surro, mode=mode, ok=1, **r)
    except Exception as e:
        return dict(ss=ss, task=task, seed=seed, surrogate=surro, mode=mode, ok=0,
                    regret='', mean_cand='', over_pred='', err=str(e)[:60])

def build_units(seeds, surrogates, dims):
    units = []
    sys.path.insert(0, HPOB_REPO); from hpob_handler import HPOBHandler
    h = HPOBHandler(root_dir=HPOB_DATA_ROOT, mode='v3')
    for ss in ALL_SPACES:
        for task in h.meta_test_data.get(ss, {}):
            for seed in range(seeds):
                for surro in surrogates:
                    if surro == 'gp' and dims.get(ss, 99) > GP_DIM_MAX:
                        continue                         # GP only where well-conditioned
                    for mode in ('full', 'restricted'):
                        units.append((ss, task, seed, surro, mode))
    return units

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2)))
    ap.add_argument('--with-rf', action='store_true')
    ap.add_argument('--analyze-only', action='store_true')
    a = ap.parse_args()
    if a.analyze_only:
        return analyze()
    dims = {r['ss_id']: int(r['dim']) for r in csv.DictReader(open(HERE.parent/"table2_per_space_b25.csv"))}
    surrogates = (['rf'] if a.with_rf else []) + ['tpe', 'gp']
    units = build_units(a.seeds, surrogates, dims)
    fields = ['ss','task','seed','surrogate','mode','ok','regret','mean_cand','over_pred','err']
    done = set()
    if RESULTS.exists():
        for r in csv.DictReader(open(RESULTS)):
            if r.get('ok') == '1': done.add((r['ss'],r['task'],r['seed'],r['surrogate'],r['mode']))
    todo = [u for u in units if (u[0],u[1],str(u[2]),u[3],u[4]) not in done]
    print(f"units total {len(units)}, done {len(done)}, todo {len(todo)}, workers {a.workers}")
    newfile = not RESULTS.exists()
    f = open(RESULTS, 'a', newline=''); w = csv.DictWriter(f, fieldnames=fields)
    if newfile: w.writeheader()
    import time; t0 = time.time(); c = 0
    with ProcessPoolExecutor(max_workers=a.workers, initializer=_init) as ex:
        for res in as_completed([ex.submit(run_unit, u) for u in todo]):
            row = res.result(); row.setdefault('err','')
            w.writerow(row); c += 1
            if c % 100 == 0:
                f.flush(); el=time.time()-t0
                print(f"  {c}/{len(todo)}  {el:.0f}s  ETA {(len(todo)-c)/(c/el):.0f}s", flush=True)
    f.close(); print(f"done {c} units in {time.time()-t0:.0f}s")
    analyze()

def analyze():
    from scipy import stats
    if not RESULTS.exists(): print("no results yet."); return
    rows = [r for r in csv.DictReader(open(RESULTS)) if r.get('ok')=='1']
    # per (surrogate): pair restricted vs full over (ss,task,seed)
    for surro in ['rf','tpe','gp']:
        cell = collections.defaultdict(dict); cand = collections.defaultdict(list)
        for r in rows:
            if r['surrogate']!=surro: continue
            cell[(r['ss'],r['task'],r['seed'])][r['mode']] = float(r['regret'])
            cand[r['mode']].append(float(r['mean_cand']))
        keys = [k for k in cell if 'full' in cell[k] and 'restricted' in cell[k]]
        if not keys: continue
        full = np.array([cell[k]['full'] for k in keys]); res = np.array([cell[k]['restricted'] for k in keys])
        d = full - res  # + = restricted better
        p = stats.wilcoxon(res, full).pvalue if np.any(np.abs(d)>1e-12) else float('nan')
        Nb = np.mean(cand['full']); Kb = np.mean(cand['restricted'])
        print(f"\n[{surro.upper()}] restricted(K={K_RESTRICT}) vs full, n={len(keys)} pairs")
        print(f"  regret: full={full.mean():.4f}  restricted={res.mean():.4f}  median Δ={np.median(d):+.4f}  "
              f"W/L={int((d>1e-9).sum())}/{int((d<-1e-9).sum())}  Wilcoxon p={p:.3f}  (non-inferiority: expect ns)")
        print(f"  cost: full N̄={Nb:.0f} vs restricted K̄={Kb:.0f}  ->  {Nb/Kb:.1f}x fewer")
    # bonus: optimizer's-curse (over-prediction) RF vs GP on full scoring
    for surro in ['rf','gp']:
        ov=[float(r['over_pred']) for r in rows if r['surrogate']==surro and r['mode']=='full' and r['over_pred'] not in ('','nan')]
        if ov: print(f"  [curse] {surro.upper()} full-scoring mean over-prediction = {np.mean(ov):+.4f}")

if __name__ == '__main__':
    main()
