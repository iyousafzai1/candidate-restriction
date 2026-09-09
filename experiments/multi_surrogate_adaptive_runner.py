#!/usr/bin/env python3
"""
Adaptive-K with TPE and GP-UCB surrogates (reviewer R2.2).

Extends multi_surrogate_runner.py to test the ADAPTIVE schedule (not just fixed
K=2000) with TPE and GP-UCB. The adaptive schedule uses each surrogate's own
uncertainty proxy:
  - RF:  tree-dispersion std (existing)
  - TPE: acquisition-score std over the probe (proxy for model uncertainty)
  - GP:  predictive std from the GP posterior (natural uncertainty)

Same frozen constants as the main study: F=2500, K_MIN=1000, C_GROW=0.75.

    HPOB_REPO=/path/to/HPO-B \
    HPOB_DATA_ROOT=/path/to/hpob-data/ \
    python3 multi_surrogate_adaptive_runner.py --seeds 5 --workers 4

    python3 multi_surrogate_adaptive_runner.py --analyze-only
"""
import sys, os, csv, argparse, collections, warnings, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
warnings.filterwarnings('ignore')

HERE = Path(__file__).resolve().parent
HPOB_REPO = os.environ.get("HPOB_REPO", "/tmp/HPO-B")
HPOB_DATA_ROOT = os.environ.get("HPOB_DATA_ROOT", "/tmp/hpob_data/hpob-data/")
BUDGET, WARMUP, BETA, K_RESTRICT, GP_DIM_MAX = 25, 5, 1.96, 2000, 8
AK_FULL_FLOOR, AK_K_MIN, AK_C_GROW = 2500, 1000, 0.75
RESULTS = HERE / "multi_surrogate_adaptive_results.csv"
ALL_SPACES = ['6767','5906','5971','5965','6794','7607','7609',
              '5527','5891','5636','5859','5889','4796','6766']

_H = {}
def _init():
    sys.path.insert(0, HPOB_REPO)
    from hpob_handler import HPOBHandler
    _H['h'] = HPOBHandler(root_dir=HPOB_DATA_ROOT, mode='v3')

def score_rf(Xo, yo, Xc, seed, step):
    from sklearn.ensemble import RandomForestRegressor
    rf = RandomForestRegressor(n_estimators=64, max_depth=None, min_samples_leaf=1,
                               random_state=seed+step, n_jobs=1).fit(Xo, yo)
    P = np.array([t.predict(Xc) for t in rf.estimators_]); mu = P.mean(0)
    return mu + BETA*P.std(0), P.std(0).mean()

def score_gp(Xo, yo, Xc, seed, step):
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
    k = ConstantKernel(1.0)*Matern(length_scale=1.0, nu=2.5) + WhiteKernel(1e-3)
    gp = GaussianProcessRegressor(kernel=k, optimizer=None, alpha=1e-6, normalize_y=True)
    gp.fit(Xo, yo)
    mu, sd = gp.predict(Xc, return_std=True)
    return mu + BETA*sd, sd.mean()

def score_tpe(Xo, yo, Xc, seed, step, rng):
    from scipy.stats import gaussian_kde
    arr = np.asarray(yo); split = np.median(arr)
    good, bad = arr >= split, arr < split
    if good.sum() >= 2 and bad.sum() >= 2:
        try:
            kg, kb = gaussian_kde(Xo[good].T), gaussian_kde(Xo[bad].T)
            acq = kg.logpdf(Xc.T) - kb.logpdf(Xc.T)
            dispersion = float(np.std(acq))
            return acq, dispersion
        except Exception:
            pass
    return rng.normal(size=len(Xc)), 1.0

def run_bo(surro, ss, task, seed, mode):
    h = _H['h']
    X = np.array(h.meta_test_data[ss][task]['X'])
    y = h.normalize(np.array(h.meta_test_data[ss][task]['y'])).ravel()
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-9)
    rng = np.random.default_rng(seed); n = len(X)
    obs, best, scored = [], 0.0, []
    s0 = None
    for step in range(BUDGET):
        if step < WARMUP:
            idx = int(rng.integers(n))
            while idx in obs: idx = int(rng.integers(n))
        else:
            uneval = [i for i in range(n) if i not in set(obs)]; U = len(uneval)
            if U == 0:
                idx = int(rng.integers(n))
            else:
                if mode == 'full':
                    cand = np.array(uneval)
                elif mode == 'restricted':
                    cand = rng.choice(uneval, min(K_RESTRICT, U), replace=False)
                elif mode == 'adaptive':
                    if U <= AK_FULL_FLOOR:
                        cand = np.array(uneval)
                    else:
                        probe = rng.choice(uneval, min(AK_K_MIN, U), replace=False)
                        Xo = X[obs] if surro != 'gp' else Xs[obs]
                        yo = np.array([y[i] for i in obs])
                        Xp = X[probe] if surro != 'gp' else Xs[probe]
                        if surro == 'rf':
                            _, s_hat = score_rf(Xo, yo, Xp, seed, step)
                        elif surro == 'gp':
                            _, s_hat = score_gp(Xo, yo, Xp, seed, step)
                        else:
                            _, s_hat = score_tpe(Xo, yo, Xp, seed, step, rng)
                        s_hat = max(s_hat, 1e-12)
                        if s0 is None: s0 = s_hat
                        K_t = int(min(U, max(AK_K_MIN, round(AK_K_MIN * (s0 / s_hat) ** AK_C_GROW))))
                        if K_t <= len(probe):
                            cand = probe
                        else:
                            remaining = list(set(uneval) - set(probe.tolist()))
                            extra = rng.choice(remaining, min(K_t - len(probe), len(remaining)), replace=False)
                            cand = np.concatenate([probe, extra])
                else:
                    raise ValueError(f"Unknown mode: {mode}")

                Xo = X[obs] if surro != 'gp' else Xs[obs]
                yo = np.array([y[i] for i in obs])
                Xc = X[cand] if surro != 'gp' else Xs[cand]
                if surro == 'rf':   acq, _ = score_rf(Xo, yo, Xc, seed, step)
                elif surro == 'gp': acq, _ = score_gp(Xo, yo, Xc, seed, step)
                else:               acq, _ = score_tpe(Xo, yo, Xc, seed, step, rng)
                j = int(np.argmax(acq)); idx = int(cand[j]); scored.append(len(cand))
        obs.append(int(idx)); best = max(best, y[idx])
    return dict(regret=max(0.0, 1.0 - best),
                mean_cand=(float(np.mean(scored)) if scored else 0.0))

def run_unit(args):
    ss, task, seed, surro, mode = args
    try:
        r = run_bo(surro, ss, task, seed, mode)
        return dict(ss=ss, task=task, seed=seed, surrogate=surro, mode=mode, ok=1, **r)
    except Exception as e:
        return dict(ss=ss, task=task, seed=seed, surrogate=surro, mode=mode, ok=0,
                    regret='', mean_cand='', err=str(e)[:80])

def build_units(seeds, surrogates, dims):
    units = []
    sys.path.insert(0, HPOB_REPO); from hpob_handler import HPOBHandler
    h = HPOBHandler(root_dir=HPOB_DATA_ROOT, mode='v3')
    for ss in ALL_SPACES:
        for task in h.meta_test_data.get(ss, {}):
            for seed in range(seeds):
                for surro in surrogates:
                    if surro == 'gp' and dims.get(ss, 99) > GP_DIM_MAX:
                        continue
                    for mode in ('full', 'restricted', 'adaptive'):
                        units.append((ss, task, seed, surro, mode))
    return units

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--workers', type=int, default=max(1, min(4, (os.cpu_count() or 2))))
    ap.add_argument('--surrogates', nargs='+', default=['tpe', 'gp'], choices=['rf','tpe','gp'])
    ap.add_argument('--analyze-only', action='store_true')
    a = ap.parse_args()
    if a.analyze_only:
        return analyze()
    dims = {r['ss_id']: int(r['dim']) for r in csv.DictReader(open(HERE.parent/"table2_per_space_b25.csv"))}
    units = build_units(a.seeds, a.surrogates, dims)
    fields = ['ss','task','seed','surrogate','mode','ok','regret','mean_cand','err']
    done = set()
    if RESULTS.exists():
        for r in csv.DictReader(open(RESULTS)):
            if r.get('ok') == '1': done.add((r['ss'],r['task'],r['seed'],r['surrogate'],r['mode']))
    todo = [u for u in units if (u[0],u[1],str(u[2]),u[3],u[4]) not in done]
    print(f"units total {len(units)}, done {len(done)}, todo {len(todo)}, workers {a.workers}")
    newfile = not RESULTS.exists()
    f = open(RESULTS, 'a', newline=''); w = csv.DictWriter(f, fieldnames=fields)
    if newfile: w.writeheader()
    t0 = time.time(); c = 0
    with ProcessPoolExecutor(max_workers=a.workers, initializer=_init) as ex:
        for res in as_completed([ex.submit(run_unit, u) for u in todo]):
            row = res.result(); row.setdefault('err','')
            w.writerow(row); c += 1
            if c % 50 == 0:
                f.flush(); el=time.time()-t0
                print(f"  {c}/{len(todo)}  {el:.0f}s  ETA {(len(todo)-c)/(c/el) if c else 0:.0f}s", flush=True)
    f.close(); print(f"done {c} units in {time.time()-t0:.0f}s")
    analyze()

def analyze():
    from scipy import stats
    if not RESULTS.exists(): print("no results yet."); return
    rows = [r for r in csv.DictReader(open(RESULTS)) if r.get('ok')=='1']
    for surro in ['rf','tpe','gp']:
        cell = collections.defaultdict(dict); cand = collections.defaultdict(list)
        for r in rows:
            if r['surrogate']!=surro: continue
            cell[(r['ss'],r['task'],r['seed'])][r['mode']] = float(r['regret'])
            cand[r['mode']].append(float(r['mean_cand']))
        for mode_a, mode_b in [('restricted','full'), ('adaptive','full'), ('adaptive','restricted')]:
            keys = [k for k in cell if mode_a in cell[k] and mode_b in cell[k]]
            if not keys: continue
            a = np.array([cell[k][mode_a] for k in keys]); b = np.array([cell[k][mode_b] for k in keys])
            d = b - a
            p = stats.wilcoxon(a, b).pvalue if np.any(np.abs(d)>1e-12) else float('nan')
            Na = np.mean(cand[mode_a]) if cand[mode_a] else 0; Nb = np.mean(cand[mode_b]) if cand[mode_b] else 0
            print(f"\n[{surro.upper()}] {mode_a} vs {mode_b}, n={len(keys)} pairs")
            print(f"  regret: {mode_b}={b.mean():.4f}  {mode_a}={a.mean():.4f}  "
                  f"W/L={int((d>1e-9).sum())}/{int((d<-1e-9).sum())}  Wilcoxon p={p:.3f}")
            if Nb > 0 and Na > 0:
                print(f"  cost: {mode_b} N̄={Nb:.0f} vs {mode_a} K̄={Na:.0f}  ->  {Nb/Na:.1f}x fewer")

if __name__ == '__main__':
    main()
