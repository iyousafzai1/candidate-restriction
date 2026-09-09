#!/usr/bin/env python3
"""
E2 -- Direct test of the COVERAGE mechanism (answers reviewer Concern 8).

The paper argues coverage (a random subset almost always contains a high-acquisition
candidate), not bias reduction, is why restriction loses little. The reviewer rightly
notes that non-inferiority does NOT prove this: a subset could contain a good point
that the acquisition fails to pick. This script measures coverage *directly* in
acquisition space. At every step where restriction is active (U>F) of a real
adaptive-K run on HPO-B, it computes the full-pool acquisition landscape and asks:

  1. gap_norm   = (max acq over FULL pool - max acq over the K-subset) / acq_range
                  -> how much acquisition value the restricted set forfeits (0 = perfect coverage).
  2. retains_top = 1 if the subset contains a candidate within 1% of the full-pool
                  acquisition maximum (coverage event).
  3. sel_obj_rank = percentile rank of the selected point's TRUE objective among the
                  unevaluated pool (does a high-acquisition pick correspond to a good config?).

Per task it then correlates the mean acquisition gap with the realised regret
difference (adaptive-K vs full scoring), testing whether the rare coverage misses are
what cause the large-pool losses.

Needs HPO-B v3 on the machine (same env vars as adaptive_k_runner.py):
    export HPOB_DATA_ROOT=/path/to/hpob-data/         # dir with the v3 meta data
    export HPOB_REPO=/path/to/HPO-B                    # repo providing hpob_handler.py
    pip install scikit-learn numpy scipy
    python3 coverage_mechanism.py --probe             # load ONE task, print shapes, verify
    python3 coverage_mechanism.py --seeds 5 --workers 6
    python3 coverage_mechanism.py --analyze-only
Frozen adaptive-K constants match the paper: F=2500, K_MIN=1000, C_GROW=0.75.
"""
import os, sys, csv, argparse, collections, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "coverage_mechanism_raw.csv"
HPOB_REPO = os.environ.get("HPOB_REPO", "/tmp/HPO-B")
HPOB_DATA_ROOT = os.environ.get("HPOB_DATA_ROOT", "/tmp/hpob_data/hpob-data/")

BUDGET, WARMUP, BETA, RF_N = 25, 5, 1.96, 64
F, K_MIN, C_GROW = 2500, 1000, 0.75            # frozen adaptive-K constants (paper values)
NEAR_TOP = 0.01                                 # "retains top" tolerance: within 1% of acq range
ALL_SPACES = ['6767','5906','5971','5965','6794','7607','7609',
              '5527','5891','5636','5859','5889','4796','6766']


def _acq(rf, X):
    P = np.array([t.predict(X) for t in rf.estimators_]); return P.mean(0) + BETA * P.std(0)


def run_task(X, y_norm, seed, restrict):
    """One BO run. If restrict: adaptive-K with per-step coverage instrumentation.
    If not: full-candidate rf_smbo. Returns (final_regret, list_of_step_records)."""
    from sklearn.ensemble import RandomForestRegressor
    rng = np.random.default_rng(seed); n = len(X)
    obs, ov, best, s0, recs = [], [], 0.0, None, []
    for step in range(BUDGET):
        if step < WARMUP:
            idx = int(rng.integers(n))
            while idx in obs: idx = int(rng.integers(n))
        else:
            rf = RandomForestRegressor(n_estimators=RF_N, max_depth=None, min_samples_leaf=1,
                                       random_state=seed + step, n_jobs=1).fit(X[obs], np.array(ov))
            uneval = [i for i in range(n) if i not in set(obs)]; U = len(uneval)
            if U == 0:
                idx = int(rng.integers(n))
            elif (not restrict) or U <= F:
                a = _acq(rf, X[uneval]); idx = uneval[int(np.argmax(a))]
            else:
                probe = rng.choice(uneval, min(K_MIN, U), replace=False)
                pstd = np.array([t.predict(X[probe]) for t in rf.estimators_]).std(0)
                s_hat = float(pstd.mean()) + 1e-12
                if s0 is None: s0 = s_hat
                K_t = int(min(U, max(K_MIN, round(K_MIN * (s0 / s_hat) ** C_GROW))))
                if K_t <= len(probe):
                    cand = probe
                else:
                    rem = list(set(uneval) - set(probe.tolist()))
                    extra = rng.choice(rem, min(K_t - len(probe), len(rem)), replace=False)
                    cand = np.concatenate([probe, extra])
                a_cand = _acq(rf, X[cand])
                idx = int(cand[int(np.argmax(a_cand))])
                # ---- instrumentation: full-pool acquisition landscape (the expensive, honest part)
                a_full = _acq(rf, X[uneval])
                fmax, fmin = float(a_full.max()), float(a_full.min())
                rng_a = fmax - fmin + 1e-12
                rmax = float(a_cand.max())
                gap_norm = (fmax - rmax) / rng_a
                retains_top = int(rmax >= fmax - NEAR_TOP * rng_a)
                # selected point's true-objective percentile among unevaluated (1 = best)
                yv = y_norm[uneval]; sel_obj_rank = float((yv <= y_norm[idx]).mean())
                recs.append((step, U, len(cand), round(gap_norm, 5), retains_top, round(sel_obj_rank, 4)))
        obs.append(int(idx)); ov.append(float(y_norm[idx])); best = max(best, float(y_norm[idx]))
    return max(0.0, 1.0 - best), recs


def _load(hdlr, ss, task):
    X = np.array(hdlr.meta_test_data[ss][task]['X'])
    y = hdlr.normalize(np.array(hdlr.meta_test_data[ss][task]['y'])).ravel()
    return X, y


_UNITS = None
def _init(units):
    global _UNITS; _UNITS = units

def work(args):
    ss, task, seed = args
    X, y = _UNITS[(ss, task)]
    reg_r, recs = run_task(X, y, seed, restrict=True)
    reg_f, _ = run_task(X, y, seed, restrict=False)
    return ss, task, seed, reg_r, reg_f, recs


def probe():
    sys.path.insert(0, HPOB_REPO); from hpob_handler import HPOBHandler
    hdlr = HPOBHandler(root_dir=HPOB_DATA_ROOT, mode='v3')
    ss = ALL_SPACES[0]; task = list(hdlr.meta_test_data[ss].keys())[0]
    X, y = _load(hdlr, ss, task)
    print(f"OK: space {ss} task {task}: X {X.shape}, y in [{y.min():.3f},{y.max():.3f}]")
    reg, recs = run_task(X, y, 0, restrict=True)
    print(f"adaptive-K run: final regret {reg:.4f}, {len(recs)} active (restricted) steps")
    if recs:
        print("sample step record (step,U,K,gap_norm,retains_top,sel_obj_rank):", recs[0])
        print("-> if gap_norm is small and retains_top=1 on most steps, coverage holds.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--probe', action='store_true'); ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--workers', type=int, default=6); ap.add_argument('--analyze-only', action='store_true')
    a = ap.parse_args()
    if a.probe: return probe()
    if a.analyze_only: return analyze()
    sys.path.insert(0, HPOB_REPO); from hpob_handler import HPOBHandler
    hdlr = HPOBHandler(root_dir=HPOB_DATA_ROOT, mode='v3')
    units = {}
    for ss in ALL_SPACES:
        for task in hdlr.meta_test_data.get(ss, {}):
            units[(ss, task)] = _load(hdlr, ss, task)
    jobs = [(ss, task, s) for (ss, task) in units for s in range(a.seeds)]
    done = set()
    if RESULTS.exists():
        for r in csv.DictReader(open(RESULTS)):
            done.add((r['ss'], r['task'], r['seed']))
    todo = [j for j in jobs if (j[0], j[1], str(j[2])) not in done]
    print(f"units {len(units)}, jobs {len(jobs)}, todo {len(todo)}, workers {a.workers}")
    fields = ['ss', 'task', 'seed', 'step', 'U', 'K', 'gap_norm', 'retains_top', 'sel_obj_rank',
              'reg_restrict', 'reg_full']
    f = open(RESULTS, 'a', newline=''); w = csv.DictWriter(f, fieldnames=fields)
    if not done: w.writeheader()
    t0 = time.time(); c = 0
    with ProcessPoolExecutor(max_workers=a.workers, initializer=_init, initargs=(units,)) as ex:
        for fut in as_completed([ex.submit(work, j) for j in todo]):
            ss, task, seed, reg_r, reg_f, recs = fut.result()
            for (step, U, K, gap, ret, rank) in recs:
                w.writerow(dict(ss=ss, task=task, seed=seed, step=step, U=U, K=K, gap_norm=gap,
                                retains_top=ret, sel_obj_rank=rank, reg_restrict=reg_r, reg_full=reg_f))
            if not recs:  # task never activated restriction (small pool) -- still record the pair
                w.writerow(dict(ss=ss, task=task, seed=seed, step='', U='', K='', gap_norm='',
                                retains_top='', sel_obj_rank='', reg_restrict=reg_r, reg_full=reg_f))
            c += 1
            if c % 20 == 0: f.flush(); print(f"  {c}/{len(todo)} {time.time()-t0:.0f}s", flush=True)
    f.close(); print(f"done {c} tasks-seeds in {time.time()-t0:.0f}s"); analyze()


def analyze():
    from scipy import stats
    if not RESULTS.exists(): print("no results"); return
    rows = [r for r in csv.DictReader(open(RESULTS)) if r['gap_norm'] not in ('', None)]
    if not rows: print("no active-step records"); return
    gap = np.array([float(r['gap_norm']) for r in rows])
    ret = np.array([int(r['retains_top']) for r in rows])
    rank = np.array([float(r['sel_obj_rank']) for r in rows])
    print(f"\n=== E2 coverage mechanism: {len(rows)} active (restricted) steps ===")
    print(f"  acquisition gap forfeited by restriction: mean {gap.mean():.4f}, median {np.median(gap):.4f}, 95th pct {np.percentile(gap,95):.4f}")
    print(f"  steps where subset retains a within-1% top-acquisition candidate: {100*ret.mean():.1f}%")
    print(f"  selected point's true-objective percentile (1=best): mean {rank.mean():.3f}")
    # per-task: correlate mean acq gap with realised regret loss (adaptive - full)
    per = collections.defaultdict(list); reg = {}
    for r in rows:
        per[(r['ss'], r['task'], r['seed'])].append(float(r['gap_norm']))
        reg[(r['ss'], r['task'], r['seed'])] = (float(r['reg_restrict']), float(r['reg_full']))
    mg = np.array([np.mean(v) for v in per.values()])
    dloss = np.array([reg[k][0] - reg[k][1] for k in per])   # +ve = restriction worse
    if len(mg) > 3 and np.std(mg) > 0:
        rho, p = stats.spearmanr(mg, dloss)
        print(f"  per-run: Spearman(mean acq gap, regret loss) = {rho:+.3f} (p={p:.3f}, n={len(mg)})")
        print("  -> positive corr means the rare coverage misses are what drive the large-pool losses.")


if __name__ == '__main__':
    main()
