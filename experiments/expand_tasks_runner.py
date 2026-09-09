#!/usr/bin/env python3
"""
Power lever: enlarge the held-out evaluation set with HPO-B's meta-VALIDATION
tasks (which this cold-start, fixed-constant method never uses for tuning), so
the paired test against rf_smbo has more tasks and can move from "directional"
to significant -- without a new benchmark and without touching the method.

Runs all five methods (random, tpe, rf_smbo, rf_candidate_2k, adaptive_k) on the
validation tasks x 20 seeds x {5,10,25}, writing into the SAME phase2_cache
(cache keys include task_id, so new tasks simply add files; existing test-task
cells are untouched). Then re-runs the paired analysis three ways:
    test-only (the primary, pre-registered split), validation-only, and combined.

    python3 expand_tasks_runner.py                # run validation tasks + analyze
    python3 expand_tasks_runner.py --with-excluded  # also add spaces 5970, 5860
    python3 expand_tasks_runner.py --analyze-only   # skip running, just analyze

The five runner functions are copied VERBATIM from hpob_phase2_extension.py and
adaptive_k_runner.py, and adaptive_k uses the FROZEN constants from the 6767
sweep (F=2500, K_MIN=1000, C_GROW=0.75), so results are consistent with the
existing cache. Requires the HPO-B data (see env vars below).
"""
import sys, os, csv, json, time, argparse, glob, collections
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
PKG  = HERE.parent
DEFAULT_CACHE = PKG / "2026-07-07-09-51-55" / "audit_workspace" / "hpob_phase2_confirmatory" / "phase2_cache"
CACHE_DIR = Path(os.environ.get("CACHE_DIR", str(DEFAULT_CACHE)))
CACHE_DIR.mkdir(parents=True, exist_ok=True)
HPOB_REPO = os.environ.get("HPOB_REPO", "/tmp/HPO-B")
HPOB_DATA_ROOT = os.environ.get("HPOB_DATA_ROOT", "/tmp/hpob_data/hpob-data/")

BUDGETS = [5, 10, 25]
SEEDS = list(range(20))
RF_N_ESTIMATORS = 64
RF_UCB_BETA = 1.96
RF_CANDIDATE_POOL = 2000
WARMUP = 5
# frozen adaptive-K constants from the 6767 sweep
AK_FULL_FLOOR = int(os.environ.get("FULL_FLOOR", "2500"))
AK_K_MIN      = int(os.environ.get("K_MIN",      "1000"))
AK_C_GROW     = float(os.environ.get("C_GROW",   "0.75"))

ALL_SPACES = ['6767','5906','5971','5965','6794','7607','7609',
              '5527','5891','5636','5859','5889','4796','6766']
EXCLUDED_SPACES = ['5970','5860']
METHODS_RUN = ['random','tpe','rf_smbo','rf_candidate_2k','adaptive_k']

def unit_path(ss, task, seed, method, budget):
    return CACHE_DIR / f"{ss}_{task}_s{seed}_{method}_B{budget}.json"

# ===========================================================================
# runner functions (VERBATIM from hpob_phase2_extension.py / adaptive_k_runner.py)
# ===========================================================================
def _prep(hdlr, ss, task):
    X = np.array(hdlr.meta_test_data[ss][task]['X']) if task in hdlr.meta_test_data.get(ss, {}) \
        else np.array(hdlr.meta_validation_data[ss][task]['X'])
    y = np.array(hdlr.meta_test_data[ss][task]['y']) if task in hdlr.meta_test_data.get(ss, {}) \
        else np.array(hdlr.meta_validation_data[ss][task]['y'])
    return X, hdlr.normalize(y)

def run_random(ss, task, seed, budget, hdlr):
    X, yn = _prep(hdlr, ss, task); rng = np.random.default_rng(seed); n = len(X)
    best, traj = 0.0, []
    for i in range(budget):
        idx = int(rng.integers(n)); v = float(yn[idx].item())
        best = max(best, v); traj.append(float(best))
    return {'trajectory': traj, 'n_evals': budget}

def run_tpe(ss, task, seed, budget, hdlr):
    from scipy.stats import gaussian_kde
    X, yn = _prep(hdlr, ss, task); rng = np.random.default_rng(seed); n = len(X)
    obs_i, obs_v, best, traj = [], [], 0.0, []
    for step in range(budget):
        if step < 3:
            idx = int(rng.integers(n))
            while idx in obs_i: idx = int(rng.integers(n))
        else:
            arr = np.array(obs_v); split = np.median(arr)
            good, bad = arr >= split, arr < split
            if good.sum() >= 2 and bad.sum() >= 2:
                gc = np.array([X[i] for i in np.array(obs_i)[good]])
                bc = np.array([X[i] for i in np.array(obs_i)[bad]])
                try:
                    kg, kb = gaussian_kde(gc.T), gaussian_kde(bc.T)
                    cands = rng.choice(n, min(2000, n), replace=False)
                    br, bc2 = -np.inf, cands[0]
                    for c in cands:
                        if c in obs_i: continue
                        try:
                            r = kg.logpdf(X[c]) - kb.logpdf(X[c])
                            if r > br: br, bc2 = r, c
                        except: pass
                    idx = bc2 if br > -np.inf and bc2 not in obs_i else int(rng.integers(n))
                except: idx = int(rng.integers(n))
            else: idx = int(rng.integers(n))
        obs_i.append(idx); obs_v.append(float(yn[idx].item()))
        best = max(best, yn[idx].item()); traj.append(float(best))
    return {'trajectory': traj, 'n_evals': len(obs_i)}

def run_rf_smbo(ss, task, seed, budget, hdlr):
    from sklearn.ensemble import RandomForestRegressor
    X, yn = _prep(hdlr, ss, task); rng = np.random.default_rng(seed); n = len(X)
    obs_i, obs_v, best, traj = [], [], 0.0, []
    for step in range(budget):
        if step < 5:
            idx = int(rng.integers(n))
            while idx in obs_i: idx = int(rng.integers(n))
        else:
            rf = RandomForestRegressor(n_estimators=RF_N_ESTIMATORS, max_depth=None,
                                       min_samples_leaf=1, random_state=seed+step, n_jobs=1)
            rf.fit(np.array([X[i] for i in obs_i]), np.array(obs_v))
            uneval = [i for i in range(n) if i not in obs_i]
            if uneval:
                Xp = np.array([X[i] for i in uneval])
                preds = np.array([t.predict(Xp) for t in rf.estimators_])
                idx = uneval[int(np.argmax(preds.mean(0) + RF_UCB_BETA*preds.std(0)))]
            else: idx = int(rng.integers(n))
        obs_i.append(idx); obs_v.append(float(yn[idx].item()))
        best = max(best, yn[idx].item()); traj.append(float(best))
    return {'trajectory': traj, 'n_evals': len(obs_i)}

def run_rf_candidate_2k(ss, task, seed, budget, hdlr):
    from sklearn.ensemble import RandomForestRegressor
    X, yn = _prep(hdlr, ss, task); rng = np.random.default_rng(seed); n = len(X)
    obs_i, obs_v, best, traj = [], [], 0.0, []
    for step in range(budget):
        if step < 5:
            idx = int(rng.integers(n))
            while idx in obs_i: idx = int(rng.integers(n))
        else:
            rf = RandomForestRegressor(n_estimators=RF_N_ESTIMATORS, max_depth=None,
                                       min_samples_leaf=1, random_state=seed+step, n_jobs=1)
            rf.fit(np.array([X[i] for i in obs_i]), np.array(obs_v))
            uneval = [i for i in range(n) if i not in obs_i]
            if uneval:
                nc = min(RF_CANDIDATE_POOL, len(uneval))
                cands = rng.choice(uneval, nc, replace=False)
                preds = np.array([t.predict(np.array([X[i] for i in cands])) for t in rf.estimators_])
                score = preds.mean(0) + RF_UCB_BETA*preds.std(0)
                idx = cands[int(rng.integers(len(cands)))] if np.max(score)-np.min(score) < 1e-6 \
                      else cands[int(np.argmax(score))]
            else: idx = int(rng.integers(n))
        obs_i.append(int(idx)); obs_v.append(float(yn[idx].item()))
        best = max(best, yn[idx].item()); traj.append(float(best))
    return {'trajectory': traj, 'n_evals': len(obs_i)}

def run_adaptive_k(ss, task, seed, budget, hdlr):
    from sklearn.ensemble import RandomForestRegressor
    X, yn = _prep(hdlr, ss, task); rng = np.random.default_rng(seed); n = len(X)
    obs_i, obs_v, best, traj, s0 = [], [], 0.0, [], None
    for step in range(budget):
        if step < WARMUP:
            idx = int(rng.integers(n))
            while idx in obs_i: idx = int(rng.integers(n))
        else:
            rf = RandomForestRegressor(n_estimators=RF_N_ESTIMATORS, max_depth=None,
                                       min_samples_leaf=1, random_state=seed+step, n_jobs=1)
            rf.fit(np.array([X[i] for i in obs_i]), np.array(obs_v))
            obs_set = set(obs_i); uneval = [i for i in range(n) if i not in obs_set]; U = len(uneval)
            if U == 0:
                idx = int(rng.integers(n))
            elif U <= AK_FULL_FLOOR:
                Xp = np.array([X[i] for i in uneval])
                preds = np.array([t.predict(Xp) for t in rf.estimators_])
                idx = uneval[int(np.argmax(preds.mean(0) + RF_UCB_BETA*preds.std(0)))]
            else:
                probe = rng.choice(uneval, min(AK_K_MIN, U), replace=False)
                pp = np.array([t.predict(np.array([X[i] for i in probe])) for t in rf.estimators_])
                pstd = pp.std(0); s_hat = float(pstd.mean()) + 1e-12
                if s0 is None: s0 = s_hat
                K_t = int(min(U, max(AK_K_MIN, round(AK_K_MIN*(s0/s_hat)**AK_C_GROW))))
                if K_t <= len(probe):
                    cand = probe; score = pp.mean(0) + RF_UCB_BETA*pstd
                else:
                    rem = list(set(uneval) - set(probe.tolist()))
                    extra = rng.choice(rem, min(K_t-len(probe), len(rem)), replace=False)
                    cand = np.concatenate([probe, extra])
                    pc = np.array([t.predict(np.array([X[i] for i in cand])) for t in rf.estimators_])
                    score = pc.mean(0) + RF_UCB_BETA*pc.std(0)
                idx = int(cand[int(rng.integers(len(cand)))]) if float(np.max(score)-np.min(score)) < 1e-6 \
                      else int(cand[int(np.argmax(score))])
        obs_i.append(int(idx)); obs_v.append(float(yn[idx].item()))
        best = max(best, yn[idx].item()); traj.append(float(best))
    return {'trajectory': traj, 'n_evals': len(obs_i)}

RUNNERS = {'random': run_random, 'tpe': run_tpe, 'rf_smbo': run_rf_smbo,
           'rf_candidate_2k': run_rf_candidate_2k, 'adaptive_k': run_adaptive_k}

# ===========================================================================
def do_run(spaces, splits):
    sys.path.insert(0, HPOB_REPO)
    from hpob_handler import HPOBHandler
    import warnings; warnings.filterwarnings('ignore')
    print(f"Loading HPO-B ...  adaptive_k frozen constants F={AK_FULL_FLOOR},K_MIN={AK_K_MIN},c={AK_C_GROW}")
    hdlr = HPOBHandler(root_dir=HPOB_DATA_ROOT, mode='v3')
    # collect new tasks by split
    manifest = []
    tasklist = []
    for ss in spaces:
        for split in splits:
            data = getattr(hdlr, f"meta_{split}_data", {}).get(ss, {})
            for task in data.keys():
                # skip tasks that are ALSO in the test split when adding validation to avoid dup
                if split == 'validation' and task in hdlr.meta_test_data.get(ss, {}):
                    continue
                tasklist.append((ss, task)); manifest.append((ss, task, split))
    # write manifest of which tasks are which split
    with open(HERE/"expand_task_manifest.csv","w",newline="") as f:
        w=csv.writer(f); w.writerow(["ss_id","task_id","split"]); w.writerows(manifest)
    total = len(tasklist)*len(SEEDS)*len(METHODS_RUN); done=skip=fail=0; t0=time.time()
    print(f"New tasks: {len(tasklist)}  ->  {total} run-units")
    for (ss, task) in tasklist:
        for seed in SEEDS:
            for method in METHODS_RUN:
                done += 1
                if all(unit_path(ss, task, seed, method, b).exists() for b in BUDGETS):
                    skip += 1; continue
                try:
                    res = RUNNERS[method](ss, task, seed, max(BUDGETS), hdlr)
                    for b in BUDGETS:
                        i = min(b, len(res['trajectory']))-1
                        if i >= 0:
                            val = float(res['trajectory'][i])
                            json.dump({'search_space_id': ss,'task_id': task,'seed': seed,
                                       'method': method,'budget': b,'best_found_value': val,
                                       'normalized_regret': max(0.0,1.0-val),'n_evals': res['n_evals']},
                                      open(unit_path(ss, task, seed, method, b),'w'))
                except Exception as e:
                    fail += 1; print(f"  FAIL {ss} {task} s{seed} {method}: {e}", flush=True)
                if done % 100 == 0:
                    el=time.time()-t0; r=done/el if el else 0
                    print(f"  {done}/{total} ({skip} cached,{fail} fail) {el:.0f}s ETA {(total-done)/r if r else 0:.0f}s",flush=True)
    print(f"Run done: {done} units, {skip} cached, {fail} failed, {time.time()-t0:.0f}s")

def analyze():
    from scipy import stats
    meta = {r['ss_id']:(int(r['dim']),int(r['n_configs']))
            for r in csv.DictReader(open(PKG/"table2_per_space_b25.csv"))}
    # split map: original test tasks vs newly-added tasks (from manifest)
    added=set()
    mpath=HERE/"expand_task_manifest.csv"
    if mpath.exists():
        for r in csv.DictReader(open(mpath)): added.add((r['ss_id'],r['task_id']))
    recs=[]
    for f in glob.glob(str(CACHE_DIR/"*.json")):
        try: recs.append(json.load(open(f)))
        except: pass
    need=['adaptive_k','rf_candidate_2k','rf_smbo']
    cell=collections.defaultdict(dict)
    for r in recs:
        if int(r['budget'])!=25: continue
        cell[(r['search_space_id'],str(r['task_id']),int(r['seed']))][r['method']]=float(r['normalized_regret'])
    agg=collections.defaultdict(lambda: collections.defaultdict(list))
    for (ss,t,s),md in cell.items():
        if all(m in md for m in need):
            for m in need: agg[(ss,t)][m].append(md[m])
    T=sorted(agg)
    def subset(kind):
        if kind=='test': return [k for k in T if k not in added]
        if kind=='added': return [k for k in T if k in added]
        return T
    def paired(tasks, base, method):
        a=np.array([np.mean(agg[k][method]) for k in tasks]); b=np.array([np.mean(agg[k][base]) for k in tasks])
        d=b-a; p2=stats.wilcoxon(a,b).pvalue if len(a)>=3 and np.any(np.abs(d)>1e-12) else float('nan')
        p1=stats.wilcoxon(a,b,alternative='less').pvalue if len(a)>=3 and np.any(np.abs(d)>1e-12) else float('nan')
        return len(a),100*d.mean()/b.mean() if b.mean() else 0,int((d>1e-9).sum()),int((d<-1e-9).sum()),p2,p1
    rows=[["method_vs_rf_smbo","taskset","n","gain_pct","wins","losses","p_two_sided","p_one_sided"]]
    print(f"\n=== expanded evaluation: adaptive_k / rf_candidate_2k vs rf_smbo (B=25, 20 seeds) ===")
    for method in ['adaptive_k','rf_candidate_2k']:
        for kind in ['test','added','combined']:
            ts=subset(kind)
            if len(ts)<3: continue
            n,g,w,l,p2,p1=paired(ts,'rf_smbo',method)
            print(f"  {method:15s} [{kind:8s}] n={n:3d} gain={g:+.1f}% W/L={w}/{l} two-sided p={p2:.4f} one-sided p={p1:.4f}")
            rows.append([method,kind,n,f"{g:+.1f}",w,l,f"{p2:.4f}",f"{p1:.4f}"])
    with open(HERE/"expanded_stat_tests.csv","w",newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"\nwrote {HERE/'expanded_stat_tests.csv'}")

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--with-excluded',action='store_true',help='also add spaces 5970, 5860')
    ap.add_argument('--splits',default='validation',help='comma list: validation[,test]')
    ap.add_argument('--analyze-only',action='store_true')
    args=ap.parse_args()
    if not args.analyze_only:
        spaces=ALL_SPACES+(EXCLUDED_SPACES if args.with_excluded else [])
        do_run(spaces, args.splits.split(','))
    analyze()
