#!/usr/bin/env python3
"""
Adaptive-K variant of rf_candidate_2k, derived from the optimizer's-curse bound
    B_t(K) = Delta * e^{-p K}  +  2 * sigma_t * sqrt(2 ln K)
whose minimiser K*_t grows as the surrogate becomes reliable (sigma_t shrinks)
and is capped at the number of unevaluated configs U_t.

Deployable schedule (three FROZEN constants, set once on space 6767 only):
    * FULL_FLOOR  F      : if U_t <= F  -> score ALL unevaluated configs, in range
                           order, exactly like rf_smbo  (=> provable no-op / weak
                           dominance in small pools; removes the small-pool drag).
    * K_MIN              : early candidate floor when the RF is most uncertain.
    * C_GROW  c          : growth exponent; K_t = K_MIN * (s0 / s_hat_t)^c, capped at U_t,
                           where s_hat_t = mean tree-prediction std over a probe
                           (an online estimate of sigma_t) and s0 is that quantity
                           at the first restricted step of the run.
Everything else (RF: 64 trees; UCB beta=1.96; 5-step warmup) is identical to the
frozen rf_candidate_2k, so the ONLY change under test is the candidate-set size.

Run (on the machine with HPO-B data):
    python3 adaptive_k_runner.py --smoke        # space 6767 only, sanity check
    python3 adaptive_k_runner.py                # all 14 spaces, 20 seeds
    python3 adaptive_k_runner.py --analyze-only # skip running, just print the verdict

Writes method='adaptive_k' cells into the SAME phase2_cache, then prints and saves
    adaptive_k_stat_tests.csv   (adaptive_k vs rf_smbo / rf_candidate_2k, task-level
                                 paired Wilcoxon + optimizer's-curse regime split)
"""
import sys, os, csv, json, time, argparse, glob, collections
from pathlib import Path
import numpy as np

# ----------------------------------------------------------------- paths / config
HERE = Path(__file__).resolve().parent
PKG  = HERE.parent
DEFAULT_CACHE = PKG / "2026-07-07-09-51-55" / "audit_workspace" / "hpob_phase2_confirmatory" / "phase2_cache"
CACHE_DIR = Path(os.environ.get("CACHE_DIR", str(DEFAULT_CACHE)))
CACHE_DIR.mkdir(parents=True, exist_ok=True)

HPOB_REPO = os.environ.get("HPOB_REPO", "/tmp/HPO-B")
HPOB_DATA_ROOT = os.environ.get("HPOB_DATA_ROOT", "/tmp/hpob_data/hpob-data/")
HPOB_DESC = os.environ.get("HPOB_DESC", "/tmp/HPO-B/hpob-data/meta-dataset-descriptors.json")

# ---- frozen hyperparameters ----
BUDGETS = [5, 10, 25]
SEEDS = list(range(20))
RF_N_ESTIMATORS = 64
RF_UCB_BETA = 1.96
WARMUP = 5
# ---- adaptive-K constants: FREEZE these on space 6767, then never touch again ----
FULL_FLOOR_G = int(os.environ.get("FULL_FLOOR", "2500"))   # F: score-all threshold (small-pool no-op)
K_MIN_G      = int(os.environ.get("K_MIN",      "1000"))   # early candidate floor (frozen)
C_GROW_G     = float(os.environ.get("C_GROW",   "0.75"))   # growth exponent (frozen)

ALL_SPACES = ['6767','5906','5971','5965','6794','7607','7609',
              '5527','5891','5636','5859','5889','4796','6766']
METHODS_ALL = ['rf_candidate_2k','rf_smbo','random','tpe','adaptive_k']

def unit_path(ss, task, seed, method, budget):
    return CACHE_DIR / f"{ss}_{task}_s{seed}_{method}_B{budget}.json"

# ===========================================================================
# adaptive-K runner
# ===========================================================================
def run_adaptive_k(ss_id, task, seed, budget, hdlr, full_floor=None, k_min=None, c_grow=None):
    # constants default to the frozen module-level values; overridable for the 6767 sweep
    FULL_FLOOR = FULL_FLOOR_G if full_floor is None else full_floor
    K_MIN      = K_MIN_G      if k_min      is None else k_min
    C_GROW     = C_GROW_G     if c_grow     is None else c_grow
    from sklearn.ensemble import RandomForestRegressor
    X_test = np.array(hdlr.meta_test_data[ss_id][task]['X'])
    y_test = np.array(hdlr.meta_test_data[ss_id][task]['y'])
    y_norm = hdlr.normalize(y_test)
    rng = np.random.default_rng(seed)
    n_configs = len(X_test)
    observed_idx, observed_vals, best, traj = [], [], 0.0, []
    s0 = None                                  # run-level reference uncertainty
    for step in range(budget):
        if step < WARMUP:
            idx = int(rng.integers(n_configs))
            while idx in observed_idx: idx = int(rng.integers(n_configs))
        else:
            X_obs = np.array([X_test[i] for i in observed_idx])
            y_obs = np.array(observed_vals)
            rf = RandomForestRegressor(n_estimators=RF_N_ESTIMATORS, max_depth=None,
                                       min_samples_leaf=1, random_state=seed+step, n_jobs=1)
            rf.fit(X_obs, y_obs)
            obs_set = set(observed_idx)
            uneval = [i for i in range(n_configs) if i not in obs_set]
            U = len(uneval)
            if U == 0:
                idx = int(rng.integers(n_configs))
            elif U <= FULL_FLOOR:
                # ---- small pool: score ALL, range order == rf_smbo (provable no-op) ----
                X_pen = np.array([X_test[i] for i in uneval])
                preds = np.array([t.predict(X_pen) for t in rf.estimators_])
                score = preds.mean(0) + RF_UCB_BETA * preds.std(0)
                idx = uneval[int(np.argmax(score))]
            else:
                # ---- large pool: adaptive restriction driven by surrogate uncertainty ----
                probe = rng.choice(uneval, min(K_MIN, U), replace=False)
                Xp = np.array([X_test[i] for i in probe])
                pp = np.array([t.predict(Xp) for t in rf.estimators_])
                pstd = pp.std(0)
                s_hat = float(pstd.mean()) + 1e-12
                if s0 is None: s0 = s_hat
                K_t = int(min(U, max(K_MIN, round(K_MIN * (s0 / s_hat) ** C_GROW))))
                if K_t <= len(probe):
                    cand = probe
                    score = pp.mean(0) + RF_UCB_BETA * pstd
                else:
                    remaining = list(set(uneval) - set(probe.tolist()))
                    extra = rng.choice(remaining, min(K_t - len(probe), len(remaining)), replace=False)
                    cand = np.concatenate([probe, extra])
                    Xc = np.array([X_test[i] for i in cand])
                    pc = np.array([t.predict(Xc) for t in rf.estimators_])
                    score = pc.mean(0) + RF_UCB_BETA * pc.std(0)
                if float(np.max(score) - np.min(score)) < 1e-6:
                    idx = int(cand[int(rng.integers(len(cand)))])
                else:
                    idx = int(cand[int(np.argmax(score))])
        observed_idx.append(idx)
        observed_vals.append(float(y_norm[idx].item()))
        if y_norm[idx].item() > best: best = float(y_norm[idx].item())
        traj.append(float(best))
    return {'trajectory': traj, 'final_acc': best, 'n_evals': len(observed_idx)}

def do_run(spaces):
    sys.path.insert(0, HPOB_REPO)
    from hpob_handler import HPOBHandler
    import warnings; warnings.filterwarnings('ignore')
    print(f"Loading HPO-B handler ...  (F={FULL_FLOOR_G}, K_MIN={K_MIN_G}, c={C_GROW_G})")
    hdlr = HPOBHandler(root_dir=HPOB_DATA_ROOT, mode='v3')
    total_units = done = failed = skipped = 0
    t0 = time.time()
    for ss in spaces:
        tasks = list(hdlr.meta_test_data.get(ss, {}).keys())
        for task in tasks:
            for seed in SEEDS:
                total_units += 1
                if all(unit_path(ss, task, seed, 'adaptive_k', b).exists() for b in BUDGETS):
                    skipped += 1; done += 1; continue
                try:
                    res = run_adaptive_k(ss, task, seed, max(BUDGETS), hdlr)
                    for b in BUDGETS:
                        i = min(b, len(res['trajectory'])) - 1
                        if i >= 0:
                            val = float(res['trajectory'][i])
                            out = {'search_space_id': ss, 'task_id': task, 'seed': seed,
                                   'method': 'adaptive_k', 'budget': b, 'best_found_value': val,
                                   'normalized_regret': max(0.0, 1.0 - val), 'n_evals': res['n_evals']}
                            json.dump(out, open(unit_path(ss, task, seed, 'adaptive_k', b), 'w'))
                except Exception as e:
                    failed += 1; print(f"  FAILED {ss} {task} s{seed}: {e}", flush=True)
                done += 1
                if done % 50 == 0:
                    el = time.time()-t0; rate = done/el if el else 0
                    print(f"  {done} units ({skipped} cached, {failed} failed) {el:.0f}s "
                          f"ETA {(total_units-done)/rate if rate else 0:.0f}s", flush=True)
    print(f"Run done: {done} units, {skipped} cached, {failed} failed, {time.time()-t0:.0f}s")

# ===========================================================================
# analysis: adaptive_k vs rf_smbo / rf_candidate_2k  (task-level paired Wilcoxon)
# ===========================================================================
def analyze():
    from scipy import stats
    meta = {r['ss_id']: (int(r['dim']), int(r['n_configs']))
            for r in csv.DictReader(open(os.path.join(PKG, "table2_per_space_b25.csv")))}
    recs = []
    for f in glob.glob(str(CACHE_DIR / "*.json")):
        try: recs.append(json.load(open(f)))
        except: pass
    cell = collections.defaultdict(dict)
    for r in recs:
        if int(r['budget']) != 25: continue
        cell[(r['search_space_id'], str(r['task_id']), int(r['seed']))][r['method']] = float(r['normalized_regret'])
    need = ['adaptive_k', 'rf_smbo', 'rf_candidate_2k']
    agg = collections.defaultdict(lambda: collections.defaultdict(list))
    for (ss, t, s), md in cell.items():
        if all(m in md for m in need):
            for m in need: agg[(ss, t)][m].append(md[m])
    T = sorted(agg)
    if not T:
        print("No adaptive_k results in cache yet — run without --analyze-only first.")
        return
    data = {m: np.array([np.mean(agg[k][m]) for k in T]) for m in need}
    N = np.array([meta[k[0]][1] for k in T], float)
    rows = [["comparison", "subset", "n", "mean_gain", "gain_pct", "wins", "losses", "wilcoxon_p"]]
    print(f"\n=== adaptive_k results on {len(T)} tasks (B=25, 20-seed grid) ===")
    print(f"    constants: FULL_FLOOR={FULL_FLOOR_G}, K_MIN={K_MIN_G}, C_GROW={C_GROW_G}")
    def rep(base_name, subset, mask=None):
        a = data['adaptive_k']; b = data[base_name]
        if mask is not None: a, b, dd = a[mask], b[mask], None
        d = b - a
        p = stats.wilcoxon(a, b).pvalue if len(a) >= 3 and np.any(np.abs(d) > 1e-12) else float('nan')
        w = int((d > 1e-9).sum()); l = int((d < -1e-9).sum())
        pct = 100 * d.mean() / b.mean() if b.mean() else 0
        print(f"  adaptive_k vs {base_name:15s} [{subset:16s}] n={len(a):2d} "
              f"gain={d.mean():+.4f} ({pct:+.1f}%) W/L={w}/{l} p={p:.4g}")
        rows.append([f"adaptive_k_vs_{base_name}", subset, len(a), f"{d.mean():+.4f}",
                     f"{pct:+.1f}", w, l, f"{p:.4g}"])
    rep('rf_smbo', 'all')
    rep('rf_smbo', 'pool>FULL_FLOOR', N > FULL_FLOOR_G)
    rep('rf_smbo', 'pool<=FULL_FLOOR', N <= FULL_FLOOR_G)
    rep('rf_candidate_2k', 'all')
    # sanity: is small-pool a true no-op vs rf_smbo?
    smallmask = N <= FULL_FLOOR_G
    if smallmask.any():
        identical = np.allclose(data['adaptive_k'][smallmask], data['rf_smbo'][smallmask], atol=1e-9)
        print(f"  [check] small-pool adaptive_k == rf_smbo exactly? {identical}  "
              f"(max|diff|={np.max(np.abs(data['adaptive_k'][smallmask]-data['rf_smbo'][smallmask])):.2e})")
    out = HERE / "adaptive_k_stat_tests.csv"
    with open(out, 'w', newline='') as f:
        csv.writer(f).writerows(rows)
    print(f"\nwrote {out}")

# ===========================================================================
# sweep: pick the frozen constants on space 6767 ONLY (in-memory, no cache writes)
# ===========================================================================
SWEEP_SETTINGS = [
    # (FULL_FLOOR, K_MIN, C_GROW)
    (2000,  500, 1.0),
    (2500,  750, 1.0),
    (2500, 1000, 0.75),
    (3000,  750, 1.5),
    (3000, 1000, 1.0),
]
SWEEP_SPACE = '6767'

def do_sweep():
    """Run each candidate constant-setting on 6767 (test tasks x 20 seeds) in memory,
    compare to the cached rf_smbo / rf_candidate_2k. Nothing is written to the shared
    cache, so this cannot corrupt the frozen-method results. Pick the winner, freeze it."""
    from scipy import stats
    sys.path.insert(0, HPOB_REPO)
    from hpob_handler import HPOBHandler
    import warnings; warnings.filterwarnings('ignore')
    print(f"Sweep on space {SWEEP_SPACE} only (in-memory, no cache writes).")
    hdlr = HPOBHandler(root_dir=HPOB_DATA_ROOT, mode='v3')
    tasks = list(hdlr.meta_test_data.get(SWEEP_SPACE, {}).keys())

    # baselines from cache for the same (task,seed) at B=25
    def cached(method):
        out = {}
        for t in tasks:
            for s in SEEDS:
                p = unit_path(SWEEP_SPACE, t, s, method, 25)
                if p.exists():
                    out[(t, s)] = json.load(open(p))['normalized_regret']
        return out
    smbo_c = cached('rf_smbo'); cand_c = cached('rf_candidate_2k')
    if not smbo_c:
        print("  rf_smbo not cached for 6767 — run fill_missing_cells.py first."); return

    def task_means(reg):  # (task,seed)->reg  ->  per-task mean over seeds
        by = collections.defaultdict(list)
        for (t, s), v in reg.items(): by[t].append(v)
        return {t: float(np.mean(v)) for t, v in by.items()}
    smbo_tm = task_means(smbo_c); cand_tm = task_means(cand_c)

    print(f"  {len(tasks)} test tasks x {len(SEEDS)} seeds\n")
    hdr = f"  {'F':>5} {'K_MIN':>6} {'c':>5} | {'adaptK_reg':>10} {'vs_rf_smbo':>11} {'W/L':>6} {'p':>8} {'vs_cand2k':>10}"
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    results = []
    for (F, KM, C) in SWEEP_SETTINGS:
        reg = {}
        for t in tasks:
            for s in SEEDS:
                res = run_adaptive_k(SWEEP_SPACE, t, s, 25, hdlr, full_floor=F, k_min=KM, c_grow=C)
                reg[(t, s)] = max(0.0, 1.0 - res['trajectory'][-1])
        ak_tm = task_means(reg)
        common = [t for t in tasks if t in smbo_tm and t in ak_tm]
        a = np.array([ak_tm[t] for t in common]); b = np.array([smbo_tm[t] for t in common])
        d = b - a
        p = stats.wilcoxon(a, b).pvalue if len(a) >= 3 and np.any(np.abs(d) > 1e-12) else float('nan')
        gain_pct = 100 * d.mean() / b.mean() if b.mean() else 0
        cd = np.array([cand_tm[t] for t in common if t in cand_tm])
        ca = np.array([ak_tm[t] for t in common if t in cand_tm])
        cand_pct = 100 * (cd - ca).mean() / cd.mean() if len(cd) and cd.mean() else 0
        w = int((d > 1e-9).sum()); l = int((d < -1e-9).sum())
        print(f"  {F:>5} {KM:>6} {C:>5} | {a.mean():>10.4f} {gain_pct:>+10.1f}% {w:>3}/{l:<2} {p:>8.3g} {cand_pct:>+9.1f}%")
        results.append(((F, KM, C), a.mean(), gain_pct, p))
    best = min(results, key=lambda r: r[1])
    print(f"\n  Recommended freeze on 6767 (lowest regret): FULL_FLOOR={best[0][0]}, "
          f"K_MIN={best[0][1]}, C_GROW={best[0][2]}  (reg={best[1]:.4f}, vs rf_smbo {best[2]:+.1f}%)")
    print("  -> set these three constants, then run the full 14-space job with them and DO NOT retune.")

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true', help='only space 6767')
    ap.add_argument('--space', help='single space id')
    ap.add_argument('--sweep', action='store_true', help='pick frozen constants on 6767 (in-memory, no cache writes)')
    ap.add_argument('--analyze-only', action='store_true', help='skip running, just analyze cache')
    args = ap.parse_args()
    if args.sweep:
        do_sweep()
    elif args.analyze_only:
        analyze()
    else:
        spaces = ['6767'] if args.smoke else ([args.space] if args.space else ALL_SPACES)
        do_run(spaces)
        analyze()
