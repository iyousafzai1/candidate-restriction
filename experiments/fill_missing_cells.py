#!/usr/bin/env python3
"""
Fill exactly the missing (space, task, seed, method, budget) cells listed in
missing_cells.csv, writing into the SAME phase2_cache the re-analysis reads.

The four runner functions below are copied VERBATIM from
hpob_phase2_extension.py, so every filled cell is bit-for-bit identical to what
the original 20-seed run would have produced (each runner is deterministic given
its seed). Cells already present are skipped, so this is safe to re-run / resume.

Usage (on the machine that has the HPO-B data):
    python3 fill_missing_cells.py                  # fill everything in missing_cells.csv
    python3 fill_missing_cells.py --smoke          # just space 6767, to sanity-check
    python3 fill_missing_cells.py --space 5965     # one space only

Requirements (same as the original runs):
    - HPO-B repo on the path (default /tmp/HPO-B), importable `hpob_handler`
    - HPO-B v3 data (default handler root /tmp/hpob_data/hpob-data/)
    - descriptors json (default /tmp/HPO-B/hpob-data/meta-dataset-descriptors.json)
  Override any of these with env vars: HPOB_REPO, HPOB_DATA_ROOT, HPOB_DESC.

After it finishes, refresh every table:
    python3 reanalyze.py
"""
import sys, os, csv, json, time, argparse
from pathlib import Path
from collections import defaultdict
import numpy as np

# ---------------------------------------------------------------- paths / config
HERE = Path(__file__).resolve().parent
PKG  = HERE.parent
# Same cache the re-analysis reads (the folder you copied in). Override with CACHE_DIR.
DEFAULT_CACHE = PKG / "2026-07-07-09-51-55" / "audit_workspace" / "hpob_phase2_confirmatory" / "phase2_cache"
CACHE_DIR = Path(os.environ.get("CACHE_DIR", str(DEFAULT_CACHE)))
CACHE_DIR.mkdir(parents=True, exist_ok=True)
MISSING_CSV = Path(os.environ.get("MISSING_CSV", str(HERE / "missing_cells.csv")))

HPOB_REPO = os.environ.get("HPOB_REPO", "/tmp/HPO-B")
HPOB_DATA_ROOT = os.environ.get("HPOB_DATA_ROOT", "/tmp/hpob_data/hpob-data/")
HPOB_DESC = os.environ.get("HPOB_DESC", "/tmp/HPO-B/hpob-data/meta-dataset-descriptors.json")

sys.path.insert(0, HPOB_REPO)
from hpob_handler import HPOBHandler
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import gaussian_kde
import warnings
warnings.filterwarnings('ignore')

# ---- frozen hyperparameters (identical to hpob_phase2_extension.py) ----
BUDGETS = [5, 10, 25]
RF_CANDIDATE_POOL = 2000
RF_N_ESTIMATORS = 64
RF_UCB_BETA = 1.96

print(f"Loading HPO-B handler from {HPOB_DATA_ROOT} ...")
hdlr = HPOBHandler(root_dir=HPOB_DATA_ROOT, mode='v3')

def unit_path(ss_id, task, seed, method, budget):
    return CACHE_DIR / f"{ss_id}_{task}_s{seed}_{method}_B{budget}.json"

# ============================================================================
# RUNNER FUNCTIONS — copied verbatim from hpob_phase2_extension.py (do not edit)
# ============================================================================
def run_random(ss_id, task, seed, budget, hdlr):
    X_test = np.array(hdlr.meta_test_data[ss_id][task]['X'])
    y_test = np.array(hdlr.meta_test_data[ss_id][task]['y'])
    y_norm = hdlr.normalize(y_test)
    rng = np.random.default_rng(seed)
    n_configs = len(X_test)
    best, traj = 0.0, []
    for i in range(budget):
        idx = int(rng.integers(n_configs))
        val = float(y_norm[idx].item())
        if val > best: best = val
        traj.append(float(best))
    return {'trajectory': traj, 'final_acc': best, 'n_evals': budget}

def run_tpe(ss_id, task, seed, budget, hdlr):
    X_test = np.array(hdlr.meta_test_data[ss_id][task]['X'])
    y_test = np.array(hdlr.meta_test_data[ss_id][task]['y'])
    y_norm = hdlr.normalize(y_test)
    rng = np.random.default_rng(seed)
    n_configs = len(X_test)
    observed_idx, observed_vals, best, traj = [], [], 0.0, []
    for step in range(budget):
        if step < 3:
            idx = int(rng.integers(n_configs))
            while idx in observed_idx: idx = int(rng.integers(n_configs))
        else:
            obs_arr = np.array(observed_vals)
            split = np.median(obs_arr)
            good, bad = obs_arr >= split, obs_arr < split
            if good.sum() >= 2 and bad.sum() >= 2:
                gc = np.array([X_test[i] for i in np.array(observed_idx)[good]])
                bc = np.array([X_test[i] for i in np.array(observed_idx)[bad]])
                try:
                    kde_g, kde_b = gaussian_kde(gc.T), gaussian_kde(bc.T)
                    candidates = rng.choice(n_configs, min(2000, n_configs), replace=False)
                    best_ratio, best_c = -np.inf, candidates[0]
                    for c in candidates:
                        if c in observed_idx: continue
                        try:
                            ratio = kde_g.logpdf(X_test[c]) - kde_b.logpdf(X_test[c])
                            if ratio > best_ratio: best_ratio, best_c = ratio, c
                        except: pass
                    idx = best_c if best_ratio > -np.inf and best_c not in observed_idx else int(rng.integers(n_configs))
                except: idx = int(rng.integers(n_configs))
            else: idx = int(rng.integers(n_configs))
        observed_idx.append(idx)
        observed_vals.append(float(y_norm[idx].item()))
        if y_norm[idx].item() > best: best = float(y_norm[idx].item())
        traj.append(float(best))
    return {'trajectory': traj, 'final_acc': best, 'n_evals': len(observed_idx)}

def run_rf_smbo(ss_id, task, seed, budget, hdlr):
    X_test = np.array(hdlr.meta_test_data[ss_id][task]['X'])
    y_test = np.array(hdlr.meta_test_data[ss_id][task]['y'])
    y_norm = hdlr.normalize(y_test)
    rng = np.random.default_rng(seed)
    n_configs = len(X_test)
    observed_idx, observed_vals, best, traj = [], [], 0.0, []
    for step in range(budget):
        if step < 5:
            idx = int(rng.integers(n_configs))
            while idx in observed_idx: idx = int(rng.integers(n_configs))
        else:
            X_obs = np.array([X_test[i] for i in observed_idx])
            y_obs_arr = np.array(observed_vals)
            rf = RandomForestRegressor(n_estimators=RF_N_ESTIMATORS, max_depth=None, min_samples_leaf=1, random_state=seed+step, n_jobs=1)
            rf.fit(X_obs, y_obs_arr)
            uneval = [i for i in range(n_configs) if i not in observed_idx]
            if uneval:
                X_pen = np.array([X_test[i] for i in uneval])
                preds = np.array([t.predict(X_pen) for t in rf.estimators_])
                mean, std = preds.mean(axis=0), preds.std(axis=0)
                idx = uneval[int(np.argmax(mean + RF_UCB_BETA * std))]
            else: idx = int(rng.integers(n_configs))
        observed_idx.append(idx)
        observed_vals.append(float(y_norm[idx].item()))
        if y_norm[idx].item() > best: best = float(y_norm[idx].item())
        traj.append(float(best))
    return {'trajectory': traj, 'final_acc': best, 'n_evals': len(observed_idx)}

def run_rf_candidate_2k(ss_id, task, seed, budget, hdlr):
    X_test = np.array(hdlr.meta_test_data[ss_id][task]['X'])
    y_test = np.array(hdlr.meta_test_data[ss_id][task]['y'])
    y_norm = hdlr.normalize(y_test)
    rng = np.random.default_rng(seed)
    n_configs = len(X_test)
    observed_idx, observed_vals, best, traj = [], [], 0.0, []
    for step in range(budget):
        if step < 5:
            idx = int(rng.integers(n_configs))
            while idx in observed_idx: idx = int(rng.integers(n_configs))
        else:
            X_obs = np.array([X_test[i] for i in observed_idx])
            y_obs_arr = np.array(observed_vals)
            rf = RandomForestRegressor(n_estimators=RF_N_ESTIMATORS, max_depth=None, min_samples_leaf=1, random_state=seed+step, n_jobs=1)
            rf.fit(X_obs, y_obs_arr)
            uneval = [i for i in range(n_configs) if i not in observed_idx]
            if uneval:
                n_cand = min(RF_CANDIDATE_POOL, len(uneval))
                candidates = rng.choice(uneval, n_cand, replace=False)
                X_pen = np.array([X_test[i] for i in candidates])
                preds = np.array([t.predict(X_pen) for t in rf.estimators_])
                mean, std = preds.mean(axis=0), preds.std(axis=0)
                score = mean + RF_UCB_BETA * std
                if np.max(score) - np.min(score) < 1e-6:
                    idx = candidates[int(rng.integers(len(candidates)))]
                else: idx = candidates[int(np.argmax(score))]
            else: idx = int(rng.integers(n_configs))
        observed_idx.append(idx)
        observed_vals.append(float(y_norm[idx].item()))
        if y_norm[idx].item() > best: best = float(y_norm[idx].item())
        traj.append(float(best))
    return {'trajectory': traj, 'final_acc': best, 'n_evals': len(observed_idx)}

RUNNERS = {'random': run_random, 'tpe': run_tpe, 'rf_smbo': run_rf_smbo, 'rf_candidate_2k': run_rf_candidate_2k}

# ============================================================================
def load_missing(space_filter=None):
    """Group missing_cells.csv into unique (space,task,seed,method) run-units.
    Each unit is run once at max budget; all requested budgets are written."""
    units = defaultdict(set)   # (ss,task,seed,method) -> set of budgets needed
    with open(MISSING_CSV) as f:
        for r in csv.DictReader(f):
            if space_filter and r['ss_id'] != space_filter:
                continue
            units[(r['ss_id'], r['task_id'], int(r['seed']), r['method'])].add(int(r['budget']))
    return units

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--space', help='fill a single search space only')
    ap.add_argument('--smoke', action='store_true', help='fill only space 6767 as a smoke test')
    args = ap.parse_args()
    sfilter = '6767' if args.smoke else args.space

    if not MISSING_CSV.exists():
        sys.exit(f"missing_cells.csv not found at {MISSING_CSV} — run reanalyze.py first")

    units = load_missing(sfilter)
    total = len(units)
    print(f"Missing run-units to fill: {total}  (cache: {CACHE_DIR})")
    if total == 0:
        print("Nothing to do — grid already complete for this selection.")
        return

    done = 0; skipped = 0; failed = 0; t0 = time.time()
    # sort for stable, space-grouped progress
    for (ss, task, seed, method) in sorted(units.keys()):
        budgets_needed = sorted(units[(ss, task, seed, method)])
        # skip if every needed budget already on disk (resumable)
        if all(unit_path(ss, task, seed, method, b).exists() for b in budgets_needed):
            skipped += 1; done += 1; continue
        try:
            result = RUNNERS[method](ss, task, seed, max(BUDGETS), hdlr)
            for b in BUDGETS:                      # write all 3 budgets from the one run
                idx = min(b, len(result['trajectory'])) - 1
                if idx >= 0:
                    val = float(result['trajectory'][idx])
                    out = {'search_space_id': ss, 'task_id': task, 'seed': seed,
                           'method': method, 'budget': b, 'best_found_value': val,
                           'normalized_regret': max(0.0, 1.0 - val), 'n_evals': result['n_evals']}
                    json.dump(out, open(unit_path(ss, task, seed, method, b), 'w'))
        except Exception as e:
            failed += 1
            print(f"  FAILED {ss} {task} s{seed} {method}: {e}", flush=True)
        done += 1
        if done % 50 == 0 or done == total:
            el = time.time() - t0
            rate = done / el if el > 0 else 0
            eta = (total - done) / rate if rate > 0 else 0
            print(f"  {done}/{total} units  ({skipped} already cached, {failed} failed)  "
                  f"{el:.0f}s elapsed, ETA {eta:.0f}s", flush=True)

    print(f"\nDone: {done} units processed, {skipped} were already cached, {failed} failed "
          f"in {time.time()-t0:.0f}s.")
    print("Now run:  python3 reanalyze.py   to refresh every table on the completed grid.")

if __name__ == '__main__':
    main()
