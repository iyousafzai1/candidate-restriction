#!/usr/bin/env python3
"""Phase 2: Confirmatory validation of frozen rf_candidate_2k.
Extends Stage E: 10 seeds, all PASS spaces, statistical analysis."""

import sys, json, csv, os, time, argparse
from pathlib import Path
from collections import defaultdict
import numpy as np

ROOT = Path("/tmp")
sys.path.insert(0, str(ROOT / "HPO-B"))

from hpob_handler import HPOBHandler
from sklearn.ensemble import RandomForestRegressor
import warnings
import os as _os
from pathlib import Path as _Path
_RUNS = _Path(_os.environ.get("RUNS_DIR", _Path(__file__).resolve().parent.parent / "runs"))
warnings.filterwarnings('ignore')

OUT_DIR = _RUNS / "hpob_phase2_confirmatory"
CACHE_DIR = OUT_DIR / "phase2_cache"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

BUDGETS = [5, 10, 25]
SEEDS = list(range(10))  # 10 seeds
N_JOBS = 4

# Frozen rf_candidate_2k parameters
RF_CANDIDATE_POOL = 2000
RF_N_ESTIMATORS = 64
RF_UCB_BETA = 1.96

# Methods to run
METHODS_RUN = ['random', 'tpe', 'rf_smbo', 'rf_candidate_2k']

# All spaces for Phase 2 (13 PASS from screening + 5971 BORDERLINE)
# Sorted by dimension descending
ALL_SPACES = ['6767', '5906', '5971', '5965', '6794', '7607', '7609',
              '5527', '5891', '5636', '5859', '5889', '4796', '6766']

# Algorithm families for reference
ALGO_MAP = {
    '6767': 'xgboost', '5971': 'xgboost', '5906': 'xgboost',
    '6794': 'ranger', '5965': 'ranger', '7607': 'ranger', '7609': 'ranger', '5889': 'ranger',
    '5527': 'svm', '5891': 'svm',
    '5636': 'rpart', '5859': 'rpart',
    '4796': 'glmnet', '5970': 'glmnet', '6766': 'glmnet',
}

# Config sizes (from Stage E screen, cached)
CONFIG_MAP = {
    '4796': 300, '5527': 44790, '5636': 43841, '5859': 2878,
    '5889': 298, '5891': 2584, '5906': 254, '5965': 11228,
    '6766': 39155, '6767': 44258, '6794': 52448, '7607': 647, '7609': 1269, '5971': 2265,
}

print("Loading HPO-B handler...")
hdlr = HPOBHandler(root_dir='/tmp/hpob_data/hpob-data/', mode='v3')

# Load descriptor for dimension info
with open('/tmp/HPO-B/hpob-data/meta-dataset-descriptors.json') as f:
    desc = json.load(f)

def get_test_tasks(ss_id):
    return list(hdlr.meta_test_data.get(ss_id, {}).keys())

def unit_path(ss_id, task, seed, method, budget):
    fname = f"{ss_id}_{task}_s{seed}_{method}_B{budget}.json"
    return CACHE_DIR / fname

# Also check Stage E cache
stageE_cache = _Path(_os.environ.get("STAGEE_CACHE", _RUNS / "stageE_cache"))

def check_any_cache(ss_id, task, seed, method, budget):
    p = unit_path(ss_id, task, seed, method, budget)
    if p.exists():
        return p
    # Also check Stage E cache
    se = stageE_cache / p.name
    if se.exists():
        return se
    return None

def run_random(ss_id, task, seed, budget, hdlr):
    X_test = np.array(hdlr.meta_test_data[ss_id][task]['X'])
    y_test = np.array(hdlr.meta_test_data[ss_id][task]['y'])
    y_norm = hdlr.normalize(y_test)
    rng = np.random.default_rng(seed)
    n_configs = len(X_test)
    best = 0.0
    traj = []
    for i in range(budget):
        idx = int(rng.integers(n_configs))
        val = float(y_norm[idx].item())
        if val > best:
            best = val
        traj.append(float(best))
    return {'trajectory': traj, 'final_acc': best, 'n_evals': budget}

def run_tpe(ss_id, task, seed, budget, hdlr):
    from scipy.stats import gaussian_kde
    X_test = np.array(hdlr.meta_test_data[ss_id][task]['X'])
    y_test = np.array(hdlr.meta_test_data[ss_id][task]['y'])
    y_norm = hdlr.normalize(y_test)
    rng = np.random.default_rng(seed)
    n_configs = len(X_test)
    observed_idx, observed_vals = [], []
    best = 0.0
    traj = []
    for step in range(budget):
        if step < 3:
            idx = int(rng.integers(n_configs))
            while idx in observed_idx:
                idx = int(rng.integers(n_configs))
        else:
            observed = np.array(observed_vals)
            split = np.median(observed)
            good = observed >= split
            bad = observed < split
            n_good, n_bad = good.sum(), bad.sum()
            if n_good >= 2 and n_bad >= 2:
                good_configs = np.array([X_test[i] for i in np.array(observed_idx)[good]])
                bad_configs = np.array([X_test[i] for i in np.array(observed_idx)[bad]])
                try:
                    kde_good = gaussian_kde(good_configs.T)
                    kde_bad = gaussian_kde(bad_configs.T)
                    n_candidate = min(2000, n_configs)
                    candidates = rng.choice(len(X_test), n_candidate, replace=False)
                    best_ratio, best_c = -np.inf, candidates[0]
                    for c in candidates:
                        if c in observed_idx: continue
                        try:
                            ratio = kde_good.logpdf(X_test[c]) - kde_bad.logpdf(X_test[c])
                            if ratio > best_ratio:
                                best_ratio, best_c = ratio, c
                        except: pass
                    idx = best_c if best_ratio > -np.inf and best_c not in observed_idx else int(rng.integers(n_configs))
                except:
                    idx = int(rng.integers(n_configs))
            else:
                idx = int(rng.integers(n_configs))
        observed_idx.append(idx)
        observed_vals.append(float(y_norm[idx].item()))
        if y_norm[idx].item() > best:
            best = float(y_norm[idx].item())
        traj.append(float(best))
    return {'trajectory': traj, 'final_acc': best, 'n_evals': len(observed_idx)}

def run_rf_smbo(ss_id, task, seed, budget, hdlr):
    X_test = np.array(hdlr.meta_test_data[ss_id][task]['X'])
    y_test = np.array(hdlr.meta_test_data[ss_id][task]['y'])
    y_norm = hdlr.normalize(y_test)
    rng = np.random.default_rng(seed)
    n_configs = len(X_test)
    observed_idx, observed_vals = [], []
    best, traj = 0.0, []
    for step in range(budget):
        if step < 5:
            idx = int(rng.integers(n_configs))
            while idx in observed_idx:
                idx = int(rng.integers(n_configs))
        else:
            X_obs = np.array([X_test[i] for i in observed_idx])
            y_obs_arr = np.array(observed_vals)
            rf = RandomForestRegressor(n_estimators=RF_N_ESTIMATORS, max_depth=None, min_samples_leaf=1,
                                       random_state=seed + step, n_jobs=1)
            rf.fit(X_obs, y_obs_arr)
            uneval = [i for i in range(n_configs) if i not in observed_idx]
            if uneval:
                X_pen = np.array([X_test[i] for i in uneval])
                preds = np.array([t.predict(X_pen) for t in rf.estimators_])
                mean, std = preds.mean(axis=0), preds.std(axis=0)
                score = mean + RF_UCB_BETA * std
                idx = uneval[int(np.argmax(score))]
            else:
                idx = int(rng.integers(n_configs))
        observed_idx.append(idx)
        observed_vals.append(float(y_norm[idx].item()))
        if y_norm[idx].item() > best:
            best = float(y_norm[idx].item())
        traj.append(float(best))
    return {'trajectory': traj, 'final_acc': best, 'n_evals': len(observed_idx)}

def run_rf_candidate_2k(ss_id, task, seed, budget, hdlr):
    X_test = np.array(hdlr.meta_test_data[ss_id][task]['X'])
    y_test = np.array(hdlr.meta_test_data[ss_id][task]['y'])
    y_norm = hdlr.normalize(y_test)
    rng = np.random.default_rng(seed)
    n_configs = len(X_test)
    observed_idx, observed_vals = [], []
    best, traj = 0.0, []
    for step in range(budget):
        if step < 5:
            idx = int(rng.integers(n_configs))
            while idx in observed_idx:
                idx = int(rng.integers(n_configs))
        else:
            X_obs = np.array([X_test[i] for i in observed_idx])
            y_obs_arr = np.array(observed_vals)
            rf = RandomForestRegressor(n_estimators=RF_N_ESTIMATORS, max_depth=None, min_samples_leaf=1,
                                       random_state=seed + step, n_jobs=1)
            rf.fit(X_obs, y_obs_arr)
            uneval = [i for i in range(n_configs) if i not in observed_idx]
            if uneval:
                n_candidate = min(RF_CANDIDATE_POOL, len(uneval))
                candidates = rng.choice(uneval, n_candidate, replace=False)
                X_pen = np.array([X_test[i] for i in candidates])
                preds = np.array([t.predict(X_pen) for t in rf.estimators_])
                mean, std = preds.mean(axis=0), preds.std(axis=0)
                score = mean + RF_UCB_BETA * std
                if np.max(score) - np.min(score) < 1e-6:
                    idx = candidates[int(rng.integers(len(candidates)))]
                else:
                    idx = candidates[int(np.argmax(score))]
            else:
                idx = int(rng.integers(n_configs))
        observed_idx.append(idx)
        observed_vals.append(float(y_norm[idx].item()))
        if y_norm[idx].item() > best:
            best = float(y_norm[idx].item())
        traj.append(float(best))
    return {'trajectory': traj, 'final_acc': best, 'n_evals': len(observed_idx)}

RUNNERS = {'random': run_random, 'tpe': run_tpe, 'rf_smbo': run_rf_smbo, 'rf_candidate_2k': run_rf_candidate_2k}

def run_phase2(ss_id, smoke=False):
    test_tasks = get_test_tasks(ss_id)
    info = desc.get(ss_id, {})
    dim = len(info.get('variables', {}))
    
    seeds = SEEDS[:3] if smoke else SEEDS
    
    for task in test_tasks:
        for method in METHODS_RUN:
            for seed in seeds:
                # Check all budgets cached
                all_cached = all(check_any_cache(ss_id, task, seed, method, b) for b in BUDGETS)
                if all_cached:
                    continue
                
                try:
                    max_b = max(BUDGETS)
                    result = RUNNERS[method](ss_id, task, seed, max_b, hdlr)
                    
                    for b in BUDGETS:
                        idx = min(b, len(result['trajectory'])) - 1
                        if idx >= 0:
                            val = float(result['trajectory'][idx])
                            regret = max(0.0, 1.0 - val)
                            out = {
                                'search_space_id': ss_id, 'task_id': task, 'seed': seed,
                                'method': method, 'budget': b,
                                'best_found_value': val, 'normalized_regret': regret,
                                'n_evals': result['n_evals'],
                            }
                            path = unit_path(ss_id, task, seed, method, b)
                            json.dump(out, open(path, 'w'))
                except Exception as e:
                    print(f"  {task} {method} s{seed} FAILED: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true', help='Smoke test (3 seeds, 1 space)')
    parser.add_argument('--space', help='Single space to process')
    args = parser.parse_args()
    
    if args.smoke:
        run_phase2('6767', smoke=True)
    elif args.space:
        run_phase2(args.space)
    else:
        for ss_id in ALL_SPACES:
            info = desc.get(ss_id, {})
            dim = len(info.get('variables', {}))
            n_test = len(get_test_tasks(ss_id))
            print(f"\n{'='*50}\nSpace {ss_id} (dim={dim}, {n_test} test tasks)\n{'='*50}")
            run_phase2(ss_id)
