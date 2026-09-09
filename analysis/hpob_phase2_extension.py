#!/usr/bin/env python3
"""Extend Phase 2 from 10 to 20 seeds.
Only runs seeds 10-19 (seeds 0-9 already cached).
Uses the same frozen rf_candidate_2k and experimental setup."""

import sys, json, csv, os, time, argparse
from pathlib import Path
from collections import defaultdict
import numpy as np

ROOT = Path("/tmp")
sys.path.insert(0, str(ROOT / "HPO-B"))

from hpob_handler import HPOBHandler
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import gaussian_kde
import warnings
import os as _os
from pathlib import Path as _Path
_RUNS = _Path(_os.environ.get("RUNS_DIR", _Path(__file__).resolve().parent.parent / "runs"))
warnings.filterwarnings('ignore')

OUT_DIR = _RUNS / "hpob_phase2_confirmatory"
CACHE_DIR = OUT_DIR / "phase2_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

BUDGETS = [5, 10, 25]
NEW_SEEDS = list(range(10, 20))  # only seeds 10-19
RF_CANDIDATE_POOL = 2000
RF_N_ESTIMATORS = 64
RF_UCB_BETA = 1.96
METHODS_RUN = ['random', 'tpe', 'rf_smbo', 'rf_candidate_2k']

# All 14 spaces
ALL_SPACES = ['6767', '5906', '5971', '5965', '6794', '7607', '7609',
              '5527', '5891', '5636', '5859', '5889', '4796', '6766']

print("Loading HPO-B handler...")
hdlr = HPOBHandler(root_dir='/tmp/hpob_data/hpob-data/', mode='v3')

with open('/tmp/HPO-B/hpob-data/meta-dataset-descriptors.json') as f:
    import json as _j
    desc = _j.load(f)

def get_test_tasks(ss_id):
    return list(hdlr.meta_test_data.get(ss_id, {}).keys())

def unit_path(ss_id, task, seed, method, budget):
    return CACHE_DIR / f"{ss_id}_{task}_s{seed}_{method}_B{budget}.json"

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

def run_space(ss_id):
    test_tasks = get_test_tasks(ss_id)
    info = desc.get(ss_id, {})
    dim = len(info.get('variables', {}))
    
    total_units = len(test_tasks) * len(NEW_SEEDS) * len(METHODS_RUN)
    cached = 0
    for t in test_tasks:
        for s in NEW_SEEDS:
            for m in METHODS_RUN:
                if all(unit_path(ss_id, t, s, m, b).exists() for b in BUDGETS):
                    cached += 1
    
    if cached == total_units:
        return f"{ss_id}: all {total_units} units cached, SKIP"
    
    start = time.time()
    units_run = 0
    for t_idx, task in enumerate(test_tasks):
        for method in METHODS_RUN:
            for seed in NEW_SEEDS:
                if all(unit_path(ss_id, task, seed, method, b).exists() for b in BUDGETS):
                    continue
                try:
                    result = RUNNERS[method](ss_id, task, seed, max(BUDGETS), hdlr)
                    for b in BUDGETS:
                        idx = min(b, len(result['trajectory'])) - 1
                        if idx >= 0:
                            val = float(result['trajectory'][idx])
                            out = {'search_space_id': ss_id, 'task_id': task, 'seed': seed,
                                   'method': method, 'budget': b, 'best_found_value': val,
                                   'normalized_regret': max(0.0, 1.0 - val), 'n_evals': result['n_evals']}
                            json.dump(out, open(unit_path(ss_id, task, seed, method, b), 'w'))
                    units_run += 1
                except Exception as e:
                    print(f"  {task} {method} s{seed} FAILED: {e}")
    
    elapsed = time.time() - start
    return f"{ss_id}: ran {units_run}/{total_units} units in {elapsed:.0f}s"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--space', help='Single space')
    parser.add_argument('--smoke', action='store_true', help='Smoke test on 1 space')
    args = parser.parse_args()
    
    spaces = [args.space] if args.space else (['6767'] if args.smoke else ALL_SPACES)
    
    for ss_id in spaces:
        msg = run_space(ss_id)
        print(msg, flush=True)
    
    print("\n=== 20-seed extension complete ===")
