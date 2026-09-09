#!/usr/bin/env python3
"""Phase 2 analysis: comprehensive confirmatory validation."""

import sys, json, csv
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
from scipy import stats
import os as _os
from pathlib import Path as _Path
_RUNS = _Path(_os.environ.get("RUNS_DIR", _Path(__file__).resolve().parent.parent / "runs"))

OUT_DIR = _RUNS / "hpob_phase2_confirmatory"
CACHE_DIR = OUT_DIR / "phase2_cache"
stageE_cache = _Path(_os.environ.get("STAGEE_CACHE", _RUNS / "stageE_cache"))

# Load descriptor
with open('/tmp/HPO-B/hpob-data/meta-dataset-descriptors.json') as f:
    desc = json.load(f)

ALGO_MAP = {
    '6767': 'xgboost', '5971': 'xgboost', '5906': 'xgboost',
    '6794': 'ranger', '5965': 'ranger', '7607': 'ranger', '7609': 'ranger', '5889': 'ranger',
    '5527': 'svm', '5891': 'svm',
    '5636': 'rpart', '5859': 'rpart',
    '4796': 'glmnet', '6766': 'glmnet',
}
CONFIG_MAP = {
    '4796': 300, '5527': 44790, '5636': 43841, '5859': 2878, '5889': 298, '5891': 2584,
    '5906': 254, '5965': 11228, '5971': 2265, '6766': 39155, '6767': 44258, '6794': 52448,
    '7607': 647, '7609': 1269,
}

# Load all results from both caches
all_results = []

# Phase 2 cache
if CACHE_DIR.exists():
    for f in sorted(CACHE_DIR.glob("*.json")):
        d = json.load(open(f))
        all_results.append(d)
    print(f"Phase 2 cache: {sum(1 for _ in CACHE_DIR.glob('*.json'))} files")

# Stage E cache (for seeds 0-4)
if stageE_cache.exists():
    for f in sorted(stageE_cache.glob("*.json")):
        d = json.load(open(f))
        # Only add if not already in Phase 2 cache
        p2_path = CACHE_DIR / f.name
        if not p2_path.exists() and d['search_space_id'] in ALGO_MAP:
            all_results.append(d)

print(f"Total results loaded: {len(all_results)}")

# Add algorithm info
for r in all_results:
    ss = r['search_space_id']
    info = desc.get(ss, {})
    r['algorithm'] = ALGO_MAP.get(ss, '?')
    r['dimension'] = len(info.get('variables', {}))
    r['n_configs'] = CONFIG_MAP.get(ss, 0)

METHODS = ['random', 'tpe', 'rf_smbo', 'rf_candidate_2k']
BUDGETS = [5, 10, 25]
SPACES = sorted(set(r['search_space_id'] for r in all_results if r['search_space_id'] in ALGO_MAP))

print(f"Spaces: {SPACES}")
print(f"Total results: {len(all_results)}")

# ================================================================
# 1. Task-level results CSV
# ================================================================
task_fields = ['search_space_id', 'algorithm', 'dimension', 'n_configs', 'task_id', 'seed', 'budget',
               'method', 'best_found_value', 'normalized_regret', 'n_evals']
with open(OUT_DIR / "phase2_task_level_results.csv", 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=task_fields, extrasaction='ignore')
    w.writeheader()
    for r in all_results:
        w.writerow(r)
print(f"Saved phase2_task_level_results.csv ({len(all_results)} rows)")

# ================================================================
# 2. Method summary by space
# ================================================================
def mean_regret(ss, method, budget):
    vals = [r['normalized_regret'] for r in all_results 
            if r['search_space_id']==ss and r['method']==method and r['budget']==budget]
    return float(np.mean(vals)) if vals else None

summary_rows = []
for ss_id in SPACES:
    info = desc.get(ss_id, {})
    dim = len(info.get('variables', {}))
    algo = ALGO_MAP.get(ss_id, '?')
    n_configs = CONFIG_MAP.get(ss_id, 0)
    
    for budget in BUDGETS:
        for method in METHODS:
            m_rows = [r for r in all_results if r['search_space_id']==ss_id and r['method']==method and r['budget']==budget]
            if not m_rows:
                continue
            regrets = [r['normalized_regret'] for r in m_rows]
            tasks = len(set(r['task_id'] for r in m_rows))
            seeds = len(set(r['seed'] for r in m_rows))
            
            mean_r = float(np.mean(regrets))
            rand_r = mean_regret(ss_id, 'random', budget)
            rf_r = mean_regret(ss_id, 'rf_smbo', budget)
            tpe_r = mean_regret(ss_id, 'tpe', budget)
            
            gain_vs_rand = ((rand_r - mean_r) / rand_r * 100) if (rand_r and rand_r > 0) else 0
            gain_vs_rf = ((rf_r - mean_r) / rf_r * 100) if (rf_r and rf_r > 0) else 0
            gain_vs_tpe = ((tpe_r - mean_r) / tpe_r * 100) if (tpe_r and tpe_r > 0) else 0
            
            summary_rows.append({
                'search_space_id': ss_id, 'algorithm': algo, 'dimension': dim, 'n_configs': n_configs,
                'budget': budget, 'method': method,
                'mean_regret': f'{mean_r:.6f}', 'median_regret': f'{np.median(regrets):.6f}',
                'std_regret': f'{np.std(regrets):.6f}',
                'min_regret': f'{min(regrets):.6f}', 'max_regret': f'{max(regrets):.6f}',
                'near_oracle_rate': f'{sum(1 for r in regrets if r < 0.05) / len(regrets):.4f}',
                'n_tasks': tasks, 'n_seeds': seeds,
                'gain_vs_random': f'{gain_vs_rand:+.2f}%',
                'gain_vs_rf_smbo': f'{gain_vs_rf:+.2f}%',
                'gain_vs_tpe': f'{gain_vs_tpe:+.2f}%',
            })

with open(OUT_DIR / "phase2_method_summary_by_space.csv", 'w', newline='') as f:
    if summary_rows:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)
print(f"Saved phase2_method_summary_by_space.csv ({len(summary_rows)} rows)")

# ================================================================
# 3. Aggregate results
# ================================================================
agg_rows = []
for budget in BUDGETS:
    macro = defaultdict(list)
    micro = defaultdict(list)
    
    for ss_id in SPACES:
        for method in METHODS:
            m_regrets = [r['normalized_regret'] for r in all_results 
                        if r['search_space_id']==ss_id and r['method']==method and r['budget']==budget]
            if m_regrets:
                macro[method].append(float(np.mean(m_regrets)))
                micro[method].extend(m_regrets)
    
    for method in METHODS:
        if not macro[method]:
            continue
        macro_mean = float(np.mean(macro[method]))
        micro_mean = float(np.mean(micro[method]))
        median = float(np.median(micro[method]))
        std = float(np.std(micro[method]))
        n_units = len(micro[method])
        n_spaces = len(macro[method])
        
        # Gains vs baselines
        rand_m = float(np.mean(macro.get('random', [1])))
        rf_m = float(np.mean(macro.get('rf_smbo', [1])))
        tpe_m = float(np.mean(macro.get('tpe', [1])))
        
        gain_vs_rand = ((rand_m - macro_mean) / rand_m * 100) if rand_m > 0 else 0
        gain_vs_rf = ((rf_m - macro_mean) / rf_m * 100) if rf_m > 0 else 0
        gain_vs_tpe = ((tpe_m - macro_mean) / tpe_m * 100) if tpe_m > 0 else 0
        
        # Paired win rates (task-level)
        def win_rate(m1, m2, ulist):
            wins = 0
            for u in ulist:
                v1 = [r['normalized_regret'] for r in all_results if r['search_space_id']==u[0] and r['method']==m1 and r['budget']==budget]
                v2 = [r['normalized_regret'] for r in all_results if r['search_space_id']==u[0] and r['method']==m2 and r['budget']==budget]
                if v1 and v2:
                    if np.mean(v1) < np.mean(v2):
                        wins += 1
            return wins / len(ulist) if ulist else 0
        
        used_spaces = [(s,) for s in SPACES if macro[method] and macro.get('random') and macro.get('rf_smbo')]
        
        agg_rows.append({
            'budget': budget, 'method': method,
            'macro_mean_regret': f'{macro_mean:.6f}', 'micro_mean_regret': f'{micro_mean:.6f}',
            'median_regret': f'{median:.6f}', 'std_regret': f'{std:.6f}',
            'n_spaces': n_spaces, 'n_units': n_units,
            'gain_vs_random': f'{gain_vs_rand:+.2f}%',
            'gain_vs_rf_smbo': f'{gain_vs_rf:+.2f}%',
            'gain_vs_tpe': f'{gain_vs_tpe:+.2f}%',
            'paired_win_rate_vs_random': f'{win_rate(method, "random", used_spaces):.3f}',
            'paired_win_rate_vs_rf_smbo': f'{win_rate(method, "rf_smbo", used_spaces):.3f}',
        })

with open(OUT_DIR / "phase2_aggregate_results.csv", 'w', newline='') as f:
    if agg_rows:
        w = csv.DictWriter(f, fieldnames=list(agg_rows[0].keys()))
        w.writeheader()
        w.writerows(agg_rows)
print(f"Saved phase2_aggregate_results.csv")

# Print B=25 summary
print(f"\n{'='*60}")
print(f"B=25 AGGREGATE RESULTS")
print(f"{'='*60}")
for r in agg_rows:
    if r['budget'] == 25:
        print(f"  {r['method']:<20} macro={r['macro_mean_regret']} micro={r['micro_mean_regret']} "
              f"win_vs_random={r['paired_win_rate_vs_random']}")

# ================================================================
# 4. High-dim subgroup analysis
# ================================================================
def subgroup_analysis(spaces, label, budget=25):
    rows = []
    for method in METHODS:
        regrets = []
        for ss_id in spaces:
            vals = [r['normalized_regret'] for r in all_results 
                    if r['search_space_id']==ss_id and r['method']==method and r['budget']==budget]
            regrets.extend(vals)
        if regrets:
            mean_r = float(np.mean(regrets))
            n_tasks = len(set(r['task_id'] for r in all_results if r['search_space_id'] in spaces))
            rows.append({'method': method, 'mean_regret': mean_r, 'n_tasks': n_tasks})
    return rows

subgroups = {
    'high_dim_ge15': [s for s in SPACES if len(desc.get(s, {}).get('variables', [])) >= 15],
    'med_dim_8to14': [s for s in SPACES if 8 <= len(desc.get(s, {}).get('variables', [])) < 15],
    'low_dim_lt8': [s for s in SPACES if len(desc.get(s, {}).get('variables', [])) < 8],
    'large_config_ge10k': [s for s in SPACES if CONFIG_MAP.get(s, 0) >= 10000],
    'small_config_lt10k': [s for s in SPACES if CONFIG_MAP.get(s, 0) < 10000],
}

subgroup_rows = []
for sg_name, sg_spaces in subgroups.items():
    if not sg_spaces:
        continue
    info = desc.get(sg_spaces[0], {})
    
    for budget in [25]:
        results = subgroup_analysis(sg_spaces, sg_name, budget)
        
        cand = next((r for r in results if r['method'] == 'rf_candidate_2k'), None)
        rand = next((r for r in results if r['method'] == 'random'), None)
        rf = next((r for r in results if r['method'] == 'rf_smbo'), None)
        tpe = next((r for r in results if r['method'] == 'tpe'), None)
        
        for r in results:
            gain_vs_rand = ((rand['mean_regret'] - r['mean_regret']) / rand['mean_regret'] * 100) if (rand and rand['mean_regret'] > 0) else 0
            gain_vs_rf = ((rf['mean_regret'] - r['mean_regret']) / rf['mean_regret'] * 100) if (rf and rf['mean_regret'] > 0) else 0
            
            mean_reg_val = r['mean_regret']
            subgroup_rows.append({
                'subgroup': sg_name,
                'budget': budget,
                'n_spaces': len(sg_spaces),
                'n_tasks': r['n_tasks'],
                'method': r['method'],
                'mean_regret': f'{mean_reg_val:.6f}',
                'gain_vs_random': f'{gain_vs_rand:+.2f}%',
                'gain_vs_rf_smbo': f'{gain_vs_rf:+.2f}%',
            })

with open(OUT_DIR / "phase2_high_dim_subgroup_results.csv", 'w', newline='') as f:
    if subgroup_rows:
        w = csv.DictWriter(f, fieldnames=list(subgroup_rows[0].keys()))
        w.writeheader()
        w.writerows(subgroup_rows)
print(f"Saved phase2_high_dim_subgroup_results.csv")

print(f"\n{'='*60}")
print(f"SUBGROUP ANALYSIS (B=25)")
print(f"{'='*60}")
for sgr in subgroup_rows:
    print(f"  {sgr['subgroup']:<20} {sgr['method']:<20} regret={sgr['mean_regret']} vs_rand={sgr['gain_vs_random']} vs_rf={sgr['gain_vs_rf_smbo']}")

# ================================================================
# 5. Statistical tests
# ================================================================
def bootstrap_ci(values1, values2, n_bootstrap=5000, alpha=0.05):
    """Bootstrap 95% CI for mean difference (v1 - v2)."""
    paired = list(zip(values1, values2))
    diffs = [v1 - v2 for v1, v2 in paired]
    boot_means = []
    rng = np.random.default_rng(42)
    for _ in range(n_bootstrap):
        sample = rng.choice(diffs, len(diffs), replace=True)
        boot_means.append(np.mean(sample))
    boot_means.sort()
    low = boot_means[int(n_bootstrap * alpha / 2)]
    high = boot_means[int(n_bootstrap * (1 - alpha / 2))]
    return low, high

stat_rows = []
for budget in [25]:
    # Per-space paired comparison
    for m1, m2 in [('rf_candidate_2k', 'random'), ('rf_candidate_2k', 'rf_smbo'), ('rf_candidate_2k', 'tpe')]:
        # Space-level pairing
        pairs = []
        for ss_id in SPACES:
            v1 = [r['normalized_regret'] for r in all_results if r['search_space_id']==ss_id and r['method']==m1 and r['budget']==budget]
            v2 = [r['normalized_regret'] for r in all_results if r['search_space_id']==ss_id and r['method']==m2 and r['budget']==budget]
            if v1 and v2:
                pairs.append((float(np.mean(v1)), float(np.mean(v2))))
        
        if len(pairs) >= 3:
            v1_means = [p[0] for p in pairs]
            v2_means = [p[1] for p in pairs]
            diffs = [v2 - v1 for v1, v2 in pairs]  # positive = m1 better
            mean_gain = float(np.mean(diffs))
            median_gain = float(np.median(diffs))
            win_rate = sum(1 for d in diffs if d > 0) / len(diffs)
            
            # Bootstrap CI
            low, high = bootstrap_ci(v2_means, v1_means)
            
            # Wilcoxon
            if len(diffs) >= 6:
                w_stat, w_p = stats.wilcoxon(diffs, alternative='greater')
            else:
                w_stat, w_p = None, None
            
            stat_rows.append({
                'budget': budget,
                'comparison': f'{m1} vs {m2}',
                'paired_unit': 'space-level',
                'n_units': len(pairs),
                'mean_gain': f'{mean_gain:.6f}',
                'median_gain': f'{median_gain:.6f}',
                'win_rate': f'{win_rate:.3f}',
                'bootstrap_CI_low': f'{low:.6f}',
                'bootstrap_CI_high': f'{high:.6f}',
                'wilcoxon_stat': f'{w_stat:.4f}' if w_stat else '',
                'wilcoxon_p': f'{w_p:.6f}' if w_p else '',
                'interpretation': f'{m1} has {win_rate:.0%} win rate vs {m2} with mean gain {mean_gain:.4f}',
            })
            
            side = 'better' if mean_gain > 0 else 'worse'
            print(f"  {m1} vs {m2}: {win_rate:.0%} win rate, mean Δ={mean_gain:.4f} ({side}), "
                  f"boot 95% CI=[{low:.4f}, {high:.4f}]")

with open(OUT_DIR / "phase2_statistical_tests.csv", 'w', newline='') as f:
    if stat_rows:
        w = csv.DictWriter(f, fieldnames=list(stat_rows[0].keys()))
        w.writeheader()
        w.writerows(stat_rows)
print(f"Saved phase2_statistical_tests.csv")

# ================================================================
# 6. Winner distribution
# ================================================================
winner_rows = []
for budget in BUDGETS:
    tasks = set((r['search_space_id'], r['task_id']) for r in all_results if r['budget']==budget)
    
    unique_wins = Counter()
    tied_wins = Counter()
    
    for (ss_id, task) in sorted(tasks):
        t_rows = [r for r in all_results if r['search_space_id']==ss_id and r['task_id']==task and r['budget']==budget]
        best_regret = min(r['normalized_regret'] for r in t_rows)
        best_methods = [r['method'] for r in t_rows if abs(r['normalized_regret'] - best_regret) < 1e-8]
        if len(best_methods) == 1:
            unique_wins[best_methods[0]] += 1
        else:
            for m in best_methods:
                tied_wins[m] += 1
    
    for method in METHODS:
        uw = unique_wins.get(method, 0)
        tw = tied_wins.get(method, 0)
        if uw + tw > 0:
            winner_rows.append({
                'budget': budget, 'method': method, 'unique_wins': uw, 'tied_wins': tw,
                'win_percent': f'{uw / len(tasks) * 100:.1f}%',
            })

with open(OUT_DIR / "phase2_winner_distribution.csv", 'w', newline='') as f:
    if winner_rows:
        w = csv.DictWriter(f, fieldnames=list(winner_rows[0].keys()))
        w.writeheader()
        w.writerows(winner_rows)
print(f"Saved phase2_winner_distribution.csv")

# ================================================================
# 7. Failure analysis
# ================================================================
failure_rows = []
for ss_id in SPACES:
    info = desc.get(ss_id, {})
    dim = len(info.get('variables', {}))
    algo = ALGO_MAP.get(ss_id, '?')
    n_configs = CONFIG_MAP.get(ss_id, 0)
    
    budget = 25
    regrets = {}
    for m in METHODS:
        v = [r['normalized_regret'] for r in all_results if r['search_space_id']==ss_id and r['method']==m and r['budget']==budget]
        if v:
            regrets[m] = float(np.mean(v))
    
    if not regrets:
        continue
    
    best_method = min(regrets, key=regrets.get)
    cand_r = regrets.get('rf_candidate_2k', 1.0)
    best_r = regrets[best_method]
    
    # Rank
    sorted_m = sorted(regrets, key=regrets.get)
    rank = sorted_m.index('rf_candidate_2k') + 1 if 'rf_candidate_2k' in sorted_m else '-'
    
    # Failure type
    if best_method == 'rf_candidate_2k':
        failure_type = 'SUCCESS'
        explanation = 'rf_candidate_2k is the best method'
    elif cand_r < 0.05 and regrets.get('random', 1.0) < 0.05 and regrets.get('rf_smbo', 1.0) < 0.05:
        failure_type = 'ALL_SATURATED'
        explanation = 'All methods achieve near-zero regret'
    elif dim < 5:
        failure_type = 'LOW_DIMENSIONAL'
        explanation = 'Low-dim space where random search covers well'
    elif n_configs < 500:
        failure_type = 'SMALL_CONFIG_SPACE'
        explanation = 'Small config space, limited exploration advantage'
    elif best_method == 'rf_smbo':
        failure_type = 'RF_SMBO_DOMINATES'
        explanation = 'Full-candidate RF-SMBO slightly better'
    elif best_method == 'random':
        failure_type = 'RANDOM_DOMINATES'
        explanation = 'Random search sufficient on this space'
    elif best_method == 'tpe':
        failure_type = 'TPE_DOMINATES'
        explanation = 'TPE unexpectedly strong on this space'
    else:
        failure_type = 'OTHER'
        explanation = f'Best method: {best_method}'
    
    included = 'yes' if (failure_type == 'SUCCESS' or (cand_r < 0.1 and cand_r - best_r < 0.02)) else 'appendix_only'
    
    failure_rows.append({
        'search_space_id': ss_id, 'algorithm': algo, 'dimension': dim, 'n_configs': n_configs,
        'best_method': best_method, 'rf_candidate_2k_rank': rank,
        'rf_candidate_2k_regret': f'{cand_r:.6f}', 'best_method_regret': f'{best_r:.6f}',
        'failure_type': failure_type, 'explanation': explanation,
        'should_include_in_paper_main': 'yes' if failure_type == 'SUCCESS' else 'no',
        'should_include_in_appendix': included,
    })

with open(OUT_DIR / "phase2_failure_analysis.csv", 'w', newline='') as f:
    if failure_rows:
        w = csv.DictWriter(f, fieldnames=list(failure_rows[0].keys()))
        w.writeheader()
        w.writerows(failure_rows)
print(f"Saved phase2_failure_analysis.csv")

# Print summary
print()
print(f"{'='*60}")
print("FAILURE ANALYSIS (B=25)")
print(f"{'='*60}")
success = [r for r in failure_rows if r['failure_type'] == 'SUCCESS']
for r in failure_rows:
    print(f"  {r['search_space_id']:>6} ({r['algorithm']:<7} d={r['dimension']:<2} c={r['n_configs']:<6}): "
          f"{r['failure_type']:<20} best={r['best_method']:<15} cand_rank={r['rf_candidate_2k_rank']}")

print(f"\nSUCCESS: {len(success)}/{len(failure_rows)} spaces where rf_candidate_2k is the best method")
print(f"Paper includes: {sum(1 for r in failure_rows if r['should_include_in_paper_main']=='yes')}/{len(failure_rows)}")

# ================================================================
# 8. Generate final summary for report
# ================================================================
print()
print(f"{'='*60}")
print("FINAL PHASE 2 SUMMARY")
print(f"{'='*60}")

for r in agg_rows:
    if r['budget'] == 25:
        print(f"  {r['method']:<20} macro={r['macro_mean_regret']} {r['gain_vs_random']:>8} vs random, {r['gain_vs_rf_smbo']:>8} vs rf_smbo")

print(f"\nTotal test tasks: {len(set((r['search_space_id'], r['task_id']) for r in all_results))}")
print(f"Total seeds: {max(r['seed'] for r in all_results) + 1}")
print(f"Total spaces: {len(SPACES)}")
print(f"Total result rows: {len(all_results)}")
