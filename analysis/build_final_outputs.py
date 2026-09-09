#!/usr/bin/env python3
"""Build final paper outputs: tables, figure data, sensitivity analyses, method section, skeleton."""

import sys, json, csv
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np

ROOT = _RUNS
OUT_DIR = ROOT / "audit_workspace" / "hpob_final_paper_package"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Sources
P2_DIR = ROOT / "audit_workspace" / "hpob_phase2_confirmatory"
CACHE_DIR = P2_DIR / "phase2_cache"
SE_CACHE = ROOT / "audit_workspace" / "hpob_phase1_go_no_go" / "stageE_expansion" / "stageE_cache"
STAGE_D_RESULTS = ROOT / "audit_workspace" / "hpob_phase1_go_no_go" / "hpob_stageD_hybrid_results_6767.csv"

ALGO_MAP = {
    '6767': 'xgboost', '5971': 'xgboost', '5906': 'xgboost',
    '6794': 'ranger', '5965': 'ranger', '7607': 'ranger', '7609': 'ranger', '5889': 'ranger',
    '5527': 'svm', '5891': 'svm',
    '5636': 'rpart', '5859': 'rpart',
    '4796': 'glmnet', '6766': 'glmnet',
}
CONFIG_MAP = {k: {'6767':44258,'5971':2265,'5906':254,'5965':11228,'6794':52448,
                   '7607':647,'7609':1269,'5527':44790,'5891':2584,'5636':43841,
                   '5859':2878,'5889':298,'4796':300,'6766':39155}[k] for k in ALGO_MAP}

print("Loading all data...")
all_results = []
for d in [CACHE_DIR, SE_CACHE]:
    if d.exists():
        for f in sorted(d.glob("*.json")):
            all_results.append(json.load(open(f)))
print(f"Total results loaded: {len(all_results)}")

# Add metadata
for r in all_results:
    ss = r['search_space_id']
    r['algorithm'] = ALGO_MAP.get(ss, '?')
    info = {k: len(v.get('variables', {})) for k, v in {}}
    try:
        pass
    except:
        pass

# Load descriptor
import json as _json
import os as _os
from pathlib import Path as _Path
_RUNS = _Path(_os.environ.get("RUNS_DIR", _Path(__file__).resolve().parent.parent / "runs"))
with open('/tmp/HPO-B/hpob-data/meta-dataset-descriptors.json') as f:
    desc = _json.load(f)

for r in all_results:
    ss = r['search_space_id']
    info = desc.get(ss, {})
    r['dimension'] = len(info.get('variables', {}))
    r['n_configs'] = CONFIG_MAP.get(ss, 0)

METHODS = ['random', 'tpe', 'rf_smbo', 'rf_candidate_2k']
BUDGETS = [5, 10, 25]
SPACES = sorted(set(r['search_space_id'] for r in all_results if r['search_space_id'] in ALGO_MAP))
seeds_present = {ss: len(set(r['seed'] for r in all_results if r['search_space_id']==ss)) for ss in SPACES}

print(f"Spaces: {SPACES}")
print(f"Seed coverage: {dict(sorted(seeds_present.items()))}")

# ======================================================================
# TABLE 1: Main results (macro-average at B=5,10,25)
# ======================================================================
print("\n=== TABLE 1: Main Results ===")
t1_rows = []
for budget in BUDGETS:
    macro = defaultdict(list)
    for ss in SPACES:
        for m in METHODS:
            v = [r['normalized_regret'] for r in all_results if r['search_space_id']==ss and r['method']==m and r['budget']==budget]
            if v:
                macro[m].append(float(np.mean(v)))
    
    row = {'budget': budget}
    for m in METHODS:
        if macro[m]:
            row[m] = f"{np.mean(macro[m]):.6f}"
            row[f'{m}_std'] = f"{np.std(macro[m]):.6f}"
    row['n_spaces'] = len(macro.get('rf_candidate_2k', []))
    t1_rows.append(row)
    print(f"  B={budget}: cand={row.get('rf_candidate_2k','-')}  rf={row.get('rf_smbo','-')}  rand={row.get('random','-')}  tpe={row.get('tpe','-')}")

with open(OUT_DIR / "table1_main_results.csv", 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['budget','rf_candidate_2k','rf_candidate_2k_std','rf_smbo','rf_smbo_std','random','random_std','tpe','tpe_std','n_spaces'])
    w.writeheader()
    for r in t1_rows:
        w.writerow(r)

# ======================================================================
# TABLE 2: Per-space comparison at B=25
# ======================================================================
print("\n=== TABLE 2: Per-space comparison ===")
t2_rows = []
for ss in SPACES:
    info = desc.get(ss, {})
    dim = len(info.get('variables', {}))
    row = {'ss_id': ss, 'algorithm': ALGO_MAP.get(ss,'?'), 'dim': dim, 'n_configs': CONFIG_MAP.get(ss,0),
           'n_tasks': len(set(r['task_id'] for r in all_results if r['search_space_id']==ss))}
    
    regrets = {}
    for m in METHODS:
        v = [r['normalized_regret'] for r in all_results if r['search_space_id']==ss and r['method']==m and r['budget']==25]
        if v:
            regrets[m] = float(np.mean(v))
            row[m] = f"{regrets[m]:.6f}"
    
    if regrets:
        best = min(regrets, key=regrets.get)
        row['best_method'] = best
        ranks = sorted(regrets, key=regrets.get)
        row['cand_rank'] = ranks.index('rf_candidate_2k') + 1 if 'rf_candidate_2k' in ranks else '-'
    t2_rows.append(row)
    
    cand = row.get('rf_candidate_2k', '?')
    rf = row.get('rf_smbo', '?')
    marker = ' <<' if 'rf_candidate_2k' in row.get('best_method','') else ''
    print(f"  {ss}: cand={cand}  rf={rf}  best={row.get('best_method','?')}{marker}")

with open(OUT_DIR / "table2_per_space_b25.csv", 'w', newline='') as f:
    fieldnames = ['ss_id','algorithm','dim','n_configs','n_tasks','random','tpe','rf_smbo','rf_candidate_2k','best_method','cand_rank']
    w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
    w.writeheader(); w.writerows(t2_rows)

# ======================================================================
# TABLE 3: Subgroup analysis
# ======================================================================
print("\n=== TABLE 3: Subgroups ===")
def subgroup_table(members, label):
    rows = []
    for budget in [25]:
        for m in METHODS:
            vals = []
            for ss in members:
                vals.extend([r['normalized_regret'] for r in all_results if r['search_space_id']==ss and r['method']==m and r['budget']==budget])
            if vals:
                rows.append({'subgroup': label, 'budget': budget, 'method': m, 
                             'mean_regret': f"{np.mean(vals):.6f}", 'n': len(vals)})
    return rows

subgroups = {
    'High-dim (≥15D)': [s for s in SPACES if len(desc.get(s,{}).get('variables',[])) >= 15],
    'Med-dim (8–14D)': [s for s in SPACES if 8 <= len(desc.get(s,{}).get('variables',[])) < 15],
    'Low-dim (<8D)': [s for s in SPACES if len(desc.get(s,{}).get('variables',[])) < 8],
    'Large config (≥10K)': [s for s in SPACES if CONFIG_MAP.get(s,0) >= 10000],
    'Small config (<10K)': [s for s in SPACES if CONFIG_MAP.get(s,0) < 10000],
}

t3_rows = []
for sg_name, sg_spaces in subgroups.items():
    if sg_spaces:
        t3_rows.extend(subgroup_table(sg_spaces, sg_name))

with open(OUT_DIR / "table3_subgroup_analysis.csv", 'w', newline='') as f:
    if t3_rows:
        w = csv.DictWriter(f, fieldnames=list(t3_rows[0].keys()))
        w.writeheader(); w.writerows(t3_rows)

for sg_name, sg_spaces in subgroups.items():
    if not sg_spaces: continue
    cand = next((r for r in t3_rows if r['subgroup']==sg_name and r['method']=='rf_candidate_2k'), None)
    rf = next((r for r in t3_rows if r['subgroup']==sg_name and r['method']=='rf_smbo'), None)
    rand = next((r for r in t3_rows if r['subgroup']==sg_name and r['method']=='random'), None)
    if cand and rf:
        gain = ((float(rf['mean_regret']) - float(cand['mean_regret'])) / float(rf['mean_regret']) * 100)
        print(f"  {sg_name:<20} cand={cand['mean_regret']}  rf={rf['mean_regret']}  gain={gain:+.1f}%")

# ======================================================================
# TABLE 4: Statistical tests
# ======================================================================
print("\n=== TABLE 4: Statistics ===")
def bootstrap_ci(v1, v2, n=5000):
    from numpy.random import default_rng
    rng = default_rng(42)
    diffs = [a-b for a,b in zip(v1,v2)]
    boots = [np.mean(rng.choice(diffs, len(diffs), replace=True)) for _ in range(n)]
    boots.sort()
    return boots[125], boots[4875]

t4_rows = []
for budget in [25]:
    for m1, m2 in [('rf_candidate_2k','random'), ('rf_candidate_2k','rf_smbo'), ('rf_candidate_2k','tpe')]:
        pairs = []
        for ss in SPACES:
            v1 = [r['normalized_regret'] for r in all_results if r['search_space_id']==ss and r['method']==m1 and r['budget']==budget]
            v2 = [r['normalized_regret'] for r in all_results if r['search_space_id']==ss and r['method']==m2 and r['budget']==budget]
            if v1 and v2:
                pairs.append((float(np.mean(v1)), float(np.mean(v2))))
        
        if len(pairs) >= 3:
            d = [b-a for a,b in pairs]
            mean_g = float(np.mean(d))
            win_r = sum(1 for x in d if x > 0) / len(d)
            lo, hi = bootstrap_ci([p[0] for p in pairs], [p[1] for p in pairs])
            t4_rows.append({
                'budget': budget, 'comparison': f'{m1} vs {m2}', 'n_spaces': len(pairs),
                'mean_gain': f'{mean_g:.6f}', 'win_rate': f'{win_r:.3f}',
                'bootstrap_CI_low': f'{lo:.6f}', 'bootstrap_CI_high': f'{hi:.6f}',
                'significant': 'yes' if lo > 0 else 'no',
            })
            print(f"  {m1} vs {m2}: win={win_r:.0%}  gain={mean_g:.4f}  CI=[{lo:.4f},{hi:.4f}]  sig={'YES' if lo>0 else 'no'}")

with open(OUT_DIR / "table4_statistical_tests.csv", 'w', newline='') as f:
    if t4_rows:
        w = csv.DictWriter(f, fieldnames=list(t4_rows[0].keys()))
        w.writeheader(); w.writerows(t4_rows)

# ======================================================================
# TABLE 5: Winner distribution
# ======================================================================
print("\n=== TABLE 5: Winners ===")
t5_rows = []
for budget in BUDGETS:
    tasks = set((r['search_space_id'],r['task_id']) for r in all_results if r['budget']==budget)
    unique = Counter()
    for ss,task in tasks:
        best = min((m for m in METHODS if any(r['search_space_id']==ss and r['task_id']==task and r['method']==m and r['budget']==budget for r in all_results)),
                   key=lambda m: np.mean([r['normalized_regret'] for r in all_results if r['search_space_id']==ss and r['task_id']==task and r['method']==m and r['budget']==budget]))
        unique[best] += 1
    for m in METHODS:
        t5_rows.append({'budget': budget, 'method': m, 'unique_task_wins': unique.get(m,0), 'win_pct': f'{unique.get(m,0)/len(tasks)*100:.1f}%'})
    print(f"  B={budget}: {dict(unique)}")

with open(OUT_DIR / "table5_winner_distribution.csv", 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['budget','method','unique_task_wins','win_pct'])
    w.writeheader(); w.writerows(t5_rows)

# ======================================================================
# TABLE 6: Pool-size sensitivity (from Stage D)
# ======================================================================
print("\n=== TABLE 6: Pool-size sensitivity (6767, B=25) ===")
t6_rows = []
if STAGE_D_RESULTS.exists():
    with open(STAGE_D_RESULTS) as f:
        for r in csv.DictReader(f):
            if r['budget'] == '25':
                rr = float(r['normalized_regret'])
                t6_rows.append({'method': r['method'], 'mean_regret': f'{rr:.6f}'})
    # Aggregate
    from collections import OrderedDict
    pools = OrderedDict()
    for r in t6_rows:
        m = r['method']
        pools[m] = r['mean_regret']
    for m, v in pools.items():
        print(f"  {m:<20} {v}")

with open(OUT_DIR / "table6_pool_sensitivity.csv", 'w', newline='') as f:
    if t6_rows:
        w = csv.DictWriter(f, fieldnames=['method','mean_regret'])
        w.writeheader(); w.writerows(t6_rows)

# ======================================================================
# TABLE 7: Failure analysis
# ======================================================================
print("\n=== TABLE 7: Failure modes ===")
t7_rows = []
for ss in SPACES:
    info = desc.get(ss,{})
    dim = len(info.get('variables',{}))
    regrets = {}
    for m in METHODS:
        v = [r['normalized_regret'] for r in all_results if r['search_space_id']==ss and r['method']==m and r['budget']==25]
        if v: regrets[m] = float(np.mean(v))
    if not regrets: continue
    
    best = min(regrets, key=regrets.get)
    cand_r = regrets.get('rf_candidate_2k',1)
    best_r = regrets[best]
    
    if all(r < 0.03 for r in regrets.values()):
        fail = 'Saturated'
    elif dim < 5 and best == 'rf_candidate_2k':
        fail = 'Success'
    elif best == 'rf_candidate_2k':
        fail = 'Success'
    elif best in ['rf_smbo','random'] and CONFIG_MAP.get(ss,0) < 500:
        fail = 'Small config'
    elif best == 'rf_smbo' and cand_r - best_r < 0.01:
        fail = 'Near-tie'
    elif best == 'rf_smbo':
        fail = 'RF-smi beats cand'
    elif best == 'random':
        fail = 'Random beats cand'
    elif best == 'tpe':
        fail = 'TPE beats cand'
    else:
        fail = 'Other'
    
    t7_rows.append({
        'ss_id': ss, 'algo': ALGO_MAP.get(ss,'?'), 'dim': dim, 'n_configs': CONFIG_MAP.get(ss,0),
        'best_method': best, 'cand_regret': f'{cand_r:.4f}', 'best_regret': f'{best_r:.4f}',
        'failure_type': fail, 'diff': f'{cand_r - best_r:.4f}',
    })
    marker = ' ✱' if fail == 'Success' else ''
    print(f"  {ss}: {fail:<20} best={best}  cand={cand_r:.4f}  best={best_r:.4f}{marker}")

success = sum(1 for r in t7_rows if r['failure_type'] == 'Success')
print(f"\n  Success: {success}/{len(t7_rows)} spaces")

with open(OUT_DIR / "table7_failure_analysis.csv", 'w', newline='') as f:
    if t7_rows:
        w = csv.DictWriter(f, fieldnames=list(t7_rows[0].keys()))
        w.writeheader(); w.writerows(t7_rows)

# ======================================================================
# FIGURE DATA: Main result bar chart
# ======================================================================
print("\n=== Figure data ===")
fig_rows = []
for budget in BUDGETS:
    macro = defaultdict(list)
    for ss in SPACES:
        for m in METHODS:
            v = [r['normalized_regret'] for r in all_results if r['search_space_id']==ss and r['method']==m and r['budget']==budget]
            if v:
                macro[m].append(float(np.mean(v)))
    for m in METHODS:
        if macro[m]:
            fig_rows.append({'budget': budget, 'method': m, 'mean_regret': f'{np.mean(macro[m]):.6f}', 'std_regret': f'{np.std(macro[m]):.6f}'})

with open(OUT_DIR / "fig2_main_bars.csv", 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['budget','method','mean_regret','std_regret'])
    w.writeheader(); w.writerows(fig_rows)

# Per-space scatter data
fig3 = []
for ss in SPACES:
    cand = np.mean([r['normalized_regret'] for r in all_results if r['search_space_id']==ss and r['method']=='rf_candidate_2k' and r['budget']==25])
    rf = np.mean([r['normalized_regret'] for r in all_results if r['search_space_id']==ss and r['method']=='rf_smbo' and r['budget']==25])
    fig3.append({'ss_id': ss, 'algo': ALGO_MAP.get(ss,'?'), 'dim': len(desc.get(ss,{}).get('variables',[])),
                 'rf_candidate_2k': f'{cand:.6f}', 'rf_smbo': f'{rf:.6f}'})

with open(OUT_DIR / "fig3_scatter_b25.csv", 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['ss_id','algo','dim','rf_candidate_2k','rf_smbo'])
    w.writeheader(); w.writerows(fig3)

# Budget trajectory data
fig6 = []
for budget in BUDGETS:
    for m in METHODS:
        vals = [r['normalized_regret'] for r in all_results if r['method']==m and r['budget']==budget]
        if vals:
            fig6.append({'budget': budget, 'method': m, 'mean': f'{np.mean(vals):.6f}', 'std': f'{np.std(vals):.6f}'})

with open(OUT_DIR / "fig6_budget_trajectory.csv", 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['budget','method','mean','std'])
    w.writeheader(); w.writerows(fig6)

print("\nAll tables and figure data saved to:", OUT_DIR)
print(f"Files created: {len(list(OUT_DIR.glob('*.csv')))} CSV files")
