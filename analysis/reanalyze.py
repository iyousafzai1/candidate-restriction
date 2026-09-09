#!/usr/bin/env python3
"""
Canonical re-analysis for rf_candidate_2k (HPO-B Phase 2).
Reads the phase2_cache JSON records, rebuilds:
  - missing_cells.csv        : method x task x seed x budget cells needed to complete the grid
  - grid_coverage.csv        : per (space,task) seed coverage
  - canonical_per_space_b25.csv
  - canonical_macro.csv       : B=5/10/25 macro-average
  - canonical_stat_tests.csv  : task-level paired Wilcoxon (+ regime split)
Everything is computed on a BALANCED basis: a (space,task,seed) cell is used
for a comparison only if BOTH compared methods are present.
Target grid: seeds 0..TARGET_SEEDS-1, methods {random,tpe,rf_smbo,rf_candidate_2k}, budgets {5,10,25}.
"""
import json, glob, collections, csv, os, sys, numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
PKG  = os.path.dirname(HERE)
# cache location (edit if you move it)
CACHE = os.environ.get("CACHE_DIR", os.path.join(PKG, "2026-07-07-09-51-55",
        "audit_workspace","hpob_phase2_confirmatory","phase2_cache"))
TARGET_SEEDS = int(os.environ.get("TARGET_SEEDS", "20"))
METHODS = ['random','tpe','rf_smbo','rf_candidate_2k']
BUDGETS = [5,10,25]
OUT = HERE

meta = {r['ss_id']:(int(r['dim']),int(r['n_configs']))
        for r in csv.DictReader(open(os.path.join(PKG,"table2_per_space_b25.csv")))}

recs=[]
for f in glob.glob(os.path.join(CACHE,"*.json")):
    try: recs.append(json.load(open(f)))
    except: pass
print(f"loaded {len(recs)} cache records from {CACHE}")

# present[(space,task,seed,budget,method)] = regret
present={}
tasks_by_space=collections.defaultdict(set)
for r in recs:
    key=(r['search_space_id'], str(r['task_id']), int(r['seed']), int(r['budget']), r['method'])
    present[key]=float(r['normalized_regret'])
    tasks_by_space[r['search_space_id']].add(str(r['task_id']))

all_tasks=sorted({(ss,t) for ss,ts in tasks_by_space.items() for t in ts})
print(f"{len(all_tasks)} (space,task) pairs across {len(tasks_by_space)} spaces")

# ---- missing cells to reach the target grid ----
missing=[]
for ss,t in all_tasks:
    for seed in range(TARGET_SEEDS):
        for b in BUDGETS:
            for m in METHODS:
                if (ss,t,seed,b,m) not in present:
                    missing.append((ss,t,seed,b,m))
with open(os.path.join(OUT,"missing_cells.csv"),"w",newline="") as f:
    w=csv.writer(f); w.writerow(["ss_id","task_id","seed","budget","method"])
    w.writerows(missing)
print(f"missing cells to reach {TARGET_SEEDS} seeds x {len(METHODS)} methods x {len(BUDGETS)} budgets: {len(missing)}")

# ---- coverage per (space,task): seeds where ALL methods present at B=25 ----
with open(os.path.join(OUT,"grid_coverage.csv"),"w",newline="") as f:
    w=csv.writer(f); w.writerow(["ss_id","task_id","dim","n_configs","seeds_full_coverage_b25"])
    cov={}
    for ss,t in all_tasks:
        c=sum(all((ss,t,s,25,m) in present for m in METHODS) for s in range(TARGET_SEEDS))
        cov[(ss,t)]=c
        w.writerow([ss,t,meta[ss][0],meta[ss][1],c])
from collections import Counter
print("seed-coverage histogram (B=25):", dict(sorted(Counter(cov.values()).items())))

def task_means(budget, seeds, require_full=False):
    """return dict method-> {(ss,t):mean regret} using common seeds where all methods present."""
    out=collections.defaultdict(dict)
    for ss,t in all_tasks:
        good=[s for s in seeds if all((ss,t,s,budget,m) in present for m in METHODS)]
        if require_full and len(good)<len(seeds): continue
        if not good: continue
        for m in METHODS:
            out[m][(ss,t)]=np.mean([present[(ss,t,s,budget,m)] for m in [m] for s in good])
    return out

def holm(p):
    p=np.array(p,float); idx=np.argsort(p); adj=np.empty(len(p)); mm=len(p); prev=0
    for rank,i in enumerate(idx):
        prev=max(prev,(mm-rank)*p[i]); adj[i]=min(prev,1.0)
    return adj

# ---- canonical macro (space-mean over task-means over common seeds), all budgets ----
seeds=set(range(TARGET_SEEDS))
with open(os.path.join(OUT,"canonical_macro.csv"),"w",newline="") as f:
    w=csv.writer(f); w.writerow(["budget"]+METHODS+["cand_vs_smbo_pct","cand_vs_random_pct","cand_vs_tpe_pct","basis"])
    for b in BUDGETS:
        for basis,rf in [("all_common_seeds",False),("balanced_full20",True)]:
            tm=task_means(b,seeds,require_full=rf)
            common=set.intersection(*[set(tm[m].keys()) for m in METHODS])
            spaces=sorted({ss for ss,_ in common})
            macro={m:np.mean([np.mean([tm[m][(ss,t)] for (s2,t) in common if s2==ss]) for ss in spaces]) for m in METHODS}
            def pct(a): return 100*(macro['rf_smbo' if a=='rf_smbo' else a]-macro['rf_candidate_2k'])/macro[a]
            w.writerow([b]+[f"{macro[m]:.4f}" for m in METHODS]+
                       [f"{100*(macro['rf_smbo']-macro['rf_candidate_2k'])/macro['rf_smbo']:+.1f}",
                        f"{100*(macro['random']-macro['rf_candidate_2k'])/macro['random']:+.1f}",
                        f"{100*(macro['tpe']-macro['rf_candidate_2k'])/macro['tpe']:+.1f}", basis])

# ---- canonical per-space B=25 (all-common-seeds basis) ----
tm=task_means(25,seeds,require_full=False)
common=set.intersection(*[set(tm[m].keys()) for m in METHODS])
spaces=sorted({ss for ss,_ in common})
with open(os.path.join(OUT,"canonical_per_space_b25.csv"),"w",newline="") as f:
    w=csv.writer(f); w.writerow(["ss_id","dim","n_configs","n_tasks"]+METHODS+["gain_vs_smbo","best"])
    for ss in spaces:
        ts=[(s2,t) for (s2,t) in common if s2==ss]
        row={m:np.mean([tm[m][k] for k in ts]) for m in METHODS}
        best=min(METHODS,key=lambda m:row[m])
        w.writerow([ss,meta[ss][0],meta[ss][1],len(ts)]+[f"{row[m]:.4f}" for m in METHODS]+
                   [f"{row['rf_smbo']-row['rf_candidate_2k']:+.4f}",best])

# ---- canonical statistical tests (task-level paired Wilcoxon) ----
def run_tests(require_full, label, rows_out):
    tm=task_means(25,seeds,require_full=require_full)
    common=sorted(set.intersection(*[set(tm[m].keys()) for m in METHODS]))
    cand=np.array([tm['rf_candidate_2k'][k] for k in common])
    nconf=np.array([meta[k[0]][1] for k in common])
    praw=[]; specs=[]
    for name in ['rf_smbo','random','tpe']:
        b=np.array([tm[name][k] for k in common]); d=b-cand
        p=stats.wilcoxon(cand,b).pvalue; praw.append(p)
        specs.append((name,d,b,p))
    adj=holm(praw)
    for i,(name,d,b,p) in enumerate(specs):
        rows_out.append([label,len(common),f"cand_vs_{name}","all",
                         f"{d.mean():+.4f}",f"{100*d.mean()/b.mean():+.1f}",
                         int((d>1e-9).sum()),int((d<-1e-9).sum()),f"{p:.4g}",f"{adj[i]:.4g}"])
    # regime split vs rf_smbo
    smbo=np.array([tm['rf_smbo'][k] for k in common]); d=smbo-cand
    for rlab,mask in [("pool_gt_2000",nconf>2000),("pool_le_2000",nconf<=2000)]:
        if mask.sum()>=3:
            p=stats.wilcoxon(cand[mask],smbo[mask]).pvalue
            rows_out.append([label,int(mask.sum()),"cand_vs_rf_smbo",rlab,
                             f"{d[mask].mean():+.4f}",f"{100*d[mask].mean()/smbo[mask].mean():+.1f}",
                             int((d[mask]>1e-9).sum()),int((d[mask]<-1e-9).sum()),f"{p:.4g}",""])

rows_out=[]
run_tests(False,"all_common_seeds",rows_out)
run_tests(True ,"balanced_full20",rows_out)
with open(os.path.join(OUT,"canonical_stat_tests.csv"),"w",newline="") as f:
    w=csv.writer(f); w.writerow(["basis","n","comparison","subset","mean_gain","gain_pct","wins","losses","wilcoxon_p","holm_p"])
    w.writerows(rows_out)

print("\nwrote: missing_cells.csv, grid_coverage.csv, canonical_macro.csv, canonical_per_space_b25.csv, canonical_stat_tests.csv")
