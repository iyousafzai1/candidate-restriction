"""DDF-Deploy: evaluate a leak-free, deployable DDF variant by offline replay.

DDF-Oracle (the version reported as the paper's headline result) uses oracle
AQ = (y* - max_anchor) / (y* - y_worst), which needs the test task's true
optimum -- unavailable at real routing time.

DDF-Deploy replaces oracle AQ with a leak-free surrogate AQ_hat (matrix-SVD
completion from anchors only; see ceiling_estimator.py) and a conservative
three-zone rule (calibrate_aq_thresholds.py):
    AQ_hat <= tau1  -> confident ceiling
    AQ_hat >= tau2  -> confident headroom
    otherwise       -> abstain -> safe default (bo_lrtc_feature)

Per the reviewer discussion, DDF-Deploy intentionally DROPS Probes 3
(Tucker identifiability -- itself not leak-free, and empirically never
fires: Regime B = 0/144 tasks) and 4 (task cohesion -- leak-free, but kept
out of the primary deployable headline and left as a diagnostic/ablation
extension). The primary deployable rule is:

    d <= 2                : ceiling->active_pmf, headroom->tpe, abstain->bo_lrtc_feature
    3 <= d <= 5            : ceiling->(d<=3: bo_lrtc_feature, d>3: bo_cold),
                             headroom->bo_lrtc_feature, abstain->bo_lrtc_feature
    d >= 6                 : rgpe (Probe 2 alone is already decisive here in
                             DDF-Oracle too -- Regime G never consults AQ)

This is evaluated by OFFLINE POLICY REPLAY, not a live rerun: all component
methods were already evaluated under the identical, deterministic,
method-independent anchor protocol (verified separately -- the k0=5 probe
anchors are a pure function of the training tensor, identical across every
method/seed/task). So for each of the 144 held-out test tasks, the routing
rule selects a method, and that method's regret is read from its
precomputed trajectory in results/raw/main/.

Reports DDF-Deploy-Scenario (per-scenario calibrated thresholds) and
DDF-Deploy-Global (one pooled threshold pair), alongside DDF-Oracle and
Best-Fixed (bo_lrtc_feature) for comparison.
"""
from __future__ import annotations

import sys
import csv
import json
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import tensorize                                              # noqa: E402
from prep import Problem                                      # noqa: E402
from ceiling_estimator import estimate_task, K0 as DEFAULT_K0  # noqa: E402
from ddf import (DIM_LOW, DIM_MID_SPLIT, DIM_HIGH,             # noqa: E402
                 AQ_CEILING_THR, recommend)

SCENARIOS = ["rbv2_svm_radial", "rbv2_glmnet", "rbv2_rpart",
             "rbv2_ranger", "lcbench"]
BUDGET = 25
RESULTS_DIR = ROOT / "results" / "raw" / "main"
CALIB_DIR = ROOT / "analysis" / "ddf_deploy"
OUT_DIR = ROOT / "analysis" / "ddf_deploy"
SAFE_DEFAULT = "bo_lrtc_feature"


def load_task_regret(results_dir: Path, budget: int = BUDGET) -> dict:
    """data[scenario][task][method] = mean regret at `budget` across seeds."""
    data: dict = {}
    for scen_dir in sorted(results_dir.iterdir()):
        if not scen_dir.is_dir():
            continue
        scenario = scen_dir.name
        data[scenario] = defaultdict(dict)
        for method_dir in sorted(scen_dir.iterdir()):
            if not method_dir.is_dir():
                continue
            method = method_dir.name
            task_seeds = defaultdict(list)
            for fpath in method_dir.glob("*.json"):
                task = fpath.stem.split("__seed")[0]
                try:
                    with open(fpath) as f:
                        d = json.load(f)
                    regrets = {cp["budget"]: cp["normalized_regret"]
                               for cp in d["checkpoints"]}
                    b = max((b for b in regrets if b <= budget), default=None)
                    if b is None:
                        b = min(regrets)
                    task_seeds[task].append(regrets[b])
                except Exception:
                    continue
            for task, regrets in task_seeds.items():
                data[scenario][task][method] = float(np.mean(regrets))
    return data


def route_deploy(d: int, aq_hat: float, tau1: float, tau2: float) -> tuple[str, str]:
    """Return (method, zone) for the leak-free deployable rule."""
    if d >= DIM_HIGH + 1:                       # d >= 6
        return "rgpe", "dim-only"
    if aq_hat <= tau1:
        zone = "confident_ceiling"
        if d <= DIM_LOW:                        # d <= 2
            return "active_pmf", zone
        return ("bo_lrtc_feature" if d <= DIM_MID_SPLIT else "bo_cold"), zone
    if aq_hat >= tau2:
        zone = "confident_headroom"
        if d <= DIM_LOW:
            return "tpe", zone
        return "bo_lrtc_feature", zone
    return SAFE_DEFAULT, "abstain"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k0", type=int, default=DEFAULT_K0)
    ap.add_argument("--budget", type=int, default=BUDGET)
    ap.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    ap.add_argument("--calib-dir", type=Path, default=CALIB_DIR)
    ap.add_argument("--outdir", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    k0 = args.k0
    budget = args.budget
    results_dir = args.results_dir
    calib_dir = args.calib_dir
    out_dir = args.outdir

    thresholds = json.load(open(calib_dir / "aq_thresholds.json"))
    regret_lookup = load_task_regret(results_dir, budget=budget)

    rows = []
    for scen in SCENARIOS:
        gdir = Path(tensorize.grid_dir(ROOT, "paper_d", scen))
        g = tensorize.load_grid(gdir)
        tensor = g["tensor"]
        task_ids = g["task_ids"]
        train_tasks = list(g["meta"]["train_task_ids"])
        test_tasks = list(g["meta"]["test_task_ids"])

        thr_scen = thresholds["per_scenario"][scen]
        thr_glob = thresholds["global"]

        for task in test_tasks:
            prob = Problem(tensor, task_ids, train_tasks, task, k0=k0,
                           anchor_mode="informed")
            est = estimate_task(prob, k0)
            d = est["d"]
            aq_hat = est["aq_hat_matrix"]
            aq_true = est["aq_true"]

            m_scen, zone_scen = route_deploy(
                d, aq_hat, thr_scen["tau1_confident_ceiling"],
                thr_scen["tau2_confident_headroom"])
            m_glob, zone_glob = route_deploy(
                d, aq_hat, thr_glob["tau1_confident_ceiling"],
                thr_glob["tau2_confident_headroom"])

            oracle_rec = recommend(prob)
            m_oracle = oracle_rec["recommended"]

            task_key = str(task)
            lut = regret_lookup.get(scen, {}).get(task_key, {})

            rows.append({
                "scenario": scen, "task": task_key, "d": d,
                "aq_true": aq_true, "aq_hat_matrix": aq_hat,
                "method_deploy_scenario": m_scen, "zone_scenario": zone_scen,
                "regret_deploy_scenario": lut.get(m_scen),
                "method_deploy_global": m_glob, "zone_global": zone_glob,
                "regret_deploy_global": lut.get(m_glob),
                "method_oracle": m_oracle,
                "regret_oracle": lut.get(m_oracle),
                "regret_best_fixed": lut.get("bo_lrtc_feature"),
            })

    out_dir.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with open(out_dir / "ddf_deploy_task_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    def summarize(col: str) -> float:
        vals = [r[col] for r in rows if r[col] is not None]
        missing = len(rows) - len(vals)
        if missing:
            print(f"  [warn] {col}: {missing} tasks missing a regret lookup")
        return float(np.mean(vals))

    print(f"\nn_tasks = {len(rows)}\n")
    print(f"{'Variant':<28}{'Task-weighted mean regret':>28}")
    print(f"{'Best-Fixed (bo-lrtc-feature)':<28}{summarize('regret_best_fixed'):>28.5f}")
    print(f"{'DDF-Oracle':<28}{summarize('regret_oracle'):>28.5f}")
    print(f"{'DDF-Deploy-Scenario':<28}{summarize('regret_deploy_scenario'):>28.5f}")
    print(f"{'DDF-Deploy-Global':<28}{summarize('regret_deploy_global'):>28.5f}")

    # zone usage breakdown
    for variant, zonecol in [("Scenario", "zone_scenario"), ("Global", "zone_global")]:
        zones = defaultdict(int)
        for r in rows:
            zones[r[zonecol]] += 1
        print(f"\nZone usage ({variant}): {dict(zones)}")


if __name__ == "__main__":
    main()
