"""Main BO results analysis for the ML Springer characterization paper.

Reads:
  results/raw/main/              — 23,760 JSON files (informed anchor mode)
  results/raw/anchor_robustness/ — 18,700 JSON files (informed + random anchor modes)

Produces:
  tables/ml_springer/T2_regret_table.csv    — mean regret ± std @ {10,25,50}, all methods x scenarios
  tables/ml_springer/T2_wilcoxon.csv        — Wilcoxon pairwise p-values (Holm-corrected)
  tables/ml_springer/T4_anchor_winners.csv  — per-scenario winner shifts informed→random
  figures/ml_springer/F2_regret_curves.pdf  — 5-panel regret curves, all methods, budgets 5-50
  figures/ml_springer/F3_anchor_delta.pdf   — warm-cold Δregret bar + AQ scatter

Usage:
    cd <repository-root>
    PYTHONPATH=src python3 scripts/analyze_main_results.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.stats import wilcoxon

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))

MAIN_DIR = ROOT / "results" / "raw" / "main"
AR_DIR   = ROOT / "results" / "raw" / "anchor_robustness"
FIG_DIR  = ROOT / "figures" / "ml_springer"
TAB_DIR  = ROOT / "tables" / "ml_springer"

BUDGETS     = [5, 10, 15, 25, 50]
KEY_BUDGETS = [10, 25, 50]
BUDGET_PRIMARY = 25

SCEN_ORDER = [
    "rbv2_svm_radial",
    "rbv2_glmnet",
    "rbv2_ranger",
    "rbv2_rpart",
    "lcbench",
]
SCEN_LABEL = {
    "rbv2_svm_radial": "svm 2-D",
    "rbv2_glmnet":     "glmnet 2-D",   # 2 HP axes: alpha, s
    "rbv2_ranger":     "ranger 3-D",   # 3 HP axes: num.trees, sample.frac, min.node.size
    "rbv2_rpart":      "rpart 4-D",    # 4 HP axes: cp, maxdepth, minbucket, minsplit
    "lcbench":         "lcbench 7-D",
}

# Method display order (best to worst by expected rank)
METHOD_ORDER = [
    "rgpe", "bo_lrtc_feature", "active_pmf", "bo_lrtc_prior", "bo_lrtc_refit",
    "bo_pmf_prior", "bo_lrtc_pibo", "bo_cold", "bo_lrtc_hybrid",
    "bo_lrtc_portfolio", "tpe", "lrtc_oneshot", "rs",
]
METHOD_LABEL = {
    "rgpe":             "RGPE",
    "bo_lrtc_feature":  "bo-lrtc-feat",
    "active_pmf":       "active-PMF",
    "bo_lrtc_prior":    "bo-lrtc-prior",
    "bo_lrtc_refit":    "bo-lrtc-refit",
    "bo_pmf_prior":     "bo-pmf-prior",
    "bo_lrtc_pibo":     "bo-lrtc-πBO",
    "bo_cold":          "bo-cold",
    "bo_lrtc_hybrid":   "bo-lrtc-hyb",
    "bo_lrtc_portfolio":"bo-lrtc-port",
    "tpe":              "TPE",
    "lrtc_oneshot":     "lrtc-oneshot",
    "rs":               "RS",
}

# Colorblind-safe palette (Wong 2011, 8 colors) + extras
_BASE_COLORS = [
    "#0072B2",  # blue     — rgpe
    "#D55E00",  # vermilion — bo_lrtc_feature (highlight)
    "#009E73",  # green    — active_pmf
    "#CC79A7",  # pink     — bo_lrtc_prior
    "#56B4E9",  # sky blue — bo_lrtc_refit
    "#E69F00",  # orange   — bo_pmf_prior
    "#F0E442",  # yellow   — bo_lrtc_pibo
    "#999999",  # grey     — bo_cold (baseline)
    "#AA4499",  # purple   — bo_lrtc_hybrid
    "#44AA99",  # teal     — bo_lrtc_portfolio
    "#332288",  # indigo   — tpe
    "#BBBBBB",  # light grey — lrtc_oneshot
    "#DDDDDD",  # very light — rs
]
METHOD_COLOR = {m: _BASE_COLORS[i % len(_BASE_COLORS)]
                for i, m in enumerate(METHOD_ORDER)}

WARM_METHODS = [m for m in METHOD_ORDER if m not in ("bo_cold", "rs", "tpe")]


# ── I/O helpers ────────────────────────────────────────────────────────────

def _load_dir(directory: Path) -> pd.DataFrame:
    """Load all JSONs under directory into a long DataFrame.

    Returns columns: scenario, method, task, seed, budget,
                     normalized_regret, anchor_mode, regret_at_anchors.
    """
    rows = []
    for p in directory.rglob("*.json"):
        try:
            with open(p) as f:
                r = json.load(f)
        except Exception:
            continue
        mode = r.get("anchor_mode", "informed")
        aq   = r.get("regret_at_anchors", float("nan"))
        scen = r.get("scenario", "")
        meth = r.get("method", "")
        task = str(r.get("task", ""))
        seed = int(r.get("seed", 0))
        for cp in r.get("checkpoints", []):
            rows.append({
                "scenario":        scen,
                "method":          meth,
                "task":            task,
                "seed":            seed,
                "budget":          int(cp["budget"]),
                "normalized_regret": float(cp["normalized_regret"]),
                "anchor_mode":     mode,
                "regret_at_anchors": aq,
            })
    return pd.DataFrame(rows)


def _task_means(df: pd.DataFrame) -> pd.DataFrame:
    """Average seeds within (scenario, method, task, budget, anchor_mode)."""
    return (
        df.groupby(["scenario", "method", "task", "budget", "anchor_mode"],
                   as_index=False)
          .agg(normalized_regret=("normalized_regret", "mean"),
               regret_at_anchors=("regret_at_anchors", "mean"))
    )


# ── Table T2 ───────────────────────────────────────────────────────────────

def make_t2(tm: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """T2: mean ± std regret @ {10,25,50}, all methods × scenarios.

    Also computes Wilcoxon pairwise for key pairs with Holm correction.
    Returns (regret_table, wilcoxon_table).
    """
    # use informed anchor mode from main experiment
    df = tm[(tm.anchor_mode == "informed") & (tm.budget.isin(KEY_BUDGETS))].copy()

    # mean ± std across tasks (task is the obs unit; already seed-averaged)
    t2 = (
        df.groupby(["scenario", "method", "budget"], as_index=False)
          .agg(mean_regret=("normalized_regret", "mean"),
               std_regret=("normalized_regret", "std"),
               n_tasks=("task", "nunique"))
    )

    print("\n=== T2: Mean regret @ budget 25, all methods × scenarios ===")
    pivot = t2[t2.budget == BUDGET_PRIMARY].pivot_table(
        index="method", columns="scenario", values="mean_regret")
    pivot["AVG"] = pivot.mean(axis=1)
    pivot = pivot.sort_values("AVG")
    with pd.option_context("display.float_format", "{:.4f}".format,
                           "display.max_columns", 20, "display.width", 120):
        print(pivot.to_string())

    # ── Wilcoxon pairwise + Holm correction ────────────────────────
    KEY_PAIRS = [
        ("bo_lrtc_feature", "bo_cold"),
        ("rgpe",            "bo_cold"),
        ("bo_lrtc_feature", "rgpe"),
        ("active_pmf",      "bo_cold"),
    ]
    wrows = []
    for scen in SCEN_ORDER:
        for budget in KEY_BUDGETS:
            sub = df[(df.scenario == scen) & (df.budget == budget)]
            task_pivot = sub.pivot_table(
                index="task", columns="method", values="normalized_regret")
            for (a, b) in KEY_PAIRS:
                if a not in task_pivot.columns or b not in task_pivot.columns:
                    continue
                diff = (task_pivot[a] - task_pivot[b]).dropna().values
                if len(diff) < 5 or not np.any(diff != 0):
                    wrows.append(dict(scenario=scen, budget=budget,
                                      pair=f"{a}_vs_{b}", n=len(diff),
                                      p_raw=float("nan"), p_holm=float("nan"),
                                      sig="—"))
                    continue
                _, p = wilcoxon(diff, alternative="two-sided", zero_method="wilcox")
                wrows.append(dict(scenario=scen, budget=budget,
                                  pair=f"{a}_vs_{b}", n=len(diff), p_raw=p,
                                  p_holm=float("nan"), sig=""))

    wdf = pd.DataFrame(wrows)
    # Holm correction within each (scenario, budget) group separately
    # — corrects only across the key pairs for that scenario/budget
    for (scen, bud), grp_idx in wdf.groupby(["scenario", "budget"]).groups.items():
        sub = wdf.loc[grp_idx]
        valid_mask = sub["p_raw"].notna()
        if valid_mask.sum() == 0:
            continue
        valid_idx = sub.index[valid_mask]
        p_arr = sub.loc[valid_idx, "p_raw"].values
        order = np.argsort(p_arr)
        n_total = len(p_arr)
        p_holm = p_arr.copy()
        for rank_i, idx in enumerate(order):
            p_holm[idx] = min(1.0, p_arr[idx] * (n_total - rank_i))
        for i in range(len(order) - 1, 0, -1):
            p_holm[order[i - 1]] = min(p_holm[order[i - 1]], p_holm[order[i]])
        wdf.loc[valid_idx, "p_holm"] = p_holm
        wdf.loc[valid_idx, "sig"] = [
            "**" if p < 0.01 else ("*" if p < 0.05 else "ns") for p in p_holm
        ]

    print("\n=== Wilcoxon pairwise (Holm-corrected) @ budget 25 ===")
    b25 = wdf[wdf.budget == BUDGET_PRIMARY].copy()
    print(b25[["scenario", "pair", "n", "p_raw", "p_holm", "sig"]].to_string(index=False))

    return t2, wdf


# ── Table T4 ───────────────────────────────────────────────────────────────

def make_t4(ar_tm: pd.DataFrame) -> pd.DataFrame:
    """T4: per-scenario winner under informed vs random anchors @ budget 25."""
    df = ar_tm[ar_tm.budget == BUDGET_PRIMARY].copy()
    rows = []
    for scen in SCEN_ORDER:
        for mode in ("informed", "random"):
            sub = df[(df.scenario == scen) & (df.anchor_mode == mode)]
            if sub.empty:
                continue
            means = sub.groupby("method")["normalized_regret"].mean()
            winner = means.idxmin()
            rows.append(dict(scenario=scen, anchor_mode=mode,
                             winner=winner,
                             winner_regret=means[winner],
                             bo_cold_regret=means.get("bo_cold", float("nan")),
                             rgpe_regret=means.get("rgpe", float("nan")),
                             bo_lrtc_feature_regret=means.get("bo_lrtc_feature",
                                                               float("nan"))))
    t4 = pd.DataFrame(rows)
    print("\n=== T4: Per-scenario winner under informed vs random anchors @ budget 25 ===")
    print(t4.to_string(index=False))
    return t4


# ── Figure F2 ───────────────────────────────────────────────────────────────

def make_f2(tm: pd.DataFrame, out: Path) -> None:
    """F2: 5-panel regret curves (all methods, budgets 5-50, 95% CI shading)."""
    df = tm[(tm.anchor_mode == "informed") & (tm.budget.isin(BUDGETS))].copy()

    n_scen = len(SCEN_ORDER)
    fig, axes = plt.subplots(1, n_scen, figsize=(4.0 * n_scen, 3.6),
                             sharey=False)
    if n_scen == 1:
        axes = [axes]

    highlight = {"rgpe", "bo_lrtc_feature", "bo_cold"}

    for ax, scen in zip(axes, SCEN_ORDER):
        sub = df[df.scenario == scen]
        # order methods for legend readability
        ordered = [m for m in METHOD_ORDER if m in sub.method.unique()]
        for meth in ordered:
            ms = sub[sub.method == meth]
            grp = ms.groupby("budget")["normalized_regret"]
            means = grp.mean()
            sems  = grp.std() / np.sqrt(grp.count().clip(lower=1))
            budgets = means.index.values
            y = means.values
            lo, hi = y - 1.96 * sems.values, y + 1.96 * sems.values
            color  = METHOD_COLOR[meth]
            lw     = 1.8 if meth in highlight else 0.9
            alpha  = 0.85 if meth in highlight else 0.5
            zorder = 3 if meth in highlight else 1
            label  = METHOD_LABEL.get(meth, meth)
            ax.plot(budgets, y, color=color, lw=lw, alpha=alpha,
                    zorder=zorder, label=label)
            ax.fill_between(budgets, lo, hi, color=color, alpha=0.10, zorder=0)

        ax.set_title(SCEN_LABEL[scen], fontsize=8, pad=3)
        ax.set_xlabel("Budget (evaluations)", fontsize=7)
        if ax == axes[0]:
            ax.set_ylabel("Normalized regret", fontsize=7)
        ax.tick_params(labelsize=7)
        ax.xaxis.set_major_locator(mticker.FixedLocator(BUDGETS))
        ax.grid(True, lw=0.4, alpha=0.5)
        ax.set_ylim(bottom=0)

    # shared legend below the figure
    handles = [plt.Line2D([0], [0], color=METHOD_COLOR[m], lw=1.8 if m in highlight else 0.9,
                           label=METHOD_LABEL.get(m, m))
               for m in METHOD_ORDER if m in df.method.unique()]
    fig.legend(handles=handles, loc="lower center",
               ncol=7, fontsize=6.5, frameon=True,
               bbox_to_anchor=(0.5, -0.18))
    plt.suptitle("Normalized regret vs budget (informed anchors, k₀=3)",
                 fontsize=9, y=1.01)
    plt.tight_layout(pad=0.8)
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.savefig(Path(str(out).replace(".pdf", ".svg")), format="svg", bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ── Figure F3 ───────────────────────────────────────────────────────────────

def make_f3(ar_tm: pd.DataFrame, main_tm: pd.DataFrame, out: Path) -> None:
    """F3: warm-cold Δregret bar chart (informed vs random) + AQ scatter.

    Panel A: grouped bar chart — Δregret = bo_cold_regret - best_warmstart_regret
             for each scenario × anchor_mode.
    Panel B: scatter — mean AQ vs Δregret under each mode (shows AQ predicts benefit).
    """
    df = ar_tm[ar_tm.budget == BUDGET_PRIMARY].copy()

    rows = []
    for scen in SCEN_ORDER:
        for mode in ("informed", "random"):
            sub = df[(df.scenario == scen) & (df.anchor_mode == mode)]
            if sub.empty:
                continue
            means = sub.groupby("method")["normalized_regret"].mean()
            cold = means.get("bo_cold", float("nan"))
            warm_methods = [m for m in WARM_METHODS if m in means.index]
            if not warm_methods:
                continue
            best_warm = means[warm_methods].min()
            delta = cold - best_warm
            aq_mean = sub[sub.method == "bo_cold"]["regret_at_anchors"].mean()
            best_warm_method = means[warm_methods].idxmin()
            rows.append(dict(scenario=scen, anchor_mode=mode,
                             delta=delta, aq=aq_mean,
                             cold=cold, best_warm=best_warm,
                             best_warm_method=best_warm_method))

    delta_df = pd.DataFrame(rows)

    print("\n=== F3 data: warm-cold Δregret ===")
    print(delta_df[["scenario", "anchor_mode", "delta", "aq",
                     "best_warm_method"]].to_string(index=False))

    # compute per-scenario AQ from main experiment
    aq_main = (main_tm[(main_tm.anchor_mode == "informed") &
                        (main_tm.budget == BUDGET_PRIMARY)]
               .groupby("scenario")["regret_at_anchors"].mean())

    fig, (ax_bar, ax_scat) = plt.subplots(1, 2, figsize=(10, 3.6))

    # ── Panel A: grouped bar chart ─────────────────────────────────────
    scen_labels = [SCEN_LABEL[s] for s in SCEN_ORDER]
    x = np.arange(len(SCEN_ORDER))
    w = 0.35
    mode_colors = {"informed": "#0072B2", "random": "#D55E00"}

    for offset, mode in zip([-w / 2, w / 2], ["informed", "random"]):
        deltas = []
        for scen in SCEN_ORDER:
            row = delta_df[(delta_df.scenario == scen) &
                           (delta_df.anchor_mode == mode)]
            deltas.append(float(row["delta"].values[0]) if not row.empty else 0.0)
        bars = ax_bar.bar(x + offset, deltas, width=w,
                          color=mode_colors[mode], label=mode.capitalize(),
                          alpha=0.85, edgecolor="white", linewidth=0.5)
        for bar, d in zip(bars, deltas):
            ax_bar.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.001,
                        f"{d:.3f}", ha="center", va="bottom", fontsize=6.5)

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(scen_labels, rotation=20, ha="right", fontsize=7.5)
    ax_bar.set_ylabel("Δregret  (bo_cold − best_warm)  @budget 25", fontsize=7.5)
    ax_bar.set_title("(A) Warm-start benefit under each anchor mode", fontsize=8)
    ax_bar.legend(fontsize=8)
    ax_bar.axhline(0, color="#444444", lw=0.7, ls="--")
    ax_bar.tick_params(labelsize=7)
    ax_bar.grid(axis="y", lw=0.4, alpha=0.5)

    # ── Panel B: scatter AQ vs Δregret ─────────────────────────────────
    marker_map = {"informed": "o", "random": "s"}
    for mode in ["informed", "random"]:
        sub = delta_df[delta_df.anchor_mode == mode]
        ax_scat.scatter(sub["aq"], sub["delta"],
                        color=mode_colors[mode], marker=marker_map[mode],
                        s=60, label=mode.capitalize(), zorder=3, alpha=0.9)
        for _, row in sub.iterrows():
            ax_scat.annotate(SCEN_LABEL[row["scenario"]],
                             (row["aq"], row["delta"]),
                             textcoords="offset points", xytext=(4, 3),
                             fontsize=6, color=mode_colors[mode])

    ax_scat.set_xlabel("Mean AQ (regret_at_anchors before BO)", fontsize=7.5)
    ax_scat.set_ylabel("Δregret  (bo_cold − best_warm)", fontsize=7.5)
    ax_scat.set_title("(B) AQ predicts warm-start benefit", fontsize=8)
    ax_scat.legend(fontsize=8)
    ax_scat.axhline(0, color="#444444", lw=0.7, ls="--")
    ax_scat.tick_params(labelsize=7)
    ax_scat.grid(lw=0.4, alpha=0.5)

    plt.suptitle("Figure F3 — Warm-start benefit vs anchor mode", fontsize=9, y=1.02)
    plt.tight_layout(pad=0.8)
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.savefig(Path(str(out).replace(".pdf", ".svg")), format="svg", bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ── Headline summary ────────────────────────────────────────────────────────

def print_headline_summary(tm: pd.DataFrame) -> None:
    """Print the headline numbers using scenario-equal (Method C) aggregation.

    Method C: for each (scenario, method), average task-level means, then
    average the 5 scenario means equally.  This is the manuscript-preferred
    aggregation; it gives lcbench 1/5 weight regardless of task count.
    See docs/phase1_audit_report.md for full derivation.
    """
    df = tm[(tm.anchor_mode == "informed") & (tm.budget == BUDGET_PRIMARY)].copy()
    # Scenario-equal: scenario means first, then average across 5 scenarios
    scen_means = (df.groupby(["scenario", "method"])["normalized_regret"]
                    .mean())
    means = (scen_means.groupby("method").mean().sort_values())
    # Also compute task-equal for comparison
    task_means = (df.groupby(["method"])["normalized_regret"].mean().sort_values())
    print(f"\n=== Global rankings @ budget {BUDGET_PRIMARY} (scenario-equal, Method C) ===")
    for rank, (m, v) in enumerate(means.items(), 1):
        v_b = task_means.get(m, float("nan"))
        print(f"  {rank:>2}. {m:<26} {v:.4f}  (task-equal: {v_b:.4f})")

    print(f"\n=== Per-scenario winner @ budget {BUDGET_PRIMARY} ===")
    for scen in SCEN_ORDER:
        sub = df[df.scenario == scen]
        means_s = sub.groupby("method")["normalized_regret"].mean()
        winner  = means_s.idxmin()
        print(f"  {SCEN_LABEL[scen]:<15}: {winner} ({means_s[winner]:.4f})  "
              f"| bo_cold={means_s.get('bo_cold', float('nan')):.4f}  "
              f"| rgpe={means_s.get('rgpe', float('nan')):.4f}  "
              f"| bo_lrtc_feature={means_s.get('bo_lrtc_feature', float('nan')):.4f}")

    print(f"\n=== Warm-start vs cold BO @ budget {BUDGET_PRIMARY} ===")
    for scen in SCEN_ORDER:
        sub = df[df.scenario == scen]
        means_s = sub.groupby("method")["normalized_regret"].mean()
        cold = means_s.get("bo_cold", float("nan"))
        warm_m = [m for m in WARM_METHODS if m in means_s.index]
        best_w = means_s[warm_m].min()
        best_wm = means_s[warm_m].idxmin()
        gain = (cold - best_w) / cold * 100
        print(f"  {SCEN_LABEL[scen]:<15}: cold={cold:.4f}  best_warm={best_w:.4f} ({best_wm})  "
              f"gain={gain:+.1f}%")


# ── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load main results ─────────────────────────────────────────────
    print("Loading main results …", flush=True)
    main_df = _load_dir(MAIN_DIR)
    print(f"  {len(main_df):,} rows (from {main_df.groupby(['scenario','method','task','seed']).ngroups:,} units)")

    # ── Load anchor_robustness results ────────────────────────────────
    print("Loading anchor_robustness results …", flush=True)
    ar_df = _load_dir(AR_DIR)
    print(f"  {len(ar_df):,} rows (from {ar_df.groupby(['scenario','method','task','seed','anchor_mode']).ngroups:,} units)")

    # ── Task-level means ──────────────────────────────────────────────
    print("Computing task-level means …")
    main_tm = _task_means(main_df)
    ar_tm   = _task_means(ar_df)

    # ── Headline numbers (verification) ──────────────────────────────
    print_headline_summary(main_tm)

    # ── T2 ───────────────────────────────────────────────────────────
    print("\nBuilding T2 …")
    t2, wilcoxon_df = make_t2(main_tm)
    t2.to_csv(TAB_DIR / "T2_regret_table.csv", index=False)
    wilcoxon_df.to_csv(TAB_DIR / "T2_wilcoxon.csv", index=False)
    print(f"  Saved: {TAB_DIR / 'T2_regret_table.csv'}")
    print(f"  Saved: {TAB_DIR / 'T2_wilcoxon.csv'}")

    # ── T4 ───────────────────────────────────────────────────────────
    print("\nBuilding T4 …")
    t4 = make_t4(ar_tm)
    t4.to_csv(TAB_DIR / "T4_anchor_winners.csv", index=False)
    print(f"  Saved: {TAB_DIR / 'T4_anchor_winners.csv'}")

    # ── F2 ───────────────────────────────────────────────────────────
    print("\nBuilding F2 …")
    make_f2(main_tm, FIG_DIR / "F2_regret_curves.pdf")

    # ── F3 ───────────────────────────────────────────────────────────
    print("\nBuilding F3 …")
    make_f3(ar_tm, main_tm, FIG_DIR / "F3_anchor_delta.pdf")

    print("\n=== Done. All outputs in figures/ml_springer/ and tables/ml_springer/ ===")


if __name__ == "__main__":
    main()
