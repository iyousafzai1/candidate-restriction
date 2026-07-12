"""Aggregation and statistics for Phase 2 results (MVE plan §9).

Pairing unit for tests = task (seeds averaged within task first).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def records_to_frame(records: list[dict]) -> pd.DataFrame:
    """Flatten run-unit JSONs (with 'scenario'/'task' attached) to long format:
    one row per (scenario, method, task, seed, budget)."""
    rows = []
    passthrough = [
        "anchor_mode",
        "k0",
        "regret_at_anchors",
        "runtime_seconds",
        "_config_hash",
    ]
    for r in records:
        for cp in r["checkpoints"]:
            row = {
                "scenario": r["scenario"], "method": r["method"],
                "task": r["task"], "seed": r["seed"],
                "budget": cp["budget"],
                "normalized_regret": cp["normalized_regret"],
                "best_found": cp["best_found"],
            }
            if "n_evaluated" in cp:
                row["n_evaluated"] = cp["n_evaluated"]
            for key in passthrough:
                if key in r:
                    row[key] = r[key]
            rows.append(row)
    return pd.DataFrame(rows)


def per_task_means(df: pd.DataFrame) -> pd.DataFrame:
    """Average over seeds within (scenario, method, task, budget)."""
    return (df.groupby(["scenario", "method", "task", "budget"], as_index=False)
              .agg(normalized_regret=("normalized_regret", "mean"),
                   best_found=("best_found", "mean")))


def summary_table(task_means: pd.DataFrame) -> pd.DataFrame:
    """Mean/median regret per (scenario, method, budget)."""
    return (task_means.groupby(["scenario", "method", "budget"], as_index=False)
            .agg(mean_regret=("normalized_regret", "mean"),
                 median_regret=("normalized_regret", "median"),
                 n_tasks=("task", "nunique")))


def average_ranks(task_means: pd.DataFrame) -> pd.DataFrame:
    """Rank methods per (scenario, task, budget) by regret (1 = best),
    then average over tasks."""
    df = task_means.copy()
    df["rank"] = (df.groupby(["scenario", "task", "budget"])["normalized_regret"]
                    .rank(method="average"))
    return (df.groupby(["scenario", "method", "budget"], as_index=False)
              .agg(avg_rank=("rank", "mean")))


def _wilcoxon_normal_approx(diff: np.ndarray, alternative: str) -> float:
    """Signed-rank test p-value via normal approximation (fallback when
    scipy is unavailable; adequate for n >= 10). Zeros dropped (wilcox)."""
    d = diff[diff != 0]
    n = len(d)
    if n == 0:
        return float("nan")
    ranks = np.argsort(np.argsort(np.abs(d))) + 1.0
    # midranks for ties
    order = np.abs(d)
    uniq = np.unique(order)
    for u in uniq:
        m = order == u
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    w_pos = float(ranks[d > 0].sum())
    mu = n * (n + 1) / 4.0
    sigma = np.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    if sigma == 0:
        return float("nan")
    z = (w_pos - mu) / sigma
    from math import erf, sqrt
    cdf = 0.5 * (1 + erf(z / sqrt(2)))
    if alternative == "less":       # H1: diffs tend negative
        return cdf
    if alternative == "greater":
        return 1 - cdf
    return 2 * min(cdf, 1 - cdf)


def paired_wilcoxon(task_means: pd.DataFrame, method_a: str, method_b: str,
                    alternative: str = "less") -> pd.DataFrame:
    """H1 (default): method_a regret < method_b regret. Paired by task,
    per (scenario, budget). Also reports the median paired difference
    (effect size; negative favors method_a)."""
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        wilcoxon = None

    out = []
    for (scen, budget), g in task_means.groupby(["scenario", "budget"]):
        a = (g[g.method == method_a].set_index("task")["normalized_regret"])
        b = (g[g.method == method_b].set_index("task")["normalized_regret"])
        common = a.index.intersection(b.index)
        diff = (a.loc[common] - b.loc[common]).values
        row = {"scenario": scen, "budget": budget,
               "comparison": f"{method_a} < {method_b}",
               "n_tasks": len(common),
               "median_paired_diff": float(np.median(diff)) if len(diff) else np.nan}
        if len(diff) >= 5 and np.any(diff != 0):
            if wilcoxon is not None:
                _, p = wilcoxon(diff, alternative=alternative,
                                zero_method="wilcox")
            else:
                p = _wilcoxon_normal_approx(diff, alternative)
            row.update(p_value=float(p), significant_005=bool(p < 0.05))
        else:
            row.update(p_value=np.nan, significant_005=False)
        out.append(row)
    return pd.DataFrame(out)
