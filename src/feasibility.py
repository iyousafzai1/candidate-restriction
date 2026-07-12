"""Gate 1 — tensorization feasibility (MVE plan §3).

Two entry points:
- check_grid(): for grids we built ourselves from surrogates (the MVE path).
- cartesian_report(): for tabular benchmark data (HPO-B-style), reporting
  whether observed configurations form a Cartesian product. Unused in the
  MVE main path but required by the plan for later tabular benchmarks.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def cartesian_report(df: pd.DataFrame, hp_cols: list[str]) -> dict:
    """How grid-like is a table of observed configurations?"""
    distinct = {c: int(df[c].nunique()) for c in hp_cols}
    full_size = int(np.prod(list(distinct.values()), dtype=np.float64))
    observed = int(df[hp_cols].drop_duplicates().shape[0])
    ratio = observed / full_size if full_size > 0 else 0.0
    return {
        "hp_cols": hp_cols,
        "distinct_values_per_hp": distinct,
        "full_cartesian_size": full_size,
        "observed_distinct_configs": observed,
        "cartesian_ratio": ratio,
        "is_grid_like": bool(ratio > 0.95),
        "note": ("looks like a Cartesian grid" if ratio > 0.95 else
                 "NOT a Cartesian grid; tensorization requires binning"),
    }


def check_grid(grid: dict, thresholds: dict) -> dict:
    """Gate 1 checks on a built grid (output of tensorize.load_grid)."""
    tensor, meta = grid["tensor"], grid["meta"]
    checks = []

    def add(name, value, threshold, passed, severity="fail"):
        checks.append({"check": name, "value": value, "threshold": threshold,
                       "passed": bool(passed), "severity": severity})

    n_tasks = tensor.shape[-1]
    add("n_tasks >= min_tasks", n_tasks, thresholds["min_tasks"],
        n_tasks >= thresholds["min_tasks"])

    n_cells = int(np.prod(tensor.shape[:-1]))
    add("n_cells >= min_cells", n_cells, thresholds["min_cells"],
        n_cells >= thresholds["min_cells"])

    density = float(np.isfinite(tensor).mean())
    add("density >= min_density", round(density, 6), thresholds["min_density"],
        density >= thresholds["min_density"])

    # Degenerate-surrogate check: every task must show some variation.
    stds = np.array([np.nanstd(tensor[..., k]) for k in range(n_tasks)])
    n_flat = int((stds < 1e-9).sum())
    add("n_constant_task_slices == 0", n_flat, 0, n_flat == 0)

    # Plausibility warning only (accuracy-like metrics should sit in [0, 1]).
    vmin, vmax = float(np.nanmin(tensor)), float(np.nanmax(tensor))
    in_unit = (-0.01 <= vmin) and (vmax <= 1.01)
    add("values within [0,1] (warn only)", [round(vmin, 4), round(vmax, 4)],
        "[0,1]", in_unit, severity="warn")

    hard = [c for c in checks if c["severity"] == "fail"]
    overall = all(c["passed"] for c in hard)
    return {
        "scenario_key": meta["scenario_key"],
        "shape": meta["shape"],
        "metric": meta["metric"],
        "checks": checks,
        "gate1_pass": bool(overall),
        "warnings": [c["check"] for c in checks
                     if c["severity"] == "warn" and not c["passed"]],
    }
