"""Phase 2 data preparation: normalization, cold-start anchors, problem setup.

Conventions (MVE plan §4/§8):
- Historical (train) slices are z-scored using their own FULL grid statistics.
- The test slice is z-scored using statistics of its OBSERVED entries only
  (>= k0 anchors); falls back to std=1 if degenerate.
- Anchors: top-k0 grid cells by average z-scored performance across train
  tasks. Deterministic (ties broken by flat index). Same anchors for every
  model-based method on a given scenario/split.
- All metrics are computed on RAW (unnormalized) values.
"""
from __future__ import annotations

import numpy as np


def zscore_slice(sl: np.ndarray):
    mu = float(np.nanmean(sl))
    sd = float(np.nanstd(sl))
    sd = sd if sd > 1e-12 else 1.0
    return (sl - mu) / sd, mu, sd


def zscore_train(tensor: np.ndarray) -> np.ndarray:
    """Z-score each task slice (last mode) with its full-grid stats."""
    out = np.empty_like(tensor, dtype=np.float64)
    for k in range(tensor.shape[-1]):
        out[..., k], _, _ = zscore_slice(tensor[..., k])
    return out


def zscore_observed(values: np.ndarray):
    """Stats from observed test entries only. Returns (mu, sd)."""
    mu = float(np.mean(values))
    sd = float(np.std(values))
    return mu, (sd if sd > 1e-8 else 1.0)


def select_anchors(train_z: np.ndarray, k0: int,
                   mode: str = "informed", seed: int = 0) -> np.ndarray:
    """Choose k0 flat cell indices to seed every method on this task.

    mode="informed" (default): top-k0 cells by mean z-scored train
        performance. This is the strong-default protocol — but note the
        globally-best-on-average config is itself a very strong baseline,
        so it can leave little headroom and make "cold" BO not truly cold.
    mode="random": k0 cells drawn uniformly at random (seeded). Use this
        as the truly-cold protocol to expose how much of any method's
        performance comes from the informed anchors vs. the method itself.
    """
    n_cells = int(np.prod(train_z.shape[:-1]))
    if mode == "random":
        rng = np.random.default_rng(seed)
        return np.sort(rng.choice(n_cells, size=min(k0, n_cells),
                                  replace=False))
    if mode != "informed":
        raise ValueError(f"unknown anchor mode '{mode}'")
    mean_perf = train_z.mean(axis=-1).ravel()
    order = np.argsort(-mean_perf, kind="stable")
    return order[:k0].copy()


class Problem:
    """One (scenario grid, test task) instance seen by all methods.

    Attributes:
      train_z      : z-scored train tensor (n1..nd, N_train)
      gt_slice     : RAW ground-truth slice of the test task (n1..nd)
      anchors      : flat indices of cold-start cells
      grid_shape   : (n1..nd)
      n_cells      : number of grid cells
    """

    def __init__(self, tensor: np.ndarray, task_ids: list[str],
                 train_task_ids: list[str], test_task: str, k0: int,
                 anchor_mode: str = "informed", anchor_seed: int = 0):
        if test_task not in task_ids:
            raise ValueError(f"test task '{test_task}' not in grid task_ids")
        idx = {t: i for i, t in enumerate(task_ids)}
        train_idx = [idx[t] for t in train_task_ids if t in idx]
        if idx[test_task] in train_idx:
            raise ValueError(f"'{test_task}' is a TRAIN task — refusing to leak")

        self.grid_shape = tensor.shape[:-1]
        self.n_cells = int(np.prod(self.grid_shape))
        self.train_raw = tensor[..., train_idx]
        self.train_z = zscore_train(self.train_raw)
        self.gt_slice = tensor[..., idx[test_task]].astype(np.float64)
        self.anchor_mode = anchor_mode
        self.anchors = select_anchors(self.train_z, k0, mode=anchor_mode,
                                      seed=anchor_seed)

        gt_flat = self.gt_slice.ravel()
        self.y_best = float(np.max(gt_flat))
        self.y_worst = float(np.min(gt_flat))
        if self.y_best - self.y_worst <= 1e-12:
            raise ValueError("constant test slice reached Problem(); "
                             "build-time filtering should have removed it")

    def evaluate(self, flat_idx) -> np.ndarray:
        """'Run' configurations on the test task = look up ground truth."""
        return self.gt_slice.ravel()[np.asarray(flat_idx, dtype=int)]

    def normalized_regret(self, best_found: float) -> float:
        return (self.y_best - best_found) / (self.y_best - self.y_worst)
