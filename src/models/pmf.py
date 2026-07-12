"""Masked matrix factorization (PMF-style) via alternating ridge regression.

M ≈ W H with W: rows(tasks) x r, H: r x cols(cells). Observed-entry loss +
L2 on factors. Efficient for the Phase 2/4 mask structure (train rows fully
observed, test row sparse) by batching rows/columns that share the same mask
pattern — but written generically, so any mask works. Mask-pattern groups are
computed ONCE per fit (the mask is constant within a fit), not per iteration.

Phase 4 additions: optional warm starts (W_init/H_init) and state return,
used by the active loop to make per-step refits cheap.

Phase 4 optimizations: optional precomputed_row_groups / precomputed_col_groups
bypass the np.unique call in _mask_groups (significant for large n_cells).
_Member in active.py computes these directly from the train/test structure.
"""
from __future__ import annotations

import numpy as np


def _mask_groups(mask: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Group row indices by identical mask pattern; returns [(rows, cols)]."""
    patterns, inverse = np.unique(mask, axis=0, return_inverse=True)
    groups = []
    for g in range(len(patterns)):
        rows = np.where(inverse == g)[0]
        cols = np.where(patterns[g])[0]
        groups.append((rows, cols))
    return groups


def _solve_rows(M: np.ndarray, groups, H: np.ndarray, reg: float) -> np.ndarray:
    """Ridge-solve W given H, batched per mask-pattern group."""
    n_rows, r = M.shape[0], H.shape[0]
    W = np.zeros((n_rows, r))
    for rows, cols in groups:
        if len(cols) == 0:
            continue
        Hp = H[:, cols]                              # r x c
        A = Hp @ Hp.T + reg * np.eye(r)
        W[rows] = np.linalg.solve(A, Hp @ M[np.ix_(rows, cols)].T).T
    return W


def masked_als(M: np.ndarray, mask: np.ndarray, rank: int, reg: float = 1.0,
               iters: int = 50, seed: int = 0, tol: float = 1e-6,
               W_init: np.ndarray | None = None,
               H_init: np.ndarray | None = None,
               return_state: bool = False,
               precomputed_row_groups=None,
               precomputed_col_groups=None):
    """Returns pred (or (pred, W, H) if return_state). mask True = observed.

    precomputed_row_groups / precomputed_col_groups: if provided, skip the
    _mask_groups computation (significant speedup for large n_cells).
    These must match the structure of mask / mask.T respectively.
    """
    M = np.where(mask, M, 0.0)
    rng = np.random.default_rng(seed)
    n_rows, n_cols = M.shape
    r = int(min(rank, n_rows, n_cols))

    H = H_init.copy() if H_init is not None and H_init.shape == (r, n_cols) \
        else 0.1 * rng.standard_normal((r, n_cols))
    W = W_init.copy() if W_init is not None and W_init.shape == (n_rows, r) \
        else 0.1 * rng.standard_normal((n_rows, r))

    row_groups = precomputed_row_groups if precomputed_row_groups is not None \
        else _mask_groups(mask)
    col_groups = precomputed_col_groups if precomputed_col_groups is not None \
        else _mask_groups(mask.T)
    Mt = M.T

    prev = None
    for _ in range(iters):
        W = _solve_rows(M, row_groups, H, reg)
        H = _solve_rows(Mt, col_groups, W.T, reg).T
        pred = W @ H
        if prev is not None:
            num = np.linalg.norm(pred - prev)
            den = np.linalg.norm(prev) + 1e-12
            if num / den < tol:
                break
        prev = pred
    pred = W @ H
    if return_state:
        return pred, W, H
    return pred, W, H  # backward-compatible triple


def pmf_complete_test_row(train_block: np.ndarray, test_row: np.ndarray,
                          test_mask: np.ndarray, rank: int, reg: float,
                          iters: int, seed: int,
                          W_init=None, H_init=None,
                          return_state: bool = False,
                          precomputed_row_groups=None,
                          precomputed_col_groups=None):
    """Joint completion of [train_block (fully observed); test_row (sparse)].
    Returns predicted full test row (and (W, H) state if return_state).

    precomputed_row_groups / precomputed_col_groups: bypass mask_groups if
    provided (see masked_als docstring).
    """
    M = np.vstack([train_block, test_row[None, :]])
    mask = np.ones_like(M, dtype=bool)
    mask[-1] = test_mask
    pred, W, H = masked_als(M, mask, rank=rank, reg=reg, iters=iters,
                            seed=seed, W_init=W_init, H_init=H_init,
                            precomputed_row_groups=precomputed_row_groups,
                            precomputed_col_groups=precomputed_col_groups)
    out = pred[-1].copy()
    out[test_mask] = test_row[test_mask]    # keep observed values exact
    if return_state:
        return out, W, H
    return out
