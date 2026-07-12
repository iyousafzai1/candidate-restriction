"""One-shot LRTC: Deng & Xiao (TPAMI 2023) SNN model via factorized BCD.

Faithful reimplementation of their Algorithm 1 (Eqs. 3, 5-8) with:
- mean-fill initialization of missing test entries (their Eq. 22),
- per-mode factorization X_(i) = W_i H_i with ridge beta_i,
- observed entries re-imposed after every mode update (Eq. 8),
- final recovery X = sum_i alpha_i * fold(X_(i)) with uniform alpha.

Phase 4 performance notes (active loop):
- snn_bcd_opt() accepts pre-allocated T_buf and mask_buf to avoid large
  per-refit allocations, and uses an in-place fold-accumulation that
  reduces peak temporary memory from O(d * tensor_size) to O(tensor_size).
- np.copyto(dst, T, where=mask) replaces dst[mask] = T[mask] to avoid
  the intermediate gather array.
- The task mode (last axis, with small dim = n_train+1) uses a contiguous
  reshape view instead of a full moveaxis copy.
"""
from __future__ import annotations

import numpy as np


def unfold(t: np.ndarray, mode: int) -> np.ndarray:
    return np.moveaxis(t, mode, 0).reshape(t.shape[mode], -1)


def fold(mat: np.ndarray, mode: int, shape: tuple) -> np.ndarray:
    rest = [s for i, s in enumerate(shape) if i != mode]
    return np.moveaxis(mat.reshape([shape[mode]] + rest), 0, mode)


def snn_bcd(T: np.ndarray, mask: np.ndarray, rank: int, beta: float = 1.0,
            iters: int = 30, seed: int = 0, tol: float = 1e-5,
            Ws_init: list | None = None, return_state: bool = False):
    """Complete tensor T (NaN/garbage allowed where mask=False).
    `rank` is capped per mode at the unfolding's row dimension.
    Ws_init warm-starts the per-mode factor matrices (active-loop refits);
    with return_state=True returns (X, Ws).

    For the active loop, prefer snn_bcd_opt() which accepts pre-allocated
    buffers and uses optimized inner-loop memory management.
    """
    shape = T.shape
    d = T.ndim
    rng = np.random.default_rng(seed)

    # Mean-fill init of missing entries (Deng & Xiao Eq. 22).
    obs_mean = float(np.mean(T[mask])) if mask.any() else 0.0
    X = np.where(mask, T, obs_mean).astype(np.float64)

    alphas = np.full(d, 1.0 / d)
    ranks = [int(min(rank, shape[i])) for i in range(d)]
    if Ws_init is not None and all(
            Ws_init[i].shape == (shape[i], ranks[i]) for i in range(d)):
        Ws = [w.copy() for w in Ws_init]
    else:
        Ws = [0.1 * rng.standard_normal((shape[i], ranks[i])) for i in range(d)]

    prev = X.copy()
    for _ in range(iters):
        folds = []
        for i in range(d):
            Xi = unfold(X, i)                                  # n_i x prod
            Wi = Ws[i]
            ri = ranks[i]
            # Eq. 5: H = (W'W + beta I)^+ W' X
            Hi = np.linalg.solve(Wi.T @ Wi + beta * np.eye(ri), Wi.T @ Xi)
            # Eq. 6: W = X H' (H H' + beta I)^+
            Wi = np.linalg.solve(Hi @ Hi.T + beta * np.eye(ri), Hi @ Xi.T).T
            Ws[i] = Wi
            # Eq. 7 + Eq. 8: low-rank estimate, observed entries re-imposed
            Zi = fold(Wi @ Hi, i, shape)
            np.copyto(Zi, T, where=mask)                       # avoid gather temp
            folds.append(Zi)
        X = folds[0] * alphas[0]
        for a, Z in zip(alphas[1:], folds[1:]):
            X += a * Z
        np.copyto(X, T, where=mask)

        num = np.linalg.norm(X - prev)
        den = np.linalg.norm(prev) + 1e-12
        if num / den < tol:
            break
        prev = X.copy()
    if return_state:
        return X, Ws
    return X


def snn_bcd_opt(T_buf: np.ndarray, mask_buf: np.ndarray,
                rank: int, beta: float = 1.0,
                iters: int = 30, seed: int = 0, tol: float = 1e-5,
                Ws_init: list | None = None, return_state: bool = False):
    """Optimized snn_bcd for the active loop.

    Differences from snn_bcd (all correctness-preserving):
    1. Takes pre-allocated T_buf / mask_buf (caller updates last slice in-place;
       no numpy concatenation per refit).
    2. In-place fold accumulation into a single X_acc buffer (avoids building
       a list of d fold tensors then summing — cuts peak temp alloc from
       O(d * n) to O(n) per BCD iter).
    3. np.copyto(dst, T_buf, where=mask_buf) replaces dst[mask] = T[mask]
       everywhere, eliminating the O(n_observed) gather intermediates.
    4. Last-mode unfold uses X.reshape(-1, N+1).T (a C/F view, no copy)
       because the last axis is contiguous in C-order storage.
    5. Pre-allocated prev and X_acc buffers reused across BCD iterations.
    """
    shape = T_buf.shape
    d = T_buf.ndim
    rng = np.random.default_rng(seed)

    # Mean-fill init: unobserved entries <- mean of observed.
    obs_mean = float(np.mean(T_buf[mask_buf])) if mask_buf.any() else 0.0
    X = np.where(mask_buf, T_buf, obs_mean).astype(np.float64)

    alpha = 1.0 / d
    ranks = [int(min(rank, shape[i])) for i in range(d)]
    if Ws_init is not None and all(
            Ws_init[i].shape == (shape[i], ranks[i]) for i in range(d)):
        Ws = [w.copy() for w in Ws_init]
    else:
        Ws = [0.1 * rng.standard_normal((shape[i], ranks[i])) for i in range(d)]

    # Pre-allocate reusable buffers (avoid per-iter allocations).
    prev = X.copy()
    X_acc = np.empty_like(X)        # accumulator; swapped with X each iter
    diff_buf = np.empty_like(X)     # for convergence norm
    last_mode = d - 1
    N_plus_1 = shape[last_mode]     # n_train_member + 1
    n_cells = X.size // N_plus_1

    for _ in range(iters):
        X_acc.fill(0.0)
        for i in range(d):
            Wi = Ws[i]
            ri = ranks[i]
            # --- unfold X along mode i ---
            if i == last_mode:
                # Last axis is contiguous in C order; .T is an F-order view (no copy).
                Xi = X.reshape(n_cells, N_plus_1).T   # (N+1, n_cells)
            else:
                Xi = np.moveaxis(X, i, 0).reshape(shape[i], -1)  # copy needed
            # Eq. 5: H = (W'W + beta I)^{-1} W' X
            WtW = Wi.T @ Wi
            WtW.flat[::ri + 1] += beta        # beta*I in-place (no extra alloc)
            Hi = np.linalg.solve(WtW, Wi.T @ Xi)
            # Eq. 6: W = X H' (H H' + beta I)^{-1}
            HHt = Hi @ Hi.T
            HHt.flat[::ri + 1] += beta
            Wi_new = np.linalg.solve(HHt, Hi @ Xi.T).T
            Ws[i] = Wi_new
            # Eq. 7: fold the low-rank mode-i estimate
            Zi = fold(Wi_new @ Hi, i, shape)
            # Eq. 8: re-impose observed entries (np.copyto avoids gather temp)
            np.copyto(Zi, T_buf, where=mask_buf)
            # In-place accumulate: Zi *= alpha, then X_acc += Zi (no extra alloc)
            Zi *= alpha
            X_acc += Zi

        # Final re-imposition on the sum.
        np.copyto(X_acc, T_buf, where=mask_buf)

        # Convergence check using pre-allocated diff buffer.
        np.subtract(X_acc, prev, out=diff_buf)
        num = np.linalg.norm(diff_buf)
        den = np.linalg.norm(X_acc) + 1e-12
        # Swap X <-> prev so prev holds X_acc for next iter without copy.
        # X_acc becomes the next iteration's accumulator (will be filled to 0).
        prev, X_acc = X_acc, prev
        X = prev                               # X now points at the new estimate

        if num / den < tol:
            break

    if return_state:
        return X, Ws
    return X


def lrtc_complete_test_slice(train_z: np.ndarray, test_slice: np.ndarray,
                             test_mask: np.ndarray, rank: int, beta: float,
                             iters: int, seed: int) -> np.ndarray:
    """One-shot LRTC for Phase 2: stack z-scored train slices (fully observed)
    with the sparse test slice as the last frontal slice; complete; return
    the predicted test slice (observed entries kept exact)."""
    T = np.concatenate([train_z, test_slice[..., None]], axis=-1)
    mask = np.ones(T.shape, dtype=bool)
    mask[..., -1] = test_mask
    X = snn_bcd(T, mask, rank=rank, beta=beta, iters=iters, seed=seed)
    out = X[..., -1].copy()
    out[test_mask] = test_slice[test_mask]
    return out
