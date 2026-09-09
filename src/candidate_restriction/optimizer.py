"""Random-forest SMBO with optional candidate restriction.

``RFSMBO`` runs the loop studied in the paper over a finite, tabular search
space: fit a random forest to what has been evaluated, score candidates with
UCB, evaluate the maximiser.  The only thing that varies between the three
modes is *how many* candidates are scored:

``"full"``
    Score every unevaluated configuration (full-candidate RF-SMBO).
``"fixed"``
    Score a random subset of ``FIXED_K`` candidates (``rf-cand-2k``).
``"adaptive"``
    Size the subset online with :func:`~candidate_restriction.schedule.adaptive_k_size`.

The implementation mirrors ``experiments/adaptive_k_runner.py``, which produced
the published results.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .constants import (
    C_GROW,
    FIXED_K,
    FULL_FLOOR,
    K_MIN,
    RF_N_ESTIMATORS,
    RF_UCB_BETA,
    WARMUP,
)
from .schedule import SCORE_ALL, adaptive_k_size, fixed_k_size

__all__ = ["RFSMBO", "OptimizationResult"]


@dataclass
class OptimizationResult:
    """Outcome of one optimisation run."""

    #: Best value seen after each evaluation.
    trajectory: list[float]
    #: Indices of the configurations evaluated, in order.
    evaluated: list[int]
    #: Candidates scored at each surrogate step (empty during warm-up).
    candidates_scored: list[int] = field(default_factory=list)

    @property
    def best(self) -> float:
        """Best value found."""
        return self.trajectory[-1] if self.trajectory else float("nan")

    @property
    def mean_candidates_scored(self) -> float:
        """Mean candidates scored per surrogate step -- the paper's cost metric."""
        if not self.candidates_scored:
            return float("nan")
        return float(np.mean(self.candidates_scored))


class RFSMBO:
    """Random-forest SMBO over a tabular search space.

    Parameters
    ----------
    mode:
        ``"full"``, ``"fixed"`` or ``"adaptive"``.
    n_estimators, beta, warmup:
        Surrogate and acquisition settings; the defaults are the frozen values.
    k_min, c_grow, full_floor, fixed_k:
        Restriction constants; the defaults are the frozen values.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(300, 3))
    >>> y = -(X ** 2).sum(axis=1)
    >>> opt = RFSMBO(mode="adaptive", full_floor=50)
    >>> res = opt.optimize(X, y, budget=15, seed=0)
    >>> len(res.trajectory)
    15
    >>> bool(res.best <= y.max())
    True
    """

    def __init__(
        self,
        mode: str = "adaptive",
        *,
        n_estimators: int = RF_N_ESTIMATORS,
        beta: float = RF_UCB_BETA,
        warmup: int = WARMUP,
        k_min: int = K_MIN,
        c_grow: float = C_GROW,
        full_floor: int = FULL_FLOOR,
        fixed_k: int = FIXED_K,
    ) -> None:
        if mode not in ("full", "fixed", "adaptive"):
            raise ValueError(f"mode must be 'full', 'fixed' or 'adaptive', got {mode!r}")
        self.mode = mode
        self.n_estimators = n_estimators
        self.beta = beta
        self.warmup = warmup
        self.k_min = k_min
        self.c_grow = c_grow
        self.full_floor = full_floor
        self.fixed_k = fixed_k

    def optimize(
        self,
        X: np.ndarray,
        y: np.ndarray,
        budget: int,
        seed: int = 0,
    ) -> OptimizationResult:
        """Maximise ``y`` over the rows of ``X`` within ``budget`` evaluations."""
        from sklearn.ensemble import RandomForestRegressor

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        if len(X) != len(y):
            raise ValueError("X and y must have the same length")

        rng = np.random.default_rng(seed)
        n_configs = len(X)
        observed_idx: list[int] = []
        observed_vals: list[float] = []
        best = -np.inf
        traj: list[float] = []
        scored: list[int] = []
        s0: float | None = None

        for step in range(budget):
            if step < self.warmup or len(observed_idx) == 0:
                idx = int(rng.integers(n_configs))
                while idx in observed_idx:
                    idx = int(rng.integers(n_configs))
            else:
                rf = RandomForestRegressor(
                    n_estimators=self.n_estimators, max_depth=None,
                    min_samples_leaf=1, random_state=seed + step, n_jobs=1,
                )
                rf.fit(X[observed_idx], np.asarray(observed_vals))

                obs = set(observed_idx)
                uneval = [i for i in range(n_configs) if i not in obs]
                U = len(uneval)
                if U == 0:
                    break

                k = self._candidate_budget(U, rf, X, uneval, rng, s0)
                if isinstance(k, tuple):          # adaptive path returns its probe
                    k, probe, probe_preds, s_hat = k
                    if s0 is None:
                        s0 = s_hat
                else:
                    probe = probe_preds = None

                idx, n_scored = self._select(rf, X, uneval, k, probe, probe_preds, rng)
                scored.append(n_scored)

            observed_idx.append(idx)
            observed_vals.append(float(y[idx]))
            best = max(best, float(y[idx]))
            traj.append(best)

        return OptimizationResult(trajectory=traj, evaluated=observed_idx,
                                  candidates_scored=scored)

    # -- internals -------------------------------------------------------

    def _candidate_budget(self, U, rf, X, uneval, rng, s0):
        if self.mode == "full":
            return SCORE_ALL
        if self.mode == "fixed":
            return fixed_k_size(U, self.fixed_k, full_floor=self.full_floor)
        if U <= self.full_floor:
            return SCORE_ALL
        probe = rng.choice(uneval, min(self.k_min, U), replace=False)
        preds = np.array([t.predict(X[probe]) for t in rf.estimators_])
        s_hat = float(preds.std(0).mean()) + 1e-12
        ref = s_hat if s0 is None else s0
        k = adaptive_k_size(U, s_hat, ref, k_min=self.k_min,
                            c_grow=self.c_grow, full_floor=self.full_floor)
        return k, probe, preds, s_hat

    def _select(self, rf, X, uneval, k, probe, probe_preds, rng):
        if k is SCORE_ALL:
            cand = np.asarray(uneval)
            preds = np.array([t.predict(X[cand]) for t in rf.estimators_])
        elif probe is not None and k <= len(probe):
            cand, preds = probe, probe_preds
        elif probe is not None:
            remaining = list(set(uneval) - set(probe.tolist()))
            extra = rng.choice(remaining, min(k - len(probe), len(remaining)), replace=False)
            cand = np.concatenate([probe, extra])
            preds = np.array([t.predict(X[cand]) for t in rf.estimators_])
        else:
            cand = rng.choice(uneval, min(k, len(uneval)), replace=False)
            preds = np.array([t.predict(X[cand]) for t in rf.estimators_])

        score = preds.mean(0) + self.beta * preds.std(0)
        if float(np.max(score) - np.min(score)) < 1e-6:
            pick = int(rng.integers(len(cand)))
        else:
            pick = int(np.argmax(score))
        return int(cand[pick]), len(cand)
