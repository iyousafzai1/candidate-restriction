"""Frozen constants of the adaptive-K schedule.

These were selected once on a single HPO-B development space (``6767``) from a
grid of five ``(FULL_FLOOR, K_MIN, C_GROW)`` triples and never retuned on any
evaluation task.  They are reported in the paper's experimental-setup section.

Changing them changes the published method; they are deliberately module-level
constants rather than tunable defaults.
"""

#: F -- pools of at most this many unevaluated configurations are scored in
#: full, making the method an exact no-op on small grids.
FULL_FLOOR = 2500

#: Early candidate floor, used while the surrogate is least reliable.
K_MIN = 1000

#: c -- growth exponent of the schedule ``K_t = K_MIN * (s0 / s_hat) ** c``.
C_GROW = 0.75

#: Candidate-set size of the frozen fixed-K variant (``rf-cand-2k``).
FIXED_K = 2000

#: Trees in the random-forest surrogate.
RF_N_ESTIMATORS = 64

#: Exploration weight of the UCB acquisition.
RF_UCB_BETA = 1.96

#: Random evaluations before the surrogate is first fitted.
WARMUP = 5

__all__ = [
    "FULL_FLOOR", "K_MIN", "C_GROW", "FIXED_K",
    "RF_N_ESTIMATORS", "RF_UCB_BETA", "WARMUP",
]
