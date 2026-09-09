"""Adaptive candidate restriction for scalable random-forest Bayesian optimisation.

Reference implementation accompanying Khan, Alam, Ayyasamy and Zhang,
*Adaptive Candidate Restriction for Scalable Random-Forest Bayesian
Optimisation: Matched Accuracy at a Fraction of the Acquisition Cost*.

The headline result is a one-line change to an RF-SMBO loop::

    from candidate_restriction import adaptive_k_size

    K_t = adaptive_k_size(n_unevaluated, dispersion, reference_dispersion)

or, for the full optimiser::

    from candidate_restriction import RFSMBO

    result = RFSMBO(mode="adaptive").optimize(X, y, budget=25, seed=0)
"""

from .constants import (
    C_GROW,
    FIXED_K,
    FULL_FLOOR,
    K_MIN,
    RF_N_ESTIMATORS,
    RF_UCB_BETA,
    WARMUP,
)
from .optimizer import OptimizationResult, RFSMBO
from .schedule import SCORE_ALL, adaptive_k_size, fixed_k_size

__version__ = "1.0.0"

__all__ = [
    "RFSMBO", "OptimizationResult",
    "adaptive_k_size", "fixed_k_size", "SCORE_ALL",
    "FULL_FLOOR", "K_MIN", "C_GROW", "FIXED_K",
    "RF_N_ESTIMATORS", "RF_UCB_BETA", "WARMUP",
    "__version__",
]
