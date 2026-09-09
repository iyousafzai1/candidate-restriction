"""The adaptive-K candidate-restriction rule.

This module is the paper's contribution in isolation: a pure, dependency-free
function that decides *how many* candidates to score at one step of a
sequential model-based optimisation loop.

The schedule is derived from the optimiser's-curse bound

    B_t(K) = Delta * exp(-p * K) + 2 * sigma_t * sqrt(2 * ln K)

whose minimiser grows as the surrogate becomes reliable (``sigma_t`` shrinks).
Rather than track that minimiser analytically, the deployable rule uses the
surrogate's own predictive dispersion as an online proxy for ``sigma_t``:

    K_t = clip(K_MIN * (s0 / s_hat) ** c,  low=K_MIN,  high=U_t)

where ``s_hat`` is the mean per-tree prediction standard deviation over a
random probe of the unevaluated pool and ``s0`` is that quantity at the first
restricted step of the run.  Below a pool-size floor ``F`` the rule returns
``None``, meaning "score everything" -- so on small grids the method is exactly
equivalent to full-candidate RF-SMBO.
"""

from __future__ import annotations

import math

from .constants import C_GROW, FULL_FLOOR, K_MIN

__all__ = ["adaptive_k_size", "SCORE_ALL"]

#: Returned by :func:`adaptive_k_size` when the whole pool should be scored.
SCORE_ALL = None


def adaptive_k_size(
    n_unevaluated: int,
    dispersion: float,
    reference_dispersion: float,
    *,
    k_min: int = K_MIN,
    c_grow: float = C_GROW,
    full_floor: int = FULL_FLOOR,
) -> int | None:
    """Return how many candidates to score at this step.

    Parameters
    ----------
    n_unevaluated:
        ``U_t``, the number of configurations not yet evaluated.
    dispersion:
        ``s_hat``, the surrogate's current mean predictive standard deviation
        over a random probe of the unevaluated pool.
    reference_dispersion:
        ``s0``, the same quantity measured at the first restricted step of the
        run.  Held fixed for the remainder of the run.
    k_min, c_grow, full_floor:
        The frozen constants; override only to reproduce the selection sweep.

    Returns
    -------
    int or None
        The candidate-set size ``K_t``, or :data:`SCORE_ALL` (``None``) when the
        pool is at or below ``full_floor`` and must be scored exhaustively.

    Notes
    -----
    ``K_t`` is floored at ``k_min`` and capped at ``n_unevaluated``, so the rule
    never scores more candidates than exist and never collapses to a handful
    while the surrogate is still unreliable.

    Examples
    --------
    Small pool -- score everything, exactly like full RF-SMBO:

    >>> adaptive_k_size(2000, dispersion=0.5, reference_dispersion=0.5) is SCORE_ALL
    True

    Large pool, surrogate still as uncertain as at the start -- the floor holds:

    >>> adaptive_k_size(40000, dispersion=0.5, reference_dispersion=0.5)
    1000

    The surrogate has become four times more confident -- score more:

    >>> adaptive_k_size(40000, dispersion=0.125, reference_dispersion=0.5)
    2828

    Never more candidates than remain in the pool:

    >>> adaptive_k_size(1500, dispersion=0.001, reference_dispersion=0.5,
    ...                 full_floor=1000)
    1500
    """
    if n_unevaluated <= 0:
        raise ValueError("n_unevaluated must be positive")
    if dispersion <= 0 or reference_dispersion <= 0:
        raise ValueError("dispersions must be positive")

    if n_unevaluated <= full_floor:
        return SCORE_ALL

    grown = k_min * (reference_dispersion / dispersion) ** c_grow
    return int(min(n_unevaluated, max(k_min, round(grown))))


def fixed_k_size(n_unevaluated: int, k: int, *, full_floor: int = FULL_FLOOR) -> int | None:
    """Candidate-set size for the frozen fixed-K variant (``rf-cand-2k``).

    Identical small-pool behaviour to :func:`adaptive_k_size`; above the floor it
    scores a constant ``k`` candidates regardless of surrogate state.
    """
    if n_unevaluated <= full_floor:
        return SCORE_ALL
    return int(min(n_unevaluated, k))
