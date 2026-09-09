"""Tests for the adaptive-K rule itself."""

import numpy as np
import pytest

from candidate_restriction import (
    C_GROW, FIXED_K, FULL_FLOOR, K_MIN, RF_N_ESTIMATORS, RF_UCB_BETA,
    SCORE_ALL, adaptive_k_size, fixed_k_size,
)


class TestFrozenConstants:
    """The published constants must not drift."""

    def test_values_match_the_paper(self):
        assert (FULL_FLOOR, K_MIN, C_GROW) == (2500, 1000, 0.75)
        assert FIXED_K == 2000
        assert (RF_N_ESTIMATORS, RF_UCB_BETA) == (64, 1.96)


class TestSmallPoolIsExact:
    """At or below the floor the method must be a no-op vs full scoring."""

    @pytest.mark.parametrize("u", [1, 100, 2499, 2500])
    def test_returns_score_all(self, u):
        assert adaptive_k_size(u, 0.3, 0.3) is SCORE_ALL

    def test_just_above_floor_restricts(self):
        assert adaptive_k_size(2501, 0.3, 0.3) is not SCORE_ALL

    def test_fixed_variant_matches_floor_behaviour(self):
        assert fixed_k_size(2500, FIXED_K) is SCORE_ALL
        assert fixed_k_size(50000, FIXED_K) == FIXED_K


class TestScheduleShape:
    def test_floor_holds_when_surrogate_has_not_improved(self):
        assert adaptive_k_size(40000, 0.5, 0.5) == K_MIN

    def test_grows_as_dispersion_falls(self):
        sizes = [adaptive_k_size(40000, s, 0.5) for s in (0.5, 0.25, 0.125, 0.05)]
        assert sizes == sorted(sizes)
        assert sizes[0] < sizes[-1]

    def test_matches_closed_form(self):
        k = adaptive_k_size(40000, 0.125, 0.5)
        assert k == int(round(K_MIN * 4.0 ** C_GROW))

    def test_never_below_k_min(self):
        # dispersion worse than the reference would shrink K below the floor
        assert adaptive_k_size(40000, 5.0, 0.5) == K_MIN

    def test_never_exceeds_pool(self):
        assert adaptive_k_size(3000, 1e-6, 0.5) == 3000

    def test_capped_result_is_an_int(self):
        assert isinstance(adaptive_k_size(40000, 0.1, 0.5), int)


class TestValidation:
    @pytest.mark.parametrize("u", [0, -1])
    def test_rejects_empty_pool(self, u):
        with pytest.raises(ValueError):
            adaptive_k_size(u, 0.3, 0.3)

    @pytest.mark.parametrize("kw", [{"dispersion": 0.0}, {"reference_dispersion": -1.0}])
    def test_rejects_nonpositive_dispersion(self, kw):
        args = {"dispersion": 0.3, "reference_dispersion": 0.3, **kw}
        with pytest.raises(ValueError):
            adaptive_k_size(40000, **args)
