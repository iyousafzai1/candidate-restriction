"""Tests for the RF-SMBO loop and the three candidate-set modes."""

import numpy as np
import pytest

from candidate_restriction import RFSMBO


@pytest.fixture
def problem():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(4000, 4))
    y = -(X ** 2).sum(axis=1)          # maximised at the origin
    return X, y


class TestModes:
    @pytest.mark.parametrize("mode", ["full", "fixed", "adaptive"])
    def test_runs_and_respects_budget(self, problem, mode):
        X, y = problem
        res = RFSMBO(mode=mode, full_floor=500, fixed_k=200, k_min=100).optimize(
            X, y, budget=12, seed=0)
        assert len(res.trajectory) == 12
        assert len(res.evaluated) == 12
        assert len(set(res.evaluated)) == 12          # no configuration twice

    def test_rejects_unknown_mode(self):
        with pytest.raises(ValueError):
            RFSMBO(mode="greedy")

    def test_trajectory_is_monotone(self, problem):
        X, y = problem
        res = RFSMBO(mode="adaptive", full_floor=500, k_min=100).optimize(
            X, y, budget=12, seed=1)
        assert res.trajectory == sorted(res.trajectory)
        assert res.best == max(res.trajectory)


class TestRestrictionSavesWork:
    def test_restricted_scores_fewer_candidates_than_full(self, problem):
        X, y = problem
        common = dict(full_floor=500, k_min=100)
        full = RFSMBO(mode="full", **common).optimize(X, y, budget=12, seed=0)
        adapt = RFSMBO(mode="adaptive", **common).optimize(X, y, budget=12, seed=0)
        assert adapt.mean_candidates_scored < full.mean_candidates_scored

    def test_below_floor_adaptive_scores_everything(self, problem):
        X, y = problem
        # floor above the pool size => identical workload to full scoring
        common = dict(full_floor=10_000, k_min=100)
        full = RFSMBO(mode="full", **common).optimize(X, y, budget=10, seed=3)
        adapt = RFSMBO(mode="adaptive", **common).optimize(X, y, budget=10, seed=3)
        assert adapt.evaluated == full.evaluated
        assert adapt.candidates_scored == full.candidates_scored


class TestDeterminism:
    def test_same_seed_same_run(self, problem):
        X, y = problem
        a = RFSMBO(mode="adaptive", full_floor=500, k_min=100).optimize(X, y, 10, seed=7)
        b = RFSMBO(mode="adaptive", full_floor=500, k_min=100).optimize(X, y, 10, seed=7)
        assert a.evaluated == b.evaluated
        assert a.trajectory == b.trajectory

    def test_different_seeds_differ(self, problem):
        X, y = problem
        a = RFSMBO(mode="adaptive", full_floor=500, k_min=100).optimize(X, y, 10, seed=1)
        b = RFSMBO(mode="adaptive", full_floor=500, k_min=100).optimize(X, y, 10, seed=2)
        assert a.evaluated != b.evaluated


class TestValidation:
    def test_rejects_mismatched_shapes(self):
        with pytest.raises(ValueError):
            RFSMBO().optimize(np.zeros((10, 2)), np.zeros(9), budget=3)
