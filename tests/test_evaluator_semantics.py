from __future__ import annotations

import numpy as np
import pytest

from config.settings import EvaluationConfig
from metrics import evaluator
from metrics.evaluator import (
    EpisodeStats,
    _aggregate_episode_stats,
    _wilson_ci,
    evaluate_across_seeds,
)


def _make_episode(
    *,
    crashed: bool,
    progress_m: float,
    mean_speed_mps: float,
    duration_s: float = 4.0,
    ttcs: list[float] | None = None,
) -> EpisodeStats:
    n_steps = int(duration_s)
    speeds = [float(mean_speed_mps)] * max(n_steps, 1)
    return EpisodeStats(
        speeds=speeds,
        ttcs=(ttcs or []),
        distance_travelled=float(mean_speed_mps) * float(len(speeds)),
        longitudinal_progress=float(progress_m),
        duration_s=float(duration_s),
        collision=bool(crashed),
        n_steps=len(speeds),
    )


def _cfg() -> EvaluationConfig:
    return EvaluationConfig(
        min_success_progress_m=300.0,
        min_success_mean_speed_mps=5.0,
        ttc_thresholds_s=(1.0, 2.0, 4.0),
        bootstrap_samples=200,
        bootstrap_seed=0,
    )


def test_survival_without_success():
    ep = _make_episode(crashed=False, progress_m=100.0, mean_speed_mps=10.0)
    r = _aggregate_episode_stats([ep], dt=1.0, cfg=_cfg())
    assert r.survival_rate == pytest.approx(1.0)
    assert r.success_rate == pytest.approx(0.0)


def test_slow_surviving_policy_is_not_successful():
    ep = _make_episode(crashed=False, progress_m=400.0, mean_speed_mps=2.0)
    r = _aggregate_episode_stats([ep], dt=1.0, cfg=_cfg())
    assert r.survival_rate == pytest.approx(1.0)
    assert r.success_rate == pytest.approx(0.0)


def test_true_success_requires_progress_and_speed_and_no_collision():
    ep = _make_episode(crashed=False, progress_m=400.0, mean_speed_mps=10.0)
    r = _aggregate_episode_stats([ep], dt=1.0, cfg=_cfg())
    assert r.survival_rate == pytest.approx(1.0)
    assert r.success_rate == pytest.approx(1.0)


def test_collision_prevents_success_even_with_progress_and_speed():
    ep = _make_episode(crashed=True, progress_m=400.0, mean_speed_mps=10.0)
    r = _aggregate_episode_stats([ep], dt=1.0, cfg=_cfg())
    assert r.collision_rate == pytest.approx(1.0)
    assert r.survival_rate == pytest.approx(0.0)
    assert r.success_rate == pytest.approx(0.0)


def test_goal_completion_is_deprecated_alias_for_success_rate():
    ep = _make_episode(crashed=False, progress_m=400.0, mean_speed_mps=10.0)
    r = _aggregate_episode_stats([ep], dt=1.0, cfg=_cfg())
    assert r.goal_completion == pytest.approx(r.success_rate)


def test_ttc_exposure_uses_valid_ttc_denominator():
    ep = _make_episode(
        crashed=False,
        progress_m=350.0,
        mean_speed_mps=10.0,
        ttcs=[0.8, 1.5, 3.0, 5.0],
    )
    r = _aggregate_episode_stats([ep], dt=1.0, cfg=_cfg())
    assert r.ttc_valid_samples == 4
    assert r.ttc_exposure[1.0] == pytest.approx(1 / 4)
    assert r.ttc_exposure[2.0] == pytest.approx(2 / 4)
    assert r.ttc_exposure[4.0] == pytest.approx(3 / 4)
    assert r.ttc_below_1s_frac == pytest.approx(1 / 4)
    assert r.ttc_below_2s_frac == pytest.approx(2 / 4)
    assert r.ttc_below_4s_frac == pytest.approx(3 / 4)


def test_no_finite_ttc_returns_missing_severity_metrics():
    ep = _make_episode(
        crashed=False,
        progress_m=350.0,
        mean_speed_mps=10.0,
        ttcs=[np.inf, np.inf, np.nan],
    )
    r = _aggregate_episode_stats([ep], dt=1.0, cfg=_cfg())
    assert r.ttc_valid_samples == 0
    assert r.minimum_finite_ttc_s is None
    assert r.finite_ttc_p05_s is None
    assert r.mean_min_ttc is None


def test_finite_ttc_p05_is_deterministic():
    ep = _make_episode(
        crashed=False,
        progress_m=350.0,
        mean_speed_mps=10.0,
        ttcs=[1.0, 2.0, 3.0, 4.0, 5.0],
    )
    r = _aggregate_episode_stats([ep], dt=1.0, cfg=_cfg())
    assert r.finite_ttc_p05_s == pytest.approx(1.2)


def test_wilson_zero_events_interval_is_valid():
    low, high = _wilson_ci(0, 20)
    assert 0.0 <= low <= high <= 1.0
    assert high > 0.0


def test_wilson_nonzero_events_interval_contains_observed_rate():
    low, high = _wilson_ci(3, 10)
    assert 0.0 <= low <= high <= 1.0
    assert low <= 0.3 <= high


def test_aggregate_seed_weighting_uses_counts_not_mean_of_rates(monkeypatch):
    # First seed: 2 episodes, 1 collision. Second seed: 8 episodes, 0 collisions.
    # Correct global collision rate is 1/10 = 0.1, not mean(0.5, 0.0) = 0.25.
    call_idx = {"i": 0}

    def _fake_collect_episode_stats(policy, env, n_episodes, seed, fault_horizon_steps):
        if call_idx["i"] == 0:
            call_idx["i"] += 1
            return [
                _make_episode(crashed=True, progress_m=400.0, mean_speed_mps=10.0),
                _make_episode(crashed=False, progress_m=400.0, mean_speed_mps=10.0),
            ], 1.0
        call_idx["i"] += 1
        return [
            _make_episode(crashed=False, progress_m=400.0, mean_speed_mps=10.0)
            for _ in range(8)
        ], 1.0

    monkeypatch.setattr(evaluator, "_collect_episode_stats", _fake_collect_episode_stats)

    class _DummyPolicy:
        def act(self, obs):
            return 1

    class _DummyUnwrapped:
        config = {"policy_frequency": 1}

    class _DummyEnv:
        unwrapped = _DummyUnwrapped()

    r = evaluate_across_seeds(
        policy=_DummyPolicy(),
        env=_DummyEnv(),
        n_seeds=2,
        episodes_per_seed=1,
        base_seed=0,
        config=_cfg(),
    )
    assert r.n_episodes == 10
    assert r.n_collisions == 1
    assert r.collision_rate == pytest.approx(0.1)
