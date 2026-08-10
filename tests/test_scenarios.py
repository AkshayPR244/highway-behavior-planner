"""
Regression tests for adversarial scenarios.

Each test asserts that the IDM expert meets a minimum safety bar on a
named scenario.  These serve as:

  1. Sanity checks: if IDM fails a test, the scenario setup is wrong.
  2. Regression gates: a trained policy must at least match these thresholds.

The thresholds are intentionally conservative — the IDM expert is the
floor, not the ceiling.  Learned policies should exceed these.
"""
import pytest
import numpy as np

from envs.highway_wrapper import make_env
from policies.idm_expert import IDMExpert
from scenarios.adversarial import run_scenario, SCENARIOS


@pytest.fixture(scope="module")
def idm_scenario_results():
    """Run all scenarios once and cache results for the whole module."""
    env = make_env(seed=0)
    policy = IDMExpert(env)
    results = {
        s.name: run_scenario(s, policy, env, n_episodes=10, seed=0)
        for s in SCENARIOS
    }
    env.close()
    return results


class TestSuddenBrake:
    """
    NPC 20 m ahead at 10 m/s (TTC ≈ 1.3 s).
    IDM should escape more than half the time via braking or lane change.
    """

    def test_survival_rate_above_threshold(self, idm_scenario_results):
        r = idm_scenario_results["sudden_brake"]
        assert r.survival_rate >= 0.4, (
            f"IDM should survive ≥40% of sudden_brake episodes, got {r.survival_rate:.2f}"
        )

    def test_ego_fault_rate_bounded(self, idm_scenario_results):
        """IDM should not cause more than half its crashes by its own action."""
        r = idm_scenario_results["sudden_brake"]
        assert r.ego_fault_rate <= 0.5, (
            f"Ego fault rate too high in sudden_brake: {r.ego_fault_rate:.2f}"
        )

    def test_lc_frequency_elevated(self, idm_scenario_results):
        """Under braking pressure IDM should attempt more lane changes than baseline (0.041)."""
        r = idm_scenario_results["sudden_brake"]
        assert r.lc_frequency > 0.05, (
            f"Expected elevated LC frequency in sudden_brake, got {r.lc_frequency:.3f}"
        )


class TestCloseMerge:
    """
    NPC in adjacent lane at ego's position (unsafe lane change target).
    IDM+MOBIL safety criterion should prevent the collision in most episodes.
    """

    def test_survival_rate_high(self, idm_scenario_results):
        r = idm_scenario_results["close_merge"]
        assert r.survival_rate >= 0.6, (
            f"IDM should handle close_merge in ≥60% of episodes, got {r.survival_rate:.2f}"
        )

    def test_ego_fault_zero_or_low(self, idm_scenario_results):
        """When crashes occur they should be NPC-fault (unavoidable), not ego-fault."""
        r = idm_scenario_results["close_merge"]
        assert r.ego_fault_rate <= 0.1, (
            f"Ego should not cause close_merge crashes; ego_fault_rate={r.ego_fault_rate:.2f}"
        )


class TestAggressiveRear:
    """
    NPC 5 m behind at ego+10 m/s (rear TTC = 0.5 s).
    At 1 Hz this is physically unavoidable — all crashes should be NPC fault.
    """

    def test_all_crashes_are_npc_fault(self, idm_scenario_results):
        """
        The rear NPC is too close and too fast for the policy to escape
        in a single 1 Hz step.  classify_fault(IDLE) should always
        detect the crash as unavoidable → NPC fault.
        """
        r = idm_scenario_results["aggressive_rear"]
        # If there are collisions, they must all be NPC fault
        if r.collision_rate > 0:
            total_crashes = round(r.collision_rate * r.n_episodes)
            npc_crashes   = round(r.npc_fault_rate * r.n_episodes)
            assert npc_crashes == total_crashes, (
                f"Expected all aggressive_rear crashes to be NPC fault; "
                f"got ego_fault_rate={r.ego_fault_rate:.2f}"
            )

    def test_ego_fault_rate_is_zero(self, idm_scenario_results):
        r = idm_scenario_results["aggressive_rear"]
        assert r.ego_fault_rate == pytest.approx(0.0), (
            f"IDM ego should not be at fault for unavoidable rear collision; "
            f"got {r.ego_fault_rate:.2f}"
        )


class TestDenseCorridor:
    """
    3 NPCs at 15/30/45 m ahead at 20 m/s (TTC 3/6/9 s).
    IDM should escape in most episodes by changing lanes early.
    """

    def test_survival_rate_above_threshold(self, idm_scenario_results):
        r = idm_scenario_results["dense_corridor"]
        assert r.survival_rate >= 0.6, (
            f"IDM should navigate dense_corridor in ≥60% of episodes, "
            f"got {r.survival_rate:.2f}"
        )

    def test_lc_frequency_elevated(self, idm_scenario_results):
        """Dense traffic should trigger more lane changes than open highway baseline."""
        r = idm_scenario_results["dense_corridor"]
        assert r.lc_frequency > 0.04, (
            f"Expected elevated LC frequency in dense_corridor, got {r.lc_frequency:.3f}"
        )

    def test_min_ttc_meaningful(self, idm_scenario_results):
        """Convoy scenario should produce finite (non-inf) mean min TTC."""
        r = idm_scenario_results["dense_corridor"]
        assert np.isfinite(r.mean_min_ttc), (
            "mean_min_ttc should be finite in dense_corridor scenario"
        )
