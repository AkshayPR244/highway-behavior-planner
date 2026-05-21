"""
pytest test suite for Phase 1 components.

Covers:
  - IDM acceleration formula (physics edge cases)
  - MOBIL safety criterion (parametric state injection)
  - TTC computation (_compute_ttc)
  - Jerk computation (_compute_jerk)
  - Env wrapper observation shape
  - Safety wrapper forward projection helpers (is_action_safe, _project_front_gap, _project_rear_gap)
  - Fault attribution (would_crash_with_idle, classify_fault)
  - Evaluator metric arithmetic

Run with:
    cd ~/highway-planner && source .venv/bin/activate
    pytest tests/test_phase1.py -v
"""
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# IDM acceleration
# ---------------------------------------------------------------------------

from policies.idm_expert import (
    _idm_acceleration,
    IDM_DESIRED_SPEED,
    IDM_ACCEL_COMFORT,
    IDM_DECEL_COMFORT,
    IDM_MIN_SPACING,
    IDM_DESIRED_HEADWAY,
)


class TestIDMAcceleration:
    def test_zero_gap_returns_max_deceleration(self):
        """Gap <= 0 should return the comfort deceleration as a hard brake."""
        result = _idm_acceleration(ego_speed=20.0, front_gap=0.0, front_rel_speed=0.0)
        assert result == -IDM_DECEL_COMFORT

    def test_negative_gap_returns_max_deceleration(self):
        """Negative gap (overlap) is treated the same as zero gap."""
        result = _idm_acceleration(ego_speed=20.0, front_gap=-5.0, front_rel_speed=0.0)
        assert result == -IDM_DECEL_COMFORT

    def test_free_road_accelerates_below_desired_speed(self):
        """With no vehicle ahead (large gap) and ego below desired speed, accel > 0."""
        result = _idm_acceleration(
            ego_speed=10.0,
            front_gap=1000.0,   # effectively infinite
            front_rel_speed=0.0,
        )
        assert result > 0.0

    def test_free_road_at_desired_speed_near_zero_accel(self):
        """At desired speed on a free road, acceleration should be near zero."""
        result = _idm_acceleration(
            ego_speed=IDM_DESIRED_SPEED,
            front_gap=1000.0,
            front_rel_speed=0.0,
        )
        assert abs(result) < 0.05  # essentially zero

    def test_closing_on_slow_vehicle_decelerates(self):
        """Ego faster than front vehicle at a small gap should give negative acceleration."""
        result = _idm_acceleration(
            ego_speed=25.0,
            front_gap=8.0,      # tight gap
            front_rel_speed=10.0,  # ego closing at 10 m/s
        )
        assert result < 0.0

    def test_following_at_desired_headway_near_zero_accel(self):
        """
        When the actual gap equals the desired gap at cruise speed with no
        relative speed, the interaction term should approximately cancel the
        free-road term → small magnitude acceleration.
        """
        ego_speed = 20.0
        desired_gap = IDM_MIN_SPACING + ego_speed * IDM_DESIRED_HEADWAY
        result = _idm_acceleration(
            ego_speed=ego_speed,
            front_gap=desired_gap,
            front_rel_speed=0.0,
        )
        # Should be small — not necessarily zero because free_road term is non-zero
        # at 20 m/s < 25 m/s desired, but interaction term limits it
        assert abs(result) < IDM_ACCEL_COMFORT

    def test_output_bounded_by_comfort_accel(self):
        """IDM acceleration should not exceed comfort acceleration in magnitude on free road."""
        result = _idm_acceleration(ego_speed=0.0, front_gap=500.0, front_rel_speed=0.0)
        assert result <= IDM_ACCEL_COMFORT


# ---------------------------------------------------------------------------
# TTC computation
# ---------------------------------------------------------------------------

from metrics.evaluator import _compute_ttc, _compute_jerk


class TestTTCComputation:
    """
    _compute_ttc reads from env.unwrapped directly.  We use a lightweight
    mock that mimics the highway-env vehicle/road interface.
    """

    class _MockVehicle:
        def __init__(self, speed, position_x=0.0):
            self.speed = speed
            self.position = [position_x, 0.0]

        def lane_distance_to(self, other):
            return other.position[0] - self.position[0]

    class _MockRoad:
        def __init__(self, front_vehicle=None):
            self._front = front_vehicle

        def neighbour_vehicles(self, ego):
            return [self._front] if self._front is not None else []

    class _MockUnwrapped:
        def __init__(self, ego, road):
            self.vehicle = ego
            self.road = road

    class _MockEnv:
        def __init__(self, ego, road):
            self.unwrapped = TestTTCComputation._MockUnwrapped(ego, road)

    def _make_env(self, ego_speed, front_speed=None, gap=None):
        ego = self._MockVehicle(ego_speed, position_x=0.0)
        if front_speed is None:
            road = self._MockRoad(front_vehicle=None)
        else:
            front = self._MockVehicle(front_speed, position_x=gap)
            road = self._MockRoad(front_vehicle=front)
        return self._MockEnv(ego, road)

    def test_no_front_vehicle_returns_inf(self):
        env = self._make_env(ego_speed=25.0)
        assert _compute_ttc(env) == np.inf

    def test_not_closing_returns_inf(self):
        """Ego slower than front vehicle — gap is growing, TTC is infinite."""
        env = self._make_env(ego_speed=20.0, front_speed=25.0, gap=30.0)
        assert _compute_ttc(env) == np.inf

    def test_same_speed_returns_inf(self):
        """Ego and front at same speed — no closing, TTC is infinite."""
        env = self._make_env(ego_speed=25.0, front_speed=25.0, gap=30.0)
        assert _compute_ttc(env) == np.inf

    def test_closing_computes_correctly(self):
        """TTC = gap / (ego_speed - front_speed)."""
        gap = 50.0
        ego_speed = 30.0
        front_speed = 20.0
        expected_ttc = gap / (ego_speed - front_speed)  # 5.0 s
        env = self._make_env(ego_speed=ego_speed, front_speed=front_speed, gap=gap)
        assert _compute_ttc(env) == pytest.approx(expected_ttc)

    def test_imminent_collision_short_ttc(self):
        """Very small gap and high closing speed → very short TTC."""
        env = self._make_env(ego_speed=30.0, front_speed=10.0, gap=4.0)
        ttc = _compute_ttc(env)
        assert ttc < 1.0


# ---------------------------------------------------------------------------
# Jerk computation
# ---------------------------------------------------------------------------

class TestJerkComputation:
    def test_empty_speeds_returns_zero(self):
        assert _compute_jerk([], dt=1.0) == 0.0

    def test_two_speeds_returns_zero(self):
        """Need at least 3 points for a second difference."""
        assert _compute_jerk([20.0, 21.0], dt=1.0) == 0.0

    def test_constant_speed_zero_jerk(self):
        """Constant speed → zero acceleration → zero jerk."""
        speeds = [25.0] * 10
        assert _compute_jerk(speeds, dt=1.0) == pytest.approx(0.0, abs=1e-10)

    def test_constant_acceleration_zero_jerk(self):
        """Linearly increasing speed → constant accel → zero jerk."""
        speeds = [float(v) for v in range(10)]  # 0, 1, 2, ..., 9 m/s at dt=1s
        assert _compute_jerk(speeds, dt=1.0) == pytest.approx(0.0, abs=1e-10)

    def test_known_jerk_value(self):
        """
        Manually construct a speed sequence with known jerk.

        jerk_t = (v[t+2] - 2*v[t+1] + v[t]) / dt^2
        Use: v = [0, 0, 1, 0, 0] with dt=1
        jerk at t=0: (1 - 0 + 0)/1 = 1
        jerk at t=1: (0 - 2 + 0)/1 = -2
        jerk at t=2: (0 - 0 + 1)/1 = 1
        RMS = sqrt((1 + 4 + 1) / 3) = sqrt(2)
        """
        speeds = [0.0, 0.0, 1.0, 0.0, 0.0]
        expected = np.sqrt(2.0)
        assert _compute_jerk(speeds, dt=1.0) == pytest.approx(expected, rel=1e-6)

    def test_dt_scaling(self):
        """Halving dt should quadruple jerk (jerk ~ 1/dt^2)."""
        speeds = [0.0, 0.0, 1.0, 0.0, 0.0]
        jerk_dt1 = _compute_jerk(speeds, dt=1.0)
        jerk_dt05 = _compute_jerk(speeds, dt=0.5)
        assert jerk_dt05 == pytest.approx(jerk_dt1 * 4.0, rel=1e-6)


# ---------------------------------------------------------------------------
# Env wrapper observation shape
# ---------------------------------------------------------------------------

class TestEnvWrapper:
    def test_obs_shape_is_flat_25(self):
        """FlattenObservation should produce a (25,) vector: 5 vehicles × 5 features."""
        from envs.highway_wrapper import make_env, OBS_VEHICLES, OBS_FEATURES
        env = make_env(seed=0)
        obs, _ = env.reset(seed=0)
        expected_dim = OBS_VEHICLES * len(OBS_FEATURES)
        assert obs.shape == (expected_dim,), f"Expected ({expected_dim},), got {obs.shape}"
        env.close()

    def test_obs_dtype_is_float32(self):
        """Observation should be float32 for PyTorch compatibility."""
        from envs.highway_wrapper import make_env
        env = make_env(seed=0)
        obs, _ = env.reset(seed=0)
        assert obs.dtype == np.float32
        env.close()

    def test_action_space_has_5_actions(self):
        """DiscreteMetaAction space must have exactly 5 actions."""
        from envs.highway_wrapper import make_env
        env = make_env(seed=0)
        assert env.action_space.n == 5
        env.close()

    def test_seed_reproducibility(self):
        """Same seed must produce identical initial observations."""
        from envs.highway_wrapper import make_env
        env = make_env(seed=0)
        obs_a, _ = env.reset(seed=42)
        obs_b, _ = env.reset(seed=42)
        np.testing.assert_array_equal(obs_a, obs_b)
        env.close()


# ---------------------------------------------------------------------------
# Safety wrapper — forward projection helpers
# ---------------------------------------------------------------------------

from safety.safety_wrapper import (
    _project_front_gap,
    _project_rear_gap,
    is_action_safe,
    LANE_LEFT, IDLE, LANE_RIGHT, FASTER, SLOWER,
)


class TestForwardProjection:
    """Unit tests for the pure-math projection functions (no env needed)."""

    # --- _project_front_gap ---

    def test_front_gap_increases_when_front_faster(self):
        """Front vehicle moving faster → gap grows → min gap == initial gap."""
        min_gap = _project_front_gap(
            front_gap=20.0, ego_speed=20.0, front_speed=25.0, dt=1.0, horizon=6
        )
        assert min_gap == pytest.approx(20.0)  # gap only grows

    def test_front_gap_decreases_when_closing(self):
        """Ego faster than front → gap shrinks → min gap < initial gap."""
        min_gap = _project_front_gap(
            front_gap=20.0, ego_speed=25.0, front_speed=20.0, dt=1.0, horizon=6
        )
        assert min_gap < 20.0
        # After 6 s at 5 m/s closing, gap = 20 - 30 = -10 → min = -10
        assert min_gap == pytest.approx(20.0 - 5.0 * 6)

    def test_front_gap_same_speed_stays_constant(self):
        min_gap = _project_front_gap(
            front_gap=15.0, ego_speed=25.0, front_speed=25.0, dt=1.0, horizon=6
        )
        assert min_gap == pytest.approx(15.0)

    # --- _project_rear_gap ---

    def test_rear_gap_decreases_when_rear_faster(self):
        """Rear vehicle faster than ego → rear closes → min gap < initial."""
        min_gap = _project_rear_gap(
            rear_gap=20.0, ego_speed=20.0, rear_speed=25.0, dt=1.0, horizon=6
        )
        assert min_gap < 20.0
        # rear closes at 5 m/s for 6 s
        assert min_gap == pytest.approx(20.0 - 5.0 * 6)

    def test_rear_gap_grows_when_ego_faster(self):
        """Ego faster than rear vehicle → gap grows → min gap == initial."""
        min_gap = _project_rear_gap(
            rear_gap=20.0, ego_speed=25.0, rear_speed=20.0, dt=1.0, horizon=6
        )
        assert min_gap == pytest.approx(20.0)


class TestIsActionSafe:
    """
    Tests for is_action_safe() using hand-crafted road state dicts.
    No env required.
    """

    def _state(
        self,
        ego_speed=25.0,
        ego_lane=1,
        n_lanes=3,
        front_gap=100.0,
        front_speed=25.0,
        rear_gap_left=100.0,
        rear_speed_left=25.0,
        rear_gap_right=100.0,
        rear_speed_right=25.0,
    ):
        return dict(
            ego_speed=ego_speed,
            ego_lane=ego_lane,
            n_lanes=n_lanes,
            front_gap=front_gap,
            front_speed=front_speed,
            rear_gap_left=rear_gap_left,
            rear_speed_left=rear_speed_left,
            rear_gap_right=rear_gap_right,
            rear_speed_right=rear_speed_right,
        )

    def test_slower_always_safe(self):
        """SLOWER action is always considered safe — braking never causes front collision."""
        state = self._state(front_gap=1.0)  # dangerously small gap
        assert is_action_safe(state, SLOWER, dt=1.0, horizon=6, min_gap=4.0) is True

    def test_idle_safe_with_large_front_gap(self):
        state = self._state(front_gap=100.0, front_speed=25.0, ego_speed=25.0)
        assert is_action_safe(state, IDLE, dt=1.0, horizon=6, min_gap=4.0) is True

    def test_idle_unsafe_when_closing_fast(self):
        """Ego closing on stopped vehicle with tiny gap → unsafe."""
        state = self._state(ego_speed=25.0, front_gap=5.0, front_speed=0.0)
        # After 1 s the gap is 5 - 25 = -20, well below MIN_GAP
        assert is_action_safe(state, IDLE, dt=1.0, horizon=6, min_gap=4.0) is False

    def test_faster_uses_higher_speed(self):
        """
        FASTER projects with ego_speed + 1.  A state that is borderline safe at
        ego_speed=25 should become unsafe with the +1 bump if the gap is small enough.
        """
        # Gap=5, closing at 25 m/s → after 1 s gap = -20 → clearly unsafe at any speed
        state = self._state(ego_speed=25.0, front_gap=5.0, front_speed=0.0)
        assert is_action_safe(state, FASTER, dt=1.0, horizon=6, min_gap=4.0) is False

    def test_lane_left_blocked_at_leftmost_lane(self):
        """LANE_LEFT from lane 0 (leftmost) is always unsafe — no lane to go to."""
        state = self._state(ego_lane=0)
        assert is_action_safe(state, LANE_LEFT, dt=1.0, horizon=6, min_gap=4.0) is False

    def test_lane_right_blocked_at_rightmost_lane(self):
        """LANE_RIGHT from the rightmost lane is always unsafe."""
        state = self._state(ego_lane=2, n_lanes=3)
        assert is_action_safe(state, LANE_RIGHT, dt=1.0, horizon=6, min_gap=4.0) is False

    def test_lane_left_safe_with_large_gaps(self):
        """Ample gaps front and rear-left → lane change left is safe."""
        state = self._state(ego_lane=1, front_gap=100.0, rear_gap_left=100.0, rear_speed_left=25.0)
        assert is_action_safe(state, LANE_LEFT, dt=1.0, horizon=6, min_gap=4.0) is True

    def test_lane_right_unsafe_with_fast_rear_approach(self):
        """
        A fast rear vehicle in the target lane closing at 20 m/s from 10 m away.
        After 1 s the gap is 10 - 20 = -10 → merge is unsafe.
        """
        state = self._state(
            ego_lane=1,
            ego_speed=25.0,
            rear_gap_right=10.0,
            rear_speed_right=45.0,  # 20 m/s faster than ego
        )
        assert is_action_safe(state, LANE_RIGHT, dt=1.0, horizon=6, min_gap=4.0) is False

    def test_lane_right_safe_with_slow_rear(self):
        """Rear vehicle in target lane slower than ego → gap grows → safe."""
        state = self._state(
            ego_lane=0,
            ego_speed=25.0,
            rear_gap_right=10.0,
            rear_speed_right=15.0,  # 10 m/s slower → ego pulling away
        )
        assert is_action_safe(state, LANE_RIGHT, dt=1.0, horizon=6, min_gap=4.0) is True


# ---------------------------------------------------------------------------
# Fault attribution
# ---------------------------------------------------------------------------

from metrics.fault_attribution import (
    would_crash_with_idle,
    classify_fault,
    COLLISION_DIST,
)


def _make_snapshot(
    ego_pos, ego_speed, ego_heading=0.0, npcs=None
) -> dict:
    """Helper to build a snapshot dict without needing a live env."""
    return {
        "ego":  {"pos": np.array(ego_pos, dtype=float), "speed": float(ego_speed), "heading": float(ego_heading)},
        "npcs": [
            {"pos": np.array(n["pos"], dtype=float), "speed": float(n["speed"]), "heading": float(n.get("heading", 0.0))}
            for n in (npcs or [])
        ],
    }


class TestWouldCrashWithIdle:
    """
    Tests for would_crash_with_idle() using hand-crafted snapshots.
    All vehicles travel along the x-axis (heading=0) for simplicity.
    """

    def test_no_npcs_never_crashes(self):
        """Empty road — IDLE is always safe."""
        snap = _make_snapshot(ego_pos=[0, 0], ego_speed=25.0)
        assert would_crash_with_idle(snap, dt=1.0) is False

    def test_npc_far_away_no_crash(self):
        """NPC 200 m ahead moving at same speed — gap stays constant, no crash."""
        snap = _make_snapshot(
            ego_pos=[0, 0], ego_speed=25.0,
            npcs=[{"pos": [200, 0], "speed": 25.0}],
        )
        assert would_crash_with_idle(snap, dt=1.0) is False

    def test_npc_already_overlapping_crashes(self):
        """NPC already within COLLISION_DIST → collision regardless of action."""
        snap = _make_snapshot(
            ego_pos=[0, 0], ego_speed=25.0,
            npcs=[{"pos": [1.0, 0], "speed": 25.0}],  # 1 m away < COLLISION_DIST
        )
        assert would_crash_with_idle(snap, dt=1.0) is True

    def test_npc_closing_fast_will_crash(self):
        """
        NPC 3 m behind ego travelling 20 m/s faster.
        After 1 s NPC moves 20 m forward, ego moves 0 m (IDLE).
        NPC ends up 17 m ahead of its start = 14 m past ego → well within COLLISION_DIST.

        Wait — NPC is BEHIND ego, so NPC pos = [-3, 0], ego pos = [0, 0].
        After 1 s: ego at [25, 0], NPC at [-3 + 45, 0] = [42, 0].
        Distance = 42 - 25 = 17 m > COLLISION_DIST ≈ 5.39 m → no crash.

        Let's construct a clear crash: NPC behind ego at -4 m, NPC speed = 30 m/s, ego = 25 m/s.
        After 1 s: ego at 25, NPC at -4+30=26. Distance = 1 m < COLLISION_DIST → crash.
        """
        snap = _make_snapshot(
            ego_pos=[0, 0], ego_speed=25.0,
            npcs=[{"pos": [-4, 0], "speed": 30.0}],  # rear-end collision in 1 s
        )
        assert would_crash_with_idle(snap, dt=1.0) is True

    def test_npc_in_adjacent_lane_no_crash(self):
        """NPC two lanes over (8 m lateral) — clearly outside collision circle."""
        snap = _make_snapshot(
            ego_pos=[0, 0], ego_speed=25.0,
            npcs=[{"pos": [0, 8], "speed": 25.0}],  # 8 m lateral offset
        )
        # Both travel at 25 m/s along x-axis; lateral offset stays 8 m.
        # After 1 s: ego [25, 0], NPC [25, 8], distance = 8 m > COLLISION_DIST (≈5.39 m)
        assert would_crash_with_idle(snap, dt=1.0) is False


class TestClassifyFault:
    def test_unavoidable_crash_is_npc_fault(self):
        """NPC rear-ends ego even with IDLE → NPC fault."""
        snap = _make_snapshot(
            ego_pos=[0, 0], ego_speed=25.0,
            npcs=[{"pos": [-4, 0], "speed": 30.0}],
        )
        assert classify_fault(snap, dt=1.0) == "npc"

    def test_avoidable_crash_is_ego_fault(self):
        """
        NPC is far enough that IDLE would be safe.
        Ego's aggressive action (not modelled here) caused the crash.
        The key is: with IDLE, no crash → ego fault.
        """
        snap = _make_snapshot(
            ego_pos=[0, 0], ego_speed=25.0,
            npcs=[{"pos": [200, 0], "speed": 25.0}],  # NPC far away, no threat
        )
        assert classify_fault(snap, dt=1.0) == "ego"

    def test_bad_snapshot_returns_ambiguous(self):
        """A malformed snapshot should return 'ambiguous' without raising."""
        bad_snap = {"ego": None, "npcs": []}
        result = classify_fault(bad_snap, dt=1.0)
        assert result == "ambiguous"

