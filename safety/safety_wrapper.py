"""SafetyWrapper — conservative bounded-acceleration safety filtering.

The wrapper sits between the policy and the environment. It validates the
proposed action against a bounded-acceleration forward model, fails closed on
unknown or malformed state, and only falls back to another action after that
fallback has also been safety-checked.

Lane-change safety is evaluated against both current-lane and target-lane
front/rear interactions so the filter does not treat lateral manoeuvres as
purely one-dimensional decisions.
"""
from __future__ import annotations

import numpy as np
import gymnasium as gym
from typing import Protocol

from config.settings import (
    BehaviorAction,
    EGO_EXPECTED_DECELERATION_MPS2,
    EGO_MAX_ACCELERATION_MPS2,
    FRONT_VEHICLE_MAX_BRAKING_MPS2,
    MIN_FRONT_GAP_M,
    MIN_REAR_GAP_M,
    REAR_VEHICLE_MAX_ACCELERATION_MPS2,
    SAFETY_HORIZON,
    SAFETY_PREDICTION_DT_S,
    SafetyConfig,
)

# Backward-compatible aliases exported for tests/importers.
HORIZON = SAFETY_HORIZON
LANE_LEFT = int(BehaviorAction.LANE_LEFT)
IDLE = int(BehaviorAction.IDLE)
LANE_RIGHT = int(BehaviorAction.LANE_RIGHT)
FASTER = int(BehaviorAction.FASTER)
SLOWER = int(BehaviorAction.SLOWER)
_VALID_ACTIONS = frozenset({LANE_LEFT, IDLE, LANE_RIGHT, FASTER, SLOWER})


# ---------------------------------------------------------------------------
# Policy protocol — same as evaluator.py
# ---------------------------------------------------------------------------

class Policy(Protocol):
    def act(self, obs: np.ndarray) -> int: ...


# ---------------------------------------------------------------------------
# Constant-velocity forward projection helpers
# ---------------------------------------------------------------------------

def _get_road_state(env: gym.Env) -> dict:
    """
    Extract raw road state needed for forward projection.

    Returns a dict with:
      ego_speed     : float, current ego speed (m/s)
      ego_lane      : int, current lane index
      n_lanes       : int, total lane count
      front_gap     : float, gap to front vehicle in current lane (m), inf if none
      front_speed   : float, speed of front vehicle (m/s), ego_speed if none
      rear_gap_left : float, gap to rear vehicle in left lane (m), inf if none
      rear_speed_left : float, speed of left-lane rear vehicle, 0 if none
      rear_gap_right: float, gap to rear vehicle in right lane (m), inf if none
      rear_speed_right: float, speed of right-lane rear vehicle, 0 if none
    front_gap_left: float, gap to front vehicle in left lane (m), inf if none
    front_speed_left: float, speed of left-lane front vehicle, ego_speed if none
    front_gap_right: float, gap to front vehicle in right lane (m), inf if none
    front_speed_right: float, speed of right-lane front vehicle, ego_speed if none
    """
    try:
        road = env.unwrapped.road
        ego  = env.unwrapped.vehicle
        n_lanes = env.unwrapped.config.get("lanes_count", 3)
        ego_lane: int = ego.lane_index[2]
        ego_speed = float(ego.speed)
    except (AttributeError, IndexError, TypeError, ValueError):
        return {}

    # Front vehicle in current lane. If road queries fail, mark unknown as unsafe.
    try:
        neighbours = road.neighbour_vehicles(ego)
        front = neighbours[0] if neighbours else None
        front_gap   = ego.lane_distance_to(front) if front is not None else np.inf
        front_speed = front.speed if front is not None else ego.speed
    except (AttributeError, IndexError, TypeError, ValueError):
        return {}

    # Rear vehicle in left lane (lane_index - 1)
    rear_left_gap, rear_left_speed = _lane_rear_gap(road, ego, ego_lane - 1)

    # Rear vehicle in right lane (lane_index + 1)
    rear_right_gap, rear_right_speed = _lane_rear_gap(road, ego, ego_lane + 1)

    # Front vehicles in adjacent lanes
    front_left_gap, front_left_speed = _lane_front_gap(road, ego, ego_lane - 1)
    front_right_gap, front_right_speed = _lane_front_gap(road, ego, ego_lane + 1)

    return dict(
        ego_speed=ego_speed,
        ego_lane=ego_lane,
        n_lanes=n_lanes,
        front_gap=front_gap,
        front_speed=front_speed,
        rear_gap_left=rear_left_gap,
        rear_speed_left=rear_left_speed,
        rear_gap_right=rear_right_gap,
        rear_speed_right=rear_right_speed,
        front_gap_left=front_left_gap,
        front_speed_left=front_left_speed,
        front_gap_right=front_right_gap,
        front_speed_right=front_right_speed,
    )


def _lane_rear_gap(road, ego, target_lane_idx: int) -> tuple[float, float]:
    """
    Return (gap_to_rear_vehicle, rear_vehicle_speed) for `target_lane_idx`.

    The rear vehicle is the closest vehicle behind the ego in that lane.
    Returns (inf, 0.0) if the lane is empty behind ego.
    Returns conservative fail-safe values on query errors.
    """
    try:
        lane_vehicles = [
            v for v in road.vehicles
            if v is not ego and v.lane_index[2] == target_lane_idx
        ]
        if not lane_vehicles:
            return np.inf, 0.0
        # Ego's longitudinal position on the road
        ego_s = ego.position[0]
        # Vehicles behind ego in that lane: position[0] < ego_s
        behind = [v for v in lane_vehicles if v.position[0] < ego_s]
        if not behind:
            return np.inf, 0.0
        closest_rear = max(behind, key=lambda v: v.position[0])
        gap = ego_s - closest_rear.position[0]
        return max(0.0, gap), closest_rear.speed
    except (AttributeError, IndexError, TypeError, ValueError):
        return np.inf, np.nan


def _lane_front_gap(road, ego, target_lane_idx: int) -> tuple[float, float]:
    """
    Return (gap_to_front_vehicle, front_vehicle_speed) for `target_lane_idx`.

    The front vehicle is the closest vehicle ahead of ego in that lane.
    Returns (inf, ego.speed) if lane is empty ahead of ego.
    Returns conservative fail-safe values on query errors.
    """
    try:
        lane_vehicles = [
            v for v in road.vehicles
            if v is not ego and v.lane_index[2] == target_lane_idx
        ]
        if not lane_vehicles:
            return np.inf, ego.speed

        ego_s = ego.position[0]
        ahead = [v for v in lane_vehicles if v.position[0] > ego_s]
        if not ahead:
            return np.inf, ego.speed

        closest_front = min(ahead, key=lambda v: v.position[0])
        gap = closest_front.position[0] - ego_s
        return max(0.0, gap), closest_front.speed
    except (AttributeError, IndexError, TypeError, ValueError):
        return np.inf, np.nan


def _project_front_gap(
    front_gap: float,
    ego_speed: float,
    front_speed: float,
    dt: float,
    horizon: int,
    ego_accel: float = 0.0,
    front_accel: float = 0.0,
) -> float:
    """
    Project the front gap forward using a constant-velocity model.

    gap_{t+1} = gap_t + (front_speed - ego_speed) * dt

    Returns the minimum gap across all projected timesteps.
    """
    gap = front_gap
    min_gap = gap
    for _ in range(horizon):
        gap += (front_speed - ego_speed) * dt + 0.5 * (front_accel - ego_accel) * dt * dt
        ego_speed += ego_accel * dt
        front_speed += front_accel * dt
        min_gap = min(min_gap, gap)
    return min_gap


def _project_rear_gap(
    rear_gap: float,
    ego_speed: float,
    rear_speed: float,
    dt: float,
    horizon: int,
    ego_accel: float = 0.0,
    rear_accel: float = 0.0,
) -> float:
    """
    Project the gap from the rear vehicle to the ego (in target lane after merge).

    After the lane change the rear vehicle approaches at (rear_speed - ego_speed).
    gap_{t+1} = gap_t + (ego_speed - rear_speed) * dt   [rear closes if rear faster]

    Returns the minimum gap across all projected timesteps.
    """
    gap = rear_gap
    min_gap = gap
    for _ in range(horizon):
        gap += (ego_speed - rear_speed) * dt + 0.5 * (ego_accel - rear_accel) * dt * dt
        ego_speed += ego_accel * dt
        rear_speed += rear_accel * dt
        min_gap = min(min_gap, gap)
    return min_gap


# ---------------------------------------------------------------------------
# Safety predicate
# ---------------------------------------------------------------------------

def is_action_safe(
    state: dict,
    action: int,
    dt: float,
    horizon: int,
    min_gap: float,
    safety_config: SafetyConfig | None = None,
) -> bool:
    """
    Return True if `action` passes the forward-projection safety check.

    Checks performed:
    - FASTER / IDLE: front gap must stay >= min_gap for `horizon` steps.
      (FASTER slightly increases closing speed; we use ego_speed+1 as worst case.)
    - LANE_LEFT / LANE_RIGHT: same front-gap check in target lane, plus the
      rear vehicle in the target lane must also stay >= min_gap post-merge.
    - SLOWER: always safe from a collision standpoint (increases front gap).
    """
    cfg = safety_config or SafetyConfig()

    if action not in _VALID_ACTIONS:
        return False

    required = ["ego_speed", "ego_lane", "n_lanes", "front_gap", "front_speed"]
    if not state or any(key not in state for key in required):
        return False

    ego_speed = float(state["ego_speed"])
    ego_lane = int(state["ego_lane"])
    n_lanes = int(state["n_lanes"])
    front_gap = float(state["front_gap"])
    front_speed = float(state["front_speed"])

    if not np.isfinite([ego_speed, front_speed]).all() or np.isnan(front_gap):
        return False

    front_brake = -abs(cfg.front_vehicle_max_braking_mps2)
    rear_accel = abs(cfg.rear_vehicle_max_acceleration_mps2)

    if action in (IDLE, FASTER):
        ego_accel = 0.0 if action == IDLE else abs(cfg.ego_max_acceleration_mps2)
        proj = _project_front_gap(
            front_gap,
            ego_speed,
            front_speed,
            dt,
            horizon,
            ego_accel=ego_accel,
            front_accel=front_brake,
        )
        return proj >= min_gap

    if action == SLOWER:
        proj = _project_front_gap(
            front_gap,
            ego_speed,
            front_speed,
            dt,
            horizon,
            ego_accel=-abs(cfg.ego_expected_deceleration_mps2),
            front_accel=front_brake,
        )
        return proj >= min_gap

    if action == LANE_LEFT:
        if ego_lane == 0:
            return False
        rear_gap   = state["rear_gap_left"]
        rear_speed = state["rear_speed_left"]
        front_gap_target = state.get("front_gap_left", front_gap)
        front_speed_target = state.get("front_speed_left", front_speed)
        if not np.isfinite([rear_gap, rear_speed, front_speed_target]).all() or np.isnan(front_gap_target):
            return False
        proj_rear = _project_rear_gap(
            rear_gap,
            ego_speed,
            rear_speed,
            dt,
            horizon,
            ego_accel=0.0,
            rear_accel=rear_accel,
        )
        proj_front = _project_front_gap(
            front_gap_target,
            ego_speed,
            front_speed_target,
            dt,
            horizon,
            ego_accel=0.0,
            front_accel=front_brake,
        )
        return proj_rear >= min_gap and proj_front >= min_gap

    if action == LANE_RIGHT:
        if ego_lane >= n_lanes - 1:
            return False
        rear_gap   = state["rear_gap_right"]
        rear_speed = state["rear_speed_right"]
        front_gap_target = state.get("front_gap_right", front_gap)
        front_speed_target = state.get("front_speed_right", front_speed)
        if not np.isfinite([rear_gap, rear_speed, front_speed_target]).all() or np.isnan(front_gap_target):
            return False
        proj_rear = _project_rear_gap(
            rear_gap,
            ego_speed,
            rear_speed,
            dt,
            horizon,
            ego_accel=0.0,
            rear_accel=rear_accel,
        )
        proj_front = _project_front_gap(
            front_gap_target,
            ego_speed,
            front_speed_target,
            dt,
            horizon,
            ego_accel=0.0,
            front_accel=front_brake,
        )
        return proj_rear >= min_gap and proj_front >= min_gap

    return False


# ---------------------------------------------------------------------------
# SafetyWrapper
# ---------------------------------------------------------------------------

class SafetyWrapper:
    """
    Wraps any policy with a forward-projection safety filter.

    Parameters
    ----------
    inner : Policy
        The policy to guard (BC, DAgger, PPO, random, etc.).
    env : gym.Env
        The live highway-v0 environment (must be unwrapped to read road state).
    fallback_policy : Policy, optional
        Policy to call when the inner action is unsafe.  Defaults to IDMExpert.
    horizon : int
        Number of policy steps to project forward (default 3).
    min_gap : float
        Minimum acceptable gap in metres (default 4.0).
    """

    def __init__(
        self,
        inner: Policy,
        env: gym.Env,
        fallback_policy: Policy | None = None,
        horizon: int = HORIZON,
        min_gap: float = MIN_FRONT_GAP_M,
    ) -> None:
        self.inner    = inner
        self.env      = env
        self.horizon  = horizon
        self.min_gap  = min_gap
        self.config   = SafetyConfig(
            prediction_horizon_s=float(horizon),
            prediction_dt_s=float(env.unwrapped.config.get("policy_frequency", 1) ** -1),
            minimum_front_gap_m=float(min_gap),
            minimum_rear_gap_m=float(MIN_REAR_GAP_M),
        )

        if fallback_policy is None:
            from policies.idm_expert import IDMExpert
            self.fallback = IDMExpert(env)
        else:
            self.fallback = fallback_policy

        policy_freq = env.unwrapped.config.get("policy_frequency", 1)
        self.dt = 1.0 / policy_freq

    def _fallback_candidates(self, proposed_action: int, fallback_action: int) -> list[int]:
        ordered = [fallback_action, SLOWER, IDLE, LANE_LEFT, LANE_RIGHT, FASTER]
        result: list[int] = []
        for candidate in [proposed_action, *ordered]:
            if candidate not in result:
                result.append(candidate)
        return result

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def act(self, obs: np.ndarray) -> int:
        """
        Return the safe action for this observation.

        Does NOT set info["fallback"] — use act_with_info() or
        SafetyFilteredEnv for full metric integration.
        """
        action, _ = self.act_with_info(obs)
        return action

    def filter_action(self, obs: np.ndarray, proposed_action: int) -> tuple[int, dict]:
        """
        Filter a caller-provided action and return (safe_action, {"fallback": bool}).

        This is the API used by SafetyFilteredEnv so the wrapper stays
        compatible with the gymnasium step(action) contract.
        """
        state = _get_road_state(self.env)
        if is_action_safe(state, proposed_action, self.dt, self.horizon, self.min_gap, self.config):
            return proposed_action, {"fallback": False, "safety_checked": True, "fallback_safe": True}

        fallback_action = self.fallback.act(obs)
        telemetry = {
            "fallback": True,
            "safety_checked": True,
            "proposed_action_safe": False,
            "fallback_action": int(fallback_action),
        }
        for candidate in self._fallback_candidates(proposed_action, int(fallback_action)):
            if is_action_safe(state, candidate, self.dt, self.horizon, self.min_gap, self.config):
                telemetry["fallback_safe"] = True
                telemetry["selected_action"] = int(candidate)
                return int(candidate), telemetry

        telemetry["fallback_safe"] = False
        telemetry["selected_action"] = int(SLOWER)
        return int(SLOWER), telemetry

    def act_with_info(self, obs: np.ndarray) -> tuple[int, dict]:
        """
        Return (action, {"fallback": bool}).

        Callers that manage the env.step() loop should merge the returned dict
        into the info dict from env.step().
        """
        proposed = self.inner.act(obs)
        return self.filter_action(obs, proposed)


# ---------------------------------------------------------------------------
# SafetyFilteredEnv — convenience wrapper around the gym env
# ---------------------------------------------------------------------------

class SafetyFilteredEnv(gym.Wrapper):
    """
    Gym wrapper that applies SafetyWrapper inside env.step().

    The inner policy is queried for an action, the safety filter decides
    whether to override it, and the final action is stepped into the env.
    info["fallback"] is set on every step.

    Usage
    -----
        env    = make_env()
        inner  = SomePolicy()
        safe_env = SafetyFilteredEnv(env, inner)

        obs, _ = safe_env.reset()
        action = some_policy.act(obs)
        obs, reward, term, trunc, info = safe_env.step(action)
        # info["fallback"] is always present

    For the evaluator, pass SafetyWrapper directly:
        safe_policy = SafetyWrapper(inner, env)
        evaluate(safe_policy, env, ...)
    """

    def __init__(
        self,
        env: gym.Env,
        inner: Policy,
        fallback_policy: Policy | None = None,
        horizon: int = HORIZON,
        min_gap: float = MIN_FRONT_GAP_M,
    ) -> None:
        super().__init__(env)
        self._last_obs: np.ndarray | None = None
        self._safety = SafetyWrapper(
            inner, env,
            fallback_policy=fallback_policy,
            horizon=horizon,
            min_gap=min_gap,
        )

    def reset(self, **kwargs):
        """Reset the env and store the observation for safety filtering."""
        obs, info = self.env.reset(**kwargs)
        self._last_obs = np.asarray(obs, dtype=np.float32)
        return obs, info

    def step(self, action):
        """
        Apply safety filter to the proposed action and step the wrapped env.

        The signature intentionally follows gymnasium's step(action) API.
        """
        if self._last_obs is None:
            raise RuntimeError("Call reset() before step() in SafetyFilteredEnv")

        safe_action, extras = self._safety.filter_action(self._last_obs, int(action))
        next_obs, reward, terminated, truncated, info = self.env.step(safe_action)
        info.update(extras)
        self._last_obs = np.asarray(next_obs, dtype=np.float32)
        return next_obs, reward, terminated, truncated, info
