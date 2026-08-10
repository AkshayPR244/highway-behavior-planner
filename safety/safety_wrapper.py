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
_EMERGENCY_ACTION_PRIORITY = (SLOWER, IDLE, LANE_LEFT, LANE_RIGHT, FASTER)
_LAST_RESORT_ACTION = IDLE


# ---------------------------------------------------------------------------
# Policy protocol — same as evaluator.py
# ---------------------------------------------------------------------------

class Policy(Protocol):
    def act(self, obs: np.ndarray) -> int: ...


# ---------------------------------------------------------------------------
# Bounded-acceleration forward projection helpers
# ---------------------------------------------------------------------------


def _prediction_grid_from_config(cfg: SafetyConfig) -> tuple[float, int]:
    """Return (dt, n_steps) for bounded-acceleration prediction."""
    horizon_s = float(cfg.prediction_horizon_s)
    dt = float(cfg.prediction_dt_s)
    if not np.isfinite([horizon_s, dt]).all() or horizon_s <= 0.0 or dt <= 0.0:
        raise ValueError("SafetyConfig prediction_horizon_s and prediction_dt_s must be finite and > 0")
    if dt > horizon_s:
        raise ValueError("SafetyConfig prediction_dt_s must be <= prediction_horizon_s")
    n_steps = int(np.ceil(horizon_s / dt))
    return dt, max(1, n_steps)


def _ego_acceleration_for_action(action: int, cfg: SafetyConfig) -> float:
    """Deterministic ego longitudinal acceleration assumption by action."""
    if action == FASTER:
        return float(abs(cfg.ego_max_acceleration_mps2))
    if action == SLOWER:
        return -float(abs(cfg.ego_expected_deceleration_mps2))
    if action in (IDLE, LANE_LEFT, LANE_RIGHT):
        return 0.0
    return np.nan


def _propagate_longitudinal(position: float, speed: float, acceleration: float, dt: float) -> tuple[float, float]:
    """Propagate one timestep with non-negative speed clamping."""
    if not np.isfinite([position, speed, acceleration, dt]).all() or dt <= 0.0:
        return np.nan, np.nan

    pos = float(position)
    vel = max(0.0, float(speed))
    acc = float(acceleration)

    if vel <= 0.0:
        return pos, 0.0

    if acc >= 0.0:
        next_pos = pos + vel * dt + 0.5 * acc * dt * dt
        next_vel = max(0.0, vel + acc * dt)
        return next_pos, next_vel

    # Braking branch: stop within the timestep, then remain stopped.
    t_stop = vel / (-acc)
    if t_stop >= dt:
        next_pos = pos + vel * dt + 0.5 * acc * dt * dt
        next_vel = max(0.0, vel + acc * dt)
        return next_pos, next_vel

    dist_to_stop = vel * t_stop + 0.5 * acc * t_stop * t_stop
    next_pos = pos + max(0.0, dist_to_stop)
    return next_pos, 0.0

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
        ego = env.unwrapped.vehicle
        n_lanes = int(env.unwrapped.config.get("lanes_count", 3))
        ego_lane = int(ego.lane_index[2])
        ego_speed = float(ego.speed)
    except (AttributeError, IndexError, TypeError, ValueError) as exc:
        return {"state_error": f"road_state_unavailable:{type(exc).__name__}"}

    front_gap, front_speed = _lane_front_gap(road, ego, ego_lane)
    rear_gap, rear_speed = _lane_rear_gap(road, ego, ego_lane)
    front_left_gap, front_left_speed = _lane_front_gap(road, ego, ego_lane - 1)
    front_right_gap, front_right_speed = _lane_front_gap(road, ego, ego_lane + 1)
    rear_left_gap, rear_left_speed = _lane_rear_gap(road, ego, ego_lane - 1)
    rear_right_gap, rear_right_speed = _lane_rear_gap(road, ego, ego_lane + 1)

    current_vals = [front_gap, front_speed, rear_gap, rear_speed, ego_speed]
    if any(np.isnan(v) for v in current_vals):
        return {"state_error": "road_state_query_failed"}

    lanes = {
        "current": {
            "front_gap": front_gap,
            "front_speed": front_speed,
            "rear_gap": rear_gap,
            "rear_speed": rear_speed,
        },
        "left": {
            "front_gap": front_left_gap,
            "front_speed": front_left_speed,
            "rear_gap": rear_left_gap,
            "rear_speed": rear_left_speed,
        },
        "right": {
            "front_gap": front_right_gap,
            "front_speed": front_right_speed,
            "rear_gap": rear_right_gap,
            "rear_speed": rear_right_speed,
        },
    }

    return dict(
        ego_speed=ego_speed,
        ego_lane=ego_lane,
        n_lanes=n_lanes,
        front_gap=front_gap,
        front_speed=front_speed,
        rear_gap=rear_gap,
        rear_speed=rear_speed,
        rear_gap_left=rear_left_gap,
        rear_speed_left=rear_left_speed,
        rear_gap_right=rear_right_gap,
        rear_speed_right=rear_right_speed,
        front_gap_left=front_left_gap,
        front_speed_left=front_left_speed,
        front_gap_right=front_right_gap,
        front_speed_right=front_right_speed,
        lanes=lanes,
    )


def _lane_rear_gap(road, ego, target_lane_idx: int) -> tuple[float, float]:
    """
    Return (gap_to_rear_vehicle, rear_vehicle_speed) for `target_lane_idx`.

    The rear vehicle is the closest vehicle behind the ego in that lane.
    Returns (inf, 0.0) if the lane is empty behind ego.
    Returns conservative fail-safe values on query errors.
    """
    try:
        n_lanes = len(getattr(road.network, "graph", {}).get("0", {}).get("1", []))
        if n_lanes <= 0:
            n_lanes = int(getattr(getattr(road, "config", {}), "get", lambda *_: 3)("lanes_count", 3))
        if target_lane_idx < 0 or target_lane_idx >= n_lanes:
            return np.nan, np.nan

        lane_vehicles = [
            v for v in road.vehicles
            if v is not ego and v.lane_index[2] == target_lane_idx
        ]
        if not lane_vehicles:
            return np.inf, float(ego.speed)
        # Ego's longitudinal position on the road
        ego_s = ego.position[0]
        # Vehicles behind ego in that lane: position[0] < ego_s
        behind = [v for v in lane_vehicles if v.position[0] < ego_s]
        if not behind:
            return np.inf, 0.0
        closest_rear = max(behind, key=lambda v: v.position[0])
        gap = ego_s - closest_rear.position[0]
        return max(0.0, gap), float(closest_rear.speed)
    except (AttributeError, IndexError, TypeError, ValueError):
        return np.nan, np.nan


def _lane_front_gap(road, ego, target_lane_idx: int) -> tuple[float, float]:
    """
    Return (gap_to_front_vehicle, front_vehicle_speed) for `target_lane_idx`.

    The front vehicle is the closest vehicle ahead of ego in that lane.
    Returns (inf, ego.speed) if lane is empty ahead of ego.
    Returns conservative fail-safe values on query errors.
    """
    try:
        n_lanes = len(getattr(road.network, "graph", {}).get("0", {}).get("1", []))
        if n_lanes <= 0:
            n_lanes = int(getattr(getattr(road, "config", {}), "get", lambda *_: 3)("lanes_count", 3))
        if target_lane_idx < 0 or target_lane_idx >= n_lanes:
            return np.nan, np.nan

        lane_vehicles = [
            v for v in road.vehicles
            if v is not ego and v.lane_index[2] == target_lane_idx
        ]
        if not lane_vehicles:
            return np.inf, float(ego.speed)

        ego_s = ego.position[0]
        ahead = [v for v in lane_vehicles if v.position[0] > ego_s]
        if not ahead:
            return np.inf, float(ego.speed)

        closest_front = min(ahead, key=lambda v: v.position[0])
        gap = closest_front.position[0] - ego_s
        return max(0.0, gap), float(closest_front.speed)
    except (AttributeError, IndexError, TypeError, ValueError):
        return np.nan, np.nan


def _project_front_gap(
    front_gap: float,
    ego_speed: float,
    front_speed: float,
    dt: float,
    horizon: int,
    ego_accel: float = 0.0,
    front_accel: float = 0.0,
) -> float:
    """Project front clearance using bounded-acceleration kinematics."""
    if not np.isfinite([front_gap, ego_speed, front_speed, dt, ego_accel, front_accel]).all() or horizon <= 0:
        return np.nan

    ego_x, ego_v = 0.0, max(0.0, float(ego_speed))
    front_x, front_v = float(front_gap), max(0.0, float(front_speed))
    min_gap = front_x - ego_x

    for _ in range(int(horizon)):
        ego_x, ego_v = _propagate_longitudinal(ego_x, ego_v, ego_accel, dt)
        front_x, front_v = _propagate_longitudinal(front_x, front_v, front_accel, dt)
        if not np.isfinite([ego_x, ego_v, front_x, front_v]).all():
            return np.nan
        min_gap = min(min_gap, front_x - ego_x)

    return float(min_gap)


def _project_rear_gap(
    rear_gap: float,
    ego_speed: float,
    rear_speed: float,
    dt: float,
    horizon: int,
    ego_accel: float = 0.0,
    rear_accel: float = 0.0,
) -> float:
    """Project rear clearance using bounded-acceleration kinematics."""
    if not np.isfinite([rear_gap, ego_speed, rear_speed, dt, ego_accel, rear_accel]).all() or horizon <= 0:
        return np.nan

    ego_x, ego_v = 0.0, max(0.0, float(ego_speed))
    rear_x, rear_v = -float(rear_gap), max(0.0, float(rear_speed))
    min_gap = ego_x - rear_x

    for _ in range(int(horizon)):
        ego_x, ego_v = _propagate_longitudinal(ego_x, ego_v, ego_accel, dt)
        rear_x, rear_v = _propagate_longitudinal(rear_x, rear_v, rear_accel, dt)
        if not np.isfinite([ego_x, ego_v, rear_x, rear_v]).all():
            return np.nan
        min_gap = min(min_gap, ego_x - rear_x)

    return float(min_gap)


def _evaluate_action_safety(
    state: dict,
    action: int,
    dt: float,
    horizon: int,
    min_gap: float,
    safety_config: SafetyConfig | None = None,
) -> tuple[bool, dict]:
    """Evaluate safety and return (is_safe, diagnostics)."""
    cfg = safety_config or SafetyConfig()
    diagnostics = {
        "minimum_predicted_front_gap": np.nan,
        "minimum_predicted_rear_gap": np.nan,
        "prediction_failure": False,
    }

    if action not in _VALID_ACTIONS:
        return False, diagnostics
    if isinstance(state, dict) and state.get("state_error"):
        diagnostics["prediction_failure"] = True
        return False, diagnostics

    required = [
        "ego_speed",
        "ego_lane",
        "n_lanes",
        "front_gap",
        "front_speed",
        "rear_gap",
        "rear_speed",
    ]
    if not state or any(key not in state for key in required):
        diagnostics["prediction_failure"] = True
        return False, diagnostics

    ego_speed = float(state["ego_speed"])
    ego_lane = int(state["ego_lane"])
    n_lanes = int(state["n_lanes"])
    front_gap = float(state["front_gap"])
    front_speed = float(state["front_speed"])
    rear_gap = float(state["rear_gap"])
    rear_speed = float(state["rear_speed"])

    if not np.isfinite([ego_speed, front_speed, rear_speed, front_gap, rear_gap]).all():
        diagnostics["prediction_failure"] = True
        return False, diagnostics

    ego_accel = _ego_acceleration_for_action(action, cfg)
    if not np.isfinite(ego_accel):
        diagnostics["prediction_failure"] = True
        return False, diagnostics

    min_front_gap = max(float(min_gap), float(cfg.minimum_front_gap_m))
    min_rear_gap = float(cfg.minimum_rear_gap_m)
    front_brake = -abs(float(cfg.front_vehicle_max_braking_mps2))
    rear_accel = abs(float(cfg.rear_vehicle_max_acceleration_mps2))

    if action in (IDLE, FASTER):
        projected_front = _project_front_gap(
            front_gap,
            ego_speed,
            front_speed,
            dt,
            horizon,
            ego_accel=ego_accel,
            front_accel=front_brake,
        )
        diagnostics["minimum_predicted_front_gap"] = projected_front
        if not np.isfinite(projected_front):
            diagnostics["prediction_failure"] = True
            return False, diagnostics
        return bool(projected_front >= min_front_gap), diagnostics

    if action == SLOWER:
        projected_front = _project_front_gap(
            front_gap,
            ego_speed,
            front_speed,
            dt,
            horizon,
            ego_accel=ego_accel,
            front_accel=front_brake,
        )
        projected_rear = _project_rear_gap(
            rear_gap,
            ego_speed,
            rear_speed,
            dt,
            horizon,
            ego_accel=ego_accel,
            rear_accel=rear_accel,
        )
        diagnostics["minimum_predicted_front_gap"] = projected_front
        diagnostics["minimum_predicted_rear_gap"] = projected_rear
        if not np.isfinite([projected_front, projected_rear]).all():
            diagnostics["prediction_failure"] = True
            return False, diagnostics
        return bool(projected_front >= min_front_gap and projected_rear >= min_rear_gap), diagnostics

    if action == LANE_LEFT:
        if ego_lane == 0:
            return False, diagnostics
        rear_gap_target = float(state.get("rear_gap_left", np.nan))
        rear_speed_target = float(state.get("rear_speed_left", np.nan))
        front_gap_target = float(state.get("front_gap_left", front_gap))
        front_speed_target = float(state.get("front_speed_left", front_speed))
        if not np.isfinite([rear_gap_target, rear_speed_target, front_gap_target, front_speed_target]).all():
            diagnostics["prediction_failure"] = True
            return False, diagnostics
        projected_rear = _project_rear_gap(
            rear_gap_target,
            ego_speed,
            rear_speed_target,
            dt,
            horizon,
            ego_accel=ego_accel,
            rear_accel=rear_accel,
        )
        projected_front = _project_front_gap(
            front_gap_target,
            ego_speed,
            front_speed_target,
            dt,
            horizon,
            ego_accel=ego_accel,
            front_accel=front_brake,
        )
        diagnostics["minimum_predicted_front_gap"] = projected_front
        diagnostics["minimum_predicted_rear_gap"] = projected_rear
        if not np.isfinite([projected_front, projected_rear]).all():
            diagnostics["prediction_failure"] = True
            return False, diagnostics
        return bool(projected_front >= min_front_gap and projected_rear >= min_rear_gap), diagnostics

    if action == LANE_RIGHT:
        if ego_lane >= n_lanes - 1:
            return False, diagnostics
        rear_gap_target = float(state.get("rear_gap_right", np.nan))
        rear_speed_target = float(state.get("rear_speed_right", np.nan))
        front_gap_target = float(state.get("front_gap_right", front_gap))
        front_speed_target = float(state.get("front_speed_right", front_speed))
        if not np.isfinite([rear_gap_target, rear_speed_target, front_gap_target, front_speed_target]).all():
            diagnostics["prediction_failure"] = True
            return False, diagnostics
        projected_rear = _project_rear_gap(
            rear_gap_target,
            ego_speed,
            rear_speed_target,
            dt,
            horizon,
            ego_accel=ego_accel,
            rear_accel=rear_accel,
        )
        projected_front = _project_front_gap(
            front_gap_target,
            ego_speed,
            front_speed_target,
            dt,
            horizon,
            ego_accel=ego_accel,
            front_accel=front_brake,
        )
        diagnostics["minimum_predicted_front_gap"] = projected_front
        diagnostics["minimum_predicted_rear_gap"] = projected_rear
        if not np.isfinite([projected_front, projected_rear]).all():
            diagnostics["prediction_failure"] = True
            return False, diagnostics
        return bool(projected_front >= min_front_gap and projected_rear >= min_rear_gap), diagnostics

    return False, diagnostics


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
    safe, _ = _evaluate_action_safety(state, action, dt, horizon, min_gap, safety_config)
    return safe


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
        self.inner = inner
        self.env = env
        self.min_gap = min_gap

        policy_freq = env.unwrapped.config.get("policy_frequency", 1)
        policy_dt = 1.0 / policy_freq
        self.config = SafetyConfig(
            prediction_horizon_s=float(horizon) * float(policy_dt),
            prediction_dt_s=float(policy_dt),
            minimum_front_gap_m=float(min_gap),
            minimum_rear_gap_m=float(MIN_REAR_GAP_M),
        )

        self.dt, self.horizon = _prediction_grid_from_config(self.config)

        if fallback_policy is None:
            from policies.idm_expert import IDMExpert
            self.fallback = IDMExpert(env)
        else:
            self.fallback = fallback_policy

    def _fallback_candidates(self, proposed_action: int, fallback_action: int) -> list[int]:
        ordered = [fallback_action, *_EMERGENCY_ACTION_PRIORITY]
        result: list[int] = []
        for candidate in ordered:
            if candidate not in result:
                result.append(candidate)
        return result

    @staticmethod
    def _normalize_action(action: int) -> int | None:
        try:
            action_int = int(action)
        except (TypeError, ValueError):
            return None
        if action_int not in _VALID_ACTIONS:
            return None
        return action_int

    def _is_candidate_safe(self, state: dict, action: int | None) -> tuple[bool, dict]:
        if action is None:
            return False, {
                "minimum_predicted_front_gap": np.nan,
                "minimum_predicted_rear_gap": np.nan,
                "prediction_failure": True,
            }
        return _evaluate_action_safety(state, action, self.dt, self.horizon, self.min_gap, self.config)

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
        state_ok = bool(state) and not bool(state.get("state_error"))

        primary_action = self._normalize_action(proposed_action)
        primary_safe, primary_diag = self._is_candidate_safe(state, primary_action)

        telemetry = {
            "fallback": False,
            "safety_checked": True,
            "state_query_failed": not state_ok,
            "primary_action": int(proposed_action) if primary_action is not None else proposed_action,
            "primary_rejected": not primary_safe,
            "fallback_action": None,
            "fallback_rejected": False,
            "executed_action": None,
            "no_safe_action_found": False,
            "minimum_predicted_front_gap": primary_diag.get("minimum_predicted_front_gap"),
            "minimum_predicted_rear_gap": primary_diag.get("minimum_predicted_rear_gap"),
            "prediction_failure": bool(primary_diag.get("prediction_failure", False)),
        }

        if primary_safe and primary_action is not None:
            telemetry["executed_action"] = primary_action
            return primary_action, telemetry

        telemetry["fallback"] = True
        fallback_action = self._normalize_action(self.fallback.act(obs))
        telemetry["fallback_action"] = fallback_action
        fallback_safe, fallback_diag = self._is_candidate_safe(state, fallback_action)
        telemetry["fallback_rejected"] = not fallback_safe
        if telemetry["fallback_rejected"]:
            telemetry["prediction_failure"] = telemetry["prediction_failure"] or bool(
                fallback_diag.get("prediction_failure", False)
            )

        if fallback_safe and fallback_action is not None:
            telemetry["executed_action"] = fallback_action
            return fallback_action, telemetry

        for candidate in self._fallback_candidates(
            proposed_action=primary_action if primary_action is not None else _LAST_RESORT_ACTION,
            fallback_action=fallback_action if fallback_action is not None else _LAST_RESORT_ACTION,
        ):
            candidate_safe, candidate_diag = self._is_candidate_safe(state, candidate)
            if candidate_safe:
                telemetry["minimum_predicted_front_gap"] = candidate_diag.get("minimum_predicted_front_gap")
                telemetry["minimum_predicted_rear_gap"] = candidate_diag.get("minimum_predicted_rear_gap")
                telemetry["executed_action"] = int(candidate)
                return int(candidate), telemetry
            telemetry["prediction_failure"] = telemetry["prediction_failure"] or bool(
                candidate_diag.get("prediction_failure", False)
            )

        telemetry["no_safe_action_found"] = True
        telemetry["executed_action"] = int(_LAST_RESORT_ACTION)
        return int(_LAST_RESORT_ACTION), telemetry

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
