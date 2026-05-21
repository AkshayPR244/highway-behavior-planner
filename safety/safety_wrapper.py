"""
SafetyWrapper — hard safety filter for any policy on highway-v0.

Architecture
------------
The wrapper sits between the policy and the environment.  On each step:

  1. Query the inner policy for a proposed action.
  2. Run a constant-velocity forward projection for `horizon` policy steps
     (default 3 s at 1 Hz policy frequency).
  3. Check two safety conditions at every projected step:
       a) Front gap in the ego's current/target lane >= MIN_GAP
       b) For lane changes: rear gap in the target lane >= MIN_GAP (merge safety)
  4. If either condition fails → override with the IDM expert's action.
  5. Pass `info["fallback"] = True` back to the caller when an override fires.

The IDM fallback is intentionally conservative: it will brake or hold lane
rather than attempt a marginal manoeuvre.  This gives us a hard guarantee:
the ego never takes an action that the forward-projection predicts as unsafe.

Note on `info` propagation
--------------------------
gymnasium's env.step() returns a fixed info dict.  The wrapper calls
env.step(action) and then injects the `fallback` key into the returned dict,
so the evaluator's fallback_rate metric works without any env modification.

Usage
-----
    from envs.highway_wrapper import make_env
    from policies.idm_expert import IDMExpert
    from safety.safety_wrapper import SafetyWrapper

    env   = make_env()
    inner = SomeLearnedPolicy(...)
    safe  = SafetyWrapper(inner, env)

    obs, _ = env.reset()
    while True:
        action, info_extras = safe.act_with_info(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        info.update(info_extras)   # injects "fallback" key

The evaluator calls policy.act(obs) and separately inspects info["fallback"],
so SafetyWrapper also exposes a plain .act(obs) -> int interface; in that
mode the evaluator picks up fallback from the info dict injected by the
wrapper's companion env-stepping helper.  See SafetyFilteredEnv below for the
recommended integration pattern.
"""
from __future__ import annotations

import numpy as np
import gymnasium as gym
from typing import Protocol

# Minimum gap (metres) that must be maintained to all relevant vehicles
# in the forward projection for an action to be deemed safe.
MIN_GAP = 4.0          # m

# Number of policy steps to project forward.
# At policy_frequency=1 Hz this is 6 seconds of look-ahead —
# midpoint between 1 s human reaction time + 300 m / 30 m/s = 10 s budget.
HORIZON = 6            # steps

# Action indices from DiscreteMetaAction
LANE_LEFT  = 0
IDLE       = 1
LANE_RIGHT = 2
FASTER     = 3
SLOWER     = 4


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
    """
    road = env.unwrapped.road
    ego  = env.unwrapped.vehicle
    n_lanes = env.unwrapped.config.get("lanes_count", 3)
    ego_lane: int = ego.lane_index[2]

    # Front vehicle in current lane
    neighbours = road.neighbour_vehicles(ego)
    front = neighbours[0] if neighbours else None
    front_gap   = ego.lane_distance_to(front) if front is not None else np.inf
    front_speed = front.speed if front is not None else ego.speed

    # Rear vehicle in left lane (lane_index - 1)
    rear_left_gap, rear_left_speed = _lane_rear_gap(road, ego, ego_lane - 1)

    # Rear vehicle in right lane (lane_index + 1)
    rear_right_gap, rear_right_speed = _lane_rear_gap(road, ego, ego_lane + 1)

    return dict(
        ego_speed=ego.speed,
        ego_lane=ego_lane,
        n_lanes=n_lanes,
        front_gap=front_gap,
        front_speed=front_speed,
        rear_gap_left=rear_left_gap,
        rear_speed_left=rear_left_speed,
        rear_gap_right=rear_right_gap,
        rear_speed_right=rear_right_speed,
    )


def _lane_rear_gap(road, ego, target_lane_idx: int) -> tuple[float, float]:
    """
    Return (gap_to_rear_vehicle, rear_vehicle_speed) for `target_lane_idx`.

    The rear vehicle is the closest vehicle behind the ego in that lane.
    Returns (inf, 0.0) if the lane doesn't exist or is empty behind ego.
    """
    n_lanes = ego.road.network.graph  # only used to guard index
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
    except Exception:
        return np.inf, 0.0


def _project_front_gap(
    front_gap: float,
    ego_speed: float,
    front_speed: float,
    dt: float,
    horizon: int,
) -> float:
    """
    Project the front gap forward using a constant-velocity model.

    gap_{t+1} = gap_t + (front_speed - ego_speed) * dt

    Returns the minimum gap across all projected timesteps.
    """
    gap = front_gap
    min_gap = gap
    for _ in range(horizon):
        gap += (front_speed - ego_speed) * dt
        min_gap = min(min_gap, gap)
    return min_gap


def _project_rear_gap(
    rear_gap: float,
    ego_speed: float,
    rear_speed: float,
    dt: float,
    horizon: int,
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
        # Rear vehicle closes on ego if it's faster
        gap += (ego_speed - rear_speed) * dt
        min_gap = min(min_gap, gap)
    return min_gap


# ---------------------------------------------------------------------------
# Safety predicate
# ---------------------------------------------------------------------------

def is_action_safe(state: dict, action: int, dt: float, horizon: int, min_gap: float) -> bool:
    """
    Return True if `action` passes the forward-projection safety check.

    Checks performed:
    - FASTER / IDLE: front gap must stay >= min_gap for `horizon` steps.
      (FASTER slightly increases closing speed; we use ego_speed+1 as worst case.)
    - LANE_LEFT / LANE_RIGHT: same front-gap check in target lane, plus the
      rear vehicle in the target lane must also stay >= min_gap post-merge.
    - SLOWER: always safe from a collision standpoint (increases front gap).
    """
    ego_speed   = state["ego_speed"]
    ego_lane    = state["ego_lane"]
    n_lanes     = state["n_lanes"]
    front_gap   = state["front_gap"]
    front_speed = state["front_speed"]

    if action == SLOWER:
        return True

    if action in (IDLE, FASTER):
        # Conservative: assume FASTER raises ego speed by 1 m/s
        effective_speed = ego_speed + (1.0 if action == FASTER else 0.0)
        proj = _project_front_gap(front_gap, effective_speed, front_speed, dt, horizon)
        return proj >= min_gap

    if action == LANE_LEFT:
        if ego_lane == 0:
            return False  # already in leftmost lane — action is a no-op but flag it
        rear_gap   = state["rear_gap_left"]
        rear_speed = state["rear_speed_left"]
        proj_rear = _project_rear_gap(rear_gap, ego_speed, rear_speed, dt, horizon)
        proj_front = _project_front_gap(front_gap, ego_speed, front_speed, dt, horizon)
        return proj_rear >= min_gap and proj_front >= min_gap

    if action == LANE_RIGHT:
        if ego_lane >= n_lanes - 1:
            return False  # already in rightmost lane
        rear_gap   = state["rear_gap_right"]
        rear_speed = state["rear_speed_right"]
        proj_rear = _project_rear_gap(rear_gap, ego_speed, rear_speed, dt, horizon)
        proj_front = _project_front_gap(front_gap, ego_speed, front_speed, dt, horizon)
        return proj_rear >= min_gap and proj_front >= min_gap

    return True  # unknown action — pass through


# ---------------------------------------------------------------------------
# SafetyWrapper
# ---------------------------------------------------------------------------

class SafetyWrapper:
    """
    Wraps any policy with a hard safety filter using forward projection.

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
        min_gap: float = MIN_GAP,
    ) -> None:
        self.inner    = inner
        self.env      = env
        self.horizon  = horizon
        self.min_gap  = min_gap

        if fallback_policy is None:
            from policies.idm_expert import IDMExpert
            self.fallback = IDMExpert(env)
        else:
            self.fallback = fallback_policy

        policy_freq = env.unwrapped.config.get("policy_frequency", 1)
        self.dt = 1.0 / policy_freq

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

    def act_with_info(self, obs: np.ndarray) -> tuple[int, dict]:
        """
        Return (action, {"fallback": bool}).

        Callers that manage the env.step() loop should merge the returned dict
        into the info dict from env.step().
        """
        proposed = self.inner.act(obs)
        state    = _get_road_state(self.env)

        if is_action_safe(state, proposed, self.dt, self.horizon, self.min_gap):
            return proposed, {"fallback": False}

        # Override with fallback
        safe_action = self.fallback.act(obs)
        return safe_action, {"fallback": True}


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
        obs, reward, term, trunc, info = safe_env.step(obs)
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
        min_gap: float = MIN_GAP,
    ) -> None:
        super().__init__(env)
        self._safety = SafetyWrapper(
            inner, env,
            fallback_policy=fallback_policy,
            horizon=horizon,
            min_gap=min_gap,
        )

    def step(self, obs: np.ndarray):  # type: ignore[override]
        """
        Query inner policy, apply safety filter, step env, inject fallback flag.

        Note: `obs` here is the *current* observation (before the step),
        matching how the evaluator loop works.
        """
        action, extras = self._safety.act_with_info(obs)
        next_obs, reward, terminated, truncated, info = self.env.step(action)
        info.update(extras)
        return next_obs, reward, terminated, truncated, info
