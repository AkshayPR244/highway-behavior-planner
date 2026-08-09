"""
IDM (Intelligent Driver Model) expert policy.

Computes expert actions by directly reading the ego vehicle's road context
(front gap, relative speed, lane occupancy) and applying IDM + MOBIL rules.

DiscreteMetaAction mapping (highway-env default):
  0: LANE_LEFT
  1: IDLE
  2: LANE_RIGHT
  3: FASTER
  4: SLOWER
"""
import numpy as np
import gymnasium as gym

from config.settings import (
    BehaviorAction,
    IDM_ACCEL_COMFORT as _IDM_ACCEL_COMFORT,
    IDM_DECEL_COMFORT as _IDM_DECEL_COMFORT,
    IDM_DESIRED_HEADWAY as _IDM_DESIRED_HEADWAY,
    IDM_DESIRED_SPEED as _IDM_DESIRED_SPEED,
    IDM_MIN_SPACING as _IDM_MIN_SPACING,
    MOBIL_ACCEL_GAIN_THRESHOLD as _MOBIL_ACCEL_GAIN_THRESHOLD,
    MOBIL_B_SAFE as _MOBIL_B_SAFE,
    MOBIL_POLITENESS as _MOBIL_POLITENESS,
)

from envs.highway_wrapper import make_env

# IDM parameters
# Re-exported aliases preserve existing test/public imports.
IDM_DESIRED_SPEED = _IDM_DESIRED_SPEED
IDM_MIN_SPACING = _IDM_MIN_SPACING
IDM_DESIRED_HEADWAY = _IDM_DESIRED_HEADWAY
IDM_ACCEL_COMFORT = _IDM_ACCEL_COMFORT
IDM_DECEL_COMFORT = _IDM_DECEL_COMFORT
ACCEL_THRESHOLD = 0.5         # m/s^2

# MOBIL parameters
MOBIL_B_SAFE = _MOBIL_B_SAFE
MOBIL_POLITENESS = _MOBIL_POLITENESS
MOBIL_ACCEL_GAIN_THRESHOLD = _MOBIL_ACCEL_GAIN_THRESHOLD


def _idm_acceleration(ego_speed: float, front_gap: float, front_rel_speed: float) -> float:
    """Pure IDM acceleration formula."""
    if front_gap <= 0:
        return -IDM_DECEL_COMFORT
    desired_gap = (
        IDM_MIN_SPACING
        + max(0.0, ego_speed * IDM_DESIRED_HEADWAY
              + ego_speed * front_rel_speed / (2 * (IDM_ACCEL_COMFORT * IDM_DECEL_COMFORT) ** 0.5))
    )
    free_road = 1.0 - (ego_speed / IDM_DESIRED_SPEED) ** 4
    interaction = -(desired_gap / front_gap) ** 2
    return IDM_ACCEL_COMFORT * (free_road + interaction)


class IDMExpert:
    """
    Produces expert actions via IDM+MOBIL rules using the env's road state.

    Usage:
        expert = IDMExpert(env)
        obs, _ = env.reset()
        action = expert.act(obs)   # int in [0, 4]
    """

    def __init__(self, env: gym.Env):
        self.env = env

    def _neighbour_gap(self, road, ego, lane_index: tuple, front: bool) -> tuple:
        """
        Return (vehicle, gap_m) for the front or rear neighbour in lane_index.
        gap_m is np.inf if the lane is empty.
        """
        try:
            neighbours = road.neighbour_vehicles(ego, lane_index)
            # highway-env returns [front, rear] neighbours
            vehicle = neighbours[0] if front else (neighbours[1] if len(neighbours) > 1 else None)
            gap = ego.lane_distance_to(vehicle) if vehicle is not None else np.inf
            return vehicle, abs(gap)
        except (AttributeError, IndexError, TypeError, ValueError):
            return None, np.inf

    def _mobil_safe(self, road, ego, target_lane_idx: int) -> bool:
        """
        Full MOBIL criterion for a lane change toward target_lane_idx.

        Two conditions must both hold:

        1. Safety: the new rear vehicle in the target lane must not be forced
           to brake harder than MOBIL_B_SAFE after the merge.

        2. Incentive: the ego's acceleration gain from the lane change must
           exceed the politeness-weighted loss imposed on both followers
           (current-lane follower gains by our departure, target-lane
           follower loses by our arrival).
        """
        target_lane = (*ego.lane_index[:2], target_lane_idx)
        current_lane = ego.lane_index

        # --- Vehicles we need ---
        # Ego's current front (for ego accel before/after)
        curr_front, curr_front_gap = self._neighbour_gap(road, ego, current_lane, front=True)
        # Front in target lane (ego's new leader after change)
        tgt_front, tgt_front_gap = self._neighbour_gap(road, ego, target_lane, front=True)
        # Rear in target lane (becomes ego's new follower)
        tgt_rear, tgt_rear_gap = self._neighbour_gap(road, ego, target_lane, front=False)
        # Rear in current lane (gains headroom when ego leaves)
        curr_rear, curr_rear_gap = self._neighbour_gap(road, ego, current_lane, front=False)

        # --- Safety check ---
        if tgt_rear is not None:
            # What accel would the new follower need after our merge?
            follower_accel = _idm_acceleration(
                tgt_rear.speed,
                tgt_rear_gap,
                tgt_rear.speed - ego.speed,
            )
            if follower_accel < -MOBIL_B_SAFE:
                return False  # unsafe — would force emergency braking

        # --- Incentive check ---
        ego_speed = ego.speed

        # Ego acceleration in current lane
        curr_rel = (ego_speed - curr_front.speed) if curr_front else 0.0
        accel_ego_before = _idm_acceleration(ego_speed, curr_front_gap, curr_rel)

        # Ego acceleration after moving to target lane
        tgt_rel = (ego_speed - tgt_front.speed) if tgt_front else 0.0
        accel_ego_after = _idm_acceleration(ego_speed, tgt_front_gap, tgt_rel)

        # Current-lane rear follower: gains headroom when ego leaves
        if curr_rear is not None:
            accel_curr_rear_before = _idm_acceleration(
                curr_rear.speed, curr_rear_gap, curr_rear.speed - ego.speed
            )
            accel_curr_rear_after = _idm_acceleration(
                curr_rear.speed, curr_rear_gap + curr_front_gap, curr_rear.speed - ego.speed
            )
            curr_rear_gain = accel_curr_rear_after - accel_curr_rear_before
        else:
            curr_rear_gain = 0.0

        # Target-lane rear follower: loses headroom when ego arrives
        if tgt_rear is not None:
            accel_tgt_rear_before = _idm_acceleration(
                tgt_rear.speed, tgt_rear_gap, tgt_rear.speed - tgt_front.speed if tgt_front else 0.0
            )
            accel_tgt_rear_after = _idm_acceleration(
                tgt_rear.speed, tgt_rear_gap, tgt_rear.speed - ego.speed
            )
            tgt_rear_loss = accel_tgt_rear_before - accel_tgt_rear_after
        else:
            tgt_rear_loss = 0.0

        net_gain = (
            (accel_ego_after - accel_ego_before)
            + MOBIL_POLITENESS * (curr_rear_gain - tgt_rear_loss)
        )
        return net_gain > MOBIL_ACCEL_GAIN_THRESHOLD

    def act(self, obs: np.ndarray) -> int:
        """Return the IDM/MOBIL action for the current env state."""
        road = self.env.unwrapped.road
        ego = self.env.unwrapped.vehicle

        ego_speed = ego.speed
        ego_lane_idx = ego.lane_index[2]
        n_lanes = len(road.network.graph[ego.lane_index[0]][ego.lane_index[1]])

        # Longitudinal: IDM via front vehicle gap
        try:
            neighbours = road.neighbour_vehicles(ego)
            front_vehicle = neighbours[0] if neighbours else None
            front_gap = (
                ego.lane_distance_to(front_vehicle)
                if front_vehicle is not None else np.inf
            )
        except (AttributeError, IndexError, TypeError, ValueError):
            front_vehicle, front_gap = None, np.inf

        front_rel_speed = (ego_speed - front_vehicle.speed) if front_vehicle is not None else 0.0
        accel = _idm_acceleration(ego_speed, front_gap, front_rel_speed)

        # Lateral: MOBIL-like — overtake left if blocked, keep right when free
        can_go_left = ego_lane_idx > 0
        can_go_right = ego_lane_idx < n_lanes - 1

        if can_go_left and self._mobil_safe(road, ego, ego_lane_idx - 1):
            return int(BehaviorAction.LANE_LEFT)

        if can_go_right and self._mobil_safe(road, ego, ego_lane_idx + 1):
            return int(BehaviorAction.LANE_RIGHT)

        if accel > ACCEL_THRESHOLD:
            return int(BehaviorAction.FASTER)
        if accel < -ACCEL_THRESHOLD:
            return int(BehaviorAction.SLOWER)
        return int(BehaviorAction.IDLE)


def collect_expert_rollouts(
    n_episodes: int = 50,
    seed: int = 42,
) -> list[dict]:
    """
    Roll out the IDM expert and collect (obs, action) transitions.

    Returns a list of episode dicts:
        {"observations": np.ndarray (T, obs_dim),
         "actions":      np.ndarray (T,)}
    """
    env = make_env(seed=seed)
    expert = IDMExpert(env)
    rollouts = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        episode_obs, episode_acts = [], []
        done = False

        while not done:
            action = expert.act(obs)
            episode_obs.append(obs)
            episode_acts.append(action)
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

        rollouts.append({
            "observations": np.array(episode_obs, dtype=np.float32),
            "actions": np.array(episode_acts, dtype=np.int64),
        })

    env.close()
    return rollouts
