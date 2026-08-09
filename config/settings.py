"""Shared constants, enums, and typed configs used across the planning stack."""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class BehaviorAction(IntEnum):
    LANE_LEFT = 0
    IDLE = 1
    LANE_RIGHT = 2
    FASTER = 3
    SLOWER = 4


BEHAVIOR_ACTION_NAMES = {
    BehaviorAction.LANE_LEFT: "LANE_LEFT",
    BehaviorAction.IDLE: "IDLE",
    BehaviorAction.LANE_RIGHT: "LANE_RIGHT",
    BehaviorAction.FASTER: "FASTER",
    BehaviorAction.SLOWER: "SLOWER",
}

# Env defaults
OBS_FEATURES = ["presence", "x", "y", "vx", "vy"]
OBS_VEHICLES = 5
N_ACTIONS = len(BehaviorAction)


def build_env_config() -> dict:
    """Return the default highway-v0 config used by this project."""
    return {
        "observation": {
            "type": "Kinematics",
            "vehicles_count": OBS_VEHICLES,
            "features": OBS_FEATURES,
            "normalize": True,
            "absolute": False,
        },
        "action": {
            "type": "DiscreteMetaAction",
        },
        "lanes_count": 3,
        "vehicles_count": 10,
        "duration": 40,
        "initial_spacing": 2,
        "collision_reward": -1,
        "reward_speed_range": [20, 30],
        "simulation_frequency": 15,
        "policy_frequency": 1,
        "screen_width": 600,
        "screen_height": 150,
        "centering_position": [0.3, 0.5],
        "scaling": 5.5,
        "show_trajectories": False,
        "render_agent": True,
        "offscreen_rendering": True,
    }


# Safety / eval defaults
MIN_FRONT_GAP_M = 4.0
MIN_REAR_GAP_M = 4.0
SAFETY_HORIZON = 6
SAFETY_PREDICTION_HORIZON_S = 6.0
SAFETY_PREDICTION_DT_S = 1.0
ANTICIPATORY_TTC_THRESHOLD = 4.0
EGO_MAX_ACCELERATION_MPS2 = 2.0
EGO_EXPECTED_DECELERATION_MPS2 = 2.5
FRONT_VEHICLE_MAX_BRAKING_MPS2 = 4.0
REAR_VEHICLE_MAX_ACCELERATION_MPS2 = 2.0


@dataclass(frozen=True)
class SafetyConfig:
    prediction_horizon_s: float = SAFETY_PREDICTION_HORIZON_S
    prediction_dt_s: float = SAFETY_PREDICTION_DT_S
    minimum_front_gap_m: float = MIN_FRONT_GAP_M
    minimum_rear_gap_m: float = MIN_REAR_GAP_M
    front_vehicle_max_braking_mps2: float = FRONT_VEHICLE_MAX_BRAKING_MPS2
    rear_vehicle_max_acceleration_mps2: float = REAR_VEHICLE_MAX_ACCELERATION_MPS2
    ego_max_acceleration_mps2: float = EGO_MAX_ACCELERATION_MPS2
    ego_expected_deceleration_mps2: float = EGO_EXPECTED_DECELERATION_MPS2


@dataclass(frozen=True)
class EvaluationConfig:
    min_success_progress_m: float = 300.0
    min_success_mean_speed_mps: float = 5.0
    ttc_thresholds_s: tuple[float, ...] = (1.0, 2.0, 4.0)
    bootstrap_samples: int = 2000
    bootstrap_seed: int = 0

# IDM + MOBIL defaults
IDM_DESIRED_SPEED = 25.0
IDM_MIN_SPACING = 5.0
IDM_DESIRED_HEADWAY = 1.5
IDM_ACCEL_COMFORT = 3.0
IDM_DECEL_COMFORT = 3.0
MOBIL_B_SAFE = 4.0
MOBIL_POLITENESS = 0.2
MOBIL_ACCEL_GAIN_THRESHOLD = 0.1
