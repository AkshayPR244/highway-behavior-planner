"""Shared constants and enums used across the planning stack."""
from __future__ import annotations

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
MIN_GAP = 4.0
SAFETY_HORIZON = 6
ANTICIPATORY_TTC_THRESHOLD = 4.0

# IDM + MOBIL defaults
IDM_DESIRED_SPEED = 25.0
IDM_MIN_SPACING = 5.0
IDM_DESIRED_HEADWAY = 1.5
IDM_ACCEL_COMFORT = 3.0
IDM_DECEL_COMFORT = 3.0
MOBIL_B_SAFE = 4.0
MOBIL_POLITENESS = 0.2
MOBIL_ACCEL_GAIN_THRESHOLD = 0.1
