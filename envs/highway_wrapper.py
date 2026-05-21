"""
Thin wrapper around highway-env's highway-v0 that:
- Enforces a consistent observation/action space for all policies
- Exposes raw vehicle state needed by the IDM expert and safety checker
- Disables rendering by default (headless training)
"""
import gymnasium as gym
import numpy as np
from gymnasium.wrappers import FlattenObservation


# Observation: 5 vehicles x 5 features [presence, x, y, vx, vy] — normalized
OBS_FEATURES = ["presence", "x", "y", "vx", "vy"]
OBS_VEHICLES = 5

ENV_CONFIG = {
    "observation": {
        "type": "Kinematics",
        "vehicles_count": OBS_VEHICLES,
        "features": OBS_FEATURES,
        "normalize": True,
        "absolute": False,  # relative to ego
    },
    "action": {
        "type": "DiscreteMetaAction",
    },
    "lanes_count": 3,
    "vehicles_count": 10,
    "duration": 40,          # seconds per episode
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
    "offscreen_rendering": True,  # no display required
}


def make_env(render_mode: str | None = None, seed: int | None = None) -> gym.Env:
    """Create a configured highway-v0 environment with flat observations."""
    env = gym.make(
        "highway-v0",
        render_mode=render_mode,
        config=ENV_CONFIG,
    )
    env = FlattenObservation(env)
    if seed is not None:
        env.reset(seed=seed)
    return env


def obs_shape() -> tuple[int, ...]:
    """Return the flat observation shape."""
    return (OBS_VEHICLES * len(OBS_FEATURES),)


def n_actions() -> int:
    """Return the number of discrete actions."""
    env = make_env()
    n = env.action_space.n
    env.close()
    return n
