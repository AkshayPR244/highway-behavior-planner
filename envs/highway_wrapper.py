"""
Thin wrapper around highway-env's highway-v0 that:
- Enforces a consistent observation/action space for all policies
- Exposes raw vehicle state needed by the IDM expert and safety checker
- Disables rendering by default (headless training)
"""
import gymnasium as gym
import numpy as np
from gymnasium.wrappers.flatten_observation import FlattenObservation

from config.settings import N_ACTIONS, OBS_FEATURES, OBS_VEHICLES, build_env_config


ENV_CONFIG = build_env_config()


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
    return N_ACTIONS
