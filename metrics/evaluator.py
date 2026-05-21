"""
Metric suite for evaluating any policy on highway-v0.

Metrics collected per episode, then aggregated across episodes:

  collision_rate      — fraction of episodes ending in collision (safety)
  mean_min_ttc        — mean of per-episode minimum TTC in seconds (safety margin)
  rms_jerk            — RMS longitudinal jerk in m/s^3 (comfort)
  goal_completion     — fraction of episodes reaching timeout without collision (efficiency)
  fallback_rate       — fraction of steps where the safety wrapper overrode the policy
  lc_frequency        — lane-change initiations per step (lateral activity)
  lc_completion_rate  — fraction of initiated lane changes where ego reached the new lane
  lc_anticipatory_frac— fraction of LCs issued while closing on a slow vehicle with TTC > threshold
                        (anticipatory = acting before forced; reactive = acting under pressure)
  ego_fault_rate      — fraction of episodes where a collision was caused by ego's action
  npc_fault_rate      — fraction of episodes where a collision was unavoidable (NPC at fault)

Usage:
    from metrics.evaluator import evaluate
    from policies.idm_expert import IDMExpert
    from envs.highway_wrapper import make_env

    env = make_env()
    results = evaluate(policy=IDMExpert(env), env=env, n_episodes=20)
    print_table(results)
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Protocol

import gymnasium as gym

from metrics.fault_attribution import snapshot_pre_step, classify_fault

# Threshold in seconds: a lane change issued while closing on a front vehicle
# with TTC above this value is classified as anticipatory (acting early).
ANTICIPATORY_TTC_THRESHOLD = 4.0


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class EpisodeStats:
    """Raw per-step data collected during one episode."""
    speeds: list[float] = field(default_factory=list)       # ego speed each step (m/s)
    ttcs: list[float] = field(default_factory=list)         # TTC each step (s), inf if no vehicle ahead
    collision: bool = False
    n_fallbacks: int = 0                                     # steps where safety wrapper fired
    n_steps: int = 0
    n_lc_initiated: int = 0      # steps where policy issued LANE_LEFT or LANE_RIGHT
    n_lc_completed: int = 0      # steps where ego's integer lane number actually changed
    n_lc_anticipatory: int = 0   # LCs issued while closing on front vehicle with TTC > threshold
    fault: str = "none"           # "none" | "ego" | "npc" | "ambiguous" (set on crash)


@dataclass
class EvalResults:
    """Aggregated metrics across all evaluation episodes."""
    collision_rate: float       # [0, 1]
    mean_min_ttc: float         # seconds; higher is safer
    rms_jerk: float             # m/s^3; lower is more comfortable
    goal_completion: float      # [0, 1]
    fallback_rate: float         # [0, 1]
    lc_frequency: float          # lane-change initiations per step
    lc_completion_rate: float    # [0, 1]; NaN if no lane changes attempted
    lc_anticipatory_frac: float  # [0, 1]; NaN if no lane changes attempted
    ego_fault_rate: float        # ego-caused collisions / n_episodes
    npc_fault_rate: float        # unavoidable collisions / n_episodes
    n_episodes: int
    n_steps_total: int


# ---------------------------------------------------------------------------
# Policy protocol — anything with an .act(obs) -> int method works
# ---------------------------------------------------------------------------

class Policy(Protocol):
    def act(self, obs: np.ndarray) -> int: ...


# ---------------------------------------------------------------------------
# Per-step metric helpers
# ---------------------------------------------------------------------------

def _approach_state(env: gym.Env) -> tuple[bool, float]:
    """
    Return (is_closing, ttc) for the front vehicle in ego's lane.

    is_closing is True when the ego is faster than the vehicle ahead
    (i.e., the gap is shrinking).  ttc is the time-to-collision in seconds
    (np.inf when the lane is clear or ego is not closing).

    Used to classify lane-change initiations as anticipatory vs reactive:
      - anticipatory: is_closing=True  AND  ttc > ANTICIPATORY_TTC_THRESHOLD
      - reactive:     is_closing=True  AND  ttc <= ANTICIPATORY_TTC_THRESHOLD
      - unpressured:  is_closing=False (no threat; neither category)
    """
    road = env.unwrapped.road
    ego = env.unwrapped.vehicle
    try:
        neighbours = road.neighbour_vehicles(ego)
        front = neighbours[0] if neighbours else None
        if front is None:
            return False, np.inf
        gap = ego.lane_distance_to(front)
        closing_speed = ego.speed - front.speed
        if closing_speed <= 0 or gap <= 0:
            return False, np.inf
        return True, gap / closing_speed
    except Exception:
        return False, np.inf


def _compute_ttc(env: gym.Env) -> float:
    """
    Compute time-to-collision with the front vehicle in ego's lane.

    TTC = gap / closing_speed, where closing_speed = ego_speed - front_speed.
    Only meaningful when ego is faster than the vehicle ahead (closing gap).
    Returns np.inf when the lane is clear or ego is slower than the leader.
    """
    road = env.unwrapped.road
    ego = env.unwrapped.vehicle

    try:
        neighbours = road.neighbour_vehicles(ego)
        front = neighbours[0] if neighbours else None
        if front is None:
            return np.inf
        gap = ego.lane_distance_to(front)
        closing_speed = ego.speed - front.speed
        if closing_speed <= 0 or gap <= 0:
            return np.inf
        return gap / closing_speed
    except Exception:
        return np.inf


def _compute_jerk(speeds: list[float], dt: float) -> float:
    """
    Compute RMS longitudinal jerk from a speed time-series.

    jerk_t ≈ (v_{t+1} - 2*v_t + v_{t-1}) / dt^2  (second difference of speed)
    RMS over all valid timesteps in the episode.
    """
    if len(speeds) < 3:
        return 0.0
    v = np.array(speeds)
    # Second-order finite difference approximation of jerk
    jerk = (v[2:] - 2 * v[1:-1] + v[:-2]) / (dt ** 2)
    return float(np.sqrt(np.mean(jerk ** 2)))


# ---------------------------------------------------------------------------
# Core evaluation loop
# ---------------------------------------------------------------------------

def evaluate(
    policy: Policy,
    env: gym.Env,
    n_episodes: int = 20,
    seed: int = 0,
) -> EvalResults:
    """
    Roll out `policy` in `env` for `n_episodes` and return aggregated metrics.

    The policy must expose:  action = policy.act(obs: np.ndarray) -> int
    An optional safety wrapper is supported: if the wrapper sets
    info["fallback"] = True on a step, that step is counted as a fallback.
    """
    sim_freq = env.unwrapped.config.get("simulation_frequency", 15)
    policy_freq = env.unwrapped.config.get("policy_frequency", 1)
    dt = 1.0 / policy_freq   # seconds per policy step

    all_episode_stats: list[EpisodeStats] = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        stats = EpisodeStats()
        done = False

        prev_lane: int | None = None

        while not done:
            action = policy.act(obs)

            # Lane-change initiation: LANE_LEFT=0, LANE_RIGHT=2
            if action in (0, 2):
                stats.n_lc_initiated += 1
                # Classify as anticipatory if closing on front vehicle with comfortable TTC
                is_closing, ttc = _approach_state(env)
                if is_closing and ttc > ANTICIPATORY_TTC_THRESHOLD:
                    stats.n_lc_anticipatory += 1

            # Snapshot state BEFORE the step for counterfactual fault analysis
            pre_step = snapshot_pre_step(env)

            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Speed and TTC
            ego_speed = env.unwrapped.vehicle.speed
            stats.speeds.append(ego_speed)
            stats.ttcs.append(_compute_ttc(env))
            stats.n_steps += 1

            # Lane-change completion: did ego's integer lane number change?
            current_lane: int = env.unwrapped.vehicle.lane_index[2]
            if prev_lane is not None and current_lane != prev_lane:
                stats.n_lc_completed += 1
            prev_lane = current_lane

            # Fallback tracking (populated by safety wrapper in Phase 1+)
            if info.get("fallback", False):
                stats.n_fallbacks += 1

        stats.collision = env.unwrapped.vehicle.crashed

        # Fault attribution: classify who caused the collision
        if stats.collision:
            stats.fault = classify_fault(pre_step, dt)

        all_episode_stats.append(stats)

    # --- Aggregate ---
    n_collisions = sum(s.collision for s in all_episode_stats)
    n_total_steps = sum(s.n_steps for s in all_episode_stats)
    n_total_fallbacks = sum(s.n_fallbacks for s in all_episode_stats)
    n_total_lc_initiated = sum(s.n_lc_initiated for s in all_episode_stats)
    n_total_lc_completed = sum(s.n_lc_completed for s in all_episode_stats)
    n_total_lc_anticipatory = sum(s.n_lc_anticipatory for s in all_episode_stats)

    min_ttcs = [min(s.ttcs) for s in all_episode_stats if s.ttcs]
    rms_jerks = [_compute_jerk(s.speeds, dt) for s in all_episode_stats]

    _nan = float("nan")
    lc_completion_rate = (
        n_total_lc_completed / n_total_lc_initiated
        if n_total_lc_initiated > 0 else _nan
    )
    lc_anticipatory_frac = (
        n_total_lc_anticipatory / n_total_lc_initiated
        if n_total_lc_initiated > 0 else _nan
    )
    n_ego_fault = sum(1 for s in all_episode_stats if s.fault == "ego")
    n_npc_fault = sum(1 for s in all_episode_stats if s.fault == "npc")

    return EvalResults(
        collision_rate=n_collisions / n_episodes,
        mean_min_ttc=float(np.mean(min_ttcs)) if min_ttcs else np.inf,
        rms_jerk=float(np.mean(rms_jerks)),
        goal_completion=1.0 - (n_collisions / n_episodes),
        fallback_rate=n_total_fallbacks / n_total_steps if n_total_steps > 0 else 0.0,
        lc_frequency=n_total_lc_initiated / n_total_steps if n_total_steps > 0 else 0.0,
        lc_completion_rate=lc_completion_rate,
        lc_anticipatory_frac=lc_anticipatory_frac,
        ego_fault_rate=n_ego_fault / n_episodes,
        npc_fault_rate=n_npc_fault / n_episodes,
        n_episodes=n_episodes,
        n_steps_total=n_total_steps,
    )


# ---------------------------------------------------------------------------
# Pretty-print
# ---------------------------------------------------------------------------

def print_table(results: EvalResults, label: str = "Policy") -> None:
    """Print a formatted metric table to stdout."""
    border = "+" + "-" * 30 + "+" + "-" * 12 + "+"
    print(border)
    print(f"| {'Metric':<28} | {'Value':>10} |")
    print(border)
    def _fmt(v: float) -> str:
        return "N/A" if v != v else f"{v:.3f}"  # NaN check via reflexivity

    rows = [
        ("Policy",               label),
        ("Episodes",             str(results.n_episodes)),
        ("Collision rate",       f"{results.collision_rate:.3f}"),
        ("Goal completion",      f"{results.goal_completion:.3f}"),
        ("Mean min TTC (s)",     f"{results.mean_min_ttc:.2f}"),
        ("RMS jerk (m/s³)",      f"{results.rms_jerk:.3f}"),
        ("Fallback rate",        f"{results.fallback_rate:.3f}"),
        ("LC frequency (/step)", f"{results.lc_frequency:.3f}"),
        ("LC completion rate",   _fmt(results.lc_completion_rate)),
        ("LC anticipatory frac", _fmt(results.lc_anticipatory_frac)),
        ("Ego fault rate",       f"{results.ego_fault_rate:.3f}"),
        ("NPC fault rate",       f"{results.npc_fault_rate:.3f}"),
    ]
    for name, val in rows:
        print(f"| {name:<28} | {val:>10} |")
    print(border)
