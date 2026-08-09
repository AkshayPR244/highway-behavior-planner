"""
Adversarial scenario library for highway-v0.

Each scenario places the ego vehicle into a specific challenging initial state
by modifying NPC positions and speeds immediately after env.reset().  The
policy under test must then respond over the subsequent episode using only the
information available from its observations.

Why adversarial scenarios?
--------------------------
Aggregate metrics over random seeds are necessary but not sufficient.  A
policy can achieve 0% collision rate on average yet fail catastrophically on
specific situations (cut-ins, sudden braking, dense merges).  Named scenarios
make failure modes explicit and reproducible — the same methodology used in
safety case arguments at AV companies.

Scenarios
---------
  sudden_brake    : NPC directly ahead, 20 m gap, 10 m/s slow (TTC ≈ 1.3 s).
                    Tests emergency deceleration or lane-escape response.

  close_merge     : NPC in adjacent lane at ego's exact longitudinal position.
                    A lane change toward that side is immediately unsafe.
                    Tests whether the safety wrapper blocks the manoeuvre.

  aggressive_rear : NPC 5 m behind at ego_speed + 10 m/s (rear TTC ≈ 0.5 s).
                    Tests rearward situational awareness and forward escape.

  dense_corridor  : Three NPCs at 15 / 30 / 45 m ahead, all at 20 m/s.
                    Ego at 25 m/s is closing on all three simultaneously.
                    Tests multi-vehicle gap management.

Methodology
-----------
1. env.reset(seed) — highway-env places NPCs normally.
2. scenario.setup(env) — overwrite the positions/speeds of selected NPCs.
3. policy rollout — evaluate as usual; metrics are collected per-episode.

The first observation the policy receives is from env.reset() (before setup),
so the policy has at most 1 policy step of lag before encountering the
adversarial state.  At 1 Hz this is 1 second — still well within the reaction
window for sudden_brake (1.3 s) and dense_corridor (> 3 s).

Usage
-----
    from envs.highway_wrapper import make_env
    from policies.idm_expert import IDMExpert
    from scenarios.adversarial import run_all_scenarios, print_scenario_comparison

    env = make_env()
    policy = IDMExpert(env)
    results = run_all_scenarios(policy, env, n_episodes=10)
    print_scenario_comparison(results)
"""
from __future__ import annotations

import numpy as np
import gymnasium as gym
from dataclasses import dataclass
from typing import Callable

from metrics.evaluator import (
    EpisodeStats,
    EvalResults,
    _compute_ttc,
    _compute_jerk,
    _approach_state,
    ANTICIPATORY_TTC_THRESHOLD,
)
from metrics.fault_attribution import snapshot_pre_step, classify_fault

# highway-env default lane width (metres).  Lanes are parallel horizontal
# strips; adjacent lane center is exactly LANE_WIDTH away in y.
LANE_WIDTH = 4.0


# ---------------------------------------------------------------------------
# Vehicle placement helpers
# ---------------------------------------------------------------------------

def _get_npcs(env: gym.Env) -> list:
    """Return all vehicles on the road except the ego."""
    road = env.unwrapped.road
    ego  = env.unwrapped.vehicle
    return [v for v in road.vehicles if v is not ego]


def _place_vehicle(
    env: gym.Env,
    vehicle,
    lane_offset: int,   # lanes relative to ego: -1 = left, 0 = same, +1 = right
    x_offset: float,    # metres ahead (+) or behind (-) of ego
    speed: float,
) -> None:
    """
    Move `vehicle` to a position relative to the ego vehicle.

    Updates position, speed, heading, and lane_index so that downstream
    road.neighbour_vehicles() lookups find the vehicle in the correct lane.

    Parameters
    ----------
    lane_offset : int
        -1  → one lane to the left of ego
         0  → same lane as ego
        +1  → one lane to the right of ego
    x_offset : float
        Positive = ahead of ego, negative = behind.
    speed : float
        Absolute speed in m/s for the placed vehicle.
    """
    ego     = env.unwrapped.vehicle
    n_lanes = env.unwrapped.config.get("lanes_count", 3)

    from_node, to_node, ego_lane = ego.lane_index
    target_lane = int(np.clip(ego_lane + lane_offset, 0, n_lanes - 1))

    vehicle.position = np.array(
        [ego.position[0] + x_offset,
         ego.position[1] + lane_offset * LANE_WIDTH],
        dtype=float,
    )
    vehicle.speed    = float(speed)
    vehicle.heading  = float(ego.heading)   # straight highway → same heading
    vehicle.lane_index = (from_node, to_node, target_lane)


def _park_far(env: gym.Env, vehicles: list, start_x_offset: float = 200.0) -> None:
    """
    Move `vehicles` far ahead of ego so they don't interfere with the scenario.
    Each vehicle is placed 20 m further than the previous one.
    """
    ego = env.unwrapped.vehicle
    for i, v in enumerate(vehicles):
        _place_vehicle(env, v, lane_offset=0,
                       x_offset=start_x_offset + i * 20.0,
                       speed=ego.speed)


# ---------------------------------------------------------------------------
# Setup functions
# ---------------------------------------------------------------------------

def _setup_sudden_brake(env: gym.Env) -> None:
    """
    NPC directly ahead in ego's lane, 20 m away, crawling at 10 m/s.

    At ego's typical cruise speed of 25 m/s the closing speed is 15 m/s,
    giving TTC = 20 / 15 ≈ 1.3 s.  The policy must brake hard or escape
    into an adjacent lane within 1–2 steps.
    """
    npcs = _get_npcs(env)
    if not npcs:
        return
    _place_vehicle(env, npcs[0], lane_offset=0, x_offset=20.0, speed=10.0)
    _park_far(env, npcs[1:])


def _setup_close_merge(env: gym.Env) -> None:
    """
    NPC in the adjacent left lane (or right if ego is in lane 0),
    at ego's exact longitudinal position, matching ego's speed.

    From the safety wrapper's perspective, a LANE_LEFT action would place
    the ego directly on top of this NPC.  The wrapper must block it and
    the fallback_rate should spike.
    """
    npcs = _get_npcs(env)
    if not npcs:
        return
    ego         = env.unwrapped.vehicle
    lane_offset = -1 if ego.lane_index[2] > 0 else 1
    _place_vehicle(env, npcs[0], lane_offset=lane_offset, x_offset=0.0, speed=ego.speed)
    _park_far(env, npcs[1:])


def _setup_aggressive_rear(env: gym.Env) -> None:
    """
    NPC 5 m behind ego, travelling 10 m/s faster.

    Rear TTC = 5 / 10 = 0.5 s.  The ego cannot brake (that reduces the gap
    further).  It must accelerate or change lanes to escape.  Tests whether
    the policy can respond to rear threats and whether the safety wrapper
    (which checks rear gaps on lane changes) allows the escape.
    """
    npcs = _get_npcs(env)
    if not npcs:
        return
    ego = env.unwrapped.vehicle
    _place_vehicle(env, npcs[0], lane_offset=0, x_offset=-5.0, speed=ego.speed + 10.0)
    _park_far(env, npcs[1:])


def _setup_dense_corridor(env: gym.Env) -> None:
    """
    Three NPCs at 15, 30, and 45 m ahead of ego, all at 20 m/s.

    Ego at 25 m/s is closing on all three simultaneously.
    TTC values: 15/5 = 3 s, 30/5 = 6 s, 45/5 = 9 s.
    The policy must change lanes early to avoid the first vehicle before
    the gap closes completely.
    """
    npcs = _get_npcs(env)
    ego  = env.unwrapped.vehicle
    convoy_x = [15.0, 30.0, 45.0]
    for i, x in enumerate(convoy_x):
        if i < len(npcs):
            _place_vehicle(env, npcs[i], lane_offset=0, x_offset=x, speed=20.0)
    _park_far(env, npcs[len(convoy_x):])


# ---------------------------------------------------------------------------
# Scenario registry
# ---------------------------------------------------------------------------

@dataclass
class Scenario:
    """Named adversarial scenario with a setup function."""
    name:        str
    description: str
    setup:       Callable[[gym.Env], None]


SCENARIOS: list[Scenario] = [
    Scenario(
        name        = "sudden_brake",
        description = "NPC 20m ahead, 10 m/s slow — TTC ≈ 1.3s",
        setup       = _setup_sudden_brake,
    ),
    Scenario(
        name        = "close_merge",
        description = "NPC in adjacent lane at ego's position — unsafe LC",
        setup       = _setup_close_merge,
    ),
    Scenario(
        name        = "aggressive_rear",
        description = "NPC 5m behind at ego+10 m/s — rear TTC ≈ 0.5s",
        setup       = _setup_aggressive_rear,
    ),
    Scenario(
        name        = "dense_corridor",
        description = "3 NPCs at 15/30/45m, all at 20 m/s — closing convoy",
        setup       = _setup_dense_corridor,
    ),
]


# ---------------------------------------------------------------------------
# Scenario runner
# ---------------------------------------------------------------------------

def run_scenario(
    scenario:    Scenario,
    policy,
    env:         gym.Env,
    n_episodes:  int = 10,
    seed:        int = 0,
) -> EvalResults:
    """
    Evaluate `policy` on `scenario` for `n_episodes`.

    Workflow per episode:
        env.reset(seed) → scenario.setup(env) → policy rollout → collect metrics

    Returns the same EvalResults structure as metrics.evaluator.evaluate(),
    so results are directly comparable with the baseline table.
    """
    dt         = 1.0 / env.unwrapped.config.get("policy_frequency", 1)
    max_steps  = env.unwrapped.config.get("duration", 40)

    all_stats: list[EpisodeStats] = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        scenario.setup(env)          # ← adversarial NPC placement

        stats    = EpisodeStats()
        done     = False
        prev_lane: int | None = None
        pre_step = snapshot_pre_step(env)

        while not done and stats.n_steps < max_steps:
            action = policy.act(obs)

            if action in (0, 2):
                stats.n_lc_initiated += 1
                is_closing, ttc = _approach_state(env)
                if is_closing and ttc > ANTICIPATORY_TTC_THRESHOLD:
                    stats.n_lc_anticipatory += 1

            pre_step = snapshot_pre_step(env)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            stats.speeds.append(env.unwrapped.vehicle.speed)
            stats.ttcs.append(_compute_ttc(env))
            stats.n_steps += 1

            current_lane = env.unwrapped.vehicle.lane_index[2]
            if prev_lane is not None and current_lane != prev_lane:
                stats.n_lc_completed += 1
            prev_lane = current_lane

            if info.get("fallback", False):
                stats.n_fallbacks += 1

        stats.collision = env.unwrapped.vehicle.crashed
        if stats.collision:
            stats.fault = classify_fault(pre_step, dt)

        all_stats.append(stats)

    # Aggregate — identical arithmetic to metrics.evaluator.evaluate()
    n_col   = sum(s.collision for s in all_stats)
    n_steps = sum(s.n_steps   for s in all_stats)
    n_fb    = sum(s.n_fallbacks      for s in all_stats)
    n_lci   = sum(s.n_lc_initiated   for s in all_stats)
    n_lcc   = sum(s.n_lc_completed   for s in all_stats)
    n_lca   = sum(s.n_lc_anticipatory for s in all_stats)
    n_ego   = sum(1 for s in all_stats if s.fault == "ego")
    n_npc   = sum(1 for s in all_stats if s.fault == "npc")

    nan      = float("nan")
    min_ttcs = [min(s.ttcs) for s in all_stats if s.ttcs]
    rms_jerks= [_compute_jerk(s.speeds, dt) for s in all_stats]
    episode_speeds = [float(np.mean(s.speeds)) for s in all_stats if s.speeds]
    episode_distances = [float(np.sum(s.speeds) * dt) for s in all_stats]
    episode_progress = [float(s.speeds[-1] * s.n_steps * dt) if s.speeds else 0.0 for s in all_stats]
    finite_ttcs = [ttc for s in all_stats for ttc in s.ttcs if np.isfinite(ttc)]
    survival_rate = 1.0 - (n_col / n_episodes)
    success_rate = sum(
        1.0
        for s in all_stats
        if not s.collision and float(np.mean(s.speeds)) >= 5.0 and float(np.sum(s.speeds) * dt) >= 300.0
    ) / n_episodes if n_episodes else 0.0

    return EvalResults(
        collision_rate     = n_col / n_episodes,
        mean_min_ttc       = float(np.mean(min_ttcs)) if min_ttcs else np.inf,
        rms_jerk           = float(np.mean(rms_jerks)),
        survival_rate      = survival_rate,
        success_rate       = success_rate,
        fallback_rate      = n_fb  / n_steps if n_steps else 0.0,
        lc_frequency       = n_lci / n_steps if n_steps else 0.0,
        lc_completion_rate = n_lcc / n_lci   if n_lci   else nan,
        lc_anticipatory_frac = n_lca / n_lci if n_lci   else nan,
        ego_fault_rate     = n_ego / n_episodes,
        npc_fault_rate     = n_npc / n_episodes,
        mean_distance_travelled = float(np.mean(episode_distances)) if episode_distances else 0.0,
        mean_longitudinal_progress = float(np.mean(episode_progress)) if episode_progress else 0.0,
        mean_speed = float(np.mean(episode_speeds)) if episode_speeds else 0.0,
        mean_episode_duration = float(np.mean([s.n_steps * dt for s in all_stats])) if all_stats else 0.0,
        ttc_below_1s_frac = float(np.mean([ttc < 1.0 for ttc in finite_ttcs])) if finite_ttcs else 0.0,
        ttc_below_2s_frac = float(np.mean([ttc < 2.0 for ttc in finite_ttcs])) if finite_ttcs else 0.0,
        ttc_below_4s_frac = float(np.mean([ttc < 4.0 for ttc in finite_ttcs])) if finite_ttcs else 0.0,
        ttc_p5_finite = float(np.percentile(finite_ttcs, 5.0)) if finite_ttcs else np.inf,
        n_episodes         = n_episodes,
        n_steps_total      = n_steps,
    )


def run_all_scenarios(
    policy,
    env:        gym.Env,
    n_episodes: int = 10,
    seed:       int = 0,
) -> dict[str, EvalResults]:
    """Run every scenario in SCENARIOS and return results keyed by name."""
    return {
        s.name: run_scenario(s, policy, env, n_episodes=n_episodes, seed=seed)
        for s in SCENARIOS
    }


# ---------------------------------------------------------------------------
# Pretty-print
# ---------------------------------------------------------------------------

def print_scenario_comparison(results: dict[str, EvalResults]) -> None:
    """
    Print a side-by-side metric table comparing all scenarios.

    Example output (4 scenarios):

      | Metric                 | sudden_brake    | close_merge     | ...  |
      +---------+------+...
      | Collision rate         | 0.500           | 0.100           | ...  |
    """
    scenario_names = list(results.keys())
    col_w    = 16
    metric_w = 24

    def _fmt(v: float) -> str:
        return "N/A" if v != v else f"{v:.3f}"

    metric_rows: list[tuple[str, Callable]] = [
        ("Collision rate",        lambda r: f"{r.collision_rate:.3f}"),
        ("Survival rate",         lambda r: f"{r.survival_rate:.3f}"),
        ("Success rate",          lambda r: f"{r.success_rate:.3f}"),
        ("Mean min TTC (s)",      lambda r: f"{r.mean_min_ttc:.2f}"),
        ("RMS jerk (m/s³)",       lambda r: f"{r.rms_jerk:.3f}"),
        ("Fallback rate",         lambda r: f"{r.fallback_rate:.3f}"),
        ("LC frequency (/step)",  lambda r: f"{r.lc_frequency:.3f}"),
        ("Ego fault rate",        lambda r: f"{r.ego_fault_rate:.3f}"),
        ("NPC fault rate",        lambda r: f"{r.npc_fault_rate:.3f}"),
    ]

    sep = (
        "+" + "-" * (metric_w + 2)
        + ("+" + "-" * (col_w + 2)) * len(scenario_names)
        + "+"
    )
    def _row(label, values):
        return (
            f"| {label:<{metric_w}} |"
            + "".join(f" {v:<{col_w}} |" for v in values)
        )

    print(sep)
    print(_row("Scenario", [s[:col_w] for s in scenario_names]))
    print(sep)
    for name, fn in metric_rows:
        print(_row(name, [fn(results[s]) for s in scenario_names]))
    print(sep)
