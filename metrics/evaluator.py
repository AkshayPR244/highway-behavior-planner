"""
Metric suite for evaluating any policy on highway-v0.

Metrics collected per episode, then aggregated across episodes:

    collision_rate      — fraction of episodes ending in collision (safety)
    mean_min_ttc        — mean of per-episode minimum TTC in seconds (safety margin)
    rms_jerk            — RMS longitudinal jerk in m/s^3 (comfort)
    survival_rate       — fraction of episodes reaching timeout without collision
    success_rate        — fraction of episodes that survive and make adequate progress
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
from typing import Protocol

import gymnasium as gym

from config.settings import ANTICIPATORY_TTC_THRESHOLD, EvaluationConfig
from metrics.fault_attribution import snapshot_pre_step, classify_fault


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class EpisodeStats:
    """Raw per-step data collected during one episode."""
    speeds: list[float] = field(default_factory=list)       # ego speed each step (m/s)
    ttcs: list[float] = field(default_factory=list)         # TTC each step (s), inf if no vehicle ahead
    distance_travelled: float = 0.0
    longitudinal_progress: float = 0.0
    duration_s: float = 0.0
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
    survival_rate: float        # [0, 1]
    success_rate: float         # [0, 1]
    fallback_rate: float         # [0, 1]
    lc_frequency: float          # lane-change initiations per step
    lc_completion_rate: float    # [0, 1]; NaN if no lane changes attempted
    lc_anticipatory_frac: float  # [0, 1]; NaN if no lane changes attempted
    ego_fault_rate: float        # ego-caused collisions / n_episodes
    npc_fault_rate: float        # unavoidable collisions / n_episodes
    mean_distance_travelled: float
    mean_longitudinal_progress: float
    mean_speed: float
    mean_episode_duration: float
    ttc_below_1s_frac: float
    ttc_below_2s_frac: float
    ttc_below_4s_frac: float
    ttc_p5_finite: float
    n_episodes: int
    n_steps_total: int
    collision_rate_ci_low: float = float("nan")
    collision_rate_ci_high: float = float("nan")
    mean_speed_ci_low: float = float("nan")
    mean_speed_ci_high: float = float("nan")
    mean_progress_ci_low: float = float("nan")
    mean_progress_ci_high: float = float("nan")
    mean_distance_travelled_ci_low: float = float("nan")
    mean_distance_travelled_ci_high: float = float("nan")

    @property
    def goal_completion(self) -> float:
        return self.survival_rate


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
    except (AttributeError, IndexError, TypeError, ValueError):
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
    except (AttributeError, IndexError, TypeError, ValueError):
        return np.inf


def _wilson_ci(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson score interval for a Bernoulli proportion."""
    if n <= 0:
        return float("nan"), float("nan")
    p = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    margin = (z / denom) * np.sqrt((p * (1.0 - p) / n) + (z2 / (4.0 * n * n)))
    return max(0.0, center - margin), min(1.0, center + margin)


def _bootstrap_mean_ci(values: list[float], n_samples: int, seed: int) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    arr = np.asarray(values, dtype=np.float32)
    if len(arr) == 1:
        return float(arr[0]), float(arr[0])
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(n_samples):
        resampled = rng.choice(arr, size=len(arr), replace=True)
        samples.append(float(np.mean(resampled)))
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


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
    fault_horizon_steps: int = 1,
    config: EvaluationConfig | None = None,
) -> EvalResults:
    """
    Roll out `policy` in `env` for `n_episodes` and return aggregated metrics.

    The policy must expose:  action = policy.act(obs: np.ndarray) -> int
    An optional safety wrapper is supported: if the wrapper sets
    info["fallback"] = True on a step, that step is counted as a fallback.
    """
    cfg = config or EvaluationConfig()
    policy_freq = env.unwrapped.config.get("policy_frequency", 1)
    dt = 1.0 / policy_freq   # seconds per policy step

    all_episode_stats: list[EpisodeStats] = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        stats = EpisodeStats()
        done = False

        prev_lane: int | None = None
        start_position = float(env.unwrapped.vehicle.position[0])

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
            stats.duration_s += dt
            stats.distance_travelled += ego_speed * dt
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
        stats.longitudinal_progress = float(env.unwrapped.vehicle.position[0] - start_position)

        # Fault attribution: classify who caused the collision
        if stats.collision:
            stats.fault = classify_fault(pre_step, dt, horizon_steps=fault_horizon_steps)

        if not stats.collision:
            stats.fault = stats.fault if stats.fault != "none" else "none"

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
    finite_ttcs = [ttc for s in all_episode_stats for ttc in s.ttcs if np.isfinite(ttc)]
    episode_speed_means = [float(np.mean(s.speeds)) for s in all_episode_stats if s.speeds]
    episode_distances = [s.distance_travelled for s in all_episode_stats]
    episode_progress = [s.longitudinal_progress for s in all_episode_stats]
    episode_survival = [0.0 if s.collision else 1.0 for s in all_episode_stats]
    episode_success = [
        1.0 if (not s.collision and s.longitudinal_progress >= cfg.min_success_progress_m and float(np.mean(s.speeds)) >= cfg.min_success_mean_speed_mps) else 0.0
        for s in all_episode_stats
    ]

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
    coll_ci_low, coll_ci_high = _wilson_ci(n_collisions, n_episodes)
    mean_speed_ci_low, mean_speed_ci_high = _bootstrap_mean_ci(episode_speed_means, cfg.bootstrap_samples, cfg.bootstrap_seed)
    mean_progress_ci_low, mean_progress_ci_high = _bootstrap_mean_ci(episode_progress, cfg.bootstrap_samples, cfg.bootstrap_seed)
    mean_distance_ci_low, mean_distance_ci_high = _bootstrap_mean_ci(episode_distances, cfg.bootstrap_samples, cfg.bootstrap_seed)

    return EvalResults(
        collision_rate=n_collisions / n_episodes,
        mean_min_ttc=float(np.mean(min_ttcs)) if min_ttcs else np.inf,
        rms_jerk=float(np.mean(rms_jerks)),
        survival_rate=float(np.mean(episode_survival)) if episode_survival else 0.0,
        success_rate=float(np.mean(episode_success)) if episode_success else 0.0,
        fallback_rate=n_total_fallbacks / n_total_steps if n_total_steps > 0 else 0.0,
        lc_frequency=n_total_lc_initiated / n_total_steps if n_total_steps > 0 else 0.0,
        lc_completion_rate=lc_completion_rate,
        lc_anticipatory_frac=lc_anticipatory_frac,
        ego_fault_rate=n_ego_fault / n_episodes,
        npc_fault_rate=n_npc_fault / n_episodes,
        mean_distance_travelled=float(np.mean(episode_distances)) if episode_distances else 0.0,
        mean_longitudinal_progress=float(np.mean(episode_progress)) if episode_progress else 0.0,
        mean_speed=float(np.mean(episode_speed_means)) if episode_speed_means else 0.0,
        mean_episode_duration=float(np.mean([s.duration_s for s in all_episode_stats])) if all_episode_stats else 0.0,
        ttc_below_1s_frac=float(np.mean([ttc < 1.0 for ttc in finite_ttcs])) if finite_ttcs else 0.0,
        ttc_below_2s_frac=float(np.mean([ttc < 2.0 for ttc in finite_ttcs])) if finite_ttcs else 0.0,
        ttc_below_4s_frac=float(np.mean([ttc < 4.0 for ttc in finite_ttcs])) if finite_ttcs else 0.0,
        ttc_p5_finite=float(np.percentile(finite_ttcs, 5.0)) if finite_ttcs else np.inf,
        n_episodes=n_episodes,
        n_steps_total=n_total_steps,
        collision_rate_ci_low=coll_ci_low,
        collision_rate_ci_high=coll_ci_high,
        mean_speed_ci_low=mean_speed_ci_low,
        mean_speed_ci_high=mean_speed_ci_high,
        mean_progress_ci_low=mean_progress_ci_low,
        mean_progress_ci_high=mean_progress_ci_high,
        mean_distance_travelled_ci_low=mean_distance_ci_low,
        mean_distance_travelled_ci_high=mean_distance_ci_high,
    )


def evaluate_across_seeds(
    policy: Policy,
    env: gym.Env,
    n_seeds: int,
    episodes_per_seed: int,
    base_seed: int = 0,
    fault_horizon_steps: int = 1,
    config: EvaluationConfig | None = None,
) -> EvalResults:
    """
    Evaluate a policy across multiple seeds and aggregate metrics.

    Rate metrics are averaged with sensible denominators:
      - per-episode rates by total episodes
      - per-step rates by total steps
      - lane-change fractions by per-seed mean (ignoring NaNs)
    """
    if n_seeds < 1:
        raise ValueError("n_seeds must be >= 1")
    if episodes_per_seed < 1:
        raise ValueError("episodes_per_seed must be >= 1")

    per_seed: list[EvalResults] = []
    for s in range(n_seeds):
        per_seed.append(
            evaluate(
                policy=policy,
                env=env,
                n_episodes=episodes_per_seed,
                seed=base_seed + s,
                fault_horizon_steps=fault_horizon_steps,
                config=config,
            )
        )

    total_episodes = n_seeds * episodes_per_seed
    total_steps = sum(r.n_steps_total for r in per_seed)

    collision_count = int(round(sum(r.collision_rate * r.n_episodes for r in per_seed)))
    ego_fault_count = int(round(sum(r.ego_fault_rate * r.n_episodes for r in per_seed)))
    npc_fault_count = int(round(sum(r.npc_fault_rate * r.n_episodes for r in per_seed)))

    lc_completion_vals = [r.lc_completion_rate for r in per_seed if r.lc_completion_rate == r.lc_completion_rate]
    lc_anticipatory_vals = [r.lc_anticipatory_frac for r in per_seed if r.lc_anticipatory_frac == r.lc_anticipatory_frac]
    speed_vals = [r.mean_speed for r in per_seed]
    progress_vals = [r.mean_longitudinal_progress for r in per_seed]
    distance_vals = [r.mean_distance_travelled for r in per_seed]

    coll_ci_low, coll_ci_high = _wilson_ci(collision_count, total_episodes)
    cfg = config or EvaluationConfig()
    mean_speed_ci_low, mean_speed_ci_high = _bootstrap_mean_ci(speed_vals, cfg.bootstrap_samples, cfg.bootstrap_seed)
    mean_progress_ci_low, mean_progress_ci_high = _bootstrap_mean_ci(progress_vals, cfg.bootstrap_samples, cfg.bootstrap_seed)
    mean_distance_ci_low, mean_distance_ci_high = _bootstrap_mean_ci(distance_vals, cfg.bootstrap_samples, cfg.bootstrap_seed)

    return EvalResults(
        collision_rate=collision_count / total_episodes,
        mean_min_ttc=float(np.mean([r.mean_min_ttc for r in per_seed])),
        rms_jerk=float(np.mean([r.rms_jerk for r in per_seed])),
        survival_rate=float(np.mean([r.survival_rate for r in per_seed])),
        success_rate=float(np.mean([r.success_rate for r in per_seed])),
        fallback_rate=(
            sum(r.fallback_rate * r.n_steps_total for r in per_seed) / total_steps
            if total_steps > 0 else 0.0
        ),
        lc_frequency=(
            sum(r.lc_frequency * r.n_steps_total for r in per_seed) / total_steps
            if total_steps > 0 else 0.0
        ),
        lc_completion_rate=(
            float(np.mean(lc_completion_vals)) if lc_completion_vals else float("nan")
        ),
        lc_anticipatory_frac=(
            float(np.mean(lc_anticipatory_vals)) if lc_anticipatory_vals else float("nan")
        ),
        ego_fault_rate=ego_fault_count / total_episodes,
        npc_fault_rate=npc_fault_count / total_episodes,
        mean_distance_travelled=float(np.mean(distance_vals)),
        mean_longitudinal_progress=float(np.mean(progress_vals)),
        mean_speed=float(np.mean(speed_vals)),
        mean_episode_duration=float(np.mean([r.mean_episode_duration for r in per_seed])),
        ttc_below_1s_frac=float(np.mean([r.ttc_below_1s_frac for r in per_seed])),
        ttc_below_2s_frac=float(np.mean([r.ttc_below_2s_frac for r in per_seed])),
        ttc_below_4s_frac=float(np.mean([r.ttc_below_4s_frac for r in per_seed])),
        ttc_p5_finite=float(np.mean([r.ttc_p5_finite for r in per_seed])),
        n_episodes=total_episodes,
        n_steps_total=total_steps,
        collision_rate_ci_low=coll_ci_low,
        collision_rate_ci_high=coll_ci_high,
        mean_speed_ci_low=mean_speed_ci_low,
        mean_speed_ci_high=mean_speed_ci_high,
        mean_progress_ci_low=mean_progress_ci_low,
        mean_progress_ci_high=mean_progress_ci_high,
        mean_distance_travelled_ci_low=mean_distance_ci_low,
        mean_distance_travelled_ci_high=mean_distance_ci_high,
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
        (
            "Collision 95% CI",
            _fmt(results.collision_rate_ci_low) + "-" + _fmt(results.collision_rate_ci_high),
        ),
        ("Survival rate",        f"{results.survival_rate:.3f}"),
        ("Success rate",         f"{results.success_rate:.3f}"),
        ("Mean min TTC (s)",     f"{results.mean_min_ttc:.2f}"),
        ("RMS jerk (m/s³)",      f"{results.rms_jerk:.3f}"),
        ("Fallback rate",        f"{results.fallback_rate:.3f}"),
        ("LC frequency (/step)", f"{results.lc_frequency:.3f}"),
        ("LC completion rate",   _fmt(results.lc_completion_rate)),
        ("LC anticipatory frac", _fmt(results.lc_anticipatory_frac)),
        ("Ego fault rate",       f"{results.ego_fault_rate:.3f}"),
        ("NPC fault rate",       f"{results.npc_fault_rate:.3f}"),
        ("Mean speed (m/s)",     f"{results.mean_speed:.3f}"),
        ("Mean progress (m)",    f"{results.mean_longitudinal_progress:.3f}"),
        ("TTC < 1s frac",        f"{results.ttc_below_1s_frac:.3f}"),
        ("TTC < 2s frac",        f"{results.ttc_below_2s_frac:.3f}"),
        ("TTC < 4s frac",        f"{results.ttc_below_4s_frac:.3f}"),
    ]
    for name, val in rows:
        print(f"| {name:<28} | {val:>10} |")
    print(border)
