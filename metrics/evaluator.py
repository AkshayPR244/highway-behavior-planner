"""Metric suite for evaluating policies on highway-v0.

Metrics are collected per episode, then aggregated across episodes.
This module keeps collision, survival, and mission success semantics distinct,
and reports TTC risk as exposure over valid TTC samples.
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

    speeds: list[float] = field(default_factory=list)
    ttcs: list[float] = field(default_factory=list)
    distance_travelled: float = 0.0
    longitudinal_progress: float = 0.0
    duration_s: float = 0.0
    collision: bool = False
    n_fallbacks: int = 0
    n_steps: int = 0
    n_lc_initiated: int = 0
    n_lc_completed: int = 0
    n_lc_anticipatory: int = 0
    fault: str = "none"  # "none" | "ego" | "npc" | "ambiguous"


@dataclass
class EvalResults:
    """Aggregated metrics across all evaluation episodes."""

    collision_rate: float
    mean_min_ttc: float | None
    rms_jerk: float
    survival_rate: float
    success_rate: float
    fallback_rate: float
    lc_frequency: float
    lc_completion_rate: float
    lc_anticipatory_frac: float
    ego_fault_rate: float
    npc_fault_rate: float
    mean_distance_travelled: float
    mean_longitudinal_progress: float
    mean_speed: float
    mean_episode_duration: float

    # Legacy convenience fields derived from configured TTC thresholds.
    ttc_below_1s_frac: float
    ttc_below_2s_frac: float
    ttc_below_4s_frac: float
    # Deprecated alias for finite TTC p05.
    ttc_p5_finite: float | None

    n_episodes: int
    n_steps_total: int

    # Explicit counts for correct multi-seed aggregation and diagnostics.
    n_collisions: int = 0
    n_survivals: int = 0
    n_successes: int = 0

    # TTC exposure internals and robust severity metrics.
    ttc_valid_samples: int = 0
    ttc_exposure: dict[float, float] = field(default_factory=dict)
    ttc_exposure_counts: dict[float, int] = field(default_factory=dict)
    minimum_finite_ttc_s: float | None = None
    finite_ttc_p05_s: float | None = None

    # Wilson 95% CIs for Bernoulli metrics.
    collision_rate_ci_low: float = float("nan")
    collision_rate_ci_high: float = float("nan")
    survival_rate_ci_low: float = float("nan")
    survival_rate_ci_high: float = float("nan")
    success_rate_ci_low: float = float("nan")
    success_rate_ci_high: float = float("nan")

    # Bootstrap 95% CIs for continuous metrics.
    mean_speed_ci_low: float = float("nan")
    mean_speed_ci_high: float = float("nan")
    mean_progress_ci_low: float = float("nan")
    mean_progress_ci_high: float = float("nan")
    mean_distance_travelled_ci_low: float = float("nan")
    mean_distance_travelled_ci_high: float = float("nan")

    @property
    def goal_completion(self) -> float:
        """Deprecated compatibility alias for success_rate.

        Do not use this in new reporting code.
        """
        return self.success_rate


# ---------------------------------------------------------------------------
# Policy protocol
# ---------------------------------------------------------------------------


class Policy(Protocol):
    def act(self, obs: np.ndarray) -> int: ...


# ---------------------------------------------------------------------------
# Per-step metric helpers
# ---------------------------------------------------------------------------


def _approach_state(env: gym.Env) -> tuple[bool, float]:
    """Return (is_closing, ttc) for the front vehicle in ego's lane."""
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
    """Compute TTC with the front vehicle in ego's lane."""
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
    """Wilson score interval for a Bernoulli proportion (95% by default)."""
    if n <= 0:
        return float("nan"), float("nan")
    p = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    margin = (z / denom) * np.sqrt((p * (1.0 - p) / n) + (z2 / (4.0 * n * n)))
    return max(0.0, center - margin), min(1.0, center + margin)


def _bootstrap_mean_ci(values: list[float], n_samples: int, seed: int) -> tuple[float, float]:
    """Percentile bootstrap 95% CI for a mean."""
    if not values:
        return float("nan"), float("nan")
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 1:
        return float(arr[0]), float(arr[0])
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(n_samples):
        resampled = rng.choice(arr, size=len(arr), replace=True)
        samples.append(float(np.mean(resampled)))
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def _compute_jerk(speeds: list[float], dt: float) -> float:
    """Compute RMS longitudinal jerk from a speed series."""
    if len(speeds) < 3:
        return 0.0
    v = np.asarray(speeds, dtype=np.float64)
    jerk = (v[2:] - 2 * v[1:-1] + v[:-2]) / (dt ** 2)
    return float(np.sqrt(np.mean(jerk ** 2)))


def _collect_episode_stats(
    policy: Policy,
    env: gym.Env,
    n_episodes: int,
    seed: int,
    fault_horizon_steps: int,
) -> tuple[list[EpisodeStats], float]:
    """Collect episode-level rollouts and raw statistics."""
    policy_freq = env.unwrapped.config.get("policy_frequency", 1)
    dt = 1.0 / policy_freq

    all_episode_stats: list[EpisodeStats] = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        stats = EpisodeStats()
        done = False

        prev_lane: int | None = None
        start_position = float(env.unwrapped.vehicle.position[0])
        pre_step = snapshot_pre_step(env)

        while not done:
            action = policy.act(obs)

            if action in (0, 2):
                stats.n_lc_initiated += 1
                is_closing, ttc = _approach_state(env)
                if is_closing and ttc > ANTICIPATORY_TTC_THRESHOLD:
                    stats.n_lc_anticipatory += 1

            pre_step = snapshot_pre_step(env)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            ego_speed = float(env.unwrapped.vehicle.speed)
            stats.speeds.append(ego_speed)
            stats.ttcs.append(_compute_ttc(env))
            stats.duration_s += dt
            stats.distance_travelled += ego_speed * dt
            stats.n_steps += 1

            current_lane = int(env.unwrapped.vehicle.lane_index[2])
            if prev_lane is not None and current_lane != prev_lane:
                stats.n_lc_completed += 1
            prev_lane = current_lane

            if info.get("fallback", False):
                stats.n_fallbacks += 1

        stats.collision = bool(env.unwrapped.vehicle.crashed)
        stats.longitudinal_progress = float(env.unwrapped.vehicle.position[0] - start_position)

        if stats.collision:
            stats.fault = classify_fault(pre_step, dt, horizon_steps=fault_horizon_steps)

        all_episode_stats.append(stats)

    return all_episode_stats, dt


def _ttc_exposure_from_samples(
    finite_ttcs: list[float],
    thresholds_s: tuple[float, ...],
) -> tuple[dict[float, float], dict[float, int], int]:
    """Compute TTC exposure rates and raw counts over valid TTC samples.

    Exposure denominator is the number of finite TTC samples only.
    """
    valid = int(len(finite_ttcs))
    thresholds = tuple(sorted({float(t) for t in thresholds_s}))

    if valid == 0:
        return ({t: 0.0 for t in thresholds}, {t: 0 for t in thresholds}, 0)

    arr = np.asarray(finite_ttcs, dtype=np.float64)
    counts = {t: int(np.sum(arr < t)) for t in thresholds}
    exposure = {t: counts[t] / valid for t in thresholds}
    return exposure, counts, valid


def _aggregate_episode_stats(
    all_episode_stats: list[EpisodeStats],
    dt: float,
    cfg: EvaluationConfig,
) -> EvalResults:
    """Aggregate raw per-episode data into evaluation metrics."""
    n_episodes = len(all_episode_stats)

    if n_episodes == 0:
        return EvalResults(
            collision_rate=0.0,
            mean_min_ttc=None,
            rms_jerk=0.0,
            survival_rate=0.0,
            success_rate=0.0,
            fallback_rate=0.0,
            lc_frequency=0.0,
            lc_completion_rate=float("nan"),
            lc_anticipatory_frac=float("nan"),
            ego_fault_rate=0.0,
            npc_fault_rate=0.0,
            mean_distance_travelled=0.0,
            mean_longitudinal_progress=0.0,
            mean_speed=0.0,
            mean_episode_duration=0.0,
            ttc_below_1s_frac=float("nan"),
            ttc_below_2s_frac=float("nan"),
            ttc_below_4s_frac=float("nan"),
            ttc_p5_finite=None,
            n_episodes=0,
            n_steps_total=0,
        )

    n_collisions = sum(1 for s in all_episode_stats if s.collision)
    n_survivals = n_episodes - n_collisions

    n_total_steps = sum(s.n_steps for s in all_episode_stats)
    n_total_fallbacks = sum(s.n_fallbacks for s in all_episode_stats)
    n_total_lc_initiated = sum(s.n_lc_initiated for s in all_episode_stats)
    n_total_lc_completed = sum(s.n_lc_completed for s in all_episode_stats)
    n_total_lc_anticipatory = sum(s.n_lc_anticipatory for s in all_episode_stats)

    episode_speed_means = [float(np.mean(s.speeds)) if s.speeds else 0.0 for s in all_episode_stats]
    episode_distances = [float(s.distance_travelled) for s in all_episode_stats]
    episode_progress = [float(s.longitudinal_progress) for s in all_episode_stats]
    episode_durations = [float(s.duration_s) for s in all_episode_stats]

    episode_success_flags = [
        1
        if (
            (not s.collision)
            and (float(s.longitudinal_progress) >= cfg.min_success_progress_m)
            and ((float(np.mean(s.speeds)) if s.speeds else 0.0) >= cfg.min_success_mean_speed_mps)
        )
        else 0
        for s in all_episode_stats
    ]
    n_successes = int(sum(episode_success_flags))

    rms_jerks = [_compute_jerk(s.speeds, dt) for s in all_episode_stats]

    finite_ttcs = [ttc for s in all_episode_stats for ttc in s.ttcs if np.isfinite(ttc)]
    min_finite_ttc_per_episode: list[float] = []
    for s in all_episode_stats:
        finite_episode = [ttc for ttc in s.ttcs if np.isfinite(ttc)]
        if finite_episode:
            min_finite_ttc_per_episode.append(float(min(finite_episode)))

    ttc_exposure, ttc_exposure_counts, ttc_valid_samples = _ttc_exposure_from_samples(
        finite_ttcs, cfg.ttc_thresholds_s
    )

    minimum_finite_ttc_s = float(np.min(finite_ttcs)) if finite_ttcs else None
    finite_ttc_p05_s = float(np.percentile(finite_ttcs, 5.0)) if finite_ttcs else None
    mean_min_ttc = float(np.mean(min_finite_ttc_per_episode)) if min_finite_ttc_per_episode else None

    _nan = float("nan")
    lc_completion_rate = n_total_lc_completed / n_total_lc_initiated if n_total_lc_initiated > 0 else _nan
    lc_anticipatory_frac = n_total_lc_anticipatory / n_total_lc_initiated if n_total_lc_initiated > 0 else _nan

    n_ego_fault = sum(1 for s in all_episode_stats if s.fault == "ego")
    n_npc_fault = sum(1 for s in all_episode_stats if s.fault == "npc")

    coll_ci_low, coll_ci_high = _wilson_ci(n_collisions, n_episodes)
    surv_ci_low, surv_ci_high = _wilson_ci(n_survivals, n_episodes)
    succ_ci_low, succ_ci_high = _wilson_ci(n_successes, n_episodes)

    mean_speed_ci_low, mean_speed_ci_high = _bootstrap_mean_ci(
        episode_speed_means, cfg.bootstrap_samples, cfg.bootstrap_seed
    )
    mean_progress_ci_low, mean_progress_ci_high = _bootstrap_mean_ci(
        episode_progress, cfg.bootstrap_samples, cfg.bootstrap_seed
    )
    mean_distance_ci_low, mean_distance_ci_high = _bootstrap_mean_ci(
        episode_distances, cfg.bootstrap_samples, cfg.bootstrap_seed
    )

    return EvalResults(
        collision_rate=n_collisions / n_episodes,
        mean_min_ttc=mean_min_ttc,
        rms_jerk=float(np.mean(rms_jerks)),
        survival_rate=n_survivals / n_episodes,
        success_rate=n_successes / n_episodes,
        fallback_rate=n_total_fallbacks / n_total_steps if n_total_steps > 0 else 0.0,
        lc_frequency=n_total_lc_initiated / n_total_steps if n_total_steps > 0 else 0.0,
        lc_completion_rate=lc_completion_rate,
        lc_anticipatory_frac=lc_anticipatory_frac,
        ego_fault_rate=n_ego_fault / n_episodes,
        npc_fault_rate=n_npc_fault / n_episodes,
        mean_distance_travelled=float(np.mean(episode_distances)),
        mean_longitudinal_progress=float(np.mean(episode_progress)),
        mean_speed=float(np.mean(episode_speed_means)),
        mean_episode_duration=float(np.mean(episode_durations)),
        ttc_below_1s_frac=ttc_exposure.get(1.0, _nan),
        ttc_below_2s_frac=ttc_exposure.get(2.0, _nan),
        ttc_below_4s_frac=ttc_exposure.get(4.0, _nan),
        ttc_p5_finite=finite_ttc_p05_s,
        n_episodes=n_episodes,
        n_steps_total=n_total_steps,
        n_collisions=n_collisions,
        n_survivals=n_survivals,
        n_successes=n_successes,
        ttc_valid_samples=ttc_valid_samples,
        ttc_exposure=ttc_exposure,
        ttc_exposure_counts=ttc_exposure_counts,
        minimum_finite_ttc_s=minimum_finite_ttc_s,
        finite_ttc_p05_s=finite_ttc_p05_s,
        collision_rate_ci_low=coll_ci_low,
        collision_rate_ci_high=coll_ci_high,
        survival_rate_ci_low=surv_ci_low,
        survival_rate_ci_high=surv_ci_high,
        success_rate_ci_low=succ_ci_low,
        success_rate_ci_high=succ_ci_high,
        mean_speed_ci_low=mean_speed_ci_low,
        mean_speed_ci_high=mean_speed_ci_high,
        mean_progress_ci_low=mean_progress_ci_low,
        mean_progress_ci_high=mean_progress_ci_high,
        mean_distance_travelled_ci_low=mean_distance_ci_low,
        mean_distance_travelled_ci_high=mean_distance_ci_high,
    )


# ---------------------------------------------------------------------------
# Core evaluation API
# ---------------------------------------------------------------------------


def evaluate(
    policy: Policy,
    env: gym.Env,
    n_episodes: int = 20,
    seed: int = 0,
    fault_horizon_steps: int = 1,
    config: EvaluationConfig | None = None,
) -> EvalResults:
    """Evaluate one policy for n_episodes.

    Success is defined as:
      not collided
      and longitudinal_progress_m >= min_success_progress_m
      and mean_speed_mps >= min_success_mean_speed_mps

    Continuous 95% CIs are percentile bootstrap CIs.
    """
    cfg = config or EvaluationConfig()
    all_episode_stats, dt = _collect_episode_stats(
        policy=policy,
        env=env,
        n_episodes=n_episodes,
        seed=seed,
        fault_horizon_steps=fault_horizon_steps,
    )
    return _aggregate_episode_stats(all_episode_stats, dt=dt, cfg=cfg)


def evaluate_across_seeds(
    policy: Policy,
    env: gym.Env,
    n_seeds: int,
    episodes_per_seed: int,
    base_seed: int = 0,
    fault_horizon_steps: int = 1,
    config: EvaluationConfig | None = None,
) -> EvalResults:
    """Evaluate a policy across seeds and aggregate from raw episode stats.

    Aggregation is performed using episode/step/sample counts, not naïve means of
    already-aggregated percentages.
    """
    if n_seeds < 1:
        raise ValueError("n_seeds must be >= 1")
    if episodes_per_seed < 1:
        raise ValueError("episodes_per_seed must be >= 1")

    cfg = config or EvaluationConfig()
    all_episode_stats: list[EpisodeStats] = []
    dt: float | None = None

    for s in range(n_seeds):
        seed_stats, seed_dt = _collect_episode_stats(
            policy=policy,
            env=env,
            n_episodes=episodes_per_seed,
            seed=base_seed + s,
            fault_horizon_steps=fault_horizon_steps,
        )
        if dt is None:
            dt = seed_dt
        all_episode_stats.extend(seed_stats)

    return _aggregate_episode_stats(all_episode_stats, dt=(dt if dt is not None else 1.0), cfg=cfg)


# ---------------------------------------------------------------------------
# Pretty-print
# ---------------------------------------------------------------------------


def print_table(results: EvalResults, label: str = "Policy") -> None:
    """Print a formatted metric table to stdout."""
    border = "+" + "-" * 30 + "+" + "-" * 14 + "+"
    print(border)
    print(f"| {'Metric':<28} | {'Value':>12} |")
    print(border)

    def _fmt(v: float | None) -> str:
        if v is None:
            return "N/A"
        if isinstance(v, float) and v != v:
            return "N/A"
        return f"{v:.3f}"

    rows = [
        ("Policy", label),
        ("Episodes", str(results.n_episodes)),
        ("Collision rate", f"{results.collision_rate:.3f}"),
        ("Collision 95% CI", _fmt(results.collision_rate_ci_low) + "-" + _fmt(results.collision_rate_ci_high)),
        ("Survival rate", f"{results.survival_rate:.3f}"),
        ("Survival 95% CI", _fmt(results.survival_rate_ci_low) + "-" + _fmt(results.survival_rate_ci_high)),
        ("Success rate", f"{results.success_rate:.3f}"),
        ("Success 95% CI", _fmt(results.success_rate_ci_low) + "-" + _fmt(results.success_rate_ci_high)),
        ("Mean min TTC (s)", _fmt(results.mean_min_ttc)),
        ("Min finite TTC (s)", _fmt(results.minimum_finite_ttc_s)),
        ("Finite TTC p05 (s)", _fmt(results.finite_ttc_p05_s)),
        ("TTC < 1s exposure", _fmt(results.ttc_below_1s_frac)),
        ("TTC < 2s exposure", _fmt(results.ttc_below_2s_frac)),
        ("TTC < 4s exposure", _fmt(results.ttc_below_4s_frac)),
        ("RMS jerk (m/s^3)", f"{results.rms_jerk:.3f}"),
        ("Fallback rate", f"{results.fallback_rate:.3f}"),
        ("LC frequency (/step)", f"{results.lc_frequency:.3f}"),
        ("LC completion rate", _fmt(results.lc_completion_rate)),
        ("LC anticipatory frac", _fmt(results.lc_anticipatory_frac)),
        ("Ego fault rate", f"{results.ego_fault_rate:.3f}"),
        ("NPC fault rate", f"{results.npc_fault_rate:.3f}"),
        ("Mean speed (m/s)", f"{results.mean_speed:.3f}"),
        ("Mean progress (m)", f"{results.mean_longitudinal_progress:.3f}"),
    ]
    for name, val in rows:
        print(f"| {name:<28} | {val:>12} |")
    print(border)
