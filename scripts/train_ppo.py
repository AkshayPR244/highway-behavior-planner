"""
Phase 4 entry point: train PPO-unconstrained and PPO-CMDP, then compare.

Usage
-----
    cd ~/highway-planner && source .venv/bin/activate
    python -m scripts.train_ppo                    # full run
    python -m scripts.train_ppo --iterations 50   # quick smoke-test
    python -m scripts.train_ppo --unconstrained-only
    python -m scripts.train_ppo --cmdp-only
    python -m scripts.train_ppo --eval-episodes 100 --eval-seeds 10

Output
------
  results/ppo_unconstrained_final.pt   — best unconstrained PPO checkpoint
  results/cmdp_final.pt                — best CMDP checkpoint
  results/ppo_training.png             — collision rate + λ curves

The script prints a four-row comparison table at the end:
    IDM Expert | IRL Policy | PPO-unconstrained | PPO-CMDP
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from shutil import copy2

import numpy as np

from envs.highway_wrapper import make_env
from metrics.evaluator import EvalResults, evaluate_across_seeds, print_table
from optimizer.irl_optimizer import IRLPolicy
from policies.idm_expert import collect_expert_rollouts
from rl.cmdp_trainer import CMDPConfig, CMDPTrainer
from rl.ppo_agent import ActorCritic
from rl.ppo_trainer import PPOConfig, PPOTrainer
from rl.reward_shaping import IRLRewardShaper
from scripts.seed_partitions import (
    SeedPartition,
    enumerate_dagger_training_env_seeds,
    enumerate_expert_rollout_env_seeds,
    validate_seed_partitions,
)

RESULTS_DIR = Path(__file__).parent.parent / "results"
DAGGER_CKPT = RESULTS_DIR / "dagger_iter5_policy.pt"
IRL_WEIGHTS = RESULTS_DIR / "irl_weights.npy"
CMDP_COLLISION_EPSILON = 0.10
DEFAULT_DAGGER_LINEAGE_SEED = 42
DEFAULT_DAGGER_LINEAGE_EXPERT_EPISODES = 50
DEFAULT_DAGGER_LINEAGE_ITERS = 5
DEFAULT_DAGGER_LINEAGE_ROLLOUT_EPISODES = 20
DEFAULT_DAGGER_LINEAGE_EVAL_EPISODES = 20
DEFAULT_DAGGER_LINEAGE_SCENARIO_EPISODES = 5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_reward_shaper(n_expert_episodes: int = 20, seed: int = 0) -> IRLRewardShaper:
    """
    Build IRLRewardShaper with reward_scale set from expert rollouts.

    Collecting a small set of expert rollouts (~20 episodes) gives enough
    statistics for a robust scale estimate without taking long.
    """
    print(f"Collecting {n_expert_episodes} expert episodes for reward scale estimate...")
    rollouts = collect_expert_rollouts(n_episodes=n_expert_episodes, seed=seed)
    shaper = IRLRewardShaper.from_rollouts(rollouts, weights_path=IRL_WEIGHTS)
    print(f"  reward_scale = {shaper.reward_scale:.4f}")
    return shaper


def _build_agent(device: str = "cpu") -> ActorCritic:
    """
    Create an ActorCritic warm-started from DAgger-5.
    Falls back to random init if DAgger checkpoint not found.
    """
    ac = ActorCritic(device=device)
    if DAGGER_CKPT.exists():
        ac.load_actor_weights_from_mlp(DAGGER_CKPT)
        print(f"Warm-started actor from {DAGGER_CKPT}")
    else:
        print(f"Warning: {DAGGER_CKPT} not found — using random initialisation")
    return ac


def _safe_json_value(value: object) -> object:
    """Convert NaN/Inf floats into JSON-safe values."""
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    return value


def _eval_results_to_dict(results) -> dict:
    """Serialize EvalResults dataclass into a JSON-friendly dict."""
    return {k: _safe_json_value(v) for k, v in vars(results).items()}


def _append_metrics_jsonl(
    out_path: Path,
    run_config: dict,
    policy_results: dict[str, dict[str, object]],
    partition_metadata: dict[str, dict[str, object]],
) -> None:
    """Append one structured run record to a JSONL file."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "run_config": run_config,
        "partitions": partition_metadata,
        "results": {
            name: value
            for name, value in policy_results.items()
            if value is not None
        },
    }
    with out_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, separators=(",", ":")) + "\n")


@dataclass(frozen=True)
class CheckpointScore:
    checkpoint_path: Path
    metrics: EvalResults


def _rank_checkpoint_scores(
    scored: list[CheckpointScore],
    collision_epsilon: float | None,
) -> list[CheckpointScore]:
    """Rank checkpoints with transparent lexicographic ordering.

    Ordering when collision_epsilon is set and any checkpoint is feasible:
      1) feasible (collision <= epsilon)
      2) success_rate (descending)
      3) mean_longitudinal_progress (descending)
      4) mean_speed (descending)
      5) rms_jerk (ascending)
      6) checkpoint name (ascending, deterministic tie-break)

    If no checkpoint is feasible:
      1) collision_rate (ascending)
      2) success_rate (descending)
      3) mean_longitudinal_progress (descending)
      4) mean_speed (descending)
      5) rms_jerk (ascending)
      6) checkpoint name (ascending)
    """
    if not scored:
        return []

    if collision_epsilon is None:
        return sorted(
            scored,
            key=lambda item: (
                -item.metrics.success_rate,
                -item.metrics.mean_longitudinal_progress,
                -item.metrics.mean_speed,
                item.metrics.rms_jerk,
                item.checkpoint_path.name,
            ),
        )

    feasible = [item for item in scored if item.metrics.collision_rate <= collision_epsilon]
    if feasible:
        return sorted(
            scored,
            key=lambda item: (
                0 if item.metrics.collision_rate <= collision_epsilon else 1,
                -item.metrics.success_rate,
                -item.metrics.mean_longitudinal_progress,
                -item.metrics.mean_speed,
                item.metrics.rms_jerk,
                item.checkpoint_path.name,
            ),
        )

    return sorted(
        scored,
        key=lambda item: (
            item.metrics.collision_rate,
            -item.metrics.success_rate,
            -item.metrics.mean_longitudinal_progress,
            -item.metrics.mean_speed,
            item.metrics.rms_jerk,
            item.checkpoint_path.name,
        ),
    )


def _select_best_scored_checkpoint(
    scored: list[CheckpointScore],
    collision_epsilon: float | None,
) -> CheckpointScore:
    ranked = _rank_checkpoint_scores(scored, collision_epsilon)
    if not ranked:
        raise ValueError("No checkpoint scores provided")
    return ranked[0]


def _training_seed_set(
    *,
    train_seed: int,
    n_expert_episodes: int,
    include_dagger_lineage: bool,
    dagger_lineage_seed: int,
    dagger_lineage_expert_episodes: int,
    dagger_lineage_iters: int,
    dagger_lineage_rollout_episodes: int,
    dagger_lineage_eval_episodes: int,
    dagger_lineage_scenario_episodes: int,
) -> set[int]:
    """Return explicit enumerable training-related seeds used by this script.

    Includes:
      - demonstration seeds used by reward shaper expert rollouts,
      - unconstrained PPO training env/PPO seed,
      - CMDP PPO training env/PPO seed.

    When a DAgger warm-start checkpoint is present, includes deterministic
    DAgger training/data-generation seeds from the known lineage config.
    """
    demo_seeds = enumerate_expert_rollout_env_seeds(train_seed, n_expert_episodes)
    seeds = demo_seeds | {train_seed, train_seed + 100}

    if include_dagger_lineage:
        seeds |= enumerate_dagger_training_env_seeds(
            base_seed=dagger_lineage_seed,
            n_expert_episodes=dagger_lineage_expert_episodes,
            n_dagger_iters=dagger_lineage_iters,
            n_rollout_episodes=dagger_lineage_rollout_episodes,
            eval_episodes=dagger_lineage_eval_episodes,
            scenario_eval_episodes=dagger_lineage_scenario_episodes,
        )

    return seeds


def _evaluate_on_partition(
    *,
    policy,
    env,
    partition: SeedPartition,
    fault_horizon_steps: int,
) -> EvalResults:
    return evaluate_across_seeds(
        policy=policy,
        env=env,
        n_seeds=partition.num_seeds,
        episodes_per_seed=partition.episodes_per_seed,
        base_seed=partition.base_seed,
        fault_horizon_steps=fault_horizon_steps,
    )


def _result_with_metadata(
    *,
    policy_id: str,
    metrics: EvalResults,
    partition: SeedPartition,
    checkpoint_id: str | None,
) -> dict[str, object]:
    return {
        "policy_id": policy_id,
        "checkpoint_id": checkpoint_id,
        "partition": partition.name,
        "seeds": partition.seeds(),
        "episodes_per_seed": partition.episodes_per_seed,
        "total_episodes": partition.total_episodes,
        "metrics": _eval_results_to_dict(metrics),
    }


def _select_best_checkpoint(
    checkpoint_dir: Path,
    pattern: str,
    validation_partition: SeedPartition,
    fault_horizon_steps: int,
    device: str,
    collision_epsilon: float | None,
) -> tuple[Path, EvalResults]:
    """Select the best checkpoint using validation partition only."""
    candidates = sorted(checkpoint_dir.glob(pattern))
    if not candidates:
        raise FileNotFoundError(f"No checkpoints found matching {pattern} in {checkpoint_dir}")

    env = make_env(seed=validation_partition.base_seed)
    scored: list[CheckpointScore] = []
    for ckpt in candidates:
        model = ActorCritic.load(ckpt, device=device)
        metrics = _evaluate_on_partition(
            policy=model,
            env=env,
            partition=validation_partition,
            fault_horizon_steps=fault_horizon_steps,
        )
        scored.append(CheckpointScore(checkpoint_path=ckpt, metrics=metrics))
    env.close()

    selected = _select_best_scored_checkpoint(scored, collision_epsilon)
    return selected.checkpoint_path, selected.metrics


# ---------------------------------------------------------------------------
# Training runs
# ---------------------------------------------------------------------------

def run_ppo_unconstrained(
    n_iterations: int,
    shaper:       IRLRewardShaper,
    device:       str,
    train_seed:   int,
    lr:           float,
    save_every:   int,
    verbose:      bool,
) -> ActorCritic:
    """Train PPO without safety constraint, return the final actor-critic."""
    print("\n" + "=" * 60)
    print("PPO — Unconstrained")
    print("=" * 60)

    ac  = _build_agent(device=device)
    cfg = PPOConfig(
        n_steps=512,
        n_epochs=4,
        batch_size=64,
        lr=lr,
        device=device,
        seed=train_seed,
    )
    trainer = PPOTrainer(actor_critic=ac, config=cfg, reward_shaper=shaper)
    env     = make_env(seed=train_seed)

    trainer.train(
        env=env,
        n_iterations=n_iterations,
        save_dir=RESULTS_DIR / "ppo_unconstrained",
        save_every=save_every,
        verbose=verbose,
    )
    env.close()

    final_path = RESULTS_DIR / "ppo_unconstrained_final.pt"
    ac.save(final_path, history=trainer.history)
    print(f"Saved → {final_path}")
    return ac


def run_ppo_cmdp(
    n_iterations: int,
    shaper:       IRLRewardShaper,
    device:       str,
    train_seed:   int,
    lr:           float,
    save_every:   int,
    verbose:      bool,
) -> ActorCritic:
    """Train PPO with CMDP safety constraint, return the final actor-critic."""
    print("\n" + "=" * 60)
    print("PPO — CMDP (λ-constrained)")
    print("=" * 60)

    ac       = _build_agent(device=device)
    ppo_cfg  = PPOConfig(
        n_steps=512,
        n_epochs=4,
        batch_size=64,
        lr=lr,
        device=device,
        seed=train_seed,
    )
    cmdp_cfg = CMDPConfig(
        collision_rate_threshold=0.10,
        lambda_lr=0.05,
        lambda_init=0.0,
        lambda_max=10.0,
    )
    trainer = CMDPTrainer(
        actor_critic=ac,
        ppo_config=ppo_cfg,
        cmdp_config=cmdp_cfg,
        reward_shaper=shaper,
    )
    env = make_env(seed=train_seed)

    trainer.train(
        env=env,
        n_iterations=n_iterations,
        save_dir=RESULTS_DIR / "cmdp",
        save_every=save_every,
        verbose=verbose,
    )
    env.close()

    final_path = RESULTS_DIR / "cmdp_final.pt"
    ac.save(final_path, history=trainer.ppo.history)
    print(f"Saved → {final_path}")
    print(f"Final λ = {trainer.lambda_:.4f}")
    return ac


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------

def print_comparison(
    ppo_unc_ac: ActorCritic | None,
    cmdp_ac:    ActorCritic | None,
    test_partition: SeedPartition,
    fault_horizon_steps: int,
) -> dict[str, object]:
    """Evaluate all policies on the explicit TEST partition."""
    print("\n" + "=" * 60)
    print(
        "Policy Comparison "
        f"(partition={test_partition.name}, "
        f"{test_partition.episodes_per_seed} eps/seed, "
        f"{test_partition.num_seeds} seeds, fault_horizon={fault_horizon_steps})"
    )
    print("=" * 60)

    env = make_env(seed=test_partition.base_seed)

    # IDM Expert
    from policies.idm_expert import IDMExpert
    idm_env = make_env(seed=test_partition.base_seed)
    idm_results = _evaluate_on_partition(
        policy=IDMExpert(idm_env),
        env=idm_env,
        partition=test_partition,
        fault_horizon_steps=fault_horizon_steps,
    )
    print_table(idm_results, label="IDM Expert")
    idm_env.close()

    # IRL Policy
    irl_policy   = IRLPolicy.load(IRL_WEIGHTS)
    irl_results = _evaluate_on_partition(
        policy=irl_policy,
        env=env,
        partition=test_partition,
        fault_horizon_steps=fault_horizon_steps,
    )
    print_table(irl_results, label="IRL Policy")

    # PPO-unconstrained
    if ppo_unc_ac is not None:
        ppo_results = _evaluate_on_partition(
            policy=ppo_unc_ac,
            env=env,
            partition=test_partition,
            fault_horizon_steps=fault_horizon_steps,
        )
        print_table(ppo_results, label="PPO-unconstrained")

    # PPO-CMDP
    if cmdp_ac is not None:
        cmdp_results = _evaluate_on_partition(
            policy=cmdp_ac,
            env=env,
            partition=test_partition,
            fault_horizon_steps=fault_horizon_steps,
        )
        print_table(cmdp_results, label="PPO-CMDP")

    env.close()

    return {
        "idm_expert": idm_results,
        "irl_policy": irl_results,
        "ppo_unconstrained": ppo_results if ppo_unc_ac is not None else None,
        "ppo_cmdp": cmdp_results if cmdp_ac is not None else None,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4: PPO + CMDP training")
    parser.add_argument(
        "--iterations", type=int, default=100,
        help="Number of PPO iterations per run (default 100)"
    )
    parser.add_argument(
        "--unconstrained-only", action="store_true",
        help="Only run the unconstrained PPO"
    )
    parser.add_argument(
        "--cmdp-only", action="store_true",
        help="Only run the CMDP-PPO"
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="PyTorch device (default cpu)"
    )
    parser.add_argument(
        "--lr", type=float, default=3e-4,
        help="PPO learning rate for both unconstrained and CMDP runs (default 3e-4)"
    )
    parser.add_argument(
        "--train-seed", type=int, default=1,
        help="Training environment and PPO RNG seed (default 1)"
    )
    parser.add_argument(
        "--save-every", type=int, default=10,
        help="Save checkpoint every N iterations during training (default 10)"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress per-iteration output"
    )
    parser.add_argument(
        "--validation-base-seed", type=int, default=2026,
        help="Validation partition base seed for checkpoint selection (default 2026)"
    )
    parser.add_argument(
        "--validation-num-seeds", type=int, default=5,
        help="Validation partition number of seeds (default 5)"
    )
    parser.add_argument(
        "--validation-episodes-per-seed", type=int, default=20,
        help="Validation partition episodes per seed (default 20)"
    )
    parser.add_argument(
        "--test-base-seed", type=int, default=4042,
        help="Test partition base seed for final reporting (default 4042)"
    )
    parser.add_argument(
        "--test-num-seeds", type=int, default=5,
        help="Test partition number of seeds (default 5)"
    )
    parser.add_argument(
        "--test-episodes-per-seed", type=int, default=20,
        help="Test partition episodes per seed (default 20)"
    )
    parser.add_argument(
        "--adversarial-base-seed", type=int, default=8080,
        help="Reserved adversarial partition base seed (default 8080)"
    )
    parser.add_argument(
        "--adversarial-num-seeds", type=int, default=3,
        help="Reserved adversarial partition number of seeds (default 3)"
    )
    parser.add_argument(
        "--adversarial-episodes-per-seed", type=int, default=10,
        help="Reserved adversarial partition episodes per seed (default 10)"
    )
    parser.add_argument(
        "--dagger-lineage-seed", type=int, default=DEFAULT_DAGGER_LINEAGE_SEED,
        help=(
            "Base seed assumed for DAgger warm-start lineage when "
            "results/dagger_iter5_policy.pt is present (default 42)"
        ),
    )
    parser.add_argument(
        "--dagger-lineage-expert-episodes", type=int, default=DEFAULT_DAGGER_LINEAGE_EXPERT_EPISODES,
        help="DAgger lineage n_expert_episodes for seed overlap checks (default 50)",
    )
    parser.add_argument(
        "--dagger-lineage-iters", type=int, default=DEFAULT_DAGGER_LINEAGE_ITERS,
        help="DAgger lineage n_dagger_iters for seed overlap checks (default 5)",
    )
    parser.add_argument(
        "--dagger-lineage-rollout-episodes", type=int, default=DEFAULT_DAGGER_LINEAGE_ROLLOUT_EPISODES,
        help="DAgger lineage n_rollout_episodes for seed overlap checks (default 20)",
    )
    parser.add_argument(
        "--dagger-lineage-eval-episodes", type=int, default=DEFAULT_DAGGER_LINEAGE_EVAL_EPISODES,
        help="DAgger lineage eval_episodes for seed overlap checks (default 20)",
    )
    parser.add_argument(
        "--dagger-lineage-scenario-episodes", type=int, default=DEFAULT_DAGGER_LINEAGE_SCENARIO_EPISODES,
        help="DAgger lineage scenario eval episodes for seed overlap checks (default 5)",
    )
    parser.add_argument(
        "--fault-horizon-steps", type=int, default=5,
        help="Counterfactual IDLE horizon for fault attribution (default 5)"
    )
    parser.add_argument(
        "--metrics-jsonl",
        type=str,
        default=str(RESULTS_DIR / "train_ppo_metrics.jsonl"),
        help=(
            "Path to append structured run metrics as JSONL "
            "(default results/train_ppo_metrics.jsonl)"
        ),
    )
    parser.add_argument(
        "--select-best-checkpoint",
        action="store_true",
        help=(
            "After training, evaluate saved checkpoints and copy the best one "
            "(min collision rate, then max goal completion) to final checkpoint path"
        ),
    )
    parser.add_argument(
        "--checkpoint-eval-episodes", type=int, default=None,
        help="Deprecated alias for --validation-episodes-per-seed"
    )
    parser.add_argument(
        "--checkpoint-eval-seeds", type=int, default=None,
        help="Deprecated alias for --validation-num-seeds"
    )
    parser.add_argument(
        "--checkpoint-eval-base-seed", type=int, default=None,
        help="Deprecated alias for --validation-base-seed"
    )
    args = parser.parse_args()

    if args.checkpoint_eval_episodes is not None:
        args.validation_episodes_per_seed = args.checkpoint_eval_episodes
    if args.checkpoint_eval_seeds is not None:
        args.validation_num_seeds = args.checkpoint_eval_seeds
    if args.checkpoint_eval_base_seed is not None:
        args.validation_base_seed = args.checkpoint_eval_base_seed

    validation_partition = SeedPartition(
        name="validation",
        base_seed=args.validation_base_seed,
        num_seeds=args.validation_num_seeds,
        episodes_per_seed=args.validation_episodes_per_seed,
    )
    test_partition = SeedPartition(
        name="test",
        base_seed=args.test_base_seed,
        num_seeds=args.test_num_seeds,
        episodes_per_seed=args.test_episodes_per_seed,
    )
    adversarial_partition = SeedPartition(
        name="adversarial",
        base_seed=args.adversarial_base_seed,
        num_seeds=args.adversarial_num_seeds,
        episodes_per_seed=args.adversarial_episodes_per_seed,
    )

    training_seeds = _training_seed_set(
        train_seed=args.train_seed,
        n_expert_episodes=20,
        include_dagger_lineage=DAGGER_CKPT.exists(),
        dagger_lineage_seed=args.dagger_lineage_seed,
        dagger_lineage_expert_episodes=args.dagger_lineage_expert_episodes,
        dagger_lineage_iters=args.dagger_lineage_iters,
        dagger_lineage_rollout_episodes=args.dagger_lineage_rollout_episodes,
        dagger_lineage_eval_episodes=args.dagger_lineage_eval_episodes,
        dagger_lineage_scenario_episodes=args.dagger_lineage_scenario_episodes,
    )
    validate_seed_partitions(
        training_seeds=training_seeds,
        validation_partition=validation_partition,
        test_partition=test_partition,
        adversarial_partition=adversarial_partition,
    )

    verbose = not args.quiet
    shaper  = _make_reward_shaper(n_expert_episodes=20, seed=args.train_seed)

    ppo_unc_ac = None
    cmdp_ac = None
    selected_unconstrained_ckpt: str | None = None
    selected_cmdp_ckpt: str | None = None

    if not args.cmdp_only:
        ppo_unc_ac = run_ppo_unconstrained(
            n_iterations=args.iterations,
            shaper=shaper,
            device=args.device,
            train_seed=args.train_seed,
            lr=args.lr,
            save_every=max(1, args.save_every),
            verbose=verbose,
        )

        if args.select_best_checkpoint:
            best_ckpt, best_metrics = _select_best_checkpoint(
                checkpoint_dir=RESULTS_DIR / "ppo_unconstrained",
                pattern="ppo_iter*.pt",
                validation_partition=validation_partition,
                fault_horizon_steps=args.fault_horizon_steps,
                device=args.device,
                collision_epsilon=None,
            )
            final_path = RESULTS_DIR / "ppo_unconstrained_final.pt"
            copy2(best_ckpt, final_path)
            ppo_unc_ac = ActorCritic.load(final_path, device=args.device)
            selected_unconstrained_ckpt = best_ckpt.name
            print(
                "Selected best unconstrained checkpoint "
                f"{best_ckpt.name} (collision={best_metrics.collision_rate:.3f}, "
                f"survival={best_metrics.survival_rate:.3f}, "
                f"success={best_metrics.success_rate:.3f})"
            )

    if not args.unconstrained_only:
        cmdp_ac = run_ppo_cmdp(
            n_iterations=args.iterations,
            shaper=shaper,
            device=args.device,
            train_seed=args.train_seed + 100,
            lr=args.lr,
            save_every=max(1, args.save_every),
            verbose=verbose,
        )

        if args.select_best_checkpoint:
            best_ckpt, best_metrics = _select_best_checkpoint(
                checkpoint_dir=RESULTS_DIR / "cmdp",
                pattern="cmdp_iter*.pt",
                validation_partition=validation_partition,
                fault_horizon_steps=args.fault_horizon_steps,
                device=args.device,
                collision_epsilon=CMDP_COLLISION_EPSILON,
            )
            final_path = RESULTS_DIR / "cmdp_final.pt"
            copy2(best_ckpt, final_path)
            cmdp_ac = ActorCritic.load(final_path, device=args.device)
            selected_cmdp_ckpt = best_ckpt.name
            print(
                "Selected best CMDP checkpoint "
                f"{best_ckpt.name} (collision={best_metrics.collision_rate:.3f}, "
                f"survival={best_metrics.survival_rate:.3f}, "
                f"success={best_metrics.success_rate:.3f})"
            )

    comparison_results = print_comparison(
        ppo_unc_ac,
        cmdp_ac,
        test_partition=test_partition,
        fault_horizon_steps=args.fault_horizon_steps,
    )

    policy_results_payload = {
        "idm_expert": _result_with_metadata(
            policy_id="idm_expert",
            metrics=comparison_results["idm_expert"],
            partition=test_partition,
            checkpoint_id=None,
        ),
        "irl_policy": _result_with_metadata(
            policy_id="irl_policy",
            metrics=comparison_results["irl_policy"],
            partition=test_partition,
            checkpoint_id=str(IRL_WEIGHTS.name),
        ),
        "ppo_unconstrained": (
            _result_with_metadata(
                policy_id="ppo_unconstrained",
                metrics=comparison_results["ppo_unconstrained"],
                partition=test_partition,
                checkpoint_id=selected_unconstrained_ckpt or "ppo_unconstrained_final.pt",
            )
            if comparison_results.get("ppo_unconstrained") is not None
            else None
        ),
        "ppo_cmdp": (
            _result_with_metadata(
                policy_id="ppo_cmdp",
                metrics=comparison_results["ppo_cmdp"],
                partition=test_partition,
                checkpoint_id=selected_cmdp_ckpt or "cmdp_final.pt",
            )
            if comparison_results.get("ppo_cmdp") is not None
            else None
        ),
    }

    metrics_path = Path(args.metrics_jsonl)
    run_config = {
        "iterations": args.iterations,
        "unconstrained_only": args.unconstrained_only,
        "cmdp_only": args.cmdp_only,
        "device": args.device,
        "lr": args.lr,
        "train_seed": args.train_seed,
        "training_seeds_enumerable": sorted(training_seeds),
        "dagger_lineage_included": DAGGER_CKPT.exists(),
        "dagger_lineage_seed": args.dagger_lineage_seed,
        "dagger_lineage_expert_episodes": args.dagger_lineage_expert_episodes,
        "dagger_lineage_iters": args.dagger_lineage_iters,
        "dagger_lineage_rollout_episodes": args.dagger_lineage_rollout_episodes,
        "dagger_lineage_eval_episodes": args.dagger_lineage_eval_episodes,
        "dagger_lineage_scenario_episodes": args.dagger_lineage_scenario_episodes,
        "save_every": args.save_every,
        "select_best_checkpoint": args.select_best_checkpoint,
        "validation_partition": validation_partition.to_metadata(),
        "test_partition": test_partition.to_metadata(),
        "adversarial_partition": adversarial_partition.to_metadata(),
        "fault_horizon_steps": args.fault_horizon_steps,
    }
    _append_metrics_jsonl(
        out_path=metrics_path,
        run_config=run_config,
        policy_results=policy_results_payload,
        partition_metadata={
            "validation": validation_partition.to_metadata(),
            "test": test_partition.to_metadata(),
            "adversarial": adversarial_partition.to_metadata(),
        },
    )
    print(f"Appended structured metrics → {metrics_path}")


if __name__ == "__main__":
    main()
