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

RESULTS_DIR = Path(__file__).parent.parent / "results"
DAGGER_CKPT = RESULTS_DIR / "dagger_iter5_policy.pt"
IRL_WEIGHTS = RESULTS_DIR / "irl_weights.npy"


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
    policy_results: dict[str, object],
) -> None:
    """Append one structured run record to a JSONL file."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "run_config": run_config,
        "results": {
            name: _eval_results_to_dict(res)
            for name, res in policy_results.items()
            if res is not None
        },
    }
    with out_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, separators=(",", ":")) + "\n")


def _select_best_checkpoint(
    checkpoint_dir: Path,
    pattern: str,
    eval_episodes: int,
    eval_seeds: int,
    eval_base_seed: int,
    fault_horizon_steps: int,
    device: str,
) -> tuple[Path, EvalResults]:
    """Select the best checkpoint by min collision rate, then max completion."""
    candidates = sorted(checkpoint_dir.glob(pattern))
    if not candidates:
        raise FileNotFoundError(f"No checkpoints found matching {pattern} in {checkpoint_dir}")

    env = make_env(seed=eval_base_seed)
    scored: list[tuple[Path, EvalResults]] = []
    for ckpt in candidates:
        model = ActorCritic.load(ckpt, device=device)
        metrics = evaluate_across_seeds(
            policy=model,
            env=env,
            n_seeds=eval_seeds,
            episodes_per_seed=eval_episodes,
            base_seed=eval_base_seed,
            fault_horizon_steps=fault_horizon_steps,
        )
        scored.append((ckpt, metrics))
    env.close()

    best_ckpt, best_metrics = min(
        scored,
        key=lambda item: (
            item[1].collision_rate,
            -item[1].success_rate,
            -item[1].mean_longitudinal_progress,
            item[1].rms_jerk,
        ),
    )
    return best_ckpt, best_metrics


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
    eval_episodes: int,
    eval_seeds: int,
    fault_horizon_steps: int,
) -> dict[str, object]:
    """Evaluate all policies and print a four-row comparison."""
    print("\n" + "=" * 60)
    print(
        "Policy Comparison "
        f"({eval_episodes} eps/seed, {eval_seeds} seeds, fault_horizon={fault_horizon_steps})"
    )
    print("=" * 60)

    env = make_env(seed=42)

    # IDM Expert
    from policies.idm_expert import IDMExpert
    idm_env     = make_env(seed=42)
    idm_results = evaluate_across_seeds(
        policy=IDMExpert(idm_env),
        env=idm_env,
        n_seeds=eval_seeds,
        episodes_per_seed=eval_episodes,
        base_seed=42,
        fault_horizon_steps=fault_horizon_steps,
    )
    print_table(idm_results, label="IDM Expert")
    idm_env.close()

    # IRL Policy
    irl_policy   = IRLPolicy.load(IRL_WEIGHTS)
    irl_results  = evaluate_across_seeds(
        policy=irl_policy,
        env=env,
        n_seeds=eval_seeds,
        episodes_per_seed=eval_episodes,
        base_seed=42,
        fault_horizon_steps=fault_horizon_steps,
    )
    print_table(irl_results, label="IRL Policy")

    # PPO-unconstrained
    if ppo_unc_ac is not None:
        ppo_results = evaluate_across_seeds(
            policy=ppo_unc_ac,
            env=env,
            n_seeds=eval_seeds,
            episodes_per_seed=eval_episodes,
            base_seed=42,
            fault_horizon_steps=fault_horizon_steps,
        )
        print_table(ppo_results, label="PPO-unconstrained")

    # PPO-CMDP
    if cmdp_ac is not None:
        cmdp_results = evaluate_across_seeds(
            policy=cmdp_ac,
            env=env,
            n_seeds=eval_seeds,
            episodes_per_seed=eval_episodes,
            base_seed=42,
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
        "--eval-episodes", type=int, default=20,
        help="Episodes per seed for final comparison table (default 20)"
    )
    parser.add_argument(
        "--eval-seeds", type=int, default=1,
        help="Number of seeds for final comparison table (default 1)"
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
        "--checkpoint-eval-episodes", type=int, default=20,
        help="Episodes per seed used for checkpoint selection (default 20)"
    )
    parser.add_argument(
        "--checkpoint-eval-seeds", type=int, default=3,
        help="Number of seeds used for checkpoint selection (default 3)"
    )
    parser.add_argument(
        "--checkpoint-eval-base-seed", type=int, default=2026,
        help="Base seed used for checkpoint selection eval (default 2026)"
    )
    args = parser.parse_args()

    verbose = not args.quiet
    shaper  = _make_reward_shaper(n_expert_episodes=20, seed=args.train_seed)

    ppo_unc_ac = None
    cmdp_ac    = None

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
                eval_episodes=args.checkpoint_eval_episodes,
                eval_seeds=args.checkpoint_eval_seeds,
                eval_base_seed=args.checkpoint_eval_base_seed,
                fault_horizon_steps=args.fault_horizon_steps,
                device=args.device,
            )
            final_path = RESULTS_DIR / "ppo_unconstrained_final.pt"
            copy2(best_ckpt, final_path)
            ppo_unc_ac = ActorCritic.load(final_path, device=args.device)
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
                eval_episodes=args.checkpoint_eval_episodes,
                eval_seeds=args.checkpoint_eval_seeds,
                eval_base_seed=args.checkpoint_eval_base_seed,
                fault_horizon_steps=args.fault_horizon_steps,
                device=args.device,
            )
            final_path = RESULTS_DIR / "cmdp_final.pt"
            copy2(best_ckpt, final_path)
            cmdp_ac = ActorCritic.load(final_path, device=args.device)
            print(
                "Selected best CMDP checkpoint "
                f"{best_ckpt.name} (collision={best_metrics.collision_rate:.3f}, "
                f"survival={best_metrics.survival_rate:.3f}, "
                f"success={best_metrics.success_rate:.3f})"
            )

    comparison_results = print_comparison(
        ppo_unc_ac,
        cmdp_ac,
        eval_episodes=args.eval_episodes,
        eval_seeds=args.eval_seeds,
        fault_horizon_steps=args.fault_horizon_steps,
    )

    metrics_path = Path(args.metrics_jsonl)
    run_config = {
        "iterations": args.iterations,
        "unconstrained_only": args.unconstrained_only,
        "cmdp_only": args.cmdp_only,
        "device": args.device,
        "lr": args.lr,
        "train_seed": args.train_seed,
        "save_every": args.save_every,
        "select_best_checkpoint": args.select_best_checkpoint,
        "checkpoint_eval_episodes": args.checkpoint_eval_episodes,
        "checkpoint_eval_seeds": args.checkpoint_eval_seeds,
        "checkpoint_eval_base_seed": args.checkpoint_eval_base_seed,
        "eval_episodes": args.eval_episodes,
        "eval_seeds": args.eval_seeds,
        "fault_horizon_steps": args.fault_horizon_steps,
    }
    _append_metrics_jsonl(
        out_path=metrics_path,
        run_config=run_config,
        policy_results=comparison_results,
    )
    print(f"Appended structured metrics → {metrics_path}")


if __name__ == "__main__":
    main()
