"""
Phase 4 entry point: train PPO-unconstrained and PPO-CMDP, then compare.

Usage
-----
    cd ~/highway-planner && source .venv/bin/activate
    python -m scripts.train_ppo                    # full run
    python -m scripts.train_ppo --iterations 50   # quick smoke-test
    python -m scripts.train_ppo --unconstrained-only
    python -m scripts.train_ppo --cmdp-only

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
from pathlib import Path

import numpy as np

from envs.highway_wrapper import make_env
from metrics.evaluator import evaluate, print_table
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

def _make_reward_shaper(n_expert_episodes: int = 20) -> IRLRewardShaper:
    """
    Build IRLRewardShaper with reward_scale set from expert rollouts.

    Collecting a small set of expert rollouts (~20 episodes) gives enough
    statistics for a robust scale estimate without taking long.
    """
    print(f"Collecting {n_expert_episodes} expert episodes for reward scale estimate...")
    rollouts = collect_expert_rollouts(n_episodes=n_expert_episodes, seed=0)
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


# ---------------------------------------------------------------------------
# Training runs
# ---------------------------------------------------------------------------

def run_ppo_unconstrained(
    n_iterations: int,
    shaper:       IRLRewardShaper,
    device:       str,
    verbose:      bool,
) -> ActorCritic:
    """Train PPO without safety constraint, return the final actor-critic."""
    print("\n" + "=" * 60)
    print("PPO — Unconstrained")
    print("=" * 60)

    ac  = _build_agent(device=device)
    cfg = PPOConfig(n_steps=512, n_epochs=4, batch_size=64, lr=3e-4, device=device)
    trainer = PPOTrainer(actor_critic=ac, config=cfg, reward_shaper=shaper)
    env     = make_env(seed=1)

    trainer.train(
        env=env,
        n_iterations=n_iterations,
        save_dir=RESULTS_DIR / "ppo_unconstrained",
        save_every=max(1, n_iterations // 5),
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
    verbose:      bool,
) -> ActorCritic:
    """Train PPO with CMDP safety constraint, return the final actor-critic."""
    print("\n" + "=" * 60)
    print("PPO — CMDP (λ-constrained)")
    print("=" * 60)

    ac       = _build_agent(device=device)
    ppo_cfg  = PPOConfig(n_steps=512, n_epochs=4, batch_size=64, lr=3e-4, device=device)
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
    env = make_env(seed=2)

    trainer.train(
        env=env,
        n_iterations=n_iterations,
        save_dir=RESULTS_DIR / "cmdp",
        save_every=max(1, n_iterations // 5),
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
) -> None:
    """Evaluate all policies and print a four-row comparison."""
    print("\n" + "=" * 60)
    print("Policy Comparison (20 episodes, seed=42)")
    print("=" * 60)

    env = make_env(seed=42)

    # IDM Expert
    from policies.idm_expert import IDMExpert
    idm_env     = make_env(seed=42)
    idm_results = evaluate(policy=IDMExpert(idm_env), env=idm_env, n_episodes=20, seed=42)
    print_table(idm_results, label="IDM Expert")
    idm_env.close()

    # IRL Policy
    irl_policy   = IRLPolicy.load(IRL_WEIGHTS)
    irl_results  = evaluate(policy=irl_policy, env=env, n_episodes=20, seed=42)
    print_table(irl_results, label="IRL Policy")

    # PPO-unconstrained
    if ppo_unc_ac is not None:
        ppo_results = evaluate(policy=ppo_unc_ac, env=env, n_episodes=20, seed=42)
        print_table(ppo_results, label="PPO-unconstrained")

    # PPO-CMDP
    if cmdp_ac is not None:
        cmdp_results = evaluate(policy=cmdp_ac, env=env, n_episodes=20, seed=42)
        print_table(cmdp_results, label="PPO-CMDP")

    env.close()


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
        "--quiet", action="store_true",
        help="Suppress per-iteration output"
    )
    args = parser.parse_args()

    verbose = not args.quiet
    shaper  = _make_reward_shaper(n_expert_episodes=20)

    ppo_unc_ac = None
    cmdp_ac    = None

    if not args.cmdp_only:
        ppo_unc_ac = run_ppo_unconstrained(
            n_iterations=args.iterations,
            shaper=shaper,
            device=args.device,
            verbose=verbose,
        )

    if not args.unconstrained_only:
        cmdp_ac = run_ppo_cmdp(
            n_iterations=args.iterations,
            shaper=shaper,
            device=args.device,
            verbose=verbose,
        )

    print_comparison(ppo_unc_ac, cmdp_ac)


if __name__ == "__main__":
    main()
