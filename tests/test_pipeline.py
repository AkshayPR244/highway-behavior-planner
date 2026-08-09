"""
Fast end-to-end pipeline smoke test.

Purpose
-------
Catch interface regressions across phases by running a tiny version of:
  expert rollouts -> BC -> one DAgger-style aggregation/retrain -> IRL -> PPO -> CMDP

This is intentionally small and deterministic. It validates that phase outputs
can be consumed by the next phase without shape/type mismatches.
"""
from __future__ import annotations

import numpy as np
import torch

from envs.highway_wrapper import make_env
from optimizer.irl_optimizer import IRLPolicy, train_irl
from policies.idm_expert import collect_expert_rollouts
from rl.cmdp_trainer import CMDPConfig, CMDPTrainer
from rl.ppo_agent import ActorCritic
from rl.ppo_trainer import PPOConfig, PPOTrainer
from rl.reward_shaping import IRLRewardShaper
from training.bc_train import ExpertDataset, split_rollouts, train_bc
from training.dagger_train import _retrain_on_dataset, aggregate, rollout_policy


def test_fast_end_to_end_pipeline(tmp_path):
    """Run a tiny pipeline and assert each phase produces usable artifacts."""
    seed = 7

    # ------------------------------------------------------------------
    # Phase 1: expert dataset
    # ------------------------------------------------------------------
    expert_rollouts = collect_expert_rollouts(n_episodes=2, seed=seed)
    assert len(expert_rollouts) == 2
    assert all(ep["observations"].ndim == 2 for ep in expert_rollouts)

    # ------------------------------------------------------------------
    # Phase 2a: BC warm-start
    # ------------------------------------------------------------------
    bc_ckpt = tmp_path / "bc_policy.pt"
    bc_policy = train_bc(
        n_episodes=2,
        n_epochs=2,
        batch_size=32,
        lr=1e-3,
        val_frac=0.5,
        patience=1,
        seed=seed,
        save_path=bc_ckpt,
        device="cpu",
        verbose=False,
    )
    assert bc_ckpt.exists()

    # ------------------------------------------------------------------
    # Phase 2b: one DAgger-style aggregate + retrain
    # ------------------------------------------------------------------
    tagged = []
    for ep in expert_rollouts:
        tagged.append({
            "observations": ep["observations"],
            "actions": ep["actions"],
            "source": "expert",
        })

    rollout_eps = rollout_policy(
        policy=bc_policy,
        n_episodes=1,
        seed=seed + 100,
        beta=0.5,
    )
    agg = aggregate(tagged, rollout_eps)
    train_rollouts, val_rollouts = split_rollouts(agg, val_frac=0.5, seed=seed)

    dagger_policy = _retrain_on_dataset(
        train_ds=ExpertDataset(train_rollouts),
        val_ds=ExpertDataset(val_rollouts),
        n_epochs=2,
        batch_size=32,
        lr=1e-3,
        patience=1,
        save_path=tmp_path / "dagger_iter1_policy.pt",
        device="cpu",
        verbose=False,
        sample_weights=None,
    )

    obs_sample = expert_rollouts[0]["observations"][0]
    action = dagger_policy.act(obs_sample)
    assert isinstance(action, int)
    assert 0 <= action < 5

    # ------------------------------------------------------------------
    # Phase 3: IRL on aggregated data
    # ------------------------------------------------------------------
    irl_input = [{"observations": ep["observations"], "actions": ep["actions"]} for ep in agg]
    weights, history = train_irl(
        rollouts=irl_input,
        n_epochs=10,
        lr=5e-2,
        l2_reg=1e-3,
        patience=3,
        save_path=tmp_path / "irl_weights.npy",
        device="cpu",
        verbose=False,
    )

    assert weights.shape == (8,)
    assert np.all(np.isfinite(weights))
    assert len(history) >= 1

    irl_policy = IRLPolicy(weights)
    shaper = IRLRewardShaper.from_irl_policy(irl_policy, env_reward_alpha=0.1, reward_scale=1.0)

    # ------------------------------------------------------------------
    # Phase 4a: PPO smoke train
    # ------------------------------------------------------------------
    env = make_env(seed=seed)
    ac = ActorCritic(device="cpu")
    ppo_cfg = PPOConfig(
        n_steps=32,
        n_epochs=1,
        batch_size=16,
        lr=3e-4,
        device="cpu",
        seed=seed,
    )
    ppo = PPOTrainer(actor_critic=ac, config=ppo_cfg, reward_shaper=shaper)
    hist_ppo = ppo.train(env=env, n_iterations=1, verbose=False)
    assert len(hist_ppo) == 1
    assert np.isfinite(hist_ppo[0]["pg_loss"])

    # ------------------------------------------------------------------
    # Phase 4b: CMDP smoke train
    # ------------------------------------------------------------------
    ac_cmdp = ActorCritic(device="cpu")
    cmdp = CMDPTrainer(
        actor_critic=ac_cmdp,
        ppo_config=ppo_cfg,
        cmdp_config=CMDPConfig(collision_rate_threshold=0.5, lambda_lr=0.05),
        reward_shaper=shaper,
    )
    hist_cmdp = cmdp.train(env=env, n_iterations=1, verbose=False)
    env.close()

    assert len(hist_cmdp) == 1
    assert "collision_rate" in hist_cmdp[0]
    assert np.isfinite(hist_cmdp[0]["collision_rate"])

    # Ensure the new history artifacts are numerically sane.
    assert torch.isfinite(torch.tensor(cmdp.lambda_))
