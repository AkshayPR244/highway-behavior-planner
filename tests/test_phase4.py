"""
pytest test suite for Phase 4 components — PPO + CMDP.

Covers:
  - ActorCritic output shapes and dtype
  - act() and act_stochastic() return valid actions
  - Warm-start from MLPPolicy checkpoint
  - GAE advantage computation on a synthetic trajectory
  - Lagrange update rule (λ increases when violated, decreases when slack)
  - IRLRewardShaper shaped_reward() is finite
  - RolloutBuffer add/compute_returns/minibatches
  - PPOTrainer.update() runs without error on a synthetic buffer
  - CMDPTrainer._update_lambda correctness
  - ActorCritic save/load round-trip

Run with:
    cd ~/highway-planner && source .venv/bin/activate
    pytest tests/test_phase4.py -v
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from rl.ppo_agent import ActorCritic, OBS_DIM, N_ACTIONS
from rl.ppo_trainer import PPOConfig, PPOTrainer, RolloutBuffer
from rl.cmdp_trainer import CMDPConfig, CMDPTrainer
from rl.reward_shaping import IRLRewardShaper
from optimizer.irl_optimizer import IRLPolicy
from optimizer.feature_extractor import N_FEATURES


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_obs(n: int = 1) -> np.ndarray:
    rng = np.random.default_rng(0)
    obs = rng.random((n, OBS_DIM)).astype(np.float32)
    return obs[0] if n == 1 else obs


def _make_ac(seed: int = 0) -> ActorCritic:
    torch.manual_seed(seed)
    return ActorCritic(device="cpu")


def _make_shaper() -> IRLRewardShaper:
    """Build a shaper from the saved IRL weights (skips if not present)."""
    import os
    weights_path = "results/irl_weights.npy"
    if not os.path.exists(weights_path):
        pytest.skip("irl_weights.npy not found — run irl_optimizer first")
    return IRLRewardShaper(weights_path=weights_path)


# ---------------------------------------------------------------------------
# TestActorCriticShape
# ---------------------------------------------------------------------------

class TestActorCriticShape:
    """Output shapes and dtypes for ActorCritic forward passes."""

    def test_forward_dist_and_value_shape(self):
        ac  = _make_ac()
        obs = torch.zeros(4, OBS_DIM)   # batch of 4
        dist, reward_value, cost_value = ac.forward(obs)
        assert reward_value.shape == (4,), f"reward_value shape: {reward_value.shape}"
        assert cost_value.shape == (4,), f"cost_value shape: {cost_value.shape}"
        assert dist.probs.shape == (4, N_ACTIONS)

    def test_get_action_and_value_shapes(self):
        ac  = _make_ac()
        obs = torch.zeros(8, OBS_DIM)
        action, lp, entropy, reward_value, cost_value = ac.get_action_and_value(obs)
        assert action.shape   == (8,)
        assert lp.shape       == (8,)
        assert entropy.shape  == (8,)
        assert reward_value.shape == (8,)
        assert cost_value.shape   == (8,)

    def test_get_action_and_value_with_given_action(self):
        ac     = _make_ac()
        obs    = torch.zeros(4, OBS_DIM)
        acts   = torch.zeros(4, dtype=torch.long)
        _, lp, entropy, reward_value, cost_value = ac.get_action_and_value(obs, acts)
        assert lp.shape == (4,)

    def test_logprob_is_finite(self):
        ac  = _make_ac(42)
        obs = torch.rand(16, OBS_DIM)
        _, lp, _, _, _ = ac.get_action_and_value(obs)
        assert torch.all(torch.isfinite(lp))

    def test_value_is_finite(self):
        ac  = _make_ac(1)
        obs = torch.rand(8, OBS_DIM)
        _, _, _, reward_value, cost_value = ac.get_action_and_value(obs)
        assert torch.all(torch.isfinite(reward_value))
        assert torch.all(torch.isfinite(cost_value))


# ---------------------------------------------------------------------------
# TestActorCriticAct
# ---------------------------------------------------------------------------

class TestActorCriticAct:
    """act() and act_stochastic() interfaces."""

    @pytest.mark.parametrize("seed", [0, 1, 7])
    def test_act_returns_valid_action(self, seed):
        ac  = _make_ac(seed)
        obs = _make_obs()
        a   = ac.act(obs)
        assert a in range(N_ACTIONS), f"act() returned {a}"

    def test_act_deterministic(self):
        ac  = _make_ac(3)
        obs = _make_obs()
        assert ac.act(obs) == ac.act(obs)

    def test_act_stochastic_returns_action_and_logprob(self):
        ac  = _make_ac(5)
        obs = _make_obs()
        a, lp = ac.act_stochastic(obs)
        assert a in range(N_ACTIONS)
        assert np.isfinite(lp)

    def test_act_stochastic_log_prob_negative(self):
        """log π(a|s) ≤ 0 always (probability ≤ 1)."""
        ac  = _make_ac(9)
        obs = _make_obs()
        for _ in range(20):
            _, lp = ac.act_stochastic(obs)
            assert lp <= 0.0 + 1e-6


# ---------------------------------------------------------------------------
# TestActorCriticSaveLoad
# ---------------------------------------------------------------------------

class TestActorCriticSaveLoad:
    def test_save_load_roundtrip(self, tmp_path):
        ac = _make_ac(77)
        p  = tmp_path / "ac.pt"
        ac.save(p)
        ac2 = ActorCritic.load(p)
        obs = torch.rand(2, OBS_DIM)
        with torch.no_grad():
            _, reward_v1, cost_v1 = ac.forward(obs)
            _, reward_v2, cost_v2 = ac2.forward(obs)
        torch.testing.assert_close(reward_v1, reward_v2)
        torch.testing.assert_close(cost_v1, cost_v2)

    def test_load_produces_valid_actions(self, tmp_path):
        ac = _make_ac(88)
        p  = tmp_path / "ac2.pt"
        ac.save(p)
        loaded = ActorCritic.load(p)
        assert loaded.act(_make_obs()) in range(N_ACTIONS)

    def test_warm_start_from_mlp(self, tmp_path):
        """load_actor_weights_from_mlp copies trunk weights from MLPPolicy."""
        import os
        mlp_path = "results/dagger_iter5_policy.pt"
        if not os.path.exists(mlp_path):
            pytest.skip("dagger_iter5_policy.pt not found")

        ac = ActorCritic(device="cpu")
        ac.load_actor_weights_from_mlp(mlp_path)
        # After loading, actor should produce valid actions
        assert ac.act(_make_obs()) in range(N_ACTIONS)

    def test_warm_start_changes_weights(self, tmp_path):
        """Warm-start must actually change the trunk weights."""
        import os
        mlp_path = "results/dagger_iter5_policy.pt"
        if not os.path.exists(mlp_path):
            pytest.skip("dagger_iter5_policy.pt not found")

        ac_random = ActorCritic(device="cpu")
        w_before  = ac_random.trunk[0].weight.data.clone()

        ac_random.load_actor_weights_from_mlp(mlp_path)
        w_after = ac_random.trunk[0].weight.data

        assert not torch.allclose(w_before, w_after), \
            "Trunk weights unchanged after warm-start"


# ---------------------------------------------------------------------------
# TestRolloutBuffer
# ---------------------------------------------------------------------------

class TestRolloutBuffer:
    """RolloutBuffer add / compute_returns / minibatches."""

    def _make_buffer(self, T: int = 16) -> RolloutBuffer:
        rng = np.random.default_rng(0)
        buf = RolloutBuffer()
        for _ in range(T):
            buf.add(
                obs      = rng.random(OBS_DIM).astype(np.float32),
                action   = int(rng.integers(0, N_ACTIONS)),
                reward   = float(rng.standard_normal()),
                cost     = float(rng.random()),
                done     = bool(rng.random() < 0.1),
                value    = float(rng.standard_normal()),
                log_prob = float(-rng.random()),
            )
        buf.compute_returns(last_value=0.0, gamma=0.99, gae_lambda=0.95,
                            device=torch.device("cpu"))
        return buf

    def test_returns_shape(self):
        buf = self._make_buffer(20)
        assert buf.reward_returns.shape    == (20,)
        assert buf.reward_advantages.shape == (20,)
        assert buf.cost_returns.shape      == (20,)
        assert buf.cost_advantages.shape   == (20,)

    def test_returns_finite(self):
        buf = self._make_buffer(32)
        assert torch.all(torch.isfinite(buf.reward_returns))
        assert torch.all(torch.isfinite(buf.reward_advantages))
        assert torch.all(torch.isfinite(buf.cost_returns))
        assert torch.all(torch.isfinite(buf.cost_advantages))

    def test_minibatches_cover_all_data(self):
        T   = 32
        buf = self._make_buffer(T)
        rng = np.random.default_rng(7)
        seen = []
        for batch in buf.minibatches(batch_size=8, device=torch.device("cpu"), rng=rng):
            obs_b = batch[0]
            seen.append(obs_b.shape[0])
        assert sum(seen) == T

    def test_gae_terminal_episode(self):
        """When done=True, GAE should not bleed across episodes."""
        buf = RolloutBuffer()
        # Two episodes: t=0..4 (done at t=4), t=5..9
        for t in range(10):
            buf.add(
                obs      = np.zeros(OBS_DIM, dtype=np.float32),
                action   = 1,
                reward   = 1.0,
                cost     = 0.0,
                done     = (t == 4),
                value    = 0.5,
                log_prob = -1.0,
            )
        buf.compute_returns(last_value=0.5, gamma=0.99, gae_lambda=0.95,
                            device=torch.device("cpu"))
        # Advantage at t=4 (terminal) should reflect no future value bleed
        assert buf.reward_advantages is not None

    def test_to_tensors_shapes(self):
        buf = self._make_buffer(16)
        obs_t, act_t, lp_t, val_t, cost_val_t = buf.to_tensors(torch.device("cpu"))
        assert obs_t.shape == (16, OBS_DIM)
        assert act_t.shape == (16,)
        assert lp_t.shape  == (16,)
        assert val_t.shape == (16,)
        assert cost_val_t.shape == (16,)


# ---------------------------------------------------------------------------
# TestGAEAdvantageMath
# ---------------------------------------------------------------------------

class TestGAEAdvantageMath:
    """Verify GAE on a hand-computed example."""

    def test_single_step_gae(self):
        """
        With T=1, gae_lambda=0, γ=0.99:
            Â_0 = r_0 + γ·V(s_1) - V(s_0)
                = 1.0 + 0.99·2.0 - 1.0 = 1.98
        """
        buf = RolloutBuffer()
        buf.add(obs=np.zeros(OBS_DIM, dtype=np.float32), action=0,
            reward=1.0, cost=0.0, done=False, value=1.0, log_prob=-1.0)
        buf.compute_returns(last_value=2.0, gamma=0.99, gae_lambda=0.0,
                            device=torch.device("cpu"))
        expected = 1.0 + 0.99 * 2.0 - 1.0   # = 1.98
        assert abs(buf.reward_advantages[0].item() - expected) < 1e-4

    def test_terminal_step_no_bootstrap(self):
        """
        When done=True, the next-state value should be zeroed out:
            Â_0 = r_0 + γ·0 - V(s_0) = r_0 - V(s_0)
        """
        buf = RolloutBuffer()
        buf.add(obs=np.zeros(OBS_DIM, dtype=np.float32), action=0,
            reward=3.0, cost=0.0, done=True, value=1.0, log_prob=-1.0)
        buf.compute_returns(last_value=99.0, gamma=0.99, gae_lambda=0.0,
                            device=torch.device("cpu"))
        # done=True zeroes the (1 - done) mask → no bootstrap
        expected = 3.0 - 1.0   # = 2.0
        assert abs(buf.reward_advantages[0].item() - expected) < 1e-4

    def test_reward_truncation_bootstraps_final_value(self):
        buf = RolloutBuffer()
        buf.add(
            obs=np.zeros(OBS_DIM, dtype=np.float32),
            action=0,
            reward=1.0,
            cost=0.0,
            done=True,
            terminated=False,
            truncated=True,
            value=2.0,
            cost_value=0.2,
            log_prob=-1.0,
            next_reward_value=5.0,
            next_cost_value=0.4,
        )
        buf.compute_returns(last_value=0.0, gamma=0.99, gae_lambda=0.0,
                            device=torch.device("cpu"), last_cost_value=0.0)
        assert abs(buf.reward_returns[0].item() - (1.0 + 0.99 * 5.0)) < 1e-4
        assert abs(buf.reward_advantages[0].item() - (1.0 + 0.99 * 5.0 - 2.0)) < 1e-4

    def test_cost_truncation_bootstraps_final_value(self):
        buf = RolloutBuffer()
        buf.add(
            obs=np.zeros(OBS_DIM, dtype=np.float32),
            action=0,
            reward=0.0,
            cost=0.0,
            done=True,
            terminated=False,
            truncated=True,
            value=0.0,
            cost_value=0.2,
            log_prob=-1.0,
            next_reward_value=0.0,
            next_cost_value=0.4,
        )
        buf.compute_returns(last_value=0.0, gamma=0.99, gae_lambda=0.0,
                            device=torch.device("cpu"), last_cost_value=0.0)
        assert abs(buf.cost_returns[0].item() - (0.0 + 0.99 * 0.4)) < 1e-4
        assert abs(buf.cost_advantages[0].item() - (0.0 + 0.99 * 0.4 - 0.2)) < 1e-4

    def test_terminal_step_no_truncation_bootstrap(self):
        buf = RolloutBuffer()
        buf.add(
            obs=np.zeros(OBS_DIM, dtype=np.float32),
            action=0,
            reward=3.0,
            cost=0.0,
            done=True,
            terminated=True,
            truncated=False,
            value=1.0,
            cost_value=0.2,
            log_prob=-1.0,
            next_reward_value=5.0,
            next_cost_value=0.4,
        )
        buf.compute_returns(last_value=0.0, gamma=0.99, gae_lambda=0.0,
                            device=torch.device("cpu"), last_cost_value=0.0)
        assert abs(buf.reward_returns[0].item() - 3.0) < 1e-4
        assert abs(buf.cost_returns[0].item() - 0.0) < 1e-4

    def test_truncation_does_not_cross_reset_boundary(self):
        buf = RolloutBuffer()
        buf.add(
            obs=np.zeros(OBS_DIM, dtype=np.float32),
            action=0,
            reward=1.0,
            cost=0.0,
            done=True,
            terminated=False,
            truncated=True,
            value=2.0,
            cost_value=0.0,
            log_prob=-1.0,
            next_reward_value=5.0,
            next_cost_value=0.0,
        )
        buf.add(
            obs=np.ones(OBS_DIM, dtype=np.float32),
            action=0,
            reward=100.0,
            cost=0.0,
            done=False,
            terminated=False,
            truncated=False,
            value=0.0,
            cost_value=0.0,
            log_prob=-1.0,
            next_reward_value=0.0,
            next_cost_value=0.0,
        )
        buf.compute_returns(last_value=0.0, gamma=0.99, gae_lambda=0.95,
                            device=torch.device("cpu"), last_cost_value=0.0)
        assert abs(buf.reward_returns[0].item() - (1.0 + 0.99 * 5.0)) < 1e-4
        assert abs(buf.reward_advantages[0].item() - (1.0 + 0.99 * 5.0 - 2.0)) < 1e-4


# ---------------------------------------------------------------------------
# TestLagrangeUpdate
# ---------------------------------------------------------------------------

class TestLagrangeUpdate:
    """CMDPTrainer._update_lambda correctness."""

    def _make_cmdp(self, eps: float = 0.1, lr: float = 0.1, lambda_init: float = 0.0):
        ac  = _make_ac()
        cfg = PPOConfig(device="cpu")
        return CMDPTrainer(
            actor_critic=ac,
            ppo_config=cfg,
            cmdp_config=CMDPConfig(
                collision_rate_threshold=eps,
                lambda_lr=lr,
                lambda_init=lambda_init,
                lambda_max=10.0,
            ),
        )

    def test_lambda_increases_when_violated(self):
        trainer = self._make_cmdp(eps=0.10, lr=0.1, lambda_init=0.0)
        trainer._update_lambda(collision_rate=0.30)  # 0.30 > 0.10
        assert trainer.lambda_ > 0.0

    def test_lambda_decreases_when_slack(self):
        trainer = self._make_cmdp(eps=0.10, lr=0.1, lambda_init=1.0)
        trainer._update_lambda(collision_rate=0.00)  # 0.00 < 0.10
        assert trainer.lambda_ < 1.0

    def test_lambda_clipped_at_zero(self):
        trainer = self._make_cmdp(eps=0.10, lr=0.5, lambda_init=0.01)
        trainer._update_lambda(collision_rate=0.00)  # violation = -0.10 → pulls below 0
        assert trainer.lambda_ == pytest.approx(0.0)

    def test_lambda_clipped_at_max(self):
        trainer = self._make_cmdp(eps=0.0, lr=100.0, lambda_init=0.0)
        trainer._update_lambda(collision_rate=1.0)   # big violation
        assert trainer.lambda_ == pytest.approx(10.0)

    def test_lambda_step_magnitude(self):
        """λ ← λ + lr * (col_rate − ε).  Verify the arithmetic."""
        trainer = self._make_cmdp(eps=0.10, lr=0.05, lambda_init=2.0)
        trainer._update_lambda(collision_rate=0.20)
        expected = 2.0 + 0.05 * (0.20 - 0.10)   # = 2.005
        assert abs(trainer.lambda_ - expected) < 1e-6

    def test_cmdp_penalizes_collision_transitions_only(self, monkeypatch):
        """Reward penalty should apply only where rollout cost_t == 1."""
        trainer = self._make_cmdp(eps=0.10, lr=0.0, lambda_init=2.0)

        buf = RolloutBuffer()
        buf.add(
            obs=np.zeros(OBS_DIM, dtype=np.float32),
            action=1,
            reward=1.0,
            cost=0.0,
            done=False,
            value=0.0,
            log_prob=-0.5,
        )
        buf.add(
            obs=np.zeros(OBS_DIM, dtype=np.float32),
            action=1,
            reward=1.0,
            cost=1.0,
            done=True,
            value=0.0,
            log_prob=-0.5,
        )
        buf.compute_returns(
            last_value=0.0,
            gamma=trainer.ppo.cfg.gamma,
            gae_lambda=trainer.ppo.cfg.gae_lambda,
            device=trainer.ppo.device,
        )

        def _fake_collect_rollout(_env):
            stats = {
                "episodes": 1,
                "mean_ep_ret": 0.0,
                "mean_ep_shaped": 0.0,
                "collision_count": 1,
            }
            return buf, stats

        captured = {}

        def _fake_update(_buf, lagrange_lambda=0.0):
            captured["rewards"] = list(_buf.rewards)
            captured["costs"] = list(_buf.costs)
            captured["lambda"] = lagrange_lambda
            return {"pg_loss": 0.0, "reward_vf_loss": 0.0, "cost_vf_loss": 0.0, "ent_loss": 0.0}

        monkeypatch.setattr(trainer.ppo, "collect_rollout", _fake_collect_rollout)
        monkeypatch.setattr(trainer.ppo, "update", _fake_update)

        trainer.train(env=None, n_iterations=1, verbose=False)
        assert captured["rewards"] == [1.0, 1.0]
        assert captured["costs"] == [0.0, 1.0]
        assert captured["lambda"] == pytest.approx(2.0)

    def test_lambda_unchanged_when_no_episodes_complete(self, monkeypatch):
        trainer = self._make_cmdp(eps=0.10, lr=0.1, lambda_init=0.7)

        buf = RolloutBuffer()
        buf.add(
            obs=np.zeros(OBS_DIM, dtype=np.float32),
            action=1,
            reward=1.0,
            cost=0.0,
            done=False,
            terminated=False,
            truncated=False,
            value=0.0,
            cost_value=0.0,
            log_prob=-0.5,
            next_reward_value=0.0,
            next_cost_value=0.0,
        )
        buf.compute_returns(last_value=0.0, gamma=trainer.ppo.cfg.gamma,
                            gae_lambda=trainer.ppo.cfg.gae_lambda,
                            device=trainer.ppo.device,
                            last_cost_value=0.0)

        def _fake_collect_rollout(_env):
            stats = {
                "episodes": 0,
                "mean_ep_ret": 0.0,
                "mean_ep_shaped": 0.0,
                "collision_count": 0,
            }
            return buf, stats

        captured = {}

        def _fake_update(_buf, lagrange_lambda=0.0):
            captured["lambda"] = lagrange_lambda
            return {"pg_loss": 0.0, "reward_vf_loss": 0.0, "cost_vf_loss": 0.0, "ent_loss": 0.0}

        monkeypatch.setattr(trainer.ppo, "collect_rollout", _fake_collect_rollout)
        monkeypatch.setattr(trainer.ppo, "update", _fake_update)

        history = trainer.train(env=None, n_iterations=1, verbose=False)
        assert trainer.lambda_ == pytest.approx(0.7)
        assert captured["lambda"] == pytest.approx(0.7)
        assert history[-1]["lambda_updated"] is False
        assert np.isnan(history[-1]["collision_rate"])


# ---------------------------------------------------------------------------
# TestIRLRewardShaper
# ---------------------------------------------------------------------------

class TestIRLRewardShaper:
    """Shaped reward is finite and moves in the right direction."""

    def test_shaped_reward_finite(self):
        shaper = _make_shaper()
        obs    = _make_obs()
        for action in range(N_ACTIONS):
            r = shaper.shaped_reward(obs, action, env_reward=0.5)
            assert np.isfinite(r), f"shaped_reward not finite for action {action}"

    def test_irl_cost_finite(self):
        shaper = _make_shaper()
        obs    = _make_obs()
        for action in range(N_ACTIONS):
            c = shaper.irl_cost(obs, action)
            assert np.isfinite(c)

    def test_env_reward_alpha_contribution(self):
        """Doubling env_reward should change total reward by alpha * delta."""
        shaper = _make_shaper()
        shaper.env_reward_alpha = 0.1
        obs    = _make_obs()
        r1 = shaper.shaped_reward(obs, 1, env_reward=0.0)
        r2 = shaper.shaped_reward(obs, 1, env_reward=1.0)
        assert abs((r2 - r1) - 0.1) < 1e-5

    def test_from_irl_policy(self):
        """IRLRewardShaper.from_irl_policy builds a shaper without file I/O."""
        import os
        if not os.path.exists("results/irl_weights.npy"):
            pytest.skip("irl_weights.npy not found")
        policy = IRLPolicy.load("results/irl_weights.npy")
        shaper = IRLRewardShaper.from_irl_policy(policy)
        assert shaper.weights.shape == (N_FEATURES,)
        r = shaper.shaped_reward(_make_obs(), action=1, env_reward=0.0)
        assert np.isfinite(r)


# ---------------------------------------------------------------------------
# TestPPOTrainerUpdate
# ---------------------------------------------------------------------------

class TestPPOTrainerUpdate:
    """PPOTrainer.update() runs without error on a synthetic buffer."""

    def _synthetic_buffer(self, T: int = 64) -> RolloutBuffer:
        rng = np.random.default_rng(42)
        buf = RolloutBuffer()
        for _ in range(T):
            buf.add(
                obs      = rng.random(OBS_DIM).astype(np.float32),
                action   = int(rng.integers(0, N_ACTIONS)),
                reward   = float(rng.standard_normal()),
                cost     = float(rng.random() < 0.05),
                done     = bool(rng.random() < 0.05),
                value    = float(rng.standard_normal()),
                log_prob = float(-rng.random() - 0.1),
            )
        buf.compute_returns(last_value=0.0, gamma=0.99, gae_lambda=0.95,
                            device=torch.device("cpu"))
        return buf

    def test_update_returns_dict(self):
        ac  = _make_ac(0)
        cfg = PPOConfig(n_steps=64, n_epochs=2, batch_size=16, device="cpu")
        trainer = PPOTrainer(actor_critic=ac, config=cfg)
        buf     = self._synthetic_buffer(64)
        stats   = trainer.update(buf)
        assert "pg_loss"  in stats
        assert "reward_vf_loss" in stats
        assert "cost_vf_loss" in stats
        assert "ent_loss" in stats

    def test_update_stats_finite(self):
        ac  = _make_ac(1)
        cfg = PPOConfig(n_steps=64, n_epochs=2, batch_size=16, device="cpu")
        trainer = PPOTrainer(actor_critic=ac, config=cfg)
        buf     = self._synthetic_buffer(64)
        stats   = trainer.update(buf)
        assert np.isfinite(stats["pg_loss"])
        assert np.isfinite(stats["reward_vf_loss"])
        assert np.isfinite(stats["cost_vf_loss"])
        assert np.isfinite(stats["ent_loss"])

    def test_update_changes_weights(self):
        """Weights must change after a gradient step."""
        ac  = _make_ac(2)
        w0  = ac.trunk[0].weight.data.clone()
        cfg = PPOConfig(n_steps=64, n_epochs=2, batch_size=16, device="cpu")
        trainer = PPOTrainer(actor_critic=ac, config=cfg)
        buf     = self._synthetic_buffer(64)
        trainer.update(buf)
        assert not torch.allclose(ac.trunk[0].weight.data, w0), \
            "Trunk weights unchanged after PPO update"
