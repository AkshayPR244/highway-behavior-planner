from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator

import numpy as np
import torch
import torch.nn as nn

from rl.ppo_agent import ActorCritic


@dataclass
class PPOConfig:
    # Rollout
    n_steps: int = 512
    n_envs: int = 1

    # Discount and GAE
    gamma: float = 0.99
    gae_lambda: float = 0.95

    # PPO objective
    clip_eps: float = 0.2
    vf_coef: float = 0.5
    cost_vf_coef: float = 0.5
    ent_coef: float = 0.01

    # Optimization
    n_epochs: int = 10
    batch_size: int = 64
    lr: float = 3e-4
    max_grad_norm: float = 0.5

    # Misc
    device: str = "cpu"
    seed: int = 42


@dataclass
class RolloutBuffer:
    obs: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    rewards: list = field(default_factory=list)
    costs: list = field(default_factory=list)

    # Episode bookkeeping
    dones: list = field(default_factory=list)
    terminateds: list = field(default_factory=list)
    truncateds: list = field(default_factory=list)

    # Critic predictions at current and next transition states
    values: list = field(default_factory=list)
    cost_values: list = field(default_factory=list)
    next_values: list = field(default_factory=list)
    next_cost_values: list = field(default_factory=list)

    log_probs: list = field(default_factory=list)

    reward_returns: torch.Tensor | None = None
    reward_advantages: torch.Tensor | None = None
    cost_returns: torch.Tensor | None = None
    cost_advantages: torch.Tensor | None = None

    last_value: float = 0.0
    last_cost_value: float = 0.0

    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        cost: float,
        done: bool | None,
        value: float,
        log_prob: float,
        cost_value: float | None = None,
        terminated: bool | None = None,
        truncated: bool | None = None,
        next_reward_value: float | None = None,
        next_cost_value: float | None = None,
    ) -> None:
        if cost_value is None:
            cost_value = 0.0

        # Backward-compatible inference for callers that only provide done.
        if terminated is None and truncated is None:
            terminated = bool(done)
            truncated = False
        elif terminated is None:
            terminated = bool(done) if done is not None else False
        elif truncated is None:
            truncated = bool(done) and not bool(terminated)

        if done is None:
            done = bool(terminated or truncated)

        self.obs.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.costs.append(cost)

        self.dones.append(bool(done))
        self.terminateds.append(bool(terminated))
        self.truncateds.append(bool(truncated))

        self.values.append(float(value))
        self.cost_values.append(float(cost_value))
        self.next_values.append(None if next_reward_value is None else float(next_reward_value))
        self.next_cost_values.append(None if next_cost_value is None else float(next_cost_value))

        self.log_probs.append(float(log_prob))

    def compute_returns(
        self,
        last_value: float,
        gamma: float,
        gae_lambda: float,
        device: torch.device,
        last_cost_value: float = 0.0,
    ) -> None:
        t_steps = len(self.rewards)
        reward_advantages = np.zeros(t_steps, dtype=np.float32)
        cost_advantages = np.zeros(t_steps, dtype=np.float32)
        reward_gae = 0.0
        cost_gae = 0.0

        use_explicit_bootstrap = (
            len(self.next_values) == t_steps
            and len(self.next_cost_values) == t_steps
            and len(self.terminateds) == t_steps
            and len(self.truncateds) == t_steps
            and all(v is not None for v in self.next_values)
            and all(v is not None for v in self.next_cost_values)
        )

        if use_explicit_bootstrap:
            for t in reversed(range(t_steps)):
                terminated = bool(self.terminateds[t])
                truncated = bool(self.truncateds[t])

                # Bootstrap from next state on truncation, but never on true terminal.
                bootstrap_mask = 0.0 if terminated else 1.0
                # Stop recursion at any episode boundary (terminated or truncated).
                recursion_mask = 0.0 if (terminated or truncated) else 1.0

                reward_delta = (
                    self.rewards[t]
                    + gamma * float(self.next_values[t]) * bootstrap_mask
                    - self.values[t]
                )
                reward_gae = reward_delta + gamma * gae_lambda * recursion_mask * reward_gae
                reward_advantages[t] = reward_gae

                cost_delta = (
                    self.costs[t]
                    + gamma * float(self.next_cost_values[t]) * bootstrap_mask
                    - self.cost_values[t]
                )
                cost_gae = cost_delta + gamma * gae_lambda * recursion_mask * cost_gae
                cost_advantages[t] = cost_gae
        else:
            # Legacy path for buffers built without explicit next-state values.
            values_ext = self.values + [last_value]
            cost_values_ext = self.cost_values + [last_cost_value]
            for t in reversed(range(t_steps)):
                not_done = 1.0 - float(self.dones[t])
                reward_delta = self.rewards[t] + gamma * values_ext[t + 1] * not_done - values_ext[t]
                reward_gae = reward_delta + gamma * gae_lambda * not_done * reward_gae
                reward_advantages[t] = reward_gae

                cost_delta = self.costs[t] + gamma * cost_values_ext[t + 1] * not_done - cost_values_ext[t]
                cost_gae = cost_delta + gamma * gae_lambda * not_done * cost_gae
                cost_advantages[t] = cost_gae

        reward_returns = reward_advantages + np.asarray(self.values, dtype=np.float32)
        cost_returns = cost_advantages + np.asarray(self.cost_values, dtype=np.float32)

        self.last_value = float(last_value)
        self.last_cost_value = float(last_cost_value)

        self.reward_advantages = torch.tensor(reward_advantages, dtype=torch.float32, device=device)
        self.reward_returns = torch.tensor(reward_returns, dtype=torch.float32, device=device)
        self.cost_advantages = torch.tensor(cost_advantages, dtype=torch.float32, device=device)
        self.cost_returns = torch.tensor(cost_returns, dtype=torch.float32, device=device)

    def to_tensors(
        self, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        obs_t = torch.tensor(np.asarray(self.obs), dtype=torch.float32, device=device)
        actions_t = torch.tensor(np.asarray(self.actions), dtype=torch.long, device=device)
        log_probs_t = torch.tensor(np.asarray(self.log_probs), dtype=torch.float32, device=device)
        values_t = torch.tensor(np.asarray(self.values), dtype=torch.float32, device=device)
        cost_values_t = torch.tensor(np.asarray(self.cost_values), dtype=torch.float32, device=device)
        return obs_t, actions_t, log_probs_t, values_t, cost_values_t

    def minibatches(
        self,
        batch_size: int,
        device: torch.device,
        rng: np.random.Generator,
    ) -> Generator[tuple, None, None]:
        t_steps = len(self.rewards)
        obs_t, actions_t, lp_t, val_t, cost_val_t = self.to_tensors(device)
        indices = rng.permutation(t_steps)

        for start in range(0, t_steps, batch_size):
            idx = torch.tensor(indices[start : start + batch_size], device=device)
            yield (
                obs_t[idx],
                actions_t[idx],
                lp_t[idx],
                val_t[idx],
                cost_val_t[idx],
                self.reward_advantages[idx],
                self.reward_returns[idx],
                self.cost_advantages[idx],
                self.cost_returns[idx],
            )


class PPOTrainer:
    def __init__(
        self,
        actor_critic: ActorCritic,
        config: PPOConfig,
        reward_shaper=None,
    ) -> None:
        self.ac = actor_critic
        self.cfg = config
        self.shaper = reward_shaper
        self.device = torch.device(config.device)
        self.rng = np.random.default_rng(config.seed)
        self.optimizer = torch.optim.Adam(self.ac.parameters(), lr=config.lr, eps=1e-5)
        self.history: list[dict] = []

    def collect_rollout(self, env) -> tuple[RolloutBuffer, dict]:
        buf = RolloutBuffer()
        stats = {
            "episodes": 0,
            "ep_lens": [],
            "ep_rets": [],
            "ep_shaped": [],
            "collision_count": 0,
        }

        obs, _ = env.reset()
        ep_ret = 0.0
        ep_shaped = 0.0
        ep_len = 0

        for _ in range(self.cfg.n_steps):
            obs_arr = np.asarray(obs, dtype=np.float32)
            obs_t = torch.from_numpy(obs_arr).unsqueeze(0).to(self.device, dtype=torch.float32)

            with torch.no_grad():
                dist, reward_value, cost_value = self.ac.forward(obs_t)
                action = dist.sample()
                log_prob = dist.log_prob(action)

            action_int = int(action.item())
            next_obs, env_reward, terminated, truncated, info = env.step(action_int)
            episode_done = bool(terminated or truncated)

            # Evaluate value heads on the episode's actual final transition state
            # before any reset happens.
            next_obs_arr = np.asarray(next_obs, dtype=np.float32)
            next_obs_t = torch.from_numpy(next_obs_arr).unsqueeze(0).to(self.device, dtype=torch.float32)
            with torch.no_grad():
                _, next_reward_value, next_cost_value = self.ac.forward(next_obs_t)

            if self.shaper is not None:
                reward = self.shaper.shaped_reward(obs_arr, action_int, float(env_reward))
            else:
                reward = float(env_reward)

            buf.add(
                obs=obs_arr,
                action=action_int,
                reward=reward,
                cost=float(info.get("crashed", False)),
                done=episode_done,
                value=float(reward_value.item()),
                cost_value=float(cost_value.item()),
                log_prob=float(log_prob.item()),
                terminated=bool(terminated),
                truncated=bool(truncated),
                next_reward_value=float(next_reward_value.item()),
                next_cost_value=float(next_cost_value.item()),
            )

            ep_ret += float(env_reward)
            ep_shaped += reward
            ep_len += 1

            if episode_done:
                stats["episodes"] += 1
                stats["ep_lens"].append(ep_len)
                stats["ep_rets"].append(ep_ret)
                stats["ep_shaped"].append(ep_shaped)
                if info.get("crashed", False):
                    stats["collision_count"] += 1
                obs, _ = env.reset()
                ep_ret = 0.0
                ep_shaped = 0.0
                ep_len = 0
            else:
                obs = next_obs

        # Explicit next-state values are already stored transition-by-transition.
        buf.compute_returns(
            last_value=0.0,
            gamma=self.cfg.gamma,
            gae_lambda=self.cfg.gae_lambda,
            device=self.device,
            last_cost_value=0.0,
        )

        stats["mean_ep_len"] = float(np.mean(stats["ep_lens"])) if stats["ep_lens"] else 0.0
        stats["mean_ep_ret"] = float(np.mean(stats["ep_rets"])) if stats["ep_rets"] else 0.0
        stats["mean_ep_shaped"] = float(np.mean(stats["ep_shaped"])) if stats["ep_shaped"] else 0.0

        return buf, stats

    def update(self, buf: RolloutBuffer, lagrange_lambda: float = 0.0) -> dict:
        pg_losses, reward_vf_losses, cost_vf_losses, ent_losses = [], [], [], []

        for _ in range(self.cfg.n_epochs):
            for (
                obs_b,
                act_b,
                old_lp_b,
                _old_val_b,
                _old_cost_val_b,
                reward_adv_b,
                reward_ret_b,
                cost_adv_b,
                cost_ret_b,
            ) in buf.minibatches(self.cfg.batch_size, self.device, self.rng):
                _, new_lp, entropy, new_reward_val, new_cost_val = self.ac.get_action_and_value(obs_b, act_b)

                log_ratio = new_lp - old_lp_b
                ratio = log_ratio.exp()
                policy_adv_b = reward_adv_b - lagrange_lambda * cost_adv_b
                policy_adv_b = (policy_adv_b - policy_adv_b.mean()) / (policy_adv_b.std() + 1e-8)

                pg_loss1 = -policy_adv_b * ratio
                pg_loss2 = -policy_adv_b * ratio.clamp(1 - self.cfg.clip_eps, 1 + self.cfg.clip_eps)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                reward_vf_loss = nn.functional.mse_loss(new_reward_val, reward_ret_b)
                cost_vf_loss = nn.functional.mse_loss(new_cost_val, cost_ret_b)
                ent_loss = -entropy.mean()

                loss = (
                    pg_loss
                    + self.cfg.vf_coef * reward_vf_loss
                    + self.cfg.cost_vf_coef * cost_vf_loss
                    + self.cfg.ent_coef * ent_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.ac.parameters(), self.cfg.max_grad_norm)
                self.optimizer.step()

                pg_losses.append(pg_loss.item())
                reward_vf_losses.append(reward_vf_loss.item())
                cost_vf_losses.append(cost_vf_loss.item())
                ent_losses.append(ent_loss.item())

        return {
            "pg_loss": float(np.mean(pg_losses)),
            "reward_vf_loss": float(np.mean(reward_vf_losses)),
            "cost_vf_loss": float(np.mean(cost_vf_losses)),
            "ent_loss": float(np.mean(ent_losses)),
        }

    def train(
        self,
        env,
        n_iterations: int,
        save_dir: Path | None = None,
        save_every: int = 10,
        verbose: bool = True,
    ) -> list[dict]:
        if save_dir is not None:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

        if verbose:
            print("=" * 60)
            print("PPO Training")
            print(f"  n_iterations={n_iterations}, n_steps={self.cfg.n_steps}")
            print(f"  lr={self.cfg.lr}, clip_eps={self.cfg.clip_eps}")
            print(f"  gamma={self.cfg.gamma}, gae_lambda={self.cfg.gae_lambda}")
            print("=" * 60)

        for i in range(1, n_iterations + 1):
            buf, rollout_stats = self.collect_rollout(env)
            update_stats = self.update(buf)

            record = {
                "iteration": i,
                "episodes": rollout_stats["episodes"],
                "mean_ep_ret": rollout_stats["mean_ep_ret"],
                "mean_ep_shaped": rollout_stats["mean_ep_shaped"],
                "collision_count": rollout_stats["collision_count"],
                **update_stats,
            }
            self.history.append(record)

            if verbose and (i % 5 == 0 or i == 1):
                print(
                    f"  iter {i:4d}/{n_iterations}"
                    f" | eps={rollout_stats['episodes']:3d}"
                    f" | ret={rollout_stats['mean_ep_ret']:+.3f}"
                    f" | shaped={rollout_stats['mean_ep_shaped']:+.3f}"
                    f" | coll={rollout_stats['collision_count']}"
                    f" | pg={update_stats['pg_loss']:+.4f}"
                    f" | rvf={update_stats['reward_vf_loss']:.4f}"
                )

            if save_dir is not None and (i % save_every == 0 or i == n_iterations):
                ckpt = save_dir / f"ppo_iter{i:04d}.pt"
                self.ac.save(ckpt)
                if verbose:
                    print(f"    saved {ckpt}")

        return self.history
