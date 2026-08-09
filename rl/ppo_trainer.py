"""
PPO training loop for highway-v0.

Implements Proximal Policy Optimization (Schulman et al. 2017) with:
    - Clipped surrogate objective
    - Generalized Advantage Estimation (GAE, Schulman et al. 2016)
    - Entropy bonus for exploration
    - Value function coefficient in the combined loss
    - Mini-batch updates with multiple epochs per rollout

Algorithm outline (one iteration)
-----------------------------------
1. Collect T steps using current policy π_old (stochastic)
2. Compute returns G_t and advantages Â_t via GAE
3. For K epochs:
      For each mini-batch of size M:
        a. Compute new log-probs and values under π_θ
        b. ratio r_t = π_θ(a|s) / π_old(a|s)
        c. L_clip = E[min(r_t Â_t, clip(r_t, 1-ε, 1+ε) Â_t)]
        d. L_vf   = MSE(V_θ(s), G_t)
        e. L_ent  = H[π_θ(·|s)]
        f. L_total = −L_clip + c_vf·L_vf − c_ent·L_ent
        g. Gradient step on L_total

GAE
----
The advantage at step t is:
    δ_t    = r_t + γ V(s_{t+1}) − V(s_t)
    Â_t = δ_t + (γλ) δ_{t+1} + (γλ)² δ_{t+2} + ...

Higher λ → lower variance, higher bias (closer to MC return).
We use λ=0.95, γ=0.99 — standard for continuous control tasks.

Notes on the reward signal
---------------------------
The reward used here is the IRL-shaped reward from `rl/reward_shaping.py`,
not the raw env reward.  The CMDP trainer (cmdp_trainer.py) adds the
Lagrange safety term on top after collecting the rollout.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator

import numpy as np
import torch
import torch.nn as nn

from rl.ppo_agent import ActorCritic

# ---------------------------------------------------------------------------
# Hyperparameters dataclass
# ---------------------------------------------------------------------------

@dataclass
class PPOConfig:
    """All PPO hyperparameters in one place."""

    # Rollout
    n_steps:    int   = 512    # steps collected per iteration
    n_envs:     int   = 1      # number of parallel environments (we use 1)

    # Discount and GAE
    gamma:      float = 0.99
    gae_lambda: float = 0.95

    # PPO clipping
    clip_eps:   float = 0.2    # ε in the clipped surrogate

    # Loss coefficients
    vf_coef:    float = 0.5    # critic loss weight
    ent_coef:   float = 0.01   # entropy bonus (encourages exploration)

    # Optimisation
    n_epochs:     int   = 4      # epochs per rollout batch
    batch_size:   int   = 64     # mini-batch size
    lr:           float = 3e-4
    max_grad_norm: float = 0.5   # gradient clipping

    # Misc
    device:   str = "cpu"
    seed:     int = 42


# ---------------------------------------------------------------------------
# Rollout buffer
# ---------------------------------------------------------------------------

@dataclass
class RolloutBuffer:
    """
    Stores one rollout of (obs, action, reward, done, value, log_prob).

    All tensors have shape (T,) or (T, obs_dim).  After `compute_returns`,
    `returns` and `advantages` are also populated.
    """
    obs:       list = field(default_factory=list)
    actions:   list = field(default_factory=list)
    rewards:   list = field(default_factory=list)
    costs:     list = field(default_factory=list)
    dones:     list = field(default_factory=list)
    values:    list = field(default_factory=list)
    log_probs: list = field(default_factory=list)

    # Populated by compute_returns()
    returns:    torch.Tensor | None = None
    advantages: torch.Tensor | None = None
    last_value: float = 0.0

    def add(
        self,
        obs:      np.ndarray,
        action:   int,
        reward:   float,
        cost:     float,
        done:     bool,
        value:    float,
        log_prob: float,
    ) -> None:
        self.obs.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.costs.append(cost)
        self.dones.append(done)
        self.values.append(value)
        self.log_probs.append(log_prob)

    def compute_returns(
        self,
        last_value: float,
        gamma:      float,
        gae_lambda: float,
        device:     torch.device,
    ) -> None:
        """
        Compute GAE advantages and discounted returns.

        Parameters
        ----------
        last_value : float
            Bootstrap value V(s_{T+1}).  Zero if the episode ended; otherwise
            the critic's estimate at the final state.
        gamma, gae_lambda : float
            Standard PPO discount parameters.
        device : torch.device
        """
        T = len(self.rewards)
        advantages = np.zeros(T, dtype=np.float32)
        gae        = 0.0

        # Extend values with the bootstrap
        values_ext = self.values + [last_value]

        for t in reversed(range(T)):
            not_done = 1.0 - float(self.dones[t])
            delta    = (
                self.rewards[t]
                + gamma * values_ext[t + 1] * not_done
                - values_ext[t]
            )
            gae          = delta + gamma * gae_lambda * not_done * gae
            advantages[t] = gae

        returns = advantages + np.array(self.values, dtype=np.float32)
        self.last_value = float(last_value)

        self.advantages = torch.tensor(advantages, dtype=torch.float32, device=device)
        self.returns    = torch.tensor(returns,    dtype=torch.float32, device=device)

    def to_tensors(
        self, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (obs_t, actions_t, log_probs_t, values_t) tensors."""
        obs_t      = torch.tensor(np.array(self.obs),      dtype=torch.float32, device=device)
        actions_t  = torch.tensor(np.array(self.actions),  dtype=torch.long,    device=device)
        log_probs_t = torch.tensor(np.array(self.log_probs), dtype=torch.float32, device=device)
        values_t   = torch.tensor(np.array(self.values),   dtype=torch.float32, device=device)
        return obs_t, actions_t, log_probs_t, values_t

    def minibatches(
        self,
        batch_size: int,
        device:     torch.device,
        rng:        np.random.Generator,
    ) -> Generator[tuple, None, None]:
        """
        Yield random mini-batches from the buffer.

        Each mini-batch is a tuple:
            (obs, actions, old_log_probs, old_values, advantages, returns)
        """
        T = len(self.rewards)
        obs_t, actions_t, lp_t, val_t = self.to_tensors(device)
        indices = rng.permutation(T)

        for start in range(0, T, batch_size):
            idx = torch.tensor(indices[start: start + batch_size], device=device)
            yield (
                obs_t[idx],
                actions_t[idx],
                lp_t[idx],
                val_t[idx],
                self.advantages[idx],
                self.returns[idx],
            )


# ---------------------------------------------------------------------------
# PPO Trainer
# ---------------------------------------------------------------------------

class PPOTrainer:
    """
    Runs the PPO training loop.

    Parameters
    ----------
    actor_critic : ActorCritic
    config       : PPOConfig
    reward_shaper : callable(obs, action, env_reward) → float
        Any object with a `shaped_reward(obs, action, env_reward)` method.
        Typically an IRLRewardShaper.  If None, raw env reward is used.
    """

    def __init__(
        self,
        actor_critic:  ActorCritic,
        config:        PPOConfig,
        reward_shaper=None,
    ) -> None:
        self.ac      = actor_critic
        self.cfg     = config
        self.shaper  = reward_shaper
        self.device  = torch.device(config.device)
        self.rng     = np.random.default_rng(config.seed)
        self.optimizer = torch.optim.Adam(
            self.ac.parameters(), lr=config.lr, eps=1e-5
        )

        # Training history
        self.history: list[dict] = []

    # ------------------------------------------------------------------
    # Rollout collection
    # ------------------------------------------------------------------

    def collect_rollout(self, env) -> tuple[RolloutBuffer, dict]:
        """
        Collect exactly cfg.n_steps steps from *env* using the current policy.

        Steps may span multiple episodes.  A stats dict is returned with:
            episodes          : number of completed episodes
            mean_ep_len       : mean episode length
            mean_ep_ret       : mean raw (env) episode return
            mean_ep_shaped    : mean shaped episode return
            collision_count   : total collisions
        """
        buf   = RolloutBuffer()
        stats = {
            "episodes": 0, "ep_lens": [], "ep_rets": [], "ep_shaped": [],
            "collision_count": 0,
        }

        obs, _  = env.reset()
        ep_ret  = 0.0
        ep_shpd = 0.0
        ep_len  = 0

        for _ in range(self.cfg.n_steps):
            obs_arr = np.asarray(obs, dtype=np.float32)
            obs_t   = torch.from_numpy(obs_arr).unsqueeze(0).to(self.device, dtype=torch.float32)

            with torch.no_grad():
                dist, value = self.ac.forward(obs_t)
                action      = dist.sample()
                log_prob    = dist.log_prob(action)

            action_int = int(action.item())
            next_obs, env_reward, terminated, truncated, info = env.step(action_int)
            done = terminated or truncated

            # Shaped reward (or raw if no shaper)
            if self.shaper is not None:
                reward = self.shaper.shaped_reward(obs_arr, action_int, float(env_reward))
            else:
                reward = float(env_reward)

            buf.add(
                obs=obs_arr,
                action=action_int,
                reward=reward,
                cost=float(info.get("crashed", False)),
                done=done,
                value=float(value.item()),
                log_prob=float(log_prob.item()),
            )

            ep_ret  += float(env_reward)
            ep_shpd += reward
            ep_len  += 1

            if done:
                # Record episode stats
                stats["episodes"]        += 1
                stats["ep_lens"].append(ep_len)
                stats["ep_rets"].append(ep_ret)
                stats["ep_shaped"].append(ep_shpd)
                if info.get("crashed", False):
                    stats["collision_count"] += 1
                obs, _ = env.reset()
                ep_ret = ep_shpd = ep_len = 0.0
            else:
                obs = next_obs

        # Bootstrap value at the end of the rollout
        if not done:
            obs_t = torch.from_numpy(np.asarray(obs, dtype=np.float32)).unsqueeze(0).to(
                self.device, dtype=torch.float32
            )
            with torch.no_grad():
                _, last_val = self.ac.forward(obs_t)
            last_value = float(last_val.item())
        else:
            last_value = 0.0

        buf.compute_returns(
            last_value=last_value,
            gamma=self.cfg.gamma,
            gae_lambda=self.cfg.gae_lambda,
            device=self.device,
        )

        # Summarise
        stats["mean_ep_len"]    = float(np.mean(stats["ep_lens"]))    if stats["ep_lens"]    else 0.0
        stats["mean_ep_ret"]    = float(np.mean(stats["ep_rets"]))    if stats["ep_rets"]    else 0.0
        stats["mean_ep_shaped"] = float(np.mean(stats["ep_shaped"]))  if stats["ep_shaped"]  else 0.0

        return buf, stats

    # ------------------------------------------------------------------
    # PPO update
    # ------------------------------------------------------------------

    def update(self, buf: RolloutBuffer, extra_loss_fn=None) -> dict:
        """
        Run K epochs of mini-batch PPO updates on *buf*.

        Parameters
        ----------
        buf : RolloutBuffer  (with returns and advantages populated)
        extra_loss_fn : callable(obs_batch) → scalar Tensor or None
            Hook for the CMDP trainer to inject the Lagrange safety penalty.

        Returns
        -------
        dict with mean pg_loss, vf_loss, ent_loss, total_loss across all batches.
        """
        # Normalise advantages over the entire buffer (reduces variance)
        adv = buf.advantages
        buf.advantages = (adv - adv.mean()) / (adv.std() + 1e-8)

        pg_losses, vf_losses, ent_losses = [], [], []

        for _ in range(self.cfg.n_epochs):
            for (obs_b, act_b, old_lp_b, old_val_b, adv_b, ret_b) in buf.minibatches(
                self.cfg.batch_size, self.device, self.rng
            ):
                _, new_lp, entropy, new_val = self.ac.get_action_and_value(obs_b, act_b)

                # --- Clipped surrogate (actor) ---
                log_ratio   = new_lp - old_lp_b
                ratio       = log_ratio.exp()
                pg_loss1    = -adv_b * ratio
                pg_loss2    = -adv_b * ratio.clamp(1 - self.cfg.clip_eps, 1 + self.cfg.clip_eps)
                pg_loss     = torch.max(pg_loss1, pg_loss2).mean()

                # --- Value loss ---
                vf_loss     = nn.functional.mse_loss(new_val, ret_b)

                # --- Entropy bonus ---
                ent_loss    = -entropy.mean()

                # --- Combined loss ---
                loss = (
                    pg_loss
                    + self.cfg.vf_coef  * vf_loss
                    + self.cfg.ent_coef * ent_loss
                )

                # Optional hook (CMDP Lagrange penalty)
                if extra_loss_fn is not None:
                    loss = loss + extra_loss_fn(obs_b)

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.ac.parameters(), self.cfg.max_grad_norm)
                self.optimizer.step()

                pg_losses.append(pg_loss.item())
                vf_losses.append(vf_loss.item())
                ent_losses.append(ent_loss.item())

        return {
            "pg_loss":  float(np.mean(pg_losses)),
            "vf_loss":  float(np.mean(vf_losses)),
            "ent_loss": float(np.mean(ent_losses)),
        }

    # ------------------------------------------------------------------
    # Full training run
    # ------------------------------------------------------------------

    def train(
        self,
        env,
        n_iterations:   int,
        save_dir:       Path | None = None,
        save_every:     int = 10,
        extra_loss_fn=None,
        verbose:        bool = True,
    ) -> list[dict]:
        """
        Run *n_iterations* of PPO.

        Parameters
        ----------
        env           : gymnasium environment
        n_iterations  : int
        save_dir      : if given, save checkpoints here every save_every iters
        extra_loss_fn : passed to update() at every iteration
        verbose       : print per-iteration summary

        Returns
        -------
        list[dict]  Training history (one dict per iteration).
        """
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
            update_stats       = self.update(buf, extra_loss_fn=extra_loss_fn)

            record = {
                "iteration":          i,
                "episodes":           rollout_stats["episodes"],
                "mean_ep_ret":        rollout_stats["mean_ep_ret"],
                "mean_ep_shaped":     rollout_stats["mean_ep_shaped"],
                "collision_count":    rollout_stats["collision_count"],
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
                    f" | vf={update_stats['vf_loss']:.4f}"
                )

            if save_dir is not None and (i % save_every == 0 or i == n_iterations):
                ckpt = save_dir / f"ppo_iter{i:04d}.pt"
                self.ac.save(ckpt)
                if verbose:
                    print(f"    ↳ saved {ckpt}")

        return self.history
