"""
CMDP (Constrained MDP) trainer wrapping PPO with a Lagrange multiplier λ.

The safety constraint
----------------------
We want to find the policy π that maximises expected IRL reward subject to:

    J_cost(π) ≤ ε       (collision rate ≤ threshold)

The Lagrangian relaxation turns this into an unconstrained saddle-point problem:

    max_π  min_{λ≥0}  J_reward(π) − λ · (J_cost(π) − ε)

λ is the Lagrange multiplier.  When the constraint is *violated* (collision
rate > ε), λ rises — increasing the penalty on the cost and pushing the policy
back toward safety.  When the constraint is *slack* (collision rate < ε), λ
falls — relaxing the penalty and letting the policy optimise reward more freely.

Dual update rule (gradient ascent on the dual):
    λ ← max(0, λ + α_λ · (J_cost − ε))

where J_cost is estimated from the most recent rollout batch.

The safety cost signal
-----------------------
We use a binary step cost per episode: 1 if the episode ended in a collision,
0 otherwise.  This is the simplest formulation that directly minimises the
collision rate.

The policy-side penalty uses a lagrangian advantage term during PPO updates:

    A_L(s, a) = A_r(s, a) − λ · A_c(s, a)

where the reward and cost critics are trained separately and the cost signal
is a binary per-step collision label (1 on collision transitions, 0 otherwise)
stored in the rollout buffer.

λ history
----------
`CMDPTrainer` records λ at every iteration so you can plot how the constraint
binding tightness evolves.  A healthy run shows λ rising early (policy is
unsafe), then stabilising once the constraint is satisfied.

References
----------
- Achiam et al., "Constrained Policy Optimization", ICML 2017
- Stooke et al., "Responsive Safety in Reinforcement Learning by PID Lagrangian
  Methods", ICML 2020 (uses a PID controller for λ; we use simple gradient)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from rl.ppo_agent import ActorCritic
from rl.ppo_trainer import PPOConfig, PPOTrainer
from rl.reward_shaping import IRLRewardShaper


class CMDPConfig:
    """CMDP-specific hyperparameters (on top of PPOConfig)."""

    def __init__(
        self,
        collision_rate_threshold: float = 0.10,   # ε: target max collision rate
        lambda_lr:                float = 0.05,   # α_λ: dual update step size
        lambda_init:              float = 0.0,    # initial λ
        lambda_max:               float = 10.0,   # clip λ to prevent divergence
    ) -> None:
        self.collision_rate_threshold = collision_rate_threshold
        self.lambda_lr                = lambda_lr
        self.lambda_init              = lambda_init
        self.lambda_max               = lambda_max


class CMDPTrainer:
    """
    PPO + Lagrange multiplier safety constraint.

    Parameters
    ----------
    actor_critic  : ActorCritic
    ppo_config    : PPOConfig
    cmdp_config   : CMDPConfig
    reward_shaper : IRLRewardShaper or None
    """

    def __init__(
        self,
        actor_critic:  ActorCritic,
        ppo_config:    PPOConfig,
        cmdp_config:   CMDPConfig | None = None,
        reward_shaper: IRLRewardShaper | None = None,
    ) -> None:
        self.cmdp_cfg = cmdp_config or CMDPConfig()
        self._lambda  = float(self.cmdp_cfg.lambda_init)

        self.ppo = PPOTrainer(
            actor_critic=actor_critic,
            config=ppo_config,
            reward_shaper=reward_shaper,
        )

        # Records per-iteration: λ, collision rate, reward
        self.lambda_history:    list[float] = []
        self.collision_history: list[float] = []

    @property
    def lambda_(self) -> float:
        return self._lambda

    # ------------------------------------------------------------------
    # Lagrange update
    # ------------------------------------------------------------------

    def _update_lambda(self, collision_rate: float) -> None:
        """
        Dual gradient ascent step.

        λ ← clip(λ + α_λ · (collision_rate − ε), 0, λ_max)

        If the collision rate is above ε, λ rises → stronger safety penalty.
        If below ε, λ falls → policy can be more aggressive.
        """
        violation    = collision_rate - self.cmdp_cfg.collision_rate_threshold
        new_lambda   = self._lambda + self.cmdp_cfg.lambda_lr * violation
        self._lambda = float(np.clip(new_lambda, 0.0, self.cmdp_cfg.lambda_max))

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(
        self,
        env,
        n_iterations: int,
        save_dir:     Path | None = None,
        save_every:   int = 10,
        verbose:      bool = True,
    ) -> list[dict]:
        """
        Run *n_iterations* of CMDP-PPO.

                At each iteration:
                    1. Collect rollout with current policy
                    2. Compute per-rollout collision rate
                    3. Run PPO-Lagrangian update using reward and cost advantages
                    4. Update λ via dual gradient step

        Returns
        -------
        list[dict]  One record per iteration with PPO stats + λ + collision rate.
        """
        if save_dir is not None:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

        if verbose:
            print("=" * 60)
            print("CMDP-PPO Training")
            print(f"  n_iterations={n_iterations}")
            print(f"  collision_threshold={self.cmdp_cfg.collision_rate_threshold}")
            print(f"  lambda_lr={self.cmdp_cfg.lambda_lr}")
            print(f"  initial λ={self._lambda:.3f}")
            print("=" * 60)

        for i in range(1, n_iterations + 1):
            # ---- 1. Collect rollout ----
            buf, rollout_stats = self.ppo.collect_rollout(env)

            n_steps = len(buf.rewards)
            collision_rate = (
                rollout_stats["collision_count"] / max(rollout_stats["episodes"], 1)
            )

            # ---- 2. PPO-Lagrangian update ----
            update_stats = self.ppo.update(buf, lagrange_lambda=self._lambda)

            # ---- 3. Update λ ----
            self._update_lambda(collision_rate)
            self.lambda_history.append(self._lambda)
            self.collision_history.append(collision_rate)

            record = {
                "iteration":       i,
                "episodes":        rollout_stats["episodes"],
                "mean_ep_ret":     rollout_stats["mean_ep_ret"],
                "mean_ep_shaped":  rollout_stats["mean_ep_shaped"],
                "collision_rate":  collision_rate,
                "lambda":          self._lambda,
                **update_stats,
            }
            self.ppo.history.append(record)

            if verbose and (i % 5 == 0 or i == 1):
                print(
                    f"  iter {i:4d}/{n_iterations}"
                    f" | eps={rollout_stats['episodes']:3d}"
                    f" | ret={rollout_stats['mean_ep_ret']:+.3f}"
                    f" | coll={collision_rate:.3f}"
                    f" | λ={self._lambda:.4f}"
                    f" | pg={update_stats['pg_loss']:+.4f}"
                )

            if save_dir is not None and (i % save_every == 0 or i == n_iterations):
                ckpt = save_dir / f"cmdp_iter{i:04d}.pt"
                self.ppo.ac.save(ckpt)
                if verbose:
                    print(f"    ↳ saved {ckpt}")

        return self.ppo.history
