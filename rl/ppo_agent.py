"""
Actor-Critic network for PPO on highway-v0.

Architecture
------------
Shared trunk: 25 → Linear(256) → Tanh → Linear(256) → Tanh

Three heads on top of the trunk:
    Actor head: Linear(256 → 5) → Categorical distribution
    Reward critic: Linear(256 → 1) → scalar V_r(s)
    Cost critic: Linear(256 → 1) → scalar V_c(s)

Design notes
------------
Tanh activations (not ReLU) in the trunk are standard for PPO actor-critics.
ReLU's dying-neuron problem is more damaging for the value head than for a
pure classifier because V(s) must remain finite and smooth throughout training.
Tanh bounds activations, which stabilises value estimates early in training when
the policy is still random.

The actor and critic share the trunk weights.  This means the feature
representation learned to predict V(s) is also used for π(a|s).  The
trade-off: shared features learn faster (fewer parameters, shared signal) but
a bad value gradient can corrupt the policy.  For highway-v0 (low-dimensional
obs, short episodes) the speed benefit outweighs the risk.

Warm-starting from DAgger
--------------------------
The trunk is initialized from the DAgger-5 MLPPolicy checkpoint.  MLPPolicy
uses the same 25→256→256 trunk with ReLU — we copy the weight tensors directly
and replace activation functions.  This requires overriding the trunk weights
after construction (see `load_actor_weights_from_mlp`).

Why warm-start?  PPO on a random policy in highway-v0 takes ~500 episodes to
stop crashing constantly.  Starting from DAgger-5 (collision=0.10) means PPO
explores from a policy that already keeps the car alive, so the Lagrange
multiplier λ doesn't immediately saturate.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from config.settings import N_ACTIONS
from envs.highway_wrapper import OBS_FEATURES, OBS_VEHICLES

OBS_DIM   = OBS_VEHICLES * len(OBS_FEATURES)  # 25


class ActorCritic(nn.Module):
    """
    Shared-trunk Actor-Critic for PPO.

    Parameters
    ----------
    obs_dim : int
        Flat observation dimension (default 25).
    n_actions : int
        Number of discrete actions (default 5).
    hidden : int
        Width of each hidden layer (default 256).
    device : str
        PyTorch device string.
    """

    def __init__(
        self,
        obs_dim:   int = OBS_DIM,
        n_actions: int = N_ACTIONS,
        hidden:    int = 256,
        device:    str = "cpu",
    ) -> None:
        super().__init__()
        self.device = torch.device(device)

        # Shared trunk — Tanh for stable value estimates
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )

        # Policy head: logits → Categorical distribution
        self.actor_head = nn.Linear(hidden, n_actions)
        # Reward value head: scalar V_r(s)
        self.reward_value_head = nn.Linear(hidden, 1)
        # Cost value head: scalar V_c(s)
        self.cost_value_head = nn.Linear(hidden, 1)

        self.to(self.device)

        # Orthogonal initialization (standard for PPO).
        # Smaller gain on actor head → initial policy close to uniform
        # (don't commit to any action before seeing data).
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.trunk:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor_head.weight,  gain=0.01)
        nn.init.zeros_(self.actor_head.bias)
        nn.init.orthogonal_(self.reward_value_head.weight, gain=1.0)
        nn.init.zeros_(self.reward_value_head.bias)
        nn.init.orthogonal_(self.cost_value_head.weight, gain=1.0)
        nn.init.zeros_(self.cost_value_head.bias)

    # ------------------------------------------------------------------
    # Core forward passes
    # ------------------------------------------------------------------

    def forward(self, obs: torch.Tensor) -> tuple[Categorical, torch.Tensor, torch.Tensor]:
        """
        Full forward pass.

        Parameters
        ----------
        obs : torch.Tensor, shape (B, obs_dim)

        Returns
        -------
        dist  : Categorical
            Action distribution π(·|obs).
        reward_value : torch.Tensor, shape (B,)
            Reward critic estimates V_r(obs).
        cost_value : torch.Tensor, shape (B,)
            Cost critic estimates V_c(obs).
        """
        features = self.trunk(obs)                       # (B, hidden)
        logits   = self.actor_head(features)             # (B, n_actions)
        reward_value = self.reward_value_head(features).squeeze(-1)  # (B,)
        cost_value   = self.cost_value_head(features).squeeze(-1)    # (B,)
        dist     = Categorical(logits=logits)
        return dist, reward_value, cost_value

    def get_action_and_value(
        self, obs: torch.Tensor, action: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Used inside the PPO update loop.

        Parameters
        ----------
        obs    : (B, obs_dim)
        action : (B,) int or None.  If None, samples from the distribution.

        Returns
        -------
        action    : (B,) int
        log_prob  : (B,) log π(action | obs)
        entropy   : (B,) H[π(·|obs)]
        reward_value : (B,)
        cost_value   : (B,)
        """
        dist, reward_value, cost_value = self.forward(obs)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy  = dist.entropy()
        return action, log_prob, entropy, reward_value, cost_value

    # ------------------------------------------------------------------
    # act() — matches the policy interface used by evaluate()
    # ------------------------------------------------------------------

    def act(self, obs: np.ndarray) -> int:
        """
        Greedy (argmax) action selection for a single observation.

        Greedy is used at *evaluation* time.  During *training* the PPO
        rollout uses sample() to encourage exploration.

        Parameters
        ----------
        obs : np.ndarray, shape (obs_dim,)

        Returns
        -------
        int  Action in [0, 4].
        """
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            dist, _, _ = self.forward(obs_t)
        return int(dist.probs.argmax(dim=-1).item())

    def act_stochastic(self, obs: np.ndarray) -> tuple[int, float]:
        """
        Stochastic action + log-prob for a single obs (used during rollout).

        Returns
        -------
        action   : int
        log_prob : float
        """
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            dist, _, _ = self.forward(obs_t)
            action   = dist.sample()
            log_prob = dist.log_prob(action)
        return int(action.item()), float(log_prob.item())

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def save(self, path: str | Path, history: list[dict] | None = None) -> None:
        """Save state_dict (and optional training history) to *path*."""
        payload: dict = {"state_dict": self.state_dict()}
        if history is not None:
            payload["history"] = history
        torch.save(payload, path)

    @classmethod
    def load(
        cls,
        path:      str | Path,
        obs_dim:   int = OBS_DIM,
        n_actions: int = N_ACTIONS,
        hidden:    int = 256,
        device:    str = "cpu",
    ) -> "ActorCritic":
        """Load from a previously saved checkpoint."""
        model = cls(obs_dim=obs_dim, n_actions=n_actions, hidden=hidden, device=device)
        data  = torch.load(path, map_location=device, weights_only=False)
        # Support both old format (bare state_dict) and new format (dict with key)
        if isinstance(data, dict) and "state_dict" in data:
            model.load_state_dict(data["state_dict"], strict=False)
        else:
            model.load_state_dict(data, strict=False)
        return model

    def load_actor_weights_from_mlp(self, mlp_path: str | Path) -> None:
        """
        Warm-start the trunk from an MLPPolicy (DAgger) checkpoint.

        MLPPolicy's state_dict keys:
            net.0.weight, net.0.bias   (Linear 25→256)
            net.2.weight, net.2.bias   (Linear 256→256)
            net.4.weight, net.4.bias   (Linear 256→5, actor logits)

        We copy layers 0 and 2 into trunk.0 and trunk.2 respectively.
        The actor head is copied from net.4.  The value heads are left at their
        orthogonal initialisations — we have no DAgger value targets to warm-start them.

        Note: MLPPolicy uses ReLU; we use Tanh.  The weight magnitudes are
        compatible (both ~√2/fan_in scale), but activations differ.  In practice
        Tanh is slightly more conservative early on, which is fine for PPO.
        """
        mlp_sd = torch.load(mlp_path, map_location=self.device)

        # Trunk layer 0
        self.trunk[0].weight.data.copy_(mlp_sd["net.0.weight"])
        self.trunk[0].bias.data.copy_(mlp_sd["net.0.bias"])
        # Trunk layer 2
        self.trunk[2].weight.data.copy_(mlp_sd["net.2.weight"])
        self.trunk[2].bias.data.copy_(mlp_sd["net.2.bias"])
        # Actor head (net.4 in MLPPolicy → actor_head here)
        self.actor_head.weight.data.copy_(mlp_sd["net.4.weight"])
        self.actor_head.bias.data.copy_(mlp_sd["net.4.bias"])
