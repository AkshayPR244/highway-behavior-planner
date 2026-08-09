"""
MLP policy for highway-v0.

Architecture: 25 → Linear(256) → ReLU → Linear(256) → ReLU → Linear(5 logits)

Design notes:
- Outputs raw logits, NOT softmax probabilities.
  CrossEntropyLoss expects logits; applying softmax first would double-apply
  the normalisation and corrupt gradients.
- act() is always greedy (argmax).  Greedy rollouts are sufficient for DAgger
  because compounding positional errors naturally push the policy off the
  expert's trajectory — stochastic sampling would add noise without benefit.
- torch.no_grad() in act() prevents autograd from tracking rollout tensors,
  saving memory and compute during data collection.
- obs is cast to float32 explicitly because gymnasium returns float32 but numpy
  can silently upcast; a float64 tensor causes a dtype mismatch in nn.Linear.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

from config.settings import N_ACTIONS
from envs.highway_wrapper import OBS_FEATURES, OBS_VEHICLES

OBS_DIM = OBS_VEHICLES * len(OBS_FEATURES)   # 25


class MLPPolicy(nn.Module):
    """
    Two-hidden-layer MLP that maps a flat kinematic observation to action logits.

    Parameters
    ----------
    obs_dim : int
        Dimension of the flat observation vector (default: 25).
    n_actions : int
        Number of discrete actions (default: 5).
    hidden : int
        Width of each hidden layer (default: 256).
    device : str
        PyTorch device string — "cpu", "mps", or "cuda".
        Defaults to "cpu" for numerical stability; switch to "mps" on M-series
        Macs once you've verified correctness on cpu.
    """

    def __init__(
        self,
        obs_dim: int = OBS_DIM,
        n_actions: int = N_ACTIONS,
        hidden: int = 256,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.device = torch.device(device)

        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
            # No activation here — logits are passed directly to CrossEntropyLoss
        )
        self.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return action logits for a batch of observations."""
        return self.net(x)

    def act(self, obs: np.ndarray) -> int:
        """
        Greedy action selection for a single observation.

        Parameters
        ----------
        obs : np.ndarray, shape (obs_dim,)
            Flat kinematic observation from the env.

        Returns
        -------
        int
            Chosen action index in [0, 4].
        """
        # Cast to float32 — env returns float32, but numpy can silently upcast
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            logits = self.forward(obs_t)          # shape (1, n_actions)
        return int(logits.argmax(dim=-1).item())  # .item() → plain Python int

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save model weights to *path* (state_dict only, not the full model)."""
        torch.save(self.state_dict(), path)

    @classmethod
    def load(
        cls,
        path: str | Path,
        obs_dim: int = OBS_DIM,
        n_actions: int = N_ACTIONS,
        hidden: int = 256,
        device: str = "cpu",
    ) -> "MLPPolicy":
        """
        Instantiate an MLPPolicy and load weights from *path*.

        Saving/loading state_dict (not the whole model) is more robust:
        the full-model pickle encodes the class definition path, so any
        rename or refactor breaks the load.  state_dict is just tensors.
        """
        policy = cls(obs_dim=obs_dim, n_actions=n_actions, hidden=hidden, device=device)
        policy.load_state_dict(torch.load(path, map_location=device))
        return policy
