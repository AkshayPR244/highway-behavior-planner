"""
IRL-to-PPO reward bridge for Phase 4.

Converts the learned MaxEnt IRL cost weights into a step-level shaped
reward that PPO can optimise.

Cost vs. reward sign convention
---------------------------------
IRL learns a *cost* function:  c(s, a; w) = w · φ(s, a)
    high cost = undesirable action (IDM expert avoids it)
    low cost  = desirable action   (IDM expert prefers it)

PPO maximises *reward*, so we negate:
    r_IRL(s, a) = −w · φ(s, a) = −c(s, a; w)

We also blend in the raw environment reward at a small weight α (default 0.1).
The env reward is mostly speed-based and goal-completion; it keeps the policy
goal-directed when the IRL signal is ambiguous.

    r_total(s, a) = r_IRL(s, a) + α · r_env

The α parameter is intentionally small.  The IRL reward is the primary signal;
the env reward is a regulariser that prevents the policy from learning to idle
(which has low IRL cost but doesn't complete the goal).

Reward scale
------------
IRL weights are O(1)–O(5); the env reward is in [−1, 1].  Blending them
directly would mean the env term is invisible at α=0.1.  We therefore
normalise r_IRL to zero mean and unit variance using statistics computed
from the expert rollouts at construction time.  This makes the blend ratio
α interpretable: α=0.1 means the env reward contributes ~10% of the signal.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from optimizer.feature_extractor import N_FEATURES, extract
from optimizer.irl_optimizer import IRLPolicy

_RESULTS_DIR = Path(__file__).parent.parent / "results"
_DEFAULT_WEIGHTS = _RESULTS_DIR / "irl_weights.npy"


class IRLRewardShaper:
    """
    Wraps IRL weights and computes shaped reward at every step.

    Parameters
    ----------
    weights_path : path-like
        Path to the .npy file produced by irl_optimizer.
    env_reward_alpha : float
        Blend weight for the raw environment reward (default 0.1).
    reward_scale : float or None
        If given, divide r_IRL by this constant before blending.
        If None, scale is set to 1.0 (no normalisation).
        Pass the standard deviation of IRL rewards over the expert
        dataset to normalise (see `from_rollouts` classmethod).
    """

    def __init__(
        self,
        weights_path:      str | Path = _DEFAULT_WEIGHTS,
        env_reward_alpha:  float = 0.1,
        reward_scale:      float = 1.0,
    ) -> None:
        weights = np.load(weights_path)
        if len(weights) != N_FEATURES:
            raise ValueError(
                f"Weights have {len(weights)} features but extractor expects {N_FEATURES}."
            )
        self.weights          = weights.astype(np.float32)
        self.env_reward_alpha = env_reward_alpha
        self.reward_scale     = max(reward_scale, 1e-8)   # avoid div/0

    # ------------------------------------------------------------------
    # Primary interface
    # ------------------------------------------------------------------

    def shaped_reward(
        self,
        obs:        np.ndarray,
        action:     int,
        env_reward: float,
    ) -> float:
        """
        Compute the total shaped reward for one step.

        r_total = (−w · φ(s, a)) / scale  +  α · r_env

        Parameters
        ----------
        obs        : np.ndarray, shape (25,)
        action     : int in [0, 4]
        env_reward : float  Raw reward returned by the environment step.

        Returns
        -------
        float
        """
        phi   = extract(obs, action)                       # (N_FEATURES,)
        cost  = float(self.weights @ phi)                  # w · φ
        r_irl = -cost / self.reward_scale                  # negate and scale
        return r_irl + self.env_reward_alpha * env_reward

    def irl_cost(self, obs: np.ndarray, action: int) -> float:
        """Raw IRL cost c(s,a) = w·φ — useful for logging."""
        return float(self.weights @ extract(obs, action))

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_rollouts(
        cls,
        rollouts:         list[dict],
        weights_path:     str | Path = _DEFAULT_WEIGHTS,
        env_reward_alpha: float = 0.1,
    ) -> "IRLRewardShaper":
        """
        Build a shaper with reward_scale set to the std of IRL costs over
        the expert dataset.  This makes r_IRL ≈ N(0, 1) over expert steps,
        so env_reward_alpha is interpretable as a fraction.

        Parameters
        ----------
        rollouts : list of episode dicts with keys "observations", "actions"
        weights_path : path to irl_weights.npy
        env_reward_alpha : blend weight

        Returns
        -------
        IRLRewardShaper
        """
        weights = np.load(weights_path).astype(np.float32)
        costs   = []
        for ep in rollouts:
            obs_arr = ep["observations"]
            act_arr = ep["actions"]
            for obs, action in zip(obs_arr, act_arr):
                phi  = extract(obs, int(action))
                costs.append(float(weights @ phi))
        costs = np.array(costs, dtype=np.float32)
        scale = float(costs.std()) if costs.std() > 1e-8 else 1.0
        return cls(
            weights_path=weights_path,
            env_reward_alpha=env_reward_alpha,
            reward_scale=scale,
        )

    @classmethod
    def from_irl_policy(
        cls,
        irl_policy:       IRLPolicy,
        env_reward_alpha: float = 0.1,
        reward_scale:     float = 1.0,
    ) -> "IRLRewardShaper":
        """
        Build a shaper directly from an already-loaded IRLPolicy object,
        without needing a file path.  Useful in tests.
        """
        shaper = cls.__new__(cls)
        shaper.weights          = irl_policy.weights.astype(np.float32)
        shaper.env_reward_alpha = env_reward_alpha
        shaper.reward_scale     = max(reward_scale, 1e-8)
        return shaper
