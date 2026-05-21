"""
Maximum-entropy IRL optimizer for highway-v0.

Learns a linear cost function c(s, a; w) = w · φ(s, a) by maximising the
likelihood of expert demonstrations under a Boltzmann policy:

    π_w(a | s) = softmax_a(-w · φ(s, a))     [low cost = high probability]

The negative log-likelihood is equivalent to CrossEntropyLoss with logits
  logit(a) = -w · φ(s, a)

which PyTorch can differentiate directly.  No rollouts, no partition-function
enumeration — this is the single-step MaxEnt approximation (also called
"maximum causal entropy IRL in the immediate-reward limit").

Gradient (what Adam computes):
    ∇_w L = E_expert[φ] - E_{π_w}[φ]
    → push w toward features the expert uses more than the current policy.

Usage:
    from policies.idm_expert import collect_expert_rollouts
    from optimizer.irl_optimizer import train_irl, IRLPolicy

    rollouts = collect_expert_rollouts(n_episodes=50, seed=42)
    weights, history = train_irl(rollouts)
    policy = IRLPolicy(weights)
    action = policy.act(obs)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from optimizer.feature_extractor import (
    FEATURE_NAMES,
    N_FEATURES,
    extract_all_actions_batch,
    extract_dataset,
)

RESULTS_DIR = Path(__file__).parent.parent / "results"


# ---------------------------------------------------------------------------
# IRL Policy
# ---------------------------------------------------------------------------

class IRLPolicy:
    """
    Greedy policy derived from a learned linear cost function.

    Acts by computing the cost for all 5 actions and choosing the cheapest:
        a* = argmin_a  w · φ(s, a)

    Parameters
    ----------
    weights : np.ndarray or torch.Tensor, shape (4,)
        Learned cost weights.  Positive weight = feature is costly.
    """

    def __init__(self, weights: np.ndarray | torch.Tensor) -> None:
        if isinstance(weights, torch.Tensor):
            weights = weights.detach().cpu().numpy()
        self.weights = np.array(weights, dtype=np.float32)

    def act(self, obs: np.ndarray) -> int:
        """
        Select the action with minimum cost at *obs*.

        Parameters
        ----------
        obs : np.ndarray, shape (25,) or (25,) float32/float64

        Returns
        -------
        int
            Discrete action in [0, 4].
        """
        obs = np.asarray(obs, dtype=np.float32)
        phi_all = extract_all_actions_batch(obs[np.newaxis])[0]   # (5, 4)
        costs   = phi_all @ self.weights                           # (5,)
        return int(np.argmin(costs))

    def save(self, path: str | Path) -> None:
        """Save weights to a .npy file."""
        np.save(path, self.weights)

    @classmethod
    def load(cls, path: str | Path) -> "IRLPolicy":
        """Load weights from a .npy file."""
        weights = np.load(path)
        return cls(weights)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_irl(
    rollouts: list[dict],
    n_epochs: int = 500,
    lr: float = 5e-2,
    l2_reg: float = 1e-3,
    feature_l2: np.ndarray | None = None,
    patience: int = 30,
    save_path: Path | None = None,
    device: str = "cpu",
    verbose: bool = True,
) -> tuple[np.ndarray, list[float]]:
    """
    Learn cost weights w ∈ ℝ⁸ from expert rollouts via MaxEnt IRL.

    Parameters
    ----------
    rollouts : list[dict]
        Expert episode dicts with keys "observations" (T, 25) and
        "actions" (T,).  Typically the output of collect_expert_rollouts().
    n_epochs : int
        Maximum number of gradient steps.
    lr : float
        Adam learning rate.  Higher than BC (5e-2 vs 3e-4) because the
        parameter space is only 8D — we can take larger steps.
    l2_reg : float
        Baseline L2 regularisation coefficient applied to all weights.
        Prevents w from growing unboundedly (same action ranking but
        sharper distribution with larger ||w||).
    feature_l2 : np.ndarray of shape (N_FEATURES,) or None
        Per-feature L2 strengths.  If provided, weight k is regularised
        with ``feature_l2[k]`` instead of ``l2_reg``.

        Use this to soften specific features without distorting others.
        Example — soften lane_change (index 6) and accel (index 7)::

            feature_l2 = np.full(N_FEATURES, 1e-3)  # baseline
            feature_l2[6] = 0.5   # strong pull toward w[6]=0 → less LC cost
            feature_l2[7] = 0.1   # mild pull on accel

        Higher λ_k → stronger shrinkage → weight closer to 0 → lower cost
        for that feature → policy uses that action more freely.
    patience : int
        Early stopping: stop if loss doesn't improve for this many epochs.
    save_path : Path or None
        If provided, save the learned weights as a .npy file here.
    device : str
        PyTorch device string.
    verbose : bool
        Print loss every 50 epochs.

    Returns
    -------
    weights : np.ndarray, shape (4,)
        Learned cost weights.
    history : list[float]
        Per-epoch NLL loss (before regularisation).
    """
    dev = torch.device(device)

    # ------------------------------------------------------------------ #
    # 1. Build (logits_table, expert_actions) tensors                      #
    # ------------------------------------------------------------------ #
    # We precompute φ(s, a) for every (step, action) pair upfront.
    # Shape: (N, 5, 4)  — N steps, 5 actions, 4 features.
    # This lets the inner loop be a single batched matmul.
    all_obs  = np.concatenate([ep["observations"] for ep in rollouts])  # (N, 25)
    all_acts = np.concatenate([ep["actions"]      for ep in rollouts])  # (N,)

    phi_all  = extract_all_actions_batch(all_obs)  # (N, 5, 4), float32
    phi_all_t = torch.tensor(phi_all, dtype=torch.float32, device=dev)  # (N, 5, 4)
    acts_t    = torch.tensor(all_acts, dtype=torch.long, device=dev)    # (N,)

    # Per-feature L2 vector: broadcast scalar l2_reg if feature_l2 not given
    if feature_l2 is None:
        l2_vec = torch.full((N_FEATURES,), l2_reg, dtype=torch.float32, device=dev)
    else:
        l2_vec = torch.tensor(
            np.asarray(feature_l2, dtype=np.float32), device=dev
        )
        assert l2_vec.shape == (N_FEATURES,), \
            f"feature_l2 must have length {N_FEATURES}, got {len(feature_l2)}"

    N = len(all_obs)
    if verbose:
        print(f"IRL: {N} expert steps, {N_FEATURES} features, {n_epochs} max epochs")
        print(f"     lr={lr}  l2_reg={l2_reg}  patience={patience}")
        if feature_l2 is not None:
            overrides = [
                f"{FEATURE_NAMES[i]}={feature_l2[i]:.3g}"
                for i in range(N_FEATURES)
                if abs(feature_l2[i] - l2_reg) > 1e-9
            ]
            if overrides:
                print(f"     per-feature overrides: {', '.join(overrides)}")

    # ------------------------------------------------------------------ #
    # 2. Initialise weights                                                #
    # ------------------------------------------------------------------ #
    # Small random init — zero init would work too (gradient is non-zero
    # at w=0 because expert features != uniform-policy expected features).
    w = nn.Parameter(torch.randn(N_FEATURES, device=dev) * 0.1)
    optimizer = torch.optim.Adam([w], lr=lr)
    criterion = nn.CrossEntropyLoss()  # expects logits (N, 5), targets (N,)

    # ------------------------------------------------------------------ #
    # 3. Training loop                                                     #
    # ------------------------------------------------------------------ #
    # logits for action a at step t: -w · φ(s_t, a)
    # shape: (N, 5) = (N, 5, 4) @ (4,) = (N, 5) using einsum/matmul.
    #
    # CrossEntropyLoss(logits, targets) = -log softmax(logits)[targets]
    # = -logit[a*] + log Σ_a exp(logit[a])
    # which is exactly the MaxEnt NLL we derived above.

    history: list[float] = []
    best_loss    = float("inf")
    best_w       = w.data.clone()
    patience_ctr = 0

    for epoch in range(1, n_epochs + 1):
        optimizer.zero_grad()

        # Compute logits: (N, 5, 4) @ (4,) → (N, 5)
        # Negative because low cost = high logit = high probability
        logits = -(phi_all_t @ w)   # (N, 5)

        nll   = criterion(logits, acts_t)
        reg   = (l2_vec * w ** 2).sum()   # per-feature L2
        loss  = nll + reg

        loss.backward()
        optimizer.step()

        history.append(nll.item())

        # Early stopping on NLL (not regularised loss)
        if nll.item() < best_loss:
            best_loss    = nll.item()
            best_w       = w.data.clone()
            patience_ctr = 0
            marker = " *"
        else:
            patience_ctr += 1
            marker = ""

        if verbose and (epoch % 50 == 0 or epoch == 1):
            print(f"  epoch {epoch:4d}/{n_epochs} | nll={nll.item():.4f}{marker}")

        if patience_ctr >= patience:
            if verbose:
                print(f"  Early stopping at epoch {epoch}")
            break

    # Restore best weights
    w.data.copy_(best_w)
    weights = best_w.cpu().numpy()

    if verbose:
        print(f"\nLearned weights:")
        for name, val in zip(FEATURE_NAMES, weights):
            sign = "↓ cost" if val < 0 else "↑ cost"
            print(f"  {name:12s}: {val:+.4f}  ({sign})")

    if save_path is not None:
        save_path = Path(save_path)
        np.save(save_path, weights)
        if verbose:
            print(f"\nWeights saved to {save_path}")

    return weights, history


# ---------------------------------------------------------------------------
# Convenience: print a comparison of expert vs. IRL-policy feature usage
# ---------------------------------------------------------------------------

def print_feature_match(rollouts: list[dict], weights: np.ndarray) -> None:
    """
    Print how well the IRL policy's expected feature counts match the expert.

    Under perfect MaxEnt IRL the two columns should be equal (that's the
    optimality condition ∇L = 0).  A large discrepancy indicates the
    feature is mis-specified or the learning hasn't converged.

    Parameters
    ----------
    rollouts : list[dict]
        Expert rollout episodes.
    weights : np.ndarray, shape (4,)
        Learned cost weights.
    """
    all_obs  = np.concatenate([ep["observations"] for ep in rollouts])
    all_acts = np.concatenate([ep["actions"]      for ep in rollouts])

    # Expert feature means
    phi_expert = extract_all_actions_batch(all_obs)   # (N, 5, 4)
    expert_feats = phi_expert[np.arange(len(all_acts)), all_acts, :]  # (N, 4)
    expert_mean  = expert_feats.mean(axis=0)

    # IRL policy feature means (expected under π_w)
    logits    = -(phi_expert @ weights)                 # (N, 5)
    probs     = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs    /= probs.sum(axis=1, keepdims=True)        # (N, 5) softmax
    # E[φ] = Σ_a π(a|s) φ(s,a)  for each step, then average over steps
    policy_mean = (probs[:, :, np.newaxis] * phi_expert).sum(axis=1).mean(axis=0)

    print("\nFeature matching (convergence check):")
    print(f"  {'Feature':12s}  {'Expert':>8s}  {'IRL policy':>10s}  {'Diff':>8s}")
    print(f"  {'-'*12}  {'-'*8}  {'-'*10}  {'-'*8}")
    for name, e, p in zip(FEATURE_NAMES, expert_mean, policy_mean):
        diff = e - p
        print(f"  {name:12s}  {e:8.4f}  {p:10.4f}  {diff:+8.4f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from policies.idm_expert import collect_expert_rollouts

    print("=" * 60)
    print("MaxEnt IRL — Phase 3")
    print("=" * 60)

    rollouts = collect_expert_rollouts(n_episodes=50, seed=42)

    # Per-feature L2: apply stronger regularisation on the two action-only
    # features (lane_change idx=6, accel idx=7) so their weights are pulled
    # toward 0 — reducing unconditional action cost and allowing more
    # dynamic lane changes while keeping the physics-grounded interaction
    # weights free to grow.
    feat_l2 = np.full(N_FEATURES, 1e-3, dtype=np.float32)   # baseline
    feat_l2[6] = 0.5    # lane_change: strong shrinkage → less LC penalty
    feat_l2[7] = 0.05   # accel: mild shrinkage

    weights, history = train_irl(
        rollouts,
        feature_l2=feat_l2,
        save_path=RESULTS_DIR / "irl_weights.npy",
    )

    print_feature_match(rollouts, weights)

    print("\nDone. Run scripts/eval_irl.py for full metric comparison.")
