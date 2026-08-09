"""
Behavioural Cloning (BC) training loop for highway-v0.

BC is supervised learning on expert demonstrations:
    loss = CrossEntropy( policy(obs), expert_action )

The core limitation — distribution shift — is NOT fixed here.
At test time the policy drifts into states the expert never visited,
causing cascading errors.  DAgger (dagger_train.py) addresses this.

Training design decisions:
- Train/val split is done by EPISODE, not by step.  Splitting by step
  leaks correlated consecutive observations across the split, making
  val loss optimistically misleading.
- Early stopping on val loss with weight restoration prevents overfitting
  on the small (~2000 step) dataset.
- Per-class accuracy is logged every epoch to catch action-class collapse
  early (e.g. the model always predicting IDLE).

Usage:
    cd ~/highway-planner && source .venv/bin/activate
    python -m training.bc_train
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from config.settings import BEHAVIOR_ACTION_NAMES
from envs.highway_wrapper import make_env
from policies.idm_expert import collect_expert_rollouts
from policies.mlp_policy import MLPPolicy

# Action names for readable per-class accuracy output
ACTION_NAMES = {int(k): v for k, v in BEHAVIOR_ACTION_NAMES.items()}

RESULTS_DIR = Path(__file__).parent.parent / "results"


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ExpertDataset(Dataset):
    """
    Flat (obs, action) dataset built from a list of episode rollout dicts.

    Each rollout dict has:
        "observations": np.ndarray  shape (T, obs_dim)
        "actions":      np.ndarray  shape (T,)

    All episodes are concatenated into a single pool of transitions.
    The DataLoader handles batching and shuffling from there.
    """

    def __init__(self, rollouts: list[dict]) -> None:
        obs_list, act_list = [], []
        for ep in rollouts:
            obs_list.append(ep["observations"])
            act_list.append(ep["actions"])

        # Stack all episodes into single arrays
        all_obs = np.concatenate(obs_list, axis=0).astype(np.float32)  # (N, obs_dim)
        all_acts = np.concatenate(act_list, axis=0).astype(np.int64)   # (N,)

        # Convert once to tensors — faster than converting per __getitem__ call
        self.obs = torch.from_numpy(all_obs)
        self.actions = torch.from_numpy(all_acts)

    def __len__(self) -> int:
        return len(self.actions)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.obs[idx], self.actions[idx]


# ---------------------------------------------------------------------------
# Train / val split
# ---------------------------------------------------------------------------

def split_rollouts(
    rollouts: list[dict],
    val_frac: float = 0.2,
    seed: int = 0,
) -> tuple[list[dict], list[dict]]:
    """
    Split a list of episode rollouts into train and val sets.

    Split is by EPISODE to prevent consecutive-step correlation leaking
    across the split boundary.  With val_frac=0.2 and 50 episodes, 10
    episodes (~400 steps) are held out for validation.
    """
    rng = random.Random(seed)
    shuffled = rollouts[:]
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_frac))
    return shuffled[n_val:], shuffled[:n_val]   # train, val


# ---------------------------------------------------------------------------
# Per-class accuracy helper
# ---------------------------------------------------------------------------

def _class_accuracy(
    policy: MLPPolicy,
    loader: DataLoader,
    device: torch.device,
) -> dict[int, float]:
    """
    Compute per-action-class accuracy over *loader*.

    Returns a dict mapping action index → accuracy in [0, 1].
    Classes with no samples in *loader* return NaN.

    This is the class-collapse detector: if LANE_LEFT accuracy is 0%
    while IDLE is 99%, the network has learned to ignore rare actions.
    """
    policy.eval()
    correct = {a: 0 for a in range(5)}
    total   = {a: 0 for a in range(5)}

    with torch.no_grad():
        for obs_batch, act_batch in loader:
            obs_batch = obs_batch.to(device)
            act_batch = act_batch.to(device)
            preds = policy(obs_batch).argmax(dim=-1)
            for a in range(5):
                mask = act_batch == a
                total[a]   += mask.sum().item()
                correct[a] += (preds[mask] == a).sum().item()

    return {
        a: (correct[a] / total[a] if total[a] > 0 else float("nan"))
        for a in range(5)
    }


# ---------------------------------------------------------------------------
# Core training function
# ---------------------------------------------------------------------------

def _class_weights(dataset: ExpertDataset, n_actions: int = 5) -> torch.Tensor:
    """
    Compute inverse-frequency class weights for CrossEntropyLoss.

    weight[c] = total_samples / (n_classes * count[c])

    Rare classes (e.g. LANE_LEFT at ~2%) get high weight so a wrong
    prediction on them hurts proportionally more than a wrong prediction
    on IDLE (~50%).  Without this, the network collapses to never predicting
    lane changes — they're too cheap to get wrong under uniform loss.
    """
    counts = torch.zeros(n_actions)
    all_acts = dataset.actions  # already a LongTensor
    for a in range(n_actions):
        counts[a] = (all_acts == a).sum().float()
    # Replace any zero counts with 1 to avoid division by zero
    # (a class absent from training data gets weight 1, effectively neutral)
    counts = counts.clamp(min=1.0)
    weights = len(dataset) / (n_actions * counts)
    return weights


def train_bc(
    n_episodes: int = 50,
    n_epochs: int = 100,
    batch_size: int = 64,
    lr: float = 3e-4,
    val_frac: float = 0.2,
    patience: int = 10,
    seed: int = 42,
    save_path: str | Path = RESULTS_DIR / "bc_policy.pt",
    device: str = "cpu",
    verbose: bool = True,
) -> MLPPolicy:
    """
    Train an MLPPolicy via Behavioural Cloning on IDM expert rollouts.

    Parameters
    ----------
    n_episodes : int
        Number of expert episodes to collect.  More episodes = more data
        but quadratic training time per epoch.
    n_epochs : int
        Maximum training epochs.  Early stopping usually fires before this.
    batch_size : int
        Mini-batch size for SGD.  64 gives ~25 batches/epoch with 1600
        training steps — enough gradient noise for generalisation.
    lr : float
        Adam learning rate.  3e-4 is the standard starting point.
    val_frac : float
        Fraction of episodes held out for validation and early stopping.
    patience : int
        Stop training if val loss does not improve for this many epochs.
    seed : int
        Random seed for rollout collection and dataset split.
    save_path : Path
        Where to write the best checkpoint (state_dict).
    device : str
        PyTorch device.  Use "cpu" for stability; "mps" for M-series speed.
    verbose : bool
        Print per-epoch metrics when True.

    Returns
    -------
    MLPPolicy
        Policy loaded with the best val-loss weights found during training.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Collect expert rollouts
    # ------------------------------------------------------------------
    if verbose:
        print(f"Collecting {n_episodes} expert episodes...")
    rollouts = collect_expert_rollouts(n_episodes=n_episodes, seed=seed)
    total_steps = sum(len(r["actions"]) for r in rollouts)
    if verbose:
        print(f"  {total_steps} total transitions collected")

    # ------------------------------------------------------------------
    # 2. Split by episode, build DataLoaders
    # ------------------------------------------------------------------
    train_rollouts, val_rollouts = split_rollouts(rollouts, val_frac=val_frac, seed=seed)

    train_dataset = ExpertDataset(train_rollouts)
    val_dataset   = ExpertDataset(val_rollouts)

    # shuffle=True for train so each epoch sees a different batch ordering
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)

    if verbose:
        print(f"  Train: {len(train_dataset)} steps | Val: {len(val_dataset)} steps")

    # ------------------------------------------------------------------
    # 3. Instantiate policy, optimiser, loss
    # ------------------------------------------------------------------
    dev = torch.device(device)
    policy = MLPPolicy(device=device)

    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    # Inverse-frequency class weights: rare actions (LANE_LEFT, LANE_RIGHT)
    # get higher weight so the network cannot ignore them to minimise loss.
    # Without this, BC collapses to always predicting IDLE/FASTER/SLOWER.
    weights = _class_weights(train_dataset).to(dev)
    criterion = nn.CrossEntropyLoss(weight=weights)

    # ------------------------------------------------------------------
    # 4. Training loop with early stopping
    # ------------------------------------------------------------------
    best_val_loss = float("inf")
    patience_ctr  = 0

    for epoch in range(1, n_epochs + 1):

        # --- Train phase ---
        policy.train()
        train_loss_sum = 0.0
        for obs_batch, act_batch in train_loader:
            obs_batch = obs_batch.to(dev)
            act_batch = act_batch.to(dev)

            # zero_grad BEFORE forward — forgetting accumulates gradients
            # across batches and makes the loss explode
            optimizer.zero_grad()
            logits = policy(obs_batch)            # (B, 5)
            loss   = criterion(logits, act_batch) # scalar
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * len(act_batch)

        train_loss = train_loss_sum / len(train_dataset)

        # --- Val phase ---
        policy.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for obs_batch, act_batch in val_loader:
                obs_batch = obs_batch.to(dev)
                act_batch = act_batch.to(dev)
                logits = policy(obs_batch)
                loss   = criterion(logits, act_batch)
                val_loss_sum += loss.item() * len(act_batch)

        val_loss = val_loss_sum / len(val_dataset)

        # --- Early stopping ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            policy.save(save_path)
            patience_ctr  = 0
            improved_marker = " *"
        else:
            patience_ctr += 1
            improved_marker = ""

        # --- Logging ---
        if verbose:
            print(f"Epoch {epoch:3d}/{n_epochs} | "
                  f"train_loss={train_loss:.4f} | "
                  f"val_loss={val_loss:.4f}{improved_marker}")

        if patience_ctr >= patience:
            if verbose:
                print(f"Early stopping at epoch {epoch} (patience={patience})")
            break

    # ------------------------------------------------------------------
    # 5. Restore best weights and report per-class accuracy
    # ------------------------------------------------------------------
    # Restore the best checkpoint saved during training.
    policy = MLPPolicy.load(save_path, device=device)

    if verbose:
        acc = _class_accuracy(policy, val_loader, dev)
        print("\nPer-class accuracy on val set (class collapse check):")
        for a, name in ACTION_NAMES.items():
            pct = acc[a] * 100 if not np.isnan(acc[a]) else float("nan")
            print(f"  {name:12s}: {pct:.1f}%")
        print(f"\nBest val loss: {best_val_loss:.4f}")
        print(f"Checkpoint saved to: {save_path}")

    return policy


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    policy = train_bc(n_episodes=50, n_epochs=50, verbose=True)
