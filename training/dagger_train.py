"""
DAgger (Dataset Aggregation) training loop for highway-v0.

DAgger fixes BC's distribution-shift problem by iteratively collecting states
that the *learner* visits and re-labelling them with the expert oracle.  After
N iterations the training dataset covers both the expert's trajectory and the
off-distribution states the policy drifts into, eliminating the compounding
error that makes BC brittle.

Algorithm (Ross et al., AISTATS 2011):
    D ← collect_expert_rollouts()          # initial dataset
    π ← train_bc(D)                        # warm-start from BC
    for i = 1 … N:
        states ← rollout_policy(π)         # policy drives, states recorded
        labels ← query_expert(states)      # IDM labels every visited state
        D ← D ∪ {(state, label)}           # aggregate — never discard old data
        π ← train_bc(D)                    # retrain from scratch on full D

Key design choices:
- Policy rollouts use the LIVE env road state so IDMExpert.act() can read
  vehicle positions directly — obs alone is insufficient for IDM.
- We retrain from scratch each iteration (not fine-tune) to match the paper's
  theoretical guarantee and avoid catastrophic forgetting analysis.
- init_from_bc=True warm-starts iteration 1 from a BC checkpoint.  Without
  this, a random policy produces degenerate crash-heavy rollouts that add
  noise rather than signal to the dataset.
- A per-iteration evaluation table is printed so convergence is visible.

Usage:
    cd ~/highway-planner && source .venv/bin/activate
    python -m training.dagger_train
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from envs.highway_wrapper import make_env
from metrics.evaluator import evaluate, print_table
from policies.idm_expert import IDMExpert, collect_expert_rollouts
from policies.mlp_policy import MLPPolicy
from scenarios.adversarial import run_all_scenarios as _run_scenarios
from training.bc_train import ExpertDataset, split_rollouts, train_bc

RESULTS_DIR = Path(__file__).parent.parent / "results"

# Action names for the comparison table header
_POLICY_LABEL = "DAgger-{i}"


# ---------------------------------------------------------------------------
# Step 1: Roll out the current policy, collecting visited observations
# ---------------------------------------------------------------------------

def rollout_policy(
    policy: MLPPolicy,
    n_episodes: int,
    seed: int,
    beta: float = 0.0,
) -> list[dict]:
    """
    Drive the environment with *policy* and record every (obs, expert_action)
    pair encountered.

    The environment is stepped using the POLICY's action so that the states
    visited are exactly those the policy would reach — this is the DAgger
    requirement.  However, the action *recorded* in the returned rollout is
    the IDM expert's action for that state, not the policy's action.

    This is the oracle query step: we visit policy-induced states and
    immediately label them with what the expert would have done there.

    Parameters
    ----------
    policy : MLPPolicy
        Current learner policy.
    n_episodes : int
        Number of episodes to roll out.
    seed : int
        Base seed; episode i uses seed+i for reproducibility.

    Returns
    -------
    list[dict]
        Episode dicts with keys "observations", "actions", and "source".
        Actions are IDM expert labels (not necessarily the executed actions).
    """
    rng = np.random.default_rng(seed)
    env = make_env(seed=seed)
    expert = IDMExpert(env)
    rollouts = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        episode_obs, episode_acts = [], []
        done = False

        while not done:
            # Expert labels the CURRENT state (live road state, not obs alone)
            expert_action = expert.act(obs)

            # Record: where we are + what expert says to do here
            episode_obs.append(obs.copy())
            episode_acts.append(expert_action)

            # β-mixing (Fix 3): step with expert action with prob beta so
            # the early policy doesn't visit deeply off-distribution crash
            # states that produce contradictory corrective labels.
            if beta > 0.0 and rng.random() < beta:
                step_action = expert_action
            else:
                # Step with POLICY action — this is what makes it DAgger not BC.
                # The policy drives; it will drift off the expert's trajectory;
                # those drifted states get labelled and added to the dataset.
                step_action = policy.act(obs)

            obs, _, terminated, truncated, _ = env.step(step_action)
            done = terminated or truncated

        rollouts.append({
            "observations": np.array(episode_obs, dtype=np.float32),
            "actions":      np.array(episode_acts, dtype=np.int64),
            "source":       "rollout",
        })

    env.close()
    return rollouts


# ---------------------------------------------------------------------------
# Step 2: Aggregate datasets
# ---------------------------------------------------------------------------

def aggregate(existing: list[dict], new: list[dict]) -> list[dict]:
    """
    Return a new dataset list containing all episodes from both inputs.

    DAgger's guarantee depends on NEVER discarding old data — the aggregated
    dataset grows monotonically.  Replacing instead of aggregating would cause
    the policy to forget how to handle states from earlier iterations.
    """
    return existing + new


# ---------------------------------------------------------------------------
# Main DAgger loop
# ---------------------------------------------------------------------------

def train_dagger(
    n_dagger_iters: int = 5,
    n_expert_episodes: int = 50,
    n_rollout_episodes: int = 20,
    n_bc_epochs: int = 60,
    batch_size: int = 64,
    lr: float = 3e-4,
    val_frac: float = 0.2,
    patience: int = 8,
    eval_episodes: int = 20,
    seed: int = 42,
    init_from_bc: bool = True,
    device: str = "cpu",
    verbose: bool = True,
    beta_decay: float = 0.5,
    expert_weight_fraction: float = 0.5,
) -> MLPPolicy:
    """
    Train an MLPPolicy with DAgger on highway-v0.

    Parameters
    ----------
    n_dagger_iters : int
        Number of DAgger iterations after the BC warm-start.
    n_expert_episodes : int
        Expert episodes for the initial dataset D_0.
    n_rollout_episodes : int
        Policy episodes to roll out per DAgger iteration.
        Fewer needed than BC because policy-visited states are more
        informative (they are the hard cases the policy struggles with).
    n_bc_epochs : int
        Max BC epochs per iteration.  Fewer than standalone BC because
        early stopping fires sooner on larger datasets.
    batch_size : int
        Mini-batch size for BC retraining.
    lr : float
        Adam learning rate for BC retraining.
    val_frac : float
        Fraction of episodes held out for val loss / early stopping.
    patience : int
        Early-stopping patience per BC retraining run.
    eval_episodes : int
        Episodes used for the per-iteration metric evaluation.
    seed : int
        Base random seed.
    init_from_bc : bool
        If True, warm-start DAgger iteration 1 from a BC checkpoint.
        If False, start from random weights — produces noisier early rollouts.
    device : str
        PyTorch device string.
    beta_decay : float
        β-mixing decay base (Fix 3).  At iteration i the rollout mixes in
        expert actions with probability ``beta_decay ** i``.  Prevents the
        early policy from visiting crash-heavy off-distribution states.
        Set to 0.0 to disable (pure policy rollout from iter 1).
    expert_weight_fraction : float
        Target fraction of gradient signal from expert steps (Fix 1).
        A WeightedRandomSampler ensures expert steps contribute this fraction
        of every mini-batch regardless of how large the rollout dataset grows.
        Set to 0.0 to disable (uniform sampling).

    Returns
    -------
    MLPPolicy
        Best policy found across all DAgger iterations (lowest val loss).
    """
    # ------------------------------------------------------------------
    # 0. Initial expert dataset D_0
    # ------------------------------------------------------------------
    if verbose:
        print("=" * 60)
        print("DAgger Phase 2")
        print("=" * 60)
        print(f"\n[Init] Collecting {n_expert_episodes} expert episodes...")

    dataset = collect_expert_rollouts(n_episodes=n_expert_episodes, seed=seed)
    for ep in dataset:         # tag source so the weighted sampler can tell
        ep["source"] = "expert"  # expert steps from rollout steps apart
    _print_dataset_stats(dataset, label="D_0 (expert only)", verbose=verbose)

    # ------------------------------------------------------------------
    # 1. BC warm-start (iteration 0)
    # ------------------------------------------------------------------
    bc_ckpt = RESULTS_DIR / "bc_policy.pt"

    if init_from_bc and bc_ckpt.exists():
        if verbose:
            print(f"\n[Init] Loading BC warm-start from {bc_ckpt}")
        policy = MLPPolicy.load(bc_ckpt, device=device)
    else:
        if verbose:
            print(f"\n[Init] Training BC warm-start on D_0...")
        policy = train_bc(
            n_episodes=n_expert_episodes,
            n_epochs=n_bc_epochs,
            batch_size=batch_size,
            lr=lr,
            val_frac=val_frac,
            patience=patience,
            seed=seed,
            save_path=bc_ckpt,
            device=device,
            verbose=verbose,
        )

    # Evaluate BC baseline so the comparison table starts at iteration 0
    if verbose:
        print("\n[Iter 0 / BC baseline] Evaluating...")
    _eval_and_print(policy, eval_episodes, seed, label="BC (iter 0)", verbose=verbose)

    # ------------------------------------------------------------------
    # 2. DAgger iterations
    # ------------------------------------------------------------------
    best_policy = policy
    best_corridor_cr = float("inf")  # Fix 2: track dense_corridor collision rate

    for i in range(1, n_dagger_iters + 1):
        if verbose:
            print(f"\n{'=' * 60}")
            print(f"DAgger iteration {i}/{n_dagger_iters}")
            print(f"{'=' * 60}")

        # β schedule: iter 1 → beta_decay^1, iter 2 → beta_decay^2, …
        # decays toward 0 so later iterations become pure-policy rollouts
        beta_i = beta_decay ** i
        if verbose and beta_i > 1e-3:
            print(f"  β-mixing: expert action probability = {beta_i:.3f}")

        # --- 2a. Roll out current policy, label with expert ---
        if verbose:
            print(f"  Rolling out policy for {n_rollout_episodes} episodes...")
        new_rollouts = rollout_policy(
            policy=policy,
            n_episodes=n_rollout_episodes,
            seed=seed + i * 1000,   # different seed each iter for diverse states
            beta=beta_i,
        )

        # --- 2b. Aggregate ---
        dataset = aggregate(dataset, new_rollouts)
        _print_dataset_stats(dataset, label=f"D_{i} (after aggregation)", verbose=verbose)

        # --- 2c. Retrain from scratch on full dataset ---
        if verbose:
            print(f"  Retraining BC on D_{i} (from scratch)...")

        iter_ckpt = RESULTS_DIR / f"dagger_iter{i}_policy.pt"
        train_rollouts, val_rollouts = split_rollouts(dataset, val_frac=val_frac, seed=seed)
        train_ds = ExpertDataset(train_rollouts)
        val_ds   = ExpertDataset(val_rollouts)

        # Fix 1: weighted sampler keeps expert steps at ≥ expert_weight_fraction
        # of gradient signal regardless of how large the rollout dataset grows.
        sample_weights = _compute_sample_weights(train_rollouts, expert_weight_fraction)

        # Reuse train_bc internals by passing the already-collected dataset
        # directly rather than re-collecting — avoids duplicate env calls.
        policy = _retrain_on_dataset(
            train_ds=train_ds,
            val_ds=val_ds,
            n_epochs=n_bc_epochs,
            batch_size=batch_size,
            lr=lr,
            patience=patience,
            save_path=iter_ckpt,
            device=device,
            verbose=verbose,
            sample_weights=sample_weights,
        )

        # --- 2d. Evaluate ---
        if verbose:
            print(f"\n  [Iter {i}] Evaluating...")
        _eval_and_print(
            policy, eval_episodes, seed,
            label=f"DAgger-{i}", verbose=verbose,
        )

        # --- 2e. Scenario-based early stopping (Fix 2) ---
        # Run dense_corridor for 5 episodes — the scenario most sensitive to
        # lane-change thrashing.  If collision rate worsens, the policy has
        # overfit to rollout pathologies and further iterations will not help.
        _sc_env = make_env(seed=seed)
        sc_results = _run_scenarios(policy, _sc_env, n_episodes=5, seed=seed)
        _sc_env.close()
        corridor_cr = sc_results["dense_corridor"].collision_rate
        if verbose:
            print(f"  [Scenario] dense_corridor collision_rate={corridor_cr:.3f} "
                  f"(best so far: {best_corridor_cr:.3f})")
        if corridor_cr <= best_corridor_cr:
            best_corridor_cr = corridor_cr
            best_policy = policy
        else:
            if verbose:
                print(f"  [Scenario stop] Regression on dense_corridor — "
                      f"reverting to iter {i-1} policy and stopping early.")
            policy = best_policy
            break

    if verbose:
        print(f"\n{'=' * 60}")
        print("DAgger training complete.")
        print(f"Per-iteration checkpoints saved to: {RESULTS_DIR}/")
        print(f"{'=' * 60}")

    return policy


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compute_sample_weights(
    train_rollouts: list[dict],
    expert_fraction: float = 0.5,
) -> np.ndarray | None:
    """
    Compute per-step sample weights for WeightedRandomSampler (Fix 1).

    Expert steps receive weight ``expert_fraction / n_expert_steps``;
    rollout steps receive ``(1 - expert_fraction) / n_rollout_steps``.
    This guarantees expert data contributes *expert_fraction* of gradient
    signal every epoch regardless of how many rollout episodes have
    accumulated.

    Returns None when all data comes from one source (no mixing needed)
    or when expert_fraction is 0 (weighting disabled).
    """
    if expert_fraction <= 0.0:
        return None

    n_expert  = sum(len(ep["actions"]) for ep in train_rollouts
                    if ep.get("source") == "expert")
    n_rollout = sum(len(ep["actions"]) for ep in train_rollouts
                    if ep.get("source") != "expert")

    if n_expert == 0 or n_rollout == 0:
        return None  # single-source dataset; uniform sampling is fine

    w_expert  = expert_fraction / n_expert
    w_rollout = (1.0 - expert_fraction) / n_rollout
    weights: list[float] = []
    for ep in train_rollouts:
        w = w_expert if ep.get("source") == "expert" else w_rollout
        weights.extend([w] * len(ep["actions"]))
    return np.array(weights, dtype=np.float64)


def _retrain_on_dataset(
    train_ds: ExpertDataset,
    val_ds: ExpertDataset,
    n_epochs: int,
    batch_size: int,
    lr: float,
    patience: int,
    save_path: Path,
    device: str,
    verbose: bool,
    sample_weights: np.ndarray | None = None,
) -> MLPPolicy:
    """
    Run the BC training loop on pre-built Dataset objects.

    Separated from train_bc() to avoid re-collecting rollouts when we
    already have the aggregated dataset in memory.  Mirrors the training
    loop in bc_train.py exactly — same loss, same early stopping, same
    checkpoint discipline.

    If *sample_weights* is provided, a WeightedRandomSampler is used for
    the training DataLoader so expert steps can be upweighted relative to
    rollout steps (Fix 1).
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, WeightedRandomSampler
    from training.bc_train import _class_weights

    dev = torch.device(device)
    policy = MLPPolicy(device=device)  # fresh random weights each iteration
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    weights = _class_weights(train_ds).to(dev)
    criterion = nn.CrossEntropyLoss(weight=weights)

    if sample_weights is not None:
        sampler = WeightedRandomSampler(
            weights=torch.from_numpy(sample_weights),
            num_samples=len(train_ds),
            replacement=True,
        )
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    best_val_loss = float("inf")
    patience_ctr  = 0

    for epoch in range(1, n_epochs + 1):
        # Train
        policy.train()
        for obs_b, act_b in train_loader:
            obs_b, act_b = obs_b.to(dev), act_b.to(dev)
            optimizer.zero_grad()
            loss = criterion(policy(obs_b), act_b)
            loss.backward()
            optimizer.step()

        # Val
        policy.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for obs_b, act_b in val_loader:
                obs_b, act_b = obs_b.to(dev), act_b.to(dev)
                val_loss_sum += criterion(policy(obs_b), act_b).item() * len(act_b)
        val_loss = val_loss_sum / len(val_ds)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            policy.save(save_path)
            patience_ctr = 0
            marker = " *"
        else:
            patience_ctr += 1
            marker = ""

        if verbose:
            print(f"    epoch {epoch:3d}/{n_epochs} | val_loss={val_loss:.4f}{marker}")

        if patience_ctr >= patience:
            if verbose:
                print(f"    Early stopping at epoch {epoch}")
            break

    return MLPPolicy.load(save_path, device=device)


def _print_dataset_stats(dataset: list[dict], label: str, verbose: bool) -> None:
    """Print total step count and action distribution for a dataset."""
    if not verbose:
        return
    all_acts = np.concatenate([ep["actions"] for ep in dataset])
    n = len(all_acts)
    action_names = {0: "LANE_LEFT", 1: "IDLE", 2: "LANE_RIGHT", 3: "FASTER", 4: "SLOWER"}
    print(f"\n  Dataset {label}: {len(dataset)} episodes, {n} steps")
    for a, name in action_names.items():
        count = (all_acts == a).sum()
        print(f"    {name:12s}: {count:4d} ({100 * count / n:.1f}%)")


def _eval_and_print(
    policy: MLPPolicy,
    n_episodes: int,
    seed: int,
    label: str,
    verbose: bool,
) -> None:
    """Evaluate policy and print the metric table."""
    if not verbose:
        return
    env = make_env(seed=seed)
    results = evaluate(policy=policy, env=env, n_episodes=n_episodes, seed=seed)
    env.close()
    print_table(results, label=label)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    train_dagger(
        n_dagger_iters=5,
        n_expert_episodes=50,
        n_rollout_episodes=20,
        beta_decay=0.5,
        expert_weight_fraction=0.5,
        verbose=True,
    )
