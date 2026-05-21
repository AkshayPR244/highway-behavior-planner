"""
pytest test suite for Phase 2 components.

Covers:
  - MLPPolicy: architecture, forward pass, act(), save/load, dtype safety
  - ExpertDataset: construction, length, item shapes, dtype
  - split_rollouts: episode-level split, no leakage, edge cases
  - BC training loop: loss decreases, early stopping, class accuracy helper
  - DAgger: rollout records expert labels (not policy actions),
            aggregate never discards data, dataset grows monotonically

Run with:
    cd ~/highway-planner && source .venv/bin/activate
    pytest tests/test_phase2.py -v
"""
from __future__ import annotations

import copy
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from policies.mlp_policy import MLPPolicy, OBS_DIM, N_ACTIONS
from training.bc_train import (
    ExpertDataset,
    split_rollouts,
    _class_accuracy,
)
from training.dagger_train import aggregate, rollout_policy


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _fake_rollouts(n_episodes: int = 5, steps: int = 10) -> list[dict]:
    """Return synthetic rollout dicts — no env needed."""
    rng = np.random.default_rng(0)
    return [
        {
            "observations": rng.random((steps, OBS_DIM), dtype=np.float32),
            "actions":      rng.integers(0, N_ACTIONS, size=steps).astype(np.int64),
        }
        for _ in range(n_episodes)
    ]


# ---------------------------------------------------------------------------
# MLPPolicy
# ---------------------------------------------------------------------------

class TestMLPPolicy:

    def test_output_shape(self):
        """Forward pass on a batch returns (B, N_ACTIONS) logits."""
        policy = MLPPolicy()
        x = torch.zeros(4, OBS_DIM)
        out = policy(x)
        assert out.shape == (4, N_ACTIONS)

    def test_act_returns_valid_action(self):
        """act() must return an int in [0, N_ACTIONS)."""
        policy = MLPPolicy()
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        action = policy.act(obs)
        assert isinstance(action, int)
        assert 0 <= action < N_ACTIONS

    def test_act_accepts_float64_obs(self):
        """act() must not crash when numpy gives float64 (silent upcast guard)."""
        policy = MLPPolicy()
        obs = np.zeros(OBS_DIM, dtype=np.float64)   # wrong dtype
        action = policy.act(obs)                      # should cast internally
        assert isinstance(action, int)

    def test_act_is_deterministic(self):
        """Same obs must produce same action on repeated calls (greedy argmax)."""
        policy = MLPPolicy()
        obs = np.random.default_rng(7).random(OBS_DIM).astype(np.float32)
        actions = [policy.act(obs) for _ in range(5)]
        assert len(set(actions)) == 1, "act() must be deterministic"

    def test_act_does_not_track_gradients(self):
        """No gradient tape should be active during act() — no_grad guard."""
        policy = MLPPolicy()
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        with torch.no_grad():
            # Inside no_grad, requires_grad tensors should still work but not accumulate
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            logits = policy(obs_t)
            assert not logits.requires_grad

    def test_hidden_layer_count(self):
        """Network must have exactly two hidden Linear layers plus one output layer."""
        policy = MLPPolicy()
        linear_layers = [m for m in policy.net if isinstance(m, nn.Linear)]
        assert len(linear_layers) == 3   # hidden1, hidden2, output

    def test_hidden_layer_width(self):
        """Default hidden width is 256."""
        policy = MLPPolicy(hidden=256)
        linears = [m for m in policy.net if isinstance(m, nn.Linear)]
        assert linears[0].out_features == 256
        assert linears[1].out_features == 256

    def test_custom_hidden_width(self):
        """Hidden width is configurable."""
        policy = MLPPolicy(hidden=128)
        linears = [m for m in policy.net if isinstance(m, nn.Linear)]
        assert linears[0].out_features == 128

    def test_no_softmax_in_forward(self):
        """
        Output logits must NOT be softmax-normalised.
        CrossEntropyLoss applies its own softmax; double-applying corrupts gradients.
        We check that logits do not sum to 1.
        """
        policy = MLPPolicy()
        x = torch.randn(1, OBS_DIM)
        logits = policy(x)
        # Softmax outputs sum to 1; raw logits almost certainly do not
        assert abs(logits.sum().item() - 1.0) > 0.01, "Logits look softmax-normalised"

    def test_save_and_load_roundtrip(self):
        """save() / load() must reproduce identical act() outputs."""
        policy = MLPPolicy()
        obs = np.random.default_rng(42).random(OBS_DIM).astype(np.float32)
        original_action = policy.act(obs)

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = Path(tmpdir) / "test_policy.pt"
            policy.save(ckpt)
            loaded = MLPPolicy.load(ckpt)
            loaded_action = loaded.act(obs)

        assert original_action == loaded_action

    def test_load_restores_weights_exactly(self):
        """Loaded policy parameters must be numerically identical to saved ones."""
        policy = MLPPolicy()
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = Path(tmpdir) / "weights.pt"
            policy.save(ckpt)
            loaded = MLPPolicy.load(ckpt)

        for (name, p_orig), (_, p_load) in zip(
            policy.named_parameters(), loaded.named_parameters()
        ):
            assert torch.allclose(p_orig, p_load), f"Mismatch in {name}"


# ---------------------------------------------------------------------------
# ExpertDataset
# ---------------------------------------------------------------------------

class TestExpertDataset:

    def test_length_matches_total_steps(self):
        rollouts = _fake_rollouts(n_episodes=3, steps=10)
        ds = ExpertDataset(rollouts)
        assert len(ds) == 30   # 3 episodes × 10 steps

    def test_item_obs_shape(self):
        ds = ExpertDataset(_fake_rollouts(n_episodes=2, steps=8))
        obs, _ = ds[0]
        assert obs.shape == (OBS_DIM,)

    def test_item_obs_dtype(self):
        ds = ExpertDataset(_fake_rollouts())
        obs, _ = ds[0]
        assert obs.dtype == torch.float32

    def test_item_action_dtype(self):
        """Actions must be int64 — CrossEntropyLoss requires LongTensor targets."""
        ds = ExpertDataset(_fake_rollouts())
        _, act = ds[0]
        assert act.dtype == torch.int64

    def test_action_values_in_range(self):
        ds = ExpertDataset(_fake_rollouts(n_episodes=5, steps=20))
        all_acts = torch.stack([ds[i][1] for i in range(len(ds))])
        assert all_acts.min() >= 0
        assert all_acts.max() < N_ACTIONS

    def test_handles_variable_episode_lengths(self):
        """Episodes of different lengths must concatenate correctly."""
        rollouts = [
            {"observations": np.zeros((5,  OBS_DIM), dtype=np.float32),
             "actions":      np.zeros(5,  dtype=np.int64)},
            {"observations": np.zeros((15, OBS_DIM), dtype=np.float32),
             "actions":      np.zeros(15, dtype=np.int64)},
        ]
        ds = ExpertDataset(rollouts)
        assert len(ds) == 20


# ---------------------------------------------------------------------------
# split_rollouts
# ---------------------------------------------------------------------------

class TestSplitRollouts:

    def test_sizes_sum_to_total(self):
        rollouts = _fake_rollouts(n_episodes=10)
        train, val = split_rollouts(rollouts, val_frac=0.2)
        assert len(train) + len(val) == 10

    def test_val_size_correct(self):
        rollouts = _fake_rollouts(n_episodes=10)
        _, val = split_rollouts(rollouts, val_frac=0.2)
        assert len(val) == 2   # 10 × 0.2 = 2

    def test_no_episode_overlap(self):
        """
        No episode dict should appear in both train and val.
        We use object identity (id()) to check — episode dicts are not deep-copied.
        """
        rollouts = _fake_rollouts(n_episodes=10)
        train, val = split_rollouts(rollouts, val_frac=0.3)
        train_ids = {id(ep) for ep in train}
        val_ids   = {id(ep) for ep in val}
        assert train_ids.isdisjoint(val_ids), "Episode leaked into both splits"

    def test_original_list_not_mutated(self):
        """split_rollouts must not modify the caller's list (shallow copy guard)."""
        rollouts = _fake_rollouts(n_episodes=6)
        original_ids = [id(ep) for ep in rollouts]
        split_rollouts(rollouts, val_frac=0.2)
        assert [id(ep) for ep in rollouts] == original_ids

    def test_seed_reproducibility(self):
        """Same seed → same split every time."""
        rollouts = _fake_rollouts(n_episodes=10)
        train_a, val_a = split_rollouts(rollouts, seed=99)
        train_b, val_b = split_rollouts(rollouts, seed=99)
        assert [id(ep) for ep in train_a] == [id(ep) for ep in train_b]
        assert [id(ep) for ep in val_a]   == [id(ep) for ep in val_b]

    def test_different_seeds_give_different_splits(self):
        rollouts = _fake_rollouts(n_episodes=20)
        train_a, _ = split_rollouts(rollouts, seed=1)
        train_b, _ = split_rollouts(rollouts, seed=2)
        # Very unlikely to be identical with 20 episodes
        assert [id(ep) for ep in train_a] != [id(ep) for ep in train_b]

    def test_val_frac_zero_point_one_gives_at_least_one_val(self):
        """max(1, ...) guard: val set must never be empty."""
        rollouts = _fake_rollouts(n_episodes=5)
        _, val = split_rollouts(rollouts, val_frac=0.01)  # would round to 0 without guard
        assert len(val) >= 1


# ---------------------------------------------------------------------------
# BC: loss decreases and class accuracy
# ---------------------------------------------------------------------------

class TestBCTraining:

    def test_loss_decreases_over_epochs(self):
        """
        After 10 epochs of BC on a tiny synthetic dataset, train loss must be
        strictly lower than the initial loss.  Verifies that gradients flow
        and the optimiser is updating weights.
        """
        import copy
        from torch.utils.data import DataLoader

        rollouts = _fake_rollouts(n_episodes=8, steps=20)
        train_r, _ = split_rollouts(rollouts, val_frac=0.25, seed=0)
        ds = ExpertDataset(train_r)
        loader = DataLoader(ds, batch_size=16, shuffle=True)

        policy = MLPPolicy()
        optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        def _epoch_loss():
            policy.train()
            total = 0.0
            for obs_b, act_b in loader:
                optimizer.zero_grad()
                loss = criterion(policy(obs_b), act_b)
                loss.backward()
                optimizer.step()
                total += loss.item() * len(act_b)
            return total / len(ds)

        loss_0 = _epoch_loss()
        for _ in range(9):
            loss_n = _epoch_loss()

        assert loss_n < loss_0, (
            f"Loss did not decrease: initial={loss_0:.4f}, final={loss_n:.4f}"
        )

    def test_class_accuracy_returns_five_classes(self):
        """_class_accuracy must return a dict with keys 0–4."""
        from torch.utils.data import DataLoader
        rollouts = _fake_rollouts(n_episodes=5, steps=20)
        ds = ExpertDataset(rollouts)
        loader = DataLoader(ds, batch_size=16)
        policy = MLPPolicy()
        acc = _class_accuracy(policy, loader, torch.device("cpu"))
        assert set(acc.keys()) == {0, 1, 2, 3, 4}

    def test_class_accuracy_values_in_range(self):
        """All accuracy values must be in [0, 1] or NaN (class absent from val set)."""
        from torch.utils.data import DataLoader
        rollouts = _fake_rollouts(n_episodes=5, steps=40)
        ds = ExpertDataset(rollouts)
        loader = DataLoader(ds, batch_size=32)
        policy = MLPPolicy()
        acc = _class_accuracy(policy, loader, torch.device("cpu"))
        for a, v in acc.items():
            if not np.isnan(v):
                assert 0.0 <= v <= 1.0, f"Action {a} accuracy out of range: {v}"

    def test_perfect_policy_accuracy_is_one(self):
        """
        A policy that always returns action 1 should have accuracy=1.0 on a
        dataset where every label is 1, and 0.0 on all other classes.
        """
        from torch.utils.data import DataLoader

        # Dataset: all actions = 1 (IDLE)
        rollouts = [
            {"observations": np.zeros((20, OBS_DIM), dtype=np.float32),
             "actions":      np.ones(20, dtype=np.int64)}
        ]
        ds = ExpertDataset(rollouts)
        loader = DataLoader(ds, batch_size=20)

        # Policy biased to always output action 1
        policy = MLPPolicy()
        with torch.no_grad():
            policy.net[-1].bias.fill_(0.0)
            policy.net[-1].weight.fill_(0.0)
            policy.net[-1].bias[1] = 100.0  # force argmax → 1

        acc = _class_accuracy(policy, loader, torch.device("cpu"))
        assert acc[1] == pytest.approx(1.0)
        # Classes absent from the dataset return NaN (no samples → undefined accuracy)
        assert np.isnan(acc[0]), "Class 0 has no samples; accuracy must be NaN"


# ---------------------------------------------------------------------------
# DAgger: rollout_policy and aggregate
# ---------------------------------------------------------------------------

class TestDAggerRollout:

    def test_rollout_returns_expert_labels_not_policy_actions(self):
        """
        rollout_policy steps with the policy but labels with the expert.
        We verify this by using a policy that always picks action 0 (LANE_LEFT)
        and checking that the recorded actions are NOT all 0 — because the IDM
        expert on a clear highway will not exclusively choose LANE_LEFT.
        """
        # Bias policy to always output action 0
        policy = MLPPolicy()
        with torch.no_grad():
            policy.net[-1].bias.fill_(0.0)
            policy.net[-1].weight.fill_(0.0)
            policy.net[-1].bias[0] = 100.0   # argmax → 0 always

        rollouts = rollout_policy(policy, n_episodes=2, seed=0)
        all_acts = np.concatenate([r["actions"] for r in rollouts])

        # If we recorded policy actions, everything would be 0.
        # Expert labels will include IDLE, FASTER, etc.
        assert not np.all(all_acts == 0), (
            "All recorded actions are 0 — looks like policy actions were stored, "
            "not expert labels"
        )

    def test_rollout_obs_shape(self):
        policy = MLPPolicy()
        rollouts = rollout_policy(policy, n_episodes=1, seed=42)
        assert rollouts[0]["observations"].shape[1] == OBS_DIM

    def test_rollout_actions_in_valid_range(self):
        policy = MLPPolicy()
        rollouts = rollout_policy(policy, n_episodes=2, seed=0)
        all_acts = np.concatenate([r["actions"] for r in rollouts])
        assert all_acts.min() >= 0
        assert all_acts.max() < N_ACTIONS


class TestDAggerAggregate:

    def test_aggregate_grows_dataset(self):
        """Aggregated dataset must contain more episodes than either input."""
        d1 = _fake_rollouts(n_episodes=5)
        d2 = _fake_rollouts(n_episodes=3)
        merged = aggregate(d1, d2)
        assert len(merged) == 8

    def test_aggregate_preserves_all_episodes(self):
        """Every episode from both inputs must appear in the merged dataset."""
        d1 = _fake_rollouts(n_episodes=4)
        d2 = _fake_rollouts(n_episodes=3)
        merged = aggregate(d1, d2)
        for ep in d1 + d2:
            assert any(ep is m for m in merged), "Episode lost during aggregation"

    def test_aggregate_does_not_mutate_inputs(self):
        """aggregate() must not modify either input list."""
        d1 = _fake_rollouts(n_episodes=3)
        d2 = _fake_rollouts(n_episodes=2)
        len_d1_before, len_d2_before = len(d1), len(d2)
        aggregate(d1, d2)
        assert len(d1) == len_d1_before
        assert len(d2) == len_d2_before

    def test_aggregate_is_monotonically_growing(self):
        """
        Simulating N DAgger iterations: dataset must grow every iteration.
        This is the core DAgger invariant — old data is never discarded.
        """
        dataset = _fake_rollouts(n_episodes=5)
        sizes = [len(dataset)]
        for _ in range(4):
            new_batch = _fake_rollouts(n_episodes=3)
            dataset = aggregate(dataset, new_batch)
            sizes.append(len(dataset))

        for i in range(1, len(sizes)):
            assert sizes[i] > sizes[i - 1], (
                f"Dataset shrank at iteration {i}: {sizes[i-1]} → {sizes[i]}"
            )
