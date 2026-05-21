"""
pytest test suite for Phase 3 components — MaxEnt IRL.

Covers:
  - Feature extractor shape and dtype contracts
  - Feature extractor value correctness (action indicators, state interaction)
  - extract_batch vs extract consistency
  - extract_all_actions / extract_all_actions_batch shape contracts
  - IRLPolicy.act() validity
  - IRLPolicy save/load round-trip
  - Trained weight file shape and sign conventions

Run with:
    cd ~/highway-planner && source .venv/bin/activate
    pytest tests/test_phase3.py -v
"""
from __future__ import annotations

import numpy as np
import pytest

from optimizer.feature_extractor import (
    N_FEATURES,
    FEATURE_NAMES,
    extract,
    extract_batch,
    extract_all_actions,
    extract_all_actions_batch,
)
from optimizer.irl_optimizer import IRLPolicy

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_obs(
    ego_vx: float = 0.5,
    npc_present: bool = True,
    npc_x: float = 0.2,
    npc_y: float = 0.0,
    npc_vx: float = 0.0,
) -> np.ndarray:
    """
    Build a minimal 25-float observation.

    By default places one NPC directly ahead in the same lane.
    With npc_y=0, the NPC is in the same lane (|y| < 0.2 threshold).
    """
    obs = np.zeros(25, dtype=np.float32)
    # Row 0: ego
    obs[0] = 1.0          # presence
    obs[1] = 0.0          # x = 0 (ego at origin)
    obs[2] = 0.0          # y = 0
    obs[3] = ego_vx       # normalised longitudinal speed
    obs[4] = 0.0          # vy
    # Row 1: first NPC
    obs[5]  = 1.0 if npc_present else 0.0
    obs[6]  = npc_x
    obs[7]  = npc_y
    obs[8]  = npc_vx
    obs[9]  = 0.0
    # Rows 2–4: absent
    return obs


# ---------------------------------------------------------------------------
# TestFeatureExtractorShape
# ---------------------------------------------------------------------------

class TestFeatureExtractorShape:
    """extract() and extract_batch() must return the right shape and dtype."""

    def test_extract_shape(self):
        obs = _make_obs()
        phi = extract(obs, action=1)
        assert phi.shape == (N_FEATURES,), f"expected ({N_FEATURES},) got {phi.shape}"

    def test_extract_dtype(self):
        obs = _make_obs()
        phi = extract(obs, action=1)
        assert phi.dtype == np.float32

    @pytest.mark.parametrize("action", [0, 1, 2, 3, 4])
    def test_extract_shape_all_actions(self, action):
        obs = _make_obs()
        phi = extract(obs, action=action)
        assert phi.shape == (N_FEATURES,)

    def test_extract_batch_shape(self):
        batch = np.stack([_make_obs() for _ in range(10)])
        acts  = np.zeros(10, dtype=int)
        phi   = extract_batch(batch, acts)
        assert phi.shape == (10, N_FEATURES)

    def test_extract_batch_dtype(self):
        batch = np.stack([_make_obs() for _ in range(5)])
        acts  = np.ones(5, dtype=int)
        phi   = extract_batch(batch, acts)
        assert phi.dtype == np.float32

    def test_extract_batch_single(self):
        """Batch of size 1 should match extract()."""
        obs  = _make_obs(ego_vx=0.3)
        phi1 = extract(obs, action=3)
        phi2 = extract_batch(obs[np.newaxis], np.array([3]))
        np.testing.assert_allclose(phi1, phi2[0], rtol=1e-5)


# ---------------------------------------------------------------------------
# TestFeatureExtractorValues
# ---------------------------------------------------------------------------

class TestFeatureExtractorValues:
    """Check numeric correctness for known inputs."""

    # ---- action indicators ------------------------------------------------

    def test_idle_action_zeros_lc_and_accel(self):
        """IDLE (1): features 6 (lane_change) and 7 (accel) must be 0."""
        obs = _make_obs()
        phi = extract(obs, action=1)
        assert phi[6] == pytest.approx(0.0), "lane_change should be 0 for IDLE"
        assert phi[7] == pytest.approx(0.0), "accel should be 0 for IDLE"

    def test_idle_action_zeros_speed_faster_slower(self):
        """IDLE: speed×faster (0) and speed×slower (1) must be 0."""
        obs = _make_obs(ego_vx=0.7)
        phi = extract(obs, action=1)
        assert phi[0] == pytest.approx(0.0)
        assert phi[1] == pytest.approx(0.0)

    def test_idle_action_speed_idle_nonzero(self):
        """IDLE: speed×idle (2) must equal ego speed."""
        obs = _make_obs(ego_vx=0.6)
        phi = extract(obs, action=1)
        assert phi[2] == pytest.approx(0.6, abs=1e-5)

    def test_faster_action_lane_change_zero(self):
        """FASTER (3): feature 6 (lane_change) must be 0."""
        obs = _make_obs()
        phi = extract(obs, action=3)
        assert phi[6] == pytest.approx(0.0)

    def test_faster_action_accel_one(self):
        """FASTER (3): feature 7 (accel) must be 1."""
        obs = _make_obs()
        phi = extract(obs, action=3)
        assert phi[7] == pytest.approx(1.0)

    def test_lane_left_action_lane_change_one(self):
        """LANE_LEFT (0): feature 6 (lane_change) must be 1."""
        obs = _make_obs()
        phi = extract(obs, action=0)
        assert phi[6] == pytest.approx(1.0)

    def test_lane_right_action_lane_change_one(self):
        """LANE_RIGHT (2): feature 6 (lane_change) must be 1."""
        obs = _make_obs()
        phi = extract(obs, action=2)
        assert phi[6] == pytest.approx(1.0)

    def test_slower_action_accel_one(self):
        """SLOWER (4): feature 7 (accel) must be 1."""
        obs = _make_obs()
        phi = extract(obs, action=4)
        assert phi[7] == pytest.approx(1.0)

    # ---- state interaction ------------------------------------------------

    def test_speed_faster_equals_ego_vx(self):
        """speed×faster = ego_vx when action=FASTER."""
        obs = _make_obs(ego_vx=0.42)
        phi = extract(obs, action=3)
        assert phi[0] == pytest.approx(0.42, abs=1e-5)

    def test_speed_faster_zero_for_other_actions(self):
        """speed×faster = 0 when action is not FASTER."""
        obs = _make_obs(ego_vx=0.8)
        for action in [0, 1, 2, 4]:
            phi = extract(obs, action=action)
            assert phi[0] == pytest.approx(0.0), f"action {action}: feature 0 should be 0"

    def test_closeness_zero_when_no_npc(self):
        """No NPC present → closeness = 0 → features 3, 4, 5 all zero."""
        obs = _make_obs(npc_present=False)
        for action in [0, 1, 2, 3, 4]:
            phi = extract(obs, action=action)
            assert phi[3] == pytest.approx(0.0), f"action {action}: close×slower should be 0"
            assert phi[4] == pytest.approx(0.0), f"action {action}: close×lc should be 0"
            assert phi[5] == pytest.approx(0.0), f"action {action}: close×idle should be 0"

    def test_closeness_zero_when_npc_behind(self):
        """NPC behind ego (npc_x < 0) should not affect same-lane closeness."""
        obs = _make_obs(npc_x=-0.3)
        phi = extract(obs, action=4)  # SLOWER
        assert phi[3] == pytest.approx(0.0)

    def test_closeness_positive_when_npc_ahead(self):
        """NPC directly ahead in same lane → closeness > 0 → close×slower > 0 for SLOWER."""
        obs = _make_obs(npc_x=0.2, npc_y=0.0)  # same lane, ahead
        phi = extract(obs, action=4)
        assert phi[3] > 0.0

    def test_closeness_close_lc_for_lc_action(self):
        """NPC directly ahead → close×lc > 0 for LANE_LEFT and LANE_RIGHT."""
        obs = _make_obs(npc_x=0.15, npc_y=0.0)
        for action in [0, 2]:
            phi = extract(obs, action=action)
            assert phi[4] > 0.0, f"action {action}: close×lc should be > 0 with NPC ahead"

    def test_npc_in_adjacent_lane_not_counted_for_same_lane_closeness(self):
        """NPC in left adjacent lane (y_rel ~ -0.333) should not raise same-lane closeness."""
        obs = _make_obs(npc_x=0.15, npc_y=-0.333)  # adjacent lane
        phi = extract(obs, action=4)  # SLOWER
        assert phi[3] == pytest.approx(0.0), "Adjacent-lane NPC should not raise same-lane closeness"

    # ---- non-negativity ---------------------------------------------------

    @pytest.mark.parametrize("action", [0, 1, 2, 3, 4])
    def test_all_features_non_negative(self, action):
        """All features must be ≥ 0 (they are products of non-negative indicators)."""
        obs = _make_obs()
        phi = extract(obs, action=action)
        assert (phi >= 0.0).all(), f"action {action}: got negative feature(s): {phi}"


# ---------------------------------------------------------------------------
# TestExtractBatchConsistency
# ---------------------------------------------------------------------------

class TestExtractBatchConsistency:
    """extract_batch must agree with extract on every row."""

    def test_batch_matches_single_random(self):
        rng = np.random.default_rng(7)
        obs_batch = rng.random((32, 25)).astype(np.float32)
        actions   = rng.integers(0, 5, size=32)

        phi_batch = extract_batch(obs_batch, actions)
        for i in range(32):
            phi_single = extract(obs_batch[i], int(actions[i]))
            np.testing.assert_allclose(
                phi_single, phi_batch[i], atol=1e-5,
                err_msg=f"Mismatch at row {i}, action {actions[i]}"
            )


# ---------------------------------------------------------------------------
# TestExtractAllActions
# ---------------------------------------------------------------------------

class TestExtractAllActions:
    """extract_all_actions and extract_all_actions_batch shape contracts."""

    def test_extract_all_actions_shape(self):
        obs   = _make_obs()
        phi   = extract_all_actions(obs)
        assert phi.shape == (5, N_FEATURES), f"expected (5, {N_FEATURES}) got {phi.shape}"

    def test_extract_all_actions_batch_shape(self):
        obs_batch = np.stack([_make_obs() for _ in range(8)])
        phi       = extract_all_actions_batch(obs_batch)
        assert phi.shape == (8, 5, N_FEATURES)

    def test_extract_all_actions_matches_single(self):
        """Row a of extract_all_actions(obs) must equal extract(obs, a)."""
        obs = _make_obs(ego_vx=0.55)
        phi_all = extract_all_actions(obs)
        for action in range(5):
            phi_single = extract(obs, action)
            np.testing.assert_allclose(phi_all[action], phi_single, atol=1e-5)

    def test_feature_names_length_matches_n_features(self):
        assert len(FEATURE_NAMES) == N_FEATURES


# ---------------------------------------------------------------------------
# TestIRLPolicy
# ---------------------------------------------------------------------------

class TestIRLPolicy:
    """IRLPolicy API: act, save, load."""

    def _random_weights(self, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return rng.standard_normal(N_FEATURES).astype(np.float32)

    def test_act_returns_valid_action(self):
        weights = self._random_weights()
        policy  = IRLPolicy(weights)
        obs     = _make_obs()
        action  = policy.act(obs)
        assert action in range(5), f"act() returned {action}, not in [0,4]"

    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
    def test_act_valid_for_various_weights(self, seed):
        policy = IRLPolicy(self._random_weights(seed))
        obs    = _make_obs(ego_vx=0.4)
        assert policy.act(obs) in range(5)

    def test_act_deterministic(self):
        """Same obs + weights must give same action every call."""
        policy = IRLPolicy(self._random_weights(42))
        obs    = _make_obs()
        a1     = policy.act(obs)
        a2     = policy.act(obs)
        assert a1 == a2

    def test_save_load_roundtrip(self, tmp_path):
        weights = self._random_weights(99)
        policy  = IRLPolicy(weights)
        path    = tmp_path / "weights.npy"
        policy.save(path)
        loaded  = IRLPolicy.load(path)
        np.testing.assert_array_equal(policy.weights, loaded.weights)

    def test_save_load_act_consistent(self, tmp_path):
        weights = self._random_weights(7)
        policy  = IRLPolicy(weights)
        obs     = _make_obs(ego_vx=0.3)
        a_orig  = policy.act(obs)

        path   = tmp_path / "w.npy"
        policy.save(path)
        loaded = IRLPolicy.load(path)
        assert loaded.act(obs) == a_orig

    def test_weights_shape_after_load(self, tmp_path):
        weights = np.zeros(N_FEATURES, dtype=np.float32)
        IRLPolicy(weights).save(tmp_path / "z.npy")
        loaded = IRLPolicy.load(tmp_path / "z.npy")
        assert loaded.weights.shape == (N_FEATURES,)

    def test_accepts_torch_tensor(self):
        """IRLPolicy should accept a torch.Tensor as weights input."""
        import torch
        w = torch.randn(N_FEATURES)
        policy = IRLPolicy(w)
        assert policy.weights.shape == (N_FEATURES,)
        assert isinstance(policy.weights, np.ndarray)


# ---------------------------------------------------------------------------
# TestTrainedWeights
# ---------------------------------------------------------------------------

class TestTrainedWeights:
    """
    Verify properties of the saved irl_weights.npy produced by training.

    These tests are skipped if the weights file does not exist (useful in
    fresh CI environments where training has not been run).
    """

    WEIGHTS_PATH = "results/irl_weights.npy"

    @pytest.fixture
    def weights(self):
        import os
        if not os.path.exists(self.WEIGHTS_PATH):
            pytest.skip("irl_weights.npy not found — run irl_optimizer first")
        return np.load(self.WEIGHTS_PATH)

    def test_weights_shape(self, weights):
        assert weights.shape == (N_FEATURES,), \
            f"expected ({N_FEATURES},) got {weights.shape}"

    def test_weights_finite(self, weights):
        assert np.all(np.isfinite(weights)), "weights contain NaN or Inf"

    def test_speed_idle_positive(self, weights):
        """
        w[2] (speed×idle) > 0: idling while going fast is costly.
        The IDM expert prefers FASTER when going fast, never IDLE.
        """
        assert weights[2] > 0.0, \
            f"speed×idle weight should be > 0 (costly), got {weights[2]:.4f}"

    def test_close_lc_positive(self, weights):
        """
        w[4] (close×lc) > 0: lane-changing into a crowded lane is costly.
        The IDM expert avoids LC when there is a vehicle close ahead.
        """
        assert weights[4] > 0.0, \
            f"close×lc weight should be > 0 (costly), got {weights[4]:.4f}"

    def test_irl_policy_produces_valid_actions(self, weights):
        """End-to-end: loaded weights → IRLPolicy → valid action on real obs."""
        policy = IRLPolicy(weights)
        obs    = _make_obs(ego_vx=0.5, npc_x=0.25, npc_y=0.0)
        action = policy.act(obs)
        assert action in range(5)
