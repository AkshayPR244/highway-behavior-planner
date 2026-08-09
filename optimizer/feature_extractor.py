"""
Feature extractor for MaxEnt IRL on highway-v0.

Computes an 8-dimensional feature vector φ(s, a) from the flat observation
and discrete action.  These features are the inputs to the linear cost
function c(s, a; w) = w · φ(s, a) that IRL will learn.

Observation layout (25 floats, all normalised and relative to ego):
    row 0 (ego):   [presence, x=0, y=0, vx_norm, vy_norm]
    row 1–4 (NPCs): [presence, x_rel_norm, y_rel_norm, vx_rel_norm, vy_rel_norm]

    x   = longitudinal position (positive = ahead of ego)
    vx  = longitudinal speed (relative to ego)
    y   = lateral position
    absolute=False means ego is always at origin

Actions (DiscreteMetaAction):
    0 = LANE_LEFT   1 = IDLE   2 = LANE_RIGHT
    3 = FASTER      4 = SLOWER

Feature design — why 8 and not 4
---------------------------------
In single-step MaxEnt IRL the gradient for weight w_k is:
    ∇w_k = E_expert[φ_k] - E_{π_w}[φ_k]

If φ_k(s, a) is the *same* for all 5 actions at a given state s (i.e. it is
action-independent), then E_{π_w}[φ_k] = φ_k(s) always, which equals the
expert value — gradient is identically zero.  Weights for such features are
unidentifiable from single-step data.

The fix is **interaction features**: multiply each state feature by an action
indicator so the value differs across actions at the same state.

Feature index  Name                     Formula
    0  speed_faster     vx_ego × 𝟙[a=FASTER]  — fast while accelerating
    1  speed_slower     vx_ego × 𝟙[a=SLOWER]  — fast while braking
    2  speed_idle       vx_ego × 𝟙[a=IDLE]    — fast while maintaining
    3  close_slower     closeness × 𝟙[a=SLOWER] — braking because tailgating
    4  close_lc         closeness × 𝟙[a=LC]    — changing lanes when close
    5  close_idle       closeness × 𝟙[a=IDLE]  — idling when close
    6  lane_change      𝟙[a ∈ {LC}]           — pure lane-change cost
    7  accel            𝟙[a ∈ {FASTER,SLOWER}] — pure accel cost

The 12-feature extension (indices 8–11: lcL×left_gap, lcR×right_gap,
lcL×left_side, lcR×right_side) is documented but intentionally disabled in
the active extraction path because it fails in practice
with 50 expert episodes (~16 LC steps).  With only 16 LC steps driving
features 8–11, MaxEnt learns wrong-sign weights (+1.6 instead of -1.6
on the gap incentives) due to distribution-shift artefacts in the single-
step approximation.  The result is a regression to 0.300 collision vs.
0.050 with 8 features.  Activating the 12-feature model requires either
≥500 expert episodes (giving ~80 LC steps) or a trajectory-level MaxEnt
algorithm (Ziebart et al. 2008) that avoids the single-step approximation.

Features 6–7 are still action-only (no state interaction), but their gradients
are non-zero because the policy doesn't assign equal probability to all actions.
They capture the unconditional cost of the manoeuvre independent of state.

Design choices and pitfalls are documented inline.
"""
from __future__ import annotations

import numpy as np
import torch

# Observation structure constants (must match envs/highway_wrapper.py)
_N_VEHICLES  = 5   # rows in the (N, 5) kinematics matrix
_N_FEATURES  = 5   # [presence, x, y, vx, vy]
_IDX_PRESENCE = 0
_IDX_X        = 1  # longitudinal, relative to ego, normalised
_IDX_Y        = 2  # lateral
_IDX_VX       = 3  # longitudinal speed, normalised
_IDX_VY       = 4

# Actions that constitute a lane change
_LC_ACTIONS    = frozenset({0, 2})   # LANE_LEFT, LANE_RIGHT
# Actions that involve acceleration/deceleration (jerk proxy)
_ACCEL_ACTIONS = frozenset({3, 4})   # FASTER, SLOWER

_Y_SAME_THR  = 0.2    # |y_rel| < this → same lane

N_FEATURES = 8   # active feature count (see module docstring for 12-feature notes)
FEATURE_NAMES = [
    "speed×faster",     # 0
    "speed×slower",     # 1
    "speed×idle",       # 2
    "close×slower",     # 3
    "close×lc",         # 4
    "close×idle",       # 5
    "lane_change",      # 6
    "accel",            # 7
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract(obs: np.ndarray, action: int) -> np.ndarray:
    """
    Compute φ(s, a) ∈ ℝ¹² for a single (observation, action) pair.

    Parameters
    ----------
    obs : np.ndarray, shape (25,)
        Flat normalised Kinematics observation.
    action : int
        Discrete action in [0, 4].

    Returns
    -------
    np.ndarray, shape (12,), dtype float32
        Interaction feature vector.  See module docstring for index mapping.
    """
    mat = obs.reshape(_N_VEHICLES, _N_FEATURES)   # (5, 5)

    # ------------------------------------------------------------------ #
    # State factors (action-independent scalars)                          #
    # ------------------------------------------------------------------ #
    speed = float(np.clip(mat[0, _IDX_VX], 0.0, 1.0))

    npc_rows  = mat[1:, :]
    present   = npc_rows[:, _IDX_PRESENCE] > 0.5
    npc_x     = npc_rows[:, _IDX_X]
    npc_y     = npc_rows[:, _IDX_Y]

    # Same-lane closeness ahead (used by features 3–5)
    same_lane = np.abs(npc_y) < _Y_SAME_THR
    ahead     = npc_x > 0.0
    mask_same_ahead = present & same_lane & ahead
    if mask_same_ahead.any():
        min_x     = float(np.clip(npc_x[mask_same_ahead].min(), 0.0, 1.0))
        closeness = max(0.0, 1.0 - min_x)
    else:
        closeness = 0.0

    # ------------------------------------------------------------------ #
    # Action indicators                                                    #
    # ------------------------------------------------------------------ #
    is_lc_left  = 1.0 if action == 0 else 0.0
    is_lc_right = 1.0 if action == 2 else 0.0
    is_faster   = 1.0 if action == 3 else 0.0
    is_slower   = 1.0 if action == 4 else 0.0
    is_idle     = 1.0 if action == 1 else 0.0
    is_lc       = is_lc_left + is_lc_right
    is_accel    = is_faster + is_slower

    # ------------------------------------------------------------------ #
    # Interaction features                                                 #
    # ------------------------------------------------------------------ #
    return np.array([
        speed     * is_faster,      # 0
        speed     * is_slower,      # 1
        speed     * is_idle,        # 2
        closeness * is_slower,      # 3
        closeness * is_lc,          # 4
        closeness * is_idle,        # 5
        is_lc,                      # 6
        is_accel,                   # 7
        # Features 8–11 (LC incentive/safety) are computed but not included
        # in the active feature vector. Enable by changing N_FEATURES to 12
        # and collecting ≥500 expert episodes for sufficient LC signal.
        # is_lc_left  * left_gap,     # 8
        # is_lc_right * right_gap,    # 9
        # is_lc_left  * left_side_close,   # 10
        # is_lc_right * right_side_close,  # 11
    ], dtype=np.float32)


def extract_batch(
    obs_batch: np.ndarray,
    actions: np.ndarray,
) -> np.ndarray:
    """
    Vectorised feature extraction for a batch of (obs, action) pairs.

    Parameters
    ----------
    obs_batch : np.ndarray, shape (N, 25)
        Batch of flat normalised observations.
    actions : np.ndarray, shape (N,), dtype int
        Batch of discrete actions.

    Returns
    -------
    np.ndarray, shape (N, 12), dtype float32
    """
    n = len(obs_batch)
    mat = obs_batch.reshape(n, _N_VEHICLES, _N_FEATURES)  # (N, 5, 5)

    npc_x       = mat[:, 1:, _IDX_X]                          # (N, 4)
    npc_y       = mat[:, 1:, _IDX_Y]                          # (N, 4)
    npc_present = mat[:, 1:, _IDX_PRESENCE] > 0.5             # (N, 4)

    # Speed
    speed = np.clip(mat[:, 0, _IDX_VX], 0.0, 1.0).astype(np.float32)  # (N,)

    # Same-lane closeness ahead
    same_lane = np.abs(npc_y) < _Y_SAME_THR                   # (N, 4)
    ahead     = npc_x > 0.0                                    # (N, 4)
    valid_same = npc_present & same_lane & ahead
    filled_x_same = np.where(valid_same, npc_x, 2.0)
    min_x_same    = np.clip(filled_x_same.min(axis=1), 0.0, 1.0)
    closeness     = np.maximum(0.0, 1.0 - min_x_same).astype(np.float32)

    # Action indicators
    is_lc_left  = (actions == 0).astype(np.float32)
    is_lc_right = (actions == 2).astype(np.float32)
    is_faster   = (actions == 3).astype(np.float32)
    is_slower   = (actions == 4).astype(np.float32)
    is_idle     = (actions == 1).astype(np.float32)
    is_lc       = is_lc_left + is_lc_right
    is_accel    = is_faster + is_slower

    return np.stack([
        speed     * is_faster,           # 0
        speed     * is_slower,           # 1
        speed     * is_idle,             # 2
        closeness * is_slower,           # 3
        closeness * is_lc,               # 4
        closeness * is_idle,             # 5
        is_lc,                           # 6
        is_accel,                        # 7
        # Features 8–11 disabled (see N_FEATURES note above)
        # is_lc_left  * left_gap,          # 8
        # is_lc_right * right_gap,         # 9
        # is_lc_left  * left_side_close,   # 10
        # is_lc_right * right_side_close,  # 11
    ], axis=1)  # (N, 8)


def extract_dataset(rollouts: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract (features, actions) arrays from a list of episode dicts.

    Each dict has keys "observations" (T, 25) and "actions" (T,).

    Returns
    -------
    features : np.ndarray, shape (total_steps, 4), float32
    actions  : np.ndarray, shape (total_steps,), int64
    """
    all_obs = np.concatenate([ep["observations"] for ep in rollouts], axis=0)
    all_acts = np.concatenate([ep["actions"]      for ep in rollouts], axis=0)
    features = extract_batch(all_obs, all_acts)
    return features, all_acts


def extract_all_actions(obs: np.ndarray) -> np.ndarray:
    """
    Compute φ(s, a) for ALL 5 actions at a single state s.

    Useful for the IRL softmax over actions:
        π_w(a|s) ∝ exp(-w · φ(s, a))

    Parameters
    ----------
    obs : np.ndarray, shape (25,)

    Returns
    -------
    np.ndarray, shape (5, 4), dtype float32
        Row a is the feature vector for action a.
    """
    return np.stack([extract(obs, a) for a in range(5)], axis=0)


def extract_all_actions_batch(obs_batch: np.ndarray) -> np.ndarray:
    """
    Compute φ(s, a) for ALL 5 actions for a batch of states.

    Parameters
    ----------
    obs_batch : np.ndarray, shape (N, 25)

    Returns
    -------
    np.ndarray, shape (N, 5, 4), dtype float32
        phi[i, a] is the feature vector for sample i, action a.
    """
    n = len(obs_batch)
    mat = obs_batch.reshape(n, _N_VEHICLES, _N_FEATURES)  # (N, 5, 5)

    npc_x       = mat[:, 1:, _IDX_X]                          # (N, 4)
    npc_y       = mat[:, 1:, _IDX_Y]                          # (N, 4)
    npc_present = mat[:, 1:, _IDX_PRESENCE] > 0.5             # (N, 4)

    speed = np.clip(mat[:, 0, _IDX_VX], 0.0, 1.0).astype(np.float32)  # (N,)

    same_lane = np.abs(npc_y) < _Y_SAME_THR
    ahead     = npc_x > 0.0
    valid_same = npc_present & same_lane & ahead
    filled_x_same = np.where(valid_same, npc_x, 2.0)
    min_x_same    = np.clip(filled_x_same.min(axis=1), 0.0, 1.0)
    closeness     = np.maximum(0.0, 1.0 - min_x_same).astype(np.float32)

    # Layout: actions 0..4 = [LANE_LEFT, IDLE, LANE_RIGHT, FASTER, SLOWER]
    result = np.zeros((n, 5, N_FEATURES), dtype=np.float32)

    # speed×faster / speed×slower / speed×idle
    result[:, 3, 0] = speed
    result[:, 4, 1] = speed
    result[:, 1, 2] = speed

    # close×slower / close×lc / close×idle
    result[:, 4, 3] = closeness
    result[:, 0, 4] = closeness
    result[:, 2, 4] = closeness
    result[:, 1, 5] = closeness

    # lane_change and accel indicator features
    result[:, 0, 6] = 1.0
    result[:, 2, 6] = 1.0
    result[:, 3, 7] = 1.0
    result[:, 4, 7] = 1.0

    return result
