"""
Counterfactual fault attribution for highway-v0 collisions.

After a collision, we ask: would this collision have occurred if the ego had
done IDLE (held its current speed and heading) instead of the action it took?

  - If yes → the crash was unavoidable given the pre-step physics. Fault: NPC.
  - If no  → the ego's action caused (or failed to prevent) the crash. Fault: EGO.

How it works
------------
Before every env.step(), call `snapshot_pre_step(env)` to capture the full
kinematic state of every vehicle.  If a crash is detected after the step,
call `classify_fault(snapshot, dt)` to run a single constant-velocity
forward projection of that saved state with ego doing IDLE and check whether
any NPC center comes within COLLISION_DIST of the ego center.

Constant-velocity projection
-----------------------------
All vehicles (ego and NPCs) are propagated as:

    pos_new = pos + speed * dt * [cos(heading), sin(heading)]

This is the same model used in the safety wrapper — cheap, no env interaction,
and accurate enough over 1-second policy steps at highway speeds.

Collision check
---------------
We use a circle approximation with radius VEHICLE_HALF_DIAG derived from
highway-env's default vehicle dimensions (LENGTH=5m, WIDTH=2m).  Two circles
collide when their center distance < 2 * VEHICLE_HALF_DIAG.

    VEHICLE_HALF_DIAG = sqrt((5/2)^2 + (2/2)^2) ≈ 2.69 m
    COLLISION_DIST = 2 * 2.69 ≈ 5.39 m

Fault categories
----------------
    "ego"       — IDLE would have avoided the crash; ego's action caused it
    "npc"       — IDLE would still have crashed; NPC was the unavoidable cause
    "ambiguous" — snapshot or geometry computation failed
"""
from __future__ import annotations

import math
import numpy as np
import gymnasium as gym

# highway-env default vehicle dimensions
_VEHICLE_LENGTH = 5.0   # m
_VEHICLE_WIDTH  = 2.0   # m

# Circle approximation: half-diagonal of vehicle bounding box
VEHICLE_HALF_DIAG: float = math.sqrt((_VEHICLE_LENGTH / 2) ** 2 + (_VEHICLE_WIDTH / 2) ** 2)

# Two vehicles collide when their circle centers are closer than this
COLLISION_DIST: float = 2 * VEHICLE_HALF_DIAG   # ≈ 5.39 m


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------

def snapshot_pre_step(env: gym.Env) -> dict:
    """
    Capture the kinematic state of every vehicle before a step is taken.

    Must be called BEFORE env.step() on every timestep so the snapshot is
    available for counterfactual analysis if that step causes a crash.

    Returns
    -------
    dict with keys:
        ego   : {"pos": ndarray(2,), "speed": float, "heading": float}
        npcs  : list of the same structure, one per non-ego vehicle
    """
    road = env.unwrapped.road
    ego  = env.unwrapped.vehicle
    return {
        "ego": {
            "pos":     ego.position.copy(),
            "speed":   float(ego.speed),
            "heading": float(ego.heading),
        },
        "npcs": [
            {
                "pos":     v.position.copy(),
                "speed":   float(v.speed),
                "heading": float(v.heading),
            }
            for v in road.vehicles
            if v is not ego
        ],
    }


# ---------------------------------------------------------------------------
# Constant-velocity forward projection
# ---------------------------------------------------------------------------

def _project(pos: np.ndarray, speed: float, heading: float, dt: float) -> np.ndarray:
    """Move a vehicle forward by dt seconds at constant speed and heading."""
    return pos + speed * dt * np.array([math.cos(heading), math.sin(heading)])


# ---------------------------------------------------------------------------
# Counterfactual check
# ---------------------------------------------------------------------------

def would_crash_with_idle(snapshot: dict, dt: float) -> bool:
    """
    Simulate one policy step from `snapshot` with ego doing IDLE.

    IDLE = constant speed, constant heading (no acceleration, no steering).
    Returns True if any NPC center comes within COLLISION_DIST of the ego center.
    """
    ego = snapshot["ego"]
    ego_pos_after = _project(ego["pos"], ego["speed"], ego["heading"], dt)

    for npc in snapshot["npcs"]:
        npc_pos_after = _project(npc["pos"], npc["speed"], npc["heading"], dt)
        dist = float(np.linalg.norm(ego_pos_after - npc_pos_after))
        if dist < COLLISION_DIST:
            return True
    return False


# ---------------------------------------------------------------------------
# Fault classification
# ---------------------------------------------------------------------------

def classify_fault(snapshot: dict, dt: float) -> str:
    """
    Classify who caused a collision that just occurred.

    Call this immediately after detecting `env.unwrapped.vehicle.crashed = True`.

    Parameters
    ----------
    snapshot : dict
        Output of `snapshot_pre_step()` captured before the crashing step.
    dt : float
        Policy step duration in seconds (1 / policy_frequency).

    Returns
    -------
    "ego"       — ego's action caused the crash (IDLE would have been safe)
    "npc"       — crash was unavoidable even with IDLE (NPC at fault)
    "ambiguous" — geometry check failed (snapshot missing or malformed)
    """
    try:
        if would_crash_with_idle(snapshot, dt):
            return "npc"
        return "ego"
    except Exception:
        return "ambiguous"
