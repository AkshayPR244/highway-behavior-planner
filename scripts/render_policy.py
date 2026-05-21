"""
Render a trained policy as an MP4 video.

Usage
-----
    # Show all available policies:
    python -m scripts.render_policy --list

    # Record PPO-CMDP (default):
    python -m scripts.render_policy

    # Record a specific policy:
    python -m scripts.render_policy --policy idm
    python -m scripts.render_policy --policy irl
    python -m scripts.render_policy --policy ppo_unc
    python -m scripts.render_policy --policy cmdp

    # Record multiple episodes, choose output path:
    python -m scripts.render_policy --policy cmdp --episodes 3 --out results/cmdp_demo.mp4

    # Record all four policies back-to-back into one video:
    python -m scripts.render_policy --all --out results/all_policies.mp4

Output
------
    results/<policy>_demo.mp4   — one episode per policy by default
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import imageio.v2 as iio

from envs.highway_wrapper import make_env
from optimizer.irl_optimizer import IRLPolicy
from policies.idm_expert import IDMExpert
from rl.ppo_agent import ActorCritic

RESULTS_DIR = Path(__file__).parent.parent / "results"

IRL_WEIGHTS   = RESULTS_DIR / "irl_weights.npy"
PPO_UNC_CKPT  = RESULTS_DIR / "ppo_unconstrained_final.pt"
CMDP_CKPT     = RESULTS_DIR / "cmdp_final.pt"

# Frames per second for the output video.
# highway-env renders at simulation_frequency (15 Hz) but policy steps at 1 Hz,
# so each policy step spans 15 rendered frames.  We record one frame per policy
# step and play back at 8 fps — a comfortable viewing speed.
VIDEO_FPS = 8


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_render_env(seed: int = 42):
    """Create an env configured for rgb_array rendering."""
    import gymnasium as gym
    from gymnasium.wrappers import FlattenObservation
    from envs.highway_wrapper import ENV_CONFIG

    cfg = dict(ENV_CONFIG)
    cfg["offscreen_rendering"] = False  # needed so render() returns pixels

    env = gym.make("highway-v0", render_mode="rgb_array", config=cfg)
    env = FlattenObservation(env)
    env.reset(seed=seed)
    return env


def _load_policy(name: str):
    """Return a callable policy object for the given name."""
    if name == "idm":
        # IDM needs the env — build a temporary one to get road access;
        # the render env will be passed separately during rollout.
        return None  # special-cased in record_episode

    if name == "irl":
        if not IRL_WEIGHTS.exists():
            raise FileNotFoundError(f"IRL weights not found: {IRL_WEIGHTS}\n"
                                    "Run: python -m optimizer.irl_optimizer")
        return IRLPolicy.load(IRL_WEIGHTS)

    if name == "ppo_unc":
        if not PPO_UNC_CKPT.exists():
            raise FileNotFoundError(f"PPO-unconstrained checkpoint not found: {PPO_UNC_CKPT}\n"
                                    "Run: python -m scripts.train_ppo --unconstrained-only")
        return ActorCritic.load(PPO_UNC_CKPT)

    if name == "cmdp":
        if not CMDP_CKPT.exists():
            raise FileNotFoundError(f"CMDP checkpoint not found: {CMDP_CKPT}\n"
                                    "Run: python -m scripts.train_ppo --cmdp-only")
        return ActorCritic.load(CMDP_CKPT)

    raise ValueError(f"Unknown policy: {name!r}. Choose from: idm, irl, ppo_unc, cmdp")


def _policy_label(name: str) -> str:
    return {
        "idm":     "IDM Expert",
        "irl":     "IRL Policy",
        "ppo_unc": "PPO-unconstrained",
        "cmdp":    "PPO-CMDP",
    }[name]


# ---------------------------------------------------------------------------
# Core recording loop
# ---------------------------------------------------------------------------

def record_episode(policy_name: str, seed: int = 42) -> tuple[list[np.ndarray], dict]:
    """
    Run one episode and collect RGB frames + episode stats.

    Returns
    -------
    frames : list of (H, W, 3) uint8 arrays
    stats  : dict with collision, goal, steps, total_reward
    """
    env = _make_render_env(seed=seed)
    obs, _ = env.reset(seed=seed)

    if policy_name == "idm":
        policy = IDMExpert(env)
        act_fn = lambda o: policy.act(o)
    else:
        policy = _load_policy(policy_name)
        act_fn = lambda o: policy.act(o)

    frames = []
    total_reward = 0.0
    steps = 0
    collided = False
    goal = False

    done = False
    while not done:
        frame = env.render()
        if frame is not None:
            frames.append(frame.astype(np.uint8))

        action = act_fn(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        steps += 1
        done = terminated or truncated

        if terminated and info.get("crashed", False):
            collided = True
        if truncated:
            goal = True

    # Capture the final frame
    frame = env.render()
    if frame is not None:
        frames.append(frame.astype(np.uint8))

    env.close()

    stats = {
        "collision": collided,
        "goal":      goal,
        "steps":     steps,
        "reward":    total_reward,
    }
    return frames, stats


def annotate_frames(
    frames: list[np.ndarray],
    label: str,
    stats: dict,
) -> list[np.ndarray]:
    """
    Burn a text overlay onto every frame using matplotlib.

    We do this in pure numpy/matplotlib to avoid a cv2 dependency.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from io import BytesIO

    annotated = []
    outcome = "COLLISION" if stats["collision"] else ("GOAL" if stats["goal"] else "TIMEOUT")
    outcome_color = "red" if stats["collision"] else "limegreen"

    for i, frame in enumerate(frames):
        h, w = frame.shape[:2]
        dpi = 80
        fig, ax = plt.subplots(figsize=(w / dpi, h / dpi), dpi=dpi)
        ax.imshow(frame)
        ax.axis("off")
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

        # Top-left: policy label
        ax.text(
            8, 12, label,
            fontsize=9, color="white", fontweight="bold",
            bbox=dict(facecolor="black", alpha=0.55, pad=2, boxstyle="round"),
        )
        # Top-right: step counter
        ax.text(
            w - 8, 12, f"step {i + 1:3d}/{stats['steps']}",
            fontsize=8, color="white", ha="right",
            bbox=dict(facecolor="black", alpha=0.45, pad=2, boxstyle="round"),
        )
        # Bottom-right: outcome (shown on last 8 frames)
        if i >= len(frames) - 8:
            ax.text(
                w / 2, h - 12, outcome,
                fontsize=11, color=outcome_color, fontweight="bold", ha="center",
                bbox=dict(facecolor="black", alpha=0.65, pad=3, boxstyle="round"),
            )

        buf = BytesIO()
        fig.savefig(buf, format="rgba", dpi=dpi)
        plt.close(fig)
        buf.seek(0)
        rgba = np.frombuffer(buf.read(), dtype=np.uint8).reshape(h, w, 4)
        annotated.append(rgba[:, :, :3])

    return annotated


# ---------------------------------------------------------------------------
# Multi-episode helpers
# ---------------------------------------------------------------------------

def record_policy(
    policy_name: str,
    n_episodes:  int,
    seed:        int,
    annotate:    bool,
) -> list[np.ndarray]:
    """Collect frames across n_episodes for one policy."""
    all_frames: list[np.ndarray] = []
    label = _policy_label(policy_name)

    for ep in range(n_episodes):
        ep_seed = seed + ep
        print(f"  [{label}] episode {ep + 1}/{n_episodes} (seed={ep_seed}) ...", end=" ")
        frames, stats = record_episode(policy_name, seed=ep_seed)
        outcome = "COLLISION" if stats["collision"] else ("GOAL ✓" if stats["goal"] else "timeout")
        print(f"{stats['steps']} steps — {outcome}")

        if annotate:
            frames = annotate_frames(frames, label, stats)

        # Add a short black pause between episodes (0.5 s = ~4 frames at 8 fps)
        if all_frames:
            h, w = frames[0].shape[:2]
            pause = [np.zeros((h, w, 3), dtype=np.uint8)] * 4
            all_frames.extend(pause)

        all_frames.extend(frames)

    return all_frames


def save_video(frames: list[np.ndarray], path: Path, fps: int = VIDEO_FPS) -> None:
    """Write frames to MP4 or GIF depending on the file extension."""
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.suffix.lower() == ".gif":
        # GIF: use palettisation for smaller file size
        with iio.get_writer(str(path), fps=fps, loop=0, palettesize=128) as writer:
            for frame in frames:
                writer.append_data(frame)
    else:
        # MP4 (default)
        with iio.get_writer(str(path), fps=fps, format="FFMPEG", codec="libx264",
                            output_params=["-pix_fmt", "yuv420p"]) as writer:
            for frame in frames:
                # Ensure even dimensions (libx264 requirement)
                h, w = frame.shape[:2]
                h = h - (h % 2)
                w = w - (w % 2)
                writer.append_data(frame[:h, :w])

    print(f"Saved → {path}  ({len(frames)} frames @ {fps} fps)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

POLICIES = ["idm", "irl", "ppo_unc", "cmdp"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a trained policy to video")
    parser.add_argument(
        "--policy", choices=POLICIES, default="cmdp",
        help="Which policy to render (default: cmdp)",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Render all four policies sequentially into one video",
    )
    parser.add_argument(
        "--episodes", type=int, default=1,
        help="Episodes to record per policy (default: 1)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Starting seed (default: 42)",
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help="Output path (default: results/<policy>_demo.mp4)",
    )
    parser.add_argument(
        "--no-annotate", action="store_true",
        help="Skip text overlays (faster, but no labels/outcome)",
    )
    parser.add_argument(
        "--fps", type=int, default=VIDEO_FPS,
        help=f"Output video FPS (default: {VIDEO_FPS})",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List available policies and their checkpoint status, then exit",
    )
    args = parser.parse_args()

    if args.list:
        print("Available policies:")
        for name in POLICIES:
            label = _policy_label(name)
            if name == "idm":
                status = "✅ always available (analytical)"
            elif name == "irl":
                status = "✅ ready" if IRL_WEIGHTS.exists() else "❌ missing — run: python -m optimizer.irl_optimizer"
            elif name == "ppo_unc":
                status = "✅ ready" if PPO_UNC_CKPT.exists() else "❌ missing — run: python -m scripts.train_ppo --unconstrained-only"
            elif name == "cmdp":
                status = "✅ ready" if CMDP_CKPT.exists() else "❌ missing — run: python -m scripts.train_ppo --cmdp-only"
            print(f"  {name:10s}  {label:22s}  {status}")
        return

    annotate = not args.no_annotate
    policies_to_render = POLICIES if args.all else [args.policy]

    print(f"Recording {args.episodes} episode(s) per policy, seed={args.seed}")
    print(f"Annotation: {'on' if annotate else 'off'}")
    print()

    all_frames: list[np.ndarray] = []

    for name in policies_to_render:
        frames = record_policy(
            policy_name=name,
            n_episodes=args.episodes,
            seed=args.seed,
            annotate=annotate,
        )
        # Add a longer separator between policies in --all mode
        if all_frames and args.all:
            h, w = frames[0].shape[:2]
            separator = [np.zeros((h, w, 3), dtype=np.uint8)] * args.fps  # 1 s black
            all_frames.extend(separator)
        all_frames.extend(frames)

    if args.out is not None:
        out_path = args.out
    elif args.all:
        out_path = RESULTS_DIR / "all_policies.mp4"
    else:
        out_path = RESULTS_DIR / f"{args.policy}_demo.mp4"

    print()
    save_video(all_frames, out_path, fps=args.fps)


if __name__ == "__main__":
    main()
