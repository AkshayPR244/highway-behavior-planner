"""
Plot PPO and CMDP training curves from saved checkpoints.

Reads the training history embedded in the checkpoint .pt files and produces:
  results/training_curves.png  — 3-panel figure:
    Top:    collision count per rollout (unconstrained vs. CMDP)
    Middle: mean episode return per rollout (unconstrained vs. CMDP)
    Bottom: Lagrange multiplier λ over iterations (CMDP only)

Usage
-----
    python -m scripts.plot_training_curves

    # Custom output path:
    python -m scripts.plot_training_curves --out results/my_curves.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import torch

RESULTS_DIR = Path(__file__).parent.parent / "results"
PPO_UNC_DIR = RESULTS_DIR / "ppo_unconstrained"
CMDP_DIR    = RESULTS_DIR / "cmdp"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_history(ckpt_dir: Path, final_ckpt: Path) -> list[dict] | None:
    """
    Reconstruct per-iteration history from the final checkpoint.

    PPOTrainer.train() returns history and ActorCritic.save() saves the
    actor-critic weights only.  The training history is stored alongside
    the checkpoint as <name>_history.npy when available, or embedded in
    the checkpoint dict under key 'history'.
    """
    history_path = final_ckpt.with_name(final_ckpt.stem + "_history.npz")
    if history_path.exists():
        data = np.load(history_path, allow_pickle=True)
        return data["history"].tolist()

    # Fall back: scan per-iteration checkpoints for embedded history
    iters = sorted(ckpt_dir.glob("*.pt")) if ckpt_dir.exists() else []
    if not iters:
        return None

    # Each checkpoint has only weights — we can't reconstruct history without
    # the history npz.  Signal clearly.
    return None


def _load_history_from_pt(path: Path) -> list[dict] | None:
    """Try to load history from a .pt file that may embed it."""
    try:
        data = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(data, dict) and "history" in data:
            return data["history"]
    except Exception:
        pass
    return None


def _smooth(values: list[float], window: int = 5) -> np.ndarray:
    """Uniform moving-average smoothing."""
    arr = np.array(values, dtype=float)
    if len(arr) < window:
        return arr
    kernel = np.ones(window) / window
    padded = np.pad(arr, (window // 2, window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")[:len(arr)]


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_curves(
    ppo_history:  list[dict] | None,
    cmdp_history: list[dict] | None,
    out_path:     Path,
) -> None:
    if ppo_history is None and cmdp_history is None:
        print("No training history found.  Re-run training with the patched script "
              "that saves history, or see note in plot_training_curves.py.")
        return

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=False)
    fig.suptitle("PPO Training Curves", fontsize=13, fontweight="bold", y=0.98)

    colours = {"ppo_unc": "#E07B39", "cmdp": "#4A90D9"}

    # ── Panel 1: collision count per rollout ─────────────────────────────────
    ax1 = axes[0]
    ax1.set_title("Collision count per rollout")
    ax1.set_ylabel("collisions")

    if ppo_history:
        iters   = [r["iteration"]       for r in ppo_history]
        # unconstrained records use collision_count (raw integer)
        colls   = [r.get("collision_count", r.get("collision_rate", 0)) for r in ppo_history]
        ax1.plot(iters, colls, color=colours["ppo_unc"], alpha=0.25, lw=1)
        ax1.plot(iters, _smooth(colls), color=colours["ppo_unc"], lw=2,
                 label="PPO-unconstrained")

    if cmdp_history:
        iters   = [r["iteration"]       for r in cmdp_history]
        # CMDP records use collision_rate (fraction per episode)
        colls   = [r.get("collision_rate", r.get("collision_count", 0)) for r in cmdp_history]
        ax1.plot(iters, colls, color=colours["cmdp"], alpha=0.25, lw=1)
        ax1.plot(iters, _smooth(colls), color=colours["cmdp"], lw=2,
                 label="PPO-CMDP")

    ax1.legend(fontsize=9)
    ax1.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax1.grid(True, alpha=0.3)

    # ── Panel 2: mean episode return ─────────────────────────────────────────
    ax2 = axes[1]
    ax2.set_title("Mean episode return (IRL-shaped reward)")
    ax2.set_ylabel("return")

    if ppo_history:
        iters   = [r["iteration"]   for r in ppo_history]
        returns = [r["mean_ep_ret"] for r in ppo_history]
        ax2.plot(iters, returns, color=colours["ppo_unc"], alpha=0.25, lw=1)
        ax2.plot(iters, _smooth(returns), color=colours["ppo_unc"], lw=2,
                 label="PPO-unconstrained")

    if cmdp_history:
        iters   = [r["iteration"]   for r in cmdp_history]
        returns = [r["mean_ep_ret"] for r in cmdp_history]
        ax2.plot(iters, returns, color=colours["cmdp"], alpha=0.25, lw=1)
        ax2.plot(iters, _smooth(returns), color=colours["cmdp"], lw=2,
                 label="PPO-CMDP")

    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # ── Panel 3: λ trajectory (CMDP only) ────────────────────────────────────
    ax3 = axes[2]
    ax3.set_title("Lagrange multiplier λ (CMDP only)")
    ax3.set_ylabel("λ")
    ax3.set_xlabel("iteration")

    if cmdp_history and ("lambda" in cmdp_history[0] or "lambda_" in cmdp_history[0]):
        iters  = [r["iteration"] for r in cmdp_history]
        lambdas = [r.get("lambda", r.get("lambda_", 0.0)) for r in cmdp_history]
        ax3.plot(iters, lambdas, color=colours["cmdp"], lw=2, label="λ")
        # Threshold line
        ax3.axhline(0.0, color="gray", lw=0.8, linestyle="--")
        ax3.fill_between(iters, 0, lambdas, color=colours["cmdp"], alpha=0.12)
        ax3.legend(fontsize=9)
    else:
        ax3.text(0.5, 0.5, "λ history not available\n(re-run training to capture it)",
                 ha="center", va="center", transform=ax3.transAxes, color="gray")

    ax3.grid(True, alpha=0.3)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Plot PPO + CMDP training curves")
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "training_curves.png",
                        help="Output path (default: results/training_curves.png)")
    args = parser.parse_args()

    ppo_ckpt  = RESULTS_DIR / "ppo_unconstrained_final.pt"
    cmdp_ckpt = RESULTS_DIR / "cmdp_final.pt"

    ppo_history  = _load_history_from_pt(ppo_ckpt)
    cmdp_history = _load_history_from_pt(cmdp_ckpt)

    if ppo_history is None:
        print(f"Note: no history in {ppo_ckpt.name} — PPO-unconstrained panel will be empty.")
    if cmdp_history is None:
        print(f"Note: no history in {cmdp_ckpt.name} — CMDP panel will be empty.")

    plot_curves(ppo_history, cmdp_history, args.out)


if __name__ == "__main__":
    main()
