"""
Weight visualizer for the MaxEnt IRL cost function.

Produces a horizontal bar chart of learned cost weights w[k], one bar per
feature.  Positive weight (red) means the feature is *costly* — the policy
avoids it.  Negative weight (green) means it is *cheap* — the policy seeks
it.

Usage:
    cd ~/highway-planner && source .venv/bin/activate
    python -m optimizer.weight_viz                        # saves to results/
    python -m optimizer.weight_viz --show                 # also opens window
    python -m optimizer.weight_viz --weights path/to.npy --out fig.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # headless default; overridden if --show is passed
import matplotlib.pyplot as plt
import numpy as np

from optimizer.feature_extractor import FEATURE_NAMES, N_FEATURES

_RESULTS_DIR = Path(__file__).parent.parent / "results"
_DEFAULT_WEIGHTS = _RESULTS_DIR / "irl_weights.npy"
_DEFAULT_OUT     = _RESULTS_DIR / "irl_weights.png"


def plot_weights(
    weights: np.ndarray,
    out_path: Path | str | None = None,
    show: bool = False,
    title: str = "Learned IRL cost weights  w[k]",
) -> plt.Figure:
    """
    Draw a horizontal bar chart of *weights* and return the figure.

    Parameters
    ----------
    weights : np.ndarray, shape (N_FEATURES,)
    out_path : path-like or None
        If given, save the figure to this path.
    show : bool
        If True, call plt.show() after rendering.
    title : str
        Figure title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if len(weights) != N_FEATURES:
        raise ValueError(
            f"Expected {N_FEATURES} weights, got {len(weights)}. "
            "Did you save weights from a different N_FEATURES run?"
        )

    fig, ax = plt.subplots(figsize=(8, 4))

    y_pos  = np.arange(N_FEATURES)
    colors = ["#d62728" if w >= 0 else "#2ca02c" for w in weights]

    bars = ax.barh(y_pos, weights, color=colors, edgecolor="white", height=0.65)

    # Annotate each bar with its numeric value
    for bar, val in zip(bars, weights):
        xoff = 0.03 if val >= 0 else -0.03
        ha   = "left" if val >= 0 else "right"
        ax.text(
            val + xoff, bar.get_y() + bar.get_height() / 2,
            f"{val:+.3f}", va="center", ha=ha, fontsize=8.5,
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(FEATURE_NAMES, fontsize=9)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Weight value  (positive = costly, negative = cheap)")
    ax.set_title(title, fontsize=11, pad=10)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#d62728", label="Costly  (policy avoids)"),
        Patch(facecolor="#2ca02c", label="Cheap  (policy seeks)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8)

    fig.tight_layout()

    if out_path is not None:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved weight chart → {out_path}")

    if show:
        matplotlib.use("TkAgg")
        plt.show()

    return fig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualise IRL cost weights")
    parser.add_argument(
        "--weights", type=Path, default=_DEFAULT_WEIGHTS,
        help="Path to the .npy weights file (default: results/irl_weights.npy)",
    )
    parser.add_argument(
        "--out", type=Path, default=_DEFAULT_OUT,
        help="Output PNG path (default: results/irl_weights.png)",
    )
    parser.add_argument(
        "--show", action="store_true",
        help="Open a matplotlib window in addition to saving",
    )
    args = parser.parse_args()

    if not args.weights.exists():
        raise FileNotFoundError(
            f"Weights not found: {args.weights}\n"
            "Run `python -m optimizer.irl_optimizer` first."
        )

    weights = np.load(args.weights)
    print(f"Loaded {len(weights)}-feature weights from {args.weights}")
    print("\nWeight table:")
    print(f"  {'Feature':<22} {'Weight':>8}  Sign")
    print("  " + "-" * 38)
    for name, w in zip(FEATURE_NAMES, weights):
        sign = "↑ cost" if w >= 0 else "↓ cost"
        print(f"  {name:<22} {w:>+8.4f}  ({sign})")

    plot_weights(weights, out_path=args.out, show=args.show)
