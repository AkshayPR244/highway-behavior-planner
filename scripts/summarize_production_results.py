"""
Summarise the most recent stabilized PPO/CMDP retrain sweep.

Reads a production sweep directory created by scripts.train_ppo with
`--select-best-checkpoint`, then prints the promoted winners, their metrics,
and a compact comparison against the baseline checkpoints.

Usage
-----
    python -m scripts.summarize_production_results
    python -m scripts.summarize_production_results --run-dir results/retrain_eval/<run>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


RESULTS_DIR = Path(__file__).parent.parent / "results"
RETRAIN_DIR = RESULTS_DIR / "retrain_eval"


def _latest_run_dir() -> Path:
    marker = RETRAIN_DIR / "LATEST_PRODUCTION_DIR.txt"
    if marker.exists():
        candidate = Path(marker.read_text().strip())
        if candidate.exists():
            return candidate

    candidates = [
        path
        for path in RETRAIN_DIR.iterdir()
        if path.is_dir() and path.name.endswith("_production_stability")
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No production sweep directory found under {RETRAIN_DIR}"
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _fmt(value: object) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _print_policy_block(label: str, metrics: dict) -> None:
    print(f"{label}")
    print(f"  collision_rate: {_fmt(metrics.get('collision_rate'))}")
    print(f"  goal_completion: {_fmt(metrics.get('goal_completion'))}")
    print(f"  mean_min_ttc: {_fmt(metrics.get('mean_min_ttc'))}")
    print(f"  rms_jerk: {_fmt(metrics.get('rms_jerk'))}")
    if "collision_rate_ci_low" in metrics and "collision_rate_ci_high" in metrics:
        low = _fmt(metrics.get('collision_rate_ci_low'))
        high = _fmt(metrics.get('collision_rate_ci_high'))
        print(f"  collision_rate_ci: {low}–{high}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarise the latest stabilized PPO/CMDP retrain sweep"
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Production sweep directory (defaults to the latest one)",
    )
    args = parser.parse_args()

    run_dir = args.run_dir or _latest_run_dir()
    summary_path = run_dir / "production_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary file: {summary_path}")

    summary = json.loads(summary_path.read_text())

    print(f"Run dir: {run_dir}")
    print()

    baseline = summary["baseline"]
    print("Baseline")
    _print_policy_block("  PPO-unconstrained", baseline["ppo_unconstrained"])
    _print_policy_block("  PPO-CMDP", baseline["ppo_cmdp"])
    print()

    best = summary["best"]
    print("Promoted winners")
    print(
        f"  PPO-unconstrained: {best['ppo_unconstrained']['seed']} "
        f"(collision={_fmt(best['ppo_unconstrained']['metrics'].get('collision_rate'))}, "
        f"goal={_fmt(best['ppo_unconstrained']['metrics'].get('goal_completion'))})"
    )
    print(
        f"  PPO-CMDP: {best['ppo_cmdp']['seed']} "
        f"(collision={_fmt(best['ppo_cmdp']['metrics'].get('collision_rate'))}, "
        f"goal={_fmt(best['ppo_cmdp']['metrics'].get('goal_completion'))})"
    )
    print()

    print("All scored candidates")
    for policy_name, by_seed in summary["scored"].items():
        print(f"  {policy_name}")
        for seed_name, metrics in by_seed.items():
            print(
                f"    {seed_name}: collision={_fmt(metrics.get('collision_rate'))}, "
                f"goal={_fmt(metrics.get('goal_completion'))}, "
                f"rms_jerk={_fmt(metrics.get('rms_jerk'))}"
            )


if __name__ == "__main__":
    main()