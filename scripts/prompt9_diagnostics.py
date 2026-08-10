from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from envs.highway_wrapper import make_env
from metrics.evaluator import evaluate_across_seeds
from rl.ppo_agent import ActorCritic
from rl.ppo_trainer import PPOConfig, PPOTrainer
from scripts.train_ppo import CMDP_COLLISION_EPSILON, CheckpointScore, _rank_checkpoint_scores


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OUT_DIR = RESULTS / "diagnostics" / "prompt9"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _step_from_name(path: Path) -> int:
    match = re.search(r"(\d+)", path.stem)
    return int(match.group(1)) if match else -1


def _normalize_sd(obj: object) -> dict[str, torch.Tensor]:
    if isinstance(obj, dict) and "state_dict" in obj:
        return obj["state_dict"]
    if isinstance(obj, dict):
        return obj
    raise TypeError(f"Unsupported checkpoint payload type: {type(obj)!r}")


def _same_state_dict(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> bool:
    if set(a.keys()) != set(b.keys()):
        return False
    for key in a.keys():
        ta = a[key]
        tb = b[key]
        if ta.shape != tb.shape:
            return False
        if not torch.equal(ta, tb):
            return False
    return True


def _eval_checkpoints(
    ckpts: list[Path],
    family: str,
    val_base_seed: int,
    val_num_seeds: int,
    val_eps_per_seed: int,
    fault_horizon_steps: int,
) -> pd.DataFrame:
    env = make_env(seed=val_base_seed)
    rows: list[dict[str, object]] = []
    total = len(ckpts)
    for ckpt in ckpts:
        idx = len(rows) + 1
        print(f"[{family}] evaluating checkpoint {idx}/{total}: {ckpt.name}", flush=True)
        model = ActorCritic.load(ckpt, device="cpu")
        metrics = evaluate_across_seeds(
            policy=model,
            env=env,
            n_seeds=val_num_seeds,
            episodes_per_seed=val_eps_per_seed,
            base_seed=val_base_seed,
            fault_horizon_steps=fault_horizon_steps,
        )
        rows.append(
            {
                "family": family,
                "checkpoint": ckpt.name,
                "step": _step_from_name(ckpt),
                "collision_rate": metrics.collision_rate,
                "collision_ci_low": metrics.collision_rate_ci_low,
                "collision_ci_high": metrics.collision_rate_ci_high,
                "success_rate": metrics.success_rate,
                "survival_rate": metrics.survival_rate,
                "progress": metrics.mean_longitudinal_progress,
                "mean_speed": metrics.mean_speed,
                "rms_jerk": metrics.rms_jerk,
                "constraint_satisfied": bool(metrics.collision_rate <= CMDP_COLLISION_EPSILON),
                "fallback_rate": metrics.fallback_rate,
                "n_episodes": metrics.n_episodes,
            }
        )
    env.close()
    return pd.DataFrame(rows).sort_values("step")


def main() -> None:
    bench_path = RESULTS / "final_benchmark_test_partition.json"
    if bench_path.exists():
        bench = json.loads(bench_path.read_text())
        val_meta = bench.get("validation_partition") or {}
        val_base = int(val_meta.get("base_seed", 10042))
        val_num = int(val_meta.get("num_seeds", 10))
        val_eps = int(val_meta.get("episodes_per_seed", 20))
    else:
        val_base, val_num, val_eps = 10042, 10, 20

    val_seeds = [val_base + i for i in range(val_num)]
    fault_horizon = 5
    train_cmdp_env_seed = 142

    cmdp_ckpts = sorted((RESULTS / "cmdp").glob("cmdp_iter*.pt"))
    ppo_ckpts = sorted((RESULTS / "ppo_unconstrained").glob("ppo_iter*.pt"))

    print(
        f"validation partition: base={val_base}, seeds={val_num}, episodes_per_seed={val_eps}",
        flush=True,
    )
    print(f"candidate checkpoints: cmdp={len(cmdp_ckpts)}, ppo={len(ppo_ckpts)}", flush=True)

    cmdp_df = _eval_checkpoints(cmdp_ckpts, "cmdp", val_base, val_num, val_eps, fault_horizon)
    ppo_df = _eval_checkpoints(ppo_ckpts, "ppo_unconstrained", val_base, val_num, val_eps, fault_horizon)

    cmdp_df.to_csv(OUT_DIR / "cmdp_validation_checkpoints.csv", index=False)
    ppo_df.to_csv(OUT_DIR / "ppo_validation_checkpoints.csv", index=False)

    scored: list[CheckpointScore] = []
    for row in cmdp_df.to_dict("records"):
        class Dummy:
            pass

        d = Dummy()
        d.collision_rate = float(row["collision_rate"])
        d.success_rate = float(row["success_rate"])
        d.mean_longitudinal_progress = float(row["progress"])
        d.mean_speed = float(row["mean_speed"])
        d.rms_jerk = float(row["rms_jerk"])
        scored.append(CheckpointScore(checkpoint_path=Path(str(row["checkpoint"])), metrics=d))

    ranked = _rank_checkpoint_scores(scored, CMDP_COLLISION_EPSILON)
    ranked_names = [entry.checkpoint_path.name for entry in ranked]
    selected_by_policy = ranked_names[0] if ranked_names else None

    promoted_cmdp_sd = _normalize_sd(torch.load(RESULTS / "cmdp_final.pt", map_location="cpu", weights_only=False))
    promoted_ppo_sd = _normalize_sd(
        torch.load(RESULTS / "ppo_unconstrained_final.pt", map_location="cpu", weights_only=False)
    )

    cmdp_candidates = list(cmdp_ckpts)
    cmdp_candidates += list((RESULTS / "retrain_eval").rglob("cmdp_seed*.pt"))
    cmdp_candidates += list((RESULTS / "retrain_eval").rglob("cmdp_iter*.pt"))
    promoted_cmdp_matches: list[str] = []
    for path in cmdp_candidates:
        try:
            sd = _normalize_sd(torch.load(path, map_location="cpu", weights_only=False))
        except Exception:
            continue
        if _same_state_dict(promoted_cmdp_sd, sd):
            promoted_cmdp_matches.append(str(path.relative_to(ROOT)))

    ppo_candidates = list(ppo_ckpts)
    ppo_candidates += list((RESULTS / "retrain_eval").rglob("ppo_unconstrained_seed*.pt"))
    ppo_candidates += list((RESULTS / "retrain_eval").rglob("ppo_iter*.pt"))
    promoted_ppo_matches: list[str] = []
    for path in ppo_candidates:
        try:
            sd = _normalize_sd(torch.load(path, map_location="cpu", weights_only=False))
        except Exception:
            continue
        if _same_state_dict(promoted_ppo_sd, sd):
            promoted_ppo_matches.append(str(path.relative_to(ROOT)))

    history_path = RESULTS / "retrain_eval" / "20260809_131815" / "cmdp_final_post.pt"
    cmdp_history: list[dict[str, object]] = []
    if history_path.exists():
        payload = torch.load(history_path, map_location="cpu", weights_only=False)
        if isinstance(payload, dict):
            cmdp_history = payload.get("history", []) or []

    lambda_stats: dict[str, object] = {}
    if cmdp_history:
        lambdas = np.array([float(row.get("lambda", np.nan)) for row in cmdp_history], dtype=float)
        coll = np.array([float(row.get("collision_rate", np.nan)) for row in cmdp_history], dtype=float)
        completed = np.array([int(row.get("completed_episodes", 0)) for row in cmdp_history], dtype=int)
        previous = np.concatenate([[0.0], lambdas[:-1]])
        delta = lambdas - previous

        lambda_stats = {
            "iterations": int(len(cmdp_history)),
            "lambda_start": float(lambdas[0]),
            "lambda_end": float(lambdas[-1]),
            "lambda_min": float(np.nanmin(lambdas)),
            "lambda_max": float(np.nanmax(lambdas)),
            "lambda_mean": float(np.nanmean(lambdas)),
            "collision_rate_mean": float(np.nanmean(coll)),
            "collision_rate_min": float(np.nanmin(coll)),
            "collision_rate_max": float(np.nanmax(coll)),
            "updates_with_zero_completed_episodes_frac": float(np.mean(completed == 0)),
            "updates_with_one_completed_episode_frac": float(np.mean(completed == 1)),
            "updates_with_zero_collisions_frac": float(np.mean(np.isfinite(coll) & (coll == 0.0))),
            "updates_with_full_collisions_frac": float(np.mean(np.isfinite(coll) & (coll == 1.0))),
            "lambda_delta_mean": float(np.nanmean(delta)),
            "lambda_delta_std": float(np.nanstd(delta)),
            "lambda_delta_p05": float(np.nanpercentile(delta, 5)),
            "lambda_delta_p95": float(np.nanpercentile(delta, 95)),
        }

        hist_df = pd.DataFrame(cmdp_history)
        hist_df["lambda_delta"] = delta
        hist_df.to_csv(OUT_DIR / "cmdp_history_20260809_131815.csv", index=False)

    cmdp_model = ActorCritic.load(RESULTS / "cmdp_final.pt", device="cpu")
    ppo_cfg = PPOConfig(n_steps=512, n_epochs=4, batch_size=64, lr=1e-4, device="cpu", seed=train_cmdp_env_seed)
    trainer = PPOTrainer(actor_critic=cmdp_model, config=ppo_cfg, reward_shaper=None)

    env = make_env(seed=train_cmdp_env_seed)
    rollout_rows: list[dict[str, object]] = []
    all_reward_adv: list[np.ndarray] = []
    all_cost_adv: list[np.ndarray] = []
    all_combined_adv: list[np.ndarray] = []
    all_cost_returns: list[np.ndarray] = []
    all_cost_values: list[np.ndarray] = []

    lambda_for_adv = float(lambda_stats.get("lambda_end", 0.0)) if lambda_stats else 0.0

    for idx in range(30):
        if idx % 5 == 0:
            print(f"[rollout diagnostics] collecting rollout {idx+1}/30", flush=True)
        buf, stats = trainer.collect_rollout(env)
        episodes = int(stats["episodes"])
        collisions = int(stats["collision_count"])
        collision_rate = (collisions / episodes) if episodes > 0 else float("nan")

        reward_adv = buf.reward_advantages.detach().cpu().numpy()
        cost_adv = buf.cost_advantages.detach().cpu().numpy()
        combined = reward_adv - lambda_for_adv * cost_adv

        rollout_rows.append(
            {
                "rollout_idx": idx + 1,
                "n_steps": len(buf.rewards),
                "completed_episodes": episodes,
                "collision_episodes": collisions,
                "collision_rate": collision_rate,
                "mean_ep_ret": float(stats["mean_ep_ret"]),
                "mean_ep_shaped": float(stats["mean_ep_shaped"]),
                "mean_abs_reward_adv": float(np.mean(np.abs(reward_adv))),
                "mean_abs_cost_adv": float(np.mean(np.abs(cost_adv))),
                "mean_abs_lambda_cost_adv": float(np.mean(np.abs(lambda_for_adv * cost_adv))),
                "std_reward_adv": float(np.std(reward_adv)),
                "std_cost_adv": float(np.std(cost_adv)),
                "std_combined_adv_pre_norm": float(np.std(combined)),
            }
        )

        all_reward_adv.append(reward_adv)
        all_cost_adv.append(cost_adv)
        all_combined_adv.append(combined)
        all_cost_returns.append(buf.cost_returns.detach().cpu().numpy())
        all_cost_values.append(np.asarray(buf.cost_values, dtype=float))

    env.close()

    rollout_df = pd.DataFrame(rollout_rows)
    rollout_df.to_csv(OUT_DIR / "cmdp_rollout_diagnostics_train_partition.csv", index=False)

    val_env = make_env(seed=val_base)
    per_episode: list[dict[str, object]] = []
    per_step: list[dict[str, object]] = []
    for seed in val_seeds:
        for ep in range(val_eps):
            if (seed - val_base) % 2 == 0 and ep == 0:
                print(f"[cost critic] validation seed {seed}", flush=True)
            obs, _ = val_env.reset(seed=seed + ep)
            done = False
            preds: list[float] = []
            costs: list[float] = []
            while not done:
                obs_arr = np.asarray(obs, dtype=np.float32)
                obs_t = torch.from_numpy(obs_arr).unsqueeze(0)
                with torch.no_grad():
                    dist, _, cost_value = cmdp_model.forward(obs_t)
                action = int(dist.probs.argmax(dim=-1).item())
                next_obs, _, terminated, truncated, info = val_env.step(action)
                cost = float(info.get("crashed", False))
                preds.append(float(cost_value.item()))
                costs.append(cost)
                obs = next_obs
                done = bool(terminated or truncated)

            episode_collision = 1.0 if any(c > 0.5 for c in costs) else 0.0
            episode_pred = float(np.mean(preds)) if preds else 0.0
            per_episode.append(
                {
                    "episode_collision": episode_collision,
                    "mean_predicted_cost_value": episode_pred,
                    "len": len(preds),
                    "sum_cost": float(np.sum(costs)),
                }
            )
            for pred in preds:
                per_step.append({"pred": pred, "episode_collision": episode_collision})

    val_env.close()

    episode_df = pd.DataFrame(per_episode)
    step_df = pd.DataFrame(per_step)

    mean_pred_collide = (
        float(episode_df.loc[episode_df["episode_collision"] == 1.0, "mean_predicted_cost_value"].mean())
        if (episode_df["episode_collision"] == 1.0).any()
        else float("nan")
    )
    mean_pred_safe = (
        float(episode_df.loc[episode_df["episode_collision"] == 0.0, "mean_predicted_cost_value"].mean())
        if (episode_df["episode_collision"] == 0.0).any()
        else float("nan")
    )
    mse_episode = float(
        np.mean(
            (
                episode_df["mean_predicted_cost_value"].to_numpy()
                - episode_df["episode_collision"].to_numpy()
            )
            ** 2
        )
    )

    step_df["pred_clip"] = step_df["pred"].clip(lower=0.0, upper=1.0)
    bins = np.arange(0.0, 1.01, 0.1)
    step_df["bucket"] = pd.cut(step_df["pred_clip"], bins=bins, include_lowest=True, right=True)
    calibration = (
        step_df.groupby("bucket", observed=True)
        .agg(
            n=("episode_collision", "size"),
            mean_pred=("pred_clip", "mean"),
            observed_collision_freq=("episode_collision", "mean"),
        )
        .reset_index()
    )
    calibration.to_csv(OUT_DIR / "cmdp_cost_value_calibration_validation.csv", index=False)

    val_env2 = make_env(seed=val_base)
    collided_episode_cost_counts: list[int] = []
    for seed in val_seeds[:3]:
        for ep in range(min(10, val_eps)):
            obs, _ = val_env2.reset(seed=seed + ep)
            done = False
            costs: list[float] = []
            while not done:
                obs_arr = np.asarray(obs, dtype=np.float32)
                obs_t = torch.from_numpy(obs_arr).unsqueeze(0)
                with torch.no_grad():
                    dist, _, _ = cmdp_model.forward(obs_t)
                action = int(dist.probs.argmax(dim=-1).item())
                obs, _, terminated, truncated, info = val_env2.step(action)
                costs.append(float(info.get("crashed", False)))
                done = bool(terminated or truncated)
            if any(c > 0.5 for c in costs):
                collided_episode_cost_counts.append(int(np.sum(np.asarray(costs) > 0.5)))
    val_env2.close()

    reward_adv_all = np.concatenate(all_reward_adv) if all_reward_adv else np.array([])
    cost_adv_all = np.concatenate(all_cost_adv) if all_cost_adv else np.array([])
    combined_adv_all = np.concatenate(all_combined_adv) if all_combined_adv else np.array([])
    cost_returns_all = np.concatenate(all_cost_returns) if all_cost_returns else np.array([])
    cost_values_all = np.concatenate(all_cost_values) if all_cost_values else np.array([])

    summary = {
        "validation_partition": {
            "base_seed": val_base,
            "num_seeds": val_num,
            "episodes_per_seed": val_eps,
            "seeds": val_seeds,
            "total_episodes": val_num * val_eps,
            "fault_horizon_steps": fault_horizon,
        },
        "checkpoint_ranking": {
            "cmdp_selected_by_constraint_first_policy": selected_by_policy,
            "cmdp_ranking_top5": ranked_names[:5],
            "promoted_cmdp_final_matches": promoted_cmdp_matches,
            "promoted_ppo_final_matches": promoted_ppo_matches,
        },
        "lambda_history_supporting_run": lambda_stats,
        "rollout_sparsity_train_partition": {
            "n_rollouts": int(len(rollout_df)),
            "rollout_length_steps": int(ppo_cfg.n_steps),
            "completed_episodes_per_rollout_mean": float(rollout_df["completed_episodes"].mean()),
            "completed_episodes_per_rollout_min": int(rollout_df["completed_episodes"].min()),
            "completed_episodes_per_rollout_max": int(rollout_df["completed_episodes"].max()),
            "collision_episodes_per_rollout_mean": float(rollout_df["collision_episodes"].mean()),
            "fraction_rollouts_zero_collisions": float(
                np.mean(rollout_df["collision_episodes"].to_numpy() == 0)
            ),
            "fraction_rollouts_zero_completed_episodes": float(
                np.mean(rollout_df["completed_episodes"].to_numpy() == 0)
            ),
            "fraction_rollouts_one_completed_episode": float(
                np.mean(rollout_df["completed_episodes"].to_numpy() == 1)
            ),
        },
        "advantage_magnitude_train_partition": {
            "lambda_used_for_scaling": lambda_for_adv,
            "mean_abs_reward_adv": float(np.mean(np.abs(reward_adv_all))) if reward_adv_all.size else None,
            "mean_abs_cost_adv": float(np.mean(np.abs(cost_adv_all))) if cost_adv_all.size else None,
            "mean_abs_lambda_cost_adv": (
                float(np.mean(np.abs(lambda_for_adv * cost_adv_all))) if cost_adv_all.size else None
            ),
            "std_reward_adv": float(np.std(reward_adv_all)) if reward_adv_all.size else None,
            "std_cost_adv": float(np.std(cost_adv_all)) if cost_adv_all.size else None,
            "std_combined_adv_pre_norm": float(np.std(combined_adv_all)) if combined_adv_all.size else None,
        },
        "cost_critic_validation": {
            "mean_predicted_Vc_episode_collide": mean_pred_collide,
            "mean_predicted_Vc_episode_no_collision": mean_pred_safe,
            "episode_level_mse": mse_episode,
        },
        "cost_signal_timing": {
            "n_collided_episodes_checked": len(collided_episode_cost_counts),
            "cost_positive_count_values": sorted(set(collided_episode_cost_counts)),
            "fraction_collided_with_single_positive_cost": (
                float(np.mean(np.asarray(collided_episode_cost_counts) == 1))
                if collided_episode_cost_counts
                else None
            ),
        },
        "cost_return_numerics_train_partition": {
            "gamma_cost_shared_with_gamma": 0.99,
            "cost_returns_min": float(np.min(cost_returns_all)) if cost_returns_all.size else None,
            "cost_returns_max": float(np.max(cost_returns_all)) if cost_returns_all.size else None,
            "cost_returns_mean": float(np.mean(cost_returns_all)) if cost_returns_all.size else None,
            "cost_values_min": float(np.min(cost_values_all)) if cost_values_all.size else None,
            "cost_values_max": float(np.max(cost_values_all)) if cost_values_all.size else None,
            "cost_values_mean": float(np.mean(cost_values_all)) if cost_values_all.size else None,
            "cost_adv_min": float(np.min(cost_adv_all)) if cost_adv_all.size else None,
            "cost_adv_max": float(np.max(cost_adv_all)) if cost_adv_all.size else None,
            "cost_adv_mean": float(np.mean(cost_adv_all)) if cost_adv_all.size else None,
            "frac_all_zero_cost_batches": float(
                np.mean(rollout_df["collision_episodes"].to_numpy() == 0)
            ),
        },
        "availability_notes": {
            "approx_kl_logged": False,
            "mean_cost_return_logged_per_iteration": False,
            "checkpoint_lambda_persisted_per_ckpt": False,
        },
    }

    (OUT_DIR / "prompt9_summary.json").write_text(json.dumps(summary, indent=2))

    print("Wrote diagnostics to", OUT_DIR, flush=True)


if __name__ == "__main__":
    main()
