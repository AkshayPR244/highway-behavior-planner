from __future__ import annotations

from pathlib import Path

import pytest

from metrics.evaluator import EvalResults
from scripts import train_ppo
from scripts.seed_partitions import SeedPartition, validate_seed_partitions


def _fake_eval_results(
    *,
    collision: float,
    success: float,
    progress: float,
    speed: float,
    jerk: float,
) -> EvalResults:
    return EvalResults(
        collision_rate=collision,
        mean_min_ttc=None,
        rms_jerk=jerk,
        survival_rate=1.0 - collision,
        success_rate=success,
        fallback_rate=0.0,
        lc_frequency=0.0,
        lc_completion_rate=float("nan"),
        lc_anticipatory_frac=float("nan"),
        ego_fault_rate=0.0,
        npc_fault_rate=0.0,
        mean_distance_travelled=progress,
        mean_longitudinal_progress=progress,
        mean_speed=speed,
        mean_episode_duration=40.0,
        ttc_below_1s_frac=0.0,
        ttc_below_2s_frac=0.0,
        ttc_below_4s_frac=0.0,
        ttc_p5_finite=None,
        n_episodes=20,
        n_steps_total=800,
    )


def test_default_validation_and_test_partitions_are_disjoint():
    validation = SeedPartition("validation", base_seed=2026, num_seeds=5, episodes_per_seed=20)
    test = SeedPartition("test", base_seed=4042, num_seeds=5, episodes_per_seed=20)
    adversarial = SeedPartition("adversarial", base_seed=8080, num_seeds=3, episodes_per_seed=10)
    validate_seed_partitions(
        training_seeds={1, 101},
        validation_partition=validation,
        test_partition=test,
        adversarial_partition=adversarial,
    )


def test_validation_test_overlap_raises():
    validation = SeedPartition("validation", base_seed=100, num_seeds=3, episodes_per_seed=20)
    test = SeedPartition("test", base_seed=102, num_seeds=3, episodes_per_seed=20)
    with pytest.raises(ValueError, match="Validation and test seed partitions overlap"):
        validate_seed_partitions(
            training_seeds={1, 2},
            validation_partition=validation,
            test_partition=test,
        )


def test_training_test_overlap_raises_when_training_seeds_are_enumerable():
    validation = SeedPartition("validation", base_seed=200, num_seeds=3, episodes_per_seed=20)
    test = SeedPartition("test", base_seed=101, num_seeds=3, episodes_per_seed=20)
    with pytest.raises(ValueError, match="Training and test seed partitions overlap"):
        validate_seed_partitions(
            training_seeds={101, 5000},
            validation_partition=validation,
            test_partition=test,
        )


def test_derived_expert_training_seed_overlap_triggers_error():
    validation = SeedPartition("validation", base_seed=2026, num_seeds=3, episodes_per_seed=20)
    test = SeedPartition("test", base_seed=3, num_seeds=1, episodes_per_seed=20)

    training_seeds = train_ppo._training_seed_set(
        train_seed=1,
        n_expert_episodes=20,
        include_dagger_lineage=False,
        dagger_lineage_seed=42,
        dagger_lineage_expert_episodes=50,
        dagger_lineage_iters=5,
        dagger_lineage_rollout_episodes=20,
        dagger_lineage_eval_episodes=20,
        dagger_lineage_scenario_episodes=5,
    )

    assert 3 in training_seeds
    with pytest.raises(ValueError, match="Training and test seed partitions overlap"):
        validate_seed_partitions(
            training_seeds=training_seeds,
            validation_partition=validation,
            test_partition=test,
        )


def test_derived_dagger_training_seed_overlap_triggers_error():
    validation = SeedPartition("validation", base_seed=2026, num_seeds=3, episodes_per_seed=20)
    # DAgger rollout seeds include base + i*1000 + ep. With base=42, i=1, ep=0 => 1042.
    test = SeedPartition("test", base_seed=1042, num_seeds=1, episodes_per_seed=20)

    training_seeds = train_ppo._training_seed_set(
        train_seed=1,
        n_expert_episodes=20,
        include_dagger_lineage=True,
        dagger_lineage_seed=42,
        dagger_lineage_expert_episodes=50,
        dagger_lineage_iters=5,
        dagger_lineage_rollout_episodes=20,
        dagger_lineage_eval_episodes=20,
        dagger_lineage_scenario_episodes=5,
    )

    assert 1042 in training_seeds
    with pytest.raises(ValueError, match="Training and test seed partitions overlap"):
        validate_seed_partitions(
            training_seeds=training_seeds,
            validation_partition=validation,
            test_partition=test,
        )


def test_checkpoint_selection_uses_validation_partition_only(monkeypatch, tmp_path):
    ckpt_a = tmp_path / "ppo_iter0001.pt"
    ckpt_b = tmp_path / "ppo_iter0002.pt"
    ckpt_a.write_text("a", encoding="utf-8")
    ckpt_b.write_text("b", encoding="utf-8")

    class _DummyEnv:
        def close(self):
            return None

    monkeypatch.setattr(train_ppo, "make_env", lambda seed: _DummyEnv())
    monkeypatch.setattr(train_ppo.ActorCritic, "load", lambda path, device: object())

    calls: list[str] = []

    def _fake_evaluate_on_partition(*, policy, env, partition, fault_horizon_steps):
        calls.append(partition.name)
        if "0001" in calls and False:
            pass
        # rank so second checkpoint wins in unconstrained sorting
        if len(calls) == 1:
            return _fake_eval_results(collision=0.10, success=0.70, progress=300.0, speed=20.0, jerk=1.0)
        return _fake_eval_results(collision=0.10, success=0.80, progress=320.0, speed=21.0, jerk=1.0)

    monkeypatch.setattr(train_ppo, "_evaluate_on_partition", _fake_evaluate_on_partition)

    validation = SeedPartition("validation", base_seed=2026, num_seeds=3, episodes_per_seed=20)
    selected_path, _ = train_ppo._select_best_checkpoint(
        checkpoint_dir=tmp_path,
        pattern="ppo_iter*.pt",
        validation_partition=validation,
        fault_horizon_steps=5,
        device="cpu",
        collision_epsilon=None,
    )

    assert calls == ["validation", "validation"]
    assert selected_path.name == "ppo_iter0002.pt"


def test_feasible_constrained_checkpoint_beats_unsafe_higher_success():
    a = train_ppo.CheckpointScore(
        checkpoint_path=Path("a.pt"),
        metrics=_fake_eval_results(collision=0.05, success=0.80, progress=300.0, speed=20.0, jerk=1.0),
    )
    b = train_ppo.CheckpointScore(
        checkpoint_path=Path("b.pt"),
        metrics=_fake_eval_results(collision=0.15, success=0.95, progress=350.0, speed=22.0, jerk=1.0),
    )
    selected = train_ppo._select_best_scored_checkpoint([a, b], collision_epsilon=0.10)
    assert selected.checkpoint_path.name == "a.pt"


def test_among_feasible_higher_success_wins():
    a = train_ppo.CheckpointScore(
        checkpoint_path=Path("a.pt"),
        metrics=_fake_eval_results(collision=0.05, success=0.70, progress=300.0, speed=20.0, jerk=1.0),
    )
    b = train_ppo.CheckpointScore(
        checkpoint_path=Path("b.pt"),
        metrics=_fake_eval_results(collision=0.09, success=0.80, progress=280.0, speed=19.0, jerk=1.0),
    )
    selected = train_ppo._select_best_scored_checkpoint([a, b], collision_epsilon=0.10)
    assert selected.checkpoint_path.name == "b.pt"


def test_when_no_feasible_checkpoint_lowest_collision_wins():
    a = train_ppo.CheckpointScore(
        checkpoint_path=Path("a.pt"),
        metrics=_fake_eval_results(collision=0.20, success=0.80, progress=300.0, speed=20.0, jerk=1.0),
    )
    b = train_ppo.CheckpointScore(
        checkpoint_path=Path("b.pt"),
        metrics=_fake_eval_results(collision=0.15, success=0.70, progress=350.0, speed=22.0, jerk=1.0),
    )
    selected = train_ppo._select_best_scored_checkpoint([a, b], collision_epsilon=0.10)
    assert selected.checkpoint_path.name == "b.pt"


def test_tie_breaking_is_deterministic_by_checkpoint_name():
    a = train_ppo.CheckpointScore(
        checkpoint_path=Path("a.pt"),
        metrics=_fake_eval_results(collision=0.10, success=0.80, progress=300.0, speed=20.0, jerk=1.0),
    )
    b = train_ppo.CheckpointScore(
        checkpoint_path=Path("b.pt"),
        metrics=_fake_eval_results(collision=0.10, success=0.80, progress=300.0, speed=20.0, jerk=1.0),
    )
    selected = train_ppo._select_best_scored_checkpoint([b, a], collision_epsilon=0.10)
    assert selected.checkpoint_path.name == "a.pt"


def test_final_benchmark_path_uses_test_partition(monkeypatch):
    calls: list[str] = []

    class _DummyEnv:
        def close(self):
            return None

    class _DummyPolicy:
        def act(self, obs):
            return 1

    monkeypatch.setattr(train_ppo, "make_env", lambda seed: _DummyEnv())
    monkeypatch.setattr(train_ppo.IRLPolicy, "load", lambda _path: _DummyPolicy())

    import policies.idm_expert as idm_module

    class _DummyIDM:
        def __init__(self, env):
            self.env = env

        def act(self, obs):
            return 1

    monkeypatch.setattr(idm_module, "IDMExpert", _DummyIDM)
    monkeypatch.setattr(train_ppo, "print_table", lambda results, label: None)

    def _fake_eval(*, policy, env, partition, fault_horizon_steps):
        calls.append(partition.name)
        return _fake_eval_results(collision=0.1, success=0.8, progress=300.0, speed=20.0, jerk=1.0)

    monkeypatch.setattr(train_ppo, "_evaluate_on_partition", _fake_eval)

    test_partition = SeedPartition("test", base_seed=4042, num_seeds=2, episodes_per_seed=10)
    _ = train_ppo.print_comparison(
        ppo_unc_ac=None,
        cmdp_ac=None,
        test_partition=test_partition,
        fault_horizon_steps=5,
    )

    assert calls == ["test", "test"]


def test_result_metadata_records_partition_and_seed_list():
    partition = SeedPartition("test", base_seed=4042, num_seeds=3, episodes_per_seed=20)
    payload = train_ppo._result_with_metadata(
        policy_id="ppo_cmdp",
        metrics=_fake_eval_results(collision=0.1, success=0.8, progress=300.0, speed=20.0, jerk=1.0),
        partition=partition,
        checkpoint_id="cmdp_iter0100.pt",
    )
    assert payload["partition"] == "test"
    assert payload["seeds"] == [4042, 4043, 4044]
    assert payload["episodes_per_seed"] == 20
    assert payload["total_episodes"] == 60
    assert payload["checkpoint_id"] == "cmdp_iter0100.pt"
