from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SeedPartition:
    """Explicit enumerable seed partition for evaluation workflows."""

    name: str
    base_seed: int
    num_seeds: int
    episodes_per_seed: int

    def seeds(self) -> list[int]:
        if self.num_seeds < 1:
            raise ValueError(f"{self.name} num_seeds must be >= 1")
        return [self.base_seed + i for i in range(self.num_seeds)]

    @property
    def total_episodes(self) -> int:
        if self.episodes_per_seed < 1:
            raise ValueError(f"{self.name} episodes_per_seed must be >= 1")
        return self.num_seeds * self.episodes_per_seed

    def to_metadata(self) -> dict[str, object]:
        return {
            "partition": self.name,
            "base_seed": self.base_seed,
            "num_seeds": self.num_seeds,
            "episodes_per_seed": self.episodes_per_seed,
            "seeds": self.seeds(),
            "total_episodes": self.total_episodes,
        }


def make_seed_range(base_seed: int, num_seeds: int) -> list[int]:
    if num_seeds < 1:
        raise ValueError("num_seeds must be >= 1")
    return [base_seed + i for i in range(num_seeds)]


def enumerate_expert_rollout_env_seeds(base_seed: int, n_episodes: int) -> set[int]:
    """Environment seeds touched by collect_expert_rollouts.

    Includes:
      - make_env(seed=base_seed)
      - env.reset(seed=base_seed + ep) for ep in [0, n_episodes)
    """
    if n_episodes < 1:
        raise ValueError("n_episodes must be >= 1")
    return {base_seed, *{base_seed + ep for ep in range(n_episodes)}}


def enumerate_dagger_training_env_seeds(
    *,
    base_seed: int,
    n_expert_episodes: int,
    n_dagger_iters: int,
    n_rollout_episodes: int,
    eval_episodes: int,
    scenario_eval_episodes: int = 5,
) -> set[int]:
    """Deterministic environment seed set touched by training.dagger_train.

    Captures explicit env.reset(seed=...) usage in DAgger data generation,
    rollout collection, per-iteration evaluation, and scenario-based early stop.
    """
    if n_dagger_iters < 0:
        raise ValueError("n_dagger_iters must be >= 0")
    if n_rollout_episodes < 1:
        raise ValueError("n_rollout_episodes must be >= 1")
    if eval_episodes < 1:
        raise ValueError("eval_episodes must be >= 1")
    if scenario_eval_episodes < 1:
        raise ValueError("scenario_eval_episodes must be >= 1")

    seeds = set()

    # Initial expert dataset D0 (collect_expert_rollouts).
    seeds |= enumerate_expert_rollout_env_seeds(base_seed, n_expert_episodes)

    # BC baseline evaluation and per-iteration evaluation: evaluate(..., seed=base_seed)
    seeds |= {base_seed + ep for ep in range(eval_episodes)}

    # Scenario-based early stop: run_all_scenarios(..., seed=base_seed, n_episodes=5)
    seeds |= {base_seed + ep for ep in range(scenario_eval_episodes)}

    # Iteration rollouts: rollout_policy(..., seed=base_seed + i * 1000)
    for i in range(1, n_dagger_iters + 1):
        rollout_seed = base_seed + i * 1000
        # make_env(seed=rollout_seed) and env.reset(seed=rollout_seed + ep)
        seeds.add(rollout_seed)
        seeds |= {rollout_seed + ep for ep in range(n_rollout_episodes)}

    return seeds


def validate_seed_partitions(
    *,
    training_seeds: set[int] | None,
    validation_partition: SeedPartition,
    test_partition: SeedPartition,
    adversarial_partition: SeedPartition | None = None,
) -> None:
    """Fail loudly when explicit seed partitions overlap."""
    validation_seeds = set(validation_partition.seeds())
    test_seeds = set(test_partition.seeds())
    overlap_val_test = validation_seeds & test_seeds
    if overlap_val_test:
        raise ValueError(
            "Validation and test seed partitions overlap: "
            f"{sorted(overlap_val_test)}"
        )

    if training_seeds is not None:
        overlap_train_val = set(training_seeds) & validation_seeds
        if overlap_train_val:
            raise ValueError(
                "Training and validation seed partitions overlap: "
                f"{sorted(overlap_train_val)}"
            )

        overlap_train_test = set(training_seeds) & test_seeds
        if overlap_train_test:
            raise ValueError(
                "Training and test seed partitions overlap: "
                f"{sorted(overlap_train_test)}"
            )

    if adversarial_partition is not None:
        adversarial_seeds = set(adversarial_partition.seeds())
        overlap_adv_val = adversarial_seeds & validation_seeds
        if overlap_adv_val:
            raise ValueError(
                "Adversarial and validation seed partitions overlap: "
                f"{sorted(overlap_adv_val)}"
            )
        overlap_adv_test = adversarial_seeds & test_seeds
        if overlap_adv_test:
            raise ValueError(
                "Adversarial and test seed partitions overlap: "
                f"{sorted(overlap_adv_test)}"
            )
        if training_seeds is not None:
            overlap_adv_train = adversarial_seeds & set(training_seeds)
            if overlap_adv_train:
                raise ValueError(
                    "Adversarial and training seed partitions overlap: "
                    f"{sorted(overlap_adv_train)}"
                )
