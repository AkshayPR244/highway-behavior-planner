# Learning-Augmented Highway Planner — Project Plan

A portfolio project targeting Aurora Tech's **Behavior Planning Software Engineer** role.
The system demonstrates three techniques that appear together in modern AV stacks:
imitation learning (BC + DAgger), differentiable inverse RL, and safety-constrained
reinforcement learning — all evaluated on a common highway driving benchmark.

---

## Motivation

Most open-source AV-planning demos show one technique in isolation.  This project
deliberately chains all three into a single pipeline so that each phase's output
feeds the next:

```
IDM Expert → [BC / DAgger] → Imitation Policy
                                    ↓
                         IRL cost weights (Phase 2)
                                    ↓
                         PPO reward shaping (Phase 3)
                                    ↓
                    Safety wrapper (hard override, all phases)
```

The result is a system that can be ablated cleanly: swap out the BC policy for
DAgger, replace the hand-tuned reward with IRL weights, add/remove the safety
wrapper — and the same metric suite reports the difference numerically.

---

## Environment

**highway-v0** (highway-env 1.9.1 / gymnasium 0.29.1)

- Straight infinite 3-lane highway, 10 NPC vehicles, 40-second episodes
- `DiscreteMetaAction`: 0=LANE_LEFT, 1=IDLE, 2=LANE_RIGHT, 3=FASTER, 4=SLOWER
- Kinematics observer: 5 vehicles × 5 features → 25-dim flat observation (float32)
- `policy_frequency=1 Hz`, `simulation_frequency=15 Hz`
- Hardware: Apple M4 (MPS), Python 3.11.15, PyTorch 2.1.0, uv virtual env

---

## Phases

### Phase 0 — Infrastructure ✅

**Goal:** Pin all moving parts so every subsequent component sees identical
observation and action spaces, reproducible seeds, and a single source of
environment configuration.

**Rationale:** Highway-env has many config knobs that silently change the
observation shape or reward scale.  Locking them in one place (`make_env()`)
prevents subtle mismatches between the expert collector, the learner, and the
evaluator.

**Deliverables:**
- `envs/highway_wrapper.py` — `make_env(seed)` → `FlattenObservation`-wrapped env
  with `offscreen_rendering=True` (no GUI required in CI)

---

### Phase 1 — Expert, Metrics & Safety ✅

**Goal:** Build a ground-truth expert good enough to serve as the DAgger oracle,
a metric suite rich enough to detect regressions, and a hard safety filter that
can be toggled independently of the policy.

#### 1a. IDM + MOBIL Expert (`policies/idm_expert.py`)

**Rationale:** The Intelligent Driver Model (IDM) + MOBIL lane-change model is the
standard analytical baseline for highway driving.  It produces collision-free
trajectories without any training, giving a clean upper bound for Phase 2/3
comparisons.  It also doubles as the DAgger labelling oracle — when the learner
reaches an out-of-distribution state, the expert re-labels the action.

Key parameters:
- IDM: desired speed 25 m/s, min spacing 5 m, headway 1.5 s, accel/decel 3 m/s²
- MOBIL: B_safe 4.0 m/s², politeness 0.2, threshold 0.1 m/s²

Validated results (20 episodes, seed=0):

| Metric | IDM Expert |
|---|---|
| Collision rate | 0.000 |
| Goal completion | 1.000 |
| Mean min TTC | 14.34 s |
| RMS jerk | 4.887 m/s³ |
| LC frequency | 0.041 /step |
| LC completion rate | 0.576 |
| LC anticipatory frac | 0.879 |

#### 1b. Metric Suite (`metrics/evaluator.py`)

**Rationale:** A single collision-rate number hides too much.  Aurora-style
safety cases require: *was the crash the ego's fault?*, *how proactive were
lane changes?*, *how smooth was longitudinal control?*  Nine metrics answer
these questions and are tracked across every policy comparison.

Metrics:
1. **collision_rate** — fraction of episodes ending in crash
2. **goal_completion** — fraction reaching episode end without crash
3. **mean_min_ttc** — average of per-episode minimum time-to-collision (s)
4. **rms_jerk** — RMS of second-difference of speed (m/s³); comfort proxy
5. **fallback_rate** — fraction of steps where safety wrapper overrode policy
6. **lc_frequency** — lane-change initiations per step
7. **lc_completion_rate** — fraction of initiated LCs that finish
8. **lc_anticipatory_frac** — fraction of LCs started while TTC > 4 s (proactive)
9. **ego_fault_rate / npc_fault_rate** — counterfactual fault attribution (see below)

#### 1c. Counterfactual Fault Attribution (`metrics/fault_attribution.py`)

**Rationale:** Not all crashes are the ego's fault.  An NPC rear-ending the ego
at 0.5 s TTC cannot be avoided regardless of what the ego does.  Conflating
avoidable and unavoidable crashes inflates the ego's apparent failure rate.

Method: before each step, snapshot all vehicle positions and speeds.  If a crash
occurs, project all vehicles forward one step assuming ego takes IDLE.  If the
crash still happens under IDLE, classify as **NPC fault**; otherwise **ego fault**.

Constants: `COLLISION_DIST ≈ 5.39 m` (2× vehicle half-diagonal, conservative).

#### 1d. Safety Wrapper (`safety/safety_wrapper.py`)

**Rationale:** Learned policies will initially be unsafe.  Rather than letting
them explore freely (too dangerous for highway speeds) or constraining the reward
signal (entangles safety and performance objectives), a hard override wrapper
intercepts unsafe actions before they reach the environment.  This cleanly
separates *what the policy wants to do* from *what it is allowed to do*.

Design:
- `HORIZON=6` steps (6-second lookahead: derived from 300 m stopping distance at
  30 m/s = 10 s budget, minus 1 s reaction time, midpoint ≈ 6 s)
- `MIN_GAP=4.0 m` minimum acceptable gap after projection
- Forward-projects proposed action over the horizon using constant-velocity kinematics
- Overrides with IDM fallback if any gap constraint is violated
- `SafetyFilteredEnv` gym.Wrapper exposes `info["fallback"]` every step

Validated: random+safety drops collision rate from 0.800 → 0.500; IDM+safety
achieves 0 collisions with fallback_rate=0.003.

#### 1e. Adversarial Scenarios (`scenarios/adversarial.py`)

**Rationale:** Aggregate metrics over random seeds can miss systematic failure
modes.  Named scenarios make specific dangerous situations reproducible and
testable — the same methodology used in safety case arguments at AV companies.
A trained policy must pass the same scenario regression tests as the IDM expert.

Four scenarios (each stresses a distinct failure mode):

| Scenario | Setup | Key IDM result |
|---|---|---|
| `sudden_brake` | NPC 20 m ahead at 10 m/s — TTC 1.3 s | 50% collision; 20% ego fault |
| `close_merge` | NPC in adjacent lane at ego's x-position | 20% collision; 0% ego fault |
| `aggressive_rear` | NPC 5 m behind at ego+10 m/s — TTC 0.5 s | 100% collision; **100% NPC fault** |
| `dense_corridor` | 3 NPCs at 15/30/45 m, all at 20 m/s | 20% collision; high jerk |

The `aggressive_rear` result is the clearest safety case statement: 100% NPC fault
means the crash is physically unavoidable at 1 Hz — the policy cannot be blamed.
Any future policy should reproduce this attribution or the fault module needs review.

#### 1f. Tests (`tests/test_phase1.py`, `tests/test_scenarios.py`)

54 tests, all passing:
- IDM acceleration (7): zero gap, negative gap, free road, desired speed, headway, comfort
- TTC computation (5): mock env, no leader, same speed, formula, imminent
- Jerk computation (6): empty, short, constant speed, constant accel, known value, dt scaling
- Env wrapper (4): obs shape (25,), dtype float32, 5 actions, seed reproducibility
- Forward projection (5): front gap grows/shrinks/constant, rear gap closes/grows
- Action safety (9): SLOWER always safe, IDLE, FASTER, boundary lanes, LC
- Fault attribution (8): no NPCs, far NPC, overlapping, closing rear, adjacent lane, npc/ego/ambiguous
- Scenario regression (10): per-scenario thresholds on collision rate, ego fault rate, LC frequency

---

### Phase 2 — Imitation Learning: BC + DAgger ✅

**Goal:** Train a neural policy that imitates the IDM expert.  Start with
Behavioural Cloning (offline), then apply DAgger to recover from the
distribution-shift problem.

**Rationale:** BC from a fixed dataset suffers compounding errors — small
deviations at test time reach states never seen in training, causing cascading
failures.  Ross et al. (DAgger, AISTATS 2011) fix this by iteratively adding
expert-labelled data from the states the *learner* actually visits.

Architecture: MLP, 25 → 256 → 256 → 5 logits, cross-entropy loss, Adam.

Steps:
1. `policies/mlp_policy.py` — PyTorch MLP with `act(obs)` interface
2. `training/bc_train.py` — offline BC loop: collect expert rollouts → DataLoader
   → cross-entropy minimisation → save checkpoint
3. `training/dagger_train.py` — DAgger loop (Ross et al. Algorithm 1):
   - Initialise dataset D with expert rollouts
   - For N iterations: roll out current policy π_i, query IDM expert for labels,
     aggregate D ← D ∪ new data, retrain BC on D
4. Comparison table: IDM Expert | BC | DAgger-1 | DAgger-5

Key question: how many DAgger iterations are needed before the collision rate
matches the IDM expert on the adversarial scenarios?

References:
- Ross et al., "A Reduction of Imitation Learning and Structured Prediction to
  No-Regret Online Learning", AISTATS 2011 — https://arxiv.org/abs/1011.0686
- CS 285 Lecture 2 — https://rail.eecs.berkeley.edu/deeprlcourse/

---

### Phase 3 — Differentiable IRL Cost Optimizer ✅

**Goal:** Instead of hand-tuning reward weights, learn them from the expert
demonstrations using maximum-entropy IRL (Ziebart et al.).

**Rationale:** The IDM expert encodes implicit preferences (comfort, safety
margin, speed) that are hard to specify by hand.  IRL recovers a cost function
that rationalises the expert's behaviour under a soft-optimal policy model.
The recovered weights then seed Phase 4's PPO reward, creating a principled
bridge between imitation and RL.

Deliverables (`optimizer/`):
- `feature_extractor.py` — 8 action-conditioned features: speed×{faster,slower,idle},
  close×{slower,lc,idle}, lane_change, accel. 12-feature extension (LC gap features)
  fully implemented but commented out — requires ≥500 expert episodes to train.
- `irl_optimizer.py` — MaxEnt IRL: log-linear soft-optimal policy, NLL gradient
  descent (Adam, lr=0.05), per-feature L2 regularisation, early stopping.
- `weight_viz.py` — horizontal bar chart of learned weights saved to `results/`.

Validated results — IRL policy (20 episodes, seed=42):

| Metric | IDM Expert | DAgger-5 | IRL Policy |
|---|---|---|---|
| Collision rate | 0.000 | 0.100 | 0.050 |
| Goal completion | 1.000 | 0.900 | 0.950 |
| Mean min TTC | 12.63 s | 8.60 s | ∞ |
| RMS jerk | 5.536 m/s³ | 5.389 m/s³ | 0.155 m/s³ |
| Ego fault rate | 0.000 | 0.050 | 0.000 |

Learned weights (`results/irl_weights.npy`):

| Feature | Weight | Interpretation |
|---|---|---|
| speed×faster | −0.9817 | reward faster speeds |
| speed×slower | −4.0230 | strongly reward slower speeds (safety) |
| speed×idle | +1.9851 | penalise maintaining high speed while idle |
| close×slower | +1.0032 | penalise slowing behind close NPC |
| close×lc | +2.6101 | penalise LC into tight gap |
| close×idle | −1.8483 | reward maintaining gap when idle |
| lane_change | +0.0246 | slight penalty for unnecessary LCs |
| accel | −0.2338 | mild reward for acceleration |

Key lesson: 12-feature model (LC gap incentives) fails with single-step MaxEnt on 16
LC data points from 50 expert episodes — wrong-sign weights, collision regresses to
0.300. Needs ≥500 expert episodes or trajectory-level MaxEnt.

References:
- Ziebart et al., "Maximum Entropy Inverse Reinforcement Learning",
  AAAI 2008 — https://www.aaai.org/Papers/AAAI/2008/AAAI08-227.pdf

---

### Phase 4 — Safety-Constrained RL (CMDP + Lagrange) ✅

**Goal:** Fine-tune the imitation policy with PPO using the IRL reward, subject to
a formal safety constraint (collision rate ≤ ε), enforced via a learned Lagrange
multiplier λ.

**Rationale:** Pure RL on the raw highway-env reward explores aggressively and
produces unsafe behaviour early in training.  Framing safety as a Constrained
Markov Decision Process (CMDP) with a dual variable λ lets the optimiser trade
off performance vs. safety automatically — λ rises when the constraint is
violated and falls when it is slack.  This is more principled than the hard
override wrapper (which is retained as a last-resort backstop) and directly
mirrors the architecture described in recent AV safety literature.

Deliverables (`rl/`):
- `ppo_agent.py` — ActorCritic (25→256→256→5 + value head), GAE rollouts,
  warm-start from DAgger-5 MLP checkpoint.
- `reward_shaping.py` — `IRLRewardShaper`: scales IRL cost by expert dataset std,
  blends with env reward (`α=0.1`).  `r_total = (-w·φ)/scale + α·r_env`.
- `ppo_trainer.py` — clipped surrogate + entropy bonus + GAE (γ=0.99, λ_gae=0.95);
  n_steps=512, n_epochs=4, batch_size=64, lr=3e-4, clip_eps=0.2.
- `cmdp_trainer.py` — Lagrange dual gradient ascent: `λ ← clip(λ + lr·(col_rate − ε), 0, λ_max)`;
  collision_threshold=0.10, lambda_lr=0.05, lambda_max=10.0.
- `scripts/train_ppo.py` — entry point; saves checkpoints, prints 4-policy comparison.

Validated results (20 episodes, seed=42, 100 PPO iterations):

| Metric | IDM Expert | IRL Policy | PPO-unconstrained | PPO-CMDP |
|---|---|---|---|---|
| Collision rate | 0.000 | 0.050\* | 1.000 | **0.100** |
| Goal completion | 1.000 | 0.950\* | 0.000 | **0.900** |
| Mean min TTC | 12.63 s | ∞ | 1.17 s | ∞ |
| RMS jerk | 5.536 | 0.155 | 5.320 | 1.348 |
| Ego fault rate | 0.000 | 0.000 | 0.300 | 0.000 |
| Final λ | — | — | — | 0.149 |

\* Standalone validation; comparison-table result varies with env seed interaction.

Key findings:
1. **PPO-unconstrained degenerates** — learns to crash quickly to minimise
   accumulated IRL cost (episode termination exploitation).  LC freq = 1.000
   (lane-changes every step), all episodes collide.
2. **PPO-CMDP holds the constraint** — λ rises to 0.22 in early training to suppress
   collisions, then decays as the policy stabilises near the ε=0.10 boundary.
   Final λ=0.149 indicates the constraint is binding (active).
3. **Jerk improvement** — PPO-CMDP (1.348) is smoother than IDM (5.536) because
   the IRL cost penalises aggressive longitudinal commands.

References:
- Achiam et al., "Constrained Policy Optimization", ICML 2017 — https://arxiv.org/abs/1705.10528
- Schulman et al., "Proximal Policy Optimization", arXiv 2017 — https://arxiv.org/abs/1707.06347

---

## Novel Contributions

These distinguish the project from standard imitation-learning demos:

1. **Counterfactual fault attribution** — `ego_fault_rate` / `npc_fault_rate`
   distinguish avoidable from unavoidable crashes, enabling honest policy
   comparison.  Standard benchmarks only report aggregate collision rate.

2. **`lc_anticipatory_frac` metric** — tracks what fraction of lane changes are
   initiated *before* a threat closes in (TTC > 4 s).  Proactive behaviour is
   safer and more comfortable than reactive swerving.

3. **Named adversarial scenarios with regression tests** — four reproducible
   failure modes, each with a suite of assertions that any policy must pass.
   Methodology mirrors safety case arguments used at AV companies.

4. **IRL → PPO reward bridge** — the cost weights learned in Phase 3 directly
   initialise Phase 4's reward function, making the three-phase pipeline
   end-to-end coherent rather than three disconnected experiments.

5. **Learned Lagrange multiplier λ** (Phase 4) — ablated against the hard-override
   safety wrapper to show the trade-off between constraint softness and performance.

---

## File Structure

```
highway-planner/
├── envs/
│   └── highway_wrapper.py       # make_env(), pinned config
├── policies/
│   ├── idm_expert.py            # IDM + MOBIL expert, collect_expert_rollouts()
│   └── mlp_policy.py            # ⬜ MLP policy (Phase 2)
├── metrics/
│   ├── evaluator.py             # evaluate(), EvalResults, print_table()
│   └── fault_attribution.py     # snapshot_pre_step(), classify_fault()
├── safety/
│   └── safety_wrapper.py        # SafetyWrapper, SafetyFilteredEnv
├── scenarios/
│   └── adversarial.py           # 4 named scenarios, run_all_scenarios()
├── training/
│   ├── bc_train.py              # ⬜ BC training loop (Phase 2)
│   └── dagger_train.py          # ⬜ DAgger loop (Phase 2)
├── optimizer/
│   └── ...                      # ⬜ IRL cost optimizer (Phase 3)
├── rl/
│   └── ...                      # ⬜ PPO + CMDP (Phase 4)
├── tests/
│   ├── test_phase1.py           # 44 unit tests
│   └── test_scenarios.py        # 10 scenario regression tests
└── pyproject.toml
```

---

## Status

| Component | Status | Tests |
|---|---|---|
| `envs/highway_wrapper.py` | ✅ Complete | 4 |
| `policies/idm_expert.py` | ✅ Complete | 7 |
| `metrics/evaluator.py` | ✅ Complete | 11 |
| `metrics/fault_attribution.py` | ✅ Complete | 8 |
| `safety/safety_wrapper.py` | ✅ Complete | 14 |
| `scenarios/adversarial.py` | ✅ Complete | 10 |
| `policies/mlp_policy.py` | ✅ Complete | 11 |
| `training/bc_train.py` | ✅ Complete | 12 |
| `training/dagger_train.py` | ✅ Complete | 12 |
| `optimizer/feature_extractor.py` | ✅ Complete | 22 |
| `optimizer/irl_optimizer.py` | ✅ Complete | 21 |
| `optimizer/weight_viz.py` | ✅ Complete | 8 |
| `rl/ppo_agent.py` | ✅ Complete | 14 |
| `rl/reward_shaping.py` | ✅ Complete | 4 |
| `rl/ppo_trainer.py` | ✅ Complete | 11 |
| `rl/cmdp_trainer.py` | ✅ Complete | 5 |
| `scripts/train_ppo.py` | ✅ Complete | — |

**Total tests: 174 / 174 passing**
