# PowerGridReconfig

PowerGridReconfig is a research and engineering project for automatic power-grid emergency control.

The project focuses on learning how to improve stressed grid states by changing the network topology - mainly by switching transmission branches - and, later, by combining topology actions with redispatch and self-play planning.

The long-term direction is an AlphaZero-like control system for power grids:

```text
grid state
-> graph neural network policy/value model
-> planning/search
-> corrective action
-> updated grid state
-> new training experience
-> improved model
```

The current pipeline is teacher-based supervised learning. A search-based teacher generates corrective trajectories, and a graph neural network learns:

* a policy over topology-switching actions;
* a value estimate for the current grid state.

---

## Table of Contents

1. [Project Goal](#1-project-goal)
2. [Current Control Problem](#2-current-control-problem)
3. [Action Space](#3-action-space)
4. [High-Level Architecture](#4-high-level-architecture)
5. [Repository Structure](#5-repository-structure)
6. [Data Layout](#6-data-layout)
7. [Core Package](#7-core-package)
8. [Search and Planning](#8-search-and-planning)
9. [Machine Learning Components](#9-machine-learning-components)
10. [Main Scripts](#10-main-scripts)
11. [Training Pipeline](#11-training-pipeline)
12. [Value Targets](#12-value-targets)
13. [Policy Targets](#13-policy-targets)
14. [Current Working Datasets](#14-current-working-datasets)
15. [Testing](#15-testing)
16. [Temporary Runs and Archive Policy](#16-temporary-runs-and-archive-policy)
17. [Naming Conventions](#17-naming-conventions)
18. [Developer Workflow](#18-developer-workflow)
19. [Known Issues and Roadmap](#19-known-issues-and-roadmap)
20. [Project Summary](#20-project-summary)

---

## 1. Project Goal

The goal of PowerGridReconfig is to build an intelligent topology-control system for overloaded power grids.

Given a stressed grid state, the system should be able to:

1. detect overloads;
2. represent the grid as a graph;
3. generate valid topology-switching actions;
4. estimate the effect of each action;
5. select a corrective action;
6. continue planning if needed;
7. stop and hand off to redispatch when topology switching is no longer the right tool.

The project is not intended to be only a static classifier. The intended final system should learn through a closed loop:

```text
teacher/planner -> dataset -> GNN training -> model-guided planning -> replay data -> retraining
```

The current stage is supervised learning from a strong teacher. The next stage is model-guided planning and self-play.

---

## 2. Current Control Problem

The project currently focuses on topology switching for overloaded power grids.

The system receives a grid state with:

* bus features;
* branch features;
* branch loading information;
* topology information;
* action mask;
* overload indicators.

The system must decide whether to:

* switch off a branch;
* continue topology optimization;
* stop and hand the case over to redispatch.

A good action should improve the grid state by reducing overloads and avoiding invalid or unstable configurations.

---

## 3. Action Space

The current action space is branch-based.

```text
action 0      -> stop / handoff to redispatch
action k > 0  -> switch off branch with branch_pos = k - 1
```

For a grid with `N` branches:

```text
num_actions = N + 1
```

For the current IEEE case118 setup:

```text
num_buses    = 118
num_branches = 186
num_actions  = 187
```

The stop/handoff action is important. It allows the model to decide that topology switching should stop and the case should be passed to a redispatch module.

---

## 4. High-Level Architecture

The project is organized around five layers.

```text
1. Data layer
   Raw states, transitions, generated PF data, teacher datasets.

2. Simulation layer
   Environment, power-flow backend, reward function.

3. Planning layer
   Beam search, LODF screening, MCTS experiments.

4. Machine learning layer
   Graph dataset, GNN policy-value model, training and evaluation.

5. Future self-play layer
   Model-guided planning, replay buffer and retraining loop.
```

At the current stage, the strongest part of the project is the supervised teacher-learning pipeline.

The next major milestone is closing the full self-play loop.

---

## 5. Repository Structure

```text
PowerGridReconfig/
  commands/
  configs/
  data/
  grid_topology_ai/
  notebooks/
  requirements.txt
  scripts/
  set_temp_gridfm/
  tests/
```

### `commands/`

Helper commands and command templates.

This folder is useful for storing repeatable command lines for generation, training, evaluation and debugging.

### `configs/`

Configuration files for experiments and data generation.

This folder should contain stable configuration files, not temporary run outputs.

### `data/`

Datasets, generated states, training data, debug runs, archived runs and generated GridFM/PF data.

The `data/` directory should stay organized. Temporary data belongs in `data/_scratch/`; old runs belong in `data/_archive/`.

### `grid_topology_ai/`

Main Python package.

It contains:

* action space;
* data adapter;
* environment;
* power-flow backend;
* reward logic;
* state store;
* transition generator;
* search/planning components;
* graph neural network models.

### `notebooks/`

Exploratory notebooks.

Notebooks should be used for analysis, visualization and research experiments. Production logic should eventually move into Python modules or scripts.

### `scripts/`

Runnable scripts for data generation, teacher generation, training, evaluation and planning.

### `set_temp_gridfm/`

Temporary or setup-related GridFM files.

This folder should be reviewed periodically. If some files become obsolete, move them to `data/_archive/` or remove them.

### `tests/`

Unit tests and regression tests.

Tests are used to verify that dataset loading, graph models and core training components still work after code changes.

---

## 6. Data Layout

The `data/` directory must remain organized. It should not become a dump for experiments.

Recommended structure:

```text
data/
  datasets/
  gridfm_generated/
  gridfm_transitions/
  profiles/
  self_play/
  training/
  _scratch/
  _archive/
```

### `data/datasets/`

Main prepared grid datasets.

Current working dataset:

```text
data/datasets/
  case118_balanced_v1/
    raw/
    transitions/
    configs/
    manifest/
```

Purpose:

* `raw/` - raw grid states;
* `transitions/` - transition/scenario tables;
* `configs/` - generation configuration files;
* `manifest/` - dataset metadata.

Build artifacts such as `chunks/` and `logs/` should not stay here permanently. They should be moved to:

```text
data/_archive/dataset_build_artifacts/
```

### `data/gridfm_generated/`

Large generated GridFM/PF datasets.

Current examples:

```text
data/gridfm_generated/
  case118_1000_pf/
  case118_60000_pf/
  case118_subsets/
```

This data is generated source data, not temporary debug output.

### `data/gridfm_transitions/`

Transition data used by GridFM-related workflows.

This folder should contain reusable transition artifacts, not debug transitions.

### `data/profiles/`

Load, generation or scenario profiles.

This data is usually used as an input source for dataset generation.

### `data/self_play/`

Working teacher/self-play/training datasets.

Current working structure:

```text
data/self_play/
  impact_teacher_balanced_v1_simple/
  impact_teacher_balanced_v1_medium/
  impact_teacher_balanced_v1_hard_lodf_k100/
  impact_teacher_balanced_v1_mixed_lodf/
  impact_teacher_v5_train5000_split/
```

Purpose:

* `impact_teacher_balanced_v1_simple/` - simple teacher scenarios;
* `impact_teacher_balanced_v1_medium/` - medium-difficulty scenarios;
* `impact_teacher_balanced_v1_hard_lodf_k100/` - hard scenarios generated with LODF screening;
* `impact_teacher_balanced_v1_mixed_lodf/` - mixed dataset built from simple, medium and hard data;
* `impact_teacher_v5_train5000_split/` - older baseline dataset/checkpoint kept for comparison.

### `data/training/`

Training outputs.

This folder should contain final or important training runs.

Recommended examples:

```text
data/training/
  gnn_v2_mixed_lodf_v1/
  gnn_v2_mixed_lodf_outcome_v1/
```

Debug training runs should not go here. They should go to:

```text
data/_scratch/training/
```

### `data/_scratch/`

Temporary runs.

Everything with names like the following should go here:

```text
debug_*
test_*
smoke_*
cache_*
*_test*
```

Example:

```text
data/_scratch/
  datasets/
  gridfm/
  self_play/
  training/
```

Rule:

```text
If it can be deleted without losing the main research result, it belongs in _scratch.
```

The whole scratch directory can be removed when needed:

```bash
rm -rf data/_scratch
```

### `data/_archive/`

Old experiments and build artifacts that are no longer part of the active workflow but should not be deleted yet.

Example:

```text
data/_archive/
  self_play_old_runs/
  dataset_build_artifacts/
```

Archive data should only be deleted after a successful new full cycle:

```text
generation -> training -> evaluation -> comparison with baseline
```

---

## 7. Core Package

Main package:

```text
grid_topology_ai/
```

### `action_space.py`

Defines the topology-switching action space.

Main convention:

```text
0      -> stop / handoff
1..N   -> switch off branch 0..N-1
```

This file should stay stable because many components depend on the action indexing convention.

### `data_adapter.py`

Converts stored grid data into the internal representation used by the environment and neural network.

Responsible for:

* bus features;
* branch features;
* graph connectivity;
* branch/action indexing;
* feature column definitions.

Typical tensors:

```text
bus_features
branch_features
edge_index
action_mask
```

### `environment.py`

Applies topology actions to a grid state.

Conceptually:

```text
state + action -> next_state + reward + done + info
```

It coordinates:

* current grid state;
* action application;
* power-flow backend calls;
* reward calculation;
* terminal condition logic.

### `pypower_backend.py`

Runs power-flow calculations.

It determines whether a topology action produces a valid grid state and returns resulting flows, loadings and convergence information.

Used by:

* environment;
* teacher generation;
* planning;
* reward evaluation.

### `reward.py`

Defines the reward function.

Reward is used by the teacher/search system to score actions and trajectories.

It should reward:

* reducing overloads;
* reducing severe overloads;
* improving the safety score;
* reaching a solved state.

It should penalize:

* invalid actions;
* worsening the grid;
* non-convergent power flow;
* remaining overloaded states.

Important distinction:

```text
reward is used by teacher/search
value target is used by the neural network
```

These are related, but they are not the same thing.

### `state_store.py`

Stores and loads grid states.

Teacher datasets usually consist of:

```text
examples.csv
states/*.npz
```

`state_store.py` is responsible for writing and reading those `.npz` files.

### `transition_generator.py`

Generates transition/scenario tables.

This normally runs before teacher generation.

---

## 8. Search and Planning

Search code lives under:

```text
grid_topology_ai/search/
```

### `impact_beam_search.py`

Beam-search planner used by the impact teacher.

Rough workflow:

```text
current state
-> candidate actions
-> simulate each action
-> score resulting states
-> keep best beams
-> repeat
```

This is one of the main components of the teacher generation pipeline.

### `continuation_gate.py`

Decides whether topology switching should continue or whether the system should stop and hand off to redispatch.

This is important because not every overloaded situation should be handled only by switching lines.

### MCTS-related components

MCTS is part of the future AlphaZero-like direction.

The intended future connection is:

```text
neural evaluator
-> MCTS planning
-> improved policy target
-> replay buffer
-> retraining
```

At the moment, MCTS is experimental infrastructure rather than the main training pipeline.

---

## 9. Machine Learning Components

Model code lives under:

```text
grid_topology_ai/models/
```

### `graph_self_play_dataset.py`

Main dataset class for graph-based training.

It reads:

```text
examples.csv
states/*.npz
```

And returns graph tensors:

```text
bus_features
branch_features
edge_index
action_mask
target_policy
target_value
```

Value target priority:

```text
1. outcome_value_target
2. value_target
3. legacy discounted_return_from_step / value_scale
```

This keeps old datasets usable while allowing new datasets to use better value targets.

### `graph_policy_value_net.py`

Older graph policy-value model.

Mostly kept as a baseline or legacy reference.

### `graph_policy_value_net_v2.py`

Current main GNN model.

Design:

* pure PyTorch;
* no `torch_geometric`;
* no `torch_scatter`;
* edge-aware message passing;
* branch embeddings are central because actions are branch actions;
* policy head outputs logits for all actions;
* value head outputs a scalar in `[-1, 1]`.

The value head uses a bounded output, so value targets must also be bounded.

### `neural_evaluator.py`

Wrapper for using a trained neural network during planning.

Expected role:

```text
state -> policy probabilities + value estimate
```

This is required for MCTS and model-guided planning.

---

## 10. Main Scripts

### `scripts/self_play/generate_impact_teacher_parallel_fast.py`

Main fast teacher generator.

Responsibilities:

* read transition scenarios;
* create environments;
* run impact beam search;
* apply topology actions;
* save state files;
* write `examples.csv`;
* use multiprocessing;
* use LODF screening when enabled;
* generate teacher policy targets;
* generate value targets.

New generated datasets should include:

```text
value_target
outcome_value_target
outcome_class
outcome_steps_to_terminal
outcome_value_target_mode
```

The main value target for training should be:

```text
outcome_value_target
```

### `scripts/self_play/train_graph_baseline.py`

Main training script for graph policy-value models.

Responsibilities:

* load train and validation datasets;
* create `GraphSelfPlayDataset`;
* create `GraphPolicyValueNet` or `GraphPolicyValueNetV2`;
* train policy and value heads;
* log metrics;
* save checkpoints;
* save best checkpoint variants;
* store dataset/model metadata;
* print value target diagnostics.

Checkpoint variants:

```text
*_best_loss.pt
*_best_top1.pt
*_best_top5.pt
*_best_switch.pt
*_best_policy.pt
*_last.pt
```

### `scripts/evaluation/evaluate_examples_with_checkpoint.py`

Evaluation script.

It loads a checkpoint and evaluates it on an examples CSV.

Main metrics:

```text
loss
policy_loss
value_loss
top1
top3
top5
stop_acc
switch_acc
```

### `scripts/generate_transitions.py`

Generates transition/scenario tables from prepared grid data.

Runs before teacher generation.

### `scripts/planning/run_beam_search.py`

Runs beam search planning on selected scenarios.

Useful for debugging the planner.

### `scripts/planning/run_mcts.py`

Runs MCTS experiments.

Part of the future self-play direction.

### `scripts/check_gridfm_reward.py`

Checks reward behavior on selected states.

Useful when reward logic changes.

---

## 11. Training Pipeline

### 11.1 Prepare base dataset

Base grid data lives here:

```text
data/datasets/case118_balanced_v1/
```

Important subfolders:

```text
raw/
transitions/
configs/
manifest/
```

### 11.2 Generate teacher trajectories

Teacher generation creates:

```text
examples.csv
states/*.npz
```

Example output folders:

```text
data/self_play/impact_teacher_balanced_v1_simple/
data/self_play/impact_teacher_balanced_v1_medium/
data/self_play/impact_teacher_balanced_v1_hard_lodf_k100/
```

### 11.3 Build mixed dataset

The mixed dataset combines simple, medium and hard examples.

Current main mixed dataset:

```text
data/self_play/impact_teacher_balanced_v1_mixed_lodf/
  examples.csv
  examples_train.csv
  examples_val.csv
  states/
```

The split should be scenario-based, not row-based.
Examples from the same scenario must not appear in both train and validation.

### 11.4 Train GNN

Example command:

```bash
python -u -m scripts.self_play.train_graph_baseline \
  data/self_play/impact_teacher_balanced_v1_mixed_lodf/examples_train.csv \
  --val-examples-csv data/self_play/impact_teacher_balanced_v1_mixed_lodf/examples_val.csv \
  --output data/training/gnn_v2_mixed_lodf_v1/graph_policy_value_net_v2.pt \
  --metrics-csv data/training/gnn_v2_mixed_lodf_v1/metrics.csv \
  --model-type graph_v2 \
  --hidden-dim 128 \
  --num-layers 4 \
  --dropout 0.10 \
  --epochs 120 \
  --batch-size 64 \
  --lr 0.001 \
  --value-loss-weight 1.0 \
  --device cuda \
  --amp \
  --num-workers 0 \
  --save-best \
  --save-multiple-best
```

### 11.5 Evaluate checkpoint

```bash
python -m scripts.evaluation.evaluate_examples_with_checkpoint \
  --examples-csv data/self_play/impact_teacher_balanced_v1_mixed_lodf/examples_val.csv \
  --checkpoint data/training/gnn_v2_mixed_lodf_v1/graph_policy_value_net_v2_best_policy.pt \
  --device cuda \
  --amp \
  --batch-size 64
```

---

## 12. Value Targets

The old value target logic was:

```text
discounted_return_from_step / value_scale
clip(-1, 1)
```

Problems:

* manual scale selection;
* saturated targets;
* weak value gradients;
* inconsistent behavior across datasets;
* target based on reward shaping instead of final outcome.

The new target system should be:

```text
1. outcome_value_target - main target for value head
2. value_target - dense normalized reward target
3. legacy fallback - for old datasets only
```

### 12.1 Dense `value_target`

Dense reward target:

```text
r_norm_t = tanh(step_reward_t / 7000)

value_target_t =
  sum(gamma^k * r_norm_{t+k})
  /
  sum(gamma^k)
```

This stays in `[-1, 1]`.

It is useful for diagnostics and possible auxiliary training.

### 12.2 `outcome_value_target`

Outcome-based AlphaZero-like target:

```text
solved                         -> +1.0
handoff_to_redispatch_teacher  ->  0.0
max_steps_reached              -> -1.0
```

For each step:

```text
outcome_value_target_t = terminal_value * gamma^(steps_to_terminal)
```

This is the preferred target for the value head.

---

## 13. Policy Targets

Policy targets are produced by the teacher.

They are stored in:

```text
mcts_policy_json
```

Even when the teacher is not true MCTS, this field is used as the generic policy distribution field.

Most current teacher examples are one-hot:

```text
best teacher action -> probability 1.0
all other actions   -> probability 0.0
```

Later, MCTS should produce softer policy distributions.

---

## 14. Current Working Datasets

### `impact_teacher_balanced_v1_simple`

Simple scenarios.

Used to check that the model can learn easy cases and stop/handoff logic.

### `impact_teacher_balanced_v1_medium`

Medium difficulty.

Used for realistic multi-step switching decisions.

### `impact_teacher_balanced_v1_hard_lodf_k100`

Hard scenarios with LODF candidate screening.

Important for teaching difficult topology decisions.

### `impact_teacher_balanced_v1_mixed_lodf`

Main current mixed training dataset.

Built from simple, medium and hard examples.

### `impact_teacher_v5_train5000_split`

Older baseline.

Kept for comparison until a stronger new model is trained and evaluated.

---

## 15. Testing

Tests live in:

```text
tests/
```

Important tests:

```text
test_graph_policy_value_net_v2.py
test_graph_self_play_dataset.py
```

Run:

```bash
python -m pytest tests -q
```

Before serious training or generation, run:

```bash
python -m py_compile scripts/self_play/generate_impact_teacher_parallel_fast.py
python -m py_compile scripts/self_play/train_graph_baseline.py
python -m py_compile grid_topology_ai/models/graph_self_play_dataset.py
python -m pytest tests -q
```

---

## 16. Temporary Runs and Archive Policy

Debug runs should go into `_scratch`.

Example:

```bash
python -u -m scripts.self_play.generate_impact_teacher_parallel_fast \
  data/datasets/case118_balanced_v1/raw \
  --transitions data/datasets/case118_balanced_v1/transitions/transitions_teacher_hard100.csv \
  --output-dir data/_scratch/self_play/debug_generation_test20 \
  --limit 20 \
  --num-workers 10 \
  --batch-size 2 \
  --use-lodf-screening \
  --lodf-screen-top-k 100 \
  --lodf-min-candidate-count 8 \
  --max-steps 6 \
  --max-teacher-steps 6 \
  --quiet-success
```

Rule:

```text
If it is a debug run, test run, smoke run or cache - put it in _scratch.
```

---

## 17. Naming Conventions

### Production teacher datasets

```text
data/self_play/impact_teacher_balanced_v2_simple/
data/self_play/impact_teacher_balanced_v2_medium/
data/self_play/impact_teacher_balanced_v2_hard_lodf_k100/
data/self_play/impact_teacher_balanced_v2_mixed_lodf/
```

### Training outputs

```text
data/training/gnn_v2_mixed_lodf_v1/
data/training/gnn_v2_mixed_lodf_outcome_v1/
```

### Debug outputs

```text
data/_scratch/self_play/debug_...
data/_scratch/training/debug_...
```

### Archived outputs

```text
data/_archive/...
```

---

## 18. Developer Workflow

### 18.1 Check syntax and tests

```bash
python -m py_compile scripts/self_play/generate_impact_teacher_parallel_fast.py
python -m py_compile scripts/self_play/train_graph_baseline.py
python -m py_compile grid_topology_ai/models/graph_self_play_dataset.py
python -m pytest tests -q
```

### 18.2 Run a small teacher generation

```bash
python -u -m scripts.self_play.generate_impact_teacher_parallel_fast \
  data/datasets/case118_balanced_v1/raw \
  --transitions data/datasets/case118_balanced_v1/transitions/transitions_teacher_hard100.csv \
  --output-dir data/_scratch/self_play/debug_generation_test20 \
  --limit 20 \
  --num-workers 10 \
  --batch-size 2 \
  --use-lodf-screening \
  --lodf-screen-top-k 100 \
  --lodf-min-candidate-count 8 \
  --max-steps 6 \
  --max-teacher-steps 6 \
  --quiet-success
```

### 18.3 Inspect output CSV

```bash
python -c "import pandas as pd; p='data/_scratch/self_play/debug_generation_test20/examples.csv'; df=pd.read_csv(p); print(df.head()); print(df.columns); print(df.shape)"
```

### 18.4 Run a short training test

```bash
python -u -m scripts.self_play.train_graph_baseline \
  data/_scratch/self_play/debug_generation_test20/examples.csv \
  --val-examples-csv data/_scratch/self_play/debug_generation_test20/examples.csv \
  --output data/_scratch/training/debug_train/graph_policy_value_net_v2.pt \
  --metrics-csv data/_scratch/training/debug_train/metrics.csv \
  --model-type graph_v2 \
  --epochs 1 \
  --batch-size 32 \
  --device cuda \
  --amp \
  --num-workers 0 \
  --save-best \
  --save-multiple-best \
  --no-tensorboard
```

---

## 19. Known Issues and Roadmap

### Recently improved

* Vectorized aggregation in GNN v2.
* Train/validation normalization stats handling.
* Graph dataset and model tests.
* Checkpoint metadata.
* Dataset versioning.
* Better bounded dense `value_target`.
* Cleaner `data/` directory structure.

### Next tasks

1. Make `outcome_value_target` the default value target for future generations.
2. Regenerate simple, medium and hard datasets with the new target logic.
3. Build a new mixed dataset.
4. Train GNN v2 on the new mixed dataset.
5. Compare the new model with the old baseline.
6. Close the self-play loop:

   * model inference;
   * planning;
   * replay buffer;
   * retraining.
7. Add more grid topologies.
8. Add N-k perturbations instead of only N-1.
9. Add generation dispatch variation.
10. Add curriculum learning by scenario difficulty.

---

## 20. Project Summary

PowerGridReconfig builds an intelligent topology-control system for overloaded power grids. The grid is treated as a graph, actions correspond to topology changes, a teacher/planner generates corrective trajectories, and a graph neural network learns both a policy over switching actions and a value estimate for the current state.

The current pipeline is teacher-based supervised learning. The long-term goal is a closed AlphaZero-like loop where planning, self-play and retraining continuously improve the model.
