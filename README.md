# PowerGridReconfig

## Project overview

PowerGridReconfig is a Python 3.11 research framework for emergency topology control in power-system simulations. The current system implements a pool-guided AlphaZero-like self-play loop for the case118 research setup: a fixed physical scenario pool is sampled, a model-guided neural MCTS planner generates replay, a graph policy-value checkpoint is fine-tuned, a fixed evaluation set is evaluated, and the candidate checkpoint is accepted or rejected.

This is not a full classical AlphaZero system. The scenario pool is fixed, generation is model-guided, the continuation gate can change the executed action, replay accumulates across iterations, candidates are fine-tuned from checkpoints, evaluation is fixed, and acceptance/rejection is based on configured metrics.

For implementation details, see [docs/self_play.md](docs/self_play.md).

## Current implemented pipeline

```text
fixed physical scenario pool
  -> prioritized scenario sampling
  -> neural MCTS generation
  -> replay update
  -> scenario-level train/validation split
  -> checkpoint fine-tuning
  -> fixed evaluation
  -> acceptance or rejection
  -> next iteration
```

## Scientific invariants

- `solved=True` means exactly `assessment.physically_secure=True`; thermal
  feasibility alone is never a solved episode.
- Generation `PF_ALG` must equal evaluation `PF_ALG`.
- The current canonical `PF_ALG` is `3`.
- The MCTS visit distribution is the policy target.
- The continuation gate can change the executed action, but it does not change the MCTS visit policy target.
- `outcome_value_target` is required for graph training examples.
- Feature normalization is part of the checkpoint contract.
- Fine-tuning preserves normalization statistics from the parent checkpoint.
- The train/validation split is performed by `scenario_id`, not by individual rows.
- The candidate checkpoint is selected by validation loss when a validation set exists.
- The evaluation set is fixed across candidates.
- The acceptance metric is normally `solve_rate`.
- Bootstrap metrics must include compatible `pf_alg` provenance.

## Physical success contract

The authoritative success predicate is:

```text
physically_secure =
    power_flow_converged
    and all_values_finite
    and topology_connected
    and thermal_feasible
    and voltage_feasible
    and generator_p_feasible
    and generator_q_feasible
    and angle_difference_feasible
```

Limits are evaluated on the actual PYPOWER/GridFM bus, branch, and generator
arrays. Active generators are checked individually. Disabled branches and
generators are ignored, `RATE_A=0` is unconstrained, branch angle limits use
MATPOWER semantics, and non-finite or missing mandatory data fail closed.

`thermal_solved` is retained as a diagnostic meaning that no active, rated
branch exceeds its thermal limit. `physically_secure` is the only success
criterion. `done` only means the episode ended: a PF failure, maximum step
count, or explicit handoff is done but not solved. A handoff transfers control
to redispatch and remains separate from solved.

GridFM parquet rows do not carry trustworthy PF convergence provenance, so
environment reset performs a no-op AC power flow before the initial state can
be classified or searched.

## Artifact compatibility

The physical-objective, outcome/value-target, evaluation-metrics, checkpoint,
and replay contracts are versioned. The current versions are all `2`. Examples
and replay rows carry the physical and outcome versions; checkpoints carry all
training-semantic versions; evaluation JSON contains the evaluation version
and the nested physical contract. Consumers require exact versions.

Artifacts produced under the former thermal-only solved definition are
scientifically incompatible. They are rejected with a regeneration command;
they are never relabeled or upgraded in place. Regenerate in this order:

```bash
# 1. Fresh episodes and outcome targets (the generator now writes both).
python -m scripts.self_play.generate <POOL_RAW_DIR> --transitions <POOL_TRANSITIONS.csv> --output-dir <NEW_SELF_PLAY_DIR> --pf-alg 3

# 2. Fresh checkpoint; do not initialize from a legacy checkpoint.
python -m scripts.self_play.train_graph_baseline <NEW_EXAMPLES_CSV> --output <NEW_CHECKPOINT.pt> --device cpu

# 3. Fresh fixed-evaluation rows and metrics.
python -m scripts.evaluation.evaluate_checkpoint <EVAL_RAW_DIR> --transitions <EVAL_TRANSITIONS.csv> --checkpoint <NEW_CHECKPOINT.pt> --pf-alg 3 --output-csv <NEW_EVAL_RESULTS.csv> --output-json <NEW_EVAL_METRICS.json>
```

Archive the old replay/run directory, point the self-play YAML bootstrap paths
at the new checkpoint and metrics, then start a new run. Do not mix old and new
examples, replay chunks, checkpoints, or evaluation metrics.

## Action space

The topology-control action space is:

- `0` -> stop/handoff;
- `1..N` -> branch opening actions.

Handoff means the topology-control episode ends and the case is passed to an external or future redispatch layer. Production redispatch optimization is not implemented here.

## Installation

Use Python 3.11.

Windows PowerShell:

```powershell
py -3.11 -m venv .venv311
.\.venv311\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

Linux:

```bash
python3.11 -m venv .venv311
source .venv311/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

Dependency files:

- `pyproject.toml` is the canonical dependency definition.
- `constraints/py311.txt` records the tested Python 3.11 compatibility constraints.
- `requirements.txt` is the full developer installation entry point.

## Quick validation

```bash
python -m compileall -q grid_topology_ai scripts tests
python -m pytest -q
python -m scripts.self_play.loop --help
python -m scripts.self_play.loop configs/self_play_loop_pilot.yaml --plan-only
```

## Running self-play

Run the pilot loop with:

```bash
python -m scripts.self_play.loop configs/self_play_loop_pilot.yaml
```

Useful modes:

- `--plan-only` prints and validates the execution plan without running generation, training, or evaluation.
- `--validate-only` validates configuration and required artifact references.
- `--resume` continues only after the latest iteration that has a valid `iteration_complete.json` marker. Incomplete iteration directories cause refusal instead of silent reuse.

## Configuration

The self-play YAML is organized into these sections:

- `pool`: fixed scenario pool transitions and raw state directory.
- `replay_buffer`: accumulated replay storage and sampling limits.
- `generation`: neural MCTS generation settings, including `PF_ALG`.
- `training`: graph policy-value fine-tuning settings.
- `evaluation`: fixed evaluation transitions, raw states, checkpoint evaluation, and `PF_ALG`.
- `acceptance`: candidate acceptance metric and thresholds.
- `metadata`: run naming and reproducibility metadata.

## Iteration artifacts

A typical run directory contains:

```text
runs/<run_name>/
  run_state.json
  learning_curve.csv
  replay/
  checkpoints/
    best.pt
    best_metrics.json
  iter_001/
    selected_scenario_ids.txt
    raw/
      selected_transitions.csv
      examples.csv
      generate.log
    train_batch.csv
    train_examples.csv
    validation_examples.csv
    train_validation_split.json
    candidate_checkpoint.pt
    train_metrics.csv
    eval_results.csv
    eval_metrics.json
    metadata.json
    iteration_complete.json
```

The atomic completion marker is `iteration_complete.json`.

## Package structure

- `grid_topology_ai/config`: typed configuration and validation.
- `grid_topology_ai/self_play`: pool state, sampling, replay, acceptance, artifacts, and loop support.
- `grid_topology_ai/training`: graph policy-value training, checkpoints, metrics, and splits.
- `grid_topology_ai/evaluation`: checkpoint evaluation and metrics.
- `grid_topology_ai/search`: MCTS planning components.
- `grid_topology_ai/models`: graph datasets and neural models.
- `scripts/self_play`: self-play loop and training CLIs.
- `scripts/evaluation`: evaluation CLIs including `python -m scripts.evaluation.evaluate_checkpoint`.
- `tests`: unit, contract, and smoke tests.

Public entry points kept stable:

```bash
python -m scripts.self_play.loop
python -m scripts.self_play.train_graph_baseline
python -m scripts.evaluation.evaluate_checkpoint
```

## Bootstrap preparation

Before the first real run, prepare:

- scenario pool transitions CSV;
- raw state directory for the pool;
- bootstrap checkpoint;
- fixed evaluation transitions CSV;
- fixed evaluation raw state directory;
- bootstrap evaluation metrics.

Bootstrap evaluation metrics must be computed with the same `PF_ALG` configured for generation and evaluation.

## Testing and CI

GitHub Actions cover:

- Ubuntu tests;
- Windows tests;
- package build;
- data tools smoke.

Local graph dataset integration tests are opt-in because they need prepared local data artifacts.

## Current limitations

- Actions are topology branch openings plus stop/handoff.
- The main research setup is one case118 configuration.
- Scenarios come from a fixed pool rather than unrestricted environment generation.
- There is no production redispatch optimizer.
- No iteration is guaranteed to improve `solve_rate`.
- Real self-play is expensive and requires prepared data and checkpoint artifacts.
- This is research code, not operational grid control software.

## Legacy teacher pipeline

Teacher generators remain useful for bootstrap datasets, baseline comparison, and debugging. They are no longer the only documented training route; the implemented self-play loop is the current integrated pipeline.

## Reproducibility

Reproducibility relies on Python 3.11, pinned constraints, explicit seeds, artifact hashes, fixed evaluation data, checkpoint provenance, and CI checks. Checkpoints store selection metadata, normalization metadata, dataset metadata, and training configuration.
