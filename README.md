# PowerGridReconfig

## Project overview

PowerGridReconfig is a Python 3.11 research framework for emergency topology control in power-system simulations. The Light runtime exposes explicit teacher, self-play, training, and checkpoint-evaluation workflows instead of an integrated acceptance/promotion loop.

The main learned-control path uses model-guided neural MCTS over a fixed physical scenario pool. Teacher generation remains available as an optional deterministic bootstrap path. Training and evaluation are separate explicit operations, and evaluation runs one selected policy behavior per invocation.

For implementation details, see [docs/self_play.md](docs/self_play.md). The stable bidirectional branch-status policy and artifact compatibility rules are specified in [docs/topology_action_contract.md](docs/research/topology_action_contract.md).

## Light reference pipeline

```text
GridFM/raw scenario
  -> canonical state and scenario data (`grid_topology_ai.state`, `.data`)
  -> optimized AC power flow (`grid_topology_ai.power_flow`)
  -> stable topology action space (`grid_topology_ai.actions`)
  -> deterministic teacher search (`grid_topology_ai.search.teacher`)
  -> graph examples and dataset (`grid_topology_ai.dataset`)
  -> GraphPolicyValueNetV2 training (`grid_topology_ai.model`, `.training`)
  -> neural MCTS self-play (`grid_topology_ai.search.mcts`, `.self_play`)
  -> scientific evaluation (`grid_topology_ai.evaluation`)
```

Each stage is invoked directly. The Light runtime does not contain a checkpoint arena, automatic promotion/acceptance loop, curriculum controller, or sealed final-test orchestrator.

The memory-mapped scenario representation and worker-safe adapters have one
runtime owner, `grid_topology_ai.runtime`. Model inference is intentionally
separate from scientific evaluation and is owned by
`grid_topology_ai.evaluator`. The command-line entry point is
`power-grid-reconfig` (equivalently `python -m grid_topology_ai.cli`).

## Evaluation contract

Evaluation runs exactly one policy mode per invocation. `EvaluationConfig.policy_mode` is the canonical selector:

- `ungated` is the default learned-controller behavior: neural policy plus MCTS without continuation-gate filtering;
- `constrained` applies continuation filtering to the root policy before action selection and is selected by CLI with `--use-continuation-gate`.

Evaluation produces canonical per-scenario physical outcome fields and aggregate metrics for that single run. It does not automatically run paired ungated/constrained comparisons, bootstrap confidence intervals, checkpoint ranking, promotion, or acceptance gates.

## Scientific invariants

- `solved=True` means exactly `assessment.physically_secure=True`; thermal
  feasibility alone is never a solved episode.
- Generation `PF_ALG` must equal evaluation `PF_ALG`.
- The current canonical `PF_ALG` is `3`.
- The MCTS visit distribution is the self-play policy target.
- Every branch has one stable policy slot. Its executable direction is state-dependent: active means open, inactive and explicitly allowed means close.
- Self-play examples, checkpoints, and evaluation require exact topology-action configuration and ordered action-layout provenance.
- `outcome_value_target` is required for graph training examples.
- Feature normalization is part of the checkpoint contract.
- Fine-tuning preserves normalization statistics from the parent checkpoint.
- The train/validation split is performed by `scenario_id`, not by individual rows.
- Evaluation scenario selection is explicit and deterministic for a fixed input set and seed.

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

## State feature schema

Graph states use the versioned state feature schema `v3`. The schema is shared
by initial states and states reconstructed from PYPOWER results.

Bus features contain 30 ordered columns:

```text
Pd
Qd
Pg
Qg
Vm
Va
PQ
PV
REF
vn_kv
GS
BS
min_vm_pu
max_vm_pu
gen_online_count
gen_available
gen_p_min_mw
gen_p_max_mw
gen_q_min_mvar
gen_q_max_mvar
gen_p_down_margin_mw
gen_p_up_margin_mw
gen_q_down_margin_mvar
gen_q_up_margin_mvar
gen_min_p_down_margin_mw
gen_min_p_up_margin_mw
gen_min_q_down_margin_mvar
gen_min_q_up_margin_mvar
gen_p_limit_violation_count
gen_q_limit_violation_count
```

Generator outputs and limits are aggregated by bus using active generators
only. Offline generators do not contribute to generation, availability,
limits, headroom, worst-unit margins, or violation counts.

The aggregate directional margins describe the combined generation range at a
bus. The `gen_min_*` columns preserve the worst directional margin of any
active generator on that bus, preventing one unit's spare capacity from hiding
another unit's limit violation. The P/Q violation counts record how many active
generators are outside their raw configured limits. Negative margins are
preserved because they represent an actual limit violation rather than missing
data.

Branch features contain 16 ordered columns:

```text
pf
qf
pt
qt
r
x
b
tap
shift
rate_a
br_status
s_from_mva
s_to_mva
s_max_mva
loading_percent
unlimited_rating
```

`RATE_A=0` is represented explicitly by `unlimited_rating=1`. Its
`loading_percent` is stored as `0`; consumers must use `unlimited_rating`
instead of interpreting a sentinel loading value.

`bus_ids` stores the original GridFM bus identifiers. `edge_index` never stores
raw bus identifiers: it stores contiguous zero-based bus positions in the range
`0 <= index < number_of_buses`.

State construction fails closed for non-finite values, duplicate identifiers,
unknown branch endpoints, generators attached to unknown buses, invalid binary
statuses, and inverted voltage or generator limits.

The exact schema version, ordered feature columns, schema fingerprint,
`edge_index` semantics, and bus-ID semantics are stored in state NPZ files,
self-play rows, replay metadata, and graph checkpoints.

## Artifact compatibility

The current exact contract versions are:

| Contract | Version |
| --- | ---: |
| physical objective | `3` |
| outcome/value target | `5` |
| state feature schema | `3` |
| evaluation metrics | `6` |
| checkpoint | `7` |
| replay buffer schema | `6` |
| physics configuration | `1` |
| topology action | `1` |

State NPZ files, examples, replay rows, and checkpoints carry exact
state-feature schema provenance, including the schema version, fingerprint,
ordered bus and branch columns, contiguous `edge_index` semantics, and original
bus-ID semantics.

Examples and replay rows also carry physical, value-target,
physics-configuration, and topology-action provenance. Checkpoints additionally
carry the ordered `stop_plus_branch_status_v1` action layout and its
fingerprint. Evaluation loads the topology-action configuration and layout from
the checkpoint and requires an exact match before creating its runtime action
space. Consumers fail closed on missing, old, reordered, or mismatched
provenance.

Artifacts produced under former solved semantics, before topology-action
provenance, or before state feature schema `v3` are scientifically
incompatible. They are never relabeled or upgraded in place. Regenerate states,
self-play examples, replay, and checkpoints in this order:

```bash
# 1. Fresh episodes and outcome targets.
# Repeat --closeable-branch-id for each verified normally-open/tie branch.
python -m grid_topology_ai.cli self-play <POOL_RAW_DIR> \
  --transitions <POOL_TRANSITIONS.csv> \
  --output <NEW_SELF_PLAY_DIR> \
  --pf-alg 3 \
  --require-connected-after-switch \
  --min-loading-for-switch-percent 0.0

# 2. Fresh checkpoint; do not initialize from a legacy checkpoint.
python -m grid_topology_ai.cli train <NEW_EXAMPLES_CSV> --output <NEW_CHECKPOINT.pt> --device cpu

# 3. Fresh fixed-evaluation rows and metrics.
python -m grid_topology_ai.cli evaluate <EVAL_RAW_DIR> --transitions <EVAL_TRANSITIONS.csv> --checkpoint <NEW_CHECKPOINT.pt> --pf-alg 3 --output-csv <NEW_EVAL_RESULTS.csv> --output-json <NEW_EVAL_METRICS.json>
```

Archive incompatible old artifacts and do not mix them with regenerated examples, checkpoints, or evaluation outputs.

## Action space

The topology-control policy layout is `stop_plus_branch_status_v1`:

- `0` -> stop/handoff;
- `1 + branch_pos` -> the stable branch-status slot for `branch_ids[branch_pos]`.

The command represented by a branch slot depends on the current state:

- active branch -> `switch_off_branch`, `target_status=0`;
- inactive branch listed in `closeable_branch_ids` -> `switch_on_branch`, `target_status=1`.

`require_connected_after_switch` and `min_loading_for_switch_percent` constrain
branch opening. The loading threshold never filters a permitted closure.
Inactive branches not present in the explicit allowlist remain masked. Handoff
means the topology-control episode ends and the case is passed to an external
or future redispatch layer. Production redispatch optimization is not
implemented here.

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
python -m grid_topology_ai.cli --help
```

## Running self-play

Generate examples directly with the canonical CLI. The caller selects the input
checkpoint explicitly; omit `--checkpoint` to use the heuristic evaluator.

```bash
python -m grid_topology_ai.cli self-play RAW_DIR \
  --transitions TRANSITIONS.csv \
  --output data/self_play/mcts_v0 \
  --checkpoint CHECKPOINT.pt
```

Generation writes self-play examples only. Training and evaluation are separate,
explicit operations; generation does not promote checkpoints or run a final test.

## Package structure

- `grid_topology_ai/config`: typed configuration and validation.
- `grid_topology_ai/self_play`: direct example generation, example contracts, and small artifact helpers.
- `grid_topology_ai/training`: graph policy-value training, checkpoints, metrics, and splits.
- `grid_topology_ai/evaluation.py`: checkpoint evaluation and metrics.
- `grid_topology_ai/teacher_runtime.py`: deterministic teacher generation runtime.
- `grid_topology_ai/search`: MCTS planning components.
- `grid_topology_ai/models`: graph datasets and neural models.
- `grid_topology_ai/cli.py`: the unified teacher, training, self-play, and evaluation CLI.
- `tests`: unit, contract, and smoke tests.

Unified Light commands:

```bash
python -m grid_topology_ai.cli teacher --help
python -m grid_topology_ai.cli self-play --help
python -m grid_topology_ai.cli train --help
python -m grid_topology_ai.cli evaluate --help
```

The installed console script exposes the same command surface:

```bash
power-grid-reconfig --help
```

## Self-play inputs

Prepare a transitions CSV, its raw state directory, and—when neural-guided MCTS
is desired—the exact checkpoint to pass via `--checkpoint`.

## Testing and CI

GitHub Actions cover:

- Ubuntu tests;
- Windows tests;
- package build;
- data tools smoke.

Local graph dataset integration tests are opt-in because they need prepared local data artifacts.

## Current limitations

- Topology actions are branch-status changes plus stop/handoff; inactive branches can be closed only when explicitly allowlisted.
- The main research setup is one case118 configuration.
- Scenarios come from a fixed pool rather than unrestricted environment generation.
- There is no production redispatch optimizer.
- Real self-play is expensive and requires prepared data and checkpoint artifacts.
- This is research code, not operational grid control software.

## Teacher bootstrap

Teacher generation remains available for bootstrap datasets, baseline comparison, and debugging through the unified `teacher` subcommand. Its runtime owner is packaged in `grid_topology_ai.teacher_runtime` so the installed CLI does not depend on repository-only `scripts` modules.

## Reproducibility

Reproducibility relies on Python 3.11, pinned constraints, explicit seeds,
artifact contracts, fixed input data, checkpoint provenance, and CI checks.
Checkpoints store normalization metadata, dataset metadata, physics configuration,
topology-action configuration, the ordered action layout, and training configuration.
