# PowerGridReconfig

## Project overview

PowerGridReconfig is a Python 3.11 research framework for emergency topology control in power-system simulations. The current system implements a pool-guided AlphaZero-like self-play loop for the case118 research setup: a fixed physical scenario pool is sampled, a model-guided neural MCTS planner generates replay, a graph policy-value checkpoint is fine-tuned, a fixed evaluation set is evaluated, and the candidate checkpoint is accepted or rejected.

This is not a full classical AlphaZero system. The scenario pool is fixed, generation is model-guided, the continuation gate is diagnostic only, replay accumulates across iterations, candidates are fine-tuned from checkpoints, evaluation is fixed, and acceptance/rejection is based on configured metrics.

For implementation details, see [docs/self_play.md](docs/self_play.md). The stable bidirectional branch-status policy and artifact compatibility rules are specified in [docs/topology_action_contract.md](docs/research/topology_action_contract.md).

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

## Evaluation and checkpoint selection contract

The learned controller is evaluated in the `ungated` policy mode: neural policy
plus MCTS, without continuation-gate filtering. When continuation analysis is
enabled, `constrained` is evaluated as a secondary hybrid-controller diagnostic.
It does not replace the learned controller's result.

Checkpoint-arena ranking, regular promotion, paired confidence checks, aggregate
acceptance gates, learning-curve headline values, and final-test headline values
all use `ungated`. A checkpoint cannot be selected or promoted only because the
continuation gate repairs a weaker learned policy.

Evaluation artifacts expose the contract explicitly:

- `primary_policy_mode` is `ungated`;
- top-level metrics are copies of the primary ungated metrics;
- complete groups are stored under `mode_metrics.ungated` and, when enabled,
  `mode_metrics.constrained`;
- `continuation_gate_gain` is constrained minus ungated physical-security rate
  and measures the external gate's contribution, not learned-policy quality;
- `checkpoint_selection.json` records `policy_mode: ungated`;
- the sealed final-test report records the primary mode and both mode-specific
  physical-security rates.

Selection paths fail closed. Missing ungated metrics, a non-ungated primary mode,
an inconsistent top-level headline, or a missing required policy-mode group is a
contract error rather than a fallback to constrained results.

## Scientific invariants

- `solved=True` means exactly `assessment.physically_secure=True`; thermal
  feasibility alone is never a solved episode.
- Generation `PF_ALG` must equal evaluation `PF_ALG`.
- The current canonical `PF_ALG` is `3`.
- The MCTS visit distribution is the policy target.
- The continuation gate records diagnostics but never rewrites the executed action or the MCTS visit policy target.
- Every branch has one stable policy slot. Its executable direction is state-dependent: active means open, inactive and explicitly allowed means close.
- Self-play, replay, checkpoints, and evaluation require exact topology-action configuration and ordered action-layout provenance.
- `outcome_value_target` is required for graph training examples.
- Feature normalization is part of the checkpoint contract.
- Fine-tuning preserves normalization statistics from the parent checkpoint.
- The train/validation split is performed by `scenario_id`, not by individual rows.
- Validation objectives retain candidate epoch checkpoints; the optional closed-loop tuning arena selects the canonical candidate using ungated metrics.
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
python -m scripts.self_play.generate <POOL_RAW_DIR> \
  --transitions <POOL_TRANSITIONS.csv> \
  --output-dir <NEW_SELF_PLAY_DIR> \
  --pf-alg 3 \
  --require-connected-after-switch \
  --min-loading-for-switch-percent 0.0

# 2. Fresh checkpoint; do not initialize from a legacy checkpoint.
python -m scripts.self_play.train_graph_baseline <NEW_EXAMPLES_CSV> --output <NEW_CHECKPOINT.pt> --device cpu

# 3. Fresh fixed-evaluation rows and metrics.
python -m scripts.evaluation.evaluate_checkpoint <EVAL_RAW_DIR> --transitions <EVAL_TRANSITIONS.csv> --checkpoint <NEW_CHECKPOINT.pt> --pf-alg 3 --output-csv <NEW_EVAL_RESULTS.csv> --output-json <NEW_EVAL_METRICS.json>
```

## Repository layout

```text
grid_topology_ai/
  config/
  evaluation/
  models/
  search/
  self_play/
  training/

scripts/
  data/
  diagnostics/
  evaluation/
  planning/
  self_play/

configs/
  self_play_loop_pilot.yaml
  self_play_loop_smoke.yaml
  self_play_loop.yaml

tests/
```

## Validation

Run the complete suite with:

```bash
python -m pytest -q
```

Run the self-play configuration preflight with:

```bash
python -m scripts.self_play.loop configs/self_play_loop.yaml --validate-only
```
