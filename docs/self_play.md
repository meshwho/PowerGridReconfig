# Self-play pipeline

## Scope

The current Light runtime exposes self-play, training, and evaluation as direct,
independent operations. There is no integrated arena, automatic checkpoint
promotion, acceptance controller, replay orchestration loop, curriculum manager,
or sealed final-test runner.

The implemented learned-control flow is:

```text
GridFM raw scenario
  -> canonical physical state
  -> AC power flow
  -> topology action space
  -> MCTS self-play examples
  -> GraphPolicyValueNetV2 training
  -> explicit checkpoint evaluation
```

Teacher generation is an optional deterministic bootstrap path and is not part
of self-play control logic.

## Physical success contract

`solved=True` means exactly `assessment.physically_secure=True`:

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

Thermal feasibility alone is diagnostic and never defines a solved episode.
Initial GridFM states receive a no-op AC power flow because parquet input does
not contain trustworthy convergence provenance.

The current canonical power-flow algorithm for generated research artifacts is
`PF_ALG=3`. Generation and evaluation must use the same physical configuration.

## Topology action contract

The graph policy layout is `stop_plus_branch_status_v1`:

- policy index `0` is stop/handoff;
- policy index `1 + branch_pos` is the stable action slot for
  `branch_ids[branch_pos]`;
- an active branch slot opens that branch;
- an inactive branch slot closes it only when the branch ID is explicitly
  present in `closeable_branch_ids`.

`require_connected_after_switch` and `min_loading_for_switch_percent` constrain
opening actions. The loading threshold does not filter an explicitly allowed
closure.

The complete topology-action configuration and ordered action layout are stored
as artifact provenance. Evaluation reconstructs its runtime action space from
the checkpoint and rejects mismatched or incomplete provenance.

See [topology_action_contract.md](research/topology_action_contract.md) for the
serialized layout and fingerprints.

## MCTS and value contract

MCTS uses undiscounted terminal utility. `gamma` is exactly `1.0` throughout the
current generation and evaluation API.

Dense environment rewards remain diagnostics. They do not enter the MCTS value
backup or the policy-value target.

Every state in a completed episode receives the terminal outcome value target
associated with that episode. The current target is bounded to `[-1, 1]`, so no
checkpoint `value_scale` field is used.

The self-play policy target is the MCTS root visit distribution. The executed
action is selected from that distribution according to the configured
temperature schedule. Root Dirichlet noise and action-temperature sampling are
independent sources of exploration.

## Progressive widening

`top_k` is the initial switch-action width, not a permanent pruning limit.
Additional retained legal actions are activated as node visits grow:

```text
top_k + floor(
    widening_coefficient
    * visit_count ** widening_exponent
)
```

The width is capped by the number of retained legal switch actions. Stop does
not consume a switch-action slot.

## Random streams

Self-play keeps MCTS exploration and real-action sampling on independent random
streams. Scenario-specific seeds are derived from the configured stream seed and
scenario ID, so reordering scenarios does not reassign their random streams.

## Generation inputs

A direct self-play invocation requires:

- a GridFM raw directory containing `bus_data.parquet`, `branch_data.parquet`,
  and `gen_data.parquet`;
- a transitions CSV containing the selected scenario IDs;
- an output directory;
- optionally, a current Graph V2 checkpoint for neural-guided MCTS.

Example:

```bash
python -m grid_topology_ai.cli self-play RAW_DIR \
  --transitions TRANSITIONS.csv \
  --output data/self_play/run_001 \
  --checkpoint CHECKPOINT.pt \
  --pf-alg 3 \
  --gamma 1.0
```

Omit `--checkpoint` to use the non-neural evaluator path.

## Self-play artifacts

Generation writes `examples.csv`, state NPZ files under `states/`, and
`progress.json`.

Persisted examples identify states by `state_id`; persisted `state_path` is not a
supported schema field. Runtime file paths are derived from the artifact
location and `state_id`.

Source identity used for resume is content-based. Raw GridFM files, transitions,
and an optional checkpoint are identified by content rather than absolute file
location. Moving an unchanged input set does not change its semantic identity.

Existing artifacts are not migrated to the current schema. Missing, old, or
mismatched required fields fail closed and must be regenerated.

## Resume contract

`--resume` requires an existing `progress.json` whose semantic identity exactly
matches the current request. Existing `examples.csv` and referenced NPZ state
files are validated against the current contract before generation continues.

A non-resume run refuses to reuse an output directory containing existing
progress or examples.

## Training

Training consumes current validated example CSVs and state NPZ files. Graph
batches are packed without node or edge padding and use the physical
`edge_active_mask` derived from branch status.

A current Graph V2 checkpoint contains, among other required fields:

- `model_type=graph_policy_value_net_v2`;
- `topology_cardinality_independent=True`;
- the model state dictionary and exact architecture dimensions;
- feature normalization arrays;
- physics configuration;
- topology-action configuration and ordered action layout.

Checkpoint loading is strict. Missing required Graph V2 fields are errors; they
are not supplied from defaults or inferred from an older checkpoint format.

## Evaluation

Evaluation runs exactly one policy mode per invocation:

- `ungated`: the learned-controller behavior, neural policy plus MCTS;
- `constrained`: applies the current root-policy constraint analysis before
  action selection.

The mode is selected explicitly:

```bash
python -m grid_topology_ai.cli evaluate RAW_DIR \
  --transitions EVAL.csv \
  --checkpoint CHECKPOINT.pt \
  --policy-mode constrained \
  --min-hard-improvement 50 \
  --min-soft-improvement 15 \
  --min-constraint-visits 5 \
  --min-constraint-visit-fraction 0.01 \
  --pf-alg 3 \
  --gamma 1.0
```

`--policy-mode ungated` is the default. The removed
`--use-continuation-gate` flag is not part of the current CLI.

Evaluation records canonical physical outcome fields and aggregate metrics for
that single run. It does not automatically run paired modes or promote a
checkpoint.

## Reproducibility

Reproducibility depends on explicit seeds, content-based source identities,
strict artifact contracts, exact physical configuration, checkpoint provenance,
and deterministic scenario selection for a fixed request.

Useful validation commands are:

```bash
python -m compileall -q grid_topology_ai scripts tests
python -m pytest -q
python -m grid_topology_ai.cli self-play --help
python -m grid_topology_ai.cli evaluate --help
```
