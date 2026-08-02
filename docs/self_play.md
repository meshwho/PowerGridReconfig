# Self-play pipeline

## 1. Execution plan

The loop reads a YAML configuration, resolves run directories, verifies artifact paths, and executes iterations in order. `--plan-only` prints the intended work; `--validate-only` validates references; normal execution runs generation, replay update, training, evaluation, and acceptance.

## 2. Bootstrap initialization

A run starts from a bootstrap checkpoint and bootstrap fixed-evaluation metrics. The metrics must include `pf_alg` provenance compatible with generation and evaluation settings. Both artifacts must also carry the current semantic contract versions. The bootstrap checkpoint must match the configured topology-action fingerprint and ordered action layout before it is copied to the canonical best path.

## 2.1 Physical success and episode termination

`solved`, `TerminationReason.SOLVED`, positive solved bonuses, positive terminal
outcome targets, and `solve_rate` all use one authoritative predicate:

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

The calculator uses the raw PYPOWER result before feature sanitization and the
static GridFM/PYPOWER limits: `VM/VMIN/VMAX/VA`, branch status, `RATE_A`, angle
limits, endpoints and terminal flows, plus each active generator's `PG/PMIN/PMAX`
and `QG/QMIN/QMAX`. Bus IDs are mapped to array positions for angle checks.
Disabled elements do not create violations. `RATE_A=0` and angle bounds at
`-360/360` follow MATPOWER's unconstrained semantics. Invalid mandatory data,
unknown active endpoints, NaN, and infinity fail closed.

The related terms are deliberately different:

- `thermal_solved` / `thermal_feasible`: diagnostic only; no active rated branch is thermally overloaded.
- `physically_secure`: all eight physical components above are simultaneously true; this is the exact definition of solved.
- `done`: the control episode ended. Solved, PF failure, max steps, and explicit stop/handoff can all be done.
- `handoff`: topology control stops and transfers the case to redispatch; it is terminal but never solved unless the state was already physically secure, in which case the reason is `SOLVED` rather than handoff.

`stop_policy=solved_only` exposes stop only for a physically secure state.
Thermal-safe but voltage-, generator-, angle-, or connectivity-infeasible states
continue when steps remain and no explicit stop was chosen. Initial GridFM
states receive a no-op AC power flow because parquet input alone has no reliable
convergence provenance.

## 3. Pool metadata

The pool describes fixed physical scenarios, transition rows, raw state references, scenario identifiers, and hashes used to audit reproducibility.

## 4. Prioritized sampling

Each iteration samples scenario ids from the fixed pool. Sampling may prioritize weak or unsolved scenarios while keeping the pool itself fixed.

## 5. Generation request

Generation uses the current accepted checkpoint, configured MCTS settings, raw states, and `PF_ALG`. The canonical pilot value is `PF_ALG=3`.

### 5.1 Bidirectional topology-action configuration

The graph policy layout is `stop_plus_branch_status_v1`. Policy index `0` is
stop/handoff; policy index `1 + branch_pos` is the stable branch-status slot for
`branch_ids[branch_pos]`. The slot identity does not change with the current
status:

- active branch -> `switch_off_branch`, `target_status=0`;
- inactive branch -> `switch_on_branch`, `target_status=1`.

A closure is legal only when the inactive branch ID is explicitly present in
`closeable_branch_ids`. The generation block exposes the complete semantic
action-space configuration:

```yaml
generation:
  require_connected_after_switch: true
  min_loading_for_switch_percent: 0.0
  # Populate only with verified normally-open/tie branch IDs.
  closeable_branch_ids: []
```

`require_connected_after_switch` and `min_loading_for_switch_percent` constrain
opening actions only. The loading threshold never filters an allowed closure.
An empty `closeable_branch_ids` list preserves opening-only behavior. The list is
canonicalized and becomes part of the topology-action fingerprint written to
examples, replay, and checkpoints.

Evaluation does not define an independent topology-action override. It loads the
exact action configuration and ordered layout from the checkpoint, reconstructs
the runtime action space from that provenance, and rejects a mismatch before an
episode is evaluated.

See [topology_action_contract.md](topology_action_contract.md) for the serialized
layout, fingerprints, and artifact compatibility rules.

### 5.2 Progressive widening

`top_k` is the initial number of switch actions exposed to PUCT, not a permanent pruning limit. Each node retains the complete neural/DC/loading ranking and activates more legal switch actions as its visit count grows.

The active switch width is:

```text
top_k + floor(
    widening_coefficient
    * visit_count ** widening_exponent
)
```

The result is capped by the number of legal switch actions. Existing wider shortlists are never reduced, stop actions do not consume switch slots, and `widening_coefficient: 0` disables growth.

### 5.3 MCTS action coverage

Every root search reports two action-space coverage measures:

- `action_coverage` is the fraction of legal root actions activated
  for PUCT selection;
- `visited_action_coverage` is the fraction of legal root actions
  that received at least one simulation.

The denominator is the complete retained legal root action ranking.
The considered count includes actions activated by the initial
shortlist, progressive widening, and the off-prior exploration quota.

Self-play stores the counts and coverage values with every generated
example. Evaluation stores per-search counts in the episode CSV and
aggregates mean and minimum coverage values in the metrics JSON.

### 5.4 Action temperature schedule

Root Dirichlet noise changes the MCTS search priors. Action
temperature controls how the real self-play action is selected from
the resulting root visit distribution. These are separate sources of
exploration.

A positive `selection_temperature` is used only when both conditions
hold:

- the one-based self-play iteration is not greater than
  `temperature_iterations`;
- the zero-based episode step is less than `temperature_steps`.

After either cutoff, action selection uses temperature `0.0` and
therefore deterministic argmax. Setting either cutoff to zero disables
temperature-based sampling and preserves the previous behavior.

### 5.5 Independent random streams

Each self-play iteration expands the configured base seed and
one-based iteration number through `numpy.random.SeedSequence`.
Separate child streams are used for:

- scenario sampling;
- MCTS exploration and root noise;
- action sampling from the behavior policy.

Generation derives scenario-specific MCTS and action-sampling seeds
from the corresponding stream seed and scenario ID. Reordering
scenarios therefore does not change the random stream assigned to a
particular scenario, and action sampling does not consume random values
from MCTS.

### 5.6 Exploration diagnostics

Every production self-play example records the action-selection
temperature and mode, policy-target entropy, normalized policy-target
entropy, and MCTS root action coverage.

Policy-target entropy is measured in nats. Normalized entropy divides
the entropy by `log(k)`, where `k` is the number of actions with
positive probability. It is defined as zero when `k <= 1`.

At the end of each iteration, diagnostics are aggregated over the
newly generated self-play steps. They are stored under
`extra.self_play_exploration` in the iteration metadata and as
`self_play_*` columns in `learning_curve.csv`.

The aggregation uses current-iteration raw examples only. Replay
examples from earlier iterations are not included.

### 5.7 Terminal utility contract

The policy-value model and MCTS use undiscounted terminal utility.

Every state in one completed episode receives the same value target:

- physically solved: `+1`;
- executed and physically validated redispatch: `0`;
- every other terminal outcome: `-1`.

`outcome_gamma` is fixed at `1.0`. `outcome_steps_to_terminal` records
the remaining number of transitions for diagnostics but does not scale the
target. MCTS backs up the leaf utility unchanged across every traversed edge.
Dense environment rewards and their accumulated return remain diagnostic and
never enter the policy-value target or MCTS Q backup.

## 6. MCTS target versus executed action

The policy target is the MCTS visit distribution. The real self-play action is selected from that distribution according to the configured temperature schedule. Continuation analysis is diagnostic only: it records allowed/recommended actions and whether the selected action agrees, but it does not override the executed action or rewrite the policy target.

## 7. Replay buffer

Generated examples are appended to replay. Replay accumulation allows later iterations to train on current and prior experience according to configured limits. Replay manifests and every replay row are checked against the current physical, outcome/value-target, physics-configuration, topology-action configuration, and ordered action-layout contracts before loading or mutation. Mixed topology configurations or layouts are rejected before the buffer is changed.

## 8. Train/validation split

Training batches are split by `scenario_id`. A scenario cannot appear in both train and validation files for the same candidate.

## 9. Normalization contract

Normalization arrays are part of the checkpoint contract. Fine-tuning from an initial checkpoint requires normalized features and reuses the parent checkpoint normalization statistics.

## 10. Checkpoint selection

The main candidate checkpoint records `checkpoint_selection_metric=validation_loss` when validation data exists and `training_loss` otherwise. Additional variants record exact selector metadata: best loss uses `validation_loss`, best top-1 uses `validation_top1`, best top-5 uses `validation_top5`, best switch uses `validation_switch_accuracy`, best policy uses `policy_selection_score`, and last uses `last_epoch`.

## 11. Fixed evaluation

Candidate checkpoints are evaluated on the fixed evaluation transitions and raw states. This keeps acceptance comparable across iterations. `solve_count` and `solve_rate` count only physically secure outcomes and therefore equal `physically_secure_count` and `physically_secure_rate`. Thermal feasibility remains a separate diagnostic rate. Evaluation also records counts/rates for PF convergence, finite values, topology connectivity, thermal, voltage, generator P/Q, and angle feasibility, plus violation diagnostics.

Before workers are initialized, evaluation loads the checkpoint's exact
topology-action configuration and ordered layout. Each worker constructs
`GridFMActionSpace` from that configuration and validates the evaluator
checkpoint against the same fingerprints. A checkpoint without topology
action provenance, or with a different allowlist/layout, is rejected rather
than evaluated under different semantics.

The regular evaluation set is a selection set: it is used after each iteration
to decide whether a candidate checkpoint is promoted. The final test set is
independent and is never used for training, self-play generation, candidate
selection, or promotion. It is reserved for one evaluation of the final best
checkpoint after the loop has completed.

## 12. PF_ALG provenance

Generation config, evaluation config, evaluation requests, and fixed metrics must use exact integer `PF_ALG` values. Fractional or boolean values are rejected instead of rounded.

## 13. Acceptance

Acceptance compares candidate metrics with the best accepted metrics. The primary configured metric is usually `solve_rate`; thresholds and safety constraints decide whether the candidate replaces the best checkpoint. Candidate and best metrics must match the configured `PF_ALG`, current evaluation/physical semantic versions, topology-action configuration fingerprint, and ordered action-layout fingerprint. Historical metrics or checkpoints from different action semantics cannot influence promotion.

## 14. Atomic completion marker

An iteration is complete only when `iteration_complete.json` exists and is valid. This marker is written after all required artifacts for the iteration are complete.

## 15. Resume behavior

`--resume` continues after the latest valid completed iteration. If a later iteration directory exists without a valid `iteration_complete.json`, the loop refuses to continue until the operator removes or repairs the incomplete directory.

## 16. Artifact hashes

Dataset and state references are hashed in metadata so a run can be audited against the inputs used to create checkpoints and metrics.

## 17. Learning curve columns

`learning_curve.csv` tracks iteration-level progress such as iteration index, candidate checkpoint, evaluation metrics, acceptance decision, and best metric state.
It also records `self_play_*` exploration diagnostics so policy entropy
and action coverage can be compared across iterations.

## 18. Failure recovery

For config or artifact validation failures, fix the referenced paths or metadata and rerun validation. For incomplete iteration directories, inspect the partial artifacts and either delete the incomplete iteration or restart from a clean run directory.

## 19. Pilot workflow

A pilot workflow is: prepare bootstrap artifacts, set verified
`closeable_branch_ids` in the pilot YAML when tie-line closing is intended, run
`--validate-only`, run `--plan-only`, execute one iteration, inspect training and
evaluation artifacts, then resume for additional iterations.

## 20. Bootstrap metrics recalculation rules

Recompute bootstrap metrics whenever the fixed evaluation set, raw states, checkpoint, `PF_ALG`, evaluation settings, topology-action configuration, ordered action layout, or metrics schema changes. Do not reuse metrics with missing, fractional, boolean, or mismatched `pf_alg` values or with incompatible topology provenance.

## 21. Semantic artifact versions and regeneration

The current exact contract versions are:

| Contract | Version |
| --- | ---: |
| physical objective | `3` |
| outcome objective | `1` |
| outcome/value target | `5` |
| checkpoint | `7` |
| replay buffer schema | `6` |
| evaluation metrics | `6` |
| physics configuration | `1` |
| topology action | `1` |

The version boundaries are intentional. Former `solved` labels meant only
thermal-feasible, and artifacts created before topology provenance did not
record the exact branch identity/order or semantic action allowlist. Missing or
old versions are rejected. `ensure_outcome_value_targets` refuses to stamp
current targets onto legacy solved labels. User artifacts are not deleted
automatically.

Create a clean artifact chain in this order (replace angle-bracket paths):

```bash
# Fresh physical episodes, topology provenance, and versioned outcome targets.
# Repeat --closeable-branch-id for each verified normally-open/tie branch.
python -m scripts.self_play.generate <POOL_RAW_DIR> \
  --transitions <POOL_TRANSITIONS.csv> \
  --output-dir <NEW_SELF_PLAY_DIR> \
  --pf-alg 3 \
  --require-connected-after-switch \
  --min-loading-for-switch-percent 0.0

# Fresh checkpoint, without a legacy --init-checkpoint.
python -m scripts.self_play.train_graph_baseline <NEW_SELF_PLAY_DIR>/examples.csv --output <NEW_CHECKPOINT.pt> --device cpu

# Fresh fixed evaluation and summary metrics. Evaluation uses checkpoint topology provenance.
python -m scripts.evaluation.evaluate_checkpoint <EVAL_RAW_DIR> --transitions <EVAL_TRANSITIONS.csv> --checkpoint <NEW_CHECKPOINT.pt> --pf-alg 3 --output-csv <NEW_EVAL_RESULTS.csv> --output-json <NEW_EVAL_METRICS.json>
```

Archive the old replay/run directory, update `bootstrap_checkpoint` and
`bootstrap_eval_metrics` in the YAML, and start a new run. Existing evaluation
metrics cannot be compared across the version boundary, and a checkpoint trained
on legacy targets or a different topology-action layout cannot be used as a
compatible parent.
