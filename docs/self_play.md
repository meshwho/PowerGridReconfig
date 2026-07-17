# Self-play pipeline

## 1. Execution plan

The loop reads a YAML configuration, resolves run directories, verifies artifact paths, and executes iterations in order. `--plan-only` prints the intended work; `--validate-only` validates references; normal execution runs generation, replay update, training, evaluation, and acceptance.

## 2. Bootstrap initialization

A run starts from a bootstrap checkpoint and bootstrap fixed-evaluation metrics. The metrics must include `pf_alg` provenance compatible with generation and evaluation settings. Both artifacts must also carry the current semantic contract versions; validation happens before a bootstrap checkpoint is copied to the canonical best path.

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

## 6. MCTS target versus executed action

The policy target is the MCTS visit distribution. The continuation gate may alter the executed action for safety or episode-control semantics, but it does not rewrite the MCTS visit target.

## 7. Replay buffer

Generated examples are appended to replay. Replay accumulation allows later iterations to train on current and prior experience according to configured limits. Replay manifests and every replay row are checked against the current physical and outcome/value-target versions before loading or mutation.

## 8. Train/validation split

Training batches are split by `scenario_id`. A scenario cannot appear in both train and validation files for the same candidate.

## 9. Normalization contract

Normalization arrays are part of the checkpoint contract. Fine-tuning from an initial checkpoint requires normalized features and reuses the parent checkpoint normalization statistics.

## 10. Checkpoint selection

The main candidate checkpoint records `checkpoint_selection_metric=validation_loss` when validation data exists and `training_loss` otherwise. Additional variants record exact selector metadata: best loss uses `validation_loss`, best top-1 uses `validation_top1`, best top-5 uses `validation_top5`, best switch uses `validation_switch_accuracy`, best policy uses `policy_selection_score`, and last uses `last_epoch`.

## 11. Fixed evaluation

Candidate checkpoints are evaluated on the fixed evaluation transitions and raw states. This keeps acceptance comparable across iterations. `solve_count` and `solve_rate` count only physically secure outcomes and therefore equal `physically_secure_count` and `physically_secure_rate`. Thermal feasibility remains a separate diagnostic rate. Evaluation also records counts/rates for PF convergence, finite values, topology connectivity, thermal, voltage, generator P/Q, and angle feasibility, plus violation diagnostics.

## 12. PF_ALG provenance

Generation config, evaluation config, evaluation requests, and fixed metrics must use exact integer `PF_ALG` values. Fractional or boolean values are rejected instead of rounded.

## 13. Acceptance

Acceptance compares candidate metrics with the best accepted metrics. The primary configured metric is usually `solve_rate`; thresholds and safety constraints decide whether the candidate replaces the best checkpoint. Candidate and best metrics must match both the configured `PF_ALG` and the current evaluation/physical semantic versions, so thermal-only historical metrics cannot influence checkpoint promotion.

## 14. Atomic completion marker

An iteration is complete only when `iteration_complete.json` exists and is valid. This marker is written after all required artifacts for the iteration are complete.

## 15. Resume behavior

`--resume` continues after the latest valid completed iteration. If a later iteration directory exists without a valid `iteration_complete.json`, the loop refuses to continue until the operator removes or repairs the incomplete directory.

## 16. Artifact hashes

Dataset and state references are hashed in metadata so a run can be audited against the inputs used to create checkpoints and metrics.

## 17. Learning curve columns

`learning_curve.csv` tracks iteration-level progress such as iteration index, candidate checkpoint, evaluation metrics, acceptance decision, and best metric state.

## 18. Failure recovery

For config or artifact validation failures, fix the referenced paths or metadata and rerun validation. For incomplete iteration directories, inspect the partial artifacts and either delete the incomplete iteration or restart from a clean run directory.

## 19. Pilot workflow

A pilot workflow is: prepare bootstrap artifacts, run `--validate-only`, run `--plan-only`, execute one iteration, inspect training and evaluation artifacts, then resume for additional iterations.

## 20. Bootstrap metrics recalculation rules

Recompute bootstrap metrics whenever the fixed evaluation set, raw states, checkpoint, `PF_ALG`, evaluation settings, or metrics schema changes. Do not reuse metrics with missing, fractional, boolean, or mismatched `pf_alg` values.

## 21. Semantic artifact versions and regeneration

The current incompatible contract versions are:

- `PHYSICAL_OBJECTIVE_SCHEMA_VERSION=2`;
- `OUTCOME_VALUE_TARGET_CONTRACT_VERSION=2`;
- `EVALUATION_METRICS_CONTRACT_VERSION=2`;
- `CHECKPOINT_CONTRACT_VERSION=2`;
- replay buffer schema `2`.

The version bump is intentional: former `solved` labels meant only
thermal-feasible, so their value targets, trained weights, solve rates, and
acceptance comparisons have different scientific meaning. Missing or old
versions are rejected. `ensure_outcome_value_targets` refuses to stamp current
targets onto legacy solved labels. User artifacts are not deleted automatically.

Create a clean artifact chain in this order (replace angle-bracket paths):

```bash
# Fresh physical episodes and versioned outcome targets.
python -m scripts.self_play.generate <POOL_RAW_DIR> --transitions <POOL_TRANSITIONS.csv> --output-dir <NEW_SELF_PLAY_DIR> --pf-alg 3

# Fresh checkpoint, without a legacy --init-checkpoint.
python -m scripts.self_play.train_graph_baseline <NEW_SELF_PLAY_DIR>/examples.csv --output <NEW_CHECKPOINT.pt> --device cpu

# Fresh fixed evaluation and summary metrics.
python -m scripts.evaluation.evaluate_checkpoint <EVAL_RAW_DIR> --transitions <EVAL_TRANSITIONS.csv> --checkpoint <NEW_CHECKPOINT.pt> --pf-alg 3 --output-csv <NEW_EVAL_RESULTS.csv> --output-json <NEW_EVAL_METRICS.json>
```

Archive the old replay/run directory, update `bootstrap_checkpoint` and
`bootstrap_eval_metrics` in the YAML, and start a new run. Existing evaluation
metrics cannot be compared across the version boundary, and a checkpoint trained
on legacy targets cannot be used as a compatible parent.
