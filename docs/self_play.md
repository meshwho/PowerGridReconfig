# Self-play pipeline

## 1. Execution plan

The loop reads a YAML configuration, resolves run directories, verifies artifact paths, and executes iterations in order. `--plan-only` prints the intended work; `--validate-only` validates references; normal execution runs generation, replay update, training, evaluation, and acceptance.

## 2. Bootstrap initialization

A run starts from a bootstrap checkpoint and bootstrap fixed-evaluation metrics. The metrics must include `pf_alg` provenance compatible with generation and evaluation settings.

## 3. Pool metadata

The pool describes fixed physical scenarios, transition rows, raw state references, scenario identifiers, and hashes used to audit reproducibility.

## 4. Prioritized sampling

Each iteration samples scenario ids from the fixed pool. Sampling may prioritize weak or unsolved scenarios while keeping the pool itself fixed.

## 5. Generation request

Generation uses the current accepted checkpoint, configured MCTS settings, raw states, and `PF_ALG`. The canonical pilot value is `PF_ALG=3`.

## 6. MCTS target versus executed action

The policy target is the MCTS visit distribution. The continuation gate may alter the executed action for safety or episode-control semantics, but it does not rewrite the MCTS visit target.

## 7. Replay buffer

Generated examples are appended to replay. Replay accumulation allows later iterations to train on current and prior experience according to configured limits.

## 8. Train/validation split

Training batches are split by `scenario_id`. A scenario cannot appear in both train and validation files for the same candidate.

## 9. Normalization contract

Normalization arrays are part of the checkpoint contract. Fine-tuning from an initial checkpoint requires normalized features and reuses the parent checkpoint normalization statistics.

## 10. Checkpoint selection

The main candidate checkpoint records `checkpoint_selection_metric=validation_loss` when validation data exists and `training_loss` otherwise. Additional variants record exact selector metadata: best loss uses `validation_loss`, best top-1 uses `validation_top1`, best top-5 uses `validation_top5`, best switch uses `validation_switch_accuracy`, best policy uses `policy_selection_score`, and last uses `last_epoch`.

## 11. Fixed evaluation

Candidate checkpoints are evaluated on the fixed evaluation transitions and raw states. This keeps acceptance comparable across iterations.

## 12. PF_ALG provenance

Generation config, evaluation config, evaluation requests, and fixed metrics must use exact integer `PF_ALG` values. Fractional or boolean values are rejected instead of rounded.

## 13. Acceptance

Acceptance compares candidate metrics with the best accepted metrics. The primary configured metric is usually `solve_rate`; thresholds and safety constraints decide whether the candidate replaces the best checkpoint.

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

## Physical objective schema v2

Self-play acceptance uses `physically_secure_rate_requested` as the primary success metric. `solve_rate` remains an alias for physically secure success in new evaluation output, while `thermal_solved_rate` is diagnostic only.

Schema version 1 evaluation metrics and training examples are incompatible with schema version 2 and must be regenerated rather than converted. New examples include `value_target_schema_version=2` and `physical_objective_schema_version=2`. Bootstrap checkpoints trained on schema version 1 targets should be retrained.
