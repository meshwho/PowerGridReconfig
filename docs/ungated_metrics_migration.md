# Ungated evaluation metrics migration

The self-play loop now uses the ungated neural-policy-plus-MCTS controller as the only primary policy for checkpoint-arena ranking, promotion, paired confidence checks, and final headline metrics. Continuation-gate results remain available as secondary constrained-controller diagnostics.

## Affected artifacts

Evaluation JSON files produced before this contract change are not valid bootstrap or canonical best metrics when either of the following is true:

- `primary_policy_mode` is `constrained`;
- `task_config.primary_policy_mode` is missing or is not `ungated`.

The loop rejects these artifacts before copying bootstrap metrics, loading existing best metrics, or promoting a candidate. Do not edit old JSON files to relabel them: the metric values were calculated under different controller semantics.

The checkpoint weights themselves do not need to be retrained solely for this migration. They must be evaluated again with the current code and the same fixed evaluation dataset intended for the new run.

## Recalculate bootstrap metrics

Evaluate the bootstrap checkpoint with the current evaluation implementation:

```bash
python -m grid_topology_ai.cli evaluate \
  <EVAL_RAW_DIR> \
  --transitions <EVAL_TRANSITIONS.csv> \
  --checkpoint <BOOTSTRAP_CHECKPOINT.pt> \
  --pf-alg 3 \
  --output-csv <NEW_BOOTSTRAP_EVAL_RESULTS.csv> \
  --output-json <NEW_BOOTSTRAP_EVAL_METRICS.json>
```

Use the same power-flow algorithm, physics settings, topology-action provenance, action layout, and fixed evaluation set configured for the self-play run.

Before using the new JSON, verify that it contains:

```text
primary_policy_mode = ungated
task_config.primary_policy_mode = ungated
```

When continuation comparison is enabled, the artifact should also retain both `mode_metrics.ungated` and `mode_metrics.constrained`. The top-level metrics must match the ungated group; `continuation_gate_gain` is diagnostic only.

Point `bootstrap_eval_metrics` in the self-play YAML at the newly generated JSON.

## Existing runs

Do not resume an old run whose previous checkpoint selection or promotion decisions used constrained headline metrics. Those decisions are not comparable with the current ungated contract.

For a scientifically clean migration:

1. archive the old run directory;
2. keep the desired bootstrap checkpoint if its checkpoint and physics contracts remain current;
3. recalculate its fixed-evaluation metrics with the current code;
4. choose a new `run_name` and checkpoint directory;
5. run `--validate-only` and `--plan-only` before starting the new loop.

Do not replace only `best_metrics.json` inside an old completed run and continue from its completion markers. The earlier arena and promotion history would still have been selected under the constrained-primary semantics.

## Result interpretation

Report the two controllers separately:

- **ungated**: learned neural policy plus MCTS; used for training alignment, checkpoint selection, promotion, and headline final-test results;
- **constrained**: neural policy plus MCTS plus continuation-gate filtering; reported as a secondary hybrid-controller result.

A positive `continuation_gate_gain` measures the contribution of the external gate and must not be attributed to the learned policy itself.
