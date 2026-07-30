# Topology action contract

This document defines the stable policy layout, executable branch-status actions, provenance fields, and compatibility rules used by topology-control artifacts.

## Scope

The topology agent controls branch status only:

- stop/handoff;
- open an active branch;
- close an explicitly allowed inactive branch.

Redispatch, reactive-power control, and other continuous controls remain separate phases. They are not encoded as additional topology-policy slots.

## Stable policy layout

The graph policy uses the layout name:

```text
stop_plus_branch_status_v1
```

For a state with branch IDs stored in order as `branch_ids[0..N-1]`, the policy vector has `N + 1` positions:

| Policy index | Slot kind | Target |
| --- | --- | --- |
| `0` | `stop` | no branch target |
| `1 + branch_pos` | `branch_status` | `branch_ids[branch_pos]` |

A serialized slot contains exactly:

```json
{
  "action_id": 1,
  "kind": "branch_status",
  "target_id": 42,
  "target_pos": 0
}
```

The layout identifies the controlled object. It deliberately does not contain `target_status`.

## State-dependent executable action

One stable branch slot maps to the command required by the current state:

- active branch -> `switch_off_branch`, `target_status=0`;
- inactive branch -> `switch_on_branch`, `target_status=1`.

The policy index therefore remains stable while the executable direction changes with `branch_status`. There is no separate permanent policy slot for opening and closing the same branch.

`GridFMAction` is the executable action and carries:

- `action_id`;
- `action_type`;
- `branch_id`;
- `branch_pos`;
- `target_status`.

## Legality and the action mask

The action layout defines identity and order. The action mask defines whether each slot is legal in the current state.

Structural rules:

- slot `0` is the stop/handoff slot;
- opening is available only for an active branch;
- when `require_connected_after_switch=true`, opening a branch that would disconnect the active grid is masked out;
- closing is available only for an inactive branch whose ID is listed in `closeable_branch_ids`;
- configured closeable branch IDs that are absent from the grid are rejected;
- the branch order in `branch_ids` is never inferred from the mask.

Operational rules:

- `min_loading_for_switch_percent` filters opening candidates only;
- it does not filter closure of an inactive tie-line;
- `valid_action_mask()` is the compatibility alias for the operational mask.

The mask is not topology provenance. Two artifacts with masks of the same length are not compatible unless their action layouts also match exactly.

## Action-space configuration contract

The semantic action configuration contains:

```json
{
  "require_connected_after_switch": true,
  "min_loading_for_switch_percent": 0.0,
  "closeable_branch_ids": []
}
```

`enable_cache` is intentionally excluded. Cache use changes execution performance, not action semantics, so it does not affect the configuration fingerprint.

Changing any semantic field changes the topology action configuration fingerprint.

## Provenance fields

Every compatible topology artifact carries these five fields:

| Field | Meaning |
| --- | --- |
| `topology_action_contract_version` | schema version of the topology action contract |
| `topology_action_config` | canonical semantic action-space configuration |
| `topology_action_config_fingerprint` | SHA-256 fingerprint of the canonical configuration |
| `action_layout` | ordered serialized list of policy slots |
| `action_layout_fingerprint` | SHA-256 fingerprint of the ordered layout |

Canonical fingerprints use UTF-8 JSON with sorted keys, compact separators, and no NaN values.

Because the layout fingerprint includes ordered branch identity, all of the following are incompatible even when `num_actions` is unchanged:

- a different branch ID;
- a different branch order;
- a missing or additional branch;
- a different slot kind or target position.

## Artifact storage

### Self-play CSV

Each example row stores all five provenance fields. `topology_action_config` and `action_layout` are compact JSON strings.

### NPZ state

The NPZ contains `branch_ids` as a state array. Its `metadata_json` contains the same five provenance fields as the CSV row.

Validation rebuilds the layout from NPZ `branch_ids` and requires it to match both the NPZ metadata and the corresponding CSV row.

### Replay buffer

Every replay row carries topology provenance. `buffer_manifest.json` also stores the canonical action configuration and layout.

On load and mutation:

- the replay schema version must match exactly;
- every row must match the manifest configuration;
- every row must match the manifest layout;
- mixed topology layouts are rejected before the buffer is mutated.

### Checkpoint

A graph checkpoint stores the five provenance fields at the checkpoint root and inside dataset metadata. It also stores:

```text
policy_layout = stop_plus_branch_status_v1
```

Fine-tuning requires the initial checkpoint to match the training dataset's exact action configuration, ordered layout, layout fingerprint, and policy layout.

The neural evaluator rebuilds the layout from the evaluated state's `branch_ids` before inference. A same-sized graph with a different branch identity or order is rejected.

## Current contract versions

The current versions are:

| Contract | Version |
| --- | ---: |
| physical objective | `3` |
| outcome/value target | `4` |
| evaluation metrics | `6` |
| checkpoint | `6` |
| replay buffer schema | `5` |
| physics configuration | `1` |
| topology action | `1` |

Consumers require exact current versions. Missing fields and older versions fail closed.

## Compatibility boundary

Legacy datasets and checkpoints created before topology provenance cannot be upgraded by inserting metadata after the fact. The original branch identity and order must be known at generation time.

Do not:

- infer a layout from `num_actions` alone;
- infer branch identity from `edge_index` alone;
- relabel an old checkpoint with the current contract version;
- mix old replay chunks with current rows;
- reuse a checkpoint trained against a different branch order.

## Regeneration

Archive the old replay/run directory and rebuild a clean artifact chain:

```bash
# 1. Generate new examples and NPZ states with topology provenance.
python -m scripts.self_play.generate <POOL_RAW_DIR> --transitions <POOL_TRANSITIONS.csv> --output-dir <NEW_SELF_PLAY_DIR> --pf-alg 3

# 2. Train a fresh checkpoint from the new examples.
python -m scripts.self_play.train_graph_baseline <NEW_SELF_PLAY_DIR>/examples.csv --output <NEW_CHECKPOINT.pt> --device cpu

# 3. Recompute fixed evaluation artifacts for the new checkpoint.
python -m scripts.evaluation.evaluate_checkpoint <EVAL_RAW_DIR> --transitions <EVAL_TRANSITIONS.csv> --checkpoint <NEW_CHECKPOINT.pt> --pf-alg 3 --output-csv <NEW_EVAL_RESULTS.csv> --output-json <NEW_EVAL_METRICS.json>
```

Do not initialize the new training run from a legacy checkpoint.

## Developer invariants

When extending topology actions:

1. Keep slot identity separate from the state-dependent command.
2. Version any incompatible serialized layout or policy-head interpretation.
3. Include semantic configuration in provenance; exclude performance-only settings.
4. Compare ordered layout fingerprints across CSV, NPZ, replay, training, checkpoints, and inference.
5. Reject incompatible artifacts instead of silently adapting them.
6. Keep redispatch and reactive control outside this topology policy unless a separately versioned policy layout is introduced.
