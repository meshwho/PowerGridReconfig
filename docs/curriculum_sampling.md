# Curriculum sampling policy

This document defines how the self-play loop samples scenarios from the fixed
physical pool. The policy is designed to prevent permanently unsolved and hard
cases from disappearing behind a high aggregate solve rate, while keeping the
selection reproducible and auditable.

The pool itself is not regenerated or reordered by this policy. Only the
scenario selection probabilities and mandatory composition of each iteration
change.

## Configuration

Curriculum settings live under `pool.curriculum`:

```yaml
pool:
  transitions_csv: data/pool/transitions.csv
  raw_dir: data/pool/raw
  metadata_path: runs/self_play/inputs/pool_metadata.json

  curriculum:
    never_solved_min_fraction: 0.20
    hard_min_fraction: 0.25
    simple_max_fraction: 0.35
    frontier_max_fraction: 0.50

    frontier_solve_rate_min: 0.25
    frontier_solve_rate_max: 0.75

    learning_progress_weight: 1.00
    uncertainty_weight: 0.75
    staleness_weight: 0.50
    frontier_weight: 0.35

    stale_after_iterations: 3
    priority_floor: 0.05
```

The section is optional. Missing values use the defaults shown above.
Fractions must be in `[0, 1]`, the lower frontier bound must be smaller than the
upper bound, weights must be finite and non-negative, and
`stale_after_iterations` and `priority_floor` must be positive.

## Scenario groups

The staged sampler uses the following groups:

- **never solved**: `times_attempted > 0` and `times_solved == 0`;
- **hard**: `difficulty_class` is `hard`, case-insensitively;
- **simple**: `difficulty_class` is `simple`, case-insensitively;
- **frontier**: the current solve rate lies inside the configured inclusive
  interval `[frontier_solve_rate_min, frontier_solve_rate_max]`.

An unattempted scenario is not yet classified as never solved. It still starts
with maximum uncertainty and staleness, so it receives a high sampling priority.

Groups may overlap. In particular, a hard scenario that has never been solved
counts toward both minimum quotas when selected.

## Learning signals

Pool metadata schema version 3 stores five curriculum signals for every
scenario:

- `last_iteration_solve_rate`;
- `solve_rate_delta`;
- `learning_progress`;
- `uncertainty`;
- `staleness`.

### Learning progress

For the current iteration:

```text
iteration_solve_rate = solved_episodes / attempted_episodes
solve_rate_delta = iteration_solve_rate - previous_smoothed_solve_rate
```

The first observed learning-progress value is `abs(solve_rate_delta)`.
Subsequent values use an exponential moving average:

```text
learning_progress =
    (1 - ema_alpha) * previous_learning_progress
    + ema_alpha * abs(solve_rate_delta)
```

The default `ema_alpha` is `0.30`. Both improvement and deterioration therefore
increase the signal: either means the scenario is currently informative.
The stored solve rate itself is updated with the same EMA coefficient.

### Uncertainty

Uncertainty is the normalized standard deviation of a Beta posterior with a
uniform prior:

```text
alpha = times_solved + 1
beta = times_attempted - times_solved + 1

posterior_variance =
    alpha * beta
    / ((alpha + beta)^2 * (alpha + beta + 1))

uncertainty =
    sqrt(posterior_variance) / sqrt(1 / 12)
```

The result is clipped to `[0, 1]`. New scenarios start at `1.0`; uncertainty
usually decreases as evidence accumulates.

### Staleness

Unattempted scenarios have staleness `1.0`. Otherwise:

```text
staleness = min(
    (current_iteration - last_attempted_iteration)
    / stale_after_iterations,
    1.0,
)
```

Thus `stale_after_iterations` controls how quickly an ignored scenario reaches
maximum staleness.

Selected scenarios with no generated episode rows still count as attempted.
Their `last_attempted_iter` advances, their `solve_rate_delta` is set to zero,
and uncertainty and staleness are refreshed. This prevents failed generation
from making a selected scenario look unseen forever.

## Priority score

Schema-v3 priorities are additive and auditable:

```text
frontier_signal = 4 * solve_rate * (1 - solve_rate)

priority =
    priority_floor
    + learning_progress_weight * learning_progress
    + uncertainty_weight * uncertainty
    + staleness_weight * staleness
    + frontier_weight * frontier_signal
    + difficulty_bonus
```

The difficulty bonus is:

| Difficulty | Bonus |
| --- | ---: |
| `simple` | `0.0` |
| `medium` or unknown | `0.1` |
| `hard` | `0.2` |

All continuous signals are clipped to `[0, 1]` before weighting. The frontier
term is intentionally only one component; a solve rate of zero no longer
collapses the whole priority to zero. Each scenario stores
`priority_components`, including the final total, so the score can be audited.

## Staged sampling

Let:

```text
target_count = min(requested_count, pool_size)
```

Sampling is without replacement and proceeds in four stages.

### 1. Never-solved quota

The sampler first requests:

```text
ceil(target_count * never_solved_min_fraction)
```

never-solved scenarios, limited by availability.

### 2. Hard quota

It then requests enough additional hard scenarios to reach:

```text
ceil(target_count * hard_min_fraction)
```

Hard scenarios already selected in the first stage count toward this target.

### 3. Capped residual fill

The remaining slots are selected one at a time using normalized priority
weights. During this residual fill, the sampler applies:

```text
simple_limit = floor(target_count * simple_max_fraction)
frontier_limit = floor(target_count * frontier_max_fraction)
```

Minimum quotas take precedence over maximum caps. A quota-stage selection may
therefore make the final simple or frontier fraction exceed its nominal cap.
The caps constrain only candidates considered while filling the remaining
slots; they never remove an already selected mandatory scenario.

### 4. Deterministic fallback

If the remaining pool cannot fill the batch under both caps, the sampler relaxes
caps only as needed:

1. `frontier_max_fraction`;
2. `simple_max_fraction`.

Every relaxation is written to the sampling report. If a mandatory group has
fewer candidates than its target, the sampler fills the batch from the rest of
the pool and records the quota shortfall instead of failing the iteration.

Within each stage, candidates are drawn with probability proportional to their
current priority. Scenario IDs are unique in the result.

## Reproducibility

Scenario sampling uses the iteration-specific scenario-sampling child seed
derived from the configured base seed. The same pool metadata, configuration,
iteration and seed produce the same ordered scenario selection.

Priority refresh is performed before sampling, using the current iteration for
staleness. The diagnostic report is prepared from a copy of pool metadata, so
pre-generation reporting does not mutate the authoritative scenario state.
The persisted pool state is refreshed after the iteration completes.

## Diagnostics and artifacts

Every schema-v3 iteration writes:

```text
iteration_XXXX/curriculum_sampling.json
```

The report includes:

- requested, target and selected counts;
- pool and selected counts by difficulty;
- available, target, selected, fraction and shortfall for never-solved cases;
- the same fields for hard cases;
- selected count, fraction and configured limit for simple and frontier cases;
- any cap relaxations;
- the scenario-sampling seed;
- pool and selected coverage of scenarios unvisited for at least
  `stale_after_iterations`;
- unvisited coverage split by difficulty;
- mean priority, learning progress, uncertainty and staleness of the selected
  scenarios.

The self-play loop also prints a compact version of this summary before
generation.

After the iteration, `metadata.json` stores:

- `hashes.curriculum_sampling_sha256`;
- `extra.curriculum_sampling_path`;
- `extra.curriculum_sampling_sha256`;
- `extra.curriculum_sampling` with the complete report.

The actual selected scenario IDs must exactly match the IDs used to prepare the
report. A mismatch aborts the iteration finalization instead of publishing
misleading diagnostics.

`learning_curve.csv` receives these columns:

```text
curriculum_never_solved_fraction
curriculum_hard_fraction
curriculum_simple_fraction
curriculum_frontier_fraction
curriculum_unvisited_pool_fraction
curriculum_unvisited_selected_fraction
curriculum_never_solved_shortfall
curriculum_hard_shortfall
curriculum_cap_relaxation_count
curriculum_mean_priority
curriculum_mean_learning_progress
curriculum_mean_uncertainty
curriculum_mean_staleness
```

These columns make coverage drift visible across iterations without requiring
operators to parse every JSON report.

## Pool metadata migration

Pool metadata schema version 2 is migrated in place to version 3 when loaded.
Migration preserves historical attempt, solve, solve-rate and timing statistics,
then initializes and refreshes the new learning signals.

This migration does not change the transition CSV, raw state schema or saved NPZ
state files. No pool regeneration is required solely because curriculum
sampling was enabled.

Code paths that receive an unmigrated schema-v2 object directly retain the
legacy priority and sampling behavior. Normal self-play pipeline initialization
loads and migrates the metadata before iteration sampling.

## Tuning guidance

Minimum quotas and maximum caps are independent constraints; their configured
fractions do not need to sum to one. For small batches, quotas use `ceil` and
caps use `floor`, so one scenario can materially change the realized fraction.
Inspect the JSON report rather than inferring counts from configuration alone.

Recommended first adjustments:

- increase `never_solved_min_fraction` when persistent failures receive too few
  attempts;
- increase `hard_min_fraction` when difficulty coverage is insufficient;
- reduce `simple_max_fraction` if easy cases dominate residual sampling;
- reduce `frontier_max_fraction` if medium solve-rate cases crowd out zero-rate
  cases;
- increase `staleness_weight` or decrease `stale_after_iterations` when cases
  remain unvisited too long;
- increase `uncertainty_weight` when under-observed cases need more exploration;
- increase `learning_progress_weight` when rapidly changing cases should be
  revisited more often.

A persistent non-zero shortfall means the fixed pool does not contain enough
eligible scenarios to satisfy the requested quota. Changing priority weights
cannot fix an availability shortfall; the pool composition or quota must change.
