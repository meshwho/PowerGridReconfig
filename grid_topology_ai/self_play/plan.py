from __future__ import annotations

from pathlib import Path

import pandas as pd

from grid_topology_ai.config import SelfPlayConfig
from grid_topology_ai.self_play.paths import SelfPlayPaths


def _count_unique_scenarios(path: Path) -> int:
    if not path.exists():
        return 0

    df = pd.read_csv(path)

    if "scenario_id" not in df.columns:
        return 0

    return int(df["scenario_id"].nunique())


def _path_status(path: Path) -> str:
    if path.exists():
        if path.is_dir():
            return "OK dir"

        if path.is_file():
            return "OK file"

        return "OK exists"

    return "MISSING"


def render_execution_plan(
    *,
    config: SelfPlayConfig,
    paths: SelfPlayPaths,
    config_path: Path,
) -> str:
    pool_size = _count_unique_scenarios(paths.pool_transitions_csv)
    eval_size = _count_unique_scenarios(paths.eval_csv)
    estimated_examples_per_iteration = (
        config.n_scenarios_per_iteration * config.generation.max_steps
    )
    examples_per_iteration = (
        config.training.examples_per_iteration
        or estimated_examples_per_iteration
    )
    first_iter_dir = paths.iteration_dir(1)

    lines: list[str] = []
    lines.extend(
        [
            "",
            "=" * 100,
            "Self-play execution plan",
            "=" * 100,
            f"Project root:              {paths.project_root}",
            f"Config:                    {config_path}",
            f"Run name:                  {config.run_name}",
            f"Output dir:                {paths.run_dir}",
            "",
            "Scenario pool:",
            f"  transitions_csv:          {paths.pool_transitions_csv}",
            f"  transitions status:       {_path_status(paths.pool_transitions_csv)}",
            f"  raw_dir:                  {paths.pool_raw_dir}",
            f"  raw status:               {_path_status(paths.pool_raw_dir)}",
            f"  metadata_path:            {paths.pool_metadata}",
            f"  metadata status:          {_path_status(paths.pool_metadata)}",
            f"  unique scenarios:         {pool_size}",
            "",
            "Evaluation set:",
            f"  eval_csv:                 {paths.eval_csv}",
            f"  eval csv status:          {_path_status(paths.eval_csv)}",
            f"  eval_raw_dir:             {paths.eval_raw_dir}",
            f"  eval raw status:          {_path_status(paths.eval_raw_dir)}",
            f"  unique eval scenarios:    {eval_size}",
            "",
            "Bootstrap:",
            f"  checkpoint:               {paths.bootstrap_checkpoint}",
            f"  checkpoint status:        {_path_status(paths.bootstrap_checkpoint)}",
            f"  metrics:                  {paths.bootstrap_metrics}",
            f"  metrics status:           {_path_status(paths.bootstrap_metrics)}",
            "",
            "Canonical self-play best:",
            f"  best checkpoint:          {paths.best_checkpoint}",
            f"  best metrics:             {paths.best_metrics}",
            "",
            "Loop:",
            f"  iterations:               {config.n_iterations}",
            f"  scenarios per iteration:  {config.n_scenarios_per_iteration}",
            f"  max_steps:                {config.generation.max_steps}",
            f"  estimated raw examples:   {estimated_examples_per_iteration} per iteration",
            "",
            "Replay buffer:",
            f"  max_size:                 {config.replay_buffer.max_size}",
            f"  min_size_to_train:        {config.replay_buffer.min_size_to_train}",
            f"  fresh_fraction:           {config.replay_buffer.fresh_fraction}",
            "",
            "Generation:",
            f"  simulations:              {config.generation.simulations}",
            f"  depth:                    {config.generation.depth}",
            f"  top_k:                    {config.generation.top_k}",
            f"  gamma:                    {config.generation.gamma}",
            f"  use_root_noise:           {config.generation.use_root_noise}",
            f"  use_continuation_gate:    {config.generation.use_continuation_gate}",
            "",
            "Training:",
            f"  examples_per_iteration:   {examples_per_iteration}",
            f"  epochs_per_iteration:     {config.training.epochs}",
            f"  batch_size:               {config.training.batch_size}",
            f"  learning_rate:            {config.training.learning_rate}",
            f"  model_type:               {config.training.model_type}",
            f"  hidden_dim:                {config.training.hidden_dim}",
            f"  num_layers:                {config.training.num_layers}",
            f"  dropout:                   {config.training.dropout}",
            "",
            "Evaluation:",
            f"  simulations:              {config.evaluation.simulations}",
            f"  depth:                    {config.evaluation.depth}",
            f"  max_steps:                {config.evaluation.max_steps}",
            f"  device:                   {config.evaluation.device}",
            "",
            "Acceptance:",
            f"  metric:                   {config.acceptance.metric}",
            f"  min_improvement:          {config.acceptance.min_improvement}",
            f"  max_simple_drop:          {config.acceptance.max_simple_solve_rate_drop}",
            "",
            "First iteration would write:",
            f"  {first_iter_dir / 'selected_scenario_ids.txt'}",
            f"  {first_iter_dir / 'raw' / 'examples.csv'}",
            f"  {first_iter_dir / 'train_batch.csv'}",
            f"  {first_iter_dir / 'candidate_checkpoint.pt'}",
            f"  {first_iter_dir / 'eval_metrics.json'}",
            f"  {first_iter_dir / 'metadata.json'}",
            f"  {paths.learning_curve}",
            "",
            "No generation, training, evaluation, or file creation was performed.",
        ]
    )

    return "\n".join(lines)
