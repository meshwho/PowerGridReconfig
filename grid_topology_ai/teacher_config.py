from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


CHECKPOINT_CONFIG_SCHEMA_VERSION = 1

# These settings control execution resources only. Exact PF caching is physically
# transparent, so enabling or disabling it cannot change teacher semantics.
RUNTIME_TASK_CONFIG_FIELDS = frozenset(
    {
        "disable_cache",
        "clear_caches_every",
        "max_worker_memory_mb",
        "print_memory_events",
        "max_tasks_per_child",
        "min_free_system_memory_mb",
        "memory_registry_max_age_sec",
        "auto_worker_memory_mb",
        "auto_worker_memory_reserve_mb",
        "auto_worker_cpu_util_target",
        "auto_worker_cpu_mode",
        "auto_worker_cpu_fraction",
        "auto_worker_max",
    }
)

_CHECKPOINT_CONFIG_KEYS = frozenset(
    {
        "checkpoint_config_schema_version",
        "task_config",
        "semantic_task_config",
        "runtime_task_config",
    }
)


def _json_normalize(value: Any) -> Any:
    return json.loads(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
    )


def split_teacher_task_config(
    task_config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split teacher settings into semantic and execution-only mappings."""

    if not isinstance(task_config, Mapping):
        raise TypeError("task_config must be a mapping.")

    semantic: dict[str, Any] = {}
    runtime: dict[str, Any] = {}

    for key, value in task_config.items():
        name = str(key)
        target = runtime if name in RUNTIME_TASK_CONFIG_FIELDS else semantic
        target[name] = value

    return semantic, runtime


def semantic_teacher_task_config(
    task_config: Mapping[str, Any],
) -> dict[str, Any]:
    semantic, _runtime = split_teacher_task_config(task_config)
    return semantic


def runtime_teacher_task_config(
    task_config: Mapping[str, Any],
) -> dict[str, Any]:
    _semantic, runtime = split_teacher_task_config(task_config)
    return runtime


def checkpoint_config_payload(
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the canonical persisted checkpoint-config representation."""

    if not isinstance(config, Mapping):
        raise TypeError("checkpoint config must be a mapping.")

    if isinstance(config.get("semantic_task_config"), Mapping):
        stored_semantic = dict(config["semantic_task_config"])
        runtime_value = config.get("runtime_task_config", {})
        stored_runtime = (
            dict(runtime_value) if isinstance(runtime_value, Mapping) else {}
        )
        # Re-split the stored sections with the current classification. This
        # lets a setting become runtime-only once its implementation is proven
        # physically transparent without invalidating an existing checkpoint.
        semantic, runtime = split_teacher_task_config(
            {**stored_semantic, **stored_runtime}
        )
    else:
        task_config = config.get("task_config")
        if not isinstance(task_config, Mapping):
            raise ValueError("Teacher checkpoint config is missing task_config.")
        semantic, runtime = split_teacher_task_config(task_config)

    payload = {
        str(key): value
        for key, value in config.items()
        if str(key) not in _CHECKPOINT_CONFIG_KEYS
    }
    payload.update(
        {
            "checkpoint_config_schema_version": CHECKPOINT_CONFIG_SCHEMA_VERSION,
            "semantic_task_config": semantic,
            "runtime_task_config": runtime,
        }
    )
    return _json_normalize(payload)


def semantic_checkpoint_identity(
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Return only fields allowed to decide checkpoint compatibility."""

    payload = checkpoint_config_payload(config)
    return {
        key: value
        for key, value in payload.items()
        if key not in {
            "checkpoint_config_schema_version",
            "runtime_task_config",
        }
    }


def ensure_teacher_checkpoint_config(
    config_path: Path,
    config: Mapping[str, Any],
) -> None:
    """Persist config and reject only changes that alter teacher semantics."""

    normalized = checkpoint_config_payload(config)
    current_identity = semantic_checkpoint_identity(normalized)

    if config_path.exists():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(existing, dict):
            raise RuntimeError(
                f"Teacher checkpoint config must contain an object: {config_path}"
            )

        if semantic_checkpoint_identity(existing) != current_identity:
            raise RuntimeError(
                "Teacher checkpoint configuration does not match "
                "the current command. Use the original semantic settings, "
                "a different --run-name, or --force."
            )
        return

    config_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = config_path.with_suffix(config_path.suffix + ".tmp")
    temp_path.write_text(
        json.dumps(
            normalized,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    temp_path.replace(config_path)


def load_teacher_task_config(path: Path) -> dict[str, Any]:
    """Load either a plain task config or old/new checkpoint-config file."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Teacher task config must contain a JSON object: {path}")

    legacy = payload.get("task_config")
    if isinstance(legacy, dict):
        return dict(legacy)

    semantic = payload.get("semantic_task_config")
    if isinstance(semantic, dict):
        runtime = payload.get("runtime_task_config", {})
        if runtime is not None and not isinstance(runtime, dict):
            raise ValueError(
                f"runtime_task_config must contain an object: {path}"
            )
        return {**semantic, **dict(runtime or {})}

    return dict(payload)


def _run_identity_task_config(
    states_dir: Path,
    current_task_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Preserve legacy run IDs while making new run IDs runtime-independent."""

    config_path = states_dir.parent / "teacher_checkpoint_config.json"
    if config_path.exists():
        try:
            payload = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            payload = None

        if isinstance(payload, dict):
            # Old checkpoints hashed the full task_config. Keep doing that for an
            # in-progress legacy run so resumed rows retain the existing run_id.
            legacy = payload.get("task_config")
            if isinstance(legacy, dict):
                return dict(legacy)

            # A split checkpoint has already emitted rows using the semantic
            # section that was stored at creation time. Preserve that exact
            # historical identity even if a field is reclassified later.
            semantic = payload.get("semantic_task_config")
            if isinstance(semantic, dict):
                return dict(semantic)

    return semantic_teacher_task_config(current_task_config)


def teacher_run_id(
    states_dir: str | Path,
    task_config: Mapping[str, Any],
) -> str:
    """Build provenance identity from semantic teacher settings only."""

    resolved_states_dir = Path(states_dir).resolve()
    payload = {
        "states_dir": str(resolved_states_dir),
        "task_config": _run_identity_task_config(
            resolved_states_dir,
            task_config,
        ),
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    return f"impact_teacher_{digest[:24]}"
