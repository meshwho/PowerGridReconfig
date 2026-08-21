from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

_RAW_SOURCE_FILES = ("bus_data.parquet", "branch_data.parquet", "gen_data.parquet")
_RUNTIME_FIELDS = frozenset({"disable_cache"})


def _normalize(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False))


def teacher_source_identity(raw_dir: str | Path, transitions_path: str | Path) -> dict[str, Any]:
    """Describe the current inputs without reading large source files."""
    raw_dir = Path(raw_dir).resolve()
    transitions_path = Path(transitions_path).resolve()
    paths = [raw_dir / name for name in _RAW_SOURCE_FILES] + [transitions_path]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Required teacher source file not found: {missing[0]}")
    return {
        "raw_dir": str(raw_dir),
        "transitions_path": str(transitions_path),
        "files": {
            str(path): {
                "size": path.stat().st_size,
                "mtime_ns": path.stat().st_mtime_ns,
            }
            for path in paths
        },
    }


def semantic_teacher_task_config(task_config: Mapping[str, Any]) -> dict[str, Any]:
    return _normalize(
        {str(key): value for key, value in task_config.items() if str(key) not in _RUNTIME_FIELDS}
    )


def ensure_teacher_checkpoint_config(config_path: Path, config: Mapping[str, Any]) -> None:
    """Persist the one current Light resume identity and reject mismatches."""
    task_config = config.get("task_config")
    if not isinstance(task_config, Mapping):
        raise ValueError("Teacher checkpoint config is missing task_config.")
    identity = _normalize(
        {
            "source_identity": config.get("source_identity"),
            "scenario_ids": config.get("scenario_ids"),
            "task_config": semantic_teacher_task_config(task_config),
        }
    )
    if config_path.exists():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if existing != identity:
            raise RuntimeError(
                "Teacher checkpoint configuration does not match the current command. "
                "Use the original semantic settings, a different --run-name, or --force."
            )
        return
    config_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = config_path.with_suffix(config_path.suffix + ".tmp")
    temp_path.write_text(
        json.dumps(identity, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    temp_path.replace(config_path)


def load_teacher_task_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Teacher task config must contain a JSON object: {path}")
    task_config = payload.get("task_config", payload)
    if not isinstance(task_config, dict):
        raise ValueError(f"Teacher task config must contain a JSON object: {path}")
    return dict(task_config)


def teacher_run_id(states_dir: str | Path, task_config: Mapping[str, Any]) -> str:
    payload = {
        "states_dir": str(Path(states_dir).resolve()),
        "task_config": semantic_teacher_task_config(task_config),
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return f"impact_teacher_{hashlib.sha256(encoded).hexdigest()[:24]}"
