"""Loading and validating task-specific benchmark definitions."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

from everbench.metrics import metric_definition

REQUIRED_TASK_MEMBERS = (
    "TASK_NAME",
    "EVENT_STREAM_URL",
    "LABEL_STREAM_URL",
    "accepts_event",
    "event_id",
    "features_for",
    "label_for",
    "PROBLEM_TYPE",
    "METRICS",
    "DESCRIPTION_HTML",
)


def load_task(path: str | Path) -> ModuleType:
    task_path = Path(path).resolve()
    spec = importlib.util.spec_from_file_location(f"everbench_task_{task_path.stem}", task_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load task file: {task_path}")
    task = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(task)
    missing = [name for name in REQUIRED_TASK_MEMBERS if not hasattr(task, name)]
    if missing:
        raise ValueError(f"task {task_path} is missing: {', '.join(missing)}")
    if not isinstance(task.DESCRIPTION_HTML, str):
        raise ValueError("DESCRIPTION_HTML must be a string")
    metric_definition(task.PROBLEM_TYPE, task.METRICS)
    if getattr(task, "NEGATIVE_LABEL_DELAY_SECONDS", None) is not None and task.PROBLEM_TYPE != "binary_classification":
        raise ValueError("NEGATIVE_LABEL_DELAY_SECONDS is only supported for binary_classification tasks")
    return task


def load_task_named(task_name: str, directory: str | Path = "tasks") -> ModuleType:
    """Find a local task definition by its stable task name."""
    for task_path in sorted(Path(directory).glob("*.py")):
        task = load_task(task_path)
        if task.TASK_NAME == task_name:
            return task
    raise LookupError(f"no task definition found for {task_name!r}")


def discover_tasks(directory: str | Path = "tasks") -> list[ModuleType]:
    """Load every top-level task definition from ``directory``.

    Supporting files can live in task subdirectories. Only top-level Python
    files are benchmark definitions, which makes a deployment automatically
    pick up a new task after its next restart.
    """
    task_paths = sorted(Path(directory).glob("*.py"))
    if not task_paths:
        raise LookupError(f"no task definitions found in {Path(directory)}")

    tasks = [load_task(path) for path in task_paths]
    names = [task.TASK_NAME for task in tasks]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"task names must be unique: {', '.join(duplicates)}")
    return tasks
