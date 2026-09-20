"""Loading task modules into validated, immutable runtime definitions."""

from __future__ import annotations

import importlib.util
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from everbench.auto.config import AutoResearchConfig

from everbench.metrics import metric_definition
from everbench.sources import Source

TASK_FILENAME = "task.py"


@dataclass(frozen=True, kw_only=True)
class LabelPolicy:
    """The target horizon, matching tolerance, and result when no label arrives."""

    delay_seconds: float
    default_label: Any = None
    tolerance_seconds: float = 0
    grace_seconds: float = 60

    def __post_init__(self) -> None:
        if self.delay_seconds <= 0 or self.tolerance_seconds < 0 or self.grace_seconds < 0:
            raise ValueError("label horizon must be positive; tolerance and grace must be nonnegative")

    @property
    def close_after_seconds(self) -> float:
        return self.delay_seconds + self.tolerance_seconds + self.grace_seconds


@dataclass(frozen=True, kw_only=True)
class TaskDefinition:
    TASK_NAME: str
    PROBLEM_TYPE: str
    METRICS: tuple[Any, ...]
    DESCRIPTION_HTML: str
    sources: tuple[Source, ...]
    label_policy: LabelPolicy | None = None
    LEADERBOARD_PRIMARY_METRIC: str | None = None
    metric_inputs_for: Callable[..., tuple[Any, Any]] | None = None
    AUTO_RESEARCH: AutoResearchConfig | None = None
    __file__: str = ""

    def __post_init__(self) -> None:
        if not self.TASK_NAME or not self.sources:
            raise ValueError("tasks need a name and at least one source")
        names = [source.name for source in self.sources]
        if len(names) != len(set(names)):
            raise ValueError("source names must be unique within a task")
        metric_definition(problem_type=self.PROBLEM_TYPE, prototypes=self.METRICS)
        if self.LEADERBOARD_PRIMARY_METRIC is not None and self.LEADERBOARD_PRIMARY_METRIC not in {
            type(metric).__name__ for metric in self.METRICS
        }:
            raise ValueError("LEADERBOARD_PRIMARY_METRIC must name a configured metric")


def task_paths(*, directory: str | Path) -> list[Path]:
    """Return canonical task definitions, one per task directory."""
    return sorted(Path(directory).glob(f"*/{TASK_FILENAME}"))


def _module(*, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"everbench_task_{path.parent.name}", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load task file: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_task(*, path: str | Path) -> TaskDefinition:
    return _load_task(task_path=Path(path).resolve())


@lru_cache
def _load_task(*, task_path: Path) -> TaskDefinition:
    task = getattr(_module(path=task_path), "TASK", None)
    if not isinstance(task, TaskDefinition):
        raise ValueError(f"{task_path} must export TASK = TaskDefinition(...)")
    return replace(task, __file__=str(task_path))


def load_task_named(*, task_name: str, directory: str | Path = "tasks") -> TaskDefinition:
    """Find a local task definition by its stable task name."""
    for task_path in task_paths(directory=directory):
        task = load_task(path=task_path)
        if task.TASK_NAME == task_name:
            return task
    raise LookupError(f"no task definition found for {task_name!r}")


def discover_tasks(*, directory: str | Path = "tasks", task_names: Iterable[str] = ()) -> list[TaskDefinition]:
    paths = task_paths(directory=directory)
    if not paths:
        raise LookupError(f"no task definitions found in {Path(directory)}")
    tasks = [load_task(path=path) for path in paths]
    names = [task.TASK_NAME for task in tasks]
    duplicates = sorted(name for name in set(names) if names.count(name) > 1)
    if duplicates:
        raise ValueError(f"task names must be unique: {', '.join(duplicates)}")
    requested = set(task_names)
    if requested:
        missing = requested - set(names)
        if missing:
            raise LookupError(f"task definitions not found: {', '.join(sorted(missing))}")
        tasks = [task for task in tasks if task.TASK_NAME in requested]
    return tasks
