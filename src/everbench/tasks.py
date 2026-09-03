"""Loading task modules into validated, immutable runtime definitions."""

from __future__ import annotations

import importlib.util
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from threading import Event
from types import ModuleType
from typing import Any

from everbench.metrics import metric_definition

TASK_FILENAME = "task.py"


@dataclass(frozen=True)
class TaskDefinition:
    TASK_NAME: str
    EVENT_STREAM_URL: str
    LABEL_STREAM_URL: str
    accepts_event: Callable[[dict[str, Any]], bool]
    event_id: Callable[[dict[str, Any]], str | None]
    label_for: Callable[[dict[str, Any]], tuple[str, Any, str] | None]
    PROBLEM_TYPE: str
    METRICS: tuple[Any, ...]
    DESCRIPTION_HTML: str
    __file__: str
    NEGATIVE_LABEL_DELAY_SECONDS: float | None = None
    event_stream: Callable[[Event], Iterable[dict[str, Any]]] | None = None
    label_stream: Callable[[Event], Iterable[dict[str, Any]]] | None = None
    event_timestamp: Callable[[dict[str, Any]], float] | None = None
    label_timestamp: Callable[[dict[str, Any]], float] | None = None
    metric_inputs_for: Callable[[Any, Any, Any], tuple[Any, Any]] | None = None


def task_paths(directory: str | Path) -> list[Path]:
    """Return canonical task definitions, one per task directory."""
    return sorted(Path(directory).glob(f"*/{TASK_FILENAME}"))


def _module(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"everbench_task_{path.parent.name}", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load task file: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _callable(module: ModuleType, name: str) -> Callable:
    value = getattr(module, name, None)
    if not callable(value):
        raise ValueError(f"task {module.__file__} must define callable {name}()")
    return value


def _optional_callable(module: ModuleType, name: str) -> Callable | None:
    value = getattr(module, name, None)
    if value is not None and not callable(value):
        raise ValueError(f"task {module.__file__} must define callable {name}()")
    return value


def load_task(path: str | Path) -> TaskDefinition:
    return _load_task(Path(path).resolve())


@lru_cache
def _load_task(task_path: Path) -> TaskDefinition:
    module = _module(task_path)
    required_values = (
        "TASK_NAME",
        "EVENT_STREAM_URL",
        "LABEL_STREAM_URL",
        "PROBLEM_TYPE",
        "METRICS",
        "DESCRIPTION_HTML",
    )
    missing = [name for name in required_values if not hasattr(module, name)]
    if missing:
        raise ValueError(f"task {task_path} is missing: {', '.join(missing)}")
    if not isinstance(module.TASK_NAME, str) or not module.TASK_NAME:
        raise ValueError("TASK_NAME must be a non-empty string")
    if not isinstance(module.DESCRIPTION_HTML, str):
        raise ValueError("DESCRIPTION_HTML must be a string")
    metrics = tuple(module.METRICS)
    metric_definition(module.PROBLEM_TYPE, metrics)
    delay = getattr(module, "NEGATIVE_LABEL_DELAY_SECONDS", None)
    if delay is not None and module.PROBLEM_TYPE != "binary_classification":
        raise ValueError("NEGATIVE_LABEL_DELAY_SECONDS is only supported for binary_classification tasks")
    if delay is not None and float(delay) <= 0:
        raise ValueError("NEGATIVE_LABEL_DELAY_SECONDS must be positive")
    event_stream = _optional_callable(module, "event_stream")
    label_stream = _optional_callable(module, "label_stream")
    for stream_name, stream, url in (
        ("event", event_stream, module.EVENT_STREAM_URL),
        ("label", label_stream, module.LABEL_STREAM_URL),
    ):
        if stream is None and (not isinstance(url, str) or not url):
            raise ValueError(f"{stream_name} tasks must define a stream callable or a non-empty URL")
    return TaskDefinition(
        TASK_NAME=module.TASK_NAME,
        EVENT_STREAM_URL=module.EVENT_STREAM_URL,
        LABEL_STREAM_URL=module.LABEL_STREAM_URL,
        accepts_event=_callable(module, "accepts_event"),
        event_id=_callable(module, "event_id"),
        label_for=_callable(module, "label_for"),
        PROBLEM_TYPE=module.PROBLEM_TYPE,
        METRICS=metrics,
        DESCRIPTION_HTML=module.DESCRIPTION_HTML,
        __file__=str(task_path),
        NEGATIVE_LABEL_DELAY_SECONDS=float(delay) if delay is not None else None,
        event_stream=event_stream,
        label_stream=label_stream,
        event_timestamp=_optional_callable(module, "event_timestamp"),
        label_timestamp=_optional_callable(module, "label_timestamp"),
        metric_inputs_for=_optional_callable(module, "metric_inputs_for"),
    )


def load_task_named(task_name: str, directory: str | Path = "tasks") -> TaskDefinition:
    """Find a local task definition by its stable task name."""
    for task_path in task_paths(directory):
        task = load_task(task_path)
        if task.TASK_NAME == task_name:
            return task
    raise LookupError(f"no task definition found for {task_name!r}")


def discover_tasks(directory: str | Path = "tasks") -> list[TaskDefinition]:
    paths = task_paths(directory)
    if not paths:
        raise LookupError(f"no task definitions found in {Path(directory)}")
    tasks = [load_task(path) for path in paths]
    names = [task.TASK_NAME for task in tasks]
    duplicates = sorted(name for name in set(names) if names.count(name) > 1)
    if duplicates:
        raise ValueError(f"task names must be unique: {', '.join(duplicates)}")
    return tasks
