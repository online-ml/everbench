"""Task-configured River metrics and their durable runtime checkpoint."""

from __future__ import annotations

import hashlib
import math
import pickle
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

PROBLEM_TYPES = frozenset(
    {
        "regression",
        "binary_classification",
        "multiclass_classification",
        "clustering",
        "anomaly_detection",
    }
)


def _metric_name(metric: Any) -> str:
    return type(metric).__name__


def metric_definition(problem_type: str, prototypes: Iterable[Any]) -> dict[str, Any]:
    """Return a stable description of a task's metric configuration.

    River metric instances are task configuration, not shared mutable state.
    The learner clones them once for every model.
    """
    metrics = tuple(prototypes)
    names = [_metric_name(metric) for metric in metrics]
    if problem_type not in PROBLEM_TYPES:
        choices = ", ".join(sorted(PROBLEM_TYPES))
        raise ValueError(f"unknown PROBLEM_TYPE {problem_type!r}; choose one of {choices}")
    if not metrics:
        raise ValueError("METRICS must contain at least one River metric")
    if len(names) != len(set(names)):
        raise ValueError("METRICS cannot contain two metrics with the same class name")
    for metric in metrics:
        if not all(callable(getattr(metric, name, None)) for name in ("clone", "get", "update")):
            raise TypeError("each METRICS entry must be a River metric instance")
        try:
            float(metric.get())
        except (TypeError, ValueError) as error:
            raise TypeError("each METRICS entry must produce one numeric value for the leaderboard") from error
    return {
        "problem_type": problem_type,
        "metric_names": names,
        # This catches a task changing, for example, ROCAUC's threshold count.
        "fingerprint": hashlib.sha256(pickle.dumps(metrics, protocol=pickle.HIGHEST_PROTOCOL)).hexdigest(),
    }


@dataclass
class MetricTracker:
    """Independent River metric instances for one (task, model) pair."""

    definition: dict[str, Any]
    metrics: dict[str, Any]
    predictions: int = 0
    observations: int = 0

    @classmethod
    def fresh(cls, problem_type: str, prototypes: Iterable[Any], predictions: int = 0) -> MetricTracker:
        prototypes = tuple(prototypes)
        definition = metric_definition(problem_type, prototypes)
        return cls(
            definition=definition,
            metrics={_metric_name(metric): metric.clone() for metric in prototypes},
            predictions=predictions,
        )

    @classmethod
    def restore(cls, definition: dict[str, Any], payload: bytes) -> MetricTracker:
        tracker = pickle.loads(payload)
        if not isinstance(tracker, cls):
            raise TypeError("metric checkpoint has an unexpected type")
        if tracker.definition != definition:
            raise RuntimeError(
                "task metric configuration changed after metrics were recorded; "
                "use a new task name or reset this model's metric state"
            )
        return tracker

    def update(
        self, y_true: Any, prediction: Any, inputs_for: Callable[[Any, Any, Any], tuple[Any, Any]] | None = None
    ) -> None:
        for metric in self.metrics.values():
            metric_y_true, metric_prediction = (
                inputs_for(metric, y_true, prediction) if inputs_for else (y_true, prediction)
            )
            metric.update(metric_y_true, metric_prediction)
        self.observations += 1

    def values(self) -> dict[str, float | None]:
        values: dict[str, float | None] = {}
        for name, metric in self.metrics.items():
            value = float(metric.get())
            values[name] = value if math.isfinite(value) else None
        return values

    def payload(self) -> bytes:
        return pickle.dumps(self, protocol=pickle.HIGHEST_PROTOCOL)
