"""Reusable model implementations, independent of individual tasks."""

from __future__ import annotations

from typing import Any

from everbench import artifacts
from everbench.metrics import MetricTracker
from everbench.tasks import TaskDefinition


def prediction_for(task: TaskDefinition, model: PickledModel, event_id: str, event: dict[str, Any]) -> Any:
    """Apply a task's prediction semantics to a model."""
    if task.PROBLEM_TYPE in {"binary_classification", "multiclass_classification"}:
        if model.supports_probabilities:
            probabilities = model.predict_proba_one(event_id, event)
            if task.PROBLEM_TYPE == "multiclass_classification":
                return probabilities
            return probabilities.get(True, probabilities.get(1, probabilities.get("true", 0.0)))
        return model.predict_one(event_id, event)
    if task.PROBLEM_TYPE == "anomaly_detection":
        if model.supports_scoring:
            return model.score_one(event_id, event)
        return model.predict_one(event_id, event)
    return model.predict_one(event_id, event)


def metric_inputs_for(task: TaskDefinition, metric: Any, y_true: Any, prediction: Any) -> tuple[Any, Any]:
    """Let a task adapt a stored prediction to a metric's expected input."""
    hook = task.metric_inputs_for
    return hook(metric, y_true, prediction) if hook is not None else (y_true, prediction)


class PickledModel:
    """Adapter for trusted online or scoring-only predictor pickles."""

    def __init__(self, model_id: str, model: Any):
        self._predict_one = getattr(model, "predict_one", None)
        self._predict_proba_one = getattr(model, "predict_proba_one", None)
        self._score_one = getattr(model, "score_one", None)
        if not (callable(self._predict_one) or callable(self._predict_proba_one) or callable(self._score_one)):
            raise TypeError(
                "pickled models must provide predict_one(event_id, event), predict_proba_one(event_id, event), or score_one(event_id, event)"
            )
        self.model_id = model_id
        self.model = model

    @property
    def supports_learning(self) -> bool:
        return callable(getattr(self.model, "learn_one", None))

    @property
    def supports_probabilities(self) -> bool:
        return callable(self._predict_proba_one)

    @property
    def supports_scoring(self) -> bool:
        return callable(self._score_one)

    def predict_one(self, event_id: str, event: dict[str, Any]) -> Any:
        if not callable(self._predict_one):
            raise AttributeError("underlying model does not provide predict_one")
        return self._predict_one(event_id, event)

    def predict_proba_one(self, event_id: str, event: dict[str, Any]) -> Any:
        if not callable(self._predict_proba_one):
            raise AttributeError("underlying model does not provide predict_proba_one")
        return self._predict_proba_one(event_id, event)

    def score_one(self, event_id: str, event: dict[str, Any]) -> Any:
        if not callable(self._score_one):
            raise AttributeError("underlying model does not provide score_one")
        return self._score_one(event_id, event)

    def learn_one(self, event_id: str, event: dict[str, Any], label: Any) -> None:
        learner = getattr(self.model, "learn_one", None)
        if learner is None:
            raise AttributeError("underlying model does not provide learn_one")
        learner(event_id, event, label)

    def payload(self) -> bytes:
        return artifacts.dumps(self.model)


def validate_model(
    task: TaskDefinition,
    candidate: PickledModel,
    examples: list[tuple[str, dict[str, Any], object]],
) -> int:
    """Check a model protocol against recent examples."""
    if not examples:
        return 0
    tracker = MetricTracker.fresh(task.PROBLEM_TYPE, task.METRICS)
    for event_id, event, y in examples[-5:]:
        prediction = prediction_for(task, candidate, event_id, event)
        tracker.update(y, prediction, lambda metric, target, value: metric_inputs_for(task, metric, target, value))
        if candidate.supports_learning:
            candidate.learn_one(event_id, event, y)
    tracker.values()
    return min(len(examples), 5)
