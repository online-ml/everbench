"""Reusable model implementations, independent of individual tasks."""

from __future__ import annotations

import copy
from types import ModuleType
from typing import Any

from everbench import artifacts
from everbench.metrics import MetricTracker


def prediction_for(task: ModuleType, model: Any, event_id: str, event: dict[str, Any]) -> Any:
    """Apply a task's prediction semantics to a model."""
    if task.PROBLEM_TYPE in {"binary_classification", "multiclass_classification"}:
        try:
            probabilities = model.predict_proba_one(event_id, event)
        except AttributeError:
            return model.predict_one(event_id, event)
        if task.PROBLEM_TYPE == "multiclass_classification":
            return probabilities
        return probabilities.get(True, probabilities.get(1, probabilities.get("true", 0.0)))
    if task.PROBLEM_TYPE == "anomaly_detection":
        try:
            return model.score_one(event_id, event)
        except AttributeError:
            return model.predict_one(event_id, event)
    return model.predict_one(event_id, event)


def metric_inputs_for(task: ModuleType, metric: Any, y_true: Any, prediction: Any) -> tuple[Any, Any]:
    """Let a task adapt a stored prediction to a metric's expected input."""
    hook = getattr(task, "metric_inputs_for", None)
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


def supports_learning(model: Any) -> bool:
    """Whether a predictor should receive labels for an in-place update."""
    capability = getattr(model, "supports_learning", None)
    return capability if isinstance(capability, bool) else callable(getattr(model, "learn_one", None))


def validate_uploaded_model(
    task: ModuleType,
    payload: bytes,
    signature: str,
    examples: list[tuple[str, dict[str, Any], object]],
) -> int:
    """Check a signed upload against available labelled examples without mutating it.

    This deliberately lives alongside the model protocol: the validation is a
    generic protocol check, not a separate model subsystem.
    """
    candidate = PickledModel("validation", copy.deepcopy(artifacts.loads(payload, signature)))
    if not examples:
        return 0
    tracker = MetricTracker.fresh(task.PROBLEM_TYPE, task.METRICS)
    for event_id, event, y in examples[-5:]:
        prediction = prediction_for(task, candidate, event_id, event)
        tracker.update(y, prediction, lambda metric, target, value: metric_inputs_for(task, metric, target, value))
        if supports_learning(candidate):
            candidate.learn_one(event_id, event, y)
    tracker.values()
    return min(len(examples), 5)
