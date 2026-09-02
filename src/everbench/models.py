"""Reusable model implementations, independent of individual tasks."""

from __future__ import annotations

import copy
from types import ModuleType
from typing import Any

from everbench import artifacts
from everbench.metrics import MetricTracker


def prediction_for(task: ModuleType, model: Any, features: dict[str, float], event_id: str | None = None) -> Any:
    """Apply a task's prediction semantics to a model."""
    if getattr(model, "uses_event_context", False):
        if event_id is None:
            raise ValueError("this model requires an event ID to make a prediction")
        return model.predict_event(event_id, features)
    hook = getattr(task, "predict_for", None)
    if hook is not None:
        return hook(model, features)
    if task.PROBLEM_TYPE == "binary_classification":
        try:
            probabilities = model.predict_proba_one(features)
        except AttributeError:
            return model.predict_one(features)
        return probabilities.get(True, probabilities.get(1, probabilities.get("true", 0.0)))
    if task.PROBLEM_TYPE == "anomaly_detection":
        try:
            return model.score_one(features)
        except AttributeError:
            return model.predict_one(features)
    return model.predict_one(features)


def metric_inputs_for(task: ModuleType, metric: Any, y_true: Any, prediction: Any) -> tuple[Any, Any]:
    """Let a task adapt a stored prediction to a metric's expected input."""
    hook = getattr(task, "metric_inputs_for", None)
    return hook(metric, y_true, prediction) if hook is not None else (y_true, prediction)


class PickledModel:
    """Adapter for trusted online or scoring-only predictor pickles."""

    def __init__(self, model_id: str, model: Any):
        self.uses_event_context = callable(getattr(model, "predict_event", None))
        if not self.uses_event_context and not (
            callable(getattr(model, "predict_one", None)) or callable(getattr(model, "predict_proba_one", None))
        ):
            raise TypeError(
                "pickled models must provide predict_one(features), predict_proba_one(features), or predict_event(event_id, features)"
            )
        self.model_id = model_id
        self.model = model

    @property
    def supports_learning(self) -> bool:
        return callable(getattr(self.model, "learn_one", None))

    def predict_one(self, features: dict[str, float]) -> Any:
        return self.model.predict_one(features)

    def predict_proba_one(self, features: dict[str, float]) -> Any:
        predictor = getattr(self.model, "predict_proba_one", None)
        if predictor is None:
            raise AttributeError("underlying model does not provide predict_proba_one")
        return predictor(features)

    def score_one(self, features: dict[str, float]) -> Any:
        scorer = getattr(self.model, "score_one", None)
        if scorer is None:
            raise AttributeError("underlying model does not provide score_one")
        return scorer(features)

    def predict_event(self, event_id: str, features: dict[str, float]) -> Any:
        predictor = getattr(self.model, "predict_event", None)
        if predictor is None:
            raise AttributeError("underlying model does not provide predict_event")
        return predictor(event_id, features)

    def learn_one(self, features: dict[str, float], y: Any) -> None:
        learner = getattr(self.model, "learn_one", None)
        if learner is None:
            raise AttributeError("underlying model does not provide learn_one")
        learner(features, y)

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
    examples: list[tuple[str, dict[str, float], object]],
) -> int:
    """Check a signed upload against available labelled examples without mutating it.

    This deliberately lives alongside the model protocol: the validation is a
    generic protocol check, not a separate model subsystem.
    """
    candidate = PickledModel("validation", copy.deepcopy(artifacts.loads(payload, signature)))
    if not examples:
        return 0
    tracker = MetricTracker.fresh(task.PROBLEM_TYPE, task.METRICS)
    for event_id, features, y in examples[-5:]:
        prediction = prediction_for(task, candidate, features, event_id)
        tracker.update(y, prediction, lambda metric, target, value: metric_inputs_for(task, metric, target, value))
        if supports_learning(candidate):
            candidate.learn_one(features, y)
    tracker.values()
    return min(len(examples), 5)
