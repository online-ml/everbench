"""River-backed delayed progressive validation for online classifiers."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from time import perf_counter
from typing import Any, TypeAlias

from river import base, evaluate, metrics

from everbench.auto.research import ConstraintResult, Evaluation, Objective

ArchiveExample: TypeAlias = tuple[
    str,
    int,
    dict[Any, Any],
    base.typing.ClfTarget,
    datetime,
    datetime,
]


@dataclass(frozen=True)
class EvaluationOutcome:
    """Comparison evidence plus the candidate state produced by causal replay."""

    evaluation: Evaluation
    trained_candidate: base.Classifier
    checkpoint_label_available_at: datetime
    checkpoint_event_sequence: int
    candidate_predict_seconds: float
    champion_predict_seconds: float


class _TimedClassifier(base.Classifier):
    """Measure prediction calls while hiding evaluator metadata from a model."""

    def __init__(self, model: base.Classifier) -> None:
        self.model = model
        self.predict_seconds = 0.0

    @property
    def _multiclass(self) -> bool:
        return self.model._multiclass

    def predict_one(self, x: dict[Any, Any], **kwargs: Any) -> base.typing.ClfTarget | None:
        started = perf_counter()
        prediction = self.model.predict_one(x, **kwargs)
        self.predict_seconds += perf_counter() - started
        return prediction

    def predict_proba_one(self, x: dict[Any, Any], **kwargs: Any) -> dict[base.typing.ClfTarget, float]:
        started = perf_counter()
        prediction = self.model.predict_proba_one(x, **kwargs)
        self.predict_seconds += perf_counter() - started
        return prediction

    def learn_one(self, x: dict[Any, Any], y: base.typing.ClfTarget, **kwargs: Any) -> None:
        self.model.learn_one(x, y, **kwargs)


def _river_dataset(
    observations: Sequence[ArchiveExample],
    timing: list[tuple[datetime, timedelta] | None],
) -> Iterator[tuple[dict[Any, Any], base.typing.ClfTarget]]:
    """Adapt archive rows to River without exposing timestamps as features."""
    previous: tuple[datetime, int] | None = None
    for _, sequence, features, target, available_at, label_available_at in observations:
        if label_available_at < available_at:
            raise ValueError("a label is available before its observation")
        position = (available_at, sequence)
        if previous is not None and position < previous:
            raise ValueError("observations must be ordered by availability time and sequence")
        previous = position
        timing[0] = (available_at, label_available_at - available_at)
        yield dict(features), target


def _evaluate_model(
    model: base.Classifier,
    observations: Sequence[ArchiveExample],
    metric_definitions: tuple[metrics.base.ClassificationMetric, ...],
) -> tuple[_TimedClassifier, list[metrics.base.ClassificationMetric]]:
    timed = _TimedClassifier(model.clone())
    metric = metrics.base.Metrics([definition.clone() for definition in metric_definitions])
    timing: list[tuple[datetime, timedelta] | None] = [None]

    def moment(features: dict[Any, Any]) -> datetime:
        del features
        assert timing[0] is not None
        return timing[0][0]

    def delay(features: dict[Any, Any], target: Any) -> timedelta:
        del features, target
        assert timing[0] is not None
        return timing[0][1]

    evaluate.progressive_val_score(
        dataset=_river_dataset(observations, timing),
        model=timed,
        metric=metric,
        moment=moment,
        delay=delay,
    )
    return timed, list(metric)


def progressive_validate(
    champion: base.Classifier,
    candidate: base.Classifier,
    observations: Sequence[ArchiveExample],
    objective: Objective,
    *,
    max_prediction_time_ratio: float | None = None,
) -> EvaluationOutcome:
    """Compare fresh definitions with River's delayed progressive validation.

    Both models see the same complete archive in event-availability order.
    River predicts when each event arrived, reveals its label after the recorded
    delay, updates every metric, and only then learns from that label.
    """
    if not observations:
        raise ValueError("evaluation contains no observations")

    metric_definitions = (objective.metric,) + tuple(constraint.metric for constraint in objective.metric_constraints)
    timed_champion, champion_metrics = _evaluate_model(champion, observations, metric_definitions)
    timed_candidate, candidate_metrics = _evaluate_model(candidate, observations, metric_definitions)

    constraints = tuple(
        constraint.evaluate(champion_metric.get(), candidate_metric.get())
        for constraint, champion_metric, candidate_metric in zip(
            objective.metric_constraints,
            champion_metrics[1:],
            candidate_metrics[1:],
            strict=True,
        )
    )
    if max_prediction_time_ratio is not None:
        ratio = timed_candidate.predict_seconds / max(timed_champion.predict_seconds, 1e-12)
        constraints += (
            ConstraintResult(
                name="prediction_time_ratio",
                passed=ratio <= max_prediction_time_ratio,
                detail=f"{ratio:.3f}x <= {max_prediction_time_ratio:.3f}x champion time",
            ),
        )

    checkpoint = max(observations, key=lambda row: (row[5], row[1]))
    return EvaluationOutcome(
        evaluation=Evaluation(
            champion_score=champion_metrics[0].get(),
            candidate_score=candidate_metrics[0].get(),
            observations=len(observations),
            constraints=constraints,
        ),
        trained_candidate=timed_candidate.model,
        checkpoint_label_available_at=checkpoint[5],
        checkpoint_event_sequence=checkpoint[1],
        candidate_predict_seconds=timed_candidate.predict_seconds,
        champion_predict_seconds=timed_champion.predict_seconds,
    )
