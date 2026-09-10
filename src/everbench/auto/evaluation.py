"""Causal temporal evaluation for online classification candidates."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import datetime
from time import perf_counter
from typing import Any

from river import base

from everbench.auto.research import ConstraintResult, Evaluation, Objective


@dataclass(frozen=True)
class TemporalObservation:
    """An observation with separate prediction and label availability times."""

    observation_id: str
    sequence: int
    x: dict[Any, Any]
    y: base.typing.ClfTarget
    available_at: datetime
    label_available_at: datetime

    def __post_init__(self) -> None:
        if self.label_available_at < self.available_at:
            raise ValueError(f"label for {self.observation_id!r} is available before its observation")


@dataclass(frozen=True)
class TemporalSplit:
    """Agent-visible research observations and evaluator-only promotion evidence."""

    research: tuple[TemporalObservation, ...]
    promotion: tuple[TemporalObservation, ...]


@dataclass(frozen=True)
class EvaluationOutcome:
    """Sealed evidence plus the candidate state produced by causal replay."""

    evaluation: Evaluation
    trained_candidate: base.Classifier
    checkpoint_label_available_at: datetime
    checkpoint_event_sequence: int
    candidate_predict_seconds: float
    champion_predict_seconds: float


def temporal_split(
    observations: tuple[TemporalObservation, ...],
    promotion_observations: int,
    min_research_observations: int = 1,
) -> TemporalSplit:
    """Reserve the latest event cohort as sealed promotion evidence."""
    if promotion_observations <= 0:
        raise ValueError("promotion_observations must be positive")
    if min_research_observations <= 0:
        raise ValueError("min_research_observations must be positive")
    ordered = tuple(sorted(observations, key=lambda row: (row.available_at, row.sequence)))
    required = promotion_observations + min_research_observations
    if len(ordered) < required:
        raise ValueError(f"evaluation needs at least {required} complete observations; received {len(ordered)}")
    split_at = len(ordered) - promotion_observations
    return TemporalSplit(research=ordered[:split_at], promotion=ordered[split_at:])


def _predictions(
    model: base.Classifier,
    metrics: tuple[Any, ...],
    x: dict[Any, Any],
) -> list[Any]:
    hard_prediction = model.predict_one(x) if any(metric.requires_labels for metric in metrics) else None
    probabilities = model.predict_proba_one(x) if any(not metric.requires_labels for metric in metrics) else None
    return [hard_prediction if metric.requires_labels else probabilities for metric in metrics]


def evaluate_temporally(
    champion: base.Classifier,
    candidate: base.Classifier,
    split: TemporalSplit,
    objective: Objective,
    *,
    max_prediction_time_ratio: float | None = None,
) -> EvaluationOutcome:
    """Replay prediction and delayed learning without exposing sealed rows.

    Both definitions start fresh, see identical observations, and receive each
    label only at its historical availability time. Metrics are updated only for
    the sealed promotion cohort, while the research prefix acts as warm-up.
    """
    promotion_ids = {row.observation_id for row in split.promotion}
    rows = split.research + split.promotion
    ids = [row.observation_id for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("observation IDs must be unique")

    events = iter(sorted(rows, key=lambda row: (row.available_at, row.sequence)))
    labels = iter(sorted(rows, key=lambda row: (row.label_available_at, row.sequence)))
    return evaluate_temporal_actions(
        champion,
        candidate,
        _merge_temporal_actions(events, labels, promotion_ids),
        objective,
        promotion_observations=len(split.promotion),
        max_prediction_time_ratio=max_prediction_time_ratio,
    )


def _next_or_none(rows: Iterator[TemporalObservation]) -> TemporalObservation | None:
    return next(rows, None)


def _merge_temporal_actions(
    events: Iterable[TemporalObservation],
    labels: Iterable[TemporalObservation],
    promotion_ids: set[str],
) -> Iterable[tuple[int, TemporalObservation, bool]]:
    """Merge event- and label-ordered rows without constructing a 2N timeline."""
    event_iterator = iter(events)
    label_iterator = iter(labels)
    event = _next_or_none(event_iterator)
    label = _next_or_none(label_iterator)
    while event is not None or label is not None:
        event_key = (event.available_at, 0, event.sequence) if event is not None else None
        label_key = (label.label_available_at, 1, label.sequence) if label is not None else None
        if label_key is None or (event_key is not None and event_key < label_key):
            assert event is not None
            yield 0, event, event.observation_id in promotion_ids
            event = _next_or_none(event_iterator)
        else:
            assert label is not None
            yield 1, label, label.observation_id in promotion_ids
            label = _next_or_none(label_iterator)


def evaluate_temporal_actions(
    champion: base.Classifier,
    candidate: base.Classifier,
    actions: Iterable[tuple[int, TemporalObservation, bool]],
    objective: Objective,
    *,
    promotion_observations: int,
    max_prediction_time_ratio: float | None = None,
) -> EvaluationOutcome:
    """Evaluate an already causal action stream with bounded prediction state."""
    champion = champion.clone()
    candidate = candidate.clone()

    metric_definitions = (objective.metric,) + tuple(constraint.metric for constraint in objective.metric_constraints)
    champion_metrics = [metric.clone() for metric in metric_definitions]
    candidate_metrics = [metric.clone() for metric in metric_definitions]
    predictions: dict[str, tuple[list[Any], list[Any]]] = {}
    candidate_predict_seconds = 0.0
    champion_predict_seconds = 0.0
    last_label: tuple[datetime, int] | None = None
    for action, row, is_promotion in actions:
        if action == 0:
            started = perf_counter()
            champion_predictions = _predictions(champion, metric_definitions, row.x)
            champion_predict_seconds += perf_counter() - started
            started = perf_counter()
            candidate_predictions = _predictions(candidate, metric_definitions, row.x)
            candidate_predict_seconds += perf_counter() - started
            if is_promotion:
                predictions[row.observation_id] = (champion_predictions, candidate_predictions)
            continue

        if is_promotion:
            champion_predictions, candidate_predictions = predictions.pop(row.observation_id)
            for metric, prediction in zip(champion_metrics, champion_predictions, strict=True):
                metric.update(row.y, prediction)
            for metric, prediction in zip(candidate_metrics, candidate_predictions, strict=True):
                metric.update(row.y, prediction)
        champion.learn_one(row.x, row.y)
        candidate.learn_one(row.x, row.y)
        last_label = (row.label_available_at, row.sequence)

    if last_label is None:
        raise ValueError("evaluation contains no labels")
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
        ratio = candidate_predict_seconds / max(champion_predict_seconds, 1e-12)
        constraints += (
            ConstraintResult(
                name="prediction_time_ratio",
                passed=ratio <= max_prediction_time_ratio,
                detail=f"{ratio:.3f}x <= {max_prediction_time_ratio:.3f}x champion time",
            ),
        )
    return EvaluationOutcome(
        evaluation=Evaluation(
            champion_score=champion_metrics[0].get(),
            candidate_score=candidate_metrics[0].get(),
            observations=promotion_observations,
            constraints=constraints,
        ),
        trained_candidate=candidate,
        checkpoint_label_available_at=last_label[0],
        checkpoint_event_sequence=last_label[1],
        candidate_predict_seconds=candidate_predict_seconds,
        champion_predict_seconds=champion_predict_seconds,
    )
