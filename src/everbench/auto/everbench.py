"""Thin adapter between portable AutoClassifier and Everbench's model protocol."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from river import base
from sqlalchemy import select
from sqlalchemy.orm import Session

from everbench.auto.classifier import AutoClassifier
from everbench.auto.evaluation import TemporalObservation
from everbench.schema import BenchmarkEvent, BenchmarkLabel
from everbench.tasks import TaskDefinition


class EverbenchAutoClassifier:
    """Expose an AutoClassifier through Everbench's event-ID-aware protocol."""

    def __init__(self, auto_classifier: AutoClassifier) -> None:
        self.auto_classifier = auto_classifier

    def predict_one(self, event_id: str, event: dict[str, Any]) -> Any:
        del event_id
        return self.auto_classifier.predict_one(event)

    def predict_proba_one(self, event_id: str, event: dict[str, Any]) -> dict[base.typing.ClfTarget, float]:
        del event_id
        return self.auto_classifier.predict_proba_one(event)

    def learn_one(self, event_id: str, event: dict[str, Any], label: Any) -> None:
        del event_id
        self.auto_classifier.learn_one(event, label)


def complete_observations(
    session: Session,
    task: TaskDefinition,
    limit: int,
    maturity_margin_seconds: float = 0.0,
    *,
    now: datetime | None = None,
) -> tuple[TemporalObservation, ...]:
    """Load a recent event-time cohort whose labels have fully matured.

    For delayed-negative tasks, filtering on event time is essential. Merely
    selecting rows that already have labels would over-sample fast positives.
    """
    statement = (
        select(
            BenchmarkEvent.event_id,
            BenchmarkEvent.sequence,
            BenchmarkEvent.event,
            BenchmarkEvent.inserted_at,
            BenchmarkLabel.y,
            BenchmarkLabel.available_at,
        )
        .join(
            BenchmarkLabel,
            (BenchmarkLabel.task_name == BenchmarkEvent.task_name)
            & (BenchmarkLabel.event_id == BenchmarkEvent.event_id),
        )
        .where(BenchmarkEvent.task_name == task.TASK_NAME)
    )
    if task.NEGATIVE_LABEL_DELAY_SECONDS is not None:
        cutoff = (now or datetime.now(UTC)) - timedelta(
            seconds=task.NEGATIVE_LABEL_DELAY_SECONDS + maturity_margin_seconds
        )
        statement = statement.where(BenchmarkEvent.event_time <= cutoff)
    rows = session.execute(
        statement.order_by(BenchmarkEvent.inserted_at.desc(), BenchmarkEvent.sequence.desc()).limit(limit)
    ).all()
    observations = []
    for event_id, sequence, event, event_available_at, y, label_available_at in reversed(rows):
        # Labels can enter the inbox before their event. Learning cannot happen
        # until both are available, so clamp to the event's availability time.
        effective_label_at = max(label_available_at, event_available_at)
        if not isinstance(event, dict):
            raise TypeError(f"event {event_id!r} is not a mapping")
        observations.append(
            TemporalObservation(
                observation_id=event_id,
                sequence=sequence,
                x=event,
                y=y,
                available_at=event_available_at,
                label_available_at=effective_label_at,
            )
        )
    return tuple(observations)
