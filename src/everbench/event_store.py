"""Database operations for incoming events, labels, and per-model event state."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import func, select, text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from everbench.db import advisory_key
from everbench.schema import BenchmarkEvent, BenchmarkLabel, ModelEventState, StreamCursor


@dataclass(frozen=True)
class LabelInput:
    event_id: str
    y: Any
    reason: str
    available_at: datetime = field(default_factory=lambda: datetime.now(UTC))


def lock_task_ingest(session: Session, task_name: str) -> None:
    """Serialize a task's event, label, and horizon-finalizer writes."""
    session.execute(
        text("SELECT pg_advisory_xact_lock(:lock_key)"),
        {"lock_key": advisory_key("ingest", task_name)},
    )


def add_events(
    session: Session,
    task_name: str,
    events: list[tuple[str, float, dict[str, Any]]],
    delay_seconds: float | None = None,
) -> list[str]:
    """Insert a collector batch and return the event IDs that became durable."""
    if not events:
        return []
    values = [
        {
            "task_name": task_name,
            "event_id": event_id,
            "event_time": datetime.fromtimestamp(timestamp, UTC),
            "event": event,
        }
        for event_id, timestamp, event in events
    ]
    statement = (
        insert(BenchmarkEvent)
        .values(values)
        .on_conflict_do_nothing(index_elements=["task_name", "event_id"])
        .returning(BenchmarkEvent.event_id)
    )
    inserted_event_ids = list(session.scalars(statement).all())
    # A label can arrive before its event batch. Once the event is available,
    # apply the configured positive-label horizon to that pending label too.
    if delay_seconds is not None:
        session.execute(
            text(
                """DELETE FROM benchmark_labels AS label
                   USING benchmark_events AS event
                   WHERE label.task_name = :task_name
                     AND label.event_id = ANY(CAST(:event_ids AS text[]))
                     AND event.task_name = label.task_name AND event.event_id = label.event_id
                     AND label.y = '1'::jsonb
                     AND label.available_at
                         > event.event_time + make_interval(secs => :delay_seconds)"""
            ),
            {
                "task_name": task_name,
                "event_ids": inserted_event_ids,
                "delay_seconds": delay_seconds,
            },
        )
    return inserted_event_ids


def add_expired_negative_labels(session: Session, task_name: str, delay_seconds: float | None) -> list[str]:
    if delay_seconds is None:
        return []
    return list(
        session.scalars(
            text(
                """INSERT INTO benchmark_labels (task_name, event_id, y, reason)
               SELECT event.task_name, event.event_id, to_jsonb(0), 'not-positive-within-horizon'
               FROM benchmark_events AS event
               LEFT JOIN benchmark_labels AS label USING (task_name, event_id)
               WHERE event.task_name = :task_name
                 AND event.event_time <= now() - make_interval(secs => :delay_seconds)
                 AND label.event_id IS NULL
               ON CONFLICT (task_name, event_id) DO NOTHING
               RETURNING event_id"""
            ),
            {"task_name": task_name, "delay_seconds": delay_seconds},
        ).all()
    )


def purge_orphan_labels(session: Session, task_name: str, cutoff: datetime) -> int:
    """Bound the label inbox when a corresponding accepted event never arrives."""
    result = session.execute(
        text(
            """DELETE FROM benchmark_labels AS label
                 WHERE label.task_name = :task_name AND label.inserted_at < :cutoff
                   AND NOT EXISTS (
                     SELECT 1 FROM benchmark_events AS event
                      WHERE event.task_name = label.task_name AND event.event_id = label.event_id
                   )
               RETURNING label.event_id"""
        ),
        {"task_name": task_name, "cutoff": cutoff},
    )
    return len(list(result.scalars()))


def add_labels(session: Session, task_name: str, labels: list[LabelInput], delay_seconds: float | None) -> list[str]:
    """Insert a label-collector batch with one Postgres statement.

    A positive label arriving after its configured horizon is ignored: the
    delayed-negative finalizer is the authority once that horizon has passed.
    """
    if not labels:
        return []
    if delay_seconds is not None:
        event_times = {
            event_id: event_time
            for event_id, event_time in session.execute(
                select(BenchmarkEvent.event_id, BenchmarkEvent.event_time).where(
                    BenchmarkEvent.task_name == task_name,
                    BenchmarkEvent.event_id.in_([label.event_id for label in labels]),
                )
            )
            if event_id is not None and event_time is not None
        }
        labels = [
            label
            for label in labels
            if label.event_id not in event_times
            or label.y != 1
            or label.available_at <= event_times[label.event_id] + timedelta(seconds=delay_seconds)
        ]
        if not labels:
            return []
    values = [
        {
            "task_name": task_name,
            "event_id": label.event_id,
            "y": label.y,
            "reason": label.reason,
            "available_at": label.available_at,
        }
        for label in labels
    ]
    statement = (
        insert(BenchmarkLabel)
        .values(values)
        .on_conflict_do_nothing(index_elements=["task_name", "event_id"])
        .returning(BenchmarkLabel.event_id)
    )
    return list(session.scalars(statement).all())


def stream_cursor(session: Session, task_name: str, stream_name: str) -> str | None:
    row = session.get(StreamCursor, {"task_name": task_name, "stream_name": stream_name})
    return row.event_id if row else None


def save_stream_cursor(session: Session, task_name: str, stream_name: str, event_id: str) -> None:
    session.execute(
        insert(StreamCursor)
        .values(task_name=task_name, stream_name=stream_name, event_id=event_id)
        .on_conflict_do_update(
            index_elements=["task_name", "stream_name"], set_={"event_id": event_id, "updated_at": func.now()}
        )
    )


def untrained_labels(
    session: Session, task_name: str, model_id: str, limit: int = 500
) -> list[tuple[str, Any, datetime, int]]:
    rows = session.execute(
        text(
            """SELECT label.event_id, label.y, label.available_at, event.sequence
               FROM benchmark_labels AS label
               JOIN benchmark_events AS event USING (task_name, event_id)
               JOIN benchmark_model_events AS model_event
                 ON model_event.task_name = label.task_name AND model_event.event_id = label.event_id
                 AND model_event.model_id = :model_id
               WHERE label.task_name = :task_name
                 AND model_event.trained_at IS NULL
               ORDER BY label.available_at, event.sequence LIMIT :limit"""
        ),
        {"task_name": task_name, "model_id": model_id, "limit": limit},
    )
    return [(event_id, y, available_at, sequence) for event_id, y, available_at, sequence in rows]


def unevaluated_labels(session: Session, task_name: str, model_id: str, limit: int = 500) -> list[tuple[str, Any, Any]]:
    """Return labelled predictions not yet incorporated into River metrics."""
    rows = session.execute(
        text(
            """SELECT label.event_id, label.y, model_event.prediction
               FROM benchmark_labels AS label
               JOIN benchmark_events AS event USING (task_name, event_id)
               JOIN benchmark_model_events AS model_event USING (task_name, event_id)
               WHERE label.task_name = :task_name AND model_event.model_id = :model_id
                 AND model_event.prediction_status = 'predicted'
                 AND model_event.evaluated_at IS NULL
               ORDER BY label.available_at, event.sequence LIMIT :limit"""
        ),
        {"task_name": task_name, "model_id": model_id, "limit": limit},
    )
    return [(event_id, y, prediction) for event_id, y, prediction in rows]


def add_trainings(session: Session, task_name: str, model_id: str, event_ids: list[str]) -> list[str]:
    if not event_ids:
        return []
    return list(
        session.scalars(
            text(
                """UPDATE benchmark_model_events
                     SET trained_at = now()
                   WHERE task_name = :task_name AND model_id = :model_id
                     AND event_id = ANY(CAST(:event_ids AS text[]))
                     AND trained_at IS NULL
                   RETURNING event_id"""
            ),
            {"task_name": task_name, "model_id": model_id, "event_ids": event_ids},
        )
    )


def add_metric_updates(session: Session, task_name: str, model_id: str, event_ids: list[str]) -> None:
    if event_ids:
        session.execute(
            text(
                """UPDATE benchmark_model_events
                     SET evaluated_at = now()
                   WHERE task_name = :task_name AND model_id = :model_id
                     AND event_id = ANY(CAST(:event_ids AS text[]))
                     AND prediction_status = 'predicted' AND evaluated_at IS NULL"""
            ),
            {"task_name": task_name, "model_id": model_id, "event_ids": event_ids},
        )


def _model_processing_pending_clause() -> str:
    """SQL predicate for work that prevents a labelled event from archiving."""
    return """(
        NOT EXISTS (
          SELECT 1 FROM benchmark_model_events AS model_event
          WHERE model_event.task_name = event.task_name AND model_event.event_id = event.event_id
            AND model_event.model_id = model.model_id
            AND model_event.trained_at IS NOT NULL
            AND (model_event.prediction_status = 'skipped' OR model_event.evaluated_at IS NOT NULL)
        )
        OR NOT EXISTS (
          SELECT 1 FROM model_snapshots AS snapshot
          WHERE snapshot.task_name = model.task_name AND snapshot.model_id = model.model_id
        )
        OR EXISTS (
          SELECT 1 FROM model_snapshots AS snapshot
          WHERE snapshot.task_name = model.task_name AND snapshot.model_id = model.model_id
            AND (
              snapshot.checkpoint_label_available_at IS NULL
              OR snapshot.checkpoint_label_available_at < label.available_at
              OR (snapshot.checkpoint_label_available_at = label.available_at
                  AND snapshot.checkpoint_event_sequence < event.sequence)
            )
        )
    )"""


def completed_labelled_events(session: Session, task_name: str, event_ids: list[str]) -> list[str]:
    """Find labelled events no eligible active model still needs to process."""
    if not event_ids:
        return []
    # Checkpoints are updated through the ORM, while this eligibility query is
    # deliberately raw SQL. Flush first so a just-completed learning batch can
    # release its hot-cache entries in the same transaction.
    session.flush()
    rows = session.execute(
        text(
            f"""SELECT event.event_id
                 FROM benchmark_events AS event
                 JOIN benchmark_labels AS label USING (task_name, event_id)
                 WHERE event.task_name = :task_name
                   AND event.event_id = ANY(CAST(:event_ids AS text[]))
                   AND NOT EXISTS (
                     SELECT 1 FROM benchmark_models AS model
                     WHERE model.task_name = event.task_name AND model.active
                       AND event.sequence >= model.start_sequence
                       AND {_model_processing_pending_clause()}
                   )"""
        ),
        {"task_name": task_name, "event_ids": event_ids},
    )
    return [event_id for (event_id,) in rows]


def unpredicted_events(
    session: Session, task_name: str, model_id: str, start_sequence: int, limit: int = 500
) -> list[str]:
    rows = session.execute(
        text(
            """SELECT event.event_id
               FROM benchmark_events AS event
               LEFT JOIN benchmark_model_events AS model_event
                 ON model_event.task_name = event.task_name AND model_event.event_id = event.event_id
                 AND model_event.model_id = :model_id
               LEFT JOIN benchmark_labels AS label
                 ON label.task_name = event.task_name AND label.event_id = event.event_id
               WHERE event.task_name = :task_name AND event.sequence >= :start_sequence
                 AND model_event.event_id IS NULL
                 AND label.event_id IS NULL
               ORDER BY event.sequence LIMIT :limit"""
        ),
        {"task_name": task_name, "model_id": model_id, "start_sequence": start_sequence, "limit": limit},
    )
    return [event_id for (event_id,) in rows]


def event_payloads(session: Session, task_name: str, event_ids: list[str]) -> dict[str, dict[str, Any]]:
    """Bulk database fallback for raw event cache misses."""
    if not event_ids:
        return {}
    rows = session.execute(
        select(BenchmarkEvent.event_id, BenchmarkEvent.event).where(
            BenchmarkEvent.task_name == task_name,
            BenchmarkEvent.event_id.in_(event_ids),
        )
    )
    return {event_id: event for event_id, event in rows}


def latest_labelled_examples(session: Session, task_name: str, limit: int = 5) -> list[tuple[str, dict[str, Any], Any]]:
    """Read recent labelled events before archives are available for a task."""
    rows = session.execute(
        text(
            """SELECT event.event_id, event.event, label.y
               FROM benchmark_labels AS label
               JOIN benchmark_events AS event USING (task_name, event_id)
               WHERE label.task_name = :task_name
               ORDER BY label.available_at DESC, event.sequence DESC LIMIT :limit"""
        ),
        {"task_name": task_name, "limit": limit},
    )
    return list(reversed([(event_id, event, y) for event_id, event, y in rows]))


def labelled_unpredicted_events(
    session: Session, task_name: str, model_id: str, start_sequence: int, limit: int = 500
) -> list[str]:
    """Events that became labelled before this model persisted a prediction."""
    rows = session.execute(
        text(
            """SELECT event.event_id
               FROM benchmark_events AS event
               JOIN benchmark_labels AS label USING (task_name, event_id)
               LEFT JOIN benchmark_model_events AS model_event
                 ON model_event.task_name = event.task_name AND model_event.event_id = event.event_id
                 AND model_event.model_id = :model_id
               WHERE event.task_name = :task_name AND event.sequence >= :start_sequence
                 AND model_event.event_id IS NULL
               ORDER BY label.available_at, event.sequence LIMIT :limit"""
        ),
        {"task_name": task_name, "model_id": model_id, "start_sequence": start_sequence, "limit": limit},
    )
    return [event_id for (event_id,) in rows]


def add_prediction_skips(
    session: Session,
    task_name: str,
    model_id: str,
    event_ids: list[str],
    reason: str = "label-available-before-prediction",
) -> list[str]:
    if event_ids:
        return list(
            session.scalars(
                insert(ModelEventState)
                .values(
                    [
                        {
                            "task_name": task_name,
                            "event_id": event_id,
                            "model_id": model_id,
                            "prediction_status": "skipped",
                            "prediction_reason": reason,
                        }
                        for event_id in event_ids
                    ]
                )
                .on_conflict_do_nothing()
                .returning(ModelEventState.event_id)
            )
        )
    return []


def add_predictions(session: Session, task_name: str, model_id: str, predictions: list[tuple[str, Any]]) -> list[str]:
    """Persist only predictions whose label is still unavailable.

    The final label check happens in the INSERT statement, closing the race
    between selecting candidate events and writing their predictions.
    """
    if not predictions:
        return []
    rows = json.dumps([{"event_id": event_id, "prediction": prediction} for event_id, prediction in predictions])
    inserted = session.execute(
        text(
            """INSERT INTO benchmark_model_events
                 (task_name, event_id, model_id, prediction, prediction_status)
               SELECT :task_name, incoming.event_id, :model_id, incoming.prediction, 'predicted'
               FROM jsonb_to_recordset(CAST(:rows AS jsonb)) AS incoming(event_id text, prediction jsonb)
               LEFT JOIN benchmark_labels AS label
                 ON label.task_name = :task_name AND label.event_id = incoming.event_id
               WHERE label.event_id IS NULL
               ON CONFLICT (task_name, event_id, model_id) DO NOTHING
               RETURNING event_id"""
        ),
        {"task_name": task_name, "model_id": model_id, "rows": rows},
    )
    return list(inserted.scalars())
