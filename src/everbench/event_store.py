"""Database operations for incoming events, labels, and per-model event state."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import func, select, text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from everbench.db import advisory_key
from everbench.records import LabelInput, LabelledExample, Observation, PendingPrediction, ReadyObservation
from everbench.schema import (
    BenchmarkEvent,
    BenchmarkLabel,
    LabelSchedule,
    ModelEventState,
    StreamCursor,
)
from everbench.tasks import LabelPolicy


def lock_task_ingest(*, session: Session, task_name: str) -> None:
    """Serialize a task's event, label, and horizon-finalizer writes."""
    session.execute(
        text("SELECT pg_advisory_xact_lock(:lock_key)"),
        {"lock_key": advisory_key(parts=("ingest", task_name))},
    )


def add_events(
    *, session: Session, task_name: str, events: list[Observation], policy: LabelPolicy | None = None
) -> list[str]:
    """Persist observations and resolve earlier forecasts in the same transaction."""
    if not events:
        return []
    statement = (
        insert(BenchmarkEvent)
        .values(
            [
                {
                    "task_name": task_name,
                    "event_id": event.event_id,
                    "event_time": datetime.fromtimestamp(event.timestamp, UTC),
                    "event": event.payload,
                }
                for event in events
            ]
        )
        .on_conflict_do_nothing(index_elements=["task_name", "event_id"])
        .returning(BenchmarkEvent.event_id)
    )
    inserted = list(session.scalars(statement))
    inserted_set = set(inserted)
    accepted = [event for event in events if event.event_id in inserted_set]
    if policy is not None and accepted:
        # An inbox label may precede the observation. Apply its source-time
        # horizon only to newly inserted observations, never to existing defaults.
        session.execute(
            text(
                """DELETE FROM benchmark_labels AS label USING benchmark_events AS event
               WHERE event.task_name = :task AND label.task_name = event.task_name
                 AND label.event_id = event.event_id AND event.event_id = ANY(:ids)
                 AND label.available_at > event.event_time + make_interval(secs => :horizon)"""
            ),
            {"task": task_name, "ids": inserted, "horizon": policy.delay_seconds + policy.tolerance_seconds},
        )
        session.execute(
            insert(LabelSchedule)
            .values(
                [
                    {
                        "task_name": task_name,
                        "event_id": event.event_id,
                        "entity_key": event.entity_key,
                        "target_at": datetime.fromtimestamp(event.timestamp + policy.delay_seconds, UTC),
                        "due_at": datetime.fromtimestamp(event.timestamp + policy.close_after_seconds, UTC),
                    }
                    for event in accepted
                ]
            )
            .on_conflict_do_nothing()
        )
        match_forecasts(session=session, task_name=task_name, observations=accepted, policy=policy)
    _mark_ready_labels(session=session, task_name=task_name, event_ids=inserted)
    return inserted


def _mark_ready_labels(*, session: Session, task_name: str, event_ids: list[str]) -> list[str]:
    """Queue a resolution only once both its observation and outcome exist."""
    if not event_ids:
        return []
    session.execute(
        text(
            """DELETE FROM benchmark_label_schedule AS schedule USING benchmark_labels AS label
           WHERE schedule.task_name = :task AND label.task_name = schedule.task_name
             AND label.event_id = schedule.event_id AND schedule.event_id = ANY(:ids)"""
        ),
        {"task": task_name, "ids": event_ids},
    )
    return list(
        session.scalars(
            text(
                """INSERT INTO benchmark_ready_labels (task_name, event_id, has_target)
           SELECT event.task_name, event.event_id, label.y <> 'null'::jsonb
             FROM benchmark_events AS event JOIN benchmark_labels AS label USING (task_name, event_id)
            WHERE event.task_name = :task AND event.event_id = ANY(:ids)
           ON CONFLICT (task_name, event_id) DO NOTHING RETURNING event_id"""
            ),
            {"task": task_name, "ids": event_ids},
        )
    )


def match_forecasts(
    *, session: Session, task_name: str, observations: list[Observation], policy: LabelPolicy
) -> list[str]:
    """A station snapshot supplies targets for pending forecasts of that station.

    The entity/deadline index limits reads to the incoming snapshot's matching
    window. No history scan, in-memory station history, or second poller is needed.
    """
    snapshots = [event for event in observations if event.entity_key is not None]
    if not snapshots:
        return []
    by_entity: dict[str, list[Observation]] = {}
    for event in sorted(snapshots, key=lambda event: event.timestamp):
        assert event.entity_key is not None
        by_entity.setdefault(event.entity_key, []).append(event)
    earliest = min(event.timestamp for event in snapshots) - policy.tolerance_seconds
    latest = max(event.timestamp for event in snapshots)
    schedules = session.scalars(
        select(LabelSchedule).where(
            LabelSchedule.task_name == task_name,
            LabelSchedule.entity_key.in_(list(by_entity)),
            LabelSchedule.target_at >= datetime.fromtimestamp(earliest, UTC),
            LabelSchedule.target_at <= datetime.fromtimestamp(latest, UTC),
        )
    )
    labels = []
    for schedule in schedules:
        assert schedule.entity_key is not None
        for snapshot in by_entity[schedule.entity_key]:
            if 0 <= snapshot.timestamp - schedule.target_at.timestamp() <= policy.tolerance_seconds:
                labels.append(LabelInput(event_id=schedule.event_id, y=snapshot.value, reason="forecast-snapshot"))
                break
    return add_labels(session=session, task_name=task_name, labels=labels)


def resolve_due(*, session: Session, task_name: str, policy: LabelPolicy | None, limit: int = 500) -> list[str]:
    """Finalize a bounded batch of deadlines; None is an unavailable target."""
    if policy is None:
        return []
    due = list(
        session.scalars(
            select(LabelSchedule.event_id)
            .where(LabelSchedule.task_name == task_name, LabelSchedule.due_at <= func.now())
            .order_by(LabelSchedule.due_at)
            .limit(limit)
        )
    )
    reason = "target-unavailable" if policy.default_label is None else "deadline-default"
    return add_labels(
        session=session,
        task_name=task_name,
        labels=[LabelInput(event_id=identifier, y=policy.default_label, reason=reason) for identifier in due],
    )


def purge_orphan_labels(*, session: Session, task_name: str, cutoff: datetime) -> int:
    """Bound the label inbox when a corresponding accepted event never arrives."""
    result = session.execute(
        text(
            """DELETE FROM benchmark_labels AS label
           WHERE label.task_name = :task AND label.inserted_at < :cutoff
             AND NOT EXISTS (SELECT 1 FROM benchmark_events AS event
                             WHERE event.task_name = label.task_name AND event.event_id = label.event_id)
           RETURNING label.event_id"""
        ),
        {"task": task_name, "cutoff": cutoff},
    )
    return len(list(result.scalars()))


def add_labels(
    *, session: Session, task_name: str, labels: list[LabelInput], policy: LabelPolicy | None = None
) -> list[str]:
    """Persist first resolutions; explicit source labels obey the task horizon."""
    if not labels:
        return []
    if policy is not None:
        event_times = {
            identifier: timestamp
            for identifier, timestamp in session.execute(
                select(BenchmarkEvent.event_id, BenchmarkEvent.event_time).where(
                    BenchmarkEvent.task_name == task_name,
                    BenchmarkEvent.event_id.in_([label.event_id for label in labels]),
                )
            )
        }
        labels = [
            label
            for label in labels
            if label.event_id not in event_times
            or label.available_at
            <= event_times[label.event_id] + timedelta(seconds=policy.delay_seconds + policy.tolerance_seconds)
        ]
    if not labels:
        return []
    # A source batch can include retries for the same ID; first resolution wins.
    unique = {label.event_id: label for label in reversed(labels)}
    inserted = list(
        session.scalars(
            insert(BenchmarkLabel)
            .values(
                [
                    {
                        "task_name": task_name,
                        "event_id": label.event_id,
                        "y": label.y,
                        "reason": label.reason,
                        "available_at": label.available_at,
                    }
                    for label in unique.values()
                ]
            )
            .on_conflict_do_nothing(index_elements=["task_name", "event_id"])
            .returning(BenchmarkLabel.event_id)
        )
    )
    _mark_ready_labels(session=session, task_name=task_name, event_ids=inserted)
    return inserted


def stream_cursor(*, session: Session, task_name: str, stream_name: str) -> str | None:
    row = session.get(StreamCursor, {"task_name": task_name, "stream_name": stream_name})
    return row.event_id if row else None


def save_stream_cursor(*, session: Session, task_name: str, stream_name: str, event_id: str) -> None:
    session.execute(
        insert(StreamCursor)
        .values(task_name=task_name, stream_name=stream_name, event_id=event_id)
        .on_conflict_do_update(
            index_elements=["task_name", "stream_name"], set_={"event_id": event_id, "updated_at": func.now()}
        )
    )


def events_after_cursor(
    *, session: Session, task_name: str, model_id: str, cursor_sequence: int, start_sequence: int, limit: int = 500
) -> list[PendingPrediction]:
    """Read each accepted event once per model using the event sequence index."""
    rows = session.execute(
        text(
            """SELECT event.event_id, event.sequence,
                      label.event_id IS NOT NULL AS resolved,
                      state.event_id IS NOT NULL AS has_state
                 FROM benchmark_events AS event
                 LEFT JOIN benchmark_labels AS label USING (task_name, event_id)
                 LEFT JOIN benchmark_model_events AS state
                   ON state.task_name = event.task_name AND state.event_id = event.event_id
                  AND state.model_id = :model_id
                WHERE event.task_name = :task_name
                  AND event.sequence > :cursor_sequence
                  AND event.sequence >= :start_sequence
                ORDER BY event.sequence
                LIMIT :limit"""
        ),
        {
            "task_name": task_name,
            "model_id": model_id,
            "cursor_sequence": cursor_sequence,
            "start_sequence": start_sequence,
            "limit": limit,
        },
    )
    return [PendingPrediction(**row) for row in rows.mappings()]


def ready_labels_after_cursor(
    *, session: Session, task_name: str, model_id: str, cursor_sequence: int, start_sequence: int, limit: int = 500
) -> list[ReadyObservation]:
    """Read each matched label once per model from its compact append-only queue."""
    rows = session.execute(
        text(
            """SELECT ready.sequence, ready.event_id, label.y AS target, label.available_at, event.sequence AS event_sequence,
                      state.prediction, state.prediction_status,
                      state.evaluated_at IS NOT NULL AS evaluated, state.trained_at IS NOT NULL AS trained
                 FROM benchmark_ready_labels AS ready
                 JOIN benchmark_events AS event USING (task_name, event_id)
                 JOIN benchmark_labels AS label USING (task_name, event_id)
                 LEFT JOIN benchmark_model_events AS state
                   ON state.task_name = ready.task_name AND state.event_id = ready.event_id
                  AND state.model_id = :model_id
                WHERE ready.task_name = :task_name
                  AND ready.sequence > :cursor_sequence
                  AND event.sequence >= :start_sequence
                ORDER BY ready.sequence
                LIMIT :limit"""
        ),
        {
            "task_name": task_name,
            "model_id": model_id,
            "cursor_sequence": cursor_sequence,
            "start_sequence": start_sequence,
            "limit": limit,
        },
    )
    return [ReadyObservation(**row) for row in rows.mappings()]


def add_trainings(
    *, session: Session, task_name: str, model_id: str, event_ids: list[str], skipped: bool = False
) -> list[str]:
    if not event_ids:
        return []
    return list(
        session.scalars(
            text(
                """UPDATE benchmark_model_events
                     SET trained_at = now(), training_skipped = :skipped
                   WHERE task_name = :task_name AND model_id = :model_id
                     AND event_id = ANY(CAST(:event_ids AS text[]))
                     AND trained_at IS NULL
                   RETURNING event_id"""
            ),
            {"task_name": task_name, "model_id": model_id, "event_ids": event_ids, "skipped": skipped},
        )
    )


def add_metric_updates(*, session: Session, task_name: str, model_id: str, event_ids: list[str]) -> None:
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
          SELECT 1
            FROM model_snapshots AS snapshot
            JOIN benchmark_ready_labels AS ready
              ON ready.task_name = event.task_name AND ready.event_id = event.event_id
           WHERE snapshot.task_name = model.task_name AND snapshot.model_id = model.model_id
             AND snapshot.checkpoint_ready_sequence < ready.sequence
        )
    )"""


def completed_labelled_events(*, session: Session, task_name: str, event_ids: list[str]) -> list[str]:
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
    *, session: Session, task_name: str, model_id: str, start_sequence: int, limit: int = 500
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


def event_payloads(*, session: Session, task_name: str, event_ids: list[str]) -> dict[str, dict[str, Any]]:
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


def latest_labelled_examples(*, session: Session, task_name: str, limit: int = 5) -> list[LabelledExample]:
    """Read recent labelled events before archives are available for a task."""
    rows = session.execute(
        text(
            """SELECT event.event_id, event.event, label.y
               FROM benchmark_labels AS label
               JOIN benchmark_events AS event USING (task_name, event_id)
               WHERE label.task_name = :task_name AND label.y <> 'null'::jsonb
               ORDER BY label.available_at DESC, event.sequence DESC LIMIT :limit"""
        ),
        {"task_name": task_name, "limit": limit},
    )
    return list(reversed([LabelledExample(event_id=event_id, payload=event, target=y) for event_id, event, y in rows]))


def add_prediction_skips(
    *,
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


def add_predictions(*, session: Session, task_name: str, model_id: str, predictions: dict[str, Any]) -> list[str]:
    """Persist only predictions whose label is still unavailable.

    The final label check happens in the INSERT statement, closing the race
    between selecting candidate events and writing their predictions.
    """
    if not predictions:
        return []
    rows = json.dumps([{"event_id": event_id, "prediction": value} for event_id, value in predictions.items()])
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
