"""All database access shared by workers and the HTTP API."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from typing import Any
from uuid import uuid4

from sqlalchemy import delete, func, select, text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from everbench import artifacts
from everbench.schema import (
    ArchiveManifest,
    BenchmarkEvent,
    BenchmarkLabel,
    MetricState,
    MetricUpdate,
    ModelArtifact,
    ModelRegistration,
    ModelSnapshot,
    Prediction,
    PredictionSkip,
    StreamCursor,
    Training,
    WorkerHeartbeat,
)


def add_events(
    session: Session,
    task_name: str,
    events: list[tuple[str, float, dict[str, float]]],
    delay_seconds: float | None = None,
) -> int:
    """Insert a collector batch in one statement and one transaction."""
    if not events:
        return 0
    values = [
        {
            "task_name": task_name,
            "event_id": event_id,
            "event_time": datetime.fromtimestamp(timestamp, UTC),
            "features": features,
        }
        for event_id, timestamp, features in events
    ]
    statement = (
        insert(BenchmarkEvent)
        .values(values)
        .on_conflict_do_nothing(index_elements=["task_name", "event_id"])
        .returning(BenchmarkEvent.event_id)
    )
    inserted = len(session.scalars(statement).all())
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
                     AND event.event_time <= now() - make_interval(secs => :delay_seconds)"""
            ),
            {
                "task_name": task_name,
                "event_ids": [event_id for event_id, _, _ in events],
                "delay_seconds": delay_seconds,
            },
        )
    return inserted


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


def add_labels(
    session: Session, task_name: str, labels: list[tuple[str, Any, str]], delay_seconds: float | None
) -> int:
    """Insert a label-collector batch with one Postgres statement.

    A positive label arriving after its configured horizon is ignored: the
    delayed-negative finalizer is the authority once that horizon has passed.
    """
    if not labels:
        return 0
    if delay_seconds is not None:
        event_times = {
            event_id: event_time
            for event_id, event_time in session.execute(
                select(BenchmarkEvent.event_id, BenchmarkEvent.event_time).where(
                    BenchmarkEvent.task_name == task_name,
                    BenchmarkEvent.event_id.in_([event_id for event_id, _, _ in labels]),
                )
            )
            if event_id is not None and event_time is not None
        }
        cutoff = datetime.now(UTC) - timedelta(seconds=delay_seconds)
        labels = [
            (event_id, y, reason)
            for event_id, y, reason in labels
            if event_id not in event_times or y != 1 or event_times[event_id] > cutoff
        ]
        if not labels:
            return 0
    values = [
        {"task_name": task_name, "event_id": event_id, "y": y, "reason": reason} for event_id, y, reason in labels
    ]
    statement = (
        insert(BenchmarkLabel)
        .values(values)
        .on_conflict_do_nothing(index_elements=["task_name", "event_id"])
        .returning(BenchmarkLabel.event_id)
    )
    return len(session.scalars(statement).all())


def active_registrations(session: Session, task_name: str) -> list[ModelRegistration]:
    return list(
        session.scalars(
            select(ModelRegistration)
            .where(ModelRegistration.task_name == task_name, ModelRegistration.active)
            .order_by(ModelRegistration.model_id)
        )
    )


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
               LEFT JOIN benchmark_predictions AS prediction
                 ON prediction.task_name = label.task_name AND prediction.event_id = label.event_id
                 AND prediction.model_id = :model_id
               LEFT JOIN benchmark_prediction_skips AS prediction_skip
                 ON prediction_skip.task_name = label.task_name AND prediction_skip.event_id = label.event_id
                 AND prediction_skip.model_id = :model_id
               LEFT JOIN benchmark_trainings AS training
                 ON training.task_name = label.task_name AND training.event_id = label.event_id
                 AND training.model_id = :model_id
               WHERE label.task_name = :task_name
                 AND (prediction.event_id IS NOT NULL OR prediction_skip.event_id IS NOT NULL)
                 AND training.event_id IS NULL
               ORDER BY label.available_at, event.sequence LIMIT :limit"""
        ),
        {"task_name": task_name, "model_id": model_id, "limit": limit},
    )
    return [(event_id, y, available_at, sequence) for event_id, y, available_at, sequence in rows]


def unevaluated_labels(session: Session, task_name: str, model_id: str, limit: int = 500) -> list[tuple[str, Any, Any]]:
    """Return labelled predictions not yet incorporated into River metrics."""
    rows = session.execute(
        text(
            """SELECT label.event_id, label.y, prediction.prediction
               FROM benchmark_labels AS label
               JOIN benchmark_events AS event USING (task_name, event_id)
               JOIN benchmark_predictions AS prediction USING (task_name, event_id)
               LEFT JOIN benchmark_metric_updates AS metric_update
                 ON metric_update.task_name = label.task_name AND metric_update.event_id = label.event_id
                 AND metric_update.model_id = :model_id
               WHERE label.task_name = :task_name AND prediction.model_id = :model_id
                 AND metric_update.event_id IS NULL
               ORDER BY label.available_at, event.sequence LIMIT :limit"""
        ),
        {"task_name": task_name, "model_id": model_id, "limit": limit},
    )
    return [(event_id, y, prediction) for event_id, y, prediction in rows]


def add_trainings(session: Session, task_name: str, model_id: str, event_ids: list[str]) -> None:
    if event_ids:
        session.execute(
            insert(Training)
            .values([{"task_name": task_name, "event_id": event_id, "model_id": model_id} for event_id in event_ids])
            .on_conflict_do_nothing()
        )


def add_metric_updates(session: Session, task_name: str, model_id: str, event_ids: list[str]) -> None:
    if event_ids:
        session.execute(
            insert(MetricUpdate)
            .values([{"task_name": task_name, "event_id": event_id, "model_id": model_id} for event_id in event_ids])
            .on_conflict_do_nothing()
        )


def _model_processing_pending_clause() -> str:
    """SQL predicate for work that prevents a labelled event from archiving."""
    return """(
        (NOT EXISTS (
          SELECT 1 FROM benchmark_predictions AS prediction
          WHERE prediction.task_name = event.task_name AND prediction.event_id = event.event_id
            AND prediction.model_id = model.model_id
        ) AND NOT EXISTS (
          SELECT 1 FROM benchmark_prediction_skips AS prediction_skip
          WHERE prediction_skip.task_name = event.task_name AND prediction_skip.event_id = event.event_id
            AND prediction_skip.model_id = model.model_id
        ))
        OR NOT EXISTS (
          SELECT 1 FROM benchmark_trainings AS training
          WHERE training.task_name = event.task_name AND training.event_id = event.event_id
            AND training.model_id = model.model_id
        )
        OR (EXISTS (
          SELECT 1 FROM benchmark_predictions AS prediction
          WHERE prediction.task_name = event.task_name AND prediction.event_id = event.event_id
            AND prediction.model_id = model.model_id
        ) AND NOT EXISTS (
          SELECT 1 FROM benchmark_metric_updates AS metric_update
          WHERE metric_update.task_name = event.task_name AND metric_update.event_id = event.event_id
            AND metric_update.model_id = model.model_id
        ))
        OR NOT EXISTS (
          SELECT 1 FROM model_snapshots AS snapshot
          WHERE snapshot.task_name = model.task_name AND snapshot.model_id = model.model_id
            AND (
              snapshot.checkpoint_label_available_at IS NULL
              OR snapshot.checkpoint_label_available_at > label.available_at
              OR (snapshot.checkpoint_label_available_at = label.available_at
                  AND snapshot.checkpoint_event_sequence >= event.sequence)
            )
        )
    )"""


def completed_labelled_events(session: Session, task_name: str, event_ids: list[str]) -> list[str]:
    """Find labelled events no eligible active model still needs to process."""
    if not event_ids:
        return []
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
               LEFT JOIN benchmark_predictions AS prediction
                 ON prediction.task_name = event.task_name AND prediction.event_id = event.event_id
                 AND prediction.model_id = :model_id
               LEFT JOIN benchmark_prediction_skips AS prediction_skip
                 ON prediction_skip.task_name = event.task_name AND prediction_skip.event_id = event.event_id
                 AND prediction_skip.model_id = :model_id
               LEFT JOIN benchmark_labels AS label
                 ON label.task_name = event.task_name AND label.event_id = event.event_id
               WHERE event.task_name = :task_name AND event.sequence >= :start_sequence
                 AND prediction.event_id IS NULL AND prediction_skip.event_id IS NULL
                 AND label.event_id IS NULL
               ORDER BY event.sequence LIMIT :limit"""
        ),
        {"task_name": task_name, "model_id": model_id, "start_sequence": start_sequence, "limit": limit},
    )
    return [event_id for (event_id,) in rows]


def event_features(session: Session, task_name: str, event_ids: list[str]) -> dict[str, dict[str, float]]:
    """Bulk database fallback for read-cache misses."""
    if not event_ids:
        return {}
    rows = session.execute(
        select(BenchmarkEvent.event_id, BenchmarkEvent.features).where(
            BenchmarkEvent.task_name == task_name,
            BenchmarkEvent.event_id.in_(event_ids),
        )
    )
    return {event_id: features for event_id, features in rows}


def latest_labelled_examples(
    session: Session, task_name: str, limit: int = 5
) -> list[tuple[str, dict[str, float], Any]]:
    """Read recent labelled events before archives are available for a task."""
    rows = session.execute(
        text(
            """SELECT event.event_id, event.features, label.y
               FROM benchmark_labels AS label
               JOIN benchmark_events AS event USING (task_name, event_id)
               WHERE label.task_name = :task_name
               ORDER BY label.available_at DESC, event.sequence DESC LIMIT :limit"""
        ),
        {"task_name": task_name, "limit": limit},
    )
    return list(reversed([(event_id, features, y) for event_id, features, y in rows]))


def labelled_unpredicted_events(
    session: Session, task_name: str, model_id: str, start_sequence: int, limit: int = 500
) -> list[str]:
    """Events that became labelled before this model persisted a prediction."""
    rows = session.execute(
        text(
            """SELECT event.event_id
               FROM benchmark_events AS event
               JOIN benchmark_labels AS label USING (task_name, event_id)
               LEFT JOIN benchmark_predictions AS prediction
                 ON prediction.task_name = event.task_name AND prediction.event_id = event.event_id
                 AND prediction.model_id = :model_id
               LEFT JOIN benchmark_prediction_skips AS prediction_skip
                 ON prediction_skip.task_name = event.task_name AND prediction_skip.event_id = event.event_id
                 AND prediction_skip.model_id = :model_id
               WHERE event.task_name = :task_name AND event.sequence >= :start_sequence
                 AND prediction.event_id IS NULL AND prediction_skip.event_id IS NULL
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
) -> None:
    if event_ids:
        session.execute(
            insert(PredictionSkip)
            .values(
                [
                    {"task_name": task_name, "event_id": event_id, "model_id": model_id, "reason": reason}
                    for event_id in event_ids
                ]
            )
            .on_conflict_do_nothing()
        )


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
            """INSERT INTO benchmark_predictions (task_name, event_id, model_id, prediction)
               SELECT :task_name, incoming.event_id, :model_id, incoming.prediction
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


def model_metric_state(session: Session, task_name: str, model_id: str) -> MetricState | None:
    return session.get(MetricState, {"task_name": task_name, "model_id": model_id})


def model_prediction_count(session: Session, task_name: str, model_id: str) -> int:
    return int(
        session.scalar(
            select(func.count())
            .select_from(Prediction)
            .where(Prediction.task_name == task_name, Prediction.model_id == model_id)
        )
        or 0
    )


def save_metric_state(
    session: Session,
    task_name: str,
    model_id: str,
    definition: dict[str, Any],
    state: bytes,
    predictions: int,
    observations: int,
    values: dict[str, float | None],
) -> None:
    statement = (
        insert(MetricState)
        .values(
            task_name=task_name,
            model_id=model_id,
            definition=definition,
            state=state,
            predictions=predictions,
            observations=observations,
            values=values,
        )
        .on_conflict_do_update(
            index_elements=["task_name", "model_id"],
            set_={
                "definition": definition,
                "state": state,
                "predictions": predictions,
                "observations": observations,
                "values": values,
                "updated_at": func.now(),
            },
        )
    )
    session.execute(statement)


def register_model(
    session: Session,
    task_name: str,
    model_id: str,
    owner: str,
    artifact_id: str,
) -> tuple[ModelRegistration, bool]:
    registration = session.get(ModelRegistration, {"task_name": task_name, "model_id": model_id})
    if registration:
        if registration.artifact_id != artifact_id:
            raise ValueError("model_id already has a different artifact; choose a new model_id")
        registration.owner = owner
        registration.active = True
        registration.failure_count = 0
        registration.last_error = None
        registration.failed_at = None
        return registration, False
    start_sequence = session.scalar(
        select(func.coalesce(func.max(BenchmarkEvent.sequence) + 1, 1)).where(BenchmarkEvent.task_name == task_name)
    )
    registration = ModelRegistration(
        task_name=task_name,
        model_id=model_id,
        owner=owner,
        artifact_id=artifact_id,
        start_sequence=start_sequence,
    )
    session.add(registration)
    return registration, True


def deactivate_model(session: Session, task_name: str, model_id: str) -> bool:
    registration = session.get(ModelRegistration, {"task_name": task_name, "model_id": model_id})
    if registration is None or not registration.active:
        return False
    registration.active = False
    return True


def record_model_failure(
    session: Session, task_name: str, model_id: str, error: BaseException, max_failures: int
) -> bool:
    """Record one failed model cycle and deactivate repeated offenders.

    Returns whether the model remains active. The traceback stays in logs; the
    database retains a concise operational message for the dashboard/API.
    """
    registration = session.get(ModelRegistration, {"task_name": task_name, "model_id": model_id})
    if registration is None:
        return False
    registration.failure_count += 1
    registration.last_error = f"{type(error).__name__}: {error}"[:2_000]
    registration.failed_at = datetime.now(UTC)
    if registration.failure_count >= max_failures:
        registration.active = False
    return registration.active


def record_model_success(session: Session, task_name: str, model_id: str) -> None:
    registration = session.get(ModelRegistration, {"task_name": task_name, "model_id": model_id})
    if registration is not None and registration.failure_count:
        registration.failure_count = 0
        registration.last_error = None
        registration.failed_at = None


def model_registration(session: Session, task_name: str, model_id: str) -> ModelRegistration | None:
    return session.get(ModelRegistration, {"task_name": task_name, "model_id": model_id})


def store_artifact(session: Session, payload: bytes, signature: str, metadata: dict[str, Any]) -> ModelArtifact:
    checksum = artifacts.sha256(payload)
    artifact = session.scalar(select(ModelArtifact).where(ModelArtifact.sha256 == checksum))
    if artifact:
        # An identical artifact may be re-uploaded with a previously missing
        # source definition. Preserve existing validation metadata while
        # refreshing the human-facing artifact description.
        artifact.metadata_ = {**(artifact.metadata_ or {}), **metadata}
        return artifact
    artifact = ModelArtifact(
        artifact_id=str(uuid4()),
        sha256=checksum,
        payload=payload,
        signature=signature,
        metadata_=metadata,
    )
    session.add(artifact)
    return artifact


def record_artifact_validation(
    artifact_record: ModelArtifact, task_name: str, definition: dict[str, Any], examples: int
) -> None:
    metadata = dict(artifact_record.metadata_ or {})
    validations = dict(metadata.get("validations") or {})
    validations[task_name] = {"definition": definition, "examples": examples}
    metadata["validations"] = validations
    artifact_record.metadata_ = metadata


def artifact(session: Session, artifact_id: str) -> ModelArtifact | None:
    return session.get(ModelArtifact, artifact_id)


def latest_snapshot(session: Session, task_name: str, model_id: str) -> ModelSnapshot | None:
    return session.scalar(
        select(ModelSnapshot).where(ModelSnapshot.task_name == task_name, ModelSnapshot.model_id == model_id)
    )


def trained_examples_since_checkpoint(
    session: Session,
    task_name: str,
    model_id: str,
    checkpoint_label_available_at: datetime | None,
    checkpoint_event_sequence: int | None,
) -> list[tuple[str, dict[str, float], Any]]:
    """Return committed learning updates not represented by a checkpoint."""
    parameters: dict[str, Any] = {"task_name": task_name, "model_id": model_id}
    watermark = ""
    if checkpoint_label_available_at is not None and checkpoint_event_sequence is not None:
        watermark = """AND (label.available_at > :checkpoint_label_available_at
                          OR (label.available_at = :checkpoint_label_available_at
                              AND event.sequence > :checkpoint_event_sequence))"""
        parameters.update(
            checkpoint_label_available_at=checkpoint_label_available_at,
            checkpoint_event_sequence=checkpoint_event_sequence,
        )
    rows = session.execute(
        text(
            f"""SELECT event.event_id, event.features, label.y
                 FROM benchmark_trainings AS training
                 JOIN benchmark_events AS event USING (task_name, event_id)
                 JOIN benchmark_labels AS label USING (task_name, event_id)
                 WHERE training.task_name = :task_name AND training.model_id = :model_id
                   {watermark}
                 ORDER BY label.available_at, event.sequence"""
        ),
        parameters,
    )
    return [(event_id, features, y) for event_id, features, y in rows]


def save_pickle_snapshot(
    session: Session,
    task_name: str,
    model_id: str,
    payload: bytes,
    checkpoint_label_available_at: datetime | None,
    checkpoint_event_sequence: int | None,
) -> ModelArtifact:
    """Replace the operational checkpoint instead of retaining every batch.

    Historical model checkpoints belong in a deliberate archive policy, not in
    the always-on Postgres path. Retaining just one snapshot is sufficient for
    restart recovery and bounds database growth for large River models.
    """
    previous_artifact_id = session.scalar(
        select(ModelSnapshot.artifact_id).where(
            ModelSnapshot.task_name == task_name, ModelSnapshot.model_id == model_id
        )
    )
    artifact_record = store_artifact(session, payload, artifacts.sign(payload), {"source": "worker-snapshot"})
    statement = (
        insert(ModelSnapshot)
        .values(
            task_name=task_name,
            model_id=model_id,
            artifact_id=artifact_record.artifact_id,
            checkpoint_label_available_at=checkpoint_label_available_at,
            checkpoint_event_sequence=checkpoint_event_sequence,
        )
        .on_conflict_do_update(
            index_elements=["task_name", "model_id"],
            set_={
                "artifact_id": artifact_record.artifact_id,
                "checkpoint_label_available_at": checkpoint_label_available_at,
                "checkpoint_event_sequence": checkpoint_event_sequence,
                "created_at": func.now(),
            },
        )
    )
    session.execute(statement)
    stale_ids = {previous_artifact_id} - {artifact_record.artifact_id} if previous_artifact_id else set()
    if stale_ids:
        registered_ids = select(ModelRegistration.artifact_id).where(ModelRegistration.artifact_id.is_not(None))
        snapshot_ids = select(ModelSnapshot.artifact_id)
        session.execute(
            delete(ModelArtifact).where(
                ModelArtifact.artifact_id.in_(stale_ids),
                ModelArtifact.artifact_id.not_in(registered_ids),
                ModelArtifact.artifact_id.not_in(snapshot_ids),
            )
        )
    return artifact_record


def active_model_count(session: Session, task_name: str) -> int:
    return int(
        session.scalar(
            select(func.count())
            .select_from(ModelRegistration)
            .where(ModelRegistration.task_name == task_name, ModelRegistration.active)
        )
        or 0
    )


def record_heartbeat(
    session: Session,
    worker_id: str,
    task_name: str | None,
    role: str,
    status: str = "running",
    detail: str | None = None,
) -> None:
    statement = (
        insert(WorkerHeartbeat)
        .values(worker_id=worker_id, task_name=task_name, role=role, status=status, detail=detail)
        .on_conflict_do_update(
            index_elements=["worker_id"],
            set_={"task_name": task_name, "role": role, "status": status, "detail": detail, "last_seen_at": func.now()},
        )
    )
    session.execute(statement)


def worker_health(session: Session) -> list[WorkerHeartbeat]:
    return list(session.scalars(select(WorkerHeartbeat).order_by(WorkerHeartbeat.role, WorkerHeartbeat.worker_id)))


def task_names(session: Session) -> list[str]:
    """Tasks known to operational state, including idle registered tasks."""
    rows = session.execute(
        text(
            """SELECT task_name FROM benchmark_events
               UNION SELECT task_name FROM benchmark_labels
               UNION SELECT task_name FROM benchmark_models
               UNION SELECT task_name FROM worker_heartbeats WHERE task_name IS NOT NULL
               ORDER BY task_name"""
        )
    )
    return [task_name for (task_name,) in rows]


def task_stats(session: Session, task_name: str) -> dict[str, int]:
    row = (
        session.execute(
            text(
                """SELECT
                 (SELECT COUNT(*) FROM benchmark_events WHERE task_name = :task_name)
                   + COALESCE((SELECT SUM(row_count) FROM archive_manifest WHERE task_name = :task_name), 0) AS events,
                 (SELECT COUNT(*)
                    FROM benchmark_labels AS label
                    JOIN benchmark_events AS event USING (task_name, event_id)
                   WHERE label.task_name = :task_name)
                   + COALESCE((SELECT SUM(row_count) FROM archive_manifest WHERE task_name = :task_name), 0) AS labels"""
            ),
            {"task_name": task_name},
        )
        .mappings()
        .one()
    )
    return {key: int(value) for key, value in row.items()}


def task_leaderboard(session: Session, task_name: str) -> list[dict[str, Any]]:
    rows = session.execute(
        text(
            """SELECT model.model_id,
                      model.owner,
                      model.active,
                      model.failure_count,
                      model.last_error,
                      model.failed_at,
                      model.created_at,
                      COALESCE(metric_state.predictions, 0) AS predictions,
                      COALESCE(metric_state.observations, 0) AS labels,
                      COALESCE(metric_state.values, '{}'::jsonb) AS metrics,
                      COALESCE(octet_length(snapshot_artifact.payload), octet_length(artifact.payload), 0) AS model_bytes,
                      COALESCE(artifact.metadata ->> 'class_definition', '') AS class_definition,
                      COALESCE(artifact.metadata ->> 'class_name', 'pickle') AS class_name
               FROM benchmark_models AS model
               LEFT JOIN benchmark_metric_state AS metric_state
                 ON metric_state.task_name = model.task_name AND metric_state.model_id = model.model_id
               LEFT JOIN model_artifacts AS artifact ON artifact.artifact_id = model.artifact_id
               LEFT JOIN model_snapshots AS snapshot
                 ON snapshot.task_name = model.task_name AND snapshot.model_id = model.model_id
               LEFT JOIN model_artifacts AS snapshot_artifact ON snapshot_artifact.artifact_id = snapshot.artifact_id
               WHERE model.task_name = :task_name
               ORDER BY model.model_id"""
        ),
        {"task_name": task_name},
    ).mappings()
    return [dict(row) for row in rows]


def task_archives(session: Session, task_name: str) -> list[ArchiveManifest]:
    return list(
        session.scalars(
            select(ArchiveManifest)
            .where(ArchiveManifest.task_name == task_name)
            .order_by(ArchiveManifest.event_date.desc(), ArchiveManifest.created_at.desc())
        )
    )


def task_archive(session: Session, task_name: str, content_sha256: str) -> ArchiveManifest | None:
    return session.scalar(
        select(ArchiveManifest).where(
            ArchiveManifest.task_name == task_name, ArchiveManifest.content_sha256 == content_sha256
        )
    )


def task_metric_names(session: Session, task_name: str) -> list[str]:
    row = session.scalar(
        select(MetricState.definition)
        .where(MetricState.task_name == task_name)
        .order_by(MetricState.updated_at.desc())
        .limit(1)
    )
    return list(row.get("metric_names", [])) if row else []


def next_archive_week(session: Session, task_name: str, cutoff: datetime) -> date | None:
    """Return the oldest UTC ISO week containing eligible completed rows."""
    return session.scalar(
        text(
            f"""SELECT date_trunc('week', event.event_time AT TIME ZONE 'UTC')::date
               FROM benchmark_events AS event
               JOIN benchmark_labels AS label USING (task_name, event_id)
               WHERE event.task_name = :task_name AND event.event_time < :cutoff
                 AND NOT EXISTS (
                   SELECT 1 FROM benchmark_models AS model
                   WHERE model.task_name = event.task_name AND model.active
                     AND event.sequence >= model.start_sequence
                     AND {_model_processing_pending_clause()}
                 )
               ORDER BY date_trunc('week', event.event_time AT TIME ZONE 'UTC') LIMIT 1"""
        ),
        {"task_name": task_name, "cutoff": cutoff},
    )


def archive_rows(
    session: Session, task_name: str, week_start: date, cutoff: datetime, limit: int
) -> list[dict[str, Any]]:
    rows = session.execute(
        text(
            f"""SELECT event.event_id, event.sequence, event.event_time, event.inserted_at, event.features,
                      label.y, label.reason, label.available_at
               FROM benchmark_events AS event
               JOIN benchmark_labels AS label USING (task_name, event_id)
               WHERE event.task_name = :task_name
                 AND date_trunc('week', event.event_time AT TIME ZONE 'UTC')::date = :week_start
                 AND event.event_time < :cutoff
                 AND NOT EXISTS (
                   SELECT 1 FROM benchmark_models AS model
                   WHERE model.task_name = event.task_name AND model.active
                     AND event.sequence >= model.start_sequence
                     AND {_model_processing_pending_clause()}
                 )
               ORDER BY event.inserted_at, event.sequence LIMIT :limit"""
        ),
        {"task_name": task_name, "week_start": week_start, "cutoff": cutoff, "limit": limit},
    ).mappings()
    return [dict(row) for row in rows]


def record_archive(
    session: Session, content_sha256: str, task_name: str, event_date: date, path: str, row_count: int, byte_size: int
) -> None:
    session.execute(
        insert(ArchiveManifest)
        .values(
            content_sha256=content_sha256,
            task_name=task_name,
            event_date=event_date,
            path=path,
            row_count=row_count,
            byte_size=byte_size,
        )
        .on_conflict_do_nothing()
    )


def purge_archived_events(session: Session, task_name: str, event_ids: list[str]) -> None:
    """Only call after a manifest was committed for a durable archive target."""
    for model in (MetricUpdate, Training, PredictionSkip, Prediction, BenchmarkLabel, BenchmarkEvent):
        session.execute(delete(model).where(model.task_name == task_name, model.event_id.in_(event_ids)))
