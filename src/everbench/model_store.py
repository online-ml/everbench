"""Database operations for model registrations, artifacts, metrics, and snapshots."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

from sqlalchemy import delete, func, select, text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from everbench import artifacts, event_store
from everbench.db import advisory_key
from everbench.schema import (
    AutoExperiment,
    BenchmarkEvent,
    MetricState,
    ModelArtifact,
    ModelEventState,
    ModelRegistration,
    ModelSnapshot,
    ReadyLabel,
)


def runnable_registrations(session: Session, task_name: str) -> list[ModelRegistration]:
    """Return manually active models whose retry window has elapsed."""
    return list(
        session.scalars(
            select(ModelRegistration)
            .where(
                ModelRegistration.task_name == task_name,
                ModelRegistration.active,
                (ModelRegistration.disabled_until.is_(None)) | (ModelRegistration.disabled_until <= func.now()),
            )
            .order_by(ModelRegistration.model_id)
            .with_for_update()
        )
    )


def disabled_registrations(session: Session, task_name: str) -> list[ModelRegistration]:
    """Return active models currently paused by the circuit breaker."""
    return list(
        session.scalars(
            select(ModelRegistration)
            .where(
                ModelRegistration.task_name == task_name,
                ModelRegistration.active,
                ModelRegistration.disabled_until.is_not(None),
                ModelRegistration.disabled_until > func.now(),
            )
            .order_by(ModelRegistration.model_id)
            .with_for_update()
        )
    )


def model_metric_state(session: Session, task_name: str, model_id: str) -> MetricState | None:
    return session.get(MetricState, {"task_name": task_name, "model_id": model_id})


def model_prediction_count(session: Session, task_name: str, model_id: str) -> int:
    return int(
        session.scalar(
            select(func.count())
            .select_from(ModelEventState)
            .where(
                ModelEventState.task_name == task_name,
                ModelEventState.model_id == model_id,
                ModelEventState.prediction_status == "predicted",
            )
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
        if registration.active and registration.artifact_id == artifact_id:
            # Retrying the exact same active upload is safe and idempotent.
            return registration, False
        raise ValueError("model_id has already been used; choose a new model_id")
    start_sequence = session.scalar(
        select(func.coalesce(func.max(BenchmarkEvent.sequence) + 1, 1)).where(BenchmarkEvent.task_name == task_name)
    )
    start_sequence = int(start_sequence or 1)
    registration = ModelRegistration(
        task_name=task_name,
        model_id=model_id,
        owner=owner,
        artifact_id=artifact_id,
        start_sequence=start_sequence,
        prediction_cursor_sequence=start_sequence - 1,
        label_cursor_sequence=int(
            session.scalar(
                select(func.coalesce(func.max(ReadyLabel.sequence), 0)).where(ReadyLabel.task_name == task_name)
            )
            or 0
        ),
    )
    session.add(registration)
    return registration, True


def lock_model_registrations(session: Session, task_name: str) -> None:
    """Serialize count-and-register operations for one task until transaction end."""
    session.execute(
        text("SELECT pg_advisory_xact_lock(:lock_id)"),
        {"lock_id": advisory_key("model-registration", task_name)},
    )


def _delete_unreferenced_artifacts(session: Session, artifact_ids: set[str]) -> None:
    if not artifact_ids:
        return
    registered_ids = select(ModelRegistration.artifact_id).where(ModelRegistration.artifact_id.is_not(None))
    snapshot_ids = select(ModelSnapshot.artifact_id)
    experiment_ids = select(AutoExperiment.candidate_artifact_id).where(
        AutoExperiment.candidate_artifact_id.is_not(None)
    )
    champion_ids = select(AutoExperiment.champion_artifact_id)
    session.execute(
        delete(ModelArtifact).where(
            ModelArtifact.artifact_id.in_(artifact_ids),
            ModelArtifact.artifact_id.not_in(registered_ids),
            ModelArtifact.artifact_id.not_in(snapshot_ids),
            ModelArtifact.artifact_id.not_in(experiment_ids),
            ModelArtifact.artifact_id.not_in(champion_ids),
        )
    )


def delete_model(session: Session, task_name: str, model_id: str) -> bool:
    """Remove a registration and all state that belongs only to that model."""
    registration = session.get(ModelRegistration, {"task_name": task_name, "model_id": model_id})
    if registration is None:
        return False
    snapshot_artifact_id = session.scalar(
        select(ModelSnapshot.artifact_id).where(
            ModelSnapshot.task_name == task_name, ModelSnapshot.model_id == model_id
        )
    )
    experiment_artifact_ids = session.execute(
        select(AutoExperiment.champion_artifact_id, AutoExperiment.candidate_artifact_id).where(
            AutoExperiment.task_name == task_name,
            AutoExperiment.model_id == model_id,
        )
    ).all()
    for model_table in (MetricState, ModelSnapshot):
        session.execute(delete(model_table).where(model_table.task_name == task_name, model_table.model_id == model_id))
    artifact_ids = {artifact_id for artifact_id in (registration.artifact_id, snapshot_artifact_id) if artifact_id}
    artifact_ids.update(
        artifact_id for pair in experiment_artifact_ids for artifact_id in pair if artifact_id is not None
    )
    session.delete(registration)
    session.flush()
    _delete_unreferenced_artifacts(session, artifact_ids)
    return True


def record_model_failure(
    session: Session,
    task_name: str,
    model_id: str,
    error: BaseException,
    retry_initial_seconds: float,
    retry_max_seconds: float,
) -> datetime | None:
    """Pause a failed model with capped exponential backoff and return its retry time."""
    registration = session.get(ModelRegistration, {"task_name": task_name, "model_id": model_id})
    if registration is None:
        return None
    registration.failure_count += 1
    registration.error_count += 1
    registration.last_error = f"{type(error).__name__}: {error}"[:2_000]
    now = datetime.now(UTC)
    registration.failed_at = now
    retry_seconds = min(
        max(retry_initial_seconds, 0.0) * 2 ** min(registration.failure_count - 1, 20),
        max(retry_max_seconds, 0.0),
    )
    registration.disabled_until = now + timedelta(seconds=retry_seconds)
    return registration.disabled_until


def record_model_success(session: Session, task_name: str, model_id: str) -> None:
    registration = session.get(ModelRegistration, {"task_name": task_name, "model_id": model_id})
    if registration is not None and registration.failure_count:
        registration.failure_count = 0
        registration.last_error = None
        registration.failed_at = None
        registration.disabled_until = None


def advance_model_checkpoint(
    session: Session,
    task_name: str,
    registration: ModelRegistration,
    label_available_at: datetime,
    event_sequence: int,
    ready_sequence: int,
) -> None:
    """Mark terminally skipped labels as covered by the model's checkpoint.

    A paused model deliberately does not learn its skipped labels. Its current
    artifact therefore remains the right restart state, while the watermark
    prevents those terminal labels from indefinitely blocking compaction.
    """
    snapshot = latest_snapshot(session, task_name, registration.model_id)
    if snapshot is None:
        if registration.artifact_id is None:
            raise RuntimeError(f"model artifact missing for {registration.model_id}")
        session.add(
            ModelSnapshot(
                task_name=task_name,
                model_id=registration.model_id,
                artifact_id=registration.artifact_id,
                checkpoint_label_available_at=label_available_at,
                checkpoint_event_sequence=event_sequence,
                checkpoint_ready_sequence=ready_sequence,
            )
        )
        return

    candidate = (label_available_at, event_sequence)
    if (
        snapshot.checkpoint_label_available_at is None
        or snapshot.checkpoint_event_sequence is None
        or candidate > (snapshot.checkpoint_label_available_at, snapshot.checkpoint_event_sequence)
    ):
        snapshot.checkpoint_label_available_at = label_available_at
        snapshot.checkpoint_event_sequence = event_sequence
    if snapshot.checkpoint_ready_sequence is None or ready_sequence > snapshot.checkpoint_ready_sequence:
        snapshot.checkpoint_ready_sequence = ready_sequence


def record_disabled_work(
    session: Session, task_name: str, registration: ModelRegistration, limit: int
) -> tuple[int, int]:
    """Terminally skip bounded work for a paused model and count it durably."""
    events = event_store.events_after_cursor(
        session,
        task_name,
        registration.model_id,
        registration.prediction_cursor_sequence,
        registration.start_sequence,
        limit,
    )
    missing = [event_id for event_id, _, _, has_state in events if not has_state]
    skipped_predictions = len(
        event_store.add_prediction_skips(session, task_name, registration.model_id, missing, "model-disabled")
    )
    if events:
        registration.prediction_cursor_sequence = events[-1][1]
    labels = event_store.ready_labels_after_cursor(
        session,
        task_name,
        registration.model_id,
        registration.label_cursor_sequence,
        registration.start_sequence,
        limit,
    )
    missing_label_state = [event_id for _, event_id, *_, prediction_status, _, _ in labels if prediction_status is None]
    event_store.add_prediction_skips(session, task_name, registration.model_id, missing_label_state, "model-disabled")
    evaluations = [row for row in labels if row[6] == "predicted" and not row[7]]
    event_store.add_metric_updates(session, task_name, registration.model_id, [row[1] for row in evaluations])
    untrained = [row for row in labels if not row[8]]
    trained_event_ids = event_store.add_trainings(
        session, task_name, registration.model_id, [row[1] for row in untrained]
    )
    skipped_labels = len(trained_event_ids)
    if labels:
        last = labels[-1]
        registration.label_cursor_sequence = last[0]
        advance_model_checkpoint(session, task_name, registration, last[3], last[4], last[0])
    registration.skipped_predictions += skipped_predictions
    registration.skipped_labels += skipped_labels
    return skipped_predictions, skipped_labels


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
    checkpoint_ready_sequence: int | None,
) -> list[tuple[str, dict[str, Any], Any]]:
    """Return committed learning updates not represented by a checkpoint."""
    parameters: dict[str, Any] = {"task_name": task_name, "model_id": model_id}
    watermark = ""
    order_by = "label.available_at, event.sequence"
    if checkpoint_ready_sequence is not None:
        watermark = "AND ready.sequence > :checkpoint_ready_sequence"
        parameters["checkpoint_ready_sequence"] = checkpoint_ready_sequence
        order_by = "ready.sequence"
    elif checkpoint_label_available_at is not None and checkpoint_event_sequence is not None:
        watermark = """AND (label.available_at > :checkpoint_label_available_at
                          OR (label.available_at = :checkpoint_label_available_at
                              AND event.sequence > :checkpoint_event_sequence))"""
        parameters.update(
            checkpoint_label_available_at=checkpoint_label_available_at,
            checkpoint_event_sequence=checkpoint_event_sequence,
        )
    rows = session.execute(
        text(
            f"""SELECT event.event_id, event.event, label.y
                 FROM benchmark_model_events AS model_event
                 JOIN benchmark_events AS event USING (task_name, event_id)
                 JOIN benchmark_labels AS label USING (task_name, event_id)
                 JOIN benchmark_ready_labels AS ready USING (task_name, event_id)
                 WHERE model_event.task_name = :task_name AND model_event.model_id = :model_id
                   AND model_event.trained_at IS NOT NULL
                   {watermark}
                 ORDER BY {order_by}"""
        ),
        parameters,
    )
    return [(event_id, event, y) for event_id, event, y in rows]


def save_pickle_snapshot(
    session: Session,
    task_name: str,
    model_id: str,
    payload: bytes,
    checkpoint_label_available_at: datetime | None,
    checkpoint_event_sequence: int | None,
    checkpoint_ready_sequence: int | None,
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
            checkpoint_ready_sequence=checkpoint_ready_sequence,
        )
        .on_conflict_do_update(
            index_elements=["task_name", "model_id"],
            set_={
                "artifact_id": artifact_record.artifact_id,
                "checkpoint_label_available_at": checkpoint_label_available_at,
                "checkpoint_event_sequence": checkpoint_event_sequence,
                "checkpoint_ready_sequence": checkpoint_ready_sequence,
                "created_at": func.now(),
            },
        )
    )
    session.execute(statement)
    stale_ids = {previous_artifact_id} - {artifact_record.artifact_id} if previous_artifact_id else set()
    _delete_unreferenced_artifacts(session, stale_ids)
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
