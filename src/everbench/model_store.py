"""Database operations for model registrations, artifacts, metrics, and snapshots."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

from sqlalchemy import delete, func, select, text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from everbench import artifacts, event_store
from everbench.db import advisory_key
from everbench.records import LabelledExample
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


def runnable_registrations(*, session: Session, task_name: str) -> list[ModelRegistration]:
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


def disabled_registrations(*, session: Session, task_name: str) -> list[ModelRegistration]:
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


def model_metric_state(*, session: Session, task_name: str, model_id: str) -> MetricState | None:
    return session.get(MetricState, {"task_name": task_name, "model_id": model_id})


def model_prediction_count(*, session: Session, task_name: str, model_id: str) -> int:
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
    *,
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
    *, session: Session, task_name: str, model_id: str, owner: str, artifact_id: str
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


def lock_model_registrations(*, session: Session, task_name: str) -> None:
    """Serialize count-and-register operations for one task until transaction end."""
    session.execute(
        text("SELECT pg_advisory_xact_lock(:lock_id)"),
        {"lock_id": advisory_key(parts=("model-registration", task_name))},
    )


def _delete_unreferenced_artifacts(*, session: Session, artifact_ids: set[str]) -> None:
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


def delete_model(*, session: Session, task_name: str, model_id: str) -> bool:
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
    _delete_unreferenced_artifacts(session=session, artifact_ids=artifact_ids)
    return True


def record_model_failure(
    *,
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


def record_model_success(*, session: Session, task_name: str, model_id: str) -> None:
    registration = session.get(ModelRegistration, {"task_name": task_name, "model_id": model_id})
    if registration is not None and registration.failure_count:
        registration.failure_count = 0
        registration.last_error = None
        registration.failed_at = None
        registration.disabled_until = None


def advance_model_checkpoint(
    *, session: Session, task_name: str, registration: ModelRegistration, previous_sequence: int, ready_sequence: int
) -> None:
    """Advance a checkpoint across work that a disabled model deliberately skips."""
    snapshot = latest_snapshot(session=session, task_name=task_name, model_id=registration.model_id)
    # Never claim the saved pickle includes earlier learning that only lived in RAM.
    if (snapshot.checkpoint_ready_sequence if snapshot is not None else 0) < previous_sequence:
        return
    if snapshot is None:
        if registration.artifact_id is None:
            raise RuntimeError(f"model artifact missing for {registration.model_id}")
        session.add(
            ModelSnapshot(
                task_name=task_name,
                model_id=registration.model_id,
                artifact_id=registration.artifact_id,
                checkpoint_ready_sequence=ready_sequence,
            )
        )
    else:
        snapshot.checkpoint_ready_sequence = max(snapshot.checkpoint_ready_sequence, ready_sequence)


def start_live_generation(*, session: Session, registration: ModelRegistration) -> int:
    """Admit only observations committed after this generation's boundary.

    The caller holds the registration row lock. Taking the ingestion lock also
    prevents an in-flight collector from straddling the new event frontier.
    """
    event_store.lock_task_ingest(session=session, task_name=registration.task_name)
    registration.start_sequence = (
        int(
            session.scalar(
                select(func.coalesce(func.max(BenchmarkEvent.sequence), 0)).where(
                    BenchmarkEvent.task_name == registration.task_name
                )
            )
            or 0
        )
        + 1
    )
    registration.prediction_cursor_sequence = registration.start_sequence - 1
    registration.label_cursor_sequence = int(
        session.scalar(
            select(func.coalesce(func.max(ReadyLabel.sequence), 0)).where(
                ReadyLabel.task_name == registration.task_name
            )
        )
        or 0
    )
    registration.failure_count = 0
    registration.disabled_until = None
    return registration.label_cursor_sequence


def record_disabled_work(
    *, session: Session, task_name: str, registration: ModelRegistration, limit: int
) -> tuple[int, int]:
    """Terminally skip bounded work for a paused model and count it durably."""
    events = event_store.events_after_cursor(
        session=session,
        task_name=task_name,
        model_id=registration.model_id,
        cursor_sequence=registration.prediction_cursor_sequence,
        start_sequence=registration.start_sequence,
        limit=limit,
    )
    missing = [row.event_id for row in events if not row.has_state]
    skipped_predictions = len(
        event_store.add_prediction_skips(
            session=session,
            task_name=task_name,
            model_id=registration.model_id,
            event_ids=missing,
            reason="model-disabled",
        )
    )
    if events:
        registration.prediction_cursor_sequence = events[-1].sequence
    labels = event_store.ready_labels_after_cursor(
        session=session,
        task_name=task_name,
        model_id=registration.model_id,
        cursor_sequence=registration.label_cursor_sequence,
        start_sequence=registration.start_sequence,
        limit=limit,
    )
    missing_label_state = [row.event_id for row in labels if row.prediction_status is None]
    event_store.add_prediction_skips(
        session=session,
        task_name=task_name,
        model_id=registration.model_id,
        event_ids=missing_label_state,
        reason="model-disabled",
    )
    evaluations = [row for row in labels if row.prediction_status == "predicted" and not row.evaluated]
    event_store.add_metric_updates(
        session=session,
        task_name=task_name,
        model_id=registration.model_id,
        event_ids=[row.event_id for row in evaluations],
    )
    untrained = [row for row in labels if not row.trained]
    trained_event_ids = event_store.add_trainings(
        session=session,
        task_name=task_name,
        model_id=registration.model_id,
        event_ids=[row.event_id for row in untrained],
        skipped=True,
    )
    skipped_labels = len(trained_event_ids)
    if labels:
        last = labels[-1]
        advance_model_checkpoint(
            session=session,
            task_name=task_name,
            registration=registration,
            previous_sequence=registration.label_cursor_sequence,
            ready_sequence=last.sequence,
        )
        registration.label_cursor_sequence = last.sequence
    registration.skipped_predictions += skipped_predictions
    registration.skipped_labels += skipped_labels
    return skipped_predictions, skipped_labels


def model_registration(*, session: Session, task_name: str, model_id: str) -> ModelRegistration | None:
    return session.get(ModelRegistration, {"task_name": task_name, "model_id": model_id})


def store_artifact(*, session: Session, payload: bytes, signature: str, metadata: dict[str, Any]) -> ModelArtifact:
    checksum = artifacts.sha256(payload=payload)
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
    *, artifact_record: ModelArtifact, task_name: str, definition: dict[str, Any], examples: int
) -> None:
    metadata = dict(artifact_record.metadata_ or {})
    validations = dict(metadata.get("validations") or {})
    validations[task_name] = {"definition": definition, "examples": examples}
    metadata["validations"] = validations
    artifact_record.metadata_ = metadata


def artifact(*, session: Session, artifact_id: str) -> ModelArtifact | None:
    return session.get(ModelArtifact, artifact_id)


def latest_snapshot(*, session: Session, task_name: str, model_id: str) -> ModelSnapshot | None:
    return session.scalar(
        select(ModelSnapshot).where(ModelSnapshot.task_name == task_name, ModelSnapshot.model_id == model_id)
    )


def trained_examples_since_checkpoint(
    *, session: Session, task_name: str, model_id: str, checkpoint_ready_sequence: int
) -> Iterator[LabelledExample]:
    """Stream committed learning not yet included in the saved model state."""
    rows = session.execute(
        text(
            """SELECT event.event_id, event.event, label.y
           FROM benchmark_model_events AS state
           JOIN benchmark_events AS event USING (task_name, event_id)
           JOIN benchmark_labels AS label USING (task_name, event_id)
           JOIN benchmark_ready_labels AS ready USING (task_name, event_id)
           JOIN benchmark_models AS model USING (task_name, model_id)
           WHERE state.task_name = :task AND state.model_id = :model
             AND state.trained_at IS NOT NULL AND label.y <> 'null'::jsonb
             AND NOT state.training_skipped
             AND event.sequence >= model.start_sequence AND ready.sequence > :checkpoint
           ORDER BY ready.sequence"""
        ),
        {"task": task_name, "model": model_id, "checkpoint": checkpoint_ready_sequence},
        execution_options={"yield_per": 500},
    )
    for event_id, payload, target in rows:
        yield LabelledExample(event_id=event_id, payload=payload, target=target)


def save_pickle_snapshot(
    *, session: Session, task_name: str, model_id: str, payload: bytes, checkpoint_ready_sequence: int
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
    # A checkpoint may be byte-identical to the registered champion (notably
    # immediately after an auto bootstrap). Keep the champion's descriptive
    # metadata in that case; the snapshot role is already represented by the
    # model_snapshots row.
    artifact_record = store_artifact(
        session=session, payload=payload, signature=artifacts.sign(payload=payload), metadata={}
    )
    snapshot_metadata = dict(artifact_record.metadata_ or {})
    snapshot_metadata.setdefault("source", "worker-snapshot")
    artifact_record.metadata_ = snapshot_metadata
    statement = (
        insert(ModelSnapshot)
        .values(
            task_name=task_name,
            model_id=model_id,
            artifact_id=artifact_record.artifact_id,
            checkpoint_ready_sequence=checkpoint_ready_sequence,
        )
        .on_conflict_do_update(
            index_elements=["task_name", "model_id"],
            set_={
                "artifact_id": artifact_record.artifact_id,
                "checkpoint_ready_sequence": checkpoint_ready_sequence,
                "created_at": func.now(),
            },
        )
    )
    session.execute(statement)
    stale_ids = {previous_artifact_id} - {artifact_record.artifact_id} if previous_artifact_id else set()
    _delete_unreferenced_artifacts(session=session, artifact_ids=stale_ids)
    return artifact_record


def active_model_count(*, session: Session, task_name: str) -> int:
    return int(
        session.scalar(
            select(func.count())
            .select_from(ModelRegistration)
            .where(ModelRegistration.task_name == task_name, ModelRegistration.active)
        )
        or 0
    )
