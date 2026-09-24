"""Read models and operational state for reports and the HTTP dashboard."""

from __future__ import annotations

from typing import Any

from sqlalchemy import func, select, text
from sqlalchemy.dialects.sqlite import insert
from sqlalchemy.orm import Session

from everbench.auto import store as auto_store
from everbench.auto.documentation import documented_source
from everbench.schema import (
    JSON_TYPE,
    ArchiveManifest,
    AutoExperiment,
    BenchmarkEvent,
    BenchmarkLabel,
    TaskRegistration,
    UTCDateTime,
    WorkerHeartbeat,
)


def record_heartbeat(
    *,
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


def worker_health(*, session: Session) -> list[WorkerHeartbeat]:
    return list(session.scalars(select(WorkerHeartbeat).order_by(WorkerHeartbeat.role, WorkerHeartbeat.worker_id)))


def task_names(*, session: Session) -> list[str]:
    """List task names registered at startup or first accepted event."""
    return list(session.scalars(select(TaskRegistration.task_name).order_by(TaskRegistration.task_name)))


def register_tasks(*, session: Session, task_names: list[str]) -> None:
    """Make startup registration safe when web and worker services start together."""
    if task_names:
        session.execute(
            insert(TaskRegistration)
            .values([{"task_name": task_name} for task_name in task_names])
            .on_conflict_do_nothing(index_elements=["task_name"])
        )


def task_stats(*, session: Session, task_name: str) -> dict[str, int]:
    live_events = session.scalar(
        select(func.count()).select_from(BenchmarkEvent).where(BenchmarkEvent.task_name == task_name)
    )
    live_labels = session.scalar(
        select(func.count())
        .select_from(BenchmarkLabel)
        .join(
            BenchmarkEvent,
            (BenchmarkEvent.task_name == BenchmarkLabel.task_name)
            & (BenchmarkEvent.event_id == BenchmarkLabel.event_id),
        )
        .where(BenchmarkLabel.task_name == task_name, BenchmarkLabel.y != JSON_TYPE.NULL)
    )
    archived_events, archived_labels = session.execute(
        select(
            func.coalesce(func.sum(ArchiveManifest.row_count), 0),
            func.coalesce(func.sum(func.coalesce(ArchiveManifest.label_count, ArchiveManifest.row_count)), 0),
        ).where(ArchiveManifest.task_name == task_name)
    ).one()
    return {
        "events": int(live_events or 0) + int(archived_events),
        "labels": int(live_labels or 0) + int(archived_labels),
    }


def task_leaderboard(*, session: Session, task_name: str) -> list[dict[str, Any]]:
    rows = session.execute(
        text(
            """SELECT model.model_id,
                      model.owner,
                      model.active,
                      model.failure_count,
                      model.last_error,
                      model.failed_at,
                      model.disabled_until,
                      model.error_count,
                      model.skipped_predictions + model.skipped_labels AS skipped,
                      model.created_at,
                      COALESCE(metric_state.predictions, 0) AS predictions,
                      COALESCE(metric_state.observations, 0) AS labels,
                      COALESCE(metric_state."values", '{}') AS metrics,
                      COALESCE(length(snapshot_artifact.payload), length(artifact.payload), 0) AS model_bytes
               FROM benchmark_models AS model
               LEFT JOIN benchmark_metric_state AS metric_state
                 ON metric_state.task_name = model.task_name AND metric_state.model_id = model.model_id
               LEFT JOIN model_artifacts AS artifact ON artifact.artifact_id = model.artifact_id
               LEFT JOIN model_snapshots AS snapshot
                 ON snapshot.task_name = model.task_name AND snapshot.model_id = model.model_id
               LEFT JOIN model_artifacts AS snapshot_artifact ON snapshot_artifact.artifact_id = snapshot.artifact_id
               WHERE model.task_name = :task_name AND model.active
               ORDER BY model.model_id"""
        ).columns(metrics=JSON_TYPE, created_at=UTCDateTime(), failed_at=UTCDateTime(), disabled_until=UTCDateTime()),
        {"task_name": task_name},
    ).mappings()
    return [dict(row) for row in rows]


def model_detail(*, session: Session, task_name: str, model_id: str) -> dict[str, Any] | None:
    row = (
        session.execute(
            text(
                """SELECT model.model_id,
                      model.owner,
                      model.created_at,
                      artifact.metadata AS artifact_metadata,
                      COALESCE(
                          artifact.metadata ->> 'class_definition',
                          artifact.metadata ->> 'source_code',
                          ''
                      ) AS class_definition,
                      COALESCE(artifact.metadata ->> 'class_name', 'pickle') AS class_name
                 FROM benchmark_models AS model
                 LEFT JOIN model_artifacts AS artifact ON artifact.artifact_id = model.artifact_id
                WHERE model.task_name = :task_name AND model.model_id = :model_id AND model.active"""
            ).columns(artifact_metadata=JSON_TYPE, created_at=UTCDateTime()),
            {"task_name": task_name, "model_id": model_id},
        )
        .mappings()
        .one_or_none()
    )
    if row is None:
        return None
    detail = dict(row)
    metadata = detail.pop("artifact_metadata") or {}
    is_auto_artifact = metadata.get("source") in {"auto-bootstrap", "auto-promotion"} or (
        detail["owner"] == "everbench-auto" and "generation" in metadata and "source_code" in metadata
    )
    if is_auto_artifact:
        counts = {
            status: count
            for status, count in session.execute(
                select(AutoExperiment.status, func.count())
                .where(AutoExperiment.task_name == task_name, AutoExperiment.model_id == model_id)
                .group_by(AutoExperiment.status)
            )
        }
        detail["class_definition"] = documented_source(
            source=detail["class_definition"],
            metadata=metadata,
            experiments=auto_store.recent_experiments(session=session, task_name=task_name, model_id=model_id, limit=5),
            counts=counts,
        )
    return detail
