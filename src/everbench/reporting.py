"""Read models and operational state for reports and the HTTP dashboard."""

from __future__ import annotations

from typing import Any

from sqlalchemy import func, select, text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from everbench.schema import WorkerHeartbeat


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
                      model.disabled_until,
                      model.error_count,
                      model.skipped_predictions + model.skipped_labels AS skipped,
                      model.created_at,
                      COALESCE(metric_state.predictions, 0) AS predictions,
                      COALESCE(metric_state.observations, 0) AS labels,
                      COALESCE(metric_state.values, '{}'::jsonb) AS metrics,
                      COALESCE(octet_length(snapshot_artifact.payload), octet_length(artifact.payload), 0) AS model_bytes
               FROM benchmark_models AS model
               LEFT JOIN benchmark_metric_state AS metric_state
                 ON metric_state.task_name = model.task_name AND metric_state.model_id = model.model_id
               LEFT JOIN model_artifacts AS artifact ON artifact.artifact_id = model.artifact_id
               LEFT JOIN model_snapshots AS snapshot
                 ON snapshot.task_name = model.task_name AND snapshot.model_id = model.model_id
               LEFT JOIN model_artifacts AS snapshot_artifact ON snapshot_artifact.artifact_id = snapshot.artifact_id
               WHERE model.task_name = :task_name AND model.active
               ORDER BY model.model_id"""
        ),
        {"task_name": task_name},
    ).mappings()
    return [dict(row) for row in rows]


def model_detail(session: Session, task_name: str, model_id: str) -> dict[str, Any] | None:
    row = (
        session.execute(
            text(
                """SELECT model.model_id,
                      model.owner,
                      model.created_at,
                      COALESCE(artifact.metadata ->> 'class_definition', '') AS class_definition,
                      COALESCE(artifact.metadata ->> 'class_name', 'pickle') AS class_name
                 FROM benchmark_models AS model
                 LEFT JOIN model_artifacts AS artifact ON artifact.artifact_id = model.artifact_id
                WHERE model.task_name = :task_name AND model.model_id = :model_id AND model.active"""
            ),
            {"task_name": task_name, "model_id": model_id},
        )
        .mappings()
        .one_or_none()
    )
    return dict(row) if row is not None else None
