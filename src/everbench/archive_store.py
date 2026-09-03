"""Database operations for durable event archives."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from sqlalchemy import delete, select, text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from everbench import event_store
from everbench.schema import ArchiveManifest, BenchmarkEvent, BenchmarkLabel


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
                     AND {event_store._model_processing_pending_clause()}
                 )
               ORDER BY event.event_time LIMIT 1"""
        ),
        {"task_name": task_name, "cutoff": cutoff},
    )


def archive_rows(
    session: Session, task_name: str, week_start: date, cutoff: datetime, limit: int
) -> list[dict[str, Any]]:
    rows = session.execute(
        text(
            f"""SELECT event.event_id, event.sequence, event.event_time, event.inserted_at, event.event,
                      label.y, label.reason, label.available_at
               FROM benchmark_events AS event
               JOIN benchmark_labels AS label USING (task_name, event_id)
               WHERE event.task_name = :task_name
                 AND event.event_time >= (CAST(:week_start AS date)::timestamp AT TIME ZONE 'UTC')
                 AND event.event_time < (CAST(:week_start AS date)::timestamp AT TIME ZONE 'UTC') + INTERVAL '7 days'
                 AND event.event_time < :cutoff
                 AND NOT EXISTS (
                   SELECT 1 FROM benchmark_models AS model
                   WHERE model.task_name = event.task_name AND model.active
                     AND event.sequence >= model.start_sequence
                     AND {event_store._model_processing_pending_clause()}
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
    for model in (BenchmarkLabel, BenchmarkEvent):
        session.execute(delete(model).where(model.task_name == task_name, model.event_id.in_(event_ids)))
