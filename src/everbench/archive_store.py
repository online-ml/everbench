"""Database operations for durable event archives."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from sqlalchemy import select, text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from everbench import event_store
from everbench.schema import ArchiveManifest


def task_archives(*, session: Session, task_name: str) -> list[ArchiveManifest]:
    return list(
        session.scalars(
            select(ArchiveManifest)
            .where(ArchiveManifest.task_name == task_name)
            .order_by(ArchiveManifest.event_date.desc(), ArchiveManifest.created_at.desc())
        )
    )


def task_archive(*, session: Session, task_name: str, content_sha256: str) -> ArchiveManifest | None:
    return session.scalar(
        select(ArchiveManifest).where(
            ArchiveManifest.task_name == task_name, ArchiveManifest.content_sha256 == content_sha256
        )
    )


def archive_for_week(*, session: Session, task_name: str, event_date: date) -> ArchiveManifest | None:
    return session.scalar(
        select(ArchiveManifest).where(
            ArchiveManifest.task_name == task_name,
            ArchiveManifest.event_date == event_date,
        )
    )


def latest_complete_archive_week(*, session: Session, task_name: str, cutoff: datetime) -> date | None:
    """Return the newest closed availability week whose rows are archived."""
    return session.scalar(
        text(
            """SELECT max(manifest.event_date)
                 FROM archive_manifest AS manifest
                WHERE manifest.task_name = :task_name
                  AND manifest.event_date + 7 <= CAST(:cutoff AS date)
                  AND NOT EXISTS (
                      SELECT 1 FROM benchmark_events AS event
                       WHERE event.task_name = manifest.task_name
                         AND event.inserted_at >= (
                             manifest.event_date::timestamp AT TIME ZONE 'UTC'
                         )
                         AND event.inserted_at < (
                             (manifest.event_date + 7)::timestamp AT TIME ZONE 'UTC'
                         )
                  )"""
        ),
        {"task_name": task_name, "cutoff": cutoff},
    )


def next_archive_week(*, session: Session, task_name: str, cutoff: datetime) -> date | None:
    """Return the oldest UTC availability week old enough to archive."""
    return session.scalar(
        text(
            """SELECT date_trunc('week', event.inserted_at AT TIME ZONE 'UTC')::date
               FROM benchmark_events AS event
               WHERE event.task_name = :task_name AND event.inserted_at < :cutoff
               ORDER BY event.inserted_at LIMIT 1"""
        ),
        {"task_name": task_name, "cutoff": cutoff},
    )


def archive_week_ready(*, session: Session, task_name: str, week_start: date) -> bool:
    """Return whether every event in a closed availability week is complete."""
    incomplete = session.scalar(
        text(
            f"""SELECT EXISTS (
                   SELECT 1
                     FROM benchmark_events AS event
                     LEFT JOIN benchmark_labels AS label USING (task_name, event_id)
                    WHERE event.task_name = :task_name
                      AND event.inserted_at >= (
                          CAST(:week_start AS date)::timestamp AT TIME ZONE 'UTC'
                      )
                      AND event.inserted_at < (
                          (CAST(:week_start AS date) + 7)::timestamp AT TIME ZONE 'UTC'
                      )
                      AND (
                          label.event_id IS NULL OR EXISTS (
                              SELECT 1 FROM benchmark_models AS model
                               WHERE model.task_name = event.task_name AND model.active
                                 AND event.sequence >= model.start_sequence
                                 AND {event_store._model_processing_pending_clause()}
                          )
                      )
               )"""
        ),
        {"task_name": task_name, "week_start": week_start},
    )
    return not bool(incomplete)


def archive_rows(*, session: Session, task_name: str, week_start: date) -> list[dict[str, Any]]:
    """Load one complete availability week in deterministic replay order."""
    rows = session.execute(
        text(
            f"""SELECT event.event_id, event.sequence, event.event_time, event.inserted_at, event.event,
                      label.y, label.reason, label.available_at
               FROM benchmark_events AS event
               JOIN benchmark_labels AS label USING (task_name, event_id)
               WHERE event.task_name = :task_name
                 AND event.inserted_at >= (CAST(:week_start AS date)::timestamp AT TIME ZONE 'UTC')
                 AND event.inserted_at < (CAST(:week_start AS date)::timestamp AT TIME ZONE 'UTC') + INTERVAL '7 days'
                 AND NOT EXISTS (
                   SELECT 1 FROM benchmark_models AS model
                   WHERE model.task_name = event.task_name AND model.active
                     AND event.sequence >= model.start_sequence
                     AND {event_store._model_processing_pending_clause()}
                 )
               ORDER BY event.inserted_at, event.sequence"""
        ),
        {"task_name": task_name, "week_start": week_start},
    ).mappings()
    return [dict(row) for row in rows]


def record_archive(
    *,
    session: Session,
    content_sha256: str,
    task_name: str,
    event_date: date,
    path: str,
    row_count: int,
    byte_size: int,
    label_count: int | None = None,
) -> bool:
    inserted = session.scalar(
        insert(ArchiveManifest)
        .values(
            content_sha256=content_sha256,
            task_name=task_name,
            event_date=event_date,
            path=path,
            row_count=row_count,
            byte_size=byte_size,
            label_count=row_count if label_count is None else label_count,
        )
        .on_conflict_do_nothing(constraint="archive_manifest_task_week_key")
        .returning(ArchiveManifest.content_sha256)
    )
    return inserted is not None


def purge_archived_events(*, session: Session, task_name: str, event_ids: list[str]) -> None:
    """Only call after a manifest was committed for a durable archive target."""
    if not event_ids:
        return
    # Bind the IDs as one PostgreSQL array. Expanding a 100k-row archive batch
    # into an IN clause exceeds PostgreSQL's 65,535 bind-parameter limit.
    for table in ("benchmark_labels", "benchmark_events"):
        session.execute(
            text(
                f"""DELETE FROM {table}
                     WHERE task_name = :task_name
                       AND event_id = ANY(CAST(:event_ids AS text[]))"""
            ),
            {"task_name": task_name, "event_ids": event_ids},
        )
