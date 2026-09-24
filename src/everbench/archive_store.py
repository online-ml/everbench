"""Database operations for durable event archives."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import Any

from sqlalchemy import bindparam, delete, select, text
from sqlalchemy.dialects.sqlite import insert
from sqlalchemy.orm import Session

from everbench import event_store
from everbench.schema import JSON_TYPE, ArchiveManifest, BenchmarkEvent, BenchmarkLabel, UTCDateTime


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
    dates = session.scalars(
        select(ArchiveManifest.event_date)
        .where(ArchiveManifest.task_name == task_name, ArchiveManifest.event_date <= cutoff.date() - timedelta(days=7))
        .order_by(ArchiveManifest.event_date.desc())
    )
    for week_start in dates:
        start, end = _week_bounds(week_start=week_start)
        remaining = session.scalar(
            select(BenchmarkEvent.event_id)
            .where(
                BenchmarkEvent.task_name == task_name,
                BenchmarkEvent.inserted_at >= start,
                BenchmarkEvent.inserted_at < end,
            )
            .limit(1)
        )
        if remaining is None:
            return week_start
    return None


def next_archive_week(*, session: Session, task_name: str, cutoff: datetime) -> date | None:
    """Return the oldest UTC availability week old enough to archive."""
    earliest = session.scalar(
        select(BenchmarkEvent.inserted_at)
        .where(BenchmarkEvent.task_name == task_name, BenchmarkEvent.inserted_at < cutoff)
        .order_by(BenchmarkEvent.inserted_at)
        .limit(1)
    )
    return earliest.date() - timedelta(days=earliest.weekday()) if earliest else None


def _week_bounds(*, week_start: date) -> tuple[datetime, datetime]:
    start = datetime.combine(week_start, datetime.min.time(), tzinfo=UTC)
    return start, start + timedelta(days=7)


def archive_week_ready(*, session: Session, task_name: str, week_start: date) -> bool:
    """Return whether every event in a closed availability week is complete."""
    start, end = _week_bounds(week_start=week_start)
    incomplete = session.scalar(
        text(
            f"""SELECT EXISTS (
                   SELECT 1
                     FROM benchmark_events AS event
                     LEFT JOIN benchmark_labels AS label USING (task_name, event_id)
                    WHERE event.task_name = :task_name
                      AND event.inserted_at >= :start AND event.inserted_at < :end
                      AND (
                          label.event_id IS NULL OR EXISTS (
                              SELECT 1 FROM benchmark_models AS model
                               WHERE model.task_name = event.task_name AND model.active
                                 AND event.sequence >= model.start_sequence
                                 AND {event_store._model_processing_pending_clause()}
                          )
                      )
               )"""
        ).bindparams(bindparam("start", type_=UTCDateTime()), bindparam("end", type_=UTCDateTime())),
        {"task_name": task_name, "start": start, "end": end},
    )
    return not bool(incomplete)


def archive_rows(*, session: Session, task_name: str, week_start: date) -> list[dict[str, Any]]:
    """Load one complete availability week in deterministic replay order."""
    start, end = _week_bounds(week_start=week_start)
    rows = session.execute(
        text(
            f"""SELECT event.event_id, event.sequence, event.event_time, event.inserted_at, event.event,
                      label.y, label.reason, label.available_at
               FROM benchmark_events AS event
               JOIN benchmark_labels AS label USING (task_name, event_id)
               WHERE event.task_name = :task_name
                 AND event.inserted_at >= :start AND event.inserted_at < :end
                 AND NOT EXISTS (
                   SELECT 1 FROM benchmark_models AS model
                   WHERE model.task_name = event.task_name AND model.active
                     AND event.sequence >= model.start_sequence
                     AND {event_store._model_processing_pending_clause()}
                 )
               ORDER BY event.inserted_at, event.sequence"""
        )
        .bindparams(bindparam("start", type_=UTCDateTime()), bindparam("end", type_=UTCDateTime()))
        .columns(
            event=JSON_TYPE,
            y=JSON_TYPE,
            event_time=UTCDateTime(),
            inserted_at=UTCDateTime(),
            available_at=UTCDateTime(),
        ),
        {"task_name": task_name, "start": start, "end": end},
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
        .on_conflict_do_nothing(index_elements=["task_name", "event_date"])
        .returning(ArchiveManifest.content_sha256)
    )
    return inserted is not None


def purge_archived_events(*, session: Session, task_name: str, event_ids: list[str]) -> None:
    """Only call after a manifest was committed for a durable archive target."""
    if not event_ids:
        return
    for offset in range(0, len(event_ids), 500):
        ids = event_ids[offset : offset + 500]
        session.execute(
            delete(BenchmarkLabel).where(BenchmarkLabel.task_name == task_name, BenchmarkLabel.event_id.in_(ids))
        )
        session.execute(
            delete(BenchmarkEvent).where(BenchmarkEvent.task_name == task_name, BenchmarkEvent.event_id.in_(ids))
        )
