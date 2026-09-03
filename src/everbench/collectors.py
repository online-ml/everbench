"""Event and label stream collectors."""

from __future__ import annotations

import logging
import threading
import time
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy.orm import Session, sessionmaker

from everbench import event_store
from everbench.batching import TimedBatch
from everbench.config import CONFIG
from everbench.heartbeat import Heartbeat
from everbench.hotstore import HotStore
from everbench.sse import StreamMessage, subscribe
from everbench.tasks import TaskDefinition


def _timestamp(task: TaskDefinition, event: dict) -> float:
    extractor = task.event_timestamp
    return float(extractor(event) if extractor else event.get("timestamp", time.time()))


def _label_timestamp(task: TaskDefinition, event: dict) -> datetime:
    extractor = task.label_timestamp
    timestamp = float(extractor(event)) if extractor else time.time()
    return datetime.fromtimestamp(timestamp, UTC)


@dataclass
class StreamCursorState:
    value: str | None


def _stream(
    task: TaskDefinition,
    source_name: str,
    stream_name: str,
    url: str,
    stop: threading.Event,
    cursor_state: StreamCursorState | None,
):
    """Use a task's local generator when supplied, otherwise subscribe to SSE."""
    source = getattr(task, source_name)
    if source is not None:
        return (StreamMessage(payload=event, event_id=None) for event in source(stop))
    if cursor_state is None:
        raise RuntimeError("SSE streams require a durable cursor state")
    return subscribe(stream_name, url, stop=stop, last_event_id=lambda: cursor_state.value)


def _cursor_state(
    sessions: sessionmaker[Session], task: TaskDefinition, source_name: str, stream_name: str
) -> StreamCursorState | None:
    if getattr(task, source_name) is not None:
        return None
    with sessions() as session:
        return StreamCursorState(event_store.stream_cursor(session, task.TASK_NAME, stream_name))


def _last_cursor(items: list[tuple[Any, str | None]]) -> str | None:
    for _, event_id in reversed(items):
        if event_id is not None:
            return event_id
    return None


def _flush_before_exit(batch: TimedBatch[Any]) -> None:
    """Drain a collector batch during orderly shutdown or fail visibly."""
    deadline = time.monotonic() + CONFIG.shutdown_flush_seconds
    while not batch.flush():
        if time.monotonic() >= deadline:
            raise RuntimeError("could not checkpoint collector batch before shutdown")
        time.sleep(0.25)


def _flush_on_timer(batch: TimedBatch[Any], stop: threading.Event) -> None:
    """Give a sparse stream the same bounded write latency as a busy stream."""
    while not stop.wait(CONFIG.ingest_flush_seconds):
        batch.flush_if_due()


def _cache_durable_events(
    hot: HotStore | None, events: list[tuple[str, float, dict[str, Any]]], inserted_event_ids: list[str]
) -> None:
    """Populate RAM only after the event transaction has committed.

    A repeated source message may carry a different payload for an existing
    event ID. The database's uniqueness constraint decides the canonical
    record, so the cache must only receive newly inserted rows.
    """
    if hot is None or not inserted_event_ids:
        return
    events_by_id = {event_id: event for event_id, _, event in events}
    for event_id in inserted_event_ids:
        hot.put(event_id, events_by_id[event_id])


def collect_events(
    sessions: sessionmaker[Session],
    task: TaskDefinition,
    stop: threading.Event | None = None,
    hot: HotStore | None = None,
    heartbeat: bool = True,
) -> None:
    stop = stop or threading.Event()
    delay_seconds = task.NEGATIVE_LABEL_DELAY_SECONDS
    cursor_state = _cursor_state(sessions, task, "event_stream", "events")

    def flush(items: list[tuple[tuple[str, float, dict[str, Any]] | None, str | None]]) -> None:
        events = [event for event, _ in items if event is not None]
        cursor = _last_cursor(items)
        with sessions.begin() as session:
            event_store.lock_task_ingest(session, task.TASK_NAME)
            inserted_event_ids = event_store.add_events(session, task.TASK_NAME, events, delay_seconds)
            if cursor_state is not None and cursor is not None:
                event_store.save_stream_cursor(session, task.TASK_NAME, "events", cursor)
        _cache_durable_events(hot, events, inserted_event_ids)
        if cursor_state is not None and cursor is not None:
            cursor_state.value = cursor
        logging.debug("flushed %d/%d events", len(inserted_event_ids), len(events))

    batch = TimedBatch(CONFIG.ingest_batch_size, CONFIG.ingest_flush_seconds, flush, CONFIG.ingest_max_pending_items)
    with Heartbeat(sessions, task.TASK_NAME, "event-collector") if heartbeat else nullcontext():
        timer = threading.Thread(target=_flush_on_timer, args=(batch, stop), name="event-batch-flush", daemon=True)
        timer.start()
        try:
            for message in _stream(
                task, "event_stream", f"{task.TASK_NAME}: events", task.EVENT_STREAM_URL, stop, cursor_state
            ):
                event = message.payload
                if not task.accepts_event(event):
                    batch.add((None, message.event_id))
                    continue
                event_id = task.event_id(event)
                if event_id is not None:
                    event_time = _timestamp(task, event)
                    batch.add(((event_id, event_time, event), message.event_id))
                else:
                    batch.add((None, message.event_id))
        finally:
            stop.set()
            timer.join(timeout=2)
            _flush_before_exit(batch)


def collect_labels(
    sessions: sessionmaker[Session],
    task: TaskDefinition,
    stop: threading.Event | None = None,
    hot: HotStore | None = None,
    heartbeat: bool = True,
) -> None:
    stop = stop or threading.Event()
    delay_seconds = task.NEGATIVE_LABEL_DELAY_SECONDS
    cursor_state = _cursor_state(sessions, task, "label_stream", "labels")

    def maintain_labels() -> None:
        while not stop.is_set():
            try:
                with sessions.begin() as session:
                    event_store.lock_task_ingest(session, task.TASK_NAME)
                    event_ids = event_store.add_expired_negative_labels(session, task.TASK_NAME, delay_seconds)
                    purged = event_store.purge_orphan_labels(
                        session,
                        task.TASK_NAME,
                        datetime.now(UTC) - timedelta(days=CONFIG.label_inbox_retention_days),
                    )
                if event_ids:
                    if hot is not None:
                        hot.mark_labelled(event_ids)
                    logging.info("queued %d horizon labels", len(event_ids))
                if purged:
                    logging.info("purged %d expired orphan labels", purged)
            except Exception:
                logging.exception("label maintenance failed")
            stop.wait(60)

    maintenance = threading.Thread(target=maintain_labels, name="label-maintenance", daemon=True)
    maintenance.start()

    def flush(items: list[tuple[event_store.LabelInput | None, str | None]]) -> None:
        labels = [label for label, _ in items if label is not None]
        cursor = _last_cursor(items)
        with sessions.begin() as session:
            event_store.lock_task_ingest(session, task.TASK_NAME)
            inserted_event_ids = event_store.add_labels(session, task.TASK_NAME, labels, delay_seconds)
            if cursor_state is not None and cursor is not None:
                event_store.save_stream_cursor(session, task.TASK_NAME, "labels", cursor)
        if cursor_state is not None and cursor is not None:
            cursor_state.value = cursor
        if hot is not None:
            hot.mark_labelled(inserted_event_ids)
        logging.debug("flushed %d/%d labels", len(inserted_event_ids), len(labels))

    batch = TimedBatch(CONFIG.ingest_batch_size, CONFIG.ingest_flush_seconds, flush, CONFIG.ingest_max_pending_items)
    with Heartbeat(sessions, task.TASK_NAME, "label-collector") if heartbeat else nullcontext():
        timer = threading.Thread(target=_flush_on_timer, args=(batch, stop), name="label-batch-flush", daemon=True)
        timer.start()
        try:
            for message in _stream(
                task, "label_stream", f"{task.TASK_NAME}: labels", task.LABEL_STREAM_URL, stop, cursor_state
            ):
                event = message.payload
                label = task.label_for(event)
                collected = event_store.LabelInput(*label, _label_timestamp(task, event)) if label is not None else None
                batch.add((collected, message.event_id))
        finally:
            stop.set()
            timer.join(timeout=2)
            maintenance.join(timeout=2)
            _flush_before_exit(batch)
