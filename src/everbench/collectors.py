"""Event and label stream collectors."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from threading import RLock
from typing import Any, Generic, TypeVar

from sqlalchemy.orm import Session, sessionmaker

from everbench import event_store
from everbench.batching import TimedBatch
from everbench.config import CONFIG
from everbench.hotstore import HotStore
from everbench.records import LabelInput, Observation
from everbench.sources import Input, Source
from everbench.tasks import TaskDefinition


@dataclass(kw_only=True)
class StreamCursorState:
    value: str | None


BatchItem = TypeVar("BatchItem")


class CollectorBatch(Generic[BatchItem]):
    """Batch accepted records while checkpointing filtered SSE traffic sparsely."""

    def __init__(
        self,
        *,
        cursor_state: StreamCursorState | None,
        flush: Callable[..., None],
        checkpoint: Callable[..., None] | None,
    ) -> None:
        self.cursor_state = cursor_state
        self.latest_cursor = cursor_state.value if cursor_state is not None else None
        self._checkpoint = checkpoint
        self._checkpointed_at = time.monotonic()
        self._lock = RLock()
        self._batch = TimedBatch(
            max_items=CONFIG.ingest_batch_size,
            max_age_seconds=CONFIG.ingest_flush_seconds,
            flush=self._flush_items,
            max_pending_items=CONFIG.ingest_max_pending_items,
        )
        self._flush = flush

    def observe(self, *, item: BatchItem | None, cursor: str | None) -> None:
        with self._lock:
            if cursor is not None:
                self.latest_cursor = cursor
            if item is not None:
                self._batch.add(item=item)

    def _flush_items(self, *, items: list[BatchItem]) -> None:
        self._flush(items=items, cursor=self.latest_cursor)
        self._mark_cursor_durable()

    def _mark_cursor_durable(self) -> None:
        if self.cursor_state is not None and self.latest_cursor is not None:
            self.cursor_state.value = self.latest_cursor
        self._checkpointed_at = time.monotonic()

    def _checkpoint_cursor(self) -> bool:
        if self._batch.items and not self._batch.flush():
            return False
        if (
            self._checkpoint is not None
            and self.latest_cursor is not None
            and self.cursor_state is not None
            and self.latest_cursor != self.cursor_state.value
        ):
            self._checkpoint(cursor=self.latest_cursor)
            self._mark_cursor_durable()
        return True

    def tick(self) -> None:
        with self._lock:
            self._batch.flush_if_due()
            if time.monotonic() - self._checkpointed_at >= CONFIG.stream_cursor_checkpoint_seconds:
                self._checkpoint_cursor()

    def flush(self) -> bool:
        with self._lock:
            return self._checkpoint_cursor()


def _flush_before_exit(*, batch: CollectorBatch[Any]) -> None:
    """Drain a collector batch during orderly shutdown or fail visibly."""
    deadline = time.monotonic() + CONFIG.shutdown_flush_seconds
    while not batch.flush():
        if time.monotonic() >= deadline:
            raise RuntimeError("could not checkpoint collector batch before shutdown")
        time.sleep(0.25)


def _flush_on_timer(*, batch: CollectorBatch[Any], stop: threading.Event) -> None:
    """Give a sparse stream the same bounded write latency as a busy stream."""
    interval = min(CONFIG.ingest_flush_seconds, CONFIG.stream_cursor_checkpoint_seconds)
    while not stop.wait(interval):
        try:
            batch.tick()
        except Exception:
            logging.exception("collector cursor checkpoint failed")


def _cache_durable_events(*, hot: HotStore | None, events: list[Observation], inserted_event_ids: list[str]) -> None:
    if hot is not None:
        accepted = set(inserted_event_ids)
        for event in events:
            if event.event_id in accepted:
                hot.put(event_id=event.event_id, event=event.payload)


def collect_source(
    *,
    sessions: sessionmaker[Session],
    task: TaskDefinition,
    source: Source,
    stop: threading.Event,
    hot: HotStore | None = None,
) -> None:
    with sessions() as session:
        cursor_state = StreamCursorState(
            value=event_store.stream_cursor(session=session, task_name=task.TASK_NAME, stream_name=source.name)
        )

    def flush(*, items: list[Input], cursor: str | None) -> None:
        observations = [record for record in items if isinstance(record, Observation)]
        labels = [record for record in items if isinstance(record, LabelInput)]
        with sessions.begin() as session:
            event_store.lock_task_ingest(session=session, task_name=task.TASK_NAME)
            inserted = event_store.add_events(
                session=session, task_name=task.TASK_NAME, events=observations, policy=task.label_policy
            )
            resolved = event_store.add_labels(
                session=session, task_name=task.TASK_NAME, labels=labels, policy=task.label_policy
            )
            if cursor is not None:
                event_store.save_stream_cursor(
                    session=session, task_name=task.TASK_NAME, stream_name=source.name, event_id=cursor
                )
        _cache_durable_events(hot=hot, events=observations, inserted_event_ids=inserted)
        if hot is not None:
            hot.mark_labelled(event_ids=resolved)

    def checkpoint(*, cursor: str) -> None:
        with sessions.begin() as session:
            event_store.save_stream_cursor(
                session=session, task_name=task.TASK_NAME, stream_name=source.name, event_id=cursor
            )

    batch = CollectorBatch(cursor_state=cursor_state, flush=flush, checkpoint=checkpoint)
    timer = threading.Thread(
        target=_flush_on_timer, kwargs={"batch": batch, "stop": stop}, name=f"{source.name}-flush", daemon=True
    )
    timer.start()
    try:
        for message in source.read(stop=stop, cursor=lambda: cursor_state.value):
            # Do not checkpoint a source message until every decoded record is durable.
            for record in message.records:
                batch.observe(item=record, cursor=None)
            batch.observe(item=None, cursor=message.cursor)
    finally:
        stop.set()
        timer.join(timeout=2)
        _flush_before_exit(batch=batch)


def maintain_resolutions(
    *, sessions: sessionmaker[Session], task: TaskDefinition, stop: threading.Event, hot: HotStore | None = None
) -> None:
    """Resolve deadlines without polling or retaining source history."""
    while not stop.is_set():
        with sessions.begin() as session:
            event_store.lock_task_ingest(session=session, task_name=task.TASK_NAME)
            resolved = event_store.resolve_due(
                session=session, task_name=task.TASK_NAME, policy=task.label_policy, limit=CONFIG.ingest_batch_size
            )
            event_store.purge_orphan_labels(
                session=session,
                task_name=task.TASK_NAME,
                cutoff=datetime.now(UTC) - timedelta(days=CONFIG.label_inbox_retention_days),
            )
        if hot is not None:
            hot.mark_labelled(event_ids=resolved)
        if len(resolved) < CONFIG.ingest_batch_size:
            stop.wait(60)
