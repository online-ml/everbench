"""Long-running event, label, and learner worker implementations."""

from __future__ import annotations

import logging
import socket
import threading
import time
import zlib
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from types import ModuleType
from typing import Any

from sqlalchemy import text
from sqlalchemy.orm import Session, sessionmaker

from everbench import artifacts, store
from everbench.batching import TimedBatch
from everbench.config import CONFIG
from everbench.hotstore import HotStore
from everbench.metrics import MetricTracker, metric_definition
from everbench.models import PickledModel, metric_inputs_for, prediction_for, supports_learning
from everbench.sse import StreamMessage, subscribe


class Heartbeat(AbstractContextManager):
    """Writes a shared, durable liveness signal while a worker is running."""

    def __init__(
        self, sessions: sessionmaker[Session], task_name: str | None, role: str, detail: Callable[[], str] | None = None
    ):
        self.sessions = sessions
        self.task_name = task_name
        self.role = role
        self.detail = detail
        self.worker_id = f"{socket.gethostname()}:{task_name or 'global'}:{role}"
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, name=f"heartbeat-{role}", daemon=True)

    def _run(self) -> None:
        while not self.stop.is_set():
            try:
                with self.sessions.begin() as session:
                    store.record_heartbeat(
                        session,
                        self.worker_id,
                        self.task_name,
                        self.role,
                        detail=self.detail() if self.detail else None,
                    )
            except Exception:
                logging.exception("failed to record %s heartbeat", self.role)
            self.stop.wait(CONFIG.heartbeat_seconds)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args) -> None:
        self.stop.set()
        self.thread.join(timeout=2)


def _timestamp(task: ModuleType, event: dict) -> float:
    extractor = getattr(task, "event_timestamp", None)
    return float(extractor(event) if extractor else event.get("timestamp", time.time()))


@dataclass
class StreamCursorState:
    value: str | None


def _stream(
    task: ModuleType,
    source_name: str,
    stream_name: str,
    url: str,
    stop: threading.Event,
    cursor_state: StreamCursorState | None,
):
    """Use a task's local generator when supplied, otherwise subscribe to SSE."""
    source = getattr(task, source_name, None)
    if source is not None:
        return (StreamMessage(payload=event, event_id=None) for event in source(stop))
    if cursor_state is None:
        raise RuntimeError("SSE streams require a durable cursor state")
    return subscribe(stream_name, url, stop=stop, last_event_id=lambda: cursor_state.value)


def _cursor_state(
    sessions: sessionmaker[Session], task: ModuleType, source_name: str, stream_name: str
) -> StreamCursorState | None:
    if getattr(task, source_name, None) is not None:
        return None
    with sessions() as session:
        return StreamCursorState(store.stream_cursor(session, task.TASK_NAME, stream_name))


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
        hot.put_event(event_id, events_by_id[event_id])


def collect_events(
    sessions: sessionmaker[Session],
    task: ModuleType,
    stop: threading.Event | None = None,
    hot: HotStore | None = None,
    heartbeat: bool = True,
) -> None:
    stop = stop or threading.Event()
    delay_seconds = getattr(task, "NEGATIVE_LABEL_DELAY_SECONDS", None)
    cursor_state = _cursor_state(sessions, task, "event_stream", "events")

    def flush(items: list[tuple[tuple[str, float, dict[str, Any]] | None, str | None]]) -> None:
        events = [event for event, _ in items if event is not None]
        cursor = _last_cursor(items)
        with sessions.begin() as session:
            inserted_event_ids = store.add_events(session, task.TASK_NAME, events, delay_seconds)
            if cursor_state is not None and cursor is not None:
                store.save_stream_cursor(session, task.TASK_NAME, "events", cursor)
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
    task: ModuleType,
    stop: threading.Event | None = None,
    hot: HotStore | None = None,
    heartbeat: bool = True,
) -> None:
    stop = stop or threading.Event()
    delay_seconds = getattr(task, "NEGATIVE_LABEL_DELAY_SECONDS", None)
    cursor_state = _cursor_state(sessions, task, "label_stream", "labels")

    def finalize_negatives() -> None:
        while not stop.is_set():
            try:
                with sessions.begin() as session:
                    event_ids = store.add_expired_negative_labels(session, task.TASK_NAME, delay_seconds)
                if event_ids:
                    if hot is not None:
                        hot.mark_labelled(event_ids)
                    logging.info("queued %d horizon labels", len(event_ids))
            except Exception:
                logging.exception("negative-label finalizer failed")
            stop.wait(60)

    finalizer: threading.Thread | None = None
    if delay_seconds is not None:
        finalizer = threading.Thread(target=finalize_negatives, name="negative-label-finalizer", daemon=True)
        finalizer.start()

    def flush(items: list[tuple[tuple[str, Any, str] | None, str | None]]) -> None:
        labels = [label for label, _ in items if label is not None]
        cursor = _last_cursor(items)
        with sessions.begin() as session:
            inserted_event_ids = store.add_labels(session, task.TASK_NAME, labels, delay_seconds)
            if cursor_state is not None and cursor is not None:
                store.save_stream_cursor(session, task.TASK_NAME, "labels", cursor)
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
                batch.add((label, message.event_id))
        finally:
            stop.set()
            timer.join(timeout=2)
            if finalizer is not None:
                finalizer.join(timeout=2)
            _flush_before_exit(batch)


def _task_lock(task_name: str) -> int:
    return zlib.crc32(task_name.encode())


def _load_model(session: Session, task: ModuleType, registration):
    snapshot = store.latest_snapshot(session, task.TASK_NAME, registration.model_id)
    artifact_record = store.artifact(session, snapshot.artifact_id) if snapshot is not None else None
    if artifact_record is None and registration.artifact_id:
        artifact_record = store.artifact(session, registration.artifact_id)
    if artifact_record is None:
        raise RuntimeError(f"pickle artifact missing for {registration.model_id}")
    return PickledModel(
        registration.model_id, artifacts.loads(artifact_record.payload, artifact_record.signature)
    ), snapshot


@dataclass
class CachedModel:
    fingerprint: tuple[Any, ...]
    model: PickledModel
    tracker: MetricTracker
    checkpointed_at: float


def _model_operation(model_id: str, operation: str, callback: Callable[[], Any]) -> Any:
    """Apply a finite budget to model calls that return control to Python."""
    started_at = time.monotonic()
    result = callback()
    elapsed = time.monotonic() - started_at
    if elapsed > CONFIG.max_model_operation_seconds:
        raise TimeoutError(
            f"{model_id} {operation} took {elapsed:.1f}s; limit is {CONFIG.max_model_operation_seconds:.1f}s"
        )
    return result


def _restore_uncheckpointed_learning(
    session: Session, task: ModuleType, registration, model: PickledModel, snapshot
) -> None:
    """Recover labels learned after the last durable model checkpoint."""
    if not supports_learning(model):
        return
    # Snapshots created before checkpoint watermarks existed were written after
    # every learning batch. Treat them as authoritative to avoid replaying
    # their already-included history twice.
    if snapshot is not None and snapshot.checkpoint_label_available_at is None:
        return
    for event_id, event, y in store.trained_examples_since_checkpoint(
        session,
        task.TASK_NAME,
        registration.model_id,
        snapshot.checkpoint_label_available_at if snapshot is not None else None,
        snapshot.checkpoint_event_sequence if snapshot is not None else None,
    ):
        _model_operation(
            model.model_id, "learn", lambda event_id=event_id, event=event, y=y: model.learn_one(event_id, event, y)
        )


def _events(session: Session, task_name: str, event_ids: list[str], hot: HotStore | None) -> dict[str, dict[str, Any]]:
    """Read raw events from memory, then bulk-fall back to Postgres."""
    values = {event_id: hot.event(event_id) for event_id in event_ids} if hot is not None else {}
    missing = [event_id for event_id in event_ids if values.get(event_id) is None]
    if missing:
        recovered = store.event_payloads(session, task_name, missing)
        if hot is not None:
            for event_id, event in recovered.items():
                hot.put_payload(event_id, event)
        values.update(recovered)
    absent = {event_id for event_id in event_ids if values.get(event_id) is None}
    if absent:
        raise RuntimeError(f"raw events disappeared before processing: {', '.join(sorted(absent)[:3])}")
    return {event_id: event for event_id, event in values.items() if event is not None}


def _active_models(session: Session, task: ModuleType, cache: dict[str, CachedModel]) -> list[tuple[Any, CachedModel]]:
    """Keep models resident while noticing API additions and deactivations."""
    registrations = store.runnable_registrations(session, task.TASK_NAME)
    active_ids = {registration.model_id for registration in registrations}
    for model_id in set(cache) - active_ids:
        del cache[model_id]
    models = []
    for registration in registrations:
        definition = metric_definition(task.PROBLEM_TYPE, task.METRICS)
        fingerprint = (registration.artifact_id, definition["fingerprint"])
        cached = cache.get(registration.model_id)
        try:
            if cached is None or cached.fingerprint != fingerprint:
                persisted = store.model_metric_state(session, task.TASK_NAME, registration.model_id)
                tracker = (
                    MetricTracker.restore(definition, persisted.state)
                    if persisted is not None
                    else MetricTracker.fresh(
                        task.PROBLEM_TYPE,
                        task.METRICS,
                        store.model_prediction_count(session, task.TASK_NAME, registration.model_id),
                    )
                )
                model, snapshot = _load_model(session, task, registration)
                _restore_uncheckpointed_learning(session, task, registration, model, snapshot)
                # A legacy/no checkpoint must be upgraded on its next learning
                # batch before source rows can be archived.
                checkpointed_at = (
                    time.monotonic()
                    if snapshot is not None and snapshot.checkpoint_label_available_at is not None
                    else 0.0
                )
                cache[registration.model_id] = CachedModel(fingerprint, model, tracker, checkpointed_at)
        except Exception as error:
            cache.pop(registration.model_id, None)
            retry_at = store.record_model_failure(
                session,
                task.TASK_NAME,
                registration.model_id,
                error,
                CONFIG.model_retry_initial_seconds,
                CONFIG.model_retry_max_seconds,
            )
            logging.exception("could not load model %s; retry_at=%s", registration.model_id, retry_at)
            continue
        models.append((registration, cache[registration.model_id]))
    return models


def _learn_model(
    session: Session, task: ModuleType, registration, cached: CachedModel, hot: HotStore | None
) -> tuple[int, int, int]:
    """Process one model in the caller's savepoint."""
    model, tracker = cached.model, cached.tracker
    skipped = store.labelled_unpredicted_events(
        session, task.TASK_NAME, model.model_id, registration.start_sequence, CONFIG.learner_batch_size
    )
    store.add_prediction_skips(session, task.TASK_NAME, model.model_id, skipped)
    if hot is not None:
        hot.mark_labelled(skipped)
    evaluations = store.unevaluated_labels(session, task.TASK_NAME, model.model_id, CONFIG.learner_batch_size)
    if hot is not None:
        hot.mark_labelled([event_id for event_id, _, _ in evaluations])
    for _, y, prediction in evaluations:
        tracker.update(y, prediction, lambda metric, target, value: metric_inputs_for(task, metric, target, value))
    store.add_metric_updates(session, task.TASK_NAME, model.model_id, [event_id for event_id, _, _ in evaluations])
    labels = store.untrained_labels(session, task.TASK_NAME, model.model_id, CONFIG.learner_batch_size)
    if hot is not None:
        hot.mark_labelled([event_id for event_id, _, _, _ in labels])
    if supports_learning(model):
        label_events = _events(session, task.TASK_NAME, [event_id for event_id, _, _, _ in labels], hot)
        for event_id, y, _, _ in labels:
            _model_operation(
                model.model_id,
                "learn",
                lambda event_id=event_id, y=y: model.learn_one(event_id, label_events[event_id], y),
            )
    store.add_trainings(session, task.TASK_NAME, model.model_id, [event_id for event_id, _, _, _ in labels])
    events = store.unpredicted_events(
        session, task.TASK_NAME, model.model_id, registration.start_sequence, CONFIG.learner_batch_size
    )
    prediction_events = _events(session, task.TASK_NAME, events, hot)
    predictions = [
        (
            event_id,
            _model_operation(
                model.model_id,
                "predict",
                lambda event_id=event_id: prediction_for(task, model, event_id, prediction_events[event_id]),
            ),
        )
        for event_id in events
    ]
    inserted_predictions = set(store.add_predictions(session, task.TASK_NAME, model.model_id, predictions))
    raced_labels = [event_id for event_id, _ in predictions if event_id not in inserted_predictions]
    store.add_prediction_skips(session, task.TASK_NAME, model.model_id, raced_labels)
    tracker.predictions += len(inserted_predictions)
    if labels or evaluations or predictions:
        store.save_metric_state(
            session,
            task.TASK_NAME,
            model.model_id,
            tracker.definition,
            tracker.payload(),
            tracker.predictions,
            tracker.observations,
            tracker.values(),
        )
        if labels and time.monotonic() - cached.checkpointed_at >= CONFIG.model_checkpoint_seconds:
            _, _, label_available_at, event_sequence = labels[-1]
            payload = model.payload()
            if len(payload) > CONFIG.max_model_snapshot_bytes:
                raise ValueError(
                    f"serialized model is {len(payload):,} bytes; limit is {CONFIG.max_model_snapshot_bytes:,} bytes"
                )
            store.save_pickle_snapshot(
                session, task.TASK_NAME, model.model_id, payload, label_available_at, event_sequence
            )
            cached.checkpointed_at = time.monotonic()
    return len(labels), len(inserted_predictions), len(evaluations)


def learn_once(
    session: Session,
    task: ModuleType,
    cache: dict[str, CachedModel] | None = None,
    hot: HotStore | None = None,
    completed_hot_events: list[str] | None = None,
) -> list[tuple[str, int, int, int]]:
    cache = cache if cache is not None else {}
    results = []
    for registration in store.disabled_registrations(session, task.TASK_NAME):
        cache.pop(registration.model_id, None)
        store.record_disabled_work(session, task.TASK_NAME, registration, CONFIG.learner_batch_size)
    for registration, cached in _active_models(session, task, cache):
        try:
            with session.begin_nested():
                trained, predicted, evaluated = _learn_model(session, task, registration, cached, hot)
        except Exception as error:
            cache.pop(registration.model_id, None)
            retry_at = store.record_model_failure(
                session,
                task.TASK_NAME,
                registration.model_id,
                error,
                CONFIG.model_retry_initial_seconds,
                CONFIG.model_retry_max_seconds,
            )
            logging.exception("model %s failed; retry_at=%s", registration.model_id, retry_at)
            continue
        if trained or predicted or evaluated:
            store.record_model_success(session, task.TASK_NAME, registration.model_id)
        results.append((registration.model_id, trained, predicted, evaluated))
    for registration in store.disabled_registrations(session, task.TASK_NAME):
        cache.pop(registration.model_id, None)
        store.record_disabled_work(session, task.TASK_NAME, registration, CONFIG.learner_batch_size)
    if hot is not None:
        completed = store.completed_labelled_events(session, task.TASK_NAME, hot.labelled_event_ids())
        if completed_hot_events is None:
            hot.discard(completed)
        else:
            completed_hot_events.extend(completed)
    return results


def learner(
    sessions: sessionmaker[Session],
    task: ModuleType,
    once: bool = False,
    stop: threading.Event | None = None,
    hot: HotStore | None = None,
    heartbeat: bool = True,
) -> None:
    stop = stop or threading.Event()
    models: dict[str, CachedModel] = {}
    with sessions() as session:
        acquired = session.scalar(
            text("SELECT pg_try_advisory_lock(:lock_id)"), {"lock_id": _task_lock(task.TASK_NAME)}
        )
        if not acquired:
            raise RuntimeError(f"another learner is already running for {task.TASK_NAME}")
        with Heartbeat(sessions, task.TASK_NAME, "learner") if heartbeat else nullcontext():
            while not stop.is_set():
                try:
                    completed_hot_events: list[str] = []
                    results = learn_once(session, task, models, hot, completed_hot_events)
                    session.commit()
                    if hot is not None:
                        hot.discard(completed_hot_events)
                    for model_id, trained, predicted, evaluated in results:
                        if trained or predicted or evaluated:
                            logging.info(
                                "%s: trained=%d predicted=%d evaluated=%d", model_id, trained, predicted, evaluated
                            )
                except Exception:
                    session.rollback()
                    # A model can have learned in RAM before the surrounding
                    # database transaction fails. Discard it so the next
                    # cycle reloads the last committed checkpoint instead of
                    # learning the same labels twice.
                    models.clear()
                    logging.exception("learner cycle failed")
                if once:
                    return
                stop.wait(CONFIG.learner_idle_seconds)
