"""Resident model loading and predict-then-learn cycles."""

from __future__ import annotations

import logging
import threading
import time
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

from sqlalchemy import text
from sqlalchemy.orm import Session, sessionmaker

from everbench import artifacts, event_store, model_store
from everbench.config import CONFIG
from everbench.db import advisory_key
from everbench.heartbeat import Heartbeat
from everbench.hotstore import HotStore
from everbench.metrics import MetricTracker, metric_definition
from everbench.models import PickledModel, metric_inputs_for, prediction_for
from everbench.tasks import TaskDefinition


def _task_lock(task_name: str) -> int:
    return advisory_key("learner", task_name)


def _load_model(session: Session, task: TaskDefinition, registration):
    snapshot = model_store.latest_snapshot(session, task.TASK_NAME, registration.model_id)
    artifact_record = model_store.artifact(session, snapshot.artifact_id) if snapshot is not None else None
    if artifact_record is None and registration.artifact_id:
        artifact_record = model_store.artifact(session, registration.artifact_id)
    if artifact_record is None:
        raise RuntimeError(f"pickle artifact missing for {registration.model_id}")
    model = artifacts.loads(artifact_record.payload, artifact_record.signature)
    return PickledModel(registration.model_id, model), snapshot


@dataclass
class CachedModel:
    fingerprint: tuple[Any, ...]
    model: PickledModel
    tracker: MetricTracker
    checkpointed_at: float


def _restore_uncheckpointed_learning(
    session: Session, task: TaskDefinition, registration, model: PickledModel, snapshot
) -> None:
    """Recover labels learned after the last durable model checkpoint."""
    if not model.supports_learning:
        return
    # Snapshots created before checkpoint watermarks existed were written after
    # every learning batch. Treat them as authoritative to avoid replaying
    # their already-included history twice.
    if snapshot is not None and snapshot.checkpoint_label_available_at is None:
        return
    for event_id, event, y in model_store.trained_examples_since_checkpoint(
        session,
        task.TASK_NAME,
        registration.model_id,
        snapshot.checkpoint_label_available_at if snapshot is not None else None,
        snapshot.checkpoint_event_sequence if snapshot is not None else None,
    ):
        model.learn_one(event_id, event, y)


def _events(session: Session, task_name: str, event_ids: list[str], hot: HotStore | None) -> dict[str, dict[str, Any]]:
    """Read raw events from memory, then bulk-fall back to Postgres."""
    values = {event_id: hot.event(event_id) for event_id in event_ids} if hot is not None else {}
    missing = [event_id for event_id in event_ids if values.get(event_id) is None]
    if missing:
        recovered = event_store.event_payloads(session, task_name, missing)
        if hot is not None:
            for event_id, event in recovered.items():
                hot.put(event_id, event)
        values.update(recovered)
    absent = {event_id for event_id in event_ids if values.get(event_id) is None}
    if absent:
        raise RuntimeError(f"raw events disappeared before processing: {', '.join(sorted(absent)[:3])}")
    return {event_id: event for event_id, event in values.items() if event is not None}


def _active_models(
    session: Session, task: TaskDefinition, cache: dict[str, CachedModel]
) -> list[tuple[Any, CachedModel]]:
    """Keep models resident while noticing API additions and deactivations."""
    registrations = model_store.runnable_registrations(session, task.TASK_NAME)
    definition = metric_definition(task.PROBLEM_TYPE, task.METRICS)
    active_ids = {registration.model_id for registration in registrations}
    for model_id in set(cache) - active_ids:
        cache.pop(model_id)
    models = []
    for registration in registrations:
        fingerprint = (registration.artifact_id, definition["fingerprint"])
        cached = cache.get(registration.model_id)
        try:
            if cached is None or cached.fingerprint != fingerprint:
                cache.pop(registration.model_id, None)
                persisted = model_store.model_metric_state(session, task.TASK_NAME, registration.model_id)
                tracker = (
                    MetricTracker.restore(definition, persisted.state)
                    if persisted is not None
                    else MetricTracker.fresh(
                        task.PROBLEM_TYPE,
                        task.METRICS,
                        model_store.model_prediction_count(session, task.TASK_NAME, registration.model_id),
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
            retry_at = model_store.record_model_failure(
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
    session: Session, task: TaskDefinition, registration, cached: CachedModel, hot: HotStore | None
) -> tuple[int, int, int]:
    """Process one model in the caller's savepoint."""
    model, tracker = cached.model, cached.tracker
    skipped = event_store.labelled_unpredicted_events(
        session, task.TASK_NAME, model.model_id, registration.start_sequence, CONFIG.learner_batch_size
    )
    event_store.add_prediction_skips(session, task.TASK_NAME, model.model_id, skipped)
    if hot is not None:
        hot.mark_labelled(skipped)
    evaluations = event_store.unevaluated_labels(session, task.TASK_NAME, model.model_id, CONFIG.learner_batch_size)
    if hot is not None:
        hot.mark_labelled([event_id for event_id, _, _ in evaluations])
    for _, y, prediction in evaluations:
        tracker.update(y, prediction, lambda metric, target, value: metric_inputs_for(task, metric, target, value))
    event_store.add_metric_updates(
        session, task.TASK_NAME, model.model_id, [event_id for event_id, _, _ in evaluations]
    )
    labels = event_store.untrained_labels(session, task.TASK_NAME, model.model_id, CONFIG.learner_batch_size)
    if hot is not None:
        hot.mark_labelled([event_id for event_id, _, _, _ in labels])
    if model.supports_learning:
        label_events = _events(session, task.TASK_NAME, [event_id for event_id, _, _, _ in labels], hot)
        for event_id, y, _, _ in labels:
            model.learn_one(event_id, label_events[event_id], y)
    event_store.add_trainings(session, task.TASK_NAME, model.model_id, [event_id for event_id, _, _, _ in labels])
    events = event_store.unpredicted_events(
        session, task.TASK_NAME, model.model_id, registration.start_sequence, CONFIG.learner_batch_size
    )
    prediction_events = _events(session, task.TASK_NAME, events, hot)
    predictions = [
        (
            event_id,
            prediction_for(task, model, event_id, prediction_events[event_id]),
        )
        for event_id in events
    ]
    inserted_predictions = set(event_store.add_predictions(session, task.TASK_NAME, model.model_id, predictions))
    raced_labels = [event_id for event_id, _ in predictions if event_id not in inserted_predictions]
    event_store.add_prediction_skips(session, task.TASK_NAME, model.model_id, raced_labels)
    tracker.predictions += len(inserted_predictions)
    if labels or evaluations or predictions:
        model_store.save_metric_state(
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
            model_store.save_pickle_snapshot(
                session, task.TASK_NAME, model.model_id, payload, label_available_at, event_sequence
            )
            cached.checkpointed_at = time.monotonic()
    return len(labels), len(inserted_predictions), len(evaluations)


def learn_once(
    session: Session,
    task: TaskDefinition,
    cache: dict[str, CachedModel] | None = None,
    hot: HotStore | None = None,
    completed_hot_events: list[str] | None = None,
) -> list[tuple[str, int, int, int]]:
    cache = cache if cache is not None else {}
    results = []
    initially_disabled_model_ids = set()
    for registration in model_store.disabled_registrations(session, task.TASK_NAME):
        cache.pop(registration.model_id, None)
        model_store.record_disabled_work(session, task.TASK_NAME, registration, CONFIG.learner_batch_size)
        initially_disabled_model_ids.add(registration.model_id)
    for registration, cached in _active_models(session, task, cache):
        try:
            with session.begin_nested():
                trained, predicted, evaluated = _learn_model(session, task, registration, cached, hot)
        except Exception as error:
            cache.pop(registration.model_id, None)
            retry_at = model_store.record_model_failure(
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
            model_store.record_model_success(session, task.TASK_NAME, registration.model_id)
        results.append((registration.model_id, trained, predicted, evaluated))
    for registration in model_store.disabled_registrations(session, task.TASK_NAME):
        if registration.model_id in initially_disabled_model_ids:
            continue
        cache.pop(registration.model_id, None)
        model_store.record_disabled_work(session, task.TASK_NAME, registration, CONFIG.learner_batch_size)
    if hot is not None:
        completed = event_store.completed_labelled_events(session, task.TASK_NAME, hot.labelled_event_ids())
        if completed_hot_events is None:
            hot.discard(completed)
        else:
            completed_hot_events.extend(completed)
    return results


def learner(
    sessions: sessionmaker[Session],
    task: TaskDefinition,
    once: bool = False,
    stop: threading.Event | None = None,
    hot: HotStore | None = None,
    heartbeat: bool = True,
) -> None:
    stop = stop or threading.Event()
    models: dict[str, CachedModel] = {}
    engine = sessions.kw.get("bind")
    if engine is None:
        raise RuntimeError("learner session factory is not bound to an engine")
    # PostgreSQL session-level advisory locks belong to the physical connection,
    # not the SQLAlchemy Session. Pin that connection until this learner exits.
    with engine.connect() as connection, sessions(bind=connection) as session:
        acquired = session.scalar(
            text("SELECT pg_try_advisory_lock(:lock_id)"), {"lock_id": _task_lock(task.TASK_NAME)}
        )
        if not acquired:
            raise RuntimeError(f"another learner is already running for {task.TASK_NAME}")
        session.commit()
        try:
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
                                    "%s: trained=%d predicted=%d evaluated=%d",
                                    model_id,
                                    trained,
                                    predicted,
                                    evaluated,
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
        finally:
            models.clear()
            try:
                session.execute(text("SELECT pg_advisory_unlock(:lock_id)"), {"lock_id": _task_lock(task.TASK_NAME)})
                session.commit()
            except Exception:
                # Do not return a connection with an unknown lock state to the pool.
                connection.invalidate()
