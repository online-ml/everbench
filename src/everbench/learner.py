"""Resident model loading and predict-then-learn cycles."""

from __future__ import annotations

import logging
import threading
import time
from contextlib import nullcontext
from dataclasses import dataclass
from functools import partial
from typing import Any

from sqlalchemy.orm import Session, sessionmaker

from everbench import artifacts, event_store, model_store
from everbench.config import CONFIG
from everbench.db import lock_transaction
from everbench.heartbeat import Heartbeat
from everbench.hotstore import HotStore
from everbench.metrics import MetricTracker, metric_definition
from everbench.models import PickledModel, metric_inputs_for, prediction_for
from everbench.tasks import TaskDefinition

_learner_locks: dict[str, threading.Lock] = {}
_learner_locks_guard = threading.Lock()


def _task_lock(*, task_name: str) -> threading.Lock:
    with _learner_locks_guard:
        return _learner_locks.setdefault(task_name, threading.Lock())


def _load_model(*, session: Session, task: TaskDefinition, registration):
    snapshot = model_store.latest_snapshot(session=session, task_name=task.TASK_NAME, model_id=registration.model_id)
    artifact_record = (
        model_store.artifact(session=session, artifact_id=snapshot.artifact_id) if snapshot is not None else None
    )
    if artifact_record is None and registration.artifact_id:
        artifact_record = model_store.artifact(session=session, artifact_id=registration.artifact_id)
    if artifact_record is None:
        raise RuntimeError(f"pickle artifact missing for {registration.model_id}")
    model = artifacts.loads(payload=artifact_record.payload, signature=artifact_record.signature)
    return PickledModel(model_id=registration.model_id, model=model), snapshot


@dataclass(kw_only=True)
class CachedModel:
    fingerprint: tuple[Any, ...]
    model: PickledModel
    tracker: MetricTracker
    checkpointed_at: float
    needs_checkpoint: bool = False


@dataclass(slots=True, kw_only=True)
class LearningResult:
    model_id: str
    trained: int = 0
    predicted: int = 0
    evaluated: int = 0
    resolved: int = 0

    @property
    def processed(self) -> bool:
        return bool(self.resolved or self.predicted)


def _restore_uncheckpointed_learning(
    *, session: Session, task: TaskDefinition, registration, model: PickledModel, snapshot
) -> None:
    """Recover labels learned after the last durable model checkpoint."""
    if not model.supports_learning:
        return
    for example in model_store.trained_examples_since_checkpoint(
        session=session,
        task_name=task.TASK_NAME,
        model_id=registration.model_id,
        checkpoint_ready_sequence=snapshot.checkpoint_ready_sequence if snapshot is not None else 0,
    ):
        model.learn_one(event_id=example.event_id, event=example.payload, label=example.target)


def _events(
    *, session: Session, task_name: str, event_ids: list[str], hot: HotStore | None
) -> dict[str, dict[str, Any]]:
    """Read raw events from memory, then bulk-fall back to SQLite."""
    values = {event_id: hot.event(event_id=event_id) for event_id in event_ids} if hot is not None else {}
    missing = [event_id for event_id in event_ids if values.get(event_id) is None]
    if missing:
        recovered = event_store.event_payloads(session=session, task_name=task_name, event_ids=missing)
        if hot is not None:
            for event_id, event in recovered.items():
                hot.put(event_id=event_id, event=event)
        values.update(recovered)
    absent = {event_id for event_id in event_ids if values.get(event_id) is None}
    if absent:
        raise RuntimeError(f"raw events disappeared before processing: {', '.join(sorted(absent)[:3])}")
    return {event_id: event for event_id, event in values.items() if event is not None}


def _active_models(
    *, session: Session, task: TaskDefinition, cache: dict[str, CachedModel]
) -> list[tuple[Any, CachedModel]]:
    """Keep models resident while noticing API additions and deactivations."""
    registrations = model_store.runnable_registrations(session=session, task_name=task.TASK_NAME)
    definition = metric_definition(problem_type=task.PROBLEM_TYPE, prototypes=task.METRICS)
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
                persisted = model_store.model_metric_state(
                    session=session, task_name=task.TASK_NAME, model_id=registration.model_id
                )
                tracker = (
                    MetricTracker.restore(definition=definition, payload=persisted.state)
                    if persisted is not None
                    else MetricTracker.fresh(
                        problem_type=task.PROBLEM_TYPE,
                        prototypes=task.METRICS,
                        predictions=model_store.model_prediction_count(
                            session=session, task_name=task.TASK_NAME, model_id=registration.model_id
                        ),
                    )
                )
                model, snapshot = _load_model(session=session, task=task, registration=registration)
                _restore_uncheckpointed_learning(
                    session=session, task=task, registration=registration, model=model, snapshot=snapshot
                )
                checkpointed_at = time.monotonic() if snapshot is not None else 0.0
                cache[registration.model_id] = CachedModel(
                    fingerprint=fingerprint,
                    model=model,
                    tracker=tracker,
                    checkpointed_at=checkpointed_at,
                    needs_checkpoint=registration.label_cursor_sequence
                    > (snapshot.checkpoint_ready_sequence if snapshot is not None else 0),
                )
        except Exception as error:
            cache.pop(registration.model_id, None)
            retry_at = model_store.record_model_failure(
                session=session,
                task_name=task.TASK_NAME,
                model_id=registration.model_id,
                error=error,
                retry_initial_seconds=CONFIG.model_retry_initial_seconds,
                retry_max_seconds=CONFIG.model_retry_max_seconds,
            )
            logging.exception("could not load model %s; retry_at=%s", registration.model_id, retry_at)
            continue
        models.append((registration, cache[registration.model_id]))
    return models


def _predict_pending(*, session: Session, task: TaskDefinition, registration, cached: CachedModel, hot) -> int:
    pending = event_store.events_after_cursor(
        session=session,
        task_name=task.TASK_NAME,
        model_id=registration.model_id,
        cursor_sequence=registration.prediction_cursor_sequence,
        start_sequence=registration.start_sequence,
        limit=CONFIG.learner_batch_size,
    )
    skipped = [row.event_id for row in pending if row.resolved and not row.has_state]
    event_store.add_prediction_skips(
        session=session, task_name=task.TASK_NAME, model_id=registration.model_id, event_ids=skipped
    )
    predictable = [row.event_id for row in pending if not row.resolved and not row.has_state]
    payloads = _events(session=session, task_name=task.TASK_NAME, event_ids=predictable, hot=hot)
    predictions = {
        identifier: prediction_for(task=task, model=cached.model, event_id=identifier, event=payloads[identifier])
        for identifier in predictable
    }
    inserted = set(
        event_store.add_predictions(
            session=session, task_name=task.TASK_NAME, model_id=registration.model_id, predictions=predictions
        )
    )
    raced = [identifier for identifier in predictable if identifier not in inserted]
    event_store.add_prediction_skips(
        session=session, task_name=task.TASK_NAME, model_id=registration.model_id, event_ids=raced
    )
    cached.tracker.predictions += len(inserted)
    if pending:
        registration.prediction_cursor_sequence = pending[-1].sequence
    return len(inserted)


def _process_resolutions(
    *, session: Session, task: TaskDefinition, registration, cached: CachedModel, hot
) -> LearningResult:
    resolutions = event_store.ready_labels_after_cursor(
        session=session,
        task_name=task.TASK_NAME,
        model_id=registration.model_id,
        cursor_sequence=registration.label_cursor_sequence,
        start_sequence=registration.start_sequence,
        limit=CONFIG.learner_batch_size,
    )
    missing = [row.event_id for row in resolutions if row.prediction_status is None]
    event_store.add_prediction_skips(
        session=session, task_name=task.TASK_NAME, model_id=registration.model_id, event_ids=missing
    )
    predicted = [row for row in resolutions if row.prediction_status == "predicted" and not row.evaluated]
    evaluable = [row for row in predicted if row.target is not None]
    for row in evaluable:
        cached.tracker.update(
            y_true=row.target,
            prediction=row.prediction,
            inputs_for=partial(metric_inputs_for, task=task),
        )
    # Unavailable targets settle the prediction without contributing a score.
    event_store.add_metric_updates(
        session=session,
        task_name=task.TASK_NAME,
        model_id=registration.model_id,
        event_ids=[row.event_id for row in predicted],
    )
    untrained = [row for row in resolutions if not row.trained]
    learnable = [row for row in untrained if row.target is not None]
    if cached.model.supports_learning:
        payloads = _events(
            session=session, task_name=task.TASK_NAME, event_ids=[row.event_id for row in learnable], hot=hot
        )
        for row in learnable:
            cached.model.learn_one(event_id=row.event_id, event=payloads[row.event_id], label=row.target)
    event_store.add_trainings(
        session=session,
        task_name=task.TASK_NAME,
        model_id=registration.model_id,
        event_ids=[row.event_id for row in untrained],
    )
    if resolutions:
        cached.needs_checkpoint = True
        registration.label_cursor_sequence = resolutions[-1].sequence
        if hot is not None:
            hot.mark_labelled(event_ids=[row.event_id for row in resolutions])
    return LearningResult(
        model_id=registration.model_id, resolved=len(resolutions), trained=len(learnable), evaluated=len(evaluable)
    )


def _checkpoint(*, session: Session, task: TaskDefinition, cached: CachedModel, ready_sequence: int) -> None:
    payload = cached.model.payload()
    if len(payload) > CONFIG.max_model_snapshot_bytes:
        raise ValueError(
            f"serialized model is {len(payload):,} bytes; limit is {CONFIG.max_model_snapshot_bytes:,} bytes"
        )
    model_store.save_pickle_snapshot(
        session=session,
        task_name=task.TASK_NAME,
        model_id=cached.model.model_id,
        payload=payload,
        checkpoint_ready_sequence=ready_sequence,
    )
    cached.checkpointed_at = time.monotonic()
    cached.needs_checkpoint = False


def _learn_model(
    *, session: Session, task: TaskDefinition, registration, cached: CachedModel, hot: HotStore | None
) -> LearningResult:
    """Predict, resolve, and checkpoint within the caller's savepoint."""
    predicted = _predict_pending(session=session, task=task, registration=registration, cached=cached, hot=hot)
    result = _process_resolutions(session=session, task=task, registration=registration, cached=cached, hot=hot)
    result.predicted = predicted
    if result.processed:
        tracker = cached.tracker
        model_store.save_metric_state(
            session=session,
            task_name=task.TASK_NAME,
            model_id=registration.model_id,
            definition=tracker.definition,
            state=tracker.payload(),
            predictions=tracker.predictions,
            observations=tracker.observations,
            values=tracker.values(),
        )
    if cached.needs_checkpoint and time.monotonic() - cached.checkpointed_at >= CONFIG.model_checkpoint_seconds:
        _checkpoint(session=session, task=task, cached=cached, ready_sequence=registration.label_cursor_sequence)
    return result


def learn_once(
    *,
    session: Session,
    task: TaskDefinition,
    cache: dict[str, CachedModel] | None = None,
    hot: HotStore | None = None,
    completed_hot_events: list[str] | None = None,
) -> list[LearningResult]:
    # Claim SQLite's writer slot before loading models or reading cursors.
    # A read transaction upgraded after a collector commits fails with
    # SQLITE_BUSY_SNAPSHOT even when busy_timeout is configured.
    lock_transaction(session=session, name=f"learner:{task.TASK_NAME}")
    cache = cache if cache is not None else {}
    results = []
    initially_disabled_model_ids = set()
    for registration in model_store.disabled_registrations(session=session, task_name=task.TASK_NAME):
        cache.pop(registration.model_id, None)
        model_store.record_disabled_work(
            session=session, task_name=task.TASK_NAME, registration=registration, limit=CONFIG.learner_batch_size
        )
        initially_disabled_model_ids.add(registration.model_id)
    for registration, cached in _active_models(session=session, task=task, cache=cache):
        try:
            with session.begin_nested():
                result = _learn_model(session=session, task=task, registration=registration, cached=cached, hot=hot)
        except Exception as error:
            cache.pop(registration.model_id, None)
            retry_at = model_store.record_model_failure(
                session=session,
                task_name=task.TASK_NAME,
                model_id=registration.model_id,
                error=error,
                retry_initial_seconds=CONFIG.model_retry_initial_seconds,
                retry_max_seconds=CONFIG.model_retry_max_seconds,
            )
            logging.exception("model %s failed; retry_at=%s", registration.model_id, retry_at)
            continue
        if result.processed:
            model_store.record_model_success(session=session, task_name=task.TASK_NAME, model_id=registration.model_id)
        results.append(result)
    for registration in model_store.disabled_registrations(session=session, task_name=task.TASK_NAME):
        if registration.model_id in initially_disabled_model_ids:
            continue
        cache.pop(registration.model_id, None)
        model_store.record_disabled_work(
            session=session, task_name=task.TASK_NAME, registration=registration, limit=CONFIG.learner_batch_size
        )
    if hot is not None:
        completed = event_store.completed_labelled_events(
            session=session, task_name=task.TASK_NAME, event_ids=hot.labelled_event_ids()
        )
        if completed_hot_events is None:
            hot.discard(event_ids=completed)
        else:
            completed_hot_events.extend(completed)
    return results


def learner(
    *,
    sessions: sessionmaker[Session],
    task: TaskDefinition,
    once: bool = False,
    stop: threading.Event | None = None,
    hot: HotStore | None = None,
    heartbeat: bool = True,
) -> None:
    stop = stop or threading.Event()
    models: dict[str, CachedModel] = {}
    task_lock = _task_lock(task_name=task.TASK_NAME)
    if not task_lock.acquire(blocking=False):
        raise RuntimeError(f"another learner is already running for {task.TASK_NAME}")
    try:
        with sessions() as session:
            with Heartbeat(sessions=sessions, task_name=task.TASK_NAME, role="learner") if heartbeat else nullcontext():
                while not stop.is_set():
                    try:
                        completed_hot_events: list[str] = []
                        results = learn_once(
                            session=session, task=task, cache=models, hot=hot, completed_hot_events=completed_hot_events
                        )
                        session.commit()
                        if hot is not None:
                            hot.discard(event_ids=completed_hot_events)
                        for result in results:
                            if result.processed:
                                logging.info(
                                    "%s: trained=%d predicted=%d evaluated=%d",
                                    result.model_id,
                                    result.trained,
                                    result.predicted,
                                    result.evaluated,
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
        task_lock.release()
