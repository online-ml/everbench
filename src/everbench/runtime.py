"""Supervised single-process task runtime for small Railway deployments."""

from __future__ import annotations

import json
import logging
import queue
import signal
import threading
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, time, timedelta
from types import FrameType

from sqlalchemy.orm import Session, sessionmaker

from everbench.archive import archive_once, storage_configured
from everbench.collectors import collect_source, maintain_resolutions
from everbench.config import CONFIG
from everbench.heartbeat import Heartbeat
from everbench.hotstore import HotStore
from everbench.learner import learner
from everbench.tasks import TaskDefinition


@dataclass(frozen=True, slots=True, kw_only=True)
class Failure:
    name: str
    error: BaseException


def _log_failure(*, context: str, name: str, error: BaseException) -> None:
    logging.error(
        "%s %s failed",
        context,
        name,
        exc_info=(type(error), error, error.__traceback__),
    )


def _supervised(
    *, stop: threading.Event, failures: queue.SimpleQueue[Failure], name: str, target: Callable[[], None]
) -> Callable[[], None]:
    """Turn an unexpected thread exit into a process-level failure."""

    def run() -> None:
        try:
            target()
            if not stop.is_set():
                raise RuntimeError(f"{name} stopped unexpectedly")
        except BaseException as error:
            failures.put(Failure(name=name, error=error))
            stop.set()

    return run


@contextmanager
def _shutdown_signals(*, stop: threading.Event, enabled: bool):
    if not enabled:
        yield
        return

    def request_stop(_: int, __: FrameType | None) -> None:  # noqa: PLR0917 -- external positional protocol
        logging.info("shutdown requested; draining in-memory batches")
        stop.set()

    previous_handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    for sig in previous_handlers:
        signal.signal(sig, request_stop)
    try:
        yield
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)


def _run_threads(
    *,
    stop: threading.Event,
    failures: queue.SimpleQueue[Failure],
    threads: list[threading.Thread],
    failure_context: str,
) -> None:
    """Start, monitor, and join a set of essential runtime threads."""
    try:
        for thread in threads:
            thread.start()
        while any(thread.is_alive() for thread in threads):
            try:
                failure = failures.get(timeout=0.2)
            except queue.Empty:
                continue
            _log_failure(context=failure_context, name=failure.name, error=failure.error)
            stop.set()
            raise failure.error
        try:
            failure = failures.get_nowait()
        except queue.Empty:
            pass
        else:
            _log_failure(context=failure_context, name=failure.name, error=failure.error)
            raise failure.error
    finally:
        stop.set()
        for thread in threads:
            thread.join(timeout=CONFIG.shutdown_flush_seconds + 2)

    alive = [thread.name for thread in threads if thread.is_alive()]
    if alive:
        raise RuntimeError(f"workers did not stop before the shutdown deadline: {', '.join(alive)}")


def run_task(
    *,
    sessions: sessionmaker[Session],
    task: TaskDefinition,
    stop: threading.Event | None = None,
    install_signal_handlers: bool = True,
) -> None:
    """Run collectors and learner under one failure-propagating supervisor.

    SIGINT and SIGTERM request an orderly stop: collector batches are drained
    and the learner completes its current transaction. An unexpected failure
    in an essential loop stops the process so Railway can restart it.
    """
    stop = stop or threading.Event()
    hot = HotStore(capacity=CONFIG.hot_event_capacity, max_event_bytes=CONFIG.hot_event_max_bytes)
    failures: queue.SimpleQueue[Failure] = queue.SimpleQueue()

    def archive() -> None:
        """Periodically archive completed rows without interrupting live work."""
        if not storage_configured():
            logging.warning("archiver disabled: no durable archive target is configured")
            stop.wait()
            return
        while not stop.is_set():
            try:
                count = archive_once(sessions=sessions, task=task)
                if count:
                    logging.info("archived one weekly file with %d %s events", count, task.TASK_NAME)
            except Exception:
                # Source rows remain in SQLite and a later cycle retries.
                logging.exception("archiver cycle failed")
            stop.wait(CONFIG.archive_interval_seconds)

    threads = [
        threading.Thread(
            target=_supervised(
                stop=stop,
                failures=failures,
                name=source.name,
                target=lambda *, source=source: collect_source(
                    sessions=sessions, task=task, source=source, stop=stop, hot=hot
                ),
            ),
            name=source.name,
        )
        for source in task.sources
    ]
    threads.extend(
        [
            threading.Thread(
                target=_supervised(
                    stop=stop,
                    failures=failures,
                    name="resolutions",
                    target=lambda: maintain_resolutions(sessions=sessions, task=task, stop=stop, hot=hot),
                ),
                name="resolutions",
            ),
            threading.Thread(
                target=_supervised(
                    stop=stop,
                    failures=failures,
                    name="learner",
                    target=lambda: learner(sessions=sessions, task=task, stop=stop, hot=hot, heartbeat=False),
                ),
                name="learner",
            ),
            threading.Thread(
                target=_supervised(stop=stop, failures=failures, name="archiver", target=archive), name="archiver"
            ),
        ]
    )

    def detail() -> str:
        return json.dumps({"hot_store": hot.stats()})

    with (
        _shutdown_signals(stop=stop, enabled=install_signal_handlers),
        Heartbeat(sessions=sessions, task_name=task.TASK_NAME, role="task-runtime", detail=detail),
    ):
        _run_threads(stop=stop, failures=failures, threads=threads, failure_context="task runtime")


def _next_research_run(*, now: datetime) -> datetime:
    days = (0 - now.weekday()) % 7
    candidate = datetime.combine((now + timedelta(days=days)).date(), time(hour=1), tzinfo=UTC)
    return candidate if candidate > now else candidate + timedelta(days=7)


def _schedule_research(*, sessions: sessionmaker[Session], tasks: list[TaskDefinition], stop: threading.Event) -> None:
    from everbench.auto.service import auto_worker

    while not stop.is_set():
        due = _next_research_run(now=datetime.now(UTC))
        if stop.wait((due - datetime.now(UTC)).total_seconds()):
            return
        try:
            auto_worker(sessions=sessions, tasks=tasks)
        except Exception:
            logging.exception("weekly research failed")


def run_tasks(*, sessions: sessionmaker[Session], tasks: list[TaskDefinition], schedule_research: bool = False) -> None:
    """Run all task runtimes in one Railway worker process.

    One process lets each task keep its hot store in RAM, while the shared
    supervisor owns signal handling. A failure in any task ends the worker so
    Railway restarts it cleanly instead of silently leaving a task behind.
    """
    if not tasks:
        raise ValueError("at least one task is required")

    stop = threading.Event()
    failures: queue.SimpleQueue[Failure] = queue.SimpleQueue()

    def run(*, task: TaskDefinition) -> None:
        try:
            run_task(sessions=sessions, task=task, stop=stop, install_signal_handlers=False)
            if not stop.is_set():
                raise RuntimeError(f"task runtime {task.TASK_NAME!r} stopped unexpectedly")
        except BaseException as error:
            failures.put(Failure(name=task.TASK_NAME, error=error))
            stop.set()

    threads = [
        threading.Thread(target=run, kwargs={"task": task}, name=f"task-runtime-{task.TASK_NAME}") for task in tasks
    ]
    if schedule_research:
        threads.append(
            threading.Thread(
                target=_supervised(
                    stop=stop,
                    failures=failures,
                    name="research-scheduler",
                    target=lambda: _schedule_research(sessions=sessions, tasks=tasks, stop=stop),
                ),
                name="research-scheduler",
            )
        )
    with _shutdown_signals(stop=stop, enabled=True):
        _run_threads(stop=stop, failures=failures, threads=threads, failure_context="task runtime")
