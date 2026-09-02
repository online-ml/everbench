"""Supervised single-process task runtime for small Railway deployments."""

from __future__ import annotations

import json
import logging
import queue
import signal
import threading
from collections.abc import Callable
from types import FrameType, ModuleType

from sqlalchemy.orm import Session, sessionmaker

from everbench.archive import archive_once, storage_configured
from everbench.config import CONFIG
from everbench.hotstore import HotStore
from everbench.workers import Heartbeat, collect_events, collect_labels, learner


def run_task(
    sessions: sessionmaker[Session],
    task: ModuleType,
    *,
    stop: threading.Event | None = None,
    install_signal_handlers: bool = True,
) -> None:
    """Run collectors and learner under one failure-propagating supervisor.

    SIGINT and SIGTERM request an orderly stop: collector batches are drained
    and the learner completes its current transaction. An unexpected failure
    in an essential loop stops the process so Railway can restart it.
    """
    stop = stop or threading.Event()
    hot = HotStore(CONFIG.hot_event_capacity)
    failures: queue.SimpleQueue[BaseException] = queue.SimpleQueue()

    def supervised(name: str, target: Callable[[], None]) -> Callable[[], None]:
        def run() -> None:
            try:
                target()
                if not stop.is_set():
                    raise RuntimeError(f"{name} stopped unexpectedly")
            except BaseException as error:
                failures.put(error)
                stop.set()

        return run

    def request_stop(_: int, __: FrameType | None) -> None:
        logging.info("shutdown requested; draining in-memory batches")
        stop.set()

    def compact() -> None:
        """Periodically archive completed rows without interrupting live work."""
        if not storage_configured():
            logging.warning("archive compactor disabled: no durable archive target is configured")
            stop.wait()
            return
        while not stop.is_set():
            try:
                count = archive_once(sessions, task)
                if count:
                    logging.info("archived and compacted %d %s events", count, task.TASK_NAME)
            except Exception:
                # Source rows remain in Postgres and a later cycle retries.
                logging.exception("archive compactor cycle failed")
            stop.wait(CONFIG.archive_interval_seconds)

    threads = [
        threading.Thread(
            target=supervised("event collector", lambda: collect_events(sessions, task, stop, hot, heartbeat=False)),
            name="event-collector",
        ),
        threading.Thread(
            target=supervised("label collector", lambda: collect_labels(sessions, task, stop, hot, heartbeat=False)),
            name="label-collector",
        ),
        threading.Thread(
            target=supervised("learner", lambda: learner(sessions, task, stop=stop, hot=hot, heartbeat=False)),
            name="learner",
        ),
        threading.Thread(target=supervised("archive compactor", compact), name="archive-compactor"),
    ]

    previous_handlers = {}
    if install_signal_handlers:
        previous_handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
        for sig in previous_handlers:
            signal.signal(sig, request_stop)
    try:

        def detail() -> str:
            return json.dumps({"hot_store": hot.stats()})

        with Heartbeat(sessions, task.TASK_NAME, "task-runtime", detail):
            for thread in threads:
                thread.start()
            while any(thread.is_alive() for thread in threads):
                try:
                    error = failures.get(timeout=0.2)
                except queue.Empty:
                    continue
                logging.exception("task runtime failed", exc_info=error)
                stop.set()
                raise error
            try:
                error = failures.get_nowait()
            except queue.Empty:
                pass
            else:
                logging.exception("task runtime failed", exc_info=error)
                raise error
    finally:
        stop.set()
        for thread in threads:
            thread.join(timeout=CONFIG.shutdown_flush_seconds + 2)
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)

    alive = [thread.name for thread in threads if thread.is_alive()]
    if alive:
        raise RuntimeError(f"workers did not stop before the shutdown deadline: {', '.join(alive)}")


def run_tasks(sessions: sessionmaker[Session], tasks: list[ModuleType]) -> None:
    """Run all task runtimes in one Railway worker process.

    One process lets each task keep its hot store in RAM, while the shared
    supervisor owns signal handling. A failure in any task ends the worker so
    Railway restarts it cleanly instead of silently leaving a task behind.
    """
    if not tasks:
        raise ValueError("at least one task is required")

    stop = threading.Event()
    failures: queue.SimpleQueue[tuple[str, BaseException]] = queue.SimpleQueue()

    def run(task: ModuleType) -> None:
        try:
            run_task(sessions, task, stop=stop, install_signal_handlers=False)
            if not stop.is_set():
                raise RuntimeError(f"task runtime {task.TASK_NAME!r} stopped unexpectedly")
        except BaseException as error:
            failures.put((task.TASK_NAME, error))
            stop.set()

    def request_stop(_: int, __: FrameType | None) -> None:
        logging.info("shutdown requested; draining in-memory batches")
        stop.set()

    previous_handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    for sig in previous_handlers:
        signal.signal(sig, request_stop)
    threads = [threading.Thread(target=run, args=(task,), name=f"task-runtime-{task.TASK_NAME}") for task in tasks]
    try:
        for thread in threads:
            thread.start()
        while any(thread.is_alive() for thread in threads):
            try:
                task_name, error = failures.get(timeout=0.2)
            except queue.Empty:
                continue
            logging.exception("task runtime %s failed", task_name, exc_info=error)
            stop.set()
            raise error
    finally:
        stop.set()
        for thread in threads:
            thread.join(timeout=CONFIG.shutdown_flush_seconds + 2)
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)

    alive = [thread.name for thread in threads if thread.is_alive()]
    if alive:
        raise RuntimeError(f"task runtimes did not stop before the shutdown deadline: {', '.join(alive)}")
