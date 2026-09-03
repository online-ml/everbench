"""Durable liveness reporting for long-running workers."""

from __future__ import annotations

import logging
import socket
import threading
from collections.abc import Callable
from contextlib import AbstractContextManager

from sqlalchemy.orm import Session, sessionmaker

from everbench import reporting
from everbench.config import CONFIG


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
                    reporting.record_heartbeat(
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
