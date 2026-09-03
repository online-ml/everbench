"""Bounded in-memory batch with a record-count and elapsed-time flush policy."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from threading import RLock
from typing import Generic, TypeVar

Item = TypeVar("Item")


class TimedBatch(Generic[Item]):
    def __init__(
        self,
        max_items: int,
        max_age_seconds: float,
        flush: Callable[[list[Item]], None],
        max_pending_items: int | None = None,
    ):
        self.max_items = max_items
        self.max_age_seconds = max_age_seconds
        self.max_pending_items = max_pending_items or max_items * 10
        self.flush_callback = flush
        self.items: list[Item] = []
        self.opened_at = time.monotonic()
        self._lock = RLock()

    def add(self, item: Item) -> None:
        with self._lock:
            self.items.append(item)
            if len(self.items) > self.max_pending_items:
                raise RuntimeError(f"batch exceeded its {self.max_pending_items} item outage limit")
            if len(self.items) >= self.max_items or time.monotonic() - self.opened_at >= self.max_age_seconds:
                self._flush_locked()

    def flush(self) -> bool:
        with self._lock:
            return self._flush_locked()

    def flush_if_due(self) -> bool:
        """Flush a non-empty batch even when the source has gone quiet."""
        with self._lock:
            if self.items and time.monotonic() - self.opened_at >= self.max_age_seconds:
                return self._flush_locked()
            return False

    def _flush_locked(self) -> bool:
        if not self.items:
            return True
        # Preserve the in-memory batch if the database transaction fails. The
        # collector can retry it (and inserts are idempotent) rather than
        # silently dropping events on a transient outage.
        try:
            self.flush_callback(self.items)
        except Exception:
            # Keep the batch for the next event-driven retry. A persistent
            # outage will still be visible in logs and Railway health checks.
            logging.exception("batch flush failed; retaining %d items", len(self.items))
            self.opened_at = time.monotonic()
            return False
        self.items = []
        self.opened_at = time.monotonic()
        return True
