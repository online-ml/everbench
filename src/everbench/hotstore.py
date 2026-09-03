"""A bounded, thread-safe cache for raw events shared by one runtime."""

from __future__ import annotations

import json
from collections import OrderedDict
from copy import deepcopy
from threading import RLock
from typing import Any


class HotStore:
    """A bounded, defensive read cache; Postgres remains the source of truth."""

    def __init__(self, capacity: int, max_event_bytes: int | None = None):
        if capacity < 1:
            raise ValueError("hot store capacity must be positive")
        if max_event_bytes is not None and max_event_bytes < 1:
            raise ValueError("hot store maximum event size must be positive")
        self.capacity = capacity
        self.max_event_bytes = max_event_bytes
        self._events: OrderedDict[str, dict[str, Any]] = OrderedDict()
        # A label can arrive before the learner has processed it. Track only a
        # bounded set of those IDs so completed events can leave RAM promptly.
        self._labelled: OrderedDict[str, None] = OrderedDict()
        self._hits = 0
        self._misses = 0
        self._bypasses = 0
        self._lock = RLock()

    def put(self, event_id: str, event: dict[str, Any]) -> None:
        """Cache JSON-sized payloads without sharing mutable references."""
        if self.max_event_bytes is not None:
            try:
                encoded = json.dumps(event, separators=(",", ":")).encode()
            except (TypeError, ValueError):
                # Postgres is still the authoritative fallback. Refusing an
                # unusual payload here is safer than keeping unbounded data.
                with self._lock:
                    self._bypasses += 1
                return
            if len(encoded) > self.max_event_bytes:
                with self._lock:
                    self._bypasses += 1
                return
        with self._lock:
            self._events[event_id] = deepcopy(event)
            self._events.move_to_end(event_id)
            self._trim()

    def mark_labelled(self, event_ids: list[str]) -> None:
        """Remember newly durable labels until their cached events are settled."""
        with self._lock:
            for event_id in event_ids:
                self._labelled[event_id] = None
                self._labelled.move_to_end(event_id)
            while len(self._labelled) > self.capacity:
                self._labelled.popitem(last=False)

    def labelled_event_ids(self) -> list[str]:
        with self._lock:
            return list(self._labelled)

    def discard(self, event_ids: list[str]) -> None:
        """Release events that no active model needs in memory any longer."""
        with self._lock:
            for event_id in event_ids:
                self._events.pop(event_id, None)
                self._labelled.pop(event_id, None)

    def _trim(self) -> None:
        while len(self._events) > self.capacity:
            event_id, _ = self._events.popitem(last=False)
            self._labelled.pop(event_id, None)

    def event(self, event_id: str) -> dict[str, Any] | None:
        with self._lock:
            event = self._events.get(event_id)
            if event is None:
                self._misses += 1
                return None
            self._hits += 1
            self._events.move_to_end(event_id)
            return deepcopy(event)

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "entries": len(self._events),
                "capacity": self.capacity,
                "hits": self._hits,
                "misses": self._misses,
                "bypasses": self._bypasses,
            }

    def __len__(self) -> int:
        with self._lock:
            return len(self._events)
