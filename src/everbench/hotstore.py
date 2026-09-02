"""A bounded, thread-safe cache for event features shared by one runtime."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from threading import RLock


@dataclass(frozen=True)
class HotEvent:
    features: dict[str, float]


class HotStore:
    """Least-recently-used read cache; Postgres remains the source of truth."""

    def __init__(self, capacity: int):
        if capacity < 1:
            raise ValueError("hot store capacity must be positive")
        self.capacity = capacity
        self._events: OrderedDict[str, HotEvent] = OrderedDict()
        # A label can arrive before the learner has processed it. Track only a
        # bounded set of those IDs so completed events can leave RAM promptly.
        self._labelled: OrderedDict[str, None] = OrderedDict()
        self._hits = 0
        self._misses = 0
        self._lock = RLock()

    def put_event(self, event_id: str, features: dict[str, float]) -> None:
        with self._lock:
            self._events[event_id] = HotEvent(features)
            self._events.move_to_end(event_id)
            self._trim()

    def put_features(self, event_id: str, features: dict[str, float]) -> None:
        with self._lock:
            event = self._events.get(event_id)
            if event is None:
                self._events[event_id] = HotEvent(features)
            else:
                self._events[event_id] = HotEvent(features)
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

    def features(self, event_id: str) -> dict[str, float] | None:
        with self._lock:
            event = self._events.get(event_id)
            if event is None:
                self._misses += 1
                return None
            self._hits += 1
            self._events.move_to_end(event_id)
            return event.features

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "entries": len(self._events),
                "capacity": self.capacity,
                "hits": self._hits,
                "misses": self._misses,
            }

    def __len__(self) -> int:
        with self._lock:
            return len(self._events)
