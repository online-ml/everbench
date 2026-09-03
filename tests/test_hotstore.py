from __future__ import annotations

from everbench.collectors import _cache_durable_events
from everbench.hotstore import HotStore


def test_keeps_its_own_copy_of_an_event() -> None:
    hot = HotStore(capacity=2)
    original = {"nested": {"value": 1}}

    hot.put("one", original)
    original["nested"]["value"] = 2
    received = hot.event("one")
    assert received is not None
    received["nested"]["value"] = 3

    assert hot.event("one") == {"nested": {"value": 1}}


def test_only_caches_events_inserted_by_postgres() -> None:
    hot = HotStore(capacity=2)
    events = [("existing", 1.0, {"value": "ignored"}), ("new", 2.0, {"value": "durable"})]

    _cache_durable_events(hot, events, ["new"])

    assert hot.event("existing") is None
    assert hot.event("new") == {"value": "durable"}


def test_bypasses_an_oversized_event() -> None:
    hot = HotStore(capacity=2, max_event_bytes=10)

    hot.put("large", {"value": "too large"})

    assert hot.event("large") is None
    assert hot.stats()["bypasses"] == 1
