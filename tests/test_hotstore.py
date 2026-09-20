from __future__ import annotations

from everbench.collectors import _cache_durable_events
from everbench.hotstore import HotStore
from everbench.records import Observation


def test_keeps_its_own_copy_of_an_event() -> None:
    hot = HotStore(capacity=2)
    original = {"nested": {"value": 1}}

    hot.put(event_id="one", event=original)
    original["nested"]["value"] = 2
    received = hot.event(event_id="one")
    assert received is not None
    received["nested"]["value"] = 3

    assert hot.event(event_id="one") == {"nested": {"value": 1}}


def test_only_caches_events_inserted_by_postgres() -> None:
    hot = HotStore(capacity=2)
    events = [
        Observation(event_id="existing", timestamp=1.0, payload={"value": "ignored"}),
        Observation(event_id="new", timestamp=2.0, payload={"value": "durable"}),
    ]

    _cache_durable_events(hot=hot, events=events, inserted_event_ids=["new"])

    assert hot.event(event_id="existing") is None
    assert hot.event(event_id="new") == {"value": "durable"}


def test_bypasses_an_oversized_event() -> None:
    hot = HotStore(capacity=2, max_event_bytes=10)

    hot.put(event_id="large", event={"value": "too large"})

    assert hot.event(event_id="large") is None
    assert hot.stats()["bypasses"] == 1
