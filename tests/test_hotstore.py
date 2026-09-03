from __future__ import annotations

import unittest

from everbench.hotstore import HotStore
from everbench.workers import _cache_durable_events


class HotStoreTest(unittest.TestCase):
    def test_keeps_its_own_copy_of_an_event(self) -> None:
        hot = HotStore(capacity=2)
        original = {"nested": {"value": 1}}

        hot.put_event("one", original)
        original["nested"]["value"] = 2
        received = hot.event("one")
        assert received is not None
        received["nested"]["value"] = 3

        self.assertEqual(hot.event("one"), {"nested": {"value": 1}})

    def test_only_caches_events_inserted_by_postgres(self) -> None:
        hot = HotStore(capacity=2)
        events = [("existing", 1.0, {"value": "ignored"}), ("new", 2.0, {"value": "durable"})]

        _cache_durable_events(hot, events, ["new"])

        self.assertIsNone(hot.event("existing"))
        self.assertEqual(hot.event("new"), {"value": "durable"})

    def test_bypasses_an_oversized_event(self) -> None:
        hot = HotStore(capacity=2, max_event_bytes=10)

        hot.put_event("large", {"value": "too large"})

        self.assertIsNone(hot.event("large"))
        self.assertEqual(hot.stats()["bypasses"], 1)


if __name__ == "__main__":
    unittest.main()
