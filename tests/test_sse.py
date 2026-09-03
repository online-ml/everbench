from __future__ import annotations

import threading
import unittest
from unittest.mock import patch

import httpx

from everbench.sse import subscribe


class SseTest(unittest.TestCase):
    def test_event_ids_are_carried_forward_between_sse_messages(self) -> None:
        body = b'id: 41\ndata: {"one": 1}\n\ndata: {"two": 2}\n\n'
        client = httpx.Client(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body)
            )
        )
        stop = threading.Event()
        with patch("everbench.sse.httpx.Client", return_value=client):
            stream = subscribe("test", "https://example.test/stream", stop)
            self.assertEqual(next(stream).event_id, "41")
            self.assertEqual(next(stream).event_id, "41")
            stop.set()

    def test_reconnect_request_uses_the_durable_cursor(self) -> None:
        requests: list[httpx.Request] = []

        def respond(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=b"id: 42\ndata: {}\n\n",
            )

        client = httpx.Client(transport=httpx.MockTransport(respond))
        stop = threading.Event()
        with patch("everbench.sse.httpx.Client", return_value=client):
            stream = subscribe("test", "https://example.test/stream", stop, last_event_id=lambda: "41")
            self.assertEqual(next(stream).event_id, "42")
            stop.set()
        self.assertEqual(requests[0].headers["Last-Event-ID"], "41")


if __name__ == "__main__":
    unittest.main()
