from __future__ import annotations

import io
import threading
import unittest
from unittest.mock import patch

from everbench.sse import _events, subscribe


class Response:
    def __init__(self, body: bytes):
        self.body = body

    def __enter__(self):
        return io.BytesIO(self.body)

    def __exit__(self, *_):
        return None


class SseTest(unittest.TestCase):
    def test_event_ids_are_carried_forward_between_sse_messages(self) -> None:
        response = io.BytesIO(b'id: 41\ndata: {"one": 1}\n\ndata: {"two": 2}\n\n')
        self.assertEqual(
            list(_events(response)),
            [("message", '{"one": 1}', "41"), ("message", '{"two": 2}', "41")],
        )

    def test_reconnect_request_uses_the_durable_cursor(self) -> None:
        stop = threading.Event()
        with patch("everbench.sse.urlopen", return_value=Response(b"id: 42\ndata: {}\n\n")) as urlopen:
            stream = subscribe("test", "https://example.test/stream", stop, last_event_id=lambda: "41")
            self.assertEqual(next(stream).event_id, "42")
            stop.set()
        request = urlopen.call_args.args[0]
        self.assertEqual(request.get_header("Last-event-id"), "41")


if __name__ == "__main__":
    unittest.main()
