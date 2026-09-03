from __future__ import annotations

import threading

import httpx
import pytest

from everbench.sse import subscribe


def test_event_ids_are_carried_forward_between_sse_messages(monkeypatch: pytest.MonkeyPatch) -> None:
    body = b'id: 41\ndata: {"one": 1}\n\ndata: {"two": 2}\n\n'
    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body)
        )
    )
    stop = threading.Event()
    monkeypatch.setattr("everbench.sse.httpx.Client", lambda **kwargs: client)

    stream = subscribe("test", "https://example.test/stream", stop)

    assert next(stream).event_id == "41"
    assert next(stream).event_id == "41"
    stop.set()


def test_reconnect_request_uses_the_durable_cursor(monkeypatch: pytest.MonkeyPatch) -> None:
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
    monkeypatch.setattr("everbench.sse.httpx.Client", lambda **kwargs: client)

    stream = subscribe("test", "https://example.test/stream", stop, last_event_id=lambda: "41")

    assert next(stream).event_id == "42"
    stop.set()
    assert requests[0].headers["Last-Event-ID"] == "41"
