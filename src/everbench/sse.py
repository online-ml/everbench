"""SSE subscription with durable-cursor reconnection."""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass

import httpx
from httpx_sse import SSEError, connect_sse

USER_AGENT = "everbench/0.1 (https://github.com/online-ml/everbench)"
READ_TIMEOUT_SECONDS = 10


@dataclass(frozen=True)
class StreamMessage:
    payload: dict
    event_id: str | None


def subscribe(
    name: str,
    url: str,
    stop: threading.Event | None = None,
    last_event_id: Callable[[], str | None] | None = None,
) -> Iterator[StreamMessage]:
    """Reconnect from the caller's last durably committed SSE cursor."""
    stop = stop or threading.Event()
    backoff = 1.0
    # A bounded read timeout lets a collector observe SIGTERM even when an
    # upstream stream goes completely silent.
    timeout = httpx.Timeout(connect=30, read=READ_TIMEOUT_SECONDS, write=30, pool=30)
    with httpx.Client(timeout=timeout, headers={"User-Agent": USER_AGENT}) as client:
        while not stop.is_set():
            cursor = last_event_id() if last_event_id is not None else None
            headers = {"Last-Event-ID": cursor} if cursor else {}
            try:
                logging.info("connecting to %s", name)
                with connect_sse(client, "GET", url, headers=headers) as event_source:
                    event_source.response.raise_for_status()
                    backoff = 1.0
                    current_event_id = cursor
                    for event in event_source.iter_sse():
                        if stop.is_set():
                            return
                        if event.retry is not None:
                            backoff = min(max(event.retry / 1_000, 0.1), 30.0)
                        if event.id:
                            current_event_id = event.id
                        if event.event != "message":
                            continue
                        try:
                            payload = event.json()
                        except json.JSONDecodeError:
                            logging.warning("ignoring malformed %s event", name)
                            continue
                        if isinstance(payload, dict):
                            yield StreamMessage(payload, current_event_id)
                        else:
                            logging.warning("ignoring non-object %s event", name)
            except (httpx.HTTPError, SSEError) as error:
                if stop.is_set():
                    return
                logging.warning("%s disconnected: %s; retrying in %.1fs", name, error, backoff)
                stop.wait(backoff)
                backoff = min(backoff * 2, 30.0)
