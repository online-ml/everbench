"""Minimal standard-library SSE client shared by collector workers."""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from urllib.error import URLError
from urllib.request import Request, urlopen

USER_AGENT = "everbench/0.1 (https://github.com/MaxHalford/river-vandalism-demo)"


@dataclass(frozen=True)
class StreamMessage:
    payload: dict
    event_id: str | None


def _events(response, initial_event_id: str | None = None) -> Iterator[tuple[str, str, str | None]]:
    event_type, data_lines, event_id = "message", [], initial_event_id
    for raw_line in response:
        line = raw_line.decode("utf-8").rstrip("\r\n")
        if not line:
            if data_lines:
                yield event_type, "\n".join(data_lines), event_id
            event_type, data_lines = "message", []
        elif line.startswith("event:"):
            event_type = line.removeprefix("event:").lstrip()
        elif line.startswith("id:"):
            event_id = line.removeprefix("id:").lstrip()
        elif line.startswith("data:"):
            data_lines.append(line.removeprefix("data:").lstrip())


def subscribe(
    name: str,
    url: str,
    stop: threading.Event | None = None,
    last_event_id: Callable[[], str | None] | None = None,
) -> Iterator[StreamMessage]:
    """Reconnect from the caller's last durably committed SSE cursor."""
    stop = stop or threading.Event()
    backoff = 1
    while not stop.is_set():
        try:
            logging.info("connecting to %s", name)
            cursor = last_event_id() if last_event_id is not None else None
            headers = {"Accept": "text/event-stream", "User-Agent": USER_AGENT}
            if cursor:
                headers["Last-Event-ID"] = cursor
            request = Request(url, headers=headers)
            with urlopen(request, timeout=90) as response:
                backoff = 1
                for event_type, data, event_id in _events(response, cursor):
                    if stop.is_set():
                        return
                    if event_type != "message":
                        continue
                    try:
                        payload = json.loads(data)
                        if isinstance(payload, dict):
                            yield StreamMessage(payload, event_id)
                        else:
                            logging.warning("ignoring non-object %s event", name)
                    except json.JSONDecodeError:
                        logging.warning("ignoring malformed %s event", name)
        except (OSError, URLError) as error:
            if stop.is_set():
                return
            logging.warning("%s disconnected: %s; retrying in %ss", name, error, backoff)
            stop.wait(backoff)
            backoff = min(backoff * 2, 30)
