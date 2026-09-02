"""Minimal standard-library SSE client shared by collector workers."""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Iterator
from urllib.error import URLError
from urllib.request import Request, urlopen

USER_AGENT = "everbench/0.1 (https://github.com/MaxHalford/river-vandalism-demo)"


def _events(response) -> Iterator[tuple[str, str]]:
    event_type, data_lines = "message", []
    for raw_line in response:
        line = raw_line.decode("utf-8").rstrip("\r\n")
        if not line:
            if data_lines:
                yield event_type, "\n".join(data_lines)
            event_type, data_lines = "message", []
        elif line.startswith("event:"):
            event_type = line.removeprefix("event:").lstrip()
        elif line.startswith("data:"):
            data_lines.append(line.removeprefix("data:").lstrip())


def subscribe(name: str, url: str, stop: threading.Event | None = None) -> Iterator[dict]:
    """Reconnect until stopped; the caller owns durable de-duplication."""
    stop = stop or threading.Event()
    backoff = 1
    while not stop.is_set():
        try:
            logging.info("connecting to %s", name)
            request = Request(url, headers={"Accept": "text/event-stream", "User-Agent": USER_AGENT})
            with urlopen(request, timeout=90) as response:
                backoff = 1
                for event_type, data in _events(response):
                    if stop.is_set():
                        return
                    if event_type != "message":
                        continue
                    try:
                        yield json.loads(data)
                    except json.JSONDecodeError:
                        logging.warning("ignoring malformed %s event", name)
        except (OSError, URLError) as error:
            if stop.is_set():
                return
            logging.warning("%s disconnected: %s; retrying in %ss", name, error, backoff)
            stop.wait(backoff)
            backoff = min(backoff * 2, 30)
